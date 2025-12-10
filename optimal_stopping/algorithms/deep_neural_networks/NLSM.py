"""
Neural Network Regression for Bermudan Option Pricing (NLSM)
Based on Lapeyre, B. and Lelong, J. (2021)
"Neural network regression for bermudan option pricing"
Monte Carlo Methods and Applications, 27(3):227-247

Key idea:
- Replace polynomial basis functions in LSM with neural networks
- Train networks to approximate continuation value E[V_{t+1} | S_t]
- Use MSE loss for regression (vs policy gradient in DOS)
- Backward induction: at each time, regress discounted future value on current state

This is similar to RLSM but uses deeper networks with proper training.
"""

import time
import numpy as np

# Optional: Try to use PyTorch for GPU acceleration
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class NeuralLSM:
    """
    Neural Network LSM (Lapeyre & Lelong 2021).

    Uses neural networks to learn continuation values via regression,
    replacing polynomial basis functions in classical LSM.

    Parameters:
    -----------
    model : StockModel
        The underlying asset model
    payoff : Payoff
        The option payoff function
    hidden_size : int, optional
        Number of hidden units per layer (default: 40)
    nb_paths : int, optional
        Number of paths for training (default: 4096)
    nb_epochs : int, optional
        Number of training epochs per time step (default: 50)
    learning_rate : float, optional
        Learning rate for Adam optimizer (default: 0.001)
    use_payoff_as_input : bool, optional
        Include current payoff in network input (default: True)
    train_ITM_only : bool, optional
        Train only on in-the-money paths (like LSM, default: True)
    batch_size : int, optional
        Mini-batch size for training (default: 256)
    **kwargs : dict
        Additional arguments (ignored)
    """

    def __init__(self, model, payoff, hidden_size=40, nb_paths=4096,
                 nb_epochs=50, learning_rate=0.001, use_payoff_as_input=True,
                 train_ITM_only=True, batch_size=256, **kwargs):
        self.model = model
        self.payoff = payoff
        self.hidden_size = hidden_size
        self.nb_paths = nb_paths
        self.nb_epochs = nb_epochs
        self.learning_rate = learning_rate
        self.use_payoff_as_input = use_payoff_as_input
        self.train_ITM_only = train_ITM_only
        self.batch_size = batch_size

        # Model parameters
        self.r = model.rate
        self.dt = model.dt
        self.nb_dates = model.nb_dates
        self.nb_stocks = model.nb_stocks
        self.S0 = np.array(model.spot).flatten()

        if len(self.S0) == 1 and self.nb_stocks > 1:
            self.S0 = np.full(self.nb_stocks, self.S0[0])

        # Input dimension: d stocks + (optionally) payoff
        self.input_dim = self.nb_stocks
        if self.use_payoff_as_input:
            self.input_dim += 1

        # Networks for each time step
        self.networks = {}

        # Feature normalization
        self.feature_mean = None
        self.feature_std = None

        # Use PyTorch if available
        self.use_torch = TORCH_AVAILABLE

    def _generate_paths_timed(self, nb_paths):
        """Generate paths with explicit timing."""
        start_time = time.time()
        result = self.model.generate_paths(nb_paths=nb_paths)
        elapsed = time.time() - start_time

        if isinstance(result, tuple):
            paths = result[0]
        else:
            paths = result

        return paths, elapsed

    def _normalize_state(self, states):
        """Normalize states for neural network input."""
        return np.log(states / self.S0[np.newaxis, :])

    def _build_features(self, states, payoffs=None, fit=False):
        """Build and normalize input features."""
        features = self._normalize_state(states)

        if self.use_payoff_as_input and payoffs is not None:
            norm_payoffs = payoffs[:, np.newaxis] / self.payoff.strike
            features = np.concatenate([features, norm_payoffs], axis=1)

        if fit:
            self.feature_mean = np.mean(features, axis=0)
            self.feature_std = np.std(features, axis=0) + 1e-8

        if self.feature_mean is not None:
            features = (features - self.feature_mean) / self.feature_std

        return features.astype(np.float32)

    # ==================== NumPy Network Implementation ====================

    def _init_network_numpy(self):
        """Initialize network weights (NumPy)."""
        h = self.hidden_size
        d = self.input_dim

        # He initialization for ReLU
        W1 = np.random.randn(d, h).astype(np.float32) * np.sqrt(2.0 / d)
        b1 = np.zeros(h, dtype=np.float32)
        W2 = np.random.randn(h, h).astype(np.float32) * np.sqrt(2.0 / h)
        b2 = np.zeros(h, dtype=np.float32)
        W3 = np.random.randn(h, 1).astype(np.float32) * np.sqrt(2.0 / h)
        b3 = np.zeros(1, dtype=np.float32)

        return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}

    def _forward_numpy(self, features, params):
        """Forward pass with ReLU activation (NumPy)."""
        # Layer 1
        z1 = features @ params['W1'] + params['b1']
        a1 = np.maximum(0, z1)  # ReLU

        # Layer 2
        z2 = a1 @ params['W2'] + params['b2']
        a2 = np.maximum(0, z2)  # ReLU

        # Output layer (linear for regression)
        z3 = a2 @ params['W3'] + params['b3']
        output = z3.flatten()

        return output, {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2, 'z3': z3}

    def _backward_numpy(self, features, output, target, cache, params):
        """Backward pass for MSE loss (NumPy)."""
        n = features.shape[0]

        # MSE loss gradient: 2 * (output - target) / n
        d_output = 2 * (output - target) / n
        d_z3 = d_output[:, np.newaxis]

        d_W3 = cache['a2'].T @ d_z3
        d_b3 = np.sum(d_z3, axis=0)

        # ReLU derivative
        d_a2 = d_z3 @ params['W3'].T
        d_z2 = d_a2 * (cache['z2'] > 0)

        d_W2 = cache['a1'].T @ d_z2
        d_b2 = np.sum(d_z2, axis=0)

        d_a1 = d_z2 @ params['W2'].T
        d_z1 = d_a1 * (cache['z1'] > 0)

        d_W1 = features.T @ d_z1
        d_b1 = np.sum(d_z1, axis=0)

        return {'W1': d_W1, 'b1': d_b1, 'W2': d_W2, 'b2': d_b2, 'W3': d_W3, 'b3': d_b3}

    def _train_network_numpy(self, features, targets):
        """Train network via mini-batch SGD with Adam (NumPy)."""
        params = self._init_network_numpy()

        # Adam optimizer state
        m = {k: np.zeros_like(v) for k, v in params.items()}
        v = {k: np.zeros_like(v) for k, v in params.items()}
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        n_samples = features.shape[0]
        n_batches = max(1, n_samples // self.batch_size)

        step = 0
        for epoch in range(self.nb_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)

            for batch_idx in range(n_batches):
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, n_samples)
                batch_indices = indices[start:end]

                batch_features = features[batch_indices]
                batch_targets = targets[batch_indices]

                # Forward pass
                output, cache = self._forward_numpy(batch_features, params)

                # Backward pass
                grads = self._backward_numpy(batch_features, output, batch_targets,
                                             cache, params)

                # Adam update
                step += 1
                for k in params:
                    m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
                    v[k] = beta2 * v[k] + (1 - beta2) * (grads[k] ** 2)
                    m_hat = m[k] / (1 - beta1 ** step)
                    v_hat = v[k] / (1 - beta2 ** step)
                    params[k] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        return params

    # ==================== PyTorch Network Implementation ====================

    def _create_network_torch(self):
        """Create PyTorch network."""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )

    def _train_network_torch(self, features, targets):
        """Train network with PyTorch."""
        net = self._create_network_torch()
        optimizer = optim.Adam(net.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        features_t = torch.FloatTensor(features)
        targets_t = torch.FloatTensor(targets)

        dataset = torch.utils.data.TensorDataset(features_t, targets_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                             shuffle=True)

        for epoch in range(self.nb_epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred = net(batch_x).squeeze()
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

        return net

    # ==================== Training ====================

    def _train_continuation_networks(self, paths, payoffs_all):
        """
        Train continuation value networks using backward iteration.

        At each time t, train network to predict E[V_{t+1} | S_t]
        where V_{t+1} is the (discounted) optimal value at t+1.
        """
        n_paths = paths.shape[0]
        T = self.nb_dates
        df = self.model.df

        # Value function (updated backward)
        V = np.zeros((n_paths, T + 1), dtype=np.float32)
        V[:, T] = payoffs_all[:, T]  # Terminal value

        # Stopping decisions (for pricing)
        stop_decisions = np.zeros((n_paths, T), dtype=bool)

        # Fit feature normalization on all states
        all_states = paths[:, :, :-1].reshape(-1, self.nb_stocks)
        _ = self._build_features(all_states, fit=True)

        # Backward iteration
        for t in range(T - 1, -1, -1):
            states = paths[:, :, t]
            payoffs_t = payoffs_all[:, t]
            features = self._build_features(states, payoffs_t)

            # Target: discounted value at t+1
            target = df * V[:, t + 1]

            # Filter to ITM paths if requested (like LSM)
            if self.train_ITM_only:
                itm_mask = payoffs_t > 0
                if np.sum(itm_mask) > 10:  # Need enough samples
                    train_features = features[itm_mask]
                    train_targets = target[itm_mask]
                else:
                    train_features = features
                    train_targets = target
            else:
                train_features = features
                train_targets = target

            # Train network to predict continuation value
            if self.use_torch and TORCH_AVAILABLE:
                net = self._train_network_torch(train_features, train_targets)
                self.networks[t] = net

                # Predict continuation values
                with torch.no_grad():
                    cont_values = net(torch.FloatTensor(features)).numpy().flatten()
            else:
                params = self._train_network_numpy(train_features, train_targets)
                self.networks[t] = params

                cont_values, _ = self._forward_numpy(features, params)

            # Ensure non-negative continuation values
            cont_values = np.maximum(0, cont_values)

            # Stopping decision: exercise if payoff >= continuation
            stop_now = payoffs_t >= cont_values
            stop_decisions[:, t] = stop_now

            # Update value function
            V[:, t] = np.where(stop_now, payoffs_t, cont_values)

        return V, stop_decisions

    # ==================== Pricing ====================

    def _price_with_networks(self, paths, payoffs_all):
        """
        Price option using trained networks.

        Use learned continuation values to make stopping decisions.
        """
        n_paths = paths.shape[0]
        T = self.nb_dates
        df = self.model.df

        # Track stopping
        stopped = np.zeros(n_paths, dtype=bool)
        path_values = np.zeros(n_paths, dtype=np.float32)

        for t in range(T):
            if np.all(stopped):
                break

            states = paths[:, :, t]
            payoffs_t = payoffs_all[:, t]
            features = self._build_features(states, payoffs_t)

            # Get continuation value estimate
            if self.use_torch and TORCH_AVAILABLE:
                with torch.no_grad():
                    net = self.networks[t]
                    cont_values = net(torch.FloatTensor(features)).numpy().flatten()
            else:
                cont_values, _ = self._forward_numpy(features, self.networks[t])

            cont_values = np.maximum(0, cont_values)

            # Exercise if payoff >= continuation and not already stopped
            exercise_now = (payoffs_t >= cont_values) & ~stopped

            # Discount to time 0
            discount = df ** t
            path_values[exercise_now] = payoffs_t[exercise_now] * discount
            stopped[exercise_now] = True

        # Paths that never stopped: exercise at maturity
        discount_T = df ** T
        path_values[~stopped] = payoffs_all[~stopped, T] * discount_T

        return np.mean(path_values)

    def price(self, train_eval_split=0.5):
        """
        Price American option using Neural LSM.

        Returns:
        --------
        price : float
            Estimated option value
        path_gen_time : float
            Time spent generating paths
        """
        # Generate training paths
        train_paths, path_gen_time_train = self._generate_paths_timed(self.nb_paths)
        train_payoffs = self.payoff(train_paths)

        # Train continuation value networks
        self._train_continuation_networks(train_paths, train_payoffs)

        # Generate evaluation paths
        eval_paths, path_gen_time_eval = self._generate_paths_timed(self.nb_paths)
        eval_payoffs = self.payoff(eval_paths)

        # Price using trained networks
        price = self._price_with_networks(eval_paths, eval_payoffs)

        total_path_gen_time = path_gen_time_train + path_gen_time_eval

        return float(price), float(total_path_gen_time)