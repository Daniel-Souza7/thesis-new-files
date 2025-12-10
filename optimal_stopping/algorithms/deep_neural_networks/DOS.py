"""
Deep Optimal Stopping (DOS) for American Option Pricing
Based on Becker, Cheridito, Jentzen, and Welti (2020)
"Pricing and hedging american-style options with deep learning"
Journal of Risk and Financial Management, 13(7):158

Key idea:
- Train N neural networks (one per exercise date) to output stopping probability
- Use sigmoid output for differentiability (soft stopping decision)
- Backward iteration: at time n, maximize expected value given future networks are fixed
- Final decision uses hard threshold (stop if F_θ(S) >= 0.5)

This is a GLOBAL optimization approach (vs local regression in LSM).
"""

import time
import numpy as np

# Optional: Try to use PyTorch for GPU acceleration, fall back to NumPy
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class DeepOptimalStopping:
    """
    Deep Optimal Stopping (Becker et al. 2020).

    Trains neural networks to learn optimal stopping decisions directly
    by maximizing the expected discounted payoff.

    Parameters:
    -----------
    model : StockModel
        The underlying asset model
    payoff : Payoff
        The option payoff function
    hidden_size : int, optional
        Number of hidden units per layer (default: 40)
    nb_paths : int, optional
        Number of paths for training (default: 8192)
    nb_epochs : int, optional
        Number of training epochs per time step (default: 30)
    learning_rate : float, optional
        Learning rate for Adam optimizer (default: 0.001)
    use_payoff_as_input : bool, optional
        Include current payoff in network input (default: True)
    **kwargs : dict
        Additional arguments (ignored)
    """

    def __init__(self, model, payoff, hidden_size=40, nb_paths=8192,
                 nb_epochs=30, learning_rate=0.001, use_payoff_as_input=True,
                 **kwargs):
        self.model = model
        self.payoff = payoff
        self.hidden_size = hidden_size
        self.nb_paths = nb_paths
        self.nb_epochs = nb_epochs
        self.learning_rate = learning_rate
        self.use_payoff_as_input = use_payoff_as_input

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

        # Networks for each time step (initialized during training)
        self.networks = {}  # Dict of {time_idx: network_params}

        # Use PyTorch if available, otherwise NumPy implementation
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
        # Log-normalize by initial price
        return np.log(states / self.S0[np.newaxis, :])

    def _build_features(self, states, payoffs=None):
        """Build input features for neural network."""
        features = self._normalize_state(states)

        if self.use_payoff_as_input and payoffs is not None:
            # Normalize payoffs by strike
            norm_payoffs = payoffs[:, np.newaxis] / self.payoff.strike
            features = np.concatenate([features, norm_payoffs], axis=1)

        return features.astype(np.float32)

    # ==================== NumPy Implementation ====================

    def _init_network_numpy(self, time_idx):
        """Initialize network weights for one time step (NumPy)."""
        # Two hidden layers with tanh activation, sigmoid output
        h = self.hidden_size
        d = self.input_dim

        # Xavier initialization
        W1 = np.random.randn(d, h).astype(np.float32) * np.sqrt(2.0 / d)
        b1 = np.zeros(h, dtype=np.float32)
        W2 = np.random.randn(h, h).astype(np.float32) * np.sqrt(2.0 / h)
        b2 = np.zeros(h, dtype=np.float32)
        W3 = np.random.randn(h, 1).astype(np.float32) * np.sqrt(2.0 / h)
        b3 = np.zeros(1, dtype=np.float32)

        return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}

    def _forward_numpy(self, features, params):
        """Forward pass through network (NumPy)."""
        # Layer 1
        z1 = features @ params['W1'] + params['b1']
        a1 = np.tanh(z1)

        # Layer 2
        z2 = a1 @ params['W2'] + params['b2']
        a2 = np.tanh(z2)

        # Output layer (sigmoid for probability)
        z3 = a2 @ params['W3'] + params['b3']
        prob = 1.0 / (1.0 + np.exp(-z3.flatten()))  # Sigmoid

        return prob, {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2, 'z3': z3}

    def _backward_numpy(self, features, prob, cache, params, grad_output):
        """Backward pass for gradient computation (NumPy)."""
        n = features.shape[0]

        # Sigmoid derivative: σ'(z) = σ(z)(1-σ(z))
        d_z3 = grad_output * prob * (1 - prob)
        d_z3 = d_z3[:, np.newaxis]  # Shape (n, 1)

        d_W3 = cache['a2'].T @ d_z3 / n
        d_b3 = np.mean(d_z3, axis=0)

        # Tanh derivative: 1 - tanh²(z)
        d_a2 = d_z3 @ params['W3'].T
        d_z2 = d_a2 * (1 - cache['a2'] ** 2)

        d_W2 = cache['a1'].T @ d_z2 / n
        d_b2 = np.mean(d_z2, axis=0)

        d_a1 = d_z2 @ params['W2'].T
        d_z1 = d_a1 * (1 - cache['a1'] ** 2)

        d_W1 = features.T @ d_z1 / n
        d_b1 = np.mean(d_z1, axis=0)

        return {'W1': d_W1, 'b1': d_b1, 'W2': d_W2, 'b2': d_b2, 'W3': d_W3, 'b3': d_b3}

    def _update_params_numpy(self, params, grads, lr):
        """Update parameters with gradient descent."""
        for key in params:
            params[key] += lr * grads[key]  # Gradient ASCENT (maximizing)
        return params

    # ==================== PyTorch Implementation ====================

    def _create_network_torch(self):
        """Create PyTorch network for one time step."""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )

    # ==================== Training ====================

    def _train_networks(self, paths, payoffs_all):
        """
        Train stopping networks using backward iteration.

        At each time n (going backward), we:
        1. Use already-trained networks for times n+1, ..., N to compute
           continuation values
        2. Train network at time n to maximize:
           E[g(n,S_n) * F_θ(S_n) + V_{n+1} * (1 - F_θ(S_n))]
        """
        n_paths = paths.shape[0]
        T = self.nb_dates

        # Discount factors for each future time
        df = self.model.df  # e^{-r*dt}

        # Initialize: at maturity, always exercise
        # Value at maturity for each path
        V = np.zeros((n_paths, T + 1), dtype=np.float32)
        V[:, T] = payoffs_all[:, T]

        # Store stopping probabilities for debugging
        stop_probs = np.zeros((n_paths, T), dtype=np.float32)

        # Backward iteration
        for t in range(T - 1, -1, -1):
            # Current state features
            states = paths[:, :, t]
            payoffs_t = payoffs_all[:, t]
            features = self._build_features(states, payoffs_t)

            # Exercise value at time t
            exercise_value = payoffs_t

            # Continuation value: discounted expected future value
            # (computed from already-trained networks)
            continuation_value = df * V[:, t + 1]

            # Initialize network for this time step
            if self.use_torch and TORCH_AVAILABLE:
                self._train_network_torch(t, features, exercise_value, continuation_value)
            else:
                self._train_network_numpy(t, features, exercise_value, continuation_value)

            # Compute optimal value using trained network
            if self.use_torch and TORCH_AVAILABLE:
                with torch.no_grad():
                    net = self.networks[t]
                    features_t = torch.FloatTensor(features)
                    prob = net(features_t).numpy().flatten()
            else:
                prob, _ = self._forward_numpy(features, self.networks[t])

            stop_probs[:, t] = prob

            # Value at time t using soft stopping
            # V_t = F(S_t) * g(t,S_t) + (1-F(S_t)) * E[V_{t+1}]
            V[:, t] = prob * exercise_value + (1 - prob) * continuation_value

        return V, stop_probs

    def _train_network_numpy(self, t, features, exercise_value, continuation_value):
        """Train network at time t using NumPy (gradient ascent)."""
        params = self._init_network_numpy(t)

        # Adam optimizer state
        m = {k: np.zeros_like(v) for k, v in params.items()}
        v = {k: np.zeros_like(v) for k, v in params.items()}
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for epoch in range(self.nb_epochs):
            # Forward pass
            prob, cache = self._forward_numpy(features, params)

            # Expected value: F*g + (1-F)*C
            expected_value = prob * exercise_value + (1 - prob) * continuation_value

            # Gradient of expected value w.r.t. prob:
            # d/dF [F*g + (1-F)*C] = g - C
            grad_prob = exercise_value - continuation_value

            # Backward pass
            grads = self._backward_numpy(features, prob, cache, params, grad_prob)

            # Adam update (gradient ASCENT)
            for k in params:
                m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
                v[k] = beta2 * v[k] + (1 - beta2) * (grads[k] ** 2)
                m_hat = m[k] / (1 - beta1 ** (epoch + 1))
                v_hat = v[k] / (1 - beta2 ** (epoch + 1))
                params[k] += self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        self.networks[t] = params

    def _train_network_torch(self, t, features, exercise_value, continuation_value):
        """Train network at time t using PyTorch."""
        net = self._create_network_torch()
        optimizer = optim.Adam(net.parameters(), lr=self.learning_rate)

        features_t = torch.FloatTensor(features)
        exercise_t = torch.FloatTensor(exercise_value)
        continuation_t = torch.FloatTensor(continuation_value)

        for epoch in range(self.nb_epochs):
            optimizer.zero_grad()

            prob = net(features_t).squeeze()

            # Maximize: F*g + (1-F)*C
            expected_value = prob * exercise_t + (1 - prob) * continuation_t
            loss = -expected_value.mean()  # Negative for maximization

            loss.backward()
            optimizer.step()

        self.networks[t] = net

    # ==================== Pricing ====================

    def _price_with_networks(self, paths, payoffs_all):
        """
        Price option using trained networks with HARD stopping decision.

        Stop at first time t where F_θ(S_t) >= 0.5 (or at maturity).
        """
        n_paths = paths.shape[0]
        T = self.nb_dates
        df = self.model.df

        # Track stopping time and payoff for each path
        stopped = np.zeros(n_paths, dtype=bool)
        path_values = np.zeros(n_paths, dtype=np.float32)

        for t in range(T):
            if np.all(stopped):
                break

            states = paths[:, :, t]
            payoffs_t = payoffs_all[:, t]
            features = self._build_features(states, payoffs_t)

            # Get stopping probability
            if self.use_torch and TORCH_AVAILABLE:
                with torch.no_grad():
                    net = self.networks[t]
                    prob = net(torch.FloatTensor(features)).numpy().flatten()
            else:
                prob, _ = self._forward_numpy(features, self.networks[t])

            # Hard stopping: stop if prob >= 0.5 and not already stopped
            stop_now = (prob >= 0.5) & ~stopped

            # Discount factor from time 0 to time t
            discount = df ** t

            # Record discounted payoff for paths that stop now
            path_values[stop_now] = payoffs_t[stop_now] * discount
            stopped[stop_now] = True

        # Paths that never stopped: exercise at maturity
        discount_T = df ** T
        path_values[~stopped] = payoffs_all[~stopped, T] * discount_T

        return np.mean(path_values)

    def price(self, train_eval_split=0.5):
        """
        Price American option using Deep Optimal Stopping.

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

        # Train networks
        self._train_networks(train_paths, train_payoffs)

        # Generate evaluation paths (independent)
        eval_paths, path_gen_time_eval = self._generate_paths_timed(self.nb_paths)
        eval_payoffs = self.payoff(eval_paths)

        # Price using trained networks
        price = self._price_with_networks(eval_paths, eval_payoffs)

        total_path_gen_time = path_gen_time_train + path_gen_time_eval

        return float(price), float(total_path_gen_time)