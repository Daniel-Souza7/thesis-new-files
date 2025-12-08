"""
Randomized Stochastic Mesh Method - Version 1
Neural Network for Continuation Value Approximation

Uses stochastic mesh to generate training samples (state, continuation_value)
and trains a randomized neural network to approximate Q(t,x).

Similar to RLSM but with mesh-based targets instead of regression.
"""

import time
import numpy as np


class RandomizedStochasticMesh1:
    """
    RSM1: Use neural network to approximate continuation values from mesh samples.

    Training:
    - Generate mesh and compute weighted continuation values
    - Train RNN to map (t, state) -> continuation_value
    - Use NN predictions for pricing instead of mesh weights

    Parameters:
    -----------
    model : StockModel
        The underlying asset model
    payoff : Payoff
        The option payoff function
    hidden_size : int, optional
        Number of hidden units in randomized neural network (default: 20)
    nb_paths : int, optional
        Number of paths for training mesh (default: 1000)
        Note: Mesh complexity is O(b²T), use smaller values than Monte Carlo!
    use_payoff_as_input : bool, optional
        Include current payoff in NN input (default: False)
    **kwargs : dict
        Additional arguments (ignored)
    """

    def __init__(self, model, payoff, hidden_size=20, nb_paths=1000,
                 use_payoff_as_input=False, **kwargs):
        self.model = model
        self.payoff = payoff
        self.hidden_size = hidden_size
        self.nb_paths = nb_paths
        self.use_payoff_as_input = use_payoff_as_input

        # Extract model parameters
        self.r = model.rate
        self.dt = model.dt
        self.T = model.maturity
        self.nb_dates = model.nb_dates
        self.nb_stocks = model.nb_stocks
        self.sigma = np.array(model.volatility).flatten()
        self.S0 = np.array(model.spot).flatten()

        # Replicate scalar to all assets if needed
        if len(self.S0) == 1 and self.nb_stocks > 1:
            self.S0 = np.full(self.nb_stocks, self.S0[0])
        if len(self.sigma) == 1 and self.nb_stocks > 1:
            self.sigma = np.full(self.nb_stocks, self.sigma[0])

        # Randomized neural network weights (initialized once)
        self._initialize_network()

    def _initialize_network(self):
        """Initialize random weights for RNN (fixed, not trained)."""
        # Input dimension: d assets + 1 (time) + (optionally) 1 (current payoff)
        input_dim = self.nb_stocks + 1
        if self.use_payoff_as_input:
            input_dim += 1

        # Random input-to-hidden weights (fixed)
        self.W_in = np.random.randn(input_dim, self.hidden_size) / np.sqrt(input_dim)
        self.b_in = np.random.randn(self.hidden_size)

        # Output weights (trained via least squares)
        self.W_out = None  # Computed during training

    def _activation(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)

    def _forward(self, states, time_idx):
        """
        Forward pass through randomized neural network.

        Parameters:
        -----------
        states : array, shape (n, d)
            State variables
        time_idx : int
            Time index (normalized to [0, 1])

        Returns:
        --------
        predictions : array, shape (n,)
            Predicted continuation values
        """
        n = states.shape[0]

        # Normalize time to [0, 1]
        t_normalized = time_idx / self.nb_dates

        # Build input features
        features = np.column_stack([
            states,  # Asset prices
            np.full(n, t_normalized),  # Time
        ])

        if self.use_payoff_as_input:
            payoffs = np.array([self.payoff(states[i]) for i in range(n)])
            features = np.column_stack([features, payoffs])

        # Hidden layer: h = activation(W_in @ x + b_in)
        hidden = self._activation(features @ self.W_in + self.b_in)

        # Output layer: y = W_out @ h
        if self.W_out is None:
            return np.zeros(n)  # Not trained yet

        predictions = hidden @ self.W_out

        return predictions

    def _transition_density(self, S_from, S_to, asset_idx=0):
        """
        Compute transition density for geometric Brownian motion.
        Same as in StochasticMesh.
        """
        sigma = self.sigma[asset_idx] if self.nb_stocks > 1 else self.sigma[0]

        S_from = np.atleast_1d(S_from)
        S_to = np.atleast_1d(S_to)

        mean = (self.r - 0.5 * sigma**2) * self.dt
        var = sigma**2 * self.dt
        std = np.sqrt(var)

        log_ratio = np.log(S_to / S_from)
        density = (1.0 / (S_to * std * np.sqrt(2 * np.pi))) * \
                  np.exp(-0.5 * ((log_ratio - mean) / std)**2)

        return density

    def _compute_mesh_weights(self, paths_prev, paths_curr, t):
        """Compute stratified average density weights (same as SM)."""
        b = self.nb_paths
        weights = np.ones((b, b), dtype=np.float32)

        for asset_idx in range(self.nb_stocks):
            asset_weights = np.zeros((b, b), dtype=np.float32)

            for j in range(b):
                densities = np.array([
                    self._transition_density(
                        paths_prev[k, asset_idx],
                        paths_curr[j, asset_idx],
                        asset_idx
                    )
                    for k in range(b)
                ])

                avg_density = np.mean(densities)

                for i in range(b):
                    asset_weights[i, j] = densities[i] / (avg_density + 1e-10)

            # Clip individual asset weights to prevent extreme values
            asset_weights = np.clip(asset_weights, 1e-6, 100)

            weights *= asset_weights

            # CRITICAL: Clip after each multiplication to prevent overflow
            weights = np.clip(weights, 1e-6, 1e6)

        return weights

    def _train_network(self, paths):
        """
        Train neural network using mesh-generated continuation values.

        For each mesh node at each time, compute target continuation value
        using mesh weights, then train NN via least squares.

        Parameters:
        -----------
        paths : array, shape (b, d, T+1)
            Forward-simulated mesh paths
        """
        b = self.nb_paths
        T = self.nb_dates

        # Compute all payoffs upfront
        payoffs = self.payoff(paths)  # Shape: (b, T+1)

        # Collect training samples across all times and nodes
        all_features = []
        all_targets = []

        # Initialize value function at maturity (use float32 for memory efficiency)
        Q = np.zeros((b, T + 1), dtype=np.float32)
        Q[:, T] = payoffs[:, T]

        # Backward induction to generate training data
        for t in range(T - 1, -1, -1):
            paths_t = paths[:, :, t]
            paths_t1 = paths[:, :, t + 1]

            weights = self._compute_mesh_weights(paths_t, paths_t1, t)

            # For each mesh node, compute target continuation value
            for i in range(b):
                # Target: weighted continuation value from mesh
                continuation_target = np.sum(Q[:, t + 1] * weights[i, :]) / b

                # Handle NaN/inf from numerical issues
                if not np.isfinite(continuation_target):
                    continuation_target = 0.0

                # Store (state, time) -> continuation_value for training
                t_normalized = t / self.nb_dates
                features = np.concatenate([
                    paths_t[i],  # State
                    [t_normalized],  # Time
                ])

                if self.use_payoff_as_input:
                    features = np.concatenate([features, [payoffs[i, t]]])

                all_features.append(features)
                all_targets.append(continuation_target)

                # Update Q for this node
                exercise_value = payoffs[i, t]
                Q[i, t] = max(exercise_value, continuation_target)

        # Convert to arrays
        X = np.array(all_features)  # (b*T, input_dim)
        y = np.array(all_targets)  # (b*T,)

        # Compute hidden layer activations
        H = self._activation(X @ self.W_in + self.b_in)  # (b*T, hidden_size)

        # Train output weights via least squares: W_out = (H^T H)^{-1} H^T y
        self.W_out = np.linalg.lstsq(H, y, rcond=None)[0]

    def price(self, train_eval_split=None):
        """
        Price American option using randomized mesh with neural network.

        Returns:
        --------
        price : float
            Estimated option value
        path_gen_time : float
            Time spent generating paths (for run_algo.py to compute comp_time)
        """
        # Generate training mesh
        train_paths, path_gen_time_train = self.model.generate_paths(nb_paths=self.nb_paths)

        # Train neural network on mesh samples
        self._train_network(train_paths)

        # Price using neural network predictions (backward induction)
        eval_paths, path_gen_time_eval = self.model.generate_paths(nb_paths=1000)
        b_eval = 1000
        T = self.nb_dates

        # Compute all payoffs upfront
        eval_payoffs = self.payoff(eval_paths)  # Shape: (b_eval, T+1)

        # Initialize at maturity (use float32 for memory efficiency)
        Q = np.zeros((b_eval, T + 1), dtype=np.float32)
        Q[:, T] = eval_payoffs[:, T]

        # Backward induction using NN predictions
        for t in range(T - 1, -1, -1):
            states = eval_paths[:, :, t]

            # Predict continuation values using NN
            continuation_values = self._forward(states, t)

            # Optimal decision
            for i in range(b_eval):
                exercise_value = eval_payoffs[i, t]
                Q[i, t] = max(exercise_value, continuation_values[i])

        # Return average over all paths starting from S0
        price = np.mean(Q[:, 0])

        # Return total path generation time
        total_path_gen_time = path_gen_time_train + path_gen_time_eval

        # Handle case where path generation failed
        if total_path_gen_time is None or not np.isfinite(price):
            return 0.0, 0.0

        return price, total_path_gen_time
