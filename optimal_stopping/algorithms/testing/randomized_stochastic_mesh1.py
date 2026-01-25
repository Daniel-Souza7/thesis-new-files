"""
Randomized Stochastic Mesh Method - Version 1 - FIXED v2
Neural Network for Continuation Value Approximation

FIXES:
1. Use log-space computation for multi-asset density weights
2. Normalize input features for neural network stability
3. Handle None timing from model.generate_paths()
"""

import time
import numpy as np


class RandomizedStochasticMesh1:
    """
    RSM1: Use neural network to approximate continuation values from mesh samples.
    """

    def __init__(self, model, payoff, hidden_size=20, nb_paths=1000,
                 use_payoff_as_input=False, **kwargs):
        self.model = model
        self.payoff = payoff
        # RSM1 uses fixed hidden_size=20 as default
        if hidden_size is None:
            hidden_size = 20
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

        if len(self.S0) == 1 and self.nb_stocks > 1:
            self.S0 = np.full(self.nb_stocks, self.S0[0])
        if len(self.sigma) == 1 and self.nb_stocks > 1:
            self.sigma = np.full(self.nb_stocks, self.sigma[0])

        # Feature normalization parameters (set during training)
        self.feature_mean = None
        self.feature_std = None

        self._initialize_network()

    def _generate_paths_timed(self, nb_paths):
        """Generate paths with explicit timing (handles None from model)."""
        start_time = time.time()
        result = self.model.generate_paths(nb_paths=nb_paths)
        elapsed = time.time() - start_time

        if isinstance(result, tuple):
            paths = result[0]
        else:
            paths = result

        return paths, elapsed

    def _initialize_network(self):
        """Initialize random weights for RNN (fixed, not trained)."""
        input_dim = self.nb_stocks + 1  # normalized prices + time
        if self.use_payoff_as_input:
            input_dim += 1

        # Random input-to-hidden weights (fixed) - use smaller initialization
        self.W_in = np.random.randn(input_dim, self.hidden_size) * 0.5
        self.b_in = np.random.randn(self.hidden_size) * 0.1
        self.W_out = None

    def _activation(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)

    def _normalize_features(self, states, time_idx, payoffs=None, fit=False):
        """
        Build and normalize input features for neural network.

        Args:
            states: (n, d) array of stock prices
            time_idx: time index
            payoffs: optional (n,) array of payoff values
            fit: if True, compute normalization parameters
        """
        n = states.shape[0]
        t_normalized = time_idx / self.nb_dates

        # Use log-prices for better scaling (prices are always positive)
        log_states = np.log(states / self.S0[np.newaxis, :])  # Normalize by initial price

        features = np.column_stack([
            log_states,  # Log-normalized prices
            np.full(n, t_normalized),
        ])

        if self.use_payoff_as_input and payoffs is not None:
            # Normalize payoffs by strike
            norm_payoffs = payoffs / self.payoff.strike
            features = np.column_stack([features, norm_payoffs])

        if fit:
            self.feature_mean = np.mean(features, axis=0)
            self.feature_std = np.std(features, axis=0) + 1e-8

        if self.feature_mean is not None:
            features = (features - self.feature_mean) / self.feature_std

        return features

    def _forward(self, states, time_idx, payoffs=None):
        """Forward pass through randomized neural network."""
        features = self._normalize_features(states, time_idx, payoffs, fit=False)

        # Hidden layer
        hidden = self._activation(features @ self.W_in + self.b_in)

        if self.W_out is None:
            return np.zeros(states.shape[0])

        predictions = hidden @ self.W_out

        # Ensure non-negative predictions (continuation values must be >= 0)
        predictions = np.maximum(0, predictions)

        return predictions

    def _compute_mesh_weights(self, paths_prev, paths_curr, t):
        """
        Compute stratified average density weights using LOG-SPACE arithmetic.
        Same as in StochasticMesh - prevents overflow for multi-asset.
        """
        b = paths_prev.shape[0]
        weights = np.zeros((b, b), dtype=np.float64)

        for j in range(b):
            log_densities = np.zeros(b, dtype=np.float64)

            for asset_idx in range(self.nb_stocks):
                sigma = self.sigma[asset_idx] if len(self.sigma) > 1 else self.sigma[0]

                mean = (self.r - 0.5 * sigma ** 2) * self.dt
                var = sigma ** 2 * self.dt
                std = np.sqrt(var)

                S_from = paths_prev[:, asset_idx]
                S_to = paths_curr[j, asset_idx]

                log_ratio = np.log(S_to / S_from)
                log_density = -np.log(S_to * std * np.sqrt(2 * np.pi)) \
                              - 0.5 * ((log_ratio - mean) / std) ** 2

                log_densities += log_density

            max_log = np.max(log_densities)
            densities = np.exp(log_densities - max_log)

            avg_density = np.mean(densities)
            if avg_density > 0:
                weights[:, j] = densities / avg_density
            else:
                weights[:, j] = 1.0

        weights = np.clip(weights, 0.01, 100.0)
        return weights.astype(np.float32)

    def _train_network(self, paths):
        """Train neural network using mesh-generated continuation values."""
        b = self.nb_paths
        T = self.nb_dates

        payoffs = self.payoff(paths)

        all_features = []
        all_targets = []

        Q = np.zeros((b, T + 1), dtype=np.float64)
        Q[:, T] = payoffs[:, T]

        # First pass: collect all features to fit normalization
        all_states = []
        all_times = []
        all_payoff_vals = []

        for t in range(T - 1, -1, -1):
            all_states.append(paths[:, :, t])
            all_times.extend([t] * b)
            if self.use_payoff_as_input:
                all_payoff_vals.extend(payoffs[:, t])

        # Fit normalization on all training data
        all_states_arr = np.vstack(all_states)
        dummy_features = self._normalize_features(
            all_states_arr,
            T // 2,  # Use middle time for fitting
            fit=True
        )

        # Second pass: backward induction with normalized features
        for t in range(T - 1, -1, -1):
            paths_t = paths[:, :, t]
            paths_t1 = paths[:, :, t + 1]

            weights = self._compute_mesh_weights(paths_t, paths_t1, t)

            for i in range(b):
                # Discounted continuation value from mesh
                continuation_target = np.sum(Q[:, t + 1] * weights[i, :]) / b
                continuation_target *= self.model.df  # Discount factor

                if not np.isfinite(continuation_target):
                    continuation_target = 0.0

                # Build normalized features
                payoff_val = payoffs[i, t] if self.use_payoff_as_input else None
                features = self._normalize_features(
                    paths_t[i:i + 1], t,
                    np.array([payoff_val]) if payoff_val is not None else None,
                    fit=False
                )

                all_features.append(features[0])
                all_targets.append(continuation_target)

                exercise_value = payoffs[i, t]
                Q[i, t] = max(exercise_value, continuation_target)

        X = np.array(all_features)
        y = np.array(all_targets)

        # Compute hidden layer activations
        H = self._activation(X @ self.W_in + self.b_in)

        # Ridge regression for stability
        ridge_lambda = 0.01
        HtH = H.T @ H + ridge_lambda * np.eye(H.shape[1])
        Hty = H.T @ y
        self.W_out = np.linalg.solve(HtH, Hty)

    def price(self, train_eval_split=None):
        """Price American option using randomized mesh with neural network."""
        # Generate training mesh
        train_paths, path_gen_time_train = self._generate_paths_timed(nb_paths=self.nb_paths)

        # Train neural network on mesh samples
        self._train_network(train_paths)

        # Price using neural network predictions
        eval_paths, path_gen_time_eval = self._generate_paths_timed(nb_paths=1000)

        b_eval = 1000
        T = self.nb_dates

        eval_payoffs = self.payoff(eval_paths)

        Q = np.zeros((b_eval, T + 1), dtype=np.float64)
        Q[:, T] = eval_payoffs[:, T]

        # Backward induction using NN predictions
        for t in range(T - 1, -1, -1):
            states = eval_paths[:, :, t]
            payoff_vals = eval_payoffs[:, t] if self.use_payoff_as_input else None

            # Predict continuation values using NN
            continuation_values = self._forward(states, t, payoff_vals)

            # Apply discount factor
            continuation_values *= self.model.df

            for i in range(b_eval):
                exercise_value = eval_payoffs[i, t]
                Q[i, t] = max(exercise_value, continuation_values[i])

        price = np.mean(Q[:, 0])
        total_path_gen_time = path_gen_time_train + path_gen_time_eval

        return float(price), float(total_path_gen_time)