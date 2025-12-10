"""
Randomized Stochastic Mesh Method - Version 2 - FIXED v2
Neural Network for Optimal Weight Learning

FIXES:
1. Use log-space computation for multi-asset density weights
2. Normalize input features for neural network stability
3. Handle None timing from model.generate_paths()
4. Proper weight normalization in learned weights
"""

import time
import numpy as np


class RandomizedStochasticMesh2:
    """
    RSM2: Use neural network to learn optimal mesh weighting function.
    """

    def __init__(self, model, payoff, hidden_size=20, nb_paths=500,
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

        if len(self.S0) == 1 and self.nb_stocks > 1:
            self.S0 = np.full(self.nb_stocks, self.S0[0])
        if len(self.sigma) == 1 and self.nb_stocks > 1:
            self.sigma = np.full(self.nb_stocks, self.sigma[0])

        # Feature normalization parameters
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
        """Initialize randomized neural network for weight function."""
        # Input: time + log(S_from/S0) + log(S_to/S0) + derived features
        base_dim = 1 + 2 * self.nb_stocks  # time + from + to (log-normalized)
        derived_dim = 2  # log_distance, payoff_diff_normalized
        input_dim = base_dim + derived_dim

        if self.use_payoff_as_input:
            input_dim += 2

        self.W_in = np.random.randn(input_dim, self.hidden_size) * 0.5
        self.b_in = np.random.randn(self.hidden_size) * 0.1
        self.W_out = None

    def _activation(self, x):
        """ReLU activation."""
        return np.maximum(0, x)

    def _build_weight_features(self, S_from, S_to, t, fit=False):
        """
        Build normalized input features for weight prediction network.
        """
        t_normalized = t / self.nb_dates

        # Use log-normalized prices for stability
        log_from = np.log(S_from / self.S0)
        log_to = np.log(S_to / self.S0)

        # Log distance (more stable than linear distance)
        log_distance = np.log(1 + np.linalg.norm(S_to - S_from) / np.mean(self.S0))

        # Payoff difference normalized by strike
        S_from_3d = S_from.reshape(1, -1, 1)
        S_to_3d = S_to.reshape(1, -1, 1)

        payoff_from = self.payoff(S_from_3d)[0, 0]
        payoff_to = self.payoff(S_to_3d)[0, 0]
        payoff_diff = (payoff_to - payoff_from) / self.payoff.strike

        features = np.concatenate([
            [t_normalized],
            log_from,
            log_to,
            [log_distance, payoff_diff],
        ])

        if self.use_payoff_as_input:
            norm_payoff_from = payoff_from / self.payoff.strike
            norm_payoff_to = payoff_to / self.payoff.strike
            features = np.concatenate([features, [norm_payoff_from, norm_payoff_to]])

        return features

    def _predict_weight(self, S_from, S_to, t):
        """Predict weight using neural network."""
        features = self._build_weight_features(S_from, S_to, t)

        if self.feature_mean is not None:
            features = (features - self.feature_mean) / self.feature_std

        hidden = self._activation(features @ self.W_in + self.b_in)

        if self.W_out is None:
            return 1.0

        # Output is log-weight for numerical stability
        log_weight = hidden @ self.W_out

        # Clip log-weight to prevent extreme values
        log_weight = np.clip(log_weight, -5, 5)
        weight = np.exp(log_weight)

        return weight

    def _compute_learned_weights(self, paths_prev, paths_curr, t):
        """Compute weights using learned neural network."""
        b = paths_prev.shape[0]
        weights = np.zeros((b, b), dtype=np.float64)

        for i in range(b):
            for j in range(b):
                weights[i, j] = self._predict_weight(
                    paths_prev[i],
                    paths_curr[j],
                    t
                )

            # Normalize each row to sum to b (so dividing by b gives probability)
            row_sum = np.sum(weights[i, :])
            if row_sum > 0:
                weights[i, :] = weights[i, :] * b / row_sum
            else:
                weights[i, :] = 1.0

        return weights.astype(np.float32)

    def _compute_analytical_weights(self, paths_prev, paths_curr, t):
        """
        Compute stratified average density weights using LOG-SPACE arithmetic.
        Used as training target.
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
        return weights

    def _train_network_on_european(self, num_training_meshes=10):
        """Train weight network using analytical weights as targets."""
        all_features = []
        all_log_weights = []

        for mesh_idx in range(num_training_meshes):
            paths, _ = self._generate_paths_timed(nb_paths=self.nb_paths)
            b = self.nb_paths
            T = self.nb_dates

            # Use analytical weights as training targets
            for t in range(T - 1, T):
                paths_t = paths[:, :, t]
                paths_t1 = paths[:, :, t + 1]

                # Get analytical weights
                analytical_weights = self._compute_analytical_weights(paths_t, paths_t1, t)

                for i in range(b):
                    for j in range(b):
                        features = self._build_weight_features(
                            paths_t[i],
                            paths_t1[j],
                            t
                        )
                        all_features.append(features)

                        # Target: log of analytical weight
                        target_log_weight = np.log(analytical_weights[i, j] + 1e-10)
                        all_log_weights.append(target_log_weight)

        X = np.array(all_features)
        y = np.array(all_log_weights)

        # Fit normalization
        self.feature_mean = np.mean(X, axis=0)
        self.feature_std = np.std(X, axis=0) + 1e-8
        X_norm = (X - self.feature_mean) / self.feature_std

        # Compute hidden layer
        H = self._activation(X_norm @ self.W_in + self.b_in)

        # Ridge regression for stability
        ridge_lambda = 0.1
        HtH = H.T @ H + ridge_lambda * np.eye(H.shape[1])
        Hty = H.T @ y
        self.W_out = np.linalg.solve(HtH, Hty)

    def price(self, train_eval_split=None):
        """Price American option using learned weight function."""
        # Train weight network
        if self.W_out is None:
            self._train_network_on_european(num_training_meshes=5)

        # Generate mesh for pricing
        paths, path_gen_time = self._generate_paths_timed(nb_paths=self.nb_paths)

        b = self.nb_paths
        T = self.nb_dates

        payoffs = self.payoff(paths)

        Q = np.zeros((b, T + 1), dtype=np.float64)
        Q[:, T] = payoffs[:, T]

        # Backward induction using learned weights
        for t in range(T - 1, -1, -1):
            paths_t = paths[:, :, t]
            paths_t1 = paths[:, :, t + 1]

            weights = self._compute_learned_weights(paths_t, paths_t1, t)

            for i in range(b):
                exercise_value = payoffs[i, t]

                # Weighted continuation value
                continuation_value = np.sum(Q[:, t + 1] * weights[i, :]) / b
                continuation_value *= self.model.df  # Discount

                if not np.isfinite(continuation_value):
                    continuation_value = 0.0

                Q[i, t] = max(exercise_value, continuation_value)

        # Average over all starting paths
        price = np.mean(Q[:, 0])

        return float(price), float(path_gen_time)