"""
Stochastic Mesh Method for American Option Pricing - FIXED v2
Based on Broadie & Glasserman (2004)

FIXES:
1. Use log-space computation for multi-asset density weights (prevents overflow)
2. Proper row normalization of weights
3. Handle None timing from model.generate_paths()
"""

import time
import numpy as np
import math


class StochasticMesh:
    """
    Stochastic Mesh Method (Broadie & Glasserman 2004).
    """

    def __init__(self, model, payoff, nb_paths=500, nb_path_estimates=None,
                 use_control_variates=True, **kwargs):
        self.model = model
        self.payoff = payoff
        self.b = nb_paths
        self.nb_path_estimates = nb_path_estimates or min(nb_paths, 100)  # Limit for speed
        self.use_control_variates = use_control_variates

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

        model_name = type(model).__name__
        if model_name not in ['BlackScholes', 'BlackScholesModel']:
            import warnings
            warnings.warn(
                f"Stochastic Mesh is designed for Black-Scholes models. "
                f"Your model is {model_name}. Results may be inaccurate.",
                UserWarning
            )

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

    def _compute_mesh_weights(self, paths_prev, paths_curr, t):
        """
        Compute stratified average density weights using LOG-SPACE arithmetic.

        This prevents numerical overflow when multiplying densities across assets.

        Weight formula: w(i,j) = f(x_i, y_j) / [(1/b) * sum_k f(x_k, y_j)]

        For multi-asset with independent GBM:
        f(x, y) = prod_d f_d(x^d, y^d)
        log f(x, y) = sum_d log f_d(x^d, y^d)
        """
        b = paths_prev.shape[0]
        weights = np.zeros((b, b), dtype=np.float64)

        for j in range(b):
            # Compute log of joint density for all source nodes i to target node j
            log_densities = np.zeros(b, dtype=np.float64)

            for asset_idx in range(self.nb_stocks):
                sigma = self.sigma[asset_idx] if len(self.sigma) > 1 else self.sigma[0]

                mean = (self.r - 0.5 * sigma**2) * self.dt
                var = sigma**2 * self.dt
                std = np.sqrt(var)

                S_from = paths_prev[:, asset_idx]  # Shape: (b,)
                S_to = paths_curr[j, asset_idx]    # Scalar

                # Log of lognormal density: log f(S_from -> S_to)
                log_ratio = np.log(S_to / S_from)
                log_density = -np.log(S_to * std * np.sqrt(2 * np.pi)) \
                             - 0.5 * ((log_ratio - mean) / std)**2

                log_densities += log_density  # Sum in log space = product in linear space

            # Convert to linear space using log-sum-exp trick for stability
            max_log = np.max(log_densities)
            densities = np.exp(log_densities - max_log)

            # Stratified weight: f(x_i, y_j) / avg_k f(x_k, y_j)
            avg_density = np.mean(densities)
            if avg_density > 0:
                weights[:, j] = densities / avg_density
            else:
                weights[:, j] = 1.0  # Fallback to uniform

        # Clip to prevent extreme values (should be centered around 1.0 now)
        weights = np.clip(weights, 0.01, 100.0)

        return weights.astype(np.float32)

    def _mesh_estimator(self, paths):
        """Compute high-biased mesh estimator using backward induction."""
        b = self.b
        T = self.nb_dates

        payoffs = self.payoff(paths)  # Shape: (b, T+1)

        Q = np.zeros((b, T + 1), dtype=np.float64)
        Q[:, T] = payoffs[:, T]

        for t in range(T - 1, -1, -1):
            paths_t = paths[:, :, t]
            paths_t1 = paths[:, :, t + 1]

            weights = self._compute_mesh_weights(paths_t, paths_t1, t)

            for i in range(b):
                exercise_value = payoffs[i, t]

                # Continuation value: weighted average with discount
                # weights are normalized so we divide by b
                continuation_value = np.sum(Q[:, t + 1] * weights[i, :]) / b

                # Discount factor
                continuation_value *= self.model.df

                if not np.isfinite(continuation_value):
                    continuation_value = 0.0

                Q[i, t] = max(exercise_value, continuation_value)

        return Q[0, 0], Q

    def _path_estimator(self, Q_values, mesh_paths):
        """Compute low-biased path estimator using mesh stopping rule."""
        path_payoffs = []

        for _ in range(self.nb_path_estimates):
            path, _ = self._generate_paths_timed(nb_paths=1)
            path_payoffs_all = self.payoff(path)

            stopped = False
            for t in range(self.nb_dates):
                exercise_value = path_payoffs_all[0, t]

                S_t = path[0, :, t]
                mesh_states = mesh_paths[:, :, t]

                distances = np.linalg.norm(mesh_states - S_t[np.newaxis, :], axis=1)
                nearest_idx = np.argmin(distances)
                estimated_Q = Q_values[nearest_idx, t]

                if exercise_value >= estimated_Q and exercise_value > 0:
                    path_payoffs.append(exercise_value)
                    stopped = True
                    break

            if not stopped:
                path_payoffs.append(path_payoffs_all[0, self.nb_dates])

        return np.mean(path_payoffs) if path_payoffs else 0.0

    def _european_control(self, paths):
        """Compute European option value as control variate."""
        payoffs = self.payoff(paths)
        mesh_european = np.mean(payoffs[:, -1])

        eur_paths, _ = self._generate_paths_timed(nb_paths=10000)
        eur_payoffs = self.payoff(eur_paths)
        european_value = np.mean(eur_payoffs[:, -1])

        return european_value, mesh_european

    def price(self, train_eval_split=None):
        """Price American option using stochastic mesh method."""
        mesh_paths, path_gen_time = self._generate_paths_timed(nb_paths=self.b)

        mesh_estimate, Q_values = self._mesh_estimator(mesh_paths)
        path_estimate = self._path_estimator(Q_values, mesh_paths)

        if self.use_control_variates:
            european_value, mesh_european = self._european_control(mesh_paths)
            control_adjustment = european_value - mesh_european
            mesh_estimate += control_adjustment

        price = (mesh_estimate + path_estimate) / 2.0

        self.mesh_estimate = mesh_estimate
        self.path_estimate = path_estimate

        return float(price), float(path_gen_time)