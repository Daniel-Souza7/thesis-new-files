"""
Stochastic Mesh Method for American Option Pricing
Based on Broadie & Glasserman (2004)

Implementation of the classic mesh algorithm with:
- Stratified average density method (variance reduction)
- High-biased mesh estimator
- Low-biased path estimator
- Control variates (optional)
"""

import time
import numpy as np
import math


class StochasticMesh:
    """
    Stochastic Mesh Method (Broadie & Glasserman 2004).

    Generates a mesh of forward paths and prices American options using
    weighted backward induction. Provides both high-biased mesh estimate
    and low-biased path estimate for confidence intervals.

    Parameters:
    -----------
    model : StockModel
        The underlying asset model (must be BlackScholes for transition densities)
    payoff : Payoff
        The option payoff function
    nb_paths : int, optional
        Number of paths in mesh (denoted 'b' in paper, default: 500)
        Note: Mesh complexity is O(b²T), so use smaller values than Monte Carlo!
    nb_path_estimates : int, optional
        Number of independent paths for lower bound estimator (default: nb_paths)
    use_control_variates : bool, optional
        Whether to use European value as control variate (default: True)
    **kwargs : dict
        Ignored (for compatibility with other algorithms)
    """

    def __init__(self, model, payoff, nb_paths=500, nb_path_estimates=None,
                 use_control_variates=True, **kwargs):
        self.model = model
        self.payoff = payoff
        self.b = nb_paths  # Mesh size
        self.nb_path_estimates = nb_path_estimates or nb_paths
        self.use_control_variates = use_control_variates

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

        # Check model compatibility
        model_name = type(model).__name__
        if model_name not in ['BlackScholes', 'BlackScholesModel']:
            import warnings
            warnings.warn(
                f"Stochastic Mesh is designed for Black-Scholes models with known "
                f"transition densities. Your model is {model_name}. "
                f"Results may be inaccurate.",
                UserWarning
            )

    def _transition_density(self, S_from, S_to, asset_idx=0):
        """
        Compute transition density f(t, x, y) for geometric Brownian motion.

        For Black-Scholes: log(S_{t+1}/S_t) ~ N((r - σ²/2)Δt, σ²Δt)

        Parameters:
        -----------
        S_from : float or array
            Current state value(s)
        S_to : float or array
            Next state value(s)
        asset_idx : int
            Which asset (for multi-dimensional case)

        Returns:
        --------
        density : float or array
            Transition probability density
        """
        sigma = self.sigma[asset_idx] if self.nb_stocks > 1 else self.sigma[0]

        # Handle scalar or array inputs
        S_from = np.atleast_1d(S_from)
        S_to = np.atleast_1d(S_to)

        # Mean and variance of log(S_{t+1}/S_t)
        mean = (self.r - 0.5 * sigma**2) * self.dt
        var = sigma**2 * self.dt
        std = np.sqrt(var)

        # Lognormal density
        log_ratio = np.log(S_to / S_from)
        density = (1.0 / (S_to * std * np.sqrt(2 * np.pi))) * \
                  np.exp(-0.5 * ((log_ratio - mean) / std)**2)

        return density

    def _compute_mesh_weights(self, paths_prev, paths_curr, t):
        """
        Compute stratified average density weights.

        Weight from node i at time t-1 to node j at time t:
        w(i,j) = f(t-1, X_{t-1}(i), X_t(j)) / [(1/b) * sum_k f(t-1, X_{t-1}(k), X_t(j))]

        This is the stratified implementation of average density method.

        Parameters:
        -----------
        paths_prev : array, shape (b, d)
            Mesh points at time t-1
        paths_curr : array, shape (b, d)
            Mesh points at time t
        t : int
            Current time index

        Returns:
        --------
        weights : array, shape (b, b)
            weights[i,j] = weight from node i to node j
        """
        b = self.b
        weights = np.ones((b, b))

        # For multi-asset, compute product of marginal densities
        # (assumes independence under GBM, which holds for our case)
        for asset_idx in range(self.nb_stocks):
            asset_weights = np.zeros((b, b))

            for j in range(b):
                # Compute f(t-1, X_{t-1}(k), X_t(j)) for all k
                densities = np.array([
                    self._transition_density(
                        paths_prev[k, asset_idx],
                        paths_curr[j, asset_idx],
                        asset_idx
                    )
                    for k in range(b)
                ])

                # Average density: g(t, X_t(j)) = (1/b) * sum_k f(...)
                avg_density = np.mean(densities)

                # Stratified weights
                for i in range(b):
                    asset_weights[i, j] = densities[i] / (avg_density + 1e-10)

            # Multiply across assets (independence assumption)
            weights *= asset_weights

        return weights

    def _mesh_estimator(self, paths):
        """
        Compute high-biased mesh estimator using backward induction.

        Q(t, X_t(i)) = max(h(t, X_t(i)), (1/b) * sum_j Q(t+1, X_{t+1}(j)) * w(i,j))

        Parameters:
        -----------
        paths : array, shape (b, d, T+1)
            Forward-simulated mesh paths

        Returns:
        --------
        mesh_estimate : float
            High-biased estimate of option value
        Q_values : array, shape (b, T+1)
            Estimated continuation values at all mesh nodes (for path estimator)
        """
        b = self.b
        T = self.nb_dates

        # Compute all payoffs upfront
        payoffs = self.payoff(paths)  # Shape: (b, T+1)

        # Initialize value function at maturity
        Q = np.zeros((b, T + 1))
        Q[:, T] = payoffs[:, T]

        # Backward induction
        for t in range(T - 1, -1, -1):
            # Compute weights from time t to time t+1
            paths_t = paths[:, :, t]  # (b, d)
            paths_t1 = paths[:, :, t + 1]  # (b, d)

            weights = self._compute_mesh_weights(paths_t, paths_t1, t)

            # For each mesh node at time t
            for i in range(b):
                # Immediate exercise value
                exercise_value = payoffs[i, t]

                # Continuation value: weighted average over next time step
                continuation_value = np.sum(Q[:, t + 1] * weights[i, :]) / b

                # Optimal value
                Q[i, t] = max(exercise_value, continuation_value)

        # Mesh estimate is value at initial node
        return Q[0, 0], Q

    def _path_estimator(self, Q_values, mesh_paths):
        """
        Compute low-biased path estimator using mesh stopping rule.

        Simulates independent paths and stops at τ = min{t : h(t,S_t) >= Q(t,S_t)}

        Parameters:
        -----------
        Q_values : array, shape (b, T+1)
            Mesh continuation values
        mesh_paths : array, shape (b, d, T+1)
            Original mesh paths (for interpolation)

        Returns:
        --------
        path_estimate : float
            Low-biased estimate (average over multiple paths)
        """
        path_payoffs = []

        for _ in range(self.nb_path_estimates):
            # Generate independent path
            path, _ = self.model.generate_paths(nb_paths=1)
            # path shape: (1, d, T+1)

            # Compute payoffs for this path
            path_payoffs_all = self.payoff(path)  # Shape: (1, T+1)

            # Determine stopping time using mesh estimate
            for t in range(self.nb_dates):
                exercise_value = path_payoffs_all[0, t]

                # Estimate Q(t, S_t) by finding nearest mesh node
                # and using its continuation value
                S_t = path[0, :, t]  # Shape: (d,)
                mesh_states = mesh_paths[:, :, t]  # Shape: (b, d)

                # Find nearest neighbor in mesh (simple approach)
                distances = np.linalg.norm(mesh_states - S_t[np.newaxis, :], axis=1)
                nearest_idx = np.argmin(distances)
                estimated_Q = Q_values[nearest_idx, t]

                # Stop if exercise >= continuation
                if exercise_value >= estimated_Q:
                    path_payoffs.append(exercise_value)
                    break
            else:
                # Reached maturity without exercising
                path_payoffs.append(path_payoffs_all[0, self.nb_dates])

        return np.mean(path_payoffs)

    def _european_control(self, paths):
        """
        Compute European option value as control variate.

        European value = E[h(T, S_T)] = (1/b) * sum_i h(T, X_T(i))

        Parameters:
        -----------
        paths : array, shape (b, d, T+1)
            Forward-simulated paths

        Returns:
        --------
        european_value : float
            European option value (known/analytical if available)
        mesh_european : float
            Mesh estimate of European value
        """
        # Mesh estimate: average terminal payoffs
        payoffs = self.payoff(paths)  # Shape: (b, T+1)
        mesh_european = np.mean(payoffs[:, -1])

        # For simple payoffs, could compute analytical European value
        # For now, use Monte Carlo estimate as "known" value
        eur_paths, _ = self.model.generate_paths(nb_paths=10000)
        eur_payoffs = self.payoff(eur_paths)  # Shape: (10000, T+1)
        european_value = np.mean(eur_payoffs[:, -1])

        return european_value, mesh_european

    def price(self, train_eval_split=None):
        """
        Price American option using stochastic mesh method.

        Returns both high-biased mesh estimate and low-biased path estimate.

        Parameters:
        -----------
        train_eval_split : ignored (for compatibility)

        Returns:
        --------
        price : float
            Midpoint of [path_estimate, mesh_estimate] interval
        comp_time : float
            Computation time (path generation not included)
        """
        t_start = time.time()

        # Generate mesh: b independent forward paths
        mesh_paths, path_gen_time = self.model.generate_paths(nb_paths=self.b)
        # mesh_paths shape: (b, d, T+1)

        # Compute high-biased mesh estimator
        mesh_estimate, Q_values = self._mesh_estimator(mesh_paths)

        # Compute low-biased path estimator
        path_estimate = self._path_estimator(Q_values, mesh_paths)

        # Apply control variates if requested
        if self.use_control_variates:
            european_value, mesh_european = self._european_control(mesh_paths)
            control_adjustment = european_value - mesh_european
            mesh_estimate += control_adjustment

        # Return midpoint of confidence interval as price estimate
        price = (mesh_estimate + path_estimate) / 2.0

        comp_time = time.time() - t_start

        # Store estimates for diagnostics
        self.mesh_estimate = mesh_estimate
        self.path_estimate = path_estimate

        return price, comp_time
