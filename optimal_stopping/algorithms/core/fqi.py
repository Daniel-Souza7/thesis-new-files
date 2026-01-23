"""
Fitted Q-Iteration (FQI) for American option pricing.

Simple benchmark implementation of the FQI algorithm from:
"Regression methods for pricing complex American-style options"
(Tsitsiklis and Van Roy, 2001)
"""

import numpy as np
import math
import time
from optimal_stopping.run import configs
from optimal_stopping.algorithms.utils import basis_functions


class FQIFast:
    """
    Computes American option price using Fitted Q-Iteration (FQI).

    FQI learns the Q-function (value of exercising or continuing) using
    regression with polynomial basis functions.
    """

    def __init__(self, model, payoff, nb_epochs=20, hidden_size=None,
                 factors=None, train_ITM_only=True, use_payoff_as_input=False,
                 use_barrier_as_input=False):
        """
        Initialize FQI pricer.

        Args:
            model: Stock model
            payoff: Payoff function
            nb_epochs: Number of iterations for Q-learning
            hidden_size: Ignored (for API compatibility)
            factors: Ignored (for API compatibility)
            train_ITM_only: If True, only train on in-the-money paths
            use_payoff_as_input: If True, include payoff as feature
            use_barrier_as_input: If True, include barrier values as input hint
        """
        self.model = model
        self.payoff = payoff
        self.nb_epochs = nb_epochs
        self.train_ITM_only = train_ITM_only
        self.use_payoff_as_input = use_payoff_as_input
        self.use_barrier_as_input = use_barrier_as_input

        # Check for variance paths
        self.use_var = getattr(model, 'return_var', False)

        # Extract barrier values from payoff if available (for use as input hint)
        self.barrier_values = []
        if self.use_barrier_as_input:
            if hasattr(payoff, 'barrier') and payoff.barrier is not None:
                self.barrier_values.append(payoff.barrier)
            if hasattr(payoff, 'barrier_up') and payoff.barrier_up is not None:
                self.barrier_values.append(payoff.barrier_up)
            if hasattr(payoff, 'barrier_down') and payoff.barrier_down is not None:
                self.barrier_values.append(payoff.barrier_down)

        self.nb_barriers = len(self.barrier_values)

        # Initialize basis functions (degree 2 polynomials + time features)
        state_size = model.nb_stocks * (1 + self.use_var) + 2 + self.use_payoff_as_input * 1 + self.nb_barriers
        self.basis = basis_functions.BasisFunctions(state_size)
        self.nb_base_fcts = self.basis.nb_base_fcts

        # Q-function weights
        self.weights = np.zeros(self.nb_base_fcts)

    def price(self, train_eval_split=2):
        """
        Compute option price using FQI.

        Args:
            train_eval_split: Ratio for splitting paths into training/evaluation

        Returns:
            tuple: (price, time_for_path_generation)
        """
        t_start = time.time()

        # Generate paths
        if configs.path_gen_seed.get_seed() is not None:
            np.random.seed(configs.path_gen_seed.get_seed())

        path_result = self.model.generate_paths()
        if isinstance(path_result, tuple):
            stock_paths, var_paths = path_result
        else:
            stock_paths = path_result
            var_paths = None

        time_path_gen = time.time() - t_start
        print(f"time path gen: {time_path_gen:.4f} ", end="")

        # Compute payoffs for all paths at all times
        payoffs = self.payoff(stock_paths)

        # Split into training and evaluation sets
        self.split = len(stock_paths) // train_eval_split

        nb_paths, nb_stocks, nb_dates_from_shape = stock_paths.shape
        # Use model's nb_dates (actual time steps) for consistency across all algorithms
        disc_factor = math.exp(-self.model.rate * self.model.maturity / self.model.nb_dates)

        # Prepare state with time features
        if self.use_payoff_as_input:
            paths = np.concatenate([stock_paths, np.expand_dims(payoffs, axis=1)], axis=1)
        else:
            paths = stock_paths

        if self.use_var and var_paths is not None:
            paths = np.concatenate([paths, var_paths], axis=1)

        # Evaluate basis functions for all paths and times
        eval_bases = self._evaluate_bases_all(paths, nb_dates_from_shape)

        # FQI iterations
        for epoch in range(self.nb_epochs):
            # Predict Q-values for all states at times 1 to T-1
            q_values = self._predict(eval_bases[:self.split, 1:, :])

            # Clip to non-negative (American option value can't be negative)
            q_values = np.maximum(0, q_values)

            # Target: max(payoff_t+1, Q_t+1) discounted
            indicator_stop = np.maximum(payoffs[:self.split, 1:], q_values)

            # Filter to ITM paths if train_ITM_only is enabled (reference methodology)
            if self.train_ITM_only:
                # Create mask for ITM (path, time) pairs: payoff > 0
                itm_mask = payoffs[:self.split, :-1] > 0

                # Apply mask to training data
                train_bases = eval_bases[:self.split, :-1, :] * itm_mask[:, :, np.newaxis]
                train_targets = indicator_stop * itm_mask
            else:
                train_bases = eval_bases[:self.split, :-1, :]
                train_targets = indicator_stop

            # Compute regression matrices
            matrixU = np.tensordot(
                train_bases,
                train_bases,
                axes=([0, 1], [0, 1])
            )

            vectorV = np.sum(
                train_bases * disc_factor * np.repeat(
                    np.expand_dims(train_targets, axis=2),
                    eval_bases.shape[2],
                    axis=2
                ),
                axis=(0, 1)
            )

            # Update weights
            self._fit(matrixU, vectorV)

        # Compute final Q-values
        continuation_value = self._predict(eval_bases)

        # Clip to non-negative (American option value can't be negative)
        continuation_value = np.maximum(0, continuation_value)

        # Determine exercise strategy
        exercise = (payoffs > continuation_value).astype(int)
        exercise[:, -1] = 1  # Must exercise at maturity
        exercise[:, 0] = 0   # Cannot exercise at t=0

        # Find exercise dates
        ex_dates = np.argmax(exercise, axis=1)

        # Track exercise dates for statistics
        self._exercise_dates = ex_dates.copy()

        # Compute prices
        prices = np.take_along_axis(
            payoffs, np.expand_dims(ex_dates, axis=1), axis=1
        ).reshape(-1) * disc_factor ** ex_dates

        price = max(np.mean(prices[self.split:]), payoffs[0, 0])

        return price, time_path_gen

    def price_upper_lower_bound(self, train_eval_split=2):
        """
        Compute both lower and upper bounds using FQI.

        Lower bound: Regular FQI pricing
        Upper bound: Dual formulation

        Args:
            train_eval_split: Ratio for splitting paths

        Returns:
            tuple: (lower_bound, upper_bound, time_for_path_generation)
        """
        t_start = time.time()

        # Generate paths
        if configs.path_gen_seed.get_seed() is not None:
            np.random.seed(configs.path_gen_seed.get_seed())

        path_result = self.model.generate_paths()
        if isinstance(path_result, tuple):
            stock_paths, var_paths = path_result
        else:
            stock_paths = path_result
            var_paths = None

        time_path_gen = time.time() - t_start
        print(f"time path gen: {time_path_gen:.4f} ", end="")

        # Compute payoffs
        payoffs = self.payoff(stock_paths)

        # Split
        self.split = len(stock_paths) // train_eval_split

        nb_paths, nb_stocks, nb_dates_from_shape = stock_paths.shape
        disc_factor = math.exp(-self.model.rate * self.model.maturity / self.model.nb_dates)

        # Prepare state
        if self.use_payoff_as_input:
            paths = np.concatenate([stock_paths, np.expand_dims(payoffs, axis=1)], axis=1)
        else:
            paths = stock_paths

        if self.use_var and var_paths is not None:
            paths = np.concatenate([paths, var_paths], axis=1)

        # Evaluate basis functions
        eval_bases = self._evaluate_bases_all(paths, nb_dates_from_shape)

        # Initialize exercise dates tracking (will be updated after FQI)
        self._exercise_dates = None

        # FQI iterations
        for epoch in range(self.nb_epochs):
            q_values = self._predict(eval_bases[:self.split, 1:, :])
            q_values = np.maximum(0, q_values)
            indicator_stop = np.maximum(payoffs[:self.split, 1:], q_values)

            # Filter to ITM paths if train_ITM_only is enabled (reference methodology)
            if self.train_ITM_only:
                # Create mask for ITM (path, time) pairs: payoff > 0
                itm_mask = payoffs[:self.split, :-1] > 0

                # Apply mask to training data
                train_bases = eval_bases[:self.split, :-1, :] * itm_mask[:, :, np.newaxis]
                train_targets = indicator_stop * itm_mask
            else:
                train_bases = eval_bases[:self.split, :-1, :]
                train_targets = indicator_stop

            matrixU = np.tensordot(
                train_bases,
                train_bases,
                axes=([0, 1], [0, 1])
            )

            vectorV = np.sum(
                train_bases * disc_factor * np.repeat(
                    np.expand_dims(train_targets, axis=2),
                    eval_bases.shape[2],
                    axis=2
                ),
                axis=(0, 1)
            )

            self._fit(matrixU, vectorV)

        # Compute continuation values
        continuation_value = self._predict(eval_bases)
        continuation_value = np.maximum(0, continuation_value)

        # Lower bound
        exercise = (payoffs > continuation_value).astype(int)
        exercise[:, -1] = 1
        exercise[:, 0] = 0
        ex_dates = np.argmax(exercise, axis=1)

        # Track exercise dates for get_exercise_time()
        self._exercise_dates = ex_dates

        prices = np.take_along_axis(
            payoffs, np.expand_dims(ex_dates, axis=1), axis=1
        ).reshape(-1) * disc_factor ** ex_dates

        lower_bound = max(np.mean(prices[self.split:]), payoffs[0, 0])

        # Upper bound: Construct martingale M
        M = np.maximum(payoffs, continuation_value)

        # Compute upper bound on evaluation set using dual formulation
        # The martingale M satisfies M[t] >= payoff[t] for all t
        # Upper bound = E[M[0]] where M is constructed from Q-function
        upper_bound = np.mean(M[self.split:, 0])

        return lower_bound, upper_bound, time_path_gen

    def _evaluate_bases_all(self, stock_paths, nb_dates):
        """
        Evaluate basis functions for all paths and times.

        Args:
            stock_paths: (nb_paths, nb_stocks, nb_dates)
            nb_dates: Number of time steps

        Returns:
            basis_values: (nb_paths, nb_dates, nb_base_fcts)
        """
        nb_paths = stock_paths.shape[0]

        # Add time features: normalized time t/T and (1-t/T)
        time_grid = np.linspace(0, 1, nb_dates)
        time_features = np.stack([time_grid, 1 - time_grid], axis=0)  # (2, nb_dates)
        time_features = np.tile(time_features[np.newaxis, :, :], (nb_paths, 1, 1))  # (nb_paths, 2, nb_dates)

        # Concatenate stock prices and time features
        paths_with_time = np.concatenate([stock_paths, time_features], axis=1)  # (nb_paths, nb_stocks+2, nb_dates)

        # Add barrier values as input hint (if enabled)
        if self.nb_barriers > 0:
            spot = self.model.spot if hasattr(self.model, 'spot') else 100.0
            barrier_features = np.array([[[b / spot] for b in self.barrier_values]])  # (1, nb_barriers, 1)
            barrier_features = np.tile(barrier_features, (nb_paths, 1, nb_dates))  # (nb_paths, nb_barriers, nb_dates)
            paths_with_time = np.concatenate([paths_with_time, barrier_features], axis=1)

        # Transpose for easier iteration
        paths_transposed = np.transpose(paths_with_time, (1, 0, 2))  # (features, nb_paths, nb_dates)

        # Evaluate basis functions
        basis_values = np.zeros((nb_paths, nb_dates, self.nb_base_fcts))

        for t in range(nb_dates):
            state_t = paths_transposed[:, :, t].T  # (nb_paths, features)
            for i in range(nb_paths):
                for j in range(self.nb_base_fcts):
                    basis_values[i, t, j] = self.basis.base_fct(j, state_t[i])

        return basis_values

    def _predict(self, X):
        """Predict Q-values using current weights."""
        return np.dot(X, self.weights)

    def _fit(self, matrixU, vectorV):
        """Fit weights using least squares."""
        self.weights = np.linalg.solve(matrixU, vectorV)

    def get_exercise_time(self):
        """Return average exercise time normalized to [0, 1] (evaluation set only)."""
        if not hasattr(self, '_exercise_dates'):
            return None

        nb_dates = self.model.nb_dates
        # Only use evaluation set paths (self.split:), not training paths
        normalized_times = self._exercise_dates[self.split:] / nb_dates
        return float(np.mean(normalized_times))

    def backward_induction_on_paths(self, stock_paths, var_paths=None):
        """
        Apply learned policy using backward induction.

        This method replicates the pricing behavior for video generation by using
        the learned Q-function weights to make exercise decisions via backward induction.

        Args:
            stock_paths: (nb_paths, nb_stocks, nb_dates+1) - Stock price paths
            var_paths: (nb_paths, nb_stocks, nb_dates+1) - Variance paths (optional)

        Returns:
            exercise_dates: (nb_paths,) - Time step when each path is exercised
            payoff_values: (nb_paths,) - Payoff value at exercise for each path
            price: float - Average discounted payoff

        Raises:
            ValueError: If no learned policy available (must call price() first)
        """
        if not hasattr(self, 'weights') or np.allclose(self.weights, 0):
            raise ValueError("No learned policy available. Must call price() first to train.")

        nb_paths = stock_paths.shape[0]
        nb_dates = self.model.nb_dates

        # Compute all payoffs upfront
        payoffs = self.payoff(stock_paths)

        # Prepare stock paths with optional features
        paths = stock_paths.copy()
        if self.use_payoff_as_input:
            paths = np.concatenate([paths, np.expand_dims(payoffs, axis=1)], axis=1)
        if self.use_var and var_paths is not None:
            paths = np.concatenate([paths, var_paths], axis=1)

        # Evaluate basis functions for all paths and times
        eval_bases = self._evaluate_bases_all(paths, nb_dates + 1)

        # Initialize with terminal payoff
        values = payoffs[:, -1].copy()

        # Track exercise dates (initialize to maturity = nb_dates)
        exercise_dates = np.full(nb_paths, nb_dates, dtype=int)

        disc_factor = math.exp(-self.model.rate * self.model.maturity / nb_dates)

        # Backward induction from T-1 to 1
        for date in range(nb_dates - 1, 0, -1):
            # Current immediate exercise value
            immediate_exercise = payoffs[:, date]

            # Compute continuation values using learned weights
            continuation_values = self._predict(eval_bases[:, date, :])

            # Clip to non-negative (American option value can't be negative)
            continuation_values = np.maximum(0, continuation_values)

            # Discount future values
            discounted_values = values * disc_factor

            # Exercise decision: exercise if immediate > continuation
            exercise_now = immediate_exercise > continuation_values

            # Update values
            values = np.where(exercise_now, immediate_exercise, discounted_values)

            # Track exercise dates - only update if exercising earlier
            exercise_dates[exercise_now] = date

        # Extract payoff values at exercise time
        payoff_values = np.array([payoffs[i, exercise_dates[i]] for i in range(nb_paths)])

        # Compute price (average discounted payoff)
        price = np.mean(values)

        return exercise_dates, payoff_values, price


class FQIFastDeg1(FQIFast):
    """FQI using degree-1 polynomial basis."""

    def __init__(self, model, payoff, nb_epochs=20, hidden_size=None,
                 factors=None, train_ITM_only=True, use_payoff_as_input=False):
        super().__init__(model, payoff, nb_epochs, hidden_size, factors,
                         train_ITM_only, use_payoff_as_input)

        state_size = model.nb_stocks * (1 + self.use_var) + 2 + self.use_payoff_as_input * 1
        self.basis = basis_functions.BasisFunctionsDeg1(state_size)
        self.nb_base_fcts = self.basis.nb_base_fcts
        self.weights = np.zeros(self.nb_base_fcts)


class FQIFastLaguerre(FQIFast):
    """FQI using Laguerre polynomial basis with time features."""

    def __init__(self, model, payoff, nb_epochs=20, hidden_size=None,
                 factors=None, train_ITM_only=True, use_payoff_as_input=False):
        super().__init__(model, payoff, nb_epochs, hidden_size, factors,
                         train_ITM_only, use_payoff_as_input)

        state_size = model.nb_stocks * (1 + self.use_var) + 1 + self.use_payoff_as_input * 1
        K = getattr(payoff, 'strike', 1.0)
        self.basis = basis_functions.BasisFunctionsLaguerreTime(state_size, T=model.maturity, K=K)
        self.nb_base_fcts = self.basis.nb_base_fcts
        self.weights = np.zeros(self.nb_base_fcts)

    def _evaluate_bases_all(self, stock_paths, nb_dates):
        """
        Evaluate Laguerre basis functions with time.

        For Laguerre, only add single time feature (not both t and 1-t).
        """
        nb_paths = stock_paths.shape[0]

        # Add time feature: normalized time t/T
        time_grid = np.linspace(0, 1, nb_dates)
        time_features = time_grid[np.newaxis, np.newaxis, :]  # (1, 1, nb_dates)
        time_features = np.tile(time_features, (nb_paths, 1, 1))  # (nb_paths, 1, nb_dates)

        # Concatenate stock prices and time
        paths_with_time = np.concatenate([stock_paths, time_features], axis=1)

        # Add barrier values as input hint (if enabled)
        if self.nb_barriers > 0:
            spot = self.model.spot if hasattr(self.model, 'spot') else 100.0
            barrier_features = np.array([[[b / spot] for b in self.barrier_values]])  # (1, nb_barriers, 1)
            barrier_features = np.tile(barrier_features, (nb_paths, 1, nb_dates))  # (nb_paths, nb_barriers, nb_dates)
            paths_with_time = np.concatenate([paths_with_time, barrier_features], axis=1)

        # Transpose for easier iteration
        paths_transposed = np.transpose(paths_with_time, (1, 0, 2))

        # Evaluate basis functions
        basis_values = np.zeros((nb_paths, nb_dates, self.nb_base_fcts))

        for t in range(nb_dates):
            state_t = paths_transposed[:, :, t].T  # (nb_paths, features)
            for i in range(nb_paths):
                for j in range(self.nb_base_fcts):
                    basis_values[i, t, j] = self.basis.base_fct(j, state_t[i])

        return basis_values
