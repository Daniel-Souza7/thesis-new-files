"""
Least Squares Monte Carlo (LSM) for American option pricing.

Simple benchmark implementation of the classic LSM algorithm from:
"Valuing American Options by Simulation: A Simple Least-Squares Approach"
(Longstaff and Schwartz, 2001)
"""

import numpy as np
import math
import time
from optimal_stopping.run import configs
from optimal_stopping.algorithms.utils import basis_functions


class LeastSquaresPricer:
    """
    Computes American option price by Least Square Monte Carlo (LSM).

    Uses polynomial basis functions for regression-based continuation value estimation.
    """

    def __init__(self, model, payoff, nb_epochs=None, hidden_size=None,
                 factors=None, train_ITM_only=True, use_payoff_as_input=False,
                 use_barrier_as_input=False):
        """
        Initialize LSM pricer.

        Args:
            model: Stock model
            payoff: Payoff function
            nb_epochs: Ignored (for API compatibility)
            hidden_size: Ignored (for API compatibility)
            factors: Ignored (for API compatibility)
            train_ITM_only: If True, only train on in-the-money paths
            use_payoff_as_input: If True, include payoff as feature
            use_barrier_as_input: If True, include barrier values as input hint
        """
        self.model = model
        self.payoff = payoff
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

        # Initialize basis functions (degree 2 polynomials)
        state_size = model.nb_stocks * (1 + self.use_var) + self.use_payoff_as_input * 1 + self.nb_barriers
        self.basis = basis_functions.BasisFunctions(state_size)
        self.nb_base_fcts = self.basis.nb_base_fcts

        # Storage for learned policy (coefficients at each time step)
        self._learned_coefficients = {}  # {time_step: coefficients}

    def price(self, train_eval_split=2):
        """
        Compute option price using LSM backward induction.

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

        # Compute payoffs for all paths
        payoffs = self.payoff(stock_paths)

        # Split into training and evaluation sets
        self.split = len(stock_paths) // train_eval_split

        nb_paths, nb_stocks, nb_dates_from_shape = stock_paths.shape
        # Use model's nb_dates (actual time steps) for consistency across all algorithms
        disc_factor = math.exp(-self.model.rate * self.model.maturity / self.model.nb_dates)

        # Initialize with terminal payoff
        values = payoffs[:, -1].copy()

        # Track exercise dates (initialize to maturity = nb_dates, not nb_dates-1)
        self._exercise_dates = np.full(nb_paths, self.model.nb_dates, dtype=int)

        # Clear previous learned policy and prepare to store new one
        self._learned_coefficients = {}

        # Backward induction from T-1 to 1
        for date in range(self.model.nb_dates - 1, 0, -1):
            # Current immediate exercise value
            immediate_exercise = payoffs[:, date]

            # Prepare state for regression
            current_state = stock_paths[:, :, date]

            if self.use_payoff_as_input:
                current_state = np.concatenate([current_state, payoffs[:, date:date+1]], axis=1)

            if self.use_var and var_paths is not None:
                current_state = np.concatenate([current_state, var_paths[:, :, date]], axis=1)

            # Add barrier values as input hint (if enabled)
            if self.nb_barriers > 0:
                spot = self.model.spot if hasattr(self.model, 'spot') else 100.0
                barrier_features = np.array([[b / spot for b in self.barrier_values]])
                barrier_features = np.repeat(barrier_features, current_state.shape[0], axis=0)
                current_state = np.concatenate([current_state, barrier_features], axis=1)

            # Discount future values
            discounted_values = values * disc_factor

            # Learn continuation value using least squares regression
            continuation_values, coefficients = self._learn_continuation(
                current_state, discounted_values, immediate_exercise
            )

            # Store learned coefficients for this time step
            self._learned_coefficients[date] = coefficients

            # Update values: max(exercise now, continue)
            exercise_now = immediate_exercise > continuation_values
            values = np.where(exercise_now, immediate_exercise, discounted_values)

            # Track exercise dates - only update if exercising earlier
            self._exercise_dates[exercise_now] = date

        # Final price: average over evaluation paths
        price = np.mean(values[self.split:])

        return price, time_path_gen

    def price_upper_lower_bound(self, train_eval_split=2):
        """
        Compute both lower and upper bounds using LSM.

        Lower bound: Regular LSM pricing
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

        # Initialize with terminal payoff
        values = payoffs[:, -1].copy()

        # Initialize martingale M for upper bound
        M = np.zeros((nb_paths, self.model.nb_dates + 1))
        M[:, -1] = values.copy()

        # Clear previous learned policy
        self._learned_coefficients = {}

        # Initialize exercise dates tracking (for get_exercise_time)
        self._exercise_dates = np.full(nb_paths, self.model.nb_dates, dtype=int)

        # Backward induction
        for date in range(self.model.nb_dates - 1, 0, -1):
            immediate_exercise = payoffs[:, date]
            current_state = stock_paths[:, :, date]

            if self.use_payoff_as_input:
                current_state = np.concatenate([current_state, payoffs[:, date:date+1]], axis=1)

            if self.use_var and var_paths is not None:
                current_state = np.concatenate([current_state, var_paths[:, :, date]], axis=1)

            # Add barrier values as input hint (if enabled)
            if self.nb_barriers > 0:
                spot = self.model.spot if hasattr(self.model, 'spot') else 100.0
                barrier_features = np.array([[b / spot for b in self.barrier_values]])
                barrier_features = np.repeat(barrier_features, current_state.shape[0], axis=0)
                current_state = np.concatenate([current_state, barrier_features], axis=1)

            discounted_values = values * disc_factor

            continuation_values, coefficients = self._learn_continuation(
                current_state, discounted_values, immediate_exercise
            )

            self._learned_coefficients[date] = coefficients

            # Lower bound: update values
            exercise_now = immediate_exercise > continuation_values
            values = np.where(exercise_now, immediate_exercise, discounted_values)

            # Track exercise dates
            self._exercise_dates[exercise_now] = date

            # Upper bound: update martingale M
            M[:, date] = np.maximum(immediate_exercise, continuation_values)

        # Lower bound
        lower_bound = np.mean(values[self.split:])

        # Set M[0] based on time-0 payoff and continuation value
        payoff_0 = payoffs[:, 0]
        M[:, 0] = np.maximum(payoff_0, values)

        # Upper bound using dual formulation
        # The martingale M satisfies M[t] >= payoff[t] for all t
        # Upper bound = E[M[0]] where M is constructed via backward induction
        upper_bound = np.mean(M[self.split:, 0])

        return lower_bound, upper_bound, time_path_gen

    def _learn_continuation(self, current_state, future_values, immediate_exercise):
        """
        Learn continuation value using least squares regression on polynomial basis.

        Args:
            current_state: (nb_paths, state_size) - Current stock prices (+ payoff if used)
            future_values: (nb_paths,) - Discounted future values
            immediate_exercise: (nb_paths,) - Immediate payoff if exercised

        Returns:
            continuation_values: (nb_paths,) - Estimated continuation values
            coefficients: (nb_base_fcts,) - Learned regression coefficients
        """
        # Determine which paths are ITM
        if self.train_ITM_only:
            # Only use in-the-money paths
            itm_mask = (immediate_exercise > 0)
        else:
            # Use all paths
            itm_mask = np.ones(len(immediate_exercise), dtype=bool)

        # Initialize continuation values to 0 (like old LSM code)
        # This prevents noisy extrapolation for OTM paths
        nb_paths = current_state.shape[0]
        continuation_values = np.zeros(nb_paths)
        coefficients = np.zeros(self.nb_base_fcts)  # Default to zero coefficients

        # Only compute continuation values for ITM paths
        if itm_mask.sum() > 0:
            # Get indices of ITM paths
            itm_indices = np.where(itm_mask)[0]

            # Evaluate basis functions only for ITM paths
            basis_matrix = np.zeros((len(itm_indices), self.nb_base_fcts))
            for idx, i in enumerate(itm_indices):
                for j in range(self.nb_base_fcts):
                    basis_matrix[idx, j] = self.basis.base_fct(j, current_state[i])

            # Determine which ITM paths are in the training set
            train_itm_mask = itm_indices < self.split  # Boolean mask within itm_indices

            if train_itm_mask.sum() > 0:
                # Get basis and future values for training ITM paths
                basis_train = basis_matrix[train_itm_mask]
                train_itm_indices = itm_indices[train_itm_mask]

                coefficients = np.linalg.lstsq(
                    basis_train,
                    future_values[train_itm_indices],
                    rcond=None
                )[0]

                # Predict continuation values for all ITM paths
                continuation_values[itm_mask] = np.dot(basis_matrix, coefficients)

                # Clip to non-negative (American option value can't be negative)
                continuation_values = np.maximum(0, continuation_values)

        return continuation_values, coefficients

    def get_exercise_time(self):
        """Return average exercise time normalized to [0, 1] (evaluation set only)."""
        if not hasattr(self, '_exercise_dates'):
            return None

        nb_dates = self.model.nb_dates
        # Only use evaluation set paths (self.split:), not training paths
        normalized_times = self._exercise_dates[self.split:] / nb_dates
        return float(np.mean(normalized_times))

    def predict(self, stock_paths, var_paths=None):
        """
        Apply learned policy to new stock paths.

        Args:
            stock_paths: (nb_paths, nb_stocks, nb_dates+1) - Stock price paths
            var_paths: (nb_paths, nb_stocks, nb_dates+1) - Variance paths (optional)

        Returns:
            exercise_times: (nb_paths,) - Time step when each path is exercised
            payoff_values: (nb_paths,) - Payoff value at exercise for each path
        """
        if not self._learned_coefficients:
            raise ValueError("No learned policy available. Must call price() first to train.")

        nb_paths = stock_paths.shape[0]
        nb_dates = self.model.nb_dates

        # Compute all payoffs upfront
        payoffs = self.payoff(stock_paths)

        # Initialize tracking
        exercise_times = np.full(nb_paths, nb_dates, dtype=int)  # Default to maturity = nb_dates
        payoff_values = payoffs[:, -1].copy()  # Default to terminal payoff
        exercised = np.zeros(nb_paths, dtype=bool)  # Track which paths already exercised

        disc_factor = math.exp(-self.model.rate * self.model.maturity / self.model.nb_dates)

        # Forward simulation: check exercise decision at each time step
        for date in range(1, nb_dates):  # Start from 1, not 0
            # Skip if no coefficients learned for this time step
            if date not in self._learned_coefficients:
                continue

            # Only process paths that haven't exercised yet
            active_mask = ~exercised

            if active_mask.sum() == 0:
                break  # All paths exercised

            # Get immediate exercise value
            immediate_exercise = payoffs[active_mask, date]

            # Prepare state for continuation value prediction
            current_state = stock_paths[active_mask, :, date]

            if self.use_payoff_as_input:
                current_state = np.concatenate([current_state, payoffs[active_mask, date:date+1]], axis=1)

            if self.use_var and var_paths is not None:
                current_state = np.concatenate([current_state, var_paths[active_mask, :, date]], axis=1)

            # Add barrier values as input hint (if enabled)
            if self.nb_barriers > 0:
                spot = self.model.spot if hasattr(self.model, 'spot') else 100.0
                barrier_features = np.array([[b / spot for b in self.barrier_values]])
                barrier_features = np.repeat(barrier_features, current_state.shape[0], axis=0)
                current_state = np.concatenate([current_state, barrier_features], axis=1)

            # Predict continuation value using learned coefficients
            coefficients = self._learned_coefficients[date]
            nb_active = active_mask.sum()

            # Evaluate basis functions
            basis_matrix = np.zeros((nb_active, self.nb_base_fcts))
            for i in range(nb_active):
                for j in range(self.nb_base_fcts):
                    basis_matrix[i, j] = self.basis.base_fct(j, current_state[i])

            continuation_values = np.dot(basis_matrix, coefficients)

            # Clip continuation values to be non-negative (can't be negative for American options)
            # This prevents spurious early exercise when polynomial regression produces negative values
            continuation_values = np.maximum(0, continuation_values)

            # Exercise decision: exercise if immediate > continuation
            exercise_now = immediate_exercise > continuation_values

            if exercise_now.sum() > 0:
                # Update exercise info for paths that choose to exercise
                active_indices = np.where(active_mask)[0]
                exercise_indices = active_indices[exercise_now]

                exercise_times[exercise_indices] = date
                payoff_values[exercise_indices] = immediate_exercise[exercise_now]
                exercised[exercise_indices] = True

        return exercise_times, payoff_values

    def backward_induction_on_paths(self, stock_paths, var_paths=None):
        """
        Apply learned policy using backward induction (same as training).

        This is what should be used for create_video to replicate pricing behavior.
        Uses backward induction (not forward simulation) with learned coefficients.

        Args:
            stock_paths: (nb_paths, nb_stocks, nb_dates+1) - Stock price paths
            var_paths: (nb_paths, nb_stocks, nb_dates+1) - Variance paths (optional)

        Returns:
            exercise_times: (nb_paths,) - Time step when each path is exercised
            payoff_values: (nb_paths,) - Payoff value at exercise for each path
            price: float - Average discounted payoff
        """
        if not self._learned_coefficients:
            raise ValueError("No learned policy available. Must call price() first to train.")

        nb_paths = stock_paths.shape[0]
        nb_dates = self.model.nb_dates

        # Compute all payoffs upfront
        payoffs = self.payoff(stock_paths)

        # Initialize with terminal payoff
        values = payoffs[:, -1].copy()

        # Track exercise dates (initialize to maturity = nb_dates)
        exercise_dates = np.full(nb_paths, nb_dates, dtype=int)

        disc_factor = math.exp(-self.model.rate * self.model.maturity / nb_dates)

        # Backward induction from T-1 to 1 (same as in price())
        for date in range(nb_dates - 1, 0, -1):
            # Skip if no coefficients learned for this time step
            if date not in self._learned_coefficients:
                continue

            # Current immediate exercise value
            immediate_exercise = payoffs[:, date]

            # Prepare state
            current_state = stock_paths[:, :, date]

            if self.use_payoff_as_input:
                current_state = np.concatenate([current_state, payoffs[:, date:date+1]], axis=1)

            if self.use_var and var_paths is not None:
                current_state = np.concatenate([current_state, var_paths[:, :, date]], axis=1)

            # Add barrier values as input hint (if enabled)
            if self.nb_barriers > 0:
                spot = self.model.spot if hasattr(self.model, 'spot') else 100.0
                barrier_features = np.array([[b / spot for b in self.barrier_values]])
                barrier_features = np.repeat(barrier_features, current_state.shape[0], axis=0)
                current_state = np.concatenate([current_state, barrier_features], axis=1)

            # Get learned coefficients for this time step
            coefficients = self._learned_coefficients[date]

            # Evaluate basis functions for all paths
            basis_matrix = np.zeros((nb_paths, self.nb_base_fcts))
            for i in range(nb_paths):
                for j in range(self.nb_base_fcts):
                    basis_matrix[i, j] = self.basis.base_fct(j, current_state[i])

            # Compute continuation values
            continuation_values = np.dot(basis_matrix, coefficients)

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


class LeastSquarePricerDeg1(LeastSquaresPricer):
    """LSM using degree-1 polynomial basis."""

    def __init__(self, model, payoff, nb_epochs=None, hidden_size=None,
                 factors=None, train_ITM_only=True, use_payoff_as_input=False):
        super().__init__(model, payoff, nb_epochs, hidden_size, factors,
                         train_ITM_only, use_payoff_as_input)

        state_size = model.nb_stocks * (1 + self.use_var) + self.use_payoff_as_input * 1
        self.basis = basis_functions.BasisFunctionsDeg1(state_size)
        self.nb_base_fcts = self.basis.nb_base_fcts


class LeastSquarePricerLaguerre(LeastSquaresPricer):
    """LSM using weighted Laguerre polynomial basis."""

    def __init__(self, model, payoff, nb_epochs=None, hidden_size=None,
                 factors=None, train_ITM_only=True, use_payoff_as_input=False):
        super().__init__(model, payoff, nb_epochs, hidden_size, factors,
                         train_ITM_only, use_payoff_as_input)

        state_size = model.nb_stocks * (1 + self.use_var) + self.use_payoff_as_input * 1
        # Use strike as scaling parameter K
        K = getattr(payoff, 'strike', 1.0)
        self.basis = basis_functions.BasisFunctionsLaguerre(state_size, K=K)
        self.nb_base_fcts = self.basis.nb_base_fcts
