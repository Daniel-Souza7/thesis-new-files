"""
Randomized Least Squares Monte Carlo (RLSM) for STANDARD options.

This is the implementation from (Optimal stopping via randomized neural networks,
Herrera, Krach, Ruyssen and Teichmann, 2021).

IMPORTANT: This version is for STANDARD (non-path-dependent) options ONLY.
For barrier and lookback options, use SRLSM instead.

Compatible with:
- BasketCall, BasketPut
- MaxCall, MaxPut
- GeometricBasketCall, etc.

NOT compatible with:
- Barrier options (use SRLSM)
- Lookback options (use SRLSM)
"""

import torch
import numpy as np
import time
import math
from optimal_stopping.run import configs
from optimal_stopping.algorithms.utils import randomized_neural_networks


class RLSM:
    """
    Randomized Least Squares Monte Carlo for standard options.

    Uses randomized neural networks as basis functions for regression.
    """

    def __init__(self, model, payoff, hidden_size=100, factors=(1., 1.),
                 train_ITM_only=True, use_payoff_as_input=False, **kwargs):
        """
        Initialize RLSM pricer.

        Args:
            model: Stock model
            payoff: Payoff function (must NOT be path-dependent)
            hidden_size: Number of neurons in hidden layer
            factors: Tuple of (activation_slope, weight_scale, ...)
            train_ITM_only: If True, only use in-the-money paths for training
            use_payoff_as_input: If True, include payoff in state

        Raises:
            ValueError: If payoff is path-dependent
        """
        self.model = model
        self.payoff = payoff
        self.hidden_size = hidden_size
        self.factors = factors
        self.train_ITM_only = train_ITM_only
        self.use_payoff_as_input = use_payoff_as_input

        # Check for variance paths
        self.use_var = getattr(model, 'return_var', False)

        # CRITICAL: Verify this is NOT a path-dependent option
        if getattr(payoff, 'is_path_dependent', False):
            raise ValueError(
                f"RLSM is for STANDARD options only. "
                f"The payoff '{type(payoff).__name__}' is path-dependent. "
                f"Use SRLSM for path-dependent options (barriers, lookbacks)."
            )

        # Initialize randomized neural network
        if hidden_size < 0:
            hidden_size = 50 + abs(hidden_size) * model.nb_stocks

        state_size = model.nb_stocks * (1 + self.use_var) + self.use_payoff_as_input * 1
        self._init_reservoir(state_size, hidden_size, factors)

        # Storage for learned policy (coefficients at each time step)
        self._learned_coefficients = {}  # {time_step: coefficients}

    def _init_reservoir(self, state_size, hidden_size, factors):
        """Initialize randomized neural network (reservoir)."""
        self.reservoir = randomized_neural_networks.Reservoir2(
            hidden_size,
            state_size,
            factors=factors[1:],  # Weight scaling factors
            activation=torch.nn.LeakyReLU(factors[0] / 2)  # Activation slope
        )
        self.nb_base_fcts = hidden_size + 1  # +1 for constant term

    def price(self, train_eval_split=2):
        """
        Compute option price using backward induction with RLSM.

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

        # Add payoff to stock paths if needed
        if self.use_payoff_as_input:
            stock_paths = np.concatenate(
                [stock_paths, np.expand_dims(payoffs, axis=1)], axis=1
            )

        # Split into training and evaluation sets
        self.split = len(stock_paths) // train_eval_split

        nb_paths, nb_stocks_plus, nb_dates_from_shape = stock_paths.shape
        # Use model's nb_dates (actual time steps) for consistency across all algorithms
        disc_factor = math.exp(-self.model.rate * self.model.maturity / self.model.nb_dates)

        # Initialize with terminal payoff
        values = self.payoff.eval(stock_paths[:, :self.model.nb_stocks, -1])

        # Track exercise dates (initialize to maturity = nb_dates, not nb_dates-1)
        self._exercise_dates = np.full(nb_paths, self.model.nb_dates, dtype=int)

        # Clear previous learned policy and prepare to store new one
        self._learned_coefficients = {}

        # Backward induction from T-1 to 1
        for date in range(self.model.nb_dates - 1, 0, -1):
            # Current immediate exercise value
            immediate_exercise = self.payoff.eval(stock_paths[:, :self.model.nb_stocks, date])

            # Prepare state for regression
            if self.use_payoff_as_input:
                current_state = stock_paths[:, :, date]
            else:
                current_state = stock_paths[:, :self.model.nb_stocks, date]

            if self.use_var and var_paths is not None:
                current_state = np.concatenate([current_state, var_paths[:, :, date]], axis=1)

            # Learn continuation value using randomized NN regression
            continuation_values, coefficients = self._learn_continuation(
                current_state,
                values * disc_factor,
                immediate_exercise
            )

            # Store learned coefficients for this time step
            self._learned_coefficients[date] = coefficients

            # Update values based on optimal exercise decision
            exercise_now = immediate_exercise > continuation_values

            # Track exercise dates - only update if exercising earlier
            self._exercise_dates[exercise_now] = date

            values[exercise_now] = immediate_exercise[exercise_now]
            values[~exercise_now] *= disc_factor

        # Final payoff at t=0
        payoff_0 = self.payoff.eval(stock_paths[:, :self.model.nb_stocks, 0])

        # Return average price on evaluation set, discounted to time 0
        return max(payoff_0[0], np.mean(values[self.split:]) * disc_factor), time_path_gen

    def price_upper_lower_bound(self, train_eval_split=2):
        """
        Compute both lower and upper bounds for American option price.

        Lower bound: Regular LSM pricing (suboptimal policy)
        Upper bound: Dual formulation (Rogers 2002, Haugh-Kogan 2004)

        Args:
            train_eval_split: Ratio for splitting paths into training/evaluation

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

        # Compute payoffs for all paths
        payoffs = self.payoff(stock_paths)

        # Add payoff to stock paths if needed
        if self.use_payoff_as_input:
            stock_paths = np.concatenate(
                [stock_paths, np.expand_dims(payoffs, axis=1)], axis=1
            )

        # Split into training and evaluation sets
        self.split = len(stock_paths) // train_eval_split

        nb_paths, nb_stocks_plus, nb_dates_from_shape = stock_paths.shape
        disc_factor = math.exp(-self.model.rate * self.model.maturity / self.model.nb_dates)

        # Initialize with terminal payoff
        values = self.payoff.eval(stock_paths[:, :self.model.nb_stocks, -1])

        # Initialize martingale M for upper bound (starts at terminal payoff)
        M = np.zeros((nb_paths, self.model.nb_dates + 1))
        M[:, -1] = values.copy()

        # Clear previous learned policy
        self._learned_coefficients = {}

        # Backward induction from T-1 to 1
        for date in range(self.model.nb_dates - 1, 0, -1):
            # Current immediate exercise value
            immediate_exercise = self.payoff.eval(stock_paths[:, :self.model.nb_stocks, date])

            # Prepare state for regression
            if self.use_payoff_as_input:
                current_state = stock_paths[:, :, date]
            else:
                current_state = stock_paths[:, :self.model.nb_stocks, date]

            if self.use_var and var_paths is not None:
                current_state = np.concatenate([current_state, var_paths[:, :, date]], axis=1)

            # Learn continuation value using randomized NN regression
            continuation_values, coefficients = self._learn_continuation(
                current_state,
                values * disc_factor,
                immediate_exercise
            )

            # Store learned coefficients
            self._learned_coefficients[date] = coefficients

            # Update values for lower bound (optimal exercise decision)
            exercise_now = immediate_exercise > continuation_values
            values[exercise_now] = immediate_exercise[exercise_now]
            values[~exercise_now] *= disc_factor

            # Update martingale M for upper bound
            # M[t] = max(payoff[t], continuation_value[t])
            M[:, date] = np.maximum(immediate_exercise, continuation_values)

        # Compute lower bound (regular pricing on evaluation set)
        payoff_0 = self.payoff.eval(stock_paths[:, :self.model.nb_stocks, 0])
        lower_bound = max(payoff_0[0], np.mean(values[self.split:]) * disc_factor)

        # Compute upper bound on evaluation set using dual formulation
        # Upper bound = E[max_t (payoff[t] - M[t] + M[0])]
        eval_payoffs = payoffs[self.split:]
        eval_M = M[self.split:]

        # Compute max_t (payoff[t] - M[t]) for each path
        payoff_minus_M = eval_payoffs - eval_M
        max_diff = np.max(payoff_minus_M, axis=1)

        # Add M[0] back (martingale starting value)
        upper_bound = np.mean(max_diff + eval_M[:, 0])

        return lower_bound, upper_bound, time_path_gen

    def get_exercise_time(self):
        """Return average exercise time normalized to [0, 1] (evaluation set only)."""
        if not hasattr(self, '_exercise_dates'):
            return None

        nb_dates = self.model.nb_dates
        # Only use evaluation set paths (self.split:), not training paths
        normalized_times = self._exercise_dates[self.split:] / nb_dates
        return float(np.mean(normalized_times))

    def _learn_continuation(self, current_state, future_values, immediate_exercise):
        """
        Learn continuation value using randomized neural network regression.

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

        # Initialize continuation values to 0 (prevents noisy extrapolation for OTM paths)
        nb_paths = current_state.shape[0]
        continuation_values = np.zeros(nb_paths)
        coefficients = np.zeros(self.nb_base_fcts)  # Default to zero coefficients

        # Only compute continuation values for ITM paths
        if itm_mask.sum() > 0:
            # Get indices of ITM paths
            itm_indices = np.where(itm_mask)[0]

            # Evaluate basis functions only for ITM paths
            X_itm = current_state[itm_mask]
            X_tensor = torch.from_numpy(X_itm).type(torch.float32)
            basis_itm = self.reservoir(X_tensor).detach().numpy()

            # Add constant term (intercept)
            basis_itm = np.concatenate([basis_itm, np.ones((len(basis_itm), 1))], axis=1)

            # Determine which ITM paths are in the training set
            train_itm_mask = itm_indices < self.split  # Boolean mask within itm_indices

            if train_itm_mask.sum() > 0:
                # Get basis and future values for training ITM paths
                basis_train = basis_itm[train_itm_mask]
                train_itm_indices = itm_indices[train_itm_mask]

                coefficients = np.linalg.lstsq(
                    basis_train,
                    future_values[train_itm_indices],
                    rcond=None
                )[0]

                # Predict continuation values for all ITM paths
                continuation_values[itm_mask] = np.dot(basis_itm, coefficients)

                # Clip to non-negative (American option value can't be negative)
                continuation_values = np.maximum(0, continuation_values)

        return continuation_values, coefficients

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

        # Initialize tracking
        exercise_times = np.full(nb_paths, nb_dates, dtype=int)  # Default to maturity = nb_dates
        payoff_values = self.payoff.eval(stock_paths[:, :self.model.nb_stocks, -1])  # Terminal payoff
        exercised = np.zeros(nb_paths, dtype=bool)  # Track which paths already exercised

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
            immediate_exercise = self.payoff.eval(stock_paths[active_mask, :self.model.nb_stocks, date])

            # Prepare state for continuation value prediction
            if self.use_payoff_as_input:
                current_state = stock_paths[active_mask, :, date]
            else:
                current_state = stock_paths[active_mask, :self.model.nb_stocks, date]

            if self.use_var and var_paths is not None:
                current_state = np.concatenate([current_state, var_paths[active_mask, :, date]], axis=1)

            # Predict continuation value using learned coefficients
            coefficients = self._learned_coefficients[date]

            # Evaluate basis functions using reservoir
            X_tensor = torch.from_numpy(current_state).type(torch.float32)
            basis = self.reservoir(X_tensor).detach().numpy()
            basis = np.concatenate([basis, np.ones((len(basis), 1))], axis=1)

            continuation_values = np.dot(basis, coefficients)

            # Clip continuation values to be non-negative (can't be negative for American options)
            # This prevents spurious early exercise when regression produces negative values
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

            # Get learned coefficients for this time step
            coefficients = self._learned_coefficients[date]

            # Evaluate basis functions using reservoir
            X_tensor = torch.from_numpy(current_state).type(torch.float32)
            basis = self.reservoir(X_tensor).detach().numpy()
            basis = np.concatenate([basis, np.ones((len(basis), 1))], axis=1)

            # Compute continuation values
            continuation_values = np.dot(basis, coefficients)

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