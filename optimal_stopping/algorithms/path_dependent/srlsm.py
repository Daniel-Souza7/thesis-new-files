"""
Special Randomized Least Squares Monte Carlo (SRLSM) for PATH-DEPENDENT options.

This is an adaptation of RLSM specifically for options that require full path history:
- Barrier options (UpAndOut, DownAndOut, UpAndIn, DownAndIn)
- Lookback options

KEY DIFFERENCES from standard RLSM:
1. Passes FULL PATH HISTORY when evaluating payoffs (not just current state)
2. Properly handles initial barrier checks at t=0
3. Tracks historical extrema for lookback features

The regression part (continuation value learning) is IDENTICAL to RLSM.
Only the payoff evaluation differs.
"""

import torch
import numpy as np
import time
import math
from optimal_stopping.run import configs
from optimal_stopping.algorithms.utils import randomized_neural_networks


class SRLSM:
    """
    Special RLSM for path-dependent options (barriers, lookbacks).

    Compatible with:
    - All barrier options (UpAndOut, DownAndOut, etc.)
    - All lookback options

    NOT compatible with standard options (use RLSM instead).
    """

    def __init__(self, model, payoff, hidden_size=100, factors=(1., 1.),
                 train_ITM_only=True, use_payoff_as_input=False,
                 use_barrier_as_input=False, activation='leakyrelu', dropout=0.0,
                 ridge_coeff=1e-3, **kwargs):
        """
        Initialize SRLSM pricer.

        Args:
            model: Stock model
            payoff: Payoff function (must BE path-dependent)
            hidden_size: Number of neurons in hidden layer
            factors: Tuple of (activation_slope, weight_scale, ...)
            train_ITM_only: If True, only use in-the-money paths for training
            use_payoff_as_input: Not typically used for barriers
            use_barrier_as_input: If True, include barrier values as input hint
            activation: Activation function ('relu', 'tanh', 'elu', 'leakyrelu')
            dropout: Dropout probability (default: 0.0, SRLSM uses single layer so dropout has less effect)
            ridge_coeff: Ridge regularization coefficient (default: 1e-3, standard practice)

        Raises:
            ValueError: If payoff is NOT path-dependent
        """
        self.model = model
        self.payoff = payoff
        self.hidden_size = hidden_size
        self.factors = factors
        self.train_ITM_only = train_ITM_only
        self.use_payoff_as_input = use_payoff_as_input
        self.use_barrier_as_input = use_barrier_as_input
        self.activation = activation
        self.dropout = dropout
        self.ridge_coeff = ridge_coeff

        # Check for variance paths
        self.use_var = getattr(model, 'return_var', False)

        # CRITICAL: Verify this IS a path-dependent option
        if not getattr(payoff, 'is_path_dependent', False):
            raise ValueError(
                f"SRLSM is for PATH-DEPENDENT options only. "
                f"The payoff '{type(payoff).__name__}' is NOT path-dependent. "
                f"Use RLSM for standard options."
            )

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

        # Initialize randomized neural network (same as RLSM)
        # Use exact hidden_size from config (no modifications)
        state_size = model.nb_stocks * (1 + self.use_var) + self.use_payoff_as_input * 1 + self.nb_barriers
        self._init_reservoir(state_size, hidden_size, factors, activation, dropout)

        # Storage for learned policy (coefficients at each time step)
        self._learned_coefficients = {}  # {time_step: coefficients}

    def _init_reservoir(self, state_size, hidden_size, factors, activation, dropout):
        """Initialize randomized neural network (reservoir)."""
        # SRLSM uses single layer (num_layers=1 fixed)
        self.reservoir = randomized_neural_networks.Reservoir2(
            hidden_size,
            state_size,
            factors=factors[1:],  # Weight scaling factors
            activation=activation,  # Configurable activation function
            num_layers=1,  # SRLSM always uses 1 layer
            dropout=dropout  # Dropout probability
        )
        self.nb_base_fcts = hidden_size + 1  # +1 for constant term

    def price(self, train_eval_split=2):
        """
        Compute option price for path-dependent options.

        KEY DIFFERENCE: Passes full path history when evaluating payoffs.

        Args:
            train_eval_split: Ratio for splitting paths (default: 2 = 50/50)

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

        # Split into training and evaluation sets
        self.split = len(stock_paths) // train_eval_split

        # Setup - FIX: stock_paths.shape[2] is nb_dates+1, not nb_dates
        nb_paths, nb_stocks, nb_dates_plus_one = stock_paths.shape
        nb_dates = nb_dates_plus_one - 1  # Actual number of time steps
        disc_factor = math.exp(-self.model.rate * self.model.maturity / nb_dates)

        # Initialize with terminal payoff
        # CRITICAL: Pass FULL PATH HISTORY (all dates from 0 to T)
        values = self.payoff.eval(stock_paths)

        # Track exercise dates (initialize to maturity = nb_dates, not nb_dates-1)
        self._exercise_dates = np.full(nb_paths, nb_dates, dtype=int)

        # Clear previous learned policy and prepare to store new one
        self._learned_coefficients = {}

        # Backward induction from T-1 to 1
        for date in range(nb_dates - 1, 0, -1):  # FIX: Use nb_dates-1 as upper bound
            # Current immediate exercise value
            # CRITICAL: Pass path history from t=0 to t=date (inclusive)
            path_history = stock_paths[:, :, :date + 1]
            immediate_exercise = self.payoff.eval(path_history)

            # Prepare current state for regression (ONLY current timestep needed)
            current_state = stock_paths[:, :, date]

            if self.use_var and var_paths is not None:
                current_state = np.concatenate([current_state, var_paths[:, :, date]], axis=1)

            # Add payoff to state if requested
            if self.use_payoff_as_input:
                current_state = np.concatenate([
                    current_state,
                    immediate_exercise.reshape(-1, 1)
                ], axis=1)

            # Learn continuation value (same as RLSM - uses current state only)
            continuation_values, coefficients = self._learn_continuation(
                current_state,
                values * disc_factor,
                immediate_exercise
            )

            # Store learned coefficients for this time step
            self._learned_coefficients[date] = coefficients

            # Update values based on optimal exercise decision
            exercise_now = immediate_exercise > continuation_values
            values[exercise_now] = immediate_exercise[exercise_now]
            values[~exercise_now] *= disc_factor

            # NEW: Track exercise dates - only update if exercising earlier
            self._exercise_dates[exercise_now] = date


        # Final payoff at t=0 (check initial state)
        payoff_0 = self.payoff.eval(stock_paths[:, :, :1])  # Pass t=0 as history

        # Return average price on evaluation set, discounted to time 0
        return max(payoff_0[0], np.mean(values[self.split:]) * disc_factor), time_path_gen

    def price_upper_lower_bound(self, train_eval_split=2):
        """
        Compute both lower and upper bounds for path-dependent American options.

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

        # Split into training and evaluation sets
        self.split = len(stock_paths) // train_eval_split

        nb_paths, nb_stocks, nb_dates_plus_one = stock_paths.shape
        nb_dates = nb_dates_plus_one - 1
        disc_factor = math.exp(-self.model.rate * self.model.maturity / nb_dates)

        # Initialize with terminal payoff (pass full path history)
        values = self.payoff.eval(stock_paths)

        # Initialize martingale M for upper bound
        M = np.zeros((nb_paths, nb_dates + 1))
        M[:, -1] = values.copy()

        # Clear previous learned policy
        self._learned_coefficients = {}

        # Initialize exercise dates tracking (for get_exercise_time)
        self._exercise_dates = np.full(nb_paths, nb_dates, dtype=int)

        # Backward induction from T-1 to 1
        for date in range(nb_dates - 1, 0, -1):
            # Current immediate exercise value (path-dependent)
            path_history = stock_paths[:, :, :date + 1]
            immediate_exercise = self.payoff.eval(path_history)

            # Prepare current state for regression
            current_state = stock_paths[:, :, date]

            if self.use_var and var_paths is not None:
                current_state = np.concatenate([current_state, var_paths[:, :, date]], axis=1)

            if self.use_payoff_as_input:
                current_state = np.concatenate([
                    current_state,
                    immediate_exercise.reshape(-1, 1)
                ], axis=1)

            # Add barrier values as input hint (if enabled)
            if self.nb_barriers > 0:
                spot = self.model.spot if hasattr(self.model, 'spot') else 100.0
                barrier_features = np.array([[b / spot for b in self.barrier_values]])
                barrier_features = np.repeat(barrier_features, current_state.shape[0], axis=0)
                current_state = np.concatenate([current_state, barrier_features], axis=1)

            # Learn continuation value
            continuation_values, coefficients = self._learn_continuation(
                current_state,
                values * disc_factor,
                immediate_exercise
            )

            # Store learned coefficients
            self._learned_coefficients[date] = coefficients

            # Update values for lower bound
            exercise_now = immediate_exercise > continuation_values
            values[exercise_now] = immediate_exercise[exercise_now]
            values[~exercise_now] *= disc_factor

            # Track exercise dates
            self._exercise_dates[exercise_now] = date

            # Update martingale M for upper bound
            M[:, date] = np.maximum(immediate_exercise, continuation_values)

        # Compute lower bound
        payoff_0 = self.payoff.eval(stock_paths[:, :, :1])
        lower_bound = max(payoff_0[0], np.mean(values[self.split:]) * disc_factor)

        # Set M[0] based on time-0 payoff and continuation value
        M[:, 0] = np.maximum(payoff_0, values)

        # Compute upper bound on evaluation set using dual formulation
        # The martingale M satisfies M[t] >= payoff[t] for all t
        # Upper bound = E[M[0]] where M is constructed via backward induction
        upper_bound = np.mean(M[self.split:, 0])

        return lower_bound, upper_bound, time_path_gen

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
        Apply learned policy using backward induction for path-dependent payoffs.

        This is what should be used for create_video to replicate pricing behavior.
        Uses backward induction (not forward simulation) with learned coefficients.

        Args:
            stock_paths: (nb_paths, nb_stocks, nb_dates+1) - Stock price paths
            var_paths: (nb_paths, nb_stocks, nb_dates+1) - Variance paths (optional)

        Returns:
            exercise_dates: (nb_paths,) - Time step when each path is exercised
            payoff_values: (nb_paths,) - Payoff value at exercise for each path
            price: float - Average discounted payoff
        """
        if not self._learned_coefficients:
            raise ValueError("No learned policy available. Must call price() first to train.")

        nb_paths = stock_paths.shape[0]
        nb_dates = self.model.nb_dates

        # Compute all payoffs (path-dependent: need full history at each step)
        payoffs = np.zeros((nb_paths, nb_dates + 1))
        for date in range(nb_dates + 1):
            path_history = stock_paths[:, :, :date + 1]  # Full history up to date
            payoffs[:, date] = self.payoff.eval(path_history)

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

            if self.use_var and var_paths is not None:
                current_state = np.concatenate([current_state, var_paths[:, :, date]], axis=1)

            if self.use_payoff_as_input:
                current_state = np.concatenate([
                    current_state,
                    immediate_exercise.reshape(-1, 1)
                ], axis=1)

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

    def _learn_continuation(self, current_state, future_values, immediate_exercise):
        """
        Learn continuation value using randomized neural network regression.

        This is IDENTICAL to RLSM - regression on current state.
        The difference is in payoff evaluation, not continuation value learning.

        Args:
            current_state: (nb_paths, nb_stocks) - Current stock prices
            future_values: (nb_paths,) - Discounted future values
            immediate_exercise: (nb_paths,) - Immediate payoff if exercised

        Returns:
            continuation_values: (nb_paths,) - Estimated continuation values
            coefficients: (nb_base_fcts,) - Learned regression coefficients
        """
        # Determine which paths to use for training
        if self.train_ITM_only:
            train_mask = (immediate_exercise[:self.split] > 0)
        else:
            train_mask = np.ones(self.split, dtype=bool)

        # Evaluate basis functions
        X_tensor = torch.from_numpy(current_state).type(torch.float32)
        basis = self.reservoir(X_tensor).detach().numpy()

        # Add constant term
        basis = np.concatenate([basis, np.ones((len(basis), 1))], axis=1)

        # Ridge regression: (X^T X + λI)^{-1} X^T y
        # where λ = ridge_coeff
        basis_train = basis[:self.split][train_mask]
        targets_train = future_values[:self.split][train_mask]

        XtX = basis_train.T @ basis_train
        ridge_penalty = self.ridge_coeff * np.eye(XtX.shape[0])
        Xty = basis_train.T @ targets_train

        coefficients = np.linalg.solve(XtX + ridge_penalty, Xty)

        # Predict continuation values
        continuation_values = np.dot(basis, coefficients)

        # Clip to non-negative (American option value can't be negative)
        continuation_values = np.maximum(0, continuation_values)

        return continuation_values, coefficients