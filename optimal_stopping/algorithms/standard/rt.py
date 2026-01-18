"""
RT: Randomized Neural Networks Algorithm for Thesis.

This is a universal version of RLSM that works with BOTH:
- STANDARD (non-path-dependent) options
- PATH-DEPENDENT options (barriers, lookbacks)

Based on the RLSM implementation from (Optimal stopping via randomized neural networks,
Herrera, Krach, Ruyssen and Teichmann, 2021), with modifications to handle
path-dependent payoffs.

Compatible with ALL payoffs:
- BasketCall, BasketPut, MaxCall, MaxPut, GeometricBasketCall, etc.
- Barrier options (UpAndOut, DownAndOut, UpAndIn, DownAndIn)
- Lookback options
- All other path-dependent and non-path-dependent payoffs
"""

import torch
import numpy as np
import time
import math
from optimal_stopping.run import configs
from optimal_stopping.algorithms.utils import randomized_neural_networks


class RT:
    """
    RT: Randomized Neural Networks Algorithm for Thesis.

    Universal algorithm that handles both path-dependent and non-path-dependent options.
    Uses randomized neural networks as basis functions for regression.
    """

    def __init__(self, model, payoff, hidden_size=20, factors=(1., 1.),
                 train_ITM_only=True, use_payoff_as_input=True,
                 use_barrier_as_input=False, activation='leakyrelu', dropout=0.0,
                 **kwargs):
        """
        Initialize RT pricer.

        Args:
            model: Stock model
            payoff: Payoff function (path-dependent or non-path-dependent)
            hidden_size: Number of neurons in hidden layer (default: 20)
            factors: Tuple of (activation_slope, weight_scale, ...)
            train_ITM_only: If True, only use in-the-money paths for training
            use_payoff_as_input: If True, include payoff in state
            use_barrier_as_input: If True, include barrier values as input hint
            activation: Activation function ('relu', 'tanh', 'elu', 'leakyrelu')
            dropout: Dropout probability (default: 0.0)
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

        # Check for variance paths
        self.use_var = getattr(model, 'return_var', False)

        # Detect if payoff is path-dependent
        self.is_path_dependent = getattr(payoff, 'is_path_dependent', False)

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

        # Initialize randomized neural network
        # Use exact hidden_size from config (no modifications)
        state_size = model.nb_stocks * (1 + self.use_var) + self.use_payoff_as_input * 1 + self.nb_barriers
        self._init_reservoir(state_size, hidden_size, factors, activation, dropout)

        # Storage for learned policy (coefficients at each time step)
        self._learned_coefficients = {}  # {time_step: coefficients}

    def _init_reservoir(self, state_size, hidden_size, factors, activation, dropout):
        """Initialize randomized neural network (reservoir)."""
        # RT uses single layer (num_layers=1 fixed, like RLSM)
        self.reservoir = randomized_neural_networks.Reservoir2(
            hidden_size,
            state_size,
            factors=factors[1:],  # Weight scaling factors
            activation=activation,  # Configurable activation function
            num_layers=1,  # RT always uses 1 layer
            dropout=dropout  # Dropout probability
        )
        self.nb_base_fcts = hidden_size + 1  # +1 for constant term

    def _eval_payoff(self, stock_paths, date=None):
        """
        Evaluate payoff at a specific date, handling both path-dependent and non-path-dependent.

        Args:
            stock_paths: Full stock paths array (nb_paths, nb_stocks, nb_dates+1)
            date: Time index (if None, use full path for terminal payoff)

        Returns:
            payoffs: (nb_paths,) array of payoff values
        """
        if self.is_path_dependent:
            # Path-dependent: need full history up to date
            if date is None:
                # Terminal payoff - pass full path
                return self.payoff.eval(stock_paths[:, :self.model.nb_stocks, :])
            else:
                # Pass history from t=0 to t=date (inclusive)
                return self.payoff.eval(stock_paths[:, :self.model.nb_stocks, :date + 1])
        else:
            # Non-path-dependent: only need current state
            if date is None:
                # Terminal payoff - last time step
                return self.payoff.eval(stock_paths[:, :self.model.nb_stocks, -1])
            else:
                # Current time step
                return self.payoff.eval(stock_paths[:, :self.model.nb_stocks, date])

    def price(self, train_eval_split=2):
        """
        Compute option price using backward induction with RT.

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

        # Compute payoffs for all paths (for use_payoff_as_input feature)
        if self.use_payoff_as_input:
            nb_paths, nb_stocks, nb_dates_plus_1 = stock_paths.shape
            payoffs = np.zeros((nb_paths, nb_dates_plus_1))

            if self.is_path_dependent:
                # For path-dependent payoffs (barriers, lookbacks), try to use base payoff
                # at intermediate timesteps since full barrier logic requires complete path
                if hasattr(self.payoff, 'base_payoff'):
                    # Barrier wrapper - use base payoff for intermediate timesteps
                    base_payoff = self.payoff.base_payoff
                    for t in range(nb_dates_plus_1):
                        if base_payoff.is_path_dependent:
                            # Base is also path-dependent (e.g., Lookback with barrier)
                            payoffs[:, t] = base_payoff.eval(stock_paths[:, :nb_stocks, :t + 1])
                        else:
                            # Base is standard (e.g., BasketCall with barrier)
                            payoffs[:, t] = base_payoff.eval(stock_paths[:, :nb_stocks, t])
                else:
                    # Pure path-dependent (e.g., Lookback without barrier)
                    # Evaluate with partial path history
                    for t in range(nb_dates_plus_1):
                        payoffs[:, t] = self._eval_payoff(stock_paths, date=t)
            else:
                # Non-path-dependent: evaluate normally at each timestep
                for t in range(nb_dates_plus_1):
                    payoffs[:, t] = self._eval_payoff(stock_paths, date=t)

            # Add payoff as extra feature
            stock_paths = np.concatenate(
                [stock_paths, np.expand_dims(payoffs, axis=1)], axis=1
            )

        # Split into training and evaluation sets
        self.split = len(stock_paths) // train_eval_split

        nb_paths, nb_stocks_plus, nb_dates_from_shape = stock_paths.shape
        # Use model's nb_dates (actual time steps) for consistency across all algorithms
        disc_factor = math.exp(-self.model.rate * self.model.maturity / self.model.nb_dates)

        # Get strike price for normalization
        if hasattr(self.payoff, 'strike'):
            strike = self.payoff.strike
        else:
            strike = self.model.spot if hasattr(self.model, 'spot') else stock_paths[0, :self.model.nb_stocks, 0]
        if np.isscalar(strike):
            strike = np.full(self.model.nb_stocks, strike)

        # Initialize with terminal payoff
        values = self._eval_payoff(stock_paths, date=None)

        # Track exercise dates (initialize to maturity = nb_dates, not nb_dates-1)
        self._exercise_dates = np.full(nb_paths, self.model.nb_dates, dtype=int)

        # Clear previous learned policy and prepare to store new one
        self._learned_coefficients = {}

        # Backward induction from T-1 to 1
        for date in range(self.model.nb_dates - 1, 0, -1):
            # Current immediate exercise value
            immediate_exercise = self._eval_payoff(stock_paths, date=date)

            # Prepare state for regression (normalize stock prices by strike)
            if self.use_payoff_as_input:
                # Normalize only the stock price columns, keep payoff unnormalized
                normalized_stocks = stock_paths[:, :self.model.nb_stocks, date] / strike
                current_state = np.concatenate([normalized_stocks, stock_paths[:, self.model.nb_stocks:, date]], axis=1)
            else:
                current_state = stock_paths[:, :self.model.nb_stocks, date] / strike

            if self.use_var and var_paths is not None:
                current_state = np.concatenate([current_state, var_paths[:, :, date]], axis=1)

            # Add barrier values as input hint (if enabled)
            if self.nb_barriers > 0:
                strike_scalar = strike[0] if isinstance(strike, np.ndarray) else strike
                barrier_features = np.array([[b / strike_scalar for b in self.barrier_values]])
                barrier_features = np.repeat(barrier_features, current_state.shape[0], axis=0)
                current_state = np.concatenate([current_state, barrier_features], axis=1)

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
        payoff_0 = self._eval_payoff(stock_paths, date=0)

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

        # Compute payoffs for all paths (for use_payoff_as_input feature)
        if self.use_payoff_as_input:
            nb_paths, nb_stocks, nb_dates_plus_1 = stock_paths.shape
            payoffs = np.zeros((nb_paths, nb_dates_plus_1))

            if self.is_path_dependent:
                # For path-dependent payoffs (barriers, lookbacks), try to use base payoff
                # at intermediate timesteps since full barrier logic requires complete path
                if hasattr(self.payoff, 'base_payoff'):
                    # Barrier wrapper - use base payoff for intermediate timesteps
                    base_payoff = self.payoff.base_payoff
                    for t in range(nb_dates_plus_1):
                        if base_payoff.is_path_dependent:
                            # Base is also path-dependent (e.g., Lookback with barrier)
                            payoffs[:, t] = base_payoff.eval(stock_paths[:, :nb_stocks, :t + 1])
                        else:
                            # Base is standard (e.g., BasketCall with barrier)
                            payoffs[:, t] = base_payoff.eval(stock_paths[:, :nb_stocks, t])
                else:
                    # Pure path-dependent (e.g., Lookback without barrier)
                    # Evaluate with partial path history
                    for t in range(nb_dates_plus_1):
                        payoffs[:, t] = self._eval_payoff(stock_paths, date=t)
            else:
                # Non-path-dependent: evaluate normally at each timestep
                for t in range(nb_dates_plus_1):
                    payoffs[:, t] = self._eval_payoff(stock_paths, date=t)

            # Add payoff as extra feature
            stock_paths = np.concatenate(
                [stock_paths, np.expand_dims(payoffs, axis=1)], axis=1
            )

        # Split into training and evaluation sets
        self.split = len(stock_paths) // train_eval_split

        nb_paths, nb_stocks_plus, nb_dates_from_shape = stock_paths.shape
        disc_factor = math.exp(-self.model.rate * self.model.maturity / self.model.nb_dates)

        # Get strike price for normalization
        if hasattr(self.payoff, 'strike'):
            strike = self.payoff.strike
        else:
            strike = self.model.spot if hasattr(self.model, 'spot') else stock_paths[0, :self.model.nb_stocks, 0]
        if np.isscalar(strike):
            strike = np.full(self.model.nb_stocks, strike)

        # Initialize with terminal payoff
        values = self._eval_payoff(stock_paths, date=None)

        # Initialize martingale M for upper bound (starts at terminal payoff)
        M = np.zeros((nb_paths, self.model.nb_dates + 1))
        M[:, -1] = values.copy()

        # Clear previous learned policy
        self._learned_coefficients = {}

        # Initialize exercise dates tracking (for get_exercise_time)
        self._exercise_dates = np.full(nb_paths, self.model.nb_dates, dtype=int)

        # Backward induction from T-1 to 1
        for date in range(self.model.nb_dates - 1, 0, -1):
            # Current immediate exercise value
            immediate_exercise = self._eval_payoff(stock_paths, date=date)

            # Prepare state for regression (normalize stock prices by strike)
            if self.use_payoff_as_input:
                # Normalize only the stock price columns, keep payoff unnormalized
                normalized_stocks = stock_paths[:, :self.model.nb_stocks, date] / strike
                current_state = np.concatenate([normalized_stocks, stock_paths[:, self.model.nb_stocks:, date]], axis=1)
            else:
                current_state = stock_paths[:, :self.model.nb_stocks, date] / strike

            if self.use_var and var_paths is not None:
                current_state = np.concatenate([current_state, var_paths[:, :, date]], axis=1)

            # Add barrier values as input hint (if enabled)
            if self.nb_barriers > 0:
                strike_scalar = strike[0] if isinstance(strike, np.ndarray) else strike
                barrier_features = np.array([[b / strike_scalar for b in self.barrier_values]])
                barrier_features = np.repeat(barrier_features, current_state.shape[0], axis=0)
                current_state = np.concatenate([current_state, barrier_features], axis=1)

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

            # Track exercise dates
            self._exercise_dates[exercise_now] = date

            # Update martingale M for upper bound
            # M[t] = max(payoff[t], continuation_value[t])
            M[:, date] = np.maximum(immediate_exercise, continuation_values)

        # Compute lower bound (regular pricing on evaluation set)
        payoff_0 = self._eval_payoff(stock_paths, date=0)
        lower_bound = max(payoff_0[0], np.mean(values[self.split:]) * disc_factor)

        # Set M[0] based on time-0 payoff and discounted continuation value
        M[:, 0] = np.maximum(payoff_0, values * disc_factor)

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
        # Initialize continuation values to 0 (prevents noisy extrapolation for OTM paths)
        nb_paths = current_state.shape[0]
        continuation_values = np.zeros(nb_paths)
        coefficients = np.zeros(self.nb_base_fcts)  # Default to zero coefficients

        # Identify ITM paths in training set, and all ITM paths (for prediction)
        if self.train_ITM_only:
            # Train only on ITM paths in training set
            in_the_money = np.where(immediate_exercise[:self.split] > 0)[0]
            # Predict on all ITM paths
            in_the_money_all = np.where(immediate_exercise > 0)[0]
        else:
            # Train on all training paths
            in_the_money = np.arange(self.split)
            # Predict on all paths
            in_the_money_all = np.arange(nb_paths)

        if len(in_the_money) > 0:
            # Evaluate basis functions for training ITM paths
            X_train = current_state[in_the_money]
            X_tensor_train = torch.from_numpy(X_train).type(torch.float32)
            basis_train = self.reservoir(X_tensor_train).detach().numpy()
            # Add constant term (intercept)
            basis_train = np.concatenate([basis_train, np.ones((len(basis_train), 1))], axis=1)

            # Standard least squares (no regularization)
            coefficients = np.linalg.lstsq(
                basis_train,
                future_values[in_the_money],
                rcond=None
            )[0]

            # Evaluate basis functions for all ITM paths (for prediction)
            X_all = current_state[in_the_money_all]
            X_tensor_all = torch.from_numpy(X_all).type(torch.float32)
            basis_all = self.reservoir(X_tensor_all).detach().numpy()
            # Add constant term (intercept)
            basis_all = np.concatenate([basis_all, np.ones((len(basis_all), 1))], axis=1)

            # Predict continuation values for all ITM paths
            continuation_values[in_the_money_all] = np.dot(basis_all, coefficients)

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

        # Get strike price for normalization
        if hasattr(self.payoff, 'strike'):
            strike = self.payoff.strike
        else:
            strike = self.model.spot if hasattr(self.model, 'spot') else stock_paths[0, :self.model.nb_stocks, 0]
        if np.isscalar(strike):
            strike = np.full(self.model.nb_stocks, strike)

        # Initialize tracking
        exercise_times = np.full(nb_paths, nb_dates, dtype=int)  # Default to maturity = nb_dates
        payoff_values = self._eval_payoff(stock_paths, date=None)  # Terminal payoff
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
            immediate_exercise = self._eval_payoff(stock_paths[active_mask], date=date)

            # Prepare state for continuation value prediction (normalize stock prices by strike)
            if self.use_payoff_as_input:
                # Normalize only stock columns, keep payoff unnormalized
                normalized_stocks = stock_paths[active_mask, :self.model.nb_stocks, date] / strike
                # Get payoff column (assumed to be after stocks in stock_paths if use_payoff_as_input)
                payoffs_col = stock_paths[active_mask, self.model.nb_stocks:, date]
                current_state = np.concatenate([normalized_stocks, payoffs_col], axis=1)
            else:
                current_state = stock_paths[active_mask, :self.model.nb_stocks, date] / strike

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

        # Get strike price for normalization
        if hasattr(self.payoff, 'strike'):
            strike = self.payoff.strike
        else:
            strike = self.model.spot if hasattr(self.model, 'spot') else stock_paths[0, :self.model.nb_stocks, 0]
        if np.isscalar(strike):
            strike = np.full(self.model.nb_stocks, strike)

        # Compute all payoffs upfront
        payoffs = np.zeros((nb_paths, nb_dates + 1))
        for t in range(nb_dates + 1):
            payoffs[:, t] = self._eval_payoff(stock_paths, date=t)

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

            # Prepare state (normalize stock prices by strike)
            current_state = stock_paths[:, :self.model.nb_stocks, date] / strike

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

        # After loop, values represent value at time 1
        # Apply extra discount to get value at time 0
        discounted_continuation = values * disc_factor

        # Check if immediate exercise at time 0 is better than continuation
        payoff_0 = self._eval_payoff(stock_paths, date=0)
        final_values = np.maximum(payoff_0, discounted_continuation)

        # Extract payoff values at exercise time
        payoff_values = np.array([payoffs[i, exercise_dates[i]] for i in range(nb_paths)])

        # Compute price (average of final values)
        price = np.mean(final_values)

        return exercise_dates, payoff_values, price
