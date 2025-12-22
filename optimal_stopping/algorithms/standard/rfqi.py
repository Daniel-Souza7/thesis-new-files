"""
Randomized Fitted Q-Iteration (RFQI) for STANDARD options.

This is the implementation from (Optimal stopping via randomized neural networks,
Herrera, Krach, Ruyssen and Teichmann, 2021).

RFQI uses fitted Q-iteration with randomized neural network basis functions.
Differs from RLSM by using reinforcement learning framework instead of regression.

IMPORTANT: This version is for STANDARD (non-path-dependent) options ONLY.
For barrier and lookback options, use SRFQI instead.

Compatible with:
- BasketCall, BasketPut
- MaxCall, MaxPut
- GeometricBasketCall, etc.

NOT compatible with:
- Barrier options (use SRFQI)
- Lookback options (use SRFQI)
"""

import numpy as np
import torch
import time
import math
from optimal_stopping.run import configs
from optimal_stopping.algorithms.utils import randomized_neural_networks


class RFQI:
    """
    Randomized Fitted Q-Iteration for standard options.

    Uses fitted Q-iteration with randomized neural networks as basis functions.
    """

    def __init__(self, model, payoff, nb_epochs=20, hidden_size=20,
                 factors=(1.,), train_ITM_only=True, use_payoff_as_input=False,
                 use_barrier_as_input=False, activation='leakyrelu', num_layers=1,
                 dropout=0.0, ridge_coeff=1e-3, early_stopping_callback=None,
                 **kwargs):
        """
        Initialize RFQI pricer.

        Args:
            model: Stock model
            payoff: Payoff function (must NOT be path-dependent)
            nb_epochs: Number of training epochs for fitted Q-iteration
            hidden_size: Number of neurons in hidden layer
            factors: Tuple of (activation_slope, weight_scale, ...)
            train_ITM_only: If True, only consider ITM paths in final pricing
            use_payoff_as_input: If True, include payoff in state
            use_barrier_as_input: If True, include barrier values as input hint
            activation: Activation function ('relu', 'tanh', 'elu', 'leakyrelu')
            num_layers: Number of hidden layers (1-4, RFQI can use multiple layers)
            dropout: Dropout probability between layers (default: 0.0)
            ridge_coeff: Ridge regularization coefficient (default: 1e-3, standard practice)
            early_stopping_callback: Early stopping callback (optional)

        Raises:
            ValueError: If payoff is path-dependent
        """
        self.model = model
        self.payoff = payoff
        self.nb_epochs = nb_epochs
        self.train_ITM_only = train_ITM_only
        self.use_payoff_as_input = use_payoff_as_input
        self.use_barrier_as_input = use_barrier_as_input
        self.activation = activation
        self.num_layers = num_layers
        self.dropout = dropout
        self.ridge_coeff = ridge_coeff
        self.early_stopping_callback = early_stopping_callback

        # Check for variance paths
        self.use_var = getattr(model, 'return_var', False)

        # CRITICAL: Verify this is NOT a path-dependent option
        if getattr(payoff, 'is_path_dependent', False):
            raise ValueError(
                f"RFQI is for STANDARD options only. "
                f"The payoff '{type(payoff).__name__}' is path-dependent. "
                f"Use SRFQI for path-dependent options (barriers, lookbacks)."
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

        # Initialize randomized neural network
        # Use exact hidden_size from config (no modifications)
        self.dim_out = hidden_size
        self.nb_base_fcts = self.dim_out + 1

        # State includes: stocks + time + time_to_maturity (+ optionally payoff + barriers)
        self.state_size = model.nb_stocks * (1 + self.use_var) + 2 + self.use_payoff_as_input * 1 + self.nb_barriers

        self._init_reservoir(factors, activation, num_layers, dropout)

    def _init_reservoir(self, factors, activation, num_layers, dropout):
        """Initialize randomized neural network."""
        # RFQI can use multiple layers (1-4)
        self.reservoir2 = randomized_neural_networks.Reservoir2(
            self.dim_out,
            self.state_size,
            factors=factors[1:] if len(factors) > 1 else (),
            activation=activation,  # Configurable activation function
            num_layers=num_layers,  # RFQI can have 1-4 layers
            dropout=dropout  # Dropout probability between layers
        )

    def evaluate_bases_all(self, stock_price):
        """
        Evaluate basis functions for all paths and dates.

        Args:
            stock_price: Array of shape (nb_paths, nb_stocks, nb_dates+1)

        Returns:
            basis: Array of shape (nb_paths, nb_dates+1, nb_base_fcts)
        """
        stocks = torch.from_numpy(stock_price).type(torch.float32)
        stocks = stocks.permute(0, 2, 1)  # (nb_paths, nb_dates+1, nb_stocks)

        # Add time features
        time = torch.linspace(0, 1, stocks.shape[1]).unsqueeze(0).repeat(
            (stocks.shape[0], 1)).unsqueeze(2)
        stocks = torch.cat([stocks, time, 1 - time], dim=-1)

        # Add barrier values as input hint (if enabled)
        if self.nb_barriers > 0:
            # Normalize barriers by spot price and add as features
            spot = self.model.spot if hasattr(self.model, 'spot') else 100.0
            barrier_features = torch.tensor(
                [[b / spot for b in self.barrier_values]],
                dtype=torch.float32
            ).repeat(stocks.shape[0], stocks.shape[1], 1)
            stocks = torch.cat([stocks, barrier_features], dim=-1)

        # Evaluate reservoir
        random_base = self.reservoir2(stocks)

        # Add constant term
        random_base = torch.cat([
            random_base,
            torch.ones([stocks.shape[0], stocks.shape[1], 1])
        ], dim=-1)

        return random_base.detach().numpy()

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
        if not hasattr(self, 'weights'):
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
        eval_bases = self.evaluate_bases_all(paths)

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
            continuation_values = np.dot(eval_bases[:, date, :], self.weights)

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

    def price(self, train_eval_split=2):
        """
        Compute option price using fitted Q-iteration with RFQI.

        Algorithm:
        1. Generate Monte Carlo paths
        2. Evaluate basis functions at all timesteps
        3. Iteratively train Q-function using fitted Q-iteration
        4. Determine optimal exercise times
        5. Compute discounted payoffs

        Args:
            train_eval_split: Ratio for splitting paths

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

        # Compute payoffs
        payoffs = self.payoff(stock_paths)

        # Optionally add payoff to state
        if self.use_payoff_as_input:
            stock_paths = np.concatenate(
                [stock_paths, np.expand_dims(payoffs, axis=1)], axis=1
            )

        # Add variance if needed
        if self.use_var and var_paths is not None:
            stock_paths = np.concatenate([stock_paths, var_paths], axis=1)

        # Split
        self.split = len(stock_paths) // train_eval_split

        # Setup
        nb_paths, _, nb_dates_plus_one = stock_paths.shape
        nb_dates = self.model.nb_dates
        deltaT = self.model.maturity / nb_dates
        discount_factor = math.exp(-self.model.rate * deltaT)

        # Evaluate basis functions for all paths and dates
        eval_bases = self.evaluate_bases_all(stock_paths)

        # Initialize Q-function weights
        weights = np.zeros(self.nb_base_fcts, dtype=float)

        # Fitted Q-iteration
        for epoch in range(self.nb_epochs):
            # Compute continuation values for dates 1 to T
            continuation_value = np.dot(eval_bases[:self.split, 1:, :], weights)

            # Clip to non-negative (American option value can't be negative)
            continuation_value = np.maximum(0, continuation_value)

            # Optimal stopping decision: max(immediate payoff, continuation)
            indicator_stop = np.maximum(payoffs[:self.split, 1:], continuation_value)

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

            # Build regression matrices
            # U = sum over paths and dates: basis(t) * basis(t)^T
            matrixU = np.tensordot(
                train_bases,
                train_bases,
                axes=([0, 1], [0, 1])
            )

            # V = sum over paths and dates: basis(t) * discounted_optimal_value(t+1)
            vectorV = np.sum(
                train_bases * discount_factor * np.repeat(
                    np.expand_dims(train_targets, axis=2),
                    self.nb_base_fcts,
                    axis=2
                ),
                axis=(0, 1)
            )

            # Ridge regression: (U + λI) * weights = V
            # where λ = ridge_coeff
            ridge_penalty = self.ridge_coeff * np.eye(matrixU.shape[0])
            weights = np.linalg.solve(matrixU + ridge_penalty, vectorV)

            # Early stopping: check if we should stop training
            if self.early_stopping_callback is not None:
                # Evaluate on validation set (use eval paths for early stopping)
                val_continuation = np.dot(eval_bases[self.split:, 1:, :], weights)
                val_continuation = np.maximum(0, val_continuation)
                val_values = np.maximum(payoffs[self.split:, 1:], val_continuation)
                val_score = np.mean(val_values)  # Higher is better

                if self.early_stopping_callback(val_score, epoch):
                    print(f"Early stopping at epoch {epoch+1}/{self.nb_epochs}")
                    break

        # Store learned weights for backward_induction_on_paths()
        self.weights = weights

        # Final evaluation on all paths
        continuation_value = np.dot(eval_bases, weights)

        # Clip to non-negative (American option value can't be negative)
        continuation_value = np.maximum(0, continuation_value)

        # Determine exercise decisions
        which = (payoffs > continuation_value) * 1
        which[:, -1] = 1  # Must exercise at maturity
        which[:, 0] = 0  # Cannot exercise before t=1

        # Find optimal exercise time for each path
        ex_dates = np.argmax(which, axis=1)

        # NEW: Track exercise dates for statistics
        self._exercise_dates = ex_dates.copy()

        # Compute discounted payoffs
        prices = np.take_along_axis(
            payoffs,
            np.expand_dims(ex_dates, axis=1),
            axis=1
        ).reshape(-1) * (discount_factor ** ex_dates)

        # Return average price on evaluation set
        price = max(np.mean(prices[self.split:]), payoffs[0, 0])

        return price, time_path_gen

    def price_upper_lower_bound(self, train_eval_split=2):
        """
        Compute both lower and upper bounds using Fitted Q-Iteration.

        Lower bound: Regular FQI pricing
        Upper bound: Dual formulation using learned Q-function

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

        if self.use_payoff_as_input:
            stock_paths = np.concatenate(
                [stock_paths, np.expand_dims(payoffs, axis=1)], axis=1
            )

        if self.use_var and var_paths is not None:
            stock_paths = np.concatenate([stock_paths, var_paths], axis=1)

        # Split
        self.split = len(stock_paths) // train_eval_split

        # Setup
        nb_paths, _, nb_dates_plus_one = stock_paths.shape
        nb_dates = self.model.nb_dates
        deltaT = self.model.maturity / nb_dates
        discount_factor = math.exp(-self.model.rate * deltaT)

        # Evaluate basis functions
        eval_bases = self.evaluate_bases_all(stock_paths)

        # Initialize Q-function weights
        weights = np.zeros(self.nb_base_fcts, dtype=float)

        # Initialize exercise dates tracking (will be updated after FQI)
        self._exercise_dates = None

        # Fitted Q-iteration (same as price())
        for epoch in range(self.nb_epochs):
            continuation_value = np.dot(eval_bases[:self.split, 1:, :], weights)
            continuation_value = np.maximum(0, continuation_value)
            indicator_stop = np.maximum(payoffs[:self.split, 1:], continuation_value)

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
                train_bases * discount_factor * np.repeat(
                    np.expand_dims(train_targets, axis=2),
                    self.nb_base_fcts,
                    axis=2
                ),
                axis=(0, 1)
            )

            weights = np.linalg.solve(matrixU, vectorV)

        self.weights = weights

        # Compute continuation values for all paths
        continuation_value = np.dot(eval_bases, weights)
        continuation_value = np.maximum(0, continuation_value)

        # Lower bound: Regular FQI pricing
        which = (payoffs > continuation_value) * 1
        which[:, -1] = 1
        which[:, 0] = 0
        ex_dates = np.argmax(which, axis=1)

        # Track exercise dates for get_exercise_time()
        self._exercise_dates = ex_dates

        prices = np.take_along_axis(
            payoffs,
            np.expand_dims(ex_dates, axis=1),
            axis=1
        ).reshape(-1) * (discount_factor ** ex_dates)

        lower_bound = max(np.mean(prices[self.split:]), payoffs[0, 0])

        # Upper bound: Construct martingale M via backward induction
        # Initialize M_diff for Haugh-Kogan dual formulation
        M_diff = np.zeros((nb_paths, nb_dates + 1))

        # Track previous continuation value for M_diff computation
        prev_cont_val = np.zeros(nb_paths)

        # Backward pass to construct M_diff
        for date in range(nb_dates, 0, -1):
            # Get continuation value at this date from learned Q-function
            curr_cont_val = continuation_value[:, date]

            # M_diff[t] = max(payoff[t], prev_cont_val) - curr_cont_val
            M_diff[:, date] = np.maximum(payoffs[:, date], prev_cont_val) - curr_cont_val

            prev_cont_val = curr_cont_val

        # Compute martingale M via cumulative sum
        M = np.cumsum(M_diff, axis=1)

        # Upper bound using dual formulation: E[max_t (payoff[t] - M[t])]
        upper_bound = np.mean(np.max(payoffs[self.split:] - M[self.split:], axis=1))

        return lower_bound, upper_bound, time_path_gen