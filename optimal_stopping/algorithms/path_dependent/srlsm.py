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
                 train_ITM_only=True, use_payoff_as_input=False, **kwargs):
        """
        Initialize SRLSM pricer.

        Args:
            model: Stock model
            payoff: Payoff function (must BE path-dependent)
            hidden_size: Number of neurons in hidden layer
            factors: Tuple of (activation_slope, weight_scale, ...)
            train_ITM_only: If True, only use in-the-money paths for training
            use_payoff_as_input: Not typically used for barriers

        Raises:
            ValueError: If payoff is NOT path-dependent
        """
        self.model = model
        self.payoff = payoff
        self.hidden_size = hidden_size
        self.factors = factors
        self.train_ITM_only = train_ITM_only
        self.use_payoff_as_input = use_payoff_as_input

        # Check for variance paths
        self.use_var = getattr(model, 'return_var', False)

        # CRITICAL: Verify this IS a path-dependent option
        if not getattr(payoff, 'is_path_dependent', False):
            raise ValueError(
                f"SRLSM is for PATH-DEPENDENT options only. "
                f"The payoff '{type(payoff).__name__}' is NOT path-dependent. "
                f"Use RLSM for standard options."
            )

        # Initialize randomized neural network (same as RLSM)
        if hidden_size < 0:
            hidden_size = 50 + abs(hidden_size) * model.nb_stocks

        state_size = model.nb_stocks * (1 + self.use_var) + self.use_payoff_as_input * 1
        self._init_reservoir(state_size, hidden_size, factors)

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
            continuation_values = self._learn_continuation(
                current_state,
                values * disc_factor,
                immediate_exercise
            )

            # Update values based on optimal exercise decision
            exercise_now = immediate_exercise > continuation_values
            values[exercise_now] = immediate_exercise[exercise_now]
            values[~exercise_now] *= disc_factor

        # Final payoff at t=0 (check initial state)
        payoff_0 = self.payoff.eval(stock_paths[:, :, :1])  # Pass t=0 as history

        # Return average price on evaluation set, discounted to time 0
        return max(payoff_0[0], np.mean(values[self.split:]) * disc_factor), time_path_gen

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

        # Least squares regression on training set
        coefficients = np.linalg.lstsq(
            basis[:self.split][train_mask],
            future_values[:self.split][train_mask],
            rcond=None
        )[0]

        # Predict continuation values
        return np.dot(basis, coefficients)