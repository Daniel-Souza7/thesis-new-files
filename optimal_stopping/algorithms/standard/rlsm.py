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

        # NEW: Track exercise dates (initialize to maturity)
        self._exercise_dates = np.full(nb_paths, self.model.nb_dates - 1, dtype=int)

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
            continuation_values = self._learn_continuation(
                current_state,
                values * disc_factor,
                immediate_exercise
            )

            # Update values based on optimal exercise decision
            exercise_now = immediate_exercise > continuation_values

            # NEW: Track exercise dates - only update if exercising earlier
            self._exercise_dates[exercise_now] = date

            values[exercise_now] = immediate_exercise[exercise_now]
            values[~exercise_now] *= disc_factor

        # Final payoff at t=0
        payoff_0 = self.payoff.eval(stock_paths[:, :self.model.nb_stocks, 0])

        # Return average price on evaluation set, discounted to time 0
        return max(payoff_0[0], np.mean(values[self.split:]) * disc_factor), time_path_gen

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
        """
        # Determine which paths to use for training
        if self.train_ITM_only:
            # Only train on in-the-money paths
            train_mask = (immediate_exercise[:self.split] > 0)
            # Also identify ITM paths for all paths (training + evaluation)
            itm_mask = (immediate_exercise > 0)
        else:
            # Train on all paths
            train_mask = np.ones(self.split, dtype=bool)
            itm_mask = np.ones(len(immediate_exercise), dtype=bool)

        # Initialize continuation values to 0 (prevents noisy extrapolation for OTM paths)
        nb_paths = current_state.shape[0]
        continuation_values = np.zeros(nb_paths)

        # Only compute continuation values for ITM paths when train_ITM_only=True
        if itm_mask.sum() > 0:
            # Evaluate basis functions only for ITM paths
            X_itm = current_state[itm_mask]
            X_tensor = torch.from_numpy(X_itm).type(torch.float32)
            basis_itm = self.reservoir(X_tensor).detach().numpy()

            # Add constant term (intercept)
            basis_itm = np.concatenate([basis_itm, np.ones((len(basis_itm), 1))], axis=1)

            # Least squares regression on training ITM paths
            if train_mask.sum() > 0:
                train_itm_mask = train_mask & itm_mask[:self.split]
                if train_itm_mask.sum() > 0:
                    # Evaluate basis for training ITM paths
                    X_train_itm = current_state[train_itm_mask]
                    X_train_tensor = torch.from_numpy(X_train_itm).type(torch.float32)
                    basis_train = self.reservoir(X_train_tensor).detach().numpy()
                    basis_train = np.concatenate([basis_train, np.ones((len(basis_train), 1))], axis=1)

                    coefficients = np.linalg.lstsq(
                        basis_train,
                        future_values[train_itm_mask],
                        rcond=None
                    )[0]

                    # Predict continuation values only for ITM paths
                    continuation_values[itm_mask] = np.dot(basis_itm, coefficients)

        return continuation_values