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
                 factors=None, train_ITM_only=True, use_payoff_as_input=False):
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
        """
        self.model = model
        self.payoff = payoff
        self.train_ITM_only = train_ITM_only
        self.use_payoff_as_input = use_payoff_as_input

        # Check for variance paths
        self.use_var = getattr(model, 'return_var', False)

        # Initialize basis functions (degree 2 polynomials)
        state_size = model.nb_stocks * (1 + self.use_var) + self.use_payoff_as_input * 1
        self.basis = basis_functions.BasisFunctions(state_size)
        self.nb_base_fcts = self.basis.nb_base_fcts

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

        # Track exercise dates (initialize to maturity)
        self._exercise_dates = np.full(nb_paths, self.model.nb_dates - 1, dtype=int)

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

            # Discount future values
            discounted_values = values * disc_factor

            # Learn continuation value using least squares regression
            continuation_values = self._learn_continuation(
                current_state, discounted_values, immediate_exercise
            )

            # Update values: max(exercise now, continue)
            exercise_now = immediate_exercise > continuation_values
            values = np.where(exercise_now, immediate_exercise, discounted_values)

            # Track exercise dates - only update if exercising earlier
            self._exercise_dates[exercise_now] = date

        # Final price: average over evaluation paths
        price = np.mean(values[self.split:])

        return price, time_path_gen

    def _learn_continuation(self, current_state, future_values, immediate_exercise):
        """
        Learn continuation value using least squares regression on polynomial basis.

        Args:
            current_state: (nb_paths, state_size) - Current stock prices (+ payoff if used)
            future_values: (nb_paths,) - Discounted future values
            immediate_exercise: (nb_paths,) - Immediate payoff if exercised

        Returns:
            continuation_values: (nb_paths,) - Estimated continuation values
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

        return continuation_values

    def get_exercise_time(self):
        """Return average exercise time normalized to [0, 1] (evaluation set only)."""
        if not hasattr(self, '_exercise_dates'):
            return None

        nb_dates = self.model.nb_dates
        # Only use evaluation set paths (self.split:), not training paths
        normalized_times = self._exercise_dates[self.split:] / nb_dates
        return float(np.mean(normalized_times))


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
