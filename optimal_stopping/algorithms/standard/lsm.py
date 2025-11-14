"""
Least Squares Monte Carlo (LSM) for American option pricing.

Implementation of the Least Squares Monte Carlo introduced in
(Valuing American Options by Simulation: A Simple Least-Squares Approach,
Longstaff and Schwartz, 2001).

This is a benchmark algorithm using polynomial regression.
"""

import numpy as np
import time
import math
from optimal_stopping.run import configs
from optimal_stopping.algorithms.utils import basis_functions


class LSM:
    """
    Least Squares Monte Carlo pricer.

    Uses polynomial basis functions (degree 2) for regression.
    """

    def __init__(self, model, payoff, train_ITM_only=True, use_payoff_as_input=False, **kwargs):
        """
        Initialize LSM pricer.

        Args:
            model: Stock model
            payoff: Payoff function
            train_ITM_only: If True, only use in-the-money paths for training
            use_payoff_as_input: If True, include payoff in state
        """
        self.model = model
        self.payoff = payoff
        self.train_ITM_only = train_ITM_only
        self.use_payoff_as_input = use_payoff_as_input
        self.use_var = getattr(model, 'return_var', False)

        # Initialize basis functions
        state_size = model.nb_stocks * (1 + self.use_var) + self.use_payoff_as_input * 1
        self.bf = basis_functions.BasisFunctions(state_size)
        self.weights = np.zeros(self.bf.nb_base_fcts)

    def price(self, train_eval_split=2):
        """
        Compute option price using LSM.

        Args:
            train_eval_split: Ratio for splitting paths into training/evaluation

        Returns:
            tuple: (price, time_for_path_generation)
        """
        t_start = time.time()

        # Generate paths
        if configs.path_gen_seed.get_seed() is not None:
            np.random.seed(configs.path_gen_seed.get_seed())

        stock_paths, var_paths = self.model.generate_paths()
        time_for_path_gen = time.time() - t_start

        # Calculate payoffs
        payoffs = self.payoff(stock_paths)

        # Split data
        self.split = int(len(stock_paths) / train_eval_split)

        # Discount factor
        nb_dates = self.model.nb_dates
        deltaT = self.model.maturity / nb_dates
        discount_factor = math.exp(-self.model.rate * deltaT)

        # Initialize continuation values with payoffs at maturity
        continuation_value = payoffs[:, -1].copy()

        # Backward induction
        for t in range(nb_dates - 1, 0, -1):
            # Discount continuation value
            continuation_value = discount_factor * continuation_value

            # Current stock prices
            stock_at_t = stock_paths[:, :, t]

            # Add variance if needed
            if self.use_var:
                stock_at_t = np.concatenate([stock_at_t, var_paths[:, :, t]], axis=1)

            # Add payoff if needed
            if self.use_payoff_as_input:
                stock_at_t = np.concatenate([stock_at_t, payoffs[:, t:t+1]], axis=1)

            # Immediate exercise value
            immediate_exercise = payoffs[:, t]

            # Determine in-the-money paths
            if self.train_ITM_only:
                itm = immediate_exercise[:self.split] > 0
            else:
                itm = immediate_exercise[:self.split] < np.infty

            # Regression on ITM paths
            if np.sum(itm) > 0:
                X_train = stock_at_t[:self.split][itm]
                y_train = continuation_value[:self.split][itm]

                # Build basis matrix
                basis_matrix = np.array([[self.bf.base_fct(i, x) for i in range(self.bf.nb_base_fcts)]
                                        for x in X_train])

                # Least squares
                self.weights = np.linalg.lstsq(basis_matrix, y_train, rcond=None)[0]

                # Predict continuation values for all paths
                basis_matrix_all = np.array([[self.bf.base_fct(i, x) for i in range(self.bf.nb_base_fcts)]
                                             for x in stock_at_t])
                predicted_continuation = basis_matrix_all @ self.weights
            else:
                predicted_continuation = np.zeros(len(stock_at_t))

            # Exercise decision: exercise if immediate > continuation
            exercise = immediate_exercise > predicted_continuation
            continuation_value = np.where(exercise, immediate_exercise, continuation_value)

        # Final discounting to t=0
        continuation_value = discount_factor * continuation_value

        # Value at t=0 is max(immediate exercise, expected continuation)
        option_value = np.maximum(payoffs[:, 0], continuation_value)

        # Calculate price (average over evaluation paths)
        price = np.mean(option_value[self.split:])

        return price, time_for_path_gen
