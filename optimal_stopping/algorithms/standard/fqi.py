"""
Fitted Q-Iteration (FQI) for American option pricing.

Implementation of fitted Q-Iteration introduced in
(Regression methods for pricing complex American-style options,
Tsitsiklis and Van Roy, 2001).

This is a benchmark algorithm using Q-learning with polynomial basis functions.
"""

import numpy as np
import time
import math
from optimal_stopping.run import configs
from optimal_stopping.algorithms.utils import basis_functions


class FQI:
    """
    Fitted Q-Iteration pricer.

    Uses Q-learning with polynomial basis functions.
    """

    def __init__(self, model, payoff, nb_epochs=20, train_ITM_only=True,
                 use_payoff_as_input=False, **kwargs):
        """
        Initialize FQI pricer.

        Args:
            model: Stock model
            payoff: Payoff function
            nb_epochs: Number of Q-iteration epochs
            train_ITM_only: If True, only use in-the-money paths for training
            use_payoff_as_input: If True, include payoff in state
        """
        self.model = model
        self.payoff = payoff
        self.nb_epochs = nb_epochs
        self.train_ITM_only = train_ITM_only
        self.use_payoff_as_input = use_payoff_as_input
        self.use_var = getattr(model, 'return_var', False)

        # Initialize basis functions (with time features)
        state_size = model.nb_stocks * (1 + self.use_var) + 2 + self.use_payoff_as_input * 1  # +2 for time features
        self.bf = basis_functions.BasisFunctions(state_size)
        self.weights = np.zeros(self.bf.nb_base_fcts)

    def _evaluate_bases_all(self, stock_paths, payoffs=None):
        """
        Evaluate all basis functions for all paths and timesteps.

        Args:
            stock_paths: Stock price paths [nb_paths, nb_stocks, nb_dates+1]
            payoffs: Optional payoffs [nb_paths, nb_dates+1]

        Returns:
            Basis function evaluations [nb_paths, nb_dates+1, nb_base_fcts]
        """
        nb_paths, nb_stocks, nb_dates_plus = stock_paths.shape
        nb_dates = nb_dates_plus - 1

        # Time features - shape [1, 1, nb_dates+1] broadcast to [nb_paths, 1, nb_dates+1]
        time_vals = np.linspace(0, 1, nb_dates_plus)[np.newaxis, np.newaxis, :]  # [1, 1, nb_dates+1]
        time = np.broadcast_to(time_vals, (nb_paths, 1, nb_dates_plus))  # [nb_paths, 1, nb_dates+1]
        time_complement = 1 - time  # [nb_paths, 1, nb_dates+1]

        # Concatenate stock paths with time features
        stocks = np.concatenate([stock_paths, time, time_complement], axis=1)  # [nb_paths, nb_stocks+2, nb_dates+1]

        # Transpose for easier processing
        stocks = np.transpose(stocks, (1, 0, 2))  # [nb_stocks+2, nb_paths, nb_dates+1]

        # Evaluate basis functions
        bf_values = np.zeros((nb_paths, nb_dates_plus, self.bf.nb_base_fcts))
        for i in range(self.bf.nb_base_fcts):
            # Evaluate for all paths and timesteps at once
            # stocks[:, :, t] has shape [nb_stocks+2, nb_paths]
            for t in range(nb_dates_plus):
                state = stocks[:, :, t].T  # [nb_paths, nb_stocks+2]
                bf_values[:, t, i] = np.array([self.bf.base_fct(i, s) for s in state])

        return bf_values

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

        stock_paths, var_paths = self.model.generate_paths()
        time_for_path_gen = time.time() - t_start

        # Calculate payoffs
        payoffs = self.payoff(stock_paths)

        # Add variance if needed
        if self.use_var:
            paths = np.concatenate([stock_paths, var_paths], axis=1)
        else:
            paths = stock_paths

        # Add payoff if needed
        if self.use_payoff_as_input:
            paths = np.concatenate([paths, np.expand_dims(payoffs, axis=1)], axis=1)

        # Split data
        self.split = int(len(stock_paths) / train_eval_split)

        # Discount factor
        nb_dates = self.model.nb_dates
        deltaT = self.model.maturity / nb_dates
        discount_factor = math.exp(-self.model.rate * deltaT)

        # Evaluate basis functions for all paths and timesteps
        eval_bases = self._evaluate_bases_all(paths, payoffs)

        # Fitted Q-Iteration
        for _ in range(self.nb_epochs):
            # Predict continuation values at t=1,...,nb_dates
            continuation_value = eval_bases[:self.split, 1:, :] @ self.weights  # [split, nb_dates]

            # Q-values: max(payoff, continuation)
            q_values = np.maximum(payoffs[:self.split, 1:], continuation_value)

            # Build regression matrices
            # X: basis at t=0,...,nb_dates-1
            # y: discounted Q-values at t=1,...,nb_dates
            basis_train = eval_bases[:self.split, :-1, :]  # [split, nb_dates, nb_base_fcts]
            discounted_q = discount_factor * q_values  # [split, nb_dates]

            # Flatten and solve
            X = basis_train.reshape(-1, self.bf.nb_base_fcts)  # [split*nb_dates, nb_base_fcts]
            y = discounted_q.reshape(-1)  # [split*nb_dates]

            # Solve normal equations
            matrixU = X.T @ X
            vectorV = X.T @ y
            self.weights = np.linalg.solve(matrixU, vectorV)

        # Predict continuation values for all paths
        if self.train_ITM_only:
            continuation_value = np.maximum(eval_bases @ self.weights, 0)
        else:
            continuation_value = eval_bases @ self.weights

        # Determine exercise times
        exercise_indicator = (payoffs > continuation_value).astype(int)
        exercise_indicator[:, -1] = 1  # Always exercise at maturity
        exercise_indicator[:, 0] = 0   # Never exercise at t=0

        # Find first exercise time for each path
        exercise_times = np.argmax(exercise_indicator, axis=1)

        # Calculate prices
        prices = np.array([payoffs[i, t] * discount_factor**t
                          for i, t in enumerate(exercise_times)])

        # Average over evaluation paths
        price = np.mean(prices[self.split:])
        price = max(price, payoffs[0, 0])

        return price, time_for_path_gen
