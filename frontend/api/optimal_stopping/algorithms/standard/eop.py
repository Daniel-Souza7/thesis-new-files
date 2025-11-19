"""
European Option Price (EOP) - Benchmark algorithm.

This algorithm exercises ONLY at maturity, providing a lower bound for American option prices.
It's equivalent to pricing a European option using Monte Carlo.
"""

import numpy as np
import time
import math
from optimal_stopping.run import configs


class EuropeanOptionPrice:
    """
    Computes European option price by exercising only at maturity.

    This serves as a benchmark/lower bound for American option pricing algorithms.
    """

    def __init__(self, model, payoff, nb_epochs=None, hidden_size=None,
                 factors=None, train_ITM_only=None, use_payoff_as_input=None, **kwargs):
        """
        Initialize EOP pricer.

        Args:
            model: Stock model
            payoff: Payoff function
            nb_epochs: Ignored (for API compatibility)
            hidden_size: Ignored (for API compatibility)
            factors: Ignored (for API compatibility)
            train_ITM_only: Ignored (for API compatibility)
            use_payoff_as_input: Ignored (for API compatibility)
        """
        self.model = model
        self.payoff = payoff

    def price(self, train_eval_split=2):
        """
        Compute European option price by exercising only at maturity.

        Args:
            train_eval_split: Ratio for splitting paths (used for consistency)

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

        # Split for consistency with other algorithms
        self.split = len(stock_paths) // train_eval_split

        nb_paths, nb_stocks, nb_dates = stock_paths.shape

        # Discount factor from t=0 to maturity
        disc_factor = math.exp(-self.model.rate * self.model.maturity)

        # Exercise only at maturity (T)
        terminal_payoffs = payoffs[:, -1]

        # Track exercise dates (all paths exercise at maturity)
        self._exercise_dates = np.full(nb_paths, nb_dates - 1, dtype=int)

        # Discount terminal payoffs to present value
        discounted_payoffs = terminal_payoffs * disc_factor

        # Return average price on evaluation set
        price = np.mean(discounted_payoffs[self.split:])

        return price, time_path_gen

    def get_exercise_time(self):
        """Return average exercise time normalized to [0, 1] (evaluation set only)."""
        if not hasattr(self, '_exercise_dates'):
            return None

        nb_dates = self.model.nb_dates
        # Only use evaluation set paths (self.split:), not training paths
        normalized_times = self._exercise_dates[self.split:] / nb_dates
        return float(np.mean(normalized_times))
