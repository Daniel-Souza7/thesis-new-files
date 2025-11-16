"""
Range single option payoffs (d = 1) - PATH-DEPENDENT.

Range options depend on the difference between max and min prices over time.
"""

import numpy as np
from .payoff import Payoff


class RangeCall_Single(Payoff):
    """Range Call (single): max(0, [max_over_time(S) - min_over_time(S)] - K)

    PATH-DEPENDENT because it uses max and min over the full path.
    """
    abbreviation = "Range-Call"
    is_path_dependent = True

    def eval(self, X):
        """X shape: (nb_paths, 1, nb_dates+1) or (nb_paths, nb_dates+1)"""
        if X.ndim == 3:
            X = X[:, 0, :]  # Extract single stock: (nb_paths, nb_dates+1)

        # Max over time: (nb_paths,)
        max_price = np.max(X, axis=1)
        # Min over time: (nb_paths,)
        min_price = np.min(X, axis=1)

        range_value = max_price - min_price
        return np.maximum(0, range_value - self.strike)


class RangePut_Single(Payoff):
    """Range Put (single): max(0, K - [max_over_time(S) - min_over_time(S)])

    PATH-DEPENDENT.
    """
    abbreviation = "Range-Put"
    is_path_dependent = True

    def eval(self, X):
        """X shape: (nb_paths, 1, nb_dates+1) or (nb_paths, nb_dates+1)"""
        if X.ndim == 3:
            X = X[:, 0, :]

        max_price = np.max(X, axis=1)
        min_price = np.min(X, axis=1)

        range_value = max_price - min_price
        return np.maximum(0, self.strike - range_value)
