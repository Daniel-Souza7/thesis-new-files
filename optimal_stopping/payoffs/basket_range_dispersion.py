"""
Range & Dispersion basket option payoffs (d > 1).

Range options are PATH-DEPENDENT (need max/min over time).
Dispersion options are NOT path-dependent (based on current prices only).
"""

import numpy as np
from .payoff import Payoff


class RangeCall(Payoff):
    """Range Basket Call: max(0, [max_i(S_i) - min_i(S_i)] - K)

    Note: This is PATH-DEPENDENT because max/min are computed over the full path.
    """
    abbreviation = "Range-BskCall"
    is_path_dependent = True

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks, nb_dates+1)"""
        # Max over all stocks and all time: (nb_paths,)
        max_over_path = np.max(X, axis=(1, 2))
        # Min over all stocks and all time: (nb_paths,)
        min_over_path = np.min(X, axis=(1, 2))
        range_value = max_over_path - min_over_path
        return np.maximum(0, range_value - self.strike)


class RangePut(Payoff):
    """Range Basket Put: max(0, K - [max_i(S_i) - min_i(S_i)])

    PATH-DEPENDENT.
    """
    abbreviation = "Range-BskPut"
    is_path_dependent = True

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks, nb_dates+1)"""
        max_over_path = np.max(X, axis=(1, 2))
        min_over_path = np.min(X, axis=(1, 2))
        range_value = max_over_path - min_over_path
        return np.maximum(0, self.strike - range_value)


class DispersionCall(Payoff):
    """Dispersion Basket Call: max(0, sum(S_i - mean(S)))

    NOT path-dependent - uses current prices only.
    """
    abbreviation = "Disp-BskCall"
    is_path_dependent = False

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks)"""
        # Mean across stocks: (nb_paths,)
        basket_mean = np.mean(X, axis=1)
        # Dispersion = sum of deviations
        dispersion = np.sum(X - basket_mean[:, np.newaxis], axis=1)
        return np.maximum(0, dispersion - self.strike)


class DispersionPut(Payoff):
    """Dispersion Basket Put: max(0, K - sum(S_i - mean(S)))

    NOT path-dependent - uses current prices only.
    """
    abbreviation = "Disp-BskPut"
    is_path_dependent = False

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks)"""
        basket_mean = np.mean(X, axis=1)
        dispersion = np.sum(X - basket_mean[:, np.newaxis], axis=1)
        return np.maximum(0, self.strike - dispersion)
