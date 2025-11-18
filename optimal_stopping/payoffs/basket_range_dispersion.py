"""
Range & Dispersion basket option payoffs (d > 1).

Range options are PATH-DEPENDENT (need max/min over time).
Dispersion options are NOT path-dependent (based on current prices only).
"""

import numpy as np
from .payoff import Payoff


class RangeCall(Payoff):
    """Range Call: max(0, [max_i(S_i) - min_i(S_i)] - K)"""
    abbreviation = "Range-BskCall"
    is_path_dependent = True

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks, nb_dates+1)"""
        # Compute range over all stocks and all time
        max_over_all = np.max(X, axis=(1, 2))
        min_over_all = np.min(X, axis=(1, 2))
        range_value = max_over_all - min_over_all
        return np.maximum(0, range_value - self.strike)


class RangePut(Payoff):
    """Range Put: max(0, K - [max_i(S_i) - min_i(S_i)])"""
    abbreviation = "Range-BskPut"
    is_path_dependent = True

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks, nb_dates+1)"""
        # Compute range over all stocks and all time
        max_over_all = np.max(X, axis=(1, 2))
        min_over_all = np.min(X, axis=(1, 2))
        range_value = max_over_all - min_over_all
        return np.maximum(0, self.strike - range_value)


class DispersionCall(Payoff):
    """Dispersion Call: max(0, σ(t) - K) where σ is std dev of prices"""
    abbreviation = "Disp-BskCall"
    is_path_dependent = False

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks)"""
        # Compute mean price across stocks for each path
        mean_price = np.mean(X, axis=1, keepdims=True)  # (nb_paths, 1)

        # Compute standard deviation of prices
        std_dev = np.sqrt(np.mean((X - mean_price) ** 2, axis=1))  # (nb_paths,)

        return np.maximum(0, std_dev - self.strike)


class DispersionPut(Payoff):
    """Dispersion Put: max(0, K - σ(t)) where σ is std dev of prices"""
    abbreviation = "Disp-BskPut"
    is_path_dependent = False

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks)"""
        # Compute mean price across stocks for each path
        mean_price = np.mean(X, axis=1, keepdims=True)  # (nb_paths, 1)

        # Compute standard deviation of prices
        std_dev = np.sqrt(np.mean((X - mean_price) ** 2, axis=1))  # (nb_paths,)

        return np.maximum(0, self.strike - std_dev)
