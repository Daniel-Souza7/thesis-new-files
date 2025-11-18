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
    """Dispersion Call: max(0, sum(|S_i - mean(S)|) - K)"""
    abbreviation = "Disp-BskCall"
    is_path_dependent = False

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks)"""
        mean_price = np.mean(X, axis=1, keepdims=True)
        dispersion = np.sum(np.abs(X - mean_price), axis=1)
        return np.maximum(0, dispersion - self.strike)


class DispersionPut(Payoff):
    """Dispersion Put: max(0, K - sum(|S_i - mean(S)|))"""
    abbreviation = "Disp-BskPut"
    is_path_dependent = False

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks)"""
        mean_price = np.mean(X, axis=1, keepdims=True)
        dispersion = np.sum(np.abs(X - mean_price), axis=1)
        return np.maximum(0, self.strike - dispersion)
