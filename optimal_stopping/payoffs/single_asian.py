"""
Asian-style single option payoffs (d = 1) - PATH-DEPENDENT.

These options depend on the average price over time.
"""

import numpy as np
from .payoff import Payoff


class AsianFixedStrikeCall_Single(Payoff):
    """Asian Fixed Strike Call (single): max(0, avg_over_time(S) - K)"""
    abbreviation = "AsianFi-Call"
    is_path_dependent = True

    def eval(self, X):
        """X shape: (nb_paths, 1, nb_dates+1) or (nb_paths, nb_dates+1)"""
        if X.ndim == 3:
            X = X[:, 0, :]  # Extract single stock: (nb_paths, nb_dates+1)

        # Average over time: (nb_paths,)
        avg_price = np.mean(X, axis=1)
        return np.maximum(0, avg_price - self.strike)


class AsianFixedStrikePut_Single(Payoff):
    """Asian Fixed Strike Put (single): max(0, K - avg_over_time(S))"""
    abbreviation = "AsianFi-Put"
    is_path_dependent = True

    def eval(self, X):
        """X shape: (nb_paths, 1, nb_dates+1) or (nb_paths, nb_dates+1)"""
        if X.ndim == 3:
            X = X[:, 0, :]

        avg_price = np.mean(X, axis=1)
        return np.maximum(0, self.strike - avg_price)


class AsianFloatingStrikeCall_Single(Payoff):
    """Asian Floating Strike Call (single): max(0, S(T) - avg_over_time(S))"""
    abbreviation = "AsianFl-Call"
    is_path_dependent = True

    def eval(self, X):
        """X shape: (nb_paths, 1, nb_dates+1) or (nb_paths, nb_dates+1)"""
        if X.ndim == 3:
            X = X[:, 0, :]

        # Price at maturity: (nb_paths,)
        final_price = X[:, -1]
        # Average over time: (nb_paths,)
        avg_price = np.mean(X, axis=1)

        return np.maximum(0, final_price - avg_price)


class AsianFloatingStrikePut_Single(Payoff):
    """Asian Floating Strike Put (single): max(0, avg_over_time(S) - S(T))"""
    abbreviation = "AsianFl-Put"
    is_path_dependent = True

    def eval(self, X):
        """X shape: (nb_paths, 1, nb_dates+1) or (nb_paths, nb_dates+1)"""
        if X.ndim == 3:
            X = X[:, 0, :]

        final_price = X[:, -1]
        avg_price = np.mean(X, axis=1)

        return np.maximum(0, avg_price - final_price)
