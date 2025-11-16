"""
Simple single asset option payoffs (d = 1) - NOT path-dependent.

Standard European call and put options on a single underlying.
"""

import numpy as np
from .payoff import Payoff


class Call(Payoff):
    """European Call: max(0, S - K)"""
    abbreviation = "Call"
    is_path_dependent = False

    def eval(self, X):
        """X shape: (nb_paths, 1) or (nb_paths,)"""
        if X.ndim > 1:
            X = X[:, 0]  # Extract first (and only) stock
        return np.maximum(0, X - self.strike)


class Put(Payoff):
    """European Put: max(0, K - S)"""
    abbreviation = "Put"
    is_path_dependent = False

    def eval(self, X):
        """X shape: (nb_paths, 1) or (nb_paths,)"""
        if X.ndim > 1:
            X = X[:, 0]
        return np.maximum(0, self.strike - X)
