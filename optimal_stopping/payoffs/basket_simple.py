"""
Simple basket option payoffs (d > 1) - NOT path-dependent.

These are the 6 standard multi-asset options that depend only on current prices.
"""

import numpy as np
from .payoff import Payoff


class BasketCall(Payoff):
    """Basket Call: max(0, mean(S) - K)"""
    abbreviation = "BskCall"
    is_path_dependent = False

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks)"""
        basket = np.mean(X, axis=1)  # Average across stocks
        return np.maximum(0, basket - self.strike)


class BasketPut(Payoff):
    """Basket Put: max(0, K - mean(S))"""
    abbreviation = "BskPut"
    is_path_dependent = False

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks)"""
        basket = np.mean(X, axis=1)
        return np.maximum(0, self.strike - basket)


class GeometricCall(Payoff):
    """Geometric Call: max(0, geom_mean(S) - K)"""
    abbreviation = "GeoCall"
    is_path_dependent = False

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks)"""
        # Geometric mean = (product of all stocks)^(1/d)
        geom_mean = np.exp(np.mean(np.log(X + 1e-10), axis=1))  # Add epsilon to avoid log(0)
        return np.maximum(0, geom_mean - self.strike)


class GeometricPut(Payoff):
    """Geometric Put: max(0, K - geom_mean(S))"""
    abbreviation = "GeoPut"
    is_path_dependent = False

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks)"""
        geom_mean = np.exp(np.mean(np.log(X + 1e-10), axis=1))
        return np.maximum(0, self.strike - geom_mean)


class MaxCall(Payoff):
    """Max Call: max(0, max(S) - K)"""
    abbreviation = "MaxCall"
    is_path_dependent = False

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks)"""
        max_stock = np.max(X, axis=1)
        return np.maximum(0, max_stock - self.strike)


class MinPut(Payoff):
    """Min Put: max(0, K - min(S))"""
    abbreviation = "MinPut"
    is_path_dependent = False

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks)"""
        min_stock = np.min(X, axis=1)
        return np.maximum(0, self.strike - min_stock)
