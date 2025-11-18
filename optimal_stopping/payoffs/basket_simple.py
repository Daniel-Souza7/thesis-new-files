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
    """Max Call: max(0, max(S_i(t)/S_i(0)) - K)"""
    abbreviation = "MaxCall"
    is_path_dependent = False

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks) for standard, (nb_paths, nb_stocks, nb_dates+1) for path-dependent"""
        # Get initial prices (handles both standard and path-dependent contexts)
        initial_prices = self._get_initial_prices(X)

        # Extract current prices
        if X.ndim == 3:
            current_prices = X[:, :, -1]  # Last timestep for path-dependent
        else:
            current_prices = X  # Current timestep for standard

        # Normalize by initial prices
        normalized_returns = current_prices / initial_prices
        max_return = np.max(normalized_returns, axis=1)
        return np.maximum(0, max_return - self.strike)


class MinPut(Payoff):
    """Min Put: max(0, K - min(S_i(t)/S_i(0)))"""
    abbreviation = "MinPut"
    is_path_dependent = False

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks) for standard, (nb_paths, nb_stocks, nb_dates+1) for path-dependent"""
        # Get initial prices (handles both standard and path-dependent contexts)
        initial_prices = self._get_initial_prices(X)

        # Extract current prices
        if X.ndim == 3:
            current_prices = X[:, :, -1]  # Last timestep for path-dependent
        else:
            current_prices = X  # Current timestep for standard

        # Normalize by initial prices
        normalized_returns = current_prices / initial_prices
        min_return = np.min(normalized_returns, axis=1)
        return np.maximum(0, self.strike - min_return)
