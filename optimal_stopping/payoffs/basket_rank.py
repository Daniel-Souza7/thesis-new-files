"""
Rank-based basket option payoffs (d > 1) - NOT path-dependent.

These options rank stocks by absolute prices S_i
and apply weights/selections based on price ranking.
"""

import numpy as np
from .payoff import Payoff


class BestOfKCall(Payoff):
    """Best-of-K Basket Call: max(0, mean(top_k_prices) - K)

    Ranks stocks by absolute prices S_i, selects top k performers,
    and averages their prices.

    Uses parameter k from self.params (default k=2).
    """
    abbreviation = "BestK-BskCall"
    is_path_dependent = False

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks)"""
        k = self.params.get('k', 2)

        # Sort prices in descending order and take top k
        sorted_prices = np.sort(X, axis=1)[:, ::-1]  # Descending
        best_k_prices = sorted_prices[:, :k]  # Take top k

        # Average the top k prices
        avg_best_k = np.mean(best_k_prices, axis=1)

        return np.maximum(0, avg_best_k - self.strike)


class WorstOfKPut(Payoff):
    """Worst-of-K Basket Put: max(0, K - mean(bottom_k_prices))

    Ranks stocks by absolute prices S_i, selects bottom k performers,
    and averages their prices.

    Uses parameter k from self.params (default k=2).
    """
    abbreviation = "WorstK-BskPut"
    is_path_dependent = False

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks)"""
        k = self.params.get('k', 2)

        # Sort prices in ascending order and take bottom k
        sorted_prices = np.sort(X, axis=1)  # Ascending
        worst_k_prices = sorted_prices[:, :k]  # Take bottom k

        # Average the worst k prices
        avg_worst_k = np.mean(worst_k_prices, axis=1)

        return np.maximum(0, self.strike - avg_worst_k)


class RankWeightedBasketCall(Payoff):
    """Rank-Weighted Basket Call: max(0, sum(w_i * S_(i)) - K)

    Ranks stocks by absolute prices S_i in descending order,
    then applies weights to the ranked prices.

    Weights: w_i = (d+1-i) / sum(1..d) where S_(1) >= S_(2) >= ... >= S_(d)
    Custom weights can be provided via self.params['weights'].
    """
    abbreviation = "Rank-BskCall"
    is_path_dependent = False

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks)"""
        nb_paths, nb_stocks = X.shape
        weights = self.params.get('weights', None)

        # Sort prices in descending order (highest to lowest)
        sorted_indices = np.argsort(X, axis=1)[:, ::-1]  # Descending indices

        # Get sorted prices for each path
        sorted_prices = np.take_along_axis(X, sorted_indices, axis=1)

        # Apply weights (highest price gets highest weight)
        if weights is None:
            # Default weights: w_i = (d+1-i) / sum(1..d)
            weights = np.arange(nb_stocks, 0, -1) / np.sum(np.arange(1, nb_stocks + 1))

        weights = np.array(weights)
        weighted_sum = np.sum(sorted_prices * weights, axis=1)

        return np.maximum(0, weighted_sum - self.strike)


class RankWeightedBasketPut(Payoff):
    """Rank-Weighted Basket Put: max(0, K - sum(w_i * S_(i)))

    Ranks stocks by absolute prices S_i in descending order,
    then applies weights to the ranked prices.

    Same weighting scheme as RankWeightedBasketCall.
    """
    abbreviation = "Rank-BskPut"
    is_path_dependent = False

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks)"""
        nb_paths, nb_stocks = X.shape
        weights = self.params.get('weights', None)

        # Sort prices in descending order
        sorted_indices = np.argsort(X, axis=1)[:, ::-1]
        sorted_prices = np.take_along_axis(X, sorted_indices, axis=1)

        # Apply weights
        if weights is None:
            weights = np.arange(nb_stocks, 0, -1) / np.sum(np.arange(1, nb_stocks + 1))

        weights = np.array(weights)
        weighted_sum = np.sum(sorted_prices * weights, axis=1)

        return np.maximum(0, self.strike - weighted_sum)
