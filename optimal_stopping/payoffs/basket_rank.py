"""
Rank-based basket option payoffs (d > 1) - NOT path-dependent.

These options rank stocks by current price and apply weights/selections.
"""

import numpy as np
from .payoff import Payoff


class BestOfKCall(Payoff):
    """Best-of-K Basket Call: max(0, mean(top_k_stocks) - K)

    Uses parameter k from self.params (default k=2).
    """
    abbreviation = "BestK-BskCall"
    is_path_dependent = False

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks)"""
        k = self.params.get('k', 2)

        # Sort stocks in descending order: (nb_paths, nb_stocks)
        sorted_stocks = -np.sort(-X, axis=1)  # Negative sort for descending

        # Take top k stocks: (nb_paths, k)
        top_k = sorted_stocks[:, :k]

        # Average of top k: (nb_paths,)
        avg_top_k = np.mean(top_k, axis=1)

        return np.maximum(0, avg_top_k - self.strike)


class WorstOfKPut(Payoff):
    """Worst-of-K Basket Put: max(0, K - mean(bottom_k_stocks))

    Uses parameter k from self.params (default k=2).
    """
    abbreviation = "WorstK-BskPut"
    is_path_dependent = False

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks)"""
        k = self.params.get('k', 2)

        # Sort stocks in ascending order: (nb_paths, nb_stocks)
        sorted_stocks = np.sort(X, axis=1)

        # Take bottom k stocks: (nb_paths, k)
        bottom_k = sorted_stocks[:, :k]

        # Average of bottom k: (nb_paths,)
        avg_bottom_k = np.mean(bottom_k, axis=1)

        return np.maximum(0, self.strike - avg_bottom_k)


class RankWeightedBasketCall(Payoff):
    """Rank-Weighted Basket Call: max(0, sum(w_i * S_(i)) - K)

    Weights: w_i = (d+1-i) / sum(1..d) where S_(1) >= S_(2) >= ... >= S_(d)
    Custom weights can be provided via self.params['weights'].
    """
    abbreviation = "Rank-BskCall"
    is_path_dependent = False

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks)"""
        nb_paths, nb_stocks = X.shape

        # Get weights
        weights = self.params.get('weights', None)
        if weights is None:
            # Default: linear decreasing weights
            # w_i = (d+1-i) / (1+2+...+d) = (d+1-i) / (d*(d+1)/2)
            ranks = np.arange(1, nb_stocks + 1)
            weights = (nb_stocks + 1 - ranks) / (nb_stocks * (nb_stocks + 1) / 2)

        # Sort stocks in descending order: (nb_paths, nb_stocks)
        sorted_stocks = -np.sort(-X, axis=1)

        # Apply weights: (nb_paths,)
        weighted_sum = np.sum(sorted_stocks * weights, axis=1)

        return np.maximum(0, weighted_sum - self.strike)


class RankWeightedBasketPut(Payoff):
    """Rank-Weighted Basket Put: max(0, K - sum(w_i * S_(i)))

    Same weighting scheme as RankWeightedBasketCall.
    """
    abbreviation = "Rank-BskPut"
    is_path_dependent = False

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks)"""
        nb_paths, nb_stocks = X.shape

        weights = self.params.get('weights', None)
        if weights is None:
            ranks = np.arange(1, nb_stocks + 1)
            weights = (nb_stocks + 1 - ranks) / (nb_stocks * (nb_stocks + 1) / 2)

        sorted_stocks = -np.sort(-X, axis=1)
        weighted_sum = np.sum(sorted_stocks * weights, axis=1)

        return np.maximum(0, self.strike - weighted_sum)
