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
        nb_paths, nb_stocks = X.shape
        k = self.params.get('k', 2)

        # Validate k
        if not isinstance(k, (int, np.integer)):
            raise ValueError(f"k must be an integer, got {type(k).__name__}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if k > nb_stocks:
            raise ValueError(f"k ({k}) cannot exceed number of stocks ({nb_stocks})")

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
        nb_paths, nb_stocks = X.shape
        k = self.params.get('k', 2)

        # Validate k
        if not isinstance(k, (int, np.integer)):
            raise ValueError(f"k must be an integer, got {type(k).__name__}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if k > nb_stocks:
            raise ValueError(f"k ({k}) cannot exceed number of stocks ({nb_stocks})")

        # Sort prices in ascending order and take bottom k
        sorted_prices = np.sort(X, axis=1)  # Ascending
        worst_k_prices = sorted_prices[:, :k]  # Take bottom k

        # Average the worst k prices
        avg_worst_k = np.mean(worst_k_prices, axis=1)

        return np.maximum(0, self.strike - avg_worst_k)


class RankWeightedBasketCall(Payoff):
    """Rank-Weighted Basket Call: max(0, (1/k) * sum(w_i * S_(i)) - K) for i=1 to k

    Ranks stocks by absolute prices S_i in descending order,
    then applies user-specified weights to only the top k performers.

    Uses parameters k and weights from self.params:
    - k: number of top performers to include (default: 2)
    - weights: user-specified weights that apply to top k assets (default: equal weights summing to 1)

    Formula: (1/k) * sum_{i=1}^k (w_i * S_(i)) where S_(1) >= S_(2) >= ... >= S_(d)
    """
    abbreviation = "Rank-BskCall"
    is_path_dependent = False

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks)"""
        nb_paths, nb_stocks = X.shape

        # Get k and weights from params
        k = self.params.get('k', 2)
        weights = self.params.get('weights', None)

        # Validate k
        if not isinstance(k, (int, np.integer)):
            raise ValueError(f"k must be an integer, got {type(k).__name__}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if k > nb_stocks:
            raise ValueError(f"k ({k}) cannot exceed number of stocks ({nb_stocks})")

        # Default weights: equal weights that sum to 1
        if weights is None:
            weights = np.ones(k) / k
        weights = np.array(weights)

        # Validate weights
        if len(weights) != k:
            raise ValueError(f"Length of weights ({len(weights)}) must equal k ({k})")
        if np.any(weights < 0):
            raise ValueError(f"All weights must be non-negative, got min={np.min(weights)}")

        # Normalize weights to sum to 1 if they don't already
        weight_sum = np.sum(weights)
        if not np.isclose(weight_sum, 1.0):
            weights = weights / weight_sum

        # Sort prices in descending order (highest to lowest)
        sorted_prices = np.sort(X, axis=1)[:, ::-1]  # Descending

        # Take top k performers
        top_k_prices = sorted_prices[:, :k]

        # Apply weights and compute weighted sum
        weighted_sum = np.sum(top_k_prices * weights, axis=1)

        # Multiply by 1/k and subtract strike
        return np.maximum(0, (1.0 / k) * weighted_sum - self.strike)


class RankWeightedBasketPut(Payoff):
    """Rank-Weighted Basket Put: max(0, K - (1/k) * sum(w_i * S_(i))) for i=1 to k

    Ranks stocks by absolute prices S_i in descending order,
    then applies user-specified weights to only the top k performers.

    Uses parameters k and weights from self.params:
    - k: number of top performers to include (default: 2)
    - weights: user-specified weights that apply to top k assets (default: equal weights summing to 1)

    Formula: K - (1/k) * sum_{i=1}^k (w_i * S_(i)) where S_(1) >= S_(2) >= ... >= S_(d)
    """
    abbreviation = "Rank-BskPut"
    is_path_dependent = False

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks)"""
        nb_paths, nb_stocks = X.shape

        # Get k and weights from params
        k = self.params.get('k', 2)
        weights = self.params.get('weights', None)

        # Validate k
        if not isinstance(k, (int, np.integer)):
            raise ValueError(f"k must be an integer, got {type(k).__name__}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if k > nb_stocks:
            raise ValueError(f"k ({k}) cannot exceed number of stocks ({nb_stocks})")

        # Default weights: equal weights that sum to 1
        if weights is None:
            weights = np.ones(k) / k
        weights = np.array(weights)

        # Validate weights
        if len(weights) != k:
            raise ValueError(f"Length of weights ({len(weights)}) must equal k ({k})")
        if np.any(weights < 0):
            raise ValueError(f"All weights must be non-negative, got min={np.min(weights)}")

        # Normalize weights to sum to 1 if they don't already
        weight_sum = np.sum(weights)
        if not np.isclose(weight_sum, 1.0):
            weights = weights / weight_sum

        # Sort prices in descending order (highest to lowest)
        sorted_prices = np.sort(X, axis=1)[:, ::-1]  # Descending

        # Take top k performers
        top_k_prices = sorted_prices[:, :k]

        # Apply weights and compute weighted sum
        weighted_sum = np.sum(top_k_prices * weights, axis=1)

        # Multiply by 1/k
        return np.maximum(0, self.strike - (1.0 / k) * weighted_sum)
