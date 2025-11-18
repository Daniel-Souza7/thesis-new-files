"""
Rank-based basket option payoffs (d > 1) - NOT path-dependent.

These options rank stocks by normalized returns R_i = S_i(t)/S_i(0)
and apply weights/selections based on performance ranking.
"""

import numpy as np
from .payoff import Payoff


class BestOfKCall(Payoff):
    """Best-of-K Basket Call: max(0, mean(top_k_returns) - K)

    Ranks stocks by normalized returns R_i = S_i(t)/S_i(0), selects top k performers,
    and averages their returns.

    Uses parameter k from self.params (default k=2).
    """
    abbreviation = "BestK-BskCall"
    is_path_dependent = False

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks) for standard, (nb_paths, nb_stocks, nb_dates+1) for path-dependent"""
        k = self.params.get('k', 2)

        # Get initial prices
        initial_prices = self._get_initial_prices(X)

        # Extract current prices
        if X.ndim == 3:
            current_prices = X[:, :, -1]  # Last timestep
        else:
            current_prices = X

        # Normalize by initial prices to get returns
        normalized_returns = current_prices / initial_prices  # (nb_paths, nb_stocks)

        # Sort returns in descending order and take top k
        sorted_returns = np.sort(normalized_returns, axis=1)[:, ::-1]  # Descending
        best_k_returns = sorted_returns[:, :k]  # Take top k

        # Average the top k returns
        avg_best_k = np.mean(best_k_returns, axis=1)

        return np.maximum(0, avg_best_k - self.strike)


class WorstOfKPut(Payoff):
    """Worst-of-K Basket Put: max(0, K - mean(bottom_k_returns))

    Ranks stocks by normalized returns R_i = S_i(t)/S_i(0), selects bottom k performers,
    and averages their returns.

    Uses parameter k from self.params (default k=2).
    """
    abbreviation = "WorstK-BskPut"
    is_path_dependent = False

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks) for standard, (nb_paths, nb_stocks, nb_dates+1) for path-dependent"""
        k = self.params.get('k', 2)

        # Get initial prices
        initial_prices = self._get_initial_prices(X)

        # Extract current prices
        if X.ndim == 3:
            current_prices = X[:, :, -1]  # Last timestep
        else:
            current_prices = X

        # Normalize by initial prices to get returns
        normalized_returns = current_prices / initial_prices

        # Sort returns in ascending order and take bottom k
        sorted_returns = np.sort(normalized_returns, axis=1)  # Ascending
        worst_k_returns = sorted_returns[:, :k]  # Take bottom k

        # Average the worst k returns
        avg_worst_k = np.mean(worst_k_returns, axis=1)

        return np.maximum(0, self.strike - avg_worst_k)


class RankWeightedBasketCall(Payoff):
    """Rank-Weighted Basket Call: max(0, sum(w_i * R_(i)) - K)

    Ranks stocks by normalized returns R_i = S_i(t)/S_i(0) in descending order,
    then applies weights to the ranked returns.

    Weights: w_i = (d+1-i) / sum(1..d) where R_(1) >= R_(2) >= ... >= R_(d)
    Custom weights can be provided via self.params['weights'].
    """
    abbreviation = "Rank-BskCall"
    is_path_dependent = False

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks) for standard, (nb_paths, nb_stocks, nb_dates+1) for path-dependent"""
        # Get initial prices
        initial_prices = self._get_initial_prices(X)

        # Extract current prices
        if X.ndim == 3:
            current_prices = X[:, :, -1]  # Last timestep
            nb_paths, nb_stocks = current_prices.shape
        else:
            current_prices = X
            nb_paths, nb_stocks = X.shape

        weights = self.params.get('weights', None)

        # Normalize by initial prices
        normalized_returns = current_prices / initial_prices

        # Sort returns in descending order (best to worst)
        sorted_indices = np.argsort(normalized_returns, axis=1)[:, ::-1]  # Descending indices

        # Get sorted returns for each path
        sorted_returns = np.take_along_axis(normalized_returns, sorted_indices, axis=1)

        # Apply weights (highest return gets highest weight)
        if weights is None:
            # Default weights: w_i = (d+1-i) / sum(1..d)
            weights = np.arange(nb_stocks, 0, -1) / np.sum(np.arange(1, nb_stocks + 1))

        weights = np.array(weights)
        weighted_sum = np.sum(sorted_returns * weights, axis=1)

        return np.maximum(0, weighted_sum - self.strike)


class RankWeightedBasketPut(Payoff):
    """Rank-Weighted Basket Put: max(0, K - sum(w_i * R_(i)))

    Ranks stocks by normalized returns R_i = S_i(t)/S_i(0) in descending order,
    then applies weights to the ranked returns.

    Same weighting scheme as RankWeightedBasketCall.
    """
    abbreviation = "Rank-BskPut"
    is_path_dependent = False

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks) for standard, (nb_paths, nb_stocks, nb_dates+1) for path-dependent"""
        # Get initial prices
        initial_prices = self._get_initial_prices(X)

        # Extract current prices
        if X.ndim == 3:
            current_prices = X[:, :, -1]  # Last timestep
            nb_paths, nb_stocks = current_prices.shape
        else:
            current_prices = X
            nb_paths, nb_stocks = X.shape

        weights = self.params.get('weights', None)

        # Normalize by initial prices
        normalized_returns = current_prices / initial_prices

        # Sort returns in descending order
        sorted_indices = np.argsort(normalized_returns, axis=1)[:, ::-1]
        sorted_returns = np.take_along_axis(normalized_returns, sorted_indices, axis=1)

        # Apply weights
        if weights is None:
            weights = np.arange(nb_stocks, 0, -1) / np.sum(np.arange(1, nb_stocks + 1))

        weights = np.array(weights)
        weighted_sum = np.sum(sorted_returns * weights, axis=1)

        return np.maximum(0, self.strike - weighted_sum)
