"""
Range & Dispersion basket option payoffs (d > 1).

Range options are PATH-DEPENDENT (need max/min over time).
Dispersion options are NOT path-dependent (based on current prices only).
"""

import numpy as np
from .payoff import Payoff


class RangeCall(Payoff):
    """Range Basket Call: max(0, [max_i(S_i(t)/S_i(0)) - min_i(S_i(t)/S_i(0))] - K)

    Note: This is PATH-DEPENDENT because it evaluates over the full path.
    """
    abbreviation = "Range-BskCall"
    is_path_dependent = True

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks, nb_dates+1)"""
        # Get initial prices
        initial_prices = self._get_initial_prices(X)

        # Extract current prices
        if X.ndim == 3:
            current_prices = X[:, :, -1]
        else:
            current_prices = X

        # Normalize and compute range
        normalized_returns = current_prices / initial_prices
        range_value = np.max(normalized_returns, axis=1) - np.min(normalized_returns, axis=1)
        return np.maximum(0, range_value - self.strike)


class RangePut(Payoff):
    """Range Basket Put: max(0, K - [max_i(S_i(t)/S_i(0)) - min_i(S_i(t)/S_i(0))])

    PATH-DEPENDENT.
    """
    abbreviation = "Range-BskPut"
    is_path_dependent = True

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks, nb_dates+1)"""
        # Get initial prices
        initial_prices = self._get_initial_prices(X)

        # Extract current prices
        if X.ndim == 3:
            current_prices = X[:, :, -1]
        else:
            current_prices = X

        # Normalize and compute range
        normalized_returns = current_prices / initial_prices
        range_value = np.max(normalized_returns, axis=1) - np.min(normalized_returns, axis=1)
        return np.maximum(0, self.strike - range_value)


class DispersionCall(Payoff):
    """Dispersion Basket Call: max(0, std(S_i(t)/S_i(0)) - K)

    NOT path-dependent - uses current prices only.
    Measures standard deviation of normalized returns across stocks.
    """
    abbreviation = "Disp-BskCall"
    is_path_dependent = False

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks) for standard, (nb_paths, nb_stocks, nb_dates+1) for path-dependent"""
        # Get initial prices
        initial_prices = self._get_initial_prices(X)

        # Extract current prices
        if X.ndim == 3:
            current_prices = X[:, :, -1]
        else:
            current_prices = X

        # Normalize by initial prices
        normalized_returns = current_prices / initial_prices

        # Compute mean return across stocks for each path
        mean_return = np.mean(normalized_returns, axis=1, keepdims=True)  # (nb_paths, 1)

        # Compute standard deviation of normalized returns
        deviations_sq = (normalized_returns - mean_return) ** 2
        sigma_norm = np.sqrt(np.mean(deviations_sq, axis=1))  # (nb_paths,)

        return np.maximum(0, sigma_norm - self.strike)


class DispersionPut(Payoff):
    """Dispersion Basket Put: max(0, K - std(S_i(t)/S_i(0)))

    NOT path-dependent - uses current prices only.
    Measures standard deviation of normalized returns across stocks.
    """
    abbreviation = "Disp-BskPut"
    is_path_dependent = False

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks) for standard, (nb_paths, nb_stocks, nb_dates+1) for path-dependent"""
        # Get initial prices
        initial_prices = self._get_initial_prices(X)

        # Extract current prices
        if X.ndim == 3:
            current_prices = X[:, :, -1]
        else:
            current_prices = X

        # Normalize by initial prices
        normalized_returns = current_prices / initial_prices

        # Compute mean return across stocks for each path
        mean_return = np.mean(normalized_returns, axis=1, keepdims=True)

        # Compute standard deviation of normalized returns
        deviations_sq = (normalized_returns - mean_return) ** 2
        sigma_norm = np.sqrt(np.mean(deviations_sq, axis=1))

        return np.maximum(0, self.strike - sigma_norm)
