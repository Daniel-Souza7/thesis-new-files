"""
Standard option payoffs (non-path-dependent).

These options depend only on the current stock prices, not on the path history.
Compatible with RLSM, RFQI, LSM, FQI, NLSM, DOS algorithms.
"""

import numpy as np
from optimal_stopping.payoffs.payoff import Payoff


class BasketCall(Payoff):
    """
    Call option on arithmetic average of basket.

    Payoff: max(0, mean(S_i) - K)

    This is a STANDARD option - depends only on current prices.
    """

    is_path_dependent = False

    def __init__(self, strike):
        super().__init__(strike)

    def eval(self, X):
        """
        Evaluate basket call payoff.

        Args:
            X: Stock prices of shape (nb_paths, nb_stocks) or (nb_stocks,)

        Returns:
            payoffs: Array of shape (nb_paths,) or scalar
        """
        if X.ndim == 1:
            # Single path case: (nb_stocks,)
            basket_value = np.mean(X)
            return max(0, basket_value - self.strike)
        else:
            # Multiple paths: (nb_paths, nb_stocks)
            basket_values = np.mean(X, axis=1)
            return np.maximum(0, basket_values - self.strike)


class BasketPut(Payoff):
    """
    Put option on arithmetic average of basket.

    Payoff: max(0, K - mean(S_i))
    """

    is_path_dependent = False

    def __init__(self, strike):
        super().__init__(strike)

    def eval(self, X):
        """Evaluate basket put payoff."""
        if X.ndim == 1:
            basket_value = np.mean(X)
            return max(0, self.strike - basket_value)
        else:
            basket_values = np.mean(X, axis=1)
            return np.maximum(0, self.strike - basket_values)


class MaxCall(Payoff):
    """
    Call option on maximum of basket.

    Payoff: max(0, max(S_i) - K)
    """

    is_path_dependent = False

    def __init__(self, strike):
        super().__init__(strike)

    def eval(self, X):
        """Evaluate max call payoff."""
        if X.ndim == 1:
            max_value = np.max(X)
            return max(0, max_value - self.strike)
        else:
            max_values = np.max(X, axis=1)
            return np.maximum(0, max_values - self.strike)


class MaxPut(Payoff):
    """
    Put option on maximum of basket.

    Payoff: max(0, K - max(S_i))
    """

    is_path_dependent = False

    def __init__(self, strike):
        super().__init__(strike)

    def eval(self, X):
        """Evaluate max put payoff."""
        if X.ndim == 1:
            max_value = np.max(X)
            return max(0, self.strike - max_value)
        else:
            max_values = np.max(X, axis=1)
            return np.maximum(0, self.strike - max_values)


class GeometricBasketCall(Payoff):
    """
    Call option on geometric average of basket.

    Payoff: max(0, (∏S_i)^(1/d) - K)

    The geometric mean is less sensitive to outliers than arithmetic mean.
    """

    is_path_dependent = False

    def __init__(self, strike):
        super().__init__(strike)

    def eval(self, X):
        """Evaluate geometric basket call payoff."""
        if X.ndim == 1:
            # Single path: (nb_stocks,)
            geo_mean = np.exp(np.mean(np.log(X + 1e-10)))  # Small epsilon to avoid log(0)
            return max(0, geo_mean - self.strike)
        else:
            # Multiple paths: (nb_paths, nb_stocks)
            geo_means = np.exp(np.mean(np.log(X + 1e-10), axis=1))
            return np.maximum(0, geo_means - self.strike)


class GeometricBasketPut(Payoff):
    """
    Put option on geometric average of basket.

    Payoff: max(0, K - (∏S_i)^(1/d))
    """

    is_path_dependent = False

    def __init__(self, strike):
        super().__init__(strike)

    def eval(self, X):
        """Evaluate geometric basket put payoff."""
        if X.ndim == 1:
            geo_mean = np.exp(np.mean(np.log(X + 1e-10)))
            return max(0, self.strike - geo_mean)
        else:
            geo_means = np.exp(np.mean(np.log(X + 1e-10), axis=1))
            return np.maximum(0, self.strike - geo_means)


class MinCall(Payoff):
    """
    Call option on minimum of basket.

    Payoff: max(0, min(S_i) - K)

    This option pays based on the worst-performing asset.
    """

    is_path_dependent = False

    def __init__(self, strike):
        super().__init__(strike)

    def eval(self, X):
        """Evaluate min call payoff."""
        if X.ndim == 1:
            min_value = np.min(X)
            return max(0, min_value - self.strike)
        else:
            min_values = np.min(X, axis=1)
            return np.maximum(0, min_values - self.strike)


class MinPut(Payoff):
    """
    Put option on minimum of basket.

    Payoff: max(0, K - min(S_i))
    """

    is_path_dependent = False

    def __init__(self, strike):
        super().__init__(strike)

    def eval(self, X):
        """Evaluate min put payoff."""
        if X.ndim == 1:
            min_value = np.min(X)
            return max(0, self.strike - min_value)
        else:
            min_values = np.min(X, axis=1)
            return np.maximum(0, self.strike - min_values)