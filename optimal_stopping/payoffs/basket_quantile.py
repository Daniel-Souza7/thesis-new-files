"""
Quantile basket option payoffs (d > 1) - PATH-DEPENDENT.

Quantile options use the α-quantile of the price distribution over time.
"""

import numpy as np
from .payoff import Payoff


class QuantileBasketCall(Payoff):
    """Quantile Basket Call: max(0, Q_α{S_1(t), ..., S_d(t)} - K)

    Q_α is the α-quantile of all stock prices over the full path.
    Uses parameter alpha from self.params (default alpha=0.95).
    PATH-DEPENDENT because it uses the distribution of prices over time.
    """
    abbreviation = "Quant-BskCall"
    is_path_dependent = True

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks, nb_dates+1)"""
        alpha = self.params.get('alpha', 0.95)

        # Flatten stocks and time dimensions: (nb_paths, nb_stocks * nb_dates)
        nb_paths, nb_stocks, nb_dates = X.shape
        X_flat = X.reshape(nb_paths, -1)

        # Compute α-quantile across all stocks and time for each path: (nb_paths,)
        quantile = np.quantile(X_flat, alpha, axis=1)

        return np.maximum(0, quantile - self.strike)


class QuantileBasketPut(Payoff):
    """Quantile Basket Put: max(0, K - Q_α{S_1(t), ..., S_d(t)})

    Q_α is the α-quantile of all stock prices over the full path.
    PATH-DEPENDENT.
    """
    abbreviation = "Quant-BskPut"
    is_path_dependent = True

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks, nb_dates+1)"""
        alpha = self.params.get('alpha', 0.95)

        nb_paths, nb_stocks, nb_dates = X.shape
        X_flat = X.reshape(nb_paths, -1)

        quantile = np.quantile(X_flat, alpha, axis=1)

        return np.maximum(0, self.strike - quantile)
