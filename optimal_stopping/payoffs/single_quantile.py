"""
Quantile single option payoffs (d = 1) - PATH-DEPENDENT.

Quantile options use the α-quantile of the price distribution over time.
"""

import numpy as np
from .payoff import Payoff


class QuantileCall(Payoff):
    """Quantile Call (single): max(0, Q_α{S(t)} - K)

    Q_α is the α-quantile of stock prices over the full path.
    Uses parameter alpha from self.params (default alpha=0.95).
    PATH-DEPENDENT because it uses the distribution of prices over time.
    """
    abbreviation = "Quant-Call"
    is_path_dependent = True

    def eval(self, X):
        """X shape: (nb_paths, 1, nb_dates+1) or (nb_paths, nb_dates+1)"""
        alpha = self.params.get('alpha', 0.95)

        if X.ndim == 3:
            X = X[:, 0, :]  # Extract single stock: (nb_paths, nb_dates+1)

        # Compute α-quantile over time for each path: (nb_paths,)
        quantile = np.quantile(X, alpha, axis=1)

        return np.maximum(0, quantile - self.strike)


class QuantilePut(Payoff):
    """Quantile Put (single): max(0, K - Q_α{S(t)})

    Q_α is the α-quantile of stock prices over the full path.
    PATH-DEPENDENT.
    """
    abbreviation = "Quant-Put"
    is_path_dependent = True

    def eval(self, X):
        """X shape: (nb_paths, 1, nb_dates+1) or (nb_paths, nb_dates+1)"""
        alpha = self.params.get('alpha', 0.95)

        if X.ndim == 3:
            X = X[:, 0, :]

        quantile = np.quantile(X, alpha, axis=1)

        return np.maximum(0, self.strike - quantile)
