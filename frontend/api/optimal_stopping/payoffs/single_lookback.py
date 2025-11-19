"""
Lookback single option payoffs (d = 1) - PATH-DEPENDENT.

Lookback options depend on the maximum or minimum price reached over the path.
"""

import warnings
import numpy as np
from .payoff import Payoff


class LookbackFixedCall(Payoff):
    """Lookback Fixed Strike Call: max(0, max_over_time(S) - K)"""
    abbreviation = "LBFi-Call"
    is_path_dependent = True

    def eval(self, X):
        """X shape: (nb_paths, 1, nb_dates+1) or (nb_paths, nb_dates+1)"""
        if X.ndim == 3:
            if X.shape[1] > 1:
                warnings.warn(
                    f"LookbackFixedCall is a single-asset payoff but received {X.shape[1]} stocks. "
                    f"Using only stock 0. Set nb_stocks=1 to avoid this warning.",
                    UserWarning,
                    stacklevel=2
                )
            X = X[:, 0, :]  # Extract single stock: (nb_paths, nb_dates+1)

        # Maximum over time: (nb_paths,)
        max_price = np.max(X, axis=1)
        return np.maximum(0, max_price - self.strike)


class LookbackFixedPut(Payoff):
    """Lookback Fixed Strike Put: max(0, K - min_over_time(S))"""
    abbreviation = "LBFi-Put"
    is_path_dependent = True

    def eval(self, X):
        """X shape: (nb_paths, 1, nb_dates+1) or (nb_paths, nb_dates+1)"""
        if X.ndim == 3:
            if X.shape[1] > 1:
                warnings.warn(
                    f"LookbackFixedPut is a single-asset payoff but received {X.shape[1]} stocks. "
                    f"Using only stock 0. Set nb_stocks=1 to avoid this warning.",
                    UserWarning,
                    stacklevel=2
                )
            X = X[:, 0, :]

        # Minimum over time: (nb_paths,)
        min_price = np.min(X, axis=1)
        return np.maximum(0, self.strike - min_price)


class LookbackFloatCall(Payoff):
    """Lookback Floating Strike Call: max(0, S(T) - min_over_time(S))"""
    abbreviation = "LBFl-Call"
    is_path_dependent = True

    def eval(self, X):
        """X shape: (nb_paths, 1, nb_dates+1) or (nb_paths, nb_dates+1)"""
        if X.ndim == 3:
            if X.shape[1] > 1:
                warnings.warn(
                    f"LookbackFloatCall is a single-asset payoff but received {X.shape[1]} stocks. "
                    f"Using only stock 0. Set nb_stocks=1 to avoid this warning.",
                    UserWarning,
                    stacklevel=2
                )
            X = X[:, 0, :]

        # Price at maturity: (nb_paths,)
        final_price = X[:, -1]
        # Minimum over time: (nb_paths,)
        min_price = np.min(X, axis=1)

        return np.maximum(0, final_price - min_price)


class LookbackFloatPut(Payoff):
    """Lookback Floating Strike Put: max(0, max_over_time(S) - S(T))"""
    abbreviation = "LBFl-Put"
    is_path_dependent = True

    def eval(self, X):
        """X shape: (nb_paths, 1, nb_dates+1) or (nb_paths, nb_dates+1)"""
        if X.ndim == 3:
            if X.shape[1] > 1:
                warnings.warn(
                    f"LookbackFloatPut is a single-asset payoff but received {X.shape[1]} stocks. "
                    f"Using only stock 0. Set nb_stocks=1 to avoid this warning.",
                    UserWarning,
                    stacklevel=2
                )
            X = X[:, 0, :]

        # Price at maturity: (nb_paths,)
        final_price = X[:, -1]
        # Maximum over time: (nb_paths,)
        max_price = np.max(X, axis=1)

        return np.maximum(0, max_price - final_price)
