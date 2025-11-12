"""
Lookback option payoffs (path-dependent).

Lookback options depend on the maximum or minimum asset price observed during the life of the option.
Compatible with SRLSM and SRFQI algorithms ONLY.

CRITICAL: These payoffs check the FULL PATH HISTORY to find historical extrema.

Lookback Types:
- Fixed Strike Lookback: Payoff based on best price seen vs fixed strike
- Floating Strike Lookback: Strike is determined by worst price seen

Implementations:
- LookbackFixedCall: max(M - K, 0) where M = max price seen
- LookbackFixedPut: max(K - m, 0) where m = min price seen
- LookbackFloatCall: max(S_T - m, 0) where m = min price seen, S_T = terminal price
- LookbackFloatPut: max(M - S_T, 0) where M = max price seen, S_T = terminal price

For multi-asset (basket) versions:
- Use arithmetic mean of stocks for terminal value
- Track max/min of the basket average over time

Total: 6 lookback options
"""

import numpy as np
from optimal_stopping.payoffs.payoff import Payoff


# ============================================================================
# FIXED STRIKE LOOKBACK OPTIONS - 2 payoffs
# ============================================================================

class LookbackFixedCall(Payoff):
    """
    Fixed strike lookback call.

    Payoff: max(M - K, 0) where M = max(max over all assets) over all time

    For d assets: M = max_{t,i} S_t^{(i)}
    """

    is_path_dependent = True

    def __init__(self, strike):
        super().__init__(strike)

    def eval(self, X):
        if X.ndim == 2:
            # Terminal evaluation: just check current max
            max_val = np.max(X, axis=1)
            return np.maximum(0, max_val - self.strike)
        elif X.ndim == 3:
            # Path-dependent: find maximum over all stocks AND all time
            # X shape: (nb_paths, nb_stocks, nb_dates)
            max_along_path = np.max(X, axis=(1, 2))  # Max over both stocks and time
            payoff = np.maximum(0, max_along_path - self.strike)
            return payoff
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


class LookbackFixedPut(Payoff):
    """
    Fixed strike lookback put.

    Payoff: max(K - m, 0) where m = min(min over all assets) over all time

    For d assets: m = min_{t,i} S_t^{(i)}
    """

    is_path_dependent = True

    def __init__(self, strike):
        super().__init__(strike)

    def eval(self, X):
        if X.ndim == 2:
            # Terminal evaluation: just check current min
            min_val = np.min(X, axis=1)
            return np.maximum(0, self.strike - min_val)
        elif X.ndim == 3:
            # Path-dependent: find minimum over all stocks AND all time
            min_along_path = np.min(X, axis=(1, 2))  # Min over both stocks and time
            payoff = np.maximum(0, self.strike - min_along_path)
            return payoff
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


# ============================================================================
# FLOATING STRIKE LOOKBACK OPTIONS - 2 payoffs
# ============================================================================

class LookbackFloatCall(Payoff):
    """
    Floating strike lookback call.

    Payoff: max(S_T - m, 0) where:
    - S_T = terminal value (mean of all assets at maturity)
    - m = min over all time (min of basket average over time)

    Strike is determined by the minimum basket value seen during the life.
    """

    is_path_dependent = True

    def __init__(self, strike=0):
        """
        Note: strike parameter is not used for floating lookback.
        It's kept for compatibility with base class.
        """
        super().__init__(strike)

    def eval(self, X):
        if X.ndim == 2:
            # Terminal evaluation: payoff is 0 (need history for floating strike)
            # Without history, we can't determine the floating strike
            return np.zeros(X.shape[0])
        elif X.ndim == 3:
            # Path-dependent evaluation
            # Terminal basket value
            terminal_mean = np.mean(X[:, :, -1], axis=1)

            # Minimum basket value over all time
            # For each time t, compute mean across stocks, then take min over time
            basket_values = np.mean(X, axis=1)  # (nb_paths, nb_dates)
            min_basket = np.min(basket_values, axis=1)  # (nb_paths,)

            # Payoff: terminal value minus minimum value seen
            payoff = np.maximum(0, terminal_mean - min_basket)
            return payoff
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


class LookbackFloatPut(Payoff):
    """
    Floating strike lookback put.

    Payoff: max(M - S_T, 0) where:
    - M = max over all time (max of basket average over time)
    - S_T = terminal value (mean of all assets at maturity)

    Strike is determined by the maximum basket value seen during the life.
    """

    is_path_dependent = True

    def __init__(self, strike=0):
        """
        Note: strike parameter is not used for floating lookback.
        It's kept for compatibility with base class.
        """
        super().__init__(strike)

    def eval(self, X):
        if X.ndim == 2:
            # Terminal evaluation: payoff is 0 (need history for floating strike)
            return np.zeros(X.shape[0])
        elif X.ndim == 3:
            # Path-dependent evaluation
            # Terminal basket value
            terminal_mean = np.mean(X[:, :, -1], axis=1)

            # Maximum basket value over all time
            basket_values = np.mean(X, axis=1)  # (nb_paths, nb_dates)
            max_basket = np.max(basket_values, axis=1)  # (nb_paths,)

            # Payoff: maximum value seen minus terminal value
            payoff = np.maximum(0, max_basket - terminal_mean)
            return payoff
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


# ============================================================================
# LOOKBACK ON MAX/MIN (Alternative formulations) - 2 payoffs
# ============================================================================

class LookbackMaxCall(Payoff):
    """
    Lookback call on the maximum asset.

    Payoff: max(max_T - K, 0) where max_T is the best (max across assets)
    value at terminal time, considering the path history.

    This is essentially: call on max(S_T^{(i)}) but with lookback feature
    that the payoff knows about historical maxima.
    """

    is_path_dependent = True

    def __init__(self, strike):
        super().__init__(strike)

    def eval(self, X):
        if X.ndim == 2:
            max_val = np.max(X, axis=1)
            return np.maximum(0, max_val - self.strike)
        elif X.ndim == 3:
            # Use the global maximum seen across all stocks and all time
            max_along_path = np.max(X, axis=(1, 2))
            payoff = np.maximum(0, max_along_path - self.strike)
            return payoff
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


class LookbackMinPut(Payoff):
    """
    Lookback put on the minimum asset.

    Payoff: max(K - min_T, 0) where min_T is the worst (min across assets)
    value at terminal time, considering the path history.
    """

    is_path_dependent = True

    def __init__(self, strike):
        super().__init__(strike)

    def eval(self, X):
        if X.ndim == 2:
            min_val = np.min(X, axis=1)
            return np.maximum(0, self.strike - min_val)
        elif X.ndim == 3:
            # Use the global minimum seen across all stocks and all time
            min_along_path = np.min(X, axis=(1, 2))
            payoff = np.maximum(0, self.strike - min_along_path)
            return payoff
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")