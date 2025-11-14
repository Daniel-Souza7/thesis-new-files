"""
Advanced Double Barrier Options (10 payoffs)

These options have TWO barriers (upper and lower) and various knock-in/knock-out conditions.
All are path-dependent and use basket averaging at maturity.

Notation:
- B_U = Upper barrier (barrier_up)
- B_L = Lower barrier (barrier_down)
- S̄(t) = mean(S_1(t), ..., S_d(t)) = basket average at maturity
- max_{τ,i} S_i(τ) = maximum stock price across all paths and times
- min_{τ,i} S_i(τ) = minimum stock price across all paths and times
"""

import numpy as np
from optimal_stopping.payoffs.payoff import Payoff


class DoubleKnockOutCall(Payoff):
    """
    Double Knock-Out Call (DKO-Call)

    Pays max(S̄(T) - K, 0) if price stays BETWEEN both barriers throughout.
    Knocks out if price touches either barrier.

    Condition: B_L < min_{τ,i} S_i(τ) AND max_{τ,i} S_i(τ) < B_U
    """

    is_path_dependent = True

    def __init__(self, strike, barrier_up=None, barrier_down=None):
        super().__init__(strike)
        self.barrier_up = barrier_up if barrier_up is not None else strike * 1.2
        self.barrier_down = barrier_down if barrier_down is not None else strike * 0.8

    def eval(self, X):
        """
        Args:
            X: Array of shape (nb_paths, nb_stocks, nb_dates+1) or (nb_paths, nb_dates+1)
        """
        if X.ndim == 3:
            # Check if price stayed within barriers
            max_price = np.max(X, axis=(1, 2))  # Max across stocks and time
            min_price = np.min(X, axis=(1, 2))  # Min across stocks and time

            stays_in_range = (min_price >= self.barrier_down) & (max_price <= self.barrier_up)

            # Terminal basket payoff
            terminal_basket = np.mean(X[:, :, -1], axis=1)
            payoff = np.maximum(0, terminal_basket - self.strike)

            return payoff * stays_in_range
        else:
            # Single stock case
            max_price = np.max(X, axis=1)
            min_price = np.min(X, axis=1)
            stays_in_range = (min_price >= self.barrier_down) & (max_price <= self.barrier_up)
            payoff = np.maximum(0, X[:, -1] - self.strike)
            return payoff * stays_in_range


class DoubleKnockOutPut(Payoff):
    """
    Double Knock-Out Put (DKO-Put)

    Pays max(K - S̄(T), 0) if price stays BETWEEN both barriers throughout.
    """

    is_path_dependent = True

    def __init__(self, strike, barrier_up=None, barrier_down=None):
        super().__init__(strike)
        self.barrier_up = barrier_up if barrier_up is not None else strike * 1.2
        self.barrier_down = barrier_down if barrier_down is not None else strike * 0.8

    def eval(self, X):
        if X.ndim == 3:
            max_price = np.max(X, axis=(1, 2))
            min_price = np.min(X, axis=(1, 2))
            stays_in_range = (min_price >= self.barrier_down) & (max_price <= self.barrier_up)

            terminal_basket = np.mean(X[:, :, -1], axis=1)
            payoff = np.maximum(0, self.strike - terminal_basket)
            return payoff * stays_in_range
        else:
            max_price = np.max(X, axis=1)
            min_price = np.min(X, axis=1)
            stays_in_range = (min_price >= self.barrier_down) & (max_price <= self.barrier_up)
            payoff = np.maximum(0, self.strike - X[:, -1])
            return payoff * stays_in_range


class DoubleKnockInCall(Payoff):
    """
    Double Knock-In Call (DKI-Call)

    Pays max(S̄(T) - K, 0) if price touches EITHER barrier.
    Opposite of DoubleKnockOut.

    Condition: min_{τ,i} S_i(τ) ≤ B_L OR max_{τ,i} S_i(τ) ≥ B_U
    """

    is_path_dependent = True

    def __init__(self, strike, barrier_up=None, barrier_down=None):
        super().__init__(strike)
        self.barrier_up = barrier_up if barrier_up is not None else strike * 1.2
        self.barrier_down = barrier_down if barrier_down is not None else strike * 0.8

    def eval(self, X):
        if X.ndim == 3:
            max_price = np.max(X, axis=(1, 2))
            min_price = np.min(X, axis=(1, 2))

            # Knocked in if touches either barrier
            knocked_in = (min_price <= self.barrier_down) | (max_price >= self.barrier_up)

            terminal_basket = np.mean(X[:, :, -1], axis=1)
            payoff = np.maximum(0, terminal_basket - self.strike)
            return payoff * knocked_in
        else:
            max_price = np.max(X, axis=1)
            min_price = np.min(X, axis=1)
            knocked_in = (min_price <= self.barrier_down) | (max_price >= self.barrier_up)
            payoff = np.maximum(0, X[:, -1] - self.strike)
            return payoff * knocked_in


class DoubleKnockInPut(Payoff):
    """
    Double Knock-In Put (DKI-Put)

    Pays max(K - S̄(T), 0) if price touches EITHER barrier.
    """

    is_path_dependent = True

    def __init__(self, strike, barrier_up=None, barrier_down=None):
        super().__init__(strike)
        self.barrier_up = barrier_up if barrier_up is not None else strike * 1.2
        self.barrier_down = barrier_down if barrier_down is not None else strike * 0.8

    def eval(self, X):
        if X.ndim == 3:
            max_price = np.max(X, axis=(1, 2))
            min_price = np.min(X, axis=(1, 2))
            knocked_in = (min_price <= self.barrier_down) | (max_price >= self.barrier_up)

            terminal_basket = np.mean(X[:, :, -1], axis=1)
            payoff = np.maximum(0, self.strike - terminal_basket)
            return payoff * knocked_in
        else:
            max_price = np.max(X, axis=1)
            min_price = np.min(X, axis=1)
            knocked_in = (min_price <= self.barrier_down) | (max_price >= self.barrier_up)
            payoff = np.maximum(0, self.strike - X[:, -1])
            return payoff * knocked_in


class UpInDownOutCall(Payoff):
    """
    Up-In-Down-Out Call (UIDO-Call)

    Pays max(S̄(T) - K, 0) if:
    - Price hits upper barrier (activates option), AND
    - Price never hits lower barrier (doesn't knock out)

    Condition: max_{τ,i} S_i(τ) ≥ B_U AND min_{τ,i} S_i(τ) > B_L
    """

    is_path_dependent = True

    def __init__(self, strike, barrier_up=None, barrier_down=None):
        super().__init__(strike)
        self.barrier_up = barrier_up if barrier_up is not None else strike * 1.2
        self.barrier_down = barrier_down if barrier_down is not None else strike * 0.8

    def eval(self, X):
        if X.ndim == 3:
            max_price = np.max(X, axis=(1, 2))
            min_price = np.min(X, axis=(1, 2))

            # Up-in AND down-out
            condition = (max_price >= self.barrier_up) & (min_price >= self.barrier_down)

            terminal_basket = np.mean(X[:, :, -1], axis=1)
            payoff = np.maximum(0, terminal_basket - self.strike)
            return payoff * condition
        else:
            max_price = np.max(X, axis=1)
            min_price = np.min(X, axis=1)
            condition = (max_price >= self.barrier_up) & (min_price >= self.barrier_down)
            payoff = np.maximum(0, X[:, -1] - self.strike)
            return payoff * condition


class UpInDownOutPut(Payoff):
    """
    Up-In-Down-Out Put (UIDO-Put)

    Pays max(K - S̄(T), 0) if price hits upper barrier but not lower.
    """

    is_path_dependent = True

    def __init__(self, strike, barrier_up=None, barrier_down=None):
        super().__init__(strike)
        self.barrier_up = barrier_up if barrier_up is not None else strike * 1.2
        self.barrier_down = barrier_down if barrier_down is not None else strike * 0.8

    def eval(self, X):
        if X.ndim == 3:
            max_price = np.max(X, axis=(1, 2))
            min_price = np.min(X, axis=(1, 2))
            condition = (max_price >= self.barrier_up) & (min_price >= self.barrier_down)

            terminal_basket = np.mean(X[:, :, -1], axis=1)
            payoff = np.maximum(0, self.strike - terminal_basket)
            return payoff * condition
        else:
            max_price = np.max(X, axis=1)
            min_price = np.min(X, axis=1)
            condition = (max_price >= self.barrier_up) & (min_price >= self.barrier_down)
            payoff = np.maximum(0, self.strike - X[:, -1])
            return payoff * condition


class UpOutDownInCall(Payoff):
    """
    Up-Out-Down-In Call (UODI-Call)

    Pays max(S̄(T) - K, 0) if:
    - Price hits lower barrier (activates option), AND
    - Price never hits upper barrier (doesn't knock out)

    Condition: max_{τ,i} S_i(τ) < B_U AND min_{τ,i} S_i(τ) ≤ B_L
    """

    is_path_dependent = True

    def __init__(self, strike, barrier_up=None, barrier_down=None):
        super().__init__(strike)
        self.barrier_up = barrier_up if barrier_up is not None else strike * 1.2
        self.barrier_down = barrier_down if barrier_down is not None else strike * 0.8

    def eval(self, X):
        if X.ndim == 3:
            max_price = np.max(X, axis=(1, 2))
            min_price = np.min(X, axis=(1, 2))

            # Down-in AND up-out
            condition = (max_price <= self.barrier_up) & (min_price <= self.barrier_down)

            terminal_basket = np.mean(X[:, :, -1], axis=1)
            payoff = np.maximum(0, terminal_basket - self.strike)
            return payoff * condition
        else:
            max_price = np.max(X, axis=1)
            min_price = np.min(X, axis=1)
            condition = (max_price <= self.barrier_up) & (min_price <= self.barrier_down)
            payoff = np.maximum(0, X[:, -1] - self.strike)
            return payoff * condition


class UpOutDownInPut(Payoff):
    """
    Up-Out-Down-In Put (UODI-Put)

    Pays max(K - S̄(T), 0) if price hits lower barrier but not upper.
    """

    is_path_dependent = True

    def __init__(self, strike, barrier_up=None, barrier_down=None):
        super().__init__(strike)
        self.barrier_up = barrier_up if barrier_up is not None else strike * 1.2
        self.barrier_down = barrier_down if barrier_down is not None else strike * 0.8

    def eval(self, X):
        if X.ndim == 3:
            max_price = np.max(X, axis=(1, 2))
            min_price = np.min(X, axis=(1, 2))
            condition = (max_price <= self.barrier_up) & (min_price <= self.barrier_down)

            terminal_basket = np.mean(X[:, :, -1], axis=1)
            payoff = np.maximum(0, self.strike - terminal_basket)
            return payoff * condition
        else:
            max_price = np.max(X, axis=1)
            min_price = np.min(X, axis=1)
            condition = (max_price <= self.barrier_up) & (min_price <= self.barrier_down)
            payoff = np.maximum(0, self.strike - X[:, -1])
            return payoff * condition


class PartialTimeBarrierCall(Payoff):
    """
    Partial Time Barrier Call (PTB-Call)

    Barrier only monitored during specific time window [T_1, T_2].
    For now: T_1 = 0, T_2 = maturity/2 (first half of option lifetime)

    Pays max(S̄(T) - K, 0) if max price in [0, T/2] stays below barrier.

    Condition: max_{τ ∈ [0,T/2],i} S_i(τ) < B
    """

    is_path_dependent = True

    def __init__(self, strike, barrier_up=None, barrier_down=None):
        super().__init__(strike)
        # Only use upper barrier for this payoff
        self.barrier = barrier_up if barrier_up is not None else strike * 1.2

    def eval(self, X):
        if X.ndim == 3:
            nb_dates = X.shape[2]
            halfway = nb_dates // 2

            # Only check barrier in first half of time period
            max_price_partial = np.max(X[:, :, :halfway], axis=(1, 2))
            stays_below = max_price_partial <= self.barrier

            terminal_basket = np.mean(X[:, :, -1], axis=1)
            payoff = np.maximum(0, terminal_basket - self.strike)
            return payoff * stays_below
        else:
            nb_dates = X.shape[1]
            halfway = nb_dates // 2
            max_price_partial = np.max(X[:, :halfway], axis=1)
            stays_below = max_price_partial <= self.barrier
            payoff = np.maximum(0, X[:, -1] - self.strike)
            return payoff * stays_below


class StepBarrierCall(Payoff):
    """
    Step Barrier Call (StepB-Call)

    Barrier level changes over time: B(τ) = B_0 + α·τ
    For simplicity: α = drift/5 (barrier grows slowly with drift)

    Pays max(S̄(T) - K, 0) if max price at each time τ stays below B(τ).

    Condition: ∀τ: max_i S_i(τ) < B(τ) where B(τ) = 1 + drift/5 · τ

    Note: This uses barrier_up as B_0 (initial barrier level)
    """

    is_path_dependent = True

    def __init__(self, strike, barrier_up=None, barrier_down=None, drift=0.02):
        super().__init__(strike)
        self.barrier_initial = barrier_up if barrier_up is not None else strike * 1.2
        self.drift = drift
        self.alpha = drift / 5.0  # Barrier growth rate

    def eval(self, X):
        if X.ndim == 3:
            nb_paths, nb_stocks, nb_dates = X.shape

            # Create time-varying barrier: B(t) = B_0 + α·t
            # Time normalized to [0, 1]
            times = np.linspace(0, 1, nb_dates)
            barriers = self.barrier_initial + self.alpha * times  # Shape: (nb_dates,)

            # Check if max stock price at each time stays below barrier
            max_price_per_time = np.max(X, axis=1)  # Shape: (nb_paths, nb_dates)

            # Compare against time-varying barrier
            stays_below = np.all(max_price_per_time <= barriers, axis=1)  # Shape: (nb_paths,)

            terminal_basket = np.mean(X[:, :, -1], axis=1)
            payoff = np.maximum(0, terminal_basket - self.strike)
            return payoff * stays_below
        else:
            nb_dates = X.shape[1]
            times = np.linspace(0, 1, nb_dates)
            barriers = self.barrier_initial + self.alpha * times
            stays_below = np.all(X <= barriers, axis=1)
            payoff = np.maximum(0, X[:, -1] - self.strike)
            return payoff * stays_below

class DoubleKnockOutLookbackFloatingCall(Payoff):
    """
    Double Knock-Out Lookback Floating Strike Call (DKO-LB-Float-Call)

    Lookback call with floating strike: pays max(S̄(T) - min_{τ,i} S_i(τ), 0)
    (buys at historical minimum, sells at terminal basket average)

    Pays only if price stays BETWEEN both barriers throughout.

    Condition: B_L < min_{τ,i} S_i(τ) AND max_{τ,i} S_i(τ) < B_U
    Payoff: max(S̄(T) - S_min, 0) where S_min = min_{τ,i} S_i(τ)
    """

    is_path_dependent = True

    def __init__(self, strike, barrier_up=None, barrier_down=None):
        super().__init__(strike)
        self.barrier_up = barrier_up if barrier_up is not None else strike * 1.2
        self.barrier_down = barrier_down if barrier_down is not None else strike * 0.8

    def eval(self, X):
        if X.ndim == 3:
            # Check if price stayed within barriers
            max_price = np.max(X, axis=(1, 2))
            min_price = np.min(X, axis=(1, 2))
            stays_in_range = (min_price >= self.barrier_down) & (max_price <= self.barrier_up)

            # Lookback floating strike: buy at minimum, sell at terminal
            terminal_basket = np.mean(X[:, :, -1], axis=1)
            payoff = np.maximum(0, terminal_basket - min_price)

            return payoff * stays_in_range
        else:
            # Single stock case
            max_price = np.max(X, axis=1)
            min_price = np.min(X, axis=1)
            stays_in_range = (min_price >= self.barrier_down) & (max_price <= self.barrier_up)
            payoff = np.maximum(0, X[:, -1] - min_price)
            return payoff * stays_in_range


class DoubleKnockOutLookbackFloatingPut(Payoff):
    """
    Double Knock-Out Lookback Floating Strike Put (DKO-LB-Float-Put)

    Lookback put with floating strike: pays max(max_{τ,i} S_i(τ) - S̄(T), 0)
    (sells at historical maximum, buys at terminal basket average)

    Pays only if price stays BETWEEN both barriers throughout.

    Condition: B_L < min_{τ,i} S_i(τ) AND max_{τ,i} S_i(τ) < B_U
    Payoff: max(S_max - S̄(T), 0) where S_max = max_{τ,i} S_i(τ)
    """

    is_path_dependent = True

    def __init__(self, strike, barrier_up=None, barrier_down=None):
        super().__init__(strike)
        self.barrier_up = barrier_up if barrier_up is not None else strike * 1.2
        self.barrier_down = barrier_down if barrier_down is not None else strike * 0.8

    def eval(self, X):
        if X.ndim == 3:
            # Check if price stayed within barriers
            max_price = np.max(X, axis=(1, 2))
            min_price = np.min(X, axis=(1, 2))
            stays_in_range = (min_price >= self.barrier_down) & (max_price <= self.barrier_up)

            # Lookback floating strike: sell at maximum, buy at terminal
            terminal_basket = np.mean(X[:, :, -1], axis=1)
            payoff = np.maximum(0, max_price - terminal_basket)

            return payoff * stays_in_range
        else:
            # Single stock case
            max_price = np.max(X, axis=1)
            min_price = np.min(X, axis=1)
            stays_in_range = (min_price >= self.barrier_down) & (max_price <= self.barrier_up)
            payoff = np.maximum(0, max_price - X[:, -1])
            return payoff * stays_in_range