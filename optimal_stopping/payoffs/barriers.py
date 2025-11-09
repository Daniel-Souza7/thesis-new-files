"""
Barrier option payoffs (path-dependent).

These options depend on whether a barrier level is hit during the option's life.
Compatible with SRLSM and SRFQI algorithms ONLY.

CRITICAL: These payoffs check the FULL PATH HISTORY including initial state at t=0.

Barrier Types:
- Up-and-Out: Knocked out if max(Si) >= B
- Down-and-Out: Knocked out if min(Si) <= B
- Up-and-In: Activated if max(Si) >= B
- Down-and-In: Activated if min(Si) <= B

Underlyings:
- Basket: Arithmetic mean of stocks
- Max: Maximum of stocks
- Min: Minimum of stocks
- GeometricBasket: Geometric mean of stocks
"""

import numpy as np
from optimal_stopping.payoffs.payoff import Payoff


# ============================================================================
# UP-AND-OUT BARRIERS (Knocked out if max(Si) >= B)
# ============================================================================

class UpAndOutBasketCall(Payoff):
    """Up-and-out barrier call on basket average."""

    is_path_dependent = True

    def __init__(self, strike, barrier=None):
        super().__init__(strike)
        self.barrier = barrier if barrier is not None else strike * 1.2

    def eval(self, X):
        if X.ndim == 2:
            terminal_mean = np.mean(X, axis=1)
            barrier_not_hit = np.max(X, axis=1) < self.barrier
            payoff = np.maximum(0, terminal_mean - self.strike)
            return payoff * barrier_not_hit
        elif X.ndim == 3:
            initial_max = np.max(X[:, :, 0], axis=1)
            initially_knocked_out = initial_max >= self.barrier
            max_along_path = np.max(X, axis=(1, 2))
            barrier_hit = max_along_path >= self.barrier
            knocked_out = initially_knocked_out | barrier_hit
            barrier_not_hit = ~knocked_out
            terminal_mean = np.mean(X[:, :, -1], axis=1)
            payoff = np.maximum(0, terminal_mean - self.strike)
            return payoff * barrier_not_hit
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


class UpAndOutBasketPut(Payoff):
    """Up-and-out barrier put on basket average."""

    is_path_dependent = True

    def __init__(self, strike, barrier=None):
        super().__init__(strike)
        self.barrier = barrier if barrier is not None else strike * 1.2

    def eval(self, X):
        if X.ndim == 2:
            terminal_mean = np.mean(X, axis=1)
            barrier_not_hit = np.max(X, axis=1) < self.barrier
            payoff = np.maximum(0, self.strike - terminal_mean)
            return payoff * barrier_not_hit
        elif X.ndim == 3:
            initial_max = np.max(X[:, :, 0], axis=1)
            initially_knocked_out = initial_max >= self.barrier
            max_along_path = np.max(X, axis=(1, 2))
            barrier_hit = max_along_path >= self.barrier
            knocked_out = initially_knocked_out | barrier_hit
            barrier_not_hit = ~knocked_out
            terminal_mean = np.mean(X[:, :, -1], axis=1)
            payoff = np.maximum(0, self.strike - terminal_mean)
            return payoff * barrier_not_hit
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


class UpAndOutMaxCall(Payoff):
    """Up-and-out barrier call on maximum."""

    is_path_dependent = True

    def __init__(self, strike, barrier=None):
        super().__init__(strike)
        self.barrier = barrier if barrier is not None else strike * 1.2

    def eval(self, X):
        if X.ndim == 2:
            terminal_max = np.max(X, axis=1)
            barrier_not_hit = terminal_max < self.barrier
            payoff = np.maximum(0, terminal_max - self.strike)
            return payoff * barrier_not_hit
        elif X.ndim == 3:
            initial_max = np.max(X[:, :, 0], axis=1)
            initially_knocked_out = initial_max >= self.barrier
            max_along_path = np.max(X, axis=(1, 2))
            barrier_hit = max_along_path >= self.barrier
            knocked_out = initially_knocked_out | barrier_hit
            barrier_not_hit = ~knocked_out
            terminal_max = np.max(X[:, :, -1], axis=1)
            payoff = np.maximum(0, terminal_max - self.strike)
            return payoff * barrier_not_hit
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


class UpAndOutMaxPut(Payoff):
    """Up-and-out barrier put on maximum."""

    is_path_dependent = True

    def __init__(self, strike, barrier=None):
        super().__init__(strike)
        self.barrier = barrier if barrier is not None else strike * 1.2

    def eval(self, X):
        if X.ndim == 2:
            terminal_max = np.max(X, axis=1)
            barrier_not_hit = terminal_max < self.barrier
            payoff = np.maximum(0, self.strike - terminal_max)
            return payoff * barrier_not_hit
        elif X.ndim == 3:
            initial_max = np.max(X[:, :, 0], axis=1)
            initially_knocked_out = initial_max >= self.barrier
            max_along_path = np.max(X, axis=(1, 2))
            barrier_hit = max_along_path >= self.barrier
            knocked_out = initially_knocked_out | barrier_hit
            barrier_not_hit = ~knocked_out
            terminal_max = np.max(X[:, :, -1], axis=1)
            payoff = np.maximum(0, self.strike - terminal_max)
            return payoff * barrier_not_hit
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


class UpAndOutGeometricBasketCall(Payoff):
    """Up-and-out barrier call on geometric basket."""

    is_path_dependent = True

    def __init__(self, strike, barrier=None):
        super().__init__(strike)
        self.barrier = barrier if barrier is not None else strike * 1.2

    def eval(self, X):
        if X.ndim == 2:
            geo_mean = np.exp(np.mean(np.log(X + 1e-10), axis=1))
            barrier_not_hit = np.max(X, axis=1) < self.barrier
            payoff = np.maximum(0, geo_mean - self.strike)
            return payoff * barrier_not_hit
        elif X.ndim == 3:
            initial_max = np.max(X[:, :, 0], axis=1)
            initially_knocked_out = initial_max >= self.barrier
            max_along_path = np.max(X, axis=(1, 2))
            barrier_hit = max_along_path >= self.barrier
            knocked_out = initially_knocked_out | barrier_hit
            barrier_not_hit = ~knocked_out
            geo_mean = np.exp(np.mean(np.log(X[:, :, -1] + 1e-10), axis=1))
            payoff = np.maximum(0, geo_mean - self.strike)
            return payoff * barrier_not_hit
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


class UpAndOutGeometricBasketPut(Payoff):
    """Up-and-out barrier put on geometric basket."""

    is_path_dependent = True

    def __init__(self, strike, barrier=None):
        super().__init__(strike)
        self.barrier = barrier if barrier is not None else strike * 1.2

    def eval(self, X):
        if X.ndim == 2:
            geo_mean = np.exp(np.mean(np.log(X + 1e-10), axis=1))
            barrier_not_hit = np.max(X, axis=1) < self.barrier
            payoff = np.maximum(0, self.strike - geo_mean)
            return payoff * barrier_not_hit
        elif X.ndim == 3:
            initial_max = np.max(X[:, :, 0], axis=1)
            initially_knocked_out = initial_max >= self.barrier
            max_along_path = np.max(X, axis=(1, 2))
            barrier_hit = max_along_path >= self.barrier
            knocked_out = initially_knocked_out | barrier_hit
            barrier_not_hit = ~knocked_out
            geo_mean = np.exp(np.mean(np.log(X[:, :, -1] + 1e-10), axis=1))
            payoff = np.maximum(0, self.strike - geo_mean)
            return payoff * barrier_not_hit
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


class UpAndOutMinCall(Payoff):
    """Up-and-out barrier call on minimum."""

    is_path_dependent = True

    def __init__(self, strike, barrier=None):
        super().__init__(strike)
        self.barrier = barrier if barrier is not None else strike * 1.2

    def eval(self, X):
        if X.ndim == 2:
            terminal_min = np.min(X, axis=1)
            barrier_not_hit = np.max(X, axis=1) < self.barrier
            payoff = np.maximum(0, terminal_min - self.strike)
            return payoff * barrier_not_hit
        elif X.ndim == 3:
            initial_max = np.max(X[:, :, 0], axis=1)
            initially_knocked_out = initial_max >= self.barrier
            max_along_path = np.max(X, axis=(1, 2))
            barrier_hit = max_along_path >= self.barrier
            knocked_out = initially_knocked_out | barrier_hit
            barrier_not_hit = ~knocked_out
            terminal_min = np.min(X[:, :, -1], axis=1)
            payoff = np.maximum(0, terminal_min - self.strike)
            return payoff * barrier_not_hit
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


class UpAndOutMinPut(Payoff):
    """Up-and-out barrier put on minimum."""

    is_path_dependent = True

    def __init__(self, strike, barrier=None):
        super().__init__(strike)
        self.barrier = barrier if barrier is not None else strike * 1.2

    def eval(self, X):
        if X.ndim == 2:
            terminal_min = np.min(X, axis=1)
            barrier_not_hit = np.max(X, axis=1) < self.barrier
            payoff = np.maximum(0, self.strike - terminal_min)
            return payoff * barrier_not_hit
        elif X.ndim == 3:
            initial_max = np.max(X[:, :, 0], axis=1)
            initially_knocked_out = initial_max >= self.barrier
            max_along_path = np.max(X, axis=(1, 2))
            barrier_hit = max_along_path >= self.barrier
            knocked_out = initially_knocked_out | barrier_hit
            barrier_not_hit = ~knocked_out
            terminal_min = np.min(X[:, :, -1], axis=1)
            payoff = np.maximum(0, self.strike - terminal_min)
            return payoff * barrier_not_hit
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


# ============================================================================
# DOWN-AND-OUT BARRIERS (Knocked out if min(Si) <= B)
# ============================================================================

class DownAndOutBasketCall(Payoff):
    """Down-and-out barrier call on basket average."""

    is_path_dependent = True

    def __init__(self, strike, barrier=None):
        super().__init__(strike)
        self.barrier = barrier if barrier is not None else strike * 0.8

    def eval(self, X):
        if X.ndim == 2:
            terminal_mean = np.mean(X, axis=1)
            barrier_not_hit = np.min(X, axis=1) > self.barrier
            payoff = np.maximum(0, terminal_mean - self.strike)
            return payoff * barrier_not_hit
        elif X.ndim == 3:
            initial_min = np.min(X[:, :, 0], axis=1)
            initially_knocked_out = initial_min <= self.barrier
            min_along_path = np.min(X, axis=(1, 2))
            barrier_hit = min_along_path <= self.barrier
            knocked_out = initially_knocked_out | barrier_hit
            barrier_not_hit = ~knocked_out
            terminal_mean = np.mean(X[:, :, -1], axis=1)
            payoff = np.maximum(0, terminal_mean - self.strike)
            return payoff * barrier_not_hit
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


class DownAndOutBasketPut(Payoff):
    """Down-and-out barrier put on basket average."""

    is_path_dependent = True

    def __init__(self, strike, barrier=None):
        super().__init__(strike)
        self.barrier = barrier if barrier is not None else strike * 0.8

    def eval(self, X):
        if X.ndim == 2:
            terminal_mean = np.mean(X, axis=1)
            barrier_not_hit = np.min(X, axis=1) > self.barrier
            payoff = np.maximum(0, self.strike - terminal_mean)
            return payoff * barrier_not_hit
        elif X.ndim == 3:
            initial_min = np.min(X[:, :, 0], axis=1)
            initially_knocked_out = initial_min <= self.barrier
            min_along_path = np.min(X, axis=(1, 2))
            barrier_hit = min_along_path <= self.barrier
            knocked_out = initially_knocked_out | barrier_hit
            barrier_not_hit = ~knocked_out
            terminal_mean = np.mean(X[:, :, -1], axis=1)
            payoff = np.maximum(0, self.strike - terminal_mean)
            return payoff * barrier_not_hit
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


class DownAndOutMaxCall(Payoff):
    """Down-and-out barrier call on maximum."""

    is_path_dependent = True

    def __init__(self, strike, barrier=None):
        super().__init__(strike)
        self.barrier = barrier if barrier is not None else strike * 0.8

    def eval(self, X):
        if X.ndim == 2:
            terminal_max = np.max(X, axis=1)
            barrier_not_hit = np.min(X, axis=1) > self.barrier
            payoff = np.maximum(0, terminal_max - self.strike)
            return payoff * barrier_not_hit
        elif X.ndim == 3:
            initial_min = np.min(X[:, :, 0], axis=1)
            initially_knocked_out = initial_min <= self.barrier
            min_along_path = np.min(X, axis=(1, 2))
            barrier_hit = min_along_path <= self.barrier
            knocked_out = initially_knocked_out | barrier_hit
            barrier_not_hit = ~knocked_out
            terminal_max = np.max(X[:, :, -1], axis=1)
            payoff = np.maximum(0, terminal_max - self.strike)
            return payoff * barrier_not_hit
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


class DownAndOutMaxPut(Payoff):
    """Down-and-out barrier put on maximum."""

    is_path_dependent = True

    def __init__(self, strike, barrier=None):
        super().__init__(strike)
        self.barrier = barrier if barrier is not None else strike * 0.8

    def eval(self, X):
        if X.ndim == 2:
            terminal_max = np.max(X, axis=1)
            barrier_not_hit = np.min(X, axis=1) > self.barrier
            payoff = np.maximum(0, self.strike - terminal_max)
            return payoff * barrier_not_hit
        elif X.ndim == 3:
            initial_min = np.min(X[:, :, 0], axis=1)
            initially_knocked_out = initial_min <= self.barrier
            min_along_path = np.min(X, axis=(1, 2))
            barrier_hit = min_along_path <= self.barrier
            knocked_out = initially_knocked_out | barrier_hit
            barrier_not_hit = ~knocked_out
            terminal_max = np.max(X[:, :, -1], axis=1)
            payoff = np.maximum(0, self.strike - terminal_max)
            return payoff * barrier_not_hit
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


class DownAndOutGeometricBasketCall(Payoff):
    """Down-and-out barrier call on geometric basket."""

    is_path_dependent = True

    def __init__(self, strike, barrier=None):
        super().__init__(strike)
        self.barrier = barrier if barrier is not None else strike * 0.8

    def eval(self, X):
        if X.ndim == 2:
            geo_mean = np.exp(np.mean(np.log(X + 1e-10), axis=1))
            barrier_not_hit = np.min(X, axis=1) > self.barrier
            payoff = np.maximum(0, geo_mean - self.strike)
            return payoff * barrier_not_hit
        elif X.ndim == 3:
            initial_min = np.min(X[:, :, 0], axis=1)
            initially_knocked_out = initial_min <= self.barrier
            min_along_path = np.min(X, axis=(1, 2))
            barrier_hit = min_along_path <= self.barrier
            knocked_out = initially_knocked_out | barrier_hit
            barrier_not_hit = ~knocked_out
            geo_mean = np.exp(np.mean(np.log(X[:, :, -1] + 1e-10), axis=1))
            payoff = np.maximum(0, geo_mean - self.strike)
            return payoff * barrier_not_hit
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


class DownAndOutGeometricBasketPut(Payoff):
    """Down-and-out barrier put on geometric basket."""

    is_path_dependent = True

    def __init__(self, strike, barrier=None):
        super().__init__(strike)
        self.barrier = barrier if barrier is not None else strike * 0.8

    def eval(self, X):
        if X.ndim == 2:
            geo_mean = np.exp(np.mean(np.log(X + 1e-10), axis=1))
            barrier_not_hit = np.min(X, axis=1) > self.barrier
            payoff = np.maximum(0, self.strike - geo_mean)
            return payoff * barrier_not_hit
        elif X.ndim == 3:
            initial_min = np.min(X[:, :, 0], axis=1)
            initially_knocked_out = initial_min <= self.barrier
            min_along_path = np.min(X, axis=(1, 2))
            barrier_hit = min_along_path <= self.barrier
            knocked_out = initially_knocked_out | barrier_hit
            barrier_not_hit = ~knocked_out
            geo_mean = np.exp(np.mean(np.log(X[:, :, -1] + 1e-10), axis=1))
            payoff = np.maximum(0, self.strike - geo_mean)
            return payoff * barrier_not_hit
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


class DownAndOutMinCall(Payoff):
    """Down-and-out barrier call on minimum."""

    is_path_dependent = True

    def __init__(self, strike, barrier=None):
        super().__init__(strike)
        self.barrier = barrier if barrier is not None else strike * 0.8

    def eval(self, X):
        if X.ndim == 2:
            terminal_min = np.min(X, axis=1)
            barrier_not_hit = terminal_min > self.barrier
            payoff = np.maximum(0, terminal_min - self.strike)
            return payoff * barrier_not_hit
        elif X.ndim == 3:
            initial_min = np.min(X[:, :, 0], axis=1)
            initially_knocked_out = initial_min <= self.barrier
            min_along_path = np.min(X, axis=(1, 2))
            barrier_hit = min_along_path <= self.barrier
            knocked_out = initially_knocked_out | barrier_hit
            barrier_not_hit = ~knocked_out
            terminal_min = np.min(X[:, :, -1], axis=1)
            payoff = np.maximum(0, terminal_min - self.strike)
            return payoff * barrier_not_hit
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


class DownAndOutMinPut(Payoff):
    """Down-and-out barrier put on minimum."""

    is_path_dependent = True

    def __init__(self, strike, barrier=None):
        super().__init__(strike)
        self.barrier = barrier if barrier is not None else strike * 0.8

    def eval(self, X):
        if X.ndim == 2:
            terminal_min = np.min(X, axis=1)
            barrier_not_hit = terminal_min > self.barrier
            payoff = np.maximum(0, self.strike - terminal_min)
            return payoff * barrier_not_hit
        elif X.ndim == 3:
            initial_min = np.min(X[:, :, 0], axis=1)
            initially_knocked_out = initial_min <= self.barrier
            min_along_path = np.min(X, axis=(1, 2))
            barrier_hit = min_along_path <= self.barrier
            knocked_out = initially_knocked_out | barrier_hit
            barrier_not_hit = ~knocked_out
            terminal_min = np.min(X[:, :, -1], axis=1)
            payoff = np.maximum(0, self.strike - terminal_min)
            return payoff * barrier_not_hit
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


# ============================================================================
# UP-AND-IN BARRIERS (Activated if max(Si) >= B)
# ============================================================================

class UpAndInBasketCall(Payoff):
    """Up-and-in barrier call on basket average."""

    is_path_dependent = True

    def __init__(self, strike, barrier=None):
        super().__init__(strike)
        self.barrier = barrier if barrier is not None else strike * 1.2

    def eval(self, X):
        if X.ndim == 2:
            terminal_mean = np.mean(X, axis=1)
            barrier_hit = np.max(X, axis=1) >= self.barrier
            payoff = np.maximum(0, terminal_mean - self.strike)
            return payoff * barrier_hit
        elif X.ndim == 3:
            initial_max = np.max(X[:, :, 0], axis=1)
            initially_knocked_in = initial_max >= self.barrier
            max_along_path = np.max(X, axis=(1, 2))
            barrier_hit = max_along_path >= self.barrier
            knocked_in = initially_knocked_in | barrier_hit
            terminal_mean = np.mean(X[:, :, -1], axis=1)
            payoff = np.maximum(0, terminal_mean - self.strike)
            return payoff * knocked_in
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


class UpAndInBasketPut(Payoff):
    """Up-and-in barrier put on basket average."""

    is_path_dependent = True

    def __init__(self, strike, barrier=None):
        super().__init__(strike)
        self.barrier = barrier if barrier is not None else strike * 1.2

    def eval(self, X):
        if X.ndim == 2:
            terminal_mean = np.mean(X, axis=1)
            barrier_hit = np.max(X, axis=1) >= self.barrier
            payoff = np.maximum(0, self.strike - terminal_mean)
            return payoff * barrier_hit
        elif X.ndim == 3:
            initial_max = np.max(X[:, :, 0], axis=1)
            initially_knocked_in = initial_max >= self.barrier
            max_along_path = np.max(X, axis=(1, 2))
            barrier_hit = max_along_path >= self.barrier
            knocked_in = initially_knocked_in | barrier_hit
            terminal_mean = np.mean(X[:, :, -1], axis=1)
            payoff = np.maximum(0, self.strike - terminal_mean)
            return payoff * knocked_in
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


class UpAndInMaxCall(Payoff):
    """Up-and-in barrier call on maximum."""

    is_path_dependent = True

    def __init__(self, strike, barrier=None):
        super().__init__(strike)
        self.barrier = barrier if barrier is not None else strike * 1.2

    def eval(self, X):
        if X.ndim == 2:
            terminal_max = np.max(X, axis=1)
            barrier_hit = terminal_max >= self.barrier
            payoff = np.maximum(0, terminal_max - self.strike)
            return payoff * barrier_hit
        elif X.ndim == 3:
            initial_max = np.max(X[:, :, 0], axis=1)
            initially_knocked_in = initial_max >= self.barrier
            max_along_path = np.max(X, axis=(1, 2))
            barrier_hit = max_along_path >= self.barrier
            knocked_in = initially_knocked_in | barrier_hit
            terminal_max = np.max(X[:, :, -1], axis=1)
            payoff = np.maximum(0, terminal_max - self.strike)
            return payoff * knocked_in
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


class UpAndInMaxPut(Payoff):
    """Up-and-in barrier put on maximum."""

    is_path_dependent = True

    def __init__(self, strike, barrier=None):
        super().__init__(strike)
        self.barrier = barrier if barrier is not None else strike * 1.2

    def eval(self, X):
        if X.ndim == 2:
            terminal_max = np.max(X, axis=1)
            barrier_hit = terminal_max >= self.barrier
            payoff = np.maximum(0, self.strike - terminal_max)
            return payoff * barrier_hit
        elif X.ndim == 3:
            initial_max = np.max(X[:, :, 0], axis=1)
            initially_knocked_in = initial_max >= self.barrier
            max_along_path = np.max(X, axis=(1, 2))
            barrier_hit = max_along_path >= self.barrier
            knocked_in = initially_knocked_in | barrier_hit
            terminal_max = np.max(X[:, :, -1], axis=1)
            payoff = np.maximum(0, self.strike - terminal_max)
            return payoff * knocked_in
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


class UpAndInGeometricBasketCall(Payoff):
    """Up-and-in barrier call on geometric basket."""

    is_path_dependent = True

    def __init__(self, strike, barrier=None):
        super().__init__(strike)
        self.barrier = barrier if barrier is not None else strike * 1.2

    def eval(self, X):
        if X.ndim == 2:
            geo_mean = np.exp(np.mean(np.log(X + 1e-10), axis=1))
            barrier_hit = np.max(X, axis=1) >= self.barrier
            payoff = np.maximum(0, geo_mean - self.strike)
            return payoff * barrier_hit
        elif X.ndim == 3:
            initial_max = np.max(X[:, :, 0], axis=1)
            initially_knocked_in = initial_max >= self.barrier
            max_along_path = np.max(X, axis=(1, 2))
            barrier_hit = max_along_path >= self.barrier
            knocked_in = initially_knocked_in | barrier_hit
            geo_mean = np.exp(np.mean(np.log(X[:, :, -1] + 1e-10), axis=1))
            payoff = np.maximum(0, geo_mean - self.strike)
            return payoff * knocked_in
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


class UpAndInGeometricBasketPut(Payoff):
    """Up-and-in barrier put on geometric basket."""

    is_path_dependent = True

    def __init__(self, strike, barrier=None):
        super().__init__(strike)
        self.barrier = barrier if barrier is not None else strike * 1.2

    def eval(self, X):
        if X.ndim == 2:
            geo_mean = np.exp(np.mean(np.log(X + 1e-10), axis=1))
            barrier_hit = np.max(X, axis=1) >= self.barrier
            payoff = np.maximum(0, self.strike - geo_mean)
            return payoff * barrier_hit
        elif X.ndim == 3:
            initial_max = np.max(X[:, :, 0], axis=1)
            initially_knocked_in = initial_max >= self.barrier
            max_along_path = np.max(X, axis=(1, 2))
            barrier_hit = max_along_path >= self.barrier
            knocked_in = initially_knocked_in | barrier_hit
            geo_mean = np.exp(np.mean(np.log(X[:, :, -1] + 1e-10), axis=1))
            payoff = np.maximum(0, self.strike - geo_mean)
            return payoff * knocked_in
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


# ============================================================================
# DOWN-AND-IN BARRIERS (Activated if min(Si) <= B)
# ============================================================================

class DownAndInBasketCall(Payoff):
    """Down-and-in barrier call on basket average."""

    is_path_dependent = True

    def __init__(self, strike, barrier=None):
        super().__init__(strike)
        self.barrier = barrier if barrier is not None else strike * 0.8

    def eval(self, X):
        if X.ndim == 2:
            terminal_mean = np.mean(X, axis=1)
            barrier_hit = np.min(X, axis=1) <= self.barrier
            payoff = np.maximum(0, terminal_mean - self.strike)
            return payoff * barrier_hit
        elif X.ndim == 3:
            initial_min = np.min(X[:, :, 0], axis=1)
            initially_knocked_in = initial_min <= self.barrier
            min_along_path = np.min(X, axis=(1, 2))
            barrier_hit = min_along_path <= self.barrier
            knocked_in = initially_knocked_in | barrier_hit
            terminal_mean = np.mean(X[:, :, -1], axis=1)
            payoff = np.maximum(0, terminal_mean - self.strike)
            return payoff * knocked_in
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


class DownAndInBasketPut(Payoff):
    """Down-and-in barrier put on basket average."""

    is_path_dependent = True

    def __init__(self, strike, barrier=None):
        super().__init__(strike)
        self.barrier = barrier if barrier is not None else strike * 0.8

    def eval(self, X):
        if X.ndim == 2:
            terminal_mean = np.mean(X, axis=1)
            barrier_hit = np.min(X, axis=1) <= self.barrier
            payoff = np.maximum(0, self.strike - terminal_mean)
            return payoff * barrier_hit
        elif X.ndim == 3:
            initial_min = np.min(X[:, :, 0], axis=1)
            initially_knocked_in = initial_min <= self.barrier
            min_along_path = np.min(X, axis=(1, 2))
            barrier_hit = min_along_path <= self.barrier
            knocked_in = initially_knocked_in | barrier_hit
            terminal_mean = np.mean(X[:, :, -1], axis=1)
            payoff = np.maximum(0, self.strike - terminal_mean)
            return payoff * knocked_in
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


class DownAndInMaxCall(Payoff):
    """Down-and-in barrier call on maximum."""

    is_path_dependent = True

    def __init__(self, strike, barrier=None):
        super().__init__(strike)
        self.barrier = barrier if barrier is not None else strike * 0.8

    def eval(self, X):
        if X.ndim == 2:
            terminal_max = np.max(X, axis=1)
            barrier_hit = np.min(X, axis=1) <= self.barrier
            payoff = np.maximum(0, terminal_max - self.strike)
            return payoff * barrier_hit
        elif X.ndim == 3:
            initial_min = np.min(X[:, :, 0], axis=1)
            initially_knocked_in = initial_min <= self.barrier
            min_along_path = np.min(X, axis=(1, 2))
            barrier_hit = min_along_path <= self.barrier
            knocked_in = initially_knocked_in | barrier_hit
            terminal_max = np.max(X[:, :, -1], axis=1)
            payoff = np.maximum(0, terminal_max - self.strike)
            return payoff * knocked_in
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


class DownAndInMaxPut(Payoff):
    """Down-and-in barrier put on maximum."""

    is_path_dependent = True

    def __init__(self, strike, barrier=None):
        super().__init__(strike)
        self.barrier = barrier if barrier is not None else strike * 0.8

    def eval(self, X):
        if X.ndim == 2:
            terminal_max = np.max(X, axis=1)
            barrier_hit = np.min(X, axis=1) <= self.barrier
            payoff = np.maximum(0, self.strike - terminal_max)
            return payoff * barrier_hit
        elif X.ndim == 3:
            initial_min = np.min(X[:, :, 0], axis=1)
            initially_knocked_in = initial_min <= self.barrier
            min_along_path = np.min(X, axis=(1, 2))
            barrier_hit = min_along_path <= self.barrier
            knocked_in = initially_knocked_in | barrier_hit
            terminal_max = np.max(X[:, :, -1], axis=1)
            payoff = np.maximum(0, self.strike - terminal_max)
            return payoff * knocked_in
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


class DownAndInGeometricBasketCall(Payoff):
    """Down-and-in barrier call on geometric basket."""

    is_path_dependent = True

    def __init__(self, strike, barrier=None):
        super().__init__(strike)
        self.barrier = barrier if barrier is not None else strike * 0.8

    def eval(self, X):
        if X.ndim == 2:
            geo_mean = np.exp(np.mean(np.log(X + 1e-10), axis=1))
            barrier_hit = np.min(X, axis=1) <= self.barrier
            payoff = np.maximum(0, geo_mean - self.strike)
            return payoff * barrier_hit
        elif X.ndim == 3:
            initial_min = np.min(X[:, :, 0], axis=1)
            initially_knocked_in = initial_min <= self.barrier
            min_along_path = np.min(X, axis=(1, 2))
            barrier_hit = min_along_path <= self.barrier
            knocked_in = initially_knocked_in | barrier_hit
            geo_mean = np.exp(np.mean(np.log(X[:, :, -1] + 1e-10), axis=1))
            payoff = np.maximum(0, geo_mean - self.strike)
            return payoff * knocked_in
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


class DownAndInGeometricBasketPut(Payoff):
    """Down-and-in barrier put on geometric basket."""

    is_path_dependent = True

    def __init__(self, strike, barrier=None):
        super().__init__(strike)
        self.barrier = barrier if barrier is not None else strike * 0.8

    def eval(self, X):
        if X.ndim == 2:
            geo_mean = np.exp(np.mean(np.log(X + 1e-10), axis=1))
            barrier_hit = np.min(X, axis=1) <= self.barrier
            payoff = np.maximum(0, self.strike - geo_mean)
            return payoff * barrier_hit
        elif X.ndim == 3:
            initial_min = np.min(X[:, :, 0], axis=1)
            initially_knocked_in = initial_min <= self.barrier
            min_along_path = np.min(X, axis=(1, 2))
            barrier_hit = min_along_path <= self.barrier
            knocked_in = initially_knocked_in | barrier_hit
            geo_mean = np.exp(np.mean(np.log(X[:, :, -1] + 1e-10), axis=1))
            payoff = np.maximum(0, self.strike - geo_mean)
            return payoff * knocked_in
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")