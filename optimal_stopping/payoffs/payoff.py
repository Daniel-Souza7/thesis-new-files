"""
Base classes for option payoffs.
"""

import numpy as np


class Payoff:
    """Base class for option payoff functions."""

    is_path_dependent = False  # Override in subclass if path-dependent

    def __init__(self, strike):
        """
        Initialize payoff with strike price.

        Args:
            strike: Strike price K
        """
        self.strike = strike

    def __call__(self, stock_paths):
        """
        Evaluate payoff for all paths at all timesteps.

        Args:
            stock_paths: Array of shape (nb_paths, nb_stocks, nb_dates+1)

        Returns:
            payoffs: Array of shape (nb_paths, nb_dates+1)
        """
        nb_paths, nb_stocks, nb_dates = stock_paths.shape
        payoffs = np.zeros((nb_paths, nb_dates))

        for date in range(nb_dates):
            if self.is_path_dependent:
                # Pass full history up to this date
                payoffs[:, date] = self.eval(stock_paths[:, :, :date + 1])
            else:
                # Pass only current timestep
                payoffs[:, date] = self.eval(stock_paths[:, :, date])

        return payoffs

    def eval(self, X):
        """
        Evaluate payoff for given stock prices.

        Args:
            X: Array of shape (nb_paths, nb_stocks) or (nb_paths, nb_stocks, nb_dates+1)

        Returns:
            Array of shape (nb_paths,) or (nb_paths, nb_dates+1)
        """
        raise NotImplementedError("Subclasses must implement eval()")

    def __repr__(self):
        """String representation of payoff."""
        return f"{self.__class__.__name__}(strike={self.strike})"

# ============================================================================
# STANDARD BASKET OPTIONS
# ============================================================================

class MaxPut(Payoff):
    """Put option on maximum of basket: max(0, K - max(S_i))"""

    def __init__(self, strike):
        self.strike = strike

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        payoff = self.strike - np.max(X, axis=1)
        return payoff.clip(0, None)


class MaxCall(Payoff):
    """Call option on maximum of basket: max(0, max(S_i) - K)"""

    def __init__(self, strike):
        self.strike = strike

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        payoff = np.max(X, axis=1) - self.strike
        return payoff.clip(0, None)


class MinPut(Payoff):
    """Put option on minimum of basket: max(0, K - min(S_i))"""

    def __init__(self, strike):
        self.strike = strike

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        payoff = self.strike - np.min(X, axis=1)
        return payoff.clip(0, None)


class MinCall(Payoff):
    """Call option on minimum of basket: max(0, min(S_i) - K)"""

    def __init__(self, strike):
        self.strike = strike

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        payoff = np.min(X, axis=1) - self.strike
        return payoff.clip(0, None)


class Put1Dim(Payoff):
    """Standard put option: max(0, K - S)"""

    def __init__(self, strike):
        self.strike = strike

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        payoff = self.strike - X
        return payoff.clip(0, None)


class Call1Dim(Payoff):
    """Standard call option: max(0, S - K)"""

    def __init__(self, strike):
        self.strike = strike

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        payoff = X - self.strike
        return payoff.clip(0, None)


class GeometricPut(Payoff):
    """Put on geometric average: max(0, K - (S_1 * ... * S_d)^(1/d))"""

    def __init__(self, strike):
        self.strike = strike

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        dim = X.shape[1]
        payoff = self.strike - np.prod(X, axis=1) ** (1 / dim)
        return payoff.clip(0, None)


class GeometricCall(Payoff):
    """Call on geometric average: max(0, (S_1 * ... * S_d)^(1/d) - K)"""

    def __init__(self, strike):
        self.strike = strike

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        dim = X.shape[1]
        payoff = np.prod(X, axis=1) ** (1 / dim) - self.strike
        return payoff.clip(0, None)


class BasketCall(Payoff):
    """Call on arithmetic average: max(0, mean(S_i) - K)"""

    def __init__(self, strike):
        self.strike = strike

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        payoff = np.mean(X, axis=1) - self.strike
        return payoff.clip(0, None)


class BasketPut(Payoff):
    """Put on arithmetic average: max(0, K - mean(S_i))"""

    def __init__(self, strike):
        self.strike = strike

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        payoff = self.strike - np.mean(X, axis=1)
        return payoff.clip(0, None)


# ============================================================================
# BARRIER OPTIONS - UP-AND-OUT (Knock-Out when price goes above barrier)
# ============================================================================

class UpAndOutMaxCall(Payoff):
    """
    Up-and-out barrier call on maximum of basket.
    Payoff: max(0, max(S_i(T)) - K) if max_over_time(max(S_i)) < B, else 0

    CRITICAL: If initial spot >= barrier, option is immediately knocked out.
    EXTREME BARRIER: If barrier > 100, immediately returns 0 payoff.
    """
    is_path_dependent = True  # Flag for backward induction algorithms
    def __init__(self, strike, barrier=None):
        self.strike = strike
        self.barrier = barrier if barrier is not None else strike * 1.2

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        # Check for extreme barrier
        if self.barrier < 100:
            return np.zeros(X.shape[0])

        if X.ndim == 2:
            # Only terminal value - check if >= barrier
            terminal_max = np.max(X, axis=1)
            barrier_not_hit = terminal_max < self.barrier
            payoff = np.maximum(0, terminal_max - self.strike)
        else:
            # CRITICAL FIX: Check initial state at t=0
            initial_max = np.max(X[:, :, 0], axis=1)  # (nb_paths,)
            initially_knocked_out = initial_max >= self.barrier

            # Check if barrier was ever hit during path
            max_along_path = np.max(X, axis=(1, 2))
            barrier_not_hit = (max_along_path < self.barrier) & ~initially_knocked_out

            terminal_max = np.max(X[:, :, -1], axis=1)
            payoff = np.maximum(0, terminal_max - self.strike)

        return payoff * barrier_not_hit


class UpAndOutMaxPut(Payoff):
    """
    Up-and-out barrier put on maximum of basket.
    Payoff: max(0, K - max(S_i(T))) if max_over_time(max(S_i)) < B, else 0
    EXTREME BARRIER: If barrier > 100, immediately returns 0 payoff.
    """

    def __init__(self, strike, barrier=None):
        self.strike = strike
        self.barrier = barrier if barrier is not None else strike * 1.2

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        # Check for extreme barrier
        if self.barrier < 100:
            return np.zeros(X.shape[0])

        if X.ndim == 2:
            terminal_max = np.max(X, axis=1)
            barrier_not_hit = terminal_max < self.barrier
            payoff = np.maximum(0, self.strike - terminal_max)
        else:
            initial_max = np.max(X[:, :, 0], axis=1)
            initially_knocked_out = initial_max >= self.barrier

            max_along_path = np.max(X, axis=(1, 2))
            barrier_not_hit = (max_along_path < self.barrier) & ~initially_knocked_out

            terminal_max = np.max(X[:, :, -1], axis=1)
            payoff = np.maximum(0, self.strike - terminal_max)

        return payoff * barrier_not_hit


class UpAndOutBasketCall(Payoff):
    """
    Up-and-out barrier call on arithmetic average basket.
    EXTREME BARRIER: If barrier > 100, immediately returns 0 payoff.
    """

    def __init__(self, strike, barrier=None):
        self.strike = strike
        self.barrier = barrier if barrier is not None else strike * 1.2

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        # Check for extreme barrier
        if self.barrier < 100:
            return np.zeros(X.shape[0])

        if X.ndim == 2:
            terminal_avg = np.mean(X, axis=1)
            barrier_not_hit = terminal_avg < self.barrier
            payoff = np.maximum(0, terminal_avg - self.strike)
        else:
            initial_avg = np.mean(X[:, :, 0], axis=1)
            initially_knocked_out = initial_avg >= self.barrier

            avg_along_path = np.mean(X, axis=1)
            max_avg = np.max(avg_along_path, axis=1)
            barrier_not_hit = (max_avg < self.barrier) & ~initially_knocked_out

            terminal_avg = np.mean(X[:, :, -1], axis=1)
            payoff = np.maximum(0, terminal_avg - self.strike)

        return payoff * barrier_not_hit


class UpAndOutBasketPut(Payoff):
    """
    Up-and-out barrier put on arithmetic average basket.
    EXTREME BARRIER: If barrier > 100, immediately returns 0 payoff.
    """

    def __init__(self, strike, barrier=None):
        self.strike = strike
        self.barrier = barrier if barrier is not None else strike * 1.2

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        # Check for extreme barrier
        if self.barrier < 100:
            return np.zeros(X.shape[0])

        if X.ndim == 2:
            terminal_avg = np.mean(X, axis=1)
            barrier_not_hit = terminal_avg < self.barrier
            payoff = np.maximum(0, self.strike - terminal_avg)
        else:
            initial_avg = np.mean(X[:, :, 0], axis=1)
            initially_knocked_out = initial_avg >= self.barrier

            avg_along_path = np.mean(X, axis=1)
            max_avg = np.max(avg_along_path, axis=1)
            barrier_not_hit = (max_avg < self.barrier) & ~initially_knocked_out

            terminal_avg = np.mean(X[:, :, -1], axis=1)
            payoff = np.maximum(0, self.strike - terminal_avg)

        return payoff * barrier_not_hit

# ============================================================================
# BARRIER OPTIONS - DOWN-AND-OUT (Knock-Out when price goes below barrier)
# ============================================================================

class DownAndOutMaxCall(Payoff):
    """
    Down-and-out barrier call on maximum of basket.
    Payoff: max(0, max(S_i(T)) - K) if min_over_time(min(S_i)) > B, else 0

    CRITICAL: If initial spot <= barrier, option is immediately knocked out.
    EXTREME BARRIER: If barrier < 100, immediately returns 0 payoff.
    """

    def __init__(self, strike, barrier=None):
        self.strike = strike
        self.barrier = barrier if barrier is not None else strike * 0.8

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        # Check for extreme barrier
        if self.barrier > 100:
            return np.zeros(X.shape[0])

        if X.ndim == 2:
            terminal_max = np.max(X, axis=1)
            terminal_min = np.min(X, axis=1)
            barrier_not_hit = terminal_min > self.barrier
            payoff = np.maximum(0, terminal_max - self.strike)
        else:
            initial_min = np.min(X[:, :, 0], axis=1)
            initially_knocked_out = initial_min <= self.barrier

            min_along_path = np.min(X, axis=(1, 2))
            barrier_not_hit = (min_along_path > self.barrier) & ~initially_knocked_out

            terminal_max = np.max(X[:, :, -1], axis=1)
            payoff = np.maximum(0, terminal_max - self.strike)

        return payoff * barrier_not_hit


class DownAndOutMaxPut(Payoff):
    """
    Down-and-out barrier put on maximum of basket.
    EXTREME BARRIER: If barrier < 100, immediately returns 0 payoff.
    """

    def __init__(self, strike, barrier=None):
        self.strike = strike
        self.barrier = barrier if barrier is not None else strike * 0.8

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        # Check for extreme barrier
        if self.barrier > 100:
            return np.zeros(X.shape[0])

        if X.ndim == 2:
            terminal_max = np.max(X, axis=1)
            terminal_min = np.min(X, axis=1)
            barrier_not_hit = terminal_min > self.barrier
            payoff = np.maximum(0, self.strike - terminal_max)
        else:
            initial_min = np.min(X[:, :, 0], axis=1)
            initially_knocked_out = initial_min <= self.barrier

            min_along_path = np.min(X, axis=(1, 2))
            barrier_not_hit = (min_along_path > self.barrier) & ~initially_knocked_out

            terminal_max = np.max(X[:, :, -1], axis=1)
            payoff = np.maximum(0, self.strike - terminal_max)

        return payoff * barrier_not_hit


class DownAndOutBasketCall(Payoff):
    """
    Down-and-out barrier call on arithmetic average basket.
    EXTREME BARRIER: If barrier < 100, immediately returns 0 payoff.
    """

    def __init__(self, strike, barrier=None):
        self.strike = strike
        self.barrier = barrier if barrier is not None else strike * 0.8

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        # Check for extreme barrier
        if self.barrier > 100:
            return np.zeros(X.shape[0])

        if X.ndim == 2:
            terminal_avg = np.mean(X, axis=1)
            barrier_not_hit = terminal_avg > self.barrier
            payoff = np.maximum(0, terminal_avg - self.strike)
        else:
            initial_avg = np.mean(X[:, :, 0], axis=1)
            initially_knocked_out = initial_avg <= self.barrier

            avg_along_path = np.mean(X, axis=1)
            min_avg = np.min(avg_along_path, axis=1)
            barrier_not_hit = (min_avg > self.barrier) & ~initially_knocked_out

            terminal_avg = np.mean(X[:, :, -1], axis=1)
            payoff = np.maximum(0, terminal_avg - self.strike)

        return payoff * barrier_not_hit


class DownAndOutBasketPut(Payoff):
    """
    Down-and-out barrier put on arithmetic average basket.
    EXTREME BARRIER: If barrier < 100, immediately returns 0 payoff.
    """

    def __init__(self, strike, barrier=None):
        self.strike = strike
        self.barrier = barrier if barrier is not None else strike * 0.8

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        # Check for extreme barrier
        if self.barrier > 100:
            return np.zeros(X.shape[0])

        if X.ndim == 2:
            terminal_avg = np.mean(X, axis=1)
            barrier_not_hit = terminal_avg > self.barrier
            payoff = np.maximum(0, self.strike - terminal_avg)
        else:
            initial_avg = np.mean(X[:, :, 0], axis=1)
            initially_knocked_out = initial_avg <= self.barrier

            avg_along_path = np.mean(X, axis=1)
            min_avg = np.min(avg_along_path, axis=1)
            barrier_not_hit = (min_avg > self.barrier) & ~initially_knocked_out

            terminal_avg = np.mean(X[:, :, -1], axis=1)
            payoff = np.maximum(0, self.strike - terminal_avg)

        return payoff * barrier_not_hit


# ============================================================================
# BARRIER OPTIONS - UP-AND-IN (Activates when price goes above barrier)
# ============================================================================

class UpAndInMaxCall(Payoff):
    """
    Up-and-in barrier call on maximum of basket.
    Payoff: max(0, max(S_i(T)) - K) if max_over_time(max(S_i)) >= B, else 0

    CRITICAL: If initial spot >= barrier, option is immediately activated.
    """

    def __init__(self, strike, barrier=None):
        self.strike = strike
        self.barrier = barrier if barrier is not None else strike * 1.2

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        if X.ndim == 2:
            terminal_max = np.max(X, axis=1)
            barrier_hit = terminal_max >= self.barrier
            payoff = np.maximum(0, terminal_max - self.strike)
        else:
            initial_max = np.max(X[:, :, 0], axis=1)
            initially_knocked_in = initial_max >= self.barrier

            max_along_path = np.max(X, axis=(1, 2))
            barrier_hit = (max_along_path >= self.barrier) | initially_knocked_in

            terminal_max = np.max(X[:, :, -1], axis=1)
            payoff = np.maximum(0, terminal_max - self.strike)

        return payoff * barrier_hit


class UpAndInMaxPut(Payoff):
    """
    Up-and-in barrier put on maximum of basket.
    """

    def __init__(self, strike, barrier=None):
        self.strike = strike
        self.barrier = barrier if barrier is not None else strike * 1.2

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        if X.ndim == 2:
            terminal_max = np.max(X, axis=1)
            barrier_hit = terminal_max >= self.barrier
            payoff = np.maximum(0, self.strike - terminal_max)
        else:
            initial_max = np.max(X[:, :, 0], axis=1)
            initially_knocked_in = initial_max >= self.barrier

            max_along_path = np.max(X, axis=(1, 2))
            barrier_hit = (max_along_path >= self.barrier) | initially_knocked_in

            terminal_max = np.max(X[:, :, -1], axis=1)
            payoff = np.maximum(0, self.strike - terminal_max)

        return payoff * barrier_hit


class UpAndInBasketCall(Payoff):
    """
    Up-and-in barrier call on arithmetic average basket.
    """

    def __init__(self, strike, barrier=None):
        self.strike = strike
        self.barrier = barrier if barrier is not None else strike * 1.2

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        if X.ndim == 2:
            terminal_avg = np.mean(X, axis=1)
            barrier_hit = terminal_avg >= self.barrier
            payoff = np.maximum(0, terminal_avg - self.strike)
        else:
            initial_avg = np.mean(X[:, :, 0], axis=1)
            initially_knocked_in = initial_avg >= self.barrier

            avg_along_path = np.mean(X, axis=1)
            max_avg = np.max(avg_along_path, axis=1)
            barrier_hit = (max_avg >= self.barrier) | initially_knocked_in

            terminal_avg = np.mean(X[:, :, -1], axis=1)
            payoff = np.maximum(0, terminal_avg - self.strike)

        return payoff * barrier_hit


class UpAndInBasketPut(Payoff):
    """
    Up-and-in barrier put on arithmetic average basket.
    """

    def __init__(self, strike, barrier=None):
        self.strike = strike
        self.barrier = barrier if barrier is not None else strike * 1.2

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        if X.ndim == 2:
            terminal_avg = np.mean(X, axis=1)
            barrier_hit = terminal_avg >= self.barrier
            payoff = np.maximum(0, self.strike - terminal_avg)
        else:
            initial_avg = np.mean(X[:, :, 0], axis=1)
            initially_knocked_in = initial_avg >= self.barrier

            avg_along_path = np.mean(X, axis=1)
            max_avg = np.max(avg_along_path, axis=1)
            barrier_hit = (max_avg >= self.barrier) | initially_knocked_in

            terminal_avg = np.mean(X[:, :, -1], axis=1)
            payoff = np.maximum(0, self.strike - terminal_avg)

        return payoff * barrier_hit


# ============================================================================
# BARRIER OPTIONS - DOWN-AND-IN (Activates when price goes below barrier)
# ============================================================================

class DownAndInMaxCall(Payoff):
    """
    Down-and-in barrier call on maximum of basket.
    Payoff: max(0, max(S_i(T)) - K) if min_over_time(min(S_i)) <= B, else 0

    CRITICAL: If initial spot <= barrier, option is immediately activated.
    """

    def __init__(self, strike, barrier=None):
        self.strike = strike
        self.barrier = barrier if barrier is not None else strike * 0.8

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        if X.ndim == 2:
            terminal_max = np.max(X, axis=1)
            terminal_min = np.min(X, axis=1)
            barrier_hit = terminal_min <= self.barrier
            payoff = np.maximum(0, terminal_max - self.strike)
        else:
            initial_min = np.min(X[:, :, 0], axis=1)
            initially_knocked_in = initial_min <= self.barrier

            min_along_path = np.min(X, axis=(1, 2))
            barrier_hit = (min_along_path <= self.barrier) | initially_knocked_in

            terminal_max = np.max(X[:, :, -1], axis=1)
            payoff = np.maximum(0, terminal_max - self.strike)

        return payoff * barrier_hit


class DownAndInMaxPut(Payoff):
    """
    Down-and-in barrier put on maximum of basket.
    """

    def __init__(self, strike, barrier=None):
        self.strike = strike
        self.barrier = barrier if barrier is not None else strike * 0.8

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        if X.ndim == 2:
            terminal_max = np.max(X, axis=1)
            terminal_min = np.min(X, axis=1)
            barrier_hit = terminal_min <= self.barrier
            payoff = np.maximum(0, self.strike - terminal_max)
        else:
            initial_min = np.min(X[:, :, 0], axis=1)
            initially_knocked_in = initial_min <= self.barrier

            min_along_path = np.min(X, axis=(1, 2))
            barrier_hit = (min_along_path <= self.barrier) | initially_knocked_in

            terminal_max = np.max(X[:, :, -1], axis=1)
            payoff = np.maximum(0, self.strike - terminal_max)

        return payoff * barrier_hit


class DownAndInBasketCall(Payoff):
    """
    Down-and-in barrier call on arithmetic average basket.
    """

    def __init__(self, strike, barrier=None):
        self.strike = strike
        self.barrier = barrier if barrier is not None else strike * 0.8

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        if X.ndim == 2:
            terminal_avg = np.mean(X, axis=1)
            barrier_hit = terminal_avg <= self.barrier
            payoff = np.maximum(0, terminal_avg - self.strike)
        else:
            initial_avg = np.mean(X[:, :, 0], axis=1)
            initially_knocked_in = initial_avg <= self.barrier

            avg_along_path = np.mean(X, axis=1)
            min_avg = np.min(avg_along_path, axis=1)
            barrier_hit = (min_avg <= self.barrier) | initially_knocked_in

            terminal_avg = np.mean(X[:, :, -1], axis=1)
            payoff = np.maximum(0, terminal_avg - self.strike)

        return payoff * barrier_hit


class DownAndInBasketPut(Payoff):
    """
    Down-and-in barrier put on arithmetic average basket.
    """

    def __init__(self, strike, barrier=None):
        self.strike = strike
        self.barrier = barrier if barrier is not None else strike * 0.8

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        if X.ndim == 2:
            terminal_avg = np.mean(X, axis=1)
            barrier_hit = terminal_avg <= self.barrier
            payoff = np.maximum(0, self.strike - terminal_avg)
        else:
            initial_avg = np.mean(X[:, :, 0], axis=1)
            initially_knocked_in = initial_avg <= self.barrier

            avg_along_path = np.mean(X, axis=1)
            min_avg = np.min(avg_along_path, axis=1)
            barrier_hit = (min_avg <= self.barrier) | initially_knocked_in

            terminal_avg = np.mean(X[:, :, -1], axis=1)
            payoff = np.maximum(0, self.strike - terminal_avg)

        return payoff * barrier_hit


# ============================================================================
# LOOKBACK OPTIONS
# ============================================================================

class LookbackMaxCall(Payoff):
    """
    Fixed-strike lookback call on maximum of basket.
    Payoff: max(0, max_over_time(max(S_i)) - K)
    """

    def __init__(self, strike):
        self.strike = strike

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        if X.ndim == 2:
            running_max = np.max(X, axis=1)
        else:
            running_max = np.max(X, axis=(1, 2))

        payoff = running_max - self.strike
        return payoff.clip(0, None)


class LookbackMaxPut(Payoff):
    """
    Fixed-strike lookback put on maximum of basket.
    """

    def __init__(self, strike):
        self.strike = strike

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        if X.ndim == 2:
            terminal_max = np.max(X, axis=1)
            running_min = terminal_max
        else:
            max_across_stocks = np.max(X, axis=1)
            running_min = np.min(max_across_stocks, axis=1)

        payoff = self.strike - running_min
        return payoff.clip(0, None)


class LookbackMinCall(Payoff):
    """
    Fixed-strike lookback call on minimum of basket.
    """

    def __init__(self, strike):
        self.strike = strike

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        if X.ndim == 2:
            terminal_min = np.min(X, axis=1)
            running_max = terminal_min
        else:
            min_across_stocks = np.min(X, axis=1)
            running_max = np.max(min_across_stocks, axis=1)

        payoff = running_max - self.strike
        return payoff.clip(0, None)


class LookbackMinPut(Payoff):
    """
    Fixed-strike lookback put on minimum of basket.
    """

    def __init__(self, strike):
        self.strike = strike

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        if X.ndim == 2:
            running_min = np.min(X, axis=1)
        else:
            running_min = np.min(X, axis=(1, 2))

        payoff = self.strike - running_min
        return payoff.clip(0, None)


class LookbackBasketCall(Payoff):
    """
    Fixed-strike lookback call on arithmetic basket average.
    """

    def __init__(self, strike):
        self.strike = strike

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        if X.ndim == 2:
            running_max = np.mean(X, axis=1)
        else:
            avg_across_stocks = np.mean(X, axis=1)
            running_max = np.max(avg_across_stocks, axis=1)

        payoff = running_max - self.strike
        return payoff.clip(0, None)


class LookbackBasketPut(Payoff):
    """
    Fixed-strike lookback put on arithmetic basket average.
    """

    def __init__(self, strike):
        self.strike = strike

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        if X.ndim == 2:
            running_min = np.mean(X, axis=1)
        else:
            avg_across_stocks = np.mean(X, axis=1)
            running_min = np.min(avg_across_stocks, axis=1)

        payoff = self.strike - running_min
        return payoff.clip(0, None)


# ============================================================================
# BARRIER OPTIONS - MIN-BASED (8 classes)
# ============================================================================

class UpAndOutMinCall(Payoff):
    """
    Up-and-out barrier call on minimum: Knocks out if max(S) > barrier
    EXTREME BARRIER: If barrier > 100, immediately returns 0 payoff.
    """

    def __init__(self, strike, barrier=None):
        self.strike = strike
        self.barrier = barrier if barrier is not None else strike * 1.2

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        # Check for extreme barrier
        if self.barrier < 100:
            return np.zeros(X.shape[0])

        if X.ndim == 2:
            max_price = np.max(X, axis=1)
            knocked_out = max_price >= self.barrier
            intrinsic = np.min(X, axis=1) - self.strike
            payoff = np.where(knocked_out, 0, intrinsic)
        else:
            initial_max = np.max(X[:, :, 0], axis=1)
            initially_knocked_out = initial_max >= self.barrier

            max_over_path = np.max(X, axis=(1, 2))
            knocked_out = (max_over_path >= self.barrier) | initially_knocked_out

            intrinsic = np.min(X[:, :, -1], axis=1) - self.strike
            payoff = np.where(knocked_out, 0, intrinsic)
        return payoff.clip(0, None)


class UpAndOutMinPut(Payoff):
    """
    Up-and-out barrier put on minimum: Knocks out if max(S) > barrier
    EXTREME BARRIER: If barrier > 100, immediately returns 0 payoff.
    """

    def __init__(self, strike, barrier=None):
        self.strike = strike
        self.barrier = barrier if barrier is not None else strike * 1.2

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        # Check for extreme barrier
        if self.barrier < 100:
            return np.zeros(X.shape[0])

        if X.ndim == 2:
            max_price = np.max(X, axis=1)
            knocked_out = max_price >= self.barrier
            intrinsic = self.strike - np.min(X, axis=1)
            payoff = np.where(knocked_out, 0, intrinsic)
        else:
            initial_max = np.max(X[:, :, 0], axis=1)
            initially_knocked_out = initial_max >= self.barrier

            max_over_path = np.max(X, axis=(1, 2))
            knocked_out = (max_over_path >= self.barrier) | initially_knocked_out

            intrinsic = self.strike - np.min(X[:, :, -1], axis=1)
            payoff = np.where(knocked_out, 0, intrinsic)
        return payoff.clip(0, None)


class DownAndOutMinCall(Payoff):
    """
    Down-and-out barrier call on minimum: Knocks out if min(S) < barrier
    EXTREME BARRIER: If barrier < 100, immediately returns 0 payoff.
    """

    def __init__(self, strike, barrier=None):
        self.strike = strike
        self.barrier = barrier if barrier is not None else strike * 0.8

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        # Check for extreme barrier
        if self.barrier > 100:
            return np.zeros(X.shape[0])

        if X.ndim == 2:
            min_price = np.min(X, axis=1)
            knocked_out = min_price <= self.barrier
            intrinsic = np.min(X, axis=1) - self.strike
            payoff = np.where(knocked_out, 0, intrinsic)
        else:
            initial_min = np.min(X[:, :, 0], axis=1)
            initially_knocked_out = initial_min <= self.barrier

            min_over_path = np.min(X, axis=(1, 2))
            knocked_out = (min_over_path <= self.barrier) | initially_knocked_out

            intrinsic = np.min(X[:, :, -1], axis=1) - self.strike
            payoff = np.where(knocked_out, 0, intrinsic)
        return payoff.clip(0, None)


class DownAndOutMinPut(Payoff):
    """
    Down-and-out barrier put on minimum: Knocks out if min(S) < barrier
    EXTREME BARRIER: If barrier < 100, immediately returns 0 payoff.
    """

    def __init__(self, strike, barrier=None):
        self.strike = strike
        self.barrier = barrier if barrier is not None else strike * 0.8

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        # Check for extreme barrier
        if self.barrier > 100:
            return np.zeros(X.shape[0])

        if X.ndim == 2:
            min_price = np.min(X, axis=1)
            knocked_out = min_price <= self.barrier
            intrinsic = self.strike - np.min(X, axis=1)
            payoff = np.where(knocked_out, 0, intrinsic)
        else:
            initial_min = np.min(X[:, :, 0], axis=1)
            initially_knocked_out = initial_min <= self.barrier

            min_over_path = np.min(X, axis=(1, 2))
            knocked_out = (min_over_path <= self.barrier) | initially_knocked_out

            intrinsic = self.strike - np.min(X[:, :, -1], axis=1)
            payoff = np.where(knocked_out, 0, intrinsic)
        return payoff.clip(0, None)


class UpAndInMinCall(Payoff):
    """Up-and-in barrier call on minimum: Knocks in if max(S) > barrier"""

    def __init__(self, strike, barrier=None):
        self.strike = strike
        self.barrier = barrier if barrier is not None else strike * 1.2

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        if X.ndim == 2:
            max_price = np.max(X, axis=1)
            knocked_in = max_price >= self.barrier
            intrinsic = np.min(X, axis=1) - self.strike
            payoff = np.where(knocked_in, intrinsic, 0)
        else:
            initial_max = np.max(X[:, :, 0], axis=1)
            initially_knocked_in = initial_max >= self.barrier

            max_over_path = np.max(X, axis=(1, 2))
            knocked_in = (max_over_path >= self.barrier) | initially_knocked_in

            intrinsic = np.min(X[:, :, -1], axis=1) - self.strike
            payoff = np.where(knocked_in, intrinsic, 0)
        return payoff.clip(0, None)


class UpAndInMinPut(Payoff):
    """Up-and-in barrier put on minimum: Knocks in if max(S) > barrier"""

    def __init__(self, strike, barrier=None):
        self.strike = strike
        self.barrier = barrier if barrier is not None else strike * 1.2

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        if X.ndim == 2:
            max_price = np.max(X, axis=1)
            knocked_in = max_price >= self.barrier
            intrinsic = self.strike - np.min(X, axis=1)
            payoff = np.where(knocked_in, intrinsic, 0)
        else:
            initial_max = np.max(X[:, :, 0], axis=1)
            initially_knocked_in = initial_max >= self.barrier

            max_over_path = np.max(X, axis=(1, 2))
            knocked_in = (max_over_path >= self.barrier) | initially_knocked_in

            intrinsic = self.strike - np.min(X[:, :, -1], axis=1)
            payoff = np.where(knocked_in, intrinsic, 0)
        return payoff.clip(0, None)


class DownAndInMinCall(Payoff):
    """Down-and-in barrier call on minimum: Knocks in if min(S) < barrier"""

    def __init__(self, strike, barrier=None):
        self.strike = strike
        self.barrier = barrier if barrier is not None else strike * 0.8

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        if X.ndim == 2:
            min_price = np.min(X, axis=1)
            knocked_in = min_price <= self.barrier
            intrinsic = np.min(X, axis=1) - self.strike
            payoff = np.where(knocked_in, intrinsic, 0)
        else:
            initial_min = np.min(X[:, :, 0], axis=1)
            initially_knocked_in = initial_min <= self.barrier

            min_over_path = np.min(X, axis=(1, 2))
            knocked_in = (min_over_path <= self.barrier) | initially_knocked_in

            intrinsic = np.min(X[:, :, -1], axis=1) - self.strike
            payoff = np.where(knocked_in, intrinsic, 0)
        return payoff.clip(0, None)


class DownAndInMinPut(Payoff):
    """Down-and-in barrier put on minimum: Knocks in if min(S) < barrier"""

    def __init__(self, strike, barrier=None):
        self.strike = strike
        self.barrier = barrier if barrier is not None else strike * 0.8

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        if X.ndim == 2:
            min_price = np.min(X, axis=1)
            knocked_in = min_price <= self.barrier
            intrinsic = self.strike - np.min(X, axis=1)
            payoff = np.where(knocked_in, intrinsic, 0)
        else:
            initial_min = np.min(X[:, :, 0], axis=1)
            initially_knocked_in = initial_min <= self.barrier

            min_over_path = np.min(X, axis=(1, 2))
            knocked_in = (min_over_path <= self.barrier) | initially_knocked_in

            intrinsic = self.strike - np.min(X[:, :, -1], axis=1)
            payoff = np.where(knocked_in, intrinsic, 0)
        return payoff.clip(0, None)


# ============================================================================
# BARRIER OPTIONS - GEOMETRIC-BASED (8 classes)
# ============================================================================

class UpAndOutGeometricCall(Payoff):
    """
    Up-and-out barrier call on geometric mean: Knocks out if max(S) > barrier
    EXTREME BARRIER: If barrier > 100, immediately returns 0 payoff.
    """

    def __init__(self, strike, barrier=None):
        self.strike = strike
        self.barrier = barrier if barrier is not None else strike * 1.2

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        # Check for extreme barrier
        if self.barrier < 100:
            return np.zeros(X.shape[0])

        if X.ndim == 2:
            max_price = np.max(X, axis=1)
            knocked_out = max_price >= self.barrier
            geometric_mean = np.exp(np.mean(np.log(X + 1e-10), axis=1))
            intrinsic = geometric_mean - self.strike
            payoff = np.where(knocked_out, 0, intrinsic)
        else:
            initial_max = np.max(X[:, :, 0], axis=1)
            initially_knocked_out = initial_max >= self.barrier

            max_over_path = np.max(X, axis=(1, 2))
            knocked_out = (max_over_path >= self.barrier) | initially_knocked_out

            geometric_mean = np.exp(np.mean(np.log(X[:, :, -1] + 1e-10), axis=1))
            intrinsic = geometric_mean - self.strike
            payoff = np.where(knocked_out, 0, intrinsic)
        return payoff.clip(0, None)


class UpAndOutGeometricPut(Payoff):
    """
    Up-and-out barrier put on geometric mean: Knocks out if max(S) > barrier
    EXTREME BARRIER: If barrier > 100, immediately returns 0 payoff.
    """

    def __init__(self, strike, barrier=None):
        self.strike = strike
        self.barrier = barrier if barrier is not None else strike * 1.2

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        # Check for extreme barrier
        if self.barrier < 100:
            return np.zeros(X.shape[0])

        if X.ndim == 2:
            max_price = np.max(X, axis=1)
            knocked_out = max_price >= self.barrier
            geometric_mean = np.exp(np.mean(np.log(X + 1e-10), axis=1))
            intrinsic = self.strike - geometric_mean
            payoff = np.where(knocked_out, 0, intrinsic)
        else:
            initial_max = np.max(X[:, :, 0], axis=1)
            initially_knocked_out = initial_max >= self.barrier

            max_over_path = np.max(X, axis=(1, 2))
            knocked_out = (max_over_path >= self.barrier) | initially_knocked_out

            geometric_mean = np.exp(np.mean(np.log(X[:, :, -1] + 1e-10), axis=1))
            intrinsic = self.strike - geometric_mean
            payoff = np.where(knocked_out, 0, intrinsic)
        return payoff.clip(0, None)


class DownAndOutGeometricCall(Payoff):
    """
    Down-and-out barrier call on geometric mean: Knocks out if min(S) < barrier
    EXTREME BARRIER: If barrier < 100, immediately returns 0 payoff.
    """

    def __init__(self, strike, barrier=None):
        self.strike = strike
        self.barrier = barrier if barrier is not None else strike * 0.8

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        # Check for extreme barrier
        if self.barrier > 100:
            return np.zeros(X.shape[0])

        if X.ndim == 2:
            min_price = np.min(X, axis=1)
            knocked_out = min_price <= self.barrier
            geometric_mean = np.exp(np.mean(np.log(X + 1e-10), axis=1))
            intrinsic = geometric_mean - self.strike
            payoff = np.where(knocked_out, 0, intrinsic)
        else:
            initial_min = np.min(X[:, :, 0], axis=1)
            initially_knocked_out = initial_min <= self.barrier

            min_over_path = np.min(X, axis=(1, 2))
            knocked_out = (min_over_path <= self.barrier) | initially_knocked_out

            geometric_mean = np.exp(np.mean(np.log(X[:, :, -1] + 1e-10), axis=1))
            intrinsic = geometric_mean - self.strike
            payoff = np.where(knocked_out, 0, intrinsic)
        return payoff.clip(0, None)


class DownAndOutGeometricPut(Payoff):
    """
    Down-and-out barrier put on geometric mean: Knocks out if min(S) < barrier
    EXTREME BARRIER: If barrier < 100, immediately returns 0 payoff.
    """

    def __init__(self, strike, barrier=None):
        self.strike = strike
        self.barrier = barrier if barrier is not None else strike * 0.8

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        # Check for extreme barrier
        if self.barrier > 100:
            return np.zeros(X.shape[0])

        if X.ndim == 2:
            min_price = np.min(X, axis=1)
            knocked_out = min_price <= self.barrier
            geometric_mean = np.exp(np.mean(np.log(X + 1e-10), axis=1))
            intrinsic = self.strike - geometric_mean
            payoff = np.where(knocked_out, 0, intrinsic)
        else:
            initial_min = np.min(X[:, :, 0], axis=1)
            initially_knocked_out = initial_min <= self.barrier

            min_over_path = np.min(X, axis=(1, 2))
            knocked_out = (min_over_path <= self.barrier) | initially_knocked_out

            geometric_mean = np.exp(np.mean(np.log(X[:, :, -1] + 1e-10), axis=1))
            intrinsic = self.strike - geometric_mean
            payoff = np.where(knocked_out, 0, intrinsic)
        return payoff.clip(0, None)


class UpAndInGeometricCall(Payoff):
    """Up-and-in barrier call on geometric mean: Knocks in if max(S) > barrier"""

    def __init__(self, strike, barrier=None):
        self.strike = strike
        self.barrier = barrier if barrier is not None else strike * 1.2

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        if X.ndim == 2:
            max_price = np.max(X, axis=1)
            knocked_in = max_price >= self.barrier
            geometric_mean = np.exp(np.mean(np.log(X + 1e-10), axis=1))
            intrinsic = geometric_mean - self.strike
            payoff = np.where(knocked_in, intrinsic, 0)
        else:
            initial_max = np.max(X[:, :, 0], axis=1)
            initially_knocked_in = initial_max >= self.barrier

            max_over_path = np.max(X, axis=(1, 2))
            knocked_in = (max_over_path >= self.barrier) | initially_knocked_in

            geometric_mean = np.exp(np.mean(np.log(X[:, :, -1] + 1e-10), axis=1))
            intrinsic = geometric_mean - self.strike
            payoff = np.where(knocked_in, intrinsic, 0)
        return payoff.clip(0, None)


class UpAndInGeometricPut(Payoff):
    """Up-and-in barrier put on geometric mean: Knocks in if max(S) > barrier"""

    def __init__(self, strike, barrier=None):
        self.strike = strike
        self.barrier = barrier if barrier is not None else strike * 1.2

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        if X.ndim == 2:
            max_price = np.max(X, axis=1)
            knocked_in = max_price >= self.barrier
            geometric_mean = np.exp(np.mean(np.log(X + 1e-10), axis=1))
            intrinsic = self.strike - geometric_mean
            payoff = np.where(knocked_in, intrinsic, 0)
        else:
            initial_max = np.max(X[:, :, 0], axis=1)
            initially_knocked_in = initial_max >= self.barrier

            max_over_path = np.max(X, axis=(1, 2))
            knocked_in = (max_over_path >= self.barrier) | initially_knocked_in

            geometric_mean = np.exp(np.mean(np.log(X[:, :, -1] + 1e-10), axis=1))
            intrinsic = self.strike - geometric_mean
            payoff = np.where(knocked_in, intrinsic, 0)
        return payoff.clip(0, None)


class DownAndInGeometricCall(Payoff):
    """Down-and-in barrier call on geometric mean: Knocks in if min(S) < barrier"""

    def __init__(self, strike, barrier=None):
        self.strike = strike
        self.barrier = barrier if barrier is not None else strike * 0.8

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        if X.ndim == 2:
            min_price = np.min(X, axis=1)
            knocked_in = min_price <= self.barrier
            geometric_mean = np.exp(np.mean(np.log(X + 1e-10), axis=1))
            intrinsic = geometric_mean - self.strike
            payoff = np.where(knocked_in, intrinsic, 0)
        else:
            initial_min = np.min(X[:, :, 0], axis=1)
            initially_knocked_in = initial_min <= self.barrier

            min_over_path = np.min(X, axis=(1, 2))
            knocked_in = (min_over_path <= self.barrier) | initially_knocked_in

            geometric_mean = np.exp(np.mean(np.log(X[:, :, -1] + 1e-10), axis=1))
            intrinsic = geometric_mean - self.strike
            payoff = np.where(knocked_in, intrinsic, 0)
        return payoff.clip(0, None)


class DownAndInGeometricPut(Payoff):
    """Down-and-in barrier put on geometric mean: Knocks in if min(S) < barrier"""

    def __init__(self, strike, barrier=None):
        self.strike = strike
        self.barrier = barrier if barrier is not None else strike * 0.8

    def __call__(self, X, strike=None):
        assert strike is None or strike == self.strike
        return self.eval(X)

    def eval(self, X):
        if X.ndim == 2:
            min_price = np.min(X, axis=1)
            knocked_in = min_price <= self.barrier
            geometric_mean = np.exp(np.mean(np.log(X + 1e-10), axis=1))
            intrinsic = self.strike - geometric_mean
            payoff = np.where(knocked_in, intrinsic, 0)
        else:
            initial_min = np.min(X[:, :, 0], axis=1)
            initially_knocked_in = initial_min <= self.barrier

            min_over_path = np.min(X, axis=(1, 2))
            knocked_in = (min_over_path <= self.barrier) | initially_knocked_in

            geometric_mean = np.exp(np.mean(np.log(X[:, :, -1] + 1e-10), axis=1))
            intrinsic = self.strike - geometric_mean
            payoff = np.where(knocked_in, intrinsic, 0)
        return payoff.clip(0, None)

# UTILITY PAYOFFS (for fractional BM experiments)
# ============================================================================

class Identity(Payoff):
    """Returns first asset price (for fractional BM experiments)"""

    def __init__(self, strike):
        self.strike = strike

    def __call__(self, X, strike=None):
        return self.eval(X)

    def eval(self, X):
        return X[:, 0]


class Max(Payoff):
    """Returns maximum of basket (for fractional BM experiments)"""

    def __init__(self, strike):
        self.strike = strike

    def __call__(self, X, strike=None):
        return self.eval(X)

    def eval(self, X):
        return np.max(X, axis=1)


class Mean(Payoff):
    """Returns arithmetic mean of basket (for fractional BM experiments)"""

    def __init__(self, strike):
        self.strike = strike

    def __call__(self, X, strike=None):
        return self.eval(X)

    def eval(self, X):
        return np.mean(X, axis=1)