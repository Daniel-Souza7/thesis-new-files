"""
Niche and Specialized Option Payoffs (7 payoffs)

These options use specialized aggregation methods on multi-asset baskets.
Most are non-path-dependent, relying only on terminal stock prices.

Notation:
- S_{(i)}(t) = i-th largest stock price at time t (sorted descending)
- S_{(1)} ≥ S_{(2)} ≥ ... ≥ S_{(d)}
- S̄(t) = arithmetic mean of all stocks at time t
- k = number of stocks to include in subset averaging
- d = total number of stocks
"""

import numpy as np
from optimal_stopping.payoffs.payoff import Payoff


class BestOfKCall(Payoff):
    """
    Best-of-K Call Option

    Pays max(0, avg(top k stocks) - K)

    Averages the k highest-priced stocks at exercise time.
    Higher k → more diversification, lower premium.
    """

    is_path_dependent = False

    def __init__(self, strike, k=2):
        """
        Args:
            strike: Strike price K
            k: Number of best (highest) stocks to average (default: 2)
        """
        super().__init__(strike)
        self.k = k

    def eval(self, X):
        """
        Args:
            X: Array of shape (nb_paths, nb_stocks) for single timestep
               or (nb_paths, nb_stocks, nb_dates+1) for path
        """
        if X.ndim == 2:
            # Single timestep: (nb_paths, nb_stocks)
            # Sort stocks descending for each path
            sorted_prices = np.sort(X, axis=1)[:, ::-1]  # Descending
            # Take top k and average
            top_k_avg = np.mean(sorted_prices[:, :self.k], axis=1)
            payoff = np.maximum(0, top_k_avg - self.strike)
            return payoff
        elif X.ndim == 3:
            # Path: (nb_paths, nb_stocks, nb_dates+1)
            # Only use terminal prices
            terminal_prices = X[:, :, -1]
            sorted_prices = np.sort(terminal_prices, axis=1)[:, ::-1]
            top_k_avg = np.mean(sorted_prices[:, :self.k], axis=1)
            payoff = np.maximum(0, top_k_avg - self.strike)
            return payoff
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


class WorstOfKCall(Payoff):
    """
    Worst-of-K Call Option

    Pays max(0, avg(bottom k stocks) - K)

    Averages the k lowest-priced stocks at exercise time.
    More conservative than BestOfK, typically cheaper.
    """

    is_path_dependent = False

    def __init__(self, strike, k=2):
        """
        Args:
            strike: Strike price K
            k: Number of worst (lowest) stocks to average (default: 2)
        """
        super().__init__(strike)
        self.k = k

    def eval(self, X):
        if X.ndim == 2:
            # Sort ascending, take first k (worst)
            sorted_prices = np.sort(X, axis=1)
            bottom_k_avg = np.mean(sorted_prices[:, :self.k], axis=1)
            payoff = np.maximum(0, bottom_k_avg - self.strike)
            return payoff
        elif X.ndim == 3:
            terminal_prices = X[:, :, -1]
            sorted_prices = np.sort(terminal_prices, axis=1)
            bottom_k_avg = np.mean(sorted_prices[:, :self.k], axis=1)
            payoff = np.maximum(0, bottom_k_avg - self.strike)
            return payoff
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


class RankWeightedBasketCall(Payoff):
    """
    Rank-Weighted Basket Call Option

    Pays max(0, Σ w_i × S_{(i)} - K) where w_i = (d+1-i) / Σj

    Weights are inversely proportional to rank:
    - Highest stock gets weight (d+1-1) = d
    - Second highest gets (d+1-2) = d-1
    - Lowest gets weight 1

    Normalized by Σ_{j=1}^d j = d(d+1)/2
    """

    is_path_dependent = False

    def __init__(self, strike):
        super().__init__(strike)

    def eval(self, X):
        if X.ndim == 2:
            nb_paths, nb_stocks = X.shape
            # Sort descending
            sorted_prices = np.sort(X, axis=1)[:, ::-1]
            # Compute weights: w_i = (d+1-i) / (d(d+1)/2)
            ranks = np.arange(1, nb_stocks + 1)  # 1, 2, ..., d
            weights = (nb_stocks + 1 - ranks) / (nb_stocks * (nb_stocks + 1) / 2)
            # Weighted sum
            weighted_sum = np.sum(sorted_prices * weights, axis=1)
            payoff = np.maximum(0, weighted_sum - self.strike)
            return payoff
        elif X.ndim == 3:
            terminal_prices = X[:, :, -1]
            nb_paths, nb_stocks = terminal_prices.shape
            sorted_prices = np.sort(terminal_prices, axis=1)[:, ::-1]
            ranks = np.arange(1, nb_stocks + 1)
            weights = (nb_stocks + 1 - ranks) / (nb_stocks * (nb_stocks + 1) / 2)
            weighted_sum = np.sum(sorted_prices * weights, axis=1)
            payoff = np.maximum(0, weighted_sum - self.strike)
            return payoff
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


class ChooserBasketOption(Payoff):
    """
    Chooser Basket Option

    Pays max(|S̄(t) - K|, 0) = |S̄(t) - K|

    Investor can choose at exercise whether to exercise as:
    - Call: max(0, S̄(t) - K), or
    - Put: max(0, K - S̄(t))

    Optimal choice is whichever is in-the-money.
    Mathematically equivalent to absolute deviation from strike.
    """

    is_path_dependent = False

    def __init__(self, strike):
        super().__init__(strike)

    def eval(self, X):
        if X.ndim == 2:
            basket_mean = np.mean(X, axis=1)
            # max(call, put) = max(max(0, S-K), max(0, K-S)) = |S-K|
            payoff = np.abs(basket_mean - self.strike)
            return payoff
        elif X.ndim == 3:
            terminal_mean = np.mean(X[:, :, -1], axis=1)
            payoff = np.abs(terminal_mean - self.strike)
            return payoff
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


class RangeCall(Payoff):
    """
    Range Call Option

    Pays max(0, [max_i S_i(t) - min_i S_i(t)] - K)

    Payoff based on the spread between highest and lowest stock prices.
    Benefits from dispersion/volatility in the basket.
    """

    is_path_dependent = False

    def __init__(self, strike):
        super().__init__(strike)

    def eval(self, X):
        if X.ndim == 2:
            max_price = np.max(X, axis=1)
            min_price = np.min(X, axis=1)
            range_spread = max_price - min_price
            payoff = np.maximum(0, range_spread - self.strike)
            return payoff
        elif X.ndim == 3:
            terminal_prices = X[:, :, -1]
            max_price = np.max(terminal_prices, axis=1)
            min_price = np.min(terminal_prices, axis=1)
            range_spread = max_price - min_price
            payoff = np.maximum(0, range_spread - self.strike)
            return payoff
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


class DispersionCall(Payoff):
    """
    Dispersion Call Option

    Pays max(0, Σ |S_i(t) - S̄(t)| - K)

    Sum of absolute deviations from basket mean.
    Measures dispersion/spread in the basket.
    High dispersion → higher payoff.
    """

    is_path_dependent = False

    def __init__(self, strike):
        super().__init__(strike)

    def eval(self, X):
        if X.ndim == 2:
            basket_mean = np.mean(X, axis=1, keepdims=True)  # (nb_paths, 1)
            deviations = np.abs(X - basket_mean)  # (nb_paths, nb_stocks)
            total_dispersion = np.sum(deviations, axis=1)
            payoff = np.maximum(0, total_dispersion - self.strike)
            return payoff
        elif X.ndim == 3:
            terminal_prices = X[:, :, -1]
            basket_mean = np.mean(terminal_prices, axis=1, keepdims=True)
            deviations = np.abs(terminal_prices - basket_mean)
            total_dispersion = np.sum(deviations, axis=1)
            payoff = np.maximum(0, total_dispersion - self.strike)
            return payoff
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")
