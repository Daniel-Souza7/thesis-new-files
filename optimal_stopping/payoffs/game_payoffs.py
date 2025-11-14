"""Game Payoffs - Special challenge payoffs with varying difficulty levels.

These payoffs combine multiple concepts like barriers, lookbacks, and custom basket calculations.
Difficulty levels: MEDIUM, HARD, IMPOSSIBLE
"""

import numpy as np
from optimal_stopping.payoffs.payoff import Payoff


# ============================================================================
# MEDIUM DIFFICULTY
# ============================================================================

class UpAndOutCall(Payoff):
    """Up-and-out call option for a single stock.

    Knocks out (becomes worthless) if the stock price goes above the barrier.
    Otherwise pays max(S(T) - K, 0).

    Difficulty: MEDIUM
    Stocks: 1
    """
    is_path_dependent = True

    def __init__(self, strike, barrier):
        super().__init__(strike)
        self.barrier = barrier

    def eval(self, X):
        """Evaluate payoff.

        Args:
            X: shape (nb_paths, 1, nb_dates+1) - single stock price paths
        """
        # Check if maximum price along path exceeds barrier
        max_price = np.max(X[:, 0, :], axis=1)
        knocked_out = max_price >= self.barrier

        # Terminal payoff
        terminal_price = X[:, 0, -1]
        payoff = np.maximum(0, terminal_price - self.strike)

        # Zero out knocked-out paths
        payoff[knocked_out] = 0

        return payoff


class DownAndOutBasketPut(Payoff):
    """Down-and-out put on arithmetic basket.

    Knocks out if the basket average goes below the barrier.
    Otherwise pays max(K - basket_avg(T), 0).

    Difficulty: MEDIUM
    Stocks: Multiple
    """
    is_path_dependent = True

    def __init__(self, strike, barrier):
        super().__init__(strike)
        self.barrier = barrier

    def eval(self, X):
        """Evaluate payoff.

        Args:
            X: shape (nb_paths, nb_stocks, nb_dates+1)
        """
        # Calculate basket average at each time step
        basket_prices = np.mean(X, axis=1)  # (nb_paths, nb_dates+1)

        # Check if minimum basket price goes below barrier
        min_basket = np.min(basket_prices, axis=1)
        knocked_out = min_basket <= self.barrier

        # Terminal payoff
        terminal_basket = basket_prices[:, -1]
        payoff = np.maximum(0, self.strike - terminal_basket)

        # Zero out knocked-out paths
        payoff[knocked_out] = 0

        return payoff


class DoubleBarrierMaxCall(Payoff):
    """Double barrier max call option.

    Knocks out if the maximum of all stocks goes above barrier_up or below barrier_down.
    Otherwise pays max(max_i(S_i(T)) - K, 0).

    Difficulty: MEDIUM
    Stocks: Multiple
    """
    is_path_dependent = True

    def __init__(self, strike, barrier_up, barrier_down):
        super().__init__(strike)
        self.barrier_up = barrier_up
        self.barrier_down = barrier_down

    def eval(self, X):
        """Evaluate payoff.

        Args:
            X: shape (nb_paths, nb_stocks, nb_dates+1)
        """
        # For each path, find the maximum stock price at each time step
        max_across_stocks = np.max(X, axis=1)  # (nb_paths, nb_dates+1)

        # Check barriers
        max_overall = np.max(max_across_stocks, axis=1)
        min_overall = np.min(max_across_stocks, axis=1)

        knocked_out = (max_overall >= self.barrier_up) | (min_overall <= self.barrier_down)

        # Terminal payoff: max of all stocks
        terminal_max = max_across_stocks[:, -1]
        payoff = np.maximum(0, terminal_max - self.strike)

        # Zero out knocked-out paths
        payoff[knocked_out] = 0

        return payoff


# ============================================================================
# HARD DIFFICULTY
# ============================================================================

class GameStepBarrierCall(Payoff):
    """Call option with stochastic step barrier (upper only).

    The upper barrier starts at initial_barrier and at each time step,
    a random value from uniform(-2, 1) is added to create a random walk barrier.
    The option knocks out if price exceeds the time-varying barrier.

    Difficulty: HARD
    Stocks: Multiple (applies to basket average)
    """
    is_path_dependent = True

    def __init__(self, strike, initial_barrier, seed=42):
        super().__init__(strike)
        self.initial_barrier = initial_barrier
        self.seed = seed

    def eval(self, X):
        """Evaluate payoff.

        Args:
            X: shape (nb_paths, nb_stocks, nb_dates+1)
        """
        nb_paths, nb_stocks, nb_dates_plus_1 = X.shape
        nb_dates = nb_dates_plus_1 - 1

        # Generate stochastic barrier path (same for all paths)
        rng = np.random.RandomState(self.seed)
        barrier_steps = rng.uniform(-2, 1, size=nb_dates)
        barrier_path = np.zeros(nb_dates_plus_1)
        barrier_path[0] = self.initial_barrier

        for t in range(1, nb_dates_plus_1):
            barrier_path[t] = barrier_path[t-1] + barrier_steps[t-1]

        # Calculate basket average at each time step
        basket_prices = np.mean(X, axis=1)  # (nb_paths, nb_dates+1)

        # Check if basket ever exceeds the time-varying barrier
        knocked_out = np.any(basket_prices >= barrier_path[np.newaxis, :], axis=1)

        # Terminal payoff
        terminal_basket = basket_prices[:, -1]
        payoff = np.maximum(0, terminal_basket - self.strike)

        # Zero out knocked-out paths
        payoff[knocked_out] = 0

        return payoff


class GameUpAndOutMinPut(Payoff):
    """Up-and-out put on minimum of stocks.

    Knocks out if the minimum of all stocks goes above the barrier.
    Otherwise pays max(K - min_i(S_i(T)), 0).

    Difficulty: HARD
    Stocks: Multiple
    """
    is_path_dependent = True

    def __init__(self, strike, barrier):
        super().__init__(strike)
        self.barrier = barrier

    def eval(self, X):
        """Evaluate payoff.

        Args:
            X: shape (nb_paths, nb_stocks, nb_dates+1)
        """
        # For each path, find the minimum stock price at each time step
        min_across_stocks = np.min(X, axis=1)  # (nb_paths, nb_dates+1)

        # Check if minimum ever goes above barrier (knocks out)
        max_of_mins = np.max(min_across_stocks, axis=1)
        knocked_out = max_of_mins >= self.barrier

        # Terminal payoff: put on minimum
        terminal_min = min_across_stocks[:, -1]
        payoff = np.maximum(0, self.strike - terminal_min)

        # Zero out knocked-out paths
        payoff[knocked_out] = 0

        return payoff


class DownAndOutBestOfKCall(Payoff):
    """Down-and-out call on the average of the top K stocks.

    Knocks out if the basket average goes below the barrier.
    Otherwise pays max(avg(top_K_stocks(T)) - K, 0).

    Difficulty: HARD
    Stocks: Multiple (at least k stocks)
    """
    is_path_dependent = True

    def __init__(self, strike, barrier, k=2):
        super().__init__(strike)
        self.barrier = barrier
        self.k = k

    def eval(self, X):
        """Evaluate payoff.

        Args:
            X: shape (nb_paths, nb_stocks, nb_dates+1)
        """
        # Calculate basket average at each time step for barrier check
        basket_prices = np.mean(X, axis=1)  # (nb_paths, nb_dates+1)

        # Check if basket goes below barrier
        min_basket = np.min(basket_prices, axis=1)
        knocked_out = min_basket <= self.barrier

        # Terminal payoff: average of top k stocks
        terminal_prices = X[:, :, -1]  # (nb_paths, nb_stocks)
        sorted_prices = np.sort(terminal_prices, axis=1)[:, ::-1]  # Sort descending
        top_k_avg = np.mean(sorted_prices[:, :self.k], axis=1)

        payoff = np.maximum(0, top_k_avg - self.strike)

        # Zero out knocked-out paths
        payoff[knocked_out] = 0

        return payoff


# ============================================================================
# IMPOSSIBLE DIFFICULTY
# ============================================================================

class DoubleBarrierLookbackFloatingPut(Payoff):
    """Double barrier lookback floating strike put.

    Knocks out if price goes above barrier_up or below barrier_down.
    Otherwise pays max(max(S(0:T)) - S(T), 0) (floating strike lookback put).

    Difficulty: IMPOSSIBLE
    Stocks: 1
    """
    is_path_dependent = True

    def __init__(self, strike, barrier_up, barrier_down):
        super().__init__(strike)
        self.barrier_up = barrier_up
        self.barrier_down = barrier_down

    def eval(self, X):
        """Evaluate payoff.

        Args:
            X: shape (nb_paths, 1, nb_dates+1)
        """
        prices = X[:, 0, :]  # (nb_paths, nb_dates+1)

        # Check double barrier
        max_price = np.max(prices, axis=1)
        min_price = np.min(prices, axis=1)
        knocked_out = (max_price >= self.barrier_up) | (min_price <= self.barrier_down)

        # Lookback floating put payoff
        max_along_path = np.max(prices, axis=1)
        terminal_price = prices[:, -1]
        payoff = np.maximum(0, max_along_path - terminal_price)

        # Zero out knocked-out paths
        payoff[knocked_out] = 0

        return payoff


class DoubleBarrierRankWeightedBasketCall(Payoff):
    """Double barrier call on rank-weighted basket of exactly 3 stocks.

    At each time step, stocks are ranked by price:
    - 1st (highest): 15% weight
    - 2nd (middle): 50% weight
    - 3rd (lowest): 35% weight

    The weighted basket knocks out if it goes above barrier_up or below barrier_down.
    Otherwise pays max(weighted_basket(T) - K, 0).

    Difficulty: IMPOSSIBLE
    Stocks: Exactly 3
    """
    is_path_dependent = True

    def __init__(self, strike, barrier_up, barrier_down):
        super().__init__(strike)
        self.barrier_up = barrier_up
        self.barrier_down = barrier_down
        # Weights: [1st place, 2nd place, 3rd place]
        self.weights = np.array([0.15, 0.50, 0.35])

    def eval(self, X):
        """Evaluate payoff.

        Args:
            X: shape (nb_paths, 3, nb_dates+1) - exactly 3 stocks required
        """
        nb_paths, nb_stocks, nb_dates_plus_1 = X.shape
        assert nb_stocks == 3, "DoubleBarrierRankWeightedBasketCall requires exactly 3 stocks"

        # Calculate rank-weighted basket at each time step
        weighted_basket = np.zeros((nb_paths, nb_dates_plus_1))

        for t in range(nb_dates_plus_1):
            prices_t = X[:, :, t]  # (nb_paths, 3)

            # Sort prices in descending order (rank 1 = highest)
            sorted_prices = np.sort(prices_t, axis=1)[:, ::-1]  # (nb_paths, 3)

            # Apply weights: 1st=15%, 2nd=50%, 3rd=35%
            weighted_basket[:, t] = np.sum(sorted_prices * self.weights[np.newaxis, :], axis=1)

        # Check double barrier
        max_basket = np.max(weighted_basket, axis=1)
        min_basket = np.min(weighted_basket, axis=1)
        knocked_out = (max_basket >= self.barrier_up) | (min_basket <= self.barrier_down)

        # Terminal payoff
        terminal_basket = weighted_basket[:, -1]
        payoff = np.maximum(0, terminal_basket - self.strike)

        # Zero out knocked-out paths
        payoff[knocked_out] = 0

        return payoff


class DoubleStepBarrierDispersionCall(Payoff):
    """Dispersion call with double stochastic step barriers.

    Barriers:
    - Lower barrier: starts at barrier_down, each step adds uniform(-1, 2)
    - Upper barrier: starts at barrier_up, each step adds uniform(-2, 1)

    Payoff: max(0, Σ|S_i(T) - S̄(T)| - K)
    where S̄(T) is the basket average at maturity.

    Knocks out if the basket average breaches either time-varying barrier.

    Difficulty: IMPOSSIBLE
    Stocks: Multiple
    """
    is_path_dependent = True

    def __init__(self, strike, barrier_up, barrier_down, seed=42):
        super().__init__(strike)
        self.barrier_up = barrier_up
        self.barrier_down = barrier_down
        self.seed = seed

    def eval(self, X):
        """Evaluate payoff.

        Args:
            X: shape (nb_paths, nb_stocks, nb_dates+1)
        """
        nb_paths, nb_stocks, nb_dates_plus_1 = X.shape
        nb_dates = nb_dates_plus_1 - 1

        # Generate stochastic barrier paths
        rng = np.random.RandomState(self.seed)

        # Lower barrier: adds uniform(-1, 2) at each step
        lower_steps = rng.uniform(-1, 2, size=nb_dates)
        lower_barrier_path = np.zeros(nb_dates_plus_1)
        lower_barrier_path[0] = self.barrier_down

        for t in range(1, nb_dates_plus_1):
            lower_barrier_path[t] = lower_barrier_path[t-1] + lower_steps[t-1]

        # Upper barrier: adds uniform(-2, 1) at each step
        upper_steps = rng.uniform(-2, 1, size=nb_dates)
        upper_barrier_path = np.zeros(nb_dates_plus_1)
        upper_barrier_path[0] = self.barrier_up

        for t in range(1, nb_dates_plus_1):
            upper_barrier_path[t] = upper_barrier_path[t-1] + upper_steps[t-1]

        # Calculate basket average at each time step
        basket_prices = np.mean(X, axis=1)  # (nb_paths, nb_dates+1)

        # Check if basket breaches either barrier
        breaches_upper = np.any(basket_prices >= upper_barrier_path[np.newaxis, :], axis=1)
        breaches_lower = np.any(basket_prices <= lower_barrier_path[np.newaxis, :], axis=1)
        knocked_out = breaches_upper | breaches_lower

        # Dispersion payoff at maturity: Σ|S_i(T) - S̄(T)|
        terminal_prices = X[:, :, -1]  # (nb_paths, nb_stocks)
        terminal_basket = basket_prices[:, -1]  # (nb_paths,)

        # Calculate dispersion: sum of absolute deviations from mean
        deviations = np.abs(terminal_prices - terminal_basket[:, np.newaxis])
        total_dispersion = np.sum(deviations, axis=1)

        payoff = np.maximum(0, total_dispersion - self.strike)

        # Zero out knocked-out paths
        payoff[knocked_out] = 0

        return payoff
