"""
Leveraged Position Payoffs (4 payoffs)

These represent leveraged basket positions with optional stop-loss protection.
Unlike traditional options, these are position values (can be negative).

Notation:
- S̄(t) = basket average at time t
- S̄(0) = initial basket average
- N = notional amount invested
- L = leverage factor (e.g., 2x, 3x)
- B_L = stop-loss barrier for long positions (< 1)
- B_U = stop-loss barrier for short positions (> 1)

WARNING: These payoffs can be NEGATIVE (losses).
"""

import numpy as np
from optimal_stopping.payoffs.payoff import Payoff


class LeveragedBasketLongPosition(Payoff):
    """
    Leveraged Long Position on Basket

    Payoff: N × L × (S̄(t)/S̄(0) - 1)

    Returns the leveraged return on the basket.
    - If basket up 10%, 2x leveraged position returns 20%
    - If basket down 10%, 2x leveraged position loses 20%

    Can be NEGATIVE (losses).
    """

    is_path_dependent = False

    def __init__(self, strike=100, notional=1.0, leverage=2.0):
        """
        Args:
            strike: Not used for leverage products, kept for API compatibility
            notional: Notional amount invested (N)
            leverage: Leverage factor (L), e.g., 2.0 for 2x leverage
        """
        super().__init__(strike)
        self.notional = notional
        self.leverage = leverage

    def eval(self, X):
        """
        Args:
            X: Array of shape (nb_paths, nb_stocks, nb_dates+1)
        """
        if X.ndim == 3:
            # Initial basket average
            initial_basket = np.mean(X[:, :, 0], axis=1)  # (nb_paths,)
            # Terminal basket average
            terminal_basket = np.mean(X[:, :, -1], axis=1)  # (nb_paths,)

            # Basket return ratio
            basket_ratio = terminal_basket / initial_basket

            # Leveraged position value
            payoff = self.notional * self.leverage * (basket_ratio - 1.0)
            return payoff
        elif X.ndim == 2:
            # For 2D case, assume first column is initial, last is terminal
            # This is a simplified handling for single timestep
            initial_basket = np.mean(X[:, 0:1], axis=1) if X.shape[1] > 1 else np.ones(X.shape[0]) * self.strike
            terminal_basket = np.mean(X, axis=1)
            basket_ratio = terminal_basket / initial_basket
            payoff = self.notional * self.leverage * (basket_ratio - 1.0)
            return payoff
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


class LeveragedBasketShortPosition(Payoff):
    """
    Leveraged Short Position on Basket

    Payoff: N × L × (1 - S̄(t)/S̄(0))

    Returns the leveraged inverse return on the basket.
    - If basket up 10%, 2x short position loses 20%
    - If basket down 10%, 2x short position returns 20%

    Can be NEGATIVE (losses).
    """

    is_path_dependent = False

    def __init__(self, strike=100, notional=1.0, leverage=2.0):
        """
        Args:
            strike: Not used for leverage products, kept for API compatibility
            notional: Notional amount invested (N)
            leverage: Leverage factor (L), e.g., 2.0 for 2x leverage
        """
        super().__init__(strike)
        self.notional = notional
        self.leverage = leverage

    def eval(self, X):
        if X.ndim == 3:
            initial_basket = np.mean(X[:, :, 0], axis=1)
            terminal_basket = np.mean(X[:, :, -1], axis=1)
            basket_ratio = terminal_basket / initial_basket

            # Short position: profit when basket falls
            payoff = self.notional * self.leverage * (1.0 - basket_ratio)
            return payoff
        elif X.ndim == 2:
            initial_basket = np.mean(X[:, 0:1], axis=1) if X.shape[1] > 1 else np.ones(X.shape[0]) * self.strike
            terminal_basket = np.mean(X, axis=1)
            basket_ratio = terminal_basket / initial_basket
            payoff = self.notional * self.leverage * (1.0 - basket_ratio)
            return payoff
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


class LeveragedBasketLongStopLoss(Payoff):
    """
    Leveraged Long Position with Stop-Loss Protection

    Payoff:
    - If min(S̄(τ)/S̄(0)) > B_L for all τ ≤ t:
        N × L × (S̄(t)/S̄(0) - 1)
    - Otherwise (stop-loss triggered):
        N × L × (B_L - 1)

    Stop-loss barrier B_L (e.g., 0.9 = 10% loss limit) locks in the loss
    when basket falls below this level at any time during the path.

    PATH-DEPENDENT: Needs to track minimum basket ratio.
    """

    is_path_dependent = True

    def __init__(self, strike=100, notional=1.0, leverage=2.0, barrier_stop_loss=0.9):
        """
        Args:
            strike: Not used, kept for API compatibility
            notional: Notional amount invested (N)
            leverage: Leverage factor (L)
            barrier_stop_loss: Stop-loss barrier B_L < 1 (default: 0.9 = 10% loss)
        """
        super().__init__(strike)
        self.notional = notional
        self.leverage = leverage
        self.barrier_stop_loss = barrier_stop_loss

    def eval(self, X):
        if X.ndim == 3:
            nb_paths, nb_stocks, nb_dates = X.shape

            # Initial basket
            initial_basket = np.mean(X[:, :, 0], axis=1)  # (nb_paths,)

            # Basket at each time
            basket_over_time = np.mean(X, axis=1)  # (nb_paths, nb_dates)

            # Basket ratio over time
            basket_ratio_over_time = basket_over_time / initial_basket[:, np.newaxis]

            # Check if stop-loss was triggered
            min_basket_ratio = np.min(basket_ratio_over_time, axis=1)
            stop_loss_triggered = min_basket_ratio <= self.barrier_stop_loss

            # Terminal basket ratio
            terminal_basket_ratio = basket_ratio_over_time[:, -1]

            # Compute payoff
            # If stop-loss NOT triggered: normal leveraged return
            # If stop-loss triggered: lock in loss at barrier level
            payoff = np.where(
                stop_loss_triggered,
                self.notional * self.leverage * (self.barrier_stop_loss - 1.0),
                self.notional * self.leverage * (terminal_basket_ratio - 1.0)
            )
            return payoff
        elif X.ndim == 2:
            # Simplified 2D handling (not truly path-dependent in this case)
            initial_basket = np.mean(X[:, 0:1], axis=1) if X.shape[1] > 1 else np.ones(X.shape[0]) * self.strike
            terminal_basket = np.mean(X, axis=1)
            min_basket = np.min(X, axis=1)
            basket_ratio = terminal_basket / initial_basket
            min_ratio = min_basket / initial_basket
            stop_loss_triggered = min_ratio <= self.barrier_stop_loss
            payoff = np.where(
                stop_loss_triggered,
                self.notional * self.leverage * (self.barrier_stop_loss - 1.0),
                self.notional * self.leverage * (basket_ratio - 1.0)
            )
            return payoff
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")


class LeveragedBasketShortStopLoss(Payoff):
    """
    Leveraged Short Position with Stop-Loss Protection

    Payoff:
    - If max(S̄(τ)/S̄(0)) < B_U for all τ ≤ t:
        N × L × (1 - S̄(t)/S̄(0))
    - Otherwise (stop-loss triggered):
        N × L × (1 - B_U)

    Stop-loss barrier B_U (e.g., 1.1 = 10% loss limit) locks in the loss
    when basket rises above this level at any time during the path.

    PATH-DEPENDENT: Needs to track maximum basket ratio.
    """

    is_path_dependent = True

    def __init__(self, strike=100, notional=1.0, leverage=2.0, barrier_stop_loss=1.1):
        """
        Args:
            strike: Not used, kept for API compatibility
            notional: Notional amount invested (N)
            leverage: Leverage factor (L)
            barrier_stop_loss: Stop-loss barrier B_U > 1 (default: 1.1 = 10% loss)
        """
        super().__init__(strike)
        self.notional = notional
        self.leverage = leverage
        self.barrier_stop_loss = barrier_stop_loss

    def eval(self, X):
        if X.ndim == 3:
            nb_paths, nb_stocks, nb_dates = X.shape

            # Initial basket
            initial_basket = np.mean(X[:, :, 0], axis=1)

            # Basket at each time
            basket_over_time = np.mean(X, axis=1)  # (nb_paths, nb_dates)

            # Basket ratio over time
            basket_ratio_over_time = basket_over_time / initial_basket[:, np.newaxis]

            # Check if stop-loss was triggered
            max_basket_ratio = np.max(basket_ratio_over_time, axis=1)
            stop_loss_triggered = max_basket_ratio >= self.barrier_stop_loss

            # Terminal basket ratio
            terminal_basket_ratio = basket_ratio_over_time[:, -1]

            # Compute payoff
            # If stop-loss NOT triggered: normal short leveraged return
            # If stop-loss triggered: lock in loss at barrier level
            payoff = np.where(
                stop_loss_triggered,
                self.notional * self.leverage * (1.0 - self.barrier_stop_loss),
                self.notional * self.leverage * (1.0 - terminal_basket_ratio)
            )
            return payoff
        elif X.ndim == 2:
            # Simplified 2D handling
            initial_basket = np.mean(X[:, 0:1], axis=1) if X.shape[1] > 1 else np.ones(X.shape[0]) * self.strike
            terminal_basket = np.mean(X, axis=1)
            max_basket = np.max(X, axis=1)
            basket_ratio = terminal_basket / initial_basket
            max_ratio = max_basket / initial_basket
            stop_loss_triggered = max_ratio >= self.barrier_stop_loss
            payoff = np.where(
                stop_loss_triggered,
                self.notional * self.leverage * (1.0 - self.barrier_stop_loss),
                self.notional * self.leverage * (1.0 - basket_ratio)
            )
            return payoff
        else:
            raise ValueError(f"Invalid dimensions: {X.ndim}")
