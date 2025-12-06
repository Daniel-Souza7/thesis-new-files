"""
Leisen-Reimer (LR) Binomial Tree for American option pricing.

Implements the improved binomial tree method from:
Leisen, D., & Reimer, M. (1996).
"Binomial Models for Option Valuation - Examining and Improving Convergence."
Applied Mathematical Finance.

The LR model uses Peizer-Pratt inversion formulas to ensure the tree nodes
are centered around the strike price at maturity, achieving smooth second-order
convergence without the oscillations of the CRR model.
"""

import numpy as np
import math
import time
from scipy.stats import norm
from optimal_stopping.run import configs


class LeisenReimerTree:
    """
    Leisen-Reimer binomial tree for American option pricing.

    Uses Peizer-Pratt inversion to align tree nodes with the strike price,
    providing superior convergence properties compared to CRR.
    Suitable for non-path-dependent payoffs only.
    """

    def __init__(self, model, payoff, nb_epochs=None, hidden_size=None,
                 factors=None, train_ITM_only=True, use_payoff_as_input=False,
                 n_steps=51):
        """
        Initialize Leisen-Reimer tree pricer.

        Args:
            model: Stock model (must be BlackScholes or compatible)
            payoff: Payoff function (must be non-path-dependent)
            nb_epochs: Ignored (for API compatibility)
            hidden_size: Ignored (for API compatibility)
            factors: Ignored (for API compatibility)
            train_ITM_only: Ignored (for API compatibility)
            use_payoff_as_input: Ignored (for API compatibility)
            n_steps: Number of time steps (default: 51, should be odd for best results)
        """
        self.model = model
        self.payoff = payoff

        # Ensure odd number of steps for Peizer-Pratt inversion
        self.n_steps = n_steps if n_steps % 2 == 1 else n_steps + 1

        # Validate compatibility
        if payoff.is_path_dependent:
            raise ValueError(
                "Leisen-Reimer tree does not support path-dependent payoffs. "
                "Use standard payoffs only (e.g., BasketCall, MaxCall, Put)."
            )

        if model.nb_stocks > 1:
            raise ValueError(
                "Leisen-Reimer tree currently only supports single-asset options (nb_stocks=1). "
                "For basket options, use Monte Carlo methods instead."
            )

        # Check model compatibility
        model_name = type(model).__name__
        if model_name not in ['BlackScholes', 'BlackScholesModel']:
            import warnings
            warnings.warn(
                f"Leisen-Reimer tree is designed for constant volatility (BlackScholes) models. "
                f"Your model is {model_name}. Results may be inaccurate.",
                UserWarning
            )

        # Extract model parameters
        self.S0 = model.spot
        self.r = model.rate
        self.T = model.maturity
        self.sigma = model.vol
        self.K = payoff.strike  # Need strike for LR method

        # Ensure scalar values
        if isinstance(self.S0, np.ndarray):
            self.S0 = self.S0[0]
        if isinstance(self.sigma, np.ndarray):
            self.sigma = self.sigma[0]

        # Tree parameters
        self.dt = self.T / self.n_steps
        self.disc = math.exp(-self.r * self.dt)

        # Compute d1 and d2 from Black-Scholes
        d1 = (math.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / \
             (self.sigma * math.sqrt(self.T))
        d2 = d1 - self.sigma * math.sqrt(self.T)

        # Peizer-Pratt inversion to get probabilities
        # This ensures the tree is centered on the strike at maturity
        self.p_up = self._peizer_pratt_inversion(d1, self.n_steps)
        self.p_down = self._peizer_pratt_inversion(d2, self.n_steps)

        # Compute up and down factors from probabilities
        # These ensure the tree matches the risk-neutral drift
        self.u = math.exp(self.r * self.dt) * self.p_up / self.p_down
        self.d = (math.exp(self.r * self.dt) - self.p_down * self.u) / (1 - self.p_down)

        # Risk-neutral probability
        self.p = self.p_down

        # Storage for optimal stopping boundary
        self._stopping_boundary = None

    def _peizer_pratt_inversion(self, z, n):
        """
        Peizer-Pratt method 2 inversion formula.

        Computes h(z, n) ≈ Φ(z) where Φ is the cumulative normal distribution.
        This gives better accuracy than standard normal CDF for tree construction.

        Args:
            z: Standard normal deviate
            n: Number of time steps

        Returns:
            float: Inversion probability
        """
        # Peizer-Pratt formula
        z_prime = z / math.sqrt(n * self.dt / self.T)

        # Avoid division by zero
        if abs(z_prime) < 1e-10:
            return 0.5

        # PP inversion formula
        numerator = 0.5 + math.copysign(1, z_prime) * math.sqrt(
            0.25 - 0.25 * math.exp(
                -(z_prime / (n + 1.0/3.0 + 0.1/(n+1)))**2 * (n + 1.0/6.0)
            )
        )

        return numerator

    def price(self, train_eval_split=2):
        """
        Compute option price using Leisen-Reimer binomial tree.

        Args:
            train_eval_split: Ignored (trees don't use train/eval split)

        Returns:
            tuple: (price, computation_time)
        """
        t_start = time.time()

        # Build stock price tree
        stock_tree = np.zeros((self.n_steps + 1, self.n_steps + 1))

        # Fill the tree: S[i,j] = S0 * u^j * d^(i-j)
        for i in range(self.n_steps + 1):
            for j in range(i + 1):
                stock_tree[i, j] = self.S0 * (self.u ** j) * (self.d ** (i - j))

        # Initialize option value tree
        option_tree = np.zeros((self.n_steps + 1, self.n_steps + 1))

        # Compute payoffs at maturity
        terminal_prices = stock_tree[self.n_steps, :self.n_steps + 1].reshape(-1, 1)
        terminal_payoffs = self.payoff.eval(terminal_prices)
        option_tree[self.n_steps, :self.n_steps + 1] = terminal_payoffs

        # Storage for stopping boundary
        self._stopping_boundary = np.full(self.n_steps + 1, np.nan)

        # Backward induction
        for i in range(self.n_steps - 1, -1, -1):
            for j in range(i + 1):
                # Continuation value
                continuation = self.disc * (
                    self.p * option_tree[i + 1, j + 1] +
                    (1 - self.p) * option_tree[i + 1, j]
                )

                # Immediate exercise value
                stock_price = stock_tree[i, j].reshape(1, 1)
                exercise = self.payoff.eval(stock_price)[0]

                # American option: max of exercise and continuation
                option_tree[i, j] = max(exercise, continuation)

                # Track stopping boundary
                if exercise > continuation and np.isnan(self._stopping_boundary[i]):
                    self._stopping_boundary[i] = stock_tree[i, j]

        price = option_tree[0, 0]
        computation_time = time.time() - t_start

        return price, computation_time

    def get_exercise_time(self):
        """
        Compute expected exercise time from the optimal stopping boundary.

        Returns:
            float: Expected exercise time normalized to [0, 1]
        """
        if self._stopping_boundary is None:
            self.price()

        # Monte Carlo simulation on the tree
        n_sim = 10000
        exercise_times = np.zeros(n_sim)

        for path_idx in range(n_sim):
            S = self.S0
            for i in range(self.n_steps + 1):
                if not np.isnan(self._stopping_boundary[i]):
                    stock_price = S.reshape(1, 1) if np.isscalar(S) else S.reshape(1, 1)
                    exercise_value = self.payoff.eval(stock_price)[0]

                    if exercise_value > 0 and S >= self._stopping_boundary[i]:
                        exercise_times[path_idx] = i
                        break

                if i < self.n_steps:
                    if np.random.rand() < self.p:
                        S *= self.u
                    else:
                        S *= self.d
            else:
                exercise_times[path_idx] = self.n_steps

        avg_exercise_time = exercise_times.mean() / self.n_steps
        return avg_exercise_time

    def backward_induction_on_paths(self, stock_paths, var_paths=None):
        """
        Apply the learned optimal stopping policy to new paths.

        Args:
            stock_paths: Array of shape (nb_paths, nb_stocks, nb_dates+1)
            var_paths: Ignored

        Returns:
            tuple: (exercise_dates, payoff_values, price)
        """
        if self._stopping_boundary is None:
            self.price()

        nb_paths = stock_paths.shape[0]
        nb_dates = stock_paths.shape[2] - 1
        exercise_dates = np.full(nb_paths, nb_dates, dtype=int)
        payoff_values = np.zeros(nb_paths)

        for path_idx in range(nb_paths):
            for date_idx in range(nb_dates + 1):
                tree_step = int(date_idx * self.n_steps / nb_dates)
                S = stock_paths[path_idx, 0, date_idx]

                if not np.isnan(self._stopping_boundary[tree_step]):
                    stock_price = np.array([[S]])
                    exercise_value = self.payoff.eval(stock_price)[0]

                    if exercise_value > 0 and S >= self._stopping_boundary[tree_step]:
                        exercise_dates[path_idx] = date_idx
                        payoff_values[path_idx] = exercise_value
                        break
            else:
                stock_price = np.array([[stock_paths[path_idx, 0, -1]]])
                payoff_values[path_idx] = self.payoff.eval(stock_price)[0]

        discount_factors = np.exp(-self.r * self.T * exercise_dates / nb_dates)
        price = np.mean(payoff_values * discount_factors)

        return exercise_dates, payoff_values, price
