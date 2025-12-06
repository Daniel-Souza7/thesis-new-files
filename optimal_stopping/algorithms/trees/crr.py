"""
Cox-Ross-Rubinstein (CRR) Binomial Tree for American option pricing.

Implements the classic binomial tree method from:
Cox, J. C., Ross, S. A., & Rubinstein, M. (1979).
"Option pricing: A simplified approach." Journal of Financial Economics.

The CRR model uses a recombining binomial tree with u = exp(σ√Δt) and d = 1/u,
which ensures the tree converges to the Black-Scholes model as N → ∞.
"""

import numpy as np
import math
import time
from optimal_stopping.run import configs


class CRRTree:
    """
    Cox-Ross-Rubinstein binomial tree for American option pricing.

    Uses a recombining lattice with symmetric up/down moves to value options
    via backward induction. Suitable for non-path-dependent payoffs only.
    """

    def __init__(self, model, payoff, nb_epochs=None, hidden_size=None,
                 factors=None, train_ITM_only=True, use_payoff_as_input=False,
                 n_steps=50):
        """
        Initialize CRR tree pricer.

        Args:
            model: Stock model (must be BlackScholes or compatible)
            payoff: Payoff function (must be non-path-dependent)
            nb_epochs: Ignored (for API compatibility)
            hidden_size: Ignored (for API compatibility)
            factors: Ignored (for API compatibility)
            train_ITM_only: Ignored (for API compatibility)
            use_payoff_as_input: Ignored (for API compatibility)
            n_steps: Number of time steps in the tree (default: 50)
        """
        self.model = model
        self.payoff = payoff
        self.n_steps = n_steps

        # Validate compatibility
        if payoff.is_path_dependent:
            raise ValueError(
                "CRR tree does not support path-dependent payoffs. "
                "Use standard payoffs only (e.g., BasketCall, MaxCall, Put)."
            )

        if model.nb_stocks > 1:
            raise ValueError(
                "CRR tree currently only supports single-asset options (nb_stocks=1). "
                "For basket options, use Monte Carlo methods instead."
            )

        # Check model compatibility
        model_name = type(model).__name__
        if model_name not in ['BlackScholes', 'BlackScholesModel']:
            import warnings
            warnings.warn(
                f"CRR tree is designed for constant volatility (BlackScholes) models. "
                f"Your model is {model_name}. Results may be inaccurate.",
                UserWarning
            )

        # Extract model parameters
        self.S0 = model.spot  # Initial stock price (scalar for nb_stocks=1)
        self.r = model.rate   # Risk-free rate
        self.T = model.maturity
        self.sigma = model.vol  # Volatility (scalar for nb_stocks=1)

        # Ensure scalar values for single-asset case
        if isinstance(self.S0, np.ndarray):
            self.S0 = self.S0[0]
        if isinstance(self.sigma, np.ndarray):
            self.sigma = self.sigma[0]

        # Tree parameters
        self.dt = self.T / self.n_steps
        self.disc = math.exp(-self.r * self.dt)  # Discount factor

        # CRR parameters: u = exp(σ√Δt), d = 1/u
        self.u = math.exp(self.sigma * math.sqrt(self.dt))
        self.d = 1.0 / self.u

        # Risk-neutral probability
        self.p = (math.exp(self.r * self.dt) - self.d) / (self.u - self.d)

        # Storage for optimal stopping boundary (for exercise time calculation)
        self._stopping_boundary = None

    def price(self, train_eval_split=2):
        """
        Compute option price using CRR binomial tree.

        Args:
            train_eval_split: Ignored (trees don't use train/eval split)

        Returns:
            tuple: (price, computation_time)
        """
        t_start = time.time()

        # Build stock price tree
        stock_tree = np.zeros((self.n_steps + 1, self.n_steps + 1))

        # Fill the tree: S[i,j] = S0 * u^j * d^(i-j)
        # i = time step, j = number of up moves
        for i in range(self.n_steps + 1):
            for j in range(i + 1):
                stock_tree[i, j] = self.S0 * (self.u ** j) * (self.d ** (i - j))

        # Initialize option value tree
        option_tree = np.zeros((self.n_steps + 1, self.n_steps + 1))

        # Compute payoffs at maturity (time step n_steps)
        # Reshape for payoff.eval: needs (nb_paths, nb_stocks)
        terminal_prices = stock_tree[self.n_steps, :self.n_steps + 1].reshape(-1, 1)
        terminal_payoffs = self.payoff.eval(terminal_prices)
        option_tree[self.n_steps, :self.n_steps + 1] = terminal_payoffs

        # Storage for stopping boundary (critical stock price at each time)
        self._stopping_boundary = np.full(self.n_steps + 1, np.nan)

        # Backward induction
        for i in range(self.n_steps - 1, -1, -1):
            for j in range(i + 1):
                # Continuation value (expected discounted value)
                continuation = self.disc * (
                    self.p * option_tree[i + 1, j + 1] +
                    (1 - self.p) * option_tree[i + 1, j]
                )

                # Immediate exercise value
                stock_price = stock_tree[i, j].reshape(1, 1)  # (1, 1) for payoff.eval
                exercise = self.payoff.eval(stock_price)[0]

                # American option: max of exercise and continuation
                option_tree[i, j] = max(exercise, continuation)

                # Track stopping boundary (lowest stock price where we exercise)
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
            # Run pricing if not done yet
            self.price()

        # Compute expected exercise time by Monte Carlo simulation on the tree
        # Generate paths and see when they first cross the stopping boundary
        n_sim = 10000
        exercise_times = np.zeros(n_sim)

        for path_idx in range(n_sim):
            S = self.S0
            for i in range(self.n_steps + 1):
                # Check if we should exercise at this time
                if not np.isnan(self._stopping_boundary[i]):
                    # Immediate exercise value
                    stock_price = S.reshape(1, 1) if np.isscalar(S) else S.reshape(1, 1)
                    exercise_value = self.payoff.eval(stock_price)[0]

                    # Exercise if payoff is positive and we're at/above boundary
                    if exercise_value > 0 and S >= self._stopping_boundary[i]:
                        exercise_times[path_idx] = i
                        break

                # Move to next step
                if i < self.n_steps:
                    if np.random.rand() < self.p:
                        S *= self.u
                    else:
                        S *= self.d
            else:
                # Exercised at maturity
                exercise_times[path_idx] = self.n_steps

        # Normalize to [0, 1]
        avg_exercise_time = exercise_times.mean() / self.n_steps
        return avg_exercise_time

    def backward_induction_on_paths(self, stock_paths, var_paths=None):
        """
        Apply the learned optimal stopping policy to new paths.

        Note: For trees, this is an approximation since the policy is defined
        on a discrete lattice, not continuous paths.

        Args:
            stock_paths: Array of shape (nb_paths, nb_stocks, nb_dates+1)
            var_paths: Ignored

        Returns:
            tuple: (exercise_dates, payoff_values, price)
        """
        # For compatibility with the standard interface
        # Use the stopping boundary to determine exercise times

        if self._stopping_boundary is None:
            self.price()

        nb_paths = stock_paths.shape[0]
        exercise_dates = np.full(nb_paths, self.n_steps, dtype=int)
        payoff_values = np.zeros(nb_paths)

        # Map model dates to tree steps
        # Assuming stock_paths has shape (nb_paths, 1, nb_dates+1)
        nb_dates = stock_paths.shape[2] - 1

        for path_idx in range(nb_paths):
            for date_idx in range(nb_dates + 1):
                # Map date to tree step
                tree_step = int(date_idx * self.n_steps / nb_dates)

                S = stock_paths[path_idx, 0, date_idx]

                # Check stopping condition
                if not np.isnan(self._stopping_boundary[tree_step]):
                    stock_price = np.array([[S]])
                    exercise_value = self.payoff.eval(stock_price)[0]

                    if exercise_value > 0 and S >= self._stopping_boundary[tree_step]:
                        exercise_dates[path_idx] = date_idx
                        payoff_values[path_idx] = exercise_value
                        break
            else:
                # Exercise at maturity
                stock_price = np.array([[stock_paths[path_idx, 0, -1]]])
                payoff_values[path_idx] = self.payoff.eval(stock_price)[0]

        # Compute price with discounting
        discount_factors = np.exp(-self.r * self.T * exercise_dates / nb_dates)
        price = np.mean(payoff_values * discount_factors)

        return exercise_dates, payoff_values, price
