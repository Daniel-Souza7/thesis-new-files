"""
Trinomial Tree for American option pricing.

Implements the three-jump process from:
Boyle, P. P. (1986).
"Option Valuation using a Three-Jump Process."
International Options Journal.

The trinomial tree allows the stock to move Up, Down, or stay Middle (flat),
providing an extra degree of freedom that improves stability and flexibility
compared to binomial trees.
"""

import numpy as np
import math
import time
from optimal_stopping.run import configs


class TrinomialTree:
    """
    Trinomial tree for American option pricing.

    Uses a three-branch lattice (up, middle, down) to value options via
    backward induction. The extra degree of freedom provides better numerical
    stability and can handle time-dependent parameters more easily.
    Suitable for non-path-dependent payoffs only.
    """

    def __init__(self, model, payoff, nb_epochs=None, hidden_size=None,
                 factors=None, train_ITM_only=True, use_payoff_as_input=False,
                 n_steps=50):
        """
        Initialize trinomial tree pricer.

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
                "Trinomial tree does not support path-dependent payoffs. "
                "Use standard payoffs only (e.g., BasketCall, MaxCall, Put)."
            )

        if model.nb_stocks > 1:
            raise ValueError(
                "Trinomial tree currently only supports single-asset options (nb_stocks=1). "
                "For basket options, use Monte Carlo methods instead."
            )

        # Check model compatibility
        model_name = type(model).__name__
        if model_name not in ['BlackScholes', 'BlackScholesModel']:
            import warnings
            warnings.warn(
                f"Trinomial tree is designed for constant volatility (BlackScholes) models. "
                f"Your model is {model_name}. Results may be inaccurate.",
                UserWarning
            )

        # Extract model parameters
        self.S0 = model.spot
        self.r = model.rate
        self.T = model.maturity
        self.sigma = model.vol

        # Ensure scalar values
        if isinstance(self.S0, np.ndarray):
            self.S0 = self.S0[0]
        if isinstance(self.sigma, np.ndarray):
            self.sigma = self.sigma[0]

        # Tree parameters
        self.dt = self.T / self.n_steps
        self.disc = math.exp(-self.r * self.dt)

        # Trinomial tree parameters (Boyle's parameterization)
        # Use λ = √3 for better stability
        lambda_factor = math.sqrt(3)
        self.u = math.exp(self.sigma * math.sqrt(lambda_factor * self.dt))
        self.d = 1.0 / self.u
        self.m = 1.0  # Middle branch: no change

        # Risk-neutral probabilities
        # Match first two moments: E[S] and Var[S]
        dx = math.log(self.u)
        nu = self.r - 0.5 * self.sigma**2  # Drift in log-space

        # Probabilities from moment matching
        self.p_u = 0.5 * ((self.sigma**2 * self.dt + nu**2 * self.dt**2) / dx**2 +
                          nu * self.dt / dx)
        self.p_d = 0.5 * ((self.sigma**2 * self.dt + nu**2 * self.dt**2) / dx**2 -
                          nu * self.dt / dx)
        self.p_m = 1.0 - self.p_u - self.p_d

        # Validate probabilities
        if not (0 <= self.p_u <= 1 and 0 <= self.p_d <= 1 and 0 <= self.p_m <= 1):
            import warnings
            warnings.warn(
                f"Trinomial probabilities out of [0,1] range: "
                f"p_u={self.p_u:.4f}, p_m={self.p_m:.4f}, p_d={self.p_d:.4f}. "
                f"Try reducing n_steps or adjusting parameters.",
                UserWarning
            )

        # Storage for optimal stopping boundary
        self._stopping_boundary = None

    def price(self, train_eval_split=2):
        """
        Compute option price using trinomial tree.

        Args:
            train_eval_split: Ignored (trees don't use train/eval split)

        Returns:
            tuple: (price, computation_time)
        """
        t_start = time.time()

        # Build stock price tree (non-recombining in general, but we use recombining)
        # For trinomial: at step i, we have 2*i+1 possible nodes
        # Node j corresponds to j - i up moves (can be negative)
        stock_tree = {}
        option_tree = {}

        # Initialize at t=0
        stock_tree[(0, 0)] = self.S0

        # Forward pass: build stock price tree
        for i in range(self.n_steps):
            for j in range(-i, i + 1):
                if (i, j) not in stock_tree:
                    continue

                S = stock_tree[(i, j)]

                # Three branches
                stock_tree[(i + 1, j + 1)] = S * self.u  # Up
                stock_tree[(i + 1, j)] = S * self.m      # Middle
                stock_tree[(i + 1, j - 1)] = S * self.d  # Down

        # Compute payoffs at maturity
        i = self.n_steps
        for j in range(-i, i + 1):
            if (i, j) in stock_tree:
                stock_price = stock_tree[(i, j)].reshape(1, 1)
                option_tree[(i, j)] = self.payoff.eval(stock_price)[0]

        # Storage for stopping boundary (critical price at each time)
        self._stopping_boundary = {}

        # Backward induction
        for i in range(self.n_steps - 1, -1, -1):
            for j in range(-i, i + 1):
                if (i, j) not in stock_tree:
                    continue

                # Continuation value (expected discounted value)
                continuation = self.disc * (
                    self.p_u * option_tree.get((i + 1, j + 1), 0) +
                    self.p_m * option_tree.get((i + 1, j), 0) +
                    self.p_d * option_tree.get((i + 1, j - 1), 0)
                )

                # Immediate exercise value
                stock_price = stock_tree[(i, j)].reshape(1, 1)
                exercise = self.payoff.eval(stock_price)[0]

                # American option: max of exercise and continuation
                option_tree[(i, j)] = max(exercise, continuation)

                # Track stopping boundary
                if exercise > continuation:
                    if i not in self._stopping_boundary:
                        self._stopping_boundary[i] = stock_tree[(i, j)]
                    else:
                        # Store minimum exercise threshold
                        self._stopping_boundary[i] = min(
                            self._stopping_boundary[i],
                            stock_tree[(i, j)]
                        )

        price = option_tree[(0, 0)]
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
            j = 0  # Current position in the tree

            for i in range(self.n_steps + 1):
                # Check if we should exercise at this time
                if i in self._stopping_boundary:
                    stock_price = S.reshape(1, 1) if np.isscalar(S) else S.reshape(1, 1)
                    exercise_value = self.payoff.eval(stock_price)[0]

                    if exercise_value > 0 and S >= self._stopping_boundary[i]:
                        exercise_times[path_idx] = i
                        break

                # Move to next step
                if i < self.n_steps:
                    rand = np.random.rand()
                    if rand < self.p_u:
                        S *= self.u
                        j += 1
                    elif rand < self.p_u + self.p_m:
                        S *= self.m
                        # j stays same
                    else:
                        S *= self.d
                        j -= 1
            else:
                # Exercised at maturity
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
                # Map date to tree step
                tree_step = int(date_idx * self.n_steps / nb_dates)
                S = stock_paths[path_idx, 0, date_idx]

                # Check stopping condition
                if tree_step in self._stopping_boundary:
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
