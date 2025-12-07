"""
Leisen-Reimer (LR) Binomial Tree for American option pricing.

Implements the improved binomial tree method from:
Leisen, D., & Reimer, M. (1996).
"Binomial Models for Option Valuation - Examining and Improving Convergence."
Applied Mathematical Finance.

Now supports multi-asset options (d=1,2,3) with correlation.
For d=1: Full Peizer-Pratt inversion for optimal convergence.
For d≥2: Uses LR marginal probabilities with correlation adjustments.
"""

import numpy as np
import math
import time
from scipy.stats import norm
from optimal_stopping.run import configs
import itertools


class LeisenReimerTree:
    """
    Leisen-Reimer binomial tree for American option pricing.

    Uses Peizer-Pratt inversion to align tree nodes with the strike price,
    providing superior convergence properties compared to CRR.
    Supports single-asset (d=1) and multi-asset (d=2,3) options with correlation.
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
        self.dim = int(model.nb_stocks)  # Number of assets (ensure plain Python int, avoid collision with self.d down-movement array)

        # Ensure odd number of steps for Peizer-Pratt inversion
        n_steps = int(n_steps)  # Ensure plain Python int
        self.n_steps = n_steps if n_steps % 2 == 1 else n_steps + 1

        # Validate compatibility
        if payoff.is_path_dependent:
            raise ValueError(
                "Leisen-Reimer tree does not support path-dependent payoffs. "
                "Use standard payoffs only (e.g., BasketCall, MaxCall, Put)."
            )

        # Warn about high dimensionality
        if self.dim > 3:
            import warnings
            warnings.warn(
                f"LR tree with {self.dim} assets will have ~{(self.n_steps+1)**self.dim:,} nodes. "
                f"This may be very slow or run out of memory. Consider using Monte Carlo methods (RLSM, RFQI).",
                UserWarning
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
        self.S0 = np.array(model.spot).flatten()
        self.r = model.rate
        self.T = model.maturity
        self.sigma = np.array(model.volatility).flatten()
        self.K = payoff.strike

        # Get correlation matrix
        if hasattr(model, 'correlation_matrix'):
            self.corr_matrix = model.correlation_matrix
            print(f"DEBUG LR __init__: Found correlation_matrix =\n{self.corr_matrix}")
        else:
            self.corr_matrix = np.eye(self.dim)
            print(f"DEBUG LR __init__: No correlation_matrix found, using identity for {self.dim} assets")

        # Tree parameters
        self.dt = self.T / self.n_steps
        self.disc = math.exp(-self.r * self.dt)

        # Compute tree parameters and probabilities
        self._compute_tree_parameters()

        # Storage for optimal stopping boundary
        self._stopping_boundary = None

    def _peizer_pratt_inversion(self, z, n):
        """
        Peizer-Pratt method 2 inversion formula.

        Computes h(z, n) ≈ Φ(z) where Φ is the cumulative normal distribution.

        Args:
            z: Standard normal deviate
            n: Number of time steps

        Returns:
            float: Inversion probability
        """
        z_prime = z / math.sqrt(n * self.dt / self.T)

        if abs(z_prime) < 1e-10:
            return 0.5

        numerator = 0.5 + math.copysign(1, z_prime) * math.sqrt(
            0.25 - 0.25 * math.exp(
                -(z_prime / (n + 1.0/3.0 + 0.1/(n+1)))**2 * (n + 1.0/6.0)
            )
        )

        return numerator

    def _compute_tree_parameters(self):
        """Compute tree parameters and joint probabilities."""
        d = self.dim

        if d == 1:
            # Single asset: Full LR method with Peizer-Pratt
            d1 = (math.log(self.S0[0] / self.K) + (self.r + 0.5 * self.sigma[0]**2) * self.T) / \
                 (self.sigma[0] * math.sqrt(self.T))
            d2 = d1 - self.sigma[0] * math.sqrt(self.T)

            p_up = self._peizer_pratt_inversion(d1, self.n_steps)
            p_down = self._peizer_pratt_inversion(d2, self.n_steps)

            self.u = np.array([math.exp(self.r * self.dt) * p_up / p_down])
            self.d = np.array([(math.exp(self.r * self.dt) - p_down * self.u[0]) / (1 - p_down)])
            self.joint_probs = np.array([1 - p_down, p_down])
            self.moves = [(0,), (1,)]

        elif d == 2:
            # Two assets: LR marginal probabilities + correlation
            rho = self.corr_matrix[0, 1]
            print(f"DEBUG LR _compute_tree_parameters: d=2, rho={rho}")

            # Compute LR marginal probabilities for each asset
            p_marginal = []
            self.u = np.zeros(d)
            self.d = np.zeros(d)

            for i in range(d):
                d1 = (math.log(self.S0[i] / self.K) + (self.r + 0.5 * self.sigma[i]**2) * self.T) / \
                     (self.sigma[i] * math.sqrt(self.T))
                d2 = d1 - self.sigma[i] * math.sqrt(self.T)

                p_up = self._peizer_pratt_inversion(d1, self.n_steps)
                p_down = self._peizer_pratt_inversion(d2, self.n_steps)

                self.u[i] = math.exp(self.r * self.dt) * p_up / p_down
                self.d[i] = (math.exp(self.r * self.dt) - p_down * self.u[i]) / (1 - p_down)
                p_marginal.append(p_down)

            print(f"DEBUG LR: Marginal probs p1={p_marginal[0]:.6f}, p2={p_marginal[1]:.6f}")

            # Apply correlation to joint probabilities (Boyle's formula)
            p1, p2 = p_marginal
            p_uu = p1 * p2 + rho * math.sqrt(p1 * (1-p1) * p2 * (1-p2))
            p_ud = p1 * (1-p2) - rho * math.sqrt(p1 * (1-p1) * p2 * (1-p2))
            p_du = (1-p1) * p2 - rho * math.sqrt(p1 * (1-p1) * p2 * (1-p2))
            p_dd = (1-p1) * (1-p2) + rho * math.sqrt(p1 * (1-p1) * p2 * (1-p2))
            print(f"DEBUG LR: Joint probs (before clip) p_uu={p_uu:.6f}, p_ud={p_ud:.6f}, p_du={p_du:.6f}, p_dd={p_dd:.6f}")

            self.joint_probs = np.array([p_dd, p_du, p_ud, p_uu])
            self.joint_probs = np.clip(self.joint_probs, 0, 1)
            self.joint_probs /= self.joint_probs.sum()
            print(f"DEBUG LR: Joint probs (after normalize) {self.joint_probs}")

            self.moves = [(0, 0), (0, 1), (1, 0), (1, 1)]

        else:
            # d > 2: Use LR marginal probabilities, ignore correlation
            import warnings
            if not np.allclose(self.corr_matrix, np.eye(d)):
                warnings.warn(
                    f"LR tree with {d} assets currently ignores correlation. "
                    f"Only marginal probabilities are matched.",
                    UserWarning
                )

            p_marginal = []
            self.u = np.zeros(d)
            self.d = np.zeros(d)

            for i in range(d):
                d1 = (math.log(self.S0[i] / self.K) + (self.r + 0.5 * self.sigma[i]**2) * self.T) / \
                     (self.sigma[i] * math.sqrt(self.T))
                d2 = d1 - self.sigma[i] * math.sqrt(self.T)

                p_up = self._peizer_pratt_inversion(d1, self.n_steps)
                p_down = self._peizer_pratt_inversion(d2, self.n_steps)

                self.u[i] = math.exp(self.r * self.dt) * p_up / p_down
                self.d[i] = (math.exp(self.r * self.dt) - p_down * self.u[i]) / (1 - p_down)
                p_marginal.append(p_down)

            # Independent joint probabilities
            self.moves = list(itertools.product([0, 1], repeat=d))
            self.joint_probs = np.array([
                np.prod([p_marginal[i] if move[i] == 1 else 1 - p_marginal[i]
                         for i in range(d)])
                for move in self.moves
            ])

    def price(self, train_eval_split=2):
        """
        Compute option price using Leisen-Reimer binomial tree.

        Args:
            train_eval_split: Ignored (trees don't use train/eval split)

        Returns:
            tuple: (price, computation_time)
        """
        t_start = time.time()

        d = self.dim  # Number of assets
        print(f"DEBUG LR price(): d={d}, self.S0={self.S0}, corr_matrix=\n{self.corr_matrix}")

        if d == 1:
            price = self._price_1d()
        elif d == 2:
            price = self._price_2d()
        else:
            price = self._price_nd()

        computation_time = time.time() - t_start
        print(f"DEBUG LR price(): Computed price={price:.6f}")
        return price, computation_time

    def _price_1d(self):
        """Price single-asset option using standard LR tree."""
        stock_tree = np.zeros((self.n_steps + 1, self.n_steps + 1))

        for i in range(self.n_steps + 1):
            for j in range(i + 1):
                stock_tree[i, j] = self.S0[0] * (self.u[0] ** j) * (self.d[0] ** (i - j))

        option_tree = np.zeros((self.n_steps + 1, self.n_steps + 1))
        terminal_prices = stock_tree[self.n_steps, :self.n_steps + 1].reshape(-1, 1)
        terminal_payoffs = self.payoff.eval(terminal_prices)
        option_tree[self.n_steps, :self.n_steps + 1] = terminal_payoffs

        self._stopping_boundary = np.full(self.n_steps + 1, np.nan)

        p_up = self.joint_probs[1]
        p_down = self.joint_probs[0]

        for i in range(self.n_steps - 1, -1, -1):
            for j in range(i + 1):
                continuation = self.disc * (
                    p_up * option_tree[i + 1, j + 1] +
                    p_down * option_tree[i + 1, j]
                )

                stock_price = stock_tree[i, j].reshape(1, 1)
                exercise = self.payoff.eval(stock_price)[0]

                option_tree[i, j] = max(exercise, continuation)

                if exercise > continuation and np.isnan(self._stopping_boundary[i]):
                    self._stopping_boundary[i] = stock_tree[i, j]

        return option_tree[0, 0]

    def _price_2d(self):
        """Price 2-asset option using 2D LR tree."""
        # Reuse CRR 2D logic with LR parameters
        stock_lattice = {}
        option_lattice = {}

        stock_lattice[(0, 0, 0)] = self.S0

        for i in range(self.n_steps):
            for j1 in range(i + 1):
                for j2 in range(i + 1):
                    if (i, j1, j2) not in stock_lattice:
                        continue

                    S = stock_lattice[(i, j1, j2)]

                    stock_lattice[(i + 1, j1, j2)] = S * np.array([self.d[0], self.d[1]])
                    stock_lattice[(i + 1, j1, j2 + 1)] = S * np.array([self.d[0], self.u[1]])
                    stock_lattice[(i + 1, j1 + 1, j2)] = S * np.array([self.u[0], self.d[1]])
                    stock_lattice[(i + 1, j1 + 1, j2 + 1)] = S * np.array([self.u[0], self.u[1]])

        i = self.n_steps
        for j1 in range(i + 1):
            for j2 in range(i + 1):
                if (i, j1, j2) in stock_lattice:
                    S = stock_lattice[(i, j1, j2)]
                    option_lattice[(i, j1, j2)] = self.payoff.eval(S.reshape(1, -1))[0]

        p_dd, p_du, p_ud, p_uu = self.joint_probs

        for i in range(self.n_steps - 1, -1, -1):
            for j1 in range(i + 1):
                for j2 in range(i + 1):
                    if (i, j1, j2) not in stock_lattice:
                        continue

                    continuation = self.disc * (
                        p_dd * option_lattice.get((i + 1, j1, j2), 0) +
                        p_du * option_lattice.get((i + 1, j1, j2 + 1), 0) +
                        p_ud * option_lattice.get((i + 1, j1 + 1, j2), 0) +
                        p_uu * option_lattice.get((i + 1, j1 + 1, j2 + 1), 0)
                    )

                    S = stock_lattice[(i, j1, j2)]
                    exercise = self.payoff.eval(S.reshape(1, -1))[0]

                    option_lattice[(i, j1, j2)] = max(exercise, continuation)

        return option_lattice[(0, 0, 0)]

    def _price_nd(self):
        """Price d-asset option using general d-dimensional tree."""
        d = self.dim

        stock_lattice = {}
        option_lattice = {}

        stock_lattice[(0,) + (0,) * d] = self.S0.copy()

        for i in range(self.n_steps):
            for state_key in list(stock_lattice.keys()):
                if state_key[0] != i:
                    continue

                S_current = stock_lattice[state_key]
                jump_counts = np.array(state_key[1:])

                for move_idx, move in enumerate(self.moves):
                    S_next = S_current * np.array([self.u[k] if move[k] == 1 else self.d[k]
                                                    for k in range(d)])
                    next_jump_counts = jump_counts + np.array(move)

                    next_key = (i + 1,) + tuple(next_jump_counts)
                    stock_lattice[next_key] = S_next

        i = self.n_steps
        for state_key in stock_lattice.keys():
            if state_key[0] == i:
                S = stock_lattice[state_key]
                option_lattice[state_key] = self.payoff.eval(S.reshape(1, -1))[0]

        for i in range(self.n_steps - 1, -1, -1):
            for state_key in list(stock_lattice.keys()):
                if state_key[0] != i:
                    continue

                S_current = stock_lattice[state_key]
                jump_counts = np.array(state_key[1:])

                continuation = 0.0
                for move_idx, move in enumerate(self.moves):
                    next_jump_counts = jump_counts + np.array(move)
                    next_key = (i + 1,) + tuple(next_jump_counts)
                    continuation += self.joint_probs[move_idx] * option_lattice.get(next_key, 0)

                continuation *= self.disc

                exercise = self.payoff.eval(S_current.reshape(1, -1))[0]

                option_lattice[state_key] = max(exercise, continuation)

        return option_lattice[(0,) + (0,) * d]

    def get_exercise_time(self):
        """
        Compute expected exercise time from the optimal stopping boundary.

        Returns:
            float: Expected exercise time normalized to [0, 1]
        """
        n_sim = 10000
        exercise_times = np.zeros(n_sim)
        d = self.dim

        for path_idx in range(n_sim):
            S = self.S0.copy()

            for i in range(self.n_steps + 1):
                stock_price = S.reshape(1, -1)
                exercise_value = self.payoff.eval(stock_price)[0]

                if exercise_value > 0 and exercise_value > 0.1 * self.payoff.strike:
                    exercise_times[path_idx] = i
                    break

                if i < self.n_steps:
                    move_idx = np.random.choice(len(self.moves), p=self.joint_probs)
                    move = self.moves[move_idx]
                    S = S * np.array([self.u[k] if move[k] == 1 else self.d[k] for k in range(d)])
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
        nb_paths, nb_stocks, nb_dates_plus_1 = stock_paths.shape
        nb_dates = nb_dates_plus_1 - 1
        exercise_dates = np.full(nb_paths, nb_dates, dtype=int)
        payoff_values = np.zeros(nb_paths)

        for path_idx in range(nb_paths):
            for date_idx in range(nb_dates + 1):
                S = stock_paths[path_idx, :, date_idx]
                stock_price = S.reshape(1, -1)
                exercise_value = self.payoff.eval(stock_price)[0]

                if exercise_value > 0:
                    exercise_dates[path_idx] = date_idx
                    payoff_values[path_idx] = exercise_value
                    break
            else:
                stock_price = stock_paths[path_idx, :, -1].reshape(1, -1)
                payoff_values[path_idx] = self.payoff.eval(stock_price)[0]

        discount_factors = np.exp(-self.r * self.T * exercise_dates / nb_dates)
        price = np.mean(payoff_values * discount_factors)

        return exercise_dates, payoff_values, price
