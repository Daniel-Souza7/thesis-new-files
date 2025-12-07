"""
Cox-Ross-Rubinstein (CRR) Binomial Tree for American option pricing.

Implements the classic binomial tree method from:
Cox, J. C., Ross, S. A., & Rubinstein, M. (1979).
"Option pricing: A simplified approach." Journal of Financial Economics.

Now supports multi-asset options (d=1,2,3) with correlation.
"""

import numpy as np
import math
import time
from optimal_stopping.run import configs
import itertools


class CRRTree:
    """
    Cox-Ross-Rubinstein binomial tree for American option pricing.

    Supports single-asset (d=1) and multi-asset (d=2,3) options with correlation.
    Uses a recombining lattice with symmetric up/down moves.
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
        self.d = model.nb_stocks  # Number of assets

        # Validate compatibility
        if payoff.is_path_dependent:
            raise ValueError(
                "CRR tree does not support path-dependent payoffs. "
                "Use standard payoffs only (e.g., BasketCall, MaxCall, Put)."
            )

        # Warn about high dimensionality
        if self.d > 3:
            import warnings
            warnings.warn(
                f"CRR tree with {self.d} assets will have ~{(self.n_steps+1)**self.d:,} nodes. "
                f"This may be very slow or run out of memory. Consider using Monte Carlo methods (RLSM, RFQI).",
                UserWarning
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
        self.S0 = np.array(model.spot).flatten()  # Initial prices (d-dimensional)
        self.r = model.rate  # Risk-free rate
        self.T = model.maturity
        self.sigma = np.array(model.volatility).flatten()  # Volatilities (d-dimensional)

        # Get correlation matrix
        if hasattr(model, 'correlation_matrix'):
            self.corr_matrix = model.correlation_matrix
            print(f"DEBUG CRR __init__: Found correlation_matrix =\n{self.corr_matrix}")
        else:
            # Default: independent assets
            self.corr_matrix = np.eye(self.d)
            print(f"DEBUG CRR __init__: No correlation_matrix found, using identity for {self.d} assets")

        # Tree parameters
        self.dt = self.T / self.n_steps
        self.disc = math.exp(-self.r * self.dt)  # Discount factor

        # Per-asset CRR parameters: u_i = exp(σ_i√Δt), d_i = 1/u_i
        self.u = np.exp(self.sigma * math.sqrt(self.dt))
        self.d = 1.0 / self.u

        # Compute joint probabilities for multi-asset tree
        self._compute_probabilities()

        # Storage for optimal stopping boundary
        self._stopping_boundary = None

    def _compute_probabilities(self):
        """
        Compute joint probabilities for multi-asset tree.

        For d assets, we have 2^d possible moves at each node.
        Probabilities must match:
        1. Risk-neutral drift for each asset
        2. Correlation structure between assets
        """
        d = len(self.S0)

        if d == 1:
            # Single asset: standard CRR probability
            p_up = (math.exp(self.r * self.dt) - self.d[0]) / (self.u[0] - self.d[0])
            self.joint_probs = np.array([1 - p_up, p_up])  # [down, up]
            self.moves = [(0,), (1,)]  # 0=down, 1=up

        elif d == 2:
            # Two assets: 4 moves (dd, du, ud, uu)
            # Use correlation to split probabilities
            rho = self.corr_matrix[0, 1]
            print(f"DEBUG CRR _compute_probabilities: d=2, rho={rho}")

            # Base probabilities for each asset (marginal)
            p1 = (math.exp(self.r * self.dt) - self.d[0]) / (self.u[0] - self.d[0])
            p2 = (math.exp(self.r * self.dt) - self.d[1]) / (self.u[1] - self.d[1])
            print(f"DEBUG CRR: Marginal probs p1={p1:.6f}, p2={p2:.6f}")

            # Joint probabilities (Boyle's formula for correlated binomial)
            # For ρ correlation, adjust joint probabilities
            p_uu = p1 * p2 + rho * math.sqrt(p1 * (1-p1) * p2 * (1-p2))
            p_ud = p1 * (1-p2) - rho * math.sqrt(p1 * (1-p1) * p2 * (1-p2))
            p_du = (1-p1) * p2 - rho * math.sqrt(p1 * (1-p1) * p2 * (1-p2))
            p_dd = (1-p1) * (1-p2) + rho * math.sqrt(p1 * (1-p1) * p2 * (1-p2))
            print(f"DEBUG CRR: Joint probs (before clip) p_uu={p_uu:.6f}, p_ud={p_ud:.6f}, p_du={p_du:.6f}, p_dd={p_dd:.6f}")

            # Ensure probabilities are valid
            self.joint_probs = np.array([p_dd, p_du, p_ud, p_uu])
            self.joint_probs = np.clip(self.joint_probs, 0, 1)
            self.joint_probs /= self.joint_probs.sum()  # Normalize
            print(f"DEBUG CRR: Joint probs (after normalize) {self.joint_probs}")

            self.moves = [(0, 0), (0, 1), (1, 0), (1, 1)]  # (asset1, asset2)

        else:
            # General d assets: 2^d moves
            # Approximate: use independent probabilities (ignoring correlation for d>2)
            # TODO: Could use copula or more sophisticated methods for d>2 with correlation
            import warnings
            if not np.allclose(self.corr_matrix, np.eye(d)):
                warnings.warn(
                    f"CRR tree with {d} assets currently ignores correlation. "
                    f"Only marginal probabilities are matched.",
                    UserWarning
                )

            # Compute marginal probabilities
            p_marginal = [(math.exp(self.r * self.dt) - self.d[i]) / (self.u[i] - self.d[i])
                          for i in range(d)]

            # All 2^d combinations
            self.moves = list(itertools.product([0, 1], repeat=d))
            self.joint_probs = np.array([
                np.prod([p_marginal[i] if move[i] == 1 else 1 - p_marginal[i]
                         for i in range(d)])
                for move in self.moves
            ])

    def price(self, train_eval_split=2):
        """
        Compute option price using CRR binomial tree.

        Args:
            train_eval_split: Ignored (trees don't use train/eval split)

        Returns:
            tuple: (price, computation_time)
        """
        t_start = time.time()

        d = len(self.S0)
        print(f"DEBUG CRR price(): d={d}, corr_matrix=\n{self.corr_matrix}")

        if d == 1:
            price = self._price_1d()
        elif d == 2:
            price = self._price_2d()
        else:
            price = self._price_nd()

        computation_time = time.time() - t_start
        print(f"DEBUG CRR price(): Computed price={price:.6f}")
        return price, computation_time

    def _price_1d(self):
        """Price single-asset option using standard CRR tree."""
        # Build stock price tree
        stock_tree = np.zeros((self.n_steps + 1, self.n_steps + 1))

        # Fill the tree: S[i,j] = S0 * u^j * d^(i-j)
        for i in range(self.n_steps + 1):
            for j in range(i + 1):
                stock_tree[i, j] = self.S0[0] * (self.u[0] ** j) * (self.d[0] ** (i - j))

        # Initialize option value tree
        option_tree = np.zeros((self.n_steps + 1, self.n_steps + 1))

        # Compute payoffs at maturity
        terminal_prices = stock_tree[self.n_steps, :self.n_steps + 1].reshape(-1, 1)
        terminal_payoffs = self.payoff.eval(terminal_prices)
        option_tree[self.n_steps, :self.n_steps + 1] = terminal_payoffs

        # Storage for stopping boundary
        self._stopping_boundary = np.full(self.n_steps + 1, np.nan)

        # Backward induction
        p_up = self.joint_probs[1]
        p_down = self.joint_probs[0]

        for i in range(self.n_steps - 1, -1, -1):
            for j in range(i + 1):
                # Continuation value
                continuation = self.disc * (
                    p_up * option_tree[i + 1, j + 1] +
                    p_down * option_tree[i + 1, j]
                )

                # Immediate exercise value
                stock_price = stock_tree[i, j].reshape(1, 1)
                exercise = self.payoff.eval(stock_price)[0]

                # American option: max of exercise and continuation
                option_tree[i, j] = max(exercise, continuation)

                # Track stopping boundary
                if exercise > continuation and np.isnan(self._stopping_boundary[i]):
                    self._stopping_boundary[i] = stock_tree[i, j]

        return option_tree[0, 0]

    def _price_2d(self):
        """Price 2-asset option using 2D CRR tree."""
        # Build 2D stock price lattice using dictionary
        # Key: (time_step, num_ups_asset1, num_ups_asset2)
        stock_lattice = {}
        option_lattice = {}

        # Initialize at t=0
        stock_lattice[(0, 0, 0)] = self.S0

        # Forward pass: build stock prices
        for i in range(self.n_steps):
            for j1 in range(i + 1):
                for j2 in range(i + 1):
                    if (i, j1, j2) not in stock_lattice:
                        continue

                    S = stock_lattice[(i, j1, j2)]

                    # Four possible moves: (dd, du, ud, uu)
                    stock_lattice[(i + 1, j1, j2)] = S * np.array([self.d[0], self.d[1]])  # dd
                    stock_lattice[(i + 1, j1, j2 + 1)] = S * np.array([self.d[0], self.u[1]])  # du
                    stock_lattice[(i + 1, j1 + 1, j2)] = S * np.array([self.u[0], self.d[1]])  # ud
                    stock_lattice[(i + 1, j1 + 1, j2 + 1)] = S * np.array([self.u[0], self.u[1]])  # uu

        # Compute payoffs at maturity
        i = self.n_steps
        for j1 in range(i + 1):
            for j2 in range(i + 1):
                if (i, j1, j2) in stock_lattice:
                    S = stock_lattice[(i, j1, j2)]
                    option_lattice[(i, j1, j2)] = self.payoff.eval(S.reshape(1, -1))[0]

        # Backward induction
        p_dd, p_du, p_ud, p_uu = self.joint_probs

        for i in range(self.n_steps - 1, -1, -1):
            for j1 in range(i + 1):
                for j2 in range(i + 1):
                    if (i, j1, j2) not in stock_lattice:
                        continue

                    # Continuation value
                    continuation = self.disc * (
                        p_dd * option_lattice.get((i + 1, j1, j2), 0) +
                        p_du * option_lattice.get((i + 1, j1, j2 + 1), 0) +
                        p_ud * option_lattice.get((i + 1, j1 + 1, j2), 0) +
                        p_uu * option_lattice.get((i + 1, j1 + 1, j2 + 1), 0)
                    )

                    # Immediate exercise value
                    S = stock_lattice[(i, j1, j2)]
                    exercise = self.payoff.eval(S.reshape(1, -1))[0]

                    # American option
                    option_lattice[(i, j1, j2)] = max(exercise, continuation)

        return option_lattice[(0, 0, 0)]

    def _price_nd(self):
        """Price d-asset option using general d-dimensional tree."""
        d = len(self.S0)

        # Use dictionary for sparse storage
        stock_lattice = {}
        option_lattice = {}

        # Initialize at t=0
        stock_lattice[(0,) + (0,) * d] = self.S0.copy()

        # Forward pass: build stock prices
        for i in range(self.n_steps):
            # Iterate over all possible states at time i
            for state_key in list(stock_lattice.keys()):
                if state_key[0] != i:
                    continue

                S_current = stock_lattice[state_key]
                jump_counts = np.array(state_key[1:])  # Current number of ups for each asset

                # Generate all 2^d possible next states
                for move_idx, move in enumerate(self.moves):
                    # Apply move: 0=down, 1=up
                    S_next = S_current * np.array([self.u[k] if move[k] == 1 else self.d[k]
                                                    for k in range(d)])
                    next_jump_counts = jump_counts + np.array(move)

                    next_key = (i + 1,) + tuple(next_jump_counts)
                    stock_lattice[next_key] = S_next

        # Compute payoffs at maturity
        i = self.n_steps
        for state_key in stock_lattice.keys():
            if state_key[0] == i:
                S = stock_lattice[state_key]
                option_lattice[state_key] = self.payoff.eval(S.reshape(1, -1))[0]

        # Backward induction
        for i in range(self.n_steps - 1, -1, -1):
            for state_key in list(stock_lattice.keys()):
                if state_key[0] != i:
                    continue

                S_current = stock_lattice[state_key]
                jump_counts = np.array(state_key[1:])

                # Continuation value
                continuation = 0.0
                for move_idx, move in enumerate(self.moves):
                    next_jump_counts = jump_counts + np.array(move)
                    next_key = (i + 1,) + tuple(next_jump_counts)
                    continuation += self.joint_probs[move_idx] * option_lattice.get(next_key, 0)

                continuation *= self.disc

                # Exercise value
                exercise = self.payoff.eval(S_current.reshape(1, -1))[0]

                # American option
                option_lattice[state_key] = max(exercise, continuation)

        return option_lattice[(0,) + (0,) * d]

    def get_exercise_time(self):
        """
        Compute expected exercise time from the optimal stopping boundary.

        Returns:
            float: Expected exercise time normalized to [0, 1]
        """
        # For multi-asset, this is approximate
        # Simulate paths on the tree and track exercise times
        n_sim = 10000
        exercise_times = np.zeros(n_sim)
        d = len(self.S0)

        for path_idx in range(n_sim):
            S = self.S0.copy()

            for i in range(self.n_steps + 1):
                # Check if should exercise
                stock_price = S.reshape(1, -1)
                exercise_value = self.payoff.eval(stock_price)[0]

                if exercise_value > 0:
                    # Simple heuristic: exercise if deeply ITM
                    if exercise_value > 0.1 * self.payoff.strike:
                        exercise_times[path_idx] = i
                        break

                # Move to next step
                if i < self.n_steps:
                    # Random move based on probabilities
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

        Note: For trees, this is an approximation since the policy is defined
        on a discrete lattice, not continuous paths.

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

        # Simple heuristic: exercise when ITM
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

        # Compute price with discounting
        discount_factors = np.exp(-self.r * self.T * exercise_dates / nb_dates)
        price = np.mean(payoff_values * discount_factors)

        return exercise_dates, payoff_values, price
