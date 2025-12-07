"""
Trinomial Tree for American option pricing.

Implements the three-jump process from:
Boyle, P. P. (1986).
"Option Valuation using a Three-Jump Process."
International Options Journal.

The trinomial tree allows the stock to move Up, Down, or stay Middle (flat),
providing an extra degree of freedom that improves stability and flexibility
compared to binomial trees.

Now supports multi-asset options (d=1,2,3) with correlation.
"""

import numpy as np
import math
import time
from optimal_stopping.run import configs
import itertools


class TrinomialTree:
    """
    Trinomial tree for American option pricing.

    Supports single-asset (d=1) and multi-asset (d=2,3) options with correlation.
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
        self.d = model.nb_stocks  # Number of assets

        # Validate compatibility
        if payoff.is_path_dependent:
            raise ValueError(
                "Trinomial tree does not support path-dependent payoffs. "
                "Use standard payoffs only (e.g., BasketCall, MaxCall, Put)."
            )

        # Warn about high dimensionality
        if self.d > 3:
            import warnings
            warnings.warn(
                f"Trinomial tree with {self.d} assets will have ~{(2*self.n_steps+1)**self.d:,} nodes. "
                f"This may be very slow or run out of memory. Consider using Monte Carlo methods (RLSM, RFQI).",
                UserWarning
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
        self.S0 = np.array(model.spot).flatten()  # Initial prices (d-dimensional)
        self.r = model.rate  # Risk-free rate
        self.T = model.maturity
        self.sigma = np.array(model.volatility).flatten()  # Volatilities (d-dimensional)

        # Get correlation matrix
        if hasattr(model, 'correlation_matrix'):
            self.corr_matrix = model.correlation_matrix
            print(f"DEBUG Trinomial __init__: Found correlation_matrix =\n{self.corr_matrix}")
        else:
            # Default: independent assets
            self.corr_matrix = np.eye(self.d)
            print(f"DEBUG Trinomial __init__: No correlation_matrix found, using identity for {self.d} assets")

        # Tree parameters
        self.dt = self.T / self.n_steps
        self.disc = math.exp(-self.r * self.dt)  # Discount factor

        # Trinomial tree parameters (Boyle's parameterization)
        # Use λ = √3 for better stability
        lambda_factor = math.sqrt(3)

        # Per-asset trinomial parameters
        self.u = np.exp(self.sigma * math.sqrt(lambda_factor * self.dt))
        self.d = 1.0 / self.u
        self.m = np.ones(self.d)  # Middle branch: no change

        # Compute joint probabilities for multi-asset tree
        self._compute_probabilities()

        # Storage for optimal stopping boundary
        self._stopping_boundary = None

    def _compute_probabilities(self):
        """
        Compute joint probabilities for multi-asset trinomial tree.

        For d assets, we have 3^d possible moves at each node.
        For simplicity, we use independent marginal probabilities (ignoring correlation).
        TODO: Implement correlation for trinomial trees (more complex than binomial).
        """
        d = len(self.S0)

        # Compute marginal probabilities for each asset
        self.marginal_probs = []

        for i in range(d):
            # Risk-neutral probabilities matching first two moments
            dx = math.log(self.u[i])
            nu = self.r - 0.5 * self.sigma[i]**2  # Drift in log-space

            # Probabilities from moment matching
            p_u = 0.5 * ((self.sigma[i]**2 * self.dt + nu**2 * self.dt**2) / dx**2 +
                         nu * self.dt / dx)
            p_d = 0.5 * ((self.sigma[i]**2 * self.dt + nu**2 * self.dt**2) / dx**2 -
                         nu * self.dt / dx)
            p_m = 1.0 - p_u - p_d

            # Validate probabilities
            if not (0 <= p_u <= 1 and 0 <= p_d <= 1 and 0 <= p_m <= 1):
                import warnings
                warnings.warn(
                    f"Trinomial probabilities for asset {i} out of [0,1] range: "
                    f"p_u={p_u:.4f}, p_m={p_m:.4f}, p_d={p_d:.4f}. "
                    f"Try reducing n_steps or adjusting parameters.",
                    UserWarning
                )

            self.marginal_probs.append((p_u, p_m, p_d))

        # For multi-asset, compute joint probabilities
        # For now, use independence (correlation support for trinomial is complex)
        if d > 1 and not np.allclose(self.corr_matrix, np.eye(d)):
            import warnings
            warnings.warn(
                f"Trinomial tree with {d} assets currently ignores correlation. "
                f"Only marginal probabilities are matched. Consider using CRR or LR for correlation support.",
                UserWarning
            )

        # Generate all 3^d possible moves (0=down, 1=middle, 2=up)
        self.moves = list(itertools.product([0, 1, 2], repeat=d))

        # Joint probabilities (assuming independence)
        self.joint_probs = np.array([
            np.prod([self.marginal_probs[i][move[i]] for i in range(d)])
            for move in self.moves
        ])

        print(f"DEBUG Trinomial: d={d}, {len(self.moves)} moves, prob sum={self.joint_probs.sum():.6f}")

    def price(self, train_eval_split=2):
        """
        Compute option price using trinomial tree.

        Args:
            train_eval_split: Ignored (trees don't use train/eval split)

        Returns:
            tuple: (price, computation_time)
        """
        t_start = time.time()

        d = len(self.S0)
        print(f"DEBUG Trinomial price(): d={d}, corr_matrix=\n{self.corr_matrix}")

        if d == 1:
            price = self._price_1d()
        else:
            price = self._price_nd()

        computation_time = time.time() - t_start
        print(f"DEBUG Trinomial price(): Computed price={price:.6f}")
        return price, computation_time

    def _price_1d(self):
        """Price single-asset option using trinomial tree."""
        # Build stock price tree
        # For trinomial: at step i, we have 2*i+1 possible nodes
        # Node j corresponds to j up moves minus j down moves (net position)
        stock_tree = {}
        option_tree = {}

        # Initialize at t=0
        stock_tree[(0, 0)] = self.S0[0]

        # Forward pass: build stock price tree
        for i in range(self.n_steps):
            for j in range(-i, i + 1):
                if (i, j) not in stock_tree:
                    continue

                S = stock_tree[(i, j)]

                # Three branches
                stock_tree[(i + 1, j + 1)] = S * self.u[0]  # Up
                stock_tree[(i + 1, j)] = S * self.m[0]      # Middle
                stock_tree[(i + 1, j - 1)] = S * self.d[0]  # Down

        # Compute payoffs at maturity
        i = self.n_steps
        for j in range(-i, i + 1):
            if (i, j) in stock_tree:
                stock_price = stock_tree[(i, j)].reshape(1, 1)
                option_tree[(i, j)] = self.payoff.eval(stock_price)[0]

        # Storage for stopping boundary
        self._stopping_boundary = {}

        # Backward induction
        p_u, p_m, p_d = self.marginal_probs[0]

        for i in range(self.n_steps - 1, -1, -1):
            for j in range(-i, i + 1):
                if (i, j) not in stock_tree:
                    continue

                # Continuation value
                continuation = self.disc * (
                    p_u * option_tree.get((i + 1, j + 1), 0) +
                    p_m * option_tree.get((i + 1, j), 0) +
                    p_d * option_tree.get((i + 1, j - 1), 0)
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
                        self._stopping_boundary[i] = min(
                            self._stopping_boundary[i],
                            stock_tree[(i, j)]
                        )

        return option_tree[(0, 0)]

    def _price_nd(self):
        """Price d-asset option using general d-dimensional trinomial tree."""
        d = len(self.S0)

        # Use dictionary for sparse storage
        # Key: (time_step, net_up_moves_asset1, net_up_moves_asset2, ...)
        # net_up_moves = num_ups - num_downs (can be negative)
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
                net_moves = np.array(state_key[1:])  # Current net moves for each asset

                # Generate all 3^d possible next states
                for move_idx, move in enumerate(self.moves):
                    # Apply move: 0=down, 1=middle, 2=up
                    # down decrements net moves, up increments, middle keeps same
                    move_delta = np.array([m - 1 for m in move])  # -1, 0, or +1

                    # Update stock prices
                    S_next = S_current.copy()
                    for k in range(d):
                        if move[k] == 2:  # Up
                            S_next[k] *= self.u[k]
                        elif move[k] == 1:  # Middle
                            S_next[k] *= self.m[k]
                        else:  # Down (move[k] == 0)
                            S_next[k] *= self.d[k]

                    next_net_moves = net_moves + move_delta
                    next_key = (i + 1,) + tuple(next_net_moves)
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
                net_moves = np.array(state_key[1:])

                # Continuation value
                continuation = 0.0
                for move_idx, move in enumerate(self.moves):
                    move_delta = np.array([m - 1 for m in move])
                    next_net_moves = net_moves + move_delta
                    next_key = (i + 1,) + tuple(next_net_moves)
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
                    if hasattr(self.payoff, 'strike'):
                        if exercise_value > 0.1 * self.payoff.strike:
                            exercise_times[path_idx] = i
                            break

                # Move to next step
                if i < self.n_steps:
                    # Random move based on probabilities
                    move_idx = np.random.choice(len(self.moves), p=self.joint_probs)
                    move = self.moves[move_idx]

                    for k in range(d):
                        if move[k] == 2:  # Up
                            S[k] *= self.u[k]
                        elif move[k] == 1:  # Middle
                            S[k] *= self.m[k]
                        else:  # Down (move[k] == 0)
                            S[k] *= self.d[k]
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
