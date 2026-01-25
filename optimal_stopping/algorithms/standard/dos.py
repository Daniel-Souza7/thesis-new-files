"""
Deep Optimal Stopping (DOS) for American option pricing.

Simple benchmark implementation of the DOS algorithm from:
"Deep optimal stopping" (Becker, Cheridito and Jentzen, 2020)
"""

import numpy as np
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata
from optimal_stopping.run import configs
from optimal_stopping.algorithms.utils import neural_networks


def init_weights(m):
    """Initialize neural network weights."""
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class DeepOptimalStopping:
    """
    Computes American option price using Deep Optimal Stopping (DOS).

    Uses a neural network to learn the optimal stopping rule directly.
    """

    def __init__(self, model, payoff, nb_epochs=20, hidden_size=10,
                 factors=None, use_path=False, train_ITM_only=False,
                 use_payoff_as_input=False):
        """
        Initialize DOS pricer.

        Args:
            model: Stock model
            payoff: Payoff function
            nb_epochs: Number of training epochs
            hidden_size: Number of neurons in hidden layer
            factors: Ignored (for API compatibility)
            use_path: If True, use full path history as input
            train_ITM_only: Ignored for DOS
            use_payoff_as_input: If True, include payoff as feature
        """
        self.model = model
        self.payoff = payoff
        self.nb_epochs = nb_epochs
        # DOS uses fixed hidden_size=10 as default
        if hidden_size is None:
            hidden_size = 10
        self.hidden_size = hidden_size
        self.use_path = use_path
        self.use_payoff_as_input = use_payoff_as_input

        # Check for variance paths
        self.use_var = getattr(model, 'return_var', False)

        # Determine state size
        if self.use_path:
            # Full path history: (nb_stocks * (nb_dates+1))
            state_size = (model.nb_stocks * (1 + self.use_var) + self.use_payoff_as_input * 1) * (model.nb_dates + 1)
        else:
            # Only current state
            state_size = model.nb_stocks * (1 + self.use_var) + self.use_payoff_as_input * 1

        self.state_size = state_size

        # Create neural network once for stopping decision (reference methodology for speed)
        self.neural_network = neural_networks.NetworkDOS(
            self.state_size, hidden_size=self.hidden_size
        ).double()

        # Note: No network storage for performance (would need 2-5x more time)
        # backward_induction_on_paths() not supported with this approach
        self._learned_networks = {}

    def price(self, train_eval_split=2):
        """
        Compute option price using DOS.

        Args:
            train_eval_split: Ratio for splitting paths into training/evaluation

        Returns:
            tuple: (price, time_for_path_generation)
        """
        t_start = time.time()

        # Generate paths
        if configs.path_gen_seed.get_seed() is not None:
            np.random.seed(configs.path_gen_seed.get_seed())
            torch.manual_seed(configs.path_gen_seed.get_seed())

        path_result = self.model.generate_paths()
        if isinstance(path_result, tuple):
            stock_paths, var_paths = path_result
        else:
            stock_paths = path_result
            var_paths = None

        time_path_gen = time.time() - t_start
        print(f"time path gen: {time_path_gen:.4f} ", end="")

        # Compute payoffs for all paths
        payoffs = self.payoff(stock_paths)

        # Discount all payoffs upfront (reference methodology)
        power = np.arange(0, self.model.nb_dates + 1)
        disc_factors = np.exp(
            (-self.model.rate) * self.model.maturity / self.model.nb_dates * power
        )
        disc_factors = np.repeat(
            np.expand_dims(disc_factors, axis=0), repeats=payoffs.shape[0], axis=0
        )
        payoffs = payoffs * disc_factors

        # Split into training and evaluation sets
        self.split = len(stock_paths) // train_eval_split

        nb_paths, nb_stocks, nb_dates_from_shape = stock_paths.shape

        # Initialize with terminal payoff (already discounted)
        values = payoffs[:, -1].copy()

        # Track exercise dates (initialize to maturity = nb_dates, not nb_dates-1)
        self._exercise_dates = np.full(nb_paths, self.model.nb_dates, dtype=int)

        # Backward induction from T-1 to 1
        for date in range(self.model.nb_dates - 1, 0, -1):
            # Current immediate exercise value (already discounted)
            immediate_exercise = payoffs[:, date]

            # Prepare state
            if self.use_path:
                # Use full path history up to current time
                current_state = stock_paths[:, :, :date+1]

                if self.use_var and var_paths is not None:
                    current_state = np.concatenate([current_state, var_paths[:, :, :date+1]], axis=1)

                # Add zeros to get fixed shape (nb_stocks, nb_dates+1)
                padding_shape = (current_state.shape[0], current_state.shape[1], nb_dates_from_shape - date)
                padding = np.zeros(padding_shape)
                current_state = np.concatenate([current_state, padding], axis=-1)

                # Flatten for network input
                current_state = current_state.reshape((current_state.shape[0], -1))
            else:
                # Use only current state
                current_state = stock_paths[:, :, date]

                if self.use_payoff_as_input:
                    current_state = np.concatenate([current_state, payoffs[:, date:date+1]], axis=1)

                if self.use_var and var_paths is not None:
                    current_state = np.concatenate([current_state, var_paths[:, :, date]], axis=1)

            # Clip to non-negative (American option value can't be negative)
            values_clipped = np.maximum(0, values)

            # Filter to ITM paths if train_ITM_only is enabled
            if self.train_ITM_only:
                itm_mask = immediate_exercise[:self.split] > 0
                train_state = current_state[:self.split][itm_mask]
                train_payoff = immediate_exercise[:self.split][itm_mask]
                train_values = values_clipped[:self.split][itm_mask]
            else:
                train_state = current_state[:self.split]
                train_payoff = immediate_exercise[:self.split]
                train_values = values_clipped[:self.split]

            # Reinitialize network weights for this time step (reference methodology)
            self.neural_network.apply(init_weights)

            # Train network to maximize expected value (only on ITM if enabled)
            # Note: values already contains discounted future payoffs
            if len(train_state) > 0:
                self._train_network(train_state, train_payoff, train_values)

            # Note: Don't store networks for performance (2-5x speedup)

            # Predict stopping decision
            stopping_rule = self._evaluate_network(current_state)

            # Update values: exercise if stopping_rule > 0.5, else continue
            exercise_now = stopping_rule > 0.5
            values[exercise_now] = immediate_exercise[exercise_now]
            # Don't discount values[~exercise_now] - already discounted

            # Track exercise dates - only update if exercising earlier
            self._exercise_dates[exercise_now] = date

        # Final price: average over evaluation paths
        payoff_0 = payoffs[0, 0]
        price = max(payoff_0, np.mean(values[self.split:]))

        return price, time_path_gen

    def price_upper_lower_bound(self, train_eval_split=2):
        """
        Compute both lower and upper bounds using DOS.

        Lower bound: Regular DOS pricing
        Upper bound: Dual formulation (Haugh-Kogan method)

        Args:
            train_eval_split: Ratio for splitting paths

        Returns:
            tuple: (lower_bound, upper_bound, time_for_path_generation)
        """
        t_start = time.time()

        # Generate paths
        if configs.path_gen_seed.get_seed() is not None:
            np.random.seed(configs.path_gen_seed.get_seed())
            torch.manual_seed(configs.path_gen_seed.get_seed())

        path_result = self.model.generate_paths()
        if isinstance(path_result, tuple):
            stock_paths, var_paths = path_result
        else:
            stock_paths = path_result
            var_paths = None

        time_path_gen = time.time() - t_start
        print(f"time path gen: {time_path_gen:.4f} ", end="")

        # Compute payoffs
        payoffs = self.payoff(stock_paths)

        # Discount all payoffs upfront (reference methodology)
        power = np.arange(0, self.model.nb_dates + 1)
        disc_factors = np.exp(
            (-self.model.rate) * self.model.maturity / self.model.nb_dates * power
        )
        disc_factors = np.repeat(
            np.expand_dims(disc_factors, axis=0), repeats=payoffs.shape[0], axis=0
        )
        payoffs = payoffs * disc_factors

        # Split
        self.split = len(stock_paths) // train_eval_split

        nb_paths, nb_stocks, nb_dates_from_shape = stock_paths.shape

        # Initialize M_diff for upper bound tracking
        M_diff = np.zeros((nb_paths, self.model.nb_dates + 1))

        # Clear previous learned networks
        self._learned_networks = {}

        # Initialize exercise dates tracking (for get_exercise_time)
        self._exercise_dates = np.full(nb_paths, self.model.nb_dates, dtype=int)

        # Initialize with terminal payoff (already discounted)
        values = payoffs[:, -1].copy()
        prev_cont_val = np.zeros_like(values)

        # Backward induction from T-1 down to 0
        for date in range(self.model.nb_dates - 1, -1, -1):
            immediate_exercise = payoffs[:, date]

            # Prepare state
            if self.use_path:
                current_state = stock_paths[:, :, :date+1]
                if self.use_var and var_paths is not None:
                    current_state = np.concatenate([current_state, var_paths[:, :, :date+1]], axis=1)
                padding_shape = (current_state.shape[0], current_state.shape[1], nb_dates_from_shape - date)
                padding = np.zeros(padding_shape)
                current_state = np.concatenate([current_state, padding], axis=-1)
                current_state = current_state.reshape((current_state.shape[0], -1))
            else:
                current_state = stock_paths[:, :, date]
                if self.use_payoff_as_input:
                    current_state = np.concatenate([current_state, payoffs[:, date:date+1]], axis=1)
                if self.use_var and var_paths is not None:
                    current_state = np.concatenate([current_state, var_paths[:, :, date]], axis=1)

            values_clipped = np.maximum(0, values)

            # Filter to ITM paths if train_ITM_only is enabled
            if self.train_ITM_only:
                itm_mask = immediate_exercise[:self.split] > 0
                train_state = current_state[:self.split][itm_mask]
                train_payoff = immediate_exercise[:self.split][itm_mask]
                train_values = values_clipped[:self.split][itm_mask]
            else:
                train_state = current_state[:self.split]
                train_payoff = immediate_exercise[:self.split]
                train_values = values_clipped[:self.split]

            # Reinitialize network weights for this time step (reference methodology)
            self.neural_network.apply(init_weights)

            # Train network
            if len(train_state) > 0:
                self._train_network(train_state, train_payoff, train_values)

            # Note: Don't store networks for performance (2-5x speedup)

            # Get stopping probabilities
            stopping_rule = self._evaluate_network(current_state)

            # Compute continuation value estimate for upper bound
            continuation_estimate = stopping_rule * immediate_exercise + (1 - stopping_rule) * values_clipped

            # Update M_diff for upper bound (reference methodology)
            if date < self.model.nb_dates - 1:
                M_diff[:, date + 1] = np.maximum(
                    payoffs[:, date + 1], prev_cont_val
                ) - continuation_estimate

            # Lower bound: update values
            if date > 0:
                exercise_now = stopping_rule > 0.5
                values[exercise_now] = immediate_exercise[exercise_now]
                # Track exercise dates
                self._exercise_dates[exercise_now] = date

            prev_cont_val = continuation_estimate.copy()

        # Compute lower bound
        payoff_0 = payoffs[0, 0]
        lower_bound = max(payoff_0, np.mean(values[self.split:]))

        # Compute upper bound using dual formulation (reference methodology)
        M = np.cumsum(M_diff, axis=1)
        upper_bound = np.mean(np.max(payoffs[self.split:] - M[self.split:], axis=1))

        return lower_bound, upper_bound, time_path_gen

    def _train_network(self, stock_values, immediate_exercise, discounted_next_values, batch_size=2000):
        """
        Train neural network to learn optimal stopping rule.

        The loss is -E[value], where value = stopping_prob * immediate_payoff + (1-stopping_prob) * continuation

        Args:
            stock_values: Current state
            immediate_exercise: Payoff if exercised now
            discounted_next_values: Continuation value
            batch_size: Batch size for training
        """
        optimizer = optim.Adam(self.neural_network.parameters())

        immediate_exercise = torch.from_numpy(immediate_exercise).double().reshape(-1, 1)
        discounted_next_values = torch.from_numpy(discounted_next_values).double()
        X_inputs = torch.from_numpy(stock_values).double()

        self.neural_network.train(True)

        for epoch in range(self.nb_epochs):
            for batch in tdata.BatchSampler(
                tdata.RandomSampler(range(len(X_inputs)), replacement=False),
                batch_size=batch_size,
                drop_last=False
            ):
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    # Network outputs stopping probability
                    stopping_prob = self.neural_network(X_inputs[batch]).reshape(-1)

                    # Expected value: stop * immediate + continue * discounted_next
                    values = (
                        immediate_exercise[batch].reshape(-1) * stopping_prob +
                        discounted_next_values[batch] * (1 - stopping_prob)
                    )

                    # Maximize expected value = minimize negative expected value
                    loss = -torch.mean(values)

                    loss.backward()
                    optimizer.step()

    def _evaluate_network(self, X_inputs):
        """
        Evaluate network to get stopping probabilities.

        Args:
            X_inputs: Input states

        Returns:
            Stopping probabilities (0 to 1)
        """
        self.neural_network.train(False)
        X_inputs = torch.from_numpy(X_inputs).double()
        outputs = self.neural_network(X_inputs)
        return outputs.view(len(X_inputs)).detach().numpy()

    def get_exercise_time(self):
        """Return average exercise time normalized to [0, 1] (evaluation set only)."""
        if not hasattr(self, '_exercise_dates'):
            return None

        nb_dates = self.model.nb_dates
        # Only use evaluation set paths (self.split:), not training paths
        normalized_times = self._exercise_dates[self.split:] / nb_dates
        return float(np.mean(normalized_times))

    def backward_induction_on_paths(self, stock_paths, var_paths=None):
        """
        Apply learned policy using backward induction (same as training).

        NOTE: This method is not supported with the current speed optimization approach.
        Network weights are reinitialized at each timestep and not stored.
        To use this method, networks would need to be stored (2-5x slower).

        Args:
            stock_paths: (nb_paths, nb_stocks, nb_dates+1) - Stock price paths
            var_paths: (nb_paths, nb_stocks, nb_dates+1) - Variance paths (optional)

        Returns:
            exercise_times: (nb_paths,) - Time step when each path is exercised
            payoff_values: (nb_paths,) - Payoff value at exercise for each path
            price: float - Average discounted payoff

        Raises:
            NotImplementedError: Method not supported with current optimization approach
        """
        raise NotImplementedError(
            "backward_induction_on_paths() is not supported with the current speed optimization. "
            "Networks are not stored during training for 2-5x speedup. "
            "To use this method, network storage would need to be re-enabled."
        )

        nb_paths = stock_paths.shape[0]
        nb_dates = self.model.nb_dates
        nb_stocks = stock_paths.shape[1]
        nb_dates_from_shape = stock_paths.shape[2]

        # Compute all payoffs upfront
        payoffs = self.payoff(stock_paths)

        # Initialize with terminal payoff
        values = payoffs[:, -1].copy()

        # Track exercise dates (initialize to maturity = nb_dates)
        exercise_dates = np.full(nb_paths, nb_dates, dtype=int)

        disc_factor = math.exp(-self.model.rate * self.model.maturity / nb_dates)

        # Backward induction from T-1 to 1 (same as in price())
        for date in range(nb_dates - 1, 0, -1):
            # Skip if no network learned for this time step
            if date not in self._learned_networks:
                continue

            # Current immediate exercise value
            immediate_exercise = payoffs[:, date]

            # Prepare state (match DOS's training structure)
            if self.use_path:
                # Use full path history up to current time
                current_state = stock_paths[:, :, :date+1]

                if self.use_var and var_paths is not None:
                    current_state = np.concatenate([current_state, var_paths[:, :, :date+1]], axis=1)

                # Add zeros to get fixed shape (nb_stocks, nb_dates+1)
                padding_shape = (current_state.shape[0], current_state.shape[1], nb_dates_from_shape - date)
                padding = np.zeros(padding_shape)
                current_state = np.concatenate([current_state, padding], axis=-1)

                # Flatten for network input
                current_state = current_state.reshape((current_state.shape[0], -1))
            else:
                # Use only current state
                current_state = stock_paths[:, :, date]

                if self.use_payoff_as_input:
                    current_state = np.concatenate([current_state, payoffs[:, date:date+1]], axis=1)

                if self.use_var and var_paths is not None:
                    current_state = np.concatenate([current_state, var_paths[:, :, date]], axis=1)

            # Get learned network for this time step
            self.neural_network = self._learned_networks[date]

            # Predict stopping decision using learned network
            stopping_rule = self._evaluate_network(current_state)

            # Discount future values
            discounted_values = values * disc_factor

            # Clip to non-negative (American option value can't be negative)
            discounted_values = np.maximum(0, discounted_values)

            # Exercise decision: exercise if stopping_rule > 0.5, else continue
            exercise_now = stopping_rule > 0.5

            # Update values
            values = np.where(exercise_now, immediate_exercise, discounted_values)

            # Track exercise dates - only update if exercising earlier
            exercise_dates[exercise_now] = date

        # Extract payoff values at exercise time
        payoff_values = np.array([payoffs[i, exercise_dates[i]] for i in range(nb_paths)])

        # Compute price (average discounted payoff)
        price = np.mean(values)

        return exercise_dates, payoff_values, price
