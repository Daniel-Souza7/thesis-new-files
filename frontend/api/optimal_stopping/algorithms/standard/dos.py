"""
Deep Optimal Stopping (DOS) for American option pricing.

Simple benchmark implementation of the DOS algorithm from:
"Deep optimal stopping" (Becker, Cheridito and Jentzen, 2020)
"""

import numpy as np
import math
import time
import copy
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

        # Create neural network for stopping decision
        self.neural_network = neural_networks.NetworkDOS(
            self.state_size, hidden_size=self.hidden_size
        ).double()
        self.neural_network.apply(init_weights)

        # Store network state for each time step (for backward_induction_on_paths)
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

        # Split into training and evaluation sets
        self.split = len(stock_paths) // train_eval_split

        nb_paths, nb_stocks, nb_dates_from_shape = stock_paths.shape
        # Use model's nb_dates (actual time steps) for consistency across all algorithms
        disc_factor = math.exp(-self.model.rate * self.model.maturity / self.model.nb_dates)

        # Initialize with terminal payoff
        values = payoffs[:, -1].copy()

        # Track exercise dates (initialize to maturity = nb_dates, not nb_dates-1)
        self._exercise_dates = np.full(nb_paths, self.model.nb_dates, dtype=int)

        # Backward induction from T-1 to 1
        for date in range(self.model.nb_dates - 1, 0, -1):
            # Current immediate exercise value
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

            # Discount future values
            discounted_values = values * disc_factor

            # Clip to non-negative (American option value can't be negative)
            discounted_values = np.maximum(0, discounted_values)

            # Train network to maximize expected value
            self._train_network(
                current_state[:self.split],
                immediate_exercise[:self.split],
                discounted_values[:self.split]
            )

            # Store network state for this time step (deep copy)
            self._learned_networks[date] = copy.deepcopy(self.neural_network)

            # Predict stopping decision
            stopping_rule = self._evaluate_network(current_state)

            # Update values: exercise if stopping_rule > 0.5, else continue
            exercise_now = stopping_rule > 0.5
            values = np.where(
                exercise_now,
                immediate_exercise,
                discounted_values
            )

            # Track exercise dates - only update if exercising earlier
            self._exercise_dates[exercise_now] = date

        # Final price: average over evaluation paths
        price = np.mean(values[self.split:])

        return price, time_path_gen

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

        This is what should be used for create_video to replicate pricing behavior.
        Uses backward induction (not forward simulation) with learned networks.

        Args:
            stock_paths: (nb_paths, nb_stocks, nb_dates+1) - Stock price paths
            var_paths: (nb_paths, nb_stocks, nb_dates+1) - Variance paths (optional)

        Returns:
            exercise_times: (nb_paths,) - Time step when each path is exercised
            payoff_values: (nb_paths,) - Payoff value at exercise for each path
            price: float - Average discounted payoff
        """
        if not self._learned_networks:
            raise ValueError("No learned policy available. Must call price() first to train.")

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
