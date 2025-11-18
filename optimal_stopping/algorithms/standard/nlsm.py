"""
Neural Least Squares Monte Carlo (NLSM) for American option pricing.

Simple benchmark implementation of the NLSM algorithm from:
"Neural network regression for Bermudan option pricing"
(Lapeyre and Lelong, 2019)
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


class NeuralNetworkPricer:
    """
    Computes American option price by Neural Least Squares Monte Carlo (NLSM).

    Uses a neural network to approximate the continuation value function.
    """

    def __init__(self, model, payoff, nb_epochs=20, hidden_size=10,
                 factors=None, train_ITM_only=True, use_payoff_as_input=False):
        """
        Initialize NLSM pricer.

        Args:
            model: Stock model
            payoff: Payoff function
            nb_epochs: Number of training epochs per time step
            hidden_size: Number of neurons in hidden layer
            factors: Ignored (for API compatibility)
            train_ITM_only: If True, only train on in-the-money paths
            use_payoff_as_input: If True, include payoff as feature
        """
        self.model = model
        self.payoff = payoff
        self.nb_epochs = nb_epochs
        self.hidden_size = hidden_size
        self.train_ITM_only = train_ITM_only
        self.use_payoff_as_input = use_payoff_as_input

        # Check for variance paths
        self.use_var = getattr(model, 'return_var', False)

        # Neural network will be created during pricing
        state_size = model.nb_stocks * (1 + self.use_var) + self.use_payoff_as_input * 1
        self.state_size = state_size
        self.neural_network = None

    def price(self, train_eval_split=2):
        """
        Compute option price using NLSM backward induction.

        Args:
            train_eval_split: Ratio for splitting paths into training/evaluation

        Returns:
            tuple: (price, time_for_path_generation)
        """
        t_start = time.time()

        # Generate paths
        if configs.path_gen_seed.get_seed() is not None:
            np.random.seed(configs.path_gen_seed.get_seed())

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

        # Track exercise dates (initialize to maturity)
        self._exercise_dates = np.full(nb_paths, self.model.nb_dates - 1, dtype=int)

        # Backward induction from T-1 to 1
        for date in range(self.model.nb_dates - 1, 0, -1):
            # Current immediate exercise value
            immediate_exercise = payoffs[:, date]

            # Prepare state for regression
            current_state = stock_paths[:, :, date]

            if self.use_payoff_as_input:
                current_state = np.concatenate([current_state, payoffs[:, date:date+1]], axis=1)

            if self.use_var and var_paths is not None:
                current_state = np.concatenate([current_state, var_paths[:, :, date]], axis=1)

            # Discount future values
            discounted_values = values * disc_factor

            # Learn continuation value using neural network
            continuation_values = self._learn_continuation(
                current_state, discounted_values, immediate_exercise
            )

            # Update values: max(exercise now, continue)
            exercise_now = immediate_exercise > continuation_values
            values = np.where(exercise_now, immediate_exercise, discounted_values)

            # Track exercise dates - only update if exercising earlier
            self._exercise_dates[exercise_now] = date

        # Final price: average over evaluation paths
        price = np.mean(values[self.split:])

        return price, time_path_gen

    def _learn_continuation(self, current_state, future_values, immediate_exercise):
        """
        Learn continuation value using neural network regression.

        Args:
            current_state: (nb_paths, state_size) - Current stock prices (+ payoff if used)
            future_values: (nb_paths,) - Discounted future values
            immediate_exercise: (nb_paths,) - Immediate payoff if exercised

        Returns:
            continuation_values: (nb_paths,) - Estimated continuation values
        """
        # Determine which paths to use for training
        if self.train_ITM_only:
            # Only train on in-the-money paths in training set
            train_mask = (immediate_exercise[:self.split] > 0)
        else:
            # Train on all paths in training set
            train_mask = np.ones(self.split, dtype=bool)

        # Create new neural network for this time step
        self.neural_network = neural_networks.NetworkNLSM(
            self.state_size, hidden_size=self.hidden_size
        ).double()
        self.neural_network.apply(init_weights)

        # Train the network
        if train_mask.sum() > 0:
            self._train_network(
                current_state[:self.split][train_mask],
                future_values[:self.split][train_mask]
            )

        # Predict on all paths
        return self._evaluate_network(current_state)

    def _train_network(self, X_inputs, Y_labels, batch_size=2000):
        """
        Train neural network using gradient descent.

        Args:
            X_inputs: Training inputs (stock prices)
            Y_labels: Training targets (continuation values)
            batch_size: Batch size for training
        """
        optimizer = optim.Adam(self.neural_network.parameters())
        X_inputs = torch.from_numpy(X_inputs).double()
        Y_labels = torch.from_numpy(Y_labels).double().view(len(Y_labels), 1)

        self.neural_network.train(True)

        for epoch in range(self.nb_epochs):
            for batch in tdata.BatchSampler(
                tdata.RandomSampler(range(len(X_inputs)), replacement=False),
                batch_size=batch_size,
                drop_last=False
            ):
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = self.neural_network(X_inputs[batch])
                    loss = nn.MSELoss(reduction="mean")(outputs, Y_labels[batch])
                    loss.backward()
                    optimizer.step()

    def _evaluate_network(self, X_inputs):
        """
        Evaluate neural network on inputs.

        Args:
            X_inputs: Input states

        Returns:
            Predicted continuation values
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
