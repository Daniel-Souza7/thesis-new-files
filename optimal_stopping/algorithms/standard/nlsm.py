"""
Neural Least Squares Monte Carlo (NLSM) for American option pricing.

Implementation of the Neural Least Squares Monte Carlo introduced in
(Neural network regression for Bermudan option pricing, Lapeyre and Lelong, 2019).

This is a benchmark algorithm using trained neural networks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata
import time
import math
from optimal_stopping.run import configs
from optimal_stopping.algorithms.utils import neural_networks


def init_weights(m):
    """Initialize network weights."""
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class NLSM:
    """
    Neural Least Squares Monte Carlo pricer.

    Uses a trained neural network to approximate continuation values.
    """

    def __init__(self, model, payoff, nb_epochs=20, hidden_size=10,
                 train_ITM_only=True, use_payoff_as_input=False, **kwargs):
        """
        Initialize NLSM pricer.

        Args:
            model: Stock model
            payoff: Payoff function
            nb_epochs: Number of training epochs for neural network
            hidden_size: Number of neurons in hidden layer
            train_ITM_only: If True, only use in-the-money paths for training
            use_payoff_as_input: If True, include payoff in state
        """
        self.model = model
        self.payoff = payoff
        self.nb_epochs = nb_epochs
        self.hidden_size = hidden_size
        self.train_ITM_only = train_ITM_only
        self.use_payoff_as_input = use_payoff_as_input
        self.use_var = getattr(model, 'return_var', False)
        self.batch_size = 2000

        # State size
        self.state_size = model.nb_stocks * (1 + self.use_var) + self.use_payoff_as_input * 1

    def _train_network(self, X_inputs, y_labels):
        """
        Train neural network on given data.

        Args:
            X_inputs: Input features
            y_labels: Target continuation values
        """
        # Create network
        network = neural_networks.NetworkNLSM(self.state_size, hidden_size=self.hidden_size).double()
        network.apply(init_weights)

        # Prepare data
        X_inputs = torch.from_numpy(X_inputs).double()
        y_labels = torch.from_numpy(y_labels).double().view(len(y_labels), 1)

        # Train
        optimizer = optim.Adam(network.parameters())
        network.train(True)

        for _ in range(self.nb_epochs):
            for batch in tdata.BatchSampler(
                    tdata.RandomSampler(range(len(X_inputs)), replacement=False),
                    batch_size=self.batch_size, drop_last=False):
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = network(X_inputs[batch])
                    loss = nn.MSELoss(reduction="mean")(outputs, y_labels[batch])
                    loss.backward()
                    optimizer.step()

        return network

    def _evaluate_network(self, network, X_inputs):
        """
        Evaluate network on given inputs.

        Args:
            network: Trained network
            X_inputs: Input features

        Returns:
            Predicted continuation values
        """
        network.train(False)
        X_inputs = torch.from_numpy(X_inputs).double()
        with torch.no_grad():
            outputs = network(X_inputs)
        return outputs.view(len(X_inputs)).detach().numpy()

    def price(self, train_eval_split=2):
        """
        Compute option price using NLSM.

        Args:
            train_eval_split: Ratio for splitting paths into training/evaluation

        Returns:
            tuple: (price, time_for_path_generation)
        """
        t_start = time.time()

        # Generate paths
        if configs.path_gen_seed.get_seed() is not None:
            np.random.seed(configs.path_gen_seed.get_seed())

        stock_paths, var_paths = self.model.generate_paths()
        time_for_path_gen = time.time() - t_start

        # Calculate payoffs
        payoffs = self.payoff(stock_paths)

        # Split data
        self.split = int(len(stock_paths) / train_eval_split)

        # Discount factor
        nb_dates = self.model.nb_dates
        deltaT = self.model.maturity / nb_dates
        discount_factor = math.exp(-self.model.rate * deltaT)

        # Backward induction
        values = payoffs.copy()

        for t in range(nb_dates - 1, 0, -1):
            # Current stock prices
            stock_at_t = stock_paths[:, :, t]

            # Add variance if needed
            if self.use_var:
                stock_at_t = np.concatenate([stock_at_t, var_paths[:, :, t]], axis=1)

            # Add payoff if needed
            if self.use_payoff_as_input:
                stock_at_t = np.concatenate([stock_at_t, payoffs[:, t:t+1]], axis=1)

            # Discounted future values
            continuation_values = discount_factor * values[:, t+1]

            # Determine in-the-money paths
            immediate_exercise = payoffs[:, t]
            if self.train_ITM_only:
                itm = immediate_exercise[:self.split] > 0
            else:
                itm = immediate_exercise[:self.split] < np.infty

            # Train network on ITM paths
            predicted_continuation = np.zeros(len(stock_at_t))
            if np.sum(itm) > 0:
                X_train = stock_at_t[:self.split][itm]
                y_train = continuation_values[:self.split][itm]

                network = self._train_network(X_train, y_train)

                # Predict for all paths
                itm_all = immediate_exercise > 0 if self.train_ITM_only else immediate_exercise < np.infty
                if np.sum(itm_all) > 0:
                    predicted_continuation[itm_all] = self._evaluate_network(network, stock_at_t[itm_all])

            # Exercise decision
            exercise = immediate_exercise > predicted_continuation
            values[:, t] = np.where(exercise, immediate_exercise, continuation_values)

        # Calculate price
        price = np.mean(values[self.split:, 0])
        price = max(price, payoffs[0, 0])

        return price, time_for_path_gen
