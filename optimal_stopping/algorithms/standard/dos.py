"""
Deep Optimal Stopping (DOS) for American option pricing.

Implementation of Deep Optimal Stopping introduced in
(Deep optimal stopping, Becker, Cheridito and Jentzen, 2019).

This is a benchmark algorithm that directly learns the stopping decision.
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


class DOS:
    """
    Deep Optimal Stopping pricer.

    Uses a neural network to learn the stopping rule directly.
    """

    def __init__(self, model, payoff, nb_epochs=20, hidden_size=10,
                 use_payoff_as_input=False, **kwargs):
        """
        Initialize DOS pricer.

        Args:
            model: Stock model
            payoff: Payoff function
            nb_epochs: Number of training epochs
            hidden_size: Number of neurons in hidden layer
            use_payoff_as_input: If True, include payoff in state
        """
        self.model = model
        self.payoff = payoff
        self.nb_epochs = nb_epochs
        self.hidden_size = hidden_size
        self.use_payoff_as_input = use_payoff_as_input
        self.use_var = getattr(model, 'return_var', False)
        self.batch_size = 2000

        # State size
        self.state_size = model.nb_stocks * (1 + self.use_var) + self.use_payoff_as_input * 1

    def _train_stopping_network(self, X_inputs, immediate_payoffs, discounted_next_values):
        """
        Train neural network to make stopping decisions.

        Network outputs probability of stopping in [0, 1].
        Objective: maximize expected value = stop_prob * immediate_payoff + (1-stop_prob) * continuation

        Args:
            X_inputs: State features
            immediate_payoffs: Immediate exercise values
            discounted_next_values: Discounted future values if not stopping
        """
        # Create network
        network = neural_networks.NetworkDOS(self.state_size, hidden_size=self.hidden_size).double()
        network.apply(init_weights)

        # Prepare data
        X_inputs = torch.from_numpy(X_inputs).double()
        immediate_payoffs = torch.from_numpy(immediate_payoffs).double()
        discounted_next_values = torch.from_numpy(discounted_next_values).double()

        # Train
        optimizer = optim.Adam(network.parameters())
        network.train(True)

        for _ in range(self.nb_epochs):
            for batch in tdata.BatchSampler(
                    tdata.RandomSampler(range(len(X_inputs)), replacement=False),
                    batch_size=self.batch_size, drop_last=False):
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    # Network outputs stopping probability
                    stop_prob = network(X_inputs[batch]).reshape(-1)

                    # Expected value if we use this stopping rule
                    values = (immediate_payoffs[batch] * stop_prob +
                             discounted_next_values[batch] * (1 - stop_prob))

                    # Maximize expected value (minimize negative)
                    loss = -torch.mean(values)
                    loss.backward()
                    optimizer.step()

        return network

    def _evaluate_stopping_network(self, network, X_inputs):
        """
        Evaluate stopping network.

        Args:
            network: Trained stopping network
            X_inputs: State features

        Returns:
            Stopping probabilities [0, 1]
        """
        network.train(False)
        X_inputs = torch.from_numpy(X_inputs).double()
        with torch.no_grad():
            outputs = network(X_inputs)
        return outputs.view(len(X_inputs)).detach().numpy()

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

        # Initialize values at maturity
        values = payoffs[:, -1].copy()

        # Store networks for each timestep
        networks = [None] * nb_dates

        # Backward induction - train networks
        for t in range(nb_dates - 1, 0, -1):
            # Current stock prices
            stock_at_t = stock_paths[:, :, t]

            # Add variance if needed
            if self.use_var:
                stock_at_t = np.concatenate([stock_at_t, var_paths[:, :, t]], axis=1)

            # Add payoff if needed
            if self.use_payoff_as_input:
                stock_at_t = np.concatenate([stock_at_t, payoffs[:, t:t+1]], axis=1)

            # Immediate exercise values
            immediate_payoffs = payoffs[:, t]

            # Discounted next values
            discounted_next = discount_factor * values

            # Train network on training paths
            X_train = stock_at_t[:self.split]
            immediate_train = immediate_payoffs[:self.split]
            discounted_train = discounted_next[:self.split]

            network = self._train_stopping_network(X_train, immediate_train, discounted_train)
            networks[t] = network

            # Evaluate stopping decisions for all paths
            stop_prob = self._evaluate_stopping_network(network, stock_at_t)

            # Threshold at 0.5 for binary decision
            stop = stop_prob >= 0.5

            # Update values
            values = np.where(stop, immediate_payoffs, discounted_next)

        # Calculate price
        price = np.mean(values[self.split:])
        price = max(price, payoffs[0, 0])

        return price, time_for_path_gen
