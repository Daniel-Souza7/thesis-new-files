"""
Neural network architectures for American option pricing algorithms.

Networks:
- NetworkNLSM: For Neural Least Squares Monte Carlo (learns continuation values)
- NetworkDOS: For Deep Optimal Stopping (learns stopping decisions)
"""

import torch.nn as nn


class NetworkNLSM(nn.Module):
    """
    Neural network for NLSM algorithm.

    Architecture: Input → Linear(H) → LeakyReLU → Linear(1) → Output

    This is a simple 2-layer feedforward network that approximates the
    continuation value function in the NLSM algorithm.

    Args:
        nb_stocks: Number of input features (stock prices)
        hidden_size: Number of neurons in hidden layer (default: 10)
    """

    def __init__(self, nb_stocks, hidden_size=10):
        super(NetworkNLSM, self).__init__()
        self.layer1 = nn.Linear(nb_stocks, hidden_size)
        self.activation = nn.LeakyReLU(0.5)
        self.layer2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, nb_stocks)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


class NetworkDOS(nn.Module):
    """
    Neural network for Deep Optimal Stopping algorithm.

    Architecture: 
        BatchNorm(input) → Linear(H) → ReLU → BatchNorm(H) → 
        Linear(1) → Sigmoid → Output

    The network learns to predict the optimal stopping decision (0=continue, 1=stop).
    Batch normalization helps with training stability.

    Args:
        nb_stocks: Number of input features (stock prices)
        hidden_size: Number of neurons in hidden layer (default: 10)
    """

    def __init__(self, nb_stocks, hidden_size=10):
        super(NetworkDOS, self).__init__()
        self.bn_input = nn.BatchNorm1d(num_features=nb_stocks)
        self.layer1 = nn.Linear(nb_stocks, hidden_size)
        self.activation = nn.ReLU()
        self.bn_hidden = nn.BatchNorm1d(num_features=hidden_size)
        self.layer2 = nn.Linear(hidden_size, 1)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, nb_stocks)

        Returns:
            Output tensor of shape (batch_size, 1) with values in [0, 1]
            representing stopping probability
        """
        x = self.bn_input(x)
        x = self.layer1(x)
        x = self.activation(x)
        x = self.bn_hidden(x)
        x = self.layer2(x)
        x = self.output_activation(x)
        return x