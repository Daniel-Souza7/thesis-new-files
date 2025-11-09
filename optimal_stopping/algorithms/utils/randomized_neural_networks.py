"""
Randomized neural networks for reservoir computing.

In randomized neural networks (reservoir computing):
- Hidden layer weights are randomly initialized and frozen
- Only the output layer is trained via least squares regression
- Much faster than training full neural networks

Classes:
- Reservoir2: PyTorch randomized feedforward network for RLSM algorithms
- randomRNN: PyTorch randomized recurrent network for RRLSM algorithms
"""

import numpy as np
import torch


class Reservoir:
    """
    Legacy NumPy-based reservoir implementation.

    NOTE: This class is deprecated. Use Reservoir2 (PyTorch) instead for better
    performance and GPU support.

    Args:
        hidden_size: Number of random features
        state_size: Dimension of input state
    """

    def __init__(self, hidden_size, state_size):
        self.weight_matrix = np.random.normal(0, 1, (hidden_size, state_size))
        self.bias_vector = np.random.normal(0, 1, hidden_size)

    def activation_function(self, x):
        """Tanh activation function."""
        return np.tanh(x)

    def evaluate(self, state):
        """
        Evaluate reservoir at given state.

        Args:
            state: Input state vector of shape (state_size,)

        Returns:
            Feature vector of shape (hidden_size + 1,) with constant appended
        """
        # Vectorized computation
        features = self.weight_matrix @ state + self.bias_vector
        features = np.tanh(features)

        # Append constant feature
        features = np.append(features, 1)
        return features


def init_weights(m, mean=0., std=1.):
    """
    Initialize linear layer weights with normal distribution.

    Args:
        m: PyTorch module
        mean: Mean of normal distribution
        std: Standard deviation of normal distribution
    """
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, mean, std)
        torch.nn.init.normal_(m.bias, mean, std)


def init_weights_gen(mean=0., std=1., mean_b=0., std_b=1., dist=0):
    """
    Generate weight initialization function with configurable distributions.

    Args:
        mean: Mean for weight distribution
        std: Std for weight distribution (or upper bound for uniform)
        mean_b: Mean for bias distribution
        std_b: Std for bias distribution (or upper bound for uniform)
        dist: Distribution type:
            0 - Normal distribution (default)
            1 - Uniform distribution
            2 - Xavier uniform initialization
            3 - Xavier normal initialization

    Returns:
        Initialization function that can be passed to model.apply()
    """

    def _init_weights(m):
        if isinstance(m, torch.nn.Linear):
            if dist == 0:
                # Normal distribution
                torch.nn.init.normal_(m.weight, mean, std)
                torch.nn.init.normal_(m.bias, mean_b, std_b)
            elif dist == 1:
                # Uniform distribution
                torch.nn.init.uniform_(m.weight, mean, std)
                torch.nn.init.uniform_(m.bias, mean_b, std_b)
            elif dist == 2:
                # Xavier uniform
                torch.nn.init.xavier_uniform_(m.weight)
                # Xavier doesn't work for 1D bias, fall back to normal
                torch.nn.init.normal_(m.bias, mean_b, std_b)
            elif dist == 3:
                # Xavier normal
                torch.nn.init.xavier_normal_(m.weight)
                # Xavier doesn't work for 1D bias, fall back to normal
                torch.nn.init.normal_(m.bias, mean_b, std_b)
            else:
                raise ValueError(
                    f"Invalid dist parameter: {dist}. Must be 0, 1, 2, or 3.\n"
                    f"  0: Normal, 1: Uniform, 2: Xavier Uniform, 3: Xavier Normal"
                )

    return _init_weights


class Reservoir2(torch.nn.Module):
    """
    PyTorch-based randomized neural network (reservoir).

    Architecture: Input → Linear(random) → Activation → Output

    The Linear layer weights are randomly initialized and frozen.
    Only used for feature extraction; actual regression happens externally.

    Args:
        hidden_size: Number of random features
        state_size: Dimension of input state
        factors: Tuple of scaling factors
            - If length 1: (input_scale,)
            - If length 8: (input_scale, ..., weight_init_params[5])
        activation: Activation function (default: LeakyReLU(0.5))

    Example:
        >>> reservoir = Reservoir2(hidden_size=50, state_size=5, factors=(1.0,))
        >>> x = torch.randn(100, 5)  # 100 samples, 5 features
        >>> features = reservoir(x)  # (100, 50) random features
    """

    def __init__(self, hidden_size, state_size, factors=(1.,),
                 activation=torch.nn.LeakyReLU(0.5)):
        super().__init__()
        self.factors = factors
        self.hidden_size = hidden_size

        layers = [
            torch.nn.Linear(state_size, hidden_size, bias=True),
            activation
        ]
        self.NN = torch.nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize random weights (then freeze them)."""
        if len(self.factors) == 8:
            # Extended initialization with custom parameters
            _init_weights = init_weights_gen(*self.factors[3:])
        else:
            # Default normal initialization
            _init_weights = init_weights

        self.apply(_init_weights)

        # Freeze all parameters - these are random features, not trainable
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input):
        """
        Extract random features from input.

        Args:
            input: Tensor of shape (batch_size, state_size)

        Returns:
            Random features of shape (batch_size, hidden_size)
        """
        # Scale input by first factor
        scaled_input = input * self.factors[0]
        return self.NN(scaled_input)


class randomRNN(torch.nn.Module):
    """
    Randomized recurrent neural network for non-Markovian problems.

    Used in RRLSM algorithms for fractional Brownian motion and path-dependent
    options where history matters.

    Architecture:
        - Standard mode: h_t = Tanh(W * [x_t, h_{t-1}])
        - Extended mode: Adds auxiliary feedforward path for richer features

    Args:
        hidden_size: Number of hidden units
        state_size: Dimension of input state
        factors: Scaling factors (input_scale, hidden_scale, aux_scale)
        extend: If True, uses extended architecture with auxiliary network

    Example:
        >>> rnn = randomRNN(hidden_size=20, state_size=5, factors=(1., 1., 1.))
        >>> x = torch.randn(10, 100, 5)  # 10 timesteps, 100 paths, 5 features
        >>> features = rnn(x)  # (10, 100, 20) sequential features
    """

    def __init__(self, hidden_size, state_size, factors=(1., 1., 1.),
                 extend=False):
        super().__init__()
        self.factors = factors
        self.extend = extend

        # In extended mode, split hidden size between RNN and auxiliary net
        if self.extend:
            self.hidden_size = int(hidden_size / 2)
        else:
            self.hidden_size = hidden_size

        # Main RNN: takes concatenated [input, hidden_state]
        layers = [
            torch.nn.Linear(state_size + self.hidden_size, self.hidden_size,
                            bias=True),
            torch.nn.Tanh()
        ]
        self.NN = torch.nn.Sequential(*layers)

        # Auxiliary feedforward network (extended mode only)
        if self.extend:
            layers = [
                torch.nn.Linear(state_size, self.hidden_size, bias=True),
                torch.nn.Tanh()
            ]
            self.NN2 = torch.nn.Sequential(*layers)

        # Initialize with random weights and freeze
        self.apply(init_weights)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input):
        """
        Process sequential input through randomized RNN.

        Args:
            input: Tensor of shape (seq_len, batch_size, state_size)

        Returns:
            Sequential features of shape:
                - Standard mode: (seq_len, batch_size, hidden_size)
                - Extended mode: (seq_len, batch_size, 2*hidden_size)
        """
        # Initialize hidden state on same device as input
        h = torch.zeros(input.shape[1], self.hidden_size, device=input.device)

        if self.extend:
            # Extended mode: output includes both RNN and auxiliary features
            hs_size = list(input.shape[:-1]) + [self.hidden_size * 2]
            hs = torch.zeros(hs_size, device=input.device)

            for i in range(input.shape[0]):
                # Concatenate scaled input with scaled hidden state
                x = torch.cat([
                    torch.tanh(input[i] * self.factors[0]),
                    h * self.factors[1]
                ], dim=-1)

                # Update hidden state
                h = self.NN(x)

                # Combine RNN output with auxiliary network output
                hs[i] = torch.cat([
                    h,
                    self.NN2(input[i] * self.factors[2])
                ], dim=-1)
        else:
            # Standard mode: only RNN features
            hs_size = list(input.shape[:-1]) + [self.hidden_size]
            hs = torch.zeros(hs_size, device=input.device)

            for i in range(input.shape[0]):
                # Concatenate scaled input with scaled hidden state
                x = torch.cat([
                    input[i] * self.factors[0],
                    h * self.factors[1]
                ], dim=-1)

                # Update hidden state
                h = self.NN(x)
                hs[i] = h

        return hs