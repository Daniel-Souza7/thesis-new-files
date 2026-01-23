"""
Abstract base class for pricing algorithms.

All pricing algorithms should inherit from BaseAlgorithm and implement
the required methods for a consistent interface.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any
import numpy as np


class BaseAlgorithm(ABC):
    """
    Abstract base class for American option pricing algorithms.

    All pricing algorithms (RT, RLSM, LSM, DOS, etc.) should implement
    this interface to ensure consistency across the framework.

    Attributes:
        model: Stock model for path generation
        payoff: Payoff function defining the derivative contract
        hidden_size: Number of neurons/basis functions (if applicable)
    """

    def __init__(self, model: Any, payoff: Any, **kwargs):
        """
        Initialize the pricing algorithm.

        Args:
            model: Stock model instance (e.g., BlackScholes, Heston)
            payoff: Payoff function instance (e.g., BasketCall, MaxCall)
            **kwargs: Algorithm-specific parameters
        """
        self.model = model
        self.payoff = payoff
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate model and payoff compatibility."""
        # Check model has required attributes
        required_model_attrs = ['nb_stocks', 'nb_paths', 'nb_dates', 'maturity']
        for attr in required_model_attrs:
            if not hasattr(self.model, attr):
                raise AttributeError(
                    f"Model must have '{attr}' attribute. "
                    f"Got model of type {type(self.model).__name__}"
                )

        # Check payoff has eval method
        if not hasattr(self.payoff, 'eval') or not callable(self.payoff.eval):
            raise AttributeError(
                f"Payoff must have callable 'eval' method. "
                f"Got payoff of type {type(self.payoff).__name__}"
            )

    @abstractmethod
    def price(self, train_eval_split: int = 2) -> Tuple[float, float]:
        """
        Compute the option price.

        This is the main entry point for pricing. Implementations should:
        1. Generate or load stock paths from the model
        2. Apply backward induction with continuation value estimation
        3. Return the price and computation time

        Args:
            train_eval_split: Ratio for splitting paths into training/evaluation
                              sets (e.g., 2 means 50% train, 50% eval)

        Returns:
            tuple: (price, computation_time) where:
                - price: Estimated option price at t=0
                - computation_time: Time spent on computation (seconds)
        """
        pass

    def _eval_payoff(self, stock_paths: np.ndarray, date: Optional[int] = None) -> np.ndarray:
        """
        Evaluate payoff at a specific date.

        Handles both path-dependent and non-path-dependent payoffs.

        Args:
            stock_paths: Stock price paths array
            date: Time index for evaluation (None for terminal)

        Returns:
            payoffs: Array of payoff values for each path
        """
        is_path_dependent = getattr(self.payoff, 'is_path_dependent', False)

        if is_path_dependent:
            if date is None:
                return self.payoff.eval(stock_paths[:, :self.model.nb_stocks, :])
            else:
                return self.payoff.eval(stock_paths[:, :self.model.nb_stocks, :date + 1])
        else:
            if date is None:
                return self.payoff.eval(stock_paths[:, :self.model.nb_stocks, -1])
            else:
                return self.payoff.eval(stock_paths[:, :self.model.nb_stocks, date])

    @property
    def is_path_dependent(self) -> bool:
        """Check if payoff is path-dependent."""
        return getattr(self.payoff, 'is_path_dependent', False)

    def __repr__(self) -> str:
        """String representation of the algorithm."""
        return (
            f"{self.__class__.__name__}("
            f"model={type(self.model).__name__}, "
            f"payoff={type(self.payoff).__name__})"
        )


class RandomizedAlgorithm(BaseAlgorithm):
    """
    Base class for randomized neural network algorithms.

    Extends BaseAlgorithm with common functionality for algorithms
    that use randomized (frozen) neural networks: RT, RLSM, RFQI, etc.

    Attributes:
        hidden_size: Number of neurons in the hidden layer
        activation: Activation function name ('relu', 'tanh', 'elu', 'leakyrelu')
        use_payoff_as_input: Whether to augment state with payoff value
    """

    def __init__(
        self,
        model: Any,
        payoff: Any,
        hidden_size: int = 20,
        activation: str = 'leakyrelu',
        use_payoff_as_input: bool = True,
        dropout: float = 0.0,
        **kwargs
    ):
        """
        Initialize randomized algorithm.

        Args:
            model: Stock model instance
            payoff: Payoff function instance
            hidden_size: Number of hidden neurons (default: 20)
            activation: Activation function ('relu', 'tanh', 'elu', 'leakyrelu')
            use_payoff_as_input: Whether to include payoff in state (default: True)
            dropout: Dropout probability (default: 0.0)
            **kwargs: Additional algorithm-specific parameters
        """
        super().__init__(model, payoff, **kwargs)
        self.hidden_size = hidden_size
        self.activation = activation
        self.use_payoff_as_input = use_payoff_as_input
        self.dropout = dropout

    def _get_state_size(self) -> int:
        """
        Compute the state dimension for the neural network.

        Returns:
            State size including stocks, variance (if applicable), and payoff hint
        """
        use_var = getattr(self.model, 'return_var', False)
        state_size = self.model.nb_stocks * (1 + use_var)
        if self.use_payoff_as_input:
            state_size += 1
        return state_size


class DeepAlgorithm(BaseAlgorithm):
    """
    Base class for deep learning algorithms.

    Extends BaseAlgorithm with common functionality for algorithms
    that use fully trainable neural networks: DOS, NLSM.

    Note: These methods use non-convex optimization and do not
    have the convergence guarantees of randomized methods.
    """

    def __init__(
        self,
        model: Any,
        payoff: Any,
        learning_rate: float = 0.001,
        nb_epochs: int = 100,
        **kwargs
    ):
        """
        Initialize deep learning algorithm.

        Args:
            model: Stock model instance
            payoff: Payoff function instance
            learning_rate: Learning rate for gradient descent
            nb_epochs: Number of training epochs
            **kwargs: Additional parameters
        """
        super().__init__(model, payoff, **kwargs)
        self.learning_rate = learning_rate
        self.nb_epochs = nb_epochs
