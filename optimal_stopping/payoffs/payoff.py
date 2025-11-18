"""
Base classes for option payoffs with auto-registration.
"""

import numpy as np


# Global registry for all payoff classes
_PAYOFF_REGISTRY = {}


class Payoff:
    """Base class for option payoff functions with auto-registration."""

    is_path_dependent = False  # Override in subclass if path-dependent
    payoff_type = "base"  # Override: "standard", "barrier", "lookback", etc.
    abbreviation = None  # Override with LaTeX abbreviation (e.g., "BskCall")

    def __init__(self, strike, **kwargs):
        """
        Initialize payoff with strike price and optional parameters.

        Args:
            strike: Strike price K
            **kwargs: Additional parameters (alpha, k, weights, barriers, etc.)
        """
        self.strike = strike
        self.params = kwargs  # Store all extra parameters

    def __init_subclass__(cls, **kwargs):
        """Auto-register payoff classes when they're defined."""
        super().__init_subclass__(**kwargs)

        # Register using class name
        if cls.__name__ not in ['Payoff', 'BarrierPayoff']:
            _PAYOFF_REGISTRY[cls.__name__] = cls

            # Also register by abbreviation if provided
            if hasattr(cls, 'abbreviation') and cls.abbreviation:
                _PAYOFF_REGISTRY[cls.abbreviation] = cls

    def __call__(self, stock_paths):
        """
        Evaluate payoff for all paths at all timesteps.

        Args:
            stock_paths: Array of shape (nb_paths, nb_stocks, nb_dates+1)

        Returns:
            payoffs: Array of shape (nb_paths, nb_dates+1)

        Note:
            Sets self.initial_prices to stock_paths[:, :, 0] for payoffs that
            require normalization by initial spot prices S_i(0). This allows
            normalized payoffs to compute returns like (S_i(t) - S_i(0)) / S_i(0).
        """
        nb_paths, nb_stocks, nb_dates = stock_paths.shape
        payoffs = np.zeros((nb_paths, nb_dates))

        # Store initial prices for normalization (t=0)
        self.initial_prices = stock_paths[:, :, 0]  # Shape: (nb_paths, nb_stocks)

        for date in range(nb_dates):
            if self.is_path_dependent:
                # Pass full history up to this date
                payoffs[:, date] = self.eval(stock_paths[:, :, :date + 1])
            else:
                # Pass only current timestep
                payoffs[:, date] = self.eval(stock_paths[:, :, date])

        return payoffs

    def eval(self, X):
        """
        Evaluate payoff for given stock prices.

        Args:
            X: Array of shape (nb_paths, nb_stocks) for standard options
               OR (nb_paths, nb_stocks, nb_dates+1) for path-dependent options

        Returns:
            Array of shape (nb_paths,) with payoff values
        """
        raise NotImplementedError("Subclasses must implement eval()")

    def __repr__(self):
        """String representation of payoff."""
        params_str = f", {self.params}" if self.params else ""
        return f"{self.__class__.__name__}(strike={self.strike}{params_str})"


def get_payoff_class(name):
    """
    Get payoff class by name or abbreviation.

    Args:
        name: Class name (e.g., 'BasketCall') or abbreviation (e.g., 'BskCall')

    Returns:
        Payoff class

    Raises:
        KeyError: If payoff not found
    """
    if name in _PAYOFF_REGISTRY:
        return _PAYOFF_REGISTRY[name]
    raise KeyError(f"Payoff '{name}' not found. Available: {list(_PAYOFF_REGISTRY.keys())}")


def list_payoffs():
    """List all registered payoffs."""
    return sorted(_PAYOFF_REGISTRY.keys())
