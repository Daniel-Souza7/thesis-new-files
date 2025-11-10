"""
Base classes for option payoffs.
"""

import numpy as np


class Payoff:
    """Base class for option payoff functions."""

    is_path_dependent = False  # Override in subclass if path-dependent

    def __init__(self, strike):
        """
        Initialize payoff with strike price.

        Args:
            strike: Strike price K
        """
        self.strike = strike

    def __call__(self, stock_paths):
        """
        Evaluate payoff for all paths at all timesteps.

        Args:
            stock_paths: Array of shape (nb_paths, nb_stocks, nb_dates+1)

        Returns:
            payoffs: Array of shape (nb_paths, nb_dates+1)
        """
        nb_paths, nb_stocks, nb_dates = stock_paths.shape
        payoffs = np.zeros((nb_paths, nb_dates))

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
            X: Array of shape (nb_paths, nb_stocks) or (nb_paths, nb_stocks, nb_dates+1)

        Returns:
            Array of shape (nb_paths,) or (nb_paths, nb_dates+1)
        """
        raise NotImplementedError("Subclasses must implement eval()")

    def __repr__(self):
        """String representation of payoff."""
        return f"{self.__class__.__name__}(strike={self.strike})"