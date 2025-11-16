"""
Asian-style basket option payoffs (d > 1) - PATH-DEPENDENT.

These options depend on the average of basket prices over time.
"""

import numpy as np
from .payoff import Payoff


class AsianFixedStrikeCall(Payoff):
    """Asian Fixed Strike Basket Call: max(0, avg_over_time(mean(S)) - K)"""
    abbreviation = "AsianFi-BskCall"
    is_path_dependent = True

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks, nb_dates+1)"""
        # Compute basket at each time: (nb_paths, nb_dates+1)
        basket_over_time = np.mean(X, axis=1)
        # Average basket over time: (nb_paths,)
        avg_basket = np.mean(basket_over_time, axis=1)
        return np.maximum(0, avg_basket - self.strike)


class AsianFixedStrikePut(Payoff):
    """Asian Fixed Strike Basket Put: max(0, K - avg_over_time(mean(S)))"""
    abbreviation = "AsianFi-BskPut"
    is_path_dependent = True

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks, nb_dates+1)"""
        basket_over_time = np.mean(X, axis=1)
        avg_basket = np.mean(basket_over_time, axis=1)
        return np.maximum(0, self.strike - avg_basket)


class AsianFloatingStrikeCall(Payoff):
    """Asian Floating Strike Basket Call: max(0, mean(S_T) - avg_over_time(mean(S)))"""
    abbreviation = "AsianFl-BskCall"
    is_path_dependent = True

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks, nb_dates+1)"""
        # Basket at maturity: (nb_paths,)
        basket_final = np.mean(X[:, :, -1], axis=1)
        # Average basket over time: (nb_paths,)
        basket_over_time = np.mean(X, axis=1)
        avg_basket = np.mean(basket_over_time, axis=1)
        return np.maximum(0, basket_final - avg_basket)


class AsianFloatingStrikePut(Payoff):
    """Asian Floating Strike Basket Put: max(0, avg_over_time(mean(S)) - mean(S_T))"""
    abbreviation = "AsianFl-BskPut"
    is_path_dependent = True

    def eval(self, X):
        """X shape: (nb_paths, nb_stocks, nb_dates+1)"""
        basket_final = np.mean(X[:, :, -1], axis=1)
        basket_over_time = np.mean(X, axis=1)
        avg_basket = np.mean(basket_over_time, axis=1)
        return np.maximum(0, avg_basket - basket_final)
