"""
Basis functions for regression in LSM and FQI algorithms.

Available basis types:
- Polynomial (degree 1 or 2)
- Weighted Laguerre polynomials
- Time-dependent Laguerre (for FQI with explicit time features)
"""

import numpy as np
from itertools import combinations


class BasisFunctions:
    """
    Degree-2 polynomial basis functions.

    Basis: {1, x₁, ..., xₙ, x₁², ..., xₙ², x₁x₂, x₁x₃, ..., xₙ₋₁xₙ}

    Number of basis functions: 1 + 2n + n(n-1)/2
    """

    def __init__(self, nb_stocks):
        self.nb_stocks = nb_stocks
        lst = list(range(self.nb_stocks))
        self.combs = [list(x) for x in combinations(lst, 2)]
        self.nb_base_fcts = 1 + 2 * self.nb_stocks + len(self.combs)

    def base_fct(self, i, x, vectorized=False):
        """
        Evaluate i-th basis function at x.

        Args:
            i: Basis function index (0 to nb_base_fcts-1)
            x: Stock prices
               - If vectorized=False: shape (nb_stocks,) - single path
               - If vectorized=True: shape (nb_paths, nb_stocks) - multiple paths
            vectorized: Whether x contains multiple paths

        Returns:
            Basis function value(s) at x
        """
        if i < 0 or i >= self.nb_base_fcts:
            raise ValueError(f"Basis function index {i} out of range [0, {self.nb_base_fcts})")

        if vectorized:
            # x shape: (nb_paths, nb_stocks)
            if i == 0:
                return np.ones(x.shape[0])  # Constant
            elif i <= self.nb_stocks:
                return x[:, i - 1]  # Linear terms: x₁, x₂, ..., xₙ
            elif i <= 2 * self.nb_stocks:
                k = i - self.nb_stocks - 1
                return x[:, k] ** 2  # Quadratic terms: x₁², x₂², ..., xₙ²
            else:
                k = i - 2 * self.nb_stocks - 1
                return x[:, self.combs[k][0]] * x[:, self.combs[k][1]]  # Cross terms
        else:
            # x shape: (nb_stocks,)
            if i == 0:
                return np.ones_like(x[0])  # Constant
            elif i <= self.nb_stocks:
                return x[i - 1]  # Linear terms
            elif i <= 2 * self.nb_stocks:
                k = i - self.nb_stocks - 1
                return x[k] ** 2  # Quadratic terms
            else:
                k = i - 2 * self.nb_stocks - 1
                return x[self.combs[k][0]] * x[self.combs[k][1]]  # Cross terms


class BasisFunctionsDeg1:
    """
    Degree-1 polynomial basis functions.

    Basis: {1, x₁, ..., xₙ}

    Number of basis functions: 1 + n
    """

    def __init__(self, nb_stocks):
        self.nb_stocks = nb_stocks
        self.nb_base_fcts = 1 + self.nb_stocks

    def base_fct(self, i, x):
        """
        Evaluate i-th basis function at x.

        Args:
            i: Basis function index (0 to nb_base_fcts-1)
            x: Stock prices, shape (nb_stocks,)

        Returns:
            Basis function value at x
        """
        if i < 0 or i >= self.nb_base_fcts:
            raise ValueError(f"Basis function index {i} out of range [0, {self.nb_base_fcts})")

        if i == 0:
            return np.ones_like(x[0])  # Constant
        else:
            return x[i - 1]  # Linear terms: x₁, x₂, ..., xₙ


class BasisFunctionsLaguerre:
    """
    Weighted Laguerre polynomial basis functions.

    Basis for each stock i:
    - L₀(xᵢ) = exp(-xᵢ/2K)
    - L₁(xᵢ) = exp(-xᵢ/2K) * (1 - xᵢ/K)
    - L₂(xᵢ) = exp(-xᵢ/2K) * (1 - 2xᵢ/K + (xᵢ/K)²/2)

    Number of basis functions: 1 + 3n

    Laguerre polynomials often provide better approximation than standard
    polynomials for option pricing problems.
    """

    def __init__(self, nb_stocks, K=1):
        self.nb_stocks = nb_stocks
        self.nb_base_fcts = 1 + 3 * self.nb_stocks
        self.K = K  # Scaling parameter (typically strike price)

    def base_fct(self, i, x):
        """
        Evaluate i-th basis function at x.

        Args:
            i: Basis function index (0 to nb_base_fcts-1)
            x: Stock prices, shape (nb_stocks,)

        Returns:
            Basis function value at x
        """
        if i < 0 or i >= self.nb_base_fcts:
            raise ValueError(f"Basis function index {i} out of range [0, {self.nb_base_fcts})")

        x_scaled = x / self.K

        if i == 0:
            return np.ones_like(x[0])  # Constant
        elif i <= self.nb_stocks:
            # L₀: exp(-x/2)
            return np.exp(-x_scaled[i - 1] / 2)
        elif i <= 2 * self.nb_stocks:
            # L₁: exp(-x/2) * (1 - x)
            k = i - self.nb_stocks - 1
            return np.exp(-x_scaled[k] / 2) * (1 - x_scaled[k])
        else:
            # L₂: exp(-x/2) * (1 - 2x + x²/2)
            k = i - 2 * self.nb_stocks - 1
            x_k = x_scaled[k]
            return np.exp(-x_k / 2) * (1 - 2 * x_k + x_k ** 2 / 2)


class BasisFunctionsLaguerreTime:
    """
    Laguerre basis with special time-dependent features.

    Used in FQI algorithms where time is included as an explicit feature.
    Assumes the last element of x is the current time.

    Number of basis functions: 1 + 3n

    For stock features (i < nb_stocks):
    - Same Laguerre polynomials as BasisFunctionsLaguerre

    For time feature (i = nb_stocks, 2*nb_stocks, 3*nb_stocks):
    - Special transformations: sin, log, quadratic
    """

    def __init__(self, nb_stocks, T, K=1):
        """
        Args:
            nb_stocks: Number of stocks (including time as last element)
            T: Maturity time
            K: Scaling parameter for stock prices
        """
        self.nb_stocks = nb_stocks
        self.nb_base_fcts = 1 + 3 * self.nb_stocks
        self.T = T
        self.K = K

    def base_fct(self, i, x):
        """
        Evaluate i-th basis function at x.

        Args:
            i: Basis function index (0 to nb_base_fcts-1)
            x: State vector (stocks + time), shape (nb_stocks,)
                Last element x[-1] is assumed to be time

        Returns:
            Basis function value at x
        """
        if i < 0 or i >= self.nb_base_fcts:
            raise ValueError(f"Basis function index {i} out of range [0, {self.nb_base_fcts})")

        x_scaled = x / self.K

        if i == 0:
            return np.ones_like(x[0])  # Constant

        # First layer: L₀ for stocks, sin for time
        elif i < self.nb_stocks:
            return np.exp(-x_scaled[i - 1] / 2)
        elif i == self.nb_stocks:  # Time feature: sin
            return np.sin(-np.pi * x_scaled[i - 1] / 2 * self.K + np.pi / 2)

        # Second layer: L₁ for stocks, log for time
        elif i < 2 * self.nb_stocks:
            k = i - self.nb_stocks - 1
            return np.exp(-x_scaled[k] / 2) * (1 - x_scaled[k])
        elif i == 2 * self.nb_stocks:  # Time feature: log
            k = i - self.nb_stocks - 1
            return np.log(1 + self.T * (1 - x_scaled[k] * self.K))

        # Third layer: L₂ for stocks, quadratic for time
        elif i < 3 * self.nb_stocks:
            k = i - 2 * self.nb_stocks - 1
            x_k = x_scaled[k]
            return np.exp(-x_k / 2) * (1 - 2 * x_k + x_k ** 2 / 2)
        elif i == 3 * self.nb_stocks:  # Time feature: quadratic
            k = i - 2 * self.nb_stocks - 1
            return (x_scaled[k] * self.K) ** 2
        else:
            raise ValueError(f"Unexpected basis function index: {i}")