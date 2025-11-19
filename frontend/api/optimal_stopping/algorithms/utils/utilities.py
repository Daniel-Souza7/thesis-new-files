"""
Utility functions for Greeks computation in American option pricing.

Author: Florian Krach
"""

import numpy as np


def get_poly_basis_and_derivatives(X, d):
    """
    Compute polynomial regression basis and its derivatives for Greeks calculation.

    Creates separate polynomial bases for each stock (no cross-terms) up to degree d:
    Basis = {1, x₁, x₁², ..., x₁ᵈ, x₂, x₂², ..., x₂ᵈ, ..., xₙ, xₙ², ..., xₙᵈ}

    This structure allows computing per-stock derivatives (delta, gamma) efficiently
    without considering cross-derivative terms.

    Args:
        X: Stock prices, shape (nb_paths, nb_stocks)
        d: Maximum polynomial degree

    Returns:
        tuple of (poly_basis, poly_basis_delta, poly_basis_gamma):
            - poly_basis: Polynomial basis, shape (nb_paths, (1+d)*nb_stocks)
                First column is all ones (constant), then d powers for each stock
            - poly_basis_delta: First derivatives ∂/∂Sᵢ, shape (nb_paths, (1+d)*nb_stocks)
            - poly_basis_gamma: Second derivatives ∂²/∂Sᵢ², shape (nb_paths, (1+d)*nb_stocks)

    Example:
        >>> X = np.array([[100, 105], [98, 103]])  # 2 paths, 2 stocks
        >>> basis, delta, gamma = get_poly_basis_and_derivatives(X, d=2)
        >>> basis.shape  # (2, 5) = 1 constant + 2*2 polynomial terms
        (2, 5)
        >>> # basis columns: [1, x₁, x₁², x₂, x₂²]
    """
    nb_stock = X.shape[1]

    # Initialize basis matrices
    poly_basis = np.ones((X.shape[0], (1 + d) * nb_stock))
    poly_basis_delta = np.zeros((X.shape[0], (1 + d) * nb_stock))
    poly_basis_gamma = np.zeros((X.shape[0], (1 + d) * nb_stock))

    # Fill in polynomial terms for each stock
    for j in range(nb_stock):
        for i in range(1, d + 1):
            col_idx = i + j * nb_stock

            # Basis: xⱼⁱ
            poly_basis[:, col_idx] = X[:, j] ** i

            # First derivative: i·xⱼⁱ⁻¹
            poly_basis_delta[:, col_idx] = i * X[:, j] ** (i - 1)

            # Second derivative: i(i-1)·xⱼⁱ⁻²
            if i > 1:
                poly_basis_gamma[:, col_idx] = i * (i - 1) * X[:, j] ** (i - 2)
            else:
                poly_basis_gamma[:, col_idx] = 0.

    return poly_basis, poly_basis_delta, poly_basis_gamma


def compute_gamma_via_BS_PDE(price, delta, theta, rate, vola, spot, dividend=0.):
    """
    Compute gamma using the Black-Scholes partial differential equation.

    The Black-Scholes PDE for an option with dividends is:
        ∂V/∂t + (r-q)S∂V/∂S + ½σ²S²∂²V/∂S² = rV

    Solving for gamma (Γ = ∂²V/∂S²):
        Γ = 2(rV - θ - (r-q)SΔ) / (σ²S²)

    where:
        - V is option price
        - θ (theta) = ∂V/∂t is time decay
        - Δ (delta) = ∂V/∂S is price sensitivity
        - r is risk-free rate
        - q is dividend yield
        - σ is volatility
        - S is spot price

    Args:
        price: Option price (V)
        delta: Option delta (∂V/∂S)
        theta: Option theta (∂V/∂t), typically negative
        rate: Risk-free interest rate (r), annualized
        vola: Volatility (σ), annualized
        spot: Current stock price (S)
        dividend: Continuous dividend yield (q), annualized (default: 0)

    Returns:
        gamma: Second derivative of option price w.r.t. spot (∂²V/∂S²)

    References:
        - Black-Scholes equation: https://en.wikipedia.org/wiki/Black–Scholes_equation
        - With dividends: https://www.math.tamu.edu/~mike.stecher/425/Sp12/optionsForDividendStocks.pdf

    Note:
        This method is more stable than finite differences for computing gamma,
        especially when delta and theta are already available from regression methods.
    """
    numerator = rate * price - theta - (rate - dividend) * spot * delta
    denominator = 0.5 * (vola ** 2) * (spot ** 2)
    return numerator / denominator