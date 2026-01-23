"""
Deep learning baseline algorithms for American option pricing.

These algorithms use fully trainable neural networks (non-convex optimization)
as opposed to the randomized neural networks in core/:

- DOS: Deep Optimal Stopping (Becker et al., 2019)
- NLSM: Neural Least Squares Monte Carlo (Becker et al., 2020)

Note: These methods achieve good scalability but lack the convergence
guarantees of randomized methods due to non-convex optimization.
"""

from optimal_stopping.algorithms.deep.dos import DOS
from optimal_stopping.algorithms.deep.nlsm import NLSM

__all__ = [
    'DOS',
    'NLSM',
]
