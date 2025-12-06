"""
Tree-based algorithms for American option pricing.

This module implements lattice-based methods for optimal stopping problems:
- CRR: Cox-Ross-Rubinstein binomial tree
- LR: Leisen-Reimer binomial tree
- Trinomial: Three-jump process tree
"""

from .crr import CRRTree
from .leisen_reimer import LeisenReimerTree
from .trinomial import TrinomialTree

__all__ = ['CRRTree', 'LeisenReimerTree', 'TrinomialTree']
