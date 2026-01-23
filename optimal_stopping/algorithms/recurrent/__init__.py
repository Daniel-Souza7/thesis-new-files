"""
Recurrent algorithms for path-dependent option pricing.

These algorithms use Echo State Networks (ESN) or recurrent architectures
to process path history for path-dependent derivatives:

- RRLSM: Recurrent Randomized Least Squares Monte Carlo
- SRLSM: Special RLSM for path-dependent options
- SRFQI: Special RFQI for path-dependent options

Note: The RT algorithm in core/ provides a feedforward alternative
that often outperforms these recurrent methods (see thesis Section 3.1.4).
"""

from optimal_stopping.algorithms.recurrent.rrlsm import RRLSM
from optimal_stopping.algorithms.recurrent.srlsm import SRLSM
from optimal_stopping.algorithms.recurrent.srfqi import SRFQI

__all__ = [
    'RRLSM',
    'SRLSM',
    'SRFQI',
]
