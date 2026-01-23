"""
Experimental algorithms for research and development.

These algorithms are not part of the core thesis framework and may be
unstable or incomplete. Use at your own risk.

Available algorithms:
- StochasticMesh: Broadie & Glasserman (2004) mesh method
- RandomizedStochasticMesh1, RandomizedStochasticMesh2: Randomized variants
- ZapQ, RZapQ: Zap Q-learning variants
- DKL, RDKL: Deep kernel learning variants
- SRFQI_RBF: RBF basis variant of SRFQI
"""

from optimal_stopping.algorithms.experimental.stochastic_mesh import SM
from optimal_stopping.algorithms.experimental.randomized_stochastic_mesh1 import RSM1
from optimal_stopping.algorithms.experimental.randomized_stochastic_mesh2 import RSM2
from optimal_stopping.algorithms.experimental.zap_q import ZAPQ
from optimal_stopping.algorithms.experimental.rzapq import RZAPQ
from optimal_stopping.algorithms.experimental.dkl import DKL
from optimal_stopping.algorithms.experimental.rdkl import RDKL
from optimal_stopping.algorithms.experimental.SRFQI_RBF import SRFQI_RBF

__all__ = [
    'SM',
    'RSM1',
    'RSM2',
    'ZAPQ',
    'RZAPQ',
    'DKL',
    'RDKL',
    'SRFQI_RBF',
]
