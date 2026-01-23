"""
Thesis experiment configurations.

This module provides pre-defined configurations for reproducing
thesis experiments. Import configurations and pass to run_algo.py.

Available configurations:
- thesis_table_4_2: Algorithmic comparison across dimensions
- thesis_table_4_3: MaxCall activation function validation
- thesis_barriers: Barrier option validation
- thesis_path_dependent: Path-dependent performance comparison
"""

from experiments.configs.thesis_chapter4 import (
    thesis_table_4_2,
    thesis_table_4_3,
    thesis_barriers,
    thesis_path_dependent,
)

__all__ = [
    'thesis_table_4_2',
    'thesis_table_4_3',
    'thesis_barriers',
    'thesis_path_dependent',
]
