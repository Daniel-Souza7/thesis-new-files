"""
Hyperparameter optimization module for optimal stopping algorithms.

This module provides automated hyperparameter tuning using:
- Bayesian Optimization (TPE) via Optuna
- Random Search baseline
- Multi-fidelity optimization (reduced paths during search)
- Early stopping for iterative algorithms

Main components:
- hyperopt.py: Main optimization orchestrator
- objective.py: Objective function evaluation
- early_stopping.py: Early stopping callbacks for RFQI
- search_spaces.py: Hyperparameter search space definitions
"""

from .hyperopt import HyperparameterOptimizer
from .early_stopping import EarlyStopping

__all__ = ['HyperparameterOptimizer', 'EarlyStopping']
