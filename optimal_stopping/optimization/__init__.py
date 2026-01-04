"""
Hyperparameter optimization module for optimal stopping algorithms.

This module provides automated hyperparameter tuning using:
- Bayesian Optimization (TPE) via Optuna
- Random Search baseline
- Multi-fidelity optimization (reduced paths during search)

Main components:
- hyperopt.py: Main optimization orchestrator
- objective.py: Objective function evaluation
- search_spaces.py: Hyperparameter search space definitions
"""

from .hyperopt import HyperparameterOptimizer

__all__ = ['HyperparameterOptimizer']
