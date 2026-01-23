"""
Path storage and data management utilities.

This module provides utilities for generating, storing, and loading
Monte Carlo paths for reproducible experiments:

- StoredPathsModel: Load pre-computed paths from HDF5 files
- PathStorage: Utilities for path caching and retrieval
- UserDataModel: Load custom user-provided data
- store_paths: Functions for generating and saving paths

Example usage:
    >>> from optimal_stopping.storage import StoredPathsModel
    >>>
    >>> # Load pre-computed paths
    >>> model = StoredPathsModel(
    ...     path_file='data/stored_paths/BS_50.h5',
    ...     nb_paths=1000000
    ... )
    >>> paths = model.generate_paths()

For generating new path datasets:
    >>> from optimal_stopping.storage import store_paths
    >>> store_paths.generate_and_store(
    ...     model_name='BlackScholes',
    ...     nb_stocks=50,
    ...     nb_paths=10000000,
    ...     output_file='data/stored_paths/BS_50_custom.h5'
    ... )
"""

from optimal_stopping.storage.stored_model import StoredPathsModel
from optimal_stopping.storage.path_storage import PathStorage
from optimal_stopping.storage.user_data_model import UserDataModel

__all__ = [
    'StoredPathsModel',
    'PathStorage',
    'UserDataModel',
]
