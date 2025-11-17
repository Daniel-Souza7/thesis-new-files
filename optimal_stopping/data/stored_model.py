"""Stored paths model - loads pre-generated paths from disk.

This module provides a Model wrapper that loads paths from storage instead of
generating them on-the-fly. This is useful for:
- Reproducibility (same paths across experiments)
- Performance (skip expensive generation, especially for RealData)
- Consistency (compare algorithms on identical paths)

Usage in configs:
    # First, store paths (run once)
    storage_id = store_paths(
        stock_model='RealData',
        nb_stocks=50,
        nb_paths=100000,
        ...
    )

    # Then use in config (run many times)
    test_config = _DefaultConfig(
        stock_models=['RealDataStored{storage_id}'],  # e.g., 'RealDataStored1700000000123'
        nb_stocks=[10],  # Can use subset
        nb_paths=[50000],  # Can use subset
        ...
    )
"""

import numpy as np
from typing import Optional, Tuple
from optimal_stopping.data.stock_model import Model
from optimal_stopping.data.path_storage import load_paths


class StoredPathsModel(Model):
    """Model that loads pre-generated paths from storage.

    This is a wrapper around stored paths that provides the standard Model interface.
    Paths are loaded from HDF5 files created by store_paths().

    Args:
        base_model: Original model name (e.g., 'RealData', 'BlackScholes')
        storage_id: Storage ID from store_paths()
        nb_stocks: Number of stocks to use (must be ≤ stored)
        nb_paths: Number of paths to use (must be ≤ stored)
        nb_dates: Number of time steps (must match stored)
        maturity: Maturity in years (must match stored)
        spot: Spot price (can differ - paths will be rescaled)
        **kwargs: Other parameters passed to base Model class

    Example:
        >>> model = StoredPathsModel(
        ...     base_model='RealData',
        ...     storage_id='1700000000123',
        ...     nb_stocks=10,
        ...     nb_paths=50000,
        ...     nb_dates=252,
        ...     maturity=1.0,
        ...     spot=100,
        ... )
        >>> paths, variance_paths = model.generate_paths()
    """

    def __init__(
        self,
        base_model: str,
        storage_id: str,
        nb_stocks: int,
        nb_paths: int,
        nb_dates: int,
        maturity: float,
        spot: float = 100.0,
        drift: float = 0.05,  # Dummy defaults for base Model class
        volatility: float = 0.2,
        dividend: float = 0.0,
        **kwargs
    ):
        """Initialize stored paths model."""
        # Initialize base class with dummy drift/volatility
        # (These don't matter since we use stored paths)
        super().__init__(
            drift=drift,
            dividend=dividend,
            volatility=volatility,
            spot=spot,
            nb_stocks=nb_stocks,
            nb_paths=nb_paths,
            nb_dates=nb_dates,
            maturity=maturity,
            name=f"{base_model}Stored{storage_id}",
            **kwargs
        )

        self.base_model = base_model
        self.storage_id = storage_id

        # Load paths from storage (with validation)
        print(f"📂 Loading stored {base_model} paths (ID: {storage_id})...")
        self._stored_paths, self._stored_variance_paths, self._metadata = load_paths(
            stock_model=base_model,
            storage_id=storage_id,
            nb_stocks=nb_stocks,
            nb_paths=nb_paths,
            nb_dates=nb_dates,
            maturity=maturity,
            spot=spot,
            verbose=True
        )

        # Track which paths have been used (for sequential access)
        self._next_path_idx = 0

    def generate_paths(self, nb_paths: Optional[int] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return stored paths.

        Args:
            nb_paths: Number of paths to return (default: all stored paths)
                     If less than stored, returns first nb_paths

        Returns:
            paths: Array of shape (nb_paths, nb_stocks, nb_dates+1)
            variance_paths: Array of shape (nb_paths, nb_stocks, nb_dates+1) or None
        """
        nb_paths = nb_paths or self.nb_paths

        if nb_paths > self._stored_paths.shape[0]:
            raise ValueError(
                f"Requested {nb_paths} paths but only {self._stored_paths.shape[0]} stored. "
                f"Use nb_paths ≤ {self._stored_paths.shape[0]}"
            )

        # Return first nb_paths
        paths = self._stored_paths[:nb_paths]
        variance_paths = self._stored_variance_paths[:nb_paths] if self._stored_variance_paths is not None else None

        return paths, variance_paths

    def generate_one_path(self) -> np.ndarray:
        """Generate a single path (returns next sequential path from storage).

        Returns:
            path: Array of shape (nb_stocks, nb_dates+1)
        """
        if self._next_path_idx >= self._stored_paths.shape[0]:
            # Wrap around to beginning
            self._next_path_idx = 0

        path = self._stored_paths[self._next_path_idx]
        self._next_path_idx += 1

        return path

    def drift_fct(self, x, t):
        """Drift function (placeholder - paths already generated)."""
        # Use stored metadata if available, otherwise use base class value
        stored_drift = self._metadata.get('param_drift')
        if stored_drift is not None:
            try:
                import json
                drift_val = json.loads(stored_drift)[0]
                if drift_val is not None:
                    return drift_val * x
            except (json.JSONDecodeError, IndexError, TypeError):
                pass
        return self.drift * x

    def diffusion_fct(self, x, t, v=0):
        """Diffusion function (placeholder - paths already generated)."""
        # Use stored metadata if available, otherwise use base class value
        stored_vol = self._metadata.get('param_volatilities') or self._metadata.get('param_volatility')
        if stored_vol is not None:
            try:
                import json
                vol_val = json.loads(stored_vol)[0]
                if vol_val is not None:
                    return vol_val * x
            except (json.JSONDecodeError, IndexError, TypeError):
                pass
        return self.volatility * x


def create_stored_model_class(base_model: str, storage_id: str):
    """Factory function to create a stored model class for a specific storage.

    This is used by stock_model.py to dynamically register stored models.

    Args:
        base_model: Base model name (e.g., 'RealData')
        storage_id: Storage ID

    Returns:
        Model class that can be instantiated

    Example:
        >>> RealDataStored123 = create_stored_model_class('RealData', '123')
        >>> model = RealDataStored123(nb_stocks=10, nb_paths=1000, ...)
    """
    class DynamicStoredModel(StoredPathsModel):
        def __init__(self, **kwargs):
            super().__init__(
                base_model=base_model,
                storage_id=storage_id,
                **kwargs
            )

    # Set a descriptive name
    DynamicStoredModel.__name__ = f"{base_model}Stored{storage_id}"
    DynamicStoredModel.__qualname__ = f"{base_model}Stored{storage_id}"

    return DynamicStoredModel
