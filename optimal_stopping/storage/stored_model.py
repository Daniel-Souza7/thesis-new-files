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
from optimal_stopping.models.stock_model import Model


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
        start_index: Offset for sliding window (used for Train/Eval split)
        **kwargs: Other parameters passed to base Model class
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
        start_index: int = 0,  # <--- Added Sliding Window Parameter
        **kwargs
    ):
        """Initialize stored paths model."""
        # Remove 'name' from kwargs if present to avoid conflict
        kwargs.pop('name', None)

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
        self.start_index = start_index  # Store the offset

        # Open HDF5 file and validate (but don't load into memory yet)
        from optimal_stopping.storage.path_storage import STORAGE_DIR
        import h5py

        filename = f"{base_model}_{storage_id}.h5"
        filepath = STORAGE_DIR / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Stored paths not found: {filepath}")

        print(f"📂 Opening stored {base_model} paths (ID: {storage_id})...")

        # Keep HDF5 file open for memory-mapped access
        self._h5_file = h5py.File(filepath, 'r')
        self._metadata = dict(self._h5_file.attrs)

        # Validate parameters
        stored_stocks = self._metadata['nb_stocks']
        stored_paths = self._metadata['nb_paths']
        stored_dates = self._metadata['nb_dates']
        stored_maturity = self._metadata['maturity']
        stored_spot = self._metadata['spot']

        print(f"   Stored: {stored_stocks} stocks, {stored_paths:,} paths, {stored_dates} dates")
        print(f"   Requested: {nb_stocks} stocks, {nb_paths:,} paths, {nb_dates} dates")
        print(f"   Sliding Window: Reading {nb_paths:,} paths starting at index {start_index:,}")

        # Validate
        errors = []
        if nb_dates != stored_dates:
            errors.append(f"nb_dates mismatch: requested {nb_dates}, stored {stored_dates}")
        if not np.isclose(maturity, stored_maturity, rtol=1e-6):
            errors.append(f"maturity mismatch: requested {maturity}, stored {stored_maturity}")
        if nb_stocks > stored_stocks:
            errors.append(f"nb_stocks too large: requested {nb_stocks}, only {stored_stocks} stored")

        # VALIDATION CHANGE: Check if range [start_index, start_index + nb_paths] fits in file
        required_end_index = start_index + nb_paths
        if required_end_index > stored_paths:
            errors.append(
                f"Request out of bounds! Needed up to index {required_end_index:,}, "
                f"but file only has {stored_paths:,} paths.\n"
                f"   (Start: {start_index:,}, Count: {nb_paths:,})"
            )

        if errors:
            self._h5_file.close()
            raise ValueError("Incompatible parameters:\n" + "\n".join(f"  - {e}" for e in errors))

        # Store slicing parameters
        self._nb_stocks_requested = nb_stocks
        self._nb_paths_requested = nb_paths
        self._spot_scale = spot / stored_spot

        if not np.isclose(self._spot_scale, 1.0, rtol=1e-6):
            print(f"   📊 Will rescale paths: {stored_spot} → {spot} (×{self._spot_scale:.4f})")

        print(f"✅ Ready to use stored paths (memory-mapped)")

        # Track which paths have been used (for sequential access)
        self._next_path_idx = 0

    def generate_paths(self, nb_paths: Optional[int] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return stored paths (read from memory-mapped HDF5) with SLIDING WINDOW logic.

        Args:
            nb_paths: Number of paths to return (default: requested paths)
                     If less than stored, returns first nb_paths

        Returns:
            paths: Array of shape (nb_paths, nb_stocks, nb_dates+1)
            variance_paths: Array of shape (nb_paths, nb_stocks, nb_dates+1) or None
        """
        nb_paths = nb_paths or self._nb_paths_requested

        if nb_paths > self._nb_paths_requested:
            raise ValueError(
                f"Requested {nb_paths} paths but only {self._nb_paths_requested} configured. "
                f"Use nb_paths ≤ {self._nb_paths_requested}"
            )

        # SLIDING WINDOW LOGIC
        # We read from [start_index : start_index + nb_paths]
        start = self.start_index
        end = self.start_index + nb_paths

        # Read from memory-mapped HDF5 (only loads what we need)
        paths = self._h5_file['paths'][start:end, :self._nb_stocks_requested, :]

        # Apply spot rescaling if needed
        if not np.isclose(self._spot_scale, 1.0, rtol=1e-6):
            paths = paths * self._spot_scale

        # Read variance paths if present
        variance_paths = None
        if 'variance_paths' in self._h5_file:
            variance_paths = self._h5_file['variance_paths'][start:end, :self._nb_stocks_requested, :]

        return paths, variance_paths

    def generate_one_path(self) -> np.ndarray:
        """Generate a single path (returns next sequential path from storage).

        Returns:
            path: Array of shape (nb_stocks, nb_dates+1)
        """
        # Calculate absolute index in the H5 file
        file_idx = self.start_index + self._next_path_idx

        # Check bounds
        if file_idx >= self._metadata['nb_paths']:
            # This shouldn't happen if validation passed, but good safety
             raise IndexError("End of stored paths reached.")

        if self._next_path_idx >= self._nb_paths_requested:
            # Wrap around to beginning of OUR WINDOW (not file beginning)
            self._next_path_idx = 0
            file_idx = self.start_index # Reset to window start

        # Read single path from memory-mapped HDF5
        path = self._h5_file['paths'][file_idx, :self._nb_stocks_requested, :]

        # Apply spot rescaling if needed
        if not np.isclose(self._spot_scale, 1.0, rtol=1e-6):
            path = path * self._spot_scale

        self._next_path_idx += 1

        return path

    def __del__(self):
        """Close HDF5 file when object is destroyed."""
        if hasattr(self, '_h5_file'):
            try:
                self._h5_file.close()
            except:
                pass

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