"""Path storage and retrieval for stock models.

This module provides functionality to store generated paths to disk and load them
later for reuse across multiple experiments. This is especially useful for:
- RealData model (expensive data downloads)
- Large path counts (100k+ paths)
- Reproducibility across experiments

Usage:
    # Store paths
    storage_id = store_paths(
        stock_model='RealData',
        nb_stocks=50,
        nb_paths=100000,
        nb_dates=252,
        maturity=1.0,
        spot=100,
        drift=(None,),  # RealData-specific
        volatilities=(None,),
    )

    # Use in config
    test_config = _DefaultConfig(
        stock_models=[f'RealDataStored{storage_id}'],
        nb_stocks=[10],  # Can use subset
        nb_paths=[50000],  # Can use subset
        spot=[90, 100, 110],  # Can rescale
        ...
    )
"""

import h5py
import numpy as np
import os
import time
import json
import warnings
from typing import Dict, Any, Optional, Tuple
from pathlib import Path


# Storage directory
STORAGE_DIR = Path(__file__).parent / 'stored_paths'


def _ensure_storage_dir():
    """Create storage directory if it doesn't exist."""
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def _generate_storage_id() -> str:
    """Generate unique storage ID based on timestamp.

    Returns:
        Timestamp-based ID (e.g., '1700000000123')
    """
    return str(int(time.time() * 1000))


def store_paths(
    stock_model: str,
    nb_stocks: int,
    nb_paths: int,
    nb_dates: int,
    maturity: float,
    spot: float = 100.0,
    custom_id: Optional[str] = None,
    **model_params
) -> str:
    """Store generated paths to disk for later reuse.

    Args:
        stock_model: Model name (e.g., 'RealData', 'BlackScholes')
        nb_stocks: Number of stocks
        nb_paths: Number of paths to generate
        nb_dates: Number of time steps
        maturity: Maturity in years
        spot: Initial spot price
        custom_id: Optional custom ID (default: auto-generated timestamp)
        **model_params: Additional parameters for the model (drift, volatility, etc.)

    Returns:
        storage_id: The ID used for storage (use as '{model}Stored{id}' in configs)

    Example:
        >>> storage_id = store_paths(
        ...     stock_model='RealData',
        ...     nb_stocks=50,
        ...     nb_paths=100000,
        ...     nb_dates=252,
        ...     maturity=1.0,
        ...     drift=(None,),
        ...     volatilities=(None,),
        ... )
        >>> print(f"Use 'RealDataStored{storage_id}' in your config")
    """
    from optimal_stopping.data.stock_model import STOCK_MODELS

    _ensure_storage_dir()

    # Generate or use custom ID
    storage_id = custom_id or _generate_storage_id()

    # Validate model exists
    if stock_model not in STOCK_MODELS:
        raise ValueError(f"Unknown stock model: {stock_model}. Available: {list(STOCK_MODELS.keys())}")

    print(f"📦 Storing paths for {stock_model}...")
    print(f"   ID: {storage_id}")
    print(f"   Stocks: {nb_stocks}, Paths: {nb_paths}, Dates: {nb_dates}")
    print(f"   Maturity: {maturity} years, Spot: {spot}")

    # Create model instance
    model_class = STOCK_MODELS[stock_model]
    model = model_class(
        nb_stocks=nb_stocks,
        nb_paths=nb_paths,
        nb_dates=nb_dates,
        maturity=maturity,
        spot=spot,
        name=stock_model,  # Required by base Model class
        **model_params
    )

    # Generate paths
    print(f"🔄 Generating {nb_paths:,} paths...")
    start_time = time.time()
    paths, variance_paths = model.generate_paths(nb_paths=nb_paths)
    elapsed = time.time() - start_time
    print(f"✅ Generated in {elapsed:.2f}s ({nb_paths/elapsed:.0f} paths/s)")

    # Store to HDF5
    filename = f"{stock_model}_{storage_id}.h5"
    filepath = STORAGE_DIR / filename

    print(f"💾 Saving to {filepath}...")
    with h5py.File(filepath, 'w') as f:
        # Store paths
        f.create_dataset('paths', data=paths, compression='gzip', compression_opts=4)

        # Store variance paths (None if not applicable)
        if variance_paths is not None:
            f.create_dataset('variance_paths', data=variance_paths, compression='gzip', compression_opts=4)

        # Store metadata
        metadata = {
            'stock_model': stock_model,
            'storage_id': storage_id,
            'nb_stocks': nb_stocks,
            'nb_paths': nb_paths,
            'nb_dates': nb_dates,
            'maturity': maturity,
            'spot': spot,
            'timestamp': time.time(),
            'creation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        }

        # Add model-specific parameters to metadata
        for key, value in model_params.items():
            # Convert tuples/lists to JSON-serializable format
            if isinstance(value, (tuple, list)):
                metadata[f'param_{key}'] = json.dumps(list(value))
            elif value is not None:
                metadata[f'param_{key}'] = str(value)

        # Store metadata as attributes
        for key, value in metadata.items():
            f.attrs[key] = value

    file_size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"✅ Stored {file_size_mb:.1f} MB to {filename}")
    print(f"📝 Use in config: stock_models=['{stock_model}Stored{storage_id}']")

    return storage_id


def load_paths(
    stock_model: str,
    storage_id: str,
    nb_stocks: int,
    nb_paths: int,
    nb_dates: int,
    maturity: float,
    spot: float,
    verbose: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Load stored paths from disk with validation.

    Args:
        stock_model: Model name (e.g., 'RealData', 'BlackScholes')
        storage_id: Storage ID
        nb_stocks: Number of stocks requested (must be ≤ stored)
        nb_paths: Number of paths requested (must be ≤ stored)
        nb_dates: Number of time steps (must match stored)
        maturity: Maturity in years (must match stored)
        spot: Spot price (can differ - paths will be rescaled)
        verbose: Print validation info

    Returns:
        paths: Array of shape (nb_paths, nb_stocks, nb_dates+1)
        variance_paths: Array of shape (nb_paths, nb_stocks, nb_dates+1) or None
        metadata: Dictionary of stored metadata

    Raises:
        FileNotFoundError: If storage file doesn't exist
        ValueError: If parameters are incompatible with stored data
    """
    filename = f"{stock_model}_{storage_id}.h5"
    filepath = STORAGE_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Stored paths not found: {filepath}\n"
            f"Available storage files: {list_stored_paths()}"
        )

    if verbose:
        print(f"📂 Loading stored paths from {filename}...")

    with h5py.File(filepath, 'r') as f:
        # Load metadata
        metadata = dict(f.attrs)

        stored_stocks = metadata['nb_stocks']
        stored_paths = metadata['nb_paths']
        stored_dates = metadata['nb_dates']
        stored_maturity = metadata['maturity']
        stored_spot = metadata['spot']

        if verbose:
            print(f"   Stored: {stored_stocks} stocks, {stored_paths:,} paths, {stored_dates} dates")
            print(f"   Requested: {nb_stocks} stocks, {nb_paths:,} paths, {nb_dates} dates")

        # Validate parameters
        errors = []
        warnings_list = []

        # MUST match: nb_dates, maturity
        if nb_dates != stored_dates:
            errors.append(f"nb_dates mismatch: requested {nb_dates}, stored {stored_dates}")
        if not np.isclose(maturity, stored_maturity, rtol=1e-6):
            errors.append(f"maturity mismatch: requested {maturity}, stored {stored_maturity}")

        # CAN be subset: nb_stocks, nb_paths
        if nb_stocks > stored_stocks:
            errors.append(f"nb_stocks too large: requested {nb_stocks}, only {stored_stocks} stored")
        if nb_paths > stored_paths:
            errors.append(f"nb_paths too large: requested {nb_paths:,}, only {stored_paths:,} stored")

        # CAN rescale: spot
        if not np.isclose(spot, stored_spot, rtol=1e-6):
            scale_factor = spot / stored_spot
            warnings_list.append(
                f"spot mismatch: requested {spot}, stored {stored_spot}. "
                f"Will rescale paths by {scale_factor:.4f}"
            )

        if errors:
            raise ValueError(
                f"Incompatible parameters for stored paths:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        if verbose and warnings_list:
            for w in warnings_list:
                warnings.warn(w)

        # Load paths (subset if needed)
        stored_paths_data = f['paths']
        paths = stored_paths_data[:nb_paths, :nb_stocks, :]

        # Load variance paths if present
        variance_paths = None
        if 'variance_paths' in f:
            stored_var_data = f['variance_paths']
            variance_paths = stored_var_data[:nb_paths, :nb_stocks, :]

        # Rescale if spot differs
        if not np.isclose(spot, stored_spot, rtol=1e-6):
            scale_factor = spot / stored_spot
            paths = paths * scale_factor
            if verbose:
                print(f"   📊 Rescaled paths: {stored_spot} → {spot} (×{scale_factor:.4f})")

        if verbose:
            print(f"✅ Loaded {paths.shape[0]:,} paths × {paths.shape[1]} stocks × {paths.shape[2]} dates")

    return paths, variance_paths, metadata


def list_stored_paths(verbose: bool = True) -> Dict[str, Dict[str, Any]]:
    """List all stored path files with their metadata.

    Args:
        verbose: Print formatted list to console

    Returns:
        Dictionary mapping storage_key to metadata
        storage_key format: '{model}Stored{id}'
    """
    _ensure_storage_dir()

    stored = {}

    for filepath in sorted(STORAGE_DIR.glob('*.h5')):
        try:
            with h5py.File(filepath, 'r') as f:
                metadata = dict(f.attrs)
                stock_model = metadata.get('stock_model', 'Unknown')
                storage_id = metadata.get('storage_id', 'Unknown')
                storage_key = f"{stock_model}Stored{storage_id}"

                stored[storage_key] = {
                    'filepath': str(filepath),
                    'file_size_mb': filepath.stat().st_size / (1024 * 1024),
                    **metadata
                }
        except Exception as e:
            if verbose:
                print(f"⚠️  Error reading {filepath.name}: {e}")

    if verbose:
        if not stored:
            print("No stored paths found.")
        else:
            print(f"\n📚 Found {len(stored)} stored path file(s):\n")
            for storage_key, info in stored.items():
                print(f"  {storage_key}")
                print(f"    Model: {info['stock_model']}")
                print(f"    Stocks: {info['nb_stocks']}, Paths: {info['nb_paths']:,}, Dates: {info['nb_dates']}")
                print(f"    Maturity: {info['maturity']} years, Spot: {info['spot']}")
                print(f"    Size: {info['file_size_mb']:.1f} MB")
                print(f"    Created: {info.get('creation_date', 'Unknown')}")
                print()

    return stored


def delete_stored_paths(storage_key: str) -> bool:
    """Delete a stored path file.

    Args:
        storage_key: Storage key in format '{model}Stored{id}'

    Returns:
        True if deleted, False if not found

    Example:
        >>> delete_stored_paths('RealDataStored1700000000123')
    """
    # Parse storage key
    if 'Stored' not in storage_key:
        raise ValueError(f"Invalid storage_key format: {storage_key}. Expected '{'{model}'}Stored{'{id}'}'")

    model, storage_id = storage_key.split('Stored')
    filename = f"{model}_{storage_id}.h5"
    filepath = STORAGE_DIR / filename

    if not filepath.exists():
        print(f"❌ Not found: {storage_key}")
        return False

    filepath.unlink()
    print(f"🗑️  Deleted: {storage_key} ({filepath.name})")
    return True
