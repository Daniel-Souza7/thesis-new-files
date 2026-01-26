#!/usr/bin/env python3
"""Update metadata in a stored paths HDF5 file.

This script allows you to update the stock_model and storage_id metadata
in a stored paths file after renaming it.

Usage:
    python update_stored_paths_metadata.py <filepath> --model BS --id 1
"""

import h5py
import argparse
from pathlib import Path


def update_metadata(filepath: Path, model_name: str, storage_id: str):
    """Update metadata in HDF5 file.

    Args:
        filepath: Path to the HDF5 file
        model_name: New model name (e.g., 'BS', 'BlackScholes')
        storage_id: New storage ID (e.g., '1', '219309124')
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"📂 Opening {filepath.name}...")

    with h5py.File(filepath, 'r') as f:
        # Read existing metadata
        print("\n📋 Current metadata:")
        for key, value in dict(f.attrs).items():
            print(f"   {key}: {value}")

    # Update metadata
    print(f"\n✏️  Updating metadata...")
    print(f"   stock_model: {model_name}")
    print(f"   storage_id: {storage_id}")

    with h5py.File(filepath, 'r+') as f:
        f.attrs['stock_model'] = model_name
        f.attrs['storage_id'] = storage_id

    # Verify
    with h5py.File(filepath, 'r') as f:
        print("\n✅ Updated metadata:")
        for key, value in dict(f.attrs).items():
            print(f"   {key}: {value}")

    print(f"\n🎉 Done! Use in config as: stock_models=['{model_name}Stored{storage_id}']")


def inspect_file(filepath: Path):
    """Inspect HDF5 file without modifying it.

    Args:
        filepath: Path to the HDF5 file
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"📂 Inspecting {filepath.name}...\n")

    with h5py.File(filepath, 'r') as f:
        # Show datasets
        print("📊 Datasets:")
        for key in f.keys():
            dataset = f[key]
            print(f"   {key}: shape={dataset.shape}, dtype={dataset.dtype}")

        # Show metadata
        print("\n📋 Metadata:")
        metadata = dict(f.attrs)
        for key, value in metadata.items():
            print(f"   {key}: {value}")

        # Show recommended usage
        model = metadata.get('stock_model', 'Unknown')
        storage_id = metadata.get('storage_id', 'Unknown')
        nb_stocks = metadata.get('nb_stocks', '?')
        nb_paths = metadata.get('nb_paths', '?')
        nb_dates = metadata.get('nb_dates', '?')
        maturity = metadata.get('maturity', '?')
        spot = metadata.get('spot', '?')

        print(f"\n💡 Recommended config usage:")
        print(f"   stock_models=['{model}Stored{storage_id}']")
        print(f"   nb_stocks={nb_stocks}  # or less")
        print(f"   nb_paths={nb_paths}  # or less")
        print(f"   nb_dates={nb_dates}  # must match")
        print(f"   maturity={maturity}  # must match")
        print(f"   spot={spot}  # or different (will rescale)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Update or inspect stored paths HDF5 file metadata'
    )
    parser.add_argument(
        'filepath',
        type=Path,
        help='Path to HDF5 file'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='New model name (e.g., BS, BlackScholes)'
    )
    parser.add_argument(
        '--id',
        type=str,
        help='New storage ID (e.g., 1, 219309124)'
    )
    parser.add_argument(
        '--inspect',
        action='store_true',
        help='Only inspect file without modifying'
    )

    args = parser.parse_args()

    if args.inspect:
        inspect_file(args.filepath)
    elif args.model and args.id:
        update_metadata(args.filepath, args.model, args.id)
    else:
        print("Error: Must provide --inspect OR both --model and --id")
        parser.print_help()
