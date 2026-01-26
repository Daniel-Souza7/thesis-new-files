#!/usr/bin/env python3
"""Debug script to check registered stored models."""

from optimal_stopping.data.stock_model import STOCK_MODELS

print("🔍 Registered stock models:")
print("=" * 60)

# Show all registered models
for model_name in sorted(STOCK_MODELS.keys()):
    print(f"  {model_name}")

print("\n🔍 Looking for 'Stored' models:")
print("=" * 60)

stored_models = [m for m in STOCK_MODELS.keys() if 'Stored' in m]
if stored_models:
    for model in sorted(stored_models):
        print(f"  ✓ {model}")
else:
    print("  ❌ No stored models found!")

print("\n💡 To use a stored model, it must appear in the list above.")
print("   If your model isn't listed, check:")
print("   1. File is in optimal_stopping/data/stored_paths/")
print("   2. Filename format: {model}_{id}.h5")
print("   3. Metadata in file matches (use update_stored_paths_metadata.py --inspect)")
