#!/usr/bin/env python3
"""Debug script to verify pricing calculations."""

import numpy as np
from optimal_stopping.data.stock_model import STOCK_MODELS

def debug_model(model_name, **config):
    """Debug a model's path generation and pricing.

    Args:
        model_name: Name of the stock model
        **config: Model configuration parameters
    """
    print(f"\n{'='*80}")
    print(f"🔍 Debugging model: {model_name}")
    print(f"{'='*80}")

    # Check if model exists
    if model_name not in STOCK_MODELS:
        print(f"❌ Model '{model_name}' not found in STOCK_MODELS!")
        print(f"Available models: {list(STOCK_MODELS.keys())}")
        return

    # Create model instance
    print(f"\n📦 Creating model instance...")
    print(f"   Config: {config}")

    try:
        model = STOCK_MODELS[model_name](**config)
        print(f"✓ Model created: {model}")
        print(f"   Type: {type(model).__name__}")
        print(f"   Name: {model.name}")
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Generate paths
    print(f"\n🎲 Generating paths...")
    nb_test_paths = min(1000, config.get('nb_paths', 1000))

    try:
        paths, var_paths = model.generate_paths(nb_paths=nb_test_paths)
        print(f"✓ Generated paths shape: {paths.shape}")
        print(f"   Expected: ({nb_test_paths}, {config['nb_stocks']}, {config['nb_dates']+1})")

        # Check paths statistics
        print(f"\n📊 Path statistics:")
        print(f"   Initial price (S0): {paths[0, 0, 0]:.2f} (should be {config['spot']:.2f})")
        print(f"   Final prices (ST):")
        print(f"      Mean: {paths[:, 0, -1].mean():.2f}")
        print(f"      Min:  {paths[:, 0, -1].min():.2f}")
        print(f"      Max:  {paths[:, 0, -1].max():.2f}")
        print(f"      Std:  {paths[:, 0, -1].std():.2f}")

        # Check if paths are reasonable
        if np.any(paths <= 0):
            print(f"   ⚠️  WARNING: Some paths have non-positive values!")
        if np.any(np.isnan(paths)):
            print(f"   ⚠️  WARNING: Some paths contain NaN values!")
        if np.any(np.isinf(paths)):
            print(f"   ⚠️  WARNING: Some paths contain infinite values!")

    except Exception as e:
        print(f"❌ Failed to generate paths: {e}")
        import traceback
        traceback.print_exc()
        return

    # Calculate European option price (basket call)
    print(f"\n💰 European option pricing:")
    K = config.get('strike', 100)
    r = config.get('drift', 0.05)  # risk-free rate
    T = config.get('maturity', 1.0)

    # Average across stocks (basket)
    basket_prices = paths.mean(axis=1)  # Average over stocks
    final_basket = basket_prices[:, -1]  # Final time

    # Payoff at maturity
    payoffs = np.maximum(final_basket - K, 0)

    # Discount to present value
    discount_factor = np.exp(-r * T)
    eop_price = discount_factor * payoffs.mean()

    print(f"   Strike (K): {K:.2f}")
    print(f"   Risk-free rate (r): {r:.4f}")
    print(f"   Maturity (T): {T:.2f} years")
    print(f"   Discount factor: {discount_factor:.4f}")
    print(f"   Average final basket price: {final_basket.mean():.2f}")
    print(f"   Average payoff (undiscounted): {payoffs.mean():.2f}")
    print(f"   European option price (EOP): {eop_price:.4f}")

    # Sanity checks
    print(f"\n🔬 Sanity checks:")
    if eop_price < 0:
        print(f"   ❌ EOP is negative! This is impossible.")
    elif eop_price > config['spot']:
        print(f"   ⚠️  EOP > spot price. This can happen for high volatility/drift.")
    else:
        print(f"   ✓ EOP looks reasonable.")

    print(f"\n💡 For American option:")
    print(f"   Value should be ≥ {eop_price:.4f} (EOP)")
    print(f"   For basket call with positive drift, optimal = hold to maturity")
    print(f"   So American price should ≈ {eop_price:.4f}")
    print(f"   If you're getting values > {eop_price:.4f} by more than ~1%, something is wrong!")

    return model, paths, eop_price


if __name__ == '__main__':
    # Example usage - modify these parameters to match your setup
    config = {
        'nb_stocks': 1,
        'nb_paths': 10000,
        'nb_dates': 100,
        'maturity': 1.0,
        'spot': 100,
        'drift': 0.05,
        'volatility': 0.2,
        'strike': 100,
    }

    # Test with BlackScholes first (baseline)
    print("Testing BlackScholes model (baseline):")
    debug_model('BlackScholes', **config)

    # Then test with your stored model
    # Uncomment and modify the model name to match your file:
    # print("\n\nTesting stored model:")
    # debug_model('BSStored1', **config)  # or whatever your model is called
