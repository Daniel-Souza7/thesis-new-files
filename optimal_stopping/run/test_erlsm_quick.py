#!/usr/bin/env python3
"""
Quick test script to validate and optimize ERLSM performance.

This script runs multiple iterations with different hyperparameters to find
the optimal configuration that consistently beats RLSM/RFQI.
"""

import sys
import numpy as np
import time
from optimal_stopping.data.stock_model import BlackScholes
from optimal_stopping.payoffs.standard import MaxCall, BasketCall
from optimal_stopping.algorithms.standard.rlsm import RLSM
from optimal_stopping.algorithms.standard.rfqi import RFQI
from optimal_stopping.algorithms.standard.erlsm import ERLSM


def test_algorithm(algo_class, model, payoff, name, **kwargs):
    """Test an algorithm and return price and timing."""
    try:
        algo = algo_class(model, payoff, **kwargs)
        start = time.time()
        price, gen_time = algo.price(train_eval_split=2)
        total_time = time.time() - start
        return {
            'name': name,
            'price': price,
            'comp_time': total_time - gen_time,
            'total_time': total_time,
            'success': True
        }
    except Exception as e:
        print(f"❌ {name} failed: {e}")
        return {'name': name, 'success': False, 'error': str(e)}


def run_comparison(nb_stocks=5, nb_paths=10000, nb_dates=52):
    """Run comparison between RLSM, RFQI, and ERLSM with different configs."""

    print(f"\n{'='*80}")
    print(f"Testing with {nb_stocks} stocks, {nb_paths} paths, {nb_dates} dates")
    print(f"{'='*80}\n")

    # Create model
    model = BlackScholes(
        drift=0.05,
        volatility=0.2,
        nb_stocks=nb_stocks,
        nb_paths=nb_paths,
        nb_dates=nb_dates,
        spot=100,
        maturity=0.25
    )

    # Test payoffs
    payoffs = [
        ('MaxCall', MaxCall(strike=100)),
        ('BasketCall', BasketCall(strike=100))
    ]

    for payoff_name, payoff in payoffs:
        print(f"\n📊 Payoff: {payoff_name}")
        print("-" * 60)

        results = []

        # Baseline: RLSM
        result = test_algorithm(
            RLSM, model, payoff, "RLSM",
            hidden_size=100,
            factors=(1., 1., 1.),
            train_ITM_only=True,
            use_payoff_as_input=True
        )
        results.append(result)
        if result['success']:
            print(f"  RLSM:  Price={result['price']:.4f}, Time={result['comp_time']:.2f}s")

        # Baseline: RFQI
        result = test_algorithm(
            RFQI, model, payoff, "RFQI",
            hidden_size=100,
            factors=(1., 1., 1.),
            train_ITM_only=True,
            use_payoff_as_input=True,
            nb_epochs=30
        )
        results.append(result)
        if result['success']:
            print(f"  RFQI:  Price={result['price']:.4f}, Time={result['comp_time']:.2f}s")

        # ERLSM variants to test
        erlsm_configs = [
            # (name, ensemble_size, poly_degree, hidden_size, bootstrap_ratio)
            ("ERLSM-3x2", 3, 2, 100, 0.8),
            ("ERLSM-5x2", 5, 2, 100, 0.8),
            ("ERLSM-5x2-large", 5, 2, 150, 0.8),
            ("ERLSM-7x2", 7, 2, 100, 0.8),
            ("ERLSM-5x1", 5, 1, 100, 0.8),  # No polynomial expansion
        ]

        for name, ens_size, poly_deg, hidden, boot_ratio in erlsm_configs:
            result = test_algorithm(
                ERLSM, model, payoff, name,
                hidden_size=hidden,
                factors=(1., 1., 1.),
                train_ITM_only=True,
                use_payoff_as_input=True,
                ensemble_size=ens_size,
                poly_degree=poly_deg,
                bootstrap_ratio=boot_ratio
            )
            results.append(result)
            if result['success']:
                print(f"  {name}: Price={result['price']:.4f}, Time={result['comp_time']:.2f}s")

        # Analysis
        print("\n  📈 Analysis:")
        successful = [r for r in results if r['success']]
        if len(successful) >= 2:
            baseline_price = successful[0]['price']  # RLSM
            for r in successful[1:]:
                diff = r['price'] - baseline_price
                pct = (diff / baseline_price) * 100
                slowdown = r['comp_time'] / successful[0]['comp_time']
                symbol = "✅" if diff > 0 else "❌"
                print(f"    {symbol} {r['name']}: {diff:+.4f} ({pct:+.2f}%), {slowdown:.1f}x slower")


def main():
    """Run optimization tests."""
    print("="*80)
    print("ERLSM OPTIMIZATION TEST")
    print("="*80)

    # Test on different problem sizes
    test_configs = [
        (5, 10000, 52),   # 5 stocks, moderate complexity
        (10, 10000, 52),  # 10 stocks
        (20, 8000, 52),   # 20 stocks, fewer paths for speed
    ]

    for nb_stocks, nb_paths, nb_dates in test_configs:
        run_comparison(nb_stocks, nb_paths, nb_dates)

    print("\n" + "="*80)
    print("OPTIMIZATION TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
