"""Test script for RealData integration.

This script tests the full integration of RealData model with the pricing engine.
"""

import sys
import json

# Add parent directory to path
sys.path.insert(0, '/home/user/thesis-new-files')

from pricing_engine import PricingEngine


def test_black_scholes():
    """Test Black-Scholes pricing (baseline)."""
    print("\n" + "="*60)
    print("TEST 1: Black-Scholes Call Option")
    print("="*60)

    engine = PricingEngine()

    params = {
        'model_type': 'BlackScholes',
        'payoff_type': 'Call',
        'algorithm': 'RLSM',
        'spot': 100,
        'strike': 100,
        'drift': 0.05,
        'volatility': 0.2,
        'rate': 0.03,
        'maturity': 1.0,
        'nb_stocks': 1,
        'nb_dates': 50,
        'nb_paths': 5000,
    }

    result = engine.price_option(params)

    if result['success']:
        print(f"✓ Price: ${result['price']:.4f}")
        print(f"✓ Computation time: {result['computation_time']:.3f}s")
        print(f"✓ Algorithm: {result['algorithm']}")
        return True
    else:
        print(f"✗ Error: {result['error']}")
        return False


def test_realdata_empirical():
    """Test RealData with empirical drift/volatility."""
    print("\n" + "="*60)
    print("TEST 2: RealData Call Option (Empirical)")
    print("="*60)

    engine = PricingEngine()

    params = {
        'model_type': 'RealData',
        'payoff_type': 'Call',
        'algorithm': 'RLSM',
        'tickers': ['AAPL'],
        'start_date': '2020-01-01',
        'end_date': '2024-01-01',
        'drift_override': None,  # Use empirical
        'volatility_override': None,  # Use empirical
        'spot': 100,
        'strike': 100,
        'rate': 0.03,
        'maturity': 1.0,
        'nb_stocks': 1,
        'nb_dates': 50,
        'nb_paths': 5000,
    }

    result = engine.price_option(params)

    if result['success']:
        print(f"✓ Price: ${result['price']:.4f}")
        print(f"✓ Computation time: {result['computation_time']:.3f}s")
        print(f"✓ Empirical drift: {result['model_info']['empirical_drift']:.4f}")
        print(f"✓ Empirical volatility: {result['model_info']['empirical_volatility']:.4f}")
        print(f"✓ Block length: {result['model_info']['block_length']} days")
        print(f"✓ Data days: {result['model_info']['data_days']}")
        return True
    else:
        print(f"✗ Error: {result['error']}")
        return False


def test_realdata_override():
    """Test RealData with drift/volatility override."""
    print("\n" + "="*60)
    print("TEST 3: RealData Call Option (Override)")
    print("="*60)

    engine = PricingEngine()

    params = {
        'model_type': 'RealData',
        'payoff_type': 'Call',
        'algorithm': 'RLSM',
        'tickers': ['AAPL'],
        'start_date': '2020-01-01',
        'end_date': '2024-01-01',
        'drift_override': 0.05,  # Override to 5%
        'volatility_override': 0.2,  # Override to 20%
        'spot': 100,
        'strike': 100,
        'rate': 0.03,
        'maturity': 1.0,
        'nb_stocks': 1,
        'nb_dates': 50,
        'nb_paths': 5000,
    }

    result = engine.price_option(params)

    if result['success']:
        print(f"✓ Price: ${result['price']:.4f}")
        print(f"✓ Computation time: {result['computation_time']:.3f}s")
        print(f"✓ Empirical drift: {result['model_info']['empirical_drift']:.4f}")
        print(f"✓ Drift override: {result['model_info']['drift_override']:.4f}")
        print(f"✓ Empirical volatility: {result['model_info']['empirical_volatility']:.4f}")
        print(f"✓ Volatility override: {result['model_info']['volatility_override']:.4f}")
        return True
    else:
        print(f"✗ Error: {result['error']}")
        return False


def test_realdata_multi_stock():
    """Test RealData with multiple stocks."""
    print("\n" + "="*60)
    print("TEST 4: RealData Basket Call (Multi-Stock)")
    print("="*60)

    engine = PricingEngine()

    params = {
        'model_type': 'RealData',
        'payoff_type': 'BasketCall',
        'algorithm': 'RLSM',
        'tickers': ['AAPL', 'MSFT'],
        'start_date': '2020-01-01',
        'end_date': '2024-01-01',
        'drift_override': None,
        'volatility_override': None,
        'spot': 100,
        'strike': 100,
        'rate': 0.03,
        'maturity': 1.0,
        'nb_stocks': 2,
        'nb_dates': 50,
        'nb_paths': 5000,
    }

    result = engine.price_option(params)

    if result['success']:
        print(f"✓ Price: ${result['price']:.4f}")
        print(f"✓ Computation time: {result['computation_time']:.3f}s")
        print(f"✓ Tickers: {', '.join(result['model_info']['tickers'])}")
        print(f"✓ Empirical drift: {result['model_info']['empirical_drift']:.4f}")
        print(f"✓ Empirical volatility: {result['model_info']['empirical_volatility']:.4f}")
        return True
    else:
        print(f"✗ Error: {result['error']}")
        return False


def test_stock_info():
    """Test stock information retrieval."""
    print("\n" + "="*60)
    print("TEST 5: Stock Information Retrieval")
    print("="*60)

    engine = PricingEngine()

    result = engine.get_stock_info(
        tickers=['AAPL', 'MSFT'],
        start_date='2020-01-01',
        end_date='2024-01-01'
    )

    if result['success']:
        print(f"✓ Tickers loaded: {', '.join(result['tickers'])}")
        print(f"✓ Data days: {result['data_days']}")
        print(f"✓ Overall drift: {result['overall_drift']:.4f}")
        print(f"✓ Overall volatility: {result['overall_volatility']:.4f}")
        print(f"✓ Block length: {result['block_length']} days")

        print("\nPer-ticker statistics:")
        for stat in result['stock_statistics']:
            print(f"  {stat['ticker']}: drift={stat['empirical_drift_annual']:.4f}, "
                  f"vol={stat['empirical_volatility_annual']:.4f}")
        return True
    else:
        print(f"✗ Error: {result['error']}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("RealData Integration Test Suite")
    print("="*60)

    tests = [
        test_black_scholes,
        test_realdata_empirical,
        test_realdata_override,
        test_realdata_multi_stock,
        test_stock_info,
    ]

    results = []
    for test_func in tests:
        try:
            success = test_func()
            results.append((test_func.__name__, success))
        except Exception as e:
            print(f"\n✗ Test failed with exception: {e}")
            results.append((test_func.__name__, False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
