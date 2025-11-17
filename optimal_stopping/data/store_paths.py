"""CLI tool for storing stock model paths.

Usage:
    # Store RealData paths with empirical drift/volatility
    python -m optimal_stopping.data.store_paths \
        --stock_model=RealData \
        --nb_stocks=50 \
        --nb_paths=100000 \
        --nb_dates=252 \
        --maturity=1.0 \
        --drift=None \
        --volatilities=None

    # Store BlackScholes paths
    python -m optimal_stopping.data.store_paths \
        --stock_model=BlackScholes \
        --nb_stocks=10 \
        --nb_paths=50000 \
        --nb_dates=100 \
        --maturity=0.5 \
        --drift=0.05 \
        --volatility=0.2

    # List all stored paths
    python -m optimal_stopping.data.store_paths --list

    # Delete stored paths
    python -m optimal_stopping.data.store_paths --delete=RealDataStored1700000000123
"""

import argparse
import sys
from optimal_stopping.data.path_storage import store_paths, list_stored_paths, delete_stored_paths


def _parse_value(value_str: str):
    """Parse command-line value to appropriate Python type.

    Args:
        value_str: String value from command line

    Returns:
        Parsed value (None, float, int, or str)
    """
    if value_str.lower() in ('none', 'null'):
        return None
    try:
        # Try int first
        if '.' not in value_str:
            return int(value_str)
        # Then float
        return float(value_str)
    except ValueError:
        # Return as string
        return value_str


def main():
    parser = argparse.ArgumentParser(
        description='Store stock model paths for later reuse',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # List/delete operations
    parser.add_argument('--list', action='store_true',
                        help='List all stored paths and exit')
    parser.add_argument('--delete', type=str, metavar='STORAGE_KEY',
                        help='Delete stored paths (e.g., RealDataStored123) and exit')

    # Required parameters for storage
    parser.add_argument('--stock_model', type=str,
                        help='Stock model name (e.g., RealData, BlackScholes)')
    parser.add_argument('--nb_stocks', type=int,
                        help='Number of stocks')
    parser.add_argument('--nb_paths', type=int,
                        help='Number of paths to generate')
    parser.add_argument('--nb_dates', type=int,
                        help='Number of time steps')
    parser.add_argument('--maturity', type=float,
                        help='Maturity in years')

    # Optional parameters
    parser.add_argument('--spot', type=float, default=100.0,
                        help='Initial spot price (default: 100)')
    parser.add_argument('--custom_id', type=str,
                        help='Custom storage ID (default: auto-generated timestamp)')

    # Model-specific parameters (catch-all)
    parser.add_argument('--drift', type=str,
                        help='Drift parameter (use "None" for empirical)')
    parser.add_argument('--volatility', type=str,
                        help='Volatility parameter (use "None" for empirical)')
    parser.add_argument('--volatilities', type=str,
                        help='Volatilities parameter (use "None" for empirical)')
    parser.add_argument('--dividend', type=float, default=0.0,
                        help='Dividend rate (default: 0)')
    parser.add_argument('--correlation', type=float,
                        help='Correlation between stocks')

    # RealData-specific parameters
    parser.add_argument('--tickers', type=str, nargs='+',
                        help='Stock tickers for RealData (default: top S&P 500)')
    parser.add_argument('--start_date', type=str,
                        help='Start date for RealData (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str,
                        help='End date for RealData (YYYY-MM-DD)')
    parser.add_argument('--exclude_crisis', action='store_true',
                        help='Exclude crisis periods for RealData')

    args = parser.parse_args()

    # Handle list operation
    if args.list:
        list_stored_paths(verbose=True)
        return 0

    # Handle delete operation
    if args.delete:
        success = delete_stored_paths(args.delete)
        return 0 if success else 1

    # Validate required arguments for storage
    required = ['stock_model', 'nb_stocks', 'nb_paths', 'nb_dates', 'maturity']
    missing = [arg for arg in required if getattr(args, arg) is None]
    if missing:
        parser.error(f"Missing required arguments for storage: {', '.join('--' + m for m in missing)}")

    # Build model parameters
    model_params = {
        'dividend': args.dividend,
    }

    # Add optional parameters if provided
    # Note: Pass scalar values directly (not tuples) since we're instantiating
    # the model directly, not going through config/itertools.product
    if args.drift is not None:
        drift_val = _parse_value(args.drift)
        model_params['drift'] = drift_val

    if args.volatility is not None:
        vol_val = _parse_value(args.volatility)
        model_params['volatility'] = vol_val

    if args.volatilities is not None:
        vol_val = _parse_value(args.volatilities)
        model_params['volatilities'] = vol_val

    if args.correlation is not None:
        model_params['correlation'] = args.correlation

    # RealData-specific
    if args.tickers:
        model_params['tickers'] = args.tickers
    if args.start_date:
        model_params['start_date'] = args.start_date
    if args.end_date:
        model_params['end_date'] = args.end_date
    if args.exclude_crisis:
        model_params['exclude_crisis'] = True

    # Store paths
    try:
        storage_id = store_paths(
            stock_model=args.stock_model,
            nb_stocks=args.nb_stocks,
            nb_paths=args.nb_paths,
            nb_dates=args.nb_dates,
            maturity=args.maturity,
            spot=args.spot,
            custom_id=args.custom_id,
            **model_params
        )
        print(f"\n✅ Success! Storage ID: {storage_id}")
        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
