"""CLI tool for storing stock model paths.

Usage:
    # Store RealData paths
    python -m optimal_stopping.data.store_paths \
        --stock_model=RealData \
        --nb_stocks=50 \
        --nb_paths=100000 \
        --nb_dates=252 \
        --maturity=1.0 \
        --drift=None \
        --volatilities=None

    # Store RoughHeston paths (New Example)
    python -m optimal_stopping.data.store_paths \
        --stock_model=RoughHeston \
        --nb_stocks=10 \
        --nb_paths=10000 \
        --nb_dates=100 \
        --maturity=0.5 \
        --spot=100 \
        --drift=0.05 \
        --volatility=0.2 \
        --mean=0.04 \
        --speed=2.0 \
        --correlation=-0.7 \
        --hurst=0.1

    # List all stored paths
    python -m optimal_stopping.data.store_paths --list

    # Delete stored paths
    python -m optimal_stopping.data.store_paths --delete=RealDataStored1700000000123
"""

import argparse
import sys
from optimal_stopping.data.path_storage import store_paths, list_stored_paths, delete_stored_paths

# Telegram setup
try:
    from telegram_notifications import send_bot_message as SBM
    TELEGRAM_ENABLED = True
except:
    TELEGRAM_ENABLED = False

    class SBM:
        @staticmethod
        def send_notification(*args, **kwargs):
            pass


def _parse_value(value_str: str):
    """Parse command-line value to appropriate Python type."""
    if value_str.lower() in ('none', 'null'):
        return None
    try:
        if '.' not in value_str:
            return int(value_str)
        return float(value_str)
    except ValueError:
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
                        help='Stock model name (e.g., RealData, BlackScholes, RoughHeston)')
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

    # Model-specific parameters
    parser.add_argument('--drift', type=str,
                        help='Drift parameter (use "None" for empirical)')
    parser.add_argument('--volatility', type=str,
                        help='Volatility/Initial Vol parameter')
    parser.add_argument('--volatilities', type=str,
                        help='Volatilities parameter')
    parser.add_argument('--dividend', type=float, default=0.0,
                        help='Dividend rate (default: 0)')
    parser.add_argument('--correlation', type=float,
                        help='Correlation between stocks (or between asset and vol for Heston)')

    # Heston / Rough Heston specific parameters
    parser.add_argument('--mean', type=float,
                        help='Long-run average variance (theta) for Heston/RoughHeston')
    parser.add_argument('--speed', type=float,
                        help='Mean reversion speed (kappa) for Heston/RoughHeston')
    parser.add_argument('--hurst', type=float,
                        help='Hurst parameter (H) for Rough models')

    # RealData-specific parameters
    parser.add_argument('--tickers', type=str, nargs='+',
                        help='Stock tickers for RealData (default: top S&P 500)')
    parser.add_argument('--start_date', type=str,
                        help='Start date for RealData (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str,
                        help='End date for RealData (YYYY-MM-DD)')
    parser.add_argument('--exclude_crisis', action='store_true',
                        help='Exclude crisis periods for RealData')

    # Telegram parameters
    parser.add_argument('--telegram_token', type=str,
                        default="8239319342:AAGIIcoDaxJ1uauHbWfdByF4yzNYdQ5jpiA",
                        help='Telegram bot token')
    parser.add_argument('--telegram_chat_id', type=str,
                        default="798647521",
                        help='Telegram chat ID')
    parser.add_argument('--send_telegram', action='store_true', default=True,
                        help='Whether to send notifications via Telegram (default: True)')
    parser.add_argument('--no_telegram', action='store_true',
                        help='Disable Telegram notifications')

    args = parser.parse_args()

    if args.no_telegram:
        args.send_telegram = False

    if args.list:
        list_stored_paths(verbose=True)
        return 0

    if args.delete:
        success = delete_stored_paths(args.delete)
        return 0 if success else 1

    required = ['stock_model', 'nb_stocks', 'nb_paths', 'nb_dates', 'maturity']
    missing = [arg for arg in required if getattr(args, arg) is None]
    if missing:
        parser.error(f"Missing required arguments for storage: {', '.join('--' + m for m in missing)}")

    # Build model parameters
    model_params = {
        'dividend': args.dividend
    }

    # Add optional parameters if provided
    if args.drift is not None:
        model_params['drift'] = _parse_value(args.drift)

    if args.volatility is not None:
        model_params['volatility'] = _parse_value(args.volatility)

    if args.volatilities is not None:
        model_params['volatilities'] = _parse_value(args.volatilities)

    if args.correlation is not None:
        model_params['correlation'] = args.correlation

    # Add Heston/RoughHeston specific params
    if args.mean is not None:
        model_params['mean'] = args.mean
    if args.speed is not None:
        model_params['speed'] = args.speed
    if args.hurst is not None:
        model_params['hurst'] = args.hurst

    # RealData-specific
    if args.tickers:
        model_params['tickers'] = args.tickers
    if args.start_date:
        model_params['start_date'] = args.start_date
    if args.end_date:
        model_params['end_date'] = args.end_date
    if args.exclude_crisis:
        model_params['exclude_crisis'] = True

    print(f"⚡ Doubling requested paths ({args.nb_paths:,}) to {args.nb_paths * 2:,} "
          f"to ensure distinct Training and Evaluation sets.")
    args.nb_paths = args.nb_paths * 2

    # Send start notification
    if TELEGRAM_ENABLED and args.send_telegram:
        try:
            SBM.send_notification(
                token=args.telegram_token,
                text=f'💾 Starting path storage...\n\n'
                     f'Model: {args.stock_model}\n'
                     f'Stocks: {args.nb_stocks}, Paths: {args.nb_paths:,}\n'
                     f'Dates: {args.nb_dates}, Maturity: {args.maturity}y',
                chat_id=args.telegram_chat_id
            )
        except Exception as e:
            print(f"⚠️  Telegram notification failed: {e}")

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

        if TELEGRAM_ENABLED and args.send_telegram:
            try:
                SBM.send_notification(
                    token=args.telegram_token,
                    text=f'✅ Path storage complete!\n\n'
                         f'Storage ID: {storage_id}\n'
                         f'Model: {args.stock_model}Stored{storage_id}\n\n'
                         f'Use in config:\n'
                         f"stock_models=['{args.stock_model}Stored{storage_id}']",
                    chat_id=args.telegram_chat_id
                )
            except Exception as e:
                print(f"⚠️  Telegram notification failed: {e}")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

        if TELEGRAM_ENABLED and args.send_telegram:
            try:
                SBM.send_notification(
                    token=args.telegram_token,
                    text=f'❌ Path storage failed!\n\n'
                         f'Model: {args.stock_model}\n'
                         f'Error: {str(e)[:200]}',
                    chat_id=args.telegram_chat_id
                )
            except:
                pass

        return 1


if __name__ == '__main__':
    sys.exit(main())