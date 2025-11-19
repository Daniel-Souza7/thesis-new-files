"""
Create convergence plots showing how price varies with a parameter.

This script visualizes algorithm convergence by plotting:
- Mean price vs varying parameter (e.g., nb_paths, hidden_size)
- Error bars showing standard deviation
- Multiple algorithms in different colors
- Auto-scaled axes (log or linear)

Usage:
    python -m optimal_stopping.run.plot_convergence --configs=plot_convergence1
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from optimal_stopping.run import configs
from optimal_stopping.utilities import read_data

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


# Mapping from config parameter names to CSV column names and display names
PARAM_MAPPING = {
    'nb_paths': ('nb_paths', 'Number of Paths'),
    'nb_dates': ('nb_dates', 'Number of Time Steps'),
    'nb_stocks': ('nb_stocks', 'Number of Stocks'),
    'hidden_size': ('hidden_size', 'Hidden Layer Size'),
    'nb_epochs': ('nb_epochs', 'Number of Epochs'),
    'volatilities': ('volatility', 'Volatility'),
    'drift': ('drift', 'Drift'),
    'strikes': ('strike', 'Strike'),
    'spots': ('spot', 'Spot Price'),
    'maturities': ('maturity', 'Maturity'),
    'k': ('k', 'K (Best-of-K)'),
    'barriers': ('barrier', 'Barrier Level'),
    'barriers_up': ('barriers_up', 'Upper Barrier'),
    'barriers_down': ('barriers_down', 'Lower Barrier'),
    'hurst': ('hurst', 'Hurst Parameter'),
    'ridge_coeff': ('ridge_coeff', 'Ridge Coefficient'),
}


def validate_config(config):
    """Validate that config has exactly 1 payoff and 1 varying parameter.

    Args:
        config: Config object

    Returns:
        varying_param: str - name of the varying parameter

    Raises:
        ValueError: If validation fails
    """
    # Check exactly 1 payoff
    if len(config.payoffs) != 1:
        raise ValueError(
            f"Config must have exactly 1 payoff for convergence plot, "
            f"got {len(config.payoffs)}: {config.payoffs}"
        )

    # Find varying parameters (those with multiple values)
    varying_params = []

    for param_name in PARAM_MAPPING.keys():
        if hasattr(config, param_name):
            param_value = getattr(config, param_name)

            # Convert to list if iterable
            if isinstance(param_value, (list, tuple)):
                values = list(param_value)
            else:
                values = [param_value]

            # Check if has multiple values
            if len(values) > 1:
                varying_params.append(param_name)

    # Must have exactly 1 varying parameter
    if len(varying_params) == 0:
        raise ValueError(
            "Config must have exactly 1 parameter with multiple values. "
            "All parameters are single-valued."
        )
    elif len(varying_params) > 1:
        raise ValueError(
            f"Config must have exactly 1 parameter with multiple values, "
            f"got {len(varying_params)}: {varying_params}. "
            f"Please set all but one parameter to single values."
        )

    return varying_params[0]


def extract_convergence_data(config, varying_param):
    """Extract mean and std prices for each algorithm and parameter value.

    Args:
        config: Config object
        varying_param: str - name of varying parameter

    Returns:
        data_dict: dict mapping algo -> (x_values, mean_prices, std_prices)
        param_display_name: str - display name for parameter
        csv_column_name: str - CSV column name for parameter
    """
    # Read CSV data
    print(f"Reading results from CSV files...")
    df = read_data.read_csvs(config, remove_duplicates=False)

    if df.empty:
        raise ValueError("No data found matching the config filters")

    # Get CSV column name and display name
    csv_column_name, param_display_name = PARAM_MAPPING[varying_param]

    # Get unique algorithms
    algos = df.index.get_level_values('algo').unique().tolist()

    # Get parameter values
    if csv_column_name in df.index.names:
        param_values = sorted(df.index.get_level_values(csv_column_name).unique())
    else:
        raise ValueError(f"Parameter '{csv_column_name}' not found in CSV data")

    print(f"Found {len(algos)} algorithm(s): {algos}")
    print(f"Found {len(param_values)} values for {param_display_name}: {param_values}")

    # Extract data for each algorithm
    data_dict = {}

    for algo in algos:
        x_values = []
        mean_prices = []
        std_prices = []

        for param_val in param_values:
            # Filter to this algo and parameter value
            try:
                # Get all price values for this combination
                subset = df.xs((algo, param_val), level=('algo', csv_column_name))

                if 'price' in subset.columns:
                    prices = subset['price'].values
                else:
                    print(f"⚠️  Warning: 'price' column not found for {algo}, {param_val}")
                    continue

                # Calculate statistics
                if len(prices) > 0:
                    x_values.append(param_val)
                    mean_prices.append(np.mean(prices))
                    std_prices.append(np.std(prices))
                else:
                    print(f"⚠️  Warning: No prices found for {algo}, {param_val}")

            except KeyError:
                # This algo/param combination doesn't exist in data
                print(f"⚠️  Warning: No data for {algo} at {csv_column_name}={param_val}")
                continue

        if len(x_values) > 0:
            data_dict[algo] = (
                np.array(x_values),
                np.array(mean_prices),
                np.array(std_prices)
            )
            print(f"  {algo}: {len(x_values)} points")
        else:
            print(f"⚠️  Warning: No data collected for {algo}")

    if not data_dict:
        raise ValueError("No data could be extracted for any algorithm")

    return data_dict, param_display_name, csv_column_name


def should_use_log_scale(values):
    """Determine if x-axis should use log scale.

    Use log scale if values span more than 2 orders of magnitude.

    Args:
        values: array of parameter values

    Returns:
        bool: True if should use log scale
    """
    if len(values) < 2:
        return False

    min_val = np.min(values)
    max_val = np.max(values)

    # Avoid division by zero
    if min_val <= 0:
        return False

    # Check if range > 100x
    ratio = max_val / min_val
    return ratio > 100


def create_convergence_plot(config, data_dict, param_display_name,
                           csv_column_name, output_path):
    """Create and save convergence plot.

    Args:
        config: Config object
        data_dict: dict mapping algo -> (x_values, mean_prices, std_prices)
        param_display_name: str - display name for parameter
        csv_column_name: str - CSV column name
        output_path: Path - where to save plot
    """
    # Get payoff name for title
    payoff_name = config.payoffs[0]

    # Determine if we should use log scale
    # Check all x values across all algorithms
    all_x_values = np.concatenate([x for x, _, _ in data_dict.values()])
    use_log = should_use_log_scale(all_x_values)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color cycle for algorithms
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_dict)))

    # Plot each algorithm
    for idx, (algo, (x_values, mean_prices, std_prices)) in enumerate(data_dict.items()):
        color = colors[idx]

        # Plot line
        ax.plot(x_values, mean_prices,
                marker='o', markersize=8, linewidth=2,
                label=algo, color=color, alpha=0.8)

        # Add error bars (std dev)
        ax.errorbar(x_values, mean_prices, yerr=std_prices,
                   fmt='none', ecolor=color, alpha=0.3, capsize=5)

        # Add shaded region for std dev
        ax.fill_between(x_values,
                        mean_prices - std_prices,
                        mean_prices + std_prices,
                        alpha=0.1, color=color)

    # Set scale
    if use_log:
        ax.set_xscale('log')
        print(f"Using log scale for x-axis (range: {np.min(all_x_values):.0f} to {np.max(all_x_values):.0f})")
    else:
        ax.set_xscale('linear')
        print(f"Using linear scale for x-axis")

    # Labels and title
    ax.set_xlabel(param_display_name, fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Price', fontsize=14, fontweight='bold')
    ax.set_title(f'Convergence Plot: {payoff_name}', fontsize=16, fontweight='bold')

    # Legend
    ax.legend(loc='best', fontsize=12, framealpha=0.9)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Tight layout
    plt.tight_layout()

    # Save figure
    print(f"Saving plot to {output_path}...")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create convergence plot showing price vs varying parameter"
    )
    parser.add_argument(
        '--configs',
        type=str,
        required=True,
        help='Config name from configs.py (must have exactly 1 payoff and 1 varying parameter)'
    )
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

    # Handle no_telegram flag
    if args.no_telegram:
        args.send_telegram = False

    print("="*70)
    print("CONVERGENCE PLOT GENERATION")
    print("="*70)

    # Load config
    print(f"\nLoading config: {args.configs}")
    if not hasattr(configs, args.configs):
        raise ValueError(f"Config '{args.configs}' not found in configs.py")
    config = getattr(configs, args.configs)

    # Validate config
    print(f"\nValidating config...")
    varying_param = validate_config(config)
    print(f"✓ Config valid")
    print(f"  Payoff: {config.payoffs[0]}")
    print(f"  Varying parameter: {varying_param}")
    print(f"  Algorithms: {list(config.algos)}")

    # Send start notification
    if TELEGRAM_ENABLED and args.send_telegram:
        try:
            SBM.send_notification(
                token=args.telegram_token,
                text=f'📊 Starting convergence plot generation...\n\n'
                     f'Config: {args.configs}\n'
                     f'Payoff: {config.payoffs[0]}\n'
                     f'Varying: {varying_param}\n'
                     f'Algos: {len(config.algos)}',
                chat_id=args.telegram_chat_id
            )
        except Exception as e:
            print(f"⚠️  Telegram notification failed: {e}")

    # Extract data
    print(f"\nExtracting data...")
    data_dict, param_display_name, csv_column_name = extract_convergence_data(
        config, varying_param
    )

    # Create output directory
    output_dir = Path('results') / args.configs
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f'convergence_{args.configs}_{timestamp}.png'

    # Create plot
    print(f"\nCreating plot...")
    create_convergence_plot(
        config, data_dict, param_display_name,
        csv_column_name, output_path
    )

    # Print summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")

    for algo, (x_values, mean_prices, std_prices) in data_dict.items():
        print(f"\n{algo}:")
        print(f"  Parameter range: {x_values[0]} to {x_values[-1]}")
        print(f"  Price range: {np.min(mean_prices):.4f} to {np.max(mean_prices):.4f}")
        print(f"  Mean std dev: {np.mean(std_prices):.4f}")
        print(f"  Max std dev: {np.max(std_prices):.4f}")

        # Check for convergence (price change < 1% between last two points)
        if len(x_values) >= 2:
            price_change = abs(mean_prices[-1] - mean_prices[-2]) / mean_prices[-2] * 100
            if price_change < 1.0:
                print(f"  ✓ Converged (last change: {price_change:.2f}%)")
            else:
                print(f"  ⚠ Not yet converged (last change: {price_change:.2f}%)")

    print(f"\n{'='*70}")
    print(f"✓ Done! Plot saved to: {output_path}")
    print(f"{'='*70}\n")

    # Send completion notification
    if TELEGRAM_ENABLED and args.send_telegram:
        try:
            # Prepare statistics text
            stats_text = f"\n\nStatistics:\n"
            for algo, (x_values, mean_prices, std_prices) in data_dict.items():
                stats_text += f"\n{algo}:\n"
                stats_text += f"  Range: {x_values[0]} → {x_values[-1]}\n"
                stats_text += f"  Price: {np.min(mean_prices):.4f} → {np.max(mean_prices):.4f}\n"

            SBM.send_notification(
                token=args.telegram_token,
                text=f'✅ Convergence plot complete!\n\n'
                     f'Config: {args.configs}\n'
                     f'Payoff: {config.payoffs[0]}\n'
                     f'File: {output_path.name}'
                     f'{stats_text}',
                chat_id=args.telegram_chat_id,
                files=[str(output_path)]
            )
        except Exception as e:
            print(f"⚠️  Telegram notification failed: {e}")


if __name__ == '__main__':
    main()
