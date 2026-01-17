"""
Create convergence plots showing how price varies with a parameter.

This script visualizes algorithm convergence by:
- Running algorithms for each value of the varying parameter
- Computing mean price and standard deviation over nb_runs
- Plotting multiple algorithms in different colors
- Auto-scaled axes (log or linear)

Usage:
    python -m optimal_stopping.run.plot_convergence --configs=plot_convergence1
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
import copy

import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as st  # <--- Added for t-distribution

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from optimal_stopping.run import configs
from optimal_stopping.data.stock_model import BlackScholes, STOCK_MODELS
from optimal_stopping.payoffs import get_payoff_class
from optimal_stopping.algorithms.standard.rlsm import RLSM
from optimal_stopping.algorithms.standard.rfqi import RFQI
from optimal_stopping.algorithms.standard.rt import RT
from optimal_stopping.algorithms.standard.lsm import LeastSquaresPricer
from optimal_stopping.algorithms.standard.fqi import FQIFast
from optimal_stopping.algorithms.standard.nlsm import NeuralNetworkPricer
from optimal_stopping.algorithms.standard.dos import DeepOptimalStopping
from optimal_stopping.algorithms.standard.eop import EuropeanOptionPrice
from optimal_stopping.algorithms.path_dependent.srlsm import SRLSM
from optimal_stopping.algorithms.path_dependent.srfqi import SRFQI
from optimal_stopping.algorithms.testing.SRFQI_RBF import SRFQI_RBF

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

# Algorithm registry
_ALGOS = {
    "RLSM": RLSM,
    "SRLSM": SRLSM,
    "RT": RT,
    "RFQI": RFQI,
    "SRFQI": SRFQI,
    "LSM": LeastSquaresPricer,
    "FQI": FQIFast,
    "NLSM": NeuralNetworkPricer,
    "DOS": DeepOptimalStopping,
    "EOP": EuropeanOptionPrice,
    "SRFQI_RBF": SRFQI_RBF,
}


# Mapping from config parameter names to display names
PARAM_NAMES = {
    'nb_paths': 'Number of Paths',
    'nb_dates': 'Number of Time Steps',
    'nb_stocks': 'Number of Stocks',
    'hidden_size': 'Hidden Layer Size',
    'nb_epochs': 'Number of Epochs',
    'volatilities': 'Volatility',
    'drift': 'Drift',
    'strikes': 'Strike',
    'spots': 'Spot Price',
    'maturities': 'Maturity',
    'k': 'K (Best-of-K)',
    'barriers': 'Barrier Level',
    'barriers_up': 'Upper Barrier',
    'barriers_down': 'Lower Barrier',
    'hurst': 'Hurst Parameter',
    'ridge_coeff': 'Ridge Coefficient',
}


def validate_config(config):
    """Validate that config has exactly 1 payoff and 1 varying parameter.

    Returns:
        varying_param: str - name of the varying parameter
        param_values: list - values of the varying parameter
    """
    # Check exactly 1 payoff
    if len(config.payoffs) != 1:
        raise ValueError(
            f"Config must have exactly 1 payoff for convergence plot, "
            f"got {len(config.payoffs)}: {config.payoffs}"
        )

    # Find varying parameters
    varying_params = []
    param_values_dict = {}

    for param_name in PARAM_NAMES.keys():
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
                param_values_dict[param_name] = values

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

    varying_param = varying_params[0]
    return varying_param, param_values_dict[varying_param]


def extract_single_value(param):
    """Extract single value from potentially iterable parameter."""
    if isinstance(param, (list, tuple)):
        return param[0]
    return param


def run_single_experiment(config, algo_name, param_value, varying_param):
    """Run a single experiment and return the price.

    Args:
        config: Base config
        algo_name: Algorithm name
        param_value: Value for the varying parameter
        varying_param: Name of the varying parameter

    Returns:
        price: float
    """
    # Create a copy of config and set the varying parameter
    exp_config = copy.copy(config)
    setattr(exp_config, varying_param, [param_value])

    # Extract parameters
    payoff_name = exp_config.payoffs[0]

    # Get algorithm and payoff classes
    if algo_name not in _ALGOS:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    Algo = _ALGOS[algo_name]
    PayoffClass = get_payoff_class(payoff_name)

    # Extract config values
    nb_stocks = extract_single_value(exp_config.nb_stocks)
    spot = extract_single_value(exp_config.spots)
    strike = extract_single_value(exp_config.strikes)
    drift = extract_single_value(exp_config.drift)
    volatility = extract_single_value(exp_config.volatilities)
    nb_dates = extract_single_value(exp_config.nb_dates)
    nb_paths = extract_single_value(exp_config.nb_paths)
    maturity = extract_single_value(exp_config.maturities)
    hidden_size = extract_single_value(exp_config.hidden_size)
    nb_epochs = extract_single_value(exp_config.nb_epochs)

    dividend = extract_single_value(exp_config.dividends)
    dtype = extract_single_value(exp_config.dtype) if hasattr(exp_config, 'dtype') else 'float32'

    factors = extract_single_value(exp_config.factors)
    use_payoff_as_input = extract_single_value(exp_config.use_payoff_as_input)
    train_ITM_only = extract_single_value(exp_config.train_ITM_only)

    # Create payoff with parameters
    payoff_params = {}
    if hasattr(exp_config, 'barriers'):
        payoff_params['barrier'] = extract_single_value(exp_config.barriers)
    if hasattr(exp_config, 'barriers_up'):
        payoff_params['barrier_up'] = extract_single_value(exp_config.barriers_up)
    if hasattr(exp_config, 'barriers_down'):
        payoff_params['barrier_down'] = extract_single_value(exp_config.barriers_down)
    if hasattr(exp_config, 'k'):
        payoff_params['k'] = extract_single_value(exp_config.k)
    if hasattr(exp_config, 'weights'):
        payoff_params['weights'] = extract_single_value(exp_config.weights)
    if hasattr(exp_config, 'step_param1'):
        payoff_params['step_param1'] = extract_single_value(exp_config.step_param1)
    if hasattr(exp_config, 'step_param2'):
        payoff_params['step_param2'] = extract_single_value(exp_config.step_param2)
    if hasattr(exp_config, 'step_param3'):
        payoff_params['step_param3'] = extract_single_value(exp_config.step_param3)
    if hasattr(exp_config, 'step_param4'):
        payoff_params['step_param4'] = extract_single_value(exp_config.step_param4)

    # Pass rate and maturity for step barriers
    if 'step_param1' in payoff_params:
        payoff_params['rate'] = drift  # drift = risk-free rate
        payoff_params['maturity'] = maturity

    payoff = PayoffClass(strike=strike, **payoff_params)

    # Create stock model
    stock_model_name = extract_single_value(exp_config.stock_models)
    if stock_model_name in STOCK_MODELS:
        ModelClass = STOCK_MODELS[stock_model_name]

        # Build kwargs for model
        model_kwargs = {
            'drift': drift,
            'volatility': volatility,
            'nb_stocks': nb_stocks,
            'nb_paths': nb_paths,
            'nb_dates': nb_dates,
            'spot': spot,
            'dividend': dividend,
            'maturity': maturity,
            'dtype': dtype
        }

        # Add model-specific parameters
        if hasattr(exp_config, 'hurst') and stock_model_name == 'FractionalBlackScholes':
            model_kwargs['hurst'] = extract_single_value(exp_config.hurst)

        model = ModelClass(**model_kwargs)
    else:
        raise ValueError(f"Unknown stock model: {stock_model_name}")

    # Create and run algorithm
    if algo_name == "EOP":
        algo = Algo(model, payoff)
    else:
        algo = Algo(
            model, payoff,
            hidden_size=hidden_size,
            nb_epochs=nb_epochs,
            factors=factors,
            train_ITM_only=train_ITM_only,
            use_payoff_as_input=use_payoff_as_input
        )

    price, _ = algo.price()
    return price


def run_convergence_experiments(config, varying_param, param_values):
    """Run all experiments and collect results.

    Returns:
        data_dict: dict mapping algo -> (x_values, mean_prices, ci_margins)
    """
    algos = list(config.algos)
    nb_runs = config.nb_runs
    confidence_level = 0.99  # <--- Set to 99%

    print(f"\nRunning experiments...")
    print(f"  Algorithms: {len(algos)}")
    print(f"  Parameter values: {len(param_values)}")
    print(f"  Runs per combination: {nb_runs}")
    print(f"  Confidence Level: {confidence_level * 100}% (Student's t)")

    data_dict = {}

    for algo in algos:
        print(f"\n{algo}:")
        x_values = []
        mean_prices = []
        ci_margins = []  # <--- Renamed from std_prices to ci_margins

        for param_val in param_values:
            prices = []

            for run in range(nb_runs):
                print(f"  {varying_param}={param_val}, run {run + 1}/{nb_runs}...", end=" ")
                try:
                    price = run_single_experiment(config, algo, param_val, varying_param)
                    prices.append(price)
                    print(f"price={price:.4f}")
                except Exception as e:
                    print(f"FAILED: {e}")
                    continue

            if len(prices) > 1:
                x_values.append(param_val)
                mean_val = np.mean(prices)
                mean_prices.append(mean_val)

                # --- CI CALCULATION START ---
                n = len(prices)
                # Calculate Standard Error of the Mean (SEM)
                sem = st.sem(prices)
                # Get t-critical value for 99% confidence and (n-1) degrees of freedom
                t_crit = st.t.ppf((1 + confidence_level) / 2., n - 1)
                # Calculate Margin of Error
                margin = sem * t_crit

                ci_margins.append(margin)
                # --- CI CALCULATION END ---

                print(f"  → Mean: {mean_val:.4f}, 99% CI: ±{margin:.4f} (n={n})")
            elif len(prices) == 1:
                # Fallback for single run (cannot calc CI)
                x_values.append(param_val)
                mean_prices.append(prices[0])
                ci_margins.append(0.0)
                print(f"  → Mean: {prices[0]:.4f}, CI: N/A (n=1)")
            else:
                print(f"  → All runs failed for {varying_param}={param_val}")

        if len(x_values) > 0:
            data_dict[algo] = (
                np.array(x_values),
                np.array(mean_prices),
                np.array(ci_margins)
            )

    return data_dict

def should_use_log_scale(values):
    """Determine if x-axis should use log scale."""
    if len(values) < 2:
        return False

    min_val = np.min(values)
    max_val = np.max(values)

    if min_val <= 0:
        return False

    ratio = max_val / min_val
    return ratio > 100


def create_convergence_plot(config, data_dict, varying_param, output_path):
    """Create and save convergence plot with Confidence Intervals."""
    payoff_name = config.payoffs[0]
    param_display_name = PARAM_NAMES.get(varying_param, varying_param)

    # Determine log scale
    all_x_values = np.concatenate([x for x, _, _ in data_dict.values()])
    use_log = should_use_log_scale(all_x_values)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color cycle
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_dict)))

    # Plot each algorithm
    for idx, (algo, (x_values, mean_prices, ci_margins)) in enumerate(data_dict.items()):
        color = colors[idx]

        # Plot line
        ax.plot(x_values, mean_prices,
                marker='o', markersize=8, linewidth=2,
                label=algo, color=color, alpha=0.8)

        # Error bars (The 'yerr' here now represents the CI margin)
        ax.errorbar(x_values, mean_prices, yerr=ci_margins,
                   fmt='none', ecolor=color, alpha=0.3, capsize=5)

        # Shaded region (Visualizing the 99% Confidence Interval)
        ax.fill_between(x_values,
                        mean_prices - ci_margins,
                        mean_prices + ci_margins,
                        alpha=0.15, color=color) # Increased alpha slightly for visibility

    # Set scale
    if use_log:
        ax.set_xscale('log')
    else:
        ax.set_xscale('linear')

    # Labels and title
    ax.set_xlabel(param_display_name, fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Price', fontsize=14, fontweight='bold')
    # Updated Title
    ax.set_title(f'Convergence: {payoff_name} (99% Confidence Interval)', fontsize=16, fontweight='bold')

    # Legend
    ax.legend(loc='best', fontsize=12, framealpha=0.9)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Tight layout
    plt.tight_layout()

    # Save
    print(f"Saving plot to {output_path}...")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Plot saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Create convergence plot by running algorithms"
    )
    parser.add_argument(
        '--configs',
        type=str,
        required=True,
        help='Config name from configs.py'
    )
    parser.add_argument('--telegram_token', type=str,
                        default="8239319342:AAGIIcoDaxJ1uauHbWfdByF4yzNYdQ5jpiA")
    parser.add_argument('--telegram_chat_id', type=str,
                        default="798647521")
    parser.add_argument('--send_telegram', action='store_true', default=True)
    parser.add_argument('--no_telegram', action='store_true')

    args = parser.parse_args()

    if args.no_telegram:
        args.send_telegram = False

    print("="*70)
    print("CONVERGENCE PLOT GENERATION")
    print("="*70)

    # Load config
    print(f"\nLoading config: {args.configs}")
    if not hasattr(configs, args.configs):
        raise ValueError(f"Config '{args.configs}' not found")
    config = getattr(configs, args.configs)

    # Validate
    print(f"\nValidating config...")
    varying_param, param_values = validate_config(config)
    print(f"✓ Config valid")
    print(f"  Payoff: {config.payoffs[0]}")
    print(f"  Varying parameter: {varying_param} = {param_values}")
    print(f"  Algorithms: {list(config.algos)}")
    print(f"  Runs per point: {config.nb_runs}")

    # Send start notification
    if TELEGRAM_ENABLED and args.send_telegram:
        try:
            SBM.send_notification(
                token=args.telegram_token,
                text=f'📊 Starting convergence experiments...\n\n'
                     f'Config: {args.configs}\n'
                     f'Payoff: {config.payoffs[0]}\n'
                     f'Varying: {varying_param}\n'
                     f'Total runs: {len(config.algos) * len(param_values) * config.nb_runs}',
                chat_id=args.telegram_chat_id
            )
        except Exception as e:
            print(f"⚠️  Telegram failed: {e}")

    # Run experiments
    data_dict = run_convergence_experiments(config, varying_param, param_values)

    if not data_dict:
        raise ValueError("No data collected from experiments")

    # Create output directory
    output_dir = Path('results') / args.configs
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f'convergence_{args.configs}_{timestamp}.png'

    # Create plot
    print(f"\nCreating plot...")
    create_convergence_plot(config, data_dict, varying_param, output_path)

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")

    for algo, (x_values, mean_prices, std_prices) in data_dict.items():
        print(f"\n{algo}:")
        print(f"  Parameter range: {x_values[0]} to {x_values[-1]}")
        print(f"  Price range: {np.min(mean_prices):.4f} to {np.max(mean_prices):.4f}")
        print(f"  Mean std dev: {np.mean(std_prices):.4f}")

        if len(x_values) >= 2:
            price_change = abs(mean_prices[-1] - mean_prices[-2]) / mean_prices[-2] * 100
            if price_change < 1.0:
                print(f"  ✓ Converged (last change: {price_change:.2f}%)")
            else:
                print(f"  ⚠ Not converged (last change: {price_change:.2f}%)")

    print(f"\n{'='*70}")
    print(f"✓ Done! Plot saved to: {output_path}")
    print(f"{'='*70}\n")

    # Send completion
    if TELEGRAM_ENABLED and args.send_telegram:
        try:
            stats_text = "\n\nResults:\n"
            for algo, (x_values, mean_prices, _) in data_dict.items():
                stats_text += f"\n{algo}:\n"
                stats_text += f"  Range: {x_values[0]} → {x_values[-1]}\n"
                stats_text += f"  Price: {np.min(mean_prices):.4f} → {np.max(mean_prices):.4f}\n"

            SBM.send_notification(
                token=args.telegram_token,
                text=f'✅ Convergence plot complete!\n\n'
                     f'Config: {args.configs}\n'
                     f'Payoff: {config.payoffs[0]}'
                     f'{stats_text}',
                chat_id=args.telegram_chat_id,
                files=[str(output_path)]
            )
        except Exception as e:
            print(f"⚠️  Telegram failed: {e}")


if __name__ == '__main__':
    main()
