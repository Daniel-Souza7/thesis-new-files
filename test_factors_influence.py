#!/usr/bin/env python3
"""
Test how different factors influence SRLSM and SRFQI accuracy.

Compares performance on:
1. Simple problem: Down-and-out barrier put
2. Complex problem: Lookback put with rough volatility

Tests different activation slopes and input scaling combinations.
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path

# Add project to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from optimal_stopping.data.stock_model import BlackScholes, Heston
from optimal_stopping.payoffs import get_payoff_class, LookbackFixedPut
from optimal_stopping.algorithms.path_dependent.srlsm import SRLSM
from optimal_stopping.algorithms.path_dependent.srfqi import SRFQI


def run_single_test(algorithm_class, model, payoff, factors, algo_name, problem_name, nb_epochs=20, hidden_size=20):
    """Run single factor test and return metrics."""
    try:
        # Initialize algorithm
        algo = algorithm_class(
            model=model,
            payoff=payoff,
            nb_epochs=nb_epochs,
            hidden_size=hidden_size,
            factors=factors,
            train_ITM_only=True,
            use_payoff_as_input=False
        )

        # Time the pricing
        start = time.time()

        # Get lower and upper bounds
        lower_bound, upper_bound, time_path_gen = algo.price_upper_lower_bound(train_eval_split=2)

        total_time = time.time() - start - time_path_gen

        # Calculate gap metrics
        gap = upper_bound - lower_bound
        relative_gap = (gap / lower_bound * 100) if lower_bound > 0 else np.nan

        return {
            'Algorithm': algo_name,
            'Problem': problem_name,
            'Activation_Slope': factors[0],
            'Input_Scale': factors[1] if len(factors) > 1 else 1.0,
            'Lower_Bound': lower_bound,
            'Upper_Bound': upper_bound,
            'Gap': gap,
            'Relative_Gap_%': relative_gap,
            'Compute_Time_s': total_time,
            'Status': 'Success'
        }

    except Exception as e:
        return {
            'Algorithm': algo_name,
            'Problem': problem_name,
            'Activation_Slope': factors[0],
            'Input_Scale': factors[1] if len(factors) > 1 else 1.0,
            'Lower_Bound': np.nan,
            'Upper_Bound': np.nan,
            'Gap': np.nan,
            'Relative_Gap_%': np.nan,
            'Compute_Time_s': np.nan,
            'Status': f'Failed: {str(e)[:50]}'
        }


def setup_simple_problem():
    """
    Simple problem: Down-and-out barrier put option.

    Moderately path-dependent, smooth payoff structure.
    """
    model = BlackScholes(
        spot=100.0,
        drift=0.05,
        dividend=0.0,
        volatility=0.2,
        maturity=1.0,
        nb_stocks=1,
        nb_paths=10000,
        nb_dates=50,
        seed=42
    )

    # Get barrier payoff from registry
    DownAndOutPut = get_payoff_class('DO-Put')
    payoff = DownAndOutPut(
        strike=100.0,
        barrier=80.0
    )

    return model, payoff, "Barrier_Put_Simple"


def setup_complex_problem():
    """
    Complex problem: Lookback put with Heston stochastic volatility.

    Highly path-dependent with stochastic volatility.
    """
    model = Heston(
        drift=0.05,           # Stock drift
        volatility=0.3,       # Vol-of-vol (xi)
        mean=0.04,           # Long-term variance mean (v_bar, also initial variance)
        speed=2.0,           # Mean reversion speed (kappa)
        correlation=-0.7,    # Correlation (rho)
        nb_stocks=1,
        nb_paths=10000,
        nb_dates=50,
        spot=100.0,          # Initial stock price
        maturity=1.0,
        dividend=0.0,
        seed=42
    )

    payoff = LookbackFixedPut(
        strike=100.0
    )

    return model, payoff, "Lookback_Put_Complex"


def test_factors():
    """Main test function."""
    print("=" * 80)
    print("Testing Factor Influence on SRLSM and SRFQI")
    print("=" * 80)
    print()

    # Factor combinations to test
    activation_slopes = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    input_scales = [0.5, 0.7, 1.0, 1.3, 1.5]

    # Algorithms to test
    algorithms = [
        (SRLSM, 'SRLSM'),
        (SRFQI, 'SRFQI')
    ]

    # Problems to test
    problems = [
        setup_simple_problem(),
        setup_complex_problem()
    ]

    # Collect results
    results = []

    total_tests = len(algorithms) * len(problems) * len(activation_slopes) * len(input_scales)
    test_counter = 0

    print(f"Running {total_tests} tests...\n")

    # Test each combination
    for algo_class, algo_name in algorithms:
        for model, payoff, problem_name in problems:
            print(f"\n{algo_name} on {problem_name}:")
            print("-" * 60)

            for activation_slope in activation_slopes:
                for input_scale in input_scales:
                    test_counter += 1

                    factors = (activation_slope, input_scale)

                    print(f"  [{test_counter}/{total_tests}] Testing factors={factors}... ", end='', flush=True)

                    result = run_single_test(
                        algorithm_class=algo_class,
                        model=model,
                        payoff=payoff,
                        factors=factors,
                        algo_name=algo_name,
                        problem_name=problem_name,
                        nb_epochs=20,
                        hidden_size=20
                    )

                    results.append(result)

                    if result['Status'] == 'Success':
                        print(f"Gap: {result['Relative_Gap_%']:.2f}%, Time: {result['Compute_Time_s']:.2f}s")
                    else:
                        print(f"{result['Status']}")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save full results
    output_file = 'factor_influence_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\n{'=' * 80}")
    print(f"Full results saved to: {output_file}")
    print(f"{'=' * 80}\n")

    # Print summary statistics
    print_summary(df)

    # Print best configurations
    print_best_configs(df)

    return df


def print_summary(df):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    # Filter successful runs
    df_success = df[df['Status'] == 'Success'].copy()

    if len(df_success) == 0:
        print("No successful runs!")
        return

    # Summary by algorithm and problem
    for algo in df_success['Algorithm'].unique():
        print(f"\n{algo}:")
        print("-" * 60)

        for problem in df_success['Problem'].unique():
            subset = df_success[(df_success['Algorithm'] == algo) &
                               (df_success['Problem'] == problem)]

            if len(subset) == 0:
                continue

            print(f"\n  {problem}:")
            print(f"    Relative Gap: {subset['Relative_Gap_%'].mean():.2f}% ± {subset['Relative_Gap_%'].std():.2f}%")
            print(f"    Best Gap: {subset['Relative_Gap_%'].min():.2f}%")
            print(f"    Worst Gap: {subset['Relative_Gap_%'].max():.2f}%")
            print(f"    Avg Compute Time: {subset['Compute_Time_s'].mean():.2f}s")


def print_best_configs(df):
    """Print best factor configurations."""
    print("\n" + "=" * 80)
    print("BEST CONFIGURATIONS (Smallest Relative Gap)")
    print("=" * 80)

    df_success = df[df['Status'] == 'Success'].copy()

    if len(df_success) == 0:
        return

    # Best for each (algorithm, problem) combination
    for algo in df_success['Algorithm'].unique():
        for problem in df_success['Problem'].unique():
            subset = df_success[(df_success['Algorithm'] == algo) &
                               (df_success['Problem'] == problem)]

            if len(subset) == 0:
                continue

            # Find best configuration
            best_idx = subset['Relative_Gap_%'].idxmin()
            best = subset.loc[best_idx]

            print(f"\n{algo} on {problem}:")
            print(f"  Best factors: (activation={best['Activation_Slope']:.1f}, input_scale={best['Input_Scale']:.1f})")
            print(f"  Lower Bound: {best['Lower_Bound']:.4f}")
            print(f"  Upper Bound: {best['Upper_Bound']:.4f}")
            print(f"  Relative Gap: {best['Relative_Gap_%']:.2f}%")
            print(f"  Compute Time: {best['Compute_Time_s']:.2f}s")

    # Overall insights
    print("\n" + "=" * 80)
    print("INSIGHTS")
    print("=" * 80)

    # Best activation slopes
    print("\nBest Activation Slopes (by average relative gap):")
    activation_summary = df_success.groupby('Activation_Slope')['Relative_Gap_%'].agg(['mean', 'std', 'count'])
    activation_summary = activation_summary.sort_values('mean')
    print(activation_summary.to_string())

    print("\nBest Input Scales (by average relative gap):")
    scale_summary = df_success.groupby('Input_Scale')['Relative_Gap_%'].agg(['mean', 'std', 'count'])
    scale_summary = scale_summary.sort_values('mean')
    print(scale_summary.to_string())


def create_heatmaps(df):
    """Create heatmap visualizations of factor influence."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        df_success = df[df['Status'] == 'Success'].copy()

        if len(df_success) == 0:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Factor Influence Heatmaps: Relative Gap (%)', fontsize=16)

        plot_idx = 0
        for algo in ['SRLSM', 'SRFQI']:
            for problem in df_success['Problem'].unique():
                subset = df_success[(df_success['Algorithm'] == algo) &
                                   (df_success['Problem'] == problem)]

                if len(subset) == 0:
                    continue

                # Pivot for heatmap
                pivot = subset.pivot_table(
                    values='Relative_Gap_%',
                    index='Activation_Slope',
                    columns='Input_Scale',
                    aggfunc='mean'
                )

                ax = axes[plot_idx // 2, plot_idx % 2]
                sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn_r',
                           ax=ax, cbar_kws={'label': 'Relative Gap (%)'})
                ax.set_title(f'{algo} - {problem}')
                ax.set_xlabel('Input Scale')
                ax.set_ylabel('Activation Slope')

                plot_idx += 1

        plt.tight_layout()
        plt.savefig('factor_influence_heatmaps.png', dpi=150, bbox_inches='tight')
        print(f"\nHeatmaps saved to: factor_influence_heatmaps.png")

    except ImportError:
        print("\nNote: Install matplotlib and seaborn for heatmap visualizations")
        print("  pip install matplotlib seaborn")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Run tests
    df = test_factors()

    # Create visualizations (optional)
    try:
        create_heatmaps(df)
    except Exception as e:
        print(f"\nCould not create heatmaps: {e}")

    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)
