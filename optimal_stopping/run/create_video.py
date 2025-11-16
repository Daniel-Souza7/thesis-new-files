"""
Create animated videos of optimal stopping exercise decisions.

This script visualizes how RL algorithms exercise American options by showing:
- Stock price paths evolving over time
- Exercise decisions marked with red dots
- Paths grayed out after exercise
- Running statistics and payoff evolution

Usage:
    python -m optimal_stopping.run.create_video --config=my_config --nb_paths_to_plot=100
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from optimal_stopping.run import configs
from optimal_stopping.data.stock_model import BlackScholes
from optimal_stopping.payoffs import get_payoff_class
from optimal_stopping.algorithms.standard.rlsm import RLSM
from optimal_stopping.algorithms.standard.rfqi import RFQI
from optimal_stopping.algorithms.path_dependent.srlsm import SRLSM
from optimal_stopping.algorithms.path_dependent.srfqi import SRFQI

# Algorithm registry
_ALGOS = {
    "RLSM": RLSM,
    "SRLSM": SRLSM,
    "RFQI": RFQI,
    "SRFQI": SRFQI,
}


def validate_config(config):
    """Validate that config has exactly 1 algo and 1 payoff."""
    if len(config.algos) != 1:
        raise ValueError(
            f"Config must have exactly 1 algo for video generation, got {len(config.algos)}: {config.algos}"
        )
    if len(config.payoffs) != 1:
        raise ValueError(
            f"Config must have exactly 1 payoff for video generation, got {len(config.payoffs)}: {config.payoffs}"
        )


def run_algorithm_for_video(config, nb_paths_for_video):
    """Run the algorithm and return paths, exercise decisions, and payoffs.

    Returns:
        stock_paths: (nb_paths, nb_stocks, nb_dates+1) or (nb_paths, nb_dates+1)
        exercise_times: (nb_paths,) - time step when each path exercised (0 to nb_dates)
        payoff_values: (nb_paths,) - payoff value at exercise
        algo_name: str
        payoff_name: str
    """
    # Extract single algo and payoff
    algo_name = config.algos[0]
    payoff_name = config.payoffs[0]

    # Get algorithm class
    if algo_name not in _ALGOS:
        raise ValueError(f"Unknown algorithm: {algo_name}. Available: {list(_ALGOS.keys())}")
    Algo = _ALGOS[algo_name]

    # Get payoff class
    PayoffClass = get_payoff_class(payoff_name)

    # Extract config parameters (handle lists)
    nb_stocks = config.nb_stocks[0] if isinstance(config.nb_stocks, (list, tuple)) else config.nb_stocks
    spot = config.spots[0] if isinstance(config.spots, (list, tuple)) else config.spots
    drift = config.drift[0] if isinstance(config.drift, (list, tuple)) else config.drift
    volatility = config.volatilities[0] if isinstance(config.volatilities, (list, tuple)) else config.volatilities
    nb_dates = config.nb_dates[0] if isinstance(config.nb_dates, (list, tuple)) else config.nb_dates
    hidden_size = config.hidden_size[0] if isinstance(config.hidden_size, (list, tuple)) else config.hidden_size
    nb_epochs = config.nb_epochs[0] if isinstance(config.nb_epochs, (list, tuple)) else config.nb_epochs
    factors = config.factors[0] if isinstance(config.factors, (list, tuple)) else config.factors
    use_payoff_as_input = config.use_payoff_as_input[0] if isinstance(config.use_payoff_as_input, (list, tuple)) else config.use_payoff_as_input
    train_ITM_only = config.train_ITM_only[0] if isinstance(config.train_ITM_only, (list, tuple)) else config.train_ITM_only

    # Create stock model (using stock_model.py API)
    maturity = config.maturities[0] if isinstance(config.maturities, (list, tuple)) else config.maturities
    stock_model_obj = BlackScholes(
        drift=drift,
        volatility=volatility,
        nb_stocks=nb_stocks,
        nb_paths=nb_paths_for_video,
        nb_dates=nb_dates,
        spot=spot,
        dividend=0,
        maturity=maturity
    )

    # Generate paths
    stock_paths, _ = stock_model_obj.generate_paths()

    # Create payoff with barrier parameters if needed
    payoff_params = {}
    if 'barrier' in payoff_name.lower():
        if hasattr(config, 'barriers'):
            payoff_params['barrier'] = config.barriers[0] if isinstance(config.barriers, (list, tuple)) else config.barriers
        if hasattr(config, 'barriers_up'):
            payoff_params['barrier_up'] = config.barriers_up[0] if isinstance(config.barriers_up, (list, tuple)) else config.barriers_up
        if hasattr(config, 'barriers_down'):
            payoff_params['barrier_down'] = config.barriers_down[0] if isinstance(config.barriers_down, (list, tuple)) else config.barriers_down

    # Add other payoff parameters
    if hasattr(config, 'k'):
        payoff_params['k'] = config.k[0] if isinstance(config.k, (list, tuple)) else config.k
    if hasattr(config, 'weights'):
        payoff_params['weights'] = config.weights[0] if isinstance(config.weights, (list, tuple)) else config.weights
    if hasattr(config, 'alpha'):
        payoff_params['alpha'] = config.alpha[0] if isinstance(config.alpha, (list, tuple)) else config.alpha

    payoff = PayoffClass(
        strike=config.strikes[0] if isinstance(config.strikes, (list, tuple)) else config.strikes,
        **payoff_params
    )

    print(f"Computing exercise decisions for {payoff_name} using greedy strategy...")
    # Note: RL algorithms (RLSM/RFQI) don't expose a fit/predict interface suitable for
    # path-by-path exercise decisions. For visualization purposes, we use a greedy strategy:
    # exercise at the time that maximizes payoff. This still provides useful insights
    # into optimal stopping behavior.

    exercise_times = np.zeros(nb_paths_for_video, dtype=int)
    payoff_values = np.zeros(nb_paths_for_video)

    # Greedy strategy: exercise when payoff is maximized
    for path_idx in range(nb_paths_for_video):
        max_payoff = 0
        best_time = nb_dates

        for t in range(nb_dates + 1):
            if PayoffClass.is_path_dependent:
                # Pass full history up to time t
                X_t = stock_paths[path_idx:path_idx+1, :, :t+1]
            else:
                # Pass only current timestep
                X_t = stock_paths[path_idx:path_idx+1, :, t]

            # Use eval() directly, not __call__()
            payoff_now = payoff.eval(X_t)[0]

            # Track maximum payoff
            if payoff_now > max_payoff:
                max_payoff = payoff_now
                best_time = t

        exercise_times[path_idx] = best_time
        payoff_values[path_idx] = max_payoff

    return stock_paths, exercise_times, payoff_values, algo_name, payoff_name


def create_video(config, stock_paths, exercise_times, payoff_values, algo_name, payoff_name, output_path):
    """Create animated video of optimal stopping decisions.

    Args:
        config: Config object
        stock_paths: (nb_paths, nb_stocks, nb_dates+1) or (nb_paths, nb_dates+1)
        exercise_times: (nb_paths,) - exercise time for each path
        payoff_values: (nb_paths,) - payoff at exercise
        algo_name: str
        payoff_name: str
        output_path: Path to save video
    """
    nb_paths = stock_paths.shape[0]
    nb_dates = config.nb_dates[0] if isinstance(config.nb_dates, (list, tuple)) else config.nb_dates

    # Determine if path-dependent (3D) or not (2D)
    if stock_paths.ndim == 3:
        nb_stocks = stock_paths.shape[1]
        is_multidim = True
    else:
        nb_stocks = 1
        is_multidim = False
        stock_paths = stock_paths[:, np.newaxis, :]  # Add stock dimension

    # Colors for different stocks
    colors = plt.cm.tab10(np.linspace(0, 1, nb_stocks))

    # Create figure with main plot and stats panel
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1])

    ax_main = fig.add_subplot(gs[0, :])
    ax_stats = fig.add_subplot(gs[1, 0])
    ax_payoff = fig.add_subplot(gs[1, 1])

    # Title
    strike = config.strikes[0] if isinstance(config.strikes, (list, tuple)) else config.strikes
    fig.suptitle(f'{algo_name} Exercise Decisions: {payoff_name}\n'
                 f'Strike={strike:.1f}, d={nb_stocks}, T={nb_dates}',
                 fontsize=14, fontweight='bold')

    # Initialize main plot
    ax_main.set_xlabel('Time Step', fontsize=12)
    ax_main.set_ylabel('Stock Price', fontsize=12)
    ax_main.set_xlim(-0.5, nb_dates + 0.5)

    # Find y-axis limits
    all_prices = stock_paths.reshape(-1)
    y_min, y_max = all_prices.min(), all_prices.max()
    y_margin = (y_max - y_min) * 0.1
    ax_main.set_ylim(y_min - y_margin, y_max + y_margin)

    # Strike line
    ax_main.axhline(y=strike, color='black', linestyle='--', linewidth=2,
                    label=f'Strike = {strike:.1f}', alpha=0.7)
    ax_main.legend(loc='upper right')
    ax_main.grid(True, alpha=0.3)

    # Initialize path lines (one per path per stock)
    lines = []
    for path_idx in range(nb_paths):
        path_lines = []
        for stock_idx in range(nb_stocks):
            line, = ax_main.plot([], [], alpha=0.6, linewidth=0.8,
                                color=colors[stock_idx])
            path_lines.append(line)
        lines.append(path_lines)

    # Initialize exercise markers
    exercise_markers = []
    for path_idx in range(nb_paths):
        markers = []
        for stock_idx in range(nb_stocks):
            marker, = ax_main.plot([], [], 'ro', markersize=8, alpha=0.8)
            markers.append(marker)
        exercise_markers.append(markers)

    # Stats text
    stats_text = ax_stats.text(0.05, 0.5, '', transform=ax_stats.transAxes,
                               fontsize=11, verticalalignment='center',
                               fontfamily='monospace')
    ax_stats.axis('off')

    # Payoff evolution plot
    ax_payoff.set_xlabel('Time', fontsize=10)
    ax_payoff.set_ylabel('Avg Payoff', fontsize=10)
    ax_payoff.set_xlim(0, nb_dates)
    ax_payoff.grid(True, alpha=0.3)
    payoff_line, = ax_payoff.plot([], [], 'b-', linewidth=2)

    # Track cumulative statistics
    payoff_evolution = []

    def init():
        """Initialize animation."""
        for path_lines in lines:
            for line in path_lines:
                line.set_data([], [])
        for markers in exercise_markers:
            for marker in markers:
                marker.set_data([], [])
        stats_text.set_text('')
        payoff_line.set_data([], [])
        return [l for path_lines in lines for l in path_lines] + \
               [m for markers in exercise_markers for m in markers] + \
               [stats_text, payoff_line]

    def animate(frame):
        """Update animation at frame (with smooth interpolation)."""
        # Smooth interpolation: 10 frames per time step
        frames_per_step = 10
        t = frame / frames_per_step  # Fractional time
        t_int = int(t)  # Integer time step
        t_frac = t - t_int  # Fractional part for interpolation

        # Track exercised paths (at integer time)
        exercised = exercise_times <= t_int
        not_exercised = ~exercised

        # Update path lines with smooth interpolation
        for path_idx in range(nb_paths):
            for stock_idx in range(nb_stocks):
                line = lines[path_idx][stock_idx]

                # Show path up to current time or exercise time
                end_time_int = min(t_int, exercise_times[path_idx])

                # Build smooth path with interpolation
                if end_time_int == 0:
                    # Just the starting point
                    x_data = np.array([0])
                    y_data = stock_paths[path_idx, stock_idx, :1]
                elif t_frac > 0 and end_time_int < nb_dates and end_time_int == t_int:
                    # Interpolate between current and next point
                    x_data = np.arange(end_time_int + 1)
                    y_data = stock_paths[path_idx, stock_idx, :end_time_int + 1]

                    # Add interpolated point
                    x_interp = t
                    y_interp = stock_paths[path_idx, stock_idx, end_time_int] * (1 - t_frac) + \
                               stock_paths[path_idx, stock_idx, end_time_int + 1] * t_frac

                    x_data = np.append(x_data, x_interp)
                    y_data = np.append(y_data, y_interp)
                else:
                    # No interpolation needed
                    x_data = np.arange(end_time_int + 1)
                    y_data = stock_paths[path_idx, stock_idx, :end_time_int + 1]

                line.set_data(x_data, y_data)

                # Gray out if exercised
                if exercised[path_idx]:
                    line.set_alpha(0.2)
                    line.set_linewidth(0.5)
                else:
                    line.set_alpha(0.6)
                    line.set_linewidth(0.8)

        # Update exercise markers
        for path_idx in range(nb_paths):
            if exercised[path_idx]:
                ex_time = exercise_times[path_idx]
                for stock_idx in range(nb_stocks):
                    marker = exercise_markers[path_idx][stock_idx]
                    marker.set_data([ex_time], [stock_paths[path_idx, stock_idx, ex_time]])
            else:
                for stock_idx in range(nb_stocks):
                    exercise_markers[path_idx][stock_idx].set_data([], [])

        # Update statistics
        nb_exercised = exercised.sum()
        pct_exercised = 100 * nb_exercised / nb_paths
        avg_payoff = payoff_values[exercised].mean() if nb_exercised > 0 else 0

        stats_str = f"Time Step: {t:.1f}/{nb_dates}\n"
        stats_str += f"Exercised: {nb_exercised}/{nb_paths} ({pct_exercised:.1f}%)\n"
        stats_str += f"Avg Payoff: {avg_payoff:.2f}\n"
        stats_str += f"Active Paths: {not_exercised.sum()}"
        stats_text.set_text(stats_str)

        # Update payoff evolution (only at integer time steps to avoid jitter)
        if frame % frames_per_step == 0:
            payoff_evolution.append(avg_payoff)
            if len(payoff_evolution) > 1:
                payoff_line.set_data(range(len(payoff_evolution)), payoff_evolution)
                ax_payoff.set_ylim(0, max(payoff_evolution) * 1.1 if max(payoff_evolution) > 0 else 1)

        return [l for path_lines in lines for l in path_lines] + \
               [m for markers in exercise_markers for m in markers] + \
               [stats_text, payoff_line]

    # Create animation with smooth interpolation
    frames_per_step = 10
    total_frames = (nb_dates + 1) * frames_per_step
    print(f"Creating smooth animation with {total_frames} frames ({frames_per_step} per time step)...")
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=total_frames,
        interval=50,  # 50ms per frame = 20 fps, gives 0.5s per time step
        blit=True,
        repeat=True
    )

    # Save video
    print(f"Saving video to {output_path}...")
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, metadata=dict(artist='Optimal Stopping'), bitrate=1800)
    anim.save(output_path, writer=writer)

    plt.close(fig)
    print(f"✓ Video saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create animated video of optimal stopping exercise decisions"
    )
    parser.add_argument(
        '--configs',
        type=str,
        required=True,
        help='Config name from configs.py (must have exactly 1 algo and 1 payoff)'
    )
    parser.add_argument(
        '--nb_paths_to_plot',
        type=int,
        default=None,
        help='Number of paths to plot (default: max(100, 5*d))'
    )

    args = parser.parse_args()

    # Load config
    print(f"Loading config: {args.configs}")
    if not hasattr(configs, args.configs):
        raise ValueError(f"Config '{args.configs}' not found in configs.py")
    config = getattr(configs, args.configs)

    # Validate config
    validate_config(config)

    # Determine number of paths
    nb_stocks = config.nb_stocks if isinstance(config.nb_stocks, int) else config.nb_stocks[0]
    if args.nb_paths_to_plot is None:
        nb_paths_to_plot = max(200, 10 * nb_stocks)
    else:
        nb_paths_to_plot = args.nb_paths_to_plot

    print(f"Will plot {nb_paths_to_plot} paths for {nb_stocks} stock(s)")

    # Run algorithm
    stock_paths, exercise_times, payoff_values, algo_name, payoff_name = \
        run_algorithm_for_video(config, nb_paths_to_plot)

    # Create output directory
    output_dir = Path('results') / args.configs
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f'video_{args.configs}_{timestamp}.mp4'

    # Create video
    create_video(
        config, stock_paths, exercise_times, payoff_values,
        algo_name, payoff_name, output_path
    )

    print(f"\n✓ Done! Video saved to: {output_path}")
    print(f"  Algo: {algo_name}")
    print(f"  Payoff: {payoff_name}")
    print(f"  Paths: {nb_paths_to_plot}")
    print(f"  Avg exercise time: {exercise_times.mean():.2f}")
    print(f"  Avg payoff: {payoff_values.mean():.2f}")


if __name__ == '__main__':
    main()
