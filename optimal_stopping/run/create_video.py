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
from optimal_stopping.algorithms.standard.rt import RT
from optimal_stopping.algorithms.standard.lsm import LeastSquaresPricer
from optimal_stopping.algorithms.standard.fqi import FQIFast
from optimal_stopping.algorithms.standard.nlsm import NeuralNetworkPricer
from optimal_stopping.algorithms.standard.dos import DeepOptimalStopping
from optimal_stopping.algorithms.standard.eop import EuropeanOptionPrice
from optimal_stopping.algorithms.path_dependent.srlsm import SRLSM
from optimal_stopping.algorithms.path_dependent.srfqi import SRFQI

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


def run_algorithm_for_video(config, nb_paths_to_display):
    """Run the algorithm and return paths, exercise decisions, and payoffs.

    Args:
        config: Config object
        nb_paths_to_display: Number of paths to display in video (subset of total)

    Returns:
        stock_paths_display: (nb_paths_to_display, nb_stocks, nb_dates+1) - paths to display
        exercise_times_display: (nb_paths_to_display,) - exercise times for displayed paths
        payoff_values_display: (nb_paths_to_display,) - payoffs for displayed paths
        exercise_times_all: (nb_paths_total,) - exercise times for ALL paths
        payoff_values_all: (nb_paths_total,) - payoffs for ALL paths
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

    # Extract dividend and dtype
    dividend = config.dividends[0] if isinstance(config.dividends, (list, tuple)) else config.dividends
    dtype = config.dtype[0] if hasattr(config, 'dtype') and isinstance(config.dtype, (list, tuple)) else (config.dtype if hasattr(config, 'dtype') else 'float32')

    # Create payoff with all optional parameters
    payoff_params = {}

    # Barrier parameters (for barrier options like UO-*, DO-*, etc.)
    if hasattr(config, 'barriers'):
        payoff_params['barrier'] = config.barriers[0] if isinstance(config.barriers, (list, tuple)) else config.barriers
    if hasattr(config, 'barriers_up'):
        payoff_params['barrier_up'] = config.barriers_up[0] if isinstance(config.barriers_up, (list, tuple)) else config.barriers_up
    if hasattr(config, 'barriers_down'):
        payoff_params['barrier_down'] = config.barriers_down[0] if isinstance(config.barriers_down, (list, tuple)) else config.barriers_down

    # Step barrier parameters (for StepB-*, DStepB-*)
    if hasattr(config, 'step_param1'):
        payoff_params['step_param1'] = config.step_param1[0] if isinstance(config.step_param1, (list, tuple)) else config.step_param1
    if hasattr(config, 'step_param2'):
        payoff_params['step_param2'] = config.step_param2[0] if isinstance(config.step_param2, (list, tuple)) else config.step_param2
    if hasattr(config, 'step_param3'):
        payoff_params['step_param3'] = config.step_param3[0] if isinstance(config.step_param3, (list, tuple)) else config.step_param3
    if hasattr(config, 'step_param4'):
        payoff_params['step_param4'] = config.step_param4[0] if isinstance(config.step_param4, (list, tuple)) else config.step_param4

    # Rank-based parameters
    if hasattr(config, 'k'):
        payoff_params['k'] = config.k[0] if isinstance(config.k, (list, tuple)) else config.k
    if hasattr(config, 'weights'):
        payoff_params['weights'] = config.weights[0] if isinstance(config.weights, (list, tuple)) else config.weights

    payoff = PayoffClass(
        strike=config.strikes[0] if isinstance(config.strikes, (list, tuple)) else config.strikes,
        **payoff_params
    )

    # Create stock model for TRAINING (smaller number of paths)
    maturity = config.maturities[0] if isinstance(config.maturities, (list, tuple)) else config.maturities
    nb_training_paths = config.nb_paths[0] if isinstance(config.nb_paths, (list, tuple)) else config.nb_paths

    train_model = BlackScholes(
        drift=drift,
        volatility=volatility,
        nb_stocks=nb_stocks,
        nb_paths=nb_training_paths,
        nb_dates=nb_dates,
        spot=spot,
        dividend=dividend,
        maturity=maturity,
        dtype=dtype
    )

    # Create and train the algorithm
    print(f"Training {algo_name} on {nb_training_paths} paths...")
    if algo_name == "EOP":
        # EOP doesn't learn a policy, use greedy fallback
        print(f"EOP has no learned policy, using greedy strategy for {payoff_name}...")
        use_learned_policy = False
    else:
        try:
            algo = Algo(
                train_model, payoff,
                hidden_size=hidden_size,
                nb_epochs=nb_epochs,
                factors=factors,
                train_ITM_only=train_ITM_only,
                use_payoff_as_input=use_payoff_as_input
            )
            # Train the algorithm
            price, _ = algo.price()
            print(f"Trained! Price: {price:.4f}")
            use_learned_policy = True
        except Exception as e:
            print(f"Warning: Failed to train {algo_name}: {e}")
            print("Falling back to greedy strategy...")
            use_learned_policy = False

    # Create stock model for VISUALIZATION (use config nb_paths for accurate statistics)
    nb_paths_total = config.nb_paths[0] if isinstance(config.nb_paths, (list, tuple)) else config.nb_paths
    video_model = BlackScholes(
        drift=drift,
        volatility=volatility,
        nb_stocks=nb_stocks,
        nb_paths=nb_paths_total,
        nb_dates=nb_dates,
        spot=spot,
        dividend=dividend,
        maturity=maturity,
        dtype=dtype
    )

    # Generate ALL paths for visualization
    stock_paths_all, _ = video_model.generate_paths()
    print(f"Generated {nb_paths_total} paths for accurate statistics...")

    # Get exercise decisions using learned policy or greedy fallback
    if use_learned_policy and hasattr(algo, 'backward_induction_on_paths'):
        print(f"Applying learned {algo_name} policy via backward induction to all {nb_paths_total} paths...")
        exercise_times_all, payoff_values_all, video_price = algo.backward_induction_on_paths(stock_paths_all)
        print(f"  Price from backward induction on video paths: {video_price:.4f}")
    else:
        # Fallback: greedy strategy
        print(f"Computing exercise decisions using greedy strategy on all {nb_paths_total} paths...")
        exercise_times_all = np.zeros(nb_paths_total, dtype=int)
        payoff_values_all = np.zeros(nb_paths_total)

        for path_idx in range(nb_paths_total):
            max_payoff = 0
            best_time = nb_dates

            for t in range(nb_dates + 1):
                if PayoffClass.is_path_dependent:
                    # Pass full history up to time t
                    X_t = stock_paths_all[path_idx:path_idx+1, :, :t+1]
                else:
                    # Pass only current timestep
                    X_t = stock_paths_all[path_idx:path_idx+1, :, t]

                # Use eval() directly, not __call__()
                payoff_now = payoff.eval(X_t)[0]

                # Track maximum payoff
                if payoff_now > max_payoff:
                    max_payoff = payoff_now
                    best_time = t

            exercise_times_all[path_idx] = best_time
            payoff_values_all[path_idx] = max_payoff

    # Take a subset for display in video
    nb_display = min(nb_paths_to_display, nb_paths_total)
    print(f"Displaying {nb_display} paths in video (out of {nb_paths_total} total)")
    stock_paths_display = stock_paths_all[:nb_display]
    exercise_times_display = exercise_times_all[:nb_display]
    payoff_values_display = payoff_values_all[:nb_display]

    return (stock_paths_display, exercise_times_display, payoff_values_display,
            exercise_times_all, payoff_values_all, algo_name, payoff_name)


def create_video(config, stock_paths, exercise_times, payoff_values,
                 exercise_times_all, payoff_values_all, algo_name, payoff_name, output_path):
    """Create animated video of optimal stopping decisions.

    Args:
        config: Config object
        stock_paths: (nb_paths_display, nb_stocks, nb_dates+1) - paths to display
        exercise_times: (nb_paths_display,) - exercise times for displayed paths
        payoff_values: (nb_paths_display,) - payoffs for displayed paths
        exercise_times_all: (nb_paths_total,) - exercise times for ALL paths
        payoff_values_all: (nb_paths_total,) - payoffs for ALL paths
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

    # Calculate population statistics for DISPLAYED paths (to display on video)
    normalized_ex_times = exercise_times / nb_dates
    exercised_at_maturity = (exercise_times == nb_dates).sum()
    pop_stats_display = {
        'nb_paths': nb_paths,
        'avg_ex_time': normalized_ex_times.mean(),
        'pct_at_maturity': 100 * exercised_at_maturity / nb_paths,
        'avg_payoff': payoff_values.mean(),
        'std_payoff': payoff_values.std(),
        'median_payoff': np.median(payoff_values)
    }

    # Calculate population statistics for ALL paths
    nb_paths_total = len(exercise_times_all)
    normalized_ex_times_all = exercise_times_all / nb_dates
    exercised_at_maturity_all = (exercise_times_all == nb_dates).sum()
    pop_stats_all = {
        'nb_paths': nb_paths_total,
        'avg_ex_time': normalized_ex_times_all.mean(),
        'pct_at_maturity': 100 * exercised_at_maturity_all / nb_paths_total,
        'avg_payoff': payoff_values_all.mean(),
        'std_payoff': payoff_values_all.std(),
        'median_payoff': np.median(payoff_values_all)
    }

    # Create figure with main plot, stats panel, and TWO population stats panels
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(4, 2, height_ratios=[3, 0.8, 0.3, 0.3], width_ratios=[3, 1],
                          hspace=0.05)  # Extremely minimal vertical spacing between subplots

    ax_main = fig.add_subplot(gs[0, :])
    ax_stats = fig.add_subplot(gs[1, 0])
    ax_payoff = fig.add_subplot(gs[1, 1])
    ax_pop_stats_display = fig.add_subplot(gs[2, :])
    ax_pop_stats_all = fig.add_subplot(gs[3, :])

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
    # --- INSERT START ---

    # Plot Barrier Up
    if hasattr(config, 'barriers_up'):
        # Extract value safely (handle list or scalar)
        b_up = config.barriers_up[0] if isinstance(config.barriers_up, (list, tuple)) else config.barriers_up

        # Check range condition (between 2 and 900)
        if b_up is not None and 2 <= b_up <= 900:
            ax_main.axhline(y=b_up, color='tab:red', linestyle='-.', linewidth=2,
                            label=f'Barrier Up = {b_up:.1f}', alpha=0.7)

    # Plot Barrier Down
    if hasattr(config, 'barriers_down'):
        # Extract value safely (handle list or scalar)
        b_down = config.barriers_down[0] if isinstance(config.barriers_down, (list, tuple)) else config.barriers_down

        # Check range condition (between 2 and 900)
        if b_down is not None and 2 <= b_down <= 900:
            ax_main.axhline(y=b_down, color='tab:green', linestyle='-.', linewidth=2,
                            label=f'Barrier Down = {b_down:.1f}', alpha=0.7)

    # --- INSERT END ---
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

    # Stats text (current frame statistics)
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

    # Population statistics text for DISPLAYED paths
    pop_stats_text_display = ax_pop_stats_display.text(
        0.5, 0.0, '', transform=ax_pop_stats_display.transAxes,
        fontsize=11, verticalalignment='bottom',
        horizontalalignment='center',
        fontfamily='monospace')
    pop_stats_str_display = (
        f"DISPLAYED PATHS (n={pop_stats_display['nb_paths']:,}) | "
        f"Avg Ex Time: {pop_stats_display['avg_ex_time']:.4f} | "
        f"@ Maturity: {pop_stats_display['pct_at_maturity']:.1f}% | "
        f"Avg Payoff: {pop_stats_display['avg_payoff']:.2f} ± {pop_stats_display['std_payoff']:.2f} | "
        f"Median: {pop_stats_display['median_payoff']:.2f}")
    pop_stats_text_display.set_text(pop_stats_str_display)
    ax_pop_stats_display.axis('off')

    # Population statistics text for ALL paths
    pop_stats_text_all = ax_pop_stats_all.text(
        0.5, 1.0, '', transform=ax_pop_stats_all.transAxes,
        fontsize=11, verticalalignment='top',
        horizontalalignment='center',
        fontfamily='monospace')
    pop_stats_str_all = (
        f"ALL PATHS (n={pop_stats_all['nb_paths']:,}) | "
        f"Avg Ex Time: {pop_stats_all['avg_ex_time']:.4f} | "
        f"@ Maturity: {pop_stats_all['pct_at_maturity']:.1f}% | "
        f"Avg Payoff: {pop_stats_all['avg_payoff']:.2f} ± {pop_stats_all['std_payoff']:.2f} | "
        f"Median: {pop_stats_all['median_payoff']:.2f}")
    pop_stats_text_all.set_text(pop_stats_str_all)
    ax_pop_stats_all.axis('off')

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
        # pop_stats_text stays constant (already set above)
        return [l for path_lines in lines for l in path_lines] + \
               [m for markers in exercise_markers for m in markers] + \
               [stats_text, payoff_line, pop_stats_text_display, pop_stats_text_all]

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
               [stats_text, payoff_line, pop_stats_text_display, pop_stats_text_all]

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

    # Load config
    print(f"Loading config: {args.configs}")
    if not hasattr(configs, args.configs):
        raise ValueError(f"Config '{args.configs}' not found in configs.py")
    config = getattr(configs, args.configs)

    # Validate config
    validate_config(config)

    # Determine number of paths
    nb_stocks = config.nb_stocks if isinstance(config.nb_stocks, int) else config.nb_stocks[0]
    nb_paths_total = config.nb_paths[0] if isinstance(config.nb_paths, (list, tuple)) else config.nb_paths

    if args.nb_paths_to_plot is None:
        nb_paths_to_plot = max(200, 10 * nb_stocks)
    else:
        nb_paths_to_plot = args.nb_paths_to_plot

    print(f"Will run backward induction on {nb_paths_total:,} paths and display {nb_paths_to_plot} in video")

    # Send start notification
    if TELEGRAM_ENABLED and args.send_telegram:
        try:
            SBM.send_notification(
                token=args.telegram_token,
                text=f'🎬 Starting video creation...\n\n'
                     f'Config: {args.configs}\n'
                     f'Total paths: {nb_paths_total:,}\n'
                     f'Displayed: {nb_paths_to_plot}\n'
                     f'Stocks: {nb_stocks}',
                chat_id=args.telegram_chat_id
            )
        except Exception as e:
            print(f"⚠️  Telegram notification failed: {e}")

    # Run algorithm
    (stock_paths, exercise_times, payoff_values,
     exercise_times_all, payoff_values_all, algo_name, payoff_name) = \
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
        exercise_times_all, payoff_values_all,
        algo_name, payoff_name, output_path
    )

    # Extract nb_dates for normalization
    nb_dates = config.nb_dates[0] if isinstance(config.nb_dates, (list, tuple)) else config.nb_dates

    print(f"\n✓ Done! Video saved to: {output_path}")
    print(f"  Algo: {algo_name}")
    print(f"  Payoff: {payoff_name}")

    # Population statistics for DISPLAYED paths
    normalized_ex_times = exercise_times / nb_dates
    exercised_at_maturity = (exercise_times == nb_dates).sum()
    nb_displayed = len(exercise_times)

    # Population statistics for ALL paths
    normalized_ex_times_all = exercise_times_all / nb_dates
    exercised_at_maturity_all = (exercise_times_all == nb_dates).sum()
    nb_total = len(exercise_times_all)

    print(f"\n{'='*70}")
    print(f"DISPLAYED PATHS STATISTICS (n={nb_displayed:,} paths)")
    print(f"{'='*70}")
    print(f"  Avg Exercise Time:     {normalized_ex_times.mean():.4f} (normalized 0-1)")
    print(f"  Exercise @ Maturity:   {exercised_at_maturity} ({100*exercised_at_maturity/nb_displayed:.1f}%)")
    print(f"  Avg Payoff:            {payoff_values.mean():.4f} ± {payoff_values.std():.4f}")
    print(f"  Median Payoff:         {np.median(payoff_values):.4f}")
    print(f"{'='*70}")

    print(f"\n{'='*70}")
    print(f"ALL PATHS STATISTICS (n={nb_total:,} paths)")
    print(f"{'='*70}")
    print(f"  Avg Exercise Time:     {normalized_ex_times_all.mean():.4f} (normalized 0-1)")
    print(f"  Exercise @ Maturity:   {exercised_at_maturity_all} ({100*exercised_at_maturity_all/nb_total:.1f}%)")
    print(f"  Avg Payoff:            {payoff_values_all.mean():.4f} ± {payoff_values_all.std():.4f}")
    print(f"  Median Payoff:         {np.median(payoff_values_all):.4f}")
    print(f"{'='*70}\n")

    # Send completion notification with video
    if TELEGRAM_ENABLED and args.send_telegram:
        try:
            SBM.send_notification(
                token=args.telegram_token,
                text=f'✅ Video creation complete!\n\n'
                     f'Config: {args.configs}\n'
                     f'Algo: {algo_name}\n'
                     f'Payoff: {payoff_name}\n'
                     f'Displayed: {nb_displayed:,} paths\n'
                     f'Total: {nb_total:,} paths\n\n'
                     f'ALL PATHS STATS:\n'
                     f'Avg exercise: {normalized_ex_times_all.mean():.4f}\n'
                     f'@ Maturity: {100*exercised_at_maturity_all/nb_total:.1f}%\n'
                     f'Avg payoff: {payoff_values_all.mean():.2f}',
                files=[str(output_path)],
                chat_id=args.telegram_chat_id
            )
        except Exception as e:
            print(f"⚠️  Telegram notification failed: {e}")


if __name__ == '__main__':
    main()
