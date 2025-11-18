"""
Debug script to investigate why exercise times differ between run_algo and create_video.

Compares:
1. Exercise times from backward induction (training)
2. Exercise times from forward simulation (predict)
"""

import numpy as np
import sys
from pathlib import Path

# Add optimal_stopping to path
sys.path.insert(0, str(Path(__file__).parent))

from optimal_stopping.algorithms.standard.lsm import LeastSquaresPricer
from optimal_stopping.data.stock_model import BlackScholes
from optimal_stopping.payoffs.single_simple import Call


def debug_exercise_times():
    """Compare exercise times from training vs prediction."""

    # Parameters (matching video_testing2)
    nb_paths = 40000
    nb_stocks = 1
    nb_dates = 10
    maturity = 1.0
    spot = 100.0
    strike = 100.0
    drift = 0.06
    volatility = 0.2
    rate = 0.02
    train_ITM_only = True
    seed = 42

    print("=" * 80)
    print("EXERCISE TIME DEBUG: TRAINING VS PREDICTION")
    print("=" * 80)
    print(f"\nParameters:")
    print(f"  Paths:          {nb_paths:,}")
    print(f"  Stocks:         {nb_stocks}")
    print(f"  Time steps:     {nb_dates}")
    print(f"  Maturity:       {maturity}")
    print(f"  Spot:           {spot}")
    print(f"  Strike:         {strike}")
    print(f"  Drift:          {drift}")
    print(f"  Volatility:     {volatility}")
    print(f"  Rate:           {rate}")
    print(f"  Train ITM only: {train_ITM_only}")
    print(f"  Seed:           {seed}")

    np.random.seed(seed)

    # Create model and payoff
    model = BlackScholes(
        drift=drift,
        volatility=volatility,
        nb_stocks=nb_stocks,
        nb_paths=nb_paths,
        nb_dates=nb_dates,
        spot=spot,
        dividend=0.0,
        maturity=maturity
    )

    payoff = Call(strike=strike)

    # Create LSM pricer
    lsm = LeastSquaresPricer(
        model=model,
        payoff=payoff,
        train_ITM_only=train_ITM_only
    )

    # ========================================
    # PHASE 1: TRAIN (like run_algo does)
    # ========================================
    print("\n" + "-" * 80)
    print("PHASE 1: TRAINING (Backward Induction)")
    print("-" * 80)

    price, time_gen = lsm.price()
    ex_time_training = lsm.get_exercise_time()

    print(f"Price:              {price:.4f}")
    print(f"Exercise Time:      {ex_time_training:.4f} (from evaluation set)")

    # Get detailed exercise time distribution from training
    training_ex_dates = lsm._exercise_dates[lsm.split:]  # evaluation set only
    exercised_at_maturity_train = (training_ex_dates == nb_dates).sum()
    exercised_early_train = (training_ex_dates < nb_dates).sum()

    print(f"  Exercised @ maturity: {exercised_at_maturity_train} ({100*exercised_at_maturity_train/len(training_ex_dates):.1f}%)")
    print(f"  Exercised early:      {exercised_early_train} ({100*exercised_early_train/len(training_ex_dates):.1f}%)")

    # Show distribution of exercise times
    unique_times, counts = np.unique(training_ex_dates, return_counts=True)
    print(f"\n  Exercise Time Distribution (training eval set):")
    for t, count in zip(unique_times, counts):
        pct = 100 * count / len(training_ex_dates)
        print(f"    t={t:2d} ({t/nb_dates:.2f}): {count:5d} paths ({pct:5.2f}%)")

    # ========================================
    # PHASE 2: PREDICT ON SAME PATHS
    # ========================================
    print("\n" + "-" * 80)
    print("PHASE 2: PREDICTION ON SAME PATHS (Forward Simulation)")
    print("-" * 80)

    # Re-generate SAME paths (same seed)
    np.random.seed(seed)
    path_result = model.generate_paths()
    if isinstance(path_result, tuple):
        stock_paths, var_paths = path_result
    else:
        stock_paths = path_result
        var_paths = None

    # Apply learned policy
    ex_times_pred, payoffs_pred = lsm.predict(stock_paths)
    ex_time_pred_same = np.mean(ex_times_pred) / nb_dates

    print(f"Exercise Time:      {ex_time_pred_same:.4f} (from predict on same paths)")

    exercised_at_maturity_pred = (ex_times_pred == nb_dates).sum()
    exercised_early_pred = (ex_times_pred < nb_dates).sum()

    print(f"  Exercised @ maturity: {exercised_at_maturity_pred} ({100*exercised_at_maturity_pred/nb_paths:.1f}%)")
    print(f"  Exercised early:      {exercised_early_pred} ({100*exercised_early_pred/nb_paths:.1f}%)")

    # Show distribution
    unique_times, counts = np.unique(ex_times_pred, return_counts=True)
    print(f"\n  Exercise Time Distribution (predict on same paths):")
    for t, count in zip(unique_times, counts):
        pct = 100 * count / nb_paths
        print(f"    t={t:2d} ({t/nb_dates:.2f}): {count:5d} paths ({pct:5.2f}%)")

    # ========================================
    # PHASE 3: PREDICT ON NEW PATHS
    # ========================================
    print("\n" + "-" * 80)
    print("PHASE 3: PREDICTION ON NEW PATHS (like create_video)")
    print("-" * 80)

    # Generate NEW paths (different seed)
    np.random.seed(seed + 1000)
    path_result = model.generate_paths()
    if isinstance(path_result, tuple):
        stock_paths_new, _ = path_result
    else:
        stock_paths_new = path_result

    # Apply learned policy
    ex_times_new, payoffs_new = lsm.predict(stock_paths_new)
    ex_time_new = np.mean(ex_times_new) / nb_dates

    print(f"Exercise Time:      {ex_time_new:.4f} (from predict on new paths)")

    exercised_at_maturity_new = (ex_times_new == nb_dates).sum()
    exercised_early_new = (ex_times_new < nb_dates).sum()

    print(f"  Exercised @ maturity: {exercised_at_maturity_new} ({100*exercised_at_maturity_new/nb_paths:.1f}%)")
    print(f"  Exercised early:      {exercised_early_new} ({100*exercised_early_new/nb_paths:.1f}%)")

    # Show distribution
    unique_times, counts = np.unique(ex_times_new, return_counts=True)
    print(f"\n  Exercise Time Distribution (predict on new paths):")
    for t, count in zip(unique_times, counts):
        pct = 100 * count / nb_paths
        print(f"    t={t:2d} ({t/nb_dates:.2f}): {count:5d} paths ({pct:5.2f}%)")

    # ========================================
    # COMPARISON
    # ========================================
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"\nExercise Times:")
    print(f"  Training (eval set):      {ex_time_training:.4f}")
    print(f"  Predict (same paths):     {ex_time_pred_same:.4f}")
    print(f"  Predict (new paths):      {ex_time_new:.4f}")

    print(f"\nDifferences:")
    print(f"  Training vs Predict(same):  {abs(ex_time_training - ex_time_pred_same):.4f}")
    print(f"  Training vs Predict(new):   {abs(ex_time_training - ex_time_new):.4f}")
    print(f"  Predict(same) vs Predict(new): {abs(ex_time_pred_same - ex_time_new):.4f}")

    # Check if CALL option should exercise early
    print(f"\n" + "=" * 80)
    print(f"THEORETICAL EXPECTATION")
    print(f"=" * 80)
    print(f"\nFor American CALL with no dividends:")
    print(f"  - Optimal strategy: NEVER exercise early")
    print(f"  - Expected exercise time: 1.0 (always at maturity)")
    print(f"\nIf exercise time < 1.0, the learned policy is SUBOPTIMAL!")

    if ex_time_training < 0.999:
        print(f"\n⚠️  WARNING: Training exercise time = {ex_time_training:.4f} < 1.0")
        print(f"   This suggests a bug in the backward induction!")

    if ex_time_pred_same < 0.999:
        print(f"\n⚠️  WARNING: Predict (same paths) exercise time = {ex_time_pred_same:.4f} < 1.0")
        print(f"   This suggests a bug in the predict() method!")

    if ex_time_new < 0.999:
        print(f"\n⚠️  WARNING: Predict (new paths) exercise time = {ex_time_new:.4f} < 1.0")
        print(f"   This suggests the learned policy doesn't generalize well!")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    debug_exercise_times()
