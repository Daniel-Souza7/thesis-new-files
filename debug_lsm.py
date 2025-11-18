"""
Debug LSM implementation - Optimized version based on Longstaff-Schwartz (2001)
Outputs: American option price and mean exercise time
"""

import numpy as np
from sklearn.linear_model import LinearRegression

# Set seed for reproducibility
np.random.seed(123456789)

# Parameters
REPS = 10000        # Number of simulations
T = 1.0             # Maturity (years) - changed to 1 to match your configs
DT = 1/10           # Time step - changed to match nb_dates=10
TIME = np.arange(DT, T + DT, DT)
N = len(TIME)       # Number of time intervals
DRIFT = 0.06        # Drift for path generation (matching your config)
R = 0.02            # Risk-free rate for discounting (matching drift - 0.04)
SIGMA = 0.2         # Volatility
D = 0.0             # Dividend yield
S_0 = 100.0         # Initial stock price
K = 100.0           # Strike price


def generate_gbm_paths(n_paths, n_steps, s0, drift, sigma, dt, d=0.0):
    """
    Generate stock price paths using Geometric Brownian Motion.

    Args:
        drift: Drift parameter for path generation (mu)

    Returns:
        S: (n_paths, n_steps+1) array of stock prices
    """
    # Pre-allocate
    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = s0

    # Generate all random numbers at once
    Z = np.random.normal(0, 1, (n_paths, n_steps))

    # Vectorized GBM
    drift_term = (drift - d - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    increments = np.exp(drift_term + diffusion * Z)
    S[:, 1:] = s0 * np.cumprod(increments, axis=1)

    return S


def lsm_american_option(S, K, r, dt, option_type='put'):
    """
    Price American option using Least Squares Monte Carlo.

    Args:
        S: (n_paths, n_steps+1) stock price paths
        K: Strike price
        r: Risk-free rate
        dt: Time step
        option_type: 'call' or 'put'

    Returns:
        price: Option price
        mean_exercise_time: Average exercise time (normalized 0-1)
    """
    n_paths, n_steps = S.shape
    n_steps -= 1  # Adjust for 0-indexed

    # Initialize value and exercise time tracking
    value = np.zeros((n_paths, n_steps + 1))
    exercise_times = np.full(n_paths, n_steps, dtype=int)  # Default to maturity

    # Payoff function
    if option_type == 'call':
        payoff = lambda s: np.maximum(s - K, 0)
    else:  # put
        payoff = lambda s: np.maximum(K - s, 0)

    # Terminal payoff
    value[:, n_steps] = payoff(S[:, n_steps])

    # Backward induction
    disc_factor = np.exp(-r * dt)

    for t in range(n_steps - 1, 0, -1):
        # Current stock price and payoff
        S_t = S[:, t]
        immediate_payoff = payoff(S_t)

        # Find in-the-money paths
        itm = immediate_payoff > 0

        if itm.sum() == 0:
            # No ITM paths, just discount
            value[:, t] = disc_factor * value[:, t + 1]
            continue

        # Regression on ITM paths only
        X = np.column_stack([S_t[itm], S_t[itm]**2])  # Basis: [S, S^2]
        Y = disc_factor * value[itm, t + 1]  # Continuation value

        # Fit regression
        try:
            reg = LinearRegression(fit_intercept=True)
            reg.fit(X, Y)
            continuation_value_itm = reg.predict(X)
        except:
            # Fallback if regression fails
            continuation_value_itm = Y

        # Initialize continuation values to 0 (like your fixed code)
        continuation_value = np.zeros(n_paths)
        continuation_value[itm] = continuation_value_itm

        # Exercise decision: exercise if immediate > continuation
        exercise_now = immediate_payoff > continuation_value

        # Update values
        value[:, t] = disc_factor * value[:, t + 1]
        value[exercise_now, t] = immediate_payoff[exercise_now]

        # Track exercise times (only update if exercising earlier)
        exercise_times[exercise_now] = t

        # Zero out future values for exercised paths
        value[exercise_now, t+1:] = 0

    # Time 0: discount from time 1
    value[:, 0] = disc_factor * value[:, 1]

    # Price is mean of value at time 0
    price = np.mean(value[:, 0])

    # Mean exercise time (normalized to [0, 1])
    mean_exercise_time = np.mean(exercise_times) / n_steps

    return price, mean_exercise_time, value, exercise_times


def main():
    """Run LSM and print results."""
    print("=" * 80)
    print("LSM American Option Pricing - Debug Version")
    print("=" * 80)
    print(f"\nParameters:")
    print(f"  Paths:        {REPS:,}")
    print(f"  Time steps:   {N}")
    print(f"  Maturity:     {T} years")
    print(f"  Strike:       {K}")
    print(f"  Spot:         {S_0}")
    print(f"  Drift:        {DRIFT} (path generation)")
    print(f"  Rate:         {R} (discounting)")
    print(f"  Volatility:   {SIGMA}")
    print(f"  Dividend:     {D}")

    # Generate paths
    print(f"\nGenerating {REPS:,} stock price paths...")
    S = generate_gbm_paths(REPS, N, S_0, DRIFT, SIGMA, DT, D)

    # Price American put
    print("Running LSM backward induction...")
    price, mean_ex_time, value, ex_times = lsm_american_option(S, K, R, DT, option_type='put')

    # Results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"American Put Price:     {price:.4f}")
    print(f"Mean Exercise Time:     {mean_ex_time:.4f}")
    print(f"Exercise Time Std:      {np.std(ex_times / N):.4f}")
    print("=" * 80)

    # Exercise time distribution
    print(f"\nExercise Time Distribution:")
    unique, counts = np.unique(ex_times, return_counts=True)
    for t, count in zip(unique, counts):
        pct = 100 * count / REPS
        print(f"  t={t:2d} ({t/N:.2f}): {count:5d} paths ({pct:5.2f}%)")

    # Sample paths
    print(f"\nFirst 5 path exercise times: {ex_times[:5]}")
    print(f"First 5 path values at t=0:  {value[:5, 0]}")

    return price, mean_ex_time, S, value, ex_times


if __name__ == "__main__":
    price, mean_ex_time, S, value, ex_times = main()
