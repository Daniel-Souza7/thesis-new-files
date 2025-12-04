import numpy as np
import matplotlib.pyplot as plt

# --- Simulation Parameters (Using updated values from generate_simulations.py) ---
NB_DATES = 50  # Number of time steps in each path
MATURITY = 1.0  # Total time period (in years)
DRIFT = 0.05  # Mu (expected return or drift)
VOLATILITY = 0.3  # Sigma (volatility)
S0 = 100.0  # Initial Stock Price
NB_PATHS = 10000  # The total number of Monte Carlo paths generated

# Parameters for the plot visualization
NB_COLORED_PATHS = 10000  # Number of paths to plot individually with color
NEW_ALPHA = 0.25  # Increased transparency for better visibility

# Calculate time step size
dt = MATURITY / NB_DATES
time_array = np.linspace(0, MATURITY, NB_DATES + 1)


# --- Vectorized Geometric Brownian Motion (GBM) Simulation ---

def run_monte_carlo_gbm(S0, drift, volatility, dt, nb_dates, nb_paths):
    """
    Generates all GBM paths simultaneously using vectorized numpy operations
    for efficiency. The result is an array of shape (nb_dates + 1, nb_paths).
    """
    # 1. Generate the random components (Z)
    # Z has shape (NB_DATES, NB_PATHS)
    np.random.seed(42)  # Ensure the simulation is reproducible
    Z = np.random.standard_normal((nb_dates, nb_paths))

    # 2. Calculate the deterministic and stochastic components of the log returns
    drift_term = (drift - 0.5 * volatility ** 2) * dt
    shock_term = volatility * np.sqrt(dt) * Z

    # Total log returns (shape: NB_DATES, NB_PATHS)
    log_returns = drift_term + shock_term

    # 3. Calculate the cumulative sum of log returns over time (axis=0)
    # This gives us the log of the price ratio S(T)/S0
    # Add a row of zeros at the start for t=0 (log(S0/S0) = 0)
    log_cumulative_returns = np.vstack([np.zeros(nb_paths), np.cumsum(log_returns, axis=0)])

    # 4. Calculate the price path: S(t) = S0 * exp(log_cumulative_returns)
    price_paths = S0 * np.exp(log_cumulative_returns)

    return price_paths


# Run the simulation
all_paths = run_monte_carlo_gbm(S0, DRIFT, VOLATILITY, dt, NB_DATES, NB_PATHS)

# --- Plotting the Results ---

plt.figure(figsize=(12, 7))

# Plot a subset of paths individually to utilize Matplotlib's default color cycle
# and increase visibility/reduce transparency.
for i in range(NB_COLORED_PATHS):
    # Plotting column i (the i-th path)
    if i == 0:
        # Only label the first path for the legend
        plt.plot(time_array, all_paths[:, i],
                 linewidth=1, alpha=NEW_ALPHA,
                 label=f'{NB_COLORED_PATHS} Sampled Paths (Colored)')
    else:
        # Plot subsequent paths without a label
        plt.plot(time_array, all_paths[:, i],
                 linewidth=1, alpha=NEW_ALPHA)

# 2. Plot Initial Price
plt.scatter([0], [S0], color='black', s=50, zorder=10, label='Initial Price ($S_0$)')

# Finalize the plot look
plt.title(f'Monte Carlo Simulation: {NB_COLORED_PATHS} Individual Paths (Sample)', fontsize=18)
plt.xlabel('Time (Years)', fontsize=14)
plt.ylabel('Asset Price ($)', fontsize=14)
plt.grid(True, linestyle=':', alpha=0.8)

plt.legend(loc='upper left', fontsize=10)

plt.tight_layout()

# Save the plot
filename = "plot_1000_colored_paths_sample.png"
plt.savefig(filename)
plt.close()

print(f"Generated {filename}")
print("\nSimulation complete. The plot now shows a colorful sample of 10000 individual paths with increased visibility.")