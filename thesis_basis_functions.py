import numpy as np
import pandas as pd
from scipy.special import legendre, chebyt, laguerre, hermite
import time

# ==========================================
# 1. Configuration & Parameters
# ==========================================
PARAMS = {
    'S0_list': [90, 100, 110],
    'dims_list': [2, 10, 25],
    'basis_funcs': ['Monomial', 'Laguerre', 'Hermite', 'Legendre', 'Chebyshev'],
    'degrees_list': [2, 3],  # <--- UPDATED: Iterate over degrees
    'n_sets': 10,  # Number of Monte Carlo sets (CRN)
    'n_paths': 25000,  # Paths per set
    'r': 0.01,
    'sigma': 0.40,
    'K': 100,
    'T': 1.0,
    'N_steps': 50,  # <--- UPDATED: Reduced time steps
    'drift_param': 0.03,
    'risk_neutral': True
}

# ==========================================
# 2. Pre-Generate Random Increments (CRN)
# ==========================================
print(f"Pre-generating {PARAMS['n_sets']} sets of Brownian increments...")
# We generate enough dimensions for the maximum case (25)
MAX_DIM = max(PARAMS['dims_list'])
np.random.seed(42)
Z_SETS = np.random.normal(0, 1, (PARAMS['n_sets'], PARAMS['n_paths'], PARAMS['N_steps'], MAX_DIM))
print("Random increments generated. Starting simulation loops...\n")


# ==========================================
# 3. Helper Functions
# ==========================================
def generate_paths(Z, S0, dim, r, sigma, T, N, drift, risk_neutral):
    """ Converts Brownian increments Z into Asset Paths S. """
    M = Z.shape[0]
    dt = T / N
    mu = r if risk_neutral else drift

    S = np.zeros((M, N + 1, dim))
    S[:, 0, :] = S0

    Z_sliced = Z[:, :, :dim]

    for t in range(N):
        S[:, t + 1, :] = S[:, t, :] * np.exp((mu - 0.5 * sigma ** 2) * dt +
                                             sigma * np.sqrt(dt) * Z_sliced[:, t, :])
    return S


def generate_basis_matrix(X, basis_type, degree):
    """ Constructs the matrix of basis functions for a given degree. """
    # Scaling to prevent numerical explosion
    if basis_type in ['Legendre', 'Chebyshev']:
        epsilon = 1e-6
        X_min, X_max = X.min(), X.max()
        X_scaled = 2 * (X - X_min) / (X_max - X_min + epsilon) - 1
    elif basis_type == 'Hermite':
        X_scaled = (X - X.mean()) / (X.std() + 1e-6)
    else:
        X_scaled = X / 100.0

    n_samples = len(X)
    basis_mat = np.ones((n_samples, degree + 1))

    for d in range(1, degree + 1):
        if basis_type == 'Monomial':
            basis_mat[:, d] = X_scaled ** d
        elif basis_type == 'Chebyshev':
            basis_mat[:, d] = chebyt(d)(X_scaled)
        elif basis_type == 'Legendre':
            basis_mat[:, d] = legendre(d)(X_scaled)
        elif basis_type == 'Laguerre':
            basis_mat[:, d] = laguerre(d)(np.abs(X_scaled))
        elif basis_type == 'Hermite':
            basis_mat[:, d] = hermite(d)(X_scaled)

    return basis_mat


def run_lsm_algo(S, K, r, T, N, basis_type, degree):
    """ Performs purely the Backward Induction (Regression). """
    start_time = time.perf_counter()

    dt = T / N
    df = np.exp(-r * dt)
    M = S.shape[0]

    # Payoff Calculation (Basket Arithmetic Mean)
    B = np.mean(S, axis=2)
    h = np.maximum(K - B, 0)

    V = np.zeros_like(h)
    V[:, -1] = h[:, -1]

    for t in range(N - 1, 0, -1):
        itm_mask = h[:, t] > 0
        if np.sum(itm_mask) == 0:
            V[:, t] = V[:, t + 1] * df
            continue

        X_itm = B[itm_mask, t]
        Y_itm = V[itm_mask, t + 1] * df

        # Basis Matrix Construction (Uses current degree)
        A = generate_basis_matrix(X_itm, basis_type, degree)

        try:
            beta, _, _, _ = np.linalg.lstsq(A, Y_itm, rcond=None)
            continuation_value = np.dot(A, beta)
        except:
            continuation_value = 0

        exercise = h[itm_mask, t] > continuation_value

        V[:, t] = V[:, t + 1] * df
        full_indices = np.where(itm_mask)[0]
        exercise_indices = full_indices[exercise]
        V[exercise_indices, t] = h[exercise_indices, t]

    price = np.mean(V[:, 1] * df)
    end_time = time.perf_counter()
    return price, (end_time - start_time)


# ==========================================
# 4. Main Simulation Loop
# ==========================================
results = []

# Loop over Scenarios
for dim in PARAMS['dims_list']:
    for S0 in PARAMS['S0_list']:

        # 4a. Pre-calculate the 10 sets of Asset Paths (S)
        # All degrees and basis functions below will use THESE EXACT paths
        print(f"Generating paths for Dim={dim}, S0={S0}...")
        scenario_paths = []
        for i in range(PARAMS['n_sets']):
            Z_set = Z_SETS[i]
            path_S = generate_paths(Z_set, S0, dim, PARAMS['r'], PARAMS['sigma'],
                                    PARAMS['T'], PARAMS['N_steps'],
                                    PARAMS['drift_param'], PARAMS['risk_neutral'])
            scenario_paths.append(path_S)

        # 4b. Loop over Basis Functions
        for basis in PARAMS['basis_funcs']:

            # 4c. Loop over Degrees (NEW LOOP)
            for deg in PARAMS['degrees_list']:

                run_prices = []
                run_times = []

                # 4d. Execute 10 runs using the 10 pre-calculated path sets
                for i in range(PARAMS['n_sets']):
                    price, algo_time = run_lsm_algo(
                        scenario_paths[i], PARAMS['K'], PARAMS['r'],
                        PARAMS['T'], PARAMS['N_steps'], basis, deg
                    )
                    run_prices.append(price)
                    run_times.append(algo_time)

                # 4e. Aggregation
                avg_price = np.mean(run_prices)
                std_price = np.std(run_prices, ddof=1)
                avg_time = np.mean(run_times)

                results.append({
                    'Dim': dim,
                    'S0': S0,
                    'Basis': basis,
                    'Degree': deg,
                    'Mean Price': avg_price,
                    'Std Dev': std_price,
                    'Avg Time (s)': avg_time
                })

# ==========================================
# 5. Reporting
# ==========================================
df_final = pd.DataFrame(results)

print("\n=== Simulation Complete ===")
print(df_final.head(10))

# Save to CSV
df_final.to_csv("lsm_basis_comparison_deg2_3.csv", index=False)
print("\nFull results saved to 'lsm_basis_comparison_deg2_3.csv'")