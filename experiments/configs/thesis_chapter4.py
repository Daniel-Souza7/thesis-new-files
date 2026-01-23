"""
Thesis Chapter 4 experiment configurations.

These configurations reproduce the benchmark experiments from Chapter 4
of the thesis. Each configuration corresponds to a specific table or figure.
"""

# Import will work after the package is installed
# For now, configurations are dictionaries that can be loaded by run_algo.py


# Table 4.2: Algorithmic comparison across dimensions
# Tests basket call pricing for d = {1, 2, 7, 50, 500}
thesis_table_4_2 = {
    'name': 'thesis_table_4_2',
    'description': 'Algorithmic comparison across dimensions (Table 4.2)',

    # Algorithms to compare
    'algos': ['RT', 'RLSM', 'LSM', 'DOS', 'NLSM', 'FQI', 'EOP'],

    # Model
    'stock_models': ['BlackScholes'],

    # Payoff
    'payoffs': ['BasketCall'],

    # Dimensions
    'nb_stocks': [1, 2, 7, 50, 500],

    # Monte Carlo parameters
    'nb_paths': [8000000, 8000000, 14000000, 10000000, 10000000],  # per dimension
    'nb_dates': [100],
    'nb_runs': 1,

    # Market parameters
    'drift': 0.08,
    'volatility': 0.2,
    'dividend': 0.0,
    'spot': 100,
    'strike': 100,
    'maturity': 1.0,
    'correlation': 0.0,

    # Algorithm parameters
    'hidden_size': [20],  # RT uses adaptive sizing
    'activation': 'leakyrelu',

    # Output
    'output_dir': 'experiments/results/table_4_2',
}


# Table 4.3: MaxCall activation function validation
# Compares RT (ELU) vs RLSM (LeakyReLU) on MaxCall options
thesis_table_4_3 = {
    'name': 'thesis_table_4_3',
    'description': 'MaxCall activation function validation (Table 4.3)',

    'algos': ['RT', 'RLSM', 'EOP'],
    'stock_models': ['BlackScholes'],
    'payoffs': ['MaxCall'],
    'nb_stocks': [5, 25, 250],
    'nb_paths': [10000000],
    'nb_dates': [100],
    'nb_runs': 1,

    'drift': 0.08,
    'volatility': 0.2,
    'dividend': 0.0,
    'spot': 100,
    'strike': 100,
    'maturity': 1.0,
    'correlation': 0.0,

    # RT uses ELU for MaxCall, RLSM uses LeakyReLU
    'hidden_size': [20],

    'output_dir': 'experiments/results/table_4_3',
}


# Tables 4.5-4.7: Barrier option validation
thesis_barriers = {
    'name': 'thesis_barriers',
    'description': 'Barrier option convergence and monotonicity (Tables 4.5-4.7)',

    'algos': ['RT', 'EOP'],
    'stock_models': ['BlackScholes'],

    # Test various barrier configurations
    'payoffs': [
        'UO_BasketCall',  # Up-and-Out
        'DO_MaxCall',     # Down-and-Out
        'UI_BasketPut',   # Up-and-In
        'DI_BasketPut',   # Down-and-In
    ],

    'nb_stocks': [5, 25],
    'nb_paths': [10000000],
    'nb_dates': [100],
    'nb_runs': 1,

    'drift': 0.08,
    'volatility': 0.2,
    'spot': [90, 100, 110],  # Multiple moneyness levels
    'strike': 100,
    'maturity': 1.0,

    # Barrier levels for sweep
    'barrier_levels': [0.001, 30, 50, 70, 80, 100, 120, 150, 200, 1000],

    'output_dir': 'experiments/results/barriers',
}


# Table 4.8: Path-dependent performance (RT vs RRLSM)
thesis_path_dependent = {
    'name': 'thesis_path_dependent',
    'description': 'Path-dependent option pricing: RT vs RRLSM (Table 4.8)',

    'algos': ['RT', 'RRLSM'],
    'stock_models': ['BlackScholes'],

    'payoffs': [
        # Lookback
        'LookbackFixedCall',
        'LookbackFloatingPut',
        # Asian
        'AsianFixedPut',
        'AsianFloatingCall',
        # Barrier + exotic combinations
        'UO_DispersionCall',
        'DO_BestOfKCall',
        'UI_MinPut',
        'DI_MaxCall',
        # Complex barriers
        'DStepB_MinPut',
        'DKI_WorstOfKPut',
        'UODI_RankCall',
        'StepB_MaxDispersionCall',
    ],

    'nb_stocks': [1, 5, 25, 50, 250, 500],
    'nb_paths': [10000000],
    'nb_dates': [100],
    'nb_runs': 1,

    'drift': 0.05,
    'volatility': 0.2,
    'spot': 100,
    'strike': 100,
    'maturity': 1.0,

    'output_dir': 'experiments/results/path_dependent',
}


# Convergence study configuration (Figure 4.1)
thesis_convergence_study = {
    'name': 'thesis_convergence_study',
    'description': 'Monte Carlo convergence study (Figure 4.1)',

    'algos': ['EOP'],  # Use EOP for pure MC variance
    'stock_models': ['BlackScholes'],
    'payoffs': ['BasketCall'],
    'nb_stocks': [2, 50],

    # Path counts for convergence
    'nb_paths': [20000, 40000, 80000, 160000, 320000, 640000,
                 1280000, 2560000, 5120000, 10240000, 20480000],
    'nb_dates': [5],
    'nb_runs': 1000,  # Many runs for variance estimation

    'drift': 0.02,
    'volatility': 0.2,
    'spot': 100,
    'strike': 100,
    'maturity': 1.0,

    'output_dir': 'experiments/results/convergence',
}


# All configurations
ALL_CONFIGS = {
    'thesis_table_4_2': thesis_table_4_2,
    'thesis_table_4_3': thesis_table_4_3,
    'thesis_barriers': thesis_barriers,
    'thesis_path_dependent': thesis_path_dependent,
    'thesis_convergence_study': thesis_convergence_study,
}


def get_config(name: str) -> dict:
    """Get configuration by name."""
    if name not in ALL_CONFIGS:
        available = ', '.join(sorted(ALL_CONFIGS.keys()))
        raise ValueError(f"Unknown config '{name}'. Available: {available}")
    return ALL_CONFIGS[name]
