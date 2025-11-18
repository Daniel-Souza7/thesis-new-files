"""Read data from CSVs according to passed flags."""

import os
import pandas as pd

from optimal_stopping.run import configs
from optimal_stopping.utilities import filtering

# INDEX must match CSV columns for multi-index creation
# Based on actual CSV structure from output
INDEX = [
    "algo", "model", "payoff", "drift", "risk_free_rate", "volatility",
    "mean", "speed", "correlation", "hurst",
    "nb_stocks", "nb_paths", "nb_dates", "spot", "strike",
    "dividend", "barrier", "maturity",
    "nb_epochs", "hidden_size", "factors", "ridge_coeff",
    "use_payoff_as_input", "train_ITM_only", "barriers_up", "barriers_down",
    "alpha", "k", "weights", "step_param1", "step_param2", "step_param3", "step_param4"
]

old_new_algo_dict = {
    "L": "NLSM",
    "FQIRfast": "RFQI",
    "LNDfast": "RLSM",
    "FQIfast": "FQI",
    "LS": "LSM",
    "DO": "DOS",
    "randRNN": "RRLSM",
    "FQIRfastRNN": "RRFQI"
}


def replace_old_algo_names():
    csv_paths = get_csv_paths() + get_csv_paths_draft()
    for f in csv_paths:
        df = pd.read_csv(f, index_col=None)
        df.replace(to_replace={"algo": old_new_algo_dict}, inplace=True)
        df.to_csv(f, index=False)


def get_csv_paths():
    csvs_dir = os.path.join(os.path.dirname(__file__), "../../output/metrics")
    if not os.path.exists(csvs_dir):
        os.makedirs(csvs_dir)
    return [os.path.join(csvs_dir, fname)
            for fname in sorted(os.listdir(csvs_dir))
            if fname.endswith('.csv')]


def get_csv_paths_draft():
    csvs_dir = os.path.join(os.path.dirname(__file__), "../../output/metrics_draft")
    if not os.path.exists(csvs_dir):
        os.makedirs(csvs_dir)
    return [os.path.join(csvs_dir, fname)
            for fname in sorted(os.listdir(csvs_dir))
            if fname.endswith('.csv')]


def read_csv(path: str, config: configs._DefaultConfig,
             reverse_filtering: bool = False):
    """Reads one CSV and filters out unwanted values."""
    try:
        df = pd.read_csv(path, index_col=INDEX)
    except Exception:
        df = pd.read_csv(path, index_col=None)
        for col in INDEX:
            if col not in df.columns:
                df[col] = None
        df.to_csv(path, index=False)
        df = pd.read_csv(path, index_col=INDEX)

    return filtering.filter_df(df, config, reverse_filtering)


def read_csvs_conv(which=0):
    """Read CSVs for convergence studies without filtering."""
    if which == 0:
        csv_paths = get_csv_paths_draft()
    elif which == 1:
        csv_paths = get_csv_paths()
    else:
        csv_paths = get_csv_paths_draft() + get_csv_paths()

    if not csv_paths:
        return pd.DataFrame()

    df = pd.concat(pd.read_csv(path, index_col=None) for path in csv_paths)
    return df


def read_csvs(config: configs._DefaultConfig, remove_duplicates: bool = True):
    """Returns dataframe with all CSV(s) content, filtered according to config."""
    csv_paths = get_csv_paths() + get_csv_paths_draft()
    print(f"Reading data from {len(csv_paths)} CSV files...")

    dfs = []
    for path in csv_paths:
        try:
            df = read_csv(path, config)
            if len(df) > 0:
                dfs.append(df)
        except Exception as e:
            print(f"⚠️ Skipping {os.path.basename(path)}: {e}")

    if not dfs:
        print("⚠️ No data matched filters")
        print(f"   Config filters: algos={config.algos}, payoffs={config.payoffs}")
        raise AssertionError("No data read with given filters...")

    df = pd.concat(dfs)

    if remove_duplicates:
        df = df[~df.index.duplicated(keep='last')]

    print(f"✅ Loaded {len(df)} rows")
    return df


def extract_single_value_indexes(df):
    """Returns (df - single value indexes, description of removed params)."""
    global_params = []
    for index_name in df.index.names:
        # Skip None index names (unnamed indices)
        if index_name is None:
            continue

        values = df.index.get_level_values(index_name)
        if len(values.unique()) == 1:
            global_params.append(f"{index_name} = {values[0]}")
            df = df.reset_index(index_name)
            # Only drop if the column exists (reset_index creates it)
            if index_name in df.columns:
                df = df.drop(columns=index_name)
    global_params_caption = ", ".join(global_params)
    return df, global_params_caption