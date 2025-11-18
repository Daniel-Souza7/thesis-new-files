"""Module to filter DataFrame based on config."""
from absl import flags
from optimal_stopping.run import configs

flags.DEFINE_bool(
    "debug_csv_filtering", False,
    "Set to True to display why rows are filtered out and a snippet of data."
)

FILTERS = [
    ("algos", "algo"),
    ("payoffs", "payoff"),
    ("dividends", "dividend"),
    ("spots", "spot"),
    ("volatilities", "volatility"),
    ("maturities", "maturity"),
    ("nb_paths", "nb_paths"),
    ("nb_dates", "nb_dates"),
    ("nb_stocks", "nb_stocks"),
    ("drift", "drift"),
    ("risk_free_rate", "risk_free_rate"),
    ("stock_models", "model"),
    ("strikes", "strike"),
    ("barriers", "barrier"),
    ("hurst", "hurst"),
    ("hidden_size", "hidden_size"),
    ("factors", "factors"),
    ("ridge_coeff", "ridge_coeff"),
    ("use_payoff_as_input", "use_payoff_as_input"),
    ("barriers_up", "barriers_up"),
    ("barriers_down", "barriers_down"),
    ("alpha", "alpha"),
    ("k", "k"),
    ("weights", "weights"),
    ("step_param1", "step_param1"),
    ("step_param2", "step_param2"),
    ("step_param3", "step_param3"),
    ("step_param4", "step_param4"),
    ("custom_spots", "custom_spots")
]

FLAGS = flags.FLAGS


def filter_df(df, config: configs._DefaultConfig, reverse_filtering: bool = False):
    """Returns new DataFrame with rows removed according to passed config."""
    import pandas as pd
    import numpy as np

    # Check what payoffs are requested
    payoffs_requested = list(getattr(config, 'payoffs', []))
    has_standard = any('And' not in str(p) for p in payoffs_requested)
    has_barriers = any('And' in str(p) for p in payoffs_requested)

    for filter_name, column_name in FILTERS:
        # Skip algo filter if ONLY requesting standard payoffs
        # (standard payoffs exist with barrier=100000 in all algos)
        if filter_name == "algos" and has_standard and not has_barriers:
            if FLAGS.debug_csv_filtering:
                print(f"⚠️ Skipping algo filter (standard payoffs use barrier=100000 from any algo)")
            continue

        if column_name not in df.index.names:
            continue

        values = list(getattr(config, filter_name))
        if filter_name == "factors":
            values = [str(x) for x in values]

        # Special handling for None values (match NaN, None, empty, "None")
        if None in values:
            col_values = df.index.get_level_values(column_name)
            # Match: None, NaN, empty string, or string "None"
            idx = (col_values.isna() |
                   (col_values == None) |
                   (col_values == '') |
                   (col_values == 'None') |
                   (col_values.astype(str) == 'nan'))
            # Also include other non-None values from the filter
            other_values = [v for v in values if v is not None]
            if other_values:
                idx = idx | col_values.isin(other_values)
        else:
            idx = df.index.get_level_values(column_name).isin(values)

        if reverse_filtering:
            idx = ~idx

        if FLAGS.debug_csv_filtering and any(~idx):
            print(f"Filtering {sum(~idx)} rows: {column_name} not in {values}")

        df = df[idx]

    return df