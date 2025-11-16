"""Write PDF table to compare algorithms results - UNIFIED VERSION
Merges two approaches:
1. NEW SCRIPT (barriers present): Modern formatting with families, shortstack, Excel export
2. OLD SCRIPT (no barriers): Traditional scalebox table format
"""
import copy
import os.path
from typing import Iterable

import pandas as pd
import numpy as np
import tensorflow as tf

from optimal_stopping.run import configs
from optimal_stopping.utilities import read_data

from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_list("rm_from_index", None, "List of keys to remove from index")

# ============================================================================
# SHARED CONFIGURATION
# ============================================================================

ALGOS_ORDER = [
    "RLSM", "SRLSM", "RFQI", "SRFQI",  # New algorithms (priority)
    "LN2", "LNfast", "LSPI",
    "LSM", "LSMRidge", "LSMLaguerre", "LSMDeg1",
    "DOS", "pathDOS", "NLSM",
    "RLSMTanh", "RLSMRidge", "RLSMElu", "RLSMSilu",
    "RLSMGelu", "RLSMSoftplus", "RLSMSoftplusReinit",
    "RRLSM", "RRLSMmix", "RRLSMRidge",
    "FQI", "FQIR", "FQIRidge", "FQILasso", "FQILaguerre", "FQIDeg1",
    "RFQITanh", "RFQIRidge", "RFQILasso", "RFQISoftplus",
    "RRFQI", "pathRFQI",
    "EOP", "B", "Trinomial",
]

COLUMNS_ORDER = ["price", "duration"]

USE_PAYOFF_FOR_ALGO = {
    "RLSM": False, "SRLSM": False, "RFQI": False, "SRFQI": False,
    "LSM": True, "LSMRidge": True, "LSMLaguerre": True, "LSMDeg1": True,
    "DOS": True, "pathDOS": True, "NLSM": True,
    "RLSMTanh": True, "RLSMRidge": True,
    "RLSMElu": True, "RLSMSilu": True, "RLSMGelu": True,
    "RLSMSoftplus": True, "RLSMSoftplusReinit": True,
    "RRLSM": True, "RRLSMmix": True, "RRLSMRidge": True,
    "FQI": False, "FQIR": False, "FQIRidge": False, "FQILasso": False,
    "FQILaguerre": False, "FQIDeg1": False,
    "RFQITanh": False, "RFQIRidge": False, "RFQILasso": False,
    "RFQISoftplus": False, "RRFQI": False, "pathRFQI": False,
    "LSPI": False, "EOP": False, "B": False, "Trinomial": False,
}

# Old script template (for no-barrier case)
_PDF_TABLE_TMPL = r"""
\begin{table*}[!h]
\center
\scalebox{0.60}{
%(table)s
}
\caption{%(caption)s
}
\label{%(label)s}
\end{table*}
"""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _format_price_with_std(mean_val, std_val):
    """Format price with std underneath, handling NaN cases."""
    if pd.isna(mean_val) or pd.isna(std_val):
        return "--"
    return r'\shortstack{%.2f \\ {\tiny (%.2f)}}' % (mean_val, std_val)


def _human_time_delta(delta):
    """Convert seconds to human-readable format."""
    hours = delta // 3600
    minutes = (delta - hours*3600) // 60
    seconds = int(delta - hours*3600 - minutes*60)
    return "".join([
        "%dh" % hours if hours else "  ",
        "%2dm" % minutes if minutes else "   ",
        "%2ds" % seconds
    ])


def _has_barriers(config):
    """Check if config has barriers (not None or empty)."""
    if not hasattr(config, 'barriers'):
        return False

    barriers = config.barriers
    if barriers is None:
        return False

    # Check if it's a list/tuple with actual values (not just None)
    if isinstance(barriers, (list, tuple)):
        return len(barriers) > 0 and any(b is not None for b in barriers)

    return barriers is not None


# ============================================================================
# MAIN ENTRY POINTS
# ============================================================================

def write_table_price(label: str, config: configs._DefaultConfig, get_df=False):
    """Write price comparison table and return Excel data."""

    # DECISION LOGIC: Check if barriers are present
    has_barriers = _has_barriers(config)

    print(f"Processing {label}: barriers={'present' if has_barriers else 'absent'}")

    if has_barriers:
        # NEW SCRIPT: Modern format with families
        df, excel_df = _write_table_with_barriers(
            label, config, ["price"], "Prices returned by each algo", get_df=get_df)
    else:
        # OLD SCRIPT: Traditional scalebox format
        df, excel_df = _write_table_without_barriers(
            label, config, ["price"], "Prices returned by each algo", get_df=get_df)

    if get_df:
        return df
    return excel_df  # Return for Excel export


def write_table_duration(label: str, config: configs._DefaultConfig):
    """Write duration comparison table."""
    has_barriers = _has_barriers(config)

    if has_barriers:
        return _write_table_with_barriers(label, config, ["duration"],
                                         "Duration (s) of each algo")
    else:
        return _write_table_without_barriers(label, config, ["duration"],
                                            "Duration (s) of each algo")


def write_table_price_duration(label: str, config: configs._DefaultConfig):
    """Write combined price and duration table."""
    has_barriers = _has_barriers(config)

    caption = "Algorithms were run $10$ times"

    if has_barriers:
        return _write_table_with_barriers(label, config, ["price", "duration"], caption)
    else:
        caption_full = (caption + " and the mean and the standard deviation (in parenthesis) of "
                       "the prices as well as the median of the computation time are given.")
        return _write_table_without_barriers(label, config, ["price", "duration"],
                                            caption_full)


# ============================================================================
# NEW SCRIPT: WITH BARRIERS (Modern Format)
# ============================================================================

def _write_table_with_barriers(
        label: str, config: configs._DefaultConfig,
        column_names: Iterable[str], caption: str,
        get_df=False, which_time="comp_time",
        get_max_usepayoff=False, get_algo_specific_usepayoff=True,
):
    """NEW SCRIPT: Modern format with families, shortstack, Excel export."""
    df = read_data.read_csvs(config, remove_duplicates=False)
    if which_time != 'duration' and 'duration' in column_names:
        df.drop(columns=['duration'], inplace=True)
        df.rename(columns={which_time: 'duration'}, inplace=True)
    df = df.filter(items=column_names)

    # Replace NaNs
    df.reset_index(inplace=True)
    df[read_data.INDEX] = df[read_data.INDEX].replace(np.nan, "no_val")
    rmfi = FLAGS.rm_from_index
    index = read_data.INDEX.copy()
    if rmfi is not None:
        df.drop(columns=rmfi, inplace=True)
        for i in rmfi:
            if i in index:
                index.remove(i)
    all_algos = np.unique(df["algo"].values)
    df.set_index(index, inplace=True)

    # Calculate aggregated values BEFORE removing duplicates
    if 'price' in column_names:
        mean_price = df.groupby(df.index)['price'].mean()
        std = df.groupby(df.index)['price'].std()
    else:
        mean_price = None
        std = None

    if 'duration' in column_names:
        median_duration = df.groupby(df.index)['duration'].median()
    else:
        median_duration = None

    df = df[~df.index.duplicated(keep='last')]

    if 'duration' in column_names:
        try:
            df['duration'] = median_duration
            df['duration'] = [_human_time_delta(sec) for sec in df['duration']]
        except Exception:
            df['duration'] = None

    if 'price' in column_names:
        if get_df:
            df['price'] = mean_price
        else:
            df['mean_price'] = mean_price
            df['std_price'] = std
            # NEW FORMAT: shortstack for narrow columns
            df['price'] = [
                _format_price_with_std(m, s)
                for m, s in zip(df['mean_price'], df['std_price'])
            ]
            df = df.drop(columns='std_price')

            # Handle use_payoff_as_input filtering
            if get_algo_specific_usepayoff and 'use_payoff_as_input' in index and len(config.use_payoff_as_input) == 2:
                ii = np.where(np.array(index) == "use_payoff_as_input")[0][0]
                jj = np.where(np.array(index) == "algo")[0][0]
                for ind in df.index:
                    ind1 = list(ind)
                    ind2 = copy.copy(ind1)
                    ind2[ii] = not ind2[ii]
                    try:
                        if ind1[ii] == USE_PAYOFF_FOR_ALGO.get(ind1[jj], False):
                            df.drop(index=tuple(ind2), inplace=True)
                        elif tuple(ind2) in df.index:
                            df.drop(index=tuple(ind1), inplace=True)
                    except KeyError:
                        pass
                df.reset_index(inplace=True)
                index.remove("use_payoff_as_input")
                df.drop(columns="use_payoff_as_input", inplace=True)
                df.set_index(index, inplace=True)

            df2 = df["mean_price"]
            df = df.drop(columns='mean_price')

    # Extract single-value parameters BEFORE creating Excel export
    df_for_excel = df.copy() if 'price' in column_names else None
    df, global_params_caption = read_data.extract_single_value_indexes(df)

    # Create Excel dataframe AFTER filtering single-value params
    excel_df = None
    if df_for_excel is not None and mean_price is not None:
        # Apply same single-value filtering to the price dataframe
        df_excel_temp = pd.DataFrame({
            'mean': mean_price,
            'std': std
        })
        df_excel_temp, _ = read_data.extract_single_value_indexes(df_excel_temp)
        excel_df = df_excel_temp

    if get_df:
        df = df.unstack('algo')
        return df, None

    # Generate comprehensive LaTeX output (NEW STYLE)
    latex_output = _generate_comprehensive_latex(
        label, df, all_algos, config, global_params_caption
    )

    # Write LaTeX to file
    _table_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), f"../../../latex/tables_draft/"))
    if not os.path.exists(_table_path):
        os.makedirs(_table_path)
    table_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), f"../../../latex/tables_draft/{label}.tex"))

    with tf.io.gfile.GFile(table_path, "w") as tablef:
        tablef.write(latex_output)
    print(f"✅ LaTeX table written (WITH BARRIERS): {table_path}")

    return df, excel_df


# ============================================================================
# OLD SCRIPT: WITHOUT BARRIERS (Traditional Format)
# ============================================================================

def _write_table_without_barriers(
        label: str, config: configs._DefaultConfig,
        column_names: Iterable[str], caption: str,
        get_df=False, which_time="comp_time",
        get_max_usepayoff=False, get_algo_specific_usepayoff=True,
):
    """OLD SCRIPT: Traditional scalebox format (no barriers)."""
    df = read_data.read_csvs(config, remove_duplicates=False)
    if which_time != 'duration' and 'duration' in column_names:
        df.drop(columns=['duration'], inplace=True)
        df.rename(columns={which_time: 'duration'}, inplace=True)
    df = df.filter(items=column_names)

    # Replace NaNs
    df.reset_index(inplace=True)
    df[read_data.INDEX] = df[read_data.INDEX].replace(np.nan, "no_val")
    rmfi = FLAGS.rm_from_index
    index = read_data.INDEX.copy()
    if rmfi is not None:
        df.drop(columns=rmfi, inplace=True)
        for i in rmfi:
            if i in index:
                index.remove(i)
    all_algos = np.unique(df["algo"].values)
    df.set_index(index, inplace=True)

    # Calculate aggregated values BEFORE removing duplicates
    if 'price' in column_names:
        mean_price = df.groupby(df.index)['price'].mean()
        std = df.groupby(df.index)['price'].std()
    else:
        mean_price = None
        std = None

    if 'duration' in column_names:
        median_duration = df.groupby(df.index)['duration'].median()
    else:
        median_duration = None

    df = df[~df.index.duplicated(keep='last')]

    if 'duration' in column_names:
        try:
            df['duration'] = median_duration
            df['duration'] = [_human_time_delta(sec) for sec in df['duration']]
        except Exception:
            df['duration'] = None

    if 'price' in column_names:
        if get_df:
            df['price'] = mean_price
        else:
            df['mean_price'] = mean_price
            df['std_price'] = std
            # OLD FORMAT: inline (mean (std))
            df['price'] = ['%.2f (%.2f)' % ms
                          for ms in zip(df['mean_price'], df['std_price'])]
            df = df.drop(columns='std_price')

            # Handle use_payoff_as_input filtering
            if get_max_usepayoff and len(config.use_payoff_as_input) == 2:
                ii = np.where(np.array(index) == "use_payoff_as_input")[0][0]
                for ind in df.index:
                    ind1 = list(ind)
                    ind2 = copy.copy(ind1)
                    ind2[ii] = not ind2[ii]
                    try:
                        if df.loc[tuple(ind1), "price"] > df.loc[tuple(ind2), "price"]:
                            df.drop(index=tuple(ind2), inplace=True)
                        else:
                            df.drop(index=tuple(ind1), inplace=True)
                    except KeyError:
                        pass
                df.reset_index(inplace=True)
                index.remove("use_payoff_as_input")
                df.drop(columns="use_payoff_as_input", inplace=True)
                df.set_index(index, inplace=True)
            elif get_algo_specific_usepayoff and 'use_payoff_as_input' in index and len(config.use_payoff_as_input) == 2:
                ii = np.where(np.array(index) == "use_payoff_as_input")[0][0]
                jj = np.where(np.array(index) == "algo")[0][0]
                for ind in df.index:
                    ind1 = list(ind)
                    ind2 = copy.copy(ind1)
                    ind2[ii] = not ind2[ii]
                    try:
                        if ind1[ii] == USE_PAYOFF_FOR_ALGO.get(ind1[jj], False):
                            df.drop(index=tuple(ind2), inplace=True)
                        elif tuple(ind2) in df.index:
                            df.drop(index=tuple(ind1), inplace=True)
                    except KeyError:
                        pass
                df.reset_index(inplace=True)
                index.remove("use_payoff_as_input")
                df.drop(columns="use_payoff_as_input", inplace=True)
                df.set_index(index, inplace=True)

            df2 = df["mean_price"]
            df = df.drop(columns='mean_price')

    # Extract single-value parameters BEFORE creating Excel export
    df_for_excel = df.copy() if 'price' in column_names else None
    df, global_params_caption = read_data.extract_single_value_indexes(df)

    # Create Excel dataframe AFTER filtering single-value params
    excel_df = None
    if df_for_excel is not None and mean_price is not None:
        # Apply same single-value filtering to the price dataframe
        df_excel_temp = pd.DataFrame({
            'mean': mean_price,
            'std': std
        })
        df_excel_temp, _ = read_data.extract_single_value_indexes(df_excel_temp)
        excel_df = df_excel_temp
    df = df.unstack('algo')

    # Sort columns (OLD SCRIPT STYLE)
    def my_key(index):
        if index.name == 'algo':
            return pd.Index([ALGOS_ORDER.index(algo) if algo in ALGOS_ORDER else 999
                           for algo in index], name='algo')
        return pd.Index([COLUMNS_ORDER.index(name) if name in COLUMNS_ORDER else 999
                        for name in index], name='')

    try:
        df = df.sort_index(key=my_key, axis='columns')
    except Exception:
        df = df.sort_index(key=my_key)
        df = df.to_frame().T

    print(df)

    # Calculate relative errors (from OLD SCRIPT)
    df2, _ = read_data.extract_single_value_indexes(df2)
    df2 = df2.unstack('algo')
    df2 = df2["mean_price"]

    if "EOP" in all_algos:
        ref_algo = "EOP"
        if isinstance(df2[ref_algo], pd.Series):
            df2[ref_algo] = df2[ref_algo].ffill()
    elif "B" in all_algos:
        ref_algo = "B"
    else:
        ref_algo = None

    # Print error comparisons
    for a, b in [["LSM", "RLSM"], ["FQI", "RFQI"], ["DOS", "RFQI"], ["FQI", "DOS"],
                 ["LSM", "RFQI"], ["DOS", "RLSM"], ["NLSM", "RLSM"],
                 ["RLSM", "RRLSM"], ["pathDOS", "RLSM"]]:
        try:
            print(a, b)
            print((df2[a] - df2[b]) / df2[a])
        except Exception:
            pass

    if ref_algo:
        for a in all_algos:
            if a != ref_algo:
                df2[a] = np.abs(df2[ref_algo] - df2[a]) / df2[ref_algo]
        print(df2)
        df2_with_index = df2.reset_index()
        if "nb_stocks" in df2_with_index.columns:
            print(df2_with_index.loc[df2_with_index["nb_stocks"] <= 100].max(axis=0, numeric_only=True))
            print(df2_with_index.loc[df2_with_index["nb_stocks"] > 100].max(axis=0, numeric_only=True))
        else:
            print("Note: Errors not available for statistics")

    if get_df:
        return df, None

    # Generate LaTeX (OLD STYLE with scalebox)
    algos = df.columns.get_level_values("algo").unique()
    bold_algos = []

    _table_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), f"../../../latex/tables_draft/"))
    if not os.path.exists(_table_path):
        os.makedirs(_table_path)
    table_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), f"../../../latex/tables_draft/{label}.tex"))

    global_params_caption = global_params_caption.replace('_', '\\_')
    caption = f"{caption}. {global_params_caption}."

    mcol_format = ' '.join('>{\\bfseries}r' if algo in bold_algos else 'r'
                          for algo in algos)
    ind = df.index.names
    if ind == [None]:
        col_format = ('|' + '|'.join([mcol_format] * len(column_names)) + '|')
    else:
        col_format = ('|' + 'c' * len(ind) + '|' +
                     '|'.join([mcol_format] * len(column_names)) + '|')

    pdf_table = _PDF_TABLE_TMPL % {
        "table": df.to_latex(
            na_rep="-", multirow=True, multicolumn=True,
            multicolumn_format='c |',
            float_format="%.2f",
            column_format=col_format),
        "caption": caption,
        "label": label,
    }

    # Header manipulation (OLD SCRIPT)
    try:
        new_header = ' & '.join(df.index.names + list(algos)*2) + '\\\\'
        oneline = False
    except Exception:
        new_header = ' & '.join(list(algos) * 2) + '\\\\'
        oneline = True

    new_lines = []
    for line in pdf_table.split('\n'):
        if 'algo &' in line:
            new_lines.append(new_header)
        elif line.startswith('nb\\_stocks &') or line.startswith('hidden\\_size &') \
            or line.startswith('maturity &') or line.startswith('payoff ') \
            or line.startswith('model '):
            continue
        elif oneline and line.startswith('{} &'):
            new_lines.append(line.replace('{} &', '').replace(
                '{c |}{price}', '{| c |}{price}'))
        elif oneline and line.startswith('0 &'):
            new_lines.append(line.replace('0 &', ''))
        else:
            new_lines.append(line)

    pdf_table = '\n'.join(new_lines)

    # Substitutions (OLD SCRIPT)
    pdf_table = pdf_table.replace('nb_stocks', '$d$')
    pdf_table = pdf_table.replace('hidden_size', '$K$')
    pdf_table = pdf_table.replace('nb_epochs', 'epochs')
    pdf_table = pdf_table.replace('use_path', 'use path')
    pdf_table = pdf_table.replace('ridge_coeff', 'ridge coeff')
    pdf_table = pdf_table.replace('train_ITM_only', 'train ITM only')
    pdf_table = pdf_table.replace('nb_dates', '$N$')
    pdf_table = pdf_table.replace('spot', '$x_0$')
    pdf_table = pdf_table.replace('use_payoff_as_input', 'use P')

    with tf.io.gfile.GFile(table_path, "w") as tablef:
        tablef.write(pdf_table)
    print(f"✅ LaTeX table written (WITHOUT BARRIERS): {table_path}")

    return df, excel_df


# ============================================================================
# EXCEL EXPORT (SHARED)
# ============================================================================

def export_to_excel(label: str, excel_df: pd.DataFrame):
    """Export data to Excel file with proper structure.

    The Excel file will have:
    - Separate columns for each varying parameter (like LaTeX table)
    - One mean column per algorithm
    - One std column per algorithm
    """
    if excel_df is None:
        print(f"⚠️ No Excel data for {label}")
        return None

    excel_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), f"../../../latex/tables_draft/{label}.xlsx"))

    try:
        # Unstack by algo to get algorithms as columns
        excel_export = excel_df.unstack('algo')

        # Reorder columns by algorithm order
        available_algos = [a for a in ALGOS_ORDER
                          if a in excel_export.columns.get_level_values(1)]
        new_cols = []
        for algo in available_algos:
            if ('mean', algo) in excel_export.columns:
                new_cols.append(('mean', algo))
            if ('std', algo) in excel_export.columns:
                new_cols.append(('std', algo))

        if new_cols:
            excel_export = excel_export[new_cols]

        # Flatten column names: ('mean', 'RLSM') -> 'mean_RLSM'
        excel_export.columns = [f"{stat}_{algo}" for stat, algo in excel_export.columns]

        # Reset index to make varying parameters separate columns
        excel_export = excel_export.reset_index()

        # Write Excel with formatting
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                excel_export.to_excel(writer, sheet_name='Results', index=False)

                # Auto-adjust column widths
                worksheet = writer.sheets['Results']
                for idx, column in enumerate(worksheet.columns, 1):
                    max_length = 0
                    column = [cell for cell in column]
                    for cell in column:
                        try:
                            if cell.value:
                                max_length = max(max_length, len(str(cell.value)))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column[0].column_letter].width = adjusted_width

            print(f"✅ Excel file written: {excel_path}")
            print(f"   - Columns: {list(excel_export.columns[:5])}{'...' if len(excel_export.columns) > 5 else ''}")
            return excel_path
        except ImportError:
            # Fallback to xlsxwriter
            print("⚠️ openpyxl not available, trying xlsxwriter...")
            excel_export.to_excel(excel_path, engine='xlsxwriter', index=False)
            print(f"✅ Excel file written: {excel_path}")
            return excel_path

    except Exception as e:
        print(f"⚠️ Could not create Excel file: {e}")
        import traceback
        traceback.print_exc()

        # Fallback to CSV
        csv_path = excel_path.replace('.xlsx', '.csv')
        try:
            excel_df.to_csv(csv_path)
            print(f"✅ CSV file written instead: {csv_path}")
            return csv_path
        except Exception as e2:
            print(f"❌ Could not create CSV file either: {e2}")
            return None


# ============================================================================
# NEW SCRIPT HELPERS (for barrier tables)
# ============================================================================

def _categorize_payoffs(all_payoffs):
    """Categorize payoffs into families."""
    families = {
        'Standard Options': [],
        'Up-and-Out Barriers': [],
        'Down-and-Out Barriers': [],
        'Up-and-In Barriers': [],
        'Down-and-In Barriers': [],
        'Lookback Options': [],
        'Asian Options': []
    }

    for payoff in all_payoffs:
        if 'UpAndOut' in payoff:
            families['Up-and-Out Barriers'].append(payoff)
        elif 'DownAndOut' in payoff:
            families['Down-and-Out Barriers'].append(payoff)
        elif 'UpAndIn' in payoff:
            families['Up-and-In Barriers'].append(payoff)
        elif 'DownAndIn' in payoff:
            families['Down-and-In Barriers'].append(payoff)
        elif 'Lookback' in payoff:
            families['Lookback Options'].append(payoff)
        elif 'Asian' in payoff:
            families['Asian Options'].append(payoff)
        else:
            families['Standard Options'].append(payoff)

    for family in families:
        families[family] = sorted(families[family])

    return {k: v for k, v in families.items() if v}


def _generate_comprehensive_latex(label, df, algos, config, global_params):
    """Generate comprehensive LaTeX with sections/subsections (NEW SCRIPT)."""
    params_lines = []

    algo_display = []
    for a in algos:
        if a in ['RLSM', 'SRLSM', 'RFQI', 'SRFQI']:
            algo_display.append(f"\\textbf{{{a}}}")
        else:
            algo_display.append(a)
    params_lines.append(f"Algorithms: {', '.join(algo_display)}")

    if hasattr(config, 'nb_stocks'):
        dims = config.nb_stocks if isinstance(config.nb_stocks, (list, tuple)) else [config.nb_stocks]
        params_lines.append(f"Dimensions $d \\in \\{{{', '.join(map(str, dims))}\\}}$")

    if hasattr(config, 'payoffs'):
        n_payoffs = len(config.payoffs)
        params_lines.append(f"Payoffs: {n_payoffs} types")

    if hasattr(config, 'barriers') and 'barrier' in df.index.names:
        barriers = config.barriers if isinstance(config.barriers, (list, tuple)) else [config.barriers]
        barrier_vals = [str(b) if b is not None else 'None' for b in barriers]
        params_lines.append(f"Barriers $B \\in \\{{{', '.join(barrier_vals)}\\}}$")

    if hasattr(config, 'nb_paths'):
        paths = config.nb_paths if isinstance(config.nb_paths, (list, tuple)) else [config.nb_paths]
        params_lines.append(f"Monte Carlo paths: {paths[0]:,}")

    if hasattr(config, 'nb_dates'):
        dates = config.nb_dates if isinstance(config.nb_dates, (list, tuple)) else [config.nb_dates]
        params_lines.append(f"Exercise dates: {dates[0]}")

    for part in global_params.split(', '):
        if '=' in part and part not in str(params_lines):
            params_lines.append(part.replace('_', '\\_'))

    params_text = "\\noindent\\textbf{Experimental Setup:}\\\\\n" + ',\\\\\n'.join(params_lines) + ".\n\n"

    algo_short = {
        'RLSM': 'R', 'SRLSM': 'SR', 'RFQI': 'Q', 'SRFQI': 'SQ'
    }
    for a in algos:
        if a not in algo_short:
            algo_short[a] = a[:3]

    lines = []
    lines.append(f"\\section{{{label.replace('_', ' ').title()}}}\n")
    lines.append(params_text)

    legend_items = [f"\\texttt{{{algo_short[a]}}} = {a}" for a in algos]
    lines.append("\\noindent\\textbf{Algorithm Legend:} " +
                 ", ".join(legend_items) + ".\\\\\n")
    lines.append("\\noindent Values: mean, {\\tiny (std)}.\n")
    lines.append("")

    # Check structure
    has_payoff = 'payoff' in df.index.names
    has_barrier = 'barrier' in df.index.names
    has_nb_stocks = 'nb_stocks' in df.index.names

    if has_payoff:
        all_payoffs = sorted(df.index.get_level_values('payoff').unique())
        payoff_families = _categorize_payoffs(all_payoffs)

        for family_name, payoffs in payoff_families.items():
            lines.append(f"\\subsection{{{family_name}}}\n")
            family_df = df[df.index.get_level_values('payoff').isin(payoffs)]

            if has_barrier and has_nb_stocks:
                table_latex = _generate_barrier_table(
                    family_df, payoffs, algos, algo_short, family_name)
            elif has_nb_stocks:
                table_latex = _generate_standard_table(
                    family_df, payoffs, algos, algo_short, family_name)
            else:
                table_latex = _generate_minimal_table(
                    family_df, payoffs, algos, algo_short, family_name)

            lines.append(table_latex)
            lines.append("")
    else:
        lines.append("\\subsection*{Results}\n")
        lines.append(_generate_simple_table(df, algos, algo_short))

    return '\n'.join(lines)


def _generate_barrier_table(df, payoffs, algos, algo_short, family_name):
    """Generate table with barrier columns."""
    barriers = sorted([b for b in df.index.get_level_values('barrier').unique()
                      if b is not None and str(b) != 'no_val'])
    dimensions = sorted(df.index.get_level_values('nb_stocks').unique())

    payoff_abbrev = {p: _abbreviate_payoff(p) for p in payoffs}

    lines = []
    lines.append("\\begin{table}[!htbp]")
    lines.append("\\centering")
    lines.append("\\tiny")
    lines.append("\\setlength{\\tabcolsep}{2pt}")

    num_algo_cols = len(algos)
    header_cols = "l|c|" + "|".join(["c" * num_algo_cols] * len(barriers))
    lines.append(f"\\begin{{tabular}}{{{header_cols}}}")
    lines.append("\\toprule")

    if len(barriers) > 0:
        lines.append(f"& & \\multicolumn{{{len(barriers) * num_algo_cols}}}{{c}}{{Barrier Level}} \\\\")
        lines.append(f"\\cmidrule{{3-{2 + len(barriers) * num_algo_cols}}}")

    barrier_cols = "Payoff & $d$ & "
    barrier_cols += " & ".join([f"\\multicolumn{{{num_algo_cols}}}{{c|}}{{{b}}}"
                                for b in barriers[:-1]])
    if len(barriers) > 0:
        barrier_cols += f" & \\multicolumn{{{num_algo_cols}}}{{c}}{{{barriers[-1]}}}"
    lines.append(barrier_cols + " \\\\")

    algo_header = " & & "
    for _ in barriers:
        algo_header += " & ".join([f"\\texttt{{{algo_short[a]}}}" for a in algos]) + " & "
    lines.append(algo_header.rstrip(" & ") + " \\\\")
    lines.append("\\midrule")

    for payoff in payoffs:
        for i, dim in enumerate(dimensions):
            row_vals = []

            if i == 0:
                row_vals.append(f"\\texttt{{{payoff_abbrev[payoff]}}}")
            else:
                row_vals.append("")
            row_vals.append(str(dim))

            for barrier in barriers:
                for algo in algos:
                    try:
                        mask = (df.index.get_level_values('algo') == algo) & \
                               (df.index.get_level_values('payoff') == payoff) & \
                               (df.index.get_level_values('barrier') == barrier) & \
                               (df.index.get_level_values('nb_stocks') == dim)
                        val = df.loc[mask, 'price'].iloc[0] if mask.any() else "--"
                        row_vals.append(str(val))
                    except:
                        row_vals.append("--")

            lines.append(" & ".join(row_vals) + " \\\\")

        lines.append("\\hline")

    if lines[-1] == "\\hline":
        lines[-1] = "\\bottomrule"

    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{family_name}: Option prices by barrier level and dimension.}}")
    lines.append(f"\\label{{tab:{family_name.replace(' ', '_').lower()}}}")
    lines.append("\\end{table}")

    return '\n'.join(lines)


def _generate_standard_table(df, payoffs, algos, algo_short, family_name):
    """Generate table for standard (non-barrier) options."""
    dimensions = sorted(df.index.get_level_values('nb_stocks').unique())
    payoff_abbrev = {p: _abbreviate_payoff(p) for p in payoffs}

    lines = []
    lines.append("\\begin{table}[!htbp]")
    lines.append("\\centering")
    lines.append("\\footnotesize")
    lines.append("\\setlength{\\tabcolsep}{3pt}")

    num_algo_cols = len(algos)
    header_cols = "l|c|" + "c" * num_algo_cols
    lines.append(f"\\begin{{tabular}}{{{header_cols}}}")
    lines.append("\\toprule")

    algo_header = "Payoff & $d$ & " + " & ".join([f"\\texttt{{{algo_short[a]}}}"
                                                    for a in algos])
    lines.append(algo_header + " \\\\")
    lines.append("\\midrule")

    for payoff in payoffs:
        for i, dim in enumerate(dimensions):
            row_vals = []

            if i == 0:
                row_vals.append(f"\\texttt{{{payoff_abbrev[payoff]}}}")
            else:
                row_vals.append("")
            row_vals.append(str(dim))

            for algo in algos:
                try:
                    mask = (df.index.get_level_values('algo') == algo) & \
                           (df.index.get_level_values('payoff') == payoff) & \
                           (df.index.get_level_values('nb_stocks') == dim)
                    val = df.loc[mask, 'price'].iloc[0] if mask.any() else "--"
                    row_vals.append(str(val))
                except:
                    row_vals.append("--")

            lines.append(" & ".join(row_vals) + " \\\\")

        lines.append("\\hline")

    if lines[-1] == "\\hline":
        lines[-1] = "\\bottomrule"

    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{family_name}: Option prices by dimension.}}")
    lines.append(f"\\label{{tab:{family_name.replace(' ', '_').lower()}}}")
    lines.append("\\end{table}")

    return '\n'.join(lines)


def _generate_minimal_table(df, payoffs, algos, algo_short, family_name):
    """Generate minimal table when only payoff varies."""
    payoff_abbrev = {p: _abbreviate_payoff(p) for p in payoffs}

    lines = []
    lines.append("\\begin{table}[!htbp]")
    lines.append("\\centering")
    lines.append("\\small")

    header_cols = "l|" + "c" * len(algos)
    lines.append(f"\\begin{{tabular}}{{{header_cols}}}")
    lines.append("\\toprule")

    algo_header = "Payoff & " + " & ".join([f"\\texttt{{{algo_short[a]}}}"
                                             for a in algos])
    lines.append(algo_header + " \\\\")
    lines.append("\\midrule")

    for payoff in payoffs:
        row_vals = [f"\\texttt{{{payoff_abbrev[payoff]}}}"]

        for algo in algos:
            try:
                mask = (df.index.get_level_values('algo') == algo) & \
                       (df.index.get_level_values('payoff') == payoff)
                val = df.loc[mask, 'price'].iloc[0] if mask.any() else "--"
                row_vals.append(str(val))
            except:
                row_vals.append("--")

        lines.append(" & ".join(row_vals) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{family_name}: Option prices.}}")
    lines.append(f"\\label{{tab:{family_name.replace(' ', '_').lower()}}}")
    lines.append("\\end{table}")

    return '\n'.join(lines)


def _abbreviate_payoff(payoff):
    """Abbreviate payoff names for tables and escape for LaTeX."""
    abbrevs = {
        'UpAndOut': 'UO-',
        'DownAndOut': 'DO-',
        'UpAndIn': 'UI-',
        'DownAndIn': 'DI-',
        'Lookback': 'LB-',
        'Asian': 'AS-',
        'Basket': 'Bsk',
        'Geometric': 'Geo',
        'GeometricBasket': 'GeoBsk'
    }
    result = payoff
    for full, abbrev in abbrevs.items():
        result = result.replace(full, abbrev)

    # Escape special LaTeX characters
    result = result.replace('_', '\\_')  # Escape underscores
    result = result.replace('%', '\\%')  # Escape percent signs
    result = result.replace('&', '\\&')  # Escape ampersands
    result = result.replace('#', '\\#')  # Escape hash symbols

    return result


def _generate_simple_table(df, algos, algo_short):
    """Generate simple table without complex structure."""
    lines = []
    lines.append("\\begin{table}[!htbp]")
    lines.append("\\centering")
    lines.append("\\small")

    try:
        df_copy = df.unstack('algo')
        available_algos = [a for a in ALGOS_ORDER if a in algos]
        if 'price' in df_copy.columns.levels[0]:
            col_order = [('price', a) for a in available_algos if ('price', a) in df_copy.columns]
            df_copy = df_copy[col_order]

        df_copy.columns = [f"\\texttt{{{algo_short.get(col[1], col[1])}}}"
                          for col in df_copy.columns]

        table_str = df_copy.to_latex(na_rep="--", escape=False)
    except:
        table_str = df.to_latex(na_rep="--", escape=False)

    lines.append(table_str)
    lines.append("\\caption{Option pricing results.}")
    lines.append("\\label{tab:simple_results}")
    lines.append("\\end{table}")

    return '\n'.join(lines)