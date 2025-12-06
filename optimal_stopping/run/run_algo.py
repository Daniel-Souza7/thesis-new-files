# Lint as: python3
"""
Main module to run the algorithms.
"""
import os
import atexit
import csv
import itertools
import multiprocessing
import random
import time
import shutil

# absl needs to be upgraded to >= 0.10.0, otherwise joblib might not work
from absl import app
from absl import flags
import numpy as np
import joblib

from optimal_stopping.utilities import configs_getter
from optimal_stopping.data import stock_model

# NEW IMPORTS - Use payoff registry system with all 408 payoffs
from optimal_stopping.payoffs import get_payoff_class, _PAYOFF_REGISTRY
# NEW IMPORTS - Restructured algorithms
from optimal_stopping.algorithms.standard.rlsm import RLSM
from optimal_stopping.algorithms.standard.rfqi import RFQI
from optimal_stopping.algorithms.path_dependent.srlsm import SRLSM
from optimal_stopping.algorithms.path_dependent.srfqi import SRFQI
from optimal_stopping.algorithms.path_dependent.rrlsm import RRLSM

# BENCHMARK ALGORITHMS - Simple implementations for comparison
from optimal_stopping.algorithms.standard.lsm import LeastSquaresPricer, LeastSquarePricerDeg1, LeastSquarePricerLaguerre
from optimal_stopping.algorithms.standard.fqi import FQIFast, FQIFastDeg1, FQIFastLaguerre
from optimal_stopping.algorithms.standard.nlsm import NeuralNetworkPricer
from optimal_stopping.algorithms.standard.dos import DeepOptimalStopping
from optimal_stopping.algorithms.standard.eop import EuropeanOptionPrice

from optimal_stopping.run import write_figures
from optimal_stopping.run import configs

from optimal_stopping.algorithms.testing.zap_q import ZapQ
from optimal_stopping.algorithms.testing.rzapq import RZapQ
from optimal_stopping.algorithms.testing.dkl import DKL_LSM
from optimal_stopping.algorithms.testing.rdkl import RandDKL_LSM
from optimal_stopping.algorithms.testing.SRFQI_RBF import SRFQI_RBF

# TREE-BASED ALGORITHMS - Lattice methods for American options
from optimal_stopping.algorithms.trees.crr import CRRTree
from optimal_stopping.algorithms.trees.leisen_reimer import LeisenReimerTree
from optimal_stopping.algorithms.trees.trinomial import TrinomialTree
# GLOBAL CLASSES
class SendBotMessage:
    def __init__(self):
        pass

    @staticmethod
    def send_notification(text, token=None, chat_id=None, *args, **kwargs):
        print(text)


NUM_PROCESSORS = multiprocessing.cpu_count()
SERVER = True
NB_JOBS = int(NUM_PROCESSORS) - 1
SEND = True

try:
    from telegram_notifications import send_bot_message as SBM

    if SERVER:
        SEND = True
except Exception:
    SBM = SendBotMessage()

FLAGS = flags.FLAGS

flags.DEFINE_list("nb_stocks", None, "List of number of Stocks")
flags.DEFINE_list("algos", None, "Name of the algos to run.")
flags.DEFINE_bool("print_errors", False, "Set to True to print errors if any.")
flags.DEFINE_integer("nb_jobs", NB_JOBS, "Number of CPUs to use parallelly")
flags.DEFINE_bool("generate_pdf", False, "Whether to generate latex tables")
flags.DEFINE_integer("path_gen_seed", None, "Seed for path generation")
flags.DEFINE_bool("compute_upper_bound", False,
                  "Whether to additionally compute upper bound for price")
flags.DEFINE_bool("compute_greeks", False,
                  "Whether to compute greeks (not available for all settings)")
flags.DEFINE_string("greeks_method", "central",
                    "one of: central, forward, backward, regression")
flags.DEFINE_float("eps", 1e-9,
                   "the epsilon for the finite difference method or regression")
flags.DEFINE_float("reg_eps", 5,
                   "the epsilon for the finite difference method or regression")
flags.DEFINE_integer("poly_deg", 2,
                     "the degree for the polynomial regression")
flags.DEFINE_bool("fd_freeze_exe_boundary", True,
                  "Whether to use same exercise boundary")
flags.DEFINE_bool("fd_compute_gamma_via_PDE", True,
                  "Whether to use the PDE to compute gamma")
flags.DEFINE_bool("DEBUG", False, "Turn on debug mode")
flags.DEFINE_integer("train_eval_split", 2,
                     "divisor for the train/eval split")

_CSV_HEADERS = ['algo', 'model', 'payoff', 'drift', 'risk_free_rate', 'volatility', 'mean',
                'speed', 'correlation', 'hurst', 'nb_stocks',
                'nb_paths', 'nb_dates', 'spot', 'strike', 'dividend',
                'barrier', 'barriers_up', 'barriers_down',
                'k', 'weights', 'step_param1', 'step_param2', 'step_param3', 'step_param4',
                'user_data_file',
                'maturity', 'nb_epochs', 'hidden_size', 'factors',
                'ridge_coeff', 'use_payoff_as_input',
                'train_ITM_only',
                'price', 'duration', 'time_path_gen', 'comp_time',
                'delta', 'gamma', 'theta', 'rho', 'vega', 'greeks_method',
                'price_upper_bound',
                'exercise_time']


# NEW PAYOFFS DICTIONARY - Use the auto-generated registry (408 payoffs)
# This registry contains all 34 base payoffs + 374 barrier variants = 408 total
_PAYOFFS = _PAYOFF_REGISTRY


_STOCK_MODELS = stock_model.STOCK_MODELS

# NEW ALGORITHMS DICTIONARY - Updated for new structure
_ALGOS = {
    # NEW ALGORITHMS - Restructured
    "RLSM": RLSM,  # Standard options
    "SRLSM": SRLSM,  # Path-dependent options (barriers, lookbacks)
    "RFQI": RFQI,  # Standard options
    "SRFQI": SRFQI,  # Path-dependent options (barriers, lookbacks)
    "RRLSM": RRLSM,
    "ZAPQ": ZapQ,
    "RZAPQ": RZapQ,
    "RDKL": RandDKL_LSM,
    "DKL": DKL_LSM,
    "SRFQI_RBF": SRFQI_RBF,

    # BENCHMARK ALGORITHMS - Simple implementations for comparison
    "LSM": LeastSquaresPricer,
    "LSMDeg1": LeastSquarePricerDeg1,
    "LSMLaguerre": LeastSquarePricerLaguerre,

    "FQI": FQIFast,
    "FQIDeg1": FQIFastDeg1,
    "FQILaguerre": FQIFastLaguerre,

    "NLSM": NeuralNetworkPricer,
    "DOS": DeepOptimalStopping,
    "EOP": EuropeanOptionPrice,  # European option (exercise only at maturity)

    # TREE-BASED ALGORITHMS - Lattice methods
    "CRR": CRRTree,  # Cox-Ross-Rubinstein binomial tree
    "LR": LeisenReimerTree,  # Leisen-Reimer binomial tree
    "Trinomial": TrinomialTree,  # Trinomial tree (3-jump process)
}



def init_seed():
    random.seed(0)
    np.random.seed(0)


def _run_algos():
    start_time = time.time()

    fpath = os.path.join(os.path.dirname(__file__), "../../output/metrics_draft",
                         f'{int(time.time() * 1000)}.csv')
    tmp_dirpath = f'{fpath}.tmp_results'
    os.makedirs(tmp_dirpath, exist_ok=True)
    atexit.register(shutil.rmtree, tmp_dirpath)
    tmp_files_idx = 0

    delayed_jobs = []

    nb_stocks_flag = [int(nb) for nb in FLAGS.nb_stocks or []]
    for config_name, config in configs_getter.get_configs():
        print(f'Config {config_name}', config)
        config.algos = [a for a in config.algos
                        if FLAGS.algos is None or a in FLAGS.algos]
        if nb_stocks_flag:
            config.nb_stocks = [a for a in config.nb_stocks
                                if a in nb_stocks_flag]
        combinations = list(itertools.product(
            config.algos, config.dividends, config.maturities, config.nb_dates,
            config.nb_paths, config.nb_stocks, config.payoffs, config.drift,
            config.risk_free_rate,
            config.spots, config.stock_models, config.strikes, config.barriers,
            config.volatilities, config.mean, config.speed, config.correlation,
            config.hurst, config.nb_epochs, config.hidden_size, config.factors,
            config.ridge_coeff, config.train_ITM_only, config.use_payoff_as_input,
            config.barriers_up, config.barriers_down,
            config.k, config.weights,
            config.step_param1, config.step_param2, config.step_param3, config.step_param4))

        for params in combinations:
            for i in range(config.nb_runs):
                tmp_file_path = os.path.join(tmp_dirpath, str(tmp_files_idx))
                tmp_files_idx += 1

                # Generate different seed for each run to ensure std != 0
                if FLAGS.path_gen_seed is None:
                    # Use random seed that varies across runs (time-based + run index)
                    import random
                    run_seed = random.randint(0, 2**32 - 1)
                else:
                    # Use base seed + run index to ensure reproducibility but variation
                    run_seed = FLAGS.path_gen_seed + i

                delayed_jobs.append(joblib.delayed(_run_algo)(
                    tmp_file_path, *params,
                    run_idx=i,  # <--- PASS THE RUN INDEX (0, 1, 2, 3, 4)
                    fail_on_error=FLAGS.print_errors,
                    compute_greeks=FLAGS.compute_greeks,
                    greeks_method=FLAGS.greeks_method,
                    eps=FLAGS.eps, poly_deg=FLAGS.poly_deg,
                    fd_freeze_exe_boundary=FLAGS.fd_freeze_exe_boundary,
                    fd_compute_gamma_via_PDE=FLAGS.fd_compute_gamma_via_PDE,
                    reg_eps=FLAGS.reg_eps, path_gen_seed=run_seed,
                    compute_upper_bound=FLAGS.compute_upper_bound,
                    DEBUG=FLAGS.DEBUG, train_eval_split=FLAGS.train_eval_split))


    print(f"Running {len(delayed_jobs)} tasks using "
          f"{FLAGS.nb_jobs}/{NUM_PROCESSORS} CPUs...")
    joblib.Parallel(n_jobs=FLAGS.nb_jobs)(delayed_jobs)

    print(f'Writing results to {fpath}...')
    with open(fpath, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=_CSV_HEADERS)
        writer.writeheader()
        for idx in range(tmp_files_idx):
            tmp_file_path = os.path.join(tmp_dirpath, str(idx))
            try:
                with open(tmp_file_path, "r") as read_f:
                    csvfile.write(read_f.read())
            except FileNotFoundError:
                pass

    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print(f"\n{'=' * 80}")
    print(f"EXECUTION COMPLETED")
    print(f"{'=' * 80}")
    print(f"Total execution time: {hours}h {minutes}m {seconds}s")
    print(f"Total tasks completed: {len(delayed_jobs)}")
    print(f"Results saved to: {fpath}")
    print(f"{'=' * 80}\n")

    return fpath


def _run_algo(
        metrics_fpath, algo, dividend, maturity, nb_dates, nb_paths,
        nb_stocks, payoff_name, drift, risk_free_rate, spot, stock_model_name, strike, barrier,
        volatility, mean, speed, correlation, hurst, nb_epochs, hidden_size=10,
        factors=(1., 1., 1.), ridge_coeff=1.,
        train_ITM_only=True, use_payoff_as_input=False,
        barrier_up=None, barrier_down=None,
        k=2, weights=None,
        step_param1=-1, step_param2=1, step_param3=-1, step_param4=1,
        user_data_file=None,
        run_idx=0,
        fail_on_error=False,
        compute_greeks=False, greeks_method=None, eps=None,
        poly_deg=None, fd_freeze_exe_boundary=True,
        fd_compute_gamma_via_PDE=True, reg_eps=None, path_gen_seed=None,
        compute_upper_bound=False,
        DEBUG=False, train_eval_split=2):
    """
    Run one algorithm for option pricing.

    This is the updated version for the restructured codebase with:
    - New RLSM, SRLSM, RFQI, SRFQI algorithms
    - Proper handling of path-dependent vs standard options
    - All 34 payoffs (8 standard + 26 barriers)
    """
    if path_gen_seed is not None:
        # Set all random seeds for reproducibility
        configs.path_gen_seed.set_seed(path_gen_seed)
        import random
        random.seed(path_gen_seed)  # Python's random module
        import numpy as np
        np.random.seed(path_gen_seed)  # Numpy (will be set again in price() but good to set early)
        import torch
        torch.manual_seed(path_gen_seed)  # PyTorch for randomized neural networks

    print(f"{algo} {payoff_name} spot={spot} vol={volatility} mat={maturity} "
          f"paths={nb_paths} ... ", end="")

    # Instantiate payoff - unified approach for all 408 payoffs
    # All payoffs accept strike + any extra parameters via **kwargs
    payoff_class = _PAYOFFS.get(payoff_name)
    if payoff_class is None:
        raise ValueError(f"Unknown payoff: {payoff_name}. Available: {len(_PAYOFFS)} payoffs")

    payoff_obj = payoff_class(
        strike=strike,
        barrier=barrier,
        barrier_up=barrier_up,
        barrier_down=barrier_down,
        k=k,
        weights=weights,
        step_param1=step_param1,
        step_param2=step_param2,
        step_param3=step_param3,
        step_param4=step_param4,
        rate=risk_free_rate,  # For step barrier growth
        maturity=maturity  # For step barrier growth
    )

    # Check if payoff is path-dependent
    is_path_dependent = getattr(payoff_obj, 'is_path_dependent', False)

    # --- INSERT START ---
    start_index = 0

    # Check if we are using Stored Data (H5 file) or RealData
    if "Stored" in stock_model_name or stock_model_name == "RealData":
        # We need 2x paths (1 for train, 1 for eval)
        paths_per_run = nb_paths * 2

        # Calculate offset: Run 0 starts at 0, Run 1 at 40k, Run 2 at 80k...
        start_index = run_idx * paths_per_run

        # We request 2x paths from the model
        paths_to_load = paths_per_run
        actual_train_eval_split = 2

        print(f"🔄 Run {run_idx + 1}: Loading {paths_to_load:,} paths "
              f"starting at index {start_index:,} (Range: {start_index:,} to {start_index + paths_to_load:,})")
    else:
        # For generated data (BlackScholes), we rely on the Seed to vary the data
        paths_to_load = nb_paths
        actual_train_eval_split = train_eval_split

        # Instantiate stock model
    stock_model_obj = _STOCK_MODELS[stock_model_name](
        drift=drift, risk_free_rate=risk_free_rate, volatility=volatility, mean=mean, speed=speed, hurst=hurst,
        correlation=correlation, nb_stocks=nb_stocks,
        nb_paths=paths_to_load,  # <--- CHANGED THIS
        nb_dates=nb_dates,
        spot=spot, dividend=dividend,
        maturity=maturity, user_data_file=user_data_file)
    # Instantiate stock model
    # Note: Don't pass 'name' - each model sets its own name internally
    stock_model_obj = _STOCK_MODELS[stock_model_name](
        drift=drift, risk_free_rate=risk_free_rate, volatility=volatility, mean=mean, speed=speed, hurst=hurst,
        correlation=correlation, nb_stocks=nb_stocks,
        nb_paths=nb_paths, nb_dates=nb_dates,
        spot=spot, dividend=dividend,
        maturity=maturity, user_data_file=user_data_file)

    # Capture actual parameter values used by the model for CSV output
    # (important when drift/volatility/risk_free_rate are None and model uses empirical/default values)
    actual_drift = drift
    actual_volatility = volatility
    actual_risk_free_rate = risk_free_rate

    # Extract actual values from the created model object
    # Special handling for RealData which stores actual empirical values separately
    if stock_model_name == 'RealData' and hasattr(stock_model_obj, 'target_drift_daily'):
        # RealData stores actual drift/vol in target_drift_daily/target_vol_daily (daily values)
        # Need to annualize: daily*252 for drift, daily*sqrt(252) for vol
        actual_drift = np.mean(stock_model_obj.target_drift_daily) * 252
        actual_volatility = np.mean(stock_model_obj.target_vol_daily) * np.sqrt(252)
    else:
        # For other models, extract from model attributes
        if hasattr(stock_model_obj, 'drift'):
            # model.drift already has dividend subtracted, so add it back for reporting
            actual_drift = stock_model_obj.drift + dividend

        if hasattr(stock_model_obj, 'volatility'):
            actual_volatility = stock_model_obj.volatility

    # Get actual risk_free_rate from model (applies to all models)
    if hasattr(stock_model_obj, 'rate'):
        actual_risk_free_rate = stock_model_obj.rate

    # Instantiate pricer based on algorithm and payoff type
    try:
        if algo in ["RLSM", "RFQI"]:
            # Standard algorithms - for non-path-dependent options
            if is_path_dependent:
                raise ValueError(
                    f"{algo} is for standard options only. "
                    f"Payoff '{payoff_name}' is path-dependent. "
                    f"Use S{algo} instead."
                )
            pricer = _ALGOS[algo](
                stock_model_obj, payoff_obj,
                hidden_size=hidden_size,
                factors=factors,
                train_ITM_only=train_ITM_only,
                use_payoff_as_input=use_payoff_as_input,
                nb_epochs=nb_epochs if algo == "RFQI" else None
            )

        elif algo in ["SRLSM", "SRFQI"]:
            # Path-dependent algorithms - for barriers, lookbacks
            if not is_path_dependent:
                raise ValueError(
                    f"{algo} is for path-dependent options only. "
                    f"Payoff '{payoff_name}' is NOT path-dependent. "
                    f"Use {algo[1:]} instead."
                )
            pricer = _ALGOS[algo](
                stock_model_obj, payoff_obj,
                hidden_size=hidden_size,
                factors=factors,
                train_ITM_only=train_ITM_only,
                use_payoff_as_input=use_payoff_as_input,
                nb_epochs=nb_epochs if algo == "SRFQI" else None
            )

        # BENCHMARK ALGORITHMS - Simple implementations
        elif algo in ['NLSM']:
            pricer = _ALGOS[algo](
                stock_model_obj, payoff_obj,
                nb_epochs=nb_epochs,
                hidden_size=hidden_size,
                factors=factors,
                train_ITM_only=train_ITM_only,
                use_payoff_as_input=use_payoff_as_input
            )

        elif algo in ["DOS"]:
            pricer = _ALGOS[algo](
                stock_model_obj, payoff_obj,
                nb_epochs=nb_epochs,
                hidden_size=hidden_size,
                factors=factors,
                use_path=False,
                train_ITM_only=train_ITM_only,
                use_payoff_as_input=use_payoff_as_input
            )


        elif algo in ['RRLSM']:

            pricer = _ALGOS[algo](
                stock_model_obj, payoff_obj,
                nb_epochs=nb_epochs,
                hidden_size=hidden_size,
                factors=factors,
                train_ITM_only=train_ITM_only,
                use_payoff_as_input=use_payoff_as_input
            )

        elif algo in ["LSM", "LSMDeg1", "LSMLaguerre", "ZAPQ", "RZAPQ", "DKL", "RDKL", "SRFQI_RBF"]:
            pricer = _ALGOS[algo](
                stock_model_obj, payoff_obj,
                nb_epochs=nb_epochs,
                hidden_size=hidden_size,
                factors=factors,
                train_ITM_only=train_ITM_only,
                use_payoff_as_input=use_payoff_as_input
            )

        elif algo in ["FQI", "FQIDeg1", "FQILaguerre"]:
            pricer = _ALGOS[algo](
                stock_model_obj, payoff_obj,
                nb_epochs=nb_epochs,
                hidden_size=hidden_size,
                factors=factors,
                train_ITM_only=train_ITM_only,
                use_payoff_as_input=use_payoff_as_input
            )

        elif algo == "EOP":
            # European Option Price - exercises only at maturity
            pricer = _ALGOS[algo](
                stock_model_obj, payoff_obj
            )

        elif algo in ["CRR", "LR", "Trinomial"]:
            # Tree-based algorithms - lattice methods
            # These use n_steps instead of nb_epochs
            # Default n_steps: 50 for CRR/Trinomial, 51 for LR (odd for Peizer-Pratt)
            n_steps = 51 if algo == "LR" else 50
            pricer = _ALGOS[algo](
                stock_model_obj, payoff_obj,
                n_steps=n_steps,
                nb_epochs=nb_epochs,  # Passed but ignored
                hidden_size=hidden_size,  # Passed but ignored
                factors=factors,  # Passed but ignored
                train_ITM_only=train_ITM_only,  # Passed but ignored
                use_payoff_as_input=use_payoff_as_input  # Passed but ignored
            )

        else:
            raise ValueError(f"Unknown algorithm: {algo}")

    except Exception as e:
        if fail_on_error:
            raise
        print(f"ERROR creating pricer: {e}")
        return

    # Run pricing
    t_begin = time.time()

    if DEBUG:
        if not compute_greeks:
            price, gen_time = pricer.price(train_eval_split=train_eval_split)
            delta, gamma, theta, rho, vega, price_u = [None] * 6
        else:
            # Greeks not implemented for new algorithms yet
            price, gen_time = pricer.price(train_eval_split=train_eval_split)
            delta, gamma, theta, rho, vega, price_u = [None] * 6
        duration = time.time() - t_begin
        comp_time = duration - gen_time
        return

    try:
        if not compute_greeks:
            # CHANGE: Use actual_train_eval_split instead of train_eval_split
            price, gen_time = pricer.price(train_eval_split=actual_train_eval_split)
            delta, gamma, theta, rho, vega, price_u = [None] * 6
        else:
            # Greeks computation not yet implemented for new algorithms
            price, gen_time = pricer.price(train_eval_split=train_eval_split)
            delta, gamma, theta, rho, vega, price_u = [None] * 6
            print("Warning: Greeks computation not yet implemented for new algorithms")

        duration = time.time() - t_begin
        comp_time = duration - gen_time

        # NEW: Get exercise time if available
        exercise_time = None
        if hasattr(pricer, 'get_exercise_time'):
            try:
                exercise_time = pricer.get_exercise_time()
            except:
                exercise_time = None
    except BaseException as err:
        if fail_on_error:
            raise
        print(f"ERROR: {err}")
        return

    # Write metrics
    metrics_ = {
        'algo': algo,
        'model': stock_model_name,
        'payoff': payoff_name,
        'drift': actual_drift,
        'risk_free_rate': actual_risk_free_rate,
        'volatility': actual_volatility,
        'mean': mean,
        'speed': speed,
        'correlation': correlation,
        'hurst': hurst,
        'nb_stocks': nb_stocks,
        'nb_paths': nb_paths,
        'nb_dates': nb_dates,
        'spot': spot,
        'strike': strike,
        'barrier': barrier,
        'barriers_up': barrier_up,
        'barriers_down': barrier_down,
        'k': k,
        'weights': weights,
        'step_param1': step_param1,
        'step_param2': step_param2,
        'step_param3': step_param3,
        'step_param4': step_param4,
        'user_data_file': user_data_file,
        'dividend': dividend,
        'maturity': maturity,
        'price': price,
        'duration': duration,
        'time_path_gen': gen_time,
        'comp_time': comp_time,
        'hidden_size': hidden_size,
        'factors': factors,
        'ridge_coeff': ridge_coeff,
        'nb_epochs': nb_epochs,
        'train_ITM_only': train_ITM_only,
        'use_payoff_as_input': use_payoff_as_input,
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'rho': rho,
        'vega': vega,
        'greeks_method': greeks_method,
        'price_upper_bound': price_u,
        'exercise_time': exercise_time,
    }

    print(f"price: {price:.4f}, comp_time: {comp_time:.4f}")

    with open(metrics_fpath, "w") as metrics_f:
        writer = csv.DictWriter(metrics_f, fieldnames=_CSV_HEADERS)
        writer.writerow(metrics_)


def main(argv):
    del argv

    if FLAGS.DEBUG:
        configs.path_gen_seed.set_seed(FLAGS.path_gen_seed)
        if FLAGS.path_gen_seed is not None:
            import random
            random.seed(FLAGS.path_gen_seed)
            import torch
            torch.manual_seed(FLAGS.path_gen_seed)
        filepath = _run_algos()
        return

    try:
        if SEND:
            SBM.send_notification(
                token="8239319342:AAGIIcoDaxJ1uauHbWfdByF4yzNYdQ5jpiA",
                text='start running AMC2 with config:\n{}'.format(FLAGS.configs),
                chat_id="798647521"
            )
        configs.path_gen_seed.set_seed(FLAGS.path_gen_seed)
        if FLAGS.path_gen_seed is not None:
            import random
            random.seed(FLAGS.path_gen_seed)
            import torch
            torch.manual_seed(FLAGS.path_gen_seed)
        filepath = _run_algos()

        if FLAGS.generate_pdf:
            write_figures.write_figures()
            write_figures.generate_pdf()

        if SEND:
            time.sleep(1)
            SBM.send_notification(
                token="8239319342:AAGIIcoDaxJ1uauHbWfdByF4yzNYdQ5jpiA",
                text='finished',
                files=[filepath],
                chat_id="798647521"
            )
    except Exception as e:
        if SEND:
            SBM.send_notification(
                token="8239319342:AAGIIcoDaxJ1uauHbWfdByF4yzNYdQ5jpiA",
                text='ERROR\n{}'.format(e),
                chat_id="798647521"
            )
        else:
            print('ERROR\n{}'.format(e))


if __name__ == "__main__":
    app.run(main)