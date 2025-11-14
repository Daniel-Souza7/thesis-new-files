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

# NEW IMPORTS - Restructured payoffs
from optimal_stopping.payoffs.standard import (
    # Standard payoffs
    BasketCall, BasketPut,
    MaxCall, MaxPut,
    MinCall, MinPut,
    GeometricBasketCall, GeometricBasketPut)

from optimal_stopping.payoffs.barriers import (
    UpAndOutBasketCall, UpAndOutBasketPut,
    UpAndOutMaxCall, UpAndOutMaxPut,
    UpAndOutMinCall, UpAndOutMinPut,
    UpAndOutGeometricBasketCall, UpAndOutGeometricBasketPut,
    # Barrier payoffs - Down-and-Out
    DownAndOutBasketCall, DownAndOutBasketPut,
    DownAndOutMaxCall, DownAndOutMaxPut,
    DownAndOutMinCall, DownAndOutMinPut,
    DownAndOutGeometricBasketCall, DownAndOutGeometricBasketPut,
    # Barrier payoffs - Up-and-In
    UpAndInBasketCall, UpAndInBasketPut,
    UpAndInMaxCall, UpAndInMaxPut,
    UpAndInGeometricBasketCall, UpAndInGeometricBasketPut,
    UpAndInMinCall, UpAndInMinPut,
    # Barrier payoffs - Down-and-In
    DownAndInBasketCall, DownAndInBasketPut,
    DownAndInMaxCall, DownAndInMaxPut,
    DownAndInGeometricBasketCall, DownAndInGeometricBasketPut,
    DownAndInMinCall, DownAndInMinPut)

# Add after barrier imports
from optimal_stopping.payoffs.lookbacks import (
    LookbackFixedCall,
    LookbackFixedPut,
    LookbackFloatCall,
    LookbackFloatPut,
    LookbackMaxCall,
    LookbackMinPut,
)

from optimal_stopping.payoffs.double_barriers import (
    DoubleKnockOutCall, DoubleKnockOutPut,
    DoubleKnockInCall, DoubleKnockInPut,
    UpInDownOutCall, UpInDownOutPut,
    UpOutDownInCall, UpOutDownInPut,
    PartialTimeBarrierCall, StepBarrierCall,
    DoubleKnockOutLookbackFloatingCall, DoubleKnockOutLookbackFloatingPut
)

# Niche/specialized payoffs
from optimal_stopping.payoffs.niche import (
    BestOfKCall, WorstOfKCall,
    RankWeightedBasketCall,
    ChooserBasketOption,
    RangeCall, DispersionCall
)

# Leveraged positions
from optimal_stopping.payoffs.leverage import (
    LeveragedBasketLongPosition, LeveragedBasketShortPosition,
    LeveragedBasketLongStopLoss, LeveragedBasketShortStopLoss
)

# NEW IMPORTS - Restructured algorithms
from optimal_stopping.algorithms.standard.rlsm import RLSM
from optimal_stopping.algorithms.standard.rfqi import RFQI
from optimal_stopping.algorithms.path_dependent.srlsm import SRLSM
from optimal_stopping.algorithms.path_dependent.srfqi import SRFQI

# OLD IMPORTS - Keep for backward compatibility if needed
try:
    from optimal_stopping.algorithms.backward_induction import DOS
    from optimal_stopping.algorithms.backward_induction import LSM
    from optimal_stopping.algorithms.backward_induction import NLSM
    from optimal_stopping.algorithms.backward_induction import RRLSM
    from optimal_stopping.algorithms.reinforcement_learning import FQI
    from optimal_stopping.algorithms.reinforcement_learning import LSPI
    from optimal_stopping.algorithms.finite_difference import binomial
    from optimal_stopping.algorithms.finite_difference import trinomial
    from optimal_stopping.algorithms.backward_induction import backward_induction_pricer

    OLD_ALGOS_AVAILABLE = True
except ImportError:
    OLD_ALGOS_AVAILABLE = False
    print("Warning: Old algorithm implementations not found. Only RLSM/SRLSM/RFQI/SRFQI available.")

from optimal_stopping.run import write_figures
from optimal_stopping.run import configs


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

_CSV_HEADERS = ['algo', 'model', 'payoff', 'drift', 'volatility', 'mean',
                'speed', 'correlation', 'hurst', 'nb_stocks',
                'nb_paths', 'nb_dates', 'spot', 'strike', 'dividend',
                'barrier', 'barriers_up', 'barriers_down',
                'k', 'notional', 'leverage', 'barrier_stop_loss',  # NEW parameters
                'maturity', 'nb_epochs', 'hidden_size', 'factors',
                'ridge_coeff', 'use_payoff_as_input',
                'train_ITM_only',
                'price', 'duration', 'time_path_gen', 'comp_time',
                'delta', 'gamma', 'theta', 'rho', 'vega', 'greeks_method',
                'price_upper_bound',
                'exercise_time']


# NEW PAYOFFS DICTIONARY - Updated for new structure
_PAYOFFS = {
    # Standard payoffs (8)
    "BasketCall": BasketCall,
    "BasketPut": BasketPut,
    "MaxCall": MaxCall,
    "MaxPut": MaxPut,
    "MinCall": MinCall,
    "MinPut": MinPut,
    "GeometricBasketCall": GeometricBasketCall,
    "GeometricBasketPut": GeometricBasketPut,

    # Barrier options - Up-and-Out (8)
    "UpAndOutBasketCall": UpAndOutBasketCall,
    "UpAndOutBasketPut": UpAndOutBasketPut,
    "UpAndOutMaxCall": UpAndOutMaxCall,
    "UpAndOutMaxPut": UpAndOutMaxPut,
    "UpAndOutMinCall": UpAndOutMinCall,
    "UpAndOutMinPut": UpAndOutMinPut,
    "UpAndOutGeometricBasketCall": UpAndOutGeometricBasketCall,
    "UpAndOutGeometricBasketPut": UpAndOutGeometricBasketPut,

    # Barrier options - Down-and-Out (8)
    "DownAndOutBasketCall": DownAndOutBasketCall,
    "DownAndOutBasketPut": DownAndOutBasketPut,
    "DownAndOutMaxCall": DownAndOutMaxCall,
    "DownAndOutMaxPut": DownAndOutMaxPut,
    "DownAndOutMinCall": DownAndOutMinCall,
    "DownAndOutMinPut": DownAndOutMinPut,
    "DownAndOutGeometricBasketCall": DownAndOutGeometricBasketCall,
    "DownAndOutGeometricBasketPut": DownAndOutGeometricBasketPut,

    # Barrier options - Up-and-In (8)
    "UpAndInBasketCall": UpAndInBasketCall,
    "UpAndInBasketPut": UpAndInBasketPut,
    "UpAndInMaxCall": UpAndInMaxCall,
    "UpAndInMaxPut": UpAndInMaxPut,
    "UpAndInMinCall": UpAndInMinCall,
    "UpAndInMinPut": UpAndInMinPut,
    "UpAndInGeometricBasketCall": UpAndInGeometricBasketCall,
    "UpAndInGeometricBasketPut": UpAndInGeometricBasketPut,

    # Barrier options - Down-and-In (8)
    "DownAndInBasketCall": DownAndInBasketCall,
    "DownAndInBasketPut": DownAndInBasketPut,
    "DownAndInMaxCall": DownAndInMaxCall,
    "DownAndInMaxPut": DownAndInMaxPut,
    "DownAndInMinCall": DownAndInMinCall,
    "DownAndInMinPut": DownAndInMinPut,
    "DownAndInGeometricBasketCall": DownAndInGeometricBasketCall,
    "DownAndInGeometricBasketPut": DownAndInGeometricBasketPut,

    #Lookbacks - 6 options
    "LookbackFixedCall": LookbackFixedCall,
    "LookbackFixedPut": LookbackFixedPut,
    "LookbackFloatCall": LookbackFloatCall,
    "LookbackFloatPut": LookbackFloatPut,
    "LookbackMaxCall": LookbackMaxCall,
    "LookbackMinPut": LookbackMinPut,

    "DoubleKnockOutCall": DoubleKnockOutCall,
    "DoubleKnockOutPut": DoubleKnockOutPut,
    "DoubleKnockInCall": DoubleKnockInCall,
    "DoubleKnockInPut": DoubleKnockInPut,
    "UpInDownOutCall": UpInDownOutCall,
    "UpInDownOutPut": UpInDownOutPut,
    "UpOutDownInCall": UpOutDownInCall,
    "UpOutDownInPut": UpOutDownInPut,
    "PartialTimeBarrierCall": PartialTimeBarrierCall,
    "StepBarrierCall": StepBarrierCall,
    "DoubleKnockOutLookbackFloatingCall": DoubleKnockOutLookbackFloatingCall,
    "DoubleKnockOutLookbackFloatingPut": DoubleKnockOutLookbackFloatingPut,

    # Niche/specialized payoffs (7)
    "BestOfKCall": BestOfKCall,
    "WorstOfKCall": WorstOfKCall,
    "RankWeightedBasketCall": RankWeightedBasketCall,
    "ChooserBasketOption": ChooserBasketOption,
    "RangeCall": RangeCall,
    "DispersionCall": DispersionCall,

    # Leveraged positions (4)
    "LeveragedBasketLongPosition": LeveragedBasketLongPosition,
    "LeveragedBasketShortPosition": LeveragedBasketShortPosition,
    "LeveragedBasketLongStopLoss": LeveragedBasketLongStopLoss,
    "LeveragedBasketShortStopLoss": LeveragedBasketShortStopLoss,
}


_STOCK_MODELS = stock_model.STOCK_MODELS

# NEW ALGORITHMS DICTIONARY - Updated for new structure
_ALGOS = {
    # NEW ALGORITHMS - Restructured
    "RLSM": RLSM,  # Standard options
    "SRLSM": SRLSM,  # Path-dependent options (barriers, lookbacks)
    "RFQI": RFQI,  # Standard options
    "SRFQI": SRFQI,  # Path-dependent options (barriers, lookbacks)
}

# Add old algorithms if available
if OLD_ALGOS_AVAILABLE:
    _ALGOS.update({
        "LSM": LSM.LeastSquaresPricer,
        "LSMLaguerre": LSM.LeastSquarePricerLaguerre,
        "LSMRidge": LSM.LeastSquarePricerRidge,
        "LSMDeg1": LSM.LeastSquarePricerDeg1,

        "FQI": FQI.FQIFast,
        "FQILaguerre": FQI.FQIFastLaguerre,
        "FQIRidge": FQI.FQIFastRidge,
        "FQILasso": FQI.FQIFastLasso,
        "FQIDeg1": FQI.FQIFastDeg1,

        "LSPI": LSPI.LSPI,

        "NLSM": NLSM.NeuralNetworkPricer,
        "DOS": DOS.DeepOptimalStopping,
        "pathDOS": DOS.DeepOptimalStopping,

        "RRLSM": RRLSM.ReservoirRNNLeastSquarePricer2,
        "RRLSMmix": RRLSM.ReservoirRNNLeastSquarePricer,
        "RRLSMRidge": RRLSM.ReservoirRNNLeastSquarePricer2Ridge,

        "EOP": backward_induction_pricer.EuropeanOptionPricer,
        "B": binomial.BinomialPricer,
        "Trinomial": trinomial.TrinomialPricer,
    })


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
            config.spots, config.stock_models, config.strikes, config.barriers,
            config.volatilities, config.mean, config.speed, config.correlation,
            config.hurst, config.nb_epochs, config.hidden_size, config.factors,
            config.ridge_coeff, config.train_ITM_only, config.use_payoff_as_input,
            config.k, config.notional, config.leverage, config.barrier_stop_loss,
            config.barriers_up, config.barriers_down))

        for params in combinations:
            for i in range(config.nb_runs):
                tmp_file_path = os.path.join(tmp_dirpath, str(tmp_files_idx))
                tmp_files_idx += 1
                delayed_jobs.append(joblib.delayed(_run_algo)(
                    tmp_file_path, *params, fail_on_error=FLAGS.print_errors,
                    compute_greeks=FLAGS.compute_greeks,
                    greeks_method=FLAGS.greeks_method,
                    eps=FLAGS.eps, poly_deg=FLAGS.poly_deg,
                    fd_freeze_exe_boundary=FLAGS.fd_freeze_exe_boundary,
                    fd_compute_gamma_via_PDE=FLAGS.fd_compute_gamma_via_PDE,
                    reg_eps=FLAGS.reg_eps, path_gen_seed=FLAGS.path_gen_seed,
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
        nb_stocks, payoff_name, drift, spot, stock_model_name, strike, barrier,
        volatility, mean, speed, correlation, hurst, nb_epochs, hidden_size=10,
        factors=(1., 1., 1.), ridge_coeff=1.,
        train_ITM_only=True, use_payoff_as_input=False,
        k=None, notional=None, leverage=None, barrier_stop_loss=None,
        barrier_up=None, barrier_down=None,
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
        configs.path_gen_seed.set_seed(path_gen_seed)

    print(f"{algo} {payoff_name} spot={spot} vol={volatility} mat={maturity} "
          f"paths={nb_paths} ... ", end="")

    # Determine payoff type
    is_single_barrier = any(x in payoff_name for x in ['UpAnd', 'DownAnd'])
    is_double_barrier = any(x in payoff_name for x in ['DoubleKnock', 'DualBarrier', 'UpInDown', 'UpOutDown',
                                                        'PartialTime', 'StepBarrier'])
    is_lookback = 'Lookback' in payoff_name
    is_niche_k = payoff_name in ['BestOfKCall', 'WorstOfKCall']
    is_leverage_basic = payoff_name in ['LeveragedBasketLongPosition', 'LeveragedBasketShortPosition']
    is_leverage_stoploss = payoff_name in ['LeveragedBasketLongStopLoss', 'LeveragedBasketShortStopLoss']

    # Instantiate payoff
    if is_double_barrier:
        # Double barriers need both levels
        payoff_obj = _PAYOFFS[payoff_name](strike, barrier_up=barrier_up, barrier_down=barrier_down)
    elif is_single_barrier:
        # Single barrier
        payoff_obj = _PAYOFFS[payoff_name](strike, barrier=barrier)
    elif is_lookback:
        payoff_obj = _PAYOFFS[payoff_name](strike)
    elif is_niche_k:
        # Niche payoffs with k parameter
        payoff_obj = _PAYOFFS[payoff_name](strike, k=k if k is not None else 2)
    elif is_leverage_stoploss:
        # Leverage with stop-loss
        payoff_obj = _PAYOFFS[payoff_name](
            strike,
            notional=notional if notional is not None else 1.0,
            leverage=leverage if leverage is not None else 2.0,
            barrier_stop_loss=barrier_stop_loss if barrier_stop_loss is not None else (0.9 if 'Long' in payoff_name else 1.1)
        )
    elif is_leverage_basic:
        # Basic leverage payoffs
        payoff_obj = _PAYOFFS[payoff_name](
            strike,
            notional=notional if notional is not None else 1.0,
            leverage=leverage if leverage is not None else 2.0
        )
    else:
        # Standard payoffs
        payoff_obj = _PAYOFFS[payoff_name](strike)

    # Check if payoff is path-dependent
    is_path_dependent = getattr(payoff_obj, 'is_path_dependent', False)

    # Instantiate stock model
    stock_model_obj = _STOCK_MODELS[stock_model_name](
        drift=drift, volatility=volatility, mean=mean, speed=speed, hurst=hurst,
        correlation=correlation, nb_stocks=nb_stocks,
        nb_paths=nb_paths, nb_dates=nb_dates,
        spot=spot, dividend=dividend,
        maturity=maturity, name=stock_model_name)

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

        # OLD ALGORITHMS - Keep for backward compatibility
        elif OLD_ALGOS_AVAILABLE and algo in _ALGOS:
            # Use old algorithm initialization logic
            if algo in ['NLSM']:
                pricer = _ALGOS[algo](stock_model_obj, payoff_obj, nb_epochs=nb_epochs,
                                      hidden_size=hidden_size,
                                      train_ITM_only=train_ITM_only,
                                      use_payoff_as_input=use_payoff_as_input)
            elif algo in ["DOS", "pathDOS"]:
                use_path = (algo == "pathDOS")
                pricer = _ALGOS[algo](stock_model_obj, payoff_obj, nb_epochs=nb_epochs,
                                      hidden_size=hidden_size, use_path=use_path,
                                      use_payoff_as_input=use_payoff_as_input)
            elif algo in ["LSM", "LSMDeg1", "LSMLaguerre"]:
                pricer = _ALGOS[algo](stock_model_obj, payoff_obj, nb_epochs=nb_epochs,
                                      train_ITM_only=train_ITM_only,
                                      use_payoff_as_input=use_payoff_as_input)
            elif "FQI" in algo:
                pricer = _ALGOS[algo](stock_model_obj, payoff_obj, nb_epochs=nb_epochs,
                                      train_ITM_only=train_ITM_only,
                                      use_payoff_as_input=use_payoff_as_input)
            else:
                pricer = _ALGOS[algo](stock_model_obj, payoff_obj, nb_epochs=nb_epochs,
                                      use_payoff_as_input=use_payoff_as_input)
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
            price, gen_time = pricer.price(train_eval_split=train_eval_split)
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
        'drift': drift,
        'volatility': volatility,
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
        'notional': notional,
        'leverage': leverage,
        'barrier_stop_loss': barrier_stop_loss,
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