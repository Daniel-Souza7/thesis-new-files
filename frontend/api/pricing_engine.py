"""Python API Pricing Engine with RealData Support.

This module provides a Python API for pricing options using various models including:
- Black-Scholes (GBM)
- Heston (stochastic volatility)
- Fractional Black-Scholes (long memory)
- Rough Heston (rough volatility)
- Real Data (block bootstrap from historical data)

The API is designed to be called from Next.js API routes via subprocess.
"""

import sys
import json
import os
import numpy as np
from typing import Dict, Any, Optional, List
import warnings
from contextlib import redirect_stdout
import io

# Add parent directory to path for imports (cross-platform)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))  # Go up two levels from api/
sys.path.insert(0, project_root)

from optimal_stopping.data.stock_model import BlackScholes, Heston, FractionalBlackScholes, RoughHeston
from optimal_stopping.data.real_data import RealDataModel
from optimal_stopping.payoffs import get_payoff_class
from optimal_stopping.algorithms.standard.rlsm import RLSM
from optimal_stopping.algorithms.standard.rfqi import RFQI
from optimal_stopping.algorithms.standard.lsm import LeastSquaresPricer
from optimal_stopping.algorithms.standard.fqi import FQIFast
from optimal_stopping.algorithms.standard.eop import EuropeanOptionPrice
from optimal_stopping.algorithms.path_dependent.srlsm import SRLSM
from optimal_stopping.algorithms.path_dependent.srfqi import SRFQI
from optimal_stopping.algorithms.trees.crr import CRRTree
from optimal_stopping.algorithms.trees.leisen_reimer import LeisenReimerTree
from optimal_stopping.algorithms.trees.trinomial import TrinomialTree


# Pre-loaded common tickers for quick access
COMMON_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
    'JPM', 'BAC', 'META', 'NVDA', 'V'
]

DEFAULT_DATE_RANGE = {
    'start': '2010-01-01',
    'end': '2024-01-01'
}


class PricingEngine:
    """Main pricing engine for option valuation."""

    def __init__(self):
        """Initialize pricing engine with caching support."""
        self.stock_data_cache = {}  # Cache for downloaded stock data
        self.model_cache = {}  # Cache for initialized models

    def get_model_class(self, model_name: str):
        """Get model class by name."""
        models = {
            'BlackScholes': BlackScholes,
            'Heston': Heston,
            'FractionalBlackScholes': FractionalBlackScholes,
            'RoughHeston': RoughHeston,
            'RealData': RealDataModel,
        }
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
        return models[model_name]

    def get_algorithm_class(self, algo_name: str, is_path_dependent: bool):
        """Get algorithm class by name and path dependency."""
        if is_path_dependent:
            algos = {
                'SRLSM': SRLSM,
                'SRFQI': SRFQI,
            }
        else:
            algos = {
                'RLSM': RLSM,
                'RFQI': RFQI,
                'LSM': LeastSquaresPricer,
                'FQI': FQIFast,
                'EOP': EuropeanOptionPrice,
                'CRR': CRRTree,
                'LR': LeisenReimerTree,
                'Trinomial': TrinomialTree,
            }

        if algo_name not in algos:
            available = list(algos.keys())
            raise ValueError(f"Unknown algorithm: {algo_name}. Available for path_dependent={is_path_dependent}: {available}")
        return algos[algo_name]

    def create_model(self, params: Dict[str, Any]):
        """Create stock model from parameters.

        Args:
            params: Dictionary with model parameters including:
                - model_type: 'BlackScholes', 'Heston', 'RealData', etc.
                - spot: Initial stock price
                - drift: Annual drift (None for empirical in RealData)
                - volatility: Annual volatility (None for empirical in RealData)
                - rate: Risk-free rate
                - nb_stocks: Number of stocks
                - maturity: Option maturity in years
                - nb_dates: Number of time steps
                - nb_paths: Number of paths for Monte Carlo

                For RealData model:
                - tickers: List of ticker symbols
                - start_date: Start date for historical data
                - end_date: End date for historical data
                - drift_override: Override drift (None = use empirical)
                - volatility_override: Override volatility (None = use empirical)
                - exclude_crisis: Exclude crisis periods
                - only_crisis: Only use crisis periods

        Returns:
            Initialized model instance
        """
        model_type = params.get('model_type', 'BlackScholes')
        model_class = self.get_model_class(model_type)

        # Base parameters for all models
        base_params = {
            'spot': params.get('spot', 100.0),
            'drift': params.get('drift', 0.05),
            'volatility': params.get('volatility', 0.2),
            'rate': params.get('rate', 0.03),
            'dividend': params.get('dividend', 0.0),
            'nb_stocks': params.get('nb_stocks', 1),
            'maturity': params.get('maturity', 1.0),
            'nb_dates': params.get('nb_dates', 50),
            'nb_paths': params.get('nb_paths', 10000),
        }

        # RealData-specific parameters
        if model_type == 'RealData':
            real_data_params = {
                'tickers': params.get('tickers', COMMON_TICKERS[:base_params['nb_stocks']]),
                'start_date': params.get('start_date', DEFAULT_DATE_RANGE['start']),
                'end_date': params.get('end_date', DEFAULT_DATE_RANGE['end']),
                'drift_override': params.get('drift_override', None),  # None = use empirical
                'volatility_override': params.get('volatility_override', None),  # None = use empirical
                'exclude_crisis': params.get('exclude_crisis', False),
                'only_crisis': params.get('only_crisis', False),
                'cache_data': params.get('cache_data', True),
            }

            # Merge base and RealData params
            model_params = {**base_params, **real_data_params}

            # Create cache key
            cache_key = f"RealData_{','.join(real_data_params['tickers'])}_{real_data_params['start_date']}_{real_data_params['end_date']}"

            # Check cache
            if cache_key in self.model_cache:
                print(f"Using cached RealData model: {cache_key}")
                return self.model_cache[cache_key]

            # Create new model
            print(f"Creating new RealData model: {cache_key}")
            model = model_class(**model_params)

            # Cache the model
            self.model_cache[cache_key] = model
            return model

        # Heston-specific parameters
        elif model_type == 'Heston':
            heston_params = {
                'kappa': params.get('kappa', 2.0),  # Mean reversion speed
                'theta': params.get('theta', 0.04),  # Long-term variance
                'xi': params.get('xi', 0.3),  # Vol of vol
                'rho': params.get('rho', -0.7),  # Correlation
                'v0': params.get('v0', 0.04),  # Initial variance
            }
            model_params = {**base_params, **heston_params}
            return model_class(**model_params)

        # Fractional Black-Scholes parameters
        elif model_type == 'FractionalBlackScholes':
            fbm_params = {
                'hurst': params.get('hurst', 0.7),  # Hurst parameter
            }
            model_params = {**base_params, **fbm_params}
            return model_class(**model_params)

        # Rough Heston parameters
        elif model_type == 'RoughHeston':
            rough_params = {
                'hurst': params.get('hurst', 0.1),  # Hurst parameter (< 0.5 for rough)
                'kappa': params.get('kappa', 2.0),
                'theta': params.get('theta', 0.04),
                'xi': params.get('xi', 0.3),
                'rho': params.get('rho', -0.7),
                'v0': params.get('v0', 0.04),
            }
            model_params = {**base_params, **rough_params}
            return model_class(**model_params)

        # Default: Black-Scholes or other simple models
        else:
            return model_class(**base_params)

    def create_payoff(self, params: Dict[str, Any]):
        """Create payoff from parameters.

        Args:
            params: Dictionary with payoff parameters including:
                - payoff_type: Name or abbreviation (e.g., 'BasketCall', 'UO_MaxCall')
                - strike: Strike price
                - barrier: Barrier level (for barrier options)
                - weights: Weights for basket options
                - k: Number of best/worst stocks for rank options

        Returns:
            Initialized payoff instance
        """
        payoff_type = params.get('payoff_type', 'Call')
        PayoffClass = get_payoff_class(payoff_type)

        if PayoffClass is None:
            raise ValueError(f"Unknown payoff type: {payoff_type}")

        # Build payoff parameters
        payoff_params = {
            'strike': params.get('strike', 100.0),
        }

        # Add barrier if needed
        if 'barrier' in params:
            payoff_params['barrier'] = params['barrier']

        # Add weights if needed
        if 'weights' in params:
            payoff_params['weights'] = params['weights']

        # Add k parameter for best-of-k/worst-of-k
        if 'k' in params:
            payoff_params['k'] = params['k']

        return PayoffClass(**payoff_params)

    def price_option(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Price an option based on request parameters.

        Args:
            request: Dictionary with all pricing parameters

        Returns:
            Dictionary with pricing results including:
                - price: Option price
                - std_error: Standard error (if available)
                - computation_time: Time taken to compute
                - model_info: Information about the model used
                - payoff_info: Information about the payoff
        """
        try:
            # Create model
            model = self.create_model(request)

            # Create payoff
            payoff = self.create_payoff(request)

            # Get algorithm
            algo_name = request.get('algorithm', 'RLSM')
            is_path_dependent = payoff.is_path_dependent
            AlgoClass = self.get_algorithm_class(algo_name, is_path_dependent)

            # Create algorithm instance
            algo = AlgoClass(model=model, payoff=payoff)

            # Price the option (suppress debug prints)
            with redirect_stdout(io.StringIO()):
                price, comp_time = algo.price()

            # Get exercise time if available
            exercise_time = None
            if hasattr(algo, 'get_exercise_time'):
                try:
                    exercise_time = algo.get_exercise_time()
                except Exception as e:
                    warnings.warn(f"Could not get exercise time: {e}")

            # Generate sample paths for visualization
            paths_sample = self._generate_sample_paths(model, num_samples=5)

            # Build response
            response = {
                'success': True,
                'price': float(price),
                'computation_time': float(comp_time),
                'exercise_time': float(exercise_time) if exercise_time is not None else None,
                'paths_sample': paths_sample,
                'model_info': {
                    'type': request.get('model_type', 'BlackScholes'),
                    'spot': model.spot,
                    'drift': model.drift if hasattr(model, 'drift') else None,
                    'volatility': model.volatility if hasattr(model, 'volatility') else None,
                    'rate': model.rate,
                    'nb_stocks': model.nb_stocks,
                    'maturity': model.maturity,
                    'nb_dates': model.nb_dates,
                    'nb_paths': model.nb_paths,
                },
                'payoff_info': {
                    'type': payoff.__class__.__name__,
                    'strike': payoff.strike,
                    'is_path_dependent': payoff.is_path_dependent,
                },
                'algorithm': algo_name,
            }

            # Add RealData-specific info
            if isinstance(model, RealDataModel):
                response['model_info']['tickers'] = model.tickers
                response['model_info']['empirical_drift'] = float(model.empirical_drift_annual)
                response['model_info']['empirical_volatility'] = float(model.empirical_vol_annual)
                response['model_info']['block_length'] = model.avg_block_length
                response['model_info']['data_days'] = len(model.returns)

                if model.drift_override is not None:
                    response['model_info']['drift_override'] = float(model.drift_override)
                if model.volatility_override is not None:
                    response['model_info']['volatility_override'] = float(model.volatility_override)

            return response

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
            }

    def _generate_sample_paths(self, model, num_samples: int = 5) -> List[List[List[float]]]:
        """Generate a small number of sample paths for visualization.

        Args:
            model: Stock model instance
            num_samples: Number of sample paths to generate

        Returns:
            List of paths, where each path is a list of [time, price] tuples
        """
        try:
            # Generate a small number of paths for visualization
            paths, _ = model.generate_paths(nb_paths=num_samples)

            # Convert to list format: [[time, price], ...]
            sample_paths = []
            for path_idx in range(min(num_samples, paths.shape[0])):
                path_data = []
                for time_idx in range(paths.shape[2]):
                    # Use first stock for single-asset, or average for multi-asset
                    if model.nb_stocks == 1:
                        price = float(paths[path_idx, 0, time_idx])
                    else:
                        # Average price across all stocks
                        price = float(np.mean(paths[path_idx, :, time_idx]))

                    time_point = float(time_idx * model.dt)
                    path_data.append([time_point, price])

                sample_paths.append(path_data)

            return sample_paths
        except Exception as e:
            warnings.warn(f"Could not generate sample paths: {e}")
            return []

    def get_stock_info(self, tickers: List[str], start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Get information about stocks including empirical drift/volatility.

        Args:
            tickers: List of ticker symbols
            start_date: Start date for historical data
            end_date: End date for historical data

        Returns:
            Dictionary with stock information including empirical statistics
        """
        try:
            # Use default date range if not provided
            start_date = start_date or DEFAULT_DATE_RANGE['start']
            end_date = end_date or DEFAULT_DATE_RANGE['end']

            # Create a temporary RealData model to get statistics
            # Use minimal paths for faster initialization
            model = RealDataModel(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                nb_stocks=len(tickers),
                spot=100.0,
                nb_paths=100,
                nb_dates=10,
                maturity=1.0,
                rate=0.03,
                dividend=0.0,
                cache_data=True,
            )

            # Extract statistics
            stock_stats = []
            for i, ticker in enumerate(model.tickers):
                stock_stats.append({
                    'ticker': ticker,
                    'empirical_drift_annual': float(model.empirical_drift_daily[i] * 252),
                    'empirical_volatility_annual': float(model.empirical_vol_daily[i] * np.sqrt(252)),
                })

            return {
                'success': True,
                'tickers': model.tickers,
                'start_date': start_date,
                'end_date': end_date,
                'data_days': len(model.returns),
                'block_length': model.avg_block_length,
                'overall_drift': float(model.empirical_drift_annual),
                'overall_volatility': float(model.empirical_vol_annual),
                'stock_statistics': stock_stats,
                'correlation_matrix': model.empirical_correlation.tolist(),
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
            }


def main():
    """Main entry point for CLI usage."""
    if len(sys.argv) < 2:
        print(json.dumps({
            'success': False,
            'error': 'No command provided. Usage: python pricing_engine.py <command> <json_params>'
        }))
        sys.exit(1)

    command = sys.argv[1]

    # Parse parameters
    if len(sys.argv) > 2:
        try:
            params = json.loads(sys.argv[2])
        except json.JSONDecodeError as e:
            print(json.dumps({
                'success': False,
                'error': f'Invalid JSON parameters: {e}'
            }))
            sys.exit(1)
    else:
        params = {}

    # Create engine
    engine = PricingEngine()

    # Execute command
    if command == 'price':
        result = engine.price_option(params)
    elif command == 'stock_info':
        tickers = params.get('tickers', COMMON_TICKERS)
        start_date = params.get('start_date', None)
        end_date = params.get('end_date', None)
        result = engine.get_stock_info(tickers, start_date, end_date)
    elif command == 'list_payoffs':
        from optimal_stopping.payoffs import list_payoffs
        result = {
            'success': True,
            'payoffs': list_payoffs(),
        }
    elif command == 'payoff_info':
        payoff_name = params.get('payoff_name')
        if not payoff_name:
            result = {
                'success': False,
                'error': 'payoff_name is required for payoff_info command',
            }
        else:
            try:
                from optimal_stopping.payoffs import get_payoff_class
                payoff_class = get_payoff_class(payoff_name)

                # Determine optional parameters based on payoff name
                optional_params = []
                if any(barrier in payoff_name for barrier in ['UO', 'DO', 'UI', 'DI', 'PTB']):
                    optional_params.append('barrier')
                if any(barrier in payoff_name for barrier in ['UODO', 'UIDI', 'UIDO', 'UODI']):
                    optional_params.extend(['barriers_up', 'barriers_down'])
                if 'StepB' in payoff_name or 'DStepB' in payoff_name:
                    optional_params.extend(['step_param1', 'step_param2', 'step_param3', 'step_param4'])
                if 'BestOf' in payoff_name or 'WorstOf' in payoff_name:
                    optional_params.append('k')
                if 'RankWeighted' in payoff_name:
                    optional_params.append('weights')

                result = {
                    'success': True,
                    'name': payoff_class.__name__,
                    'abbreviation': getattr(payoff_class, 'abbreviation', None),
                    'is_path_dependent': getattr(payoff_class, 'is_path_dependent', False),
                    'required_params': ['strike'],
                    'optional_params': optional_params,
                }
            except Exception as e:
                result = {
                    'success': False,
                    'error': str(e),
                }
    elif command == 'available_tickers':
        result = {
            'success': True,
            'common_tickers': COMMON_TICKERS,
            'default_date_range': DEFAULT_DATE_RANGE,
        }
    else:
        result = {
            'success': False,
            'error': f'Unknown command: {command}. Available: price, stock_info, list_payoffs, payoff_info, available_tickers'
        }

    # Output result as JSON
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
