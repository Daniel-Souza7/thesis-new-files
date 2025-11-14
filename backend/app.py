"""
Flask Backend for Optimal Stopping Explorer
Serves API endpoints for running American option pricing algorithms
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import time
import sys
import os

# Add parent directory to path to import optimal_stopping
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optimal_stopping.algorithms.standard.rlsm import RLSM
from optimal_stopping.algorithms.standard.rfqi import RFQI
from optimal_stopping.algorithms.standard.lsm import LSM
from optimal_stopping.algorithms.standard.nlsm import NLSM
from optimal_stopping.algorithms.standard.dos import DOS
from optimal_stopping.algorithms.standard.fqi import FQI
from optimal_stopping.data.stock_model import BlackScholes
from optimal_stopping.data.real_data import RealDataModel
from optimal_stopping.payoffs import MaxCall, BasketCall, MinCall, GeometricBasketCall

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Algorithm mapping
ALGORITHMS = {
    'RLSM': RLSM,
    'RFQI': RFQI,
    'LSM': LSM,
    'NLSM': NLSM,
    'DOS': DOS,
    'FQI': FQI
}

# Payoff mapping
PAYOFFS = {
    'MaxCall': MaxCall,
    'BasketCall': BasketCall,
    'MinCall': MinCall,
    'GeometricBasketCall': GeometricBasketCall
}


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Optimal Stopping Explorer API is running'})


@app.route('/api/price', methods=['POST'])
def price_option():
    """
    Price an American option using specified algorithm.

    Request JSON:
    {
        "algorithm": "RLSM",
        "model": "BlackScholes",
        "payoff": "MaxCall",
        "nb_stocks": 5,
        "nb_paths": 10000,
        "nb_dates": 52,
        "spot": 100,
        "strike": 100,
        "maturity": 1.0,
        "volatility": 0.2,
        "drift": 0.05,
        "rate": 0.05,
        "hidden_size": 100,
        "nb_epochs": 20
    }

    Returns:
    {
        "price": 12.34,
        "time_path_gen": 0.5,
        "time_pricing": 1.2,
        "total_time": 1.7
    }
    """
    try:
        data = request.json

        # Extract parameters
        algo_name = data.get('algorithm', 'RLSM')
        model_name = data.get('model', 'BlackScholes')
        payoff_name = data.get('payoff', 'MaxCall')

        nb_stocks = data.get('nb_stocks', 5)
        nb_paths = data.get('nb_paths', 10000)
        nb_dates = data.get('nb_dates', 52)
        spot = data.get('spot', 100)
        strike = data.get('strike', 100)
        maturity = data.get('maturity', 1.0)
        volatility = data.get('volatility', 0.2)
        drift = data.get('drift', 0.05)
        rate = data.get('rate', 0.05)

        hidden_size = data.get('hidden_size', 100)
        nb_epochs = data.get('nb_epochs', 20)

        # Create model
        if model_name == 'BlackScholes':
            model = BlackScholes(
                nb_stocks=nb_stocks,
                nb_paths=nb_paths,
                nb_dates=nb_dates,
                spot=spot,
                strike=strike,
                maturity=maturity,
                volatility=volatility,
                rate=rate,
                drift=drift
            )
        else:  # RealData
            model = RealDataModel(
                nb_stocks=nb_stocks,
                nb_paths=nb_paths,
                nb_dates=nb_dates,
                spot=spot,
                strike=strike,
                maturity=maturity,
                rate=rate,
                drift=drift
            )

        # Create payoff
        PayoffClass = PAYOFFS[payoff_name]
        payoff = PayoffClass(strike)

        # Create algorithm
        AlgoClass = ALGORITHMS[algo_name]

        if algo_name in ['RLSM', 'RFQI']:
            pricer = AlgoClass(model, payoff, hidden_size=hidden_size, train_ITM_only=True)
        elif algo_name in ['NLSM', 'DOS']:
            pricer = AlgoClass(model, payoff, hidden_size=hidden_size, nb_epochs=nb_epochs, train_ITM_only=True)
        elif algo_name == 'LSM':
            pricer = AlgoClass(model, payoff, train_ITM_only=True)
        else:  # FQI
            pricer = AlgoClass(model, payoff, nb_epochs=nb_epochs, train_ITM_only=True)

        # Price option
        t_start = time.time()
        price, time_path_gen = pricer.price(train_eval_split=2)
        total_time = time.time() - t_start
        time_pricing = total_time - time_path_gen

        return jsonify({
            'price': float(price),
            'time_path_gen': float(time_path_gen),
            'time_pricing': float(time_pricing),
            'total_time': float(total_time)
        })

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in price_option: {error_trace}")
        return jsonify({
            'error': str(e),
            'details': error_trace
        }), 500


@app.route('/api/generate_paths', methods=['POST'])
def generate_paths():
    """
    Generate sample stock paths for visualization.

    Request JSON:
    {
        "model": "BlackScholes",
        "nb_stocks": 3,
        "nb_paths": 50,
        "nb_dates": 52,
        "spot": 100,
        "maturity": 1.0,
        "volatility": 0.2,
        "drift": 0.05,
        "rate": 0.05
    }

    Returns:
    {
        "paths": [...],  # shape [nb_paths, nb_stocks, nb_dates+1]
        "time_grid": [0, 0.019, 0.038, ...]
    }
    """
    try:
        data = request.json

        model_name = data.get('model', 'BlackScholes')
        nb_stocks = data.get('nb_stocks', 3)
        nb_paths = data.get('nb_paths', 50)
        nb_dates = data.get('nb_dates', 52)
        spot = data.get('spot', 100)
        maturity = data.get('maturity', 1.0)
        volatility = data.get('volatility', 0.2)
        drift = data.get('drift', 0.05)
        rate = data.get('rate', 0.05)
        strike = data.get('strike', 100)

        # Create model
        if model_name == 'BlackScholes':
            model = BlackScholes(
                nb_stocks=nb_stocks,
                nb_paths=nb_paths,
                nb_dates=nb_dates,
                spot=spot,
                strike=strike,
                maturity=maturity,
                volatility=volatility,
                rate=rate,
                drift=drift
            )
        else:
            model = RealDataModel(
                nb_stocks=nb_stocks,
                nb_paths=nb_paths,
                nb_dates=nb_dates,
                spot=spot,
                strike=strike,
                maturity=maturity,
                rate=rate,
                drift=drift
            )

        # Generate paths
        stock_paths, _ = model.generate_paths()

        # Create time grid
        time_grid = np.linspace(0, maturity, nb_dates + 1).tolist()

        return jsonify({
            'paths': stock_paths.tolist(),
            'time_grid': time_grid
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/compare', methods=['POST'])
def compare_algorithms():
    """
    Compare multiple algorithms on the same problem.

    Request JSON:
    {
        "algorithms": ["RLSM", "LSM", "NLSM"],
        "model": "BlackScholes",
        "payoff": "MaxCall",
        "nb_stocks": 5,
        ... (other parameters)
    }

    Returns:
    {
        "results": [
            {"algorithm": "RLSM", "price": 12.34, "time": 1.2},
            {"algorithm": "LSM", "price": 12.30, "time": 0.8},
            ...
        ]
    }
    """
    try:
        data = request.json
        algos = data.get('algorithms', ['RLSM', 'LSM'])

        results = []

        for algo_name in algos:
            # Create a copy of data with current algorithm
            algo_data = data.copy()
            algo_data['algorithm'] = algo_name

            # Use the price endpoint logic
            response = price_option_internal(algo_data)

            results.append({
                'algorithm': algo_name,
                'price': response['price'],
                'time': response['total_time']
            })

        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


def price_option_internal(data):
    """Internal function to price option (reused by compare endpoint)"""
    algo_name = data.get('algorithm', 'RLSM')
    model_name = data.get('model', 'BlackScholes')
    payoff_name = data.get('payoff', 'MaxCall')

    nb_stocks = data.get('nb_stocks', 5)
    nb_paths = data.get('nb_paths', 10000)
    nb_dates = data.get('nb_dates', 52)
    spot = data.get('spot', 100)
    strike = data.get('strike', 100)
    maturity = data.get('maturity', 1.0)
    volatility = data.get('volatility', 0.2)
    drift = data.get('drift', 0.05)
    rate = data.get('rate', 0.05)
    hidden_size = data.get('hidden_size', 100)
    nb_epochs = data.get('nb_epochs', 20)

    # Create model
    if model_name == 'BlackScholes':
        model = BlackScholes(
            nb_stocks=nb_stocks, nb_paths=nb_paths, nb_dates=nb_dates,
            spot=spot, strike=strike, maturity=maturity,
            volatility=volatility, rate=rate, drift=drift
        )
    else:
        model = RealDataModel(
            nb_stocks=nb_stocks, nb_paths=nb_paths, nb_dates=nb_dates,
            spot=spot, strike=strike, maturity=maturity,
            rate=rate, drift=drift
        )

    # Create payoff
    PayoffClass = PAYOFFS[payoff_name]
    payoff = PayoffClass(strike)

    # Create algorithm
    AlgoClass = ALGORITHMS[algo_name]

    if algo_name in ['RLSM', 'RFQI']:
        pricer = AlgoClass(model, payoff, hidden_size=hidden_size, train_ITM_only=True)
    elif algo_name in ['NLSM', 'DOS']:
        pricer = AlgoClass(model, payoff, hidden_size=hidden_size, nb_epochs=nb_epochs, train_ITM_only=True)
    elif algo_name == 'LSM':
        pricer = AlgoClass(model, payoff, train_ITM_only=True)
    else:  # FQI
        pricer = AlgoClass(model, payoff, nb_epochs=nb_epochs, train_ITM_only=True)

    # Price option
    t_start = time.time()
    price, time_path_gen = pricer.price(train_eval_split=2)
    total_time = time.time() - t_start

    return {
        'price': float(price),
        'time_path_gen': float(time_path_gen),
        'time_pricing': float(total_time - time_path_gen),
        'total_time': float(total_time)
    }


if __name__ == '__main__':
    print("🚀 Optimal Stopping Explorer API starting...")
    print("📍 API will be available at: http://localhost:5000")
    print("📋 Endpoints:")
    print("   GET  /api/health          - Health check")
    print("   POST /api/price           - Price an option")
    print("   POST /api/generate_paths  - Generate stock paths")
    print("   POST /api/compare         - Compare algorithms")
    app.run(debug=True, host='0.0.0.0', port=5000)
