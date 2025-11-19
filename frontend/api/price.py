"""
Option pricing endpoint for Vercel deployment.

POST /api/price - Price an American option using specified algorithm

This is a TEMPLATE. Actual implementation requires:
1. optimal_stopping module to be accessible
2. Dependencies from api-requirements.txt installed
3. Proper error handling for production use
"""

from http.server import BaseHTTPRequestHandler
import json
import sys
import os
from datetime import datetime

# IMPORTANT: Adjust this path based on your deployment structure
# For Vercel, you may need to copy the optimal_stopping module into the api directory
# or configure the Python path appropriately
# sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


class handler(BaseHTTPRequestHandler):
    """Vercel serverless function handler for option pricing."""

    def do_POST(self):
        """Handle POST requests to pricing endpoint."""
        try:
            # Parse request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)

            # Extract parameters with defaults
            algo_name = data.get('algorithm', 'RLSM')
            payoff_name = data.get('payoff', 'BasketCall')
            model_name = data.get('model', 'BlackScholes')
            nb_paths = int(data.get('nb_paths', 10000))
            nb_stocks = int(data.get('nb_stocks', 5))
            strike = float(data.get('strike', 100))
            spot = float(data.get('spot', 100))
            drift = float(data.get('drift', 0.05))
            volatility = float(data.get('volatility', 0.2))
            rate = float(data.get('rate', 0.05))
            maturity = float(data.get('maturity', 1.0))

            # Validate inputs
            if nb_paths > 50000:
                raise ValueError("nb_paths exceeds maximum allowed value (50000)")
            if nb_stocks > 20:
                raise ValueError("nb_stocks exceeds maximum allowed value (20)")
            if maturity <= 0:
                raise ValueError("maturity must be positive")

            # TODO: Uncomment when optimal_stopping is available
            # from optimal_stopping.algorithms.standard.rlsm import RLSM
            # from optimal_stopping.data.stock_model import BlackScholes
            # from optimal_stopping.payoffs import get_payoff_class

            # Create model
            # model = BlackScholes(
            #     drift=drift,
            #     volatility=volatility,
            #     spot=spot,
            #     rate=rate,
            #     nb_stocks=nb_stocks,
            #     maturity=maturity
            # )

            # Create payoff
            # PayoffClass = get_payoff_class(payoff_name)
            # payoff = PayoffClass(strike=strike)

            # Price option
            # algo = RLSM(model, payoff, nb_paths=nb_paths)
            # price, comp_time = algo.price()

            # Mock response for now
            price = 12.34
            comp_time = 0.45

            # Build response
            result = {
                'price': float(price),
                'computation_time': float(comp_time),
                'algorithm': algo_name,
                'payoff': payoff_name,
                'model': model_name,
                'parameters': {
                    'nb_paths': nb_paths,
                    'nb_stocks': nb_stocks,
                    'strike': strike,
                    'spot': spot,
                    'drift': drift,
                    'volatility': volatility,
                    'rate': rate,
                    'maturity': maturity
                },
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'status': 'success'
            }

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        except ValueError as e:
            self._send_error(400, str(e))
        except Exception as e:
            self._send_error(500, f"Internal server error: {str(e)}")

    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def _send_error(self, status_code, message):
        """Send error response."""
        error_response = {
            'status': 'error',
            'error': message,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }

        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(error_response).encode())
