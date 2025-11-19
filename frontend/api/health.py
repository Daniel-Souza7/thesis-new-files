"""
Health check endpoint for Vercel deployment.

GET /api/health - Returns status and version information
"""

from http.server import BaseHTTPRequestHandler
import json
from datetime import datetime


class handler(BaseHTTPRequestHandler):
    """Vercel serverless function handler for health checks."""

    def do_GET(self):
        """Handle GET requests to health endpoint."""
        response = {
            'status': 'ok',
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'version': '0.1.0',
            'environment': 'production',
            'python_available': True
        }

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
