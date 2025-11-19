# API Routes

Python-based serverless functions for Vercel deployment.

## Overview

This directory contains Python API routes that are deployed as serverless functions on Vercel. Each `.py` file becomes an endpoint at `/api/{filename}`.

## Available Endpoints

### GET /api/health

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2025-11-19T12:00:00Z",
  "version": "0.1.0",
  "environment": "production",
  "python_available": true
}
```

### POST /api/price

Price an American option (TEMPLATE - requires implementation).

**Request:**
```json
{
  "algorithm": "RLSM",
  "payoff": "BasketCall",
  "model": "BlackScholes",
  "nb_paths": 10000,
  "nb_stocks": 5,
  "strike": 100,
  "spot": 100,
  "drift": 0.05,
  "volatility": 0.2,
  "rate": 0.05,
  "maturity": 1.0
}
```

**Response:**
```json
{
  "price": 12.34,
  "computation_time": 0.45,
  "algorithm": "RLSM",
  "payoff": "BasketCall",
  "model": "BlackScholes",
  "parameters": { ... },
  "timestamp": "2025-11-19T12:00:00Z",
  "status": "success"
}
```

## Vercel Function Format

All Python API routes must follow this format:

```python
from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Handle GET request
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'status': 'ok'}).encode())

    def do_POST(self):
        # Handle POST request
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        data = json.loads(body)

        # Process request...

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())
```

## Dependencies

Functions have access to packages listed in `/api-requirements.txt`:

- numpy
- scipy
- scikit-learn
- pandas

**NOT available** (too large for Vercel):
- torch (use alternative algorithms)
- h5py (not needed for API)
- yfinance (not needed for API)

## Implementation Notes

### Integrating optimal_stopping Module

To use the `optimal_stopping` module in API routes:

**Option 1: Copy Module** (Recommended for Vercel)
```bash
# Copy optimal_stopping into api directory
cp -r ../optimal_stopping ./api/
```

Then import normally:
```python
from optimal_stopping.algorithms.standard.rlsm import RLSM
```

**Option 2: Adjust Python Path**
```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from optimal_stopping.algorithms.standard.rlsm import RLSM
```

### CORS Configuration

All endpoints include CORS headers:

```python
self.send_header('Access-Control-Allow-Origin', '*')
self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
self.send_header('Access-Control-Allow-Headers', 'Content-Type')
```

Adjust `Access-Control-Allow-Origin` for production to restrict access.

### Error Handling

Always include try/catch and return proper error responses:

```python
try:
    # Process request
    result = {'status': 'success'}
except ValueError as e:
    # Client error (400)
    error = {'status': 'error', 'error': str(e)}
    self.send_response(400)
    # ...
except Exception as e:
    # Server error (500)
    error = {'status': 'error', 'error': 'Internal server error'}
    self.send_response(500)
    # ...
```

### Timeout Considerations

Vercel functions have a maximum timeout:
- **Free tier**: 10 seconds
- **Pro tier**: 60 seconds

Keep computations under this limit by:
- Limiting `nb_paths` (e.g., max 50,000)
- Using efficient algorithms (RLSM, RFQI, LSM, FQI)
- Returning early if computation will exceed timeout
- Implementing async processing for longer computations

### Memory Limits

Functions can use up to 3008 MB of memory. Monitor usage:

```python
import resource

mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"Memory usage: {mem_usage:.2f} MB")
```

## Testing Locally

### Using Vercel CLI

```bash
# Install Vercel CLI
npm i -g vercel

# Run local development server
vercel dev
```

This simulates the Vercel environment locally.

### Using curl

Test health endpoint:
```bash
curl http://localhost:3000/api/health
```

Test pricing endpoint:
```bash
curl -X POST http://localhost:3000/api/price \
  -H "Content-Type: application/json" \
  -d '{
    "algorithm": "RLSM",
    "payoff": "BasketCall",
    "nb_paths": 10000,
    "nb_stocks": 5,
    "strike": 100,
    "spot": 100,
    "drift": 0.05,
    "volatility": 0.2,
    "rate": 0.05,
    "maturity": 1.0
  }'
```

## Deployment

API routes are automatically deployed with the Next.js application. No additional configuration needed beyond:

1. `vercel.json` - Function timeout and memory settings
2. `api-requirements.txt` - Python dependencies

See [DEPLOYMENT.md](../DEPLOYMENT.md) for complete deployment guide.

## Monitoring

After deployment, monitor function performance:

1. **Vercel Dashboard** → Deployments → Your Deployment → Functions
2. Check execution time, memory usage, error rate
3. Review logs for errors or warnings
4. Optimize based on metrics

## Limitations

- Max execution time: 60 seconds
- Max function size: 50 MB compressed
- Max deployment size: 250 MB uncompressed
- No PyTorch support (too large)
- Cold start latency (1-3 seconds)

See [DEPLOYMENT.md](../DEPLOYMENT.md) for workarounds and alternatives.
