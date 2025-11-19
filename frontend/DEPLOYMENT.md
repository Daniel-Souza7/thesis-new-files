# Vercel Deployment Guide

Complete guide for deploying the American Options Pricing application to Vercel.

## Quick Start

### 1. Prerequisites

- Vercel account (sign up at https://vercel.com)
- Git repository connected to Vercel
- Node.js 20+ installed locally

### 2. Install Vercel CLI (Optional)

```bash
npm i -g vercel
```

### 3. Deploy to Vercel

#### Option A: Via Vercel Dashboard (Recommended)

1. Go to https://vercel.com/new
2. Import your Git repository
3. Configure project:
   - **Framework Preset**: Next.js
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build` (auto-detected)
   - **Output Directory**: `.next` (auto-detected)
4. Add environment variables (see below)
5. Click "Deploy"

#### Option B: Via CLI

```bash
cd frontend
vercel
```

Follow the prompts to deploy.

## Environment Variables

Configure these in the Vercel dashboard under **Settings → Environment Variables**:

### Required Variables

```bash
# API Configuration
NEXT_PUBLIC_API_URL=https://your-domain.vercel.app

# Node Environment
NODE_ENV=production
```

### Optional Variables

```bash
# Feature Flags
NEXT_PUBLIC_ENABLE_REALTIME_PRICING=true
NEXT_PUBLIC_ENABLE_PATH_STORAGE=false
NEXT_PUBLIC_MAX_PATHS=10000
NEXT_PUBLIC_MAX_STOCKS=10

# Algorithm Configuration
NEXT_PUBLIC_DEFAULT_ALGO=RLSM
NEXT_PUBLIC_ENABLE_TORCH_ALGOS=false  # Must be false on Vercel

# Model Defaults
NEXT_PUBLIC_DEFAULT_MODEL=BlackScholes
NEXT_PUBLIC_DEFAULT_STRIKE=100
NEXT_PUBLIC_DEFAULT_SPOT=100
NEXT_PUBLIC_DEFAULT_RATE=0.05
NEXT_PUBLIC_DEFAULT_VOLATILITY=0.2
NEXT_PUBLIC_DEFAULT_MATURITY=1.0
```

## Python API Routes Setup

### Important Limitations

Vercel has specific constraints for Python serverless functions:

- **Max deployment size**: 250 MB (uncompressed)
- **Max function size**: 50 MB (compressed)
- **Max execution time**: 60 seconds (Pro plan)
- **Memory**: Up to 3008 MB (Pro plan)

### Recommended Approach

Due to size constraints, we use **api-requirements.txt** instead of full **requirements.txt**:

```txt
numpy==1.24.3
scipy==1.11.4
scikit-learn==1.3.2
pandas==2.1.4
```

This excludes:
- **torch** (too large - ~700MB)
- **yfinance** (not needed for pricing)
- **h5py** (not needed for API)

### Algorithms Available on Vercel

With the minimal dependencies:

**Available**:
- RLSM, RFQI (standard options)
- SRLSM, SRFQI (path-dependent options)
- LSM, FQI (baseline methods)

**NOT Available** (require PyTorch):
- NLSM (neural network)
- DOS (deep optimal stopping)

### Alternative: Separate Microservice

For full algorithm support including PyTorch-based methods:

1. Deploy Python backend separately (AWS Lambda, Google Cloud Run, Railway, etc.)
2. Set `PYTHON_API_URL` environment variable in Vercel
3. Proxy requests from Next.js to Python backend

## File Structure for Vercel

```
frontend/
├── api/                    # API routes (Python or TypeScript)
│   ├── price.py           # Main pricing endpoint
│   └── health.py          # Health check
├── app/                    # Next.js app directory
├── public/                 # Static assets
├── vercel.json            # Vercel configuration
├── next.config.mjs        # Next.js configuration
├── api-requirements.txt   # Minimal Python deps for Vercel
├── requirements.txt       # Full deps (for reference/local)
├── package.json           # Node dependencies
└── .env.example           # Environment template
```

## Creating Python API Routes

Example pricing endpoint at `frontend/api/price.py`:

```python
from http.server import BaseHTTPRequestHandler
import json
import sys
import os

# Add parent directory to path to import optimal_stopping
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from optimal_stopping.algorithms.standard.rlsm import RLSM
from optimal_stopping.data.stock_model import BlackScholes
from optimal_stopping.payoffs import get_payoff_class

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        data = json.loads(body)

        # Extract parameters
        algo_name = data.get('algorithm', 'RLSM')
        payoff_name = data.get('payoff', 'BasketCall')
        nb_paths = data.get('nb_paths', 10000)

        # Create model and payoff
        model = BlackScholes(
            drift=data.get('drift', 0.05),
            volatility=data.get('volatility', 0.2),
            spot=data.get('spot', 100),
            rate=data.get('rate', 0.05),
            nb_stocks=data.get('nb_stocks', 5),
            maturity=data.get('maturity', 1.0)
        )

        PayoffClass = get_payoff_class(payoff_name)
        payoff = PayoffClass(strike=data.get('strike', 100))

        # Price option
        algo = RLSM(model, payoff, nb_paths=nb_paths)
        price, comp_time = algo.price()

        # Return result
        result = {
            'price': float(price),
            'computation_time': float(comp_time),
            'algorithm': algo_name,
            'payoff': payoff_name
        }

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'status': 'ok'}).encode())
```

## Deployment Checklist

### Pre-Deployment

- [ ] Update `NEXT_PUBLIC_API_URL` in environment variables
- [ ] Set `NEXT_PUBLIC_ENABLE_TORCH_ALGOS=false`
- [ ] Test build locally: `npm run build`
- [ ] Check bundle size: `npm run analyze`
- [ ] Verify TypeScript: `npm run type-check`
- [ ] Lint code: `npm run lint`

### Vercel Configuration

- [ ] Root directory set to `frontend`
- [ ] Framework preset: Next.js
- [ ] Node version: 20.x
- [ ] Build command: `npm run build`
- [ ] Install command: `npm install`
- [ ] Output directory: `.next`

### Python Configuration

- [ ] Python version: 3.9
- [ ] Requirements file: `api-requirements.txt` (minimal)
- [ ] Function timeout: 60 seconds
- [ ] Memory: 3008 MB (max)

### Post-Deployment

- [ ] Test health endpoint: `https://your-domain.vercel.app/api/health`
- [ ] Test pricing endpoint with sample request
- [ ] Verify environment variables are loaded
- [ ] Check function logs in Vercel dashboard
- [ ] Monitor function execution time and memory usage

## Local Development

### Setup

```bash
# Install Node dependencies
cd frontend
npm install

# Copy environment template
cp .env.example .env.local

# Update .env.local with local values
# NEXT_PUBLIC_API_URL=http://localhost:3000

# Run development server
npm run dev
```

### Testing Python API Routes Locally

Vercel CLI simulates the production environment:

```bash
# Install Vercel CLI
npm i -g vercel

# Run local Vercel environment
vercel dev
```

This runs both Next.js and Python functions locally.

## Performance Optimization

### Bundle Size Reduction

1. Use dynamic imports for heavy components:
```typescript
const HeavyComponent = dynamic(() => import('./HeavyComponent'), {
  loading: () => <p>Loading...</p>,
})
```

2. Optimize images with Next.js Image component
3. Enable compression in `next.config.mjs`
4. Tree-shake unused dependencies

### API Route Optimization

1. Use minimal dependencies in `api-requirements.txt`
2. Cache computation results (Redis/Vercel KV)
3. Implement request debouncing on frontend
4. Use SWR for data fetching with caching

### Cold Start Reduction

Python serverless functions have cold starts (1-3 seconds). To minimize:

1. Keep dependencies minimal
2. Use function warming (ping every 5 minutes)
3. Consider upgrading to Vercel Pro for faster cold starts
4. Implement loading states in UI

## Troubleshooting

### Build Errors

**Error**: `Module not found: Can't resolve 'fs'`

**Solution**: Add to `next.config.mjs`:
```javascript
webpack: (config, { isServer }) => {
  if (!isServer) {
    config.resolve.fallback = { fs: false, path: false };
  }
  return config;
}
```

**Error**: `Deployment exceeds maximum size`

**Solution**: Use `api-requirements.txt` instead of full `requirements.txt`

### Runtime Errors

**Error**: `Function execution timed out`

**Solution**:
- Reduce `nb_paths` in pricing requests
- Increase timeout to 60s in `vercel.json`
- Consider async processing for long computations

**Error**: `ModuleNotFoundError: No module named 'torch'`

**Solution**: Remove torch-dependent algorithms or deploy Python backend separately

### Environment Variables Not Loading

**Solution**:
- Prefix client-side variables with `NEXT_PUBLIC_`
- Redeploy after adding new variables
- Check variable scope (Production/Preview/Development)

## Monitoring

### Vercel Analytics

Enable in dashboard:
- **Analytics**: Track page views and performance
- **Speed Insights**: Monitor Web Vitals
- **Log Drains**: Export logs to external services

### Custom Monitoring

Add to pricing endpoint:

```python
import time

start_time = time.time()
# ... pricing logic ...
execution_time = time.time() - start_time

# Log metrics
print(f"Execution time: {execution_time:.2f}s")
print(f"Memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.2f}MB")
```

## Scaling Considerations

### Vercel Limits

**Free Tier**:
- 100 GB bandwidth/month
- 100 hours function execution/month
- 6,000 function invocations/day

**Pro Tier** ($20/month):
- 1 TB bandwidth
- 1,000 hours function execution
- No daily invocation limit
- 60s function timeout (vs 10s)

### When to Scale Beyond Vercel

Consider alternatives when:
- Pricing computations regularly exceed 60 seconds
- Need PyTorch-based algorithms (NLSM, DOS)
- Require persistent WebSocket connections
- Need background job processing
- Want to use HDF5 path storage

**Alternatives**:
- AWS Lambda + API Gateway (longer timeouts)
- Google Cloud Run (containerized, more resources)
- Railway (full Python environment)
- Dedicated VPS (complete control)

## Additional Resources

- [Vercel Documentation](https://vercel.com/docs)
- [Next.js Documentation](https://nextjs.org/docs)
- [Vercel Python Runtime](https://vercel.com/docs/functions/runtimes/python)
- [Serverless Functions Limits](https://vercel.com/docs/functions/limitations)

## Support

For issues specific to this deployment:
1. Check function logs in Vercel dashboard
2. Review error messages in browser console
3. Test API endpoints with curl/Postman
4. Compare local vs. production behavior

For Vercel platform issues:
- Vercel Support: https://vercel.com/support
- Community: https://github.com/vercel/next.js/discussions
