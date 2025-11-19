# Deployment Quick Start

## Local Development

### Prerequisites
- **Node.js 20+** and npm 10+ ([download](https://nodejs.org/))
- **Python 3.9+** with pip
- Git

### Quick Start

```bash
# 1. Clone and install dependencies
git clone <repo-url>
cd frontend
npm install

# 2. Copy environment file
cp .env.local.example .env.local

# 3. Start development server
npm run dev
```

Visit http://localhost:3000 to see the app. The Next.js dev server auto-reloads on code changes.

### Python Backend Setup (for /api/price)

The API routes use Python. Install requirements:

```bash
# From frontend directory
pip install -r api-requirements.txt

# Or install Python packages system-wide (optional)
cd api && pip install -e .
```

### Local Testing

**Test Health Endpoint:**
```bash
curl http://localhost:3000/api/health
```

**Test Pricing Endpoint:**
```bash
curl -X POST http://localhost:3000/api/price \
  -H "Content-Type: application/json" \
  -d '{
    "algorithm": "RLSM",
    "payoff": "BasketCall",
    "nb_paths": 5000,
    "nb_stocks": 3,
    "strike": 100,
    "spot": 100,
    "drift": 0.05,
    "volatility": 0.2,
    "rate": 0.05,
    "maturity": 1.0
  }'
```

**Run Code Quality Checks:**
```bash
npm run type-check  # TypeScript type checking
npm run lint        # ESLint
npm run format:check # Prettier formatting
```

---

## Vercel Deployment

### One-Click Deploy

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fyourname%2Fthesis-new-files)

### Manual Deployment via Vercel CLI

**1. Install Vercel CLI:**
```bash
npm i -g vercel
```

**2. Configure Environment Variables:**

Create `.env.production.local` in the `frontend` directory:
```
# API Configuration
NEXT_PUBLIC_API_URL=https://your-app.vercel.app
PYTHON_API_URL=https://your-app.vercel.app/api

# Feature Flags (recommended for Vercel)
NEXT_PUBLIC_ENABLE_REALTIME_PRICING=true
NEXT_PUBLIC_ENABLE_PATH_STORAGE=false
NEXT_PUBLIC_ENABLE_TORCH_ALGOS=false
NEXT_PUBLIC_MAX_PATHS=10000
NEXT_PUBLIC_MAX_STOCKS=10
```

**3. Deploy:**
```bash
# From frontend directory
vercel

# Follow prompts to:
# - Link project to Vercel account
# - Set project name
# - Configure environment variables
```

**4. Set Environment Variables in Vercel Dashboard:**
- Go to https://vercel.com/dashboard
- Select your project
- Settings → Environment Variables
- Add variables from step 2

**5. Redeploy:**
```bash
vercel --prod
```

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | Frontend API URL | `https://your-app.vercel.app` |
| `PYTHON_API_URL` | Python backend URL (same as above) | `https://your-app.vercel.app/api` |
| `NEXT_PUBLIC_MAX_PATHS` | Max Monte Carlo paths | `10000` |
| `NEXT_PUBLIC_MAX_STOCKS` | Max basket size | `10` |

### Troubleshooting

**Cold Start Delays (1-3 seconds):**
- Normal behavior for serverless functions
- Python API requires ~1s startup time
- Consider premium plan for faster cold starts

**Timeout Errors (>60 seconds):**
- Reduce `nb_paths` in pricing requests
- Use fast algorithms (RLSM, RFQI, LSM, FQI)
- Switch to async processing for complex calculations

**API Function Not Found:**
```bash
# Verify api-requirements.txt exists
ls frontend/api-requirements.txt

# Rebuild and redeploy
vercel --prod
```

**Build Failed:**
```bash
# Check build logs
vercel logs --production

# Rebuild locally first
npm run build

# Fix TypeScript errors
npm run type-check
```

---

## Testing the Deployed App

### Test /api/price Endpoint

Once deployed, test the API directly:

```bash
curl -X POST https://your-app.vercel.app/api/price \
  -H "Content-Type: application/json" \
  -d '{
    "algorithm": "RLSM",
    "payoff": "BasketCall",
    "nb_paths": 5000,
    "nb_stocks": 3,
    "strike": 100,
    "spot": 100,
    "drift": 0.05,
    "volatility": 0.2,
    "rate": 0.05,
    "maturity": 1.0
  }'
```

Expected response (2-5 seconds):
```json
{
  "status": "success",
  "price": 12.34,
  "computation_time": 0.45,
  "algorithm": "RLSM",
  "payoff": "BasketCall"
}
```

### Calculator Page

1. Open https://your-app.vercel.app/calculator
2. **Edit Parameters:**
   - Algorithm: Select RLSM, RFQI, LSM, FQI, DOS, or NLSM
   - Payoff: Choose from 408 available payoffs
   - Model: Black-Scholes, Heston, Fractional, Rough Heston, Real Data
   - Number of Paths: 1,000-50,000 (default: 10,000)
   - Number of Stocks: 1-20 (default: 5)

3. **View Outputs:**
   - Price with confidence interval
   - Computation time
   - Greeks (Delta, Gamma, Theta, Rho, Vega)

### Interactive Page

1. Open https://your-app.vercel.app/interactive
2. **Adjust Live Sliders:**
   - Spot price, strike, volatility, interest rate, maturity
   - Real-time price updates (if enabled)
3. **Compare Algorithms:**
   - Side-by-side pricing results
   - Performance metrics
4. **Download Results:**
   - Export pricing history as CSV
   - Save parameter configurations

### Monitoring & Logs

**View Deployment Logs:**
```bash
vercel logs --prod
```

**Monitor in Dashboard:**
1. https://vercel.com/dashboard
2. Select project → Functions tab
3. View execution time, memory, error rate for each endpoint

---

## Quick Checklist

- [ ] Node 20+ and Python 3.9+ installed
- [ ] Environment variables configured (.env.local for dev, Vercel dashboard for prod)
- [ ] Local dev server runs: `npm run dev`
- [ ] Health endpoint responds: `curl http://localhost:3000/api/health`
- [ ] Pricing endpoint works with test payload
- [ ] `npm run type-check` passes (no TypeScript errors)
- [ ] Vercel CLI installed: `npm i -g vercel`
- [ ] Project linked to Vercel account
- [ ] Production deployment successful: `vercel --prod`
- [ ] API endpoints accessible on deployed URL

## Additional Resources

- [DEPLOYMENT.md](./DEPLOYMENT.md) - Detailed deployment guide
- [DEPLOYMENT_REQUIREMENTS.md](./DEPLOYMENT_REQUIREMENTS.md) - System requirements
- [API_INTEGRATION_SUMMARY.md](./API_INTEGRATION_SUMMARY.md) - API architecture
- [Vercel CLI Docs](https://vercel.com/cli)
- [Next.js Deployment Docs](https://nextjs.org/docs/deployment)
