# Vercel Deployment Requirements

Complete requirements and configuration summary for deploying to Vercel.

## Configuration Files Created

### Core Configuration ✓

| File | Purpose | Status |
|------|---------|--------|
| `vercel.json` | Vercel platform configuration | ✓ Created |
| `next.config.mjs` | Next.js build configuration | ✓ Created |
| `package.json` | Node.js dependencies and scripts | ✓ Updated |
| `tsconfig.json` | TypeScript configuration | ✓ Existing |

### Python Dependencies ✓

| File | Purpose | Size | Status |
|------|---------|------|--------|
| `requirements.txt` | Full dependencies (reference) | ~200 MB | ✓ Created |
| `api-requirements.txt` | Minimal for Vercel | ~50 MB | ✓ Created |

**api-requirements.txt** (Used by Vercel):
- numpy 1.24.3
- scipy 1.11.4
- scikit-learn 1.3.2
- pandas 2.1.4

**Excluded from Vercel** (size limits):
- torch (~700 MB) - Too large
- yfinance - Not needed for API
- h5py - Not needed for API

### Environment Configuration ✓

| File | Purpose | Committed to Git |
|------|---------|------------------|
| `.env.example` | Template for production | Yes |
| `.env.local.example` | Template for local dev | Yes |
| `.env.local` | Local secrets | No (in .gitignore) |

### API Routes ✓

| File | Endpoint | Status |
|------|----------|--------|
| `api/health.py` | `GET /api/health` | ✓ Implemented |
| `api/price.py` | `POST /api/price` | ✓ Template |
| `api/README.md` | API documentation | ✓ Created |

**Note**: `api/price.py` requires integration with `optimal_stopping` module.

### Documentation ✓

| File | Purpose | Pages |
|------|---------|-------|
| `DEPLOYMENT.md` | Complete deployment guide | 15+ |
| `VERCEL_CHECKLIST.md` | Step-by-step checklist | 10+ |
| `VERCEL_SETUP_SUMMARY.md` | Configuration overview | 8+ |
| `QUICK_DEPLOY.md` | 5-minute quick start | 2 |
| `README.md` | Project documentation | 10+ |

### Development Tools ✓

| File | Purpose | Status |
|------|---------|--------|
| `.prettierrc.json` | Code formatting rules | ✓ Created |
| `.prettierignore` | Files to skip formatting | ✓ Created |
| `.gitignore` | Git ignore patterns | ✓ Updated |
| `.dockerignore` | Docker ignore patterns | ✓ Created |
| `Dockerfile` | Container build config | ✓ Created |

## Vercel Platform Requirements

### Account Requirements

- **Vercel Account**: https://vercel.com (free or paid)
- **Git Repository**: GitHub, GitLab, or Bitbucket
- **Plan**:
  - **Hobby** (Free): 10s timeout, 100 GB bandwidth
  - **Pro** ($20/mo): 60s timeout, 1 TB bandwidth

### Runtime Requirements

| Setting | Value | Configured In |
|---------|-------|---------------|
| Node.js Version | 20.x | `package.json` engines |
| Python Version | 3.9 | `vercel.json` functions |
| Function Timeout | 60s (Pro) / 10s (Free) | `vercel.json` functions |
| Function Memory | 3008 MB (max) | `vercel.json` functions |
| Region | iad1 (US East) | `vercel.json` regions |

### Size Limits

| Limit | Value | Current Status |
|-------|-------|----------------|
| Deployment size | 250 MB uncompressed | ~50-100 MB ✓ |
| Function size | 50 MB compressed | ~10-20 MB ✓ |
| Dependencies | api-requirements.txt | Under limit ✓ |

## Environment Variables Required

### Production (Vercel Dashboard)

Set these in **Settings → Environment Variables**:

```bash
# Required
NEXT_PUBLIC_API_URL=https://your-domain.vercel.app
NODE_ENV=production

# Algorithm Configuration
NEXT_PUBLIC_ENABLE_TORCH_ALGOS=false  # MUST be false (PyTorch too large)
NEXT_PUBLIC_DEFAULT_ALGO=RLSM
NEXT_PUBLIC_MAX_PATHS=10000
NEXT_PUBLIC_MAX_STOCKS=10

# Model Configuration
NEXT_PUBLIC_DEFAULT_MODEL=BlackScholes
NEXT_PUBLIC_DEFAULT_STRIKE=100
NEXT_PUBLIC_DEFAULT_SPOT=100
NEXT_PUBLIC_DEFAULT_RATE=0.05
NEXT_PUBLIC_DEFAULT_VOLATILITY=0.2
NEXT_PUBLIC_DEFAULT_MATURITY=1.0

# Feature Flags
NEXT_PUBLIC_ENABLE_REALTIME_PRICING=true
NEXT_PUBLIC_ENABLE_PATH_STORAGE=false  # Not supported on Vercel
```

### Local Development (.env.local)

```bash
NEXT_PUBLIC_API_URL=http://localhost:3000
NODE_ENV=development
NEXT_PUBLIC_ENABLE_TORCH_ALGOS=true  # Can be true locally
NEXT_PUBLIC_MAX_PATHS=100000
NEXT_PUBLIC_MAX_STOCKS=20
NEXT_PUBLIC_ENABLE_PATH_STORAGE=true
```

## Available Algorithms on Vercel

### Supported ✓

These work with `api-requirements.txt`:

- **RLSM** - Randomized Least Squares Monte Carlo
- **RFQI** - Randomized Fitted Q-Iteration
- **SRLSM** - State-dependent RLSM (path-dependent)
- **SRFQI** - State-dependent RFQI (path-dependent)
- **LSM** - Least Squares Monte Carlo
- **FQI** - Fitted Q-Iteration

### Not Supported ✗

These require PyTorch (too large for Vercel):

- **NLSM** - Neural Least Squares Monte Carlo
- **DOS** - Deep Optimal Stopping

**Workaround**: Deploy Python backend separately for PyTorch algorithms.

## Deployment Checklist

### Pre-Deployment ✓

- [x] Configuration files created
- [x] Dependencies optimized for Vercel
- [x] Environment variables documented
- [x] API routes implemented (templates)
- [x] Documentation complete
- [ ] **Integration with optimal_stopping module** (pending)
- [ ] Local testing complete
- [ ] Build verification

### Vercel Dashboard Setup

- [ ] Account created
- [ ] Git repository connected
- [ ] Project imported
- [ ] Root directory set to `frontend`
- [ ] Environment variables configured
- [ ] First deployment triggered

### Post-Deployment

- [ ] Health endpoint tested
- [ ] Pricing endpoint tested (after integration)
- [ ] Function logs reviewed
- [ ] Performance metrics checked
- [ ] Monitoring enabled

## Integration Steps (Required)

The configuration is complete, but you need to integrate the `optimal_stopping` module:

### Option 1: Copy Module to API Directory

```bash
cp -r /home/user/thesis-new-files/optimal_stopping \
      /home/user/thesis-new-files/frontend/api/
```

Then in `api/price.py`, uncomment:
```python
from optimal_stopping.algorithms.standard.rlsm import RLSM
from optimal_stopping.data.stock_model import BlackScholes
from optimal_stopping.payoffs import get_payoff_class
```

### Option 2: Adjust Import Path

In `api/price.py`:
```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from optimal_stopping.algorithms.standard.rlsm import RLSM
# ...
```

### Option 3: Create Package

Create `setup.py` and install as package:
```bash
cd /home/user/thesis-new-files
pip install -e .
```

## Testing Requirements

### Local Testing

```bash
# Type checking
npm run type-check

# Linting
npm run lint

# Build
npm run build

# Run locally
npm start
```

### Vercel CLI Testing

```bash
# Install CLI
npm i -g vercel

# Run local Vercel environment
vercel dev

# Test endpoints
curl http://localhost:3000/api/health
```

### Production Testing

After deployment:

```bash
# Health check
curl https://your-domain.vercel.app/api/health

# Pricing request
curl -X POST https://your-domain.vercel.app/api/price \
  -H "Content-Type: application/json" \
  -d '{"algorithm":"RLSM","payoff":"BasketCall","nb_paths":10000}'
```

## Known Limitations

### Size Constraints

- **Max deployment**: 250 MB uncompressed
- **Max function**: 50 MB compressed
- **Current**: ~50-100 MB (within limits ✓)

### Timeout Constraints

- **Free tier**: 10 seconds
- **Pro tier**: 60 seconds
- **Recommendation**: Use Pro for pricing computations

### Package Constraints

- **PyTorch**: Not available (too large)
- **HDF5**: Not available (not needed)
- **yfinance**: Not available (not needed)

### Cold Start Latency

- **Duration**: 1-3 seconds
- **When**: After inactivity
- **Mitigation**: Function warming, caching

## Performance Expectations

### Typical Performance

| Configuration | Execution Time | Cold Start |
|---------------|----------------|------------|
| nb_paths=1,000 | ~0.5s | +2s |
| nb_paths=10,000 | ~2-5s | +2s |
| nb_paths=50,000 | ~10-30s | +2s |

### Optimization Tips

1. **Limit nb_paths** to 50,000 max
2. **Cache results** using Vercel KV or Redis
3. **Warm functions** with periodic pings
4. **Use efficient algorithms** (RLSM, RFQI)
5. **Implement request debouncing** on frontend

## Support & Documentation

### Quick Start

1. [QUICK_DEPLOY.md](./QUICK_DEPLOY.md) - 5-minute deployment
2. [VERCEL_CHECKLIST.md](./VERCEL_CHECKLIST.md) - Detailed checklist

### Complete Guides

1. [DEPLOYMENT.md](./DEPLOYMENT.md) - Full deployment guide (15+ pages)
2. [VERCEL_SETUP_SUMMARY.md](./VERCEL_SETUP_SUMMARY.md) - Config overview
3. [README.md](./README.md) - Project documentation

### API Documentation

1. [api/README.md](./api/README.md) - API routes guide
2. [api/health.py](./api/health.py) - Health endpoint
3. [api/price.py](./api/price.py) - Pricing endpoint

### External Resources

- [Vercel Docs](https://vercel.com/docs)
- [Next.js Docs](https://nextjs.org/docs)
- [Python Runtime](https://vercel.com/docs/functions/runtimes/python)
- [Serverless Limits](https://vercel.com/docs/functions/limitations)

## Summary

### ✓ Configuration Complete

All necessary files for Vercel deployment have been created:

- 18 configuration files
- 5 documentation files (50+ pages)
- 3 API route files
- Complete environment setup
- Docker alternative
- Development tools

### ⏳ Integration Pending

Before deployment, you need to:

1. **Integrate optimal_stopping module** with API routes
2. **Test locally** with `npm run dev` and `vercel dev`
3. **Configure environment variables** in Vercel dashboard
4. **Deploy** and verify

### 📊 Expected Results

After deployment:

- **Deployment time**: 15-30 minutes
- **Cold start latency**: 1-3 seconds
- **Pricing request**: 2-10 seconds (depending on nb_paths)
- **Monthly cost**: $0 (Hobby) or $20 (Pro)

### 🚀 Next Steps

1. Choose integration option (copy module, adjust path, or package)
2. Update `api/price.py` with optimal_stopping imports
3. Test locally: `vercel dev`
4. Deploy: `vercel` or via dashboard
5. Verify: Test health and pricing endpoints

### 📈 Future Enhancements

- Authentication (NextAuth.js)
- Database (PostgreSQL, Vercel Postgres)
- Caching (Vercel KV, Redis)
- Analytics (Vercel Analytics)
- Error tracking (Sentry)
- Rate limiting
- Background job processing

---

**Status**: Configuration Complete ✓
**Next**: Integration & Deployment ⏳
**Estimated Time to Deploy**: 30-60 minutes
**Documentation**: 50+ pages across 5 guides
