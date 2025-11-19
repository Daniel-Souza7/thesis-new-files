# Deployment Documentation Index

Quick navigation to all Vercel deployment documentation.

## Getting Started

### New to Deployment?

Start here: **[QUICK_DEPLOY.md](./QUICK_DEPLOY.md)** (5 minutes)
- Fastest path to deployment
- Essential commands only
- Quick troubleshooting

### First Time Setup?

Read: **[DEPLOYMENT_REQUIREMENTS.md](./DEPLOYMENT_REQUIREMENTS.md)** (6 pages)
- What you need before starting
- Integration options
- Complete requirements list
- Checklist format

## Complete Guides

### Full Deployment Guide

**[DEPLOYMENT.md](./DEPLOYMENT.md)** (15+ pages)
- Comprehensive step-by-step instructions
- Environment variable setup
- Python API route configuration
- Performance optimization
- Monitoring and logging
- Troubleshooting guide
- Production best practices

### Detailed Checklist

**[VERCEL_CHECKLIST.md](./VERCEL_CHECKLIST.md)** (10+ pages)
- Pre-deployment verification (7 sections)
- Vercel dashboard configuration
- First deployment steps
- Post-deployment tasks
- Known limitations & workarounds
- Rollback procedures
- Success criteria

### Configuration Overview

**[VERCEL_SETUP_SUMMARY.md](./VERCEL_SETUP_SUMMARY.md)** (8+ pages)
- All created files explained
- File structure diagram
- Key features
- Vercel-specific limitations
- Alternative deployment options
- Integration instructions
- Next steps

## Project Documentation

### Main README

**[README.md](./README.md)** (10+ pages)
- Project overview
- Quick start guide
- Technology stack
- Available scripts
- API documentation
- Development guide
- Troubleshooting

## API Documentation

### API Routes Guide

**[api/README.md](./api/README.md)**
- Available endpoints
- Request/response formats
- Vercel function format
- Implementation notes
- CORS configuration
- Error handling
- Testing locally

### Health Endpoint

**[api/health.py](./api/health.py)**
- GET /api/health
- Fully implemented
- Returns status and version

### Pricing Endpoint

**[api/price.py](./api/price.py)**
- POST /api/price
- Template implementation
- Requires integration with optimal_stopping module

## Configuration Files

### Vercel Configuration

- **[vercel.json](./vercel.json)** - Platform settings
  - Python 3.9 + Node.js 20
  - 60s timeout, 3008 MB memory
  - CORS headers

### Next.js Configuration

- **[next.config.mjs](./next.config.mjs)** - Build settings
  - Webpack + Turbopack
  - Python integration
  - Bundle optimization

### Dependencies

- **[api-requirements.txt](./api-requirements.txt)** - Vercel (minimal)
  - numpy, scipy, scikit-learn, pandas
  - ~50 MB total

- **[requirements.txt](./requirements.txt)** - Local dev (full)
  - All project dependencies
  - ~200 MB total

### Environment Variables

- **[.env.example](./.env.example)** - Production template
- **[.env.local.example](./.env.local.example)** - Local template

## Development Tools

- **[.prettierrc.json](./.prettierrc.json)** - Code formatting
- **[.gitignore](./.gitignore)** - Git exclusions
- **[Dockerfile](./Dockerfile)** - Docker alternative
- **[package.json](./package.json)** - Node dependencies

## Quick Reference

### Key Commands

```bash
# Local development
npm run dev              # Start dev server
npm run build            # Build for production
npm run type-check       # TypeScript validation
npm run lint             # ESLint

# Vercel deployment
npm i -g vercel          # Install CLI
vercel dev               # Local Vercel environment
vercel                   # Deploy preview
vercel --prod            # Deploy to production
```

### Environment Variables (Production)

Set in Vercel Dashboard → Settings → Environment Variables:

```bash
# Required
NEXT_PUBLIC_API_URL=https://your-domain.vercel.app
NODE_ENV=production
NEXT_PUBLIC_ENABLE_TORCH_ALGOS=false

# Optional
NEXT_PUBLIC_DEFAULT_ALGO=RLSM
NEXT_PUBLIC_MAX_PATHS=10000
NEXT_PUBLIC_MAX_STOCKS=10
```

### API Endpoints

```bash
# Health check
GET /api/health

# Pricing
POST /api/price
Content-Type: application/json
{
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
}
```

## File Structure

```
frontend/
├── INDEX.md                    # This file
├── QUICK_DEPLOY.md             # 5-minute guide
├── DEPLOYMENT.md               # Complete guide (15+ pages)
├── DEPLOYMENT_REQUIREMENTS.md  # Requirements summary
├── VERCEL_CHECKLIST.md         # Detailed checklist
├── VERCEL_SETUP_SUMMARY.md     # Configuration overview
├── README.md                   # Project documentation
│
├── vercel.json                 # Vercel config
├── next.config.mjs             # Next.js config
├── package.json                # Node dependencies
├── tsconfig.json               # TypeScript config
│
├── requirements.txt            # Full Python deps
├── api-requirements.txt        # Minimal Python deps
│
├── .env.example                # Env template (production)
├── .env.local.example          # Env template (local)
│
├── api/                        # Python API routes
│   ├── README.md               # API documentation
│   ├── health.py               # Health endpoint
│   └── price.py                # Pricing endpoint (template)
│
├── app/                        # Next.js app
├── components/                 # React components
├── lib/                        # Utilities
└── public/                     # Static assets
```

## Deployment Workflow

1. **Read** [DEPLOYMENT_REQUIREMENTS.md](./DEPLOYMENT_REQUIREMENTS.md)
2. **Integrate** optimal_stopping module with API routes
3. **Test** locally with `npm run dev` and `vercel dev`
4. **Follow** [VERCEL_CHECKLIST.md](./VERCEL_CHECKLIST.md)
5. **Deploy** to Vercel
6. **Verify** endpoints work
7. **Monitor** performance

## Integration Options

Before deploying, integrate the optimal_stopping module:

**Option 1: Copy Module** (Recommended)
```bash
cp -r /home/user/thesis-new-files/optimal_stopping \
      /home/user/thesis-new-files/frontend/api/
```

**Option 2: Adjust Import Path**
```python
# In api/price.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
```

**Option 3: Install as Package**
```bash
cd /home/user/thesis-new-files
pip install -e .
```

Then uncomment imports in `api/price.py`.

## Support

### Internal Documentation

- [QUICK_DEPLOY.md](./QUICK_DEPLOY.md) - Fast deployment
- [DEPLOYMENT.md](./DEPLOYMENT.md) - Comprehensive guide
- [VERCEL_CHECKLIST.md](./VERCEL_CHECKLIST.md) - Detailed checklist
- [DEPLOYMENT_REQUIREMENTS.md](./DEPLOYMENT_REQUIREMENTS.md) - Requirements

### External Resources

- [Vercel Docs](https://vercel.com/docs)
- [Next.js Docs](https://nextjs.org/docs)
- [Python Runtime](https://vercel.com/docs/functions/runtimes/python)
- [Serverless Limits](https://vercel.com/docs/functions/limitations)

### Community

- [Next.js Discussions](https://github.com/vercel/next.js/discussions)
- [Vercel Support](https://vercel.com/support)

## Status

- **Configuration**: Complete ✓
- **Documentation**: 50+ pages ✓
- **Integration**: Pending
- **Deployment**: Not started

## Next Steps

1. Integrate optimal_stopping module
2. Test locally
3. Deploy to Vercel
4. Build frontend UI

---

**Total Documentation**: 50+ pages across 6 guides
**Created**: 2025-11-19
**Ready for**: Integration & Deployment
