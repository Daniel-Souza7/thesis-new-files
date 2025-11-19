# American Options Pricing - Frontend

Web interface for pricing American-style options using reinforcement learning and Monte Carlo methods.

## Overview

This Next.js application provides an interactive frontend for the optimal stopping pricing engine. It supports:

- **408 option payoff types** (34 base + 374 barrier variants)
- **6 pricing algorithms** (RLSM, RFQI, LSM, FQI, DOS, NLSM)
- **5 stochastic models** (Black-Scholes, Heston, Fractional BS, Rough Heston, Real Data)
- Real-time pricing with configurable parameters
- Interactive visualization of results
- Export functionality for analysis

## Quick Start

### Local Development

```bash
# Install dependencies
npm install

# Copy environment template
cp .env.example .env.local

# Run development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Production Build

```bash
npm run build
npm start
```

## Project Structure

```
frontend/
├── app/                    # Next.js App Router
│   ├── page.tsx           # Home page
│   ├── layout.tsx         # Root layout
│   └── globals.css        # Global styles
├── api/                    # API routes (Python/TypeScript)
│   ├── price.py           # Pricing endpoint
│   └── health.py          # Health check
├── components/             # React components (to be created)
├── lib/                    # Utility functions (to be created)
├── public/                 # Static assets
├── vercel.json            # Vercel deployment config
├── next.config.mjs        # Next.js configuration
├── package.json           # Node dependencies
├── requirements.txt       # Python dependencies (full)
├── api-requirements.txt   # Python dependencies (minimal for Vercel)
└── DEPLOYMENT.md          # Deployment guide
```

## Available Scripts

### Development

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm start` - Start production server
- `npm run type-check` - Run TypeScript type checking
- `npm run lint` - Run ESLint
- `npm run lint:fix` - Fix ESLint issues
- `npm run format` - Format code with Prettier
- `npm run format:check` - Check code formatting

### Maintenance

- `npm run clean` - Remove build artifacts and node_modules
- `npm run analyze` - Analyze bundle size

## Environment Variables

Create a `.env.local` file for local development:

```bash
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:3000

# Feature Flags
NEXT_PUBLIC_ENABLE_REALTIME_PRICING=true
NEXT_PUBLIC_ENABLE_PATH_STORAGE=true
NEXT_PUBLIC_MAX_PATHS=100000
NEXT_PUBLIC_MAX_STOCKS=20

# Algorithm Configuration
NEXT_PUBLIC_DEFAULT_ALGO=RLSM
NEXT_PUBLIC_ENABLE_TORCH_ALGOS=true

# Model Configuration
NEXT_PUBLIC_DEFAULT_MODEL=BlackScholes
NEXT_PUBLIC_DEFAULT_STRIKE=100
NEXT_PUBLIC_DEFAULT_SPOT=100
NEXT_PUBLIC_DEFAULT_RATE=0.05
NEXT_PUBLIC_DEFAULT_VOLATILITY=0.2
NEXT_PUBLIC_DEFAULT_MATURITY=1.0
```

See `.env.example` for all available options.

## Technology Stack

### Frontend
- **Next.js 16** - React framework with App Router
- **React 19** - UI library
- **TypeScript** - Type safety
- **Tailwind CSS 4** - Utility-first CSS

### Backend (API Routes)
- **Python 3.9** - Serverless functions
- **NumPy** - Numerical computing
- **SciPy** - Scientific computing
- **scikit-learn** - Machine learning
- **pandas** - Data structures

## Deployment

### Vercel (Recommended)

See [DEPLOYMENT.md](./DEPLOYMENT.md) for comprehensive deployment guide.

Quick deploy:

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/your-repo/your-project)

### Docker

```bash
# Build image
docker build -t options-frontend .

# Run container
docker run -p 3000:3000 options-frontend
```

### Manual

Requirements:
- Node.js 20+
- Python 3.9+ (for API routes)

```bash
npm install
npm run build
npm start
```

## API Routes

### POST /api/price

Price an American option.

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
  "confidence_interval": [12.20, 12.48]
}
```

### GET /api/health

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2025-11-19T12:00:00Z",
  "version": "0.1.0"
}
```

## Features

### Current

- Basic Next.js setup
- Tailwind CSS styling
- TypeScript support
- Vercel deployment configuration

### Planned

- [ ] Interactive pricing form
- [ ] Real-time price calculations
- [ ] Parameter visualization
- [ ] Results comparison table
- [ ] Convergence plots
- [ ] Greek calculations display
- [ ] Export to CSV/Excel
- [ ] Preset configurations
- [ ] Algorithm comparison
- [ ] Historical results storage

## Performance

### Optimization Strategies

1. **Bundle Size**: Use dynamic imports for heavy components
2. **API Routes**: Minimal Python dependencies for fast cold starts
3. **Caching**: SWR for data fetching with automatic revalidation
4. **Images**: Next.js Image component for optimization
5. **Code Splitting**: Automatic route-based splitting

### Vercel Limits

- **Function Timeout**: 60 seconds (Pro), 10 seconds (Free)
- **Function Size**: 50 MB compressed
- **Deployment Size**: 250 MB uncompressed
- **Memory**: 3008 MB max

See [DEPLOYMENT.md](./DEPLOYMENT.md) for details on working within these limits.

## Development Guide

### Adding a New Page

```bash
# Create new route
mkdir -p app/pricing
touch app/pricing/page.tsx
```

```typescript
// app/pricing/page.tsx
export default function PricingPage() {
  return <div>Pricing Page</div>
}
```

### Creating Components

```bash
mkdir -p components
touch components/PricingForm.tsx
```

```typescript
// components/PricingForm.tsx
export function PricingForm() {
  return <form>...</form>
}
```

### Adding API Routes

Python API route:
```python
# api/new-endpoint.py
from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'status': 'ok'}).encode())
```

TypeScript API route:
```typescript
// app/api/new-endpoint/route.ts
export async function GET() {
  return Response.json({ status: 'ok' })
}
```

## Testing

### Unit Tests (To be added)

```bash
npm run test
```

### Integration Tests (To be added)

```bash
npm run test:integration
```

### E2E Tests (To be added)

```bash
npm run test:e2e
```

## Troubleshooting

### Build Errors

**Issue**: TypeScript errors during build

**Solution**: Run `npm run type-check` to identify issues

**Issue**: Module not found

**Solution**: Clear `.next` folder: `npm run clean && npm install`

### Runtime Errors

**Issue**: API route not responding

**Solution**: Check Vercel function logs in dashboard

**Issue**: Environment variables not loading

**Solution**: Ensure variables are prefixed with `NEXT_PUBLIC_` for client-side access

### Performance Issues

**Issue**: Slow page load

**Solution**: Enable production mode: `npm run build && npm start`

**Issue**: Large bundle size

**Solution**: Run `npm run analyze` to identify large dependencies

## Contributing

1. Create a feature branch
2. Make changes
3. Run `npm run type-check` and `npm run lint`
4. Test locally with `npm run build`
5. Submit pull request

## Resources

### Documentation

- [Next.js Documentation](https://nextjs.org/docs)
- [React Documentation](https://react.dev)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
- [TypeScript Documentation](https://www.typescriptlang.org/docs)
- [Vercel Documentation](https://vercel.com/docs)

### Related Projects

- [Main Project README](../README.md)
- [Claude.md - Development Guide](../CLAUDE.md)
- [Implementation Summary](../IMPLEMENTATION_SUMMARY.md)
- [Validation Test Plan](../VALIDATION_TEST_PLAN.md)

### API Documentation

See the main project's documentation in `../optimal_stopping/` for:
- Algorithm details
- Payoff implementations
- Stock model specifications
- Configuration options

## License

See main project LICENSE file.

## Support

For issues:
1. Check [DEPLOYMENT.md](./DEPLOYMENT.md) troubleshooting section
2. Review Vercel function logs
3. Test API endpoints independently
4. Check browser console for client errors

For questions about the pricing algorithms, see the main project documentation.
