# Vercel Deployment Checklist

Use this checklist to ensure a smooth deployment to Vercel.

## Pre-Deployment Checklist

### 1. Code Quality

- [ ] All TypeScript files compile without errors
  ```bash
  npm run type-check
  ```

- [ ] All ESLint rules pass
  ```bash
  npm run lint
  ```

- [ ] Code is formatted consistently
  ```bash
  npm run format:check
  ```

- [ ] Build succeeds locally
  ```bash
  npm run build
  ```

### 2. Configuration Files

- [ ] `vercel.json` is present and configured
  - [ ] Python runtime version set to 3.9
  - [ ] Node version set to 20.x
  - [ ] Function timeout set to 60s (Pro) or 10s (Free)
  - [ ] Memory allocation set appropriately (3008 MB max)
  - [ ] CORS headers configured

- [ ] `next.config.mjs` is present and configured
  - [ ] Webpack config for Python integration
  - [ ] API route rewrites configured
  - [ ] CORS headers set
  - [ ] Production optimizations enabled

- [ ] `package.json` updated
  - [ ] Correct Node version in engines (>=20.0.0)
  - [ ] All scripts working (dev, build, start, lint)
  - [ ] Dependencies up to date

- [ ] Python dependencies configured
  - [ ] `api-requirements.txt` present (minimal deps for Vercel)
  - [ ] `requirements.txt` present (full deps for reference)
  - [ ] Total size under Vercel limits (<250 MB uncompressed)
  - [ ] No PyTorch (too large for Vercel - use alternative deployment)

### 3. Environment Variables

- [ ] `.env.example` file created with all required variables
- [ ] `.env.local.example` file created for local development
- [ ] Environment variables documented in README.md
- [ ] Client-side variables prefixed with `NEXT_PUBLIC_`
- [ ] Sensitive keys NOT committed to git

Required variables:
- [ ] `NEXT_PUBLIC_API_URL`
- [ ] `NODE_ENV`
- [ ] Algorithm flags (e.g., `NEXT_PUBLIC_ENABLE_TORCH_ALGOS=false`)

### 4. API Routes

- [ ] API routes are in `/api` directory
- [ ] Python handler functions follow Vercel format:
  ```python
  from http.server import BaseHTTPRequestHandler

  class handler(BaseHTTPRequestHandler):
      def do_POST(self):
          # Handle request
  ```

- [ ] API routes handle CORS correctly
- [ ] API routes return proper JSON responses
- [ ] Error handling implemented for all routes
- [ ] Timeout considerations (keep under 60s)

### 5. Testing

- [ ] Local development works
  ```bash
  npm run dev
  ```

- [ ] Production build works locally
  ```bash
  npm run build
  npm start
  ```

- [ ] Vercel CLI dev environment works
  ```bash
  vercel dev
  ```

- [ ] API endpoints respond correctly
  - [ ] Health check: `GET /api/health`
  - [ ] Pricing endpoint: `POST /api/price`

- [ ] Test with realistic payloads
  - [ ] Small request (nb_paths=1000)
  - [ ] Medium request (nb_paths=10000)
  - [ ] Large request (nb_paths=50000)

### 6. Performance

- [ ] Bundle size analyzed
  ```bash
  npm run analyze
  ```

- [ ] Large dependencies dynamically imported
- [ ] Images optimized (use Next.js Image component)
- [ ] API responses cached where appropriate
- [ ] Loading states implemented for slow operations

### 7. Security

- [ ] No secrets in code
- [ ] `.env` files in `.gitignore`
- [ ] API routes validate input
- [ ] CORS configured properly (not overly permissive)
- [ ] Dependencies audited
  ```bash
  npm audit
  ```

## Vercel Dashboard Configuration

### 1. Project Settings

- [ ] Repository connected to Vercel
- [ ] Auto-deploy enabled for main/master branch
- [ ] Preview deployments enabled for PRs

### 2. Build & Development Settings

- [ ] Framework Preset: **Next.js**
- [ ] Root Directory: **frontend**
- [ ] Build Command: `npm run build` (or leave default)
- [ ] Output Directory: `.next` (or leave default)
- [ ] Install Command: `npm install` (or leave default)
- [ ] Node.js Version: **20.x**

### 3. Environment Variables

Add these in Vercel Dashboard → Settings → Environment Variables:

**Production:**
- [ ] `NEXT_PUBLIC_API_URL` = `https://your-domain.vercel.app`
- [ ] `NODE_ENV` = `production`
- [ ] `NEXT_PUBLIC_ENABLE_TORCH_ALGOS` = `false`
- [ ] `NEXT_PUBLIC_MAX_PATHS` = `10000`
- [ ] `NEXT_PUBLIC_MAX_STOCKS` = `10`

**Preview:**
- [ ] Same as production, or different for testing

**Development:**
- [ ] `NEXT_PUBLIC_API_URL` = `http://localhost:3000`
- [ ] `NODE_ENV` = `development`

### 4. Function Configuration

- [ ] Timeout: **60 seconds** (Pro plan) or **10 seconds** (Hobby)
- [ ] Memory: **3008 MB** (or lower if sufficient)
- [ ] Region: Select closest to users (e.g., `iad1` for US East)

### 5. Domain Configuration

- [ ] Custom domain added (optional)
- [ ] SSL certificate auto-provisioned
- [ ] DNS records configured
- [ ] Domain verified

## First Deployment

### 1. Deploy via Dashboard

- [ ] Click "Deploy" button
- [ ] Monitor build logs for errors
- [ ] Wait for deployment to complete
- [ ] Note the deployment URL

### 2. Verify Deployment

- [ ] Visit deployment URL
- [ ] Check homepage loads
- [ ] Test API health endpoint
  ```bash
  curl https://your-domain.vercel.app/api/health
  ```

- [ ] Test pricing endpoint
  ```bash
  curl -X POST https://your-domain.vercel.app/api/price \
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

- [ ] Check response time (should be under 60s)
- [ ] Verify no errors in Vercel function logs

### 3. Check Logs

- [ ] Navigate to Vercel Dashboard → Deployments → Your Deployment → Functions
- [ ] Check for any errors or warnings
- [ ] Verify function execution times
- [ ] Check memory usage

## Post-Deployment

### 1. Monitoring

- [ ] Enable Vercel Analytics
- [ ] Enable Speed Insights
- [ ] Set up error tracking (Sentry, etc.)
- [ ] Configure log drains (optional)

### 2. Performance Optimization

- [ ] Review function execution times
- [ ] Identify slow API routes
- [ ] Optimize cold start times
- [ ] Implement caching where beneficial

### 3. Documentation

- [ ] Update README.md with deployment URL
- [ ] Document any deployment-specific quirks
- [ ] Create runbook for common issues
- [ ] Share deployment info with team

### 4. Continuous Deployment

- [ ] Test auto-deployment on push
- [ ] Verify preview deployments for PRs
- [ ] Configure deployment notifications (Slack, email)
- [ ] Set up status checks for PRs

## Known Limitations & Workarounds

### Size Limits

**Issue**: Deployment size exceeds 250 MB

**Solutions**:
- [ ] Use `api-requirements.txt` instead of `requirements.txt`
- [ ] Remove unused dependencies
- [ ] Exclude dev dependencies from production
- [ ] Deploy Python backend separately

### Timeout Limits

**Issue**: Function execution exceeds 60 seconds

**Solutions**:
- [ ] Reduce `nb_paths` in pricing requests
- [ ] Implement request queuing for long computations
- [ ] Use async processing with status polling
- [ ] Deploy compute-intensive tasks elsewhere

### Memory Limits

**Issue**: Function runs out of memory

**Solutions**:
- [ ] Increase memory allocation (up to 3008 MB)
- [ ] Optimize algorithm memory usage
- [ ] Process data in chunks
- [ ] Use external storage for large datasets

### PyTorch Not Available

**Issue**: NLSM/DOS algorithms require PyTorch (too large)

**Solutions**:
- [ ] Disable PyTorch algorithms in Vercel deployment
- [ ] Deploy separate Python backend for PyTorch models
- [ ] Use alternative algorithms (RLSM, RFQI, LSM, FQI)
- [ ] Consider AWS Lambda with container images (up to 10 GB)

## Troubleshooting

### Build Fails

1. Check build logs in Vercel dashboard
2. Verify all dependencies are in `package.json`
3. Run `npm run build` locally
4. Check for TypeScript errors
5. Verify Node version compatibility

### Function Errors

1. Check function logs in Vercel dashboard
2. Test locally with `vercel dev`
3. Verify environment variables are set
4. Check Python dependencies are installed
5. Verify function timeout and memory settings

### CORS Errors

1. Check `next.config.mjs` headers configuration
2. Verify `vercel.json` headers settings
3. Test with curl to isolate issue
4. Check browser console for specific error

### Environment Variables Not Loading

1. Verify variables are set in Vercel dashboard
2. Ensure client variables are prefixed with `NEXT_PUBLIC_`
3. Redeploy after adding new variables
4. Check variable scope (Production/Preview/Development)

## Rollback Plan

If deployment fails:

1. **Instant Rollback**:
   - [ ] Go to Vercel Dashboard → Deployments
   - [ ] Find previous working deployment
   - [ ] Click "..." → "Promote to Production"

2. **Fix and Redeploy**:
   - [ ] Identify issue from logs
   - [ ] Fix locally
   - [ ] Test with `vercel dev`
   - [ ] Commit and push to trigger new deployment

3. **Emergency**:
   - [ ] Disable auto-deployments
   - [ ] Revert git commit
   - [ ] Manually trigger deployment from working commit

## Success Criteria

Deployment is successful when:

- [ ] Build completes without errors
- [ ] Homepage loads correctly
- [ ] API health endpoint returns 200
- [ ] Pricing endpoint returns correct results
- [ ] Response times are acceptable (<10s typical, <60s max)
- [ ] No errors in function logs
- [ ] Environment variables load correctly
- [ ] Custom domain works (if configured)
- [ ] Analytics tracking works
- [ ] Auto-deployment works for new commits

## Additional Resources

- [Vercel Documentation](https://vercel.com/docs)
- [Next.js Deployment Guide](https://nextjs.org/docs/deployment)
- [Vercel Python Runtime](https://vercel.com/docs/functions/runtimes/python)
- [Serverless Functions Limits](https://vercel.com/docs/functions/limitations)

---

**Last Updated**: 2025-11-19

**Deployment Status**: ⬜ Not Started | ⬜ In Progress | ⬜ Completed
