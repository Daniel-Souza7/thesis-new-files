# Quick Deploy Guide

5-minute deployment guide for Vercel.

## Prerequisites

- Vercel account (https://vercel.com)
- Git repository with this code
- 15 minutes

## Step 1: Install Vercel CLI (Optional)

```bash
npm i -g vercel
```

## Step 2: Configure Environment

Create `.env.local` from template:

```bash
cp .env.example .env.local
```

Edit `.env.local` with your settings (or use defaults).

## Step 3: Test Locally

```bash
# Install dependencies
npm install

# Run dev server
npm run dev
```

Visit http://localhost:3000 to verify it works.

## Step 4: Deploy to Vercel

### Option A: Via Dashboard (Easiest)

1. Go to https://vercel.com/new
2. Click "Import Project"
3. Select your Git repository
4. Configure:
   - **Root Directory**: `frontend`
   - **Framework**: Next.js (auto-detected)
5. Add environment variables:
   ```
   NEXT_PUBLIC_API_URL = https://your-domain.vercel.app
   NODE_ENV = production
   NEXT_PUBLIC_ENABLE_TORCH_ALGOS = false
   ```
6. Click "Deploy"
7. Wait 2-5 minutes

### Option B: Via CLI

```bash
cd frontend
vercel

# Follow prompts:
# - Link to existing project? No
# - Project name? (your choice)
# - Directory? ./
# - Override settings? No
```

## Step 5: Verify Deployment

Test health endpoint:
```bash
curl https://your-domain.vercel.app/api/health
```

Expected response:
```json
{
  "status": "ok",
  "timestamp": "2025-11-19T12:00:00Z",
  "version": "0.1.0"
}
```

## Step 6: Configure Production

In Vercel dashboard:

1. Go to **Settings → Environment Variables**
2. Add:
   - `NEXT_PUBLIC_API_URL` = `https://your-domain.vercel.app`
   - `NODE_ENV` = `production`
   - `NEXT_PUBLIC_ENABLE_TORCH_ALGOS` = `false`
3. **Save**

Redeploy to apply changes:
```bash
vercel --prod
```

## Common Commands

```bash
# Local development
npm run dev

# Production build (local)
npm run build
npm start

# Deploy preview
vercel

# Deploy to production
vercel --prod

# View logs
vercel logs

# List deployments
vercel ls

# Remove deployment
vercel rm <deployment-id>
```

## Troubleshooting

### Build Failed

```bash
# Check locally first
npm run type-check
npm run lint
npm run build
```

### Function Timeout

- Reduce `nb_paths` in requests
- Check Vercel plan (Free = 10s, Pro = 60s)

### Deployment Too Large

- Verify using `api-requirements.txt` (not `requirements.txt`)
- Check total size < 250 MB

### Environment Variables Not Working

- Prefix client variables with `NEXT_PUBLIC_`
- Redeploy after adding variables
- Check variable scope (Production/Preview/Development)

## Next Steps

1. **Integrate optimal_stopping**: Copy module to `/api` directory
2. **Update pricing endpoint**: Uncomment code in `api/price.py`
3. **Build frontend UI**: Create React components
4. **Test thoroughly**: Use [VERCEL_CHECKLIST.md](./VERCEL_CHECKLIST.md)

## Full Documentation

- [DEPLOYMENT.md](./DEPLOYMENT.md) - Complete deployment guide
- [VERCEL_CHECKLIST.md](./VERCEL_CHECKLIST.md) - Detailed checklist
- [VERCEL_SETUP_SUMMARY.md](./VERCEL_SETUP_SUMMARY.md) - Configuration overview
- [README.md](./README.md) - Project documentation

## Support

- **Vercel Status**: https://vercel-status.com
- **Documentation**: https://vercel.com/docs
- **Community**: https://github.com/vercel/next.js/discussions
- **Function Logs**: Vercel Dashboard → Deployments → Functions

---

**Total Time**: ~5-15 minutes
**Success Criteria**: Health endpoint returns 200 OK
**Next**: Integrate pricing engine and build UI
