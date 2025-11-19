# Backend Deployment Guide

This guide walks you through deploying the Option Pricing Calculator backend to Railway (free tier).

## Option 1: Deploy to Railway (Recommended - Easiest)

### Step 1: Prepare Repository

1. Make sure all files in `/backend` are committed to git
2. The backend needs the `optimal_stopping` package, so your git repo should include both `/backend` and `/optimal_stopping`

### Step 2: Deploy to Railway

1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub (free)
3. Click "New Project"
4. Select "Deploy from GitHub repo"
5. Choose your repository
6. Railway will auto-detect the Procfile and deploy

### Step 3: Configure Root Directory

Since the backend code is in a subdirectory:
1. In Railway dashboard, go to your service settings
2. Under "Root Directory", enter: `backend`
3. Click "Save"
4. Railway will redeploy

### Step 4: Get Your Backend URL

1. In Railway dashboard, click on your service
2. Go to "Settings" tab
3. Under "Domains", click "Generate Domain"
4. Copy the URL (e.g., `https://your-app-name.up.railway.app`)

### Step 5: Test Your Backend

Visit `https://your-backend-url.up.railway.app/` - you should see:
```json
{
  "status": "healthy",
  "service": "Option Pricing API",
  "version": "1.0.0"
}
```

## Option 2: Deploy to Render

1. Go to [render.com](https://render.com)
2. Sign up with GitHub
3. Click "New +" → "Web Service"
4. Connect your repository
5. Configure:
   - **Name**: option-pricing-backend
   - **Root Directory**: backend
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Click "Create Web Service"
7. Copy your backend URL

## Option 3: Deploy to PythonAnywhere

1. Go to [pythonanywhere.com](https://www.pythonanywhere.com)
2. Sign up for free account
3. Upload your code via Files tab
4. Create a new web app
5. Configure WSGI file to point to `main:app`
6. Install dependencies in bash console: `pip install -r requirements.txt`

## Next Steps

Once deployed, you need to:

1. Copy your backend URL (e.g., `https://your-app.up.railway.app`)
2. Update your frontend to use this URL
3. See `/frontend/DEPLOYMENT.md` for frontend deployment

## Free Tier Limits

- **Railway**: $5 credit per month, ~500 hours runtime
- **Render**: 750 hours per month, apps sleep after 15 min inactivity
- **PythonAnywhere**: Always-on apps not available on free tier

Railway is recommended for best performance and ease of use.

## Troubleshooting

### Build Fails

- Check that `optimal_stopping` directory is in your repo
- Ensure Python version compatibility (3.11+ recommended)

### App Crashes

- Check Railway logs: Dashboard → Service → Deployments → Click on deployment → Logs
- Common issue: Missing dependencies - make sure `requirements.txt` is complete

### CORS Errors

- Backend has CORS enabled for all origins (`*`)
- In production, you may want to restrict to your Vercel domain only
- Edit `main.py` line 38: `allow_origins=["https://your-app.vercel.app"]`
