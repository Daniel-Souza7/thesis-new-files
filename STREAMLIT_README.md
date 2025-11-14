# 📈 Interactive American Option Pricing Demo

Beautiful Streamlit application for demonstrating optimal stopping algorithms for American option pricing.

![Demo Screenshot](https://via.placeholder.com/800x400/667eea/ffffff?text=American+Option+Pricing+Demo)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the App

```bash
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

## ✨ Features

### 📊 Overview Tab
- Introduction to the optimal stopping problem
- Bellman equation and mathematical formulation
- Algorithm explanations with LaTeX formulas

### 🎯 Simulation Tab
- **Interactive parameter controls** - Adjust all parameters via sliders
- **Live path visualization** - Watch stock paths being simulated
- **Progress tracking** - Real-time updates during computation
- **Beautiful visualizations** - Sample paths with Plotly charts

### 📈 Results Tab
- **Option price** with detailed metrics
- **Time breakdown** - Path generation vs pricing computation
- **Payoff distribution** - Histogram of payoffs at maturity
- **Statistics** - Mean, std dev, ITM percentage

### 🔬 Algorithm Comparison
- **Side-by-side comparison** - Run multiple algorithms on same problem
- **Price comparison** - See which algorithm gives highest prices
- **Speed comparison** - Compare computation times
- **Detailed table** - All metrics in one view

## 🎨 Customization

### Adjustable Parameters

**Market Parameters:**
- Number of stocks (2-10)
- Initial stock price ($50-$150)
- Strike price ($50-$150)
- Volatility (0.1-0.5)
- Drift (-0.1 to 0.2)
- Risk-free rate (0-0.1)
- Maturity (0.25-2 years)

**Simulation:**
- Number of paths (1K-50K)
- Number of exercise dates (10-100)

**Algorithms:**
- RLSM (Randomized LSM)
- RFQI (Randomized FQI)
- LSM (Least Squares MC)
- NLSM (Neural LSM)
- DOS (Deep Optimal Stopping)
- FQI (Fitted Q-Iteration)

**Models:**
- Black-Scholes
- Real Data (S&P 500)

**Payoffs:**
- Max Call
- Basket Call
- Min Call
- Geometric Basket Call

## 🎓 For Thesis Presentation

### Recommended Flow:

1. **Start with Overview** - Explain the problem
2. **Show simple example** - 3 stocks, Black-Scholes, Max Call with RLSM
3. **Visualize paths** - Show how Monte Carlo generates scenarios
4. **Display results** - Walk through metrics and payoff distribution
5. **Compare algorithms** - Show RLSM vs benchmarks (LSM, NLSM)
6. **Increase complexity** - Try 10 stocks, Real Data model
7. **Interactive Q&A** - Let audience adjust parameters

### Tips for Live Demo:

- Keep `nb_paths` at 10,000 for fast demos
- Use 3-5 stocks initially
- Start with Black-Scholes (faster than Real Data)
- Pre-run one simulation before presenting
- Have backup screenshots if network fails

## 🌐 Deploying Online

### Option 1: Streamlit Cloud (Free, Easiest)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Deploy with one click!
5. Share URL with thesis committee

### Option 2: Local Network (No Internet Needed)

```bash
# Find your local IP
# Windows: ipconfig
# Mac/Linux: ifconfig

# Run with network access
streamlit run app.py --server.address 0.0.0.0

# Others can access at: http://YOUR_IP:8501
```

## 🐛 Troubleshooting

### Port Already in Use
```bash
streamlit run app.py --server.port 8502
```

### Cache Issues
```bash
streamlit cache clear
streamlit run app.py
```

### Slow Performance
- Reduce `nb_paths` to 5,000-10,000
- Reduce `nb_dates` to 20-50
- Use Black-Scholes instead of Real Data
- Close other browser tabs

## 📝 Customization Ideas

### Add More Visualizations:
- Exercise boundary heatmap
- Continuation value surface plots
- Training convergence curves
- Monte Carlo error bars

### More Features:
- Greeks calculation (delta, gamma)
- Sensitivity analysis
- Download results as CSV/Excel
- Upload custom parameters from file
- Real-time algorithm animation

### Styling:
- Change color scheme in the CSS section
- Add your university logo
- Custom fonts and themes

## 🔥 Pro Tips

1. **Use st.cache** for expensive computations
2. **Session state** preserves results between reruns
3. **Tabs** organize content cleanly
4. **Columns** for side-by-side layouts
5. **Plotly** for interactive charts

## 📚 Documentation

- [Streamlit Docs](https://docs.streamlit.io)
- [Plotly Docs](https://plotly.com/python/)
- [Deployment Guide](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)

## 🎉 Enjoy Your Presentation!

Questions? Feedback? Happy to help improve the demo!
