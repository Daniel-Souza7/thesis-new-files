"""
Interactive American Option Pricing Demo
For Thesis Presentation

This Streamlit app demonstrates various algorithms for pricing American options
using Monte Carlo simulation and backward induction.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
from datetime import datetime

# Import algorithms
from optimal_stopping.algorithms.standard.rlsm import RLSM
from optimal_stopping.algorithms.standard.rfqi import RFQI
from optimal_stopping.algorithms.standard.lsm import LSM
from optimal_stopping.algorithms.standard.nlsm import NLSM
from optimal_stopping.algorithms.standard.dos import DOS
from optimal_stopping.algorithms.standard.fqi import FQI

# Import models and payoffs
from optimal_stopping.data.stock_model import BlackScholes
from optimal_stopping.data.real_data import RealDataModel
from optimal_stopping.payoffs import MaxCall, BasketCall, MinCall, GeometricBasketCall

# Page config
st.set_page_config(
    page_title="American Option Pricing",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    .stApp {
        background: transparent;
    }
    .block-container {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    h1, h2, h3 {
        color: #667eea;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    .formula-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


# Title
st.title("📈 American Option Pricing: Interactive Demo")
st.markdown("### Optimal Stopping with Randomized Neural Networks")
st.markdown("---")

# Sidebar - Parameters
st.sidebar.header("⚙️ Configuration")

# Algorithm selection
st.sidebar.subheader("Algorithm")
algo_name = st.sidebar.selectbox(
    "Choose Algorithm",
    ["RLSM", "RFQI", "LSM", "NLSM", "DOS", "FQI"],
    help="RLSM/RFQI: Randomized neural networks (best performance)\nLSM/NLSM/DOS/FQI: Benchmark algorithms"
)

# Stock model
st.sidebar.subheader("Stock Model")
model_type = st.sidebar.selectbox(
    "Model Type",
    ["Black-Scholes", "Real Data (S&P 500)"]
)

# Payoff type
st.sidebar.subheader("Option Payoff")
payoff_type = st.sidebar.selectbox(
    "Payoff Type",
    ["Max Call", "Basket Call", "Min Call", "Geometric Basket Call"]
)

# Market parameters
st.sidebar.subheader("Market Parameters")
nb_stocks = st.sidebar.slider("Number of Stocks", 2, 10, 5)
spot = st.sidebar.slider("Initial Stock Price ($)", 50, 150, 100)
strike = st.sidebar.slider("Strike Price ($)", 50, 150, 100)
volatility = st.sidebar.slider("Volatility", 0.1, 0.5, 0.2, 0.05)
drift = st.sidebar.slider("Drift (μ)", -0.1, 0.2, 0.05, 0.01)
rate = st.sidebar.slider("Risk-free Rate", 0.0, 0.1, 0.05, 0.01)
maturity = st.sidebar.slider("Maturity (years)", 0.25, 2.0, 1.0, 0.25)

# Simulation parameters
st.sidebar.subheader("Simulation Parameters")
nb_paths = st.sidebar.select_slider(
    "Number of Paths",
    options=[1000, 5000, 10000, 20000, 50000],
    value=10000
)
nb_dates = st.sidebar.slider("Number of Exercise Dates", 10, 100, 52)

# Algorithm hyperparameters
st.sidebar.subheader("Algorithm Parameters")
if algo_name in ["RLSM", "RFQI"]:
    hidden_size = st.sidebar.slider("Hidden Layer Size", 50, 200, 100, 10)
elif algo_name in ["NLSM", "DOS"]:
    hidden_size = st.sidebar.slider("Hidden Layer Size", 10, 100, 50, 10)
    nb_epochs = st.sidebar.slider("Training Epochs", 5, 50, 20, 5)
else:  # LSM, FQI
    hidden_size = None
    if algo_name in ["FQI", "NLSM", "DOS"]:
        nb_epochs = st.sidebar.slider("Training Epochs", 5, 50, 20, 5)

train_ITM_only = st.sidebar.checkbox("Train on ITM paths only", value=True)


# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Simulation", "📈 Results", "🔬 Algorithm Comparison"])


with tab1:
    st.header("American Option Pricing Problem")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### The Optimal Stopping Problem

        An American option gives the holder the right to exercise at any time before maturity.
        The pricing problem is to find the optimal exercise strategy that maximizes the option value.
        """)

        st.markdown("""
        <div class="formula-box">
        <b>Bellman Equation:</b><br>
        """, unsafe_allow_html=True)

        st.latex(r"V_t(S_t) = \max\left\{g(S_t), \, \mathbb{E}\left[e^{-r\Delta t}V_{t+1}(S_{t+1}) \mid S_t\right]\right\}")

        st.markdown(r"""
        </div>

        Where:
        - $V_t(S_t)$ = Option value at time $t$
        - $g(S_t)$ = Immediate exercise payoff
        - $\mathbb{E}[\cdot]$ = Expected continuation value
        """)

    with col2:
        st.markdown("### Your Configuration")
        st.info(f"""
        **Algorithm:** {algo_name}
        **Model:** {model_type}
        **Payoff:** {payoff_type}
        **Stocks:** {nb_stocks}
        **Paths:** {nb_paths:,}
        **Dates:** {nb_dates}
        """)

    st.markdown("---")

    # Algorithm explanation
    st.subheader(f"📚 {algo_name} Algorithm")

    if algo_name == "RLSM":
        st.markdown("""
        **Randomized Least Squares Monte Carlo (RLSM)**

        Uses randomized neural networks with frozen weights as basis functions:
        """)
        st.latex(r"C_t(S_t) \approx \sum_{i=1}^{H} \beta_i \cdot \phi(\mathbf{W}_i^T S_t)")
        st.markdown(r"""
        - Random weights $\mathbf{W}_i$ are **frozen** (not trained)
        - Only output coefficients $\beta_i$ are learned via least squares
        - Fast and stable - no gradient descent needed
        """)

    elif algo_name == "RFQI":
        st.markdown("""
        **Randomized Fitted Q-Iteration (RFQI)**

        Similar to RLSM but uses Q-learning approach with randomized networks.
        Iteratively refines Q-values for optimal stopping decisions.
        """)

    elif algo_name == "LSM":
        st.markdown("""
        **Least Squares Monte Carlo (LSM)**

        Classic Longstaff-Schwartz algorithm using polynomial basis functions:
        """)
        st.latex(r"C_t(S_t) \approx \sum_{i=0}^{d} \beta_i \cdot p_i(S_t)")
        st.markdown("""
        - Uses degree-2 polynomials as basis functions
        - Benchmark algorithm from 2001 paper
        """)

    elif algo_name == "NLSM":
        st.markdown("""
        **Neural Least Squares Monte Carlo (NLSM)**

        Uses trained neural networks to approximate continuation values.
        All weights are learned via gradient descent (slower than RLSM).
        """)

    elif algo_name == "DOS":
        st.markdown("""
        **Deep Optimal Stopping (DOS)**

        Neural network directly learns the stopping decision (not continuation value):
        - Outputs probability of stopping at each time step
        - Trained to maximize expected payoff
        """)

    elif algo_name == "FQI":
        st.markdown("""
        **Fitted Q-Iteration (FQI)**

        Q-learning approach with polynomial basis functions.
        Iteratively learns Q-values for the stopping problem.
        """)


with tab2:
    st.header("🎯 Run Simulation")

    if st.button("▶️ Run Pricing Algorithm", type="primary", use_container_width=True):

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Step 1: Generate model
        status_text.text("⚙️ Setting up model...")
        progress_bar.progress(10)
        time.sleep(0.3)

        if model_type == "Black-Scholes":
            model = BlackScholes(
                nb_stocks=nb_stocks,
                nb_paths=nb_paths,
                nb_dates=nb_dates,
                spot=spot,
                strike=strike,
                maturity=maturity,
                volatility=volatility,
                rate=rate,
                drift=drift
            )
        else:  # Real Data
            model = RealDataModel(
                nb_stocks=nb_stocks,
                nb_paths=nb_paths,
                nb_dates=nb_dates,
                spot=spot,
                strike=strike,
                maturity=maturity,
                rate=rate,
                drift=drift  # Override with config drift
            )

        # Step 2: Generate payoff
        status_text.text("🎲 Setting up payoff...")
        progress_bar.progress(20)
        time.sleep(0.3)

        if payoff_type == "Max Call":
            payoff = MaxCall(strike)
        elif payoff_type == "Basket Call":
            payoff = BasketCall(strike)
        elif payoff_type == "Min Call":
            payoff = MinCall(strike)
        else:  # Geometric Basket Call
            payoff = GeometricBasketCall(strike)

        # Step 3: Generate paths
        status_text.text("🌊 Generating stock paths...")
        progress_bar.progress(30)

        t_start = time.time()
        stock_paths, _ = model.generate_paths()
        time_path_gen = time.time() - t_start

        # Visualize paths (sample)
        st.subheader("📉 Sample Stock Paths")

        fig_paths = go.Figure()
        n_samples = min(50, nb_paths)
        time_grid = np.linspace(0, maturity, nb_dates + 1)

        for i in range(n_samples):
            for j in range(min(3, nb_stocks)):  # Show first 3 stocks
                fig_paths.add_trace(go.Scatter(
                    x=time_grid,
                    y=stock_paths[i, j, :],
                    mode='lines',
                    line=dict(width=0.5),
                    opacity=0.3,
                    showlegend=(i == 0),
                    name=f'Stock {j+1}' if i == 0 else None,
                    legendgroup=f'stock{j}'
                ))

        fig_paths.update_layout(
            title=f"Sample Paths ({n_samples} paths × {min(3, nb_stocks)} stocks shown)",
            xaxis_title="Time (years)",
            yaxis_title="Stock Price ($)",
            height=400,
            hovermode='x unified'
        )

        st.plotly_chart(fig_paths, use_container_width=True)

        # Step 4: Initialize algorithm
        status_text.text(f"🧠 Initializing {algo_name} algorithm...")
        progress_bar.progress(50)
        time.sleep(0.3)

        algo_map = {
            "RLSM": RLSM,
            "RFQI": RFQI,
            "LSM": LSM,
            "NLSM": NLSM,
            "DOS": DOS,
            "FQI": FQI
        }

        algo_class = algo_map[algo_name]

        # Initialize with appropriate parameters
        if algo_name in ["RLSM", "RFQI"]:
            pricer = algo_class(
                model, payoff,
                hidden_size=hidden_size,
                train_ITM_only=train_ITM_only
            )
        elif algo_name in ["NLSM", "DOS"]:
            pricer = algo_class(
                model, payoff,
                hidden_size=hidden_size,
                nb_epochs=nb_epochs,
                train_ITM_only=train_ITM_only
            )
        elif algo_name == "LSM":
            pricer = algo_class(
                model, payoff,
                train_ITM_only=train_ITM_only
            )
        else:  # FQI
            pricer = algo_class(
                model, payoff,
                nb_epochs=nb_epochs,
                train_ITM_only=train_ITM_only
            )

        # Step 5: Run pricing
        status_text.text(f"⚡ Running {algo_name} backward induction...")
        progress_bar.progress(70)

        t_pricing_start = time.time()
        option_price, _ = pricer.price(train_eval_split=2)
        time_pricing = time.time() - t_pricing_start

        progress_bar.progress(100)
        status_text.text("✅ Pricing complete!")
        time.sleep(0.5)

        status_text.empty()
        progress_bar.empty()

        # Store results in session state
        st.session_state['results'] = {
            'algo_name': algo_name,
            'option_price': option_price,
            'time_path_gen': time_path_gen,
            'time_pricing': time_pricing,
            'total_time': time_path_gen + time_pricing,
            'stock_paths': stock_paths,
            'payoff': payoff,
            'model': model,
            'nb_paths': nb_paths,
            'nb_dates': nb_dates,
            'timestamp': datetime.now()
        }

        st.success("✨ Simulation completed successfully!")
        st.balloons()

    else:
        st.info("👆 Click the button above to run the simulation")


with tab3:
    st.header("📈 Pricing Results")

    if 'results' in st.session_state:
        results = st.session_state['results']

        # Display metrics in cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Option Price</h3>
                <h1>${results['option_price']:.4f}</h1>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Time</h3>
                <h1>{results['total_time']:.2f}s</h1>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Paths</h3>
                <h1>{results['nb_paths']:,}</h1>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Algorithm</h3>
                <h1>{results['algo_name']}</h1>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Detailed breakdown
        st.subheader("⏱️ Time Breakdown")

        time_data = pd.DataFrame({
            'Phase': ['Path Generation', 'Pricing Algorithm'],
            'Time (s)': [results['time_path_gen'], results['time_pricing']],
            'Percentage': [
                results['time_path_gen'] / results['total_time'] * 100,
                results['time_pricing'] / results['total_time'] * 100
            ]
        })

        fig_time = px.bar(
            time_data,
            x='Phase',
            y='Time (s)',
            text='Time (s)',
            color='Percentage',
            color_continuous_scale='Viridis'
        )
        fig_time.update_traces(texttemplate='%{text:.2f}s', textposition='outside')
        fig_time.update_layout(height=400)

        st.plotly_chart(fig_time, use_container_width=True)

        # Payoff distribution
        st.subheader("💰 Payoff Distribution")

        stock_paths = results['stock_paths']
        payoffs = results['payoff'](stock_paths)

        # Show payoffs at maturity
        payoffs_maturity = payoffs[:, -1]

        fig_payoff = go.Figure()
        fig_payoff.add_trace(go.Histogram(
            x=payoffs_maturity,
            nbinsx=50,
            name='Payoff Distribution'
        ))
        fig_payoff.add_vline(
            x=results['option_price'],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Option Price: ${results['option_price']:.2f}"
        )
        fig_payoff.update_layout(
            title="Payoff Distribution at Maturity",
            xaxis_title="Payoff ($)",
            yaxis_title="Frequency",
            height=400
        )

        st.plotly_chart(fig_payoff, use_container_width=True)

        # Statistics
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Mean Payoff at Maturity", f"${np.mean(payoffs_maturity):.4f}")
            st.metric("Std Dev of Payoffs", f"${np.std(payoffs_maturity):.4f}")

        with col2:
            st.metric("% Paths ITM at Maturity", f"{np.mean(payoffs_maturity > 0) * 100:.1f}%")
            st.metric("Max Payoff", f"${np.max(payoffs_maturity):.4f}")

    else:
        st.info("📊 Run a simulation first to see results here!")


with tab4:
    st.header("🔬 Algorithm Comparison")

    st.markdown("""
    Compare multiple algorithms on the same problem to see performance differences.
    """)

    algos_to_compare = st.multiselect(
        "Select algorithms to compare",
        ["RLSM", "RFQI", "LSM", "NLSM", "DOS", "FQI"],
        default=["RLSM", "LSM"]
    )

    if st.button("▶️ Run Comparison", key="compare_btn", use_container_width=True):
        if len(algos_to_compare) < 2:
            st.warning("⚠️ Please select at least 2 algorithms to compare")
        else:
            comparison_results = []

            # Setup model and payoff once
            if model_type == "Black-Scholes":
                model = BlackScholes(
                    nb_stocks=nb_stocks,
                    nb_paths=nb_paths,
                    nb_dates=nb_dates,
                    spot=spot,
                    strike=strike,
                    maturity=maturity,
                    volatility=volatility,
                    rate=rate,
                    drift=drift
                )
            else:
                model = RealDataModel(
                    nb_stocks=nb_stocks,
                    nb_paths=nb_paths,
                    nb_dates=nb_dates,
                    spot=spot,
                    strike=strike,
                    maturity=maturity,
                    rate=rate,
                    drift=drift
                )

            if payoff_type == "Max Call":
                payoff = MaxCall(strike)
            elif payoff_type == "Basket Call":
                payoff = BasketCall(strike)
            elif payoff_type == "Min Call":
                payoff = MinCall(strike)
            else:
                payoff = GeometricBasketCall(strike)

            progress_bar = st.progress(0)
            status_text = st.empty()

            algo_map = {
                "RLSM": RLSM, "RFQI": RFQI, "LSM": LSM,
                "NLSM": NLSM, "DOS": DOS, "FQI": FQI
            }

            for idx, algo in enumerate(algos_to_compare):
                status_text.text(f"Running {algo}... ({idx+1}/{len(algos_to_compare)})")

                # Initialize algorithm
                algo_class = algo_map[algo]
                if algo in ["RLSM", "RFQI"]:
                    pricer = algo_class(model, payoff, hidden_size=100, train_ITM_only=True)
                elif algo in ["NLSM", "DOS"]:
                    pricer = algo_class(model, payoff, hidden_size=50, nb_epochs=20, train_ITM_only=True)
                elif algo == "LSM":
                    pricer = algo_class(model, payoff, train_ITM_only=True)
                else:  # FQI
                    pricer = algo_class(model, payoff, nb_epochs=20, train_ITM_only=True)

                # Run pricing
                t_start = time.time()
                price, _ = pricer.price(train_eval_split=2)
                duration = time.time() - t_start

                comparison_results.append({
                    'Algorithm': algo,
                    'Price': price,
                    'Time (s)': duration
                })

                progress_bar.progress((idx + 1) / len(algos_to_compare))

            status_text.empty()
            progress_bar.empty()

            # Display results
            df = pd.DataFrame(comparison_results)

            st.success("✅ Comparison complete!")

            # Price comparison
            col1, col2 = st.columns(2)

            with col1:
                fig_price = px.bar(
                    df,
                    x='Algorithm',
                    y='Price',
                    text='Price',
                    title="Option Prices by Algorithm",
                    color='Price',
                    color_continuous_scale='Viridis'
                )
                fig_price.update_traces(texttemplate='$%{text:.4f}', textposition='outside')
                st.plotly_chart(fig_price, use_container_width=True)

            with col2:
                fig_time = px.bar(
                    df,
                    x='Algorithm',
                    y='Time (s)',
                    text='Time (s)',
                    title="Computation Time by Algorithm",
                    color='Time (s)',
                    color_continuous_scale='Reds'
                )
                fig_time.update_traces(texttemplate='%{text:.2f}s', textposition='outside')
                st.plotly_chart(fig_time, use_container_width=True)

            # Results table
            st.subheader("📊 Detailed Comparison")
            df['Price Diff vs Best (%)'] = ((df['Price'] - df['Price'].max()) / df['Price'].max() * 100).round(2)
            df['Speed Rank'] = df['Time (s)'].rank().astype(int)
            st.dataframe(df, use_container_width=True, hide_index=True)

    else:
        st.info("👆 Select algorithms and click to compare")


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>American Option Pricing with Optimal Stopping Algorithms</p>
    <p>Built with Streamlit | Thesis Demo 2025</p>
</div>
""", unsafe_allow_html=True)
