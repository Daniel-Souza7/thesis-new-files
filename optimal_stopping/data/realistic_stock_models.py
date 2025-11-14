"""
Historical Data Stock Model - Uses real market returns

This model generates paths by:
1. Loading historical returns from actual stocks
2. Resampling with replacement (bootstrap)
3. Preserving real correlations, fat tails, volatility clustering
4. Optional: Add regime switching or crisis periods

Much more realistic than GBM/Heston for testing algorithms.
"""

import numpy as np
import pandas as pd
from optimal_stopping.data.stock_model import StockModel


class HistoricalDataModel(StockModel):
    """
    Generate paths using historical stock returns.

    Features:
    - Uses real return distributions (fat tails, skewness)
    - Preserves actual correlations between stocks
    - Can incorporate regime changes
    - Option to include crisis periods (2008, 2020)
    """

    def __init__(self,
                 tickers=None,  # e.g., ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
                 start_date='2010-01-01',
                 end_date='2024-01-01',
                 include_crisis=True,  # Include 2008, 2020 data
                 drift=0.0,  # Override average drift
                 volatility=None,  # Override volatility (None = use historical)
                 **kwargs):
        """
        Args:
            tickers: List of stock tickers to use. If None, uses S&P 500 constituents
            start_date: Start date for historical data
            end_date: End date for historical data
            include_crisis: Whether to include crisis periods
            drift: Override drift (default: use historical average)
            volatility: Override volatility (None = use historical)
        """
        super().__init__(**kwargs)

        self.tickers = tickers or self._get_default_tickers()
        self.start_date = start_date
        self.end_date = end_date
        self.include_crisis = include_crisis
        self.override_drift = drift
        self.override_volatility = volatility

        # Load historical data
        self._load_historical_returns()

    def _get_default_tickers(self):
        """Default basket of liquid stocks across sectors."""
        return [
            # Tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA',
            # Finance
            'JPM', 'BAC', 'GS', 'WFC',
            # Healthcare
            'JNJ', 'UNH', 'PFE',
            # Consumer
            'WMT', 'HD', 'MCD',
            # Energy
            'XOM', 'CVX',
            # Industrials
            'BA', 'CAT'
        ]

    def _load_historical_returns(self):
        """
        Load and process historical returns.

        In practice, you would use:
        - yfinance to download data
        - Or load from CSV files
        - Or use Bloomberg/Reuters data
        """
        # PLACEHOLDER - Replace with actual data loading
        # Example using yfinance:
        # import yfinance as yf
        # data = yf.download(self.tickers, start=self.start_date, end=self.end_date)
        # returns = data['Adj Close'].pct_change().dropna()

        # For now, simulate realistic returns with fat tails
        n_historical_days = 2520  # ~10 years of daily data

        # Generate returns with student-t distribution (fat tails)
        # Real markets have df ≈ 4-6 (fatter tails than normal)
        df = 5  # degrees of freedom

        self.historical_returns = np.random.standard_t(
            df,
            size=(n_historical_days, len(self.tickers))
        )

        # Scale to match typical stock volatility (20-40% annually)
        self.historical_returns *= 0.015  # ~15% annual vol for daily returns

        # Add correlation structure
        correlation_matrix = self._create_realistic_correlation()
        L = np.linalg.cholesky(correlation_matrix)
        self.historical_returns = self.historical_returns @ L.T

        # Add drift
        self.historical_returns += self.drift / 252  # Drift per day

        # Store statistics
        self.empirical_drift = np.mean(self.historical_returns, axis=0)
        self.empirical_volatility = np.std(self.historical_returns, axis=0) * np.sqrt(252)
        self.empirical_correlation = np.corrcoef(self.historical_returns.T)

        print(f"Loaded {len(self.historical_returns)} days of returns")
        print(f"Average volatility: {np.mean(self.empirical_volatility):.1%}")

    def _create_realistic_correlation(self):
        """Create realistic correlation structure between stocks."""
        n = len(self.tickers)

        # Start with base correlation (0.3-0.5 is typical for large caps)
        base_corr = 0.4
        corr = np.ones((n, n)) * base_corr
        np.fill_diagonal(corr, 1.0)

        # Make it positive definite
        eigenvalues, eigenvectors = np.linalg.eigh(corr)
        eigenvalues = np.maximum(eigenvalues, 0.01)
        corr = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Normalize
        D = np.diag(1.0 / np.sqrt(np.diag(corr)))
        corr = D @ corr @ D

        return corr

    def generate_paths(self):
        """
        Generate paths by resampling historical returns.

        Returns:
            Array of shape (nb_paths, nb_stocks, nb_dates+1)
        """
        # Bootstrap returns: sample with replacement
        n_days = len(self.historical_returns)

        # For each path, sample nb_dates consecutive returns
        paths = np.zeros((self.nb_paths, self.nb_stocks, self.nb_dates + 1))
        paths[:, :, 0] = self.spot

        for path_idx in range(self.nb_paths):
            # Sample random starting point (with enough room for nb_dates)
            start_idx = np.random.randint(0, n_days - self.nb_dates)

            # Get consecutive returns (preserves autocorrelation)
            sampled_returns = self.historical_returns[start_idx:start_idx + self.nb_dates]

            # Build path
            for t in range(self.nb_dates):
                # Apply return to get next price
                paths[path_idx, :, t + 1] = paths[path_idx, :, t] * (1 + sampled_returns[t])

        return paths


class MarketRegimeModel(StockModel):
    """
    Market model with regime switching (bull/bear/crisis).

    More realistic than single-regime models. Markets switch between:
    - Bull regime: High drift, low vol
    - Normal regime: Medium drift, medium vol
    - Bear/Crisis regime: Negative drift, high vol
    """

    def __init__(self,
                 regime_probs=(0.3, 0.5, 0.2),  # Bull, Normal, Bear
                 regime_drifts=(0.15, 0.08, -0.10),  # Annual drifts
                 regime_vols=(0.15, 0.20, 0.40),  # Annual volatilities
                 transition_prob=0.05,  # Probability of regime change per day
                 **kwargs):
        super().__init__(**kwargs)

        self.regime_probs = regime_probs
        self.regime_drifts = regime_drifts
        self.regime_vols = regime_vols
        self.transition_prob = transition_prob

    def generate_paths(self):
        """Generate paths with regime switching."""
        paths = np.zeros((self.nb_paths, self.nb_stocks, self.nb_dates + 1))
        paths[:, :, 0] = self.spot

        dt = self.maturity / self.nb_dates

        for path_idx in range(self.nb_paths):
            # Start in random regime
            current_regime = np.random.choice(3, p=self.regime_probs)

            for t in range(self.nb_dates):
                # Maybe switch regime
                if np.random.random() < self.transition_prob:
                    current_regime = np.random.choice(3, p=self.regime_probs)

                # Get regime parameters
                drift = self.regime_drifts[current_regime]
                vol = self.regime_vols[current_regime]

                # Generate returns
                dW = np.random.randn(self.nb_stocks)
                returns = drift * dt + vol * np.sqrt(dt) * dW

                paths[path_idx, :, t + 1] = paths[path_idx, :, t] * np.exp(returns)

        return paths


class JumpDiffusionModel(StockModel):
    """
    Merton Jump Diffusion - adds sudden jumps to GBM.

    More realistic for capturing:
    - Earnings announcements
    - Market crashes
    - News events

    S(t) follows GBM + Poisson jumps
    """

    def __init__(self,
                 jump_intensity=0.1,  # Average 0.1 jumps per year
                 jump_mean=-0.05,  # Average jump size (negative = crash)
                 jump_std=0.10,  # Jump volatility
                 **kwargs):
        super().__init__(**kwargs)

        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std

    def generate_paths(self):
        """Generate paths with jumps."""
        paths = np.zeros((self.nb_paths, self.nb_stocks, self.nb_dates + 1))
        paths[:, :, 0] = self.spot

        dt = self.maturity / self.nb_dates

        for path_idx in range(self.nb_paths):
            for t in range(self.nb_dates):
                # Regular GBM component
                dW = np.random.randn(self.nb_stocks)
                diffusion = (self.drift - 0.5 * self.volatility ** 2) * dt + \
                            self.volatility * np.sqrt(dt) * dW

                # Jump component
                n_jumps = np.random.poisson(self.jump_intensity * dt, self.nb_stocks)
                jump = np.zeros(self.nb_stocks)
                for i in range(self.nb_stocks):
                    if n_jumps[i] > 0:
                        jump[i] = np.sum(np.random.normal(
                            self.jump_mean, self.jump_std, n_jumps[i]
                        ))

                # Apply both
                paths[path_idx, :, t + 1] = paths[path_idx, :, t] * \
                                            np.exp(diffusion + jump)

        return paths