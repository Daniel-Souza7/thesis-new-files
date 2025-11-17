"""Real Data Stock Model with Stationary Block Bootstrap.

This module provides a realistic stock price model based on actual market data:
- Downloads historical data using yfinance
- Implements stationary block bootstrap (Politis & Romano, 1994)
- Preserves autocorrelation, volatility clustering, and fat tails
- Configurable crisis period inclusion/exclusion
- Automatic block length selection based on autocorrelation decay
- Supports up to 250 S&P 500 stocks with 15+ years of data
"""

import numpy as np
import pandas as pd
import warnings
from typing import List, Optional, Tuple
from optimal_stopping.data.stock_model import Model


class RealDataModel(Model):
    """Stock model using real market data with stationary block bootstrap.

    Features:
    - Downloads real stock data via yfinance
    - Stationary block bootstrap preserves autocorrelation
    - Automatic optimal block length from autocorrelation structure
    - Configurable time periods and crisis handling
    - Preserves real correlations between stocks
    - Configurable drift override
    - Supports up to 250 S&P 500 stocks

    Example:
        >>> model = RealDataModel(
        ...     tickers=['AAPL', 'MSFT', 'GOOGL'],
        ...     start_date='2015-01-01',
        ...     end_date='2024-01-01',
        ...     exclude_crisis=True,
        ...     nb_stocks=3,
        ...     nb_paths=10000,
        ...     nb_dates=252,
        ...     spot=100,
        ...     maturity=1.0
        ... )
        >>> paths, _ = model.generate_paths()
    """

    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        start_date: str = '2010-01-01',
        end_date: str = '2024-01-01',
        exclude_crisis: bool = False,
        only_crisis: bool = False,
        drift_override: Optional[float] = None,
        volatility_override: Optional[float] = None,
        avg_block_length: Optional[int] = None,
        cache_data: bool = True,
        **kwargs
    ):
        """Initialize real data model.

        Args:
            tickers: List of stock tickers (default: top S&P 500 by nb_stocks)
            start_date: Start date for historical data (YYYY-MM-DD)
            end_date: End date for historical data (YYYY-MM-DD)
            exclude_crisis: If True, exclude 2008 and 2020 crisis periods
            only_crisis: If True, only use crisis periods (overrides exclude_crisis)
            drift_override: Override historical drift (None = use config drift if available, else historical)
            volatility_override: Override historical volatility (None = use config vol if available, else historical)
            avg_block_length: Average block length (None = auto-calculate from data)
            cache_data: Cache downloaded data to avoid re-downloading
            **kwargs: Additional arguments passed to Model base class
                     Note: If 'drift' or 'volatility' in kwargs, they're used as overrides
        """
        # Extract drift/volatility from kwargs if not explicitly provided
        # This allows configs to work: drift=0.05 in config → drift_override=0.05
        if drift_override is None and 'drift' in kwargs:
            drift_override = kwargs['drift']
        if volatility_override is None and 'volatility' in kwargs:
            volatility_override = kwargs['volatility']

        # Initialize base class
        super().__init__(**kwargs)

        # Set default tickers if none provided
        if tickers is None:
            tickers = self._get_default_tickers(self.nb_stocks)

        # Validate number of stocks matches tickers
        if self.nb_stocks > len(tickers):
            warnings.warn(
                f"nb_stocks ({self.nb_stocks}) > available tickers ({len(tickers)}). "
                f"Using {len(tickers)} stocks."
            )
            self.nb_stocks = len(tickers)
        elif self.nb_stocks < len(tickers):
            # Use first nb_stocks tickers
            tickers = tickers[:self.nb_stocks]

        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.exclude_crisis = exclude_crisis
        self.only_crisis = only_crisis
        self.drift_override = drift_override
        self.volatility_override = volatility_override
        self.avg_block_length = avg_block_length
        self.cache_data = cache_data

        # Download and process data
        print(f"📊 Loading historical data for {len(self.tickers)} stocks...")
        self._download_data()
        self._calculate_returns()
        self._apply_crisis_filters()
        self._calculate_statistics()

        # Determine optimal block length if not provided
        if self.avg_block_length is None:
            self.avg_block_length = self._estimate_block_length()

        print(f"✅ Loaded {len(self.tickers)} stocks: {', '.join(self.tickers)}")
        print(f"   {len(self.returns)} days of returns ({self.start_date} to {self.end_date})")
        print(f"   Average return: {self.empirical_drift_annual:.2%}")
        print(f"   Average volatility: {self.empirical_vol_annual:.2%}")
        print(f"   Block length: {self.avg_block_length} days")

    def _get_default_tickers(self, nb_stocks: int) -> List[str]:
        """Get default tickers from S&P 500 (top 250 by market cap with 15+ years data).

        Args:
            nb_stocks: Number of stocks requested

        Returns:
            List of tickers (up to 250)
        """
        # Top 250 S&P 500 stocks by market cap with 15+ years of continuous trading
        # Updated as of 2024, sorted by market cap
        # Note: BRK.B removed due to yfinance timezone issues with dotted tickers
        all_tickers = [
            # Mega caps (Top 10)
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'LLY', 'V', 'UNH',

            # Large caps (11-50)
            'JNJ', 'XOM', 'WMT', 'JPM', 'MA', 'PG', 'AVGO', 'CVX', 'HD', 'MRK',
            'ABBV', 'KO', 'PEP', 'COST', 'ADBE', 'MCD', 'CSCO', 'CRM', 'TMO', 'BAC',
            'ACN', 'ABT', 'NFLX', 'WFC', 'DHR', 'NKE', 'DIS', 'VZ', 'CMCSA', 'INTC',
            'TXN', 'NEE', 'PM', 'UNP', 'RTX', 'ORCL', 'AMD', 'COP', 'UPS', 'MS',

            # Mid-large caps (51-100)
            'LOW', 'HON', 'QCOM', 'GS', 'IBM', 'BA', 'CAT', 'SPGI', 'AXP', 'AMGN',
            'SBUX', 'INTU', 'DE', 'GE', 'BLK', 'MDT', 'PLD', 'LMT', 'BMY', 'GILD',
            'CVS', 'ISRG', 'BKNG', 'MMC', 'ADI', 'SYK', 'CB', 'C', 'SCHW', 'TJX',
            'MO', 'PGR', 'ZTS', 'VRTX', 'AMT', 'SO', 'EOG', 'REGN', 'BDX', 'CI',
            'DUK', 'PNC', 'BSX', 'USB', 'MMM', 'SLB', 'MDLZ', 'CME', 'AON', 'ITW',

            # Established companies (101-150)
            'CL', 'GD', 'EL', 'TGT', 'NOC', 'APD', 'ETN', 'FI', 'HUM', 'EQIX',
            'NSC', 'SHW', 'ICE', 'EMR', 'WM', 'PSA', 'MPC', 'MCO', 'EW', 'CSX',
            'WELL', 'HCA', 'BK', 'ADP', 'CCI', 'TT', 'MSI', 'FCX', 'OXY', 'CMG',
            'SPG', 'MAR', 'ROP', 'F', 'PSX', 'AFL', 'PH', 'APH', 'COF', 'AJG',
            'KLAC', 'MCK', 'AZO', 'GM', 'CARR', 'D', 'AIG', 'ORLY', 'ECL', 'NXPI',

            # Strong performers (151-200)
            'AEP', 'JCI', 'TRV', 'ROST', 'VLO', 'PCG', 'ADM', 'TEL', 'SRE', 'KMB',
            'PAYX', 'O', 'LRCX', 'ALL', 'GIS', 'KMI', 'YUM', 'HLT', 'DLR', 'CTVA',
            'MCHP', 'PCAR', 'ED', 'PRU', 'IQV', 'DHI', 'HSY', 'CMI', 'ADSK', 'DD',
            'EXC', 'EA', 'GWW', 'FAST', 'CTAS', 'FDX', 'IDXX', 'KHC', 'DXCM', 'BIIB',
            'BKR', 'ANET', 'ODFL', 'MSCI', 'STZ', 'VRSK', 'XEL', 'WEC', 'ROK', 'AME',

            # Diversification (201-250)
            'GLW', 'PPG', 'MTB', 'RMD', 'WBA', 'HAL', 'FITB', 'ES', 'TDG', 'DOW',
            'AWK', 'VICI', 'CTSH', 'EBAY', 'FTV', 'KEYS', 'HPQ', 'EFX', 'TROW', 'MNST',
            'SYY', 'DAL', 'EIX', 'CBRE', 'IFF', 'BALL', 'ETR', 'WBD', 'ANSS', 'HES',
            'RJF', 'EQR', 'K', 'FE', 'PPL', 'AVB', 'VMC', 'MLM', 'NTRS', 'CHD',
            'PEG', 'DTE', 'AEE', 'LH', 'HBAN', 'CAH', 'SBAC', 'SWKS', 'NVR', 'KEY',
        ]

        # Return up to nb_stocks tickers
        return all_tickers[:min(nb_stocks, len(all_tickers))]

    def _download_data(self):
        """Download historical price data using yfinance."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance is required for RealDataModel. "
                "Install with: pip install yfinance"
            )

        # Download data (suppress yfinance warnings/errors)
        import logging
        logging.getLogger('yfinance').setLevel(logging.CRITICAL)

        try:
            # Suppress stdout from yfinance
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = StringIO()

            try:
                data = yf.download(
                    self.tickers,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False,
                    auto_adjust=True  # Adjust for splits and dividends
                )
            finally:
                sys.stdout = old_stdout

            # Handle single ticker case
            if len(self.tickers) == 1:
                self.prices = pd.DataFrame(data['Close'])
                self.prices.columns = self.tickers
            else:
                self.prices = data['Close'][self.tickers]

            # Drop columns (tickers) with all NaN values (failed downloads)
            self.prices = self.prices.dropna(axis=1, how='all')

            # Check if we lost any tickers
            failed_tickers = set(self.tickers) - set(self.prices.columns)
            if failed_tickers:
                warnings.warn(
                    f"Failed to download data for {len(failed_tickers)} ticker(s): {sorted(failed_tickers)}"
                )
                # Update tickers list to only include successful downloads
                self.tickers = list(self.prices.columns)
                self.nb_stocks = len(self.tickers)

            # Drop rows with any NaN values
            self.prices = self.prices.dropna()

            if len(self.prices) == 0:
                raise ValueError("No data downloaded. Check tickers and date range.")

            if len(self.tickers) == 0:
                raise ValueError("All tickers failed to download.")

        except Exception as e:
            raise RuntimeError(
                f"Failed to download data: {e}\n"
                f"Tickers: {self.tickers}\n"
                f"Date range: {self.start_date} to {self.end_date}"
            )

    def _calculate_returns(self):
        """Calculate daily log returns."""
        # Log returns preserve multiplicative structure
        self.returns = np.log(self.prices / self.prices.shift(1)).dropna()

        # Store dates for crisis filtering
        self.dates = self.returns.index

    def _apply_crisis_filters(self):
        """Filter data based on crisis period settings."""
        if self.only_crisis:
            # Only keep crisis periods (2008 financial crisis, 2020 COVID)
            crisis_mask = (
                ((self.dates >= '2007-10-01') & (self.dates <= '2009-06-30')) |  # 2008 crisis
                ((self.dates >= '2020-02-01') & (self.dates <= '2020-05-31'))    # COVID crash
            )
            self.returns = self.returns[crisis_mask]
            print(f"   Filtered to crisis periods only: {len(self.returns)} days")

        elif self.exclude_crisis:
            # Exclude crisis periods
            crisis_mask = (
                ((self.dates >= '2007-10-01') & (self.dates <= '2009-06-30')) |
                ((self.dates >= '2020-02-01') & (self.dates <= '2020-05-31'))
            )
            self.returns = self.returns[~crisis_mask]
            print(f"   Excluded crisis periods: {len(self.returns)} days remaining")

    def _calculate_statistics(self):
        """Calculate empirical statistics from data."""
        # Convert to numpy array for calculations
        self.returns_array = self.returns.values  # Shape: (n_days, n_stocks)

        # Calculate statistics
        self.empirical_drift_daily = np.mean(self.returns_array, axis=0)
        self.empirical_drift_annual = np.mean(self.empirical_drift_daily) * 252

        self.empirical_vol_daily = np.std(self.returns_array, axis=0)
        self.empirical_vol_annual = np.mean(self.empirical_vol_daily) * np.sqrt(252)

        # Correlation matrix
        self.empirical_correlation = np.corrcoef(self.returns_array.T)

        # Use overrides if provided
        if self.drift_override is not None:
            # Convert annual drift to daily
            self.target_drift_daily = np.full(self.nb_stocks, self.drift_override / 252)
        else:
            self.target_drift_daily = self.empirical_drift_daily

        if self.volatility_override is not None:
            # Convert annual vol to daily
            self.target_vol_daily = np.full(self.nb_stocks, self.volatility_override / np.sqrt(252))
        else:
            self.target_vol_daily = self.empirical_vol_daily

    def _estimate_block_length(self) -> int:
        """Estimate optimal block length from autocorrelation decay.

        Uses the method from Politis & White (2004):
        Block length ≈ where autocorrelation falls below 2/√n

        Returns:
            Optimal average block length in days
        """
        n_days, n_stocks = self.returns_array.shape

        # Calculate autocorrelation for each stock
        max_lag = min(100, n_days // 4)  # Check up to 100 days or 1/4 of data
        block_lengths = []

        for stock_idx in range(min(n_stocks, 10)):  # Sample up to 10 stocks for speed
            returns = self.returns_array[:, stock_idx]

            # Calculate autocorrelations
            autocorr = np.zeros(max_lag)
            for lag in range(1, max_lag):
                autocorr[lag] = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]

            # Find where autocorrelation becomes insignificant
            threshold = 2.0 / np.sqrt(n_days)
            significant_lags = np.where(np.abs(autocorr) > threshold)[0]

            if len(significant_lags) > 0:
                # Block length is last significant lag
                block_lengths.append(significant_lags[-1])
            else:
                # No significant autocorrelation, use small block
                block_lengths.append(5)

        # Use average across stocks, bounded between 5 and 50 days
        avg_length = int(np.mean(block_lengths))
        optimal_length = np.clip(avg_length, 5, 50)

        return optimal_length

    def _stationary_bootstrap_indices(self, n_samples: int) -> np.ndarray:
        """Generate indices for stationary block bootstrap.

        Implements the stationary bootstrap of Politis & Romano (1994):
        - Block lengths follow geometric distribution
        - Blocks wrap around (circular)

        Args:
            n_samples: Number of samples to generate

        Returns:
            Array of indices to sample from historical data
        """
        n_days = len(self.returns_array)
        indices = np.zeros(n_samples, dtype=int)

        i = 0
        while i < n_samples:
            # Random starting point
            start_idx = np.random.randint(0, n_days)

            # Block length from geometric distribution
            # P(L = k) = (1/p) * (1 - 1/p)^(k-1) where p = avg_block_length
            block_length = np.random.geometric(1.0 / self.avg_block_length)

            # Sample consecutive indices (with wraparound)
            for j in range(block_length):
                if i >= n_samples:
                    break
                indices[i] = (start_idx + j) % n_days
                i += 1

        return indices

    def generate_paths(self, nb_paths: Optional[int] = None) -> Tuple[np.ndarray, None]:
        """Generate price paths using stationary block bootstrap.

        Args:
            nb_paths: Number of paths to generate (default: self.nb_paths)

        Returns:
            Tuple of (paths, None) where:
            - paths: Array of shape (nb_paths, nb_stocks, nb_dates+1)
            - None: Placeholder for compatibility with base class
        """
        nb_paths = nb_paths or self.nb_paths

        # Initialize paths array
        paths = np.zeros((nb_paths, self.nb_stocks, self.nb_dates + 1))
        paths[:, :, 0] = self.spot

        for path_idx in range(nb_paths):
            # Generate bootstrap indices for this path
            indices = self._stationary_bootstrap_indices(self.nb_dates)

            # Sample returns using these indices
            sampled_returns = self.returns_array[indices, :]  # Shape: (nb_dates, nb_stocks)

            # Adjust returns if drift/vol override specified
            if self.drift_override is not None or self.volatility_override is not None:
                # Demean and rescale
                sampled_returns = (sampled_returns - self.empirical_drift_daily) / self.empirical_vol_daily
                sampled_returns = sampled_returns * self.target_vol_daily + self.target_drift_daily

            # Build path from returns
            for t in range(self.nb_dates):
                # Apply log returns: S(t+1) = S(t) * exp(r(t))
                paths[path_idx, :, t + 1] = paths[path_idx, :, t] * np.exp(sampled_returns[t, :])

        return paths, None

    def drift_fct(self, x, t):
        """Drift function (not used for bootstrap, but required by base class)."""
        return np.mean(self.target_drift_daily) * 252 * x  # Annual drift

    def diffusion_fct(self, x, t, v=0):
        """Diffusion function (not used for bootstrap, but required by base class)."""
        return np.mean(self.target_vol_daily) * np.sqrt(252) * x  # Annual volatility

    def generate_one_path(self):
        """Generate a single path (not used, but required by base class)."""
        paths, _ = self.generate_paths(nb_paths=1)
        return paths[0]
