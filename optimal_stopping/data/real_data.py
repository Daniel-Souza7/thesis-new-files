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
    - Uses empirical drift/volatility by default (configurable)
    - Supports up to 250 S&P 500 stocks

    Drift/Volatility Control (in configs):
    - Set drift=(None,), volatilities=(None,) → uses empirical values
    - Set drift=(0.05,), volatilities=(0.2,) → overrides empirical
    - Don't set drift/volatilities → uses _DefaultConfig values (0.02, 0.2)

    Example configs:
        # Use empirical drift/volatility
        test_config = _DefaultConfig(
            stock_models=['RealData'],
            drift=(None,),  # Use empirical
            volatilities=(None,),  # Use empirical
            ...
        )

        # Override to use specific drift/volatility
        test_config = _DefaultConfig(
            stock_models=['RealData'],
            drift=(0.05,),  # Force 5% drift
            volatilities=(0.2,),  # Force 20% volatility
            ...
        )
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
        start_index: int = 0,
        **kwargs
    ):
        """Initialize real data model.

        Args:
            tickers: List of stock tickers (default: top S&P 500 by nb_stocks)
            start_date: Start date for historical data (YYYY-MM-DD)
            end_date: End date for historical data (YYYY-MM-DD)
            exclude_crisis: If True, exclude 2008 and 2020 crisis periods
            only_crisis: If True, only use crisis periods (overrides exclude_crisis)
            drift_override: Override historical drift (None = use empirical historical drift)
            volatility_override: Override historical volatility (None = use empirical historical volatility)
            avg_block_length: Average block length (None = auto-calculate from data)
            cache_data: Cache downloaded data to avoid re-downloading
            **kwargs: Additional arguments passed to Model base class
                     Supports 'drift' and 'volatilities' from configs:
                     - drift=(None,) → use empirical drift
                     - drift=(0.05,) → override to 5% drift
                     - volatilities=(None,) → use empirical volatility
                     - volatilities=(0.2,) → override to 20% volatility
        """
        self.start_index = start_index
        # Extract drift/volatility from kwargs if not explicitly provided
        # This allows configs to control behavior:
        # - drift=(None,) in config → use empirical (no override)
        # - drift=(0.05,) in config → use 5% drift (override empirical)
        if drift_override is None and 'drift' in kwargs:
            drift_val = kwargs['drift']
            # Extract from tuple/list if needed: (None,) → None, (0.05,) → 0.05
            if isinstance(drift_val, (tuple, list)) and len(drift_val) > 0:
                drift_override = drift_val[0]
            else:
                drift_override = drift_val

        if volatility_override is None:
            # Check both 'volatilities' (config param) and 'volatility' (singular)
            vol_val = None
            if 'volatilities' in kwargs:
                vol_val = kwargs['volatilities']
            elif 'volatility' in kwargs:
                vol_val = kwargs['volatility']

            if vol_val is not None:
                # Extract from tuple/list if needed: (None,) → None, (0.2,) → 0.2
                if isinstance(vol_val, (tuple, list)) and len(vol_val) > 0:
                    volatility_override = vol_val[0]
                else:
                    volatility_override = vol_val

        # Base Model class requires drift and volatility for calculations
        # Extract from tuple if needed
        if isinstance(kwargs.get('drift'), (tuple, list)):
            drift_first = kwargs['drift'][0] if len(kwargs['drift']) > 0 else None
            kwargs['drift'] = drift_first if drift_first is not None else 0.05
        elif 'drift' not in kwargs or kwargs.get('drift') is None:
            kwargs['drift'] = 0.05  # Default 5% annual drift for base class

        if isinstance(kwargs.get('volatility'), (tuple, list)):
            vol_first = kwargs['volatility'][0] if len(kwargs['volatility']) > 0 else None
            kwargs['volatility'] = vol_first if vol_first is not None else 0.2
        elif 'volatility' not in kwargs or kwargs.get('volatility') is None:
            kwargs['volatility'] = 0.2  # Default 20% annual volatility for base class

        # Add name parameter for base class
        kwargs['name'] = 'RealData'

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
        print(f"   Empirical return: {self.empirical_drift_annual:.2%}, volatility: {self.empirical_vol_annual:.2%}")

        # Show if overrides are active
        if self.drift_override is not None:
            print(f"   ⚠️  Using OVERRIDE drift: {self.drift_override:.2%} (ignoring empirical)")
        if self.volatility_override is not None:
            print(f"   ⚠️  Using OVERRIDE volatility: {self.volatility_override:.2%} (ignoring empirical)")

        print(f"   Block length: {self.avg_block_length} days")
        self._debug_data_structure()

    def _get_default_tickers(self, nb_stocks: int) -> List[str]:
        """Get default tickers from S&P 500 and beyond (700+ stocks with 15+ years data).

        Args:
            nb_stocks: Number of stocks requested

        Returns:
            List of tickers (requests 1.5x nb_stocks to account for download failures)
        """
        # 700+ liquid stocks with long trading history
        # Updated as of 2024, sorted roughly by market cap and liquidity
        # Note: BRK.B removed due to yfinance timezone issues with dotted tickers
        # Requesting 1.5x more tickers than needed to account for yfinance download failures
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

            # S&P 500 continued (251-300)
            'IP', 'IR', 'LYB', 'LVS', 'AES', 'BBY', 'BXP', 'CNP', 'DG', 'DLTR',
            'DRI', 'EXPE', 'FIS', 'GRMN', 'HIG', 'HST', 'JBHT', 'L', 'LEN', 'LNC',
            'LUV', 'MKC', 'NDAQ', 'NEM', 'NI', 'NWSA', 'OMC', 'PKG', 'PKI', 'POOL',
            'PFG', 'RE', 'RF', 'RSG', 'SNPS', 'STE', 'SWK', 'TECH', 'TSN', 'TYL',
            'UDR', 'ULTA', 'URI', 'VFC', 'WDC', 'WYNN', 'ZBRA', 'CDNS', 'AMAT', 'MU',

            # Tech and growth (301-350)
            'PYPL', 'PANW', 'CRWD', 'NOW', 'SHOP', 'SQ', 'ABNB', 'UBER', 'COIN', 'SNOW',
            'FTNT', 'DDOG', 'ZS', 'NET', 'OKTA', 'TWLO', 'DOCU', 'ZM', 'ROKU', 'PINS',
            'SNAP', 'LYFT', 'DASH', 'RBLX', 'U', 'PATH', 'BILL', 'S', 'MDB', 'TEAM',
            'WDAY', 'ZI', 'VEEV', 'HUBS', 'ESTC', 'SPLK', 'RNG', 'GNRC', 'PODD', 'ALGN',
            'ILMN', 'INCY', 'TTWO', 'ATVI', 'NTAP', 'JNPR', 'FFIV', 'AKAM', 'CTXS', 'MPWR',

            # More S&P 500 (351-400)
            'TFX', 'WST', 'HOLX', 'JKHY', 'CPRT', 'CDAY', 'BR', 'LII', 'CBOE', 'CINF',
            'GL', 'AIZ', 'EXPD', 'CHRW', 'WAB', 'SIVB', 'TRMB', 'NDSN', 'PAYC', 'APA',
            'DVN', 'FANG', 'MRO', 'OKE', 'WMB', 'TRGP', 'LNG', 'CHTR', 'DPZ', 'RLGY',
            'BLDR', 'BBWI', 'TPR', 'PVH', 'RL', 'NCLH', 'CCL', 'RCL', 'MGM', 'ALB',
            'CE', 'FMC', 'EMN', 'CF', 'MOS', 'SEE', 'FLR', 'JEF', 'ALLE', 'AOS',

            # Industrial and materials (401-450)
            'BG', 'AVY', 'PHM', 'KBH', 'TOL', 'LEN', 'DHI', 'NVR', 'MTD', 'SNA',
            'PNR', 'HUBB', 'LDOS', 'HII', 'TXT', 'HWM', 'AXON', 'FLS', 'GNRC', 'IEX',
            'DOV', 'ITT', 'RRX', 'CR', 'BLDR', 'SSD', 'VMI', 'MLI', 'MAS', 'OC',
            'RPM', 'NEU', 'STLD', 'NUE', 'X', 'CLF', 'AA', 'ATI', 'MP', 'HUN',
            'LYB', 'WLK', 'APD', 'ECL', 'NEM', 'GOLD', 'AEM', 'FCX', 'SCCO', 'TECK',

            # Energy expansion (451-500)
            'APA', 'DVN', 'FANG', 'MRO', 'CVE', 'CNQ', 'SU', 'IMO', 'OVV', 'MTDR',
            'PR', 'SM', 'MGY', 'CHRD', 'RRC', 'AR', 'CTRA', 'NOG', 'VTLE', 'CPE',
            'OXY', 'COP', 'CVX', 'XOM', 'EOG', 'PSX', 'VLO', 'MPC', 'HFC', 'ANDV',
            'SLB', 'HAL', 'BKR', 'FTI', 'NOV', 'HP', 'PTEN', 'LBRT', 'NINE', 'WHD',
            'KMI', 'WMB', 'OKE', 'EPD', 'ET', 'MPLX', 'PAA', 'ENLC', 'DKL', 'USAC',

            # Financial services (501-550)
            'MS', 'GS', 'JPM', 'BAC', 'C', 'WFC', 'USB', 'PNC', 'TFC', 'FITB',
            'RF', 'CFG', 'KEY', 'HBAN', 'MTB', 'ZION', 'WTFC', 'FHN', 'SNV', 'ONB',
            'SIVB', 'SBNY', 'CMA', 'CFR', 'EWBC', 'PACW', 'WAL', 'SSB', 'UMBF', 'HWC',
            'BLK', 'SCHW', 'TROW', 'BEN', 'IVZ', 'STT', 'NTRS', 'AMG', 'SEIC', 'APAM',
            'AXP', 'V', 'MA', 'PYPL', 'SQ', 'FISV', 'FIS', 'GPN', 'FLT', 'JKHY',

            # Healthcare expansion (551-600)
            'UNH', 'CVS', 'CI', 'HUM', 'CNC', 'MOH', 'ANTM', 'ELV', 'HCA', 'THC',
            'UHS', 'CYH', 'LPLA', 'CHE', 'ENSG', 'RDNT', 'DVA', 'AMED', 'LHC', 'ACHC',
            'JNJ', 'PFE', 'ABBV', 'MRK', 'LLY', 'AMGN', 'GILD', 'BMY', 'REGN', 'VRTX',
            'BIIB', 'ALNY', 'INCY', 'JAZZ', 'SGEN', 'EXEL', 'BPMC', 'NBIX', 'UTHR', 'RARE',
            'ABT', 'TMO', 'DHR', 'SYK', 'BSX', 'EW', 'MDT', 'BDX', 'ZBH', 'BAX',

            # Consumer and retail (601-650)
            'AMZN', 'WMT', 'COST', 'TGT', 'DG', 'DLTR', 'BIG', 'BBY', 'FIVE', 'OLLI',
            'TSCO', 'AZO', 'ORLY', 'AAP', 'GPC', 'KR', 'SFM', 'GO', 'TJX', 'ROST',
            'BURL', 'GPS', 'ANF', 'AEO', 'URBN', 'DKS', 'FL', 'WSM', 'RH', 'HD',
            'LOW', 'BLDR', 'BECN', 'SHW', 'NUE', 'VMC', 'MLM', 'USCR', 'MAS', 'CARR',
            'KO', 'PEP', 'MNST', 'CELH', 'KDP', 'STZ', 'TAP', 'BF', 'SAM', 'FIZZ',

            # Tech and telecom expansion (651-700)
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NVDA', 'AMD', 'INTC', 'QCOM', 'AVGO',
            'TXN', 'ADI', 'MCHP', 'SWKS', 'NXPI', 'ON', 'MPWR', 'MU', 'LRCX', 'KLAC',
            'AMAT', 'ASML', 'TSM', 'CDNS', 'SNPS', 'ANSS', 'ADSK', 'CTSH', 'ACN', 'IBM',
            'ORCL', 'CRM', 'NOW', 'WDAY', 'PANW', 'CRWD', 'ZS', 'OKTA', 'FTNT', 'NET',
            'DDOG', 'S', 'MDB', 'TEAM', 'HUBS', 'VEEV', 'TWLO', 'ZM', 'DOCU', 'BILL',

            # Final buffer (701-750) - maximum diversification
            'VZ', 'T', 'TMUS', 'CHTR', 'CMCSA', 'DIS', 'NFLX', 'PARA', 'WBD', 'LGF',
            'FOXA', 'FOX', 'DISH', 'SIRI', 'LUMN', 'FYBR', 'AMC', 'CNK', 'IMAX', 'LYV',
            'BA', 'RTX', 'LMT', 'NOC', 'GD', 'HII', 'TDG', 'HEI', 'TXT', 'LDOS',
            'SAIC', 'CACI', 'KTOS', 'AVAV', 'CW', 'WWD', 'AJRD', 'ALEX', 'AIR', 'SPR',
            'UNP', 'NSC', 'CSX', 'CP', 'CNI', 'KSU', 'UPS', 'FDX', 'XPO', 'JBHT',
        ]

        # Remove duplicates while preserving order (keep first occurrence)
        seen = set()
        unique_tickers = []
        for ticker in all_tickers:
            if ticker not in seen:
                seen.add(ticker)
                unique_tickers.append(ticker)

        # Return up to 1.5x requested stocks to account for failures
        # This ensures we get at least nb_stocks even with download failures
        max_tickers = min(int(nb_stocks * 1.5), len(unique_tickers))
        return unique_tickers[:max_tickers]

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

            if len(self.prices.columns) == 0:
                raise ValueError("All tickers failed to download - no data available.")

            # Calculate data availability for each ticker
            total_rows = len(self.prices)
            coverage = self.prices.count() / total_rows

            # CRITICAL: For large stock counts, we need stocks with excellent coverage
            # Otherwise we'll have no common dates across all stocks
            # Use stricter filtering: require 90% coverage minimum
            min_coverage_threshold = 0.90
            high_coverage_tickers = coverage[coverage >= min_coverage_threshold]

            if len(high_coverage_tickers) < self.nb_stocks:
                # Not enough high-coverage stocks, relax threshold
                min_coverage_threshold = 0.80
                high_coverage_tickers = coverage[coverage >= min_coverage_threshold]

                if len(high_coverage_tickers) < self.nb_stocks:
                    # Still not enough, relax further
                    min_coverage_threshold = 0.70
                    high_coverage_tickers = coverage[coverage >= min_coverage_threshold]

            # Sort by coverage (best to worst)
            coverage_sorted = high_coverage_tickers.sort_values(ascending=False)

            # Determine how many stocks we can actually get
            target_stocks = min(self.nb_stocks, len(coverage_sorted))

            if target_stocks < self.nb_stocks:
                warnings.warn(
                    f"Only {target_stocks} stocks have >{min_coverage_threshold:.0%} coverage "
                    f"(requested {self.nb_stocks}). Consider a shorter date range for more stocks."
                )

            # Take the top N stocks with best coverage
            good_tickers = coverage_sorted.head(target_stocks).index.tolist()

            # Make sure all good_tickers are actually in self.prices
            # (they should be since coverage was calculated from self.prices)
            available_tickers = [t for t in good_tickers if t in self.prices.columns]

            if len(available_tickers) != len(good_tickers):
                missing = set(good_tickers) - set(available_tickers)
                warnings.warn(
                    f"Warning: {len(missing)} selected tickers not in price data: {list(missing)[:5]}"
                )

            # Filter to selected tickers that are actually available
            self.prices = self.prices[available_tickers]
            good_tickers = available_tickers

            # Report removed tickers
            removed_tickers = set(self.tickers) - set(good_tickers)
            if removed_tickers and len(removed_tickers) <= 20:
                warnings.warn(
                    f"Removed {len(removed_tickers)} ticker(s) with lowest coverage: "
                    f"{sorted(list(removed_tickers))}"
                )
            elif removed_tickers:
                warnings.warn(
                    f"Removed {len(removed_tickers)} ticker(s) with lowest coverage: "
                    f"{sorted(list(removed_tickers)[:10])}... and {len(removed_tickers)-10} more"
                )

            # Update tickers list to selected stocks
            self.tickers = good_tickers
            self.nb_stocks = len(self.tickers)

            # Now drop rows with any NaN values across selected stocks
            initial_rows = len(self.prices)
            self.prices = self.prices.dropna()
            dropped_rows = initial_rows - len(self.prices)

            if dropped_rows > 0:
                print(f"   Dropped {dropped_rows} rows with missing data")

            if len(self.prices) == 0:
                raise ValueError(
                    f"No common dates found across {len(good_tickers)} tickers. "
                    f"Try a different date range or fewer stocks."
                )

            # Verify we still have the right number of stocks
            final_stock_count = len(self.prices.columns)
            if final_stock_count != target_stocks:
                warnings.warn(
                    f"After data cleaning, have {final_stock_count} stocks instead of {target_stocks}. "
                    f"This may happen with sparse data."
                )

            # Final update of tickers list
            self.tickers = self.prices.columns.tolist()
            self.nb_stocks = len(self.tickers)

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

    def _estimate_block_length(self) -> float:
        """Estimate optimal average block length using Patton, Politis & White (2009).

        This replaces the previous heuristic with the rigorous spectral density
        estimation method, including the "correction" for the Stationary Bootstrap
        variance constant.

        Returns:
            Optimal average block length (float).
        """
        print("   Computing optimal block length (Patton, Politis & White 2009)...")
        n_days, n_stocks = self.returns_array.shape

        # We calculate the optimal block length for a sample of stocks and average them.
        # Computing for all 250+ stocks might be slow, so we take a robust sample.
        sample_size = min(n_stocks, 20)
        # Pick indices evenly spaced across the available stocks to get a mix of sectors
        indices = np.linspace(0, n_stocks - 1, sample_size, dtype=int)

        block_lengths = []

        for idx in indices:
            stock_returns = self.returns_array[:, idx]
            b_opt = self._calc_ppw_block_size(stock_returns)
            block_lengths.append(b_opt)

        # Take the mean of the optimal sizes
        avg_optimal = np.mean(block_lengths)

        # Clip to reasonable bounds to prevent bootstrap failure on edge cases
        # (e.g., extremely short history or perfect white noise)
        final_block_size = np.clip(avg_optimal, 1.0, min(100.0, n_days / 5.0))

        return float(final_block_size)

    def _calc_ppw_block_size(self, x: np.ndarray) -> float:
        """Calculate optimal block size using PPW (2009) correction & PW (2004) Bandwidth rule.

        Strict implementation of:
        1. Negligibility Threshold: c * sqrt(log10(N)/N) with c=2
        2. Bandwidth Rule: M = 2 * m_hat
        3. Flat-Top Window: Trapezoidal
        4. Variance Correction: D_SB = 2 * g^2(0) [Patton, Politis, White 2009]
        """
        N = len(x)
        if N < 20: return 1.0

        # 1. Center the data
        x_centered = x - np.mean(x)

        # 2. Determine Bandwidth 'M'
        # Search limit (K_N): "max(5, log10(N))"
        # We search deeper to be safe, but checks use the threshold logic.
        search_max = int(min(N / 4, np.ceil(10 * np.sqrt(np.log10(N) / N))))

        # Threshold for negligible autocorrelation
        # |R(k)| < 2 * sqrt(log10(N)/N)
        threshold = 2.0 * np.sqrt(np.log10(N) / N)

        # Compute normalized autocorrelation
        full_corr = np.correlate(x_centered, x_centered, mode='full')
        mid = len(full_corr) // 2
        acov = full_corr[mid:mid + search_max + 1] / N
        var_x = acov[0]

        if var_x < 1e-12: return 1.0
        rho = acov / var_x

        # Identify m_hat (smallest integer where correlation becomes negligible)
        # We look for the first spot where the next K_N lags are ALL below threshold
        m_hat = 1
        K_N = int(max(5, np.log10(N)))  #

        for k in range(1, len(rho) - K_N):
            # Check if lags k+1 through k+K_N are all insignificant
            if np.all(np.abs(rho[k + 1: k + 1 + K_N]) < threshold):
                m_hat = k
                break

        # "After identifying m_hat... the recommendation is to just take M = 2 * m_hat"
        M = 2 * m_hat
        M = max(1, M)  # Safety floor

        # 3. Define Flat-Top Window
        def flat_top(t):
            abs_t = abs(t)
            if abs_t <= 0.5:
                return 1.0  # "1 if t in [0, 1/2]"
            elif abs_t <= 1.0:
                return 2.0 * (1.0 - abs_t)  # "2(1-t) if t in [1/2, 1]"
            return 0.0  # "0 otherwise"

        # 4. Calculate Spectral Estimators
        # Sum from -M to M
        w_0 = flat_top(0)
        g_hat_0 = w_0 * acov[0]
        G_hat = 0.0

        for k in range(1, M + 1):
            if k >= len(acov): break  # Safety check

            w_k = flat_top(k / M)
            R_k = acov[k]

            # Add positive and negative lags (symmetry)
            # g(w) estimator
            g_hat_0 += 2.0 * w_k * R_k

            # G estimator
            G_hat += 2.0 * w_k * k * R_k

        # 5. Apply the Variance Constant CORRECTION [Patton, Politis, White 2009]
        # Replacing Eq (8) from 2004 text which was incorrect.
        D_SB = 2.0 * (g_hat_0 ** 2)

        if D_SB < 1e-10: return 1.0

        # 6. Final Optimal Block Size
        # b_opt = ((2 * G^2) / D_SB)^(1/3) * N^(1/3)
        term_1 = (2.0 * (G_hat ** 2)) / D_SB
        b_opt = (term_1 ** (1 / 3)) * (N ** (1 / 3))

        return b_opt
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

        # Calculate total trading days to sample (252 trading days per year)
        total_trading_days = int(self.maturity * 252)
        days_per_step = total_trading_days / self.nb_dates

        # Initialize paths array
        paths = np.zeros((nb_paths, self.nb_stocks, self.nb_dates + 1))

        # All stocks start at same spot value
        paths[:, :, 0] = self.spot

        for path_idx in range(nb_paths):
            # Sample enough daily returns to cover the full maturity period
            indices = self._stationary_bootstrap_indices(total_trading_days)
            sampled_returns = self.returns_array[indices, :]  # Shape: (total_trading_days, nb_stocks)

            # Adjust returns if drift/vol override specified
            if self.drift_override is not None or self.volatility_override is not None:
                # Demean and rescale
                sampled_returns = (sampled_returns - self.empirical_drift_daily) / self.empirical_vol_daily
                sampled_returns = sampled_returns * self.target_vol_daily + self.target_drift_daily

            # Build path by aggregating returns into nb_dates periods
            for t in range(self.nb_dates):
                # Get returns for this time period
                start_day = int(t * days_per_step)
                end_day = int((t + 1) * days_per_step)

                # Aggregate log returns: sum of log returns = log of product
                period_returns = np.sum(sampled_returns[start_day:end_day, :], axis=0)

                # Apply aggregated return
                paths[path_idx, :, t + 1] = paths[path_idx, :, t] * np.exp(period_returns)

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

    def _debug_data_structure(self):
        """Diagnose why block size is 1.0 by checking correlations."""
        print("\n🔍 --- DATA DIAGNOSTICS ---")

        # 1. Check Data Integrity
        zeros = np.sum(self.returns_array == 0)
        total = self.returns_array.size
        print(f"Data Health: {zeros} zeros out of {total} elements ({zeros / total:.2%})")

        # 2. Check CROSS-Correlation (Do stocks move together?)
        # We expect this to be HIGH
        corr_matrix = np.corrcoef(self.returns_array.T)
        avg_cross_corr = np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
        print(f"Avg CROSS-Correlation (Stock vs Stock): {avg_cross_corr:.4f} (Should be High)")

        # 3. Check AUTO-Correlation (Does yesterday predict today?)
        # We expect this to be LOW (approx 0) for Raw Returns
        print(f"\nAvg AUTO-Correlation (Lag 1 to 5):")
        for lag in range(1, 6):
            corrs = []
            for i in range(self.nb_stocks):
                series = self.returns_array[:, i]
                c = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                corrs.append(c)
            print(f"  Lag {lag}: {np.mean(corrs):.4f} (Raw Returns) -> Estimator sees this")

        # 4. Check Volatility Clustering (Squared Returns Auto-Correlation)
        # We expect this to be HIGHER. This is what you WANT the block size to capture.
        print(f"\nAvg VOLATILITY-Correlation (Squared Returns, Lag 1 to 5):")
        squared_returns = self.returns_array ** 2
        for lag in range(1, 6):
            corrs = []
            for i in range(self.nb_stocks):
                series = squared_returns[:, i]
                c = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                corrs.append(c)
            print(f"  Lag {lag}: {np.mean(corrs):.4f} (Squared Returns) -> Hidden structure")
        print("--------------------------\n")