"""User Data Model with Stationary Block Bootstrap.

This module provides a stock model based on user-provided data:
- Loads price/return data from CSV or HDF5 files in user_data/ folder
- Implements stationary block bootstrap (same as RealDataModel)
- Preserves autocorrelation, volatility clustering, and fat tails
- Automatic block length selection based on autocorrelation decay
- Supports arbitrary number of stocks with consistent date ranges

Supported File Formats:
- CSV (.csv): Text format with comma-separated values
- HDF5 (.h5, .hdf5, .hdf): Binary format for large datasets

CSV Format Requirements:
- Columns: date, ticker, price (or return)
- Date format: YYYY-MM-DD or any pandas-compatible format
- One file per analysis (all stocks in one CSV), or multiple files
- Example:
    date,ticker,price
    2020-01-01,AAPL,300.00
    2020-01-01,MSFT,160.00
    2020-01-02,AAPL,302.50
    2020-01-02,MSFT,161.20
"""

import numpy as np
import pandas as pd
import warnings
import h5py
from pathlib import Path
from typing import List, Optional, Tuple, Union
from optimal_stopping.data.stock_model import Model


class UserDataModel(Model):
    """Stock model using user-provided CSV data with stationary block bootstrap.

    Features:
    - Load price or return data from CSV
    - Stationary block bootstrap preserves autocorrelation
    - Automatic optimal block length from autocorrelation structure
    - Supports drift/volatility overrides
    - Preserves real correlations between stocks

    CSV Format:
        date,ticker,price
        2020-01-01,AAPL,300.00
        2020-01-01,MSFT,160.00
        ...

    Usage in configs:
        test_config = _DefaultConfig(
            stock_models=['UserData'],
            drift=(None,),  # Use empirical from CSV
            volatilities=(None,),  # Use empirical from CSV
            user_data_file='my_data.csv',  # File in user_data/ folder
            ...
        )
    """

    def __init__(
        self,
        data_file: Optional[str] = None,
        data_folder: str = str(Path(__file__).parent / "user_data"),
        tickers: Optional[List[str]] = None,
        date_column: str = 'date',
        ticker_column: str = 'ticker',
        value_column: str = 'price',  # or 'return'
        value_type: str = 'price',  # 'price' or 'return'
        drift_override: Optional[float] = None,
        volatility_override: Optional[float] = None,
        avg_block_length: Optional[int] = None,
        **kwargs
    ):
        """Initialize user data model.

        Args:
            data_file: CSV filename (e.g., 'mydata.csv') in data_folder
            data_folder: Path to folder containing CSV files
            tickers: List of tickers to load (None = load all from CSV)
            date_column: Name of date column in CSV
            ticker_column: Name of ticker column in CSV
            value_column: Name of price/return column in CSV
            value_type: 'price' or 'return' - how to interpret value_column
            drift_override: Override empirical drift (None = use empirical)
            volatility_override: Override empirical volatility (None = use empirical)
            avg_block_length: Average block length (None = auto-calculate)
            **kwargs: Additional arguments passed to Model base class
        """
        # Extract drift/volatility from kwargs (config parameters)
        if drift_override is None and 'drift' in kwargs:
            drift_val = kwargs['drift']
            if isinstance(drift_val, (tuple, list)) and len(drift_val) > 0:
                drift_override = drift_val[0]
            else:
                drift_override = drift_val

        if volatility_override is None:
            vol_val = None
            if 'volatilities' in kwargs:
                vol_val = kwargs['volatilities']
            elif 'volatility' in kwargs:
                vol_val = kwargs['volatility']

            if vol_val is not None:
                if isinstance(vol_val, (tuple, list)) and len(vol_val) > 0:
                    volatility_override = vol_val[0]
                else:
                    volatility_override = vol_val

        # Extract data_file from kwargs if not provided
        if data_file is None and 'user_data_file' in kwargs:
            data_file = kwargs['user_data_file']

        # Provide defaults for base class if not specified
        if 'drift' not in kwargs or kwargs.get('drift') is None:
            kwargs['drift'] = 0.05
        if 'volatility' not in kwargs or kwargs.get('volatility') is None:
            kwargs['volatility'] = 0.2

        # Extract tuple if needed
        if isinstance(kwargs.get('drift'), (tuple, list)):
            drift_first = kwargs['drift'][0] if len(kwargs['drift']) > 0 else None
            kwargs['drift'] = drift_first if drift_first is not None else 0.05
        if isinstance(kwargs.get('volatility'), (tuple, list)):
            vol_first = kwargs['volatility'][0] if len(kwargs['volatility']) > 0 else None
            kwargs['volatility'] = vol_first if vol_first is not None else 0.2

        # Add name parameter for base class
        kwargs['name'] = 'UserData'

        # Initialize base class
        super().__init__(**kwargs)

        self.data_file = data_file
        self.data_folder = Path(data_folder)
        self.date_column = date_column
        self.ticker_column = ticker_column
        self.value_column = value_column
        self.value_type = value_type
        self.drift_override = drift_override
        self.volatility_override = volatility_override
        self.avg_block_length = avg_block_length
        self.tickers = tickers

        # Load and process data
        print(f"📊 Loading user data from {data_file}...")
        self._load_data()
        self._calculate_returns()
        self._calculate_statistics()

        # Determine optimal block length if not provided
        if self.avg_block_length is None:
            self.avg_block_length = self._estimate_block_length()

        print(f"✅ Loaded {len(self.tickers)} stocks: {', '.join(self.tickers)}")
        print(f"   {len(self.returns)} days of returns")
        print(f"   Empirical return: {self.empirical_drift_annual:.2%}, volatility: {self.empirical_vol_annual:.2%}")

        # Show if overrides are active
        if self.drift_override is not None:
            print(f"   ⚠️  Using OVERRIDE drift: {self.drift_override:.2%} (ignoring empirical)")
        if self.volatility_override is not None:
            print(f"   ⚠️  Using OVERRIDE volatility: {self.volatility_override:.2%} (ignoring empirical)")

        print(f"   Block length: {self.avg_block_length} days")

    def _load_data(self):
        """Load price/return data from CSV or HDF5 file."""
        if self.data_file is None:
            raise ValueError(
                "data_file must be specified. Either:\n"
                "1. Pass data_file='mydata.csv' (or .h5) to UserDataModel(...)\n"
                "2. Set user_data_file='mydata.csv' (or .h5) in config"
            )

        # Construct full path
        file_path = self.data_folder / self.data_file

        if not file_path.exists():
            # Try without folder (maybe user provided full path)
            file_path = Path(self.data_file)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {file_path}\n"
                f"Expected location: {self.data_folder / self.data_file}\n"
                f"Available files in {self.data_folder}:\n"
                f"{list(self.data_folder.glob('*')) if self.data_folder.exists() else 'Folder does not exist'}"
            )

        # Load data based on file extension
        file_ext = file_path.suffix.lower()
        try:
            if file_ext in ['.h5', '.hdf5', '.hdf']:
                # Load HDF5 file using h5py (like stored_paths does)
                with h5py.File(file_path, 'r') as f:
                    # List all keys in the file
                    keys = list(f.keys())
                    print(f"   HDF5 file keys: {keys}")

                    if not keys:
                        raise ValueError(f"HDF5 file has no datasets. Keys: {keys}")

                    # Try to find the main dataset
                    # Common names: 'data', 'df', 'table', or use first key
                    dataset_key = None
                    for common_name in ['data', 'df', 'table', 'dataframe']:
                        if common_name in keys:
                            dataset_key = common_name
                            break

                    if dataset_key is None:
                        dataset_key = keys[0]

                    print(f"   Using dataset: '{dataset_key}'")

                    # Read the dataset
                    dataset = f[dataset_key]

                    # Check if it's a stored paths file (3D array format)
                    if len(dataset.shape) == 3:
                        # This is a stored paths file (nb_paths, nb_stocks, nb_dates+1)
                        raise ValueError(
                            f"HDF5 file contains stored paths (3D array: {dataset.shape}).\n"
                            f"This appears to be a file created by store_paths().\n\n"
                            f"UserDataModel expects historical price/return data in tabular format.\n"
                            f"To use this file, either:\n"
                            f"  1. Use StoredPathsModel instead of UserDataModel\n"
                            f"  2. Or provide a CSV/HDF5 file with columns: date, ticker, price\n\n"
                            f"See stored_model.py for how to use stored paths."
                        )

                    # Check if it's a pandas table (has specific structure)
                    if hasattr(dataset, 'dtype') and dataset.dtype.names:
                        # It's a structured array (pandas table format)
                        data = dataset[:]
                        df = pd.DataFrame(data)
                    else:
                        # Try to read as regular numpy array and convert to DataFrame
                        data = np.array(dataset)

                        # Check if there are column names stored as attributes
                        if 'columns' in dataset.attrs:
                            columns = dataset.attrs['columns']
                            if isinstance(columns, bytes):
                                columns = columns.decode('utf-8').split(',')
                            df = pd.DataFrame(data, columns=columns)
                        else:
                            # No column info, assume it's already in the right format
                            # or try to infer structure
                            df = pd.DataFrame(data)

            elif file_ext == '.csv':
                # Load CSV file
                df = pd.read_csv(file_path)
            else:
                raise ValueError(
                    f"Unsupported file format: {file_ext}\n"
                    f"Supported formats: .csv, .h5, .hdf5, .hdf"
                )
        except Exception as e:
            raise ValueError(f"Failed to read file {file_path}: {e}")

        # Validate required columns
        required_cols = [self.date_column, self.ticker_column, self.value_column]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"CSV missing required columns: {missing_cols}\n"
                f"Available columns: {list(df.columns)}\n"
                f"Expected columns: {required_cols}"
            )

        # Parse dates
        try:
            df[self.date_column] = pd.to_datetime(df[self.date_column])
        except Exception as e:
            raise ValueError(f"Failed to parse dates in column '{self.date_column}': {e}")

        # Filter to requested tickers if specified
        if self.tickers is not None:
            df = df[df[self.ticker_column].isin(self.tickers)]
            if len(df) == 0:
                raise ValueError(
                    f"No data found for requested tickers: {self.tickers}\n"
                    f"Available tickers: {df[self.ticker_column].unique().tolist()}"
                )
        else:
            # Use all tickers from CSV
            self.tickers = sorted(df[self.ticker_column].unique().tolist())

        # Validate we have enough stocks
        if len(self.tickers) < self.nb_stocks:
            warnings.warn(
                f"CSV has {len(self.tickers)} stocks but nb_stocks={self.nb_stocks}. "
                f"Using {len(self.tickers)} stocks."
            )
            self.nb_stocks = len(self.tickers)
        elif len(self.tickers) > self.nb_stocks:
            # Take first nb_stocks
            self.tickers = self.tickers[:self.nb_stocks]
            df = df[df[self.ticker_column].isin(self.tickers)]

        # Pivot to wide format (dates × tickers)
        try:
            pivot_df = df.pivot(
                index=self.date_column,
                columns=self.ticker_column,
                values=self.value_column
            )
        except Exception as e:
            raise ValueError(
                f"Failed to pivot data: {e}\n"
                "Ensure each (date, ticker) pair has exactly one row"
            )

        # Sort by date and select tickers in order
        pivot_df = pivot_df.sort_index()
        self.prices = pivot_df[self.tickers]

        # Drop rows with any missing values
        initial_rows = len(self.prices)
        self.prices = self.prices.dropna()
        dropped_rows = initial_rows - len(self.prices)

        if dropped_rows > 0:
            print(f"   Dropped {dropped_rows} rows with missing data")

        if len(self.prices) == 0:
            raise ValueError(
                f"No complete data rows found for tickers: {self.tickers}\n"
                "Ensure all tickers have overlapping date ranges"
            )

        # If data is returns, we'll handle that in _calculate_returns()

    def _calculate_returns(self):
        """Calculate daily log returns from prices or use provided returns."""
        if self.value_type == 'return':
            # Data is already returns - use directly
            self.returns = self.prices
            print(f"   Using provided returns (not computing from prices)")
        else:
            # Data is prices - calculate log returns
            self.returns = np.log(self.prices / self.prices.shift(1)).dropna()

        # Store dates
        self.dates = self.returns.index

    def _calculate_statistics(self):
        """Calculate empirical statistics from data."""
        # Convert to numpy array
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
            self.target_drift_daily = np.full(self.nb_stocks, self.drift_override / 252)
        else:
            self.target_drift_daily = self.empirical_drift_daily

        if self.volatility_override is not None:
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

        # All stocks start at same spot value
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
