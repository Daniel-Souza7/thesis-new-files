"""Stock data management utilities.

This module provides utilities for managing stock ticker data:
- Pre-loading common tickers
- Caching downloaded data
- Fetching custom ticker data
- Computing empirical statistics
"""

import sys
import json
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

# Add parent directory to path for imports (cross-platform)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))  # Go up two levels from api/
sys.path.insert(0, project_root)

from optimal_stopping.data.real_data import RealDataModel


# Extended list of pre-loaded tickers (top S&P 500 stocks)
PRELOADED_TICKERS = [
    # Mega caps
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'LLY', 'V', 'UNH',
    # Large caps
    'JNJ', 'XOM', 'WMT', 'JPM', 'MA', 'PG', 'AVGO', 'CVX', 'HD', 'MRK',
    'ABBV', 'KO', 'PEP', 'COST', 'ADBE', 'MCD', 'CSCO', 'CRM', 'TMO', 'BAC',
    # More diversification
    'ACN', 'ABT', 'NFLX', 'WFC', 'DHR', 'NKE', 'DIS', 'VZ', 'CMCSA', 'INTC',
    'TXN', 'NEE', 'PM', 'UNP', 'RTX', 'ORCL', 'AMD', 'COP', 'UPS', 'MS',
    'LOW', 'HON', 'QCOM', 'GS', 'IBM', 'BA', 'CAT', 'SPGI', 'AXP', 'AMGN',
]

DEFAULT_START_DATE = '2010-01-01'
DEFAULT_END_DATE = '2024-01-01'


class StockDataManager:
    """Manages stock data fetching and caching."""

    def __init__(self):
        """Initialize stock data manager."""
        self.cache = {}  # Cache for downloaded data

    def get_tickers_info(self, tickers: List[str], start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Get detailed information about tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Dictionary with ticker information and statistics
        """
        start_date = start_date or DEFAULT_START_DATE
        end_date = end_date or DEFAULT_END_DATE

        try:
            # Check cache
            cache_key = f"{','.join(sorted(tickers))}_{start_date}_{end_date}"
            if cache_key in self.cache:
                print(f"Using cached data for {cache_key}", file=sys.stderr)
                return self.cache[cache_key]

            # Create RealData model to fetch statistics
            # Use minimal configuration for speed
            model = RealDataModel(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                nb_stocks=len(tickers),
                spot=100.0,
                nb_paths=100,  # Minimal paths
                nb_dates=10,
                maturity=1.0,
                rate=0.03,
                dividend=0.0,
                cache_data=True,
            )

            # Extract per-ticker statistics
            ticker_stats = []
            for i, ticker in enumerate(model.tickers):
                ticker_stats.append({
                    'ticker': ticker,
                    'drift_annual': float(model.empirical_drift_daily[i] * 252),
                    'volatility_annual': float(model.empirical_vol_daily[i] * np.sqrt(252)),
                })

            # Build result
            result = {
                'success': True,
                'tickers': model.tickers,
                'start_date': start_date,
                'end_date': end_date,
                'data_days': len(model.returns),
                'block_length': model.avg_block_length,
                'overall': {
                    'drift_annual': float(model.empirical_drift_annual),
                    'volatility_annual': float(model.empirical_vol_annual),
                },
                'ticker_stats': ticker_stats,
                'correlation_matrix': model.empirical_correlation.tolist(),
            }

            # Cache result
            self.cache[cache_key] = result

            return result

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
            }

    def validate_tickers(self, tickers: List[str], start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Validate that tickers have sufficient data.

        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Dictionary with validation results
        """
        start_date = start_date or DEFAULT_START_DATE
        end_date = end_date or DEFAULT_END_DATE

        try:
            import yfinance as yf
            import logging
            from io import StringIO

            # Suppress yfinance output
            logging.getLogger('yfinance').setLevel(logging.CRITICAL)
            old_stdout = sys.stdout
            sys.stdout = StringIO()

            try:
                # Download data
                data = yf.download(
                    tickers,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True
                )
            finally:
                sys.stdout = old_stdout

            # Check which tickers succeeded
            if len(tickers) == 1:
                prices = pd.DataFrame(data['Close'])
                prices.columns = tickers
            else:
                prices = data['Close'][tickers]

            # Calculate coverage
            total_rows = len(prices)
            coverage = prices.count() / total_rows if total_rows > 0 else pd.Series()

            valid_tickers = []
            invalid_tickers = []

            for ticker in tickers:
                if ticker in coverage.index and coverage[ticker] >= 0.7:
                    valid_tickers.append({
                        'ticker': ticker,
                        'coverage': float(coverage[ticker]),
                        'available_days': int(prices[ticker].count()),
                    })
                else:
                    invalid_tickers.append(ticker)

            return {
                'success': True,
                'valid_tickers': valid_tickers,
                'invalid_tickers': invalid_tickers,
                'date_range': {
                    'start': start_date,
                    'end': end_date,
                    'total_days': total_rows,
                },
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
            }

    def get_preloaded_tickers(self) -> Dict[str, Any]:
        """Get list of pre-loaded tickers.

        Returns:
            Dictionary with pre-loaded ticker information
        """
        return {
            'success': True,
            'tickers': PRELOADED_TICKERS,
            'count': len(PRELOADED_TICKERS),
            'default_date_range': {
                'start': DEFAULT_START_DATE,
                'end': DEFAULT_END_DATE,
            },
            'categories': {
                'mega_caps': PRELOADED_TICKERS[:10],
                'large_caps': PRELOADED_TICKERS[10:30],
                'diversified': PRELOADED_TICKERS[30:],
            }
        }


def main():
    """Main entry point for CLI usage."""
    if len(sys.argv) < 2:
        print(json.dumps({
            'success': False,
            'error': 'No command provided. Usage: python stock_data.py <command> <json_params>'
        }))
        sys.exit(1)

    command = sys.argv[1]

    # Parse parameters
    if len(sys.argv) > 2:
        try:
            params = json.loads(sys.argv[2])
        except json.JSONDecodeError as e:
            print(json.dumps({
                'success': False,
                'error': f'Invalid JSON parameters: {e}'
            }))
            sys.exit(1)
    else:
        params = {}

    # Create manager
    manager = StockDataManager()

    # Execute command
    if command == 'info':
        tickers = params.get('tickers', PRELOADED_TICKERS[:10])
        start_date = params.get('start_date', None)
        end_date = params.get('end_date', None)
        result = manager.get_tickers_info(tickers, start_date, end_date)

    elif command == 'validate':
        tickers = params.get('tickers', [])
        start_date = params.get('start_date', None)
        end_date = params.get('end_date', None)
        result = manager.validate_tickers(tickers, start_date, end_date)

    elif command == 'preloaded':
        result = manager.get_preloaded_tickers()

    else:
        result = {
            'success': False,
            'error': f'Unknown command: {command}. Available: info, validate, preloaded'
        }

    # Output result as JSON
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
