"""
FastAPI backend for Option Pricing Calculator.

This standalone backend can be deployed to Railway, Render, or similar platforms.
It provides the same functionality as the Next.js API routes but as a dedicated Python service.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import sys
import os
import json
from contextlib import redirect_stdout
import io

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimal_stopping.data.stock_model import (
    BlackScholes,
    Heston,
    FractionalBlackScholes,
    RoughHeston,
    RealDataModel
)
from optimal_stopping.payoffs import get_payoff_class, get_all_payoffs
from optimal_stopping.algorithms.standard.rlsm import RLSM
from optimal_stopping.algorithms.standard.rfqi import RFQI
from optimal_stopping.algorithms.path_dependent.srlsm import SRLSM
from optimal_stopping.algorithms.path_dependent.srfqi import SRFQI
from optimal_stopping.algorithms.standard.lsm import LSM
from optimal_stopping.algorithms.standard.fqi import FQI
from optimal_stopping.algorithms.standard.eop import EOP

app = FastAPI(title="Option Pricing API")

# CORS configuration - allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class PricingRequest(BaseModel):
    model_type: str
    algorithm: str
    payoff_type: str
    spot_price: float
    strike: float
    volatility: float
    drift: float
    rate: float
    maturity: float
    nb_paths: int
    nb_dates: int
    nb_stocks: int
    hidden_size: Optional[int] = 64
    epochs: Optional[int] = 100
    barrier: Optional[float] = None
    barrier_upper: Optional[float] = None
    barrier_lower: Optional[float] = None

class StockValidateRequest(BaseModel):
    tickers: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class StockInfoRequest(BaseModel):
    tickers: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Option Pricing API",
        "version": "1.0.0"
    }


@app.get("/api/payoffs")
async def list_payoffs(name: Optional[str] = None):
    """
    Get list of all available payoffs or info about a specific payoff.

    Query params:
    - name: Optional payoff name to get detailed info
    """
    try:
        if name:
            # Get specific payoff info
            try:
                PayoffClass = get_payoff_class(name)
                return {
                    "success": True,
                    "payoff": {
                        "name": PayoffClass.__name__,
                        "abbreviation": PayoffClass.abbreviation,
                        "isPathDependent": PayoffClass.is_path_dependent,
                        "requiresMultipleAssets": PayoffClass.requires_multiple_assets
                    }
                }
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
        else:
            # Get all payoffs
            all_payoffs = get_all_payoffs()
            payoffs_list = [
                {
                    "name": p.__name__,
                    "abbreviation": p.abbreviation,
                    "isPathDependent": p.is_path_dependent,
                    "requiresMultipleAssets": p.requires_multiple_assets
                }
                for p in all_payoffs
            ]
            return {
                "success": True,
                "payoffs": payoffs_list
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/price")
async def price_option(request: PricingRequest):
    """
    Price an option using the specified model and algorithm.
    """
    try:
        # Get model class
        model_map = {
            'BlackScholes': BlackScholes,
            'Heston': Heston,
            'FractionalBlackScholes': FractionalBlackScholes,
            'RoughHeston': RoughHeston,
            'RealData': RealDataModel
        }

        if request.model_type not in model_map:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model type: {request.model_type}"
            )

        ModelClass = model_map[request.model_type]

        # Get payoff class
        try:
            PayoffClass = get_payoff_class(request.payoff_type)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Get algorithm class
        algo_map = {
            'RLSM': RLSM,
            'RFQI': RFQI,
            'SRLSM': SRLSM,
            'SRFQI': SRFQI,
            'LSM': LSM,
            'FQI': FQI,
            'EOP': EOP
        }

        if request.algorithm not in algo_map:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid algorithm: {request.algorithm}"
            )

        AlgoClass = algo_map[request.algorithm]

        # Initialize model
        model_params = {
            'spot': [request.spot_price] * request.nb_stocks,
            'drift': request.drift,
            'volatility': request.volatility,
            'rate': request.rate,
            'maturity': request.maturity,
            'nb_stocks': request.nb_stocks,
            'nb_dates': request.nb_dates,
            'nb_paths': request.nb_paths
        }

        model = ModelClass(**model_params)

        # Initialize payoff
        payoff_params = {'strike': request.strike}
        if request.barrier is not None:
            payoff_params['barrier'] = request.barrier
        if request.barrier_upper is not None:
            payoff_params['barrier_upper'] = request.barrier_upper
        if request.barrier_lower is not None:
            payoff_params['barrier_lower'] = request.barrier_lower

        payoff = PayoffClass(**payoff_params)

        # Initialize algorithm
        algo_params = {
            'hidden_size': request.hidden_size,
            'epochs': request.epochs
        }
        algo = AlgoClass(model, payoff, **algo_params)

        # Price the option (suppress debug output)
        with redirect_stdout(io.StringIO()):
            price, comp_time = algo.price()

        return {
            "success": True,
            "price": float(price),
            "computation_time": float(comp_time),
            "model": request.model_type,
            "algorithm": request.algorithm,
            "payoff": request.payoff_type
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        raise HTTPException(
            status_code=500,
            detail=f"Pricing error: {str(e)}\n{traceback.format_exc()}"
        )


@app.get("/api/stocks")
async def get_preloaded_stocks():
    """
    Get list of pre-loaded ticker symbols.
    """
    # Common stock tickers
    preloaded = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "TSLA", "NVDA", "JPM", "V", "WMT"
    ]
    return {
        "success": True,
        "tickers": preloaded
    }


@app.post("/api/stocks/validate")
async def validate_stocks(request: StockValidateRequest):
    """
    Validate stock ticker symbols.
    """
    try:
        import yfinance as yf

        valid_tickers = []
        invalid_tickers = []

        for ticker in request.tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                if 'regularMarketPrice' in info or 'currentPrice' in info:
                    valid_tickers.append(ticker)
                else:
                    invalid_tickers.append(ticker)
            except:
                invalid_tickers.append(ticker)

        return {
            "success": True,
            "valid": valid_tickers,
            "invalid": invalid_tickers
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stocks/info")
async def get_stock_info(request: StockInfoRequest):
    """
    Get detailed information about stock tickers.
    """
    try:
        import yfinance as yf
        import pandas as pd

        stocks_data = {}

        for ticker in request.tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(
                    start=request.start_date or "2020-01-01",
                    end=request.end_date or pd.Timestamp.now().strftime("%Y-%m-%d")
                )

                if not hist.empty:
                    stocks_data[ticker] = {
                        "current_price": float(hist['Close'].iloc[-1]),
                        "returns": hist['Close'].pct_change().dropna().tolist()[-30:],  # Last 30 days
                        "volatility": float(hist['Close'].pct_change().std() * (252 ** 0.5))  # Annualized
                    }
            except Exception as e:
                stocks_data[ticker] = {"error": str(e)}

        return {
            "success": True,
            "data": stocks_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
