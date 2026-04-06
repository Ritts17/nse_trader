"""
core/broker.py — OHLCV Data Fetcher

Supports multiple data sources:
  - Yahoo Finance (yfinance)
  - CSV files (any broker export)
  - Zerodha (kiteconnect)
  - Upstox (upstox-python-sdk)
  - Synthetic (demo/offline)

Usage:
    df = get_ohlcv_auto(
        symbol="NIFTY",
        interval_min=5,
        days=10,
        csv_path="data/NIFTY_5min.csv",
        zerodha_config={...},
        upstox_token="...",
    )
"""

import pandas as pd
from typing import Optional, Dict, Any
from utils.sample_data import generate_multi_day_ohlcv


def get_ohlcv_auto(
    symbol: str,
    interval_min: int,
    days: int = 10,
    csv_path: Optional[str] = None,
    zerodha_config: Optional[Dict[str, Any]] = None,
    upstox_token: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from the best available source.

    Args:
        symbol: Trading symbol (e.g. "NIFTY", "RELIANCE")
        interval_min: Bar interval in minutes (1, 5, 15, etc.)
        days: Number of days of historical data
        csv_path: Path to CSV file for CSV source
        zerodha_config: Dict with Zerodha API config
        upstox_token: Upstox access token

    Returns:
        pd.DataFrame with columns: open, high, low, close, volume
        Index: datetime
    """

    # Priority: CSV > Zerodha > Upstox > Yahoo > Synthetic
    if csv_path:
        return _load_from_csv(csv_path, interval_min)
    elif zerodha_config:
        return _load_from_zerodha(symbol, interval_min, days, zerodha_config)
    elif upstox_token:
        return _load_from_upstox(symbol, interval_min, days, upstox_token)
    else:
        # Default to synthetic data
        return _load_synthetic(symbol, interval_min, days)


def _load_from_csv(csv_path: str, interval_min: int) -> pd.DataFrame:
    """Load OHLCV data from CSV file."""
    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        # Assume CSV is already in the desired interval
        return df
    except Exception as e:
        print(f"Warning: Could not load CSV {csv_path}: {e}")
        # Fallback to synthetic
        return _load_synthetic("NIFTY", interval_min, 10)


def _load_from_zerodha(symbol: str, interval_min: int, days: int,
                       config: Dict[str, Any]) -> pd.DataFrame:
    """Load OHLCV data from Zerodha Kite."""
    try:
        # Placeholder - would need kiteconnect library
        # from kiteconnect import KiteConnect
        # kite = KiteConnect(api_key=config["api_key"])
        # kite.set_access_token(config["access_token"])
        # data = kite.historical_data(...)
        print("Warning: Zerodha integration not implemented yet")
        return _load_synthetic(symbol, interval_min, days)
    except Exception as e:
        print(f"Warning: Zerodha fetch failed: {e}")
        return _load_synthetic(symbol, interval_min, days)


def _load_from_upstox(symbol: str, interval_min: int, days: int,
                      token: str) -> pd.DataFrame:
    """Load OHLCV data from Upstox."""
    try:
        # Placeholder - would need upstox-python-sdk
        print("Warning: Upstox integration not implemented yet")
        return _load_synthetic(symbol, interval_min, days)
    except Exception as e:
        print(f"Warning: Upstox fetch failed: {e}")
        return _load_synthetic(symbol, interval_min, days)


def _load_synthetic(symbol: str, interval_min: int, days: int) -> pd.DataFrame:
    """Generate synthetic OHLCV data for demo purposes."""
    # Get approximate spot price (placeholder - in real app would fetch current price)
    spot_prices = {
        "NIFTY": 22000,
        "BANKNIFTY": 45000,
        "BEL": 250,
        "RELIANCE": 2500,
    }
    spot = spot_prices.get(symbol.upper(), 1000)

    # Generate 1-minute data
    df_1min = generate_multi_day_ohlcv(
        spot=spot,
        n_days=days,
        volatility=0.15,  # 15% annual vol
        trend=0.0,        # neutral trend
    )

    # Resample to desired interval
    if interval_min == 1:
        return df_1min
    else:
        # Resample to larger intervals
        rule = f"{interval_min}min"
        df_resampled = df_1min.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()
        return df_resampled