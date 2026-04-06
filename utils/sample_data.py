"""
utils/sample_data.py — Synthetic OHLCV Generator for Backtesting

When live NSE data is unavailable (market closed / no session),
this generates realistic intraday OHLCV data for strategy testing.

Also provides a helper to convert NSE option chain data into
a pseudo-OHLCV series using the lastPrice field across multiple fetches.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_intraday_ohlcv(
    spot:          float,
    date:          str  = None,   # "YYYY-MM-DD", defaults to today
    volatility:    float = 0.15,  # annual vol (e.g. 0.15 = 15%)
    trend:         float = 0.0,   # daily drift (+ve = up, -ve = down)
    n_minutes:     int  = 375,    # 9:15 to 15:30 = 375 minutes
    seed:          int  = 42,
) -> pd.DataFrame:
    """
    Generate synthetic 1-minute OHLCV data for a trading day.

    Uses geometric Brownian motion with intraday volume profile
    (higher volume at open and close, lower in the middle).

    Returns:
        pd.DataFrame with columns: open, high, low, close, volume
        and a datetime index.
    """
    np.random.seed(seed)

    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    start = pd.Timestamp(f"{date} 09:15:00")
    idx   = pd.date_range(start, periods=n_minutes, freq="1min")

    # Geometric Brownian Motion
    dt    = 1 / (252 * 375)    # fraction of year per 1-minute bar
    sigma = volatility
    mu    = trend

    returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn(n_minutes)
    prices  = spot * np.exp(np.cumsum(returns))
    prices  = np.insert(prices[:-1], 0, spot)

    # OHLC with realistic intrabar spread
    spread = prices * 0.001   # 0.1% spread
    noise  = np.random.rand(n_minutes)

    open_  = prices
    close_ = prices + np.random.randn(n_minutes) * spread * 0.3
    high_  = np.maximum(open_, close_) + noise * spread
    low_   = np.minimum(open_, close_) - noise * spread

    # Volume: U-shaped intraday profile
    t      = np.linspace(0, 1, n_minutes)
    volume_profile = 3 * np.exp(-10 * t) + 1 + 2 * np.exp(-10 * (1 - t))
    volume_profile /= volume_profile.sum()
    total_volume = np.random.randint(5_000_000, 20_000_000)
    volume = (volume_profile * total_volume).astype(int)

    df = pd.DataFrame({
        "open":   open_,
        "high":   high_,
        "low":    low_,
        "close":  close_,
        "volume": volume,
    }, index=idx)

    return df


def generate_multi_day_ohlcv(
    spot:       float,
    n_days:     int   = 20,
    volatility: float = 0.15,
    trend:      float = 0.0,
) -> pd.DataFrame:
    """
    Generate multiple days of 1-minute OHLCV data for backtesting.

    Skips weekends automatically.
    """
    frames = []
    current_date = datetime.now().date() - timedelta(days=n_days + 10)
    day_count    = 0
    current_spot = spot

    while day_count < n_days:
        if current_date.weekday() < 5:   # Monday=0 … Friday=4
            df = generate_intraday_ohlcv(
                spot       = current_spot,
                date       = current_date.strftime("%Y-%m-%d"),
                volatility = volatility,
                trend      = trend,
                seed       = hash(str(current_date)) % (2**31),
            )
            frames.append(df)
            current_spot = df["close"].iloc[-1]   # next day opens at prev close
            day_count   += 1

        current_date += timedelta(days=1)

    return pd.concat(frames).sort_index()


def ohlcv_from_option_snapshots(snapshots: list[dict]) -> pd.DataFrame:
    """
    Convert a list of option price snapshots into a pseudo-OHLCV DataFrame.

    Each snapshot should be:
        {"timestamp": "...", "price": float, "volume": int, "oi": int}

    Useful when you've collected live option prices every few minutes
    and want to run the strategy on the option's own price movement.
    """
    if not snapshots:
        return pd.DataFrame()

    df = pd.DataFrame(snapshots)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()

    # Resample to 1-minute OHLCV
    ohlcv = df["price"].resample("1min").agg(
        open="first", high="max", low="min", close="last"
    ).dropna()
    ohlcv["volume"] = df["volume"].resample("1min").sum().reindex(ohlcv.index, fill_value=0)

    return ohlcv