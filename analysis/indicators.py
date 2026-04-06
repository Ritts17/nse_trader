"""
analysis/indicators.py — Technical Indicators

From Video 2 (Triple Confirmation Strategy):

1. SUPERTREND — Trend following indicator
   - Green signal → Uptrend → BUY
   - Red signal   → Downtrend → SELL
   - Settings: Period=10, Multiplier=3

2. MACD — Moving Average Convergence Divergence
   - Blue line = Fast EMA (12)
   - Red line  = Slow EMA (26)
   - Signal    = 9-period EMA of MACD line
   - Blue crosses Red from BELOW → BUY crossover
   - Red crosses Blue from ABOVE → SELL crossover

3. VWAP — Volume Weighted Average Price (see pcr_vwap.py for option-chain VWAP)
   - The "anchor" line for entries
   - Price near/below VWAP + buy signal  = best long entry
   - Price near/above VWAP + sell signal = best short entry

Triple Confirmation (from video):
  ALL THREE must agree → high-probability trade
  Supertrend says BUY + MACD crosses up + Price near/below VWAP → BUY
  Supertrend says SELL + MACD crosses down + Price near/above VWAP → SELL
"""

import numpy as np
import pandas as pd
from config import (
    SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
)


# ── Supertrend ────────────────────────────────────────────────────────────────

def supertrend(
    df:         pd.DataFrame,
    period:     int   = SUPERTREND_PERIOD,
    multiplier: float = SUPERTREND_MULTIPLIER,
) -> pd.DataFrame:
    """
    Calculate Supertrend indicator.

    Requires columns: high, low, close

    Returns df with added columns:
        supertrend        : actual supertrend line value
        supertrend_dir    : 1 = Uptrend (BUY), -1 = Downtrend (SELL)
        supertrend_signal : "BUY" | "SELL" | "HOLD"
        supertrend_change : True when trend just flipped (signal candle)
    """
    df = df.copy()
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    # True Range
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    # ATR via Wilder's smoothing (EWM)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()

    upper_band = ((high + low) / 2) + multiplier * atr
    lower_band = ((high + low) / 2) - multiplier * atr

    supertrend_vals = [np.nan] * len(df)
    direction       = [1]        * len(df)   # 1=up, -1=down

    for i in range(1, len(df)):
        # Upper band
        if upper_band.iloc[i] < upper_band.iloc[i - 1] or close.iloc[i - 1] > upper_band.iloc[i - 1]:
            final_upper = upper_band.iloc[i]
        else:
            final_upper = upper_band.iloc[i - 1]

        # Lower band
        if lower_band.iloc[i] > lower_band.iloc[i - 1] or close.iloc[i - 1] < lower_band.iloc[i - 1]:
            final_lower = lower_band.iloc[i]
        else:
            final_lower = lower_band.iloc[i - 1]

        upper_band.iloc[i] = final_upper
        lower_band.iloc[i] = final_lower

        # Direction
        prev_dir = direction[i - 1]
        if prev_dir == -1:
            direction[i] = 1 if close.iloc[i] > final_upper else -1
        else:
            direction[i] = -1 if close.iloc[i] < final_lower else 1

        supertrend_vals[i] = final_lower if direction[i] == 1 else final_upper

    df["supertrend"]     = supertrend_vals
    df["supertrend_dir"] = direction

    # Signal
    df["supertrend_signal"] = df["supertrend_dir"].map({1: "BUY", -1: "SELL"})

    # Change detection (trend flip candle — the most important candle)
    df["supertrend_change"] = df["supertrend_dir"].diff().ne(0) & df["supertrend_dir"].notna()
    df.loc[df.index[0], "supertrend_change"] = False

    return df


# ── MACD ─────────────────────────────────────────────────────────────────────

def macd(
    df:     pd.DataFrame,
    fast:   int = MACD_FAST,
    slow:   int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> pd.DataFrame:
    """
    Calculate MACD indicator.

    Requires column: close

    Returns df with added columns:
        macd_line       : Fast EMA - Slow EMA  (blue line in video)
        macd_signal     : Signal EMA            (red line in video)
        macd_histogram  : macd_line - macd_signal
        macd_crossover  : "BUY_CROSS" | "SELL_CROSS" | None
    """
    df = df.copy()
    ema_fast = df["close"].ewm(span=fast,   adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow,   adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig_line  = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - sig_line

    df["macd_line"]      = macd_line
    df["macd_signal"]    = sig_line
    df["macd_histogram"] = histogram

    # Crossover detection
    prev_diff = (macd_line - sig_line).shift(1)
    curr_diff = (macd_line - sig_line)

    df["macd_crossover"] = None
    # Blue crosses red from BELOW (bullish)
    buy_cross  = (prev_diff < 0) & (curr_diff >= 0)
    # Red crosses blue from ABOVE (bearish) → blue crosses red from above = bearish
    sell_cross = (prev_diff > 0) & (curr_diff <= 0)

    df.loc[buy_cross,  "macd_crossover"] = "BUY_CROSS"
    df.loc[sell_cross, "macd_crossover"] = "SELL_CROSS"

    return df


# ── VWAP (OHLCV) ─────────────────────────────────────────────────────────────

def vwap_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate VWAP for OHLCV candle data.

    Requires columns: high, low, close, volume
    Assumes df is sorted by datetime within the same trading day.

    Returns df with added column: vwap
    """
    df = df.copy()
    typical = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (typical * df["volume"]).cumsum()
    cum_vol    = df["volume"].cumsum()
    df["vwap"] = cum_tp_vol / cum_vol
    return df


# ── Triple Confirmation ───────────────────────────────────────────────────────

def triple_confirmation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all three indicators and generate triple-confirmed signals.

    From Video 2:
    BUY  signal: MACD crosses up + Supertrend flips to BUY + Price near/below VWAP
    SELL signal: MACD crosses down + Supertrend flips to SELL + Price near/above VWAP

    The best scenario (from video): "crossover and supertrend signal happen on SAME candle"
    An acceptable scenario: crossover within 1-3 candles of supertrend flip

    Returns df with:
        triple_signal     : "BUY" | "SELL" | None
        signal_quality    : "STRONG" | "MODERATE" | None
        signal_note       : explanation text
        stop_loss         : suggested SL price
        r_factor          : potential R multiple
    """
    # Ensure indicators are computed
    required = ["supertrend_dir", "macd_crossover", "vwap"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing indicator columns: {missing}. "
                         "Run supertrend(), macd(), vwap_ohlcv() first.")

    df = df.copy()
    df["triple_signal"]  = None
    df["signal_quality"] = None
    df["signal_note"]    = None
    df["stop_loss"]      = np.nan
    df["r_factor"]       = np.nan

    for i in range(2, len(df)):
        row         = df.iloc[i]
        close_price = row["close"]
        vwap_price  = row["vwap"]
        st_dir      = row["supertrend_dir"]       # 1=buy, -1=sell
        st_change   = row["supertrend_change"]
        macd_cross  = row["macd_crossover"]

        # Check MACD crossover within last 3 candles (alignment window)
        recent_macd = df["macd_crossover"].iloc[max(0, i-2):i+1]
        has_buy_cross  = (recent_macd == "BUY_CROSS").any()
        has_sell_cross = (recent_macd == "SELL_CROSS").any()

        # ── BUY signal ────────────────────────────────────────────────────
        if st_dir == 1 and has_buy_cross:
            below_vwap = close_price <= vwap_price * 1.003   # within 0.3% above or below
            quality = "STRONG" if (st_change and macd_cross == "BUY_CROSS" and close_price <= vwap_price) \
                      else "MODERATE" if below_vwap else None

            if quality:
                prev_low   = df["low"].iloc[i - 1]
                sl         = round(prev_low * 0.998, 2)     # SL below previous candle low
                risk       = close_price - sl
                target     = close_price + (risk * 3)       # 1:3 R:R
                r_factor   = round((target - close_price) / risk, 2) if risk > 0 else 0

                df.at[df.index[i], "triple_signal"]  = "BUY"
                df.at[df.index[i], "signal_quality"] = quality
                df.at[df.index[i], "signal_note"]    = (
                    f"MACD↑ + Supertrend BUY + Price {'below' if close_price <= vwap_price else 'near'} VWAP"
                )
                df.at[df.index[i], "stop_loss"]  = sl
                df.at[df.index[i], "r_factor"]   = r_factor

        # ── SELL signal ───────────────────────────────────────────────────
        elif st_dir == -1 and has_sell_cross:
            # From video: "if price is above VWAP → excellent sell"
            # "if price near VWAP → acceptable"
            # "if price below VWAP → it will try to go back to VWAP, still valid"
            above_vwap = close_price >= vwap_price          # best case
            near_vwap  = close_price >= vwap_price * 0.997  # within 0.3%
            # Even if below, the ST+MACD double confirmation is strong enough
            valid_sell = above_vwap or near_vwap or (close_price >= vwap_price * 0.990)

            if valid_sell:
                quality = "STRONG"   if (st_change and macd_cross == "SELL_CROSS" and above_vwap) \
                         else "STRONG"   if (st_change and macd_cross == "SELL_CROSS") \
                         else "MODERATE" if near_vwap \
                         else "MODERATE"   # ST+MACD double confirm is enough

                prev_high  = df["high"].iloc[i - 1]
                sl         = round(prev_high * 1.002, 2)
                risk       = sl - close_price
                if risk <= 0:
                    continue
                target     = close_price - (risk * 3)
                r_factor   = round((close_price - target) / risk, 2)

                vwap_note = "above" if above_vwap else "near" if near_vwap else "below (mean-revert risk)"
                df.at[df.index[i], "triple_signal"]  = "SELL"
                df.at[df.index[i], "signal_quality"] = quality
                df.at[df.index[i], "signal_note"]    = (
                    f"MACD↓ + Supertrend SELL + Price {vwap_note} VWAP"
                )
                df.at[df.index[i], "stop_loss"]  = sl
                df.at[df.index[i], "r_factor"]   = r_factor

    return df


# ── Helper: Resample to any timeframe ────────────────────────────────────────

def resample_ohlcv(df: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
    """
    Resample tick or 1-minute OHLCV data to any timeframe.

    Requires: datetime index, columns: open, high, low, close, volume
    """
    rule = f"{timeframe_minutes}min"
    resampled = df.resample(rule).agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()
    return resampled


def apply_all_indicators(df: pd.DataFrame, timeframe_minutes: int = 5) -> pd.DataFrame:
    """
    One-shot: resample + apply Supertrend + MACD + VWAP + Triple Confirmation.
    """
    if timeframe_minutes > 1:
        df = resample_ohlcv(df, timeframe_minutes)

    df = supertrend(df)
    df = macd(df)
    df = vwap_ohlcv(df)
    df = triple_confirmation(df)
    return df