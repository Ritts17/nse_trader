"""
analysis/pcr_vwap.py — PCR & VWAP Analysis

From Video 2 (Triple Confirmation Strategy) and Video 4 (Option Chain):

PCR (Put-Call Ratio):
  - Formula: Sum of ΔOI (Put side, 8 strikes) / Sum of ΔOI (Call side, 8 strikes)
  - PCR > 1.5  → Bullish
  - PCR < 1.0  → Bearish
  - PCR > 4.0  → Extremely Bullish
  - PCR < 0.2  → Extremely Bearish
  - PCR ~1     → Sideways / Neutral
  - Use CHANGE in OI (not absolute OI) for direction
  - Look at 8 strike prices above and below ATM only

VWAP (Volume Weighted Average Price):
  - From video: "Until price comes near VWAP, do not take entry"
  - Entry band: ±0.3% of VWAP
  - Price below VWAP + bullish signal → best buy entry
  - Price above VWAP + bearish signal → best sell entry

Range Detection (OI-based, from Video 4):
  - Highest Call OI strike = Resistance
  - Highest Put OI strike  = Support
  - Sellers defend these levels with large capital
"""

import pandas as pd
import numpy as np
from typing import Optional
from config import (
    PCR_STRIKE_RANGE, PCR_EXTREME_BULL, PCR_BULL,
    PCR_NEUTRAL_HIGH, PCR_NEUTRAL_LOW, PCR_BEAR, PCR_EXTREME_BEAR,
    VWAP_ENTRY_BAND_PCT,
)


# ── PCR Calculation ───────────────────────────────────────────────────────────

def calculate_pcr(
    df_ce:   pd.DataFrame,
    df_pe:   pd.DataFrame,
    spot:    float,
    n_strikes: int = PCR_STRIKE_RANGE,
) -> dict:
    """
    Calculate Put-Call Ratio using Change in Open Interest
    for ±n_strikes strikes around ATM.

    Returns dict with PCR value and market bias.
    """
    if df_ce.empty or df_pe.empty:
        return {"pcr": None, "bias": "UNKNOWN", "signal": "NO DATA"}

    strikes = sorted(df_ce["strikePrice"].unique())

    # Find ATM strike (closest to spot)
    atm_strike = min(strikes, key=lambda x: abs(x - spot))
    atm_idx    = strikes.index(atm_strike)

    lo = max(0, atm_idx - n_strikes)
    hi = min(len(strikes) - 1, atm_idx + n_strikes)
    relevant_strikes = strikes[lo:hi + 1]

    df_ce_rel = df_ce[df_ce["strikePrice"].isin(relevant_strikes)]
    df_pe_rel = df_pe[df_pe["strikePrice"].isin(relevant_strikes)]

    sum_ce_doi = df_ce_rel["changeInOI"].sum()
    sum_pe_doi = df_pe_rel["changeInOI"].sum()

    if sum_ce_doi == 0:
        pcr = None
        bias, signal, color = "UNKNOWN", "INSUFFICIENT DATA", "grey"
    else:
        pcr = round(sum_pe_doi / sum_ce_doi, 3)
        bias, signal, color = _interpret_pcr(pcr)

    return {
        "pcr":              pcr,
        "atm_strike":       atm_strike,
        "sum_pe_delta_oi":  sum_pe_doi,
        "sum_ce_delta_oi":  sum_ce_doi,
        "bias":             bias,
        "signal":           signal,
        "color":            color,
        "strikes_used":     len(relevant_strikes),
    }


def _interpret_pcr(pcr: float) -> tuple[str, str, str]:
    """Return (bias, signal_text, color) based on PCR value."""
    if pcr >= PCR_EXTREME_BULL:
        return "EXTREMELY BULLISH", f"PCR {pcr:.2f} → Strong Buy Signal — Call options favoured", "bright_green"
    elif pcr >= PCR_BULL:
        return "BULLISH",           f"PCR {pcr:.2f} → Buy Signal — Market likely to rise",          "green"
    elif pcr >= PCR_NEUTRAL_LOW:
        return "NEUTRAL/SIDEWAYS",  f"PCR {pcr:.2f} → No clear direction — Avoid naked options",    "yellow"
    elif pcr >= PCR_BEAR:
        return "MILDLY BEARISH",    f"PCR {pcr:.2f} → Caution — Consider Put options",               "red"
    elif pcr >= PCR_EXTREME_BEAR:
        return "BEARISH",           f"PCR {pcr:.2f} → Sell Signal — Market likely to fall",          "red"
    else:
        return "EXTREMELY BEARISH", f"PCR {pcr:.2f} → Strong Sell — Heavy Put buying",               "bright_red"


# ── OI-Based Support & Resistance ─────────────────────────────────────────────

def find_sr_levels(
    df_ce: pd.DataFrame,
    df_pe: pd.DataFrame,
    spot:  float,
    top_n: int = 2,
) -> dict:
    """
    Find Support and Resistance from Open Interest data.

    Resistance = Strike with highest Call OI (sellers don't want price above)
    Support     = Strike with highest Put OI  (sellers don't want price below)

    From Video 4: "Sellers defend their position — market won't cross their level"
    """
    if df_ce.empty or df_pe.empty:
        return {}

    # Filter to strikes within a reasonable range of spot (± 10%)
    band = spot * 0.10
    ce_near = df_ce[(df_ce["strikePrice"] >= spot) &
                    (df_ce["strikePrice"] <= spot + band)]
    pe_near = df_pe[(df_pe["strikePrice"] <= spot) &
                    (df_pe["strikePrice"] >= spot - band)]

    resistances = []
    supports    = []

    if not ce_near.empty:
        top_ce = ce_near.nlargest(top_n, "openInterest")
        resistances = sorted(top_ce["strikePrice"].tolist())

    if not pe_near.empty:
        top_pe = pe_near.nlargest(top_n, "openInterest")
        supports = sorted(top_pe["strikePrice"].tolist(), reverse=True)

    return {
        "resistance_levels": resistances,
        "support_levels":    supports,
        "primary_resistance": resistances[0] if resistances else None,
        "primary_support":    supports[0]    if supports    else None,
    }


# ── VWAP Calculation ──────────────────────────────────────────────────────────

def calculate_vwap(
    df: pd.DataFrame,
    price_col:  str = "close",
    volume_col: str = "volume",
) -> pd.Series:
    """
    Calculate intraday VWAP from OHLCV data.

    Formula: Σ(Typical Price × Volume) / Σ(Volume)
    Typical Price = (High + Low + Close) / 3

    Note: VWAP resets each trading day.
    """
    if "high" in df.columns and "low" in df.columns:
        typical = (df["high"] + df["low"] + df[price_col]) / 3
    else:
        typical = df[price_col]

    cum_tp_vol = (typical * df[volume_col]).cumsum()
    cum_vol    = df[volume_col].cumsum()
    vwap       = cum_tp_vol / cum_vol
    return vwap


def check_vwap_entry(
    current_price: float,
    vwap:          float,
    signal_type:   str,    # "BUY" or "SELL"
    band_pct:      float = VWAP_ENTRY_BAND_PCT,
) -> dict:
    """
    Check if current price is in a valid VWAP entry zone.

    From Video 2: "Entry only when price is near VWAP"
    From Video 4: "Price below VWAP + bullish → best buy entry"

    Returns:
        valid_entry: bool
        note:        explanation string
        distance_pct: how far price is from VWAP (%)
    """
    distance_pct = (current_price - vwap) / vwap

    upper_band = vwap * (1 + band_pct)
    lower_band = vwap * (1 - band_pct)
    near_vwap  = lower_band <= current_price <= upper_band

    if signal_type == "BUY":
        # Best buy: price at or below VWAP (price will try to rise to VWAP)
        if current_price <= vwap:
            valid   = True
            quality = "EXCELLENT — Price below VWAP, expects upward mean-reversion"
        elif near_vwap:
            valid   = True
            quality = "GOOD — Price near VWAP, entry acceptable"
        else:
            valid   = False
            quality = f"SKIP — Price {abs(distance_pct)*100:.2f}% above VWAP, too far for safe entry"

    else:  # SELL
        # Best sell: price at or above VWAP (price will try to fall to VWAP)
        if current_price >= vwap:
            valid   = True
            quality = "EXCELLENT — Price above VWAP, expects downward mean-reversion"
        elif near_vwap:
            valid   = True
            quality = "GOOD — Price near VWAP, entry acceptable"
        else:
            valid   = False
            quality = f"SKIP — Price {abs(distance_pct)*100:.2f}% below VWAP, too far for safe entry"

    return {
        "valid_entry":   valid,
        "quality":       quality,
        "vwap":          round(vwap, 2),
        "distance_pct":  round(distance_pct * 100, 3),
        "upper_band":    round(upper_band, 2),
        "lower_band":    round(lower_band, 2),
    }


# ── IV Analysis ───────────────────────────────────────────────────────────────

def analyse_iv(
    df_ce:       pd.DataFrame,
    df_pe:       pd.DataFrame,
    spot:        float,
    market_regime: str = "sideways",   # "bull", "bear", "sideways"
) -> dict:
    """
    Analyse Implied Volatility at ATM strikes.

    From Video 4 (Pushkar):
      Bull market:    IV > 21 = high, IV < 15 = low
      Bear market:    IV > 38 = high, IV < 21 = low
      Sideways:       IV > 20 = high, IV < 12 = low
    High IV  → smart to SELL options (premium inflated)
    Low IV   → smart to BUY options  (premium cheap)
    """
    thresholds = {
        "bull":     {"high": 21, "low": 15},
        "bear":     {"high": 38, "low": 21},
        "sideways": {"high": 20, "low": 12},
    }.get(market_regime.lower(), {"high": 20, "low": 12})

    # Find ATM strike
    strikes = sorted(df_ce["strikePrice"].unique())
    atm     = min(strikes, key=lambda x: abs(x - spot))

    ce_atm = df_ce[df_ce["strikePrice"] == atm]
    pe_atm = df_pe[df_pe["strikePrice"] == atm]

    iv_ce = ce_atm["impliedVolatility"].values[0] if not ce_atm.empty else 0
    iv_pe = pe_atm["impliedVolatility"].values[0] if not pe_atm.empty else 0
    avg_iv = (iv_ce + iv_pe) / 2 if (iv_ce and iv_pe) else max(iv_ce, iv_pe)

    if avg_iv >= thresholds["high"]:
        iv_level    = "HIGH"
        action      = "Consider SELLING options — premium is inflated"
        dont_do     = "Do NOT buy Call options when IV is high"
    elif avg_iv <= thresholds["low"]:
        iv_level    = "LOW"
        action      = "Consider BUYING options — premium is cheap"
        dont_do     = "Avoid selling — limited premium collection"
    else:
        iv_level    = "MODERATE"
        action      = "Normal trading conditions"
        dont_do     = "No specific IV caution"

    return {
        "atm_strike":    atm,
        "iv_ce":         iv_ce,
        "iv_pe":         iv_pe,
        "avg_iv":        round(avg_iv, 2),
        "iv_level":      iv_level,
        "action":        action,
        "caution":       dont_do,
        "thresholds":    thresholds,
        "market_regime": market_regime,
    }