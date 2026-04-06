"""
core/greeks.py — Option Greeks via Black-Scholes

Greeks explained (Video 3 — Pushkar Raj Thakur):
  Delta  : How much premium changes per 1-point move in underlying
           ATM ≈ 0.5, Deep ITM → 1, Deep OTM → 0
  Gamma  : Rate of change of Delta (how fast Delta moves)
  Theta  : Time decay — how much premium erodes per day
           Favours sellers; accelerates near expiry
  Vega   : Sensitivity to Implied Volatility change
           High IV → buy options; Low IV → sell options
  Rho    : Interest rate sensitivity (negligible in Indian markets)
"""

import math
from dataclasses import dataclass
from typing import Literal
import numpy as np
from scipy.stats import norm


@dataclass
class Greeks:
    delta: float
    gamma: float
    theta: float   # per calendar day
    vega:  float   # per 1% change in IV
    rho:   float
    iv:    float   # input IV used


def black_scholes_greeks(
    option_type:  Literal["CE", "PE"],
    spot:         float,   # underlying spot price
    strike:       float,   # option strike
    tte_days:     float,   # time to expiry in calendar days
    iv:           float,   # implied volatility as percentage (e.g. 20 for 20%)
    risk_free:    float = 6.5,   # Indian repo rate %
) -> Greeks:
    """
    Compute full option Greeks using Black-Scholes-Merton model.

    Args:
        option_type : "CE" or "PE"
        spot        : current price of underlying (e.g. 22713)
        strike      : option strike price (e.g. 22700)
        tte_days    : calendar days to expiry (e.g. 2)
        iv          : implied volatility in % (e.g. 15.5)
        risk_free   : risk-free rate in % (Indian repo rate ≈ 6.5%)

    Returns:
        Greeks dataclass
    """
    if tte_days <= 0:
        return Greeks(delta=0, gamma=0, theta=0, vega=0, rho=0, iv=iv)

    T  = tte_days / 365.0        # years
    r  = risk_free / 100.0
    σ  = iv / 100.0              # convert % to decimal

    if σ <= 0 or spot <= 0 or strike <= 0:
        return Greeks(delta=0, gamma=0, theta=0, vega=0, rho=0, iv=iv)

    d1 = (math.log(spot / strike) + (r + 0.5 * σ**2) * T) / (σ * math.sqrt(T))
    d2 = d1 - σ * math.sqrt(T)

    N   = norm.cdf
    n   = norm.pdf

    # ── Delta ─────────────────────────────────────────────────────────────────
    if option_type == "CE":
        delta = N(d1)
    else:
        delta = N(d1) - 1          # negative for puts

    # ── Gamma (same for CE and PE) ─────────────────────────────────────────
    gamma = n(d1) / (spot * σ * math.sqrt(T))

    # ── Theta (per calendar day, in Rs) ──────────────────────────────────
    # BSM theta formula is annualised; divide by 365 for per-day decay.
    # For ATM Nifty ~22000 with 7 DTE and 15% IV this should be ~-10 to -30 per day
    _theta_base = -(spot * n(d1) * σ) / (2 * math.sqrt(T))
    if option_type == "CE":
        theta = (_theta_base - r * strike * math.exp(-r * T) * N(d2)) / 365
    else:
        theta = (_theta_base + r * strike * math.exp(-r * T) * N(-d2)) / 365

    # ── Vega (per 1% change in IV) ─────────────────────────────────────────
    vega = spot * n(d1) * math.sqrt(T) / 100   # divide by 100 for 1% move

    # ── Rho (per 1% change in interest rate) ──────────────────────────────
    if option_type == "CE":
        rho = strike * T * math.exp(-r * T) * N(d2) / 100
    else:
        rho = -strike * T * math.exp(-r * T) * N(-d2) / 100

    return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho, iv=iv)


def enrich_chain_with_greeks(
    df,
    option_type: Literal["CE", "PE"],
    spot:        float,
    tte_days:    float,
    risk_free:   float = 6.5,
):
    """
    Add Delta, Gamma, Theta, Vega, Rho columns to an option-chain DataFrame.

    Args:
        df          : CE or PE DataFrame from NSEFetcher
        option_type : "CE" or "PE"
        spot        : underlying spot price
        tte_days    : calendar days to expiry
        risk_free   : risk-free rate %
    """
    if df.empty:
        return df

    greeks_records = []
    for _, row in df.iterrows():
        g = black_scholes_greeks(
            option_type = option_type,
            spot        = spot,
            strike      = row["strikePrice"],
            tte_days    = tte_days,
            iv          = row["impliedVolatility"] if row["impliedVolatility"] > 0 else 15.0,
            risk_free   = risk_free,
        )
        greeks_records.append({
            "delta": round(g.delta, 4),
            "gamma": round(g.gamma, 6),
            "theta": round(g.theta, 4),
            "vega":  round(g.vega,  4),
            "rho":   round(g.rho,   4),
        })

    import pandas as pd
    greeks_df = pd.DataFrame(greeks_records)
    return pd.concat([df.reset_index(drop=True), greeks_df], axis=1)


def interpret_greeks(greeks: Greeks, option_type: str) -> dict:
    """
    Return human-readable interpretation of Greeks for a single option.
    Useful for the signal display.
    """
    abs_delta = abs(greeks.delta)

    if abs_delta > 0.7:
        moneyness = "Deep ITM — high delta, moves almost like underlying"
    elif abs_delta > 0.45:
        moneyness = "ATM — delta ~0.5, balanced risk/reward"
    elif abs_delta > 0.2:
        moneyness = "OTM — lower delta, cheaper but slower premium gain"
    else:
        moneyness = "Deep OTM — very low delta, lottery ticket territory"

    theta_daily_cost = abs(greeks.theta)

    return {
        "moneyness":   moneyness,
        "delta_note":  f"₹{abs_delta:.2f} premium change per 1-pt move in underlying",
        "theta_note":  f"Loses ₹{theta_daily_cost:.2f}/day to time decay (Theta)",
        "vega_note":   f"₹{greeks.vega:.2f} gain/loss per 1% IV change (Vega)",
        "iv_note":     f"IV = {greeks.iv:.1f}% — {'HIGH (consider selling)' if greeks.iv > 30 else 'MODERATE' if greeks.iv > 15 else 'LOW (consider buying)'}",
    }