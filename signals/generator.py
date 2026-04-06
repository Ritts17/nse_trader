"""
signals/generator.py — Trade Signal Generator

Combines:
  1. PCR bias (market direction from option chain)
  2. Triple Confirmation (Supertrend + MACD + VWAP from price chart)
  3. IV analysis (whether to buy or sell options)
  4. SR levels (entry, target, stop)
  5. Greeks context (which strike to trade)

Signal output includes:
  - Action    : BUY CE / BUY PE / SELL CE / SELL PE / WAIT
  - Entry     : price zone
  - Stop Loss : previous candle low/high
  - Target    : based on R:R
  - R-Factor  : reward-to-risk
  - Timeframe : the timeframe on which signal fired
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

from config import TRADE_START_HOUR, TRADE_START_MINUTE


class SignalGenerator:
    """
    Merges option-chain analysis with technical signals to produce
    actionable trade recommendations with full context.
    """

    def __init__(self, symbol: str, spot: float, tte_days: float):
        self.symbol   = symbol
        self.spot     = spot
        self.tte_days = tte_days

    def generate(
        self,
        ohlcv_df:    pd.DataFrame,   # price data with indicators applied
        pcr_result:  dict,
        iv_result:   dict,
        sr_levels:   dict,
        df_ce:       pd.DataFrame,
        df_pe:       pd.DataFrame,
        timeframe:   int,            # minutes
        check_time:  bool = True,    # enforce 11:00 AM rule
    ) -> dict:
        """
        Generate a complete trade signal.

        Returns a signal dict with all trade details.
        """
        signal_out = {
            "symbol":     self.symbol,
            "spot":       self.spot,
            "timeframe":  f"{timeframe}min",
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action":     "WAIT",
            "reason":     [],
            "entry":      None,
            "stop_loss":  None,
            "target_1":   None,
            "target_2":   None,
            "r_factor":   None,
            "strike":     None,
            "option_type": None,
            "sr_levels":  sr_levels,
            "pcr":        pcr_result,
            "iv":         iv_result,
            "quality":    None,
        }

        # ── Time Filter (from video: "never trade before 11 AM") ──────────
        if check_time:
            now = datetime.now()
            if now.hour < TRADE_START_HOUR or (now.hour == TRADE_START_HOUR and now.minute < TRADE_START_MINUTE):
                signal_out["reason"].append(
                    f"⏰ Time filter: No trading before {TRADE_START_HOUR:02d}:{TRADE_START_MINUTE:02d} — data still forming"
                )
                return signal_out

        # ── Get latest triple signal from price chart ──────────────────────
        chart_signals = ohlcv_df[ohlcv_df["triple_signal"].notna()]
        if chart_signals.empty:
            signal_out["reason"].append("📊 No triple-confirmation signal on chart yet")
            return signal_out

        latest = chart_signals.iloc[-1]
        chart_action  = latest["triple_signal"]     # BUY or SELL
        chart_quality = latest["signal_quality"]
        chart_note    = latest["signal_note"]
        chart_sl      = latest["stop_loss"]
        chart_rf      = latest["r_factor"]
        close_price   = latest["close"]

        # ── PCR Confirmation ──────────────────────────────────────────────
        pcr_bias = pcr_result.get("bias", "UNKNOWN")
        pcr_val  = pcr_result.get("pcr")

        pcr_ok_buy  = pcr_bias in ("BULLISH", "EXTREMELY BULLISH", "NEUTRAL/SIDEWAYS")
        pcr_ok_sell = pcr_bias in ("BEARISH", "EXTREMELY BEARISH", "MILDLY BEARISH", "NEUTRAL/SIDEWAYS")

        # ── IV Guidance ───────────────────────────────────────────────────
        iv_level = iv_result.get("iv_level", "MODERATE")
        avg_iv   = iv_result.get("avg_iv", 15)

        # ── Combine signals ───────────────────────────────────────────────
        if chart_action == "BUY" and pcr_ok_buy:
            # If IV is high, DON'T buy calls (from video: "never buy calls when IV is high")
            if iv_level == "HIGH":
                signal_out["action"] = "WAIT"
                signal_out["reason"].append(
                    f"⚠️  BUY signal confirmed BUT IV is HIGH ({avg_iv:.1f}%) — "
                    "avoid buying CE; consider selling PE instead"
                )
                # Recommend selling PE as alternative
                signal_out = self._add_sell_pe_context(signal_out, df_pe, sr_levels, close_price, chart_sl)
            else:
                signal_out["action"]      = "BUY CE"
                signal_out["option_type"] = "CE"
                signal_out["entry"]       = close_price
                signal_out["stop_loss"]   = chart_sl
                signal_out["quality"]     = chart_quality
                signal_out["r_factor"]    = chart_rf
                signal_out["reason"].append(f"✅ {chart_note}")
                signal_out["reason"].append(f"✅ PCR = {pcr_val:.2f} ({pcr_bias}) confirms BUY")
                signal_out["reason"].append(f"✅ IV = {avg_iv:.1f}% ({iv_level}) — OK to buy")
                signal_out = self._add_strike_targets(signal_out, df_ce, "CE", close_price, chart_sl)

        elif chart_action == "SELL" and pcr_ok_sell:
            if iv_level == "HIGH":
                # High IV + sell signal → excellent to sell calls
                signal_out["action"]      = "SELL CE"
                signal_out["option_type"] = "CE"
                signal_out["quality"]     = chart_quality
                signal_out["reason"].append(f"✅ {chart_note}")
                signal_out["reason"].append(f"✅ PCR = {pcr_val:.2f} ({pcr_bias}) confirms SELL")
                signal_out["reason"].append(f"✅ High IV ({avg_iv:.1f}%) → excellent to SELL CE (premium inflated)")
                signal_out = self._add_strike_targets(signal_out, df_ce, "CE", close_price, chart_sl, is_sell=True)
            else:
                signal_out["action"]      = "BUY PE"
                signal_out["option_type"] = "PE"
                signal_out["entry"]       = close_price
                signal_out["stop_loss"]   = chart_sl
                signal_out["quality"]     = chart_quality
                signal_out["r_factor"]    = chart_rf
                signal_out["reason"].append(f"✅ {chart_note}")
                signal_out["reason"].append(f"✅ PCR = {pcr_val:.2f} ({pcr_bias}) confirms SELL")
                signal_out["reason"].append(f"✅ IV = {avg_iv:.1f}% ({iv_level}) — OK to buy PE")
                signal_out = self._add_strike_targets(signal_out, df_pe, "PE", close_price, chart_sl)

        elif chart_action == "BUY" and not pcr_ok_buy:
            signal_out["reason"].append(
                f"⚠️  Chart says BUY but PCR = {pcr_val:.2f} ({pcr_bias}) — conflicting signal, SKIP"
            )

        elif chart_action == "SELL" and not pcr_ok_sell:
            signal_out["reason"].append(
                f"⚠️  Chart says SELL but PCR = {pcr_val:.2f} ({pcr_bias}) — conflicting signal, SKIP"
            )

        return signal_out

    def _add_strike_targets(
        self, sig: dict, df_option: pd.DataFrame,
        opt_type: str, entry: float, sl: float,
        is_sell: bool = False,
    ) -> dict:
        """Choose best strike and calculate targets."""
        if df_option.empty:
            return sig

        atm = min(df_option["strikePrice"].unique(), key=lambda x: abs(x - self.spot))
        sig["strike"] = atm

        # Target based on SR levels
        sr = sig.get("sr_levels", {})
        if opt_type == "CE" and not is_sell:
            t1 = sr.get("primary_resistance") or (entry + abs(entry - sl) * 2)
            t2 = (sr.get("resistance_levels") or [t1])[-1]
        else:
            t1 = sr.get("primary_support") or (entry - abs(entry - sl) * 2)
            t2 = (sr.get("support_levels") or [t1])[-1]

        sig["target_1"] = round(t1, 2)
        sig["target_2"] = round(t2, 2)

        risk = abs(entry - sl)
        if risk > 0:
            reward = abs(t1 - entry)
            sig["r_factor"] = round(reward / risk, 2)

        return sig

    def _add_sell_pe_context(
        self, sig: dict, df_pe: pd.DataFrame,
        sr_levels: dict, entry: float, sl: float,
    ) -> dict:
        """When IV is high and we get buy signal, suggest selling OTM PE instead."""
        if df_pe.empty:
            return sig

        support = sr_levels.get("primary_support")
        if support:
            otm_pe = df_pe[df_pe["strikePrice"] <= support]
            if not otm_pe.empty:
                sell_strike = otm_pe.iloc[-1]["strikePrice"]
                premium     = otm_pe.iloc[-1]["lastPrice"]
                sig["reason"].append(
                    f"💡 Alternative: SELL PE at {sell_strike} strike (premium ₹{premium:.2f}) — "
                    "collect inflated IV premium, PE expires worthless if market holds above support"
                )
        return sig


# ── Breakout Detection ────────────────────────────────────────────────────────

def detect_breakout(
    df: pd.DataFrame,
    sr_levels: dict,
    lookback: int = 5,
) -> dict:
    """
    Detect breakouts above resistance or below support.

    A breakout is confirmed when price closes above/below a key level
    and the previous candle also showed momentum in that direction.
    """
    if df.empty or not sr_levels:
        return {"breakout": None}

    latest_close = df["close"].iloc[-1]
    latest_vol   = df.get("volume", pd.Series([0])).iloc[-1]
    avg_vol      = df.get("volume", pd.Series([0])).iloc[-lookback:].mean()

    results = {
        "breakout":        None,
        "level":           None,
        "price":           latest_close,
        "volume_confirm":  latest_vol > avg_vol * 1.2 if avg_vol > 0 else False,
        "note":            "",
    }

    resistance = sr_levels.get("primary_resistance")
    support    = sr_levels.get("primary_support")

    if resistance and latest_close > resistance:
        prev_close = df["close"].iloc[-2]
        if prev_close < resistance:   # candle closed above for the first time
            results["breakout"] = "BULLISH_BREAKOUT"
            results["level"]    = resistance
            results["note"]     = (
                f"🚀 Breakout above resistance {resistance}! "
                f"Volume {'confirms' if results['volume_confirm'] else 'weak — watch'}"
            )

    elif support and latest_close < support:
        prev_close = df["close"].iloc[-2]
        if prev_close > support:
            results["breakout"] = "BEARISH_BREAKDOWN"
            results["level"]    = support
            results["note"]     = (
                f"📉 Breakdown below support {support}! "
                f"Volume {'confirms' if results['volume_confirm'] else 'weak — watch'}"
            )

    return results