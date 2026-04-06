"""
core/scheduler.py — Live Auto-Refresh Scheduler

Runs the full analysis workflow on a timer, refreshing every N minutes.
Useful for monitoring live market conditions during trading hours.

Usage:
    scheduler = LiveScheduler(
        symbol="NIFTY",
        expiry="07-Apr-2026",
        timeframe_min=15,
        refresh_min=5,
        data_source="yahoo",
        csv_path=None,
        zerodha_config=None,
        upstox_token=None,
        enforce_time=True,
    )
    scheduler.run()
"""

import time
import threading
from datetime import datetime
from typing import Optional, Dict, Any
from core.fetcher import NSEFetcher
from core.broker import get_ohlcv_auto
from analysis.pcr_vwap import calculate_pcr, find_sr_levels, analyse_iv, check_vwap_entry
from analysis.indicators import apply_all_indicators
from signals.generator import SignalGenerator
from utils.display import print_header, print_pcr, print_iv_analysis, print_sr_levels, print_vwap_check, print_breakout, print_signal


class LiveScheduler:
    """
    Auto-refresh scheduler for live market monitoring.

    Runs the analysis function periodically.
    """

    def __init__(self, symbol: str, expiry: str, timeframe_min: int,
                 refresh_min: int, data_source: str,
                 csv_path: Optional[str] = None,
                 zerodha_config: Optional[Dict[str, Any]] = None,
                 upstox_token: Optional[str] = None,
                 enforce_time: bool = True):
        self.symbol = symbol
        self.expiry = expiry
        self.timeframe_min = timeframe_min
        self.refresh_min = refresh_min
        self.data_source = data_source
        self.csv_path = csv_path
        self.zerodha_config = zerodha_config
        self.upstox_token = upstox_token
        self.enforce_time = enforce_time

        self.fetcher = NSEFetcher()
        self.running = False

    def run(self):
        """Run the scheduler (blocking)."""
        print(f"\n  🔄 Starting live scheduler for {self.symbol}")
        print(f"  Refresh every {self.refresh_min} minutes")
        print("  Press Ctrl+C to stop\n")

        self.running = True

        try:
            while self.running:
                if self.enforce_time and datetime.now().hour < 11:
                    print("  ⏰ Market not open yet (waiting for 11 AM)...")
                    time.sleep(60)  # Check every minute
                    continue

                self._run_analysis()
                print(f"\n  ⏰ Next refresh in {self.refresh_min} minutes...")
                time.sleep(self.refresh_min * 60)

        except KeyboardInterrupt:
            print("\n  🛑 Scheduler stopped by user")
            self.running = False

    def _run_analysis(self):
        """Run a single analysis cycle."""
        try:
            # Fetch option chain
            df_ce, df_pe, meta = self.fetcher.get_option_chain(self.symbol, self.expiry)
            spot = meta["underlyingValue"]

            print_header(f"Live Analysis — {self.symbol} | {datetime.now().strftime('%H:%M:%S')}")

            # Option analysis
            pcr_result = calculate_pcr(df_ce, df_pe, spot)
            sr_levels = find_sr_levels(df_ce, df_pe, spot)
            regime = {
                "BULLISH": "bull",
                "EXTREMELY BULLISH": "bull",
                "BEARISH": "bear",
                "EXTREMELY BULLISH": "bear",
            }.get(pcr_result.get("bias", ""), "sideways")
            iv_result = analyse_iv(df_ce, df_pe, spot, regime)

            print_pcr(pcr_result)
            print_iv_analysis(iv_result)
            print_sr_levels(sr_levels, spot)

            # OHLCV
            source_cfg = {
                "source": self.data_source,
                "csv_path": self.csv_path,
                "zerodha_config": self.zerodha_config,
                "upstox_token": self.upstox_token,
            }
            ohlcv_df = get_ohlcv_auto(
                symbol=self.symbol,
                interval_min=self.timeframe_min,
                days=1,  # Just today for live
                **{k: v for k, v in source_cfg.items() if v is not None}
            )
            ohlcv_df = apply_all_indicators(ohlcv_df, self.timeframe_min)

            vwap_now = ohlcv_df["vwap"].iloc[-1]
            direction = "BUY" if pcr_result.get("bias") in ("BULLISH", "EXTREMELY BULLISH") else "SELL"
            vwap_check = check_vwap_entry(spot, vwap_now, direction)
            print_vwap_check(vwap_check)

            breakout = detect_breakout(ohlcv_df, sr_levels)
            print_breakout(breakout)

            # Signal
            gen = SignalGenerator(self.symbol, spot, 7.0)  # dummy TTE
            signal = gen.generate(
                ohlcv_df=ohlcv_df,
                pcr_result=pcr_result,
                iv_result=iv_result,
                sr_levels=sr_levels,
                df_ce=df_ce,
                df_pe=df_pe,
                timeframe=self.timeframe_min,
                check_time=False,
            )
            print_signal(signal)

        except Exception as e:
            print(f"  ⚠️  Analysis error: {e}")