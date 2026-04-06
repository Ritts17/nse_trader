"""
main.py — NSE Option Trader — Main Orchestrator  (v2)

Usage:
    python main.py

Menu:
  1. Analyse stock option    — full live option chain + signals
  2. Analyse index option    — NIFTY / BANKNIFTY etc.
  3. Backtest               — find best timeframe & strategy
  4. Live scheduler          — auto-refresh every N minutes
  5. Trade journal           — log trades & view performance
  6. Exit

Architecture:
  core/fetcher.py     → NSE session + cookie handling
  core/greeks.py      → Black-Scholes Greeks
  core/broker.py      → OHLCV data (CSV / Zerodha / Upstox / Yahoo / Synthetic)
  core/scheduler.py   → auto-refresh live loop
  analysis/indicators.py → Supertrend + MACD + VWAP + Triple Confirmation
  analysis/pcr_vwap.py   → PCR, IV, SR from option chain
  analysis/pcr_tracker.py → session-level PCR trend tracking
  signals/generator.py    → combined BUY/SELL/WAIT signal
  backtest/engine.py      → multi-timeframe backtester
  utils/journal.py        → trade log + performance report
  utils/display.py        → coloured terminal output
  utils/sample_data.py    → synthetic OHLCV generator
"""

import sys
from datetime import datetime, date

from core.fetcher         import NSEFetcher
from core.broker          import get_ohlcv_auto
from core.greeks          import enrich_chain_with_greeks
from core.scheduler       import LiveScheduler
from analysis.pcr_vwap    import calculate_pcr, find_sr_levels, analyse_iv, check_vwap_entry
from analysis.indicators  import apply_all_indicators
from analysis.pcr_tracker import PCRTracker, print_pcr_session
from signals.generator    import SignalGenerator, detect_breakout
from backtest.engine      import Backtester
from utils.journal        import TradeJournal
from utils.display        import (
    print_header, print_section, print_signal, print_pcr,
    print_iv_analysis, print_sr_levels, print_greeks_table,
    print_vwap_check, print_breakout,
)
from utils.sample_data    import generate_multi_day_ohlcv
from config               import TIMEFRAMES, INDEX_SYMBOLS


# ─────────────────────────────────────────────────────────────────────────────
#  Helper prompts
# ─────────────────────────────────────────────────────────────────────────────

def prompt(msg: str, default: str = "") -> str:
    val = input(f"\n  {msg}" + (f" [{default}]" if default else "") + ": ").strip()
    return val if val else default

def prompt_int(msg: str, choices: list, default: int = None) -> int:
    choice_str = "/".join(str(c) for c in choices)
    while True:
        val = prompt(f"{msg} ({choice_str})", str(default) if default else "")
        try:
            v = int(val)
            if v in choices:
                return v
        except ValueError:
            pass
        print(f"  Please choose from: {choice_str}")

def get_tte(expiry_str: str) -> float:
    for fmt in ("%d-%b-%Y", "%d-%m-%Y", "%d-%b-%y"):
        try:
            expiry = datetime.strptime(expiry_str, fmt).date()
            return max(1.0, (expiry - date.today()).days)
        except ValueError:
            continue
    return 7.0


# ─────────────────────────────────────────────────────────────────────────────
#  Data source selector
# ─────────────────────────────────────────────────────────────────────────────

def prompt_data_source() -> dict:
    print("\n  Select OHLCV data source:")
    print("    1. Yahoo Finance (free, no setup needed)")
    print("    2. CSV file      (any broker export)")
    print("    3. Zerodha       (requires kiteconnect + API key)")
    print("    4. Upstox        (requires access token)")
    print("    5. Synthetic     (demo/offline)")

    choice = prompt("Source", "1")

    if choice == "1":
        return {"source": "yahoo"}
    elif choice == "2":
        path = prompt("CSV file path", "data/NIFTY_5min.csv")
        return {"source": "csv", "csv_path": path}
    elif choice == "3":
        api_key       = prompt("Zerodha API key")
        access_token  = prompt("Zerodha access token")
        tradingsymbol = prompt("Trading symbol (e.g. NIFTY 50)", "NIFTY 50")
        exchange      = prompt("Exchange (NSE/NFO)", "NSE")
        return {
            "source": "zerodha",
            "zerodha_config": {
                "api_key":       api_key,
                "access_token":  access_token,
                "tradingsymbol": tradingsymbol,
                "exchange":      exchange,
            }
        }
    elif choice == "4":
        token = prompt("Upstox access token")
        return {"source": "upstox", "upstox_token": token}
    else:
        return {"source": "synthetic"}


# ─────────────────────────────────────────────────────────────────────────────
#  Full analysis workflow
# ─────────────────────────────────────────────────────────────────────────────

def analyse_symbol(fetcher: NSEFetcher, symbol: str, expiry: str,
                   timeframe: int, source_cfg: dict):

    print_header(f"Analysing — {symbol} | Expiry: {expiry} | {timeframe}min")

    try:
        df_ce, df_pe, meta = fetcher.get_option_chain(symbol, expiry)
    except Exception as e:
        print(f"\n  ✗ Option chain fetch failed: {e}")
        print("  → Running in demo mode with synthetic data.\n")
        _demo_mode(symbol, expiry, timeframe)
        return

    spot     = meta["underlyingValue"]
    tte_days = get_tte(expiry)

    print(f"\n  Symbol    : {symbol}")
    print(f"  Spot      : ₹{spot:,.2f}")
    print(f"  Timestamp : {meta['timestamp']}")
    print(f"  TTE       : {tte_days:.0f} days")

    # Greeks
    print("\n  Computing Greeks...")
    df_ce = enrich_chain_with_greeks(df_ce, "CE", spot, tte_days)
    df_pe = enrich_chain_with_greeks(df_pe, "PE", spot, tte_days)

    # Option chain analysis
    pcr_result = calculate_pcr(df_ce, df_pe, spot)
    sr_levels  = find_sr_levels(df_ce, df_pe, spot)
    regime     = {
        "BULLISH":           "bull",
        "EXTREMELY BULLISH": "bull",
        "BEARISH":           "bear",
        "EXTREMELY BEARISH": "bear",
    }.get(pcr_result.get("bias", ""), "sideways")
    iv_result  = analyse_iv(df_ce, df_pe, spot, regime)

    print_pcr(pcr_result)
    print_iv_analysis(iv_result)
    print_sr_levels(sr_levels, spot)
    print_greeks_table(df_ce, df_pe, spot)

    # OHLCV + indicators
    print_section(f"Technical Analysis ({timeframe}-min)")
    ohlcv_df = get_ohlcv_auto(
        symbol         = symbol,
        interval_min   = timeframe,
        days           = 10,
        csv_path       = source_cfg.get("csv_path"),
        zerodha_config = source_cfg.get("zerodha_config"),
        upstox_token   = source_cfg.get("upstox_token"),
    )
    ohlcv_df = apply_all_indicators(ohlcv_df, timeframe)

    vwap_now   = ohlcv_df["vwap"].iloc[-1]
    direction  = "BUY" if pcr_result.get("bias") in ("BULLISH", "EXTREMELY BULLISH") else "SELL"
    vwap_check = check_vwap_entry(spot, vwap_now, direction)
    print_vwap_check(vwap_check)

    breakout = detect_breakout(ohlcv_df, sr_levels)
    print_breakout(breakout)

    gen    = SignalGenerator(symbol, spot, tte_days)
    signal = gen.generate(
        ohlcv_df   = ohlcv_df,
        pcr_result = pcr_result,
        iv_result  = iv_result,
        sr_levels  = sr_levels,
        df_ce      = df_ce,
        df_pe      = df_pe,
        timeframe  = timeframe,
        check_time = False,
    )
    print_signal(signal)

    # Offer to journal the trade
    if signal.get("action") != "WAIT":
        log_it = prompt("Log this trade to journal? (y/n)", "n").lower()
        if log_it == "y":
            journal = TradeJournal()
            journal.log_entry(signal, pcr_result, iv_result, vwap=vwap_now)
            print("  Trade logged. Use 'Trade Journal' menu to view/close it.")


def _demo_mode(symbol: str, expiry: str, timeframe: int):
    spot     = 22000 if symbol in INDEX_SYMBOLS else 400
    ohlcv_df = generate_multi_day_ohlcv(spot=spot, n_days=10)
    ohlcv_df = apply_all_indicators(ohlcv_df, timeframe)
    signals  = ohlcv_df[ohlcv_df["triple_signal"].notna()]
    print_header(f"DEMO MODE — {symbol}")
    print(f"  Synthetic spot: ₹{spot:,.2f}")
    if not signals.empty:
        last = signals.iloc[-1]
        print(f"  Latest signal : {last['triple_signal']} ({last['signal_quality']})")
        print(f"  Note          : {last['signal_note']}")
        print(f"  Stop Loss     : ₹{last['stop_loss']:.2f}")
        print(f"  R-Factor      : {last['r_factor']:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
#  Backtest
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest():
    symbol  = prompt("Symbol", "NIFTY").upper()
    n_days  = int(prompt("Days of history", "30"))
    vol     = float(prompt("Annual volatility % (e.g. 20)", "20")) / 100
    spot    = 22000 if symbol in INDEX_SYMBOLS else 400

    print_header(f"BACKTESTING — {symbol} | {n_days} days")
    df_1min = generate_multi_day_ohlcv(spot=spot, n_days=n_days, volatility=vol)
    print(f"  Total 1-min candles: {len(df_1min):,}")

    bt      = Backtester(df_1min, timeframes=TIMEFRAMES)
    results = bt.run_all_timeframes()
    print("\n" + bt.generate_report(results))

    best_tf, entry_tf = bt.best_strategy(results)
    print(f"\n  ► Watch {best_tf}min chart for signal")
    print(f"  ► Drop to {entry_tf}min for precise VWAP entry")
    print(f"  ► Always confirm with PCR from option chain")


# ─────────────────────────────────────────────────────────────────────────────
#  Live scheduler
# ─────────────────────────────────────────────────────────────────────────────

def run_scheduler():
    print_header("LIVE SCHEDULER SETUP")
    symbol   = prompt("Symbol", "NIFTY").upper()
    fetcher  = NSEFetcher()
    fetcher._load_cookies()
    expiries = fetcher.get_expiry_dates(symbol)
    if expiries:
        print(f"  Expiries: {', '.join(expiries[:8])}")
        expiry = prompt("Expiry", expiries[0])
    else:
        expiry = prompt("Expiry", "07-Apr-2026")

    timeframe  = prompt_int("Timeframe (min)", TIMEFRAMES, 15)
    refresh    = int(prompt("Refresh every N minutes", "5"))
    enforce    = prompt("Enforce 11 AM rule? (y/n)", "y").lower() == "y"
    source_cfg = prompt_data_source()

    LiveScheduler(
        symbol         = symbol,
        expiry         = expiry,
        timeframe_min  = timeframe,
        refresh_min    = refresh,
        data_source    = source_cfg.get("source", "yahoo"),
        csv_path       = source_cfg.get("csv_path"),
        zerodha_config = source_cfg.get("zerodha_config"),
        upstox_token   = source_cfg.get("upstox_token"),
        enforce_time   = enforce,
    ).run()


# ─────────────────────────────────────────────────────────────────────────────
#  Trade journal
# ─────────────────────────────────────────────────────────────────────────────

def run_journal():
    journal = TradeJournal()
    while True:
        print_header("TRADE JOURNAL")
        print("  1. Performance report")
        print("  2. Close an open trade (log exit)")
        print("  3. Best setup from history")
        print("  4. Back")

        c = prompt("Choose", "1")
        if c == "1":
            print("\n" + journal.performance_report())
        elif c == "2":
            trade_id   = prompt("Trade ID (entry_time from log)")
            exit_price = float(prompt("Exit price"))
            reason     = prompt("Exit reason", "MANUAL")
            notes      = prompt("Notes", "")
            journal.log_exit(trade_id, exit_price, reason, notes)
        elif c == "3":
            best = journal.best_setup()
            if best:
                print(f"\n  Best setup: {best.get('timeframe')} | "
                      f"{best.get('quality')} | WR={best.get('win_rate',0):.1f}% | "
                      f"Trades={best.get('trades')} | PnL={best.get('total_pnl',0):+.1f}pts")
            else:
                print("  Need at least 3 closed trades for analysis.")
        elif c == "4":
            break


# ─────────────────────────────────────────────────────────────────────────────
#  Main menu
# ─────────────────────────────────────────────────────────────────────────────

def main():
    fetcher = NSEFetcher()
    print_header("NSE Option Trader  v2.0")
    print("\n  Strategy : Triple Confirmation (Supertrend + MACD + VWAP + PCR)")
    print("  Greeks   : Black-Scholes (Δ Delta, Γ Gamma, Θ Theta, ν Vega, ρ Rho)")
    print("  Sources  : NSE India + Yahoo / Zerodha / Upstox / CSV")

    while True:
        print("\n" + "─" * 65)
        print("  MENU")
        print("─" * 65)
        print("  1. Stock Option   (BEL, RELIANCE, TCS …)")
        print("  2. Index Option   (NIFTY, BANKNIFTY, FINNIFTY)")
        print("  3. Backtest       (auto-pick best TF & entry TF)")
        print("  4. Live Scheduler (auto-refresh every N min)")
        print("  5. Trade Journal  (log trades & view P&L)")
        print("  6. Exit")

        choice = prompt("Choose", "1")

        if choice in ("1", "2"):
            if choice == "2":
                print(f"  Indices: {', '.join(INDEX_SYMBOLS)}")
                symbol = prompt("Index", "NIFTY").upper()
            else:
                symbol = prompt("Stock symbol", "BEL").upper()

            fetcher._load_cookies()
            expiries = fetcher.get_expiry_dates(symbol)
            if expiries:
                print(f"  Available: {', '.join(expiries[:8])}")
                expiry = prompt("Expiry", expiries[0])
            else:
                expiry = prompt("Expiry (DD-Mon-YYYY)", "28-Apr-2026")

            tf         = prompt_int("Timeframe (min)", TIMEFRAMES, 15)
            source_cfg = prompt_data_source()
            analyse_symbol(fetcher, symbol, expiry, tf, source_cfg)

        elif choice == "3":
            run_backtest()
        elif choice == "4":
            run_scheduler()
        elif choice == "5":
            run_journal()
        elif choice == "6":
            print("\n  Trade safe! Jai Mata Di! 📈\n")
            sys.exit(0)
        else:
            print("  Enter 1–6.")


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
#  CSV backtest workflow
# ─────────────────────────────────────────────────────────────────────────────

def _run_csv_backtest():
    """
    Load real OHLCV data from a CSV/Excel file and run the backtest.

    Broker export formats supported:
      Zerodha Kite  : datetime, open, high, low, close, volume
      Upstox        : datetime, open, high, low, close, volume, oi
      TradingView   : time, open, high, low, close, Volume
      Any CSV       : any column names — auto-mapped
    """
    print_header("Backtest from CSV / Excel")
    print("\n  Supported formats:")
    print("    Zerodha Kite, Upstox, TradingView, any OHLCV CSV")
    print("  Required columns: datetime, open, high, low, close, volume")
    print()

    filepath = prompt("Path to CSV/Excel file (or 'sample' to generate one)", "sample")

    if filepath.lower() == "sample":
        filepath = create_sample_csv("sample_nifty_5min.csv", n_days=20)

    try:
        if filepath.endswith((".xlsx", ".xls")):
            df = load_ohlcv_excel(filepath)
        else:
            df = load_ohlcv_csv(filepath)
    except Exception as e:
        print(f"\n  ✗ Failed to load file: {e}")
        return

    # Validate
    issues = validate_ohlcv(df)
    if issues:
        print("\n  ⚠️  Data quality issues found:")
        for k, v in issues.items():
            print(f"    {k}: {v}")
    else:
        print("  ✓ Data quality check passed")

    detected_tf = detect_timeframe(df)
    print(f"\n  Auto-detected timeframe: {detected_tf} minutes")
    print(f"  Date range: {df.index[0]} → {df.index[-1]}")
    print(f"  Total rows: {len(df):,}")

    symbol = prompt("Symbol name (for labelling)", "CUSTOM")

    bt      = Backtester(df, timeframes=[tf for tf in TIMEFRAMES if tf >= detected_tf])
    results = bt.run_all_timeframes()
    report  = bt.generate_report(results)
    print("\n" + report)

    best_tf, entry_tf = bt.best_strategy(results)
    print(f"\n  Recommended: Trade on {best_tf}-min | Enter on {entry_tf}-min")