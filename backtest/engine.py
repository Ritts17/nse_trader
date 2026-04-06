"""
backtest/engine.py — Multi-Timeframe Backtesting Engine

Purpose:
  1. Run the triple-confirmation strategy across multiple timeframes
  2. Score each timeframe by: win rate, R-factor, Sharpe, max drawdown
  3. Pick the BEST TIMEFRAME automatically
  4. Find the BEST ENTRY TIMEFRAME (lower TF to drill down for entry)
  5. Generate a full strategy report

From Video 2: "You can backtest this strategy as much as you want"
Strategy rules implemented:
  - Enter when: MACD crossover + Supertrend flip + Price near VWAP
  - Exit  when: MACD crosses in opposite direction OR stop loss hit
  - Stop loss : low of previous candle (for BUY) / high (for SELL)
  - PCR filter: no BUY when PCR is bearish, no SELL when PCR is bullish
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from config import (
    BACKTEST_INITIAL_CAPITAL, RISK_PER_TRADE_PCT,
    TRADE_START_HOUR, TIMEFRAMES,
)
from analysis.indicators import apply_all_indicators, resample_ohlcv


@dataclass
class Trade:
    entry_idx:    int
    entry_price:  float
    direction:    str          # "BUY" or "SELL"
    stop_loss:    float
    target:       float
    exit_idx:     Optional[int] = None
    exit_price:   Optional[float] = None
    exit_reason:  str          = ""
    pnl_points:   float        = 0.0
    pnl_pct:      float        = 0.0
    r_achieved:   float        = 0.0
    quality:      str          = ""


@dataclass
class BacktestResult:
    timeframe:      int
    total_trades:   int  = 0
    wins:           int  = 0
    losses:         int  = 0
    win_rate:       float = 0.0
    avg_r:          float = 0.0
    total_pnl_pts:  float = 0.0
    max_drawdown:   float = 0.0
    sharpe:         float = 0.0
    profit_factor:  float = 0.0
    score:          float = 0.0     # composite score for ranking
    trades:         list  = field(default_factory=list)
    equity_curve:   list  = field(default_factory=list)


class Backtester:
    """
    Runs the Triple Confirmation strategy over historical OHLCV data
    across multiple timeframes and ranks them by performance.

    Usage:
        bt = Backtester(df_1min)
        results = bt.run_all_timeframes()
        best_tf, entry_tf = bt.best_strategy()
        report = bt.generate_report(results)
    """

    def __init__(
        self,
        df_1min:   pd.DataFrame,   # 1-minute OHLCV data (requires datetime index)
        capital:   float = BACKTEST_INITIAL_CAPITAL,
        risk_pct:  float = RISK_PER_TRADE_PCT,
        timeframes: list = None,
    ):
        self.df_1min   = df_1min
        self.capital   = capital
        self.risk_pct  = risk_pct
        self.timeframes = timeframes or TIMEFRAMES

    # ── Run ───────────────────────────────────────────────────────────────────

    def run_all_timeframes(self) -> dict[int, BacktestResult]:
        """Run backtest on all configured timeframes."""
        results = {}
        for tf in self.timeframes:
            print(f"[Backtest] Running {tf}-min timeframe...")
            try:
                result = self._run_single_timeframe(tf)
                results[tf] = result
                print(
                    f"  → Trades: {result.total_trades} | "
                    f"Win Rate: {result.win_rate:.1f}% | "
                    f"Avg R: {result.avg_r:.2f} | "
                    f"Score: {result.score:.2f}"
                )
            except Exception as e:
                print(f"  ✗ Failed for {tf}-min: {e}")
        return results

    def _run_single_timeframe(self, tf_minutes: int) -> BacktestResult:
        """Run strategy on a single timeframe."""
        df = apply_all_indicators(self.df_1min.copy(), tf_minutes)

        # Time filter: only signals after TRADE_START_HOUR
        if hasattr(df.index, 'hour'):
            df = df[df.index.hour >= TRADE_START_HOUR]

        signal_rows = df[df["triple_signal"].notna()].copy()

        trades      = []
        in_trade    = False
        current     = None
        equity      = self.capital
        equity_curve = [equity]
        peak_equity  = equity

        for i in range(len(df)):
            row = df.iloc[i]

            # ── Manage open trade ──────────────────────────────────────────
            if in_trade and current:
                price = row["close"]
                exit_reason = None

                if current.direction == "BUY":
                    if price <= current.stop_loss:
                        exit_reason = "SL_HIT"
                        exit_price  = current.stop_loss
                    elif price >= current.target:
                        exit_reason = "TARGET_HIT"
                        exit_price  = current.target
                    elif row["macd_crossover"] == "SELL_CROSS":
                        exit_reason = "MACD_EXIT"
                        exit_price  = price
                else:  # SELL
                    if price >= current.stop_loss:
                        exit_reason = "SL_HIT"
                        exit_price  = current.stop_loss
                    elif price <= current.target:
                        exit_reason = "TARGET_HIT"
                        exit_price  = current.target
                    elif row["macd_crossover"] == "BUY_CROSS":
                        exit_reason = "MACD_EXIT"
                        exit_price  = price

                if exit_reason:
                    current.exit_idx   = i
                    current.exit_price = exit_price
                    current.exit_reason = exit_reason

                    if current.direction == "BUY":
                        current.pnl_points = exit_price - current.entry_price
                    else:
                        current.pnl_points = current.entry_price - exit_price

                    current.pnl_pct = current.pnl_points / current.entry_price * 100

                    risk = abs(current.entry_price - current.stop_loss)
                    current.r_achieved = current.pnl_points / risk if risk > 0 else 0

                    # Update equity
                    position_size = (equity * self.risk_pct) / abs(current.entry_price - current.stop_loss)
                    pnl_money     = current.pnl_points * position_size
                    equity        = max(0, equity + pnl_money)
                    equity_curve.append(equity)
                    peak_equity   = max(peak_equity, equity)

                    trades.append(current)
                    in_trade = False
                    current  = None

            # ── Check for new signal ───────────────────────────────────────
            if not in_trade and row["triple_signal"] in ("BUY", "SELL"):
                direction  = row["triple_signal"]
                entry      = row["close"]
                sl         = row["stop_loss"]
                risk       = abs(entry - sl) if not pd.isna(sl) else entry * 0.01

                if pd.isna(sl) or risk == 0:
                    continue

                target = entry + risk * 2 if direction == "BUY" else entry - risk * 2

                current  = Trade(
                    entry_idx   = i,
                    entry_price = entry,
                    direction   = direction,
                    stop_loss   = sl,
                    target      = target,
                    quality     = row.get("signal_quality", ""),
                )
                in_trade = True

        result = self._compute_stats(tf_minutes, trades, equity_curve)
        return result

    def _compute_stats(
        self, tf: int, trades: list, equity_curve: list,
    ) -> BacktestResult:
        if not trades:
            return BacktestResult(timeframe=tf)

        wins   = [t for t in trades if t.pnl_points > 0]
        losses = [t for t in trades if t.pnl_points <= 0]

        win_rate      = len(wins) / len(trades) * 100
        avg_r         = np.mean([t.r_achieved for t in trades])
        total_pnl_pts = sum(t.pnl_points for t in trades)
        gross_profit  = sum(t.pnl_points for t in wins  if t.pnl_points > 0)
        gross_loss    = abs(sum(t.pnl_points for t in losses if t.pnl_points < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Max drawdown
        eq = np.array(equity_curve)
        peak = np.maximum.accumulate(eq)
        dd   = (peak - eq) / peak
        max_dd = dd.max() * 100

        # Sharpe (simplified, using daily returns)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe  = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0

        # Composite score — rewards consistent performance over small samples
        # Win rate 25%, Avg R 25%, Sharpe 20%, Profit Factor 15%, Trade count 15%
        # Trade count bonus: capped at 50 trades (more trades = more reliable stats)
        trade_count_bonus = min(len(trades), 50) / 50
        score = (
            (win_rate / 100)        * 0.25 +
            min(avg_r, 5) / 5       * 0.25 +
            min(max(sharpe, 0), 3) / 3 * 0.20 +
            min(profit_factor, 5) / 5  * 0.15 +
            trade_count_bonus           * 0.15
        ) * 100

        return BacktestResult(
            timeframe     = tf,
            total_trades  = len(trades),
            wins          = len(wins),
            losses        = len(losses),
            win_rate      = round(win_rate, 2),
            avg_r         = round(avg_r, 2),
            total_pnl_pts = round(total_pnl_pts, 2),
            max_drawdown  = round(max_dd, 2),
            sharpe        = round(sharpe, 2),
            profit_factor = round(profit_factor, 2),
            score         = round(score, 2),
            trades        = trades,
            equity_curve  = equity_curve,
        )

    # ── Best strategy picker ──────────────────────────────────────────────────

    def best_strategy(
        self, results: dict[int, BacktestResult]
    ) -> tuple[int, int]:
        """
        Pick the best strategy timeframe and the optimal entry timeframe.

        Best TF:   highest composite score with at least 10 trades
        Entry TF:  next lower timeframe (to drill down for precise entry)
                   e.g. best=15min → entry on 5min
                        best=5min  → entry on 1min

        Returns (best_timeframe_minutes, entry_timeframe_minutes)
        """
        valid = {
            tf: r for tf, r in results.items()
            if r.total_trades >= 5     # minimum trades to be meaningful
        }
        if not valid:
            # Fall back to best with any trades
            valid = {tf: r for tf, r in results.items() if r.total_trades > 0}
        if not valid:
            return (15, 5)

        best_tf = max(valid, key=lambda tf: valid[tf].score)

        # Entry timeframe: one step lower
        sorted_tfs = sorted(self.timeframes)
        best_idx   = sorted_tfs.index(best_tf) if best_tf in sorted_tfs else 0
        entry_tf   = sorted_tfs[max(0, best_idx - 1)]

        return best_tf, entry_tf

    # ── Report ────────────────────────────────────────────────────────────────

    def generate_report(self, results: dict[int, BacktestResult]) -> str:
        """Generate a readable text report of all backtest results."""
        lines = [
            "=" * 70,
            "  BACKTEST REPORT — Triple Confirmation Strategy",
            "=" * 70,
            f"  Initial Capital : ₹{self.capital:,.0f}",
            f"  Risk Per Trade  : {self.risk_pct*100:.1f}%",
            "",
            f"  {'TF':>5}  {'Trades':>6}  {'Win%':>6}  {'AvgR':>5}  {'PF':>5}  {'Sharpe':>6}  {'MaxDD%':>6}  {'Score':>6}",
            "  " + "-" * 62,
        ]

        sorted_results = sorted(results.values(), key=lambda r: r.score, reverse=True)

        for r in sorted_results:
            marker = " ◄ BEST" if r == sorted_results[0] else ""
            lines.append(
                f"  {r.timeframe:>4}m  {r.total_trades:>6}  {r.win_rate:>6.1f}  "
                f"{r.avg_r:>5.2f}  {r.profit_factor:>5.2f}  {r.sharpe:>6.2f}  "
                f"{r.max_drawdown:>6.1f}  {r.score:>6.2f}{marker}"
            )

        best_tf, entry_tf = self.best_strategy(results)
        best_r = results.get(best_tf)

        lines += [
            "",
            "  RECOMMENDATION:",
            f"  ► Best Strategy Timeframe : {best_tf} minutes",
            f"  ► Entry Timeframe         : {entry_tf} minutes (drill down for precise entry)",
        ]

        if best_r:
            lines += [
                "",
                f"  Best Timeframe ({best_tf}min) Statistics:",
                f"    Total Trades  : {best_r.total_trades}",
                f"    Win Rate      : {best_r.win_rate:.1f}%",
                f"    Avg R-Factor  : {best_r.avg_r:.2f}",
                f"    Profit Factor : {best_r.profit_factor:.2f}",
                f"    Sharpe Ratio  : {best_r.sharpe:.2f}",
                f"    Max Drawdown  : {best_r.max_drawdown:.1f}%",
            ]

            # Show last 5 trades
            if best_r.trades:
                lines += ["", f"  Last {min(5, len(best_r.trades))} Trades:"]
                for t in best_r.trades[-5:]:
                    result_str = "WIN" if t.pnl_points > 0 else "LOSS"
                    lines.append(
                        f"    {t.direction} @ {t.entry_price:.2f} → "
                        f"{t.exit_price:.2f} | {result_str} | "
                        f"R={t.r_achieved:.2f} | Exit: {t.exit_reason}"
                    )

        lines.append("=" * 70)
        return "\n".join(lines)