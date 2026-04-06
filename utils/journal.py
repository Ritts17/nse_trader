"""
utils/journal.py — Trade Journal & Performance Tracking

Logs trades, calculates P&L, and provides performance analytics.

Usage:
    journal = TradeJournal()
    journal.log_entry(symbol, direction, entry_price, quantity, stop_loss, target)
    journal.log_exit(trade_id, exit_price, reason, notes)
    report = journal.performance_report()
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class Trade:
    trade_id: str
    symbol: str
    direction: str  # "BUY" or "SELL"
    entry_time: str
    entry_price: float
    quantity: int
    stop_loss: float
    target: float
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    notes: str = ""
    pnl_points: float = 0.0
    pnl_pct: float = 0.0

    def close_trade(self, exit_price: float, exit_reason: str, notes: str = ""):
        """Close the trade and calculate P&L."""
        self.exit_time = datetime.now().isoformat()
        self.exit_price = exit_price
        self.exit_reason = exit_reason
        self.notes = notes

        if self.direction == "BUY":
            self.pnl_points = (exit_price - self.entry_price) * self.quantity
        else:  # SELL
            self.pnl_points = (self.entry_price - exit_price) * self.quantity

        self.pnl_pct = (self.pnl_points / (self.entry_price * self.quantity)) * 100


class TradeJournal:
    """
    Manages trade logging and performance analysis.
    """

    def __init__(self, filename: str = "trades.json"):
        self.filename = filename
        self.trades: List[Trade] = []
        self._load_trades()

    def _load_trades(self):
        """Load trades from JSON file."""
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    data = json.load(f)
                    self.trades = [Trade(**trade) for trade in data]
            except Exception as e:
                print(f"Warning: Could not load trades: {e}")
                self.trades = []

    def _save_trades(self):
        """Save trades to JSON file."""
        try:
            with open(self.filename, 'w') as f:
                json.dump([asdict(trade) for trade in self.trades], f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save trades: {e}")

    def log_entry(self, symbol: str, direction: str, entry_price: float,
                  quantity: int, stop_loss: float, target: float) -> str:
        """
        Log a new trade entry.

        Returns the trade ID.
        """
        trade_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        trade = Trade(
            trade_id=trade_id,
            symbol=symbol,
            direction=direction,
            entry_time=datetime.now().isoformat(),
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            target=target,
        )
        self.trades.append(trade)
        self._save_trades()
        return trade_id

    def log_exit(self, trade_id: str, exit_price: float,
                 exit_reason: str, notes: str = ""):
        """Log exit for an existing trade."""
        for trade in self.trades:
            if trade.trade_id == trade_id:
                trade.close_trade(exit_price, exit_reason, notes)
                self._save_trades()
                return
        raise ValueError(f"Trade {trade_id} not found")

    def performance_report(self) -> str:
        """Generate a performance report."""
        if not self.trades:
            return "No trades logged yet."

        closed_trades = [t for t in self.trades if t.exit_time]
        open_trades = [t for t in self.trades if not t.exit_time]

        total_trades = len(closed_trades)
        winning_trades = len([t for t in closed_trades if t.pnl_points > 0])
        losing_trades = len([t for t in closed_trades if t.pnl_points < 0])

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_pnl = sum(t.pnl_points for t in closed_trades)
        avg_win = sum(t.pnl_points for t in closed_trades if t.pnl_points > 0) / winning_trades if winning_trades > 0 else 0
        avg_loss = sum(t.pnl_points for t in closed_trades if t.pnl_points < 0) / losing_trades if losing_trades > 0 else 0

        report = f"""
TRADE PERFORMANCE REPORT
{'─' * 40}

Total Trades: {total_trades}
Win Rate: {win_rate:.1f}%
Total P&L: ₹{total_pnl:,.2f}

Average Win: ₹{avg_win:,.2f}
Average Loss: ₹{avg_loss:,.2f}

Open Trades: {len(open_trades)}
"""

        if open_trades:
            report += "\nOPEN TRADES:\n"
            for trade in open_trades:
                report += f"  {trade.trade_id}: {trade.direction} {trade.symbol} @ ₹{trade.entry_price}\n"

        return report.strip()

    def best_setup(self) -> Optional[Dict[str, Any]]:
        """Find the best performing setup from closed trades."""
        closed_trades = [t for t in self.trades if t.exit_time]
        if len(closed_trades) < 3:
            return None

        # Group by symbol and direction
        setups = {}
        for trade in closed_trades:
            key = f"{trade.symbol}_{trade.direction}"
            if key not in setups:
                setups[key] = []
            setups[key].append(trade)

        # Find setup with highest win rate and best avg P&L
        best_setup = None
        best_score = 0

        for key, trades in setups.items():
            wins = len([t for t in trades if t.pnl_points > 0])
            win_rate = wins / len(trades)
            avg_pnl = sum(t.pnl_points for t in trades) / len(trades)
            score = win_rate * avg_pnl

            if score > best_score:
                best_score = score
                symbol, direction = key.split('_')
                best_setup = {
                    "symbol": symbol,
                    "direction": direction,
                    "trades": len(trades),
                    "win_rate": win_rate * 100,
                    "total_pnl": sum(t.pnl_points for t in trades),
                    "avg_pnl": avg_pnl,
                    "quality": "HIGH" if win_rate > 0.7 else "MEDIUM" if win_rate > 0.5 else "LOW"
                }

        return best_setup