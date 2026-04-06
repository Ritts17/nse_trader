"""
analysis/pcr_tracker.py — PCR Session Tracking

Tracks PCR (Put/Call Ratio) trends throughout the trading session.
Identifies shifts in market sentiment and potential reversal points.

Usage:
    tracker = PCRTracker()
    tracker.update(pcr_value, timestamp)
    session_data = tracker.get_session_summary()
    print_pcr_session(session_data)
"""

from datetime import datetime, time
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class PCRDataPoint:
    timestamp: datetime
    pcr_value: float
    trend: str  # "rising", "falling", "sideways"


class PCRTracker:
    """
    Tracks PCR evolution throughout the trading session.
    """

    def __init__(self):
        self.data_points: List[PCRDataPoint] = []
        self.session_start = None

    def update(self, pcr_value: float, timestamp: datetime = None):
        """Add a new PCR data point."""
        if timestamp is None:
            timestamp = datetime.now()

        if not self.session_start:
            self.session_start = timestamp

        # Determine trend
        if len(self.data_points) >= 2:
            prev_pcr = self.data_points[-1].pcr_value
            if pcr_value > prev_pcr * 1.05:  # 5% increase
                trend = "rising"
            elif pcr_value < prev_pcr * 0.95:  # 5% decrease
                trend = "falling"
            else:
                trend = "sideways"
        else:
            trend = "sideways"

        self.data_points.append(PCRDataPoint(timestamp, pcr_value, trend))

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the session."""
        if not self.data_points:
            return {"error": "No PCR data available"}

        pcrs = [dp.pcr_value for dp in self.data_points]
        trends = [dp.trend for dp in self.data_points]

        return {
            "start_time": self.session_start.isoformat() if self.session_start else None,
            "end_time": self.data_points[-1].timestamp.isoformat(),
            "data_points": len(self.data_points),
            "current_pcr": pcrs[-1],
            "min_pcr": min(pcrs),
            "max_pcr": max(pcrs),
            "avg_pcr": sum(pcrs) / len(pcrs),
            "trend_distribution": {
                "rising": trends.count("rising"),
                "falling": trends.count("falling"),
                "sideways": trends.count("sideways"),
            },
            "dominant_trend": max(set(trends), key=trends.count) if trends else "unknown",
            "volatility": (max(pcrs) - min(pcrs)) / (sum(pcrs) / len(pcrs)) if pcrs else 0,
        }


def print_pcr_session(session_data: Dict[str, Any]):
    """Print formatted PCR session summary."""
    if "error" in session_data:
        print(f"  PCR Session: {session_data['error']}")
        return

    print("  PCR SESSION SUMMARY")
    print("  ────────────────────")
    print(f"  Data points: {session_data['data_points']}")
    print(f"  Current PCR: {session_data['current_pcr']:.2f}")
    print(f"  Range: {session_data['min_pcr']:.2f} - {session_data['max_pcr']:.2f}")
    print(f"  Average: {session_data['avg_pcr']:.2f}")
    print(f"  Dominant trend: {session_data['dominant_trend']}")
    print(f"  Volatility: {session_data['volatility']:.1%}")

    trend_dist = session_data['trend_distribution']
    print("  Trend distribution:")
    print(f"    Rising: {trend_dist['rising']}")
    print(f"    Falling: {trend_dist['falling']}")
    print(f"    Sideways: {trend_dist['sideways']}")

    # Sentiment interpretation
    current_pcr = session_data['current_pcr']
    if current_pcr > 1.2:
        sentiment = "BEARISH (High PCR)"
    elif current_pcr < 0.8:
        sentiment = "BULLISH (Low PCR)"
    else:
        sentiment = "NEUTRAL"

    print(f"  Market sentiment: {sentiment}")