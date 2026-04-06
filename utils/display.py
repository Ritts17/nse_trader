"""
utils/display.py — Terminal Display Utilities

Coloured, formatted output for signals, Greeks, PCR, and SR levels.
"""

from colorama import init, Fore, Back, Style
init(autoreset=True)


def _color(text: str, color: str) -> str:
    mapping = {
        "green":        Fore.GREEN,
        "bright_green": Fore.GREEN + Style.BRIGHT,
        "red":          Fore.RED,
        "bright_red":   Fore.RED + Style.BRIGHT,
        "yellow":       Fore.YELLOW,
        "cyan":         Fore.CYAN,
        "white":        Fore.WHITE,
        "grey":         Fore.WHITE + Style.DIM,
        "blue":         Fore.BLUE,
        "magenta":      Fore.MAGENTA,
    }
    return mapping.get(color, "") + str(text) + Style.RESET_ALL


def print_header(title: str):
    print("\n" + "═" * 65)
    print(f"  {Fore.CYAN + Style.BRIGHT}{title}{Style.RESET_ALL}")
    print("═" * 65)


def print_section(title: str):
    print(f"\n  {Fore.YELLOW + Style.BRIGHT}▸ {title}{Style.RESET_ALL}")
    print("  " + "─" * 50)


def print_signal(signal: dict):
    """Display the main trade signal in a clear format."""
    print_header(f"TRADE SIGNAL — {signal['symbol']} [{signal['timeframe']}]")
    print(f"  Time   : {signal['timestamp']}")
    print(f"  Spot   : ₹{signal['spot']:,.2f}")
    print()

    action = signal.get("action", "WAIT")
    quality = signal.get("quality", "")

    if action == "WAIT":
        print(f"  Action : {_color('⏸  WAIT — No trade', 'yellow')}")
    elif "BUY" in action:
        q_color = "bright_green" if quality == "STRONG" else "green"
        print(f"  Action : {_color(f'▲  {action}', q_color)} {_color(f'[{quality}]', q_color)}")
    elif "SELL" in action:
        q_color = "bright_red" if quality == "STRONG" else "red"
        print(f"  Action : {_color(f'▼  {action}', q_color)} {_color(f'[{quality}]', q_color)}")

    print()

    if signal.get("entry"):
        print(f"  Entry     : ₹{signal['entry']:,.2f}")
    if signal.get("stop_loss"):
        print(f"  Stop Loss : ₹{signal['stop_loss']:,.2f}  {_color('← Exit if hit immediately', 'grey')}")
    if signal.get("target_1"):
        print(f"  Target 1  : ₹{signal['target_1']:,.2f}")
    if signal.get("target_2") and signal["target_2"] != signal.get("target_1"):
        print(f"  Target 2  : ₹{signal['target_2']:,.2f}  {_color('(partial booking)', 'grey')}")
    if signal.get("r_factor"):
        rf = signal["r_factor"]
        rf_color = "bright_green" if rf >= 3 else "green" if rf >= 2 else "yellow"
        print(f"  R-Factor  : {_color(f'{rf:.1f}:1', rf_color)}  {_color('(Reward:Risk)', 'grey')}")
    if signal.get("strike"):
        print(f"  Strike    : {signal['strike']}")

    print()
    print_section("Reasons")
    for r in signal.get("reason", []):
        print(f"    {r}")


def print_pcr(pcr: dict):
    """Display PCR analysis."""
    print_section("PCR Analysis")
    if not pcr or pcr.get("pcr") is None:
        print("    No PCR data available")
        return

    pcr_val = pcr["pcr"]
    bias    = pcr["bias"]
    signal  = pcr["signal"]
    color   = pcr.get("color", "white")

    print(f"    PCR Value  : {_color(f'{pcr_val:.3f}', color)}")
    print(f"    Market Bias: {_color(bias, color)}")
    print(f"    Signal     : {_color(signal, color)}")
    print(f"    ATM Strike : {pcr.get('atm_strike', 'N/A')}")
    print(f"    Strikes Used: ±{pcr.get('strikes_used', 'N/A')//2} from ATM")
    print(f"    ΔOI (PE)   : {pcr.get('sum_pe_delta_oi', 0):,}")
    print(f"    ΔOI (CE)   : {pcr.get('sum_ce_delta_oi', 0):,}")


def print_iv_analysis(iv: dict):
    """Display IV analysis."""
    print_section("Implied Volatility Analysis")
    if not iv:
        return

    level  = iv.get("iv_level", "?")
    avg_iv = iv.get("avg_iv", 0)
    iv_color = "bright_red" if level == "HIGH" else "green" if level == "LOW" else "yellow"

    print(f"    IV Level   : {_color(level, iv_color)}")
    print(f"    ATM IV (CE): {iv.get('iv_ce', 0):.2f}%")
    print(f"    ATM IV (PE): {iv.get('iv_pe', 0):.2f}%")
    print(f"    Avg IV     : {_color(f'{avg_iv:.2f}%', iv_color)}")
    print(f"    Action     : {iv.get('action', '')}")
    print(f"    Caution    : {_color(iv.get('caution', ''), 'yellow')}")


def print_sr_levels(sr: dict, spot: float):
    """Display Support & Resistance levels."""
    print_section("Support & Resistance (from OI)")
    if not sr:
        return

    for r in sr.get("resistance_levels", []):
        dist = ((r - spot) / spot) * 100
        print(f"    Resistance : {_color(f'₹{r:,.0f}', 'red')}  (+{dist:.2f}% from spot)")

    for s in sr.get("support_levels", []):
        dist = ((spot - s) / spot) * 100
        print(f"    Support    : {_color(f'₹{s:,.0f}', 'green')}  (-{dist:.2f}% from spot)")


def print_greeks_table(df_ce: "pd.DataFrame", df_pe: "pd.DataFrame", spot: float, n: int = 5):
    """Display Greeks for strikes near ATM."""
    import pandas as pd
    print_section("Option Greeks (Near ATM Strikes)")

    cols = ["strikePrice", "lastPrice", "impliedVolatility", "delta", "theta", "vega", "gamma"]

    for label, df in [("CALL (CE)", df_ce), ("PUT (PE)", df_pe)]:
        if df.empty or "delta" not in df.columns:
            continue

        strikes = sorted(df["strikePrice"].unique())
        atm = min(strikes, key=lambda x: abs(x - spot))
        atm_idx = list(strikes).index(atm)
        near = strikes[max(0, atm_idx - 2): atm_idx + 3]
        subset = df[df["strikePrice"].isin(near)][
            [c for c in cols if c in df.columns]
        ]
        print(f"\n    ── {label} ──")
        print("    " + subset.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def print_vwap_check(vwap_check: dict):
    """Display VWAP entry check."""
    print_section("VWAP Entry Check")
    if not vwap_check:
        return
    valid = vwap_check.get("valid_entry", False)
    color = "green" if valid else "red"
    print(f"    VWAP       : ₹{vwap_check.get('vwap', 0):,.2f}")
    print(f"    Band       : ₹{vwap_check.get('lower_band', 0):,.2f} — ₹{vwap_check.get('upper_band', 0):,.2f}")
    print(f"    Distance   : {vwap_check.get('distance_pct', 0):.3f}%")
    print(f"    Valid Entry: {_color('YES ✓' if valid else 'NO ✗', color)}")
    print(f"    Note       : {_color(vwap_check.get('quality', ''), color)}")


def print_breakout(breakout: dict):
    """Display breakout info if detected."""
    if not breakout or not breakout.get("breakout"):
        return
    print_section("Breakout Alert")
    b = breakout["breakout"]
    color = "bright_green" if "BULL" in b else "bright_red"
    print(f"    {_color(breakout.get('note', b), color)}")
    vol_ok = breakout.get("volume_confirm", False)
    print(f"    Volume confirmed: {_color('YES ✓' if vol_ok else 'NO — treat with caution', 'green' if vol_ok else 'yellow')}")