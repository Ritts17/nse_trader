"""
config.py — Central configuration for NSE Option Trader
"""

# ─── NSE ENDPOINTS ────────────────────────────────────────────────────────────
NSE_BASE_URL = "https://www.nseindia.com"
NSE_HOMEPAGE  = "https://www.nseindia.com"

# Stock option chain (like BEL, RELIANCE, TCS …)
STOCK_OPTION_URL = (
    "https://www.nseindia.com/api/NextApi/apiClient/GetQuoteApi"
    "?functionName=getOptionChainData&symbol={symbol}&params=expiryDate={expiry}"
)

# Index option chain (NIFTY, BANKNIFTY, FINNIFTY …)
INDEX_OPTION_URL = (
    "https://www.nseindia.com/api/option-chain-v3"
    "?type=Indices&symbol={symbol}&expiry={expiry}"
)

# Futures quote for VWAP (needs volume)
FUTURES_URL = (
    "https://www.nseindia.com/api/quote-derivative?symbol={symbol}"
)

# ─── REQUEST HEADERS ──────────────────────────────────────────────────────────
NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/146.0.0.0 Safari/537.36"
    ),
    "Accept":          "*/*",
    "Accept-Language": "en-US,en;q=0.6",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Referer":         "https://www.nseindia.com/",
    "sec-ch-ua":       '"Chromium";v="146", "Not-A.Brand";v="24", "Brave";v="146"',
    "sec-ch-ua-mobile":   "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest":  "empty",
    "sec-fetch-mode":  "cors",
    "sec-fetch-site":  "same-origin",
    "sec-gpc":         "1",
    "Connection":      "keep-alive",
}

# ─── STRATEGY PARAMETERS ──────────────────────────────────────────────────────

# Supertrend
SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0

# MACD
MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIGNAL = 9

# PCR thresholds (from video: >1.5 bullish, <1 bearish, >4 extremely bullish, <0.2 heavy bearish)
PCR_EXTREME_BULL  = 4.0
PCR_BULL          = 1.5
PCR_NEUTRAL_HIGH  = 1.2
PCR_NEUTRAL_LOW   = 0.9
PCR_BEAR          = 1.0
PCR_EXTREME_BEAR  = 0.2

# VWAP — entry only within this % band of VWAP
VWAP_ENTRY_BAND_PCT = 0.003   # 0.3% of VWAP

# Relevant strike count above/below ATM for PCR calculation (from video: 8 strikes)
PCR_STRIKE_RANGE = 8

# Backtest
BACKTEST_INITIAL_CAPITAL = 100_000
RISK_PER_TRADE_PCT       = 0.02   # 2% of capital per trade

# Time-of-day filter (from video: never trade before 11:00)
TRADE_START_HOUR   = 11
TRADE_START_MINUTE = 0

# Timeframes available for analysis (in minutes)
TIMEFRAMES = [1, 3, 5, 10, 15, 30, 60]

# ─── INDEX SYMBOLS ────────────────────────────────────────────────────────────
INDEX_SYMBOLS = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]