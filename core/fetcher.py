"""
core/fetcher.py — NSE Session & Data Fetcher

Strategy from Video 1 (Vijay Jadia):
  - Use requests.Session()
  - Hit the NSE homepage first to get cookies
  - Then fetch the actual API endpoint with those cookies + headers
  - Cookie is the key to getting 200 instead of 401
"""

import time
import requests
import pandas as pd
from typing import Optional, Dict, Any

from config import (
    NSE_BASE_URL, NSE_HOMEPAGE,
    STOCK_OPTION_URL, INDEX_OPTION_URL,
    NSE_HEADERS, INDEX_SYMBOLS,
)


class NSEFetcher:
    """
    Handles NSE session management and option chain data retrieval.

    Usage:
        fetcher = NSEFetcher()
        df_ce, df_pe, meta = fetcher.get_option_chain("BEL", "28-Apr-2026")
        df_ce, df_pe, meta = fetcher.get_option_chain("NIFTY", "07-Apr-2026")
    """

    def __init__(self, retry: int = 3, pause: float = 1.5):
        self.retry   = retry
        self.pause   = pause
        self.session = requests.Session()
        self.session.headers.update(NSE_HEADERS)
        self._cookies_loaded = False

    # ── Session bootstrap ────────────────────────────────────────────────────

    def _load_cookies(self):
        """
        Visit NSE homepage to acquire session cookies (critical step).
        Without this the API returns 401.
        """
        try:
            resp = self.session.get(NSE_HOMEPAGE, timeout=15)
            resp.raise_for_status()
            self._cookies_loaded = True
            print("[NSEFetcher] Session cookies acquired.")
        except Exception as exc:
            print(f"[NSEFetcher] WARNING: Could not load cookies — {exc}")

    def _get(self, url: str) -> Optional[Dict[str, Any]]:
        """Make a GET request with retries and return parsed JSON."""
        if not self._cookies_loaded:
            self._load_cookies()

        for attempt in range(1, self.retry + 1):
            try:
                resp = self.session.get(url, timeout=15)
                if resp.status_code == 200:
                    return resp.json()
                print(f"[NSEFetcher] Attempt {attempt}: HTTP {resp.status_code} for {url}")
                # Reload cookies on auth failure
                if resp.status_code in (401, 403):
                    self._cookies_loaded = False
                    self._load_cookies()
                time.sleep(self.pause)
            except Exception as exc:
                print(f"[NSEFetcher] Attempt {attempt} error: {exc}")
                time.sleep(self.pause)
        return None

    # ── Public API ───────────────────────────────────────────────────────────

    def get_option_chain(
        self,
        symbol:  str,
        expiry:  str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Fetch option chain for a stock or index.

        Args:
            symbol : e.g. "BEL", "RELIANCE", "NIFTY", "BANKNIFTY"
            expiry : e.g. "28-Apr-2026"  or  "07-Apr-2026"

        Returns:
            (df_ce, df_pe, meta)
            df_ce  — CE (Call) options DataFrame
            df_pe  — PE (Put)  options DataFrame
            meta   — dict with underlyingValue, timestamp, symbol
        """
        is_index = symbol.upper() in INDEX_SYMBOLS

        if is_index:
            url = INDEX_OPTION_URL.format(symbol=symbol.upper(), expiry=expiry)
        else:
            url = STOCK_OPTION_URL.format(symbol=symbol.upper(), expiry=expiry)

        raw = self._get(url)
        if raw is None:
            raise ConnectionError(f"Failed to fetch option chain for {symbol} / {expiry}")

        return self._parse_chain(raw, is_index, symbol)

    def get_expiry_dates(self, symbol: str) -> list[str]:
        """
        Returns available expiry dates for a symbol.
        Fetches the chain once and extracts unique expiryDates.
        """
        is_index = symbol.upper() in INDEX_SYMBOLS
        # Use a dummy expiry — NSE returns all records; we parse dates from them
        if is_index:
            url = f"https://www.nseindia.com/api/option-chain-v3?type=Indices&symbol={symbol.upper()}"
        else:
            url = f"https://www.nseindia.com/api/NextApi/apiClient/GetQuoteApi?functionName=getOptionChainData&symbol={symbol.upper()}"

        raw = self._get(url)
        if raw is None:
            return []

        dates = set()
        records = raw.get("records", raw)
        data    = records.get("data", raw.get("data", []))
        for row in data:
            d = row.get("expiryDates") or row.get("expiryDate", "")
            if d:
                dates.add(d)
        return sorted(dates)

    # ── Internal parser ──────────────────────────────────────────────────────

    def _parse_chain(
        self,
        raw:      Dict,
        is_index: bool,
        symbol:   str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Normalise raw JSON into two DataFrames (CE / PE) and a meta dict.

        Handles both stock and index response shapes:
          - Stock  : raw["data"]  with raw["underlyingValue"]
          - Index  : raw["records"]["data"] with raw["records"]["underlyingValue"]
        """
        if is_index:
            records  = raw.get("records", {})
            data     = records.get("data", [])
            und_val  = records.get("underlyingValue", 0)
            ts       = records.get("timestamp", "")
        else:
            data     = raw.get("data", [])
            und_val  = raw.get("underlyingValue", 0)
            ts       = raw.get("timestamp", "")

        ce_rows, pe_rows = [], []

        for item in data:
            strike = item.get("strikePrice", 0)
            ce     = item.get("CE")
            pe     = item.get("PE")

            if ce and ce.get("strikePrice"):
                ce_rows.append(self._flatten(ce, strike))
            if pe and pe.get("strikePrice"):
                pe_rows.append(self._flatten(pe, strike))

        df_ce = pd.DataFrame(ce_rows) if ce_rows else pd.DataFrame()
        df_pe = pd.DataFrame(pe_rows) if pe_rows else pd.DataFrame()

        for df in [df_ce, df_pe]:
            if not df.empty:
                df.sort_values("strikePrice", inplace=True)
                df.reset_index(drop=True, inplace=True)

        meta = {
            "symbol":          symbol.upper(),
            "underlyingValue": und_val,
            "timestamp":       ts,
            "is_index":        is_index,
        }

        return df_ce, df_pe, meta

    @staticmethod
    def _flatten(option: Dict, strike: float) -> Dict:
        return {
            "strikePrice":           strike,
            "expiryDate":            option.get("expiryDate"),
            "openInterest":          option.get("openInterest", 0),
            "changeInOI":            option.get("changeinOpenInterest", 0),
            "pChangeInOI":           option.get("pchangeinOpenInterest", 0),
            "totalTradedVolume":     option.get("totalTradedVolume", 0),
            "impliedVolatility":     option.get("impliedVolatility", 0),
            "lastPrice":             option.get("lastPrice", 0),
            "change":                option.get("change", 0),
            "pChange":               option.get("pchange", 0),
            "totalBuyQty":           option.get("totalBuyQuantity", 0),
            "totalSellQty":          option.get("totalSellQuantity", 0),
            "buyPrice1":             option.get("buyPrice1", 0),
            "sellPrice1":            option.get("sellPrice1", 0),
            "underlyingValue":       option.get("underlyingValue", 0),
        }