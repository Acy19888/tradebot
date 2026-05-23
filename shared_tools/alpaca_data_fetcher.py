"""Equity OHLCV adapter via Alpaca Market Data API (free / paper tier).

Phase 7e — replaces yfinance for equity backtests because yfinance caps
intraday history at 7 days (1m) / 60 days (5m), which made strategies
like rh-momentum-aapl-5m unbacktestable against a 2-year ``--since``.

Alpaca's free tier (IEX feed) provides:
  * 1m bars: 5+ years back
  * 5m / 15m / 1h / 1d: same window
  * 200 requests / minute
  * pagination via `next_page_token`

Public surface matches ``shared_tools/equity_data_fetcher.load_equity_ohlcv``
exactly so the dispatcher in ``equity_data_fetcher`` can swap providers
without touching the backtest framework.

Authentication: reads ``ALPACA_API_KEY`` and ``ALPACA_API_SECRET`` from
the environment. If either is missing, ``load_equity_ohlcv`` raises so
callers can fall back to yfinance.

Why pure ``requests`` and not the ``alpaca-py`` SDK:
  * The SDK pulls in pydantic v2 + msgpack + a websocket stack we don't
    need for backtests.
  * The REST surface we use is two endpoints with a stable contract.
  * Keeps the dependency lock small (pytest + pandas + requests are
    already in the test path).

IEX feed caveat:
  IEX is one of ~12 US exchanges and covers ~2-3% of total SIP volume.
  For major-cap tickers (AAPL, NVDA, TSLA, MSFT, SPY) the price action
  tracks the consolidated tape closely enough that 5m+ backtests are
  comparable to what a Robinhood-routed order would have seen. For
  thinly-traded mid-caps the IEX print can be sparser than reality —
  document for the operator but don't auto-block.
"""
from __future__ import annotations

import functools
import os
import sys
import time
from typing import Optional

import pandas as pd
import requests


_BASE_URL = "https://data.alpaca.markets/v2/stocks/{symbol}/bars"

# Map our codebase's timeframe strings onto Alpaca's TimeFrame strings.
# Alpaca accepts e.g. "1Min" / "5Min" / "1Hour" / "1Day".
_TIMEFRAME_MAP = {
    "1m":  "1Min",
    "2m":  "2Min",
    "5m":  "5Min",
    "15m": "15Min",
    "30m": "30Min",
    "1h":  "1Hour",
    "60m": "1Hour",
    "1d":  "1Day",
    "1wk": "1Week",
    "1w":  "1Week",
    "1mo": "1Month",
}


class AlpacaCredentialsMissing(RuntimeError):
    """Raised when ALPACA_API_KEY / ALPACA_API_SECRET aren't in the env.

    Distinct from a generic Exception so the dispatcher in
    ``equity_data_fetcher`` can catch this specific failure mode and
    fall back to yfinance instead of treating it as a data error.
    """


def _credentials() -> tuple[str, str]:
    """Read API key + secret from env; raise if either is missing."""
    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_API_SECRET")
    if not key or not secret:
        raise AlpacaCredentialsMissing(
            "ALPACA_API_KEY and ALPACA_API_SECRET must be set in the "
            "environment to use the Alpaca equity data adapter"
        )
    return key, secret


def _normalise_dataframe(bars: list[dict], tz_localize_naive: bool = False) -> pd.DataFrame:
    """Turn a list of Alpaca bar dicts into the DataFrame contract that
    the rest of the backtester expects.

    Alpaca bar shape: {"t": "2024-06-01T13:30:00Z", "o": .., "h": .., "l": ..,
                       "c": .., "v": .., "n": <trade_count>, "vw": <vwap>}

    Contract (must match equity_data_fetcher.load_equity_ohlcv):
      * tz-aware DatetimeIndex in UTC, named "datetime"
      * columns: open, high, low, close, volume (float64)
      * sorted ascending by timestamp
      * NaN close rows dropped (defensive — Alpaca doesn't normally
        emit them but our contract guarantees no NaN close).
    """
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame(bars)
    # Alpaca returns ISO-8601 strings with 'Z' suffix — pandas parses
    # them as tz-aware UTC.
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df = df.rename(columns={"t": "datetime", "o": "open", "h": "high",
                            "l": "low", "c": "close", "v": "volume"})
    keep = ["datetime", "open", "high", "low", "close", "volume"]
    missing = [k for k in keep if k not in df.columns]
    if missing:
        raise ValueError(f"alpaca payload missing columns: {missing}; got {list(df.columns)}")
    df = df[keep].astype({c: "float64" for c in ("open", "high", "low", "close", "volume")})
    df = df.set_index("datetime").sort_index()
    df = df.dropna(subset=["close"])
    return df


def _request_bars(symbol: str, alpaca_tf: str, start: Optional[str],
                  end: Optional[str], key: str, secret: str,
                  feed: str = "iex") -> list[dict]:
    """Fetch ALL bars for symbol/timeframe/window — auto-paginates.

    Alpaca caps each response at 10_000 bars and returns
    ``next_page_token`` to continue. For 1m on a 2-year window we
    expect 5-6 pages per ticker; the throttle inside ``load_equity_ohlcv``
    keeps us well under the 200 req/min rate limit even with all 19
    Robinhood tickers.

    Returns the merged ``bars`` list across all pages, or [] on any
    non-200 response (caller treats empty as "no data, skip strategy").
    """
    url = _BASE_URL.format(symbol=symbol)
    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }
    params = {
        "timeframe": alpaca_tf,
        "feed": feed,
        "limit": 10000,
        "adjustment": "raw",  # match yfinance auto_adjust=False — raw close
    }
    if start:
        params["start"] = start
    if end:
        params["end"] = end

    all_bars: list[dict] = []
    page_token: Optional[str] = None
    pages = 0
    max_pages = 200  # hard ceiling — protects against runaway pagination
    while pages < max_pages:
        if page_token:
            params["page_token"] = page_token
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
        except requests.RequestException as exc:
            print(f"[alpaca] {symbol} {alpaca_tf} network error: {exc}",
                  file=sys.stderr)
            return all_bars  # return what we have — caller may still proceed
        if resp.status_code == 401 or resp.status_code == 403:
            # Credentials invalid → caller should NOT retry with these keys.
            raise AlpacaCredentialsMissing(
                f"alpaca rejected the API key (HTTP {resp.status_code}); "
                f"check ALPACA_API_KEY / ALPACA_API_SECRET"
            )
        if resp.status_code == 429:
            # Rate limit — back off and retry once.
            print(f"[alpaca] {symbol}: 429 rate limit, sleeping 5s",
                  file=sys.stderr)
            time.sleep(5)
            continue
        if resp.status_code != 200:
            print(f"[alpaca] {symbol} {alpaca_tf} HTTP {resp.status_code}: "
                  f"{resp.text[:200]}", file=sys.stderr)
            return all_bars
        payload = resp.json()
        bars = payload.get("bars") or []
        all_bars.extend(bars)
        page_token = payload.get("next_page_token")
        if not page_token:
            break
        pages += 1
        # Polite throttle between pages — ~10ms keeps us at 100 req/min
        # for a single ticker, well below the 200/min ceiling.
        time.sleep(0.01)
    return all_bars


@functools.lru_cache(maxsize=128)
def _cached_fetch(ticker: str, alpaca_tf: str, start: Optional[str],
                  end: Optional[str]) -> pd.DataFrame:
    """LRU cache around the paginated fetch + normalise."""
    try:
        key, secret = _credentials()
    except AlpacaCredentialsMissing:
        # Re-raise so the caller's `except AlpacaCredentialsMissing`
        # branches into yfinance. Don't swallow.
        raise
    try:
        bars = _request_bars(ticker, alpaca_tf, start, end, key, secret)
    except AlpacaCredentialsMissing:
        raise
    except Exception as exc:
        print(f"[alpaca] {ticker} {alpaca_tf} failed: {exc}", file=sys.stderr)
        return pd.DataFrame()
    return _normalise_dataframe(bars)


def load_equity_ohlcv(
    ticker: str,
    timeframe: str = "1d",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load OHLCV for an equity ticker via Alpaca.

    Same shape + semantics as ``equity_data_fetcher.load_equity_ohlcv``
    so the dispatcher can swap providers transparently.

    Raises:
        AlpacaCredentialsMissing: if API keys aren't in the env. Caller
            (``equity_data_fetcher``) catches this and falls back to
            yfinance — so a Bot running without Alpaca keys still works,
            just with reduced intraday history.

    Returns an empty DataFrame on any other failure (network, HTTP non-
    auth errors) so the backtest_portfolio runner reads it as "no data".
    """
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return pd.DataFrame()

    alpaca_tf = _TIMEFRAME_MAP.get(timeframe.lower())
    if not alpaca_tf:
        print(f"[alpaca] unknown timeframe '{timeframe}' for {ticker}",
              file=sys.stderr)
        return pd.DataFrame()

    # Light throttle to stay below 200 req/min even when called in a
    # tight loop across 19 tickers × multiple timeframes. 50ms is below
    # human-perception but spaces requests.
    time.sleep(0.05)
    return _cached_fetch(ticker, alpaca_tf, start_date, end_date)


def clear_cache() -> None:
    """Drop the in-memory cache. Tests use this between runs."""
    _cached_fetch.cache_clear()


def is_available() -> bool:
    """Cheap precheck — true iff credentials are in the environment.

    Used by ``equity_data_fetcher`` to decide whether to route to Alpaca
    or fall back to yfinance, WITHOUT making an actual HTTP request.
    """
    try:
        _credentials()
        return True
    except AlpacaCredentialsMissing:
        return False
