"""Equity OHLCV adapter via yfinance.

Mirrors the public surface of ``shared_tools/data_fetcher.load_cached_data``
so the backtest framework can consume equity tickers (AAPL, NVDA, TSLA,
MSFT, SPY, ...) through the same Backtester class as crypto symbols.

Why a separate adapter rather than extending data_fetcher.py:
  * The crypto path goes through CCXT + a binanceus SQLite cache. That
    architecture doesn't map onto Yahoo Finance: no exchange-specific
    pairs, different intraday-history ceilings, different rate-limits.
  * Keeping the two paths separate lets the routing decision live in
    one place (scripts/backtest_portfolio.run_one) and avoids monkey-
    patching the CCXT helper.

yfinance limitations to be aware of when reading backtest results:
  * 1m intraday history is capped at the last 7 days.
  * 5m / 15m / 30m / 1h intraday history is capped at the last 60 days
    for sub-hour bars and 730 days for 1h. For a 5m equity backtest
    against a 2-year ``--since``, the framework will silently get only
    the most recent 60 days. That's still ~4.7k bars per ticker — a
    meaningful sample — but the report's ``period`` field will reflect
    the truncated window, not the requested one.
  * 1d is unlimited.
  * Returns are NOT adjusted for splits/dividends by default — we use
    the raw ``Close`` column (not ``Adj Close``) because the existing
    crypto path uses raw close, and reporting / win-rate stats should
    match the trade history a real broker would have produced.

Output contract — DataFrame with:
  * tz-aware DatetimeIndex in UTC (named ``datetime``)
  * columns: ``open``, ``high``, ``low``, ``close``, ``volume`` (float64)

In-memory LRU cache keyed by (ticker, timeframe, start) — a single
process can run many backtests without re-fetching the same window.
yfinance has its own on-disk request cache too, so a cold start still
amortises quickly.
"""
from __future__ import annotations

import functools
import sys
import time
from typing import Optional

import pandas as pd

# yfinance is declared in pyproject.toml [dependencies]; importing at
# module load is correct because ``uv sync`` installs it before any
# test or backtest invocation. If a downstream caller wants to gate
# on availability, they should wrap the import themselves.
import yfinance as yf

# Phase 7e: Alpaca is the preferred equity data source because yfinance
# caps intraday history at 7/60 days, which makes 2-year backtests
# impossible. We import it lazily inside load_equity_ohlcv so callers
# without Alpaca creds keep the pure-yfinance path with zero overhead.


# Map our codebase's timeframe strings (matching shared_tools/data_fetcher
# and the strategy configs) onto yfinance's interval strings.
_TIMEFRAME_MAP = {
    "1m": "1m",
    "2m": "2m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "60m": "1h",
    "90m": "90m",
    "1d": "1d",
    "5d": "5d",
    "1w": "1wk",
    "1wk": "1wk",
    "1mo": "1mo",
}

# Per yfinance's own constraints — used to surface a clear log when the
# requested ``start`` window exceeds what intraday data the API will
# return. Backtests still proceed with the truncated window; we just
# annotate the user so they know the report's ``period`` won't match
# the ``--since`` they passed.
_MAX_HISTORY_DAYS = {
    "1m": 7,
    "2m": 60,
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "1h": 730,
    "1d": None,   # unlimited
    "1wk": None,
    "1mo": None,
}


def _normalise_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame matching the crypto-path contract.

    * Lower-case the OHLCV column names.
    * Drop ``Adj Close`` if present (we use raw Close — see module docstring).
    * Convert the index to UTC and rename to ``datetime``.
    * Cast OHLCV columns to ``float64`` so downstream pandas math is
      consistent across data sources.
    * Drop any rows where Close is NaN (yfinance occasionally returns
      NaN for non-trading minutes around half-days / early closes).

    yfinance can return a MultiIndex column structure when downloading
    multiple tickers in one call. We always fetch one ticker at a time,
    but flatten defensively in case the API behaviour changes.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    # Flatten MultiIndex (just in case).
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    if "adj_close" in df.columns:
        df = df.drop(columns=["adj_close"])

    keep = ["open", "high", "low", "close", "volume"]
    missing = [k for k in keep if k not in df.columns]
    if missing:
        raise ValueError(f"yfinance payload missing columns: {missing}; got {list(df.columns)}")
    df = df[keep].astype("float64")

    # tz-aware → UTC; tz-naive → assume UTC for consistency with the
    # crypto path (CCXT returns UTC timestamps).
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "datetime"
    df = df.dropna(subset=["close"])
    return df


@functools.lru_cache(maxsize=128)
def _cached_fetch(ticker: str, yf_interval: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    """yfinance.download wrapper with in-process LRU cache.

    Returns the normalised DataFrame so cache hits are zero-copy except
    for the lru_cache machinery.
    """
    try:
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            interval=yf_interval,
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,  # serial download — predictable for tests
        )
    except Exception as exc:
        print(f"[equity-fetch] {ticker} {yf_interval} failed: {exc}", file=sys.stderr)
        return pd.DataFrame()
    return _normalise_dataframe(raw)


def _try_alpaca(ticker: str, timeframe: str, start_date: Optional[str],
                end_date: Optional[str]) -> Optional[pd.DataFrame]:
    """Try the Alpaca adapter if creds are present.

    Returns:
        None if Alpaca is not configured (caller falls back to yfinance).
        Empty DataFrame on Alpaca-specific data failure (caller keeps the
        Alpaca verdict — no automatic yfinance fallback, otherwise an
        operator with bad keys would silently get truncated yfinance data
        and assume Alpaca worked).
        Non-empty DataFrame on success.

    Lazy-imported so the pure-yfinance path has no startup overhead.
    """
    try:
        # Local import keeps the yfinance-only path light. The module
        # exists in shared_tools/ alongside this one, so the import path
        # works whether callers put shared_tools/ on sys.path or import
        # via the package.
        import alpaca_data_fetcher as alpaca
    except ImportError:
        return None  # shouldn't happen — file is checked in — but be defensive
    if not alpaca.is_available():
        return None
    try:
        return alpaca.load_equity_ohlcv(ticker, timeframe, start_date, end_date)
    except alpaca.AlpacaCredentialsMissing:
        # Keys were present in env but rejected by the API → don't fall
        # through to yfinance silently. Treat it as a data error so the
        # operator notices and rotates the key.
        print(f"[equity-fetch] {ticker}: Alpaca creds rejected, "
              f"NOT falling back to yfinance (truncated history would mislead)",
              file=sys.stderr)
        return pd.DataFrame()
    except Exception as exc:
        # Network or library bug — fall through to yfinance so the bot
        # keeps working even with a partial Alpaca outage.
        print(f"[equity-fetch] {ticker}: Alpaca error '{exc}', "
              f"falling back to yfinance", file=sys.stderr)
        return None


def load_equity_ohlcv(
    ticker: str,
    timeframe: str = "1d",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load OHLCV for an equity ticker.

    Provider precedence (Phase 7e):
      1. Alpaca, if ``ALPACA_API_KEY`` + ``ALPACA_API_SECRET`` are set in
         the env. Free tier gives 5+ years of 1m history vs yfinance's
         7-day cap.
      2. yfinance fallback. Limited intraday history but no auth needed.

    Args:
        ticker: bare ticker (``"AAPL"``, ``"NVDA"``). Not a pair.
        timeframe: human label like ``"5m"`` / ``"1h"`` / ``"1d"``.
        start_date: ``"YYYY-MM-DD"`` (inclusive); ``None`` lets the
            provider pick a sensible default.
        end_date: ``"YYYY-MM-DD"`` (exclusive). ``None`` = up to now.

    Returns an empty DataFrame on any failure so callers can degrade
    gracefully (the backtest portfolio runner already treats empty data
    as a skip with reason "no data").
    """
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return pd.DataFrame()

    # Try Alpaca first — only matters when creds are set.
    alpaca_df = _try_alpaca(ticker, timeframe, start_date, end_date)
    if alpaca_df is not None:
        # Either non-empty success OR a credentials-rejected empty
        # response we should NOT mask with yfinance.
        return alpaca_df

    yf_interval = _TIMEFRAME_MAP.get(timeframe.lower())
    if not yf_interval:
        print(f"[equity-fetch] unknown timeframe '{timeframe}' for {ticker}", file=sys.stderr)
        return pd.DataFrame()

    # When the caller asks for more history than yfinance will return
    # for this interval, surface a one-line note. We do NOT clip the
    # ``start`` ourselves — yfinance silently truncates and we want to
    # preserve their semantics rather than second-guessing.
    cap = _MAX_HISTORY_DAYS.get(yf_interval)
    if cap is not None and start_date:
        try:
            requested = pd.Timestamp(start_date)
            # pd.Timestamp.utcnow() was deprecated in pandas 4.x — use
            # Timestamp.now("UTC") and strip the tz to subtract from a
            # tz-naive parsed start. Both sides need to be tz-naive for
            # the timedelta subtraction.
            age_days = (pd.Timestamp.now("UTC").tz_localize(None) - requested.tz_localize(None)).days
            if age_days > cap:
                print(
                    f"[equity-fetch] {ticker} {yf_interval}: requested "
                    f"{age_days}d back, yfinance returns at most {cap}d for this interval",
                    file=sys.stderr,
                )
        except Exception:
            # Date parsing failures here are non-fatal — proceed.
            pass

    # Light throttle: yfinance is generous but back-to-back hits for
    # every strategy in a 30-ticker portfolio can briefly rate-limit.
    # 50ms is below human-perception but spaces requests enough.
    time.sleep(0.05)
    return _cached_fetch(ticker, yf_interval, start_date, end_date)


def clear_cache() -> None:
    """Drop the in-memory cache. Tests use this between runs."""
    _cached_fetch.cache_clear()
