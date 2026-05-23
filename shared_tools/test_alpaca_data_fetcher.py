"""Unit tests for shared_tools/alpaca_data_fetcher.py.

Mocks the HTTP layer (`requests.get`) so the suite runs offline. The
intent is to lock down:
  * the auth contract (header names, credential failure path)
  * the pagination protocol (next_page_token loop)
  * the DataFrame normalisation contract — downstream consumers
    (backtest_portfolio) read these columns by name and assume UTC
    DatetimeIndex
"""
import os
import sys

import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import alpaca_data_fetcher as adf  # noqa: E402


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def _bar(t: str, o: float = 100.0, h: float = 101.0,
         l: float = 99.0, c: float = 100.5, v: float = 1000.0) -> dict:
    """Build an Alpaca-shape bar dict."""
    return {"t": t, "o": o, "h": h, "l": l, "c": c, "v": v, "n": 42, "vw": c}


class _MockResponse:
    """Minimal stand-in for requests.Response that mocker.patch can return."""
    def __init__(self, status_code: int = 200, json_data: dict | None = None,
                 text: str = ""):
        self.status_code = status_code
        self._json = json_data or {}
        self.text = text

    def json(self):
        return self._json


@pytest.fixture(autouse=True)
def _clear_cache():
    """LRU cache leaks between tests otherwise — same ticker/timeframe
    args would return a cached DataFrame from the previous test's mock."""
    adf.clear_cache()
    yield
    adf.clear_cache()


@pytest.fixture
def _alpaca_env(monkeypatch):
    """Provide fake credentials so _credentials() doesn't raise."""
    monkeypatch.setenv("ALPACA_API_KEY", "test-key-id")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-secret")


# --------------------------------------------------------------------
# Credentials gating
# --------------------------------------------------------------------

def test_credentials_missing_raises(monkeypatch):
    """Without env vars, _credentials() raises — caller routes to yfinance."""
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)
    with pytest.raises(adf.AlpacaCredentialsMissing):
        adf._credentials()


def test_credentials_only_key_set_raises(monkeypatch):
    """Half-set creds (e.g. operator forgot the secret) must fail loudly."""
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)
    with pytest.raises(adf.AlpacaCredentialsMissing):
        adf._credentials()


def test_credentials_only_secret_set_raises(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.setenv("ALPACA_API_SECRET", "s")
    with pytest.raises(adf.AlpacaCredentialsMissing):
        adf._credentials()


def test_is_available_true_when_creds_set(_alpaca_env):
    assert adf.is_available() is True


def test_is_available_false_when_creds_missing(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)
    assert adf.is_available() is False


def test_load_without_creds_raises_via_cached_fetch(monkeypatch):
    """The cached fetch path must re-raise AlpacaCredentialsMissing so
    the dispatcher in equity_data_fetcher can branch on it."""
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)
    with pytest.raises(adf.AlpacaCredentialsMissing):
        adf.load_equity_ohlcv("AAPL", "1d", "2024-01-01")


# --------------------------------------------------------------------
# DataFrame normalisation
# --------------------------------------------------------------------

def test_normalise_renames_alpaca_keys_to_lowercase_ohlcv():
    bars = [_bar("2024-06-01T13:30:00Z", 100.0, 101.0, 99.5, 100.7, 12345)]
    out = adf._normalise_dataframe(bars)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    # Index is tz-aware UTC, named "datetime".
    assert str(out.index.tz) == "UTC"
    assert out.index.name == "datetime"
    assert out.iloc[0]["close"] == 100.7


def test_normalise_dtypes_are_float64_for_pandas_math_compat():
    """Downstream backtester compares against crypto OHLCV which is
    float64 — mismatched dtypes silently broadcast strangely."""
    bars = [_bar("2024-06-01T13:30:00Z")]
    out = adf._normalise_dataframe(bars)
    for col in ("open", "high", "low", "close", "volume"):
        assert out[col].dtype.name == "float64", f"{col} dtype must be float64"


def test_normalise_sorts_ascending_even_if_payload_unordered():
    """Alpaca normally returns ascending, but the contract is explicit
    so we enforce it regardless."""
    bars = [
        _bar("2024-06-03T13:30:00Z", c=102.0),
        _bar("2024-06-01T13:30:00Z", c=100.0),
        _bar("2024-06-02T13:30:00Z", c=101.0),
    ]
    out = adf._normalise_dataframe(bars)
    closes = list(out["close"])
    assert closes == [100.0, 101.0, 102.0]


def test_normalise_empty_bars_returns_empty_df():
    assert adf._normalise_dataframe([]).empty


def test_normalise_drops_nan_close_rows():
    """Defensive — Alpaca normally never emits NaN, but the contract
    promises no NaN closes so consumers can rely on it."""
    bars = [
        _bar("2024-06-01T13:30:00Z", c=100.0),
        _bar("2024-06-01T13:31:00Z", c=float("nan")),
    ]
    out = adf._normalise_dataframe(bars)
    assert len(out) == 1


# --------------------------------------------------------------------
# HTTP request + pagination
# --------------------------------------------------------------------

def test_request_bars_sends_auth_headers_and_params(mocker, _alpaca_env):
    """The auth headers (APCA-API-KEY-ID / -SECRET-KEY) are the spec —
    any rename would silently 401 across the board."""
    resp = _MockResponse(200, {"bars": [_bar("2024-06-01T13:30:00Z")], "next_page_token": None})
    mock_get = mocker.patch.object(adf.requests, "get", return_value=resp)

    adf._request_bars("AAPL", "1Min", "2024-01-01", "2024-06-01",
                      key="kk", secret="ss")

    mock_get.assert_called_once()
    call_kwargs = mock_get.call_args.kwargs
    assert call_kwargs["headers"]["APCA-API-KEY-ID"] == "kk"
    assert call_kwargs["headers"]["APCA-API-SECRET-KEY"] == "ss"
    assert call_kwargs["params"]["timeframe"] == "1Min"
    assert call_kwargs["params"]["start"] == "2024-01-01"
    assert call_kwargs["params"]["end"] == "2024-06-01"
    # Free tier defaults to IEX feed.
    assert call_kwargs["params"]["feed"] == "iex"
    # Raw close matches yfinance auto_adjust=False (parity with crypto path).
    assert call_kwargs["params"]["adjustment"] == "raw"


def test_request_bars_follows_pagination(mocker, _alpaca_env):
    """Alpaca caps at 10k bars per response; we must follow next_page_token
    until it's None to assemble the full window."""
    responses = [
        _MockResponse(200, {"bars": [_bar("2024-06-01T13:30:00Z")], "next_page_token": "tok-a"}),
        _MockResponse(200, {"bars": [_bar("2024-06-01T13:31:00Z")], "next_page_token": "tok-b"}),
        _MockResponse(200, {"bars": [_bar("2024-06-01T13:32:00Z")], "next_page_token": None}),
    ]
    mocker.patch.object(adf.requests, "get", side_effect=responses)
    bars = adf._request_bars("AAPL", "1Min", "2024-01-01", None,
                             key="kk", secret="ss")
    # All three pages stitched together.
    assert len(bars) == 3


def test_request_bars_401_raises_credentials_missing(mocker, _alpaca_env):
    """Invalid keys must surface so the operator can rotate them —
    silently returning empty would look like a data outage."""
    mocker.patch.object(adf.requests, "get",
                        return_value=_MockResponse(401, text="forbidden"))
    with pytest.raises(adf.AlpacaCredentialsMissing):
        adf._request_bars("AAPL", "1Min", "2024-01-01", None,
                          key="kk", secret="ss")


def test_request_bars_403_also_raises_credentials_missing(mocker, _alpaca_env):
    """Alpaca uses 403 for some auth-related rejections — treat the same."""
    mocker.patch.object(adf.requests, "get",
                        return_value=_MockResponse(403, text="forbidden"))
    with pytest.raises(adf.AlpacaCredentialsMissing):
        adf._request_bars("AAPL", "1Min", "2024-01-01", None,
                          key="kk", secret="ss")


def test_request_bars_429_retries_after_backoff(mocker, _alpaca_env):
    """Rate-limit responses get a single backoff retry. We patch sleep
    so the test doesn't actually wait 5s."""
    mocker.patch.object(adf.time, "sleep", return_value=None)
    responses = [
        _MockResponse(429),
        _MockResponse(200, {"bars": [_bar("2024-06-01T13:30:00Z")], "next_page_token": None}),
    ]
    mocker.patch.object(adf.requests, "get", side_effect=responses)
    bars = adf._request_bars("AAPL", "1Min", "2024-01-01", None,
                             key="kk", secret="ss")
    assert len(bars) == 1


def test_request_bars_network_error_returns_partial(mocker, _alpaca_env, capsys):
    """Network errors mid-pagination should not lose what we already
    got — return the partial so the operator can decide whether to retry."""
    bars_page1 = [_bar("2024-06-01T13:30:00Z")]
    responses = [
        _MockResponse(200, {"bars": bars_page1, "next_page_token": "tok-a"}),
    ]
    # First call succeeds, second raises.
    def side_effect(*a, **kw):
        if responses:
            return responses.pop(0)
        raise adf.requests.ConnectionError("network down")
    mocker.patch.object(adf.requests, "get", side_effect=side_effect)
    bars = adf._request_bars("AAPL", "1Min", "2024-01-01", None,
                             key="kk", secret="ss")
    assert bars == bars_page1
    assert "network" in capsys.readouterr().err.lower()


# --------------------------------------------------------------------
# load_equity_ohlcv (end-to-end)
# --------------------------------------------------------------------

def test_load_equity_ohlcv_happy_path(mocker, _alpaca_env):
    mocker.patch.object(adf.time, "sleep", return_value=None)
    payload = {"bars": [
        _bar("2024-06-01T13:30:00Z", o=100.0, c=100.7),
        _bar("2024-06-01T13:31:00Z", o=100.7, c=101.1),
    ], "next_page_token": None}
    mocker.patch.object(adf.requests, "get",
                        return_value=_MockResponse(200, payload))
    out = adf.load_equity_ohlcv("AAPL", timeframe="1m", start_date="2024-01-01")
    assert not out.empty
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert len(out) == 2
    assert str(out.index.tz) == "UTC"


def test_load_equity_ohlcv_unknown_timeframe_returns_empty(_alpaca_env, capsys):
    out = adf.load_equity_ohlcv("AAPL", timeframe="3h7m", start_date="2024-01-01")
    assert out.empty
    assert "unknown timeframe" in capsys.readouterr().err


def test_load_equity_ohlcv_empty_ticker_returns_empty(_alpaca_env):
    assert adf.load_equity_ohlcv("").empty
    assert adf.load_equity_ohlcv("   ").empty


def test_load_equity_ohlcv_caches_repeated_calls(mocker, _alpaca_env):
    """LRU cache must suppress repeat HTTP for the same (ticker, tf,
    start, end) — important for the portfolio runner which may invoke
    the same ticker across multiple strategies."""
    mocker.patch.object(adf.time, "sleep", return_value=None)
    mock_get = mocker.patch.object(
        adf.requests, "get",
        return_value=_MockResponse(200, {"bars": [_bar("2024-06-01T13:30:00Z")], "next_page_token": None}),
    )
    adf.load_equity_ohlcv("NVDA", timeframe="1d", start_date="2024-01-01")
    adf.load_equity_ohlcv("NVDA", timeframe="1d", start_date="2024-01-01")
    adf.load_equity_ohlcv("NVDA", timeframe="1d", start_date="2024-01-01")
    assert mock_get.call_count == 1


def test_load_equity_ohlcv_invalid_creds_returns_empty_not_crash(mocker, _alpaca_env, capsys):
    """A 401 from Alpaca mid-fetch surfaces as AlpacaCredentialsMissing
    out of _cached_fetch — load_equity_ohlcv lets it bubble to the
    dispatcher (equity_data_fetcher) which decides not to mask it."""
    mocker.patch.object(adf.time, "sleep", return_value=None)
    mocker.patch.object(adf.requests, "get",
                        return_value=_MockResponse(401, text="forbidden"))
    with pytest.raises(adf.AlpacaCredentialsMissing):
        adf.load_equity_ohlcv("AAPL", timeframe="1m", start_date="2024-01-01")


def test_load_equity_ohlcv_timeframe_mapping_uppercase_5m(mocker, _alpaca_env):
    """Ensure "5m" maps to Alpaca's "5Min" (their spec is title-case
    with the explicit "Min" suffix, not lowercase like our config)."""
    mocker.patch.object(adf.time, "sleep", return_value=None)
    captured = {}
    def capture(*a, **kw):
        captured["params"] = kw["params"]
        return _MockResponse(200, {"bars": [], "next_page_token": None})
    mocker.patch.object(adf.requests, "get", side_effect=capture)
    adf.load_equity_ohlcv("AAPL", timeframe="5m", start_date="2024-01-01")
    assert captured["params"]["timeframe"] == "5Min"


def test_load_equity_ohlcv_timeframe_mapping_1h_to_1hour(mocker, _alpaca_env):
    """Ensure "1h" maps to Alpaca's "1Hour" — easy off-by-one for
    operators who'd expect "60Min"."""
    mocker.patch.object(adf.time, "sleep", return_value=None)
    captured = {}
    def capture(*a, **kw):
        captured["params"] = kw["params"]
        return _MockResponse(200, {"bars": [], "next_page_token": None})
    mocker.patch.object(adf.requests, "get", side_effect=capture)
    adf.load_equity_ohlcv("SPY", timeframe="1h", start_date="2024-01-01")
    assert captured["params"]["timeframe"] == "1Hour"
