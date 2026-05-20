"""Unit tests for shared_tools/equity_data_fetcher.py.

Tests use pytest-mock (already in [project.optional-dependencies].test)
to stub out yfinance.download so the suite runs offline. The intent is
to lock down the normalisation contract — what callers downstream
(backtest_portfolio.run_equity_backtest) can rely on — not to validate
yfinance itself.
"""
import os
import sys

import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# We patch yfinance.download via pytest-mock fixture; the import below
# only matters so equity_data_fetcher's `import yfinance as yf` resolves
# during collection. The actual symbol we patch is
# equity_data_fetcher.yf.download (the module-local reference).
import equity_data_fetcher as edf  # noqa: E402


def _yf_payload(rows):
    """Build a fake yfinance.download() DataFrame from a list of dicts."""
    return pd.DataFrame(rows).set_index("Datetime")


@pytest.fixture(autouse=True)
def _clear_cache():
    """Ensure the LRU cache doesn't leak between tests."""
    edf.clear_cache()
    yield
    edf.clear_cache()


def test_normalise_dataframe_lowercases_and_drops_adj_close():
    raw = pd.DataFrame({
        "Open": [100.0],
        "High": [101.0],
        "Low": [99.5],
        "Close": [100.7],
        "Adj Close": [100.7],
        "Volume": [123456.0],
    }, index=pd.DatetimeIndex(["2024-06-01 13:30"], tz="US/Eastern", name="Datetime"))
    out = edf._normalise_dataframe(raw)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert "adj_close" not in out.columns
    # tz converted to UTC.
    assert str(out.index.tz) == "UTC"


def test_normalise_dataframe_handles_tz_naive_index():
    """Some yfinance call paths return tz-naive timestamps; we localise
    them to UTC so the backtester (which assumes UTC throughout) stays
    consistent with the crypto path."""
    raw = pd.DataFrame({
        "Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [1.0],
    }, index=pd.DatetimeIndex(["2024-06-01"]))
    out = edf._normalise_dataframe(raw)
    assert str(out.index.tz) == "UTC"


def test_normalise_dataframe_drops_nan_close_rows():
    raw = pd.DataFrame({
        "Open": [1.0, 2.0],
        "High": [1.0, 2.0],
        "Low": [1.0, 2.0],
        "Close": [1.0, float("nan")],
        "Volume": [10.0, 20.0],
    }, index=pd.DatetimeIndex(["2024-06-01", "2024-06-02"], tz="UTC", name="Datetime"))
    out = edf._normalise_dataframe(raw)
    assert len(out) == 1


def test_normalise_dataframe_empty_input_yields_empty_output():
    out = edf._normalise_dataframe(pd.DataFrame())
    assert out.empty


def test_normalise_dataframe_missing_columns_raises():
    raw = pd.DataFrame({
        "Open": [1.0],
        "High": [1.0],
        # missing Low, Close, Volume
    }, index=pd.DatetimeIndex(["2024-06-01"], tz="UTC", name="Datetime"))
    with pytest.raises(ValueError, match="missing columns"):
        edf._normalise_dataframe(raw)


def test_load_equity_ohlcv_unknown_timeframe_returns_empty(capsys):
    out = edf.load_equity_ohlcv("AAPL", timeframe="3h7m")
    assert out.empty
    assert "unknown timeframe" in capsys.readouterr().err


def test_load_equity_ohlcv_empty_ticker_returns_empty():
    assert edf.load_equity_ohlcv("").empty
    assert edf.load_equity_ohlcv("   ").empty


def test_load_equity_ohlcv_calls_yfinance_with_correct_args(mocker):
    fake_df = pd.DataFrame({
        "Open": [100.0],
        "High": [101.0],
        "Low": [99.0],
        "Close": [100.5],
        "Volume": [1_000_000.0],
    }, index=pd.DatetimeIndex(["2024-06-01 13:30"], tz="UTC", name="Datetime"))
    mock_dl = mocker.patch.object(edf.yf, "download", return_value=fake_df)

    out = edf.load_equity_ohlcv("AAPL", timeframe="5m", start_date="2024-04-01")
    assert not out.empty
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    mock_dl.assert_called_once()
    kwargs = mock_dl.call_args.kwargs
    assert kwargs["interval"] == "5m"
    assert kwargs["start"] == "2024-04-01"
    assert kwargs["auto_adjust"] is False


def test_load_equity_ohlcv_caches_repeated_calls(mocker):
    fake_df = pd.DataFrame({
        "Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [1.0],
    }, index=pd.DatetimeIndex(["2024-06-01"], tz="UTC", name="Datetime"))
    mock_dl = mocker.patch.object(edf.yf, "download", return_value=fake_df)

    edf.load_equity_ohlcv("NVDA", timeframe="1d", start_date="2024-01-01")
    edf.load_equity_ohlcv("NVDA", timeframe="1d", start_date="2024-01-01")
    edf.load_equity_ohlcv("NVDA", timeframe="1d", start_date="2024-01-01")
    assert mock_dl.call_count == 1, "LRU cache should suppress repeat fetches"


def test_load_equity_ohlcv_intraday_history_cap_warning(mocker, capsys):
    """When the requested start exceeds yfinance's intraday ceiling
    (60 days for 5m), we emit a one-line warning to stderr but still
    return whatever yfinance gives us."""
    fake_df = pd.DataFrame({
        "Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [1.0],
    }, index=pd.DatetimeIndex(["2024-06-01"], tz="UTC", name="Datetime"))
    mocker.patch.object(edf.yf, "download", return_value=fake_df)

    # 5m has a 60-day cap; request 2 years back.
    far_back = (pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(days=730)).strftime("%Y-%m-%d")
    edf.load_equity_ohlcv("AAPL", timeframe="5m", start_date=far_back)
    err = capsys.readouterr().err
    assert "60d" in err or "60 d" in err


def test_load_equity_ohlcv_yfinance_exception_returns_empty(mocker, capsys):
    """Network / API failures must not propagate — they degrade to an
    empty DataFrame which the backtest_portfolio runner reads as 'skip'."""
    mocker.patch.object(edf.yf, "download", side_effect=RuntimeError("yfinance HTTP 500"))
    out = edf.load_equity_ohlcv("AAPL", timeframe="1d", start_date="2024-01-01")
    assert out.empty
    assert "failed" in capsys.readouterr().err.lower()


def test_load_equity_ohlcv_normalises_us_eastern_to_utc(mocker):
    """Real yfinance intraday data arrives in US/Eastern; the adapter
    must convert to UTC so downstream pandas math against crypto data
    (UTC) is comparable."""
    raw = pd.DataFrame({
        "Open": [100.0],
        "High": [101.0],
        "Low": [99.5],
        "Close": [100.7],
        "Adj Close": [100.7],
        "Volume": [1_000_000.0],
    }, index=pd.DatetimeIndex(["2024-06-03 09:30"], tz="US/Eastern", name="Datetime"))
    mocker.patch.object(edf.yf, "download", return_value=raw)
    out = edf.load_equity_ohlcv("AAPL", timeframe="5m", start_date="2024-04-01")
    assert str(out.index.tz) == "UTC"
    # 09:30 ET on June 3 = 13:30 UTC (DST in effect).
    assert out.index[0].hour == 13
