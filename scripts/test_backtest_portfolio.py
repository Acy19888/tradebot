#!/usr/bin/env python3
"""Unit tests for scripts/backtest_portfolio.py — pure helpers only.

Pytest target. Run from repo root:
    uv run --no-sync python -m pytest scripts/test_backtest_portfolio.py -v

Tests intentionally avoid invoking the backtester (which would need
historical data + take minutes). The run_single_backtest integration is
covered by the existing backtest/tests/ suite already.
"""
import os
import sys

# Make scripts/ importable when invoked from repo root.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import backtest_portfolio as bp  # noqa: E402


def test_coin_to_symbol_crypto_appends_usdt():
    assert bp.coin_to_symbol("BTC", "perps", "hyperliquid") == "BTC/USDT"
    assert bp.coin_to_symbol("eth", "perps", "hyperliquid") == "ETH/USDT"
    assert bp.coin_to_symbol("SOL", "spot", "binanceus") == "SOL/USDT"


def test_coin_to_symbol_equity_passthrough():
    # Equities pass through so caller can detect + skip.
    for sym in ("AAPL", "NVDA", "TSLA", "SPY", "MSFT"):
        assert bp.coin_to_symbol(sym, "spot", "robinhood") == sym


def test_coin_to_symbol_preexisting_pair_preserved():
    assert bp.coin_to_symbol("BTC/USDT", "perps", "hyperliquid") == "BTC/USDT"


def test_derive_kwargs_hyperliquid_perps():
    sc = {
        "id": "hl-momentum-btc",
        "type": "perps",
        "script": "shared_scripts/check_hyperliquid.py",
        "args": ["momentum", "BTC", "1h", "--mode=paper"],
        "open_strategy": {"name": "momentum", "params": {"roc_period": 12}},
        "capital": 1000,
    }
    kw = bp.derive_backtest_kwargs(sc, "2023-01-01")
    assert kw is not None
    assert kw["strategy_name"] == "momentum"
    assert kw["symbol"] == "BTC/USDT"
    assert kw["timeframe"] == "1h"
    assert kw["capital"] == 1000
    assert kw["platform"] == "hyperliquid"
    assert kw["registry"] == "spot"
    assert kw["params"] == {"roc_period": 12}
    assert kw["since"] == "2023-01-01"


def test_derive_kwargs_robinhood_equity_returns_none():
    sc = {
        "id": "rh-momentum-aapl-5m",
        "type": "spot",
        "platform": "robinhood",
        "script": "shared_scripts/check_robinhood.py",
        "args": ["momentum", "AAPL", "5m", "--mode=paper"],
        "open_strategy": {"name": "momentum"},
        "capital": 100,
    }
    assert bp.derive_backtest_kwargs(sc, "2023-01-01") is None


def test_derive_kwargs_passes_stop_loss_settings():
    sc = {
        "id": "x",
        "type": "perps",
        "script": "shared_scripts/check_hyperliquid.py",
        "args": ["range_scalper", "ETH", "5m"],
        "open_strategy": {"name": "range_scalper"},
        "capital": 100,
        "stop_loss_atr_mult": 2.5,
    }
    kw = bp.derive_backtest_kwargs(sc, "2024-01-01")
    assert kw is not None
    assert kw["stop_loss_atr_mult"] == 2.5
    # Untouched optional fields stay out — keeps run_single_backtest's
    # defaults active.
    assert "stop_loss_pct" not in kw
    assert "trailing_stop_atr_mult" not in kw


def test_derive_kwargs_missing_args_returns_none():
    sc = {"id": "broken", "args": []}
    assert bp.derive_backtest_kwargs(sc, "2024-01-01") is None


def test_derive_kwargs_platform_inference_from_script():
    sc = {
        "id": "guess-okx",
        "type": "perps",
        "script": "shared_scripts/check_okx.py",
        "args": ["momentum", "BTC", "1h"],
        "open_strategy": {"name": "momentum"},
        "capital": 500,
    }
    kw = bp.derive_backtest_kwargs(sc, "2024-01-01")
    assert kw is not None
    assert kw["platform"] == "okx"


def test_run_one_skips_equities_without_calling_framework():
    sc = {
        "id": "rh-msft-skip",
        "type": "spot",
        "platform": "robinhood",
        "script": "shared_scripts/check_robinhood.py",
        "args": ["momentum", "MSFT", "5m"],
        "open_strategy": {"name": "momentum"},
        "capital": 100,
    }
    sid, result, reason = bp.run_one(sc, "2024-01-01")
    assert sid == "rh-msft-skip"
    assert result is None
    assert "MSFT" in reason


def test_metric_handles_missing_and_none():
    assert bp.metric(None, "sharpe_ratio") == 0
    assert bp.metric({}, "sharpe_ratio") == 0
    assert bp.metric({"sharpe_ratio": None}, "sharpe_ratio") == 0
    assert bp.metric({"sharpe_ratio": 1.5}, "sharpe_ratio") == 1.5
    assert bp.metric({}, "sharpe_ratio", default=-1) == -1


def test_write_per_strategy_report_skipped_writes_warning(tmp_path):
    sc = {
        "id": "rh-aapl",
        "type": "spot",
        "args": ["momentum", "AAPL", "5m"],
        "open_strategy": {"name": "momentum"},
        "capital": 100,
    }
    path = bp.write_per_strategy_report(
        str(tmp_path), "rh-aapl", sc, None,
        "equity ticker AAPL not in backtest data path",
    )
    assert os.path.exists(path)
    body = open(path).read()
    assert "Skipped" in body
    assert "AAPL" in body
    assert "Capital" in body  # config metadata is still rendered


def test_write_per_strategy_report_success_renders_metrics(tmp_path):
    sc = {
        "id": "hl-momentum-btc",
        "type": "perps",
        "args": ["momentum", "BTC", "1h"],
        "open_strategy": {"name": "momentum", "params": {"roc_period": 12}},
        "capital": 1000,
    }
    fake_result = {
        "strategy_name": "momentum",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "start_date": "2023-01-01",
        "end_date": "2024-12-31",
        "initial_capital": 1000,
        "final_capital": 1342.50,
        "total_return_pct": 34.25,
        "annual_return_pct": 16.8,
        "sharpe_ratio": 1.42,
        "sortino_ratio": 2.18,
        "max_drawdown_pct": 12.4,
        "calmar_ratio": 1.35,
        "total_trades": 78,
        "win_rate": 56.4,
        "profit_factor": 1.71,
        "volatility_pct": 11.8,
        "avg_win_pct": 2.4,
        "avg_loss_pct": -1.7,
        "trades": [],
        "params": {"roc_period": 12},
    }
    path = bp.write_per_strategy_report(str(tmp_path), "hl-momentum-btc", sc, fake_result, None)
    body = open(path).read()
    assert "1.420" in body or "1.42" in body  # sharpe
    assert "Total Return" in body
    assert "Max Drawdown" in body
    assert "78" in body  # trades count
    # The verbatim formatted block must be embedded too.
    assert "BACKTEST REPORT" in body


def test_write_ranking_report_sorts_by_sharpe(tmp_path):
    rows = [
        {"id": "a", "status": "ok", "sharpe": 0.5, "sortino": 0.7, "return_pct": 5.0,
         "max_dd_pct": 10.0, "win_rate": 50.0, "profit_factor": 1.1, "trades": 30,
         "symbol": "BTC/USDT", "timeframe": "1h", "type": "perps"},
        {"id": "b", "status": "ok", "sharpe": 1.8, "sortino": 2.4, "return_pct": 22.0,
         "max_dd_pct": 8.0, "win_rate": 60.0, "profit_factor": 1.8, "trades": 50,
         "symbol": "ETH/USDT", "timeframe": "4h", "type": "perps"},
        {"id": "c", "status": "skipped", "reason": "no data"},
    ]
    path = bp.write_ranking_report(str(tmp_path), rows, "2023-01-01")
    body = open(path).read()
    # B should appear before A (sorted by Sharpe DESC).
    assert body.index("| `b` |") < body.index("| `a` |")
    # Skip list rendered separately.
    assert "Skipped / Failed" in body
    assert "no data" in body
    # Top-trader guide is included.
    assert "Top-Trader Reading Guide" in body


def test_write_summary_json_round_trip(tmp_path):
    import json as _json
    rows = [{"id": "x", "status": "ok", "sharpe": 1.0}]
    path = bp.write_summary_json(str(tmp_path), rows)
    decoded = _json.load(open(path))
    assert decoded == rows
