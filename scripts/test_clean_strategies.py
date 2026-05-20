#!/usr/bin/env python3
"""Unit tests for scripts/clean_strategies.py.

Pure-helper coverage — no filesystem, no real config required.
Run from repo root:
    uv run --with pytest python -m pytest scripts/test_clean_strategies.py -v
"""
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import clean_strategies as cs  # noqa: E402


# --------------------------------------------------------------------
# classify()
# --------------------------------------------------------------------

def _row(**kwargs):
    """Test helper — produces a 'status=ok' summary row with sensible defaults."""
    base = {
        "id": "x",
        "status": "ok",
        "sharpe": 0.7,
        "return_pct": 20.0,
        "max_dd_pct": 15.0,
        "trades": 100,
    }
    base.update(kwargs)
    return base


def test_classify_keep_clean_winner():
    v, _ = cs.classify(_row(sharpe=1.5, return_pct=42, max_dd_pct=18, trades=50))
    assert v == cs.KEEP


def test_classify_discard_catastrophic_return():
    v, reason = cs.classify(_row(sharpe=-12.0, return_pct=-100.0, trades=7000))
    assert v == cs.DISCARD
    assert "catastrophic" in reason or "Sharpe" in reason


def test_classify_discard_significantly_negative_sharpe():
    v, _ = cs.classify(_row(sharpe=-1.0, return_pct=-5.0, max_dd_pct=10, trades=200))
    assert v == cs.DISCARD


def test_classify_discard_confirmed_negative_edge():
    """hl-amd-btc real case: Sharpe -0.21, Return -27.9% (above -30% so
    not 'catastrophic'), 136 trades. Sample is big enough that the
    negative Sharpe isn't noise — confirmed losing edge → DISCARD."""
    v, reason = cs.classify(_row(sharpe=-0.21, return_pct=-27.9, max_dd_pct=46.9, trades=136))
    assert v == cs.DISCARD
    assert "confirmed" in reason.lower() or "negative" in reason.lower()


def test_classify_discard_negative_sharpe_small_sample_is_only_tune_or_untested():
    """Sub-50 trades with a slightly negative Sharpe is NOT enough sample
    to call it broken — falls into the marginal band, not DISCARD."""
    v, _ = cs.classify(_row(sharpe=-0.3, return_pct=-5, max_dd_pct=20, trades=30))
    # Sharpe < -0.5 doesn't trigger (-0.3 > -0.5); trades < 50 so the new
    # 'confirmed negative edge' rule doesn't fire either. Should fall
    # through to the ambiguous-result TUNE fallback.
    assert v == cs.TUNE


def test_classify_discard_liquidation_level_drawdown_negative_sign():
    """hl-range-hype-5m real case: Sharpe 1.82, +329% return, but
    Max-DD -57.8% — the backtester stores the drawdown as a NEGATIVE
    number (e.g. -57.8). The classifier must use absolute magnitude,
    otherwise the threshold check never fires for the real data shape."""
    v, reason = cs.classify(_row(sharpe=1.82, return_pct=329.2, max_dd_pct=-57.8, trades=603))
    assert v == cs.DISCARD
    assert "liquid" in reason.lower() or "kill" in reason.lower()


def test_classify_discard_liquidation_level_drawdown_positive_sign():
    """Same threshold must fire when the source happens to store the
    drawdown as a positive magnitude (legacy/test data shape)."""
    v, _ = cs.classify(_row(sharpe=1.82, return_pct=329.2, max_dd_pct=57.8, trades=603))
    assert v == cs.DISCARD


def test_classify_max_dd_50_exact_boundary_is_tune_not_discard():
    """50.0% Max-DD is the strict cutoff (>50 triggers DISCARD). At 50
    exactly we should still see TUNE so the boundary behaviour is
    deterministic and matches the docs. Tested with both signs."""
    v, _ = cs.classify(_row(sharpe=1.0, return_pct=20, max_dd_pct=-50.0, trades=100))
    assert v == cs.TUNE
    v, _ = cs.classify(_row(sharpe=1.0, return_pct=20, max_dd_pct=50.0, trades=100))
    assert v == cs.TUNE


def test_classify_tune_high_drawdown_handles_negative_sign():
    """The TUNE rule (Max-DD > 30 with positive Sharpe) must also use
    abs() — covers the hl-momentum-btc real case (Max-DD -36.6%)."""
    v, _ = cs.classify(_row(sharpe=0.58, return_pct=42.6, max_dd_pct=-36.6, trades=25))
    # 25 trades < 30 so this fires the small-sample TUNE branch even
    # before max_dd does — but BOTH branches need abs() correctness.
    assert v == cs.TUNE
    # Now isolate the max_dd branch with enough trades + sharpe >= 0.5
    # so only the DD rule can flag it.
    v, _ = cs.classify(_row(sharpe=0.8, return_pct=25, max_dd_pct=-45, trades=100))
    assert v == cs.TUNE


def test_classify_suspicious_too_high_sharpe():
    v, reason = cs.classify(_row(sharpe=5.98, return_pct=68557.0, max_dd_pct=27.1, trades=1179))
    assert v == cs.SUSPICIOUS
    assert "lookahead" in reason.lower() or "anomaly" in reason.lower()


def test_classify_tune_marginal_sharpe():
    # Real case from the user's run: hl-momentum-btc Sharpe 0.58
    v, _ = cs.classify(_row(sharpe=0.58, return_pct=42.6, max_dd_pct=36.6, trades=25))
    # Sharpe is 0.58 (> 0.5), but Max-DD 36.6% > 30 → TUNE
    assert v == cs.TUNE


def test_classify_tune_too_few_trades():
    v, _ = cs.classify(_row(sharpe=0.7, return_pct=15, max_dd_pct=8, trades=12))
    assert v == cs.TUNE


def test_classify_tune_high_drawdown_positive_sharpe():
    v, _ = cs.classify(_row(sharpe=0.8, return_pct=25, max_dd_pct=45, trades=100))
    assert v == cs.TUNE


def test_classify_equity_from_skip_reason():
    row = {"id": "rh-aapl", "status": "skipped",
           "reason": "equity ticker AAPL not in backtest data path"}
    v, _ = cs.classify(row)
    assert v == cs.EQUITY


def test_classify_equity_from_strategy_symbol():
    # status=skipped with a generic reason but the config symbol is an equity
    row = {"id": "rh-aapl", "status": "skipped", "reason": "something else"}
    v, _ = cs.classify(row, strategy_arg_symbol="AAPL")
    assert v == cs.EQUITY


def test_classify_untested_skipped_with_unknown_reason():
    row = {"id": "weird", "status": "skipped", "reason": "backtest returned None"}
    v, _ = cs.classify(row, strategy_arg_symbol="BTC")
    assert v == cs.UNTESTED


def test_classify_discard_beats_suspicious():
    # When two conditions conflict (high sharpe but catastrophic return),
    # DISCARD wins because the hard-failure check comes first.
    v, _ = cs.classify(_row(sharpe=5.0, return_pct=-50.0, trades=100))
    assert v == cs.DISCARD


# --------------------------------------------------------------------
# cleanup_config()
# --------------------------------------------------------------------

def test_cleanup_removes_discard_keeps_others():
    cfg = {
        "config_version": 16,
        "strategies": [
            {"id": "good",     "args": ["momentum", "BTC", "1h"]},
            {"id": "bad",      "args": ["momentum", "ETH", "5m"]},
            {"id": "weird",    "args": ["vwap", "HYPE", "5m"]},
            {"id": "stock",    "args": ["momentum", "AAPL", "5m"]},
        ],
    }
    summary = [
        {"id": "good",  "status": "ok", "sharpe": 1.2, "return_pct": 30, "max_dd_pct": 12, "trades": 80},
        {"id": "bad",   "status": "ok", "sharpe": -5.0, "return_pct": -100, "max_dd_pct": 100, "trades": 9999},
        {"id": "weird", "status": "ok", "sharpe": 4.5, "return_pct": 1500, "max_dd_pct": 20, "trades": 800},
        {"id": "stock", "status": "skipped", "reason": "equity ticker AAPL not in backtest data path"},
    ]
    new_cfg, decisions = cs.cleanup_config(cfg, summary)
    ids = [s["id"] for s in new_cfg["strategies"]]
    assert "good" in ids
    assert "stock" in ids
    assert "bad" not in ids
    assert "weird" not in ids
    suspended = new_cfg.get("_suspended_strategies") or []
    assert [s["id"] for s in suspended] == ["weird"]
    by_id = {d["id"]: d for d in decisions}
    assert by_id["good"]["verdict"] == cs.KEEP
    assert by_id["bad"]["verdict"] == cs.DISCARD
    assert by_id["weird"]["verdict"] == cs.SUSPICIOUS
    assert by_id["stock"]["verdict"] == cs.EQUITY


def test_cleanup_preserves_unrelated_top_level_keys():
    cfg = {
        "config_version": 16,
        "telegram": {"enabled": True, "bot_token": ""},
        "portfolio_risk": {"max_drawdown_pct": 25},
        "strategies": [
            {"id": "good", "args": ["momentum", "BTC", "1h"]},
        ],
    }
    summary = [
        {"id": "good", "status": "ok", "sharpe": 1.0, "return_pct": 20, "max_dd_pct": 10, "trades": 60},
    ]
    new_cfg, _ = cs.cleanup_config(cfg, summary)
    # Sibling keys must be untouched — config-cleanup should never silently
    # drop notifier settings or portfolio_risk.
    assert new_cfg["telegram"] == cfg["telegram"]
    assert new_cfg["portfolio_risk"] == cfg["portfolio_risk"]
    assert new_cfg["config_version"] == 16


def test_cleanup_strategy_not_in_summary_is_untested():
    cfg = {"strategies": [{"id": "orphan", "args": ["x", "BTC", "1h"]}]}
    summary: list = []
    new_cfg, decisions = cs.cleanup_config(cfg, summary)
    assert [s["id"] for s in new_cfg["strategies"]] == ["orphan"]
    assert decisions[0]["verdict"] == cs.UNTESTED


def test_cleanup_prunes_empty_suspended_key():
    cfg = {
        "strategies": [{"id": "good", "args": ["momentum", "BTC", "1h"]}],
        "_suspended_strategies": [],  # leftover from a prior run
    }
    summary = [
        {"id": "good", "status": "ok", "sharpe": 1.0, "return_pct": 20, "max_dd_pct": 10, "trades": 60},
    ]
    new_cfg, _ = cs.cleanup_config(cfg, summary)
    assert "_suspended_strategies" not in new_cfg


# --------------------------------------------------------------------
# Report rendering
# --------------------------------------------------------------------

def test_render_report_groups_by_verdict():
    decisions = [
        {"id": "win",   "verdict": cs.KEEP,     "reason": "ok",
         "summary": {"sharpe": 1.5, "return_pct": 30, "max_dd_pct": 12, "trades": 50}},
        {"id": "loss",  "verdict": cs.DISCARD,  "reason": "bad",
         "summary": {"sharpe": -5.0, "return_pct": -100, "max_dd_pct": 100, "trades": 9999}},
        {"id": "stock", "verdict": cs.EQUITY,   "reason": "equity",
         "summary": {"status": "skipped", "reason": "equity ticker AAPL..."}},
    ]
    out = cs.render_markdown_report(decisions, "scheduler/config.json", "x/portfolio_summary.json", applied=False)
    # Group order should put DISCARD before KEEP so operators see hazards first.
    assert out.index("Discarded") < out.index("Keep")
    assert "`win`" in out and "`loss`" in out and "`stock`" in out
    assert "DRY-RUN" in out
    # "To apply" block only shows in dry-run mode.
    assert "--apply" in out


def test_render_report_applied_mode_hides_apply_block():
    out = cs.render_markdown_report([], "scheduler/config.json", "x.json", applied=True)
    assert "APPLIED" in out
    assert "## To apply" not in out


# --------------------------------------------------------------------
# End-to-end via main() w/ tmp_path
# --------------------------------------------------------------------

def test_main_dry_run_creates_preview_and_report(tmp_path, monkeypatch, capsys):
    cfg = {
        "config_version": 16,
        "strategies": [
            {"id": "good", "args": ["momentum", "BTC", "1h"]},
            {"id": "bad",  "args": ["momentum", "ETH", "5m"]},
        ],
    }
    summary = [
        {"id": "good", "status": "ok", "sharpe": 1.0, "return_pct": 25, "max_dd_pct": 12, "trades": 60},
        {"id": "bad",  "status": "ok", "sharpe": -10, "return_pct": -100, "max_dd_pct": 100, "trades": 5000},
    ]
    cfg_path = tmp_path / "config.json"
    sum_path = tmp_path / "portfolio_summary.json"
    cfg_path.write_text(json.dumps(cfg))
    sum_path.write_text(json.dumps(summary))

    monkeypatch.setattr(sys, "argv", [
        "clean_strategies.py",
        "--config", str(cfg_path),
        "--summary", str(sum_path),
        "--output-dir", str(tmp_path),
    ])
    rc = cs.main()
    assert rc == 0  # DISCARD was found, so success
    out = capsys.readouterr().out
    assert "DRY-RUN" in out
    preview = tmp_path / "config.cleaned.json"
    assert preview.exists()
    preview_data = json.loads(preview.read_text())
    assert [s["id"] for s in preview_data["strategies"]] == ["good"]
    # Original config untouched in dry-run.
    assert "bad" in cfg_path.read_text()
    # Report exists.
    assert (tmp_path / "cleanup_report.md").exists()


def test_main_apply_writes_backup_and_overwrites(tmp_path, monkeypatch):
    cfg = {
        "strategies": [
            {"id": "good", "args": ["momentum", "BTC", "1h"]},
            {"id": "bad",  "args": ["momentum", "ETH", "5m"]},
        ],
    }
    summary = [
        {"id": "good", "status": "ok", "sharpe": 1.0, "return_pct": 25, "max_dd_pct": 12, "trades": 60},
        {"id": "bad",  "status": "ok", "sharpe": -10, "return_pct": -100, "max_dd_pct": 100, "trades": 5000},
    ]
    cfg_path = tmp_path / "config.json"
    sum_path = tmp_path / "portfolio_summary.json"
    cfg_path.write_text(json.dumps(cfg))
    sum_path.write_text(json.dumps(summary))

    monkeypatch.setattr(sys, "argv", [
        "clean_strategies.py",
        "--config", str(cfg_path),
        "--summary", str(sum_path),
        "--output-dir", str(tmp_path),
        "--apply",
    ])
    rc = cs.main()
    assert rc == 0
    cleaned = json.loads(cfg_path.read_text())
    assert [s["id"] for s in cleaned["strategies"]] == ["good"]
    # A timestamped backup of the original should exist.
    backups = list(tmp_path.glob("config.json.bak.*"))
    assert len(backups) == 1
    backed_up = json.loads(backups[0].read_text())
    assert any(s["id"] == "bad" for s in backed_up["strategies"])
