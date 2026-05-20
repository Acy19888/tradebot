#!/usr/bin/env python3
"""
clean_strategies.py — Auto-classify and clean a live config.json based on
backtest_portfolio.py results.

The portfolio backtest produces a portfolio_summary.json with per-strategy
metrics. This script joins that summary against the live config and
produces:

  scheduler/config.cleaned.json   — new config (dry-run: not written
                                    unless --apply is passed)
  reports/<backtest_dir>/cleanup_report.md
                                  — human-readable classification per
                                    strategy with the verdict reason

Default mode is dry-run — prints what WOULD happen and writes the report
but does NOT touch the original config. With --apply, the live config is
backed up to config.json.bak.<timestamp> and replaced with the cleaned
version.

Classification (top-trader thresholds, conservative — when in doubt the
script defers to the operator):

  DISCARD     — catastrophic, must NOT trade
                * total_return_pct <= -30.0  OR
                * sharpe < -0.5  OR
                * sharpe < 0 AND trades >= 50  (confirmed negative edge
                  — statistically significant losing strategy)  OR
                * max_drawdown_pct > 50  (would have been liquidated
                  in live trading regardless of headline Sharpe)
  SUSPICIOUS  — looks too good to be true (lookahead bias / curve-fit
                / data anomaly); needs manual review
                * sharpe >= 3.0
  TUNE        — has signal but not live-ready, keep for parameter work
                * sharpe between 0.0 and 0.5  OR
                * max_drawdown_pct > 30  OR
                * trades < 30   (with sharpe >= 0)
  KEEP        — passed the bar
                * sharpe >= 0.5 AND max_drawdown_pct <= 30 AND
                  trades >= 30 AND total_return_pct > 0
  EQUITY      — skipped in backtest (Robinhood AAPL/NVDA/...); we
                can't classify yet, preserve untouched
  UNTESTED    — no backtest result (data fetch failed, etc.); preserve
                untouched and flag for manual review

Cleanup action per class:

  DISCARD     → removed from `strategies` list
  SUSPICIOUS  → moved to `_suspended_strategies` (kept in file but the
                scheduler doesn't load that key; visible for audit)
  TUNE        → kept in `strategies` list, flagged in report
  KEEP        → kept
  EQUITY      → kept
  UNTESTED    → kept

Usage (from repo root):
  uv run --no-sync python scripts/clean_strategies.py
  uv run --no-sync python scripts/clean_strategies.py --apply
  uv run --no-sync python scripts/clean_strategies.py \\
      --config /opt/go-trader/scheduler/config.json \\
      --summary /opt/.../reports/backtest_<ts>/portfolio_summary.json

Exit codes:
  0  cleanup succeeded (dry-run or apply)
  1  catastrophic error
  2  no DISCARD strategies found AND no SUSPICIOUS — nothing to clean,
     operator can re-run later
"""
import argparse
import glob
import json
import os
import shutil
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Verdict labels used in both the cleaned config + the report.
KEEP = "KEEP"
TUNE = "TUNE"
DISCARD = "DISCARD"
SUSPICIOUS = "SUSPICIOUS"
EQUITY = "EQUITY"
UNTESTED = "UNTESTED"

# Equity tickers that the current backtest data path doesn't serve.
# Matches scripts/backtest_portfolio.py — keep in sync.
EQUITY_TICKERS = {
    "AAPL", "NVDA", "TSLA", "MSFT", "SPY", "QQQ", "GOOGL", "AMZN", "META",
    "AMD", "NFLX", "DIS", "BA", "INTC",
}


def classify(row: dict, strategy_arg_symbol: str = "") -> Tuple[str, str]:
    """Classify one summary row + strategy symbol into (verdict, reason).

    `row` is one entry from portfolio_summary.json. `strategy_arg_symbol`
    is the raw symbol from the config args[1] — needed because the
    summary doesn't carry the equity flag once the backtest skipped it.
    Returns (verdict, reason). Reason is a one-line plain-English string
    for the markdown report.
    """
    sid = row.get("id", "?")
    status = row.get("status")

    if status != "ok":
        reason_skip = (row.get("reason") or "").lower()
        if "equity ticker" in reason_skip or strategy_arg_symbol.upper() in EQUITY_TICKERS:
            return EQUITY, f"equity ticker — preserved (no backtest data path yet)"
        return UNTESTED, f"backtest did not produce a result: {row.get('reason') or 'unknown'}"

    sharpe = float(row.get("sharpe") or 0)
    ret = float(row.get("return_pct") or 0)
    max_dd = float(row.get("max_dd_pct") or 0)
    trades = int(row.get("trades") or 0)

    # Hard failures first — they trump everything else (a strategy with
    # both a "winning" Sharpe and a liquidation-level DD is still DISCARD,
    # because the equity curve would have crossed zero in live trading).
    if ret <= -30.0:
        return DISCARD, f"return {ret:+.1f}% catastrophic loss"
    if sharpe < -0.5:
        return DISCARD, f"Sharpe {sharpe:.2f} significantly negative ({trades} trades)"
    if sharpe < 0 and trades >= 50:
        # Statistically-significant negative edge — not random underperformance.
        # 50 trades is the rough sample-size floor below which a negative
        # Sharpe could plausibly be noise.
        return DISCARD, (
            f"Sharpe {sharpe:.2f} negative over {trades} trades — "
            f"confirmed losing edge"
        )
    if max_dd > 50.0:
        # A 50%+ drawdown means the realised equity path crossed below
        # half the starting capital. In live trading the position(s) on
        # the way down would either have been liquidated (perps) or
        # would have triggered the portfolio kill switch
        # (PortfolioRiskConfig.max_drawdown_pct defaults to 25%, see
        # scheduler/config.go). DISCARD regardless of headline Sharpe.
        return DISCARD, (
            f"Max-DD {max_dd:.1f}% — strategy would have been "
            f"liquidated / kill-switched in live trading"
        )

    # Too-good-to-be-true band — flag for human review, do NOT auto-keep.
    if sharpe >= 3.0:
        return SUSPICIOUS, (
            f"Sharpe {sharpe:.2f} unusually high (return {ret:+.1f}%, "
            f"{trades} trades); likely lookahead / data anomaly — manual review"
        )

    # Marginal band — has signal but not battle-tested.
    if 0.0 <= sharpe < 0.5:
        return TUNE, f"Sharpe {sharpe:.2f} marginal, needs tuning"
    if max_dd > 30.0 and sharpe >= 0:
        return TUNE, f"Max-DD {max_dd:.1f}% too high (Sharpe {sharpe:.2f})"
    if trades < 30 and sharpe >= 0:
        return TUNE, f"only {trades} trades — statistically inconclusive"

    # Anything left should be a healthy strategy.
    if sharpe >= 0.5 and ret > 0 and max_dd <= 30 and trades >= 30:
        return KEEP, (
            f"Sharpe {sharpe:.2f}, return {ret:+.1f}%, Max-DD {max_dd:.1f}%, "
            f"{trades} trades"
        )

    # Fallback — neither clear KEEP nor clear DISCARD.
    return TUNE, f"Sharpe {sharpe:.2f}, return {ret:+.1f}%, ambiguous result"


def latest_summary_path(reports_dir: str) -> Optional[str]:
    """Return path to the latest portfolio_summary.json under reports_dir,
    or None if none found."""
    matches = sorted(glob.glob(os.path.join(reports_dir, "backtest_*", "portfolio_summary.json")))
    if not matches:
        return None
    return matches[-1]


def summary_lookup(summary_rows: List[dict]) -> Dict[str, dict]:
    """Index summary rows by strategy id for O(1) lookup."""
    return {r.get("id"): r for r in summary_rows if r.get("id")}


def strategy_symbol(sc: dict) -> str:
    """Pull the raw symbol from a config StrategyConfig's args list.
    Empty string when args are missing or short — caller handles that."""
    args = sc.get("args") or []
    if len(args) >= 2:
        return (args[1] or "").upper()
    return ""


def cleanup_config(cfg: dict, summary_rows: List[dict]) -> Tuple[dict, List[dict]]:
    """Run the cleanup transformation. Returns (new_cfg, decisions_list).

    new_cfg has DISCARD removed from `strategies`, SUSPICIOUS moved to
    `_suspended_strategies` array, everything else preserved with the
    same dict identity.

    decisions_list is one dict per strategy, suitable for the markdown
    report:
      {"id": ..., "verdict": ..., "reason": ..., "summary": <row|None>}
    """
    by_id = summary_lookup(summary_rows)
    keep: List[dict] = []
    suspended: List[dict] = []
    decisions: List[dict] = []
    for sc in cfg.get("strategies", []) or []:
        sid = sc.get("id") or "?"
        row = by_id.get(sid)
        if row is None:
            verdict, reason = UNTESTED, "no entry in portfolio_summary.json"
        else:
            verdict, reason = classify(row, strategy_symbol(sc))
        decisions.append({
            "id": sid,
            "verdict": verdict,
            "reason": reason,
            "summary": row,
            "strategy": sc,
        })
        if verdict == DISCARD:
            continue  # drop entirely
        if verdict == SUSPICIOUS:
            suspended.append(sc)
            continue
        keep.append(sc)

    new_cfg = dict(cfg)
    new_cfg["strategies"] = keep
    if suspended:
        new_cfg["_suspended_strategies"] = suspended
    elif "_suspended_strategies" in new_cfg:
        # If the operator's previous cleanup left an empty key, prune it.
        del new_cfg["_suspended_strategies"]
    return new_cfg, decisions


def render_markdown_report(decisions: List[dict], cfg_path: str, summary_path: str,
                           applied: bool) -> str:
    """Generate the operator-facing cleanup report."""
    counts = {KEEP: 0, TUNE: 0, DISCARD: 0, SUSPICIOUS: 0, EQUITY: 0, UNTESTED: 0}
    for d in decisions:
        counts[d["verdict"]] = counts.get(d["verdict"], 0) + 1

    mode = "APPLIED" if applied else "DRY-RUN (no files mutated)"
    lines = [
        "# Strategy Cleanup Report",
        "",
        f"_Generated: {datetime.now().isoformat(timespec='seconds')}_",
        f"_Mode: **{mode}**_",
        f"_Source config: `{cfg_path}`_",
        f"_Backtest summary: `{summary_path}`_",
        "",
        "## Summary",
        "",
        f"- Total strategies: {len(decisions)}",
        f"- **KEEP**: {counts[KEEP]} (passed all bars — live-candidate quality)",
        f"- **TUNE**: {counts[TUNE]} (marginal — has signal but not live-ready)",
        f"- **SUSPICIOUS**: {counts[SUSPICIOUS]} (Sharpe ≥ 3.0 → moved to `_suspended_strategies`, manual review required)",
        f"- **DISCARD**: {counts[DISCARD]} (catastrophic — REMOVED from cleaned config)",
        f"- **EQUITY**: {counts[EQUITY]} (Robinhood stocks — preserved, no backtest data path yet)",
        f"- **UNTESTED**: {counts[UNTESTED]} (no result — preserved for manual review)",
        "",
        "## Classification Thresholds",
        "",
        "| Verdict | Trigger |",
        "| --- | --- |",
        "| DISCARD | Return ≤ -30% OR Sharpe < -0.5 |",
        "| SUSPICIOUS | Sharpe ≥ 3.0 (likely lookahead or data anomaly) |",
        "| TUNE | 0 ≤ Sharpe < 0.5, OR Max-DD > 30%, OR < 30 trades |",
        "| KEEP | Sharpe ≥ 0.5 AND Max-DD ≤ 30% AND ≥30 trades AND Return > 0 |",
        "| EQUITY | Robinhood ticker (untestable without yfinance/Polygon adapter) |",
        "| UNTESTED | No backtest result (data fetch failure, etc.) |",
        "",
    ]

    # One section per verdict so the operator can scan top-down.
    order = [DISCARD, SUSPICIOUS, TUNE, KEEP, UNTESTED, EQUITY]
    headings = {
        DISCARD: ("Discarded — REMOVED from cleaned config", "danger"),
        SUSPICIOUS: ("Suspicious — moved to `_suspended_strategies`", "warn"),
        TUNE: ("Needs Tuning — preserved, flagged for parameter work", "info"),
        KEEP: ("Keep — live-candidate quality", "good"),
        UNTESTED: ("Untested — preserved, manual review needed", "info"),
        EQUITY: ("Equity — preserved (no backtest data path yet)", "info"),
    }
    for verdict in order:
        items = [d for d in decisions if d["verdict"] == verdict]
        if not items:
            continue
        title, _ = headings[verdict]
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Strategy | Sharpe | Return | MaxDD | Trades | Reason |")
        lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
        for d in items:
            row = d.get("summary") or {}
            sharpe = row.get("sharpe")
            ret = row.get("return_pct")
            max_dd = row.get("max_dd_pct")
            trades = row.get("trades")
            lines.append(
                f"| `{d['id']}` "
                f"| {('%.2f' % sharpe) if sharpe is not None else '—'} "
                f"| {('%+.1f%%' % ret) if ret is not None else '—'} "
                f"| {('%.1f%%' % max_dd) if max_dd is not None else '—'} "
                f"| {trades if trades is not None else '—'} "
                f"| {d['reason']} |"
            )
        lines.append("")

    # Operator-facing next-actions block — important so the report is
    # actionable without context-switching to chat.
    lines += [
        "## Next Actions",
        "",
        "1. **Read the DISCARD section above** — these strategies will be gone after `--apply`.",
        "2. **Investigate SUSPICIOUS strategies** — open per_strategy/<id>.md from the same backtest run and inspect the trade log. Common red flags: unrealistic Sharpe on a thinly-traded asset, look-ahead in the strategy code, fitted parameters.",
        "3. **TUNE strategies** stay live for now but are NOT live-money candidates. Plan a walk-forward optimizer pass before moving any to real capital.",
        "4. **UNTESTED strategies** need either a data fix (check the backtest stderr) or removal.",
        "5. **EQUITY strategies** are parked until Phase 7c builds a yfinance/Polygon adapter.",
        "",
    ]
    if not applied:
        lines += [
            "## To apply",
            "",
            "```",
            "uv run --no-sync python scripts/clean_strategies.py --apply",
            "```",
            "",
            "On `--apply`:",
            "- A backup of the original config is written to `<config>.bak.<timestamp>`",
            "- The cleaned config is written in place",
            "- Restart the scheduler: `sudo systemctl restart go-trader`",
            "",
        ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clean a live go-trader config based on backtest_portfolio results."
    )
    parser.add_argument("--config", default="scheduler/config.json",
                        help="path to the live config (default: %(default)s)")
    parser.add_argument("--summary", default=None,
                        help="path to portfolio_summary.json; defaults to the latest under reports/")
    parser.add_argument("--reports-dir", default="reports",
                        help="reports directory for --summary auto-discovery (default: %(default)s)")
    parser.add_argument("--apply", action="store_true",
                        help="actually write the cleaned config; otherwise dry-run")
    parser.add_argument("--output-dir", default=None,
                        help="where to write the cleanup_report.md (default: alongside --summary)")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        return 1

    summary_path = args.summary or latest_summary_path(args.reports_dir)
    if not summary_path or not os.path.exists(summary_path):
        print(
            f"ERROR: portfolio_summary.json not found. Run scripts/backtest_portfolio.py first, "
            f"or pass --summary <path>.",
            file=sys.stderr,
        )
        return 1

    with open(args.config) as f:
        cfg = json.load(f)
    with open(summary_path) as f:
        summary_rows = json.load(f)

    new_cfg, decisions = cleanup_config(cfg, summary_rows)

    counts = {}
    for d in decisions:
        counts[d["verdict"]] = counts.get(d["verdict"], 0) + 1

    output_dir = args.output_dir or os.path.dirname(summary_path)
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "cleanup_report.md")

    if args.apply:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{args.config}.bak.{ts}"
        shutil.copy2(args.config, backup_path)
        cleaned_path = args.config
        with open(cleaned_path, "w") as f:
            json.dump(new_cfg, f, indent=2)
        with open(report_path, "w") as f:
            f.write(render_markdown_report(decisions, args.config, summary_path, applied=True))
        print(f"APPLIED. Backup → {backup_path}")
        print(f"Cleaned config written: {cleaned_path}")
    else:
        cleaned_path = os.path.join(
            os.path.dirname(args.config) or ".", "config.cleaned.json"
        )
        with open(cleaned_path, "w") as f:
            json.dump(new_cfg, f, indent=2)
        with open(report_path, "w") as f:
            f.write(render_markdown_report(decisions, args.config, summary_path, applied=False))
        print(f"DRY-RUN. Cleaned preview → {cleaned_path}")

    print(f"Report → {report_path}")
    print(f"  KEEP={counts.get(KEEP, 0)}  TUNE={counts.get(TUNE, 0)}  "
          f"DISCARD={counts.get(DISCARD, 0)}  SUSPICIOUS={counts.get(SUSPICIOUS, 0)}  "
          f"EQUITY={counts.get(EQUITY, 0)}  UNTESTED={counts.get(UNTESTED, 0)}")

    if counts.get(DISCARD, 0) == 0 and counts.get(SUSPICIOUS, 0) == 0:
        # Nothing actually changed structurally — let the caller know.
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
