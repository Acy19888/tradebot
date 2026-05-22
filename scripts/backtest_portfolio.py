#!/usr/bin/env python3
"""
backtest_portfolio.py — Run backtests for every strategy in a live config.

Reads scheduler/config.json (or a path you pass in), runs the existing
backtest framework on each strategy with its REAL configured params,
capital, and stop-loss settings, and writes:

  reports/backtest_<timestamp>/per_strategy/<id>.md
  reports/backtest_<timestamp>/portfolio_ranking.md
  reports/backtest_<timestamp>/portfolio_summary.json

Per-strategy report has the full single-strategy report (Sharpe, Sortino,
Max-DD, Win-Rate, Profit-Factor, Trade Log).
Ranking report sorts ALL tested strategies by Sharpe Ratio so the operator
can see at a glance which strategies actually have edge vs which are noise.

Usage (from repo root):
  uv run --no-sync python scripts/backtest_portfolio.py
  uv run --no-sync python scripts/backtest_portfolio.py --since 2023-01-01
  uv run --no-sync python scripts/backtest_portfolio.py --strategies hl-momentum-btc,hl-amd-btc

Stock strategies (Robinhood AAPL/NVDA/TSLA/MSFT/SPY) are skipped with a
"DATA NOT AVAILABLE" note — the binanceus-backed data_fetcher doesn't
serve equities. Add a yfinance/Polygon adapter to extend stock coverage.

Exit codes:
  0 = all strategies completed (with possible skips logged in report)
  1 = catastrophic error (config unreadable, output dir uncreatable)
"""
import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

# Wire up sys.path so the existing backtest framework loads even when
# invoked from any cwd. The registry_loader expects multiple paths:
#   - repo root (so `from shared_strategies.X import Y` resolves)
#   - backtest/ (for run_backtest, backtester, reporter modules)
#   - shared_strategies/open/ (registry_loader normally injects this
#     itself, but if a strategy module imports another sibling via the
#     bare module name we need the path early)
#   - shared_strategies/open/spot/
#   - shared_tools/
# Also chdir to the repo root so data_fetcher's SQLite cache lookup
# (which uses a relative ``shared_tools/`` style path internally) works.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (
    _REPO_ROOT,
    os.path.join(_REPO_ROOT, "backtest"),
    os.path.join(_REPO_ROOT, "shared_strategies", "open"),
    os.path.join(_REPO_ROOT, "shared_strategies", "open", "spot"),
    os.path.join(_REPO_ROOT, "shared_strategies", "open", "futures"),
    os.path.join(_REPO_ROOT, "shared_tools"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# Normalize cwd so any relative-path lookups (data cache, strategy
# registry discovery, etc.) resolve consistently regardless of where
# the operator invoked the script from.
os.chdir(_REPO_ROOT)

from run_backtest import run_single_backtest  # noqa: E402
from reporter import format_single_report  # noqa: E402

# Phase 7c — equity adapter. Imported lazily inside run_equity_backtest
# so the crypto-only paths don't pay the yfinance import cost.

# Equity tickers (Robinhood) — flagged as not-yet-supported by the
# crypto-only data_fetcher. We surface them in the report rather than
# attempting and getting a misleading "no data" entry per strategy.
EQUITY_TICKERS = {
    "AAPL", "NVDA", "TSLA", "MSFT", "SPY", "QQQ", "GOOGL", "AMZN", "META",
    "AMD", "NFLX", "DIS", "BA", "INTC",
}


def coin_to_symbol(coin: str, strategy_type: str, platform: str) -> str:
    """Map a config 'symbol' (e.g. 'BTC') to a backtest data symbol.

    HL perps + crypto-spot use the binanceus-cached '<COIN>/USDT' as a
    proxy for historical price action — perps and spot diverge slightly
    (basis), but for strategy-validation the spot series is the standard
    backtest input across the existing framework.

    Equities pass through unchanged so the caller can detect them and
    route to a stocks data path (future work).
    """
    coin = coin.upper().strip()
    if coin in EQUITY_TICKERS:
        return coin
    # Already a pair (e.g. 'BTC/USDT') — preserve.
    if "/" in coin:
        return coin
    return f"{coin}/USDT"


def derive_backtest_kwargs(sc: dict, since: str) -> Optional[Dict[str, Any]]:
    """Translate one StrategyConfig dict into run_single_backtest kwargs.

    Returns None when the strategy is unsupported by the current
    backtest data path (most commonly: equities). Returned dict is ready
    to splat — caller still wraps in try/except to catch unexpected
    framework errors.
    """
    args = sc.get("args") or []
    if len(args) < 3:
        return None
    raw_symbol = args[1]
    timeframe = args[2]
    strategy_type = sc.get("type", "perps")
    platform_field = sc.get("platform")
    script = sc.get("script") or ""
    # Infer platform from script when not explicit (config has historical
    # variance — newer entries set "platform", older HL ones use the
    # "hl-" id prefix + check_hyperliquid.py script).
    if not platform_field:
        if "hyperliquid" in script:
            platform_field = "hyperliquid"
        elif "robinhood" in script:
            platform_field = "robinhood"
        elif "okx" in script:
            platform_field = "okx"
        else:
            platform_field = "binanceus"

    if raw_symbol.upper() in EQUITY_TICKERS:
        return None  # equity — skipped, surfaced in summary

    symbol = coin_to_symbol(raw_symbol, strategy_type, platform_field)
    open_ref = sc.get("open_strategy") or {}
    strat_name = open_ref.get("name") or args[0]
    params = dict(open_ref.get("params") or {})

    # Registry selection — MUST match what the live script does, otherwise
    # backtest validates against a different strategy set than runs in prod.
    #
    # check_hyperliquid.py:  sys.path insert .../shared_strategies/open/futures
    # check_okx.py (swap):   sys.path insert .../shared_strategies/open/futures
    # check_okx.py (spot):   sys.path insert .../shared_strategies/open/spot
    # check_robinhood.py:    sys.path insert .../shared_strategies/open/spot
    #
    # So: perps + futures → futures registry; only true spot markets → spot.
    # The previous mapping (perps → spot) silently lost futures-only
    # strategies like tema_cross_bd and triple_ema_bidir — the backtester
    # returned None for them and we wrote them off as "no data" when really
    # they were just absent from the wrong registry. Phase 7d fix.
    if strategy_type == "spot":
        registry = "spot"
    else:
        # perps, futures, options — all use the futures-side registry in
        # production; the futures registry is a superset that includes
        # bidirectional strategies (tema_cross_bd, triple_ema_bidir, ...).
        registry = "futures"

    capital = float(sc.get("capital") or 1000)

    kwargs: Dict[str, Any] = {
        "strategy_name": strat_name,
        "symbol": symbol,
        "timeframe": timeframe,
        "since": since,
        "capital": capital,
        "params": params,
        "registry": registry,
        "platform": platform_field,
        "htf_filter": False,
        "regime_enabled": False,
        "strategy_type": strategy_type,
    }
    for opt in (
        "stop_loss_atr_mult",
        "stop_loss_pct",
        "stop_loss_margin_pct",
        "trailing_stop_atr_mult",
        "trailing_stop_pct",
    ):
        if sc.get(opt) is not None:
            kwargs[opt] = sc[opt]
    return kwargs


def run_one(sc: dict, since: str) -> Tuple[str, Optional[dict], Optional[str]]:
    """Run a single backtest with error capture. Returns (id, result_or_none, reason_skipped_or_error)."""
    sid = sc.get("id") or "unknown"
    raw_symbol = (sc.get("args") or [None, ""])[1].upper()
    if raw_symbol in EQUITY_TICKERS:
        # Phase 7c — equities now have a real data path via yfinance.
        try:
            result = run_equity_backtest(sc, since)
        except Exception as exc:
            return sid, None, f"equity backtest raised: {exc.__class__.__name__}: {exc}"
        if not result:
            return sid, None, "equity backtest returned None (no data from yfinance)"
        return sid, result, None

    kw = derive_backtest_kwargs(sc, since)
    if not kw:
        return sid, None, "could not derive backtest kwargs (missing args / type)"

    try:
        result = run_single_backtest(**kw)
    except Exception as exc:
        return sid, None, f"backtest raised: {exc.__class__.__name__}: {exc}"
    if not result:
        return sid, None, "backtest returned None (likely no data)"
    return sid, result, None


def run_equity_backtest(sc: dict, since: str) -> Optional[dict]:
    """Backtest a Robinhood equity strategy via yfinance data.

    Mirrors what run_single_backtest does for crypto (load OHLCV →
    apply_strategy → Backtester.run → return metrics dict) but routes
    the data load through shared_tools.equity_data_fetcher instead of
    the binanceus CCXT cache. Returns the same result-dict shape so
    reporter.format_single_report and the rest of backtest_portfolio
    treat equity strategies symmetrically with crypto ones.

    Strategy registry is "spot" (Robinhood equities trade as spot in
    the existing taxonomy); platform is "robinhood" so the
    CalculatePlatformSpotFee model picks the 0% / PFOF fee schedule.

    yfinance intraday history caps (5m → 60 days, 1h → 730 days) are
    documented in equity_data_fetcher; this function does not clip
    ``since`` — yfinance silently truncates and the report's period
    field reflects what was actually returned.
    """
    from registry_loader import load_registry
    from backtester import Backtester
    from atr import ensure_atr_indicator
    from equity_data_fetcher import load_equity_ohlcv

    args = sc.get("args") or []
    if len(args) < 3:
        return None
    open_ref = sc.get("open_strategy") or {}
    strategy_name = open_ref.get("name") or args[0]
    ticker = args[1]
    timeframe = args[2]
    params = dict(open_ref.get("params") or {})
    capital = float(sc.get("capital") or 100)

    reg = load_registry("spot")
    strat = reg.STRATEGY_REGISTRY.get(strategy_name)
    if not strat:
        print(f"[equity] unknown strategy '{strategy_name}' in spot registry", file=sys.stderr)
        return None

    df = load_equity_ohlcv(ticker, timeframe=timeframe, start_date=since)
    if df.empty:
        return None

    strat_params = params or strat.get("default_params", {})
    df_signals = reg.apply_strategy(strategy_name, df, strat_params)
    # Same ATR injection pattern as run_single_backtest: close evaluators
    # like tiered_tp_atr need an `atr` column; the open strategy may not
    # emit one.
    df_signals = ensure_atr_indicator(df_signals)

    bt = Backtester(
        initial_capital=capital,
        platform="robinhood",
        open_strategy={"name": strategy_name, "params": dict(strat_params)},
        close_strategies=None,
        regime_enabled=False,
        stop_loss_atr_mult=sc.get("stop_loss_atr_mult"),
        stop_loss_pct=sc.get("stop_loss_pct"),
        trailing_stop_atr_mult=sc.get("trailing_stop_atr_mult"),
        trailing_stop_pct=sc.get("trailing_stop_pct"),
        strategy_type="spot",
    )
    results = bt.run(
        df_signals,
        strategy_name=strategy_name,
        symbol=ticker,
        timeframe=timeframe,
        params=strat_params,
    )
    return results


def metric(r: dict, key: str, default=0):
    """Result accessor that's robust to missing keys."""
    if not r:
        return default
    v = r.get(key)
    return default if v is None else v


def write_per_strategy_report(out_dir: str, sid: str, sc: dict, result: Optional[dict], skipped_reason: Optional[str]) -> str:
    """Write reports/per_strategy/<id>.md and return the path."""
    path = os.path.join(out_dir, "per_strategy", f"{sid}.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    args = sc.get("args") or []
    raw_symbol = args[1] if len(args) > 1 else "?"
    timeframe = args[2] if len(args) > 2 else "?"
    strat_name = (sc.get("open_strategy") or {}).get("name") or (args[0] if args else "?")

    lines = [
        f"# Backtest: `{sid}`",
        "",
        f"- **Strategy:** `{strat_name}`",
        f"- **Symbol:** `{raw_symbol}`",
        f"- **Timeframe:** `{timeframe}`",
        f"- **Type:** `{sc.get('type', '?')}`",
        f"- **Capital:** ${float(sc.get('capital') or 0):,.2f}",
    ]
    params = (sc.get("open_strategy") or {}).get("params")
    if params:
        lines.append(f"- **Params:** `{json.dumps(params, sort_keys=True)}`")
    sl = sc.get("stop_loss_atr_mult")
    if sl is not None:
        lines.append(f"- **Stop-Loss ATR mult:** `{sl}`")
    direction = sc.get("direction")
    if direction:
        lines.append(f"- **Direction:** `{direction}`")
    lines.append("")

    if skipped_reason:
        lines.append("## ⚠️ Skipped")
        lines.append("")
        lines.append(skipped_reason)
        lines.append("")
        with open(path, "w") as f:
            f.write("\n".join(lines))
        return path

    # Success case — emit the formatted single-strategy report verbatim
    # inside a fenced block so it renders monospace in Markdown viewers.
    lines.append("## Results")
    lines.append("")
    lines.append("```")
    lines.append(format_single_report(result))
    lines.append("```")
    lines.append("")
    lines.append("## Key metrics")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"| --- | --- |")
    lines.append(f"| Total Return | {metric(result, 'total_return_pct'):+.2f}% |")
    lines.append(f"| Annual Return | {metric(result, 'annual_return_pct'):+.2f}% |")
    lines.append(f"| Sharpe Ratio | {metric(result, 'sharpe_ratio'):.3f} |")
    lines.append(f"| Sortino Ratio | {metric(result, 'sortino_ratio'):.3f} |")
    lines.append(f"| Max Drawdown | {metric(result, 'max_drawdown_pct'):.2f}% |")
    lines.append(f"| Calmar Ratio | {metric(result, 'calmar_ratio'):.3f} |")
    lines.append(f"| Total Trades | {metric(result, 'total_trades')} |")
    lines.append(f"| Win Rate | {metric(result, 'win_rate'):.1f}% |")
    lines.append(f"| Profit Factor | {metric(result, 'profit_factor'):.3f} |")
    lines.append(f"| Volatility | {metric(result, 'volatility_pct'):.2f}% |")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def write_ranking_report(out_dir: str, rows: List[dict], since: str) -> str:
    """Write reports/portfolio_ranking.md ranked by Sharpe ratio."""
    path = os.path.join(out_dir, "portfolio_ranking.md")
    successful = [r for r in rows if r["status"] == "ok"]
    skipped = [r for r in rows if r["status"] != "ok"]

    successful.sort(key=lambda r: r.get("sharpe", -999), reverse=True)

    lines = [
        "# Portfolio Backtest Ranking",
        "",
        f"_Generated {datetime.now().isoformat(timespec='seconds')}_",
        f"_Backtest period: {since} → today_",
        "",
        f"- Total strategies in config: {len(rows)}",
        f"- Successfully backtested: {len(successful)}",
        f"- Skipped or failed: {len(skipped)}",
        "",
        "## Ranking by Sharpe Ratio",
        "",
        "| Rank | Strategy | Symbol | TF | Type | Sharpe | Sortino | Return | MaxDD | WinRate | PF | Trades |",
        "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, r in enumerate(successful, 1):
        lines.append(
            f"| {i} | `{r['id']}` | {r['symbol']} | {r['timeframe']} | {r['type']} |"
            f" {r['sharpe']:.2f} | {r['sortino']:.2f} | {r['return_pct']:+.1f}% |"
            f" {r['max_dd_pct']:.1f}% | {r['win_rate']:.0f}% | {r['profit_factor']:.2f} |"
            f" {r['trades']} |"
        )

    # Top-trader interpretation rules of thumb. These thresholds are
    # commonly cited in quant lit (Carver 'Systematic Trading', Aronson
    # 'Evidence-Based TA') and serve as a quick reading guide for the
    # operator without hand-waving "this looks good".
    lines += [
        "",
        "## Top-Trader Reading Guide",
        "",
        "| Metric | Strong | Acceptable | Discard |",
        "| --- | --- | --- | --- |",
        "| Sharpe Ratio | > 1.5 | 0.8 – 1.5 | < 0.5 |",
        "| Sortino Ratio | > 2.0 | 1.0 – 2.0 | < 0.7 |",
        "| Max Drawdown | < 15% | 15 – 25% | > 30% |",
        "| Win Rate | > 55% | 45 – 55% | < 40% |",
        "| Profit Factor | > 1.6 | 1.2 – 1.6 | < 1.1 |",
        "| Trades | > 100 | 30 – 100 | < 20 (low confidence) |",
        "",
        "Strategies in the **Discard** column on >2 metrics should not go to live.",
        "Strategies with **<20 trades** are statistically inconclusive — backtest longer or run more.",
        "",
    ]

    if skipped:
        lines.append("## Skipped / Failed")
        lines.append("")
        lines.append("| Strategy | Reason |")
        lines.append("| --- | --- |")
        for r in skipped:
            lines.append(f"| `{r['id']}` | {r['reason']} |")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def write_summary_json(out_dir: str, rows: List[dict]) -> str:
    """Write reports/portfolio_summary.json — machine-readable view."""
    path = os.path.join(out_dir, "portfolio_summary.json")
    with open(path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest every strategy in a live config.")
    parser.add_argument("--config", default="scheduler/config.json",
                        help="path to scheduler/config.json (default: %(default)s)")
    parser.add_argument("--since", default=None,
                        help="backtest start (YYYY-MM-DD); default: 2 years ago")
    parser.add_argument("--output-dir", default=None,
                        help="output directory; default: reports/backtest_<timestamp>")
    parser.add_argument("--strategies", default=None,
                        help="comma-separated subset of strategy IDs (default: all)")
    args = parser.parse_args()

    if not args.since:
        args.since = (datetime.utcnow() - timedelta(days=730)).strftime("%Y-%m-%d")
    if not args.output_dir:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"reports/backtest_{ts}"

    if not os.path.exists(args.config):
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        return 1
    try:
        with open(args.config) as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"ERROR: cannot read config: {e}", file=sys.stderr)
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    strategies = cfg.get("strategies") or []
    if args.strategies:
        wanted = {s.strip() for s in args.strategies.split(",")}
        strategies = [s for s in strategies if s.get("id") in wanted]
        if not strategies:
            print(f"ERROR: no strategies matched: {args.strategies}", file=sys.stderr)
            return 1

    print(f"Running backtests for {len(strategies)} strategies, since {args.since}")
    print(f"Output: {args.output_dir}/")

    rows: List[dict] = []
    for idx, sc in enumerate(strategies, 1):
        sid = sc.get("id") or f"strategy_{idx}"
        print(f"\n[{idx}/{len(strategies)}] {sid} ...", flush=True)
        try:
            sid_back, result, reason = run_one(sc, args.since)
        except Exception:
            traceback.print_exc()
            sid_back, result, reason = sid, None, "unexpected exception in run_one"

        write_per_strategy_report(args.output_dir, sid_back, sc, result, reason)

        if result:
            rows.append({
                "id": sid_back,
                "symbol": result.get("symbol", "?"),
                "timeframe": result.get("timeframe", "?"),
                "type": sc.get("type", "?"),
                "status": "ok",
                "sharpe": metric(result, "sharpe_ratio"),
                "sortino": metric(result, "sortino_ratio"),
                "return_pct": metric(result, "total_return_pct"),
                "max_dd_pct": metric(result, "max_drawdown_pct"),
                "win_rate": metric(result, "win_rate"),
                "profit_factor": metric(result, "profit_factor"),
                "trades": int(metric(result, "total_trades")),
            })
            print(
                f"  sharpe={metric(result, 'sharpe_ratio'):.2f}"
                f"  return={metric(result, 'total_return_pct'):+.1f}%"
                f"  trades={int(metric(result, 'total_trades'))}"
            )
        else:
            rows.append({
                "id": sid_back,
                "status": "skipped",
                "reason": reason or "unknown",
            })
            print(f"  SKIPPED: {reason}")

    ranking_path = write_ranking_report(args.output_dir, rows, args.since)
    summary_path = write_summary_json(args.output_dir, rows)

    print(f"\nReports:")
    print(f"  {ranking_path}")
    print(f"  {summary_path}")
    print(f"  per-strategy: {os.path.join(args.output_dir, 'per_strategy')}/*.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
