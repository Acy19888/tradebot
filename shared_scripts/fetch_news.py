#!/usr/bin/env python3
"""
fetch_news.py — Crypto news fetcher for the go-trader news-awareness layer
(Phase 3a).

Polls CryptoPanic's public posts API, then enriches each item with:
  - severity (high | medium | low) — keyword-based classifier
  - sentiment (positive | negative | neutral) — keyword + vote-based
  - coins — affected ticker list (from CryptoPanic's currencies field)
  - event_id — stable hash of (source domain + title) for dedup

Severity classifier is intentionally simple — Wall Street's heavyweights run
trained NLP, we don't. A small word-list catches the "obvious" high-impact
classes (regulatory, hacks, ETF, FOMC) so the operator gets DM'd on items
that matter. Refinement happens iteratively after we observe real-world
output for a few days.

Output: JSON array on stdout, ASC by published_at (oldest first), one
object per news event. Always valid JSON — empty array on no items, never
HTML or partial JSON.

Subprocess contract (matches scheduler convention):
  - exit 0 on success even with empty results
  - exit 1 on fatal errors (network, JSON parse); a JSON error object is
    still emitted on stdout so the Go side can persist a structured error
    rather than guessing from stderr

CLI:
  --since-minutes N    only emit items newer than N minutes (default 1440 = 24h)
  --limit N            max items in output (default 100)
  --probe-only         exit 0 immediately without making any network call
                       (used by scheduler version_probe.go startup check)
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

CRYPTOPANIC_URL = "https://cryptopanic.com/api/v1/posts/"

# Keyword lists for the lightweight classifier. Order matters — high
# severity wins over medium wins over low when keywords overlap.
HIGH_SEVERITY_KEYWORDS = [
    # Regulatory + macro
    "sec ", "secs ", "lawsuit", "subpoena", "sanction", "court rules",
    "ruling", "regulation", "regulatory", "banned", "ban on", "outlaw",
    "fomc", "federal reserve", "fed raises", "fed cuts", "rate hike",
    "rate cut", "cpi", "ppi", "nfp", "non-farm payrolls",
    # Security incidents
    "hack", "hacked", "exploit", "exploited", "stolen", "drained",
    "rug pull", "rugpull", "exit scam", "bankruptcy", "insolvent",
    "halts withdrawals", "freeze",
    # ETF / Institutional
    "etf approved", "etf rejected", "etf approval", "etf decision",
    "institutional adoption", "blackrock", "vanguard", "fidelity",
    # Major market moves
    "all-time high", "all time high", "ath", "crash", "flash crash",
    "liquidation cascade", "liquidations exceed",
]

MEDIUM_SEVERITY_KEYWORDS = [
    "partnership", "integration", "launch", "launches", "upgrade",
    "fork", "halving", "mainnet", "testnet", "audit", "vulnerability",
    "patch", "release", "announces", "acquires", "acquisition",
    "delisting", "listed", "listing", "ipo", "funding round",
    "raises ", "raised $",
]

POSITIVE_KEYWORDS = [
    "approved", "approval", "surge", "rally", "soars", "breakthrough",
    "record high", "ath", "adoption", "institutional buy", "accumulation",
    "partnership", "integration", "upgrade", "successful", "milestone",
    "boost", "bullish", "gain", "growth", "expansion",
]

NEGATIVE_KEYWORDS = [
    "rejected", "decline", "drops", "plunge", "crash", "collapse",
    "loses", "down", "bearish", "sell-off", "selloff", "hack", "exploit",
    "stolen", "rug", "ban", "lawsuit", "fraud", "scam", "warning",
    "lawsuit", "insolvent", "bankrupt", "halt", "freeze", "delist",
    "fall", "fell", "tumble", "slump",
]


def classify_severity(title: str, votes: dict) -> str:
    t = title.lower()
    for kw in HIGH_SEVERITY_KEYWORDS:
        if kw in t:
            return "high"
    # CryptoPanic's "important" vote is a community marker — treat as
    # medium-bump signal even without a keyword match.
    if isinstance(votes, dict) and votes.get("important", 0) >= 3:
        return "high"
    for kw in MEDIUM_SEVERITY_KEYWORDS:
        if kw in t:
            return "medium"
    return "low"


def classify_sentiment(title: str, votes: dict) -> str:
    t = title.lower()
    pos = sum(1 for kw in POSITIVE_KEYWORDS if kw in t)
    neg = sum(1 for kw in NEGATIVE_KEYWORDS if kw in t)
    # Community vote signal (positive/negative columns on CryptoPanic).
    if isinstance(votes, dict):
        pos += min(votes.get("positive", 0), 5)
        neg += min(votes.get("negative", 0), 5)
    if pos - neg >= 2:
        return "positive"
    if neg - pos >= 2:
        return "negative"
    return "neutral"


def event_id_for(domain: str, title: str) -> str:
    """Stable ID per (source domain, title) so dedupe across polling
    windows works without storing CryptoPanic's internal post id."""
    h = hashlib.sha1()
    h.update((domain or "").lower().encode("utf-8"))
    h.update(b"\x00")
    h.update((title or "").strip().lower().encode("utf-8"))
    return h.hexdigest()[:16]


def fetch_cryptopanic(since_iso: str, limit: int, api_key: str = "") -> list:
    """Returns the raw `results` list from CryptoPanic, may raise on
    network/JSON failure. Caller handles errors and emits a JSON-shaped
    error response so the Go side has a structured signal.

    The free tier without an API key is rate-limited (~50 req/h). With a
    free auth_token you get higher limits. Both modes return the same
    payload shape, so no branching is needed downstream.
    """
    params = {"public": "true", "kind": "news"}
    if api_key:
        params["auth_token"] = api_key
    url = CRYPTOPANIC_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "go-trader/news-awareness 0.1"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = resp.read()
    payload = json.loads(body)
    if not isinstance(payload, dict) or "results" not in payload:
        raise ValueError("unexpected payload shape from CryptoPanic")
    return payload.get("results", [])


def normalize_item(raw: dict) -> dict:
    """Map a single CryptoPanic post into our internal news_event shape."""
    title = (raw.get("title") or "").strip()
    url = raw.get("url") or ""
    source = raw.get("source") or {}
    domain = source.get("domain") or ""
    source_title = source.get("title") or domain
    pub = raw.get("published_at") or raw.get("created_at") or ""
    votes = raw.get("votes") or {}
    currencies = raw.get("currencies") or []
    coins = sorted({(c.get("code") or "").upper() for c in currencies if c.get("code")})
    severity = classify_severity(title, votes)
    sentiment = classify_sentiment(title, votes)
    return {
        "id": event_id_for(domain, title),
        "title": title,
        "url": url,
        "source": source_title,
        "domain": domain,
        "published_at": pub,
        "coins": coins,
        "severity": severity,
        "sentiment": sentiment,
        "votes_important": int(votes.get("important", 0)) if isinstance(votes, dict) else 0,
    }


def parse_iso8601(s: str) -> datetime:
    """Tolerant ISO-8601 parser — CryptoPanic uses '2025-01-15T12:34:56Z',
    Python's fromisoformat needs '+00:00'. Empty / unparseable → epoch."""
    if not s:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    s = s.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return datetime.fromtimestamp(0, tz=timezone.utc)


def main():
    parser = argparse.ArgumentParser(description="Fetch and classify crypto news.")
    parser.add_argument("--since-minutes", type=int, default=1440,
                        help="emit only items newer than this many minutes (default 1440 = 24h)")
    parser.add_argument("--limit", type=int, default=100,
                        help="max items to emit (default 100)")
    parser.add_argument("--probe-only", action="store_true",
                        help="exit 0 immediately without making any network call (startup probe)")
    args = parser.parse_args()

    if args.probe_only:
        # Match version_probe.go convention: a successful argparse parse is
        # the only thing needed to confirm the script is invokable.
        print("[]")
        sys.exit(0)

    cutoff = datetime.now(tz=timezone.utc) - timedelta(minutes=args.since_minutes)

    api_key = os.environ.get("CRYPTOPANIC_API_KEY", "").strip()

    try:
        raw_items = fetch_cryptopanic("", args.limit, api_key=api_key)
    except Exception as e:
        # Emit a structured error JSON object so the Go side has a clean
        # path to log + alert, then exit 1. Plain stderr text is harder to
        # plumb into the existing notification surface.
        json.dump({"error": str(e)}, sys.stdout)
        sys.exit(1)

    items = []
    for raw in raw_items:
        try:
            item = normalize_item(raw)
        except Exception:
            continue
        if parse_iso8601(item["published_at"]) < cutoff:
            continue
        items.append(item)
    # Oldest first so a downstream "INSERT IGNORE" loop processes
    # chronologically and we can resume from MAX(published_at) on next run.
    items.sort(key=lambda it: parse_iso8601(it["published_at"]))
    items = items[-args.limit:]
    json.dump(items, sys.stdout)
    sys.exit(0)


if __name__ == "__main__":
    main()
