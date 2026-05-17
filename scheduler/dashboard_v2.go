package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"
)

// dashboard_v2.go — Dashboard v2 endpoints (Phase 2).
//
// Adds aggregate read-only views layered on top of the existing /status and
// /api/strategies surface:
//
//   GET /api/v2/portfolio       — single-shot portfolio overview (KPIs)
//   GET /api/v2/pnl             — realized PnL across rolling windows (24h/7d/30d/all)
//   GET /api/v2/equity-curve    — realized-equity time-series (replayed from trades)
//   GET /api/v2/trades          — filtered + paginated trade list
//
// Design notes:
//   * Read-only. No state mutation. RLock for the in-memory state read in
//     /api/v2/portfolio; DB-only for the other three (RWMutex not needed).
//   * Equity curve is REALIZED only — we walk close-trade timestamps,
//     accumulating realized_pnl - exchange_fee per strategy. Unrealized P&L
//     between trade events is intentionally omitted: it would require a
//     historical mark series we don't have yet. A future phase can add a
//     portfolio_snapshots table that the scheduler appends to each cycle.
//   * Win/loss filtering is applied in-memory after QueryTradeHistory. Fine
//     for paper trading volume; if live volume balloons past ~10k trades,
//     promote the filter into a dedicated SQL helper.

// PortfolioOverview is the shape of /api/v2/portfolio.
type PortfolioOverview struct {
	TotalInitialCapital float64 `json:"total_initial_capital"`
	TotalCash           float64 `json:"total_cash"`
	TotalRealizedPnL    float64 `json:"total_realized_pnl"` // since inception
	TotalValue          float64 `json:"total_value"`        // = total cash (paper); future: + open mark value
	StrategiesCount     int     `json:"strategies_count"`
	OpenPositionsCount  int     `json:"open_positions_count"`
	KillSwitchActive    bool    `json:"kill_switch_active"`
	PeakValue           float64 `json:"peak_value"`
	CurrentDrawdownPct  float64 `json:"current_drawdown_pct"`
	LastCycle           string  `json:"last_cycle,omitempty"`
	CycleCount          int     `json:"cycle_count"`
}

// PnLWindow is one entry in /api/v2/pnl.
type PnLWindow struct {
	Key             string  `json:"key"`              // "24h" | "7d" | "30d" | "all"
	DurationSeconds int64   `json:"duration_seconds"` // 0 for "all"
	RealizedPnL     float64 `json:"realized_pnl"`
	Fees            float64 `json:"fees"`
	NetPnL          float64 `json:"net_pnl"`
	PctOfInitial    float64 `json:"pct_of_initial"`
	TradeCount      int     `json:"trade_count"`
	Wins            int     `json:"wins"`
	Losses          int     `json:"losses"`
}

// PnLResponse is the wrapper for /api/v2/pnl.
type PnLResponse struct {
	StrategyID string      `json:"strategy_id,omitempty"` // "" = portfolio-wide
	Windows    []PnLWindow `json:"windows"`
}

// EquityPoint is one (timestamp, value) on the realized equity curve.
type EquityPoint struct {
	Timestamp string  `json:"t"`
	Value     float64 `json:"v"`
}

// EquityCurveResponse is the shape of /api/v2/equity-curve.
type EquityCurveResponse struct {
	StrategyID string        `json:"strategy_id"` // "all" for portfolio
	Window     string        `json:"window"`      // "24h"|"7d"|"30d"|"all"
	StartValue float64       `json:"start_value"`
	EndValue   float64       `json:"end_value"`
	Points     []EquityPoint `json:"points"`
}

// TradesV2Response wraps QueryTradeHistory output with totals + applied
// filters echoed back for the UI.
type TradesV2Response struct {
	Total   int     `json:"total"`
	Limit   int     `json:"limit"`
	Offset  int     `json:"offset"`
	Filter  string  `json:"filter"` // "all"|"win"|"loss"
	Strategy string `json:"strategy,omitempty"`
	Symbol  string  `json:"symbol,omitempty"`
	Trades  []Trade `json:"trades"`
}

// parseSinceParam maps "24h"|"7d"|"30d"|"all" to a since-time. "all" → zero
// (no filter). Anything else is treated as a Go duration string ("12h",
// "90m") so the UI can be extended without code changes.
func parseSinceParam(s string) (time.Time, string, error) {
	s = strings.TrimSpace(strings.ToLower(s))
	switch s {
	case "", "all":
		return time.Time{}, "all", nil
	case "24h":
		return time.Now().Add(-24 * time.Hour), "24h", nil
	case "7d":
		return time.Now().Add(-7 * 24 * time.Hour), "7d", nil
	case "30d":
		return time.Now().Add(-30 * 24 * time.Hour), "30d", nil
	}
	d, err := time.ParseDuration(s)
	if err != nil {
		return time.Time{}, s, fmt.Errorf("bad window: %s", s)
	}
	return time.Now().Add(-d), s, nil
}

// writeV2JSON writes a JSON response with v2 conventions (bearer auth if a
// status token is configured, 200 by default, application/json header).
// Returns true on success so handlers can early-return on auth failure.
func (ss *StatusServer) writeV2JSON(w http.ResponseWriter, r *http.Request, body interface{}) bool {
	if ss.statusToken != "" && r.Header.Get("Authorization") != "Bearer "+ss.statusToken {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		w.Write([]byte(`{"error":"unauthorized"}`))
		return false
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	if err := json.NewEncoder(w).Encode(body); err != nil {
		// Headers already sent; nothing useful to do but log.
		fmt.Printf("[v2] encode response: %v\n", err)
		return false
	}
	return true
}

// handleAPIV2Portfolio is GET /api/v2/portfolio.
func (ss *StatusServer) handleAPIV2Portfolio(w http.ResponseWriter, r *http.Request) {
	ss.mu.RLock()
	defer ss.mu.RUnlock()

	overview := PortfolioOverview{
		KillSwitchActive:   ss.state.PortfolioRisk.KillSwitchActive,
		PeakValue:          ss.state.PortfolioRisk.PeakValue,
		CurrentDrawdownPct: ss.state.PortfolioRisk.CurrentDrawdownPct,
		CycleCount:         ss.state.CycleCount,
	}
	if !ss.state.LastCycle.IsZero() {
		overview.LastCycle = ss.state.LastCycle.UTC().Format(time.RFC3339)
	}

	for _, s := range ss.state.Strategies {
		if s == nil {
			continue
		}
		overview.StrategiesCount++
		overview.TotalInitialCapital += s.InitialCapital
		overview.TotalCash += s.Cash
		overview.OpenPositionsCount += len(s.Positions) + len(s.OptionPositions)

		// Realized PnL since inception: trade history is the source of truth.
		// Walk in-memory TradeHistory; SQLite has the full record but for the
		// overview we want what the scheduler has loaded, which is the same
		// data path the existing /status uses.
		for _, t := range s.TradeHistory {
			if t.IsClose {
				overview.TotalRealizedPnL += t.RealizedPnL - t.ExchangeFee
			} else {
				overview.TotalRealizedPnL -= t.ExchangeFee
			}
		}
	}
	overview.TotalValue = overview.TotalCash // paper: cash IS the value

	ss.writeV2JSON(w, r, overview)
}

// handleAPIV2PnL is GET /api/v2/pnl[?strategy=<id>].
func (ss *StatusServer) handleAPIV2PnL(w http.ResponseWriter, r *http.Request) {
	if ss.stateDB == nil {
		ss.writeV2JSON(w, r, PnLResponse{Windows: []PnLWindow{}})
		return
	}
	strategy := strings.TrimSpace(r.URL.Query().Get("strategy"))

	// Portfolio-wide baseline for pct math.
	ss.mu.RLock()
	var initial float64
	if strategy == "" {
		for _, s := range ss.state.Strategies {
			if s != nil {
				initial += s.InitialCapital
			}
		}
	} else if s, ok := ss.state.Strategies[strategy]; ok && s != nil {
		initial = s.InitialCapital
	}
	ss.mu.RUnlock()

	windows := []struct {
		key string
		d   time.Duration
	}{
		{"24h", 24 * time.Hour},
		{"7d", 7 * 24 * time.Hour},
		{"30d", 30 * 24 * time.Hour},
		{"all", 0},
	}

	resp := PnLResponse{StrategyID: strategy, Windows: make([]PnLWindow, 0, len(windows))}
	for _, w := range windows {
		var since time.Time
		if w.d > 0 {
			since = time.Now().Add(-w.d)
		}
		entry := PnLWindow{
			Key:             w.key,
			DurationSeconds: int64(w.d / time.Second),
		}
		trades, _, err := ss.stateDB.QueryTradeHistory(strategy, "", since, time.Time{}, 500, 0)
		if err == nil {
			for _, t := range trades {
				entry.Fees += t.ExchangeFee
				if t.IsClose {
					entry.RealizedPnL += t.RealizedPnL
					entry.TradeCount++
					if t.RealizedPnL > 0 {
						entry.Wins++
					} else if t.RealizedPnL < 0 {
						entry.Losses++
					}
				}
			}
		}
		entry.NetPnL = entry.RealizedPnL - entry.Fees
		if initial > 0 {
			entry.PctOfInitial = entry.NetPnL / initial * 100
		}
		resp.Windows = append(resp.Windows, entry)
	}
	ss.writeV2JSON(w, r, resp)
}

// handleAPIV2EquityCurve is GET /api/v2/equity-curve[?since=24h|7d|30d|all][&strategy=<id>].
func (ss *StatusServer) handleAPIV2EquityCurve(w http.ResponseWriter, r *http.Request) {
	if ss.stateDB == nil {
		ss.writeV2JSON(w, r, EquityCurveResponse{StrategyID: "all", Window: "all", Points: []EquityPoint{}})
		return
	}

	sinceStr := r.URL.Query().Get("since")
	since, window, err := parseSinceParam(sinceStr)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"error":"` + err.Error() + `"}`))
		return
	}
	strategy := strings.TrimSpace(r.URL.Query().Get("strategy"))
	if strategy == "" {
		strategy = "all"
	}

	// Determine baseline cash. For one strategy: its InitialCapital. For
	// "all": sum of all strategies' InitialCapital. Read under RLock from the
	// in-memory state (mirrors what the rest of the dashboard uses).
	ss.mu.RLock()
	var baseline float64
	if strategy == "all" {
		for _, s := range ss.state.Strategies {
			if s != nil {
				baseline += s.InitialCapital
			}
		}
	} else if s, ok := ss.state.Strategies[strategy]; ok && s != nil {
		baseline = s.InitialCapital
	}
	ss.mu.RUnlock()

	// Trades to walk. For "all", QueryTradeHistory("",...) returns all
	// strategies. We always pull from the beginning so the curve has a
	// stable anchor point at the initial capital; the since param then
	// filters the OUTPUT to the requested window (after the equity has
	// been replayed in full to the window's start).
	queryStrategy := ""
	if strategy != "all" {
		queryStrategy = strategy
	}
	trades, _, err := ss.stateDB.QueryTradeHistory(queryStrategy, "", time.Time{}, time.Time{}, 500, 0)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error":"query trades failed"}`))
		return
	}
	// QueryTradeHistory returns DESC; we want ASC for the cumulative walk.
	sort.Slice(trades, func(i, j int) bool {
		return trades[i].Timestamp.Before(trades[j].Timestamp)
	})

	resp := EquityCurveResponse{
		StrategyID: strategy,
		Window:     window,
		StartValue: baseline,
		Points:     []EquityPoint{},
	}
	// Anchor: baseline at first trade time (or now if no trades).
	cumulative := baseline
	if len(trades) == 0 {
		resp.EndValue = baseline
		resp.Points = append(resp.Points,
			EquityPoint{Timestamp: time.Now().UTC().Format(time.RFC3339), Value: baseline},
		)
		ss.writeV2JSON(w, r, resp)
		return
	}

	// First point: baseline value at first trade's timestamp (so the chart
	// has an anchor before the first trade fires).
	first := trades[0].Timestamp.UTC().Format(time.RFC3339)
	if since.IsZero() || trades[0].Timestamp.After(since) {
		resp.Points = append(resp.Points, EquityPoint{Timestamp: first, Value: cumulative})
	}

	for _, t := range trades {
		// Every trade pays fees; close trades realize PnL.
		cumulative -= t.ExchangeFee
		if t.IsClose {
			cumulative += t.RealizedPnL
		}
		// Filter output by the since window; cumulative has already
		// absorbed pre-window trades so the anchor reflects correct PnL.
		if !since.IsZero() && t.Timestamp.Before(since) {
			continue
		}
		resp.Points = append(resp.Points, EquityPoint{
			Timestamp: t.Timestamp.UTC().Format(time.RFC3339),
			Value:     cumulative,
		})
	}
	// If the since window excluded ALL trades, seed one point at the window
	// start so the chart isn't empty.
	if len(resp.Points) == 0 {
		resp.Points = append(resp.Points,
			EquityPoint{Timestamp: time.Now().UTC().Format(time.RFC3339), Value: cumulative},
		)
	}
	resp.EndValue = cumulative
	ss.writeV2JSON(w, r, resp)
}

// handleAPIV2Trades is GET /api/v2/trades[?strategy=...][&symbol=...][&filter=all|win|loss][&since=24h...][&limit=N][&offset=N].
func (ss *StatusServer) handleAPIV2Trades(w http.ResponseWriter, r *http.Request) {
	if ss.stateDB == nil {
		ss.writeV2JSON(w, r, TradesV2Response{Trades: []Trade{}, Filter: "all"})
		return
	}
	q := r.URL.Query()
	strategy := strings.TrimSpace(q.Get("strategy"))
	symbol := strings.TrimSpace(q.Get("symbol"))
	filter := strings.ToLower(strings.TrimSpace(q.Get("filter")))
	if filter == "" {
		filter = "all"
	}
	limit, _ := strconv.Atoi(q.Get("limit"))
	if limit <= 0 {
		limit = 50
	}
	offset, _ := strconv.Atoi(q.Get("offset"))
	if offset < 0 {
		offset = 0
	}

	since, _, err := parseSinceParam(q.Get("since"))
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"error":"` + err.Error() + `"}`))
		return
	}

	// QueryTradeHistory returns DESC by timestamp — that's what we want for
	// "latest first" UI display.
	trades, total, err := ss.stateDB.QueryTradeHistory(strategy, symbol, since, time.Time{}, limit*4, offset)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error":"query trades failed"}`))
		return
	}
	// Apply win/loss filter in-memory; QueryTradeHistory doesn't expose a
	// realized_pnl-sign predicate. See file-level note about future SQL move.
	if filter == "win" || filter == "loss" {
		filtered := trades[:0]
		for _, t := range trades {
			if !t.IsClose {
				continue
			}
			if filter == "win" && t.RealizedPnL > 0 {
				filtered = append(filtered, t)
			}
			if filter == "loss" && t.RealizedPnL < 0 {
				filtered = append(filtered, t)
			}
		}
		trades = filtered
	}
	if len(trades) > limit {
		trades = trades[:limit]
	}
	resp := TradesV2Response{
		Total:    total,
		Limit:    limit,
		Offset:   offset,
		Filter:   filter,
		Strategy: strategy,
		Symbol:   symbol,
		Trades:   trades,
	}
	ss.writeV2JSON(w, r, resp)
}
