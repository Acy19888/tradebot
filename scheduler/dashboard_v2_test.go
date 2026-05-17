package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

// dashboard_v2_test.go — coverage for dashboard v2 backend endpoints.
//
// Each test wires a StatusServer against a freshly-opened SQLite StateDB so
// QueryTradeHistory exercises the real persistence path (no in-memory mock).
// AppState is populated directly to control initial capital + cash; trades
// are inserted via StateDB.InsertTrade so the rows match what RecordTrade
// would produce in production.

func newV2TestServer(t *testing.T) (*StatusServer, func()) {
	t.Helper()
	tmpDir, err := os.MkdirTemp("", "v2test-")
	if err != nil {
		t.Fatalf("temp dir: %v", err)
	}
	dbPath := filepath.Join(tmpDir, "state.db")
	sdb, err := OpenStateDB(dbPath)
	if err != nil {
		os.RemoveAll(tmpDir)
		t.Fatalf("open db: %v", err)
	}
	state := NewAppState()
	mu := &sync.RWMutex{}
	ss := &StatusServer{
		state:   state,
		mu:      mu,
		stateDB: sdb,
	}
	cleanup := func() {
		sdb.Close()
		os.RemoveAll(tmpDir)
	}
	return ss, cleanup
}

func seedStrategy(t *testing.T, ss *StatusServer, id string, initialCapital, cash float64) {
	t.Helper()
	ss.mu.Lock()
	defer ss.mu.Unlock()
	ss.state.Strategies[id] = &StrategyState{
		ID:              id,
		Type:            "perps",
		Platform:        "hyperliquid",
		Cash:            cash,
		InitialCapital:  initialCapital,
		Positions:       map[string]*Position{},
		OptionPositions: map[string]*OptionPosition{},
	}
}

func insertCloseTrade(t *testing.T, ss *StatusServer, strategyID, symbol string, ts time.Time, pnl, fee float64) {
	t.Helper()
	trade := Trade{
		Timestamp:   ts,
		StrategyID:  strategyID,
		Symbol:      symbol,
		Side:        "sell",
		Quantity:    1,
		Price:       100,
		Value:       100,
		TradeType:   "perps",
		IsClose:     true,
		RealizedPnL: pnl,
		ExchangeFee: fee,
	}
	if err := ss.stateDB.InsertTrade(strategyID, trade); err != nil {
		t.Fatalf("insert close trade: %v", err)
	}
}

func TestParseSinceParam(t *testing.T) {
	cases := []struct {
		in      string
		want    string
		wantErr bool
	}{
		{"", "all", false},
		{"all", "all", false},
		{"24h", "24h", false},
		{"7d", "7d", false},
		{"30d", "30d", false},
		{"12h", "12h", false},
		{"garbage", "garbage", true},
	}
	for _, c := range cases {
		_, got, err := parseSinceParam(c.in)
		if (err != nil) != c.wantErr {
			t.Errorf("parseSinceParam(%q) err=%v, wantErr=%v", c.in, err, c.wantErr)
		}
		if got != c.want {
			t.Errorf("parseSinceParam(%q) label=%q, want %q", c.in, got, c.want)
		}
	}
}

func TestV2Portfolio_Aggregates(t *testing.T) {
	ss, cleanup := newV2TestServer(t)
	defer cleanup()
	seedStrategy(t, ss, "s1", 1000, 1050)
	seedStrategy(t, ss, "s2", 2000, 1980)
	ss.mu.Lock()
	ss.state.Strategies["s1"].TradeHistory = []Trade{
		{IsClose: true, RealizedPnL: 50, ExchangeFee: 0.5},
	}
	ss.state.Strategies["s2"].TradeHistory = []Trade{
		{IsClose: true, RealizedPnL: -20, ExchangeFee: 0.3},
	}
	ss.state.CycleCount = 10
	ss.mu.Unlock()

	req := httptest.NewRequest(http.MethodGet, "/api/v2/portfolio", nil)
	rr := httptest.NewRecorder()
	ss.handleAPIV2Portfolio(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status %d, want 200", rr.Code)
	}
	var got PortfolioOverview
	if err := json.NewDecoder(rr.Body).Decode(&got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got.TotalInitialCapital != 3000 {
		t.Errorf("initial: got %v, want 3000", got.TotalInitialCapital)
	}
	if got.TotalCash != 3030 {
		t.Errorf("cash: got %v, want 3030", got.TotalCash)
	}
	// realized = 50 - 0.5 + (-20) - 0.3 = 29.2
	if got.TotalRealizedPnL > 29.21 || got.TotalRealizedPnL < 29.19 {
		t.Errorf("realized: got %v, want ~29.2", got.TotalRealizedPnL)
	}
	if got.StrategiesCount != 2 {
		t.Errorf("strategies: got %d, want 2", got.StrategiesCount)
	}
	if got.CycleCount != 10 {
		t.Errorf("cycle: got %d, want 10", got.CycleCount)
	}
}

func TestV2PnL_WindowsAndCounts(t *testing.T) {
	ss, cleanup := newV2TestServer(t)
	defer cleanup()
	seedStrategy(t, ss, "s1", 1000, 1000)
	now := time.Now()
	// 24h: +30 win, -10 loss
	insertCloseTrade(t, ss, "s1", "BTC", now.Add(-2*time.Hour), 30, 0.1)
	insertCloseTrade(t, ss, "s1", "BTC", now.Add(-1*time.Hour), -10, 0.1)
	// 7d window only: +25 win
	insertCloseTrade(t, ss, "s1", "BTC", now.Add(-3*24*time.Hour), 25, 0.1)
	// 30d window only: -5 loss
	insertCloseTrade(t, ss, "s1", "BTC", now.Add(-20*24*time.Hour), -5, 0.1)

	req := httptest.NewRequest(http.MethodGet, "/api/v2/pnl", nil)
	rr := httptest.NewRecorder()
	ss.handleAPIV2PnL(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("status %d, want 200", rr.Code)
	}
	var resp PnLResponse
	if err := json.NewDecoder(rr.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(resp.Windows) != 4 {
		t.Fatalf("expected 4 windows, got %d", len(resp.Windows))
	}
	byKey := map[string]PnLWindow{}
	for _, w := range resp.Windows {
		byKey[w.Key] = w
	}
	if w := byKey["24h"]; w.TradeCount != 2 || w.Wins != 1 || w.Losses != 1 {
		t.Errorf("24h: %+v, want trades=2 wins=1 losses=1", w)
	}
	// 7d trades: 24h ones (2) + 7d one = 3
	if w := byKey["7d"]; w.TradeCount != 3 || w.Wins != 2 || w.Losses != 1 {
		t.Errorf("7d: %+v, want trades=3 wins=2 losses=1", w)
	}
	if w := byKey["all"]; w.TradeCount != 4 || w.Wins != 2 || w.Losses != 2 {
		t.Errorf("all: %+v, want trades=4 wins=2 losses=2", w)
	}
}

func TestV2EquityCurve_RealizedReplay(t *testing.T) {
	ss, cleanup := newV2TestServer(t)
	defer cleanup()
	seedStrategy(t, ss, "s1", 1000, 1000)
	t0 := time.Now().Add(-6 * 24 * time.Hour)
	insertCloseTrade(t, ss, "s1", "BTC", t0.Add(1*time.Hour), 50, 1)  // +49
	insertCloseTrade(t, ss, "s1", "BTC", t0.Add(2*time.Hour), -20, 1) // -21 → cumulative +28

	req := httptest.NewRequest(http.MethodGet, "/api/v2/equity-curve?since=7d", nil)
	rr := httptest.NewRecorder()
	ss.handleAPIV2EquityCurve(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("status %d, want 200", rr.Code)
	}
	var resp EquityCurveResponse
	if err := json.NewDecoder(rr.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.StartValue != 1000 {
		t.Errorf("start: got %v, want 1000", resp.StartValue)
	}
	// 1000 + (50-1) + (-20-1) = 1028
	if resp.EndValue < 1027.99 || resp.EndValue > 1028.01 {
		t.Errorf("end: got %v, want 1028", resp.EndValue)
	}
	if len(resp.Points) < 2 {
		t.Errorf("points: got %d, want >= 2", len(resp.Points))
	}
}

func TestV2Trades_FilterWinLoss(t *testing.T) {
	ss, cleanup := newV2TestServer(t)
	defer cleanup()
	seedStrategy(t, ss, "s1", 1000, 1000)
	now := time.Now()
	insertCloseTrade(t, ss, "s1", "BTC", now.Add(-1*time.Hour), 30, 0.1)
	insertCloseTrade(t, ss, "s1", "BTC", now.Add(-2*time.Hour), -15, 0.1)
	insertCloseTrade(t, ss, "s1", "BTC", now.Add(-3*time.Hour), 5, 0.1)

	// Win filter
	req := httptest.NewRequest(http.MethodGet, "/api/v2/trades?filter=win", nil)
	rr := httptest.NewRecorder()
	ss.handleAPIV2Trades(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("win status %d, want 200", rr.Code)
	}
	var win TradesV2Response
	if err := json.NewDecoder(rr.Body).Decode(&win); err != nil {
		t.Fatalf("win decode: %v", err)
	}
	if len(win.Trades) != 2 {
		t.Errorf("win count: got %d, want 2", len(win.Trades))
	}
	for _, tr := range win.Trades {
		if tr.RealizedPnL <= 0 {
			t.Errorf("win filter returned non-win: %+v", tr)
		}
	}

	// Loss filter
	req = httptest.NewRequest(http.MethodGet, "/api/v2/trades?filter=loss", nil)
	rr = httptest.NewRecorder()
	ss.handleAPIV2Trades(rr, req)
	var loss TradesV2Response
	if err := json.NewDecoder(rr.Body).Decode(&loss); err != nil {
		t.Fatalf("loss decode: %v", err)
	}
	if len(loss.Trades) != 1 {
		t.Errorf("loss count: got %d, want 1", len(loss.Trades))
	}
}

func TestV2EquityCurve_NoTradesReturnsBaseline(t *testing.T) {
	ss, cleanup := newV2TestServer(t)
	defer cleanup()
	seedStrategy(t, ss, "s1", 500, 500)
	seedStrategy(t, ss, "s2", 1500, 1500)

	req := httptest.NewRequest(http.MethodGet, "/api/v2/equity-curve?since=24h", nil)
	rr := httptest.NewRecorder()
	ss.handleAPIV2EquityCurve(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("status %d, want 200", rr.Code)
	}
	var resp EquityCurveResponse
	if err := json.NewDecoder(rr.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.StartValue != 2000 {
		t.Errorf("start: got %v, want 2000 (sum of init capitals)", resp.StartValue)
	}
	if len(resp.Points) == 0 {
		t.Errorf("expected at least one anchor point when no trades")
	}
}
