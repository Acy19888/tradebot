package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// news_service_test.go — coverage for the Phase 3a news layer.
//
// Tests fall into three groups:
//   1. SQLite layer  — Insert idempotency, severity filtering, coin filter
//   2. Service loop  — alert-once semantics, dedupe across polls
//   3. API endpoint  — /api/v2/news returns the right shape
//
// Each test opens a fresh on-disk DB in a tmp dir; the in-process polling
// goroutine uses a stub fetcher so the subprocess is never spawned.

func openNewsTestDB(t *testing.T) (*StateDB, func()) {
	t.Helper()
	tmp, err := os.MkdirTemp("", "news-test-")
	if err != nil {
		t.Fatalf("mkdtemp: %v", err)
	}
	dbPath := filepath.Join(tmp, "state.db")
	sdb, err := OpenStateDB(dbPath)
	if err != nil {
		os.RemoveAll(tmp)
		t.Fatalf("open db: %v", err)
	}
	return sdb, func() {
		sdb.Close()
		os.RemoveAll(tmp)
	}
}

func TestInsertNewsEvent_IsIdempotent(t *testing.T) {
	sdb, cleanup := openNewsTestDB(t)
	defer cleanup()

	e := NewsEvent{
		ID: "abc123", Title: "BTC ETF approved",
		Source: "Reuters", Severity: "high", Sentiment: "positive",
		PublishedAt: "2026-04-01T12:00:00Z",
		Coins:       []string{"BTC"},
	}
	inserted, err := sdb.InsertNewsEvent(e)
	if err != nil {
		t.Fatalf("insert 1: %v", err)
	}
	if !inserted {
		t.Error("first insert should report inserted=true")
	}
	inserted2, err := sdb.InsertNewsEvent(e)
	if err != nil {
		t.Fatalf("insert 2: %v", err)
	}
	if inserted2 {
		t.Error("second insert should report inserted=false (idempotent)")
	}
}

func TestQueryRecentNewsEvents_SeverityFilter(t *testing.T) {
	sdb, cleanup := openNewsTestDB(t)
	defer cleanup()

	pub := "2026-04-01T12:00:00Z"
	for i, sev := range []string{"high", "medium", "low", "low"} {
		_, err := sdb.InsertNewsEvent(NewsEvent{
			ID: "id-" + string(rune('a'+i)), Title: "e" + sev,
			Severity: sev, Sentiment: "neutral", PublishedAt: pub,
			Coins: []string{"BTC"},
		})
		if err != nil {
			t.Fatalf("insert: %v", err)
		}
	}
	high, err := sdb.QueryRecentNewsEvents("high", "", 10)
	if err != nil {
		t.Fatalf("query high: %v", err)
	}
	if len(high) != 1 {
		t.Errorf("high: got %d, want 1", len(high))
	}
	medPlus, err := sdb.QueryRecentNewsEvents("medium", "", 10)
	if err != nil {
		t.Fatalf("query medium: %v", err)
	}
	if len(medPlus) != 2 {
		t.Errorf("medium+: got %d, want 2", len(medPlus))
	}
	all, err := sdb.QueryRecentNewsEvents("", "", 10)
	if err != nil {
		t.Fatalf("query all: %v", err)
	}
	if len(all) != 4 {
		t.Errorf("all: got %d, want 4", len(all))
	}
}

func TestQueryRecentNewsEvents_CoinFilter(t *testing.T) {
	sdb, cleanup := openNewsTestDB(t)
	defer cleanup()
	pub := "2026-04-01T12:00:00Z"
	if _, err := sdb.InsertNewsEvent(NewsEvent{ID: "1", Title: "btc only",
		Severity: "low", PublishedAt: pub, Coins: []string{"BTC"}}); err != nil {
		t.Fatal(err)
	}
	if _, err := sdb.InsertNewsEvent(NewsEvent{ID: "2", Title: "eth only",
		Severity: "low", PublishedAt: pub, Coins: []string{"ETH"}}); err != nil {
		t.Fatal(err)
	}
	if _, err := sdb.InsertNewsEvent(NewsEvent{ID: "3", Title: "btc and sol",
		Severity: "low", PublishedAt: pub, Coins: []string{"BTC", "SOL"}}); err != nil {
		t.Fatal(err)
	}
	got, err := sdb.QueryRecentNewsEvents("", "BTC", 10)
	if err != nil {
		t.Fatalf("query: %v", err)
	}
	if len(got) != 2 {
		t.Errorf("BTC filter: got %d, want 2", len(got))
	}
}

// stubNewsFetcher returns canned events; tracks call count for ordering checks.
type stubNewsFetcher struct {
	mu     sync.Mutex
	calls  int32
	events []NewsEvent
}

func (s *stubNewsFetcher) fetch(_, _ int) ([]NewsEvent, error) {
	atomic.AddInt32(&s.calls, 1)
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]NewsEvent, len(s.events))
	copy(out, s.events)
	return out, nil
}

func TestNewsService_PollOnceInsertsAndDedupes(t *testing.T) {
	sdb, cleanup := openNewsTestDB(t)
	defer cleanup()

	stub := &stubNewsFetcher{
		events: []NewsEvent{
			{ID: "n1", Title: "BTC ETF rejected", Severity: "high", Sentiment: "negative", PublishedAt: "2026-04-01T12:00:00Z", Coins: []string{"BTC"}},
			{ID: "n2", Title: "ETH upgrade live", Severity: "medium", PublishedAt: "2026-04-01T13:00:00Z", Coins: []string{"ETH"}},
		},
	}
	ns := NewNewsService(sdb, nil, NewsServiceConfig{
		PollInterval:  10 * time.Minute,
		HighSevAlerts: false, // notifier is nil; skip the alert branch
	})
	ns.fetcher = stub.fetch

	ns.pollOnce()
	got, err := sdb.QueryRecentNewsEvents("", "", 10)
	if err != nil {
		t.Fatalf("query: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("after 1st poll: got %d, want 2", len(got))
	}

	// Second poll with the same payload should not duplicate.
	ns.pollOnce()
	got2, err := sdb.QueryRecentNewsEvents("", "", 10)
	if err != nil {
		t.Fatalf("query 2: %v", err)
	}
	if len(got2) != 2 {
		t.Errorf("after 2nd poll: got %d, want 2 (dedupe by id)", len(got2))
	}
}

// mockOwnerNotifier captures owner DMs for assertions.
type mockOwnerNotifier struct {
	mu   sync.Mutex
	dms  []string
}

func (m *mockOwnerNotifier) SendMessage(channelID, content string) error {
	return nil
}
func (m *mockOwnerNotifier) SendDM(userID, content string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.dms = append(m.dms, content)
	return nil
}
func (m *mockOwnerNotifier) AskDM(userID, question string, timeout time.Duration) (string, error) {
	return "", nil
}
func (m *mockOwnerNotifier) Close() {}

func TestNewsService_HighSeverityAlertsOwnerOnce(t *testing.T) {
	sdb, cleanup := openNewsTestDB(t)
	defer cleanup()

	mn := &mockOwnerNotifier{}
	multi := NewMultiNotifier(notifierBackend{
		notifier: mn,
		ownerID:  "owner-1",
	})

	stub := &stubNewsFetcher{
		events: []NewsEvent{
			{ID: "hot-1", Title: "Coinbase hacked", Severity: "high", Sentiment: "negative", PublishedAt: "2026-04-01T12:00:00Z"},
			{ID: "warm-1", Title: "ETH upgrade live", Severity: "medium", PublishedAt: "2026-04-01T12:05:00Z"},
		},
	}
	ns := NewNewsService(sdb, multi, NewsServiceConfig{
		PollInterval:  10 * time.Minute,
		HighSevAlerts: true,
	})
	ns.fetcher = stub.fetch

	ns.pollOnce()
	mn.mu.Lock()
	first := len(mn.dms)
	mn.mu.Unlock()
	if first != 1 {
		t.Fatalf("first poll DMs: got %d, want 1 (only high-severity)", first)
	}

	// Second poll with the same data must NOT re-alert.
	ns.pollOnce()
	mn.mu.Lock()
	second := len(mn.dms)
	mn.mu.Unlock()
	if second != 1 {
		t.Errorf("second poll DMs: got %d, want 1 (alert-once)", second)
	}
}

func TestAPIV2News_ReturnsJSON(t *testing.T) {
	sdb, cleanup := openNewsTestDB(t)
	defer cleanup()
	pub := "2026-04-01T12:00:00Z"
	for i, sev := range []string{"high", "low"} {
		_, err := sdb.InsertNewsEvent(NewsEvent{
			ID: "api-" + string(rune('a'+i)), Title: "title-" + sev,
			Severity: sev, PublishedAt: pub, Coins: []string{"BTC"},
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	ss := &StatusServer{
		state:   NewAppState(),
		mu:      &sync.RWMutex{},
		stateDB: sdb,
	}
	req := httptest.NewRequest(http.MethodGet, "/api/v2/news?min_severity=high", nil)
	rr := httptest.NewRecorder()
	ss.handleAPIV2News(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("status: got %d, want 200", rr.Code)
	}
	var resp NewsV2Response
	if err := json.NewDecoder(rr.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(resp.Events) != 1 {
		t.Errorf("events: got %d, want 1 (high only)", len(resp.Events))
	}
}

func TestNewsService_StartStopIdempotent(t *testing.T) {
	sdb, cleanup := openNewsTestDB(t)
	defer cleanup()
	ns := NewNewsService(sdb, nil, NewsServiceConfig{
		PollInterval:  10 * time.Minute,
		HighSevAlerts: false,
	})
	stub := &stubNewsFetcher{events: []NewsEvent{}}
	ns.fetcher = stub.fetch

	if !ns.Start(context.Background()) {
		t.Fatal("first Start should return true")
	}
	if ns.Start(context.Background()) {
		t.Error("second Start should be no-op (return false)")
	}
	ns.Stop()
	ns.Stop() // second stop must not panic / hang
}

func TestNewsAlertFormatting(t *testing.T) {
	msg := formatHighSeverityNewsAlert(NewsEvent{
		Title:     "Coinbase hacked",
		Source:    "Reuters",
		Sentiment: "negative",
		Coins:     []string{"BTC", "ETH"},
		URL:       "https://example.com/x",
	})
	if !contains(msg, "HIGH-SEVERITY") {
		t.Errorf("missing header in: %q", msg)
	}
	if !contains(msg, "Coinbase hacked") {
		t.Errorf("missing title in: %q", msg)
	}
	if !contains(msg, "BTC, ETH") {
		t.Errorf("missing coins in: %q", msg)
	}
	if !contains(msg, "example.com") {
		t.Errorf("missing url in: %q", msg)
	}
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && (s == sub || (len(s) > 0 && (indexOf(s, sub) >= 0)))
}
func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}
