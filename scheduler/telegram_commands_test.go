package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// telegramCommandsResetGate resets the package-level start gate so the
// handler can be (re)started in a test. Tests must NOT use t.Parallel() —
// the gate is process-global, like tradeRecorder in state.go.
func telegramCommandsResetGate(t *testing.T) {
	t.Helper()
	telegramCommandHandlerStarted.Store(false)
}

// newTestBrokerNotifier spins up a fake Telegram server that answers
// getUpdates with the provided updates exactly once (then idles), and
// records every sendMessage call. Returns the notifier with broker started
// and a slice of recorded outbound messages (sender locks own access).
type recordedSend struct {
	chatID string
	text   string
}

type fakeTelegramServer struct {
	mu       sync.Mutex
	sends    []recordedSend
	pending  []telegramUpdate // drained on each getUpdates call
	server   *httptest.Server
	notifier *TelegramNotifier
}

// newFakeTelegramServer spins up a fake Telegram Bot API and an attached
// TelegramNotifier. No updates are pre-loaded — use pushUpdates to inject
// them AFTER the broker + any subscribers are running so race-free delivery
// is observable. getUpdates sleeps briefly when there are no pending updates
// to keep the broker's poll loop from tight-spinning during tests.
func newFakeTelegramServer(t *testing.T) *fakeTelegramServer {
	t.Helper()
	f := &fakeTelegramServer{}
	f.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path := r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		switch {
		case strings.HasSuffix(path, "/getMe"):
			json.NewEncoder(w).Encode(telegramResponse{OK: true})
		case strings.HasSuffix(path, "/getUpdates"):
			f.mu.Lock()
			pending := f.pending
			f.pending = nil
			f.mu.Unlock()
			// Lightweight long-poll simulation: when there are no updates,
			// hold the request briefly so the broker doesn't tight-loop.
			if len(pending) == 0 {
				time.Sleep(20 * time.Millisecond)
			}
			raw, _ := json.Marshal(pending)
			json.NewEncoder(w).Encode(telegramResponse{OK: true, Result: raw})
		case strings.HasSuffix(path, "/sendMessage"):
			var body map[string]interface{}
			json.NewDecoder(r.Body).Decode(&body)
			chatID, _ := body["chat_id"].(string)
			text, _ := body["text"].(string)
			f.mu.Lock()
			f.sends = append(f.sends, recordedSend{chatID: chatID, text: text})
			f.mu.Unlock()
			json.NewEncoder(w).Encode(telegramResponse{OK: true})
		default:
			json.NewEncoder(w).Encode(telegramResponse{OK: false, Description: "unknown method " + path})
		}
	}))
	f.notifier = &TelegramNotifier{
		botToken: "test",
		client:   &http.Client{Timeout: 2 * time.Second},
		baseURL:  f.server.URL + "/bot",
	}
	return f
}

// pushUpdates appends updates to the pending queue. The next getUpdates
// poll drains them. Call AFTER the broker + subscribers are wired up so the
// delivery is race-free.
func (f *fakeTelegramServer) pushUpdates(updates ...telegramUpdate) {
	f.mu.Lock()
	f.pending = append(f.pending, updates...)
	f.mu.Unlock()
}

// waitForSends blocks until n recorded sends are present or the deadline
// hits. Returns the recorded sends at that point.
func (f *fakeTelegramServer) waitForSends(n int, d time.Duration) []recordedSend {
	deadline := time.Now().Add(d)
	for time.Now().Before(deadline) {
		if got := f.recordedSends(); len(got) >= n {
			return got
		}
		time.Sleep(20 * time.Millisecond)
	}
	return f.recordedSends()
}

func (f *fakeTelegramServer) recordedSends() []recordedSend {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make([]recordedSend, len(f.sends))
	copy(out, f.sends)
	return out
}

func (f *fakeTelegramServer) close() {
	if f.notifier != nil {
		f.notifier.Close()
	}
	f.server.Close()
}

func TestSplitCommand(t *testing.T) {
	cases := []struct {
		in       string
		wantCmd  string
		wantRest string
	}{
		{"/status", "/status", ""},
		{"/status arg1", "/status", "arg1"},
		{"/STATUS arg1 arg2", "/status", "arg1 arg2"},
		{"/killswitch@MyBot CONFIRM", "/killswitch", "CONFIRM"},
		{"/help@MyBot", "/help", ""},
		{"  /trim   x  ", "/trim", "x"}, // splitCommand expects pre-trimmed input
	}
	for _, c := range cases {
		got, rest := splitCommand(strings.TrimSpace(c.in))
		if got != c.wantCmd || rest != c.wantRest {
			t.Errorf("splitCommand(%q) = (%q,%q), want (%q,%q)",
				c.in, got, rest, c.wantCmd, c.wantRest)
		}
	}
}

func TestTruncID(t *testing.T) {
	if got := truncID("abc", 5); got != "abc" {
		t.Errorf("short string changed: %q", got)
	}
	if got := truncID("abcdefghij", 5); got != "abcd…" {
		t.Errorf("trunc: got %q, want %q", got, "abcd…")
	}
}

func TestFormatRelTime(t *testing.T) {
	if got := formatRelTime(time.Time{}); got != "never" {
		t.Errorf("zero time: got %q, want %q", got, "never")
	}
	if got := formatRelTime(time.Now().Add(-30 * time.Second)); !strings.HasSuffix(got, "s ago") {
		t.Errorf("seconds: got %q", got)
	}
	if got := formatRelTime(time.Now().Add(-10 * time.Minute)); !strings.HasSuffix(got, "m ago") {
		t.Errorf("minutes: got %q", got)
	}
	if got := formatRelTime(time.Now().Add(-3 * time.Hour)); !strings.HasSuffix(got, "h ago") {
		t.Errorf("hours: got %q", got)
	}
}

func newCommandHandlerForTest(notifier *TelegramNotifier, state *AppState, owner string) *TelegramCommandHandler {
	mu := &sync.RWMutex{}
	cfg := &Config{}
	return NewTelegramCommandHandler(notifier, owner, state, mu, cfg, nil)
}

func TestCommandHandler_HelpAndStatus(t *testing.T) {
	telegramCommandsResetGate(t)
	state := NewAppState()
	state.CycleCount = 42
	state.LastCycle = time.Now().Add(-2 * time.Minute)
	state.PortfolioRisk.PeakValue = 10000
	state.PortfolioRisk.CurrentDrawdownPct = 3.14
	state.Strategies["hl-momentum-btc"] = &StrategyState{
		ID:             "hl-momentum-btc",
		Type:           "perps",
		Cash:           950,
		InitialCapital: 1000,
		Positions:      map[string]*Position{},
		OptionPositions: map[string]*OptionPosition{},
	}

	f := newFakeTelegramServer(t)
	defer f.close()
	f.notifier.StartUpdateBroker()
	h := newCommandHandlerForTest(f.notifier, state, "12345")
	if h == nil {
		t.Fatal("handler should not be nil")
	}

	help := h.cmdHelp()
	if !strings.Contains(help, "/status") || !strings.Contains(help, "/killswitch") {
		t.Errorf("help missing commands: %q", help)
	}

	status := h.cmdStatus()
	if !strings.Contains(status, "Cycle: 42") {
		t.Errorf("status missing cycle: %q", status)
	}
	if !strings.Contains(status, "hl-momentum-btc") {
		t.Errorf("status missing strategy: %q", status)
	}
	if !strings.Contains(status, "Peak: $10000") {
		t.Errorf("status missing peak: %q", status)
	}
	if !strings.Contains(status, "Drawdown: 3.14%") {
		t.Errorf("status missing dd: %q", status)
	}
}

func TestCommandHandler_PositionsEmpty(t *testing.T) {
	telegramCommandsResetGate(t)
	state := NewAppState()
	f := newFakeTelegramServer(t)
	defer f.close()
	f.notifier.StartUpdateBroker()
	h := newCommandHandlerForTest(f.notifier, state, "12345")

	out := h.cmdPositions()
	if !strings.Contains(out, "No open positions") {
		t.Errorf("expected 'No open positions', got %q", out)
	}
}

func TestCommandHandler_PositionsWithLongAndShort(t *testing.T) {
	telegramCommandsResetGate(t)
	state := NewAppState()
	state.Strategies["hl-momentum-btc"] = &StrategyState{
		ID:   "hl-momentum-btc",
		Type: "perps",
		Positions: map[string]*Position{
			"BTC": {Symbol: "BTC", Side: "long", Quantity: 0.5, AvgCost: 65000},
		},
		OptionPositions: map[string]*OptionPosition{},
	}
	state.Strategies["hl-short-eth"] = &StrategyState{
		ID:   "hl-short-eth",
		Type: "perps",
		Positions: map[string]*Position{
			"ETH": {Symbol: "ETH", Side: "short", Quantity: 2.0, AvgCost: 3200},
		},
		OptionPositions: map[string]*OptionPosition{},
	}

	f := newFakeTelegramServer(t)
	defer f.close()
	f.notifier.StartUpdateBroker()
	h := newCommandHandlerForTest(f.notifier, state, "12345")

	out := h.cmdPositions()
	if !strings.Contains(out, "long BTC") {
		t.Errorf("expected 'long BTC', got %q", out)
	}
	if !strings.Contains(out, "short ETH") {
		t.Errorf("expected 'short ETH', got %q", out)
	}
	if !strings.Contains(out, "2 open position(s)") {
		t.Errorf("expected count line, got %q", out)
	}
}

func TestCommandHandler_KillSwitchNeedsConfirm(t *testing.T) {
	telegramCommandsResetGate(t)
	state := NewAppState()
	f := newFakeTelegramServer(t)
	defer f.close()
	f.notifier.StartUpdateBroker()
	h := newCommandHandlerForTest(f.notifier, state, "12345")

	out := h.cmdKillSwitch("")
	if !strings.Contains(out, "CONFIRM") {
		t.Errorf("expected CONFIRM prompt, got %q", out)
	}
	if state.PortfolioRisk.KillSwitchActive {
		t.Error("kill switch must not fire without CONFIRM")
	}
}

func TestCommandHandler_KillSwitchFires(t *testing.T) {
	telegramCommandsResetGate(t)
	state := NewAppState()
	state.PortfolioRisk.PeakValue = 10000
	state.PortfolioRisk.CurrentDrawdownPct = 5.0
	f := newFakeTelegramServer(t)
	defer f.close()
	f.notifier.StartUpdateBroker()

	var saved int32
	saver := func() error { atomic.AddInt32(&saved, 1); return nil }
	mu := &sync.RWMutex{}
	cfg := &Config{}
	h := NewTelegramCommandHandler(f.notifier, "12345", state, mu, cfg, saver)

	out := h.cmdKillSwitch("CONFIRM")
	if !strings.Contains(out, "ARMED") {
		t.Errorf("expected ARMED reply, got %q", out)
	}
	if !state.PortfolioRisk.KillSwitchActive {
		t.Error("kill switch should be active")
	}
	if state.PortfolioRisk.KillSwitchAt.IsZero() {
		t.Error("KillSwitchAt should be set")
	}
	if len(state.PortfolioRisk.Events) == 0 {
		t.Error("expected kill switch event recorded")
	}
	if got := atomic.LoadInt32(&saved); got != 1 {
		t.Errorf("stateSaver call count: got %d, want 1", got)
	}

	// Second call should report already-active.
	out2 := h.cmdKillSwitch("CONFIRM")
	if !strings.Contains(out2, "already active") {
		t.Errorf("expected already-active reply, got %q", out2)
	}
}

func TestCommandHandler_NewReturnsNilWhenDisabled(t *testing.T) {
	state := NewAppState()
	mu := &sync.RWMutex{}
	cfg := &Config{}

	if h := NewTelegramCommandHandler(nil, "12345", state, mu, cfg, nil); h != nil {
		t.Error("expected nil when notifier is nil")
	}
	if h := NewTelegramCommandHandler(&TelegramNotifier{}, "", state, mu, cfg, nil); h != nil {
		t.Error("expected nil when owner is empty")
	}
	if h := NewTelegramCommandHandler(&TelegramNotifier{}, "12345", nil, mu, cfg, nil); h != nil {
		t.Error("expected nil when state is nil")
	}
}

func TestCommandHandler_StartIgnoresNonOwner(t *testing.T) {
	telegramCommandsResetGate(t)
	state := NewAppState()

	f := newFakeTelegramServer(t)
	defer f.close()
	f.notifier.StartUpdateBroker()

	h := newCommandHandlerForTest(f.notifier, state, "12345")
	if !h.Start() {
		t.Fatal("Start returned false")
	}
	defer h.Stop()

	// Subscribers are now wired; push updates so the broker delivers them.
	// One from a non-owner (must be ignored) plus one from the owner.
	now := time.Now().Unix()
	f.pushUpdates(
		telegramUpdate{
			UpdateID: 1,
			Message: &telegramMsg{
				MessageID: 1,
				From:      &telegramUser{ID: 99999}, // not the owner
				Chat:      telegramChat{ID: 99999},
				Date:      now,
				Text:      "/status",
			},
		},
		telegramUpdate{
			UpdateID: 2,
			Message: &telegramMsg{
				MessageID: 2,
				From:      &telegramUser{ID: 12345}, // owner
				Chat:      telegramChat{ID: 12345},
				Date:      now,
				Text:      "/help",
			},
		},
	)

	sends := f.waitForSends(1, 3*time.Second)
	if len(sends) != 1 {
		t.Fatalf("expected exactly 1 reply (owner only), got %d: %+v", len(sends), sends)
	}
	if sends[0].chatID != "12345" {
		t.Errorf("reply went to %q, expected owner 12345", sends[0].chatID)
	}
	if !strings.Contains(sends[0].text, "/status") {
		t.Errorf("expected /help text containing '/status', got %q", sends[0].text)
	}
}

func TestCommandHandler_UnknownCommand(t *testing.T) {
	telegramCommandsResetGate(t)
	state := NewAppState()

	f := newFakeTelegramServer(t)
	defer f.close()
	f.notifier.StartUpdateBroker()

	h := newCommandHandlerForTest(f.notifier, state, "12345")
	if !h.Start() {
		t.Fatal("Start returned false")
	}
	defer h.Stop()

	f.pushUpdates(telegramUpdate{
		UpdateID: 1,
		Message: &telegramMsg{
			MessageID: 1,
			From:      &telegramUser{ID: 12345},
			Chat:      telegramChat{ID: 12345},
			Date:      time.Now().Unix(),
			Text:      "/foobar",
		},
	})

	sends := f.waitForSends(1, 3*time.Second)
	if len(sends) != 1 || !strings.Contains(sends[0].text, "Unknown command") {
		t.Errorf("expected 'Unknown command' reply, got %+v", sends)
	}
}
