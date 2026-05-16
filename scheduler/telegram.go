package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const telegramAPIBase = "https://api.telegram.org/bot"
const telegramMaxMessageLen = 4096

// updateSubscription is a single registered listener for Telegram updates
// dispatched by the broker (see StartUpdateBroker). Filter returns true to
// deliver the update on ch; channels are buffered so a slow consumer drops
// updates rather than stalling the broker.
type updateSubscription struct {
	id     int64
	ch     chan telegramUpdate
	filter func(telegramUpdate) bool
}

// TelegramNotifier implements Notifier using the Telegram Bot API.
type TelegramNotifier struct {
	botToken    string
	ownerChatID string
	client      *http.Client
	baseURL     string // API base URL (defaults to telegramAPIBase)
	lastUpdate  int64  // offset for getUpdates polling
	mu          sync.Mutex
	closed      bool

	// Update broker (opt-in via StartUpdateBroker). When running, AskDM and
	// any other consumer (e.g. TelegramCommandHandler) Subscribe instead of
	// polling getUpdates directly — avoids racing on lastUpdate offset.
	subMu         sync.Mutex
	subscribers   []*updateSubscription
	subSeq        int64
	brokerCtx     context.Context
	brokerCancel  context.CancelFunc
	brokerWG      sync.WaitGroup
	brokerStarted atomic.Bool
}

// NewTelegramNotifier creates a new Telegram bot notifier.
func NewTelegramNotifier(botToken, ownerChatID string) (*TelegramNotifier, error) {
	t := &TelegramNotifier{
		botToken:    botToken,
		ownerChatID: ownerChatID,
		client:      &http.Client{Timeout: 35 * time.Second},
		baseURL:     telegramAPIBase,
	}

	// Verify the bot token with a getMe call.
	resp, err := t.apiCall("getMe", nil)
	if err != nil {
		return nil, fmt.Errorf("telegram getMe failed: %w", err)
	}
	if !resp.OK {
		return nil, fmt.Errorf("telegram getMe: %s", resp.Description)
	}

	return t, nil
}

// StartUpdateBroker spawns a single long-running goroutine that polls
// getUpdates and fans out each update to all matching subscribers. Idempotent:
// calling more than once is a no-op. Must be called AFTER NewTelegramNotifier
// succeeds (so getMe has verified the token). Production code calls this from
// notifier_build.go; tests that only exercise SendMessage/SendDM may skip it.
func (t *TelegramNotifier) StartUpdateBroker() {
	if !t.brokerStarted.CompareAndSwap(false, true) {
		return
	}
	t.brokerCtx, t.brokerCancel = context.WithCancel(context.Background())
	t.brokerWG.Add(1)
	go t.brokerLoop()
}

// Subscribe registers a filtered update listener. Returns the channel updates
// arrive on and an unsubscribe function that MUST be called (defer is fine) to
// remove the subscription and release the buffered channel. Filter is called
// with the broker's subMu held — keep it cheap and side-effect-free.
//
// When the broker is not running, returns a closed channel and a no-op
// unsubscribe so callers can write code that works in either mode.
func (t *TelegramNotifier) Subscribe(filter func(telegramUpdate) bool) (<-chan telegramUpdate, func()) {
	if !t.brokerStarted.Load() {
		closed := make(chan telegramUpdate)
		close(closed)
		return closed, func() {}
	}
	sub := &updateSubscription{
		id:     atomic.AddInt64(&t.subSeq, 1),
		ch:     make(chan telegramUpdate, 16),
		filter: filter,
	}
	t.subMu.Lock()
	t.subscribers = append(t.subscribers, sub)
	t.subMu.Unlock()
	return sub.ch, func() { t.unsubscribe(sub.id) }
}

func (t *TelegramNotifier) unsubscribe(id int64) {
	t.subMu.Lock()
	defer t.subMu.Unlock()
	for i, s := range t.subscribers {
		if s.id == id {
			t.subscribers = append(t.subscribers[:i], t.subscribers[i+1:]...)
			close(s.ch)
			return
		}
	}
}

// brokerLoop is the single owner of getUpdates polling once the broker is
// started. It runs until brokerCtx is cancelled (by Close). Errors are logged
// at most once per logInterval to avoid flooding when the bot is unreachable.
func (t *TelegramNotifier) brokerLoop() {
	defer t.brokerWG.Done()

	const pollTimeoutSec = 25
	const errLogInterval = 60 * time.Second
	var lastErrLog time.Time

	for {
		select {
		case <-t.brokerCtx.Done():
			return
		default:
		}
		updates, err := t.getUpdates(pollTimeoutSec)
		if err != nil {
			// Throttled error logging — broker keeps polling.
			if time.Since(lastErrLog) > errLogInterval {
				fmt.Printf("[telegram] broker getUpdates: %v\n", err)
				lastErrLog = time.Now()
			}
			// Light backoff so we don't hammer the API in a tight error loop.
			select {
			case <-t.brokerCtx.Done():
				return
			case <-time.After(2 * time.Second):
			}
			continue
		}
		for _, u := range updates {
			t.dispatch(u)
		}
	}
}

// dispatch delivers an update to every subscriber whose filter returns true.
// Non-blocking — if a subscriber's channel is full, the update is dropped for
// that subscriber (logged once). Keeps the broker loop pumping regardless of
// slow consumers.
func (t *TelegramNotifier) dispatch(u telegramUpdate) {
	t.subMu.Lock()
	subs := make([]*updateSubscription, len(t.subscribers))
	copy(subs, t.subscribers)
	t.subMu.Unlock()

	for _, s := range subs {
		if s.filter != nil && !s.filter(u) {
			continue
		}
		select {
		case s.ch <- u:
		default:
			fmt.Printf("[telegram] dropping update %d for subscriber %d (channel full)\n", u.UpdateID, s.id)
		}
	}
}

// telegramResponse is the generic Telegram Bot API response envelope.
type telegramResponse struct {
	OK          bool            `json:"ok"`
	Description string          `json:"description,omitempty"`
	Result      json.RawMessage `json:"result,omitempty"`
}

// telegramUpdate represents a single update from getUpdates.
type telegramUpdate struct {
	UpdateID int64           `json:"update_id"`
	Message  *telegramMsg    `json:"message,omitempty"`
	Callback *telegramCBData `json:"callback_query,omitempty"`
}

type telegramMsg struct {
	MessageID int64         `json:"message_id"`
	From      *telegramUser `json:"from,omitempty"`
	Chat      telegramChat  `json:"chat"`
	Date      int64         `json:"date"`
	Text      string        `json:"text"`
}

type telegramUser struct {
	ID int64 `json:"id"`
}

type telegramChat struct {
	ID int64 `json:"id"`
}

type telegramCBData struct {
	ID      string        `json:"id"`
	From    *telegramUser `json:"from,omitempty"`
	Message *telegramMsg  `json:"message,omitempty"`
	Data    string        `json:"data"`
}

// apiCall makes a POST request to the Telegram Bot API.
func (t *TelegramNotifier) apiCall(method string, payload interface{}) (*telegramResponse, error) {
	url := t.baseURL + t.botToken + "/" + method

	var body io.Reader
	if payload != nil {
		data, err := json.Marshal(payload)
		if err != nil {
			return nil, fmt.Errorf("marshal payload: %w", err)
		}
		body = bytes.NewReader(data)
	}

	req, err := http.NewRequest("POST", url, body)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := t.client.Do(req)
	if err != nil {
		// Redact bot token from error to prevent leaking in logs.
		safeMsg := strings.ReplaceAll(err.Error(), t.botToken, "[REDACTED]")
		return nil, fmt.Errorf("telegram %s: %s", method, safeMsg)
	}
	defer resp.Body.Close()

	var result telegramResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	return &result, nil
}

// SendMessage sends a message to a Telegram chat. Truncates to 4096 chars.
func (t *TelegramNotifier) SendMessage(chatID string, content string) error {
	if len(content) > telegramMaxMessageLen {
		content = content[:telegramMaxMessageLen-3] + "..."
	}

	payload := map[string]interface{}{
		"chat_id": chatID,
		"text":    content,
	}

	resp, err := t.apiCall("sendMessage", payload)
	if err != nil {
		return fmt.Errorf("telegram sendMessage: %w", err)
	}
	if !resp.OK {
		return fmt.Errorf("telegram sendMessage: %s", resp.Description)
	}
	return nil
}

// SendDM sends a direct message to a user via their chat ID.
// In Telegram, DMs and channel messages use the same sendMessage API.
func (t *TelegramNotifier) SendDM(userID, content string) error {
	return t.SendMessage(userID, content)
}

// AskDM sends a question to the user and waits for a reply within the timeout.
// When the broker is running (production), subscribes to filtered updates so
// AskDM does not race with other consumers (e.g. TelegramCommandHandler) for
// the same getUpdates offset. When the broker is not running (legacy tests),
// falls back to direct inline polling of getUpdates.
func (t *TelegramNotifier) AskDM(userID, question string, timeout time.Duration) (string, error) {
	sentAt := time.Now().Unix()

	if err := t.SendDM(userID, question); err != nil {
		return "", fmt.Errorf("send question: %w", err)
	}

	// Broker path — race-free single-poller mode.
	if t.brokerStarted.Load() {
		ch, unsub := t.Subscribe(func(u telegramUpdate) bool {
			if u.Message == nil || u.Message.From == nil {
				return false
			}
			fromID := fmt.Sprintf("%d", u.Message.From.ID)
			return fromID == userID && u.Message.Date >= sentAt-2
		})
		defer unsub()

		select {
		case u, ok := <-ch:
			if !ok {
				return "", ErrDMTimeout
			}
			return strings.TrimSpace(u.Message.Text), nil
		case <-time.After(timeout):
			return "", ErrDMTimeout
		}
	}

	// Legacy inline-polling path — preserved so tests that instantiate
	// TelegramNotifier without calling StartUpdateBroker still work.
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		t.mu.Lock()
		if t.closed {
			t.mu.Unlock()
			return "", ErrDMTimeout
		}
		t.mu.Unlock()

		remaining := time.Until(deadline)
		pollTimeout := 10 // seconds for long-polling
		if remaining < time.Duration(pollTimeout)*time.Second {
			pollTimeout = int(remaining.Seconds())
			if pollTimeout < 1 {
				pollTimeout = 1
			}
		}

		updates, err := t.getUpdates(pollTimeout)
		if err != nil {
			// Transient error — retry after a short wait.
			time.Sleep(1 * time.Second)
			continue
		}

		for _, u := range updates {
			if u.Message != nil && u.Message.From != nil {
				fromID := fmt.Sprintf("%d", u.Message.From.ID)
				if fromID == userID && u.Message.Date >= sentAt-2 {
					return strings.TrimSpace(u.Message.Text), nil
				}
			}
		}
	}

	return "", ErrDMTimeout
}

// getUpdates polls for new messages using Telegram long polling.
func (t *TelegramNotifier) getUpdates(timeoutSec int) ([]telegramUpdate, error) {
	payload := map[string]interface{}{
		"timeout": timeoutSec,
	}
	t.mu.Lock()
	if t.lastUpdate > 0 {
		payload["offset"] = t.lastUpdate + 1
	}
	t.mu.Unlock()

	resp, err := t.apiCall("getUpdates", payload)
	if err != nil {
		return nil, err
	}
	if !resp.OK {
		return nil, fmt.Errorf("getUpdates: %s", resp.Description)
	}

	var updates []telegramUpdate
	if err := json.Unmarshal(resp.Result, &updates); err != nil {
		return nil, fmt.Errorf("unmarshal updates: %w", err)
	}

	t.mu.Lock()
	for _, u := range updates {
		if u.UpdateID > t.lastUpdate {
			t.lastUpdate = u.UpdateID
		}
	}
	t.mu.Unlock()

	return updates, nil
}

// Close marks the notifier as closed and stops any pending polling. When the
// update broker has been started it also cancels the broker context and waits
// for the polling goroutine to exit, then closes any remaining subscriber
// channels so subscribers waiting on <-ch unblock cleanly.
func (t *TelegramNotifier) Close() {
	t.mu.Lock()
	if t.closed {
		t.mu.Unlock()
		return
	}
	t.closed = true
	t.mu.Unlock()

	if t.brokerStarted.Load() && t.brokerCancel != nil {
		t.brokerCancel()
		t.brokerWG.Wait()
	}

	// Close any remaining subscriber channels so consumers unblock.
	t.subMu.Lock()
	for _, s := range t.subscribers {
		close(s.ch)
	}
	t.subscribers = nil
	t.subMu.Unlock()
}

// FormatTradeDMPlain formats a Trade into a plain-text DM (no Discord markdown).
func FormatTradeDMPlain(sc StrategyConfig, trade Trade, mode string) string {
	isClose := isTradeCloseDetails(trade.Details)

	icon := "🟢"
	header := "TRADE EXECUTED"
	if isClose {
		icon = "🔴"
		header = "TRADE CLOSED"
	}

	platformLabel := sc.Platform
	if len(platformLabel) > 0 {
		platformLabel = strings.ToUpper(platformLabel[:1]) + platformLabel[1:]
	}
	typeLabel := sc.Type

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("%s %s - %s\n", icon, header, strings.ToUpper(mode)))
	sb.WriteString(fmt.Sprintf("Strategy: %s (%s %s)\n", sc.ID, platformLabel, typeLabel))
	sb.WriteString(fmt.Sprintf("%s — %s %.3f @ $%s | Value: $%s", trade.Symbol, tradeDirectionLabel(trade), trade.Quantity, fmtComma(trade.Price), fmtComma(trade.Value)))
	if oid := strings.TrimSpace(trade.ExchangeOrderID); oid != "" {
		sb.WriteString(fmt.Sprintf(" | OID: %s", oid))
	}
	sb.WriteString("\n")

	if extras := tradeAlertExtras(sc, trade, isClose); len(extras) > 0 {
		sb.WriteString(strings.Join(extras, " | "))
	}

	return sb.String()
}
