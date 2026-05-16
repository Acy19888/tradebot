package main

import (
	"fmt"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// telegramCommandHandlerStarted gates StartTelegramCommandHandler so the
// command handler is only attached once per process even if main.go is
// restructured or notifier_build is invoked twice (tests).
var telegramCommandHandlerStarted atomic.Bool

// TelegramCommandHandler wires Telegram /commands into the running scheduler.
//
// Scope (Phase 1a, read-only + kill switch):
//   - /help          — list available commands
//   - /status        — portfolio summary (PnL, drawdown, kill switch state)
//   - /positions     — open spot/perps/options positions per strategy
//   - /killswitch    — fire the portfolio kill switch (operator override)
//
// Auth: every command is gated on the message's `from.id` matching
// cfg.Telegram.OwnerChatID. Non-owner messages are silently ignored.
//
// State access uses the shared scheduler mu (RLock for reads,
// Lock for the kill switch write) — same contract as the rest of the
// scheduler. The handler keeps a reference to *Config so it can resolve
// strategy IDs and platform names without traversing the state map.
type TelegramCommandHandler struct {
	notifier *TelegramNotifier
	owner    string
	state    *AppState
	mu       *sync.RWMutex
	cfg      *Config

	// stateSaver, when non-nil, persists state after a mutating command
	// (currently only /killswitch). Wired in main.go alongside the rest of
	// the SaveStateWithDB plumbing. Nil-safe so tests can omit it.
	stateSaver func() error

	doneCh chan struct{}
	wg     sync.WaitGroup
}

// NewTelegramCommandHandler constructs a handler bound to the given notifier
// and scheduler state. Returns nil if the notifier is nil or the owner chat
// ID is empty — callers should treat nil as "commands disabled".
func NewTelegramCommandHandler(
	notifier *TelegramNotifier,
	owner string,
	state *AppState,
	mu *sync.RWMutex,
	cfg *Config,
	stateSaver func() error,
) *TelegramCommandHandler {
	if notifier == nil || strings.TrimSpace(owner) == "" || state == nil || mu == nil || cfg == nil {
		return nil
	}
	return &TelegramCommandHandler{
		notifier:   notifier,
		owner:      strings.TrimSpace(owner),
		state:      state,
		mu:         mu,
		cfg:        cfg,
		stateSaver: stateSaver,
		doneCh:     make(chan struct{}),
	}
}

// Start subscribes to Telegram updates and spawns a goroutine that processes
// owner-authored slash commands. Returns true if the handler started (broker
// must be running). Idempotent across the whole process.
func (h *TelegramCommandHandler) Start() bool {
	if h == nil {
		return false
	}
	if !telegramCommandHandlerStarted.CompareAndSwap(false, true) {
		return false
	}
	if !h.notifier.brokerStarted.Load() {
		// Broker not running — commands won't fire. Reset the gate so a
		// later wiring attempt (e.g. after StartUpdateBroker is added) can
		// succeed.
		telegramCommandHandlerStarted.Store(false)
		return false
	}

	ch, unsub := h.notifier.Subscribe(func(u telegramUpdate) bool {
		if u.Message == nil || u.Message.From == nil {
			return false
		}
		// Only the configured owner can issue commands.
		if fmt.Sprintf("%d", u.Message.From.ID) != h.owner {
			return false
		}
		return strings.HasPrefix(strings.TrimSpace(u.Message.Text), "/")
	})

	h.wg.Add(1)
	go func() {
		defer h.wg.Done()
		defer unsub()
		for {
			select {
			case <-h.doneCh:
				return
			case u, ok := <-ch:
				if !ok {
					return
				}
				h.handle(u)
			}
		}
	}()
	return true
}

// Stop signals the goroutine to exit and waits for it. Safe to call multiple
// times.
func (h *TelegramCommandHandler) Stop() {
	if h == nil {
		return
	}
	select {
	case <-h.doneCh:
		// already closed
	default:
		close(h.doneCh)
	}
	h.wg.Wait()
}

// handle parses and dispatches a single owner command. Replies are always
// sent back to the originating chat. Errors are formatted into the reply so
// the operator sees them on their phone.
func (h *TelegramCommandHandler) handle(u telegramUpdate) {
	chatID := fmt.Sprintf("%d", u.Message.Chat.ID)
	text := strings.TrimSpace(u.Message.Text)
	if text == "" {
		return
	}
	// Strip optional bot-suffix (Telegram delivers "/status@MyBot" in groups).
	cmd, rest := splitCommand(text)

	var reply string
	switch cmd {
	case "/help", "/start":
		reply = h.cmdHelp()
	case "/status":
		reply = h.cmdStatus()
	case "/positions":
		reply = h.cmdPositions()
	case "/killswitch":
		reply = h.cmdKillSwitch(rest)
	default:
		reply = fmt.Sprintf("Unknown command: %s\nSend /help for the list.", cmd)
	}

	if err := h.notifier.SendMessage(chatID, reply); err != nil {
		fmt.Printf("[telegram-cmd] reply send failed: %v\n", err)
	}
}

// splitCommand returns the lower-cased command token and the remaining
// argument string. Handles "/cmd@BotName arg1 arg2" form used in groups.
func splitCommand(text string) (cmd, rest string) {
	parts := strings.SplitN(text, " ", 2)
	cmd = parts[0]
	if at := strings.Index(cmd, "@"); at >= 0 {
		cmd = cmd[:at]
	}
	cmd = strings.ToLower(cmd)
	if len(parts) > 1 {
		rest = strings.TrimSpace(parts[1])
	}
	return cmd, rest
}

func (h *TelegramCommandHandler) cmdHelp() string {
	return strings.Join([]string{
		"go-trader bot commands:",
		"",
		"/status      — portfolio summary, drawdown, kill switch state",
		"/positions   — open positions per strategy",
		"/killswitch  — fire the portfolio kill switch (CONFIRM required)",
		"/help        — this message",
		"",
		"Owner-only. Non-owner messages are ignored.",
	}, "\n")
}

// cmdStatus renders a compact portfolio summary. Read-only under RLock.
func (h *TelegramCommandHandler) cmdStatus() string {
	h.mu.RLock()
	defer h.mu.RUnlock()

	var b strings.Builder
	b.WriteString("📊 Portfolio Status\n")
	b.WriteString(fmt.Sprintf("Cycle: %d  Last: %s\n",
		h.state.CycleCount,
		formatRelTime(h.state.LastCycle)))

	pr := h.state.PortfolioRisk
	if pr.PeakValue > 0 {
		b.WriteString(fmt.Sprintf("Peak: $%.2f  Drawdown: %.2f%%",
			pr.PeakValue, pr.CurrentDrawdownPct))
		if pr.CurrentMarginDrawdownPct > 0 {
			b.WriteString(fmt.Sprintf("  Margin-DD: %.2f%%", pr.CurrentMarginDrawdownPct))
		}
		b.WriteString("\n")
	}
	if pr.KillSwitchActive {
		b.WriteString(fmt.Sprintf("🛑 KILL SWITCH ACTIVE since %s\n",
			pr.KillSwitchAt.Format("2006-01-02 15:04 UTC")))
	}

	// Strategy summary table.
	ids := make([]string, 0, len(h.state.Strategies))
	for id := range h.state.Strategies {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	if len(ids) == 0 {
		b.WriteString("\nNo strategies in state yet.")
		return b.String()
	}

	b.WriteString("\nStrategies:\n")
	for _, id := range ids {
		s := h.state.Strategies[id]
		if s == nil {
			continue
		}
		openPos := len(s.Positions) + len(s.OptionPositions)
		pnlPct := 0.0
		if s.InitialCapital > 0 {
			pnlPct = (s.Cash - s.InitialCapital) / s.InitialCapital * 100
		}
		breaker := ""
		if s.RiskState.CircuitBreaker {
			breaker = " 🔒CB"
		}
		b.WriteString(fmt.Sprintf("• %-22s  $%.0f → $%.0f  (%+.1f%%)  pos:%d%s\n",
			truncID(id, 22), s.InitialCapital, s.Cash, pnlPct, openPos, breaker))
	}
	return b.String()
}

// cmdPositions lists open positions per strategy in compact form. Read-only.
func (h *TelegramCommandHandler) cmdPositions() string {
	h.mu.RLock()
	defer h.mu.RUnlock()

	ids := make([]string, 0, len(h.state.Strategies))
	for id := range h.state.Strategies {
		ids = append(ids, id)
	}
	sort.Strings(ids)

	var b strings.Builder
	b.WriteString("📂 Open Positions\n")

	totalOpen := 0
	for _, id := range ids {
		s := h.state.Strategies[id]
		if s == nil {
			continue
		}
		if len(s.Positions) == 0 && len(s.OptionPositions) == 0 {
			continue
		}
		b.WriteString(fmt.Sprintf("\n%s (%s)\n", id, s.Type))

		// Spot/perps.
		symbols := make([]string, 0, len(s.Positions))
		for sym := range s.Positions {
			symbols = append(symbols, sym)
		}
		sort.Strings(symbols)
		for _, sym := range symbols {
			p := s.Positions[sym]
			if p == nil {
				continue
			}
			side := "long"
			if strings.EqualFold(p.Side, "short") {
				side = "short"
			}
			b.WriteString(fmt.Sprintf("  %s %s qty=%.4f avg=$%.2f\n",
				side, sym, p.Quantity, p.AvgCost))
			totalOpen++
		}

		// Options.
		optKeys := make([]string, 0, len(s.OptionPositions))
		for k := range s.OptionPositions {
			optKeys = append(optKeys, k)
		}
		sort.Strings(optKeys)
		for _, k := range optKeys {
			o := s.OptionPositions[k]
			if o == nil {
				continue
			}
			b.WriteString(fmt.Sprintf("  opt %s %s qty=%.2f\n",
				o.Action, k, o.Quantity))
			totalOpen++
		}
	}

	if totalOpen == 0 {
		b.WriteString("\nNo open positions.")
	} else {
		b.WriteString(fmt.Sprintf("\n%d open position(s).", totalOpen))
	}
	return b.String()
}

// cmdKillSwitch latches the portfolio kill switch. Requires the argument
// "CONFIRM" to fire — protects against fat-finger taps on the phone.
//
// This does NOT execute the close path itself — that runs in main.go's
// scheduler loop on the next cycle when CheckPortfolioRisk observes the
// active latch and routes through planKillSwitchClose. So the operator sees:
//   - immediate "armed" reply here, with the next-cycle ETA
//   - the existing kill-switch DM/Discord broadcast on the next cycle
func (h *TelegramCommandHandler) cmdKillSwitch(arg string) string {
	if !strings.EqualFold(strings.TrimSpace(arg), "CONFIRM") {
		return strings.Join([]string{
			"⚠️  Kill switch is destructive — closes ALL positions across ALL platforms.",
			"",
			"To fire, send: /killswitch CONFIRM",
		}, "\n")
	}

	// Lock for the full read-modify-persist sequence. Matches the existing
	// kill-switch reset path in main.go (lock → set flag → addKillSwitchEvent
	// → SaveStateWithDB → unlock) so the persisted DB row and in-memory state
	// can't be observed in an inconsistent intermediate state by a concurrent
	// /status reader or a scheduler-loop snapshot.
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.state.PortfolioRisk.KillSwitchActive {
		return "Kill switch already active (latched at " +
			h.state.PortfolioRisk.KillSwitchAt.Format("2006-01-02 15:04 UTC") + ")."
	}
	h.state.PortfolioRisk.KillSwitchActive = true
	h.state.PortfolioRisk.KillSwitchAt = time.Now().UTC()
	addKillSwitchEvent(&h.state.PortfolioRisk, "triggered", "operator_telegram",
		h.state.PortfolioRisk.CurrentDrawdownPct,
		0, h.state.PortfolioRisk.PeakValue,
		"manual fire via Telegram /killswitch")

	if h.stateSaver != nil {
		if err := h.stateSaver(); err != nil {
			fmt.Printf("[telegram-cmd] killswitch state save failed: %v\n", err)
		}
	}

	return strings.Join([]string{
		"🛑 KILL SWITCH ARMED",
		"",
		"Scheduler will flatten all positions on the next cycle.",
		"You will receive the standard kill-switch broadcast when on-chain confirmed flat.",
	}, "\n")
}

// truncID returns a fixed-width string of length at most n, padded with
// trailing spaces. Used to keep the /status table aligned on monospace
// Telegram clients.
func truncID(s string, n int) string {
	if len(s) > n {
		return s[:n-1] + "…"
	}
	return s
}

// formatRelTime renders a time.Time as a short relative-time string suitable
// for /status. Empty time → "never".
func formatRelTime(t time.Time) string {
	if t.IsZero() {
		return "never"
	}
	d := time.Since(t)
	switch {
	case d < time.Minute:
		return fmt.Sprintf("%ds ago", int(d.Seconds()))
	case d < time.Hour:
		return fmt.Sprintf("%dm ago", int(d.Minutes()))
	case d < 24*time.Hour:
		return fmt.Sprintf("%dh ago", int(d.Hours()))
	default:
		return t.Format("2006-01-02 15:04 UTC")
	}
}
