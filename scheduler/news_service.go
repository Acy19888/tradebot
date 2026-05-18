package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"
)

// news_service.go — Phase 3a news-awareness service.
//
// Responsibilities:
//   - Polls shared_scripts/fetch_news.py on a configurable interval.
//   - Persists new (de-duped) items into the news_events SQLite table.
//   - DMs the owner on first-seen high-severity items so the operator can
//     react in real time (or pause new entries manually for now; auto
//     risk-filter lands in Phase 3b).
//
// Explicit non-goals (would make this dangerous, see Phase 3 design doc):
//   - Auto-trading on individual news items. We do not generate buy/sell
//     signals from news here; the scheduler-side risk filter that lands
//     in Phase 3b will consume news_events but never originate trades.
//   - Sentiment-as-only-signal. Sentiment is a tag, not a directive.
//
// Lifecycle: Start(ctx) spawns a single polling goroutine that runs until
// ctx is cancelled or Stop is called. Idempotent — second Start is a no-op.
// All DB writes go through (*StateDB).InsertNewsEvent which uses
// INSERT OR IGNORE on the primary key so re-scrapes are free.

// NewsEvent mirrors one row in the news_events SQLite table. The JSON tags
// match the fetch_news.py output one-to-one so the polling loop can
// json.Unmarshal directly without an intermediate struct.
type NewsEvent struct {
	ID              string    `json:"id"`
	Title           string    `json:"title"`
	URL             string    `json:"url"`
	Source          string    `json:"source"`
	Domain          string    `json:"domain"`
	PublishedAt     string    `json:"published_at"`
	Coins           []string  `json:"coins"`
	Severity        string    `json:"severity"`
	Sentiment       string    `json:"sentiment"`
	VotesImportant  int       `json:"votes_important"`
	SeenAt          time.Time `json:"-"`
	AlertedToOwner  bool      `json:"-"`
}

// NewsService runs the news polling loop. Construct with NewNewsService;
// Start spawns the goroutine; Stop cancels it and waits.
type NewsService struct {
	stateDB        *StateDB
	notifier       *MultiNotifier
	pollInterval   time.Duration
	sinceMinutes   int
	limit          int
	highSevAlerts  bool

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	mu      sync.Mutex
	running bool
	// nextRunOverride lets tests advance the schedule deterministically
	// without touching real time.Sleep.
	nextRunOverride <-chan time.Time
	// fetcher is the function that actually calls fetch_news.py. Tests
	// inject a stub to bypass the subprocess.
	fetcher func(sinceMinutes, limit int) ([]NewsEvent, error)
}

// NewsServiceConfig is the parameter bundle for NewNewsService.
type NewsServiceConfig struct {
	PollInterval  time.Duration // default 10 min
	SinceMinutes  int           // default 1440 (24h)
	Limit         int           // default 100
	HighSevAlerts bool          // default true
}

func (c NewsServiceConfig) withDefaults() NewsServiceConfig {
	if c.PollInterval <= 0 {
		c.PollInterval = 10 * time.Minute
	}
	if c.SinceMinutes <= 0 {
		c.SinceMinutes = 1440
	}
	if c.Limit <= 0 {
		c.Limit = 100
	}
	return c
}

// NewNewsService constructs a service with default real-subprocess fetcher.
// Returns nil if stateDB is nil — callers should treat nil as "news layer
// disabled".
func NewNewsService(stateDB *StateDB, notifier *MultiNotifier, cfg NewsServiceConfig) *NewsService {
	if stateDB == nil {
		return nil
	}
	cfg = cfg.withDefaults()
	ns := &NewsService{
		stateDB:       stateDB,
		notifier:      notifier,
		pollInterval:  cfg.PollInterval,
		sinceMinutes:  cfg.SinceMinutes,
		limit:         cfg.Limit,
		highSevAlerts: cfg.HighSevAlerts,
	}
	ns.fetcher = ns.defaultFetcher
	return ns
}

// Start spawns the polling goroutine. Idempotent — calls after the first
// are no-ops. Safe to call concurrently with Stop (Stop wins). Returns
// true when this call actually starts the loop.
func (ns *NewsService) Start(ctx context.Context) bool {
	if ns == nil {
		return false
	}
	ns.mu.Lock()
	if ns.running {
		ns.mu.Unlock()
		return false
	}
	ns.running = true
	ns.ctx, ns.cancel = context.WithCancel(ctx)
	ns.mu.Unlock()

	ns.wg.Add(1)
	go ns.runLoop()
	return true
}

// Stop cancels the context and waits for the goroutine to exit. Safe to
// call before Start (no-op) and to call multiple times.
func (ns *NewsService) Stop() {
	if ns == nil {
		return
	}
	ns.mu.Lock()
	cancel := ns.cancel
	running := ns.running
	ns.mu.Unlock()
	if !running || cancel == nil {
		return
	}
	cancel()
	ns.wg.Wait()
	ns.mu.Lock()
	ns.running = false
	ns.mu.Unlock()
}

func (ns *NewsService) runLoop() {
	defer ns.wg.Done()
	// First poll immediately so the operator sees data without waiting one
	// interval after boot.
	ns.pollOnce()
	for {
		var tick <-chan time.Time
		if ns.nextRunOverride != nil {
			tick = ns.nextRunOverride
		} else {
			timer := time.NewTimer(ns.pollInterval)
			tick = timer.C
			defer timer.Stop()
		}
		select {
		case <-ns.ctx.Done():
			return
		case <-tick:
			ns.pollOnce()
		}
	}
}

// pollOnce fetches the latest news batch and persists / alerts any new
// items. Always returns even on errors — the polling loop logs and moves on.
func (ns *NewsService) pollOnce() {
	events, err := ns.fetcher(ns.sinceMinutes, ns.limit)
	if err != nil {
		fmt.Printf("[news] fetch failed: %v\n", err)
		return
	}
	now := time.Now().UTC()
	highSevNew := []NewsEvent{}
	for _, e := range events {
		if e.ID == "" || e.Title == "" {
			continue
		}
		e.SeenAt = now
		inserted, err := ns.stateDB.InsertNewsEvent(e)
		if err != nil {
			fmt.Printf("[news] insert %s: %v\n", e.ID, err)
			continue
		}
		if !inserted {
			// Duplicate by primary-key (we've seen this event already);
			// do not re-alert.
			continue
		}
		if ns.highSevAlerts && e.Severity == "high" {
			highSevNew = append(highSevNew, e)
		}
	}
	if len(highSevNew) > 0 && ns.notifier != nil && ns.notifier.HasOwner() {
		ns.sendHighSeverityAlerts(highSevNew)
	}
}

func (ns *NewsService) sendHighSeverityAlerts(events []NewsEvent) {
	// Cap the alert burst — if a feed dumps 50 high-severity items in one
	// poll (which would itself be suspicious), don't carpet-bomb the owner.
	const maxAlerts = 5
	if len(events) > maxAlerts {
		events = events[:maxAlerts]
	}
	for _, e := range events {
		msg := formatHighSeverityNewsAlert(e)
		ns.notifier.SendOwnerDM(msg)
		// Mark as alerted so a future restart doesn't re-DM the same item
		// after re-classifying it as high severity.
		if err := ns.stateDB.MarkNewsEventAlerted(e.ID); err != nil {
			fmt.Printf("[news] mark alerted %s: %v\n", e.ID, err)
		}
	}
}

func formatHighSeverityNewsAlert(e NewsEvent) string {
	var sb strings.Builder
	sb.WriteString("📰 HIGH-SEVERITY NEWS\n")
	sb.WriteString(e.Title)
	sb.WriteString("\n")
	if e.Source != "" {
		sb.WriteString("Source: " + e.Source)
		if e.Sentiment != "" && e.Sentiment != "neutral" {
			sb.WriteString(" · " + strings.Title(e.Sentiment))
		}
		sb.WriteString("\n")
	}
	if len(e.Coins) > 0 {
		sb.WriteString("Coins: " + strings.Join(e.Coins, ", ") + "\n")
	}
	if e.URL != "" {
		sb.WriteString(e.URL)
	}
	return sb.String()
}

// defaultFetcher invokes the Python news fetcher script via the shared
// runPythonReadOnly helper. Tests override fetcher to skip the subprocess.
func (ns *NewsService) defaultFetcher(sinceMinutes, limit int) ([]NewsEvent, error) {
	stdout, _, err := runPythonReadOnly("shared_scripts/fetch_news.py", []string{
		"--since-minutes", fmt.Sprintf("%d", sinceMinutes),
		"--limit", fmt.Sprintf("%d", limit),
	})
	if err != nil {
		// fetch_news.py emits a JSON error object on stdout AND exits
		// non-zero. Try to decode it for a cleaner log line.
		if len(stdout) > 0 {
			var errObj struct {
				Error string `json:"error"`
			}
			if jerr := json.Unmarshal(stdout, &errObj); jerr == nil && errObj.Error != "" {
				return nil, fmt.Errorf("fetch_news.py: %s", errObj.Error)
			}
		}
		return nil, fmt.Errorf("fetch_news.py: %w", err)
	}
	var items []NewsEvent
	if err := json.Unmarshal(stdout, &items); err != nil {
		return nil, fmt.Errorf("decode news payload: %w", err)
	}
	return items, nil
}

// ---------------------------------------------------------------------------
// StateDB helpers for news_events. Kept in this file so the news layer is
// self-contained — easy to lift out as a separate package later.
// ---------------------------------------------------------------------------

// InsertNewsEvent inserts a news event using INSERT OR IGNORE on the primary
// key. Returns true if a NEW row was added (so the caller can decide whether
// to alert). Coins are serialized to a JSON array string.
func (sdb *StateDB) InsertNewsEvent(e NewsEvent) (bool, error) {
	coinsJSON, _ := json.Marshal(e.Coins)
	res, err := sdb.db.Exec(`INSERT OR IGNORE INTO news_events
        (id, title, url, source, domain, published_at, coins, severity, sentiment, votes_important, seen_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		e.ID, e.Title, e.URL, e.Source, e.Domain, e.PublishedAt,
		string(coinsJSON), e.Severity, e.Sentiment, e.VotesImportant,
		formatTime(e.SeenAt))
	if err != nil {
		return false, fmt.Errorf("insert news_events: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return false, fmt.Errorf("rows affected: %w", err)
	}
	return n > 0, nil
}

// MarkNewsEventAlerted flips alerted = 1 so restarts don't re-DM the same
// item if the classifier re-runs and re-tags it as high severity.
func (sdb *StateDB) MarkNewsEventAlerted(id string) error {
	_, err := sdb.db.Exec("UPDATE news_events SET alerted = 1 WHERE id = ?", id)
	if err != nil {
		return fmt.Errorf("mark alerted: %w", err)
	}
	return nil
}

// QueryRecentNewsEvents returns the most recent N events optionally
// filtered by minimum severity ("low"|"medium"|"high") and coin
// substring (case-insensitive prefix match against the coins JSON).
// Caller passes limit=0 for an unbounded read (capped at 500 here).
func (sdb *StateDB) QueryRecentNewsEvents(minSeverity, coin string, limit int) ([]NewsEvent, error) {
	if limit <= 0 || limit > 500 {
		limit = 50
	}
	var where []string
	var args []interface{}
	switch strings.ToLower(minSeverity) {
	case "high":
		where = append(where, "severity = 'high'")
	case "medium":
		where = append(where, "severity IN ('high','medium')")
	case "low", "":
		// no filter
	}
	if coin != "" {
		where = append(where, "UPPER(coins) LIKE ?")
		args = append(args, "%\""+strings.ToUpper(coin)+"\"%")
	}
	whereClause := ""
	if len(where) > 0 {
		whereClause = "WHERE " + strings.Join(where, " AND ")
	}
	q := fmt.Sprintf(`SELECT id, title, url, source, domain, published_at, coins, severity, sentiment, votes_important, COALESCE(seen_at, ''), COALESCE(alerted, 0)
        FROM news_events %s ORDER BY published_at DESC LIMIT ?`, whereClause)
	args = append(args, limit)
	rows, err := sdb.db.Query(q, args...)
	if err != nil {
		return nil, fmt.Errorf("query news_events: %w", err)
	}
	defer rows.Close()
	var out []NewsEvent
	for rows.Next() {
		var e NewsEvent
		var coinsJSON, seenAt string
		var alerted int
		if err := rows.Scan(&e.ID, &e.Title, &e.URL, &e.Source, &e.Domain,
			&e.PublishedAt, &coinsJSON, &e.Severity, &e.Sentiment,
			&e.VotesImportant, &seenAt, &alerted); err != nil {
			return nil, fmt.Errorf("scan news_events: %w", err)
		}
		if coinsJSON != "" {
			_ = json.Unmarshal([]byte(coinsJSON), &e.Coins)
		}
		if seenAt != "" {
			e.SeenAt = parseTime(seenAt)
		}
		e.AlertedToOwner = alerted != 0
		out = append(out, e)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate news_events: %w", err)
	}
	if out == nil {
		out = []NewsEvent{}
	}
	return out, nil
}
