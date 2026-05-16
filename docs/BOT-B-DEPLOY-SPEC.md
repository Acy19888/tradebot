# Bot-B Server-Setup — Hermes Deployment Spec

Gib diese Datei Hermes (oder einem anderen Agent) auf dem neuen Ubuntu-Server.
Das ist eine vollständige Anweisung — Hermes soll **exakt** so vorgehen.

## Kontext

- **Bot-A** läuft schon auf einem existierenden Server (Production-Baseline).
- **Bot-B** ist der neue, verbesserte Bot — gleiche Strategien, aber neue Features.
- **Modus**: ausschließlich `paper` — beide Bots traden nur Paper-Money für mindestens 4-6 Wochen.
- **Ziel**: A/B-Vergleich. Welcher Bot performt besser? Sharpe, Max-DD, Win-Rate, Total-Return.
- Damit der Vergleich gilt: **gleiche Strategien, gleiches Start-Capital, separate Telegram-Channels, separate `state.db`**.

## Server-Prereqs (Ubuntu 22.04+ / 24.04 / 26.04)

```bash
# System updates
sudo apt update && sudo apt upgrade -y

# Build-Tools + Python + Git
sudo apt install -y build-essential git curl python3.12 python3.12-venv

# Go installieren (1.23+ erforderlich)
curl -fsSL https://go.dev/dl/go1.23.4.linux-amd64.tar.gz -o /tmp/go.tgz
sudo tar -C /usr/local -xzf /tmp/go.tgz
echo 'export PATH=$PATH:/usr/local/go/bin' | sudo tee /etc/profile.d/go.sh
source /etc/profile.d/go.sh
go version  # sollte go1.23.x ausgeben

# uv (Python Package Manager) installieren
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
uv --version
```

## Bot installieren

```bash
# In /opt/go-trader installieren
sudo mkdir -p /opt/go-trader && sudo chown $USER:$USER /opt/go-trader

# Repo clonen (Cems Fork, feature-Branch mit Phase 1a)
cd /opt
git clone https://github.com/Acy19888/tradebot.git go-trader
cd go-trader
git checkout feature/telegram-commands

# Python-Deps installieren
uv sync

# Go-Binary bauen
cd scheduler
/usr/local/go/bin/go build -ldflags "-X main.Version=$(git describe --tags --always --dirty=-mod)" -o ../go-trader .
cd ..

# Tests laufen lassen — MUSS grün sein, sonst Setup stoppen
cd scheduler && /usr/local/go/bin/go test -run "Telegram|Command" -v
# Erwartet: 16/16 PASS
cd ..
```

## Config für Bot-B erzeugen

Konfiguration:

```bash
cp scheduler/config.example.json scheduler/config.json
```

Dann `scheduler/config.json` editieren — **diese Werte für Bot-B** (separat von Bot-A halten!):

```json
{
  "config_version": 12,
  "interval_seconds": 300,
  "db_file": "scheduler/state.db",
  "status_port": 8099,
  "default_stop_loss_atr_mult": 1.0,
  "portfolio_risk": {
    "max_drawdown_pct": 25,
    "max_notional_usd": 0
  },
  "strategies": [
    {
      "id": "hl-momentum-btc",
      "type": "perps",
      "script": "shared_scripts/check_hyperliquid.py",
      "args": ["momentum", "BTC", "1h", "--mode=paper"],
      "capital": 1000,
      "max_drawdown_pct": 50,
      "interval_seconds": 300
    },
    {
      "id": "hl-amd-btc",
      "type": "perps",
      "script": "shared_scripts/check_hyperliquid.py",
      "args": ["amd_ifvg", "BTC", "15m", "--mode=paper"],
      "capital": 1000,
      "max_drawdown_pct": 50,
      "interval_seconds": 900
    }
  ],
  "telegram": {
    "enabled": true,
    "bot_token": "",
    "owner_chat_id": "DEINE_OWNER_CHAT_ID",
    "channels": {
      "hyperliquid-paper": "BOT_B_CHANNEL_ID"
    },
    "dm_channels": {
      "hyperliquid-paper": "BOT_B_CHANNEL_ID"
    }
  },
  "summary_frequency": {
    "hyperliquid": "every"
  }
}
```

**WICHTIG**: 
- `BOT_B_CHANNEL_ID` muss eine **neue** Telegram-Channel-ID sein, **nicht** dieselbe wie Bot-A. Sonst mischen sich die Alerts. Cem soll einen neuen Channel/Topic für Bot-B anlegen.
- `DEINE_OWNER_CHAT_ID` ist dieselbe wie bei Bot-A (Cems persönliche Chat-ID).
- Bei den Strategien: **GENAU dieselben wie auf Bot-A**, damit der A/B-Vergleich gültig ist. Cems Bot-A-Config muss Hermes von Cem bekommen.

## Env-Vars setzen (Bot-Token niemals in config.json!)

```bash
sudo mkdir -p /etc/go-trader
sudo tee /etc/go-trader/env > /dev/null <<'EOF'
TELEGRAM_BOT_TOKEN=DEIN_BOT_TOKEN
TELEGRAM_OWNER_CHAT_ID=DEINE_OWNER_CHAT_ID
EOF
sudo chmod 600 /etc/go-trader/env
```

## Smoke-Test

```bash
cd /opt/go-trader
# Einen einzelnen Cycle laufen lassen
./go-trader --config scheduler/config.json --once
# Erwartet:
#  - "Telegram bot connected (...)"
#  - "Telegram /commands enabled (owner-authed)"
#  - Eine Cycle-Ausführung pro Strategie ohne Errors
```

Wenn das durchläuft, ist Bot-B grundsätzlich lauffähig.

## systemd-Service

```bash
sudo tee /etc/systemd/system/go-trader.service > /dev/null <<'EOF'
[Unit]
Description=go-trader (Bot-B)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=acy88
WorkingDirectory=/opt/go-trader
EnvironmentFile=/etc/go-trader/env
ExecStart=/opt/go-trader/go-trader --config /opt/go-trader/scheduler/config.json
Restart=on-failure
RestartSec=10
TimeoutStopSec=20

# Hardening
ProtectSystem=strict
ReadWritePaths=/opt/go-trader
ProtectHome=read-only
NoNewPrivileges=yes
PrivateTmp=yes

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now go-trader
sudo systemctl status go-trader
sudo journalctl -u go-trader -f -n 50
```

## Verifizieren

```bash
# Im Journal erwartet:
#   Telegram bot connected (X channels, DM owner enabled, commands enabled)
#   Telegram /commands enabled (owner-authed)
#   [server] Dashboard at http://localhost:8099/dashboard

# HTTP Status
curl -s http://localhost:8099/status | python3 -m json.tool | head -30

# Telegram-Bot anchatten
# Auf dem Handy: /status — Bot-B sollte mit "Cycle: N" und Strategy-Liste antworten.
# Wenn Antwort kommt: Bot-B läuft.
```

## Wichtige Hinweise für Hermes

1. **Niemals Token in `config.json` schreiben**. Nur in `/etc/go-trader/env`.
2. **`config.json` enthält keine Live-Mode-Args**. Alle Strategien müssen `"--mode=paper"` haben. Falls Hermes Live-Configs vorfindet: stoppen und Cem fragen.
3. **Wenn Tests nicht grün sind**: stoppen, Output an Cem zurück, NICHT weiter installieren.
4. **Wenn der Build fehlschlägt**: stoppen, Compiler-Output zurück.
5. **Wenn systemd-Service nicht startet**: `journalctl -u go-trader -n 100` ausführen, Output zurück.
6. **Cems Bot-A-Config**: Hermes braucht die `config.json` von Bot-A (per scp oder Cem kopiert sie rüber), um identische Strategien zu konfigurieren. Sonst A/B-Vergleich ungültig.

## Was Bot-B kann das Bot-A noch nicht hat

- Telegram-Slash-Commands: `/help`, `/status`, `/positions`, `/killswitch CONFIRM`
- Race-freier Telegram-Update-Broker (Subscriber-Pattern)
- Kill-Switch-Audit-Trail mit Source-Tracking ("operator_telegram")

Mehr Features kommen in den nächsten Branches (`feature/dashboard-v2`, `feature/news-alerts`, `feature/robinhood-crypto-official`).
