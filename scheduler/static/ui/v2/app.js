"use strict";

// Dashboard v2 — single-page app. No build step, runs as a plain <script>.
// Talks to /api/v2/* endpoints (see scheduler/dashboard_v2.go).

const API = {
  portfolio: "/api/v2/portfolio",
  pnl: "/api/v2/pnl",
  equity: (window, strategy) =>
    `/api/v2/equity-curve?since=${encodeURIComponent(window)}${
      strategy ? "&strategy=" + encodeURIComponent(strategy) : ""
    }`,
  trades: (params) => {
    const q = new URLSearchParams(params).toString();
    return `/api/v2/trades?${q}`;
  },
  strategies: "/api/strategies",
};

const state = {
  window: "7d",
  chart: null,
  series: null,
  filterStrategy: "",
  filterResult: "all",
  filterLimit: 50,
  view: "overview",
  refreshTimer: null,
};

function $(sel) {
  return document.querySelector(sel);
}
function $$(sel) {
  return Array.from(document.querySelectorAll(sel));
}

function fmtUSD(v, opts = {}) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  const sign = v < 0 ? "-" : "";
  const abs = Math.abs(v);
  let str;
  if (abs >= 1000) {
    str = abs.toLocaleString("en-US", { maximumFractionDigits: 0 });
  } else {
    str = abs.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }
  return (opts.signed && v > 0 ? "+" : sign) + "$" + str;
}

function fmtPct(v, opts = {}) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  const sign = v > 0 ? "+" : "";
  return sign + v.toFixed(2) + "%";
}

function fmtQty(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  if (Math.abs(v) >= 1) return v.toFixed(4);
  return v.toFixed(6);
}

function fmtAgo(iso) {
  if (!iso) return "—";
  const t = new Date(iso);
  const diff = (Date.now() - t.getTime()) / 1000;
  if (diff < 60) return Math.floor(diff) + "s ago";
  if (diff < 3600) return Math.floor(diff / 60) + "m ago";
  if (diff < 86400) return Math.floor(diff / 3600) + "h ago";
  return t.toLocaleDateString();
}

function fmtTimestamp(iso) {
  if (!iso) return "—";
  const t = new Date(iso);
  return t.toLocaleString();
}

function signClass(v) {
  if (v > 0) return "positive";
  if (v < 0) return "negative";
  return "";
}

async function getJSON(url) {
  const res = await fetch(url, { headers: { Accept: "application/json" } });
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json();
}

// ============================================================
// Overview view rendering
// ============================================================

function renderPortfolio(data) {
  $("#kpi-value").textContent = fmtUSD(data.total_value);
  $("#kpi-value-sub").textContent =
    "init " + fmtUSD(data.total_initial_capital);

  const totalPnl = data.total_realized_pnl;
  const totalPnlPct =
    data.total_initial_capital > 0
      ? (totalPnl / data.total_initial_capital) * 100
      : 0;
  const valueClass = signClass(totalPnl);
  $("#kpi-value").classList.toggle("positive", totalPnl > 0);
  $("#kpi-value").classList.toggle("negative", totalPnl < 0);

  $("#kpi-dd").textContent = fmtPct(-data.current_drawdown_pct);
  $("#kpi-dd").classList.toggle("negative", data.current_drawdown_pct > 0.1);
  $("#kpi-dd-sub").textContent = "peak " + fmtUSD(data.peak_value);

  $("#kpi-counts").textContent =
    data.cycle_count +
    " · " +
    data.strategies_count +
    " · " +
    data.open_positions_count;

  $("#last-cycle").textContent = "Last cycle: " + fmtAgo(data.last_cycle);
  $("#kill-switch-banner").hidden = !data.kill_switch_active;
}

function renderPnL(data, currentWindow) {
  const grid = $(".pnl-grid");
  grid.innerHTML = "";
  if (!data || !data.windows) return;
  data.windows.forEach((w) => {
    const card = document.createElement("div");
    card.className = "pnl-mini";
    const cls = signClass(w.net_pnl);
    card.innerHTML = `
      <div class="pnl-mini-label">${w.key.toUpperCase()}</div>
      <div class="pnl-mini-value ${cls}">${fmtUSD(w.net_pnl, { signed: true })}</div>
      <div class="pnl-mini-meta ${cls}">${fmtPct(w.pct_of_initial)} · ${w.trade_count}T · ${w.wins}W/${w.losses}L</div>
    `;
    grid.appendChild(card);

    // Mirror the active-window card into the top KPI strip.
    if (w.key === currentWindow) {
      $("#kpi-pnl-window").textContent = fmtUSD(w.net_pnl, { signed: true });
      $("#kpi-pnl-window").classList.toggle("positive", w.net_pnl > 0);
      $("#kpi-pnl-window").classList.toggle("negative", w.net_pnl < 0);
      $("#kpi-pnl-window-sub").textContent =
        fmtPct(w.pct_of_initial) +
        " · " +
        w.trade_count +
        " trades · " +
        w.wins +
        "W/" +
        w.losses +
        "L";
      $("#kpi-pnl-window-sub").className =
        "kpi-sub " + signClass(w.net_pnl);
    }
  });
}

function ensureChart() {
  if (state.chart) return state.chart;
  const host = $("#chart");
  state.chart = LightweightCharts.createChart(host, {
    width: host.clientWidth,
    height: host.clientHeight,
    layout: {
      background: { color: "transparent" },
      textColor: "#94a3b8",
    },
    grid: {
      vertLines: { color: "#334155" },
      horzLines: { color: "#334155" },
    },
    rightPriceScale: { borderColor: "#334155" },
    timeScale: { borderColor: "#334155", timeVisible: true, secondsVisible: false },
    crosshair: { mode: 0 },
  });
  state.series = state.chart.addAreaSeries({
    lineColor: "#38bdf8",
    topColor: "rgba(56, 189, 248, 0.4)",
    bottomColor: "rgba(56, 189, 248, 0.04)",
    lineWidth: 2,
  });
  window.addEventListener("resize", () => {
    if (state.chart) {
      state.chart.resize(host.clientWidth, host.clientHeight);
    }
  });
  return state.chart;
}

function renderEquity(data) {
  ensureChart();
  const points = (data.points || []).map((p) => ({
    time: Math.floor(new Date(p.t).getTime() / 1000),
    value: p.v,
  }));
  // Lightweight-Charts requires strictly-increasing, unique timestamps.
  const dedupe = new Map();
  points.forEach((p) => dedupe.set(p.time, p));
  const arr = Array.from(dedupe.values()).sort((a, b) => a.time - b.time);
  state.series.setData(arr);
  $("#chart-empty").hidden = arr.length > 0;
  $("#chart").style.display = arr.length > 0 ? "" : "none";
  $("#equity-summary").textContent =
    "realized · " +
    data.strategy_id +
    " · " +
    arr.length +
    " points · " +
    fmtUSD(data.start_value) +
    " → " +
    fmtUSD(data.end_value);
  if (arr.length > 0) state.chart.timeScale().fitContent();
}

async function renderOpenPositions() {
  // Reuse the existing /api/strategies + /api/strategies/<id>/status surface.
  const list = $("#positions-list");
  list.innerHTML = '<p class="muted">Loading…</p>';
  try {
    const { strategies } = await getJSON(API.strategies);
    const all = [];
    for (const s of strategies) {
      const status = await getJSON(`/api/strategies/${encodeURIComponent(s.id)}/status`);
      const positions = status.positions || {};
      for (const sym of Object.keys(positions)) {
        const p = positions[sym];
        if (!p) continue;
        all.push({ strategy: s.id, sym, p });
      }
      const optPositions = status.option_positions || {};
      for (const sym of Object.keys(optPositions)) {
        const p = optPositions[sym];
        if (!p) continue;
        all.push({ strategy: s.id, sym, p, isOption: true });
      }
    }
    if (all.length === 0) {
      list.innerHTML = '<p class="empty muted">No open positions.</p>';
      return;
    }
    list.innerHTML = "";
    for (const { strategy, sym, p, isOption } of all) {
      const row = document.createElement("div");
      row.className = "position-row";
      const sideRaw = (p.side || "long").toLowerCase();
      const sideLabel = isOption ? (p.action || "?").toUpperCase() : sideRaw;
      const qty = p.quantity || 0;
      const avg = p.avg_cost ?? p.entry_premium ?? p.strike ?? 0;
      row.innerHTML = `
        <div>
          <div class="position-symbol">${sym}</div>
          <div class="position-strategy">${strategy}</div>
        </div>
        <span class="position-side ${sideRaw}">${sideLabel}</span>
        <span class="position-qty">${fmtQty(qty)}</span>
        <span class="position-px">$${avg.toLocaleString("en-US", { maximumFractionDigits: 2 })}</span>
      `;
      list.appendChild(row);
    }
  } catch (err) {
    list.innerHTML = `<p class="empty">Position fetch error: ${err.message}</p>`;
  }
}

// ============================================================
// Trades view
// ============================================================

async function renderTrades() {
  const tbody = $("#trades-table tbody");
  tbody.innerHTML = '<tr><td colspan="8" class="muted">Loading…</td></tr>';
  try {
    const params = {
      filter: state.filterResult,
      limit: state.filterLimit,
    };
    if (state.filterStrategy) params.strategy = state.filterStrategy;
    const data = await getJSON(API.trades(params));
    if (!data.trades || data.trades.length === 0) {
      tbody.innerHTML = "";
      $("#trades-empty").hidden = false;
      return;
    }
    $("#trades-empty").hidden = true;
    tbody.innerHTML = "";
    for (const t of data.trades) {
      const tr = document.createElement("tr");
      const pnl = t.realized_pnl || 0;
      const pnlClass = pnl > 0 ? "positive" : pnl < 0 ? "negative" : "muted";
      const sideClass = (t.side || "").toLowerCase() === "buy" ? "side-buy" : "side-sell";
      tr.innerHTML = `
        <td>${fmtTimestamp(t.timestamp)}</td>
        <td>${t.strategy_id}</td>
        <td>${t.symbol}</td>
        <td class="${sideClass}">${(t.side || "").toUpperCase()}${t.is_close ? " (close)" : ""}</td>
        <td class="num">${fmtQty(t.quantity)}</td>
        <td class="num">$${(t.price || 0).toLocaleString("en-US", { maximumFractionDigits: 2 })}</td>
        <td class="num ${pnlClass}">${t.is_close ? fmtUSD(pnl, { signed: true }) : "—"}</td>
        <td class="num muted">${fmtUSD(t.exchange_fee || 0)}</td>
      `;
      tbody.appendChild(tr);
    }
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="8" class="empty">Trade fetch error: ${err.message}</td></tr>`;
  }
}

async function populateStrategyFilter() {
  try {
    const { strategies } = await getJSON(API.strategies);
    const sel = $("#filter-strategy");
    while (sel.options.length > 1) sel.remove(1);
    for (const s of strategies) {
      const opt = document.createElement("option");
      opt.value = s.id;
      opt.textContent = s.id;
      sel.appendChild(opt);
    }
  } catch (err) {
    // Non-fatal — filter just stays empty.
  }
}

// ============================================================
// Lifecycle
// ============================================================

async function reloadAll() {
  $("#status-text").textContent = "Loading…";
  try {
    const [portfolio, pnl, equity] = await Promise.all([
      getJSON(API.portfolio),
      getJSON(API.pnl),
      getJSON(API.equity(state.window)),
    ]);
    renderPortfolio(portfolio);
    renderPnL(pnl, state.window);
    renderEquity(equity);
    await renderOpenPositions();
    $("#status-text").textContent = "Updated " + new Date().toLocaleTimeString();
  } catch (err) {
    $("#status-text").textContent = "Error: " + err.message;
  }
}

function switchView(view) {
  state.view = view;
  $$(".tab").forEach((t) => {
    const sel = t.dataset.view === view;
    t.setAttribute("aria-selected", sel ? "true" : "false");
  });
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === "view-" + view));
  if (view === "trades") {
    renderTrades();
  }
}

function bind() {
  $("#refresh").addEventListener("click", reloadAll);
  $("#window-select").addEventListener("change", (e) => {
    state.window = e.target.value;
    reloadAll();
  });
  $$(".tab").forEach((t) => t.addEventListener("click", () => switchView(t.dataset.view)));
  $("#filter-strategy").addEventListener("change", (e) => {
    state.filterStrategy = e.target.value;
    renderTrades();
  });
  $("#filter-result").addEventListener("change", (e) => {
    state.filterResult = e.target.value;
    renderTrades();
  });
  $("#filter-limit").addEventListener("change", (e) => {
    state.filterLimit = parseInt(e.target.value, 10) || 50;
    renderTrades();
  });
}

function startAutoRefresh() {
  if (state.refreshTimer) clearInterval(state.refreshTimer);
  state.refreshTimer = setInterval(() => {
    if (state.view === "overview") reloadAll();
    else if (state.view === "trades") renderTrades();
  }, 30000);
}

(async function init() {
  bind();
  await populateStrategyFilter();
  await reloadAll();
  startAutoRefresh();
})();
