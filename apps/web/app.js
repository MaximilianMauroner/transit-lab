const viewRoot = document.querySelector("#app-view");
const notice = document.querySelector("#notice");
const systemStatus = document.querySelector("#system-status");
const networkSelector = document.querySelector("#network-selector");
const snapshotSelector = document.querySelector("#snapshot-selector");
const modelSelector = document.querySelector("#model-selector");
const dateSelector = document.querySelector("#date-selector");
const provenanceText = document.querySelector("#provenance-text");
const facetText = document.querySelector("#facet-text");
const runBadge = document.querySelector("#run-badge");

const state = {
  catalog: null,
  overview: null,
  networkId: "",
  snapshotId: "",
  modelId: "",
  date: "",
  facet: "general",
  view: currentView(),
  network: null,
  lines: [],
  selectedLineId: "",
  similarity: null,
  selectedMatch: null,
  weights: { role: 0.4, service: 0.2, geometry: 0.15, resilience: 0.25 },
  runId: "",
  runEvents: null,
  eventSource: null
};

function currentView() {
  const value = window.location.hash.slice(1).split("?")[0];
  return value || "overview";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function number(value, digits = 0) {
  const result = Number(value);
  if (!Number.isFinite(result)) return "—";
  return new Intl.NumberFormat("en", { maximumFractionDigits: digits, minimumFractionDigits: digits }).format(result);
}

function percent(value, digits = 0) {
  const result = Number(value);
  return Number.isFinite(result) ? `${number(result * 100, digits)}%` : "—";
}

function date(value) {
  if (!value) return "—";
  return new Intl.DateTimeFormat("en", { dateStyle: "medium", timeStyle: "short" }).format(new Date(value));
}

function shortId(value) {
  const text = String(value || "");
  return text.length > 20 ? `${text.slice(0, 8)}…${text.slice(-7)}` : text || "—";
}

function showNotice(message, isError = false) {
  notice.textContent = message || "";
  notice.className = `notice${isError ? " error" : ""}`;
  notice.hidden = !message;
}

async function fetchJson(path, options = {}) {
  const response = await fetch(path, { headers: { Accept: "application/json", ...(options.headers || {}) }, ...options });
  let body;
  try { body = await response.json(); } catch { body = {}; }
  if (!response.ok) throw new Error(body.error || `Request failed with HTTP ${response.status}`);
  return body;
}

function setStatus(label, className = "") {
  systemStatus.textContent = label;
  systemStatus.className = `status-pill ${className}`.trim();
}

function selectedSnapshot() {
  return state.catalog?.snapshots?.find((snapshot) => snapshot.id === state.snapshotId) || null;
}

function selectedNetwork() {
  return state.catalog?.networks?.find((network) => network.id === state.networkId) || null;
}

function selectedModel() {
  return state.catalog?.models?.find((model) => model.id === state.modelId) || null;
}

function updateProvenance() {
  const snapshot = selectedSnapshot();
  const network = selectedNetwork();
  const model = selectedModel();
  provenanceText.textContent = [
    network?.displayName || "No network",
    snapshot ? `snapshot ${shortId(snapshot.id)}` : "no snapshot",
    snapshot?.serviceDate || "no service date",
    model ? `model ${model.version}` : "no model"
  ].join(" · ");
  facetText.textContent = `Facet: ${state.facet}`;
}

function populateSelect(select, items, selected, placeholder, getValue = (item) => item.id, getLabel = (item) => item.displayName || item.id) {
  select.replaceChildren();
  const empty = new Option(placeholder, "");
  select.add(empty);
  for (const item of items || []) select.add(new Option(getLabel(item), getValue(item)));
  select.value = selected || "";
}

function populateSelectors() {
  if (!state.catalog) return;
  populateSelect(networkSelector, state.catalog.networks, state.networkId, "Select network", (item) => item.id, (item) => item.displayName);
  const snapshots = state.networkId
    ? state.catalog.snapshots.filter((snapshot) => snapshot.networkId === state.networkId)
    : state.catalog.snapshots;
  populateSelect(snapshotSelector, snapshots, state.snapshotId, "Select snapshot", (item) => item.id, (item) => `${item.serviceDate} · ${item.sourceName || item.id}`);
  populateSelect(modelSelector, state.catalog.models, state.modelId, "No model selected", (item) => item.id, (item) => `${item.version} · ${item.status}`);
  populateSelect(dateSelector, state.catalog.dates.map((value) => ({ id: value, label: value })), state.date, "No date filter", (item) => item.id, (item) => item.label);
  updateProvenance();
}

function chooseInitialContext() {
  const query = new URLSearchParams(window.location.search);
  state.networkId = query.get("network") || state.catalog.networks?.[0]?.id || "";
  const networkSnapshots = state.catalog.snapshots.filter((snapshot) => snapshot.networkId === state.networkId);
  state.snapshotId = query.get("snapshot") || networkSnapshots[0]?.id || state.catalog.snapshots?.[0]?.id || "";
  const snapshot = selectedSnapshot();
  if (snapshot) state.networkId = snapshot.networkId;
  state.modelId = query.get("model") || state.catalog.models?.[0]?.id || "";
  state.date = query.get("date") || snapshot?.serviceDate || "";
  state.facet = query.get("facet") || "general";
  for (const name of ["role", "service", "geometry", "resilience"]) {
    const raw = query.get(`${name}Weight`);
    const value = raw === null ? NaN : Number(raw);
    if (Number.isFinite(value) && value >= 0) state.weights[name] = value;
  }
}

function syncUrl() {
  const params = new URLSearchParams();
  if (state.networkId) params.set("network", state.networkId);
  if (state.snapshotId) params.set("snapshot", state.snapshotId);
  if (state.modelId) params.set("model", state.modelId);
  if (state.date) params.set("date", state.date);
  if (state.facet && state.facet !== "general") params.set("facet", state.facet);
  for (const name of ["role", "service", "geometry", "resilience"]) {
    params.set(`${name}Weight`, String(state.weights[name]));
  }
  const next = `${window.location.pathname}?${params}${window.location.hash || "#overview"}`;
  window.history.replaceState(null, "", next);
}

async function refreshCatalog() {
  setStatus("Syncing");
  const [catalog, overview] = await Promise.all([fetchJson("/api/catalog"), fetchJson("/api/overview")]);
  const hadContext = Boolean(state.catalog);
  state.catalog = catalog;
  state.overview = overview;
  if (!hadContext) chooseInitialContext();
  else {
    if (!catalog.snapshots.some((snapshot) => snapshot.id === state.snapshotId)) state.snapshotId = catalog.snapshots[0]?.id || "";
    if (!catalog.networks.some((network) => network.id === state.networkId)) state.networkId = selectedSnapshot()?.networkId || catalog.networks[0]?.id || "";
  }
  populateSelectors();
  syncUrl();
  const active = overview.activeRun;
  const runCount = active ? 1 : 0;
  runBadge.textContent = String(runCount);
  runBadge.hidden = runCount === 0;
  setStatus("Local", "ready");
}

function heading(title, description, action = "") {
  return `<div class="view-heading"><div><h1>${escapeHtml(title)}</h1><p>${escapeHtml(description)}</p></div><div class="heading-actions">${action}</div></div>`;
}

function card(title, body, className = "") {
  return `<section class="card ${className}"><div class="card-header"><h2>${escapeHtml(title)}</h2></div>${body}</section>`;
}

function statusPill(status) {
  return `<span class="status-pill ${escapeHtml(String(status || ""))}">${escapeHtml(status || "unknown")}</span>`;
}

function empty(title, message) {
  return `<div class="empty"><strong>${escapeHtml(title)}</strong>${escapeHtml(message)}</div>`;
}

function renderOverview() {
  const overview = state.overview || { readiness: { passed: 0, total: 0, gates: [] }, corpus: {}, cityReadiness: [] };
  const readiness = overview.readiness;
  const ratio = readiness.total ? readiness.passed / readiness.total * 100 : 0;
  const blockers = readiness.gates.filter((gate) => !gate.passed).slice(0, 4);
  const active = overview.activeRun;
  const action = state.snapshotId ? `<button class="button primary" data-action="queue-simulation">Queue criticality run</button>` : "";
  viewRoot.innerHTML = heading("Research overview", "A record of what exists, what produced it, and what blocks the next meaningful run.", action) + `
    <div class="readiness-grid">
      ${card("Training readiness", `<div class="card-body"><div class="readiness-score"><strong class="score-number">${number(readiness.passed)}<small> / ${number(readiness.total)}</small></strong><div class="progress-track"><i style="width:${ratio}%"></i></div></div><p class="readiness-caption">Readiness is derived from artifacts and quality checks, not a manually entered progress value.</p></div>`)}
      ${card(active ? "Active run" : "Active run", active ? `<div class="card-body run-card"><div class="run-title"><strong>${escapeHtml(active.kind)}</strong>${statusPill(active.status)}</div><p class="run-meta">${escapeHtml(active.snapshotId || active.datasetId || active.modelId || "Queued research operation")} · ${escapeHtml(active.currentStep || "waiting for worker")}</p><div class="run-progress"><div class="progress-track"><i style="width:${active.progress.total ? Math.min(100, active.progress.completed / active.progress.total * 100) : 0}%"></i></div><span>${number(active.progress.completed)} / ${number(active.progress.total) || "—"}</span></div><div class="heading-actions" style="margin-top:17px"><button class="button" data-action="open-active-run">Open live run</button>${active.status === "running" || active.status === "queued" ? `<button class="button danger" data-action="cancel-active-run">Cancel</button>` : ""}</div></div>` : `<div class="card-body run-card"><p class="run-empty">No worker-owned run is active. Queue a compile, simulation, dataset, training, evaluation, or inference run and watch its persisted events here.</p><button class="button" data-view="runs">Open runs</button></div>`)}
    </div>
    <div class="stat-strip">
      <div class="stat-card"><span>Networks</span><strong>${number(overview.corpus.cities)}</strong><small>registered research systems</small></div>
      <div class="stat-card"><span>Snapshots</span><strong>${number(overview.corpus.snapshots)}</strong><small>${number(overview.corpus.snapshotPairs)} possible pairings</small></div>
      <div class="stat-card"><span>Line instances</span><strong>${number(overview.corpus.lineInstances)}</strong><small>canonical snapshot records</small></div>
      <div class="stat-card"><span>Labels</span><strong>${number(overview.corpus.labels)}</strong><small>simulation rows indexed</small></div>
    </div>
    ${card("Current blockers", `<div class="card-body"><ul class="blocker-list">${(blockers.length ? blockers : [{ label: "All readiness gates pass", detail: "The next run can be started." }]).map((gate) => `<li class="${gate.passed ? "pass" : ""}"><i>${gate.passed ? "✓" : "!"}</i><span><strong>${escapeHtml(gate.label)}</strong><br><span class="small-muted">${escapeHtml(gate.detail)}</span></span></li>`).join("")}</ul></div>`, "section-card")}
    ${card("City readiness matrix", `<div class="table-wrap"><table class="matrix"><thead><tr><th>Network</th><th>Feed</th><th>Compile</th><th>Snapshot pairs</th><th>Labels</th><th>Infer</th><th>Valid</th></tr></thead><tbody>${(overview.cityReadiness || []).map((row) => `<tr><td>${escapeHtml(row.displayName)}</td><td>${check(row.feed)}</td><td>${check(row.compile)}</td><td>${row.snapshotPairs ? `<span class="check yes">${number(row.snapshotPairs)}</span>` : `<span class="check warn">blocked</span>`}</td><td>${row.labels ? percent(row.labels) : `<span class="check">—</span>`}</td><td>${check(row.infer)}</td><td>${check(row.valid)}</td></tr>`).join("") || `<tr><td colspan="7">${empty("No networks indexed", "Register a feed or place a compiled snapshot under data/.")}</td></tr>`}</tbody></table></div>`, "section-card")}
    ${card("Gate detail", `<div class="table-wrap"><table class="data-table"><thead><tr><th>Group</th><th>Gate</th><th>Status</th><th>Evidence</th></tr></thead><tbody>${readiness.gates.map((gate) => `<tr><td>${escapeHtml(gate.group)}</td><td>${escapeHtml(gate.label)}</td><td>${check(gate.passed)}</td><td>${escapeHtml(gate.detail)}</td></tr>`).join("")}</tbody></table></div>`, "section-card")}
  `;
  bindActions();
}

function check(value) {
  return `<span class="check ${value ? "yes" : "warn"}">${value ? "pass" : "—"}</span>`;
}

function renderData() {
  const networks = state.catalog?.networks || [];
  const feeds = networks.flatMap((network) => network.feeds.map((feed) => ({ ...feed, networkName: network.displayName })));
  const snapshots = state.catalog?.snapshots || [];
  viewRoot.innerHTML = heading("Data registry", "Raw feed revisions and compiled snapshots are immutable records. A route_id is never treated as a continuity guarantee.") + `
    <div class="data-grid">
      ${card("Feed revisions", feeds.length ? `<div class="table-wrap"><table class="data-table"><thead><tr><th>Network</th><th>Source</th><th>Downloaded</th><th>Raw SHA-256</th><th>Size</th><th>Status</th></tr></thead><tbody>${feeds.map((feed) => `<tr><td>${escapeHtml(feed.networkName)}</td><td><strong>${escapeHtml(feed.id)}</strong><br><span class="small-muted">${escapeHtml(feed.geographicalScope || feed.sourceUrl || "Local feed")}</span></td><td>${escapeHtml(date(feed.downloadedAt))}</td><td class="mono">${escapeHtml(shortId(feed.sha256))}</td><td>${number(feed.byteCount / 1024 / 1024, 1)} MB</td><td>${statusPill(feed.validationStatus)}</td></tr>`).join("")}</tbody></table></div>` : empty("No feed revisions", "Use the CLI fetch workflow or register a local source.json."))}
      ${card("Compiled snapshots", snapshots.length ? `<div class="table-wrap"><table class="data-table"><thead><tr><th>Network</th><th>Service date</th><th>Counts</th><th>Compiler</th><th>Graph</th><th>Fingerprint</th></tr></thead><tbody>${snapshots.map((snapshot) => `<tr><td><button class="line-link" data-snapshot="${escapeHtml(snapshot.id)}">${escapeHtml(snapshot.sourceName || snapshot.networkId)}</button><br><span class="small-muted">${escapeHtml(snapshot.geographicalScope)}</span></td><td>${escapeHtml(snapshot.serviceDate)}</td><td>${number(snapshot.counts.stations)} stations · ${number(snapshot.counts.lines)} lines<br><span class="small-muted">${number(snapshot.counts.patterns)} patterns · ${number(snapshot.counts.transitEdges)} edges</span></td><td>${escapeHtml(snapshot.compilerVersion || "—")}<br><span class="mono">${escapeHtml(snapshot.compilerCommit || "commit not recorded")}</span></td><td>${snapshot.graphPath ? check(true) : check(false)}</td><td class="mono">${escapeHtml(shortId(snapshot.fingerprint))}</td></tr>`).join("")}</tbody></table></div>` : empty("No compiled snapshots", "Compile a valid feed for an explicit service date."))}
    </div>`;
  viewRoot.querySelectorAll("[data-snapshot]").forEach((button) => button.addEventListener("click", () => { state.snapshotId = button.dataset.snapshot; state.networkId = selectedSnapshot()?.networkId || state.networkId; populateSelectors(); go("network"); }));
}

async function renderDataset() {
  try {
    const datasets = await fetchJson("/api/datasets");
    const graphSnapshots = (state.catalog?.snapshots || []).filter((snapshot) => snapshot.graphPath);
    const latest = datasets[0];
    const counts = latest?.objectiveCounts || {};
    const objectiveCount = (name) => counts[name] === undefined ? "—" : number(counts[name]);
    viewRoot.innerHTML = heading("Dataset builder", "Freeze the exact snapshots, schema, split, and objective counts before a model run enters the queue.", `<button class="button primary" data-action="queue-dataset">Build from compiled snapshots</button>`) + `
      ${card("Available inputs", `<div class="card-body"><div class="key-value"><dt>Compiled snapshots</dt><dd>${number(state.catalog?.snapshots?.length || 0)}</dd><dt>Graph-ready snapshots</dt><dd>${number(graphSnapshots.length)}</dd><dt>Raw feeds</dt><dd>${number(state.overview?.corpus?.feeds)}</dd><dt>Latest dataset version</dt><dd>${latest ? `<span class="mono">${escapeHtml(shortId(latest.id))}</span>` : "No dataset manifest registered"}</dd></div></div>`)}
      ${card("Registered dataset versions", datasets.length ? `<div class="table-wrap"><table class="data-table"><thead><tr><th>Dataset</th><th>Snapshots</th><th>Schema</th><th>Split</th><th>Criticality rows</th><th>Status</th></tr></thead><tbody>${datasets.map((dataset) => `<tr><td class="mono">${escapeHtml(shortId(dataset.id))}<br><span class="small-muted">${escapeHtml(shortId(dataset.fingerprint))}</span></td><td>${number(dataset.snapshotIds?.length)}</td><td>${escapeHtml(dataset.featureSchema || "—")}</td><td>${escapeHtml(dataset.split?.strategy || dataset.split?.name || "frozen")}</td><td>${objectiveCount("criticalityLines")}</td><td>${statusPill(dataset.status)}</td></tr>`).join("")}</tbody></table></div>` : empty("No dataset manifest registered", "Queue a build after compiled graph snapshots are available."), "section-card")}
      ${card("Objective inventory", `<div class="table-wrap"><table class="data-table"><thead><tr><th>Objective</th><th>Examples</th><th>Source</th><th>Readiness</th></tr></thead><tbody><tr><td>Masked reconstruction</td><td>${objectiveCount("maskedReconstruction")}</td><td>Graph tensors</td><td>${latest ? check(true) : check(false)}</td></tr><tr><td>Cross-snapshot identity</td><td>${objectiveCount("crossSnapshotPairs")}</td><td>Matched canonical lines</td><td>${latest?.snapshotIds?.length > 1 ? check(true) : check(false)}</td></tr><tr><td>Facet metric learning</td><td>${objectiveCount("roleTriplets")} / ${objectiveCount("serviceTriplets")} / ${objectiveCount("geometryTriplets")} / ${objectiveCount("resilienceTriplets")}</td><td>Role · service · geometry · resilience</td><td>${latest ? check(true) : check(false)}</td></tr><tr><td>Criticality</td><td>${objectiveCount("criticalityLines")}</td><td>Line-removal labels</td><td>${state.overview?.corpus?.labels ? check(true) : check(false)}</td></tr></tbody></table></div>`, "section-card")}
      ${card("Leakage checks", `<div class="card-body"><ul class="blocker-list"><li><i>!</i><span>Same canonical line across train and validation: ${latest?.quality?.leakageChecks || "not checked until a dataset quality report exists"}.</span></li><li class="pass"><i>✓</i><span>Public line names and raw GTFS IDs are lookup metadata, excluded from model feature arrays by the Rust graph contract.</span></li><li><i>!</i><span>City-level holdout assignment: ${latest?.split?.holdoutNetworks?.length ? `${number(latest.split.holdoutNetworks.length)} holdout network(s) frozen` : "not assigned in the current manifest"}.</span></li></ul></div>`, "section-card")}`;
    bindActions();
  } catch (error) {
    showNotice(error.message, true);
    viewRoot.innerHTML = heading("Dataset builder", "The dataset registry could not be loaded.") + empty("Dataset registry unavailable", error.message);
  }
}

async function loadNetwork() {
  if (!state.snapshotId) return null;
  if (state.network?.snapshot?.id === state.snapshotId) return state.network;
  state.network = await fetchJson(`/api/snapshots/${encodeURIComponent(state.snapshotId)}/network`);
  return state.network;
}

async function loadLines() {
  if (!state.snapshotId) return [];
  if (state.lines.length && state.lines[0].snapshotId === state.snapshotId) return state.lines;
  state.lines = await fetchJson(`/api/snapshots/${encodeURIComponent(state.snapshotId)}/lines`);
  state.selectedLineId = state.lines[0]?.id || "";
  return state.lines;
}

function routeMap(network, selectedLineId = "") {
  const stations = network?.stations || [];
  const coordinates = stations.filter((station) => Number.isFinite(station.longitude) && Number.isFinite(station.latitude));
  if (!coordinates.length) return empty("No map geometry", "This snapshot does not contain station coordinates.");
  const minX = Math.min(...coordinates.map((station) => station.longitude));
  const maxX = Math.max(...coordinates.map((station) => station.longitude));
  const minY = Math.min(...coordinates.map((station) => station.latitude));
  const maxY = Math.max(...coordinates.map((station) => station.latitude));
  const width = 760; const height = 470; const pad = 28;
  const x = (value) => pad + ((value - minX) / Math.max(1e-9, maxX - minX)) * (width - pad * 2);
  const y = (value) => height - pad - ((value - minY) / Math.max(1e-9, maxY - minY)) * (height - pad * 2);
  const palette = ["#2868dc", "#d65a61", "#158a82", "#b47b1a", "#875cc2", "#3a7a9a", "#d27942", "#4c8e57"];
  const routes = (network.routes || []).map((route) => {
    const points = route.coordinates.map(([lon, lat]) => `${x(lon).toFixed(1)},${y(lat).toFixed(1)}`).join(" ");
    const selected = network.lines.find((line) => line.lineIndex === route.lineIndex)?.id === selectedLineId;
    return `<polyline class="map-route${selected ? " selected" : ""}" points="${points}" stroke="${palette[route.lineIndex % palette.length]}" stroke-width="${selected ? 4 : 2.4}" />`;
  }).join("");
  const nodeLayer = stations.map((station) => `<circle class="map-station${station.transferDegree > 1 ? " hub" : ""}" cx="${x(station.longitude).toFixed(1)}" cy="${y(station.latitude).toFixed(1)}" r="${station.transferDegree > 1 ? 4 : 2.2}" />`).join("");
  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Transit network map"><line class="scatter-axis" x1="${pad}" x2="${width - pad}" y1="${height - pad}" y2="${height - pad}" /><line class="scatter-axis" x1="${pad}" x2="${pad}" y1="${pad}" y2="${height - pad}" />${routes}${nodeLayer}</svg><div class="map-legend"><span class="legend-item"><i class="line"></i>route geometry</span><span class="legend-item"><i></i>station</span><span class="legend-item"><i style="background:#fff0c9;border:1px solid #bd8222"></i>interchange</span></div>`;
}

function lineSummary(line) {
  const f = line.features || {};
  return `${number(f.station_count)} stops · ${number(f.route_length_metres / 1000, 1)} km · ${number(f.daily_trip_count)} trips`;
}

async function renderNetwork() {
  if (!state.snapshotId) {
    viewRoot.innerHTML = heading("Network explorer", "Select a compiled snapshot to inspect its stations, route patterns, and provenance.") + empty("No snapshot selected", "Choose a snapshot in the top bar.");
    return;
  }
  try {
    const [network, lines] = await Promise.all([loadNetwork(), loadLines()]);
    const snapshot = selectedSnapshot();
    const ranked = [...lines].sort((a, b) => (b.criticality?.primaryScore || -Infinity) - (a.criticality?.primaryScore || -Infinity));
    viewRoot.innerHTML = heading(`${snapshot?.sourceName || "Network"}`, `${snapshot?.geographicalScope || ""} · service date ${snapshot?.serviceDate || "—"}`, `<button class="button" data-view="lines">Open all lines</button>`) + `
      <div class="network-layout"><section class="card map-card"><div class="map-toolbar"><strong>Network map</strong><span>${number(network.stations.length)} stations · ${number(network.lines.length)} lines</span></div><div class="map-surface">${routeMap(network, state.selectedLineId)}</div></section><aside class="card"><div class="card-header"><h2>Network summary</h2></div><dl class="summary-list"><div><dt>Stations</dt><dd>${number(network.stations.length)}</dd></div><div><dt>Lines</dt><dd>${number(network.lines.length)}</dd></div><div><dt>Patterns</dt><dd>${number(snapshot?.counts.patterns)}</dd></div><div><dt>Interchanges</dt><dd>${number(network.interchanges.length)}</dd></div><div><dt>Service date</dt><dd>${escapeHtml(snapshot?.serviceDate || "—")}</dd></div><div><dt>Model</dt><dd>${escapeHtml(selectedModel()?.version || "none")}</dd></div></dl><div class="card-header"><h2>Ranked lines</h2><span class="subtle">available score</span></div><ol class="ranked-list">${ranked.slice(0, 8).map((line, index) => `<li><span><span class="rank-number">${String(index + 1).padStart(2, "0")}</span><button class="line-link" data-line="${escapeHtml(line.id)}">${escapeHtml(line.displayName)}</button><br><span class="small-muted">${escapeHtml(lineSummary(line))}</span></span><strong>${line.criticality ? percent(line.criticality.primaryScore) : "—"}</strong></li>`).join("")}</ol></aside></div>`;
    bindActions();
  } catch (error) { showNotice(error.message, true); viewRoot.innerHTML = heading("Network explorer", "The selected snapshot could not be loaded.") + empty("Snapshot unavailable", error.message); }
}

async function renderLines() {
  if (!state.snapshotId) { viewRoot.innerHTML = heading("Lines", "Inspect line facts, service, network role, and model outputs.") + empty("No snapshot selected", "Choose a snapshot in the top bar."); return; }
  try {
    const lines = await loadLines();
    const selected = lines.find((line) => line.id === state.selectedLineId) || lines[0];
    if (selected) state.selectedLineId = selected.id;
    const detail = selected ? await fetchJson(`/api/lines/${encodeURIComponent(selected.id)}`) : null;
    const criticality = selected?.criticality;
    const label = selected?.label || detail?.label;
    viewRoot.innerHTML = heading("Line explorer", "Facts calculated directly from the feed stay separate from predictions and simulator labels.", `<button class="button" data-view="similarity">Find similar lines</button>`) + `
      <div class="split-layout"><section class="card filter-card"><h3>Snapshot lines</h3><div class="field"><label for="line-search">Search</label><input id="line-search" placeholder="Line name or canonical ID" /></div><div id="line-choice" class="ranked-list" style="padding:0">${lines.map((line) => `<li><span><button class="line-link" data-line="${escapeHtml(line.id)}">${escapeHtml(line.displayName)}</button><br><span class="small-muted">${escapeHtml(line.modeName)} · ${escapeHtml(lineSummary(line))}</span></span><strong>${line.criticality ? percent(line.criticality.primaryScore) : "—"}</strong></li>`).join("")}</div></section><div><section class="card">${selected ? `<div class="card-header"><h2>${escapeHtml(selected.displayName)}</h2><span>${statusPill(selected.modeName)}</span></div><dl class="key-value"><dt>Snapshot</dt><dd>${escapeHtml(shortId(selected.snapshotId))} · ${escapeHtml(selectedSnapshot()?.serviceDate)}</dd><dt>Canonical identity</dt><dd class="mono">${escapeHtml(selected.canonicalId)}</dd><dt>Route facts</dt><dd>${escapeHtml(lineSummary(selected))}</dd><dt>Service span</dt><dd>${number((selected.features.service_span_seconds || 0) / 3600, 1)} h · ${number(selected.features.median_headway_seconds / 60, 1)} min median headway</dd><dt>Network role</dt><dd>${number(selected.features.transfer_station_count)} transfer stations · ${percent(selected.features.unique_station_fraction)}</dd><dt>Representation source</dt><dd>Reference pipeline / feature-separated preview</dd></dl>` : empty("No line selected", "Choose a line from the list.")}</section>${selected ? card("Criticality", `<div class="metric-cards">${metricCard("Accessibility loss", criticality?.accessibility_auc_loss, true, "", label?.accessibility_auc_loss)}${metricCard("Unreachable share", criticality?.unreachable_share, true, "", label?.unreachable_share)}${metricCard("Mean delay", criticality?.mean_delay_reachable_seconds, false, "s", label?.mean_delay_reachable_seconds)}${metricCard("P95 delay", criticality?.p95_delay_reachable_seconds, false, "s", label?.p95_delay_reachable_seconds)}${metricCard("Extra transfers", criticality?.mean_extra_transfers, false, "", label?.mean_extra_transfers)}${metricCard("Stations isolated", criticality?.stations_losing_all_service_share, true, "", label?.stations_losing_all_service_share)}</div><div class="card-body"><p class="chart-note">Predicted and simulated values are paired when both artifacts are present. A blank uncertainty field is not a zero-confidence interval.</p></div>`, "section-card") : ""}${selected && detail ? card("Provenance", `<dl class="key-value"><dt>Model</dt><dd>${escapeHtml(detail.provenance.modelId || "No inference model")}</dd><dt>Snapshot fingerprint</dt><dd class="mono">${escapeHtml(detail.snapshot.fingerprint)}</dd><dt>Label artifact</dt><dd>${escapeHtml(detail.label?.sourceArtifactId || "No simulator label")}</dd><dt>Label policy</dt><dd class="mono">${escapeHtml(label?.policy_fingerprint || "not recorded")}</dd><dt>Compiler</dt><dd>${escapeHtml(detail.snapshot.compilerVersion || "—")}</dd></dl>`, "section-card") : ""}</div></div>`;
    bindActions();
    const search = viewRoot.querySelector("#line-search");
    search?.addEventListener("input", () => { const term = search.value.toLowerCase(); viewRoot.querySelectorAll("#line-choice li").forEach((li) => { li.hidden = !li.textContent.toLowerCase().includes(term); }); });
  } catch (error) { showNotice(error.message, true); viewRoot.innerHTML = heading("Lines", "Unable to load line instances.") + empty("Line data unavailable", error.message); }
}

function metricCard(label, value, asPercent = false, suffix = "", observed = undefined) {
  const shown = value === undefined || value === null ? "—" : asPercent ? percent(value, 1) : `${number(value, 1)}${suffix}`;
  const simulated = observed === undefined || observed === null ? "" : ` · simulated ${asPercent ? percent(observed, 1) : `${number(observed, 1)}${suffix}`}`;
  return `<div class="metric-card"><span>${escapeHtml(label)}</span><strong>${escapeHtml(shown)}</strong><small>${value === undefined || value === null ? "prediction not registered" : "predicted"}${simulated}</small></div>`;
}

function facetTabs(selected) {
  return `<div class="facet-tabs">${["general", "role", "service", "geometry", "resilience"].map((facet) => `<button class="${facet === selected ? "active" : ""}" data-facet="${facet}">${facet}</button>`).join("")}</div>`;
}

function lineOptions(lines, selected) {
  return lines.map((line) => `<option value="${escapeHtml(line.id)}" ${line.id === selected ? "selected" : ""}>${escapeHtml(line.displayName)} · ${escapeHtml(line.modeName)}</option>`).join("");
}

function weightPercent(name) {
  return Math.max(0, Math.min(100, Math.round(Number(state.weights[name] || 0) * 100)));
}

async function renderSimilarity() {
  const lines = await loadLines().catch(() => []);
  const snapshots = state.catalog?.snapshots || [];
  const candidateSnapshot = snapshots.find((snapshot) => snapshot.id !== state.snapshotId) || selectedSnapshot();
  viewRoot.innerHTML = heading("Similarity explorer", "The profile states what ‘similar’ means. Results use the original high-dimensional score; explanations use measured GTFS differences.") + `
    <div class="split-layout"><form id="similarity-form" class="card filter-card"><h3>Query definition</h3><div class="field"><label for="similarity-query-snapshot">Query snapshot</label><select id="similarity-query-snapshot">${snapshots.map((snapshot) => `<option value="${escapeHtml(snapshot.id)}" ${snapshot.id === state.snapshotId ? "selected" : ""}>${escapeHtml(snapshot.serviceDate)} · ${escapeHtml(snapshot.sourceName || snapshot.networkId)}</option>`).join("")}</select></div><div class="field"><label for="similarity-query-line">Query line</label><select id="similarity-query-line">${lineOptions(lines, state.selectedLineId)}</select></div><div class="field"><label for="similarity-candidate">Candidate snapshot</label><select id="similarity-candidate">${snapshots.map((snapshot) => `<option value="${escapeHtml(snapshot.id)}" ${snapshot.id === candidateSnapshot?.id ? "selected" : ""}>${escapeHtml(snapshot.serviceDate)} · ${escapeHtml(snapshot.sourceName || snapshot.networkId)}</option>`).join("")}</select></div><div class="field"><label>Similarity facet</label>${facetTabs(state.facet)}</div><div class="field"><label for="role-weight">Role weight <output id="role-weight-value">${weightPercent("role")}%</output></label><div class="range-row"><input id="role-weight" type="range" min="0" max="100" value="${weightPercent("role")}" /><output id="role-weight-output">${weightPercent("role")}%</output></div></div><div class="field"><label for="service-weight">Service weight <output id="service-weight-value">${weightPercent("service")}%</output></label><div class="range-row"><input id="service-weight" type="range" min="0" max="100" value="${weightPercent("service")}" /><output id="service-weight-output">${weightPercent("service")}%</output></div></div><div class="field"><label for="geometry-weight">Geometry weight <output id="geometry-weight-value">${weightPercent("geometry")}%</output></label><div class="range-row"><input id="geometry-weight" type="range" min="0" max="100" value="${weightPercent("geometry")}" /><output id="geometry-weight-output">${weightPercent("geometry")}%</output></div></div><div class="field"><label for="resilience-weight">Resilience weight <output id="resilience-weight-value">${weightPercent("resilience")}%</output></label><div class="range-row"><input id="resilience-weight" type="range" min="0" max="100" value="${weightPercent("resilience")}" /><output id="resilience-weight-output">${weightPercent("resilience")}%</output></div></div><button class="button primary" type="submit">Search candidates</button></form><section class="card results-card"><div class="card-header"><h2>Candidate lines</h2><span class="subtle">${state.similarity ? `${number(state.similarity.matches.length)} results · ${escapeHtml(state.similarity.embeddingSource)}` : "No query yet"}</span></div>${state.similarity ? similarityResults(state.similarity) : empty("Choose a profile and search", "Role, service, geometry, and resilience are separate spaces.")}</section></div>`;
  const form = viewRoot.querySelector("#similarity-form");
  viewRoot.querySelectorAll("[data-facet]").forEach((button) => button.addEventListener("click", () => { state.facet = button.dataset.facet; updateProvenance(); syncUrl(); renderSimilarity(); }));
  ["role", "service", "geometry", "resilience"].forEach((name) => { const input = viewRoot.querySelector(`#${name}-weight`); const output = viewRoot.querySelector(`#${name}-weight-output`); input?.addEventListener("input", () => { state.weights[name] = Number(input.value) / 100; output.textContent = `${input.value}%`; syncUrl(); }); });
  viewRoot.querySelector("#similarity-query-snapshot")?.addEventListener("change", async (event) => { state.snapshotId = event.target.value; state.networkId = selectedSnapshot()?.networkId || state.networkId; state.lines = []; syncUrl(); await renderSimilarity(); });
  viewRoot.querySelector("#similarity-query-line")?.addEventListener("change", (event) => { state.selectedLineId = event.target.value; });
  form?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const weights = Object.fromEntries(["role", "service", "geometry", "resilience"].map((name) => [name, Number(viewRoot.querySelector(`#${name}-weight`).value) / 100]));
    state.weights = weights;
    syncUrl();
    const candidate = viewRoot.querySelector("#similarity-candidate").value;
    try {
      setStatus("Searching");
      state.similarity = await fetchJson("/api/similarity/search", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ querySnapshotId: state.snapshotId, queryLineId: state.selectedLineId, candidateSnapshotId: candidate, profile: state.facet, weights, topK: 10 }) });
      state.selectedMatch = null;
      setStatus("Local", "ready");
      renderSimilarity();
    } catch (error) { showNotice(error.message, true); setStatus("Needs attention", "failed"); }
  });
  viewRoot.querySelectorAll("[data-match]").forEach((button) => button.addEventListener("click", () => { state.selectedMatch = state.similarity.matches.find((match) => match.lineInstanceId === button.dataset.match); renderSimilarity(); }));
}

function similarityResults(result) {
  const selected = state.selectedMatch || result.matches[0];
  return `<div class="result-row result-header"><span>Candidate</span><span>Role</span><span>Service</span><span>Geometry</span><span>Resilience</span><span>Profile</span></div>${result.matches.map((match) => `<button class="result-row" data-match="${escapeHtml(match.lineInstanceId)}" style="width:100%;border:0;background:${selected?.lineInstanceId === match.lineInstanceId ? "#f6f9ff" : "#fff"};text-align:left"><span class="result-name"><strong>${escapeHtml(match.displayName)}</strong><small>${escapeHtml(match.mode)} · ${escapeHtml(shortId(match.lineInstanceId))}</small></span><span class="score">${number(match.facetScores.role, 2)}</span><span class="score">${number(match.facetScores.service, 2)}</span><span class="score">${number(match.facetScores.geometry, 2)}</span><span class="score">${number(match.facetScores.resilience, 2)}</span><span class="score primary">${number(match.similarity, 2)}<i class="score-bar"><i style="width:${Math.max(0, Math.min(100, match.similarity * 100))}%"></i></i></span></button>`).join("")}${selected ? `<div class="comparison-grid"><div class="comparison-item"><span>Same mode</span><strong>${selected.comparison.sameMode ? "Yes" : "No"}</strong></div><div class="comparison-item"><span>Transfer percentile Δ</span><strong>${number(selected.comparison.transferStationPercentileDifference, 2)}</strong></div><div class="comparison-item"><span>Frequency distance</span><strong>${number(selected.comparison.frequencyProfileDistance, 2)}</strong></div><div class="comparison-item"><span>Route length ratio</span><strong>${selected.comparison.routeLengthRatio === null ? "—" : `${number(selected.comparison.routeLengthRatio, 2)}×`}</strong></div><div class="comparison-item"><span>Stops difference</span><strong>${number(selected.comparison.stationCountDifference)}</strong></div><div class="comparison-item"><span>Criticality percentile Δ</span><strong>${selected.comparison.criticalityPercentileDifference === null ? "—" : number(selected.comparison.criticalityPercentileDifference, 2)}</strong></div></div><div class="explanation"><strong>Measured comparison</strong>${selected.comparison.sameMode ? "Same transport mode." : "Different transport modes."} Transfer-degree percentiles differ by ${number(selected.comparison.transferStationPercentileDifference, 2)}; frequency profile distance is ${number(selected.comparison.frequencyProfileDistance, 2)}; route length ratio is ${selected.comparison.routeLengthRatio === null ? "not available" : `${number(selected.comparison.routeLengthRatio, 2)}×`}. The score is ${escapeHtml(result.embeddingSource)}, not a hidden universal meaning.</div>` : ""}`;
}

async function renderCriticality() {
  const lines = await loadLines().catch(() => []);
  const ranked = [...lines].sort((a, b) => (b.criticality?.primaryScore || -Infinity) - (a.criticality?.primaryScore || -Infinity));
  const selected = ranked.find((line) => line.id === state.selectedLineId) || ranked[0];
  state.selectedLineId = selected?.id || "";
  const label = selected?.label || null;
  viewRoot.innerHTML = heading("Criticality explorer", "A line can be important for accessibility, delay, transfers, and station isolation in different ways. Inspect each output dimension.", `<button class="button" data-view="lines">Open line facts</button>`) + `
    <div class="criticality-grid"><section class="card"><div class="card-header"><h2>Ranked lines</h2><span class="subtle">primary accessibility loss</span></div>${ranked.length ? `<div class="bar-list">${ranked.map((line) => `<button data-line="${escapeHtml(line.id)}" style="border:0;background:transparent;text-align:left;padding:0"><div class="bar-item"><span class="bar-label">${escapeHtml(line.displayName)}<small>${escapeHtml(line.modeName)}</small></span><span class="bar-track"><i style="width:${Math.max(0, Math.min(100, (line.criticality?.primaryScore ?? 0) * 100))}%"></i></span><span class="bar-value">${line.criticality ? percent(line.criticality.primaryScore, 1) : "—"}</span></div></button>`).join("")}</div>` : empty("No criticality results", "Queue a simulation or inference run for this snapshot.")}</section><section>${selected ? card(`${selected.displayName} · impact detail`, `<div class="metric-cards">${metricCard("Accessibility loss", selected.criticality?.accessibility_auc_loss, true, "", label?.accessibility_auc_loss)}${metricCard("Unreachable share", selected.criticality?.unreachable_share, true, "", label?.unreachable_share)}${metricCard("Mean reachable delay", selected.criticality?.mean_delay_reachable_seconds, false, " s", label?.mean_delay_reachable_seconds)}${metricCard("P95 reachable delay", selected.criticality?.p95_delay_reachable_seconds, false, " s", label?.p95_delay_reachable_seconds)}${metricCard("Additional transfers", selected.criticality?.mean_extra_transfers, false, "", label?.mean_extra_transfers)}${metricCard("Stations losing all service", selected.criticality?.stations_losing_all_service_share, true, "", label?.stations_losing_all_service_share)}</div><div class="card-body"><dl class="key-value"><dt>Predicted vs simulated</dt><dd>${selected.criticality ? "Prediction artifact indexed" : "Prediction not available"} · ${label ? "simulator label present" : "no compatible simulator label"}</dd><dt>Uncertainty</dt><dd>${selected.criticality?.uncertainty === null || selected.criticality?.uncertainty === undefined ? "Not registered; zero is not assumed." : number(selected.criticality.uncertainty, 2)}</dd><dt>Provenance</dt><dd>${escapeHtml(shortId(selected.snapshotId))} · ${escapeHtml(selectedModel()?.version || "no model")}</dd></dl></div>`, "section-card") : empty("No line selected", "Select a ranked line.")}</section></div>`;
  bindActions();
}

async function renderEmbeddings() {
  const facet = state.facet || "general";
  if (!state.snapshotId) { viewRoot.innerHTML = heading("Embedding explorer", "Compare line representations with the original embedding distance, not only a two-dimensional projection.") + empty("No snapshot selected", "Choose a snapshot in the top bar."); return; }
  try {
    const preview = await fetchJson(`/api/embeddings?snapshotId=${encodeURIComponent(state.snapshotId)}&facet=${encodeURIComponent(facet)}`);
    const points = preview.points || [];
    const minX = Math.min(...points.map((point) => point.x), 0); const maxX = Math.max(...points.map((point) => point.x), 1); const minY = Math.min(...points.map((point) => point.y), 0); const maxY = Math.max(...points.map((point) => point.y), 1);
    const w = 760; const h = 400; const pad = 28;
    const px = (value) => pad + (value - minX) / Math.max(1e-9, maxX - minX) * (w - pad * 2);
    const py = (value) => h - pad - (value - minY) / Math.max(1e-9, maxY - minY) * (h - pad * 2);
    const svg = `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Two dimensional line feature preview"><line class="scatter-axis" x1="${pad}" x2="${w-pad}" y1="${h-pad}" y2="${h-pad}" /><line class="scatter-axis" x1="${pad}" x2="${pad}" y1="${pad}" y2="${h-pad}" /><text class="scatter-label" x="${w-pad}" y="${h-8}">projection x</text><text class="scatter-label" x="7" y="${pad}">projection y</text>${points.map((point, index) => `<circle class="scatter-point" data-line="${escapeHtml(point.lineInstanceId)}" cx="${px(point.x).toFixed(1)}" cy="${py(point.y).toFixed(1)}" r="${point.criticalityPercentile === null ? 5 : 5 + Math.max(0, Math.min(4, point.criticalityPercentile * 4))}" fill="${["#2868dc", "#d65a61", "#158a82", "#b47b1a"][index % 4]}" aria-label="${escapeHtml(point.displayName)}" />`).join("")}</svg>`;
    viewRoot.innerHTML = heading("Embedding explorer", "The selected facet controls the feature space. Coordinates are a deterministic preview until an actual embedding artifact is registered.", facetTabs(facet)) + `<div class="card scatter-card"><div class="card-header"><h2>${escapeHtml(facet)} facet</h2><span class="subtle">${number(points.length)} line points</span></div><div class="scatter-surface">${svg}<p class="chart-note">${escapeHtml(preview.warning)} Actual cosine similarity and nearest-neighbour rank remain the retrieval metric; projection distance is not used for search.</p></div></div>`;
    viewRoot.querySelectorAll("[data-facet]").forEach((button) => button.addEventListener("click", () => { state.facet = button.dataset.facet; renderEmbeddings(); updateProvenance(); }));
    bindActions();
  } catch (error) { showNotice(error.message, true); viewRoot.innerHTML = heading("Embedding explorer", "The selected snapshot could not be projected.") + empty("Projection unavailable", error.message); }
}

async function renderRuns() {
  const runs = await fetchJson("/api/runs?limit=100").catch(() => []);
  if (state.runId) {
    await renderRunDetail();
    return;
  }
  viewRoot.innerHTML = heading("Runs", "The operational record of compile, simulation, dataset, training, evaluation, and inference work.", `<button class="button primary" data-action="queue-simulation">Queue selected simulation</button>`) + card("Run queue", runs.length ? `<div class="table-wrap run-table"><table class="data-table"><thead><tr><th>Run ID</th><th>Type</th><th>Status</th><th>Progress</th><th>Started</th><th>Commit</th><th>Inputs</th></tr></thead><tbody>${runs.map((run) => `<tr><td><button class="line-link run-id" data-run="${escapeHtml(run.id)}">${escapeHtml(shortId(run.id))}</button></td><td>${escapeHtml(run.kind)}</td><td>${statusPill(run.status)}</td><td>${run.progress.total ? `${number(run.progress.completed)} / ${number(run.progress.total)} ${escapeHtml(run.progress.unit)}` : "—"}</td><td>${escapeHtml(date(run.startedAt || run.createdAt))}</td><td class="mono">${escapeHtml(run.gitCommit || "—")}</td><td class="small-muted">${escapeHtml(run.snapshotId || run.datasetId || run.modelId || "—")}</td></tr>`).join("")}</tbody></table></div>` : empty("No runs recorded", "Queue a known run type to create a durable lineage record."));
  bindActions();
}

async function renderRunDetail() {
  const run = await fetchJson(`/api/runs/${encodeURIComponent(state.runId)}`).catch((error) => ({ error: error.message }));
  if (run.error) { showNotice(run.error, true); state.runId = ""; renderRuns(); return; }
  const events = run.events || [];
  const active = ["queued", "claimed", "running"].includes(run.status);
  viewRoot.innerHTML = heading(`${run.kind}`, `${shortId(run.id)} · ${run.snapshotId || run.datasetId || run.modelId || "unscoped"}`, `<button class="button" data-action="back-runs">Back to runs</button>${active ? `<button class="button danger" data-action="cancel-run">Cancel</button>` : ""}`) + `<div class="run-detail-grid">${card("Run specification", `<dl class="key-value"><dt>Status</dt><dd>${statusPill(run.status)}</dd><dt>Fingerprint</dt><dd class="mono">${escapeHtml(run.fingerprint)}</dd><dt>Git commit</dt><dd class="mono">${escapeHtml(run.gitCommit || "not recorded")}</dd><dt>Worker</dt><dd>${escapeHtml(run.workerId || "waiting for worker")}</dd><dt>Created</dt><dd>${escapeHtml(date(run.createdAt))}</dd><dt>Finished</dt><dd>${escapeHtml(date(run.finishedAt))}</dd><dt>Cancellation</dt><dd>${run.cancelRequested ? "requested" : "not requested"}</dd></dl>`)}${card("Step timeline", `<div class="bar-list">${(run.steps || []).map((step) => `<div class="bar-item"><span class="bar-label">${escapeHtml(step.step)}</span><span class="bar-track"><i style="width:${step.status === "succeeded" ? 100 : step.status === "running" ? 55 : 0}%"></i></span><span class="bar-value">${escapeHtml(step.status)}</span></div>`).join("") || `<div class="small-muted">No step events yet.</div>`}</div>`)}</div>${card("Structured events", `<div class="event-log">${events.map((event) => `<div class="event-line"><span class="seq">#${number(event.seq)}</span><span class="event-type">${escapeHtml(event.type)}</span><span class="event-message">${escapeHtml(event.message || event.step || (event.value !== undefined ? `${event.name} = ${number(event.value, 5)}` : event.artifactKind || ""))}</span></div>`).join("") || `<div class="small-muted">Waiting for the worker.</div>`}</div>`, "section-card")}${card("Produced artifacts", run.artifacts?.length ? `<div class="table-wrap"><table class="data-table"><thead><tr><th>Kind</th><th>Fingerprint</th><th>SHA-256</th><th>URI</th></tr></thead><tbody>${run.artifacts.map((artifact) => `<tr><td>${escapeHtml(artifact.kind)}</td><td class="mono">${escapeHtml(shortId(artifact.fingerprint))}</td><td class="mono">${escapeHtml(shortId(artifact.sha256))}</td><td class="mono">${escapeHtml(artifact.uri)}</td></tr>`).join("")}</tbody></table></div>` : empty("No artifacts yet", "The worker will register outputs after the command completes."), "section-card")}${card("Raw logs", `<pre class="log-box">${escapeHtml((run.logs || []).map((log) => `[${log.stream}] ${log.line}`).join("\n") || "No raw logs recorded yet.")}</pre>`, "section-card")}`;
  bindActions();
  if (active) startEventStream(run.id);
}

function startEventStream(runId) {
  if (state.eventSource) state.eventSource.close();
  state.eventSource = new EventSource(`/api/runs/${encodeURIComponent(runId)}/events`);
  state.eventSource.onmessage = () => { if (state.view === "runs" && state.runId === runId) renderRunDetail(); };
  state.eventSource.onerror = () => { state.eventSource?.close(); state.eventSource = null; };
}

async function renderPipeline() {
  const pipeline = await fetchJson("/api/pipeline").catch(() => ({ nodes: [], edges: [] }));
  viewRoot.innerHTML = heading("Pipeline", "Every output should be traceable from feed revision to compiled snapshot, dataset, model, inference, and similarity result.") + card("Artifact graph", `<div class="pipeline"><div class="pipeline-flow">${pipeline.nodes.map((node, index) => `${index ? `<span class="pipeline-arrow"></span>` : ""}<div class="pipeline-node ${node.status === "ready" ? "ready" : ""}"><strong>${escapeHtml(node.label)}</strong><small>${escapeHtml(node.status)}</small></div>`).join("")}</div></div>`);
}

async function renderEvaluation() {
  const metrics = state.overview?.latestEvaluation || [];
  viewRoot.innerHTML = heading("Evaluation", "Metrics only become meaningful next to their split, baseline, dataset, model, and network provenance.", `<button class="button" data-view="runs">Open evaluation runs</button>`) + card("Latest registered metrics", metrics.length ? `<div class="table-wrap"><table class="data-table"><thead><tr><th>Metric</th><th>Value</th><th>Split</th><th>Network</th><th>Recorded</th></tr></thead><tbody>${metrics.map((metric) => `<tr><td>${escapeHtml(metric.name)}</td><td><strong>${number(metric.value, 3)}</strong></td><td>${escapeHtml(metric.split || "—")}</td><td>${escapeHtml(metric.networkId || "overall")}</td><td>${escapeHtml(date(metric.createdAt))}</td></tr>`).join("")}</tbody></table></div>` : empty("No evaluation metrics", "Run the evaluation suite after a dataset and model have been registered.")) + card("Required comparisons", `<div class="table-wrap"><table class="data-table"><thead><tr><th>Claim</th><th>Required metric</th><th>Baseline</th><th>State</th></tr></thead><tbody><tr><td>Criticality ranking</td><td>Spearman · NDCG@10 · top-10% recall</td><td>frequency · length · betweenness</td><td>${check(false)}</td></tr><tr><td>Snapshot identity</td><td>Recall@1 · Recall@5 · MRR</td><td>name/sequence matcher</td><td>${check(false)}</td></tr><tr><td>Facet semantics</td><td>facet NDCG · human agreement</td><td>engineered signature</td><td>${check(false)}</td></tr></tbody></table></div>`, "section-card");
}

async function renderModels() {
  const models = await fetchJson("/api/models").catch(() => []);
  viewRoot.innerHTML = heading("Models", "Checkpoint files are immutable. Aliases can move, but old model versions and saved views retain their provenance.") + card("Model registry", models.length ? `<div class="table-wrap"><table class="data-table"><thead><tr><th>Version</th><th>Status</th><th>Aliases</th><th>Backend</th><th>Heads</th><th>Dataset</th><th>Checkpoint</th></tr></thead><tbody>${models.map((model) => `<tr><td><strong>${escapeHtml(model.version)}</strong><br><span class="mono">${escapeHtml(shortId(model.id))}</span></td><td>${statusPill(model.status)}</td><td><div class="tag-list">${model.aliases.map((alias) => `<span class="tag blue">${escapeHtml(alias)}</span>`).join("") || `<span class="small-muted">none</span>`}</div></td><td>${escapeHtml(model.architecture.backend || "—")}</td><td>${model.supportedHeads.map((head) => `<span class="tag">${escapeHtml(head)}</span>`).join(" ")}</td><td class="mono">${escapeHtml(shortId(model.datasetId))}</td><td class="mono">${escapeHtml(shortId(model.checkpointArtifactId))}</td></tr>`).join("")}</tbody></table></div>` : empty("No model versions", "The seeded demo model appears after the filesystem registry is synced."));
}

async function renderExperiments() {
  viewRoot.innerHTML = heading("Experiments", "Group runs by a frozen dataset and changed configuration. This view stays small rather than becoming a second task manager.") + card("Experiment groups", empty("No experiment groups registered", "Use GitHub Issues or Projects for coding tasks; add run grouping here when an experiment manifest exists."));
}

async function renderBenchmark() {
  const annotations = await fetchJson("/api/annotation-tasks/next").catch(() => ({ task: null }));
  viewRoot.innerHTML = heading("Human benchmark", "Pairwise judgements make the ambiguity of line similarity explicit instead of hiding it under one universal score.") + card("Annotation queue", annotations.task ? empty("Task ready", "Complete the comparison below.") : empty("No benchmark task", annotations.message || "Create a dataset and candidate pool before annotation."));
}

async function renderView() {
  showNotice("");
  document.querySelectorAll("[data-view]").forEach((link) => link.classList.toggle("active", link.dataset.view === state.view));
  try {
    if (state.view === "overview") renderOverview();
    else if (state.view === "data") renderData();
    else if (state.view === "dataset") renderDataset();
    else if (state.view === "pipeline") await renderPipeline();
    else if (state.view === "runs") await renderRuns();
    else if (state.view === "evaluation") await renderEvaluation();
    else if (state.view === "models") await renderModels();
    else if (state.view === "experiments") await renderExperiments();
    else if (state.view === "network") await renderNetwork();
    else if (state.view === "lines") await renderLines();
    else if (state.view === "similarity") await renderSimilarity();
    else if (state.view === "criticality") await renderCriticality();
    else if (state.view === "embeddings") await renderEmbeddings();
    else if (state.view === "benchmark") await renderBenchmark();
    else { state.view = "overview"; renderOverview(); }
  } catch (error) { showNotice(error.message, true); viewRoot.innerHTML = heading("Transit Lab", "The selected view failed to load.") + empty("View unavailable", error.message); }
}

function go(view) {
  state.view = view;
  window.location.hash = view;
  renderView();
}

async function queueRun(spec) {
  try {
    const run = await fetchJson("/api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(spec) });
    state.runId = run.id;
    showNotice(`Queued ${run.kind}. Run fingerprint ${shortId(run.fingerprint)}. Start the worker in another terminal to claim it.`);
    go("runs");
  } catch (error) { showNotice(error.message, true); }
}

function selectedSimulationSpec() {
  return state.snapshotId ? { kind: "simulate-criticality", snapshotId: state.snapshotId, simulationConfig: "configs/experiments/vertical-slice.yaml" } : null;
}

async function cancelRun(runId) {
  try { await fetchJson(`/api/runs/${encodeURIComponent(runId)}/cancel`, { method: "POST" }); renderRunDetail(); } catch (error) { showNotice(error.message, true); }
}

function bindActions() {
  viewRoot.querySelectorAll("[data-view]").forEach((element) => element.addEventListener("click", (event) => { event.preventDefault(); go(element.dataset.view); }));
  viewRoot.querySelectorAll("[data-line]").forEach((element) => element.addEventListener("click", (event) => { event.preventDefault(); state.selectedLineId = element.dataset.line; if (state.view === "network") go("lines"); else renderView(); }));
  viewRoot.querySelectorAll("[data-run]").forEach((element) => element.addEventListener("click", () => { state.runId = element.dataset.run; renderView(); }));
  viewRoot.querySelectorAll("[data-action]").forEach((element) => element.addEventListener("click", async () => {
    const action = element.dataset.action;
    if (action === "queue-simulation") { const spec = selectedSimulationSpec(); if (spec) await queueRun(spec); else showNotice("Select a snapshot before queueing a simulation.", true); }
    if (action === "queue-dataset") {
      const snapshotIds = (state.catalog?.snapshots || []).filter((snapshot) => snapshot.graphPath).map((snapshot) => snapshot.id);
      if (snapshotIds.length) await queueRun({ kind: "build-dataset", snapshotIds, splitConfig: { strategy: "system-level", holdoutNetworks: [] }, featureSchema: "station-line-relational-v2" });
      else showNotice("No compiled graph snapshots are ready for a dataset.", true);
    }
    if (action === "open-active-run") { state.runId = state.overview?.activeRun?.id || ""; if (state.runId) go("runs"); }
    if (action === "cancel-active-run") { if (state.overview?.activeRun) await cancelRun(state.overview.activeRun.id); }
    if (action === "cancel-run") await cancelRun(state.runId);
    if (action === "back-runs") { state.runId = ""; renderRuns(); }
  }));
}

networkSelector.addEventListener("change", () => { state.networkId = networkSelector.value; state.snapshotId = state.catalog.snapshots.find((snapshot) => snapshot.networkId === state.networkId)?.id || ""; state.lines = []; populateSelectors(); syncUrl(); renderView(); });
snapshotSelector.addEventListener("change", () => { state.snapshotId = snapshotSelector.value; state.networkId = selectedSnapshot()?.networkId || state.networkId; state.date = selectedSnapshot()?.serviceDate || state.date; state.lines = []; state.network = null; populateSelectors(); syncUrl(); renderView(); });
modelSelector.addEventListener("change", () => { state.modelId = modelSelector.value; populateSelectors(); syncUrl(); renderView(); });
dateSelector.addEventListener("change", () => { state.date = dateSelector.value; syncUrl(); });
document.querySelector("#refresh-button").addEventListener("click", async () => { try { await refreshCatalog(); await renderView(); } catch (error) { showNotice(error.message, true); setStatus("Needs attention", "failed"); } });
document.querySelector("#provenance-help").addEventListener("click", () => showNotice("Every result is scoped to the selected network, snapshot, service date, dataset/model, and facet. Changing an upstream artifact creates a new fingerprint; old results remain inspectable."));
document.querySelectorAll("[data-view]").forEach((link) => link.addEventListener("click", (event) => { event.preventDefault(); go(link.dataset.view); }));
window.addEventListener("hashchange", () => { state.view = currentView(); state.runId = ""; renderView(); });

async function boot() {
  try {
    await refreshCatalog();
    await renderView();
  } catch (error) {
    setStatus("Offline", "failed");
    showNotice(error.message, true);
    viewRoot.innerHTML = heading("Transit Lab", "The local control plane is not reachable.") + empty("Start the API server", "Run bun run apps/api/src/server.js and refresh this page.");
  }
}

boot();
