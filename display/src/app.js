import { GraphRenderer } from "./renderer.js";
import { TRANSIT_MODES, modeColor, transitMode } from "./modes.js";

const elements = {
  canvas: document.querySelector("#graph-canvas"),
  canvasEmpty: document.querySelector("#canvas-empty"),
  canvasWrap: document.querySelector("#canvas-wrap"),
  clearSelection: document.querySelector("#clear-selection"),
  dataStatus: document.querySelector("#data-status"),
  fileInput: document.querySelector("#snapshot-file"),
  fileName: document.querySelector("#file-name"),
  fullscreen: document.querySelector("#fullscreen-button"),
  fullscreenSelection: document.querySelector("#fullscreen-selection"),
  fullscreenSelectionClose: document.querySelector("#fullscreen-selection-close"),
  fullscreenSelectionContent: document.querySelector("#fullscreen-selection-content"),
  fullscreenSelectionTitle: document.querySelector("#fullscreen-selection-title"),
  heightSelect: document.querySelector("#height-select"),
  hoverCard: document.querySelector("#hover-card"),
  inspectorContent: document.querySelector("#inspector-content"),
  inspectorTitle: document.querySelector("#inspector-title"),
  legendHeight: document.querySelector("#legend-height"),
  lineList: document.querySelector("#line-list"),
  modeFilters: document.querySelector("#mode-filters"),
  modeLegend: document.querySelector("#mode-legend"),
  nodeSearch: document.querySelector("#node-search"),
  renderStatus: document.querySelector("#render-status"),
  resetView: document.querySelector("#reset-view"),
  snapshotChip: document.querySelector("#snapshot-chip"),
  snapshotSelect: document.querySelector("#snapshot-select"),
  summaryCopy: document.querySelector("#summary-copy"),
  summaryTitle: document.querySelector("#summary-title"),
  statEdges: document.querySelector("#stat-edges"),
  statInterchanges: document.querySelector("#stat-interchanges"),
  statLines: document.querySelector("#stat-lines"),
  statStations: document.querySelector("#stat-stations"),
  stationsToggle: document.querySelector("#stations-toggle"),
  topView: document.querySelector("#top-view"),
  transfersToggle: document.querySelector("#transfers-toggle"),
  allLines: document.querySelector("#all-lines-button")
};

const state = {
  network: null,
  source: "",
  snapshots: [],
  selectedLine: null,
  selectedStation: null,
  hoveredStation: null,
  search: "",
  heightMode: "connectivity",
  visibleLines: new Set(),
  error: ""
};

const heightLabels = {
  connectivity: "network role",
  departures: "daily departures",
  service: "service span"
};

const renderer = new GraphRenderer(elements.canvas, {
  onHover: showHover,
  onSelect: selectStationFromCanvas
});

function formatNumber(value) {
  return new Intl.NumberFormat("en", { maximumFractionDigits: 0 }).format(Number(value) || 0);
}

function formatDecimal(value, digits = 1) {
  return new Intl.NumberFormat("en", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits
  }).format(Number(value) || 0);
}

function formatDistance(metres) {
  const value = Number(metres) || 0;
  return value >= 1000 ? `${formatDecimal(value / 1000, 1)} km` : `${formatNumber(value)} m`;
}

function formatTime(seconds) {
  const value = Number(seconds) || 0;
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}`;
}

function formatDuration(seconds) {
  const value = Number(seconds) || 0;
  const hours = Math.floor(value / 3600);
  const minutes = Math.round((value % 3600) / 60);
  if (hours) return `${hours}h ${String(minutes).padStart(2, "0")}m`;
  return `${minutes}m`;
}

function linesAtStation(stationIndex) {
  if (!state.network) return [];
  const indexes = new Set(
    state.network.patterns
      .filter((pattern) => pattern.signature?.stops?.map(Number).includes(stationIndex))
      .map((pattern) => Number(pattern.signature.line))
  );
  return [...indexes]
    .map((index) => state.network.lines[index])
    .filter(Boolean);
}

function routeStops(lineIndex) {
  const pattern = state.network?.patterns.find((item) => Number(item.signature?.line) === lineIndex);
  if (!pattern) return [];
  return (pattern.signature.stops || [])
    .map((index) => state.network.stations[Number(index)]?.name)
    .filter(Boolean);
}

function interchangeCount() {
  if (!state.network) return 0;
  const pairs = new Set(
    state.network.interchanges.map((interchange) => {
      const left = Number(interchange.from);
      const right = Number(interchange.to);
      return left < right ? `${left}:${right}` : `${right}:${left}`;
    })
  );
  return pairs.size;
}

function makeElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function getManifest() {
  return state.network?.manifest || {};
}

function getSourceName() {
  const manifest = getManifest();
  return manifest.source_name || manifest.geographical_scope || "Loaded snapshot";
}

function validateNetwork(raw) {
  const network = raw?.network && raw.network.stations ? raw.network : raw;
  if (!network || !Array.isArray(network.stations) || !Array.isArray(network.lines)) {
    throw new Error("This file does not contain a Transit Lab network.json snapshot.");
  }
  if (!network.stations.every((station) => Number.isFinite(Number(station.latitude)) && Number.isFinite(Number(station.longitude)))) {
    throw new Error("The snapshot has stations without usable latitude and longitude values.");
  }
  return {
    ...network,
    transit_edges: Array.isArray(network.transit_edges) ? network.transit_edges : [],
    transfers: Array.isArray(network.transfers) ? network.transfers : [],
    interchanges: Array.isArray(network.interchanges) ? network.interchanges : [],
    patterns: Array.isArray(network.patterns) ? network.patterns : []
  };
}

function setStatus(label, tone = "") {
  elements.dataStatus.textContent = label;
  elements.dataStatus.className = `status-pill${tone ? ` status-pill-${tone}` : ""}`;
}

function loadNetwork(raw, source) {
  try {
    const network = validateNetwork(raw);
    state.network = network;
    state.source = source || getSourceName();
    state.error = "";
    state.selectedLine = null;
    state.selectedStation = null;
    state.hoveredStation = null;
    state.search = "";
    state.visibleLines = new Set(network.lines.map((line) => Number(line.index)));
    elements.nodeSearch.value = "";
    renderer.setNetwork(network);
    renderer.setHeightMode(state.heightMode);
    renderer.setVisibleLines(state.visibleLines);
    elements.canvasEmpty.hidden = true;
    elements.renderStatus.textContent = `${formatNumber(network.stations.length)} stations in view`;
    setStatus("Loaded");
    render();
  } catch (error) {
    state.error = error instanceof Error ? error.message : "The snapshot could not be opened.";
    setStatus("Needs attention", "error");
    elements.renderStatus.textContent = "Snapshot error";
    elements.canvasEmpty.hidden = false;
    elements.canvasEmpty.querySelector("strong").textContent = "Could not open snapshot";
    elements.canvasEmpty.querySelector("span:last-child").textContent = state.error;
  }
}

async function loadSnapshotPath(path, label) {
  if (!path) return;
  setStatus("Loading");
  elements.renderStatus.textContent = "Reading snapshot…";
  try {
    const requestPath = path.startsWith("/data/")
      ? `/api/network?path=${encodeURIComponent(path)}`
      : path;
    const response = await fetch(requestPath, { headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error(`Snapshot returned HTTP ${response.status}.`);
    loadNetwork(await response.json(), label || path);
  } catch (error) {
    state.error = error instanceof Error ? error.message : "The snapshot could not be loaded.";
    setStatus("Needs attention", "error");
    elements.renderStatus.textContent = "Snapshot error";
    elements.canvasEmpty.hidden = false;
    elements.canvasEmpty.querySelector("strong").textContent = "Could not load snapshot";
    elements.canvasEmpty.querySelector("span:last-child").textContent = state.error;
  }
}

async function loadFileList(files) {
  const networkFile = [...files].find((file) => file.name === "network.json" || file.webkitRelativePath?.endsWith("/network.json"));
  if (!networkFile) {
    state.error = "Choose a compiled snapshot folder containing network.json.";
    setStatus("Needs attention", "error");
    elements.fileName.textContent = state.error;
    return;
  }
  setStatus("Loading");
  elements.fileName.textContent = `Opening ${networkFile.webkitRelativePath || networkFile.name}`;
  try {
    loadNetwork(JSON.parse(await networkFile.text()), `Local file · ${networkFile.name}`);
    elements.fileName.textContent = `${networkFile.name} loaded locally`;
  } catch (error) {
    state.error = error instanceof Error ? error.message : "The selected file is not valid JSON.";
    setStatus("Needs attention", "error");
    elements.fileName.textContent = state.error;
  }
}

function snapshotDisplay(snapshot) {
  const scope = snapshot.scope && snapshot.scope !== snapshot.label ? ` · ${snapshot.scope}` : "";
  return `${snapshot.label}${scope}`;
}

function isViennaSnapshot(snapshot) {
  return /vienna|wien/i.test(`${snapshot.id} ${snapshot.label} ${snapshot.scope} ${snapshot.path}`);
}

async function loadSnapshotIndex() {
  try {
    const response = await fetch("/api/snapshots", { headers: { Accept: "application/json" } });
    if (!response.ok) return;
    state.snapshots = await response.json();
    elements.snapshotSelect.replaceChildren();
    if (!state.snapshots.length) {
      elements.snapshotSelect.add(new Option("No local snapshots found", ""));
      return;
    }
    state.snapshots.forEach((snapshot) => {
      elements.snapshotSelect.add(new Option(snapshotDisplay(snapshot), snapshot.path));
    });
    const queryPath = new URLSearchParams(window.location.search).get("snapshot");
    const selected = state.snapshots.find((snapshot) => snapshot.path === queryPath) ||
      state.snapshots.find(isViennaSnapshot) ||
      state.snapshots[0];
    elements.snapshotSelect.value = selected.path;
    await loadSnapshotPath(selected.path, selectedDisplay(selected));
  } catch {
    elements.snapshotSelect.replaceChildren(new Option("Open a local folder", ""));
    const queryPath = new URLSearchParams(window.location.search).get("snapshot");
    if (queryPath) await loadSnapshotPath(queryPath, queryPath);
  }
}

function selectedDisplay(snapshot) {
  return snapshot ? `${snapshot.label} · ${snapshot.scope}` : "Loaded snapshot";
}

function filteredLines() {
  const term = state.search.trim().toLowerCase();
  return (state.network?.lines || []).filter((line) => {
    if (!term) return true;
    return `${line.display_name} ${line.canonical_id} ${line.agency_key}`.toLowerCase().includes(term);
  });
}

function groupedLines(lines) {
  return TRANSIT_MODES.map((mode) => ({
    mode,
    lines: lines.filter((line) => transitMode(line.mode).key === mode.key)
  })).filter((group) => group.lines.length);
}

function renderModeFilters() {
  elements.modeFilters.replaceChildren();
  if (!state.network) return;
  groupedLines(state.network.lines).forEach(({ mode, lines }) => {
    const visibleCount = lines.filter((line) => state.visibleLines.has(Number(line.index))).length;
    const row = makeElement("label", "mode-filter");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = visibleCount === lines.length;
    checkbox.indeterminate = visibleCount > 0 && visibleCount < lines.length;
    checkbox.setAttribute("aria-label", `Show ${mode.label} lines`);
    checkbox.addEventListener("change", () => {
      lines.forEach((line) => {
        if (checkbox.checked) state.visibleLines.add(Number(line.index));
        else state.visibleLines.delete(Number(line.index));
      });
      syncVisibleLines();
    });
    const swatch = makeElement("i", "mode-swatch");
    swatch.style.backgroundColor = mode.color;
    const label = makeElement("span", "mode-filter-label", mode.label);
    const count = makeElement("span", "mode-filter-count", `${visibleCount}/${lines.length}`);
    row.append(checkbox, swatch, label, count);
    elements.modeFilters.append(row);
  });
}

function renderModeLegend() {
  elements.modeLegend.replaceChildren();
  if (!state.network) return;
  groupedLines(state.network.lines).forEach(({ mode }) => {
    const item = makeElement("span", "mode-legend-item");
    const swatch = makeElement("i", "mode-legend-swatch");
    swatch.style.backgroundColor = mode.color;
    item.append(swatch, makeElement("span", "", mode.label));
    elements.modeLegend.append(item);
  });
}

function renderLineRow(line) {
  const index = Number(line.index);
  const mode = transitMode(line.mode);
  const row = makeElement("div", `line-row${state.selectedLine === index ? " is-selected" : ""}`);
  const checkboxLabel = makeElement("label", "line-visibility");
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.checked = state.visibleLines.has(index);
  checkbox.setAttribute("aria-label", `Show ${line.display_name}`);
  checkbox.addEventListener("change", () => {
    if (checkbox.checked) state.visibleLines.add(index);
    else state.visibleLines.delete(index);
    syncVisibleLines();
  });
  const swatch = makeElement("i", "line-swatch");
  swatch.style.backgroundColor = mode.color;
  checkboxLabel.append(checkbox, swatch);
  const select = makeElement("button", "line-select", line.display_name || `Line ${index}`);
  select.type = "button";
  select.addEventListener("click", () => selectLine(index));
  const meta = makeElement("span", "line-meta", `${formatNumber(line.station_count)} stops · ${formatNumber(line.daily_trip_count)} trips`);
  row.append(checkboxLabel, select, meta);
  return row;
}

function renderLineList() {
  elements.lineList.replaceChildren();
  if (!state.network) {
    elements.lineList.append(makeElement("p", "list-empty", "Load a snapshot to see its lines."));
    return;
  }
  const lines = filteredLines();
  if (!lines.length) {
    elements.lineList.append(makeElement("p", "list-empty", "No lines match that search."));
    return;
  }
  groupedLines(lines).forEach(({ mode, lines: modeLines }) => {
    const group = makeElement("div", "line-group");
    const heading = makeElement("div", "line-group-heading");
    const swatch = makeElement("i", "line-group-swatch");
    swatch.style.backgroundColor = mode.color;
    heading.append(
      makeElement("span", "line-group-name", mode.label),
      makeElement("span", "line-group-count", String(modeLines.length))
    );
    heading.prepend(swatch);
    group.append(heading, ...modeLines.map(renderLineRow));
    elements.lineList.append(group);
  });
}

function updateRenderStatus() {
  if (!state.network) return;
  const visible = state.visibleLines.size;
  elements.renderStatus.textContent = `${formatNumber(visible)} of ${formatNumber(state.network.lines.length)} lines in view`;
}

function syncVisibleLines() {
  renderer.setVisibleLines(state.visibleLines);
  if (state.selectedStation !== null && !renderer.visibleStations.has(state.selectedStation)) {
    state.selectedStation = null;
    state.selectedLine = null;
    renderer.setSelectedLine(null);
    renderInspector();
  }
  renderModeFilters();
  renderLineList();
  updateRenderStatus();
}

function renderSummary() {
  const network = state.network;
  if (!network) {
    elements.summaryTitle.textContent = "No network loaded";
    elements.summaryCopy.textContent = "A compiled snapshot turns GTFS tables into places, routes, and connections you can inspect together.";
    [elements.statStations, elements.statLines, elements.statEdges, elements.statInterchanges].forEach((element) => { element.textContent = "—"; });
    elements.snapshotChip.textContent = "No snapshot";
    return;
  }
  const manifest = getManifest();
  const name = getSourceName();
  const scope = manifest.geographical_scope && manifest.geographical_scope !== name ? manifest.geographical_scope : "compiled GTFS feed";
  elements.summaryTitle.textContent = name;
  elements.summaryCopy.textContent = `${scope} · service date ${manifest.descriptor?.service_date || "not recorded"}. Inspect route order, station activity, and transfer structure in one view.`;
  elements.statStations.textContent = formatNumber(network.stations.length);
  elements.statLines.textContent = formatNumber(network.lines.length);
  elements.statEdges.textContent = formatNumber(network.transit_edges.length);
  elements.statInterchanges.textContent = formatNumber(interchangeCount());
  const snapshotId = network.snapshot_id || manifest.snapshot_id || "local snapshot";
  elements.snapshotChip.textContent = snapshotId.length > 16 ? `${snapshotId.slice(0, 7)}…${snapshotId.slice(-5)}` : snapshotId;
  elements.snapshotChip.title = snapshotId;
}

function detailRow(label, value, accent = false) {
  const row = makeElement("div", `detail-row${accent ? " detail-row-accent" : ""}`);
  const valueElement = typeof value === "string" ? makeElement("strong", "detail-value", value) : value;
  valueElement.classList.add("detail-value");
  row.append(makeElement("span", "detail-label", label), valueElement);
  return row;
}

function servedByList(stationIndex) {
  const lines = linesAtStation(stationIndex);
  if (!lines.length) return makeElement("strong", "detail-value", "Not recorded");
  const list = makeElement("div", "served-line-list");
  lines.forEach((line) => {
    const chip = makeElement("button", "served-line-chip", line.display_name || `Line ${line.index}`);
    chip.type = "button";
    chip.style.color = modeColor(line.mode);
    chip.style.borderColor = `${modeColor(line.mode)}66`;
    chip.addEventListener("click", () => selectLine(Number(line.index)));
    list.append(chip);
  });
  return list;
}

function canvasIsFullscreen() {
  if (document.fullscreenElement === elements.canvasWrap || document.webkitFullscreenElement === elements.canvasWrap) return true;
  try {
    return elements.canvasWrap.matches(":fullscreen");
  } catch {
    return false;
  }
}

function renderFullscreenSelection() {
  if (!canvasIsFullscreen() || !state.network || (state.selectedStation === null && state.selectedLine === null)) {
    elements.fullscreenSelection.hidden = true;
    return;
  }
  elements.fullscreenSelectionContent.replaceChildren();
  if (state.selectedStation !== null) {
    const station = state.network.stations[state.selectedStation];
    elements.fullscreenSelectionTitle.textContent = station.name || `Station ${state.selectedStation}`;
    elements.fullscreenSelectionContent.append(
      makeElement("p", "fullscreen-selection-meta", `${formatNumber(station.line_count)} lines · ${formatNumber(station.daily_departures)} daily departures`),
      servedByList(state.selectedStation)
    );
  } else {
    const line = state.network.lines[state.selectedLine];
    const mode = transitMode(line.mode);
    elements.fullscreenSelectionTitle.textContent = line.display_name || `Line ${state.selectedLine}`;
    const meta = makeElement("p", "fullscreen-selection-meta", `${mode.label} · ${formatNumber(line.station_count)} stops · ${formatNumber(line.daily_trip_count)} daily trips`);
    meta.style.color = mode.color;
    elements.fullscreenSelectionContent.append(meta);
  }
  elements.fullscreenSelection.hidden = false;
}

function selectLine(index) {
  state.selectedLine = index;
  state.selectedStation = null;
  renderer.setSelectedLine(index);
  renderInspector();
  renderLineList();
}

function selectStationFromCanvas(index) {
  if (index === null || index === undefined) {
    state.selectedStation = null;
    state.selectedLine = null;
    renderer.setSelectedLine(null);
  } else {
    state.selectedStation = index;
    state.selectedLine = null;
    renderer.setSelectedLine(null);
  }
  renderInspector();
  renderLineList();
}

function renderInspector() {
  renderFullscreenSelection();
  elements.inspectorContent.replaceChildren();
  if (!state.network || (state.selectedStation === null && state.selectedLine === null)) {
    elements.inspectorTitle.textContent = "Nothing selected";
    const empty = makeElement("div", "inspector-empty");
    empty.append(makeElement("span", "selection-icon", "+"), makeElement("p", "", "Click a station or line to see the data behind it."));
    elements.inspectorContent.append(empty);
    return;
  }

  if (state.selectedStation !== null) {
    const station = state.network.stations[state.selectedStation];
    elements.inspectorTitle.textContent = station.name || `Station ${state.selectedStation}`;
    const badge = makeElement("span", "detail-badge", station.terminal ? "Terminal" : "Station");
    const block = makeElement("div", "detail-block");
    block.append(
      badge,
      makeElement("p", "detail-id", station.canonical_id || `station:${state.selectedStation}`),
      detailRow("Coordinates", `${formatDecimal(station.latitude, 4)}, ${formatDecimal(station.longitude, 4)}`),
      detailRow("Lines serving", formatNumber(station.line_count), Number(station.line_count) > 1),
      detailRow("Served by", servedByList(state.selectedStation)),
      detailRow("Daily departures", formatNumber(station.daily_departures)),
      detailRow("Daily arrivals", formatNumber(station.daily_arrivals)),
      detailRow("Patterns", formatNumber(station.pattern_count)),
      detailRow("Active window", `${formatTime(station.first_departure)}–${formatTime(station.last_departure)}`)
    );
    elements.inspectorContent.append(block);
    return;
  }

  const line = state.network.lines[state.selectedLine];
  elements.inspectorTitle.textContent = line.display_name || `Line ${state.selectedLine}`;
  const mode = transitMode(line.mode);
  const badge = makeElement("span", "detail-badge detail-badge-line", mode.label);
  badge.style.color = mode.color;
  badge.style.borderColor = `${mode.color}66`;
  const block = makeElement("div", "detail-block");
  block.append(
    badge,
    makeElement("p", "detail-id", line.canonical_id || `line:${state.selectedLine}`),
    detailRow("Stations", formatNumber(line.station_count), true),
    detailRow("Route", routeStops(state.selectedLine).join(" → ") || "Not recorded"),
    detailRow("Route length", formatDistance(line.route_length_metres)),
    detailRow("Daily trips", formatNumber(line.daily_trip_count)),
    detailRow("Median headway", formatDuration(line.median_headway_seconds)),
    detailRow("Transfer stations", formatNumber(line.transfer_station_count)),
    detailRow("Unique stations", `${formatDecimal((Number(line.unique_station_fraction) || 0) * 100, 0)}%`),
    detailRow("Agency key", line.agency_key || "Not recorded")
  );
  const action = makeElement("button", "button button-primary inspector-action", state.visibleLines.has(state.selectedLine) ? "Hide this line" : "Show this line");
  action.type = "button";
  action.addEventListener("click", () => {
    if (state.visibleLines.has(state.selectedLine)) state.visibleLines.delete(state.selectedLine);
    else state.visibleLines.add(state.selectedLine);
    syncVisibleLines();
  });
  block.append(action);
  elements.inspectorContent.append(block);
}

function showHover(index, screenPosition) {
  state.hoveredStation = index;
  if (index === null || !state.network?.stations[index] || !screenPosition) {
    elements.hoverCard.hidden = true;
    return;
  }
  const station = state.network.stations[index];
  elements.hoverCard.replaceChildren(
    makeElement("strong", "hover-title", station.name || `Station ${index}`),
    makeElement("span", "hover-meta", `${formatNumber(station.line_count)} lines · ${formatNumber(station.daily_departures)} departures`)
  );
  const bounds = elements.canvas.getBoundingClientRect();
  const x = screenPosition.x + (elements.canvas.clientWidth ? bounds.left - elements.canvasWrap.getBoundingClientRect().left : 0);
  const y = screenPosition.y + (bounds.top - elements.canvasWrap.getBoundingClientRect().top);
  elements.hoverCard.style.left = `${Math.min(elements.canvasWrap.clientWidth - 180, Math.max(10, x + 14))}px`;
  elements.hoverCard.style.top = `${Math.min(elements.canvasWrap.clientHeight - 58, Math.max(10, y - 50))}px`;
  elements.hoverCard.hidden = false;
}

function render() {
  renderModeFilters();
  renderLineList();
  renderModeLegend();
  renderInspector();
  renderSummary();
  elements.legendHeight.textContent = heightLabels[state.heightMode];
  updateRenderStatus();
}

elements.snapshotSelect.addEventListener("change", () => {
  const snapshot = state.snapshots.find((item) => item.path === elements.snapshotSelect.value);
  loadSnapshotPath(elements.snapshotSelect.value, selectedDisplay(snapshot));
});
elements.fileInput.addEventListener("change", (event) => loadFileList(event.target.files || []));
elements.nodeSearch.addEventListener("input", (event) => {
  state.search = event.target.value;
  renderLineList();
});
elements.heightSelect.addEventListener("change", (event) => {
  state.heightMode = event.target.value;
  renderer.setHeightMode(state.heightMode);
  elements.legendHeight.textContent = heightLabels[state.heightMode];
});
elements.stationsToggle.addEventListener("change", (event) => renderer.setShowStations(event.target.checked));
elements.transfersToggle.addEventListener("change", (event) => renderer.setShowTransfers(event.target.checked));
elements.allLines.addEventListener("click", () => {
  if (!state.network) return;
  state.visibleLines = new Set(state.network.lines.map((line) => Number(line.index)));
  syncVisibleLines();
});
elements.clearSelection.addEventListener("click", () => selectStationFromCanvas(null));
elements.fullscreenSelectionClose.addEventListener("click", () => selectStationFromCanvas(null));
elements.resetView.addEventListener("click", () => renderer.fit());
elements.topView.addEventListener("click", () => renderer.topView());
elements.fullscreen.addEventListener("click", async () => {
  if (!document.fullscreenElement) await elements.canvasWrap.requestFullscreen?.();
  else await document.exitFullscreen?.();
});
document.addEventListener("fullscreenchange", () => {
  const isFullscreen = Boolean(document.fullscreenElement);
  elements.fullscreen.textContent = isFullscreen ? "×" : "↗";
  elements.fullscreen.title = isFullscreen ? "Exit fullscreen" : "Enter fullscreen";
  elements.fullscreen.setAttribute("aria-label", isFullscreen ? "Exit fullscreen" : "Enter fullscreen");
  requestAnimationFrame(() => {
    renderer.resize();
    renderer.render();
    renderFullscreenSelection();
  });
});

render();
loadSnapshotIndex();
