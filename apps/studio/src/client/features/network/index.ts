import { api, escapeHtml, formatCount, shortId } from "../../api.ts";
import { navigate } from "../../routes/router.ts";
import { GraphRenderer, modeColor, transitMode, validateNetwork } from "../../../../../../packages/visualizations/src/index.ts";
import { errorCard, loading, sectionHeading } from "../../components/ui.ts";

export async function renderNetwork(container) {
  container.innerHTML = loading("Loading indexed snapshots…");
  try {
    const snapshots = await api("/api/snapshots");
    if (!snapshots.length) {
      container.innerHTML = errorCard(new Error("No compiled snapshot is indexed yet. Produce a Rust snapshot, then refresh the index."));
      return;
    }
    container.innerHTML = shell(snapshots);
    const select = container.querySelector("#network-snapshot");
    const lineList = container.querySelector("#network-lines");
    const inspector = container.querySelector("#network-inspector");
    const canvas = container.querySelector("#network-canvas");
    const search = container.querySelector("#network-search");
    const height = container.querySelector("#network-height");
    const showStations = container.querySelector("#network-show-stations");
    const showTransfers = container.querySelector("#network-show-transfers");
    const canvasEmpty = container.querySelector("#network-canvas-empty");
    let network = null;
    let renderer = null;
    let selectedLine = null;
    let selectedStation = null;

    try {
      renderer = new GraphRenderer(canvas, {
        onHover: (station, position) => {
          if (station === null) return;
          inspector.innerHTML = stationDetails(network, station);
        },
        onSelect: (station) => {
          if (station === null) return;
          selectedStation = station;
          inspector.innerHTML = stationDetails(network, station);
        }
      });
    } catch (error) {
      canvasEmpty.textContent = `3D renderer unavailable: ${error.message}`;
    }

    const renderLines = () => {
      const term = search.value.trim().toLowerCase();
      const lines = (network?.lines || []).filter((line, index) => {
        const id = Number(line.index ?? index);
        return !term || String(line.display_name || line.canonical_id || id).toLowerCase().includes(term) || String(id).includes(term);
      });
      lineList.innerHTML = lines.map((line, position) => {
        const index = Number(line.index ?? network.lines.indexOf(line));
        const selected = selectedLine === index ? " selected" : "";
        return `<button class="line-button${selected}" type="button" data-line="${index}"><span class="line-swatch" style="background:${modeColor(line.mode)}"></span><span>${escapeHtml(line.display_name || line.canonical_id || `Line ${index}`)}</span><span class="line-meta">${escapeHtml(transitMode(line.mode).label)}</span></button>`;
      }).join("") || `<div class="empty">No lines match this search.</div>`;
      lineList.querySelectorAll("[data-line]").forEach((button) => button.addEventListener("click", () => {
        selectedLine = Number(button.dataset.line);
        if (renderer) renderer.setSelectedLine(selectedLine);
        renderLines();
        inspector.innerHTML = lineDetails(network, selectedLine);
      }));
      if (renderer) renderer.setVisibleLines(lines.map((line, index) => Number(line.index ?? network.lines.indexOf(line))));
    };

    const load = async (snapshotId) => {
      canvasEmpty.textContent = "Loading network…";
      try {
        network = validateNetwork(await api(`/api/snapshots/${encodeURIComponent(snapshotId)}/network`));
        network.lines = (network.lines || []).map((line, index) => ({ ...line, index: Number(line.index ?? index) }));
        selectedLine = null;
        selectedStation = null;
        const snapshot = snapshots.find((item) => item.id === snapshotId);
        container.querySelector("#network-title").textContent = snapshot?.sourceName || snapshot?.networkId || "Compiled snapshot";
        container.querySelector("#network-subtitle").textContent = `${snapshot?.serviceDate || "Unknown date"} · ${shortId(snapshotId)} · ${formatCount(network.stations?.length)} stations`;
        container.querySelector("#network-stations").textContent = formatCount(network.stations?.length);
        container.querySelector("#network-line-count").textContent = formatCount(network.lines?.length);
        container.querySelector("#network-edges").textContent = formatCount(network.transit_edges?.length);
        container.querySelector("#network-interchanges").textContent = formatCount(network.interchanges?.length);
        if (renderer) {
          renderer.setNetwork(network);
          renderer.setHeightMode(height.value);
          renderer.setShowStations(showStations.checked);
          renderer.setShowTransfers(showTransfers.checked);
          canvasEmpty.hidden = true;
        }
        inspector.innerHTML = `<div class="empty"><strong>Select a line or station</strong><p>Use the scene to inspect the indexed snapshot.</p></div>`;
        renderLines();
      } catch (error) {
        canvasEmpty.hidden = false;
        canvasEmpty.textContent = error.message;
      }
    };

    select.addEventListener("change", () => load(select.value));
    search.addEventListener("input", renderLines);
    height.addEventListener("change", () => renderer?.setHeightMode(height.value));
    showStations.addEventListener("change", () => renderer?.setShowStations(showStations.checked));
    showTransfers.addEventListener("change", () => renderer?.setShowTransfers(showTransfers.checked));
    container.querySelector("#network-reset")?.addEventListener("click", () => renderer?.fit());
    container.querySelector("#network-top")?.addEventListener("click", () => renderer?.topView());
    await load(select.value);
  } catch (error) {
    container.innerHTML = errorCard(error);
  }
}

function shell(snapshots) {
  return `<div class="page-intro"><div><p class="eyebrow">Compiled network</p><h2 id="network-title">Network</h2><p id="network-subtitle">Choose an indexed service-day snapshot to inspect its stations, lines, and relations.</p></div><a class="button button-quiet" href="/data" data-route="data">View lineage →</a></div>
    <div class="card network-panel"><div class="network-summary"><div><span>Stations</span><strong id="network-stations">—</strong></div><div><span>Lines</span><strong id="network-line-count">—</strong></div><div><span>Transit edges</span><strong id="network-edges">—</strong></div><div><span>Interchanges</span><strong id="network-interchanges">—</strong></div></div></div>
    <div class="network-layout" style="margin-top:16px"><aside class="card network-panel scene-panel"><div class="section-head"><div><h3>Scene</h3><p>Filter the visible graph.</p></div></div><div class="scene-controls"><div class="field"><label for="network-snapshot">Snapshot</label><select class="select" id="network-snapshot">${snapshots.map((snapshot) => `<option value="${escapeHtml(snapshot.id)}">${escapeHtml(snapshot.sourceName || snapshot.networkId || shortId(snapshot.id))} · ${escapeHtml(snapshot.serviceDate)}</option>`).join("")}</select></div><div class="field"><label for="network-search">Find a line</label><input class="input" id="network-search" type="search" placeholder="Name or index" /></div><div class="field"><label for="network-height">Height shows</label><select class="select" id="network-height"><option value="connectivity">Network role</option><option value="departures">Daily departures</option><option value="service">Service span</option></select></div><label class="toggle-line"><span>Show stations</span><input id="network-show-stations" type="checkbox" checked /></label><label class="toggle-line"><span>Show transfers</span><input id="network-show-transfers" type="checkbox" checked /></label><div style="display:flex;gap:7px"><button class="button button-quiet" id="network-reset" type="button">Reset view</button><button class="button button-quiet" id="network-top" type="button">Top view</button></div></div><div class="section-head"><div><h3>Lines</h3><p>Click to isolate one.</p></div></div><div class="line-list" id="network-lines"></div></aside><div class="canvas-wrap"><canvas id="network-canvas" aria-label="Interactive 3D compiled network"></canvas><div class="canvas-empty" id="network-canvas-empty">Loading network…</div><div class="canvas-caption">Drag to orbit · scroll to zoom · click a station to inspect</div></div><aside class="card network-panel inspector-panel"><div class="section-head"><div><h3>Inspector</h3><p>Manifest-backed detail</p></div></div><div class="inspector" id="network-inspector"></div></aside></div>`;
}

function lineDetails(network, index) {
  const line = network?.lines?.find((candidate, position) => Number(candidate.index ?? position) === Number(index));
  if (!line) return `<div class="empty"><strong>Line not found</strong></div>`;
  const features = [
    ["Mode", transitMode(line.mode).label],
    ["Canonical ID", line.canonical_id || "—"],
    ["Stations", line.station_count ?? line.stationCount ?? "—"],
    ["Patterns", line.pattern_count ?? line.patternCount ?? "—"],
    ["Daily trips", line.daily_trip_count ?? line.dailyTripCount ?? "—"]
  ];
  return `<p class="eyebrow">Line ${escapeHtml(index)}</p><h3>${escapeHtml(line.display_name || line.canonical_id || `Line ${index}`)}</h3><dl class="detail-list">${features.map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("")}</dl>`;
}

function stationDetails(network, index) {
  const station = network?.stations?.[index];
  if (!station) return `<div class="empty"><strong>Station not found</strong></div>`;
  return `<p class="eyebrow">Station ${escapeHtml(index)}</p><h3>${escapeHtml(station.name || station.canonical_id || `Station ${index}`)}</h3><dl class="detail-list"><div><dt>Coordinates</dt><dd>${escapeHtml(Number(station.latitude).toFixed(4))}, ${escapeHtml(Number(station.longitude).toFixed(4))}</dd></div><div><dt>Lines</dt><dd>${escapeHtml(station.line_count ?? "—")}</dd></div><div><dt>Patterns</dt><dd>${escapeHtml(station.pattern_count ?? "—")}</dd></div><div><dt>Daily departures</dt><dd>${escapeHtml(station.daily_departures ?? "—")}</dd></div><div><dt>Terminal</dt><dd>${station.terminal ? "Yes" : "No"}</dd></div></dl>`;
}
