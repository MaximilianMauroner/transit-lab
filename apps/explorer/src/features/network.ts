import { api, escapeHtml, formatCount, shortId } from "../../../../packages/api-client/src/index.ts";
import { errorCard, loading, sectionHeading } from "../../../../packages/ui/src/index.ts";
import { GraphRenderer, validateNetwork } from "../../../../packages/visualizations/src/index.ts";

export async function renderNetwork(container: HTMLElement) {
  const snapshots = await api("/api/public/snapshots");
  if (!snapshots.length) {
    container.innerHTML = errorCard(new Error("No published snapshots are available."));
    return;
  }
  container.innerHTML = `<section class="intro"><div><p class="eyebrow">Published network</p><h2 id="network-title">Network snapshot</h2><p id="network-subtitle">Read-only station, line, pattern, and relation inspection.</p></div><select class="select" id="snapshot-select">${snapshots.map((snapshot) => `<option value="${escapeHtml(snapshot.id)}">${escapeHtml(snapshot.sourceName || snapshot.networkId || shortId(snapshot.id))} · ${escapeHtml(snapshot.serviceDate)}</option>`).join("")}</select></section><section class="network-layout"><div class="card scene"><canvas id="network-canvas" aria-label="Published transit network"></canvas><div id="network-empty" class="canvas-empty">${loading("Loading network…")}</div></div><aside class="card inspector"><div id="network-inspector">${sectionHeading("Inspector", "Select a station in the published network.")}</div></aside></section>`;
  const select = container.querySelector<HTMLSelectElement>("#snapshot-select");
  const canvas = container.querySelector<HTMLCanvasElement>("#network-canvas");
  const emptyState = container.querySelector<HTMLElement>("#network-empty");
  const inspector = container.querySelector<HTMLElement>("#network-inspector");
  let renderer;
  let network;
  try {
    renderer = new GraphRenderer(canvas, {
      onHover: (station) => { if (station !== null) inspector.innerHTML = stationDetails(network, station); },
      onSelect: (station) => { if (station !== null) inspector.innerHTML = stationDetails(network, station); }
    });
  } catch (error) {
    emptyState.textContent = error.message;
  }
  const load = async (snapshotId) => {
    emptyState.textContent = "Loading network…";
    emptyState.hidden = false;
    network = validateNetwork(await api(`/api/public/snapshots/${encodeURIComponent(snapshotId)}/network`));
    network.lines = (network.lines || []).map((line, index) => ({ ...line, index: Number(line.index ?? index) }));
    const snapshot = snapshots.find((candidate) => candidate.id === snapshotId);
    container.querySelector("#network-title").textContent = snapshot?.sourceName || snapshot?.networkId || "Network snapshot";
    container.querySelector("#network-subtitle").textContent = `${snapshot?.serviceDate || "Unknown date"} · ${shortId(snapshotId)} · ${formatCount(network.stations?.length)} stations`;
    if (renderer) {
      renderer.setNetwork(network);
      emptyState.hidden = true;
    }
  };
  select.addEventListener("change", () => load(select.value).catch((error) => { emptyState.hidden = false; emptyState.textContent = error.message; }));
  await load(select.value);
}

function stationDetails(network, index) {
  const station = network?.stations?.[index];
  if (!station) return `<div class="empty"><strong>Station not found</strong></div>`;
  return `<p class="eyebrow">Station ${escapeHtml(index)}</p><h3>${escapeHtml(station.name || station.canonical_id || `Station ${index}`)}</h3><dl><div><dt>Coordinates</dt><dd>${escapeHtml(Number(station.latitude).toFixed(4))}, ${escapeHtml(Number(station.longitude).toFixed(4))}</dd></div><div><dt>Lines</dt><dd>${escapeHtml(station.line_count ?? "—")}</dd></div><div><dt>Daily departures</dt><dd>${escapeHtml(station.daily_departures ?? "—")}</dd></div></dl>`;
}
