import { api, escapeHtml, formatCount, shortId } from "../../api.ts";
import { errorCard, loading, sectionHeading, status, dateCell } from "../../components/ui.ts";
import { table } from "../../components/table.ts";

export async function renderData(container) {
  container.innerHTML = loading();
  try {
    const [snapshots, artifacts, datasets, models] = await Promise.all([
      api("/api/snapshots"), api("/api/artifacts?limit=250"), api("/api/datasets"), api("/api/models")
    ]);
    container.innerHTML = `
      <div class="page-intro"><div><p class="eyebrow">Manifests &amp; lineage</p><h2>Data has an address.</h2><p>Every indexed item points back to a versioned Rust output. Directory names are presentation only; relationships come from manifest inputs.</p></div><button class="button button-quiet" id="refresh-inventory" type="button">Refresh from disk</button></div>
      <div class="grid grid-four">
        ${miniStat("Snapshots", snapshots.length)}${miniStat("Artifacts", artifacts.length)}${miniStat("Datasets", datasets.length)}${miniStat("Models", models.length)}
      </div>
      <section class="card section-card">${sectionHeading("Snapshots", "Compiled network snapshots available to the Studio.", "")}${table({ headers: ["Source", "Snapshot", "Date", "Entities", "Status"], rows: snapshots.map((snapshot) => `<tr><td>${escapeHtml(snapshot.sourceName || snapshot.networkId || "Unknown")}</td><td class="mono">${escapeHtml(shortId(snapshot.id))}</td><td>${escapeHtml(snapshot.serviceDate)}</td><td>${formatCount(snapshot.counts?.stations || 0)} stations · ${formatCount(snapshot.counts?.lines || 0)} lines</td><td>${status(snapshot.status)}</td></tr>`) })}</section>
      <section class="card section-card">${sectionHeading("Artifacts", "Explicit v1 manifests and their provenance fields.", "")}${table({ headers: ["Kind", "Artifact", "Schema", "SHA-256", "Created"], rows: artifacts.map((artifact) => `<tr><td>${escapeHtml(artifact.kind)}</td><td class="mono">${escapeHtml(shortId(artifact.id))}</td><td>${escapeHtml(artifact.schemaVersion ?? "legacy")}</td><td class="mono muted">${escapeHtml(shortId(artifact.sha256, 10, 8))}</td><td>${dateCell(artifact.createdAt)}</td></tr>`), emptyTitle: "No explicit artifact manifests", emptyMessage: "Rust output manifests will appear here after the next run." })}</section>
    `;
    container.querySelector("#refresh-inventory")?.addEventListener("click", async (event) => {
      event.currentTarget.disabled = true;
      try { await api("/api/inventory/refresh", { method: "POST" }); await renderData(container); }
      catch (error) { container.innerHTML = errorCard(error); }
    });
  } catch (error) { container.innerHTML = errorCard(error); }
}

function miniStat(label, value) { return `<div class="card stat-card"><span class="stat-label">${escapeHtml(label)}</span><strong class="stat-value">${formatCount(value)}</strong></div>`; }
