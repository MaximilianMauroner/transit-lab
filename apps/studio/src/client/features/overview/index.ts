import { api, escapeHtml, formatCount, shortId } from "../../api.ts";
import { navigate } from "../../routes/router.ts";
import { errorCard, loading, sectionHeading, status, dateCell } from "../../components/ui.ts";
import { table } from "../../components/table.ts";

export async function renderOverview(container) {
  container.innerHTML = loading();
  try {
    const overview = await api("/api/overview");
    const counts = overview.counts || {};
    container.innerHTML = `
      <div class="page-intro"><div><p class="eyebrow">Repository workspace</p><h2>Keep the computation traceable.</h2><p>Rust owns computation and immutable outputs. Studio indexes the manifests, follows run events, and makes the resulting network and model lineage inspectable.</p></div><button class="button button-primary" type="button" data-action="network">Explore network</button></div>
      <div class="grid grid-four">
        ${statCard("Snapshots", counts.snapshots, "compiled service-day views", "network")}
        ${statCard("Runs", counts.runs, "queued and completed jobs", "runs")}
        ${statCard("Models", counts.models, "versioned checkpoints", "data")}
        ${statCard("Artifacts", counts.explicitArtifacts ?? counts.snapshots + counts.models, "indexed immutable outputs", "data")}
      </div>
      <div class="grid grid-two">
        <section class="card section-card">${sectionHeading("Recent snapshots", "Imported from snapshot manifests.", `<a class="section-link" href="/data" data-route="data">All data →</a>`)}${table({ headers: ["Snapshot", "Source", "Service date", "Status"], rows: (overview.snapshots || []).map((snapshot) => `<tr><td><a class="accent mono" href="/network" data-route="network" data-snapshot="${escapeHtml(snapshot.id)}">${escapeHtml(shortId(snapshot.id))}</a></td><td>${escapeHtml(snapshot.sourceName || snapshot.networkId || "Unknown")}</td><td>${escapeHtml(snapshot.serviceDate || "—")}</td><td>${status(snapshot.status)}</td></tr>`) })}</section>
        <section class="card section-card">${sectionHeading("Recent runs", "Events are replayable from the run ledger.", `<a class="section-link" href="/runs" data-route="runs">Open runs →</a>`)}${table({ headers: ["Kind", "Run", "Status", "Created"], rows: (overview.recentRuns || []).map((run) => `<tr><td>${escapeHtml(run.kind)}</td><td class="mono">${escapeHtml(shortId(run.id))}</td><td>${status(run.status)}</td><td>${dateCell(run.createdAt)}</td></tr>`) })}</section>
      </div>
    `;
    container.querySelector('[data-action="network"]')?.addEventListener("click", () => navigate("network"));
  } catch (error) {
    container.innerHTML = errorCard(error);
  }
}

function statCard(label, value, note, route) {
  return `<a class="card stat-card" href="/${route}" data-route="${route}"><span class="stat-label">${escapeHtml(label)}</span><strong class="stat-value">${formatCount(value)}</strong><span class="stat-note">${escapeHtml(note)}</span></a>`;
}
