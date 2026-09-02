import { api, escapeHtml, formatCount, shortId } from "../../../../packages/api-client/src/index.ts";
import { errorCard, loading, sectionHeading } from "../../../../packages/ui/src/index.ts";
import { table } from "../../../../packages/ui/src/table.ts";

export async function renderCriticality(container: HTMLElement) {
  container.innerHTML = loading("Loading published snapshots…");
  const snapshots = await api("/api/public/snapshots");
  if (!snapshots.length) { container.innerHTML = errorCard(new Error("No published snapshots are available.")); return; }
  container.innerHTML = `<section class="intro"><div><p class="eyebrow">Published model output</p><h2>Line criticality</h2><p>Rust-produced scores and percentiles, ranked for inspection.</p></div><select class="select" id="criticality-snapshot">${snapshots.map((snapshot) => `<option value="${escapeHtml(snapshot.id)}">${escapeHtml(snapshot.sourceName || snapshot.networkId || shortId(snapshot.id))} · ${escapeHtml(snapshot.serviceDate)}</option>`).join("")}</select></section><section class="card" id="criticality-result"></section>`;
  const select = container.querySelector<HTMLSelectElement>("#criticality-snapshot");
  const result = container.querySelector<HTMLElement>("#criticality-result");
  const load = async () => {
    result.innerHTML = loading("Loading Rust-produced criticality…");
    const data = await api(`/api/public/criticality?snapshotId=${encodeURIComponent(select.value)}`);
    const rows = (data.predictions || []).map((prediction) => `<tr><td>${escapeHtml(prediction.lineName || prediction.line)}</td><td class="accent">${Number(prediction.metrics?.[0] || 0).toFixed(3)}</td><td>${Number(prediction.metricPercentiles?.[0] || 0).toFixed(3)}</td><td>${Number(prediction.structuralUniqueness || 0).toFixed(3)}</td><td>${Number(prediction.uncertainty || 0).toFixed(3)}</td></tr>`);
    result.innerHTML = `${sectionHeading("Criticality ranking", `${formatCount(rows.length)} published line predictions`, `<span class="mono muted">${escapeHtml(shortId(data.modelId))}</span>`)}${table({ headers: ["Line", data.metricNames?.[0] || "Primary score", "Percentile", "Structural uniqueness", "Uncertainty"], rows, emptyTitle: "No predictions", emptyMessage: "This publication does not include line predictions." })}`;
  };
  select.addEventListener("change", () => load().catch((error) => { result.innerHTML = errorCard(error); }));
  await load();
}
