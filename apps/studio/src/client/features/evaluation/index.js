import { api, escapeHtml, formatCount } from "../../api.js";
import { errorCard, loading, sectionHeading } from "../../components/ui.js";
import { table } from "../../components/table.js";

export async function renderEvaluation(container) {
  container.innerHTML = loading();
  try {
    const rows = await api("/api/evaluations");
    container.innerHTML = `<div class="page-intro"><div><p class="eyebrow">Quality &amp; evaluation</p><h2>Measure before you trust.</h2><p>Evaluation values are recorded by Rust runs and displayed with their model, dataset, facet, and split lineage.</p></div><span class="read-only-badge"><span class="live-dot"></span> ${formatCount(rows.length)} recorded points</span></div><section class="card section-card">${sectionHeading("Evaluation points", "No client-side metric computation.", "")}${table({ headers: ["Facet", "Metric", "Value", "Model", "Dataset", "Split"], rows: rows.map((row) => `<tr><td>${escapeHtml(row.facet)}</td><td>${escapeHtml(row.metricName)}</td><td class="accent">${Number(row.value).toFixed(4)}</td><td>${escapeHtml(row.modelVersion || row.modelId || "—")}</td><td>${escapeHtml(row.datasetId || "—")}</td><td>${escapeHtml(row.split || "—")}</td></tr>`), emptyTitle: "No evaluation points", emptyMessage: "Run an evaluation command and publish its versioned result artifact." })}</section>`;
  } catch (error) { container.innerHTML = errorCard(error); }
}
