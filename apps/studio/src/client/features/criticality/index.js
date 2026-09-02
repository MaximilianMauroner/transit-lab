import { api, escapeHtml, formatCount, shortId } from "../../api.js";
import { normalizePredictionFile, formatMetricValue, metricLabel, metricValue, primaryMetricName } from "./normalize.js";
import { errorCard, loading, sectionHeading, status } from "../../components/ui.js";
import { table } from "../../components/table.js";

export async function renderCriticality(container) {
  container.innerHTML = loading("Loading inference sets…");
  try {
    const inferences = await api("/api/inferences");
    if (!inferences.length) {
      container.innerHTML = errorCard(new Error("No versioned inference result is indexed yet. Run infer criticality from the Runs view."));
      return;
    }
    container.innerHTML = shell(inferences);
    const select = container.querySelector("#criticality-inference");
    const search = container.querySelector("#criticality-search");
    const sort = container.querySelector("#criticality-sort");
    const body = container.querySelector("#criticality-table-body");
    const summary = container.querySelector("#criticality-summary");
    let dataset = null;

    const renderRows = () => {
      if (!dataset) return;
      const query = search.value.trim().toLowerCase();
      const sortName = sort.value;
      const rows = dataset.predictions.filter((prediction) => !query || prediction.label.toLowerCase().includes(query) || prediction.lineId.toLowerCase().includes(query));
      rows.sort((left, right) => {
        if (sortName === "line") return left.lineId.localeCompare(right.lineId, undefined, { numeric: true });
        const leftValue = sortName === "structural_uniqueness" ? left.structuralUniqueness : metricValue(left, sortName) ?? 0;
        const rightValue = sortName === "structural_uniqueness" ? right.structuralUniqueness : metricValue(right, sortName) ?? 0;
        return rightValue - leftValue || left.label.localeCompare(right.label);
      });
      const primary = primaryMetricName(dataset.metricNames);
      body.innerHTML = rows.map((prediction, index) => `<tr><td class="mono muted">${String(index + 1).padStart(2, "0")}</td><td>${escapeHtml(prediction.label)} <span class="mono muted">#${escapeHtml(prediction.lineId)}</span></td><td class="accent">${escapeHtml(formatMetricValue(primary, metricValue(prediction, primary)))}</td><td>${escapeHtml(formatMetricValue("unreachable_share", metricValue(prediction, "unreachable_share")))}</td><td>${escapeHtml(formatMetricValue("mean_delay_reachable_seconds", metricValue(prediction, "mean_delay_reachable_seconds")))}</td><td>${escapeHtml(formatMetricValue("structural_uniqueness", prediction.structuralUniqueness))}</td></tr>`).join("") || `<tr><td colspan="6">${table({ headers: [], rows: [], emptyTitle: "No lines match", emptyMessage: "Try a different search term." })}</td></tr>`;
      summary.textContent = `${formatCount(rows.length)} of ${formatCount(dataset.predictions.length)} lines · sorted by ${metricLabel(sortName)}`;
    };

    const load = async (inferenceId) => {
      summary.textContent = "Loading Rust-produced predictions…";
      try {
        dataset = normalizePredictionFile(await api(`/api/criticality?inferenceId=${encodeURIComponent(inferenceId)}`));
        const inference = inferences.find((item) => item.id === inferenceId);
        container.querySelector("#criticality-model").textContent = `${shortId(inference?.modelId)} · snapshot ${shortId(inference?.snapshotId)}`;
        container.querySelector("#criticality-count").textContent = formatCount(dataset.predictions.length);
        container.querySelector("#criticality-metrics").textContent = formatCount(dataset.metricNames.length);
        search.value = "";
        sort.value = primaryMetricName(dataset.metricNames) || "line";
        renderRows();
      } catch (error) {
        dataset = null;
        summary.textContent = error.message;
        body.replaceChildren();
      }
    };
    select.addEventListener("change", () => load(select.value));
    search.addEventListener("input", renderRows);
    sort.addEventListener("change", renderRows);
    await load(select.value);
  } catch (error) { container.innerHTML = errorCard(error); }
}

function shell(inferences) {
  const options = inferences.map((inference) => `<option value="${escapeHtml(inference.id)}">${escapeHtml(shortId(inference.snapshotId))} · ${escapeHtml(shortId(inference.modelId))}</option>`).join("");
  return `<div class="page-intro"><div><p class="eyebrow">Inference inspection</p><h2>Where a disruption matters.</h2><p>Predictions, metric percentiles, and structural scores come from Rust inference artifacts. Studio only filters, sorts, and formats the result.</p></div><a class="button button-primary" href="/runs" data-route="runs">Submit a run</a></div><div class="grid grid-three"><div class="card stat-card"><span class="stat-label">Selected model / snapshot</span><strong class="stat-value" id="criticality-model" style="font-size:17px">—</strong><span class="stat-note">Versioned lineage</span></div><div class="card stat-card"><span class="stat-label">Predicted lines</span><strong class="stat-value" id="criticality-count">—</strong><span class="stat-note">Rows in the Rust result</span></div><div class="card stat-card"><span class="stat-label">Metrics</span><strong class="stat-value" id="criticality-metrics">—</strong><span class="stat-note">Named output dimensions</span></div></div><section class="card section-card"><div class="filters"><div class="field"><label for="criticality-inference">Inference result</label><select class="select" id="criticality-inference">${options}</select></div><div class="field grow"><label for="criticality-search">Search lines</label><input class="input" id="criticality-search" type="search" placeholder="Line number or name" /></div><div class="field"><label for="criticality-sort">Sort by</label><select class="select" id="criticality-sort"><option value="accessibility_auc_loss">Accessibility loss</option><option value="unreachable_share">Unreachable share</option><option value="mean_delay_reachable_seconds">Mean delay</option><option value="structural_uniqueness">Structural uniqueness</option><option value="line">Line</option></select></div></div>${sectionHeading("Ranked predictions", "The primary metric is selected by its name, never by a UI-side recalculation.", "")}<p id="criticality-summary" class="muted" style="margin:14px 20px 0;font-size:11px">Loading…</p><div class="table-wrap"><table class="data-table"><thead><tr><th>Rank</th><th>Line</th><th>Accessibility loss</th><th>Unreachable</th><th>Mean delay</th><th>Structural uniqueness</th></tr></thead><tbody id="criticality-table-body"></tbody></table></div></section>`;
}
