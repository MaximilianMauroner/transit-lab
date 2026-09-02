import {
  formatMetricValue,
  metricLabel,
  metricValue,
  normalizePredictionFile,
  primaryMetricName
} from "./normalize.js";

const elements = {
  apiForm: document.querySelector("#api-form"),
  apiUrl: document.querySelector("#api-url"),
  dashboard: document.querySelector("#dashboard"),
  dialog: document.querySelector("#line-dialog"),
  dialogClose: document.querySelector("#dialog-close"),
  dialogMetrics: document.querySelector("#dialog-metrics"),
  dialogSubtitle: document.querySelector("#dialog-subtitle"),
  dialogTitle: document.querySelector("#dialog-title"),
  emptyMessage: document.querySelector("#empty-message"),
  emptyState: document.querySelector("#empty-state"),
  emptyTitle: document.querySelector("#empty-title"),
  errorMessage: document.querySelector("#error-message"),
  errorState: document.querySelector("#error-state"),
  fileInput: document.querySelector("#file-input"),
  fileName: document.querySelector("#file-name"),
  impactFilter: document.querySelector("#impact-filter"),
  listContainer: document.querySelector("#list-container"),
  loadingMessage: document.querySelector("#loading-message"),
  loadingState: document.querySelector("#loading-state"),
  predictionList: document.querySelector("#prediction-list"),
  resultCount: document.querySelector("#result-count"),
  resultsPanel: document.querySelector("#results-panel"),
  searchInput: document.querySelector("#search-input"),
  snapshotId: document.querySelector("#snapshot-id"),
  sourceMessage: document.querySelector("#source-message"),
  sourceStatus: document.querySelector("#source-status"),
  sortSelect: document.querySelector("#sort-select"),
  statAverage: document.querySelector("#stat-average"),
  statHighest: document.querySelector("#stat-highest"),
  statHighestNote: document.querySelector("#stat-highest-note"),
  statLines: document.querySelector("#stat-lines"),
  statLinesNote: document.querySelector("#stat-lines-note")
};

const state = {
  dataset: null,
  error: "",
  loading: false,
  source: "",
  search: "",
  impactFilter: "all",
  sortBy: "accessibility_auc_loss"
};

const apiStorageKey = "transit-lab-web-wrapper.api-url";

function setHidden(element, hidden) {
  element.hidden = hidden;
}

function showLoading(message) {
  state.loading = true;
  state.error = "";
  elements.loadingMessage.textContent = message;
  renderState();
}

function showError(message) {
  state.loading = false;
  state.error = message;
  renderState();
}

function saveApiUrl(value) {
  try {
    localStorage.setItem(apiStorageKey, value);
  } catch {
    // Storage may be disabled in private browsing. Loading still works.
  }
}

function loadSavedApiUrl() {
  try {
    const saved = localStorage.getItem(apiStorageKey);
    if (saved) elements.apiUrl.value = saved;
  } catch {
    // Storage may be disabled in private browsing.
  }
}

function loadDataset(raw, source) {
  try {
    state.dataset = normalizePredictionFile(raw);
    state.source = source;
    state.error = "";
    state.loading = false;
    state.search = "";
    state.impactFilter = "all";
    state.sortBy = "accessibility_auc_loss";
    elements.searchInput.value = "";
    elements.impactFilter.value = state.impactFilter;
    elements.sortSelect.value = state.sortBy;
    render();
  } catch (error) {
    showError(error instanceof Error ? error.message : "The prediction file is invalid.");
  }
}

async function loadFile(file) {
  if (!file) return;
  showLoading(`Reading ${file.name}.`);
  try {
    const raw = JSON.parse(await file.text());
    loadDataset(raw, `Local file · ${file.name}`);
    elements.fileName.textContent = file.name;
  } catch (error) {
    showError(error instanceof Error ? error.message : "The selected file is not valid JSON.");
  }
}

async function loadApi(url) {
  const trimmedUrl = url.trim();
  if (!trimmedUrl) {
    showError("Enter an API URL before loading.");
    return;
  }

  let parsedUrl;
  try {
    parsedUrl = new URL(trimmedUrl, window.location.href);
  } catch {
    showError("Enter a valid HTTP or HTTPS API URL.");
    return;
  }

  if (!["http:", "https:"].includes(parsedUrl.protocol)) {
    showError("The API URL must use HTTP or HTTPS.");
    return;
  }

  saveApiUrl(trimmedUrl);
  showLoading(`Fetching ${parsedUrl.href}.`);
  try {
    const response = await fetch(parsedUrl.href, {
      headers: { Accept: "application/json" }
    });
    if (!response.ok) {
      throw new Error(`The API returned HTTP ${response.status}.`);
    }
    loadDataset(await response.json(), `API · ${parsedUrl.href}`);
  } catch (error) {
    showError(
      error instanceof Error
        ? `${error.message} Check the URL and CORS settings.`
        : "The API response could not be loaded."
    );
  }
}

function getPrimaryName() {
  return primaryMetricName(state.dataset?.metricNames || []) || "accessibility_auc_loss";
}

function getPrimaryValue(prediction) {
  return metricValue(prediction, getPrimaryName()) ?? 0;
}

function getAverageImpact() {
  const predictions = state.dataset?.predictions || [];
  if (!predictions.length) return 0;
  return predictions.reduce((sum, prediction) => sum + getPrimaryValue(prediction), 0) /
    predictions.length;
}

function getSortedPredictions() {
  if (!state.dataset) return [];

  const averageImpact = getAverageImpact();
  const searchTerm = state.search.trim().toLowerCase();
  const filtered = state.dataset.predictions.filter((prediction) => {
    const matchesSearch = !searchTerm ||
      prediction.label.toLowerCase().includes(searchTerm) ||
      prediction.lineId.toLowerCase().includes(searchTerm);
    const impact = getPrimaryValue(prediction);
    const matchesImpact = state.impactFilter === "all" ||
      (state.impactFilter === "positive" && impact > 0) ||
      (state.impactFilter === "above-average" && impact >= averageImpact);
    return matchesSearch && matchesImpact;
  });

  return filtered.sort((left, right) => {
    if (state.sortBy === "line") {
      const numericLeft = Number(left.lineId);
      const numericRight = Number(right.lineId);
      if (Number.isFinite(numericLeft) && Number.isFinite(numericRight)) {
        return numericLeft - numericRight;
      }
      return left.lineId.localeCompare(right.lineId, undefined, { numeric: true });
    }

    const leftValue = state.sortBy === "structural_uniqueness"
      ? left.structuralUniqueness
      : metricValue(left, state.sortBy) ?? 0;
    const rightValue = state.sortBy === "structural_uniqueness"
      ? right.structuralUniqueness
      : metricValue(right, state.sortBy) ?? 0;
    return rightValue - leftValue || left.label.localeCompare(right.label);
  });
}

function makeElement(tagName, className, text) {
  const element = document.createElement(tagName);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function makeMetricText(name, value) {
  return formatMetricValue(name, value);
}

function makePredictionCard(prediction, rank) {
  const item = makeElement("li", "prediction-item");
  const button = makeElement("button", "prediction-card");
  button.type = "button";
  button.setAttribute(
    "aria-label",
    `${prediction.label}, ${makeMetricText(getPrimaryName(), getPrimaryValue(prediction))} accessibility loss. Open details.`
  );
  button.addEventListener("click", () => openDetails(prediction));

  const rankBadge = makeElement("span", "rank-badge", String(rank).padStart(2, "0"));
  rankBadge.setAttribute("aria-hidden", "true");

  const body = makeElement("span", "prediction-card-body");
  const title = makeElement("span", "prediction-title-row");
  title.append(
    makeElement("span", "line-name", prediction.label),
    makeElement("span", "line-id", `ID ${prediction.lineId}`)
  );

  const impactRow = makeElement("span", "impact-row");
  const impact = makeElement("span", "impact-value", makeMetricText(getPrimaryName(), getPrimaryValue(prediction)));
  const impactLabel = makeElement("span", "impact-label", "accessibility loss");
  impactRow.append(impact, impactLabel);

  const meter = makeElement("span", "metric-meter");
  meter.setAttribute("aria-hidden", "true");
  const fill = makeElement("span", "metric-meter-fill");
  fill.style.width = `${Math.min(100, Math.max(0, getPrimaryValue(prediction) * 100))}%`;
  meter.append(fill);

  body.append(title, impactRow, meter);

  const side = makeElement("span", "prediction-card-side");
  side.append(
    makeElement("span", "side-label", "Structural uniqueness"),
    makeElement("strong", "side-value", formatMetricValue("structural_uniqueness", prediction.structuralUniqueness)),
    makeElement("span", "chevron", "›")
  );

  button.append(rankBadge, body, side);
  item.append(button);
  return item;
}

function renderList() {
  elements.predictionList.replaceChildren();
  if (!state.dataset) {
    elements.listContainer.hidden = true;
    elements.emptyState.hidden = true;
    return;
  }

  const predictions = getSortedPredictions();
  const total = state.dataset.predictions.length;
  elements.resultCount.textContent = `${predictions.length} of ${total} ${total === 1 ? "line" : "lines"}`;

  if (!total) {
    elements.listContainer.hidden = true;
    elements.emptyTitle.textContent = "This prediction run has no lines";
    elements.emptyMessage.textContent = "Load a run with at least one line prediction to begin exploring.";
    elements.emptyState.hidden = false;
    return;
  }

  if (!predictions.length) {
    elements.listContainer.hidden = true;
    elements.emptyTitle.textContent = "No lines match these filters";
    elements.emptyMessage.textContent = "Try a different search or show all lines.";
    elements.emptyState.hidden = false;
    return;
  }

  elements.listContainer.hidden = false;
  elements.emptyState.hidden = true;
  predictions.forEach((prediction, index) => {
    elements.predictionList.append(makePredictionCard(prediction, index + 1));
  });
}

function renderDashboard() {
  const dataset = state.dataset;
  setHidden(elements.dashboard, !dataset);
  if (!dataset) return;

  const predictions = dataset.predictions;
  const sorted = [...predictions].sort((left, right) => getPrimaryValue(right) - getPrimaryValue(left));
  const highest = sorted[0];
  const snapshotId = dataset.snapshotId;

  elements.snapshotId.textContent = snapshotId.length > 18
    ? `${snapshotId.slice(0, 8)}…${snapshotId.slice(-6)}`
    : snapshotId;
  elements.snapshotId.title = snapshotId;
  elements.statLines.textContent = String(predictions.length);
  elements.statLinesNote.textContent = state.source || "Loaded prediction run";
  elements.statAverage.textContent = formatMetricValue(getPrimaryName(), getAverageImpact());
  elements.statHighest.textContent = highest
    ? formatMetricValue(getPrimaryName(), getPrimaryValue(highest))
    : "—";
  elements.statHighestNote.textContent = highest ? highest.label : "No line predictions";
}

function renderState() {
  const hasDataset = Boolean(state.dataset);
  setHidden(elements.resultsPanel, !hasDataset && !state.loading && !state.error);
  setHidden(elements.loadingState, !state.loading);
  setHidden(elements.errorState, !state.error);
  elements.errorMessage.textContent = state.error;
  elements.sourceStatus.textContent = state.loading
    ? "Loading"
    : state.error
      ? "Needs attention"
      : hasDataset
        ? "Loaded"
        : "Waiting for data";
  elements.sourceStatus.className = `status-pill${state.error ? " status-pill-error" : ""}`;
}

function render() {
  renderState();
  renderDashboard();
  renderList();
}

function openDetails(prediction) {
  elements.dialogTitle.textContent = prediction.label;
  elements.dialogSubtitle.textContent = `Line ID ${prediction.lineId} · predicted intervention impact`;
  elements.dialogMetrics.replaceChildren();

  state.dataset.metricNames.forEach((name) => {
    const metric = makeElement("div", "detail-metric");
    metric.append(
      makeElement("dt", "detail-label", metricLabel(name)),
      makeElement("dd", "detail-value", makeMetricText(name, metricValue(prediction, name)))
    );
    elements.dialogMetrics.append(metric);
  });

  const uniqueness = makeElement("div", "detail-metric detail-metric-highlight");
  uniqueness.append(
    makeElement("dt", "detail-label", "Structural uniqueness"),
    makeElement("dd", "detail-value", formatMetricValue("structural_uniqueness", prediction.structuralUniqueness))
  );
  elements.dialogMetrics.append(uniqueness);

  if (typeof elements.dialog.showModal === "function") {
    elements.dialog.showModal();
  } else {
    elements.dialog.setAttribute("open", "");
  }
}

elements.fileInput.addEventListener("change", (event) => loadFile(event.target.files?.[0]));
elements.apiForm.addEventListener("submit", (event) => {
  event.preventDefault();
  loadApi(elements.apiUrl.value);
});
elements.searchInput.addEventListener("input", (event) => {
  state.search = event.target.value;
  renderList();
});
elements.sortSelect.addEventListener("change", (event) => {
  state.sortBy = event.target.value;
  renderList();
});
elements.impactFilter.addEventListener("change", (event) => {
  state.impactFilter = event.target.value;
  renderList();
});
elements.dialogClose.addEventListener("click", () => elements.dialog.close());
elements.dialog.addEventListener("click", (event) => {
  if (event.target === elements.dialog) elements.dialog.close();
});

loadSavedApiUrl();
render();

const apiFromQuery = new URLSearchParams(window.location.search).get("api");
if (apiFromQuery) {
  elements.apiUrl.value = apiFromQuery;
  loadApi(apiFromQuery);
}
