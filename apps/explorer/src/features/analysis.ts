import { api, escapeHtml, formatCount, shortId } from "../../../../packages/api-client/src/index.ts";
import { errorCard, loading, sectionHeading } from "../../../../packages/ui/src/index.ts";
import { table } from "../../../../packages/ui/src/table.ts";

export async function renderEmbeddings(container) {
  await renderArtifacts(container, "/api/public/embeddings", "Embeddings", "Published embedding and projection artifacts.");
}

export async function renderEvaluation(container) {
  await renderArtifacts(container, "/api/public/evaluations", "Evaluation", "Published evaluation summaries and quality signals.");
}

async function renderArtifacts(container: HTMLElement, path: string, title: string, copy: string) {
  container.innerHTML = loading("Loading published analysis…");
  const records = await api(path);
  if (!records.length) { container.innerHTML = errorCard(new Error(`No published ${title.toLowerCase()} results are available.`)); return; }
  const rows = records.map((record) => `<tr><td>${escapeHtml(record.kind || record.metricName || "Result")}</td><td>${escapeHtml(shortId(record.id || record.modelId || ""))}</td><td>${escapeHtml(record.facet || record.split || record.status || "ready")}</td><td class="accent">${record.value === undefined ? "Published" : Number(record.value).toFixed(4)}</td></tr>`);
  container.innerHTML = `<section class="intro"><div><p class="eyebrow">Published analysis</p><h2>${escapeHtml(title)}</h2><p>${escapeHtml(copy)}</p></div></section><section class="card">${sectionHeading(title, `${formatCount(records.length)} published records`, "")}${table({ headers: ["Result", "Reference", "Facet / split", "Value"], rows })}</section>`;
}
