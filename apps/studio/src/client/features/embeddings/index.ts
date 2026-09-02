import { api, escapeHtml, formatCount, shortId } from "../../api.ts";
import { errorCard, loading, sectionHeading } from "../../components/ui.ts";
import { table } from "../../components/table.ts";

export async function renderEmbeddings(container) {
  container.innerHTML = loading();
  try {
    const artifacts = await api("/api/embeddings");
    container.innerHTML = `<div class="page-intro"><div><p class="eyebrow">Representation space</p><h2>Embeddings with lineage.</h2><p>Projection files and embedding artifacts are displayable here once Rust emits their manifests. The browser does not generate or alter vectors.</p></div><span class="read-only-badge"><span class="live-dot"></span> ${formatCount(artifacts.length)} indexed artifacts</span></div><section class="card section-card">${sectionHeading("Embedding artifacts", "Projection, facet, and base representation outputs.", "")}${table({ headers: ["Kind", "Artifact", "Schema", "Files", "Status"], rows: artifacts.map((artifact) => `<tr><td>${escapeHtml(artifact.kind)}</td><td class="mono">${escapeHtml(shortId(artifact.id))}</td><td>${escapeHtml(artifact.schemaVersion ?? "legacy")}</td><td>${formatCount(artifact.files?.length || 0)}</td><td>${escapeHtml(artifact.status)}</td></tr>`), emptyTitle: "No embedding artifacts", emptyMessage: "Produce a Rust embedding or projection artifact, then refresh the index." })}</section>`;
  } catch (error) { container.innerHTML = errorCard(error); }
}
