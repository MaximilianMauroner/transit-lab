import { api, escapeHtml, formatCount, formatDate, shortId } from "../../../../packages/api-client/src/index.ts";
import { empty, sectionHeading } from "../../../../packages/ui/src/index.ts";

export async function renderOverview(container) {
  const catalog = await api("/api/public/catalog");
  const publications = catalog.publications || [];
  if (!publications.length) {
    container.innerHTML = `<section class="card">${empty("No published results", "A Studio owner must publish a result bundle before it appears here.")}</section>`;
    return;
  }
  const overview = await api("/api/public/overview").catch(() => ({ snapshots: [], counts: {} }));
  container.innerHTML = `<section class="intro"><div><p class="eyebrow">Transit Lab Explorer</p><h2>Published transit intelligence.</h2><p>Explore immutable network, model, similarity, and evaluation results selected by the research team. Private runs and unfinished artifacts stay in Studio.</p></div><span class="publication-badge">Read-only publication</span></section>
    <section class="stats"><div><span>Publications</span><strong>${formatCount(publications.length)}</strong></div><div><span>Snapshots</span><strong>${formatCount(overview.counts?.snapshots || 0)}</strong></div><div><span>Models</span><strong>${formatCount(overview.counts?.models || 0)}</strong></div></section>
    <section class="card"><div class="section-head"><div>${sectionHeading("Publication bundles", "Every item below is an intentional published view of Rust-produced artifacts.")}</div></div><div class="publication-grid">${publications.map(publicationCard).join("")}</div></section>`;
}

function publicationCard(publication) {
  return `<article class="publication-card"><p class="eyebrow">${escapeHtml(publication.slug || shortId(publication.id))}</p><h3>${escapeHtml(publication.title)}</h3><p>${formatCount(publication.snapshotIds?.length || 0)} snapshots · ${formatCount(publication.modelIds?.length || 0)} models</p><small>Updated ${escapeHtml(formatDate(publication.updatedAt))}</small></article>`;
}
