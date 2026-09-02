import { api, escapeHtml, formatCount, shortId } from "../../api.js";
import { errorCard, loading, sectionHeading } from "../../components/ui.js";
import { table } from "../../components/table.js";

export async function renderSimilarity(container) {
  container.innerHTML = loading("Loading snapshots…");
  try {
    const snapshots = await api("/api/snapshots");
    if (snapshots.length < 1) {
      container.innerHTML = errorCard(new Error("Similarity needs at least one indexed snapshot."));
      return;
    }
    container.innerHTML = shell(snapshots);
    const querySnapshot = container.querySelector("#similarity-query-snapshot");
    const candidateSnapshot = container.querySelector("#similarity-candidate-snapshot");
    const queryLine = container.querySelector("#similarity-query-line");
    const profile = container.querySelector("#similarity-profile");
    const result = container.querySelector("#similarity-result");

    const populateLines = async () => {
      const network = await api(`/api/snapshots/${encodeURIComponent(querySnapshot.value)}/network`);
      queryLine.innerHTML = (network.lines || []).map((line, index) => `<option value="${index}">${escapeHtml(line.display_name || line.canonical_id || `Line ${index}`)}</option>`).join("");
    };
    querySnapshot.addEventListener("change", () => populateLines().catch((error) => { result.innerHTML = errorCard(error); }));
    container.querySelector("#similarity-form").addEventListener("submit", async (event) => {
      event.preventDefault();
      result.innerHTML = loading("Reading Rust-produced similarity results…");
      const params = new URLSearchParams({ querySnapshotId: querySnapshot.value, candidateSnapshotId: candidateSnapshot.value, queryLineIndex: queryLine.value, profile: profile.value });
      try {
        const data = await api(`/api/similarity?${params}`);
        result.innerHTML = renderResult(data);
      } catch (error) { result.innerHTML = errorCard(error); }
    });
    await populateLines();
  } catch (error) { container.innerHTML = errorCard(error); }
}

function shell(snapshots) {
  const options = snapshots.map((snapshot) => `<option value="${escapeHtml(snapshot.id)}">${escapeHtml(snapshot.sourceName || snapshot.networkId || shortId(snapshot.id))} · ${escapeHtml(snapshot.serviceDate)}</option>`).join("");
  return `<div class="page-intro"><div><p class="eyebrow">Cross-snapshot retrieval</p><h2>Compare by meaning.</h2><p>Similarity facets and measured comparisons are computed by Rust. Studio displays a stored result artifact and never reconstructs embeddings in JavaScript.</p></div><a class="button button-quiet" href="/embeddings" data-route="embeddings">Inspect embeddings →</a></div><section class="card section-card"><form id="similarity-form" class="filters"><div class="field"><label for="similarity-query-snapshot">Query snapshot</label><select class="select" id="similarity-query-snapshot">${options}</select></div><div class="field"><label for="similarity-query-line">Query line</label><select class="select" id="similarity-query-line"><option>Loading lines…</option></select></div><div class="field"><label for="similarity-candidate-snapshot">Candidate snapshot</label><select class="select" id="similarity-candidate-snapshot">${options}</select></div><div class="field"><label for="similarity-profile">Facet profile</label><select class="select" id="similarity-profile"><option value="general">General</option><option value="role">Network role</option><option value="service">Service</option><option value="geometry">Geometry</option><option value="resilience">Resilience</option></select></div><button class="button button-primary" type="submit">Load result</button></form></section><section class="card section-card" id="similarity-result">${sectionHeading("Similarity result", "Select a Rust-produced result artifact to inspect facet scores.", "")}<div class="empty"><strong>No result selected</strong><p>Run the Rust similar-lines command and refresh the index.</p></div></section>`;
}

function renderResult(result) {
  const matches = result.matches || result.results || [];
  const facetNames = ["role", "service", "geometry", "resilience"];
  const rows = matches.map((match) => `<tr><td>${escapeHtml(match.displayName || match.lineName || match.lineInstanceId || "Line")}</td><td class="accent">${Number(match.similarity ?? 0).toFixed(3)}</td>${facetNames.map((name) => `<td>${Number(match.facetScores?.[name] ?? match.facets?.[name] ?? 0).toFixed(3)}</td>`).join("")}<td>${escapeHtml(match.comparison?.mode || "—")}</td></tr>`);
  return `${sectionHeading("Similarity result", `${escapeHtml(result.embeddingSource || "Rust result artifact")} · ${formatCount(matches.length)} matches`, `<span class="mono muted">${escapeHtml(shortId(result.artifactId || ""))}</span>`)}${table({ headers: ["Candidate", "Score", "Role", "Service", "Geometry", "Resilience", "Mode"], rows, emptyTitle: "No matching lines", emptyMessage: "The Rust result contains no candidates for this query." })}`;
}
