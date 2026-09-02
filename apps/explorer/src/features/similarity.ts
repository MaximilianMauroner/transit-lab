import { api, escapeHtml, formatCount } from "../../../../packages/api-client/src/index.ts";
import { errorCard, loading, sectionHeading } from "../../../../packages/ui/src/index.ts";
import { table } from "../../../../packages/ui/src/table.ts";

export async function renderSimilarity(container: HTMLElement) {
  const snapshots = await api("/api/public/snapshots");
  if (!snapshots.length) { container.innerHTML = errorCard(new Error("No published snapshots are available.")); return; }
  container.innerHTML = `<section class="intro"><div><p class="eyebrow">Published representation</p><h2>Similarity</h2><p>Facet scores from Rust-produced result artifacts. No vectors are reconstructed in the browser.</p></div></section><section class="card"><form id="similarity-form" class="filters"><label>Query snapshot<select class="select" id="query-snapshot">${snapshotOptions(snapshots)}</select></label><label>Query line<select class="select" id="query-line"><option>Loading lines…</option></select></label><label>Candidate snapshot<select class="select" id="candidate-snapshot">${snapshotOptions(snapshots)}</select></label><label>Facet<select class="select" id="profile"><option value="general">General</option><option value="role">Network role</option><option value="service">Service</option><option value="geometry">Geometry</option><option value="resilience">Resilience</option></select></label><button class="button" type="submit">Load result</button></form></section><section class="card" id="similarity-result">${sectionHeading("Similarity result", "Select a published result artifact to inspect.")}</section>`;
  const querySnapshot = container.querySelector<HTMLSelectElement>("#query-snapshot");
  const queryLine = container.querySelector<HTMLSelectElement>("#query-line");
  const candidateSnapshot = container.querySelector<HTMLSelectElement>("#candidate-snapshot");
  const profile = container.querySelector<HTMLSelectElement>("#profile");
  const result = container.querySelector<HTMLElement>("#similarity-result");
  const populateLines = async () => {
    const network = await api(`/api/public/snapshots/${encodeURIComponent(querySnapshot.value)}/network`);
    queryLine.innerHTML = (network.lines || []).map((line, index) => `<option value="${index}">${escapeHtml(line.display_name || line.canonical_id || `Line ${index}`)}</option>`).join("");
  };
  querySnapshot.addEventListener("change", () => populateLines().catch((error) => { result.innerHTML = errorCard(error); }));
  container.querySelector<HTMLFormElement>("#similarity-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    result.innerHTML = loading("Loading published similarity…");
    try {
      const params = new URLSearchParams({ querySnapshotId: querySnapshot.value, candidateSnapshotId: candidateSnapshot.value, queryLineIndex: queryLine.value, profile: profile.value });
      const data = await api(`/api/public/similarity?${params}`);
      const matches = data.matches || data.results || [];
      const rows = matches.map((match) => `<tr><td>${escapeHtml(match.displayName || match.lineName || match.lineInstanceId || "Line")}</td><td class="accent">${Number(match.similarity || 0).toFixed(3)}</td><td>${Number(match.facetScores?.role ?? 0).toFixed(3)}</td><td>${Number(match.facetScores?.service ?? 0).toFixed(3)}</td><td>${Number(match.facetScores?.geometry ?? 0).toFixed(3)}</td><td>${Number(match.facetScores?.resilience ?? 0).toFixed(3)}</td></tr>`);
      result.innerHTML = `${sectionHeading("Similarity result", `${formatCount(matches.length)} published matches`, "")}${table({ headers: ["Candidate", "Score", "Role", "Service", "Geometry", "Resilience"], rows })}`;
    } catch (error) { result.innerHTML = errorCard(error); }
  });
  await populateLines();
}

function snapshotOptions(snapshots) {
  return snapshots.map((snapshot) => `<option value="${escapeHtml(snapshot.id)}">${escapeHtml(snapshot.sourceName || snapshot.networkId || snapshot.id)} · ${escapeHtml(snapshot.serviceDate)}</option>`).join("");
}
