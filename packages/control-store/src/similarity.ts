import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { all, one, parseJson } from "./database.ts";

/**
 * Similarity is a Rust-owned computation. Studio only selects and returns a
 * versioned result artifact; it deliberately has no feature-vector fallback.
 */
export async function loadSimilarityResult({ db, root, querySnapshotId, candidateSnapshotId, queryLineId, queryLineIndex, profile, weights, topK }) {
  const artifacts = all(db, `SELECT * FROM artifacts WHERE kind = 'similarity-result' AND status = 'ready' ORDER BY created_at DESC`);
  for (const artifact of artifacts) {
    const metadata = parseJson(artifact.metadata_json);
    if (metadata.querySnapshotId && metadata.querySnapshotId !== querySnapshotId) continue;
    if (metadata.candidateSnapshotId && metadata.candidateSnapshotId !== candidateSnapshotId) continue;
    if (metadata.profile && metadata.profile !== profile) continue;
    const path = artifact.local_path ? resolve(root, artifact.local_path) : null;
    if (!path) continue;
    try {
      const result = JSON.parse(await readFile(path, "utf8"));
      const query = result.query || {};
      const lineMatches = queryLineId
        ? query.lineInstanceId === queryLineId || query.lineId === queryLineId
        : queryLineIndex === null || queryLineIndex === undefined || Number(query.lineIndex ?? query.line) === Number(queryLineIndex);
      if (lineMatches) return { ...result, artifactId: artifact.id };
    } catch {
      // A malformed result is not a valid Studio source; keep looking.
    }
  }
  return null;
}
