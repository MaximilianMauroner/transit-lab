import { profileWeights } from "../../../packages/contracts/src/index.js";
import { all, parseJson } from "./db.js";

const MODE_NAMES = {
  0: "tram",
  1: "metro",
  2: "rail",
  3: "bus",
  4: "other"
};

function number(value, fallback = 0) {
  const result = Number(value);
  return Number.isFinite(result) ? result : fallback;
}

function feature(line) {
  return parseJson(line.feature_json, {});
}

function criticality(line) {
  return parseJson(line.criticality_json, {});
}

function scaled(value, scale) {
  return Math.max(0, Math.min(1, number(value) / scale));
}

function modeSignal(line) {
  const mode = number(line.mode ?? feature(line).mode);
  return [mode === 0 ? 1 : 0, mode === 1 ? 1 : 0, mode === 2 ? 1 : 0, mode === 3 ? 1 : 0, mode === 4 ? 1 : 0];
}

/**
 * Build deliberately interpretable fallback facet vectors from stored GTFS
 * features. If a real embedding artifact is available, the same response
 * shape can be upgraded without changing the API contract.
 */
export function facetVector(line, facet) {
  const f = feature(line);
  const c = criticality(line);
  const mode = modeSignal(line);
  const service = [
    ...mode,
    scaled(f.service_span_seconds, 86_400),
    scaled(Math.log1p(number(f.daily_trip_count)), 12),
    1 - scaled(f.median_headway_seconds, 7_200),
    1 - scaled(f.peak_headway_seconds, 7_200),
    1 - scaled(f.off_peak_headway_seconds, 7_200)
  ];
  const geometry = [
    ...mode,
    scaled(f.station_count, 100),
    scaled(f.pattern_count, 20),
    scaled(f.route_length_metres, 100_000),
    scaled(f.end_to_end_distance_metres, 100_000),
    scaled(f.branching_factor, 8),
    scaled(f.unique_station_fraction, 1),
    scaled(f.shared_segment_fraction, 1)
  ];
  const role = [
    ...mode,
    scaled(f.transfer_station_count, 100),
    scaled(f.unique_station_fraction, 1),
    1 - scaled(f.shared_segment_fraction, 1),
    scaled(f.pattern_count, 20),
    scaled(f.branching_factor, 8),
    scaled(f.station_count, 100)
  ];
  const resilience = [
    ...mode,
    scaled(f.unique_station_fraction, 1),
    1 - scaled(f.shared_segment_fraction, 1),
    scaled(f.transfer_station_count, 100),
    scaled(c.accessibility_auc_loss, 1),
    scaled(c.unreachable_share, 1),
    scaled(c.stations_losing_all_service_share, 1)
  ];
  if (facet === "service") return service;
  if (facet === "geometry") return geometry;
  if (facet === "resilience") return resilience;
  if (facet === "role") return role;
  return [...role, ...service.slice(5), ...geometry.slice(5), ...resilience.slice(5)];
}

export function cosineSimilarity(left, right) {
  const length = Math.min(left.length, right.length);
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;
  for (let index = 0; index < length; index += 1) {
    dot += left[index] * right[index];
    leftNorm += left[index] ** 2;
    rightNorm += right[index] ** 2;
  }
  if (!leftNorm || !rightNorm) return 0;
  return dot / Math.sqrt(leftNorm * rightNorm);
}

export function facetScore(left, right, facet) {
  return Math.max(0, Math.min(1, (cosineSimilarity(facetVector(left, facet), facetVector(right, facet)) + 1) / 2));
}

function percentile(rows, field, value) {
  const values = rows.map((row) => number(feature(row)[field])).sort((a, b) => a - b);
  if (!values.length) return 0;
  const rank = values.filter((candidate) => candidate <= number(value)).length;
  return rank / values.length;
}

function frequencyDistance(left, right) {
  const lf = feature(left);
  const rf = feature(right);
  const fields = ["service_span_seconds", "median_headway_seconds", "peak_headway_seconds", "off_peak_headway_seconds", "daily_trip_count"];
  const scales = [86_400, 7_200, 7_200, 7_200, 500];
  return fields.reduce((sum, field, index) => sum + Math.min(1, Math.abs(number(lf[field]) - number(rf[field])) / scales[index]), 0) / fields.length;
}

function ratio(left, right) {
  if (!right) return null;
  return number(left) / number(right);
}

export function measuredComparison(query, candidate, queryRows, candidateRows) {
  const q = feature(query);
  const c = feature(candidate);
  const queryCriticality = criticality(query);
  const candidateCriticality = criticality(candidate);
  const queryCriticalityRows = queryRows.filter((row) => row.criticality_json);
  const candidateCriticalityRows = candidateRows.filter((row) => row.criticality_json);
  const queryPrimary = number(queryCriticality.accessibility_auc_loss);
  const candidatePrimary = number(candidateCriticality.accessibility_auc_loss);
  return {
    sameMode: number(query.mode) === number(candidate.mode),
    mode: MODE_NAMES[number(candidate.mode)] || "other",
    transferStationPercentileDifference: Math.abs(
      percentile(queryRows, "transfer_station_count", q.transfer_station_count) -
      percentile(candidateRows, "transfer_station_count", c.transfer_station_count)
    ),
    frequencyProfileDistance: frequencyDistance(query, candidate),
    routeLengthRatio: ratio(c.route_length_metres, q.route_length_metres),
    stationCountDifference: Math.abs(number(q.station_count) - number(c.station_count)),
    criticalityPercentileDifference: queryCriticalityRows.length && candidateCriticalityRows.length
      ? Math.abs(
        percentile(queryCriticalityRows, "accessibility_auc_loss", queryPrimary) -
        percentile(candidateCriticalityRows, "accessibility_auc_loss", candidatePrimary)
      )
      : null,
    queryRawCriticality: Number.isFinite(queryPrimary) ? queryPrimary : null,
    candidateRawCriticality: Number.isFinite(candidatePrimary) ? candidatePrimary : null
  };
}

export function rankSimilarLines(db, {
  querySnapshotId,
  queryLineId,
  queryLineIndex,
  candidateSnapshotId,
  profile = "general",
  weights = {},
  topK = 10
}) {
  const queryRows = all(db, `SELECT li.*, cp.values_json AS criticality_json
    FROM line_instances li
    LEFT JOIN criticality_predictions cp ON cp.line_instance_id = li.id
      AND cp.inference_id = (
        SELECT i.id FROM inference_sets i
        WHERE i.snapshot_id = li.snapshot_id AND i.status = 'ready'
        ORDER BY i.created_at DESC LIMIT 1
      )
    WHERE li.snapshot_id = ? ORDER BY li.line_index`, [querySnapshotId]);
  const candidateRows = all(db, `SELECT li.*, cp.values_json AS criticality_json
    FROM line_instances li
    LEFT JOIN criticality_predictions cp ON cp.line_instance_id = li.id
      AND cp.inference_id = (
        SELECT i.id FROM inference_sets i
        WHERE i.snapshot_id = li.snapshot_id AND i.status = 'ready'
        ORDER BY i.created_at DESC LIMIT 1
      )
    WHERE li.snapshot_id = ? ORDER BY li.line_index`, [candidateSnapshotId]);
  const query = queryRows.find((row) => queryLineId ? row.id === queryLineId : Number(row.line_index) === Number(queryLineIndex));
  if (!query) throw new Error("query line was not found in the selected snapshot");
  const selectedWeights = profileWeights(profile, weights);
  const matches = candidateRows
    .filter((candidate) => candidate.id !== query.id || querySnapshotId !== candidateSnapshotId)
    .map((candidate) => {
      const facetScores = {
        role: facetScore(query, candidate, "role"),
        service: facetScore(query, candidate, "service"),
        geometry: facetScore(query, candidate, "geometry"),
        resilience: facetScore(query, candidate, "resilience")
      };
      const similarity = Object.entries(selectedWeights).reduce((sum, [facet, weight]) => sum + facetScores[facet] * weight, 0);
      return {
        lineInstanceId: candidate.id,
        lineIndex: Number(candidate.line_index),
        displayName: candidate.display_name,
        canonicalId: candidate.canonical_id,
        mode: MODE_NAMES[number(candidate.mode)] || "other",
        similarity,
        facetScores,
        comparison: measuredComparison(query, candidate, queryRows, candidateRows)
      };
    })
    .sort((left, right) => right.similarity - left.similarity || left.displayName.localeCompare(right.displayName))
    .slice(0, Math.max(1, Math.min(100, Number(topK) || 10)));
  return {
    query: {
      lineInstanceId: query.id,
      snapshotId: querySnapshotId,
      lineIndex: Number(query.line_index),
      displayName: query.display_name,
      mode: MODE_NAMES[number(query.mode)] || "other"
    },
    candidateSnapshotId,
    profile,
    weights: selectedWeights,
    embeddingSource: "interpretable-gtfs-signature-fallback",
    matches
  };
}

export function modeName(mode) {
  return MODE_NAMES[number(mode)] || "other";
}
