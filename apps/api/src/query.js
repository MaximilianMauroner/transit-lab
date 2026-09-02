import { all, one, parseJson } from "./db.js";
import { facetVector, modeName } from "./similarity.js";

function number(value, fallback = 0) {
  const result = Number(value);
  return Number.isFinite(result) ? result : fallback;
}

function formatFeed(row) {
  return {
    id: row.id,
    networkId: row.network_id,
    sourceUrl: row.source_url,
    landingPage: row.landing_page,
    downloadedAt: row.downloaded_at,
    validFrom: row.valid_from,
    validTo: row.valid_to,
    sha256: row.sha256,
    byteCount: Number(row.byte_count || 0),
    licence: row.licence,
    geographicalScope: row.geographical_scope,
    localPath: row.local_path,
    validationStatus: row.validation_status,
    metadata: parseJson(row.metadata_json),
    createdAt: row.created_at
  };
}

export function formatSnapshot(row) {
  if (!row) return null;
  return {
    id: row.id,
    networkId: row.network_id,
    feedRevisionId: row.feed_revision_id,
    serviceDate: row.service_date,
    serviceProfile: row.service_profile,
    status: row.status,
    fingerprint: row.fingerprint,
    compilerVersion: row.compiler_version,
    compilerCommit: row.compiler_commit,
    sourceName: row.source_name,
    geographicalScope: row.geographical_scope,
    manifestPath: row.manifest_path,
    networkPath: row.network_path,
    graphPath: row.graph_path,
    counts: parseJson(row.counts_json),
    validation: parseJson(row.validation_json),
    createdAt: row.created_at,
    updatedAt: row.updated_at
  };
}

function formatPrediction(row) {
  const valuesJson = row?.criticality_json || row?.values_json;
  if (!valuesJson) return null;
  return {
    ...parseJson(valuesJson, {}),
    primaryScore: row.primary_score === null ? null : number(row.primary_score),
    uncertainty: row.uncertainty === null ? null : number(row.uncertainty)
  };
}

export function lineRows(db, snapshotId, limit = 10_000) {
  return all(db, `SELECT li.*, cp.values_json AS criticality_json, cp.primary_score, cp.uncertainty,
      (SELECT values_json FROM criticality_labels cl
       WHERE cl.snapshot_id = li.snapshot_id AND cl.line_index = li.line_index) AS label_json
    FROM line_instances li
    LEFT JOIN criticality_predictions cp ON cp.line_instance_id = li.id
      AND cp.inference_id = (
        SELECT i.id FROM inference_sets i
        WHERE i.snapshot_id = li.snapshot_id AND i.status = 'ready'
        ORDER BY i.created_at DESC LIMIT 1
      )
    WHERE li.snapshot_id = ? ORDER BY li.line_index LIMIT ?`, [snapshotId, Math.max(1, Math.min(10_000, limit))]);
}

export function formatLine(row) {
  const features = parseJson(row.feature_json, {});
  const geometry = parseJson(row.geometry_json, {});
  const prediction = formatPrediction(row);
  return {
    id: row.id,
    snapshotId: row.snapshot_id,
    canonicalLineId: row.canonical_line_id,
    lineIndex: Number(row.line_index),
    canonicalId: row.canonical_id,
    displayName: row.display_name,
    agencyKey: row.agency_key,
    mode: Number(row.mode || 0),
    modeName: modeName(row.mode),
    features,
    geometry,
    criticality: prediction,
    label: row.label_json ? parseJson(row.label_json, null) : null
  };
}

export function formatNetworkRow(row, db) {
  const snapshots = all(db, "SELECT * FROM snapshots WHERE network_id = ? ORDER BY service_date DESC, created_at DESC", [row.id]);
  const feeds = all(db, "SELECT * FROM feed_revisions WHERE network_id = ? ORDER BY downloaded_at DESC, created_at DESC", [row.id]);
  return {
    id: row.id,
    projectId: row.project_id,
    displayName: row.display_name,
    geographicalScope: row.geographical_scope,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    snapshotCount: snapshots.length,
    feedCount: feeds.length,
    snapshots: snapshots.map(formatSnapshot),
    feeds: feeds.map(formatFeed)
  };
}

export function getNetwork(db, networkId) {
  return one(db, "SELECT * FROM networks WHERE id = ?", [networkId]);
}

export function getSnapshotRow(db, snapshotId) {
  return one(db, "SELECT * FROM snapshots WHERE id = ?", [snapshotId]);
}

export function getLineRow(db, lineId) {
  return one(db, `SELECT li.*, cp.values_json AS criticality_json, cp.primary_score, cp.uncertainty,
      (SELECT values_json FROM criticality_labels cl
       WHERE cl.snapshot_id = li.snapshot_id AND cl.line_index = li.line_index) AS label_json
    FROM line_instances li
    LEFT JOIN criticality_predictions cp ON cp.line_instance_id = li.id
      AND cp.inference_id = (
        SELECT i.id FROM inference_sets i
        WHERE i.snapshot_id = li.snapshot_id AND i.status = 'ready'
        ORDER BY i.created_at DESC LIMIT 1
      )
    WHERE li.id = ?`, [lineId]);
}

export function snapshotArtifacts(db, snapshotId) {
  return all(db, "SELECT * FROM artifacts WHERE metadata_json LIKE ? ORDER BY created_at", [`%${snapshotId}%`]).map((row) => ({
    id: row.id,
    kind: row.kind,
    fingerprint: row.fingerprint,
    uri: row.uri,
    sizeBytes: Number(row.size_bytes || 0),
    sha256: row.sha256,
    schemaVersion: row.schema_version,
    status: row.status,
    metadata: parseJson(row.metadata_json),
    createdAt: row.created_at,
    supersededBy: row.superseded_by
  }));
}

export function networkPayload(db, snapshotRow, network) {
  const rows = lineRows(db, snapshotRow.id);
  const storedLines = new Map(rows.map((row) => [Number(row.line_index), formatLine(row)]));
  const lines = (network.lines || []).map((line, index) => storedLines.get(Number(line.index ?? index)) || {
    id: `${snapshotRow.id}:${Number(line.index ?? index)}`,
    snapshotId: snapshotRow.id,
    lineIndex: Number(line.index ?? index),
    canonicalId: line.canonical_id || `line:${index}`,
    displayName: line.display_name || `Line ${index}`,
    agencyKey: line.agency_key || "",
    mode: Number(line.mode || 0),
    modeName: modeName(line.mode),
    features: line,
    geometry: {}
  });
  return {
    snapshot: formatSnapshot(snapshotRow),
    provenance: {
      networkId: snapshotRow.network_id,
      snapshotId: snapshotRow.id,
      serviceDate: snapshotRow.service_date,
      modelId: one(db, "SELECT model_id FROM inference_sets WHERE snapshot_id = ? ORDER BY created_at DESC LIMIT 1", [snapshotRow.id])?.model_id || null
    },
    stations: (network.stations || []).map((station, index) => ({
      index: Number(station.index ?? index),
      name: station.name || `Station ${index}`,
      latitude: number(station.latitude),
      longitude: number(station.longitude),
      lineCount: Number(station.line_count || 0),
      patternCount: Number(station.pattern_count || 0),
      transferDegree: Number(station.transfer_degree || 0),
      terminal: Boolean(station.terminal)
    })),
    lines,
    routes: lines.flatMap((line) => (line.geometry?.patterns || []).map((pattern) => ({
      lineIndex: line.lineIndex,
      lineName: line.displayName,
      mode: line.modeName,
      direction: pattern.direction,
      coordinates: pattern.coordinates
    }))),
    transfers: (network.transfers || []).map((transfer) => ({
      from: Number(transfer.from),
      to: Number(transfer.to),
      seconds: Number(transfer.minimum_transfer_seconds || 0),
      confidence: number(transfer.confidence, 0)
    })),
    interchanges: (network.interchanges || []).map((interchange) => ({
      from: Number(interchange.from),
      to: Number(interchange.to),
      sharedStationCount: Number(interchange.shared_station_count || 0)
    }))
  };
}

export function modelRow(row, db) {
  if (!row) return null;
  const aliases = all(db, "SELECT alias FROM model_aliases WHERE model_id = ? ORDER BY alias", [row.id]).map((item) => item.alias);
  return {
    id: row.id,
    version: row.version,
    fingerprint: row.fingerprint,
    status: row.status,
    architecture: parseJson(row.architecture_json),
    datasetId: row.dataset_id,
    trainingRunId: row.training_run_id,
    checkpointArtifactId: row.checkpoint_artifact_id,
    embeddingDimensions: parseJson(row.embedding_dimensions_json),
    supportedHeads: parseJson(row.supported_heads_json, []),
    evaluation: parseJson(row.evaluation_json),
    aliases,
    createdAt: row.created_at
  };
}

export function datasetRow(row) {
  if (!row) return null;
  return {
    id: row.id,
    fingerprint: row.fingerprint,
    status: row.status,
    manifestPath: row.manifest_path,
    featureSchema: row.feature_schema,
    snapshotIds: parseJson(row.snapshot_ids_json, []),
    split: parseJson(row.split_json),
    objectiveCounts: parseJson(row.objective_counts_json),
    quality: parseJson(row.quality_json),
    createdAt: row.created_at,
    updatedAt: row.updated_at
  };
}

export function inferenceRow(row) {
  if (!row) return null;
  return {
    id: row.id,
    fingerprint: row.fingerprint,
    modelId: row.model_id,
    snapshotId: row.snapshot_id,
    status: row.status,
    embeddingsArtifactId: row.embeddings_artifact_id,
    criticalityArtifactId: row.criticality_artifact_id,
    projectionArtifactId: row.projection_artifact_id,
    config: parseJson(row.config_json),
    createdAt: row.created_at
  };
}

export function embeddingPreview(db, snapshotId, facet = "general") {
  const rows = lineRows(db, snapshotId);
  const points = rows.map((row) => {
    const vector = facetVector(row, facet);
    const x = vector.reduce((sum, value, index) => sum + value * Math.cos(index * 1.7), 0);
    const y = vector.reduce((sum, value, index) => sum + value * Math.sin(index * 1.3), 0);
    const criticality = formatPrediction(row);
    return {
      lineInstanceId: row.id,
      lineIndex: Number(row.line_index),
      displayName: row.display_name,
      mode: modeName(row.mode),
      x,
      y,
      criticalityPercentile: criticality ? number(criticality.accessibility_auc_loss) : null,
      source: "feature-space-preview"
    };
  });
  return {
    snapshotId,
    facet,
    source: "feature-space-preview",
    warning: "No model embedding artifact is registered for this snapshot; coordinates are a deterministic GTFS feature preview, not a neural embedding.",
    points
  };
}
