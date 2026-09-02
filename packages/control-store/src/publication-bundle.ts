import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, relative, resolve, sep } from "node:path";
import {
  PUBLICATION_MANIFEST_SCHEMA_VERSION,
  assertSafeId,
  validatePublicationManifest
} from "../../contracts/src/index.ts";
import { all, dataRoot, json, now, one, parseJson } from "./database.ts";

function inside(root, candidate) {
  const base = resolve(root);
  const path = resolve(candidate);
  return path === base || path.startsWith(`${base}${sep}`);
}

function bundlePath(root, value) {
  const candidates = [resolve(root, value), resolve(dataRoot(root), value)];
  const path = candidates.find((candidate) => inside(root, candidate) || inside(dataRoot(root), candidate));
  if (!path) throw new Error("publication path escaped the repository and data roots");
  return path;
}

function sourceJson(root, value) {
  if (!value) return null;
  try {
    return JSON.parse(readFileSync(bundlePath(root, value), "utf8"));
  } catch {
    return null;
  }
}

function safeSnapshot(row) {
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
    counts: parseJson(row.counts_json),
    validation: parseJson(row.validation_json),
    createdAt: row.created_at,
    updatedAt: row.updated_at
  };
}

function safeModel(row) {
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
    createdAt: row.created_at
  };
}

function safeArtifact(row) {
  return {
    id: row.id,
    kind: row.kind,
    fingerprint: row.fingerprint,
    sizeBytes: Number(row.size_bytes || 0),
    sha256: row.sha256,
    schemaVersion: row.schema_version,
    producingRunId: row.producing_run_id,
    gitCommit: row.git_commit,
    status: row.status,
    metadata: parseJson(row.metadata_json),
    createdAt: row.created_at,
    supersededBy: row.superseded_by
  };
}

function publicNetwork(network) {
  if (!network) return null;
  return {
    snapshot_id: network.snapshot_id,
    manifest: network.manifest || {},
    stations: network.stations || [],
    lines: network.lines || [],
    patterns: (network.patterns || []).map((pattern) => ({
      index: pattern.index,
      signature: pattern.signature
    })),
    transit_edges: (network.transit_edges || []).map(({ departures_by_bin, median_runtime_by_bin, ...edge }) => edge),
    transfers: network.transfers || [],
    interchanges: network.interchanges || []
  };
}

const DEFAULT_METRIC_NAMES = [
  "accessibility_auc_loss",
  "unreachable_share",
  "mean_delay_reachable_seconds",
  "p95_delay_reachable_seconds",
  "mean_extra_transfers",
  "stations_losing_all_service_share"
];

function criticalityResult(db, row) {
  const config = parseJson(row.config_json);
  const metricNames = Array.isArray(config.metricNames) && config.metricNames.length
    ? config.metricNames
    : DEFAULT_METRIC_NAMES;
  const predictions = all(db, `SELECT cp.*, li.line_index, li.display_name
    FROM criticality_predictions cp
    JOIN line_instances li ON li.id = cp.line_instance_id
    WHERE cp.inference_id = ? ORDER BY cp.primary_score DESC, li.display_name`, [row.id]).map((prediction) => {
    const values = parseJson(prediction.values_json);
    const percentiles = values.percentiles && typeof values.percentiles === "object" ? values.percentiles : null;
    const numeric = (value, fallback = 0) => Number.isFinite(Number(value)) ? Number(value) : fallback;
    return {
      line: Number(prediction.line_index),
      lineName: prediction.display_name,
      metrics: metricNames.map((name) => numeric(values[name] ?? values[name.replaceAll("_auc_loss", "_loss")])),
      metricPercentiles: percentiles
        ? metricNames.map((name) => numeric(percentiles[name]))
        : undefined,
      structuralUniqueness: numeric(values.structuralUniqueness ?? values.structural_uniqueness),
      uncertainty: numeric(prediction.uncertainty)
    };
  });
  return {
    schemaVersion: 1,
    inferenceId: row.id,
    modelId: row.model_id,
    snapshotId: row.snapshot_id,
    metricNames,
    predictions,
    status: row.status,
    createdAt: row.created_at
  };
}

function selectedRows(db, table, ids) {
  if (!ids.length) return [];
  const placeholders = ids.map(() => "?").join(", ");
  return all(db, `SELECT * FROM ${table} WHERE id IN (${placeholders})`, ids);
}

function immutableJson(path, value) {
  const encoded = `${JSON.stringify(value, null, 2)}\n`;
  try {
    writeFileSync(path, encoded, { flag: "wx" });
  } catch (error) {
    if (error?.code !== "EEXIST") throw error;
    if (readFileSync(path, "utf8") !== encoded) {
      throw new Error(`refusing to overwrite immutable publication file ${path}`);
    }
  }
}

function descriptor(bundleDirectory, path) {
  const bytes = readFileSync(path);
  return {
    path: relative(bundleDirectory, path).split(sep).join("/"),
    sizeBytes: bytes.length,
    sha256: createHash("sha256").update(bytes).digest("hex")
  };
}

function artifactPayload(root, row) {
  if (!row?.local_path) return null;
  const value = sourceJson(root, row.local_path);
  return value ? { artifactId: row.id, value } : null;
}

/** Create the immutable JSON files served by the public Explorer. */
export function writePublicationBundle(db, root, {
  id,
  slug,
  title,
  snapshotIds,
  modelIds,
  artifactIds,
  metadata = {},
  createdAt = now()
}) {
  assertSafeId(String(id), "publication id");
  assertSafeId(String(slug), "publication slug");
  const snapshots = selectedRows(db, "snapshots", snapshotIds).filter((row) => row.status === "ready");
  if (snapshots.length !== snapshotIds.length) throw new Error("publication contains an unavailable snapshot");
  const models = selectedRows(db, "model_versions", modelIds).filter((row) => row.status === "ready");
  if (models.length !== modelIds.length) throw new Error("publication contains an unavailable model");
  const artifacts = selectedRows(db, "artifacts", artifactIds).filter((row) => row.status === "ready");
  if (artifacts.length !== artifactIds.length) throw new Error("publication contains an unavailable artifact");

  const snapshotSet = new Set(snapshotIds.map(String));
  const modelSet = new Set(modelIds.map(String));
  const inferenceRows = all(db, "SELECT * FROM inference_sets WHERE status = 'ready' ORDER BY created_at DESC, id DESC")
    .filter((row) => snapshotSet.has(String(row.snapshot_id)) && (!modelSet.size || modelSet.has(String(row.model_id))));
  const inferenceArtifactIds = inferenceRows.map((row) => row.criticality_artifact_id).filter(Boolean);
  const allArtifactIds = [...new Set([...artifactIds, ...inferenceArtifactIds, ...models.map((row) => row.checkpoint_artifact_id).filter(Boolean)])];
  const allArtifacts = selectedRows(db, "artifacts", allArtifactIds).filter((row) => row.status === "ready");

  const contents = new Map();
  const entries = {
    overview: "overview.json",
    snapshots: "snapshots.json",
    models: "models.json",
    networks: {},
    criticality: "criticality.json",
    embeddings: "embeddings.json",
    evaluations: "evaluations.json",
    similarity: "similarity.json"
  };
  const snapshotValues = snapshots.map(safeSnapshot);
  const modelValues = models.map(safeModel);
  contents.set(entries.snapshots, snapshotValues);
  contents.set(entries.models, modelValues);

  const networks = {};
  for (const snapshot of snapshots) {
    const network = publicNetwork(sourceJson(root, snapshot.network_path));
    if (!network) continue;
    const path = `networks/${snapshot.id}.json`;
    entries.networks[snapshot.id] = path;
    networks[snapshot.id] = path;
    contents.set(path, network);
  }

  const criticality = {};
  for (const row of inferenceRows) {
    criticality[row.snapshot_id] = criticalityResult(db, row);
  }
  contents.set(entries.criticality, { schemaVersion: 1, results: criticality });

  const publicArtifacts = allArtifacts
    .filter((row) => /embedding|projection/i.test(row.kind))
    .map(safeArtifact);
  contents.set(entries.embeddings, publicArtifacts);

  const evaluationRows = all(db, `SELECT se.*, mv.version AS model_version
    FROM similarity_evaluations se
    LEFT JOIN model_versions mv ON mv.id = se.model_id
    ORDER BY se.created_at DESC, se.id DESC`)
    .filter((row) => (!modelSet.size || modelSet.has(String(row.model_id))) &&
      (!row.dataset_id || modelValues.some((model) => model.datasetId === row.dataset_id)))
    .map((row) => ({
      id: row.id,
      modelId: row.model_id,
      modelVersion: row.model_version,
      datasetId: row.dataset_id,
      facet: row.facet,
      metricName: row.metric_name,
      value: Number(row.value),
      split: row.split,
      createdAt: row.created_at
    }));
  contents.set(entries.evaluations, evaluationRows);

  const similarity = allArtifacts
    .filter((row) => row.kind === "similarity-result")
    .map((row) => artifactPayload(root, row))
    .filter(Boolean)
    .map(({ artifactId, value }) => ({ ...value, artifactId }));
  contents.set(entries.similarity, similarity);

  contents.set(entries.overview, {
    projectId: "project-local",
    counts: {
      publications: 1,
      snapshots: snapshotValues.length,
      models: modelValues.length,
      networks: Object.keys(networks).length,
      criticality: Object.keys(criticality).length,
      embeddings: publicArtifacts.length,
      evaluations: evaluationRows.length,
      similarity: similarity.length
    }
  });

  const bundleDirectory = resolve(dataRoot(root), "publications", slug);
  mkdirSync(bundleDirectory, { recursive: true });
  const paths = [];
  for (const [path, value] of contents) {
    const target = resolve(bundleDirectory, path);
    if (!inside(bundleDirectory, target)) throw new Error("publication content escaped its bundle");
    mkdirSync(dirname(target), { recursive: true });
    immutableJson(target, value);
    paths.push(target);
  }
  paths.sort();
  const manifest = {
    schemaVersion: PUBLICATION_MANIFEST_SCHEMA_VERSION,
    publicationId: id,
    slug,
    title: String(title).trim(),
    createdAt,
    snapshotIds,
    modelIds,
    artifactIds: allArtifactIds,
    entries,
    metadata,
    files: paths.map((path) => descriptor(bundleDirectory, path))
  };
  validatePublicationManifest(manifest);
  const manifestPath = resolve(bundleDirectory, "manifest.json");
  immutableJson(manifestPath, manifest);
  return { manifest, manifestPath };
}

function readBundleFile(bundleDirectory, file) {
  const path = resolve(bundleDirectory, file.path);
  if (!inside(bundleDirectory, path)) throw new Error("publication manifest references an escaped file");
  const bytes = readFileSync(path);
  const sha256 = createHash("sha256").update(bytes).digest("hex");
  if (bytes.length !== Number(file.sizeBytes) || sha256 !== file.sha256) {
    throw new Error(`publication file integrity check failed for ${file.path}`);
  }
  return JSON.parse(bytes.toString("utf8"));
}

/** Read and verify a publication bundle without consulting live result rows. */
export function loadPublicationBundle(root, publication) {
  const manifestPath = publication?.manifestPath
    ? bundlePath(root, publication.manifestPath)
    : resolve(dataRoot(root), "publications", publication?.slug, "manifest.json");
  let manifest;
  try {
    manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
    validatePublicationManifest(manifest);
  } catch (error) {
    throw new Error(`publication bundle is invalid: ${error instanceof Error ? error.message : String(error)}`);
  }
  const bundleDirectory = dirname(manifestPath);
  const data = {};
  for (const file of manifest.files) data[file.path] = readBundleFile(bundleDirectory, file);
  return { manifest, data, manifestPath };
}

export function publicationData(bundle, entry) {
  const path = typeof entry === "string" ? entry : null;
  return path ? bundle.data[path] : undefined;
}
