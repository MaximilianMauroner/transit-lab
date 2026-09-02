import { readdir, stat } from "node:fs/promises";
import { basename, dirname, join, relative, resolve, sep } from "node:path";
import {
  fingerprint,
  validateArtifactManifest,
  validateDatasetManifest
} from "../../../packages/contracts/src/index.js";
import {
  all,
  dataRoot,
  json,
  now,
  one,
  parseJson,
  repositoryRoot,
  run
} from "./db.js";

async function walkFiles(directory) {
  let entries;
  try {
    entries = await readdir(directory, { withFileTypes: true });
  } catch {
    return [];
  }
  const files = [];
  for (const entry of entries) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) files.push(...await walkFiles(path));
    else if (entry.isFile()) files.push(path);
  }
  return files;
}

async function readJson(path) {
  try {
    return JSON.parse(await Bun.file(path).text());
  } catch {
    return null;
  }
}

function relativePath(root, path) {
  return relative(root, path).split(sep).join("/");
}

function slug(value, fallback = "unknown") {
  const output = String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
  return output || fallback;
}

function hexBytes(value) {
  if (!Array.isArray(value)) return null;
  return value.map((byte) => Number(byte).toString(16).padStart(2, "0")).join("");
}

function networkIdForSnapshot(root, path, manifest) {
  const parts = relativePath(root, path).split("/");
  if (parts[0] === "data" && ["snapshots", "raw", "graphs"].includes(parts[1])) {
    return slug(parts[2], "unknown");
  }
  if (["snapshots", "raw", "graphs"].includes(parts[0])) {
    return slug(parts[1], "unknown");
  }
  if (/synthetic|demo/i.test(`${manifest?.source_name || ""} ${manifest?.geographical_scope || ""}`)) return "demo";
  return slug(parts[0], slug(manifest?.source_name, "unknown"));
}

function networkDisplayName(networkId, manifest) {
  return manifest?.source_name || manifest?.geographical_scope ||
    (networkId === "demo" ? "Synthetic demo" : networkId.replace(/-/g, " "));
}

async function fileInfo(path) {
  try {
    const value = await stat(path);
    return { size: value.size, mtimeMs: value.mtimeMs };
  } catch {
    return { size: 0, mtimeMs: 0 };
  }
}

function ensureProject(db) {
  const projectId = "project-local";
  run(db, `INSERT INTO projects(id, name, description, created_at)
    VALUES (?, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET name = excluded.name, description = excluded.description`, [
    projectId,
    "Transit Lab",
    "Local-first GTFS representation, simulation, and evaluation workspace",
    now()
  ]);
  return projectId;
}

function ensureNetwork(db, projectId, networkId, displayName, scope = "") {
  const timestamp = now();
  run(db, `INSERT INTO networks(id, project_id, display_name, geographical_scope, created_at, updated_at)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET display_name = excluded.display_name,
      geographical_scope = CASE WHEN excluded.geographical_scope <> '' THEN excluded.geographical_scope ELSE networks.geographical_scope END,
      updated_at = excluded.updated_at`, [networkId, projectId, displayName, scope || "", timestamp, timestamp]);
}

async function ensureArtifact(db, root, { kind, path, fingerprintValue, sha256 = null, metadata = {}, schemaVersion = null }) {
  const info = await fileInfo(path);
  const uri = relativePath(root, path);
  const artifactFingerprint = fingerprintValue || fingerprint("filesystem-artifact-v1", {
    kind,
    uri,
    size: info.size,
    modified: info.mtimeMs
  });
  const id = `artifact-${slug(kind)}-${artifactFingerprint.slice(0, 24)}`;
  run(db, `INSERT INTO artifacts(id, kind, fingerprint, uri, local_path, size_bytes, sha256, schema_version, status, metadata_json, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'ready', ?, ?)
    ON CONFLICT(fingerprint) DO UPDATE SET uri = excluded.uri, local_path = excluded.local_path,
      size_bytes = excluded.size_bytes, sha256 = COALESCE(excluded.sha256, artifacts.sha256),
      schema_version = COALESCE(excluded.schema_version, artifacts.schema_version),
      metadata_json = excluded.metadata_json, status = 'ready'`, [
    id,
    kind,
    artifactFingerprint,
    uri,
    uri,
    info.size,
    sha256,
    schemaVersion,
    json(metadata),
    now()
  ]);
  return one(db, "SELECT * FROM artifacts WHERE fingerprint = ?", [artifactFingerprint]);
}

function connectArtifacts(db, artifactId, dependencies) {
  for (const dependency of dependencies.filter(Boolean)) {
    run(db, `INSERT OR IGNORE INTO artifact_dependencies(artifact_id, depends_on_artifact_id, relation)
      VALUES (?, ?, 'input')`, [artifactId, dependency]);
  }
}

function insideRoot(root, path) {
  const base = resolve(root);
  const candidate = resolve(path);
  return candidate === base || candidate.startsWith(`${base}${sep}`);
}

/** Re-index worker-produced manifests so a fresh API process can recover all
 * outputs from the filesystem without relying on the process that created
 * them still being alive. */
async function syncExplicitArtifactManifest(db, root, manifestPath) {
  const manifest = await readJson(manifestPath);
  if (!manifest) return null;
  try {
    validateArtifactManifest(manifest);
  } catch {
    return null;
  }
  const files = (manifest.files || []).map((file) => ({ ...file }));
  for (const file of files) {
    if (!insideRoot(root, resolve(root, file.path))) return null;
  }
  const primaryPath = files[0]?.path ? resolve(root, files[0].path) : manifestPath;
  const info = await fileInfo(primaryPath);
  const uri = relativePath(root, primaryPath);
  const id = manifest.artifactId;
  run(db, `INSERT INTO artifacts(id, kind, fingerprint, uri, local_path, size_bytes, sha256, schema_version, producing_run_id, git_commit, configuration_json, files_json, status, metadata_json, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ready', ?, ?)
    ON CONFLICT(fingerprint) DO UPDATE SET id = excluded.id, kind = excluded.kind,
      uri = excluded.uri, local_path = excluded.local_path, size_bytes = excluded.size_bytes,
      sha256 = excluded.sha256, schema_version = excluded.schema_version,
      producing_run_id = excluded.producing_run_id, git_commit = excluded.git_commit,
      configuration_json = excluded.configuration_json, files_json = excluded.files_json,
      metadata_json = excluded.metadata_json, status = 'ready'`, [
    id,
    manifest.kind,
    manifest.fingerprint,
    uri,
    uri,
    info.size,
    manifest.sha256 || null,
    manifest.schemaVersion,
    manifest.producingRunId || null,
    manifest.gitCommit || "",
    json(manifest.configuration || {}),
    json(files),
    json(manifest.metadata || {}),
    manifest.createdAt || now()
  ]);
  return { artifactId: id, fingerprint: manifest.fingerprint, inputs: manifest.inputs || [] };
}

async function syncExplicitArtifactManifests(db, root, files) {
  const entries = [];
  for (const path of files.filter((candidate) => {
    const name = basename(candidate);
    return name === "artifact-manifest.json" ||
      name.endsWith(".artifact-manifest.json") ||
      name.endsWith(".worker-artifact-manifest.json");
  })) {
    const entry = await syncExplicitArtifactManifest(db, root, path);
    if (entry) entries.push(entry);
  }
  for (const entry of entries) {
    const artifact = one(db, "SELECT id FROM artifacts WHERE id = ? OR fingerprint = ? LIMIT 1", [entry.artifactId, entry.fingerprint]);
    if (!artifact) continue;
    const dependencies = entry.inputs
      .map((input) => one(db, "SELECT id FROM artifacts WHERE id = ? OR fingerprint = ? LIMIT 1", [input.artifactId, input.fingerprint]))
      .filter(Boolean)
      .map((row) => row.id);
    connectArtifacts(db, artifact.id, dependencies);
  }
  return entries.length;
}

function linePatterns(network, lineIndex) {
  return (network.patterns || [])
    .filter((pattern) => Number(pattern?.signature?.line) === Number(lineIndex))
    .slice(0, 8);
}

function lineGeometry(network, lineIndex) {
  const stations = network.stations || [];
  const patterns = linePatterns(network, lineIndex);
  return {
    patterns: patterns.map((pattern) => ({
      index: Number(pattern.index),
      direction: pattern.signature?.direction_id ?? null,
      coordinates: (pattern.signature?.stops || []).map(Number).map((stationIndex) => {
        const station = stations[stationIndex];
        return station ? [Number(station.longitude), Number(station.latitude)] : null;
      }).filter(Boolean)
    })).filter((pattern) => pattern.coordinates.length > 1)
  };
}

function ensureCanonicalLine(db, networkId, line, timestamp) {
  const id = `canonical-${networkId}-${slug(line.canonical_id || line.display_name || line.index)}`;
  run(db, `INSERT INTO canonical_lines(id, network_id, canonical_name, mode, created_at)
    VALUES (?, ?, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET canonical_name = excluded.canonical_name, mode = excluded.mode`, [
    id,
    networkId,
    line.display_name || line.canonical_id || String(line.index),
    String(line.mode ?? ""),
    timestamp
  ]);
  return id;
}

function syncLineInstances(db, networkId, snapshotId, network) {
  const timestamp = now();
  for (const [position, line] of (network.lines || []).entries()) {
    const lineIndex = Number(line.index ?? position);
    const lineId = `${snapshotId}:${lineIndex}`;
    const canonicalLineId = ensureCanonicalLine(db, networkId, line, timestamp);
    run(db, `INSERT INTO line_instances(id, snapshot_id, canonical_line_id, line_index, canonical_id, display_name, agency_key, mode, feature_json, geometry_json, created_at, updated_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(id) DO UPDATE SET canonical_line_id = excluded.canonical_line_id,
        canonical_id = excluded.canonical_id, display_name = excluded.display_name,
        agency_key = excluded.agency_key, mode = excluded.mode,
        feature_json = excluded.feature_json, geometry_json = excluded.geometry_json,
        updated_at = excluded.updated_at`, [
      lineId,
      snapshotId,
      canonicalLineId,
      lineIndex,
      line.canonical_id || `line:${lineIndex}`,
      line.display_name || line.canonical_id || `Line ${lineIndex}`,
      line.agency_key || "",
      Number(line.mode || 0),
      json(line),
      json(lineGeometry(network, lineIndex)),
      timestamp,
      timestamp
    ]);
  }
}

async function snapshotGraphPath(root, networkId, snapshotPath) {
  const candidates = [
    join(snapshotPath, "..", "graph"),
    join(root, "data", "graphs", networkId),
    join(dirname(snapshotPath), "graph")
  ];
  for (const candidate of candidates) {
    if (await Bun.file(join(candidate, "manifest.json")).exists()) return resolve(candidate);
  }
  return null;
}

async function syncFeedRevision(db, root, sourcePath, source) {
  if (!source?.feed_id || !source.sha256) return null;
  const networkId = slug(source.feed_id);
  ensureNetwork(db, "project-local", networkId, source.display_name || networkId, source.geographical_scope || "");
  const zipPath = join(dirname(sourcePath), "gtfs.zip");
  const revisionId = `feed-${networkId}-${source.sha256}`;
  run(db, `INSERT INTO feed_revisions(id, network_id, source_url, landing_page, downloaded_at, sha256, byte_count, licence, geographical_scope, local_path, validation_status, metadata_json, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'downloaded', ?, ?)
    ON CONFLICT(id) DO UPDATE SET source_url = excluded.source_url, landing_page = excluded.landing_page,
      downloaded_at = excluded.downloaded_at, byte_count = excluded.byte_count, licence = excluded.licence,
      geographical_scope = excluded.geographical_scope, local_path = excluded.local_path,
      metadata_json = excluded.metadata_json`, [
    revisionId,
    networkId,
    source.source_url || "",
    source.landing_page || "",
    source.downloaded_at || null,
    source.sha256,
    Number(source.byte_count || 0),
    source.licence || null,
    source.geographical_scope || "",
    relativePath(root, zipPath),
    json(source),
    now()
  ]);
  const artifact = await ensureArtifact(db, root, {
    kind: "raw-gtfs-feed",
    path: zipPath,
    fingerprintValue: source.sha256,
    sha256: source.sha256,
    metadata: { feedId: source.feed_id, displayName: source.display_name, scope: source.geographical_scope }
  });
  const metadataArtifact = await ensureArtifact(db, root, {
    kind: "feed-source-metadata",
    path: sourcePath,
    metadata: { feedId: source.feed_id }
  });
  connectArtifacts(db, artifact.id, []);
  connectArtifacts(db, metadataArtifact.id, [artifact.id]);
  return revisionId;
}

async function syncSnapshot(db, root, manifestPath, networkPath, knownRevisions) {
  const manifest = await readJson(manifestPath);
  const network = await readJson(networkPath);
  if (!manifest || !network || !manifest.snapshot_id) return null;
  const networkId = networkIdForSnapshot(root, manifestPath, manifest);
  ensureNetwork(db, "project-local", networkId, networkDisplayName(networkId, manifest), manifest.geographical_scope || "");
  const feedHash = hexBytes(manifest.descriptor?.feed_hashes?.[0]);
  const feedRevisionId = feedHash ? knownRevisions.get(`${networkId}:${feedHash}`) || null : null;
  const graphPath = await snapshotGraphPath(root, networkId, dirname(manifestPath));
  const counts = {
    stations: network.stations?.length || 0,
    lines: network.lines?.length || 0,
    patterns: network.patterns?.length || 0,
    transitEdges: network.transit_edges?.length || 0,
    transferEdges: network.transfers?.length || 0,
    interchanges: network.interchanges?.length || 0,
    ...Object.fromEntries(Object.entries(manifest.validation?.row_counts || {}).map(([key, value]) => [key, value]))
  };
  const timestamp = now();
  run(db, `INSERT INTO snapshots(id, network_id, feed_revision_id, service_date, service_profile, status, fingerprint, compiler_version, source_name, geographical_scope, manifest_path, network_path, graph_path, counts_json, validation_json, created_at, updated_at)
    VALUES (?, ?, ?, ?, 'selected-day', 'ready', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET feed_revision_id = excluded.feed_revision_id,
      service_date = excluded.service_date, compiler_version = excluded.compiler_version,
      source_name = excluded.source_name, geographical_scope = excluded.geographical_scope,
      manifest_path = excluded.manifest_path, network_path = excluded.network_path,
      graph_path = excluded.graph_path, counts_json = excluded.counts_json,
      validation_json = excluded.validation_json, status = 'ready', updated_at = excluded.updated_at`, [
    manifest.snapshot_id,
    networkId,
    feedRevisionId,
    manifest.descriptor?.service_date || "unknown",
    manifest.snapshot_id,
    manifest.descriptor?.compiler_version || "",
    manifest.source_name || "",
    manifest.geographical_scope || "",
    relativePath(root, manifestPath),
    relativePath(root, networkPath),
    graphPath ? relativePath(root, graphPath) : null,
    json(counts),
    json(manifest.validation || {}),
    timestamp,
    timestamp
  ]);
  const manifestArtifact = await ensureArtifact(db, root, {
    kind: "compiled-snapshot-manifest",
    path: manifestPath,
    fingerprintValue: manifest.snapshot_id,
    metadata: { snapshotId: manifest.snapshot_id, networkId, serviceDate: manifest.descriptor?.service_date },
    schemaVersion: 1
  });
  const networkArtifact = await ensureArtifact(db, root, {
    kind: "compiled-network",
    path: networkPath,
    fingerprintValue: fingerprint("compiled-network-v1", { snapshotId: manifest.snapshot_id, path: relativePath(root, networkPath) }),
    metadata: { snapshotId: manifest.snapshot_id, networkId, counts }
  });
  connectArtifacts(db, manifestArtifact.id, feedRevisionId ? [
    one(db, "SELECT id FROM artifacts WHERE sha256 = ? LIMIT 1", [feedHash])?.id
  ] : []);
  connectArtifacts(db, networkArtifact.id, [manifestArtifact.id]);
  if (graphPath) {
    const graphManifestPath = join(graphPath, "manifest.json");
    if (await Bun.file(graphManifestPath).exists()) {
      const graphManifest = await readJson(graphManifestPath);
      const graphArtifact = await ensureArtifact(db, root, {
        kind: "compiled-graph-manifest",
        path: graphManifestPath,
        fingerprintValue: fingerprint("compiled-graph-v1", graphManifest || { snapshotId: manifest.snapshot_id }),
        metadata: { snapshotId: manifest.snapshot_id, graphSchema: graphManifest?.schema_version || null },
        schemaVersion: 1
      });
      connectArtifacts(db, graphArtifact.id, [networkArtifact.id]);
    }
  }
  syncLineInstances(db, networkId, manifest.snapshot_id, network);
  return { id: manifest.snapshot_id, networkId, network, manifest };
}

async function syncLabels(db, root, path, snapshotById) {
  const lines = (await Bun.file(path).text()).split(/\r?\n/).filter(Boolean);
  if (!lines.length) return;
  const snapshotId = parseJson(lines[0], {})?.snapshot;
  if (!snapshotId || !snapshotById.has(snapshotId)) return;
  const artifact = await ensureArtifact(db, root, {
    kind: "criticality-labels",
    path,
    metadata: { snapshotId, rows: lines.length }
  });
  for (const line of lines) {
    const label = parseJson(line, null);
    if (!label || label.line === undefined) continue;
    run(db, `INSERT INTO criticality_labels(snapshot_id, line_index, values_json, source_artifact_id, created_at)
      VALUES (?, ?, ?, ?, ?)
      ON CONFLICT(snapshot_id, line_index) DO UPDATE SET values_json = excluded.values_json, source_artifact_id = excluded.source_artifact_id`, [
      snapshotId,
      Number(label.line),
      json(label),
      artifact.id,
      now()
    ]);
  }
}

async function syncDatasetManifestFile(db, root, path) {
  const manifest = await readJson(path);
  if (!manifest) return null;
  try {
    validateDatasetManifest(manifest);
  } catch {
    return null;
  }
  const relativeManifestPath = relativePath(root, path);
  const artifact = one(db, "SELECT id FROM artifacts WHERE local_path = ? ORDER BY created_at DESC LIMIT 1", [relativeManifestPath]) ||
    await ensureArtifact(db, root, {
      kind: "dataset-manifest",
      path,
      fingerprintValue: manifest.fingerprint,
      metadata: { datasetId: manifest.datasetId, featureSchema: manifest.featureSchema },
      schemaVersion: manifest.schemaVersion
    });
  const split = typeof manifest.split === "object" ? manifest.split : { strategy: manifest.split };
  run(db, `INSERT INTO datasets(id, fingerprint, status, manifest_path, feature_schema, snapshot_ids_json, split_json, objective_counts_json, quality_json, created_at, updated_at)
    VALUES (?, ?, 'ready', ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET fingerprint = excluded.fingerprint, manifest_path = excluded.manifest_path,
      feature_schema = excluded.feature_schema, snapshot_ids_json = excluded.snapshot_ids_json,
      split_json = excluded.split_json, objective_counts_json = excluded.objective_counts_json,
      quality_json = excluded.quality_json, status = 'ready', updated_at = excluded.updated_at`, [
    manifest.datasetId,
    manifest.fingerprint,
    relativeManifestPath,
    manifest.featureSchema,
    json(manifest.snapshotIds),
    json(split),
    json(manifest.counts || manifest.objectives || {}),
    json(manifest.quality || {}),
    manifest.createdAt || now(),
    now()
  ]);
  return { datasetId: manifest.datasetId, artifactId: artifact.id };
}

function inferenceModelId(root, path, result) {
  if (result.modelId) return String(result.modelId);
  const parts = relativePath(root, path).split("/");
  const marker = parts.indexOf("inference");
  return marker >= 0 ? parts[marker + 1] || null : null;
}

async function syncInferencePredictionsFile(db, root, path, snapshotById) {
  const result = await readJson(path);
  if (!result?.snapshot_id || !Array.isArray(result.metric_names) || !Array.isArray(result.predictions)) return null;
  if (!snapshotById.has(result.snapshot_id)) return null;
  const modelId = inferenceModelId(root, path, result);
  if (!modelId || !one(db, "SELECT id FROM model_versions WHERE id = ?", [modelId])) return null;
  const criticalityArtifact = await ensureArtifact(db, root, {
    kind: "criticality-predictions",
    path,
    fingerprintValue: fingerprint("inference-predictions-v1", result),
    metadata: { snapshotId: result.snapshot_id, modelId, rows: result.predictions.length }
  });
  const embeddingsArtifact = result.line_embeddings?.length
    ? await ensureArtifact(db, root, {
      kind: "line-embeddings",
      path,
      fingerprintValue: fingerprint("line-embeddings-v1", { path: relativePath(root, path), modelId, snapshotId: result.snapshot_id }),
      metadata: { snapshotId: result.snapshot_id, modelId, rows: result.line_embeddings.length }
    })
    : null;
  const inferenceId = result.inference_id || `inference-${modelId}-${result.snapshot_id}`;
  const inferenceFingerprint = fingerprint("inference-set-v1", { inferenceId, modelId, snapshotId: result.snapshot_id, source: relativePath(root, path) });
  run(db, `INSERT INTO inference_sets(id, fingerprint, model_id, snapshot_id, status, embeddings_artifact_id, criticality_artifact_id, config_json, created_at)
    VALUES (?, ?, ?, ?, 'ready', ?, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET fingerprint = excluded.fingerprint, embeddings_artifact_id = excluded.embeddings_artifact_id,
      criticality_artifact_id = excluded.criticality_artifact_id, config_json = excluded.config_json, status = 'ready'`, [
    inferenceId,
    inferenceFingerprint,
    modelId,
    result.snapshot_id,
    embeddingsArtifact?.id || null,
    criticalityArtifact.id,
    json({ source: relativePath(root, path), metricNames: result.metric_names }),
    now()
  ]);
  for (const prediction of result.predictions) {
    const lineIndex = Number(prediction.line);
    if (!Number.isInteger(lineIndex)) continue;
    const lineInstanceId = `${result.snapshot_id}:${lineIndex}`;
    if (!one(db, "SELECT id FROM line_instances WHERE id = ?", [lineInstanceId])) continue;
    const values = Object.fromEntries(result.metric_names.map((name, index) => [name, Number(prediction.metrics?.[index] ?? 0)]));
    sqlRunPrediction(db, inferenceId, lineInstanceId, values, prediction);
  }
  return { inferenceId, modelId };
}

function sqlRunPrediction(db, inferenceId, lineInstanceId, values, prediction) {
  run(db, `INSERT INTO criticality_predictions(inference_id, line_instance_id, primary_score, uncertainty, values_json, created_at)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(inference_id, line_instance_id) DO UPDATE SET primary_score = excluded.primary_score,
      uncertainty = excluded.uncertainty, values_json = excluded.values_json`, [
    inferenceId,
    lineInstanceId,
    Number(values.accessibility_auc_loss ?? values.accessibility_loss ?? Object.values(values)[0] ?? 0),
    Number.isFinite(Number(prediction.uncertainty)) ? Number(prediction.uncertainty) : null,
    json({ ...values, structural_uniqueness: Number(prediction.structural_uniqueness ?? 0) }),
    now()
  ]);
}

async function syncModelAndPredictions(db, root, modelPath, predictionsPath, snapshotById) {
  const predictions = await readJson(predictionsPath);
  if (!predictions?.snapshot_id || !snapshotById.has(predictions.snapshot_id)) return;
  const checkpoint = await readJson(modelPath);
  const modelInfo = checkpoint?.report || {};
  const modelFingerprint = fingerprint("model-checkpoint-v1", {
    path: relativePath(root, modelPath),
    size: (await fileInfo(modelPath)).size
  });
  const modelId = "model-demo-reference";
  const checkpointArtifact = await ensureArtifact(db, root, {
    kind: "model-checkpoint",
    path: modelPath,
    fingerprintValue: modelFingerprint,
    metadata: { backend: modelInfo.backend || "reference-cpu", source: "filesystem" }
  });
  run(db, `INSERT INTO model_versions(id, version, fingerprint, status, architecture_json, checkpoint_artifact_id, embedding_dimensions_json, supported_heads_json, evaluation_json, created_at)
    VALUES (?, ?, ?, 'ready', ?, ?, ?, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET version = excluded.version, fingerprint = excluded.fingerprint,
      architecture_json = excluded.architecture_json, checkpoint_artifact_id = excluded.checkpoint_artifact_id,
      supported_heads_json = excluded.supported_heads_json, evaluation_json = excluded.evaluation_json`, [
    modelId,
    "demo-reference",
    modelFingerprint,
    json({ backend: modelInfo.backend || "reference-cpu", report: modelInfo }),
    checkpointArtifact.id,
    json({ base: 32, general: 64, role: 48, service: 32, geometry: 32, resilience: 32 }),
    json(["criticality", "reconstruction", "similarity-preview"]),
    json({}),
    now()
  ]);
  run(db, `INSERT INTO model_aliases(alias, model_id, updated_at) VALUES ('candidate', ?, ?)
    ON CONFLICT(alias) DO UPDATE SET model_id = excluded.model_id, updated_at = excluded.updated_at`, [modelId, now()]);
  const inferenceId = `inference-${modelId}-${predictions.snapshot_id}`;
  const inferenceFingerprint = fingerprint("inference-v1", { modelId, snapshotId: predictions.snapshot_id });
  const criticalityArtifact = await ensureArtifact(db, root, {
    kind: "criticality-predictions",
    path: predictionsPath,
    fingerprintValue: fingerprint("prediction-file-v1", predictions),
    metadata: { snapshotId: predictions.snapshot_id, modelId, rows: predictions.predictions?.length || 0 }
  });
  run(db, `INSERT INTO inference_sets(id, fingerprint, model_id, snapshot_id, status, criticality_artifact_id, config_json, created_at)
    VALUES (?, ?, ?, ?, 'ready', ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET fingerprint = excluded.fingerprint, criticality_artifact_id = excluded.criticality_artifact_id, status = 'ready'`, [
    inferenceId,
    inferenceFingerprint,
    modelId,
    predictions.snapshot_id,
    criticalityArtifact.id,
    json({ source: relativePath(root, predictionsPath), metricNames: predictions.metric_names || [] }),
    now()
  ]);
  for (const prediction of predictions.predictions || []) {
    const lineInstanceId = `${predictions.snapshot_id}:${Number(prediction.line)}`;
    const values = Object.fromEntries((predictions.metric_names || []).map((name, index) => [name, Number(prediction.metrics?.[index] ?? 0)]));
    run(db, `INSERT INTO criticality_predictions(inference_id, line_instance_id, primary_score, uncertainty, values_json, created_at)
      VALUES (?, ?, ?, ?, ?, ?)
      ON CONFLICT(inference_id, line_instance_id) DO UPDATE SET primary_score = excluded.primary_score,
        uncertainty = excluded.uncertainty, values_json = excluded.values_json`, [
      inferenceId,
      lineInstanceId,
      Number(values.accessibility_auc_loss ?? 0),
      null,
      json({ ...values, structural_uniqueness: Number(prediction.structural_uniqueness ?? 0) }),
      now()
    ]);
  }
  return { modelId, inferenceId };
}

export async function syncFilesystem(db, root = repositoryRoot()) {
  const projectId = ensureProject(db);
  const rootData = dataRoot(root);
  const files = await walkFiles(rootData);
  const sourcePaths = files.filter((path) => basename(path) === "source.json");
  const knownRevisions = new Map();
  for (const sourcePath of sourcePaths) {
    const source = await readJson(sourcePath);
    const revisionId = await syncFeedRevision(db, root, sourcePath, source);
    if (revisionId && source) knownRevisions.set(`${slug(source.feed_id)}:${source.sha256}`, revisionId);
  }

  const snapshotPaths = files.filter((path) => basename(path) === "manifest.json" &&
    files.includes(join(dirname(path), "network.json")));
  const snapshots = new Map();
  for (const manifestPath of snapshotPaths) {
    const result = await syncSnapshot(db, root, manifestPath, join(dirname(manifestPath), "network.json"), knownRevisions);
    if (result) snapshots.set(result.id, result);
  }

  const labelPaths = files.filter((path) => basename(path) === "labels.jsonl");
  for (const path of labelPaths) await syncLabels(db, root, path, snapshots);

  const explicitArtifacts = await syncExplicitArtifactManifests(db, root, files);
  for (const path of files.filter((candidate) => basename(candidate) === "dataset-manifest.json" || basename(candidate) === "manifest.json" && candidate.includes(`${sep}datasets${sep}`))) {
    await syncDatasetManifestFile(db, root, path);
  }

  const modelPaths = files.filter((path) => basename(path) === "model.json");
  const predictionPaths = files.filter((path) => basename(path) === "predictions.json");
  for (const predictionsPath of predictionPaths) {
    const directory = dirname(predictionsPath);
    const modelPath = modelPaths.find((candidate) => dirname(candidate) === directory);
    if (modelPath) await syncModelAndPredictions(db, root, modelPath, predictionsPath, snapshots);
  }
  for (const path of predictionPaths) await syncInferencePredictionsFile(db, root, path, snapshots);

  return { ...inventorySummary(db, projectId), explicitArtifacts };
}

export function inventorySummary(db, projectId = "project-local") {
  const count = (table, where = "") => Number(one(db, `SELECT COUNT(*) AS count FROM ${table}${where ? ` WHERE ${where}` : ""}`)?.count || 0);
  return {
    projectId,
    networks: count("networks"),
    feedRevisions: count("feed_revisions"),
    snapshots: count("snapshots"),
    lineInstances: count("line_instances"),
    datasets: count("datasets"),
    models: count("model_versions"),
    inferenceSets: count("inference_sets"),
    criticalityLabels: count("criticality_labels"),
    runs: count("runs")
  };
}

export function findSnapshot(db, snapshotId) {
  const row = one(db, "SELECT * FROM snapshots WHERE id = ?", [snapshotId]);
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

export function findArtifact(db, artifactId) {
  const row = one(db, "SELECT * FROM artifacts WHERE id = ?", [artifactId]);
  if (!row) return null;
  return {
    id: row.id,
    kind: row.kind,
    fingerprint: row.fingerprint,
    uri: row.uri,
    localPath: row.local_path,
    sizeBytes: Number(row.size_bytes || 0),
    sha256: row.sha256,
    schemaVersion: row.schema_version,
    status: row.status,
    metadata: parseJson(row.metadata_json),
    createdAt: row.created_at,
    supersededBy: row.superseded_by
  };
}

export function artifactDependencies(db, artifactId) {
  return all(db, `SELECT a.* FROM artifacts a JOIN artifact_dependencies d ON d.depends_on_artifact_id = a.id
    WHERE d.artifact_id = ? ORDER BY a.created_at`, [artifactId]).map((row) => findArtifact(db, row.id));
}

export async function loadSnapshotNetwork(db, root, snapshotId) {
  const snapshot = findSnapshot(db, snapshotId);
  if (!snapshot) return null;
  const network = await readJson(resolve(root, snapshot.networkPath));
  if (!network) return null;
  return { snapshot, network };
}
