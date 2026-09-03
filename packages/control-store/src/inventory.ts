import { createHash } from "node:crypto";
import { readdir, readFile, realpath, stat } from "node:fs/promises";
import { basename, dirname, join, relative, resolve, sep } from "node:path";
import {
  assertSafeId,
  fingerprint,
  validateArtifactManifest,
  validateBenchmarkResult,
  validateDatasetManifest,
  validateEvaluationResult,
  validateInferenceResult,
  ROUTER_ALGORITHM_VERSION
} from "../../contracts/src/index.ts";
import {
  all,
  dataRoot,
  json,
  now,
  one,
  parseJson,
  repositoryRoot,
  run
} from "./database.ts";

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
  return files.sort((left, right) => left.localeCompare(right));
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

async function fileSha256(path) {
  return createHash("sha256").update(await readFile(path)).digest("hex");
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

type EnsureArtifactOptions = {
  kind: string;
  artifactId?: string | null;
  path: string;
  fingerprintValue?: string | null;
  sha256?: string | null;
  metadata?: Record<string, unknown>;
  schemaVersion?: number | null;
  producingRunId?: string | null;
  gitCommit?: string;
  configuration?: Record<string, unknown>;
  files?: Array<Record<string, unknown>>;
};

async function ensureArtifact(db, root, {
  kind,
  artifactId = null,
  path,
  fingerprintValue,
  sha256 = null,
  metadata = {},
  schemaVersion = null,
  producingRunId = null,
  gitCommit = "",
  configuration = {},
  files = []
}: EnsureArtifactOptions) {
  const info = await fileInfo(path);
  const uri = relativePath(root, path);
  const artifactFingerprint = fingerprintValue || fingerprint("filesystem-artifact-v1", {
    kind,
    uri,
    size: info.size,
    modified: info.mtimeMs
  });
  const id = artifactId || `artifact-${slug(kind)}-${artifactFingerprint.slice(0, 24)}`;
  run(db, `INSERT INTO artifacts(id, kind, fingerprint, uri, local_path, size_bytes, sha256, schema_version, producing_run_id, git_commit, configuration_json, files_json, status, metadata_json, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ready', ?, ?)
    ON CONFLICT(fingerprint) DO UPDATE SET uri = excluded.uri, local_path = excluded.local_path,
      size_bytes = excluded.size_bytes, sha256 = COALESCE(excluded.sha256, artifacts.sha256),
      schema_version = COALESCE(excluded.schema_version, artifacts.schema_version),
      producing_run_id = COALESCE(excluded.producing_run_id, artifacts.producing_run_id),
      git_commit = CASE WHEN excluded.git_commit <> '' THEN excluded.git_commit ELSE artifacts.git_commit END,
      configuration_json = excluded.configuration_json, files_json = excluded.files_json,
      metadata_json = excluded.metadata_json, status = 'ready'`, [
    id,
    kind,
    artifactFingerprint,
    uri,
    uri,
    info.size,
    sha256,
    schemaVersion,
    producingRunId,
    gitCommit,
    json(configuration),
    json(files),
    json(metadata),
    now()
  ]);
  return one(db, "SELECT * FROM artifacts WHERE fingerprint = ?", [artifactFingerprint]);
}

function indexedArtifactForPath(db, root, path, kinds) {
  const uri = relativePath(root, path);
  const placeholders = kinds.map(() => "?").join(", ");
  return one(db, `SELECT * FROM artifacts
    WHERE uri = ? AND kind IN (${placeholders})
    ORDER BY schema_version DESC, created_at DESC, id DESC LIMIT 1`, [uri, ...kinds]);
}

function explicitArtifactFileForPath(artifact, root, path, sha256, sizeBytes) {
  if (!artifact) return false;
  const descriptors = parseJson(artifact.files_json, []);
  if (!Array.isArray(descriptors)) return false;
  const expectedPaths = new Set([
    relativePath(root, path),
    relativePath(dataRoot(root), path)
  ].map((value) => value.replaceAll("\\", "/")));
  return descriptors.some((descriptor) => {
    if (!descriptor || typeof descriptor !== "object") return false;
    const descriptorPath = typeof descriptor.path === "string"
      ? descriptor.path.replaceAll("\\", "/")
      : "";
    return expectedPaths.has(descriptorPath)
      && String(descriptor.sha256 || "").toLowerCase() === sha256
      && Number(descriptor.sizeBytes) === Number(sizeBytes);
  });
}

function record(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function firstValue(sources, keys) {
  for (const source of sources) {
    const valueSource = record(source);
    for (const key of keys) {
      const value = valueSource[key];
      if (typeof value === "string" && value.trim()) return value.trim();
      if (typeof value === "number" && Number.isFinite(value)) return String(value);
    }
  }
  return null;
}

function safeIdentifier(value, field) {
  if (value === null || value === undefined || value === "") return null;
  try {
    return assertSafeId(String(value), field);
  } catch {
    return null;
  }
}

function generatedModelId(modelFingerprint) {
  const candidate = `model-${modelFingerprint}`;
  return safeIdentifier(candidate, "modelId") || `model-${fingerprint("model-id-v1", { modelFingerprint })}`;
}

function referencedId(db, table, value) {
  if (!value) return null;
  return one(db, `SELECT id FROM ${table} WHERE id = ?`, [value])?.id || null;
}

function modelLineageSources(checkpoint, artifact, identityHint) {
  const artifactMetadata = parseJson(artifact?.metadata_json, {});
  const artifactConfiguration = parseJson(artifact?.configuration_json, {});
  return [
    checkpoint,
    checkpoint?.metadata,
    checkpoint?.lineage,
    checkpoint?.model,
    artifactMetadata,
    artifactConfiguration,
    identityHint,
    identityHint?.metadata
  ];
}

function modelArchitecture(checkpoint, modelInfo) {
  const encoderConfig = record(checkpoint?.encoder?.config);
  const explicitArchitecture = record(checkpoint?.architecture);
  return {
    backend: modelInfo.backend || checkpoint?.backend || "reference-cpu",
    ...(Object.keys(encoderConfig).length ? { encoder: encoderConfig } : {}),
    ...explicitArchitecture,
    ...(Object.keys(modelInfo).length ? { report: modelInfo } : {})
  };
}

function modelSupportedHeads(checkpoint) {
  if (Array.isArray(checkpoint?.supportedHeads) && checkpoint.supportedHeads.length) {
    return checkpoint.supportedHeads;
  }
  if (Array.isArray(checkpoint?.supported_heads) && checkpoint.supported_heads.length) {
    return checkpoint.supported_heads;
  }
  return ["criticality", "reconstruction", "similarity-preview"];
}

function modelEmbeddingDimensions(checkpoint) {
  if (record(checkpoint?.embeddingDimensions) && Object.keys(record(checkpoint.embeddingDimensions)).length) {
    return checkpoint.embeddingDimensions;
  }
  if (record(checkpoint?.embedding_dimensions) && Object.keys(record(checkpoint.embedding_dimensions)).length) {
    return checkpoint.embedding_dimensions;
  }
  return { base: 32, general: 64, role: 48, service: 32, geometry: 32, resilience: 32 };
}

function modelRunHint(db, root, modelPath) {
  const parts = relativePath(dataRoot(root), modelPath).split("/");
  if (parts.length < 3 || parts[0] !== "runs") return { run: null, runId: null };
  const runId = safeIdentifier(parts[1], "trainingRunId");
  if (!runId) return { run: null, runId: null };
  const runRow = one(db, "SELECT id, dataset_id FROM runs WHERE id = ?", [runId]);
  return { run: runRow, runId: runRow?.id || null };
}

function updateIndexedModel(db, modelId, values) {
  const existing = one(db, "SELECT * FROM model_versions WHERE id = ?", [modelId]);
  if (!existing) return;
  if (existing.fingerprint !== values.fingerprint) {
    throw new Error(`model ${modelId} already has immutable fingerprint ${existing.fingerprint}`);
  }

  // The identity columns are deliberately absent from this update. A retry
  // may refresh descriptive metadata, but it can never turn one model ID into
  // another checkpoint or silently rewrite its fingerprint.
  run(db, `UPDATE model_versions SET
    status = 'ready',
    architecture_json = CASE WHEN ? <> '{}' THEN ? ELSE architecture_json END,
    dataset_id = COALESCE(dataset_id, ?),
    training_run_id = COALESCE(training_run_id, ?),
    checkpoint_artifact_id = COALESCE(checkpoint_artifact_id, ?),
    embedding_dimensions_json = CASE WHEN ? <> '{}' THEN ? ELSE embedding_dimensions_json END,
    supported_heads_json = CASE WHEN ? <> '[]' THEN ? ELSE supported_heads_json END,
    evaluation_json = CASE WHEN ? <> '{}' THEN ? ELSE evaluation_json END
    WHERE id = ?`, [
    values.architectureJson,
    values.architectureJson,
    values.datasetId,
    values.trainingRunId,
    values.checkpointArtifactId,
    values.embeddingDimensionsJson,
    values.embeddingDimensionsJson,
    values.supportedHeadsJson,
    values.supportedHeadsJson,
    values.evaluationJson,
    values.evaluationJson,
    modelId
  ]);
}

function registerModelVersion(db, {
  requestedModelId,
  fingerprint: modelFingerprint,
  version,
  architecture,
  datasetId,
  trainingRunId,
  checkpointArtifactId,
  embeddingDimensions,
  supportedHeads,
  evaluation
}) {
  const requested = safeIdentifier(requestedModelId, "modelId");
  const existingByFingerprint = one(db, "SELECT * FROM model_versions WHERE fingerprint = ?", [modelFingerprint]);
  let modelId = requested || generatedModelId(modelFingerprint);

  // Content identity wins over a duplicate producer-side label. This keeps
  // repeated scans from creating aliases for the same immutable checkpoint.
  if (existingByFingerprint) modelId = existingByFingerprint.id;

  let existing = one(db, "SELECT * FROM model_versions WHERE id = ?", [modelId]);
  if (existing && existing.fingerprint !== modelFingerprint) {
    // Legacy outputs sometimes reused a human model ID for a new checkpoint.
    // Preserve the old row and give the new immutable content a deterministic
    // ID instead of overwriting the old fingerprint.
    modelId = generatedModelId(modelFingerprint);
    existing = one(db, "SELECT * FROM model_versions WHERE id = ?", [modelId]);
  }

  const values = {
    fingerprint: modelFingerprint,
    architectureJson: json(architecture),
    datasetId,
    trainingRunId,
    checkpointArtifactId,
    embeddingDimensionsJson: json(embeddingDimensions),
    supportedHeadsJson: json(supportedHeads),
    evaluationJson: json(evaluation)
  };
  if (existing) {
    updateIndexedModel(db, modelId, values);
  } else {
    run(db, `INSERT INTO model_versions(
      id, version, fingerprint, status, architecture_json, dataset_id, training_run_id,
      checkpoint_artifact_id, embedding_dimensions_json, supported_heads_json,
      evaluation_json, created_at
    ) VALUES (?, ?, ?, 'ready', ?, ?, ?, ?, ?, ?, ?, ?)`, [
      modelId,
      version,
      modelFingerprint,
      values.architectureJson,
      datasetId,
      trainingRunId,
      checkpointArtifactId,
      values.embeddingDimensionsJson,
      values.supportedHeadsJson,
      values.evaluationJson,
      now()
    ]);
  }
  return one(db, "SELECT * FROM model_versions WHERE id = ?", [modelId]);
}

function inside(root, candidate) {
  const absoluteRoot = resolve(root);
  const absoluteCandidate = resolve(candidate);
  return absoluteCandidate === absoluteRoot || absoluteCandidate.startsWith(`${absoluteRoot}${sep}`);
}

/** Index only artifacts that carry the explicit v1 manifest contract. */
async function syncExplicitArtifactManifest(db, root, manifestPath) {
  const manifest = await readJson(manifestPath);
  if (!manifest) return null;
  try {
    validateArtifactManifest(manifest);
  } catch {
    return null;
  }
  const artifactRoot = dataRoot(root);
  const files = (manifest.files || []).map((file) => ({ ...file }));
  if (!files.length) return null;
  const resolvedFiles = [];
  for (const file of files) {
    if (typeof file.sha256 !== "string" || !/^[a-f0-9]{64}$/i.test(file.sha256) ||
        !Number.isSafeInteger(file.sizeBytes) || file.sizeBytes < 0) return null;
    const candidates = [
      resolve(artifactRoot, file.path),
      resolve(root, file.path),
      resolve(dirname(manifestPath), file.path)
    ];
    let resolvedFile = null;
    for (const candidate of candidates) {
      if (!inside(artifactRoot, candidate) && !inside(root, candidate)) continue;
      try {
        const info = await stat(candidate);
        if (!info.isFile() || Number(info.size) !== Number(file.sizeBytes)) continue;
        // Lexical containment is not enough when a malicious or stale
        // manifest points at a symlink. Resolve the target before hashing it.
        const real = await realpath(candidate);
        if (!inside(artifactRoot, real) && !inside(root, real)) continue;
        const digest = createHash("sha256").update(await readFile(real)).digest("hex");
        if (digest !== String(file.sha256).toLowerCase()) continue;
        resolvedFile = real;
        break;
      } catch {
        // Try the next supported manifest-root interpretation. A manifest is
        // indexed only if one interpretation produces a complete valid file.
      }
    }
    if (!resolvedFile) return null;
    resolvedFiles.push(resolvedFile);
  }
  const primaryPath = resolvedFiles[0] || manifestPath;
  const artifact = await ensureArtifact(db, root, {
    artifactId: manifest.artifactId,
    kind: manifest.kind,
    path: primaryPath,
    fingerprintValue: manifest.fingerprint,
    sha256: manifest.sha256,
    schemaVersion: manifest.schemaVersion,
    producingRunId: manifest.producingRunId || null,
    gitCommit: manifest.gitCommit || "",
    configuration: manifest.configuration || {},
    files,
    metadata: { ...(manifest.metadata || {}), manifestPath: relativePath(root, manifestPath) }
  });
  return { artifact, inputs: manifest.inputs || [] };
}

async function syncExplicitArtifactManifests(db, root, files) {
  const entries = [];
  for (const path of files.filter((candidate) => basename(candidate) === "artifact-manifest.json" || basename(candidate).endsWith(".artifact-manifest.json"))) {
    const entry = await syncExplicitArtifactManifest(db, root, path);
    if (entry) entries.push(entry);
  }
  for (const entry of entries) {
    const dependencies = entry.inputs
      .map((input) => one(db, "SELECT id FROM artifacts WHERE id = ? OR fingerprint = ? LIMIT 1", [input.artifactId, input.fingerprint]))
      .filter(Boolean)
      .map((row) => row.id);
    connectArtifacts(db, entry.artifact.id, dependencies);
  }
  return entries.length;
}

async function syncDatasetManifestFile(db, root, path) {
  const manifest = await readJson(path);
  if (!manifest) return null;
  try {
    validateDatasetManifest(manifest);
  } catch {
    return null;
  }
  const artifact = await ensureArtifact(db, root, {
    kind: "dataset-manifest",
    path,
    fingerprintValue: manifest.fingerprint,
    sha256: manifest.sha256 || null,
    schemaVersion: manifest.schemaVersion,
    producingRunId: manifest.producingRunId || null,
    gitCommit: manifest.gitCommit || "",
    configuration: manifest.configuration || {},
    files: manifest.files || [{ path: relativePath(root, path) }],
    metadata: { datasetId: manifest.datasetId, featureSchema: manifest.featureSchema }
  });
  run(db, `INSERT INTO datasets(id, fingerprint, status, manifest_path, feature_schema, snapshot_ids_json, split_json, objective_counts_json, quality_json, created_at, updated_at)
    VALUES (?, ?, 'ready', ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET fingerprint = excluded.fingerprint, manifest_path = excluded.manifest_path,
      feature_schema = excluded.feature_schema, snapshot_ids_json = excluded.snapshot_ids_json,
      split_json = excluded.split_json, objective_counts_json = excluded.objective_counts_json,
      quality_json = excluded.quality_json, updated_at = excluded.updated_at`, [
    manifest.datasetId,
    manifest.fingerprint,
    relativePath(root, path),
    manifest.featureSchema,
    json(manifest.snapshotIds),
    json(manifest.split),
    json(manifest.objectives),
    json(manifest.quality || {}),
    manifest.createdAt || now(),
    now()
  ]);
  connectArtifacts(db, artifact.id, (manifest.inputArtifacts || []).map((input) =>
    one(db, "SELECT id FROM artifacts WHERE id = ? OR fingerprint = ? LIMIT 1", [input.artifactId, input.fingerprint])?.id
  ));
  return manifest.datasetId;
}

async function syncInferenceResultFile(db, root, path, snapshotById) {
  const result = await readJson(path);
  if (!result) return null;
  try {
    validateInferenceResult(result);
  } catch {
    return null;
  }
  if (!snapshotById.has(result.snapshotId) || !one(db, "SELECT id FROM model_versions WHERE id = ?", [result.modelId])) {
    return null;
  }
  const sha256 = await fileSha256(path);
  const indexed = indexedArtifactForPath(db, root, path, ["inference-result"]);
  const artifact = indexed?.sha256?.toLowerCase() === sha256
    ? indexed
    : await ensureArtifact(db, root, {
      kind: "inference-result",
      path,
      fingerprintValue: fingerprint("inference-result-content-v1", { sha256 }),
      sha256,
      schemaVersion: result.schemaVersion,
      files: [{ path: relativePath(root, path), sha256 }],
      metadata: { inferenceId: result.inferenceId || null, modelId: result.modelId, snapshotId: result.snapshotId }
    });
  const inferenceId = result.inferenceId || `inference-${result.modelId}-${result.snapshotId}`;
  run(db, `INSERT INTO inference_sets(id, fingerprint, model_id, snapshot_id, status, criticality_artifact_id, config_json, created_at)
    VALUES (?, ?, ?, ?, 'ready', ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET fingerprint = excluded.fingerprint, criticality_artifact_id = excluded.criticality_artifact_id,
      config_json = excluded.config_json, status = 'ready'`, [
    inferenceId,
    fingerprint("inference-set-v1", { inferenceId, modelId: result.modelId, snapshotId: result.snapshotId }),
    result.modelId,
    result.snapshotId,
    artifact.id,
    json({ metricNames: result.metricNames, source: relativePath(root, path) }),
    now()
  ]);
  for (const prediction of result.predictions) {
    const lineIndex = Number(prediction.line);
    if (!Number.isInteger(lineIndex)) continue;
    const lineInstanceId = `${result.snapshotId}:${lineIndex}`;
    if (!one(db, "SELECT id FROM line_instances WHERE id = ?", [lineInstanceId])) continue;
    const values = Object.fromEntries(result.metricNames.map((name, index) => [name, prediction.metrics[index]]));
    if (prediction.metricPercentiles) {
      values.percentiles = Object.fromEntries(result.metricNames.map((name, index) => [name, prediction.metricPercentiles[index]]));
    }
    run(db, `INSERT INTO criticality_predictions(inference_id, line_instance_id, primary_score, uncertainty, values_json, created_at)
      VALUES (?, ?, ?, ?, ?, ?)
      ON CONFLICT(inference_id, line_instance_id) DO UPDATE SET primary_score = excluded.primary_score,
        uncertainty = excluded.uncertainty, values_json = excluded.values_json`, [
      inferenceId,
      lineInstanceId,
      Number(values.accessibility_auc_loss ?? prediction.metrics[0] ?? 0),
      null,
      json({ ...values, structuralUniqueness: prediction.structuralUniqueness }),
      now()
    ]);
  }
  return inferenceId;
}

async function syncEvaluationResultFile(db, root, path) {
  const result = await readJson(path);
  if (!result) return null;
  try {
    validateEvaluationResult(result);
  } catch {
    return null;
  }
  const dataset = one(db, "SELECT id, fingerprint FROM datasets WHERE id = ? AND status = 'ready'", [result.datasetId]);
  if (!dataset) return null;
  if (dataset.fingerprint !== result.datasetFingerprint) return null;
  if (result.modelId !== undefined && result.modelId !== null &&
      !one(db, "SELECT id FROM model_versions WHERE id = ? AND status = 'ready'", [result.modelId])) {
    return null;
  }
  const sha256 = await fileSha256(path);
  const indexed = indexedArtifactForPath(db, root, path, ["evaluation-result"]);
  const artifact = indexed?.sha256?.toLowerCase() === sha256
    ? indexed
    : await ensureArtifact(db, root, {
      kind: "evaluation-result",
      path,
      fingerprintValue: fingerprint("evaluation-result-content-v1", { sha256 }),
      sha256,
      schemaVersion: result.schemaVersion,
      producingRunId: result.producingRunId || null,
      files: [{ path: relativePath(root, path), sha256 }],
      metadata: {
        datasetId: result.datasetId,
        datasetFingerprint: result.datasetFingerprint,
        modelId: result.modelId || null,
        split: result.split
      }
    });
  const evaluationId = `evaluation-${artifact.id}`;
  run(db, `INSERT INTO evaluation_results(
    id, fingerprint, artifact_id, dataset_id, model_id, split, top_k, report_json, created_at
  ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
  ON CONFLICT(id) DO UPDATE SET fingerprint = excluded.fingerprint,
    artifact_id = excluded.artifact_id, dataset_id = excluded.dataset_id,
    model_id = excluded.model_id, split = excluded.split, top_k = excluded.top_k,
    report_json = excluded.report_json, created_at = excluded.created_at`, [
    evaluationId,
    fingerprint("indexed-evaluation-v1", { artifactId: artifact.id, result }),
    artifact.id,
    result.datasetId,
    result.modelId || null,
    result.split,
    Number(result.topK),
    json(result),
    result.createdAt || now()
  ]);
  run(db, "DELETE FROM metric_points WHERE evaluation_id = ?", [evaluationId]);
  for (const metric of result.metrics) {
    for (const [metricName, rawValue] of Object.entries(metric.values || {})) {
      if (rawValue === null || rawValue === undefined || typeof rawValue !== "number" || !Number.isFinite(rawValue)) continue;
      run(db, `INSERT INTO metric_points(
        run_id, model_id, dataset_id, evaluation_id, name, value, split, dimensions_json, created_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`, [
        artifact.producing_run_id || null,
        result.modelId || null,
        result.datasetId,
        evaluationId,
        `evaluation.${metric.baseline}.${metricName}`,
        rawValue,
        result.split,
        json({
          evaluationId,
          artifactId: artifact.id,
          baseline: metric.baseline,
          metricName,
          split: result.split
        }),
        result.createdAt || now()
      ]);
    }
  }
  return evaluationId;
}

function benchmarkRows(result) {
  const reports = Array.isArray(result.reports) && result.reports.length
    ? result.reports
    : [result];
  return reports.map((report, index) => {
    const value = report && typeof report === "object" && !Array.isArray(report) ? report : {};
    const workload = value.workload || result.workload;
    const throughput = value.throughput ?? value.queriesPerSecond ?? value.stepsPerSecond ?? result.throughput;
    if (typeof throughput !== "number" || !Number.isFinite(throughput) || throughput < 0) return null;
    const throughputUnit = value.throughputUnit || result.throughputUnit ||
      (workload === "routing" ? "queries_per_second" : "steps_per_second");
    return {
      index,
      benchmark: value.benchmark || result.benchmark,
      workload,
      snapshotId: value.snapshotId ?? result.snapshotId ?? null,
      graphId: value.graphId ?? result.graphId ?? null,
      threadCount: value.threadCount ?? value.threads ?? result.threadCount ?? null,
      warmupUnits: value.warmupUnits ?? value.warmupQueries ?? value.warmupSteps ?? result.warmupUnits ?? 0,
      measuredUnits: value.measuredUnits ?? value.measuredQueries ?? value.measuredSteps ?? result.measuredUnits ?? 0,
      estimatedWorkUnits: value.estimatedWorkUnits ?? result.estimatedWorkUnits ?? null,
      medianMilliseconds: value.medianMilliseconds ?? result.medianMilliseconds ?? null,
      p95Milliseconds: value.p95Milliseconds ?? result.p95Milliseconds ?? null,
      throughput,
      throughputUnit,
      peakResidentMemoryBytes: value.peakResidentMemoryBytes ?? result.peakResidentMemoryBytes ?? null,
      graphCounts: value.graphCounts || result.graphCounts || {},
      runtime: value.runtime || result.runtime || {},
      threadConfiguration: value.threadConfiguration || result.threadConfiguration || {},
      report: value
    };
  }).filter(Boolean);
}

async function syncBenchmarkResultFile(db, root, path) {
  const result = await readJson(path);
  if (!result) return null;
  try {
    validateBenchmarkResult(result);
  } catch {
    return null;
  }
  const rows = benchmarkRows(result);
  if (!rows.length) return null;
  const artifact = indexedArtifactForPath(db, root, path, ["benchmark-result"]) || await ensureArtifact(db, root, {
    kind: "benchmark-result",
    path,
    fingerprintValue: fingerprint("benchmark-result-v1", result),
    schemaVersion: result.schemaVersion,
    producingRunId: result.producingRunId || null,
    files: [{ path: relativePath(root, path) }],
    metadata: {
      benchmark: result.benchmark,
      workload: result.workload,
      snapshotId: result.snapshotId || null,
      graphId: result.graphId || null
    }
  });
  run(db, "DELETE FROM benchmarks WHERE artifact_id = ?", [artifact.id]);
  for (const row of rows) {
    const fingerprintValue = fingerprint("indexed-benchmark-v1", {
      artifactId: artifact.id,
      index: row.index,
      row: row.report,
      result
    });
    const id = `benchmark-${artifact.id}-${row.index}`;
    run(db, `INSERT INTO benchmarks(
      id, fingerprint, artifact_id, run_id, benchmark, workload, snapshot_id, graph_id,
      thread_count, warmup_units, measured_units, estimated_work_units,
      median_milliseconds, p95_milliseconds, throughput, throughput_unit,
      peak_resident_memory_bytes, graph_counts_json, runtime_json,
      thread_configuration_json, report_json, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(fingerprint) DO UPDATE SET
      artifact_id = excluded.artifact_id, run_id = excluded.run_id,
      benchmark = excluded.benchmark, workload = excluded.workload,
      snapshot_id = excluded.snapshot_id, graph_id = excluded.graph_id,
      thread_count = excluded.thread_count, warmup_units = excluded.warmup_units,
      measured_units = excluded.measured_units, estimated_work_units = excluded.estimated_work_units,
      median_milliseconds = excluded.median_milliseconds, p95_milliseconds = excluded.p95_milliseconds,
      throughput = excluded.throughput, throughput_unit = excluded.throughput_unit,
      peak_resident_memory_bytes = excluded.peak_resident_memory_bytes,
      graph_counts_json = excluded.graph_counts_json, runtime_json = excluded.runtime_json,
      thread_configuration_json = excluded.thread_configuration_json,
      report_json = excluded.report_json, created_at = excluded.created_at`, [
      id,
      fingerprintValue,
      artifact.id,
      artifact.producing_run_id || null,
      row.benchmark,
      row.workload,
      row.snapshotId,
      row.graphId,
      row.threadCount === null || row.threadCount === undefined ? null : Number(row.threadCount),
      Number(row.warmupUnits || 0),
      Number(row.measuredUnits || 0),
      row.estimatedWorkUnits === null || row.estimatedWorkUnits === undefined ? null : Number(row.estimatedWorkUnits),
      row.medianMilliseconds === null || row.medianMilliseconds === undefined ? null : Number(row.medianMilliseconds),
      row.p95Milliseconds === null || row.p95Milliseconds === undefined ? null : Number(row.p95Milliseconds),
      Number(row.throughput),
      String(row.throughputUnit),
      row.peakResidentMemoryBytes === null || row.peakResidentMemoryBytes === undefined ? null : Number(row.peakResidentMemoryBytes),
      json(row.graphCounts),
      json(row.runtime),
      json(row.threadConfiguration),
      json(row.report),
      result.createdAt || now()
    ]);
  }
  return rows.length;
}

function connectArtifacts(db, artifactId, dependencies) {
  for (const dependency of dependencies.filter(Boolean)) {
    run(db, `INSERT OR IGNORE INTO artifact_dependencies(artifact_id, depends_on_artifact_id, relation)
      VALUES (?, ?, 'input')`, [artifactId, dependency]);
  }
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
  const rootData = dataRoot(root);
  const candidates = [
    join(snapshotPath, "..", "graph"),
    join(rootData, "graphs", networkId),
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
  const first = parseJson(lines[0], {});
  const snapshotId = first?.snapshot;
  const routerAlgorithmVersion = first?.router_algorithm_version;
  if (routerAlgorithmVersion !== ROUTER_ALGORITHM_VERSION) return;
  if (!snapshotId || !snapshotById.has(snapshotId)) return;
  const artifact = indexedArtifactForPath(db, root, path, ["criticality-labels"]) || await ensureArtifact(db, root, {
    kind: "criticality-labels",
    path,
    metadata: { snapshotId, rows: lines.length, routerAlgorithmVersion }
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

async function syncModelCheckpointFile(db, root, modelPath, identityHint = null) {
  const checkpoint = await readJson(modelPath);
  const modelInfo = record(checkpoint?.report);
  const indexedArtifact = indexedArtifactForPath(db, root, modelPath, ["model-checkpoint"]);
  const sha256 = await fileSha256(modelPath);
  const modelInfoOnDisk = await fileInfo(modelPath);
  const modelHint = modelRunHint(db, root, modelPath);
  const checkpointArtifact = indexedArtifact?.sha256?.toLowerCase() === sha256
    || explicitArtifactFileForPath(indexedArtifact, root, modelPath, sha256, modelInfoOnDisk.size)
    ? indexedArtifact
    : await ensureArtifact(db, root, {
      kind: "model-checkpoint",
      path: modelPath,
      fingerprintValue: fingerprint("model-checkpoint-content-v1", { sha256 }),
      sha256,
      producingRunId: modelHint.runId,
      metadata: {
        backend: modelInfo.backend || checkpoint?.backend || "reference-cpu",
        source: "filesystem",
        ...(modelHint.runId ? { trainingRunId: modelHint.runId } : {}),
        ...(checkpoint?.datasetFingerprint ? { datasetFingerprint: checkpoint.datasetFingerprint } : {})
      }
    });
  const inferredHint = {
    ...(identityHint || {}),
    ...(modelHint.runId ? { trainingRunId: modelHint.runId } : {}),
    ...(modelHint.run?.dataset_id ? { datasetId: modelHint.run.dataset_id } : {})
  };
  const sources = modelLineageSources(checkpoint, checkpointArtifact, inferredHint);
  const requestedModelId = firstValue(sources, ["modelId", "model_id"]);
  const modelFingerprint = checkpointArtifact.fingerprint;
  const datasetId = referencedId(db, "datasets", safeIdentifier(
    firstValue(sources, ["datasetId", "dataset_id"]),
    "datasetId"
  ));
  const trainingRunId = referencedId(db, "runs", safeIdentifier(
    firstValue(sources, ["trainingRunId", "training_run_id", "producingRunId", "producing_run_id", "runId", "run_id"]),
    "trainingRunId"
  ) || checkpointArtifact.producing_run_id);
  const version = firstValue(sources, ["version", "modelVersion", "model_version"]) ||
    `checkpoint-${String(modelFingerprint).slice(0, 12)}`;
  const model = registerModelVersion(db, {
    requestedModelId,
    fingerprint: modelFingerprint,
    version,
    architecture: modelArchitecture(checkpoint, modelInfo),
    datasetId,
    trainingRunId,
    checkpointArtifactId: checkpointArtifact.id,
    embeddingDimensions: modelEmbeddingDimensions(checkpoint),
    supportedHeads: modelSupportedHeads(checkpoint),
    evaluation: record(checkpoint?.evaluation)
  });
  if (!model) return null;
  run(db, `INSERT INTO model_aliases(alias, model_id, updated_at) VALUES ('candidate', ?, ?)
    ON CONFLICT(alias) DO UPDATE SET model_id = excluded.model_id, updated_at = excluded.updated_at`, [model.id, now()]);
  return { model, artifact: checkpointArtifact, checkpoint };
}

async function syncModelAndPredictions(db, root, modelPath, predictionsPath, snapshotById) {
  const predictions = await readJson(predictionsPath);
  if (!predictions) return null;
  const snapshotId = predictions.snapshot_id ?? predictions.snapshotId;
  const snapshotKnown = snapshotId && (snapshotById.has(snapshotId) || one(db, "SELECT id FROM snapshots WHERE id = ?", [snapshotId]));
  if (!snapshotKnown) return null;
  const modelResult = await syncModelCheckpointFile(db, root, modelPath, predictions);
  if (!modelResult) return null;
  const modelId = modelResult.model.id;
  const metricNames = predictions.metric_names ?? predictions.metricNames ?? [];
  const criticalityArtifact = indexedArtifactForPath(db, root, predictionsPath, ["inference-result", "criticality-predictions"]) || await ensureArtifact(db, root, {
    kind: "criticality-predictions",
    path: predictionsPath,
    fingerprintValue: fingerprint("prediction-file-v1", predictions),
    metadata: { snapshotId, modelId, rows: predictions.predictions?.length || 0 }
  });
  const inferenceId = predictions.inferenceId || predictions.inference_id || `inference-${modelId}-${snapshotId}`;
  const inferenceFingerprint = fingerprint("inference-v1", { modelId, snapshotId });
  run(db, `INSERT INTO inference_sets(id, fingerprint, model_id, snapshot_id, status, criticality_artifact_id, config_json, created_at)
    VALUES (?, ?, ?, ?, 'ready', ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET fingerprint = excluded.fingerprint, model_id = excluded.model_id,
      snapshot_id = excluded.snapshot_id, criticality_artifact_id = excluded.criticality_artifact_id,
      config_json = excluded.config_json, status = 'ready'`, [
    inferenceId,
    inferenceFingerprint,
    modelId,
    snapshotId,
    criticalityArtifact.id,
    json({ source: relativePath(root, predictionsPath), metricNames }),
    now()
  ]);
  for (const prediction of predictions.predictions || []) {
    const lineIndex = Number(prediction.line ?? prediction.lineId);
    if (!Number.isInteger(lineIndex)) continue;
    const lineInstanceId = `${snapshotId}:${lineIndex}`;
    if (!one(db, "SELECT id FROM line_instances WHERE id = ?", [lineInstanceId])) continue;
    const values = Object.fromEntries(metricNames.map((name, index) => [name, Number(prediction.metrics?.[index] ?? 0)]));
    const metricPercentiles = prediction.metricPercentiles ?? prediction.metric_percentiles;
    if (Array.isArray(metricPercentiles)) {
      values.percentiles = Object.fromEntries(metricNames.map((name, index) => [name, Number(metricPercentiles[index] ?? 0)]));
    }
    const structuralUniqueness = Number(prediction.structuralUniqueness ?? prediction.structural_uniqueness ?? 0);
    run(db, `INSERT INTO criticality_predictions(inference_id, line_instance_id, primary_score, uncertainty, values_json, created_at)
      VALUES (?, ?, ?, ?, ?, ?)
      ON CONFLICT(inference_id, line_instance_id) DO UPDATE SET primary_score = excluded.primary_score,
        uncertainty = excluded.uncertainty, values_json = excluded.values_json`, [
      inferenceId,
      lineInstanceId,
      Number(values.accessibility_auc_loss ?? values[metricNames[0]] ?? 0),
      null,
      json({ ...values, structuralUniqueness, structural_uniqueness: structuralUniqueness }),
      now()
    ]);
  }
  return { modelId, inferenceId };
}

export async function syncFilesystem(db, root = repositoryRoot()) {
  const projectId = ensureProject(db);
  const rootData = dataRoot(root);
  const files = await walkFiles(rootData);
  const explicitArtifacts = await syncExplicitArtifactManifests(db, root, files);
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

  // Datasets and models are prerequisites for evaluation and inference
  // results. Index them before their consumers so a single refresh is
  // complete even when all files arrived together.
  for (const path of files.filter((candidate) => basename(candidate) === "dataset-manifest.json")) {
    await syncDatasetManifestFile(db, root, path);
  }

  const modelPaths = files.filter((path) => basename(path) === "model.json");
  const predictionPaths = files.filter((path) => basename(path) === "predictions.json");
  const pairedModelPaths = new Set();
  for (const predictionsPath of predictionPaths) {
    const directory = dirname(predictionsPath);
    const modelPath = modelPaths.find((candidate) => dirname(candidate) === directory);
    if (modelPath) {
      pairedModelPaths.add(modelPath);
      await syncModelAndPredictions(db, root, modelPath, predictionsPath, snapshots);
    }
  }
  for (const modelPath of modelPaths) {
    if (!pairedModelPaths.has(modelPath)) await syncModelCheckpointFile(db, root, modelPath);
  }

  for (const path of files.filter((candidate) => basename(candidate) === "inference-result.json")) {
    await syncInferenceResultFile(db, root, path, snapshots);
  }

  // Evaluation results refer to indexed datasets/models, so index their
  // prerequisites first even when all files arrived in one filesystem scan.
  for (const path of files.filter((candidate) => basename(candidate) === "evaluation.json" || basename(candidate) === "evaluation-result.json")) {
    await syncEvaluationResultFile(db, root, path);
  }

  for (const path of files.filter((candidate) => basename(candidate) === "benchmark.json" ||
      basename(candidate) === "benchmark-result.json" || basename(candidate).endsWith(".benchmark.json"))) {
    await syncBenchmarkResultFile(db, root, path);
  }

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
    evaluations: count("evaluation_results"),
    benchmarks: count("benchmarks"),
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
    producingRunId: row.producing_run_id,
    gitCommit: row.git_commit,
    configuration: parseJson(row.configuration_json),
    files: parseJson(row.files_json, []),
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
