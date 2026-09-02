import { hostname } from "node:os";
import { mkdir, stat } from "node:fs/promises";
import { dirname, join, relative, resolve, sep } from "node:path";
import {
  fingerprint,
  RUN_EVENT_SCHEMA_VERSION,
  validateArtifactManifest
} from "../../../packages/contracts/src/index.js";
import {
  addRunLog,
  all,
  appendRunEvent,
  appendRunEventType,
  claimNextRun,
  createDatabase,
  dataRoot,
  getRun,
  json,
  now,
  one,
  repositoryRoot,
  run as sqlRun,
  updateRun
} from "../../api/src/db.js";
import { syncFilesystem } from "../../api/src/inventory.js";
import { readStructuredEvents } from "./event-parser.js";

const ROOT = repositoryRoot();
const WORKER_ID = process.env.TRANSIT_LAB_WORKER_ID || `${hostname()}-${process.pid}`;
const SLEEP_MS = 750;

function sleep(milliseconds) {
  return new Promise((resolvePromise) => setTimeout(resolvePromise, milliseconds));
}

function insideRoot(rootPath, path) {
  const root = resolve(rootPath);
  const candidate = resolve(path);
  return candidate === root || candidate.startsWith(`${root}${sep}`);
}

function absolute(rootPath, relativePath) {
  const path = resolve(rootPath, relativePath);
  if (!insideRoot(rootPath, path)) throw new Error("artifact path escapes repository root");
  return path;
}

function relativePath(rootPath, path) {
  const result = relative(rootPath, resolve(path)).split(sep).join("/");
  if (!result || result.startsWith("../") || result === "..") {
    throw new Error("artifact path is outside the repository root");
  }
  return result;
}

function fixedRustCommand(rootPath, args) {
  const release = resolve(rootPath, "target/release/transit-cli");
  const debug = resolve(rootPath, "target/debug/transit-cli");
  if (Bun.file(release).size > 0) return [release, ...args];
  if (Bun.file(debug).size > 0) return [debug, ...args];
  return ["cargo", "run", "--quiet", "-p", "transit-cli", "--", ...args];
}

function lineFromBytes(bytes) {
  return new TextDecoder().decode(bytes);
}

async function consumeOutput(stream, db, runId, streamName) {
  if (!stream) return;
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let pending = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      pending += decoder.decode(value, { stream: true });
      const lines = pending.split(/\r?\n/);
      pending = lines.pop() || "";
      for (const line of lines) {
        if (line) addRunLog(db, runId, streamName, line);
      }
    }
    pending += decoder.decode();
    if (pending) addRunLog(db, runId, streamName, pending);
  } catch (error) {
    addRunLog(db, runId, streamName, `output reader error: ${error.message}`);
  } finally {
    reader.releaseLock();
  }
}

function stepStart(db, runId, step) {
  sqlRun(db, `INSERT INTO run_steps(run_id, step, status, started_at, metrics_json)
    VALUES (?, ?, 'running', ?, '{}')
    ON CONFLICT(run_id, step) DO UPDATE SET status = 'running', started_at = excluded.started_at`, [runId, step, now()]);
  updateRun(db, runId, { currentStep: step, status: "running", startedAt: one(db, "SELECT started_at FROM runs WHERE id = ?", [runId])?.started_at || now() });
  appendRunEventType(db, runId, "step.started", { step });
}

function stepComplete(db, runId, step, outputFingerprint = null) {
  sqlRun(db, "UPDATE run_steps SET status = 'succeeded', finished_at = ?, output_fingerprint = ? WHERE run_id = ? AND step = ?", [now(), outputFingerprint, runId, step]);
  appendRunEventType(db, runId, "step.completed", { step });
}

function stepFailed(db, runId, step) {
  sqlRun(db, "UPDATE run_steps SET status = 'failed', finished_at = ? WHERE run_id = ? AND step = ? AND status = 'running'", [now(), runId, step]);
}

function appendProgress(db, runId, payload) {
  const event = appendRunEventType(db, runId, "progress", payload);
  updateRun(db, runId, {
    progressCompleted: payload.completed,
    progressTotal: payload.total,
    progressUnit: payload.unit
  });
  return event;
}

function appendRustEvent(db, runId, event) {
  const stored = appendRunEvent(db, runId, event);
  if (event.type === "progress") {
    updateRun(db, runId, {
      progressCompleted: event.completed,
      progressTotal: event.total,
      progressUnit: event.unit
    });
  }
  return stored;
}

async function fileSha256(path) {
  const hasher = new Bun.CryptoHasher("sha256");
  hasher.update(await Bun.file(path).arrayBuffer());
  return hasher.digest("hex");
}

function currentGitCommit(rootPath) {
  try {
    const result = Bun.spawnSync(["git", "rev-parse", "--short", "HEAD"], { cwd: rootPath });
    return lineFromBytes(result.stdout).trim() || "unknown";
  } catch {
    return "unknown";
  }
}

async function fileArtifact(db, rootPath, runId, kind, path, metadata = {}, inputArtifactIds = []) {
  const info = await stat(path);
  const relativeUri = relativePath(rootPath, path);
  const sha256 = await fileSha256(path);
  const stableMetadata = Object.fromEntries(
    Object.entries(metadata).filter(([key]) => key !== "runId" && key !== "workerId")
  );
  const artifactFingerprint = fingerprint("worker-artifact-v1", {
    kind,
    sha256,
    metadata: stableMetadata,
    inputArtifactIds
  });
  const id = `artifact-${kind.replace(/[^A-Za-z0-9]+/g, "-")}-${artifactFingerprint.slice(0, 24)}`;
  const manifest = {
    schemaVersion: 1,
    artifactId: id,
    kind,
    fingerprint: artifactFingerprint,
    sha256,
    createdAt: now(),
    inputs: inputArtifactIds.map((artifactId) => ({ artifactId, fingerprint: artifactId })),
    producingRunId: runId,
    gitCommit: currentGitCommit(rootPath),
    configuration: { runId, kind },
    files: [{ path: relativeUri, sha256, sizeBytes: info.size }],
    metadata
  };
  validateArtifactManifest(manifest);
  const name = path.split(sep).pop() || "output";
  const manifestPath = resolve(dirname(path), `${name}.worker-artifact-manifest.json`);
  const existingManifest = await Bun.file(manifestPath).exists()
    ? await Bun.file(manifestPath).json()
    : null;
  if (existingManifest) {
    validateArtifactManifest(existingManifest);
    if (existingManifest.kind !== manifest.kind ||
        existingManifest.fingerprint !== manifest.fingerprint ||
        JSON.stringify(existingManifest.files) !== JSON.stringify(manifest.files)) {
      throw new Error(`refusing to overwrite immutable artifact manifest ${manifestPath}`);
    }
  } else {
    await Bun.write(manifestPath, JSON.stringify(manifest, null, 2));
  }
  sqlRun(db, `INSERT INTO artifacts(id, kind, fingerprint, uri, local_path, size_bytes, sha256, schema_version, producing_run_id, git_commit, configuration_json, files_json, status, metadata_json, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, 'ready', ?, ?)
    ON CONFLICT(fingerprint) DO UPDATE SET uri = excluded.uri, local_path = excluded.local_path,
      size_bytes = excluded.size_bytes, sha256 = excluded.sha256, status = 'ready'`, [
    id,
    kind,
    artifactFingerprint,
    relativeUri,
    relativeUri,
    info.size,
    sha256,
    runId,
    manifest.gitCommit,
    json(manifest.configuration),
    json(manifest.files),
    json(metadata),
    now()
  ]);
  for (const inputId of inputArtifactIds) {
    sqlRun(db, `INSERT OR IGNORE INTO artifact_dependencies(artifact_id, depends_on_artifact_id, relation) VALUES (?, ?, 'input')`, [id, inputId]);
  }
  appendRunEventType(db, runId, "artifact.created", { artifactId: id, artifactKind: kind, uri: relativeUri, sha256 });
  return { id, fingerprint: artifactFingerprint, uri: relativeUri, sha256, manifestPath };
}

function artifactForPath(db, localPath) {
  return one(db, "SELECT * FROM artifacts WHERE local_path = ? ORDER BY created_at DESC LIMIT 1", [localPath]);
}

async function executeRust(db, rootPath, run, step, args, expectedPaths = []) {
  stepStart(db, run.id, step);
  appendProgress(db, run.id, { step, completed: 0, total: 1, unit: "command" });
  const command = fixedRustCommand(rootPath, args);
  addRunLog(db, run.id, "system", `spawn ${JSON.stringify(command)}`);
  const eventPath = resolve(dataRoot(rootPath), "runs", run.id, "events.jsonl");
  await mkdir(dirname(eventPath), { recursive: true });
  const child = Bun.spawn(command, {
    cwd: rootPath,
    stdout: "pipe",
    stderr: "pipe",
    env: {
      ...globalThis.process.env,
      TRANSIT_LAB_ROOT: rootPath,
      TRANSIT_RUN_ID: run.id,
      TRANSIT_EVENT_FILE: eventPath
    }
  });
  let cancelTimer;
  let cancelled = false;
  const monitor = async () => {
    cancelTimer = setInterval(() => {
      const current = getRun(db, run.id);
      if (current?.cancelRequested && !cancelled) {
        cancelled = true;
        addRunLog(db, run.id, "system", "cancellation requested; terminating child process");
        child.kill();
      }
    }, 400);
  };
  await monitor();
  const outputPromise = Promise.all([
    consumeOutput(child.stdout, db, run.id, "stdout"),
    consumeOutput(child.stderr, db, run.id, "stderr")
  ]);
  const exitCode = await child.exited;
  await outputPromise;
  if (cancelTimer) clearInterval(cancelTimer);
  const structuredEvents = await readStructuredEvents(eventPath, run.id);
  for (const event of structuredEvents) appendRustEvent(db, run.id, event);
  if (cancelled || getRun(db, run.id)?.cancelRequested) {
    stepFailed(db, run.id, step);
    updateRun(db, run.id, { status: "cancelled", finishedAt: now(), currentStep: step });
    appendRunEventType(db, run.id, "run.cancelled");
    return { cancelled: true };
  }
  if (exitCode !== 0) {
    const message = `Rust command exited with status ${exitCode}`;
    stepFailed(db, run.id, step);
    updateRun(db, run.id, { status: "failed", finishedAt: now(), errorCode: "process_failed", errorMessage: message });
    appendRunEventType(db, run.id, "error", { code: "process_failed", message });
    appendRunEventType(db, run.id, "run.failed", { code: "process_failed", message });
    return { failed: true };
  }
  const artifacts = [];
  for (const expectedPath of expectedPaths) {
    if (await Bun.file(expectedPath).exists()) artifacts.push(await fileArtifact(db, rootPath, run.id, `${step}-output`, expectedPath, { runId: run.id, step }));
  }
  stepComplete(db, run.id, step, artifacts[0]?.fingerprint || null);
  appendProgress(db, run.id, { step, completed: 1, total: 1, unit: "command" });
  return { artifacts };
}

function networkForSnapshot(db, snapshotId) {
  const row = one(db, "SELECT * FROM snapshots WHERE id = ?", [snapshotId]);
  if (!row) throw new Error("snapshot does not exist");
  return row;
}

function feedForRevision(db, revisionId) {
  const row = one(db, "SELECT * FROM feed_revisions WHERE id = ?", [revisionId]);
  if (!row) throw new Error("feed revision does not exist");
  return row;
}

function safeConfigPath(rootPath, value) {
  const configured = typeof value === "string" && value.startsWith("configs/") ? value : "configs/models/multitask-v1.yaml";
  const path = absolute(rootPath, configured);
  if (!path.startsWith(resolve(rootPath, "configs") + sep)) throw new Error("model config must be inside configs/");
  return path;
}

async function runCompile(db, rootPath, run) {
  const feed = feedForRevision(db, run.spec.feedRevisionId);
  const network = one(db, "SELECT * FROM networks WHERE id = ?", [feed.network_id]);
  const output = resolve(dataRoot(rootPath), "snapshots", feed.network_id, `${run.spec.serviceDate}-${feed.sha256.slice(0, 12)}`);
  await mkdir(dirname(output), { recursive: true });
  const result = await executeRust(db, rootPath, run, "compile-snapshot", [
    "compile", "--input", absolute(rootPath, feed.local_path), "--service-date", run.spec.serviceDate, "--output", output
  ], [join(output, "manifest.json"), join(output, "network.json")]);
  if (!result.failed && !result.cancelled) {
    await syncFilesystem(db, rootPath);
    const snapshot = one(db, "SELECT id FROM snapshots WHERE manifest_path = ?", [relativePath(rootPath, join(output, "manifest.json"))]);
    if (snapshot) updateRun(db, run.id, { snapshotId: snapshot.id });
  }
  return result;
}

async function runSimulation(db, rootPath, run) {
  const snapshot = networkForSnapshot(db, run.spec.snapshotId);
  const output = resolve(dataRoot(rootPath), "labels", snapshot.network_id, `${snapshot.id}.jsonl`);
  await mkdir(dirname(output), { recursive: true });
  const result = await executeRust(db, rootPath, run, "simulate-criticality", [
    "labels", "line-removal", "--snapshot", absolute(rootPath, dirname(snapshot.network_path)), "--output", output
  ], [output]);
  if (!result.failed && !result.cancelled) await syncFilesystem(db, rootPath);
  return result;
}

function datasetManifest(db, spec) {
  const snapshots = spec.snapshotIds.map((id) => networkForSnapshot(db, id));
  const graphPaths = snapshots.map((snapshot) => snapshot.graph_path).filter(Boolean);
  if (graphPaths.length !== snapshots.length) throw new Error("every dataset snapshot needs a compiled graph");
  const labelCounts = snapshots.map((snapshot) => Number(one(db, "SELECT COUNT(*) AS count FROM criticality_labels WHERE snapshot_id = ?", [snapshot.id])?.count || 0));
  const counts = {
    snapshots: snapshots.length,
    graphSnapshots: graphPaths.length,
    criticalityLines: labelCounts.reduce((sum, value) => sum + value, 0),
    maskedReconstruction: graphPaths.length,
    crossSnapshotPairs: Math.max(0, snapshots.length - 1),
    roleTriplets: 0,
    serviceTriplets: 0,
    geometryTriplets: 0,
    resilienceTriplets: 0,
    humanComparisons: 0
  };
  const value = { featureSchema: spec.featureSchema, snapshotIds: spec.snapshotIds, split: spec.splitConfig, counts };
  const datasetFingerprint = fingerprint("dataset-manifest-v1", value);
  return {
    schemaVersion: 1,
    datasetId: `dataset-${datasetFingerprint.slice(0, 24)}`,
    fingerprint: datasetFingerprint,
    featureSchema: spec.featureSchema,
    snapshotIds: spec.snapshotIds,
    split: typeof spec.splitConfig === "object" ? spec.splitConfig : { strategy: spec.splitConfig },
    objectives: {
      maskedReconstruction: "pending",
      crossSnapshotIdentity: "pending",
      augmentationConsistency: "pending",
      facetMetricLearning: "pending",
      criticality: counts.criticalityLines > 0 ? "available" : "missing labels"
    },
    counts,
    inputs: snapshots.map((snapshot) => ({ snapshotId: snapshot.id, graphPath: snapshot.graph_path, labelRows: labelCounts[spec.snapshotIds.indexOf(snapshot.id)] }))
  };
}

async function runBuildDataset(db, rootPath, run) {
  const manifest = datasetManifest(db, run.spec);
  const output = resolve(dataRoot(rootPath), "datasets", manifest.datasetId, "manifest.json");
  await mkdir(dirname(output), { recursive: true });
  stepStart(db, run.id, "build-dataset");
  appendProgress(db, run.id, { step: "build-dataset", completed: 0, total: 1, unit: "manifest" });
  await Bun.write(output, JSON.stringify(manifest, null, 2));
  const artifact = await fileArtifact(db, rootPath, run.id, "dataset-manifest", output, { datasetId: manifest.datasetId, fingerprint: manifest.fingerprint });
  sqlRun(db, `INSERT INTO datasets(id, fingerprint, status, manifest_path, feature_schema, snapshot_ids_json, split_json, objective_counts_json, quality_json, created_at, updated_at)
    VALUES (?, ?, 'ready', ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET fingerprint = excluded.fingerprint, manifest_path = excluded.manifest_path,
      feature_schema = excluded.feature_schema, snapshot_ids_json = excluded.snapshot_ids_json,
      split_json = excluded.split_json, objective_counts_json = excluded.objective_counts_json,
      quality_json = excluded.quality_json, status = 'ready', updated_at = excluded.updated_at`, [
    manifest.datasetId,
    manifest.fingerprint,
    relativePath(rootPath, output),
    manifest.featureSchema,
    json(manifest.snapshotIds),
    json(manifest.split),
    json(manifest.counts),
    json({ leakageChecks: "pending", manifestFingerprint: manifest.fingerprint }),
    now(),
    now()
  ]);
  stepComplete(db, run.id, "build-dataset", artifact.fingerprint);
  appendProgress(db, run.id, { step: "build-dataset", completed: 1, total: 1, unit: "manifest" });
  updateRun(db, run.id, { datasetId: manifest.datasetId });
  return { artifacts: [artifact] };
}

function labelsPathForSnapshot(db, snapshotId) {
  return one(db, "SELECT local_path FROM artifacts WHERE kind = 'criticality-labels' AND metadata_json LIKE ? ORDER BY created_at DESC LIMIT 1", [`%${snapshotId}%`])?.local_path || null;
}

async function runTrain(db, rootPath, run) {
  const dataset = one(db, "SELECT * FROM datasets WHERE id = ?", [run.spec.datasetId]);
  if (!dataset) throw new Error("dataset does not exist");
  const snapshotIds = JSON.parse(dataset.snapshot_ids_json || "[]");
  const args = ["train", "multitask"];
  const snapshots = snapshotIds.map((id) => networkForSnapshot(db, id));
  for (const snapshot of snapshots) {
    if (!snapshot.graph_path) throw new Error(`snapshot ${snapshot.id} has no graph artifact`);
    args.push("--graph", absolute(rootPath, snapshot.graph_path));
  }
  const labelPaths = snapshots.map((snapshot) => labelsPathForSnapshot(db, snapshot.id));
  if (labelPaths.every(Boolean)) {
    for (const labels of labelPaths) args.push("--labels", absolute(rootPath, labels));
  } else if (labelPaths.some(Boolean)) {
    // The Rust CLI maps the nth --labels value to the nth --graph. Never
    // silently shift a later snapshot's labels onto the wrong graph.
    appendRunEventType(db, run.id, "warning", {
      code: "partial-label-set",
      message: "Some dataset snapshots have labels and others do not; training will run without labels until the set is complete."
    });
  }
  args.push("--config", safeConfigPath(rootPath, run.spec.modelConfig), "--output", resolve(dataRoot(rootPath), "models", run.id, "model.json"));
  const output = resolve(dataRoot(rootPath), "models", run.id, "model.json");
  await mkdir(dirname(output), { recursive: true });
  const result = await executeRust(db, rootPath, run, "train", args, [output]);
  if (!result.failed && !result.cancelled && await Bun.file(output).exists()) {
    const artifact = await fileArtifact(db, rootPath, run.id, "model-checkpoint", output, { datasetId: dataset.id, runId: run.id });
    const modelFingerprint = fingerprint("model-version-v1", { datasetId: dataset.id, runId: run.id, seed: run.spec.seed, artifact: artifact.fingerprint });
    const modelId = `model-${modelFingerprint.slice(0, 24)}`;
    sqlRun(db, `INSERT OR IGNORE INTO model_versions(id, version, fingerprint, status, architecture_json, dataset_id, training_run_id, checkpoint_artifact_id, embedding_dimensions_json, supported_heads_json, evaluation_json, created_at)
      VALUES (?, ?, ?, 'ready', ?, ?, ?, ?, ?, ?, ?, ?)`, [
      modelId,
      `run-${run.id.slice(-8)}`,
      modelFingerprint,
      json({ config: run.spec.modelConfig, backend: "reference-cpu-multitask" }),
      dataset.id,
      run.id,
      artifact.id,
      json({ base: 192, general: 64, role: 48, service: 32, geometry: 32, resilience: 32 }),
      json(["criticality", "reconstruction", "similarity"]),
      json({}),
      now()
    ]);
    updateRun(db, run.id, { modelId });
  }
  return result;
}

function rankMetric(predictions, labels, metric) {
  const values = predictions.map((prediction) => ({ prediction: Number(prediction[metric] || 0), target: Number(labels[prediction.line_index]?.[metric] || 0) }));
  if (values.length < 2) return null;
  const rank = (array) => [...array].sort((a, b) => a - b).map((value, index, sorted) => ({ value, rank: sorted.filter((candidate) => candidate <= value).length })).sort((a, b) => a.value - b.value).map((item) => item.rank);
  const predicted = rank(values.map((value) => value.prediction));
  const target = rank(values.map((value) => value.target));
  const mean = (array) => array.reduce((sum, value) => sum + value, 0) / array.length;
  const pm = mean(predicted); const tm = mean(target);
  const numerator = predicted.reduce((sum, value, index) => sum + (value - pm) * (target[index] - tm), 0);
  const denominator = Math.sqrt(predicted.reduce((sum, value) => sum + (value - pm) ** 2, 0) * target.reduce((sum, value) => sum + (value - tm) ** 2, 0));
  return denominator ? numerator / denominator : null;
}

async function runEvaluate(db, rootPath, run) {
  stepStart(db, run.id, "evaluate");
  const model = one(db, "SELECT * FROM model_versions WHERE id = ?", [run.spec.modelId]);
  const inference = one(db, "SELECT * FROM inference_sets WHERE model_id = ? ORDER BY created_at DESC LIMIT 1", [run.spec.modelId]);
  const metrics = [];
  if (inference) {
    const predictionRows = all(db, "SELECT * FROM criticality_predictions WHERE inference_id = ?", [inference.id]);
    const labels = Object.fromEntries(all(db, "SELECT line_index, values_json FROM criticality_labels WHERE snapshot_id = ?", [inference.snapshot_id]).map((row) => [row.line_index, JSON.parse(row.values_json || "{}")]));
    const predictionValues = predictionRows.map((row) => ({ line_index: Number(row.line_instance_id.split(":").pop()), ...JSON.parse(row.values_json || "{}") }));
    const spearman = rankMetric(predictionValues, labels, "accessibility_auc_loss");
    if (spearman !== null) {
      metrics.push({ name: "criticality_spearman", value: spearman, split: "available-labels", networkId: null });
      sqlRun(db, "INSERT INTO metric_points(model_id, name, value, split, dimensions_json, created_at) VALUES (?, ?, ?, ?, '{}', ?)", [run.spec.modelId, "criticality_spearman", spearman, "available-labels", now()]);
    }
  }
  const output = resolve(dataRoot(rootPath), "evaluations", `${run.id}.json`);
  await mkdir(dirname(output), { recursive: true });
  const report = { schemaVersion: 1, runId: run.id, modelId: run.spec.modelId, suite: run.spec.evaluationSuite, metrics, status: metrics.length ? "computed-on-available-labels" : "no-compatible-inference-labels" };
  await Bun.write(output, JSON.stringify(report, null, 2));
  const artifact = await fileArtifact(db, rootPath, run.id, "evaluation-report", output, { modelId: run.spec.modelId, metrics: metrics.length });
  stepComplete(db, run.id, "evaluate", artifact.fingerprint);
  appendRunEventType(db, run.id, "progress", { step: "evaluate", completed: 1, total: 1, unit: "report" });
  return { artifacts: [artifact], report, model };
}

async function runInfer(db, rootPath, run) {
  const model = one(db, "SELECT * FROM model_versions WHERE id = ?", [run.spec.modelId]);
  const checkpoint = model?.checkpoint_artifact_id ? one(db, "SELECT local_path FROM artifacts WHERE id = ?", [model.checkpoint_artifact_id]) : null;
  const snapshot = networkForSnapshot(db, run.spec.snapshotId);
  if (!checkpoint?.local_path || !snapshot.graph_path) throw new Error("inference requires a checkpoint and graph artifact");
  const output = resolve(dataRoot(rootPath), "inference", run.spec.modelId, run.spec.snapshotId, "predictions.json");
  await mkdir(dirname(output), { recursive: true });
  const result = await executeRust(db, rootPath, run, "infer", [
    "infer", "criticality", "--graph", absolute(rootPath, snapshot.graph_path), "--model", absolute(rootPath, checkpoint.local_path), "--output", output
  ], [output]);
  if (!result.failed && !result.cancelled) await syncFilesystem(db, rootPath);
  return result;
}

async function executeRun(db, rootPath, claimed) {
  const run = getRun(db, claimed.id);
  if (!run) return;
  updateRun(db, run.id, { status: "running", startedAt: now(), workerId: WORKER_ID });
  appendRunEventType(db, run.id, "run.started");
  try {
    let result;
    if (run.spec.kind === "compile-snapshot") result = await runCompile(db, rootPath, run);
    else if (run.spec.kind === "simulate-criticality") result = await runSimulation(db, rootPath, run);
    else if (run.spec.kind === "build-dataset") result = await runBuildDataset(db, rootPath, run);
    else if (run.spec.kind === "train") result = await runTrain(db, rootPath, run);
    else if (run.spec.kind === "evaluate") result = await runEvaluate(db, rootPath, run);
    else if (run.spec.kind === "infer") result = await runInfer(db, rootPath, run);
    else throw new Error(`unsupported worker run kind: ${run.spec.kind}`);
    if (result?.failed || result?.cancelled) return;
    updateRun(db, run.id, { status: "succeeded", finishedAt: now(), currentStep: "" });
    appendRunEventType(db, run.id, "run.completed");
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const current = getRun(db, run.id);
    if (current?.currentStep) stepFailed(db, run.id, current.currentStep);
    updateRun(db, run.id, { status: "failed", finishedAt: now(), errorCode: "worker_error", errorMessage: message });
    appendRunEventType(db, run.id, "error", { code: "worker_error", message });
    appendRunEventType(db, run.id, "run.failed", { code: "worker_error", message });
  }
}

export async function runWorker({ once = false, root = ROOT } = {}) {
  const db = createDatabase(root);
  await syncFilesystem(db, root);
  sqlRun(db, `INSERT INTO workers(id, hostname, status, last_heartbeat_at, metadata_json)
    VALUES (?, ?, 'idle', ?, ?)
    ON CONFLICT(id) DO UPDATE SET hostname = excluded.hostname, status = 'idle', last_heartbeat_at = excluded.last_heartbeat_at`, [WORKER_ID, hostname(), now(), json({ pid: process.pid })]);
  while (true) {
    sqlRun(db, "UPDATE workers SET status = 'idle', current_run_id = NULL, last_heartbeat_at = ? WHERE id = ?", [now(), WORKER_ID]);
    const claimed = claimNextRun(db, WORKER_ID);
    if (claimed) {
      sqlRun(db, "UPDATE workers SET status = 'running', current_run_id = ?, last_heartbeat_at = ? WHERE id = ?", [claimed.id, now(), WORKER_ID]);
      await executeRun(db, root, claimed);
      sqlRun(db, "UPDATE workers SET status = 'idle', current_run_id = NULL, last_heartbeat_at = ? WHERE id = ?", [now(), WORKER_ID]);
    } else if (once) {
      break;
    } else {
      await sleep(SLEEP_MS);
    }
  }
  db.close();
}

if (import.meta.main) {
  const once = process.argv.includes("--once");
  runWorker({ once }).catch((error) => { console.error(error); process.exit(1); });
}
