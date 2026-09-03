import { createHash } from "node:crypto";
import { readFileSync, realpathSync, statSync } from "node:fs";
import { dirname, join, relative, resolve, sep } from "node:path";
import { ROUTER_ALGORITHM_VERSION, validateTrainingCheckpointManifest } from "../../../packages/contracts/src/index.ts";
import { all, dataRoot, findTrainingCheckpointForRun, one, parseJson } from "../../../packages/control-store/src/database.ts";
import { findSnapshot } from "../../../packages/control-store/src/inventory.ts";

const SAFE_DATA_SEGMENT = /^[A-Za-z0-9._:-]+$/;

export const RUST_SUBCOMMANDS = Object.freeze([
  "fetch",
  "validate",
  "compile",
  "graph",
  "labels",
  "build-dataset",
  "evaluate",
  "bench",
  "train",
  "encode-dataset",
  "train-heads",
  "fine-tune",
  "infer",
  "verify",
  "similar-lines",
  "demo"
]);

function inside(root, path) {
  const base = resolve(root);
  const candidate = resolve(path);
  return candidate === base || candidate.startsWith(`${base}${sep}`);
}

function commandPath(root, path) {
  const absolute = resolve(path);
  if (inside(root, absolute)) return relative(root, absolute).split(sep).join("/");
  if (inside(dataRoot(root), absolute)) return absolute;
  throw new Error("Rust command path escaped the repository and data roots");
}

function rustArgv(root, binary, args) {
  if (binary) return [binary, ...args];
  const release = resolve(root, "target/release/transit-cli");
  const debug = resolve(root, "target/debug/transit-cli");
  if (Bun.file(release).size > 0) return [release, ...args];
  if (Bun.file(debug).size > 0) return [debug, ...args];
  return ["cargo", "run", "--quiet", "-p", "transit-cli", "--", ...args];
}

function runOutput(root, runId, name) {
  if (!SAFE_DATA_SEGMENT.test(runId) || !SAFE_DATA_SEGMENT.test(name)) throw new Error("unsafe run output name");
  return resolve(dataRoot(root), "runs", runId, name);
}

function snapshotDirectory(root, snapshot) {
  if (!snapshot?.manifestPath) throw new Error("snapshot has no indexed manifest path");
  return resolve(root, dirname(snapshot.manifestPath));
}

function graphDirectory(root, snapshot) {
  if (snapshot.graphPath) return resolve(root, snapshot.graphPath);
  throw new Error("snapshot has no indexed graph artifact; build the graph before this run");
}

function requireSnapshot(db, root, snapshotId) {
  if (!db) throw new Error("an indexed control database is required for snapshot runs");
  const snapshot = findSnapshot(db, snapshotId);
  if (!snapshot) throw new Error(`snapshot ${snapshotId} is not indexed`);
  if (!snapshot.graphPath) throw new Error(`snapshot ${snapshotId} has no indexed graph; build the graph before this run`);
  const graph = resolve(root, snapshot.graphPath);
  return {
    snapshot,
    graph: commandPath(root, graph)
  };
}

function artifactPath(root, artifact) {
  const raw = artifact?.local_path || artifact?.uri;
  if (typeof raw !== "string" || raw.trim() === "") return null;
  const path = resolve(root, raw);
  if (!inside(root, path) && !inside(dataRoot(root), path)) {
    throw new Error("indexed artifact path escaped the repository and data roots");
  }
  return path;
}

function labelArtifactForSnapshot(db, root, snapshotId) {
  if (!db) return null;
  const candidates = all(db, `SELECT kind, local_path, uri, metadata_json
    FROM artifacts
    WHERE status = 'ready' AND kind IN ('criticality-labels', 'label-batch')
    ORDER BY created_at DESC, id DESC`);
  for (const candidate of candidates) {
    const metadata = parseJson(candidate.metadata_json, {});
    if (metadata.routerAlgorithmVersion !== ROUTER_ALGORITHM_VERSION &&
        metadata.router_algorithm_version !== ROUTER_ALGORITHM_VERSION) continue;
    if (metadata.snapshotId !== snapshotId && metadata.snapshot_id !== snapshotId) continue;
    const path = artifactPath(root, candidate);
    if (!path) continue;
    try {
      const first = readFileSync(path, "utf8").split(/\r?\n/).find((line) => line.trim());
      const row = first ? JSON.parse(first) : null;
      if (row?.snapshot === snapshotId && row?.router_algorithm_version === ROUTER_ALGORITHM_VERSION) return path;
    } catch {
      // A malformed or stale artifact is not a candidate for a dataset build.
    }
  }
  return null;
}

function requireFeedRevision(db, feedRevisionId) {
  const revision = one(db, "SELECT id, local_path FROM feed_revisions WHERE id = ?", [feedRevisionId]);
  if (!revision) throw new Error(`feed revision ${feedRevisionId} is not indexed`);
  return revision;
}

function requireDataset(db, root, datasetId) {
  const dataset = one(db, "SELECT fingerprint, manifest_path FROM datasets WHERE id = ? AND status = 'ready'", [datasetId]);
  if (!dataset?.manifest_path) throw new Error(`dataset ${datasetId} is not indexed and ready`);
  const manifestPath = resolve(root, dataset.manifest_path);
  let manifest;
  try {
    manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
  } catch (error) {
    throw new Error(`could not read dataset ${datasetId} manifest: ${error instanceof Error ? error.message : String(error)}`);
  }
  const directory = dirname(manifestPath);
  return {
    dataset: commandPath(root, directory),
    datasetFingerprint: manifest.fingerprint || dataset.fingerprint
  };
}

function requireModel(db, root, modelId) {
  if (!db) throw new Error("an indexed control database is required for model runs");
  const model = one(db, `SELECT mv.dataset_id, a.local_path, a.uri
    FROM model_versions mv
    LEFT JOIN artifacts a ON a.id = mv.checkpoint_artifact_id
    WHERE mv.id = ? AND mv.status = 'ready'`, [modelId]);
  if (!model) throw new Error(`model ${modelId} is not indexed and ready`);
  const modelPath = artifactPath(root, model);
  if (!modelPath) throw new Error(`model ${modelId} has no local checkpoint artifact`);
  try {
    if (!statSync(modelPath).isFile()) throw new Error("not a file");
  } catch (error) {
    throw new Error(`model ${modelId} checkpoint artifact is missing: ${error instanceof Error ? error.message : String(error)}`);
  }
  return { datasetId: model.dataset_id || null, model: commandPath(root, modelPath) };
}

function configObject(value, field) {
  if (value === undefined || value === null || value === "") return {};
  if (typeof value === "string") {
    try {
      const parsed = JSON.parse(value);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) return parsed;
    } catch {
      return { strategy: value };
    }
    throw new Error(`${field} must be a JSON object or a named strategy`);
  }
  if (typeof value !== "object" || Array.isArray(value)) throw new Error(`${field} must be an object`);
  return value;
}

function stringList(value, field) {
  if (value === undefined || value === null) return [];
  if (!Array.isArray(value)) throw new Error(`${field} must be an array`);
  return value.map((item, index) => {
    if (typeof item !== "string" || item.trim() === "") throw new Error(`${field}[${index}] must be a non-empty string`);
    if (!SAFE_DATA_SEGMENT.test(item)) throw new Error(`${field}[${index}] is not a safe identifier`);
    return item;
  });
}

function buildDatasetSplitArgs(splitConfig) {
  const split = configObject(splitConfig, "splitConfig");
  const strategy = split.strategy || split.splitStrategy;
  const args = strategy ? ["--split-strategy", String(strategy)] : [];
  for (const [key, flag] of [
    ["validationSnapshots", "--validation-snapshot"],
    ["validation_snapshots", "--validation-snapshot"],
    ["testSnapshots", "--test-snapshot"],
    ["test_snapshots", "--test-snapshot"],
    ["validationNetworks", "--validation-network"],
    ["validation_networks", "--validation-network"],
    ["testNetworks", "--test-network"],
    ["test_networks", "--test-network"]
  ]) {
    for (const value of stringList(split[key], `splitConfig.${key}`)) args.push(flag, value);
  }
  if (split && typeof split === "object" && (split.train || split.validation || split.test) && !strategy) {
    args.push("--split-json", JSON.stringify(split));
  }
  return args;
}

function evaluationArgs(evaluationSuite) {
  const suite = configObject(evaluationSuite, "evaluationSuite");
  const split = suite.split || "test";
  if (typeof split !== "string" || !["all", "train", "validation", "test"].includes(split)) {
    throw new Error("evaluationSuite.split must be all, train, validation, or test");
  }
  const topK = suite.topK ?? suite.top_k ?? 10;
  const seed = suite.seed ?? 73;
  if (!Number.isInteger(topK) || topK < 1 || topK > 1_000_000) throw new Error("evaluationSuite.topK must be a positive integer");
  if (!Number.isInteger(seed) || seed < 0 || seed > 2 ** 31 - 1) throw new Error("evaluationSuite.seed must be a non-negative 32-bit integer");
  return ["--split", split, "--top-k", String(topK), "--seed", String(seed)];
}

function validCheckpointRow(root, row) {
  if (!row || row.status !== "committed" || typeof row.local_path !== "string") return null;
  const checkpointPath = resolve(dataRoot(root), row.local_path);
  if (!inside(dataRoot(root), checkpointPath)) return null;
  try {
    const canonicalRoot = realpathSync(dataRoot(root));
    const canonicalCheckpoint = realpathSync(checkpointPath);
    if (!inside(canonicalRoot, canonicalCheckpoint)) return null;
    const manifest = JSON.parse(readFileSync(join(checkpointPath, "manifest.json"), "utf8"));
    validateTrainingCheckpointManifest(manifest);
    if (manifest.runId !== row.run_id || Number(manifest.globalStep) !== Number(row.global_step)) return null;
    if (String(manifest.checkpointFingerprint).toLowerCase() !== String(row.sha256).toLowerCase()) return null;
    for (const file of manifest.files) {
      const payloadPath = resolve(checkpointPath, file.path);
      if (!inside(checkpointPath, payloadPath)) return null;
      const canonicalPayload = realpathSync(payloadPath);
      if (!inside(canonicalCheckpoint, canonicalPayload)) return null;
      const payload = readFileSync(canonicalPayload);
      const digest = createHash("sha256").update(payload).digest("hex");
      if (payload.byteLength !== file.sizeBytes || digest !== file.sha256) return null;
    }
    return { row, path: checkpointPath, manifest };
  } catch {
    return null;
  }
}

/**
 * Return the newest physically committed checkpoint for a run. The database
 * is an index, not proof that a checkpoint is still usable: a process can die
 * after a DB update, or a payload can be damaged later. Validate the manifest
 * and every payload before handing a path to the Rust CLI.
 */
export function findLatestValidTrainingCheckpoint(db, root, runId, { includeResume = true } = {}) {
  if (!db) return null;
  const runRecord = one(db, "SELECT resume_checkpoint_id FROM runs WHERE id = ?", [runId]);
  const currentRows = all(db, `SELECT * FROM training_checkpoints
    WHERE run_id = ? AND status = 'committed'
    ORDER BY global_step DESC, id DESC`, [runId]);
  const current = currentRows.map((row) => validCheckpointRow(root, row)).filter(Boolean);
  if (current.length) return current[0];
  if (!includeResume || !runRecord?.resume_checkpoint_id) return null;
  const sourceRow = findTrainingCheckpointForRun(db, runId, runRecord.resume_checkpoint_id);
  return validCheckpointRow(root, sourceRow);
}

function trainingResumePath(db, root, runId) {
  if (!db) return null;
  const runRecord = one(db, "SELECT resume_checkpoint_id FROM runs WHERE id = ?", [runId]);
  const checkpoint = findLatestValidTrainingCheckpoint(db, root, runId);
  if (!checkpoint) {
    if (runRecord?.resume_checkpoint_id) {
      throw new Error(`resume training checkpoint ${runRecord.resume_checkpoint_id} is not a valid committed checkpoint`);
    }
    return null;
  }
  return commandPath(root, checkpoint.path);
}

/**
 * Convert a validated run specification to an argv array. No shell syntax is
 * accepted or constructed here; callers must pass the returned argv directly
 * to Bun.spawn.
 */
export function buildRustCommand({
  db = null,
  root,
  runId,
  spec,
  binary = process.env.TRANSIT_LAB_BINARY,
  maxAttemptSeconds = undefined,
  checkpointGraceSeconds = undefined,
  forkFromCheckpoint = false
}) {
  const outputDirectory = resolve(dataRoot(root), "runs", runId);
  switch (spec.kind) {
    case "compile-snapshot": {
      const revision = requireFeedRevision(db, spec.feedRevisionId);
      const output = runOutput(root, runId, "snapshot");
      return {
        argv: rustArgv(root, binary, ["compile", "--input", commandPath(root, revision.local_path), "--service-date", spec.serviceDate, "--output", commandPath(root, output)]),
        step: "compile-snapshot",
        outputs: [{ path: output, kind: "compiled-snapshot" }],
        outputDirectory
      };
    }
    case "simulate-criticality": {
      const snapshot = findSnapshot(db, spec.snapshotId);
      if (!snapshot) throw new Error(`snapshot ${spec.snapshotId} is not indexed`);
      const labels = runOutput(root, runId, "criticality-labels.jsonl");
      return {
        argv: rustArgv(root, binary, ["labels", "line-removal", "--snapshot", commandPath(root, snapshotDirectory(root, snapshot)), "--output", commandPath(root, labels)]),
        step: "simulate-criticality",
        outputs: [{ path: labels, kind: "criticality-labels" }],
        outputDirectory
      };
    }
    case "train": {
      const dataset = requireDataset(db, root, spec.datasetId);
      const config = runOutput(root, runId, "resolved-config.json");
      if (Bun.file(config).size <= 0) throw new Error("resolved experiment config is missing");
      const output = runOutput(root, runId, "model.json");
      const checkpointDirectory = runOutput(root, runId, "checkpoints");
      const controlFile = runOutput(root, runId, "control.json");
      const runtime = spec.runtime || {};
      const checkpointEverySteps = runtime.checkpointEverySteps || 500;
      const checkpointEverySeconds = runtime.checkpointEverySeconds || 900;
      const resumePath = trainingResumePath(db, root, runId);
      const effectiveMaxAttemptSeconds = maxAttemptSeconds ?? runtime.maxAttemptSeconds;
      const effectiveCheckpointGraceSeconds = checkpointGraceSeconds ?? runtime.checkpointGraceSeconds;
      return {
        argv: rustArgv(root, binary, [
          "train", "multitask",
          "--dataset", dataset.dataset,
          "--split", "train",
          "--config", commandPath(root, config),
          "--seed", String(spec.seed),
          "--output", commandPath(root, output),
          "--checkpoint-dir", commandPath(root, checkpointDirectory),
          "--control-file", commandPath(root, controlFile),
          "--run-id", runId,
          "--checkpoint-every-steps", String(checkpointEverySteps),
          "--checkpoint-every-seconds", String(checkpointEverySeconds),
          ...(runtime.device ? ["--device", String(runtime.device)] : []),
          ...(runtime.precision ? ["--dtype", String(runtime.precision)] : []),
          ...(runtime.backend ? ["--backend", String(runtime.backend)] : []),
          ...(runtime.workerThreads ? ["--cpu-threads", String(runtime.workerThreads)] : []),
          ...(runtime.rayonThreads ? ["--rayon-threads", String(runtime.rayonThreads)] : []),
          ...(runtime.gradientAccumulation ? ["--gradient-accumulation", String(runtime.gradientAccumulation)] : []),
          ...(resumePath ? ["--resume", resumePath] : []),
          ...(resumePath && forkFromCheckpoint ? ["--fork-from-checkpoint"] : []),
          ...(effectiveMaxAttemptSeconds ? ["--max-wall-time-seconds", String(effectiveMaxAttemptSeconds)] : []),
          ...(effectiveCheckpointGraceSeconds ? ["--checkpoint-grace-seconds", String(effectiveCheckpointGraceSeconds)] : [])
        ]),
        step: "train",
        outputs: [{ path: output, kind: "model-checkpoint" }],
        outputDirectory,
        checkpointDirectory,
        controlFile,
        datasetFingerprint: dataset.datasetFingerprint
      };
    }
    case "infer": {
      const snapshot = findSnapshot(db, spec.snapshotId);
      if (!snapshot) throw new Error(`snapshot ${spec.snapshotId} is not indexed`);
      const model = requireModel(db, root, spec.modelId);
      const output = runOutput(root, runId, "inference-result.json");
      const graph = graphDirectory(root, snapshot);
      try {
        if (!statSync(graph).isDirectory()) throw new Error("not a directory");
      } catch (error) {
        throw new Error(`snapshot ${spec.snapshotId} graph artifact is missing: ${error instanceof Error ? error.message : String(error)}`);
      }
      return {
        argv: rustArgv(root, binary, ["infer", "criticality", "--graph", commandPath(root, graph), "--model", model.model, "--model-id", spec.modelId, "--output", commandPath(root, output)]),
        step: "infer",
        outputs: [{ path: output, kind: "inference-result" }],
        outputDirectory
      };
    }
    case "build-dataset": {
      if (!db) throw new Error("an indexed control database is required for dataset runs");
      const snapshots = spec.snapshotIds.map((snapshotId) => requireSnapshot(db, root, snapshotId));
      const labels = snapshots.map(({ snapshot }) => labelArtifactForSnapshot(db, root, snapshot.id));
      // A missing label in the middle cannot be represented by clap's
      // repeated Vec<PathBuf> argument without shifting later labels. Build a
      // graph-only dataset in that case and let the user run label generation
      // or a second immutable build when labels are available.
      const allLabels = labels.length > 0 && labels.every(Boolean);
      const output = runOutput(root, runId, "dataset");
      return {
        argv: rustArgv(root, binary, [
          "build-dataset",
          ...snapshots.flatMap(({ graph }) => ["--graph", graph]),
          ...(allLabels ? labels.flatMap((label) => ["--labels", label]) : []),
          "--output", commandPath(root, output),
          ...buildDatasetSplitArgs(spec.splitConfig)
        ]),
        step: "build-dataset",
        outputs: [{ path: output, kind: "dataset-manifest" }],
        outputDirectory,
        datasetOutput: output,
        labelMode: allLabels ? "labelled" : "graph-only"
      };
    }
    case "evaluate": {
      const model = requireModel(db, root, spec.modelId);
      const datasetId = spec.datasetId || model.datasetId;
      if (!datasetId) throw new Error(`model ${spec.modelId} is not associated with a dataset; pass datasetId`);
      const dataset = requireDataset(db, root, datasetId);
      const output = runOutput(root, runId, "evaluation.json");
      return {
        argv: rustArgv(root, binary, [
          "evaluate",
          "--dataset", dataset.dataset,
          "--model", model.model,
          "--model-id", spec.modelId,
          "--output", commandPath(root, output),
          ...evaluationArgs(spec.evaluationSuite)
        ]),
        step: "evaluate",
        outputs: [{ path: output, kind: "evaluation-result" }],
        outputDirectory,
        evaluationOutput: output,
        datasetId
      };
    }
    default:
      throw new Error(`unsupported run kind: ${spec.kind}`);
  }
}
