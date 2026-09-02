import { dirname, relative, resolve, sep } from "node:path";
import { readFileSync } from "node:fs";
import { dataRoot, one } from "../../../packages/control-store/src/database.ts";
import { findSnapshot } from "../../../packages/control-store/src/inventory.ts";

const SAFE_DATA_SEGMENT = /^[A-Za-z0-9._:-]+$/;

export const RUST_SUBCOMMANDS = Object.freeze([
  "fetch",
  "validate",
  "compile",
  "graph",
  "labels",
  "train",
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

function requireFeedRevision(db, feedRevisionId) {
  const revision = one(db, "SELECT id, local_path FROM feed_revisions WHERE id = ?", [feedRevisionId]);
  if (!revision) throw new Error(`feed revision ${feedRevisionId} is not indexed`);
  return revision;
}

function requireDataset(db, root, datasetId) {
  const dataset = one(db, "SELECT manifest_path FROM datasets WHERE id = ? AND status = 'ready'", [datasetId]);
  if (!dataset?.manifest_path) throw new Error(`dataset ${datasetId} is not indexed and ready`);
  const manifestPath = resolve(root, dataset.manifest_path);
  let manifest;
  try {
    manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
  } catch (error) {
    throw new Error(`could not read dataset ${datasetId} manifest: ${error instanceof Error ? error.message : String(error)}`);
  }
  const directory = dirname(manifestPath);
  const graph = resolve(directory, manifest.graph_directory || manifest.graphDirectory || "graph");
  const labels = resolve(directory, manifest.label_file || manifest.labelFile || "labels.jsonl");
  return {
    graph: commandPath(root, graph),
    labels: commandPath(root, labels)
  };
}

/**
 * Convert a validated run specification to an argv array. No shell syntax is
 * accepted or constructed here; callers must pass the returned argv directly
 * to Bun.spawn.
 */
export function buildRustCommand({ db = null, root, runId, spec, binary = process.env.TRANSIT_LAB_BINARY }) {
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
      return {
        argv: rustArgv(root, binary, [
          "train", "multitask",
          "--graph", dataset.graph,
          "--labels", dataset.labels,
          "--config", commandPath(root, config),
          "--seed", String(spec.seed),
          "--output", commandPath(root, output)
        ]),
        step: "train",
        outputs: [{ path: output, kind: "model-checkpoint" }],
        outputDirectory
      };
    }
    case "infer": {
      const snapshot = findSnapshot(db, spec.snapshotId);
      const model = one(db, "SELECT checkpoint_artifact_id FROM model_versions WHERE id = ?", [spec.modelId]);
      if (!model) throw new Error(`model ${spec.modelId} is not indexed`);
      const checkpoint = one(db, "SELECT local_path FROM artifacts WHERE id = ?", [model.checkpoint_artifact_id]);
      if (!checkpoint?.local_path) throw new Error(`model ${spec.modelId} has no local checkpoint artifact`);
      const output = runOutput(root, runId, "inference-result.json");
      return {
        argv: rustArgv(root, binary, ["infer", "criticality", "--graph", commandPath(root, graphDirectory(root, snapshot)), "--model", commandPath(root, checkpoint.local_path), "--model-id", spec.modelId, "--output", commandPath(root, output)]),
        step: "infer",
        outputs: [{ path: output, kind: "inference-result" }],
        outputDirectory
      };
    }
    case "build-dataset":
      throw new Error("build-dataset is not exposed by the current Rust CLI; no fallback command is permitted");
    case "evaluate":
      throw new Error("evaluate is not exposed by the current Rust CLI; no fallback command is permitted");
    default:
      throw new Error(`unsupported run kind: ${spec.kind}`);
  }
}
