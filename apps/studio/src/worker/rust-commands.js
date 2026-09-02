import { dirname, relative, resolve, sep } from "node:path";
import { dataRoot, one } from "../server/database/db.js";
import { findSnapshot } from "../server/artifacts/inventory.js";

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

function relativeFromRoot(root, path) {
  const result = relative(root, resolve(path)).split(sep).join("/");
  if (!result || result.startsWith("../") || result === "..") throw new Error("Rust command path escaped the repository root");
  return result;
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

/**
 * Convert a validated run specification to an argv array. No shell syntax is
 * accepted or constructed here; callers must pass the returned argv directly
 * to Bun.spawn.
 */
export function buildRustCommand({ db, root, runId, spec, binary = process.env.TRANSIT_LAB_BINARY || "transit" }) {
  const outputDirectory = resolve(dataRoot(root), "runs", runId);
  switch (spec.kind) {
    case "compile-snapshot": {
      const revision = requireFeedRevision(db, spec.feedRevisionId);
      const output = runOutput(root, runId, "snapshot");
      return {
        argv: [binary, "compile", "--input", relativeFromRoot(root, revision.local_path), "--service-date", spec.serviceDate, "--output", relativeFromRoot(root, output)],
        step: "compile-snapshot",
        outputs: [{ path: output, kind: "compiled-snapshot" }],
        outputDirectory
      };
    }
    case "simulate-criticality": {
      const snapshot = findSnapshot(db, spec.snapshotId);
      const labels = runOutput(root, runId, "criticality-labels.jsonl");
      return {
        argv: [binary, "labels", "line-removal", "--snapshot", relativeFromRoot(root, snapshotDirectory(root, snapshot)), "--output", relativeFromRoot(root, labels)],
        step: "simulate-criticality",
        outputs: [{ path: labels, kind: "criticality-labels" }],
        outputDirectory
      };
    }
    case "train": {
      const dataset = one(db, "SELECT manifest_path FROM datasets WHERE id = ?", [spec.datasetId]);
      if (!dataset) throw new Error(`dataset ${spec.datasetId} is not indexed`);
      throw new Error("train runs need an explicit graph-to-dataset adapter before they can be submitted");
    }
    case "infer": {
      const snapshot = findSnapshot(db, spec.snapshotId);
      const model = one(db, "SELECT checkpoint_artifact_id FROM model_versions WHERE id = ?", [spec.modelId]);
      if (!model) throw new Error(`model ${spec.modelId} is not indexed`);
      const checkpoint = one(db, "SELECT local_path FROM artifacts WHERE id = ?", [model.checkpoint_artifact_id]);
      if (!checkpoint?.local_path) throw new Error(`model ${spec.modelId} has no local checkpoint artifact`);
      const output = runOutput(root, runId, "inference-result.json");
      return {
        argv: [binary, "infer", "criticality", "--graph", relativeFromRoot(root, graphDirectory(root, snapshot)), "--model", relativeFromRoot(root, checkpoint.local_path), "--model-id", spec.modelId, "--output", relativeFromRoot(root, output)],
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
