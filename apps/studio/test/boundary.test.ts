import { expect, test } from "bun:test";
import { copyFile, mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { randomUUID } from "node:crypto";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createDatabase, one, run } from "../../../packages/control-store/src/database.ts";
import { createApiHandler } from "../../api/src/routes.ts";
import { createArtifactManifest, describeArtifactFile, writeArtifactManifest } from "../../../packages/control-store/src/manifest.ts";
import { findArtifact, syncFilesystem } from "../../../packages/control-store/src/inventory.ts";
import { materializeResolvedRunConfig, readResolvedRunConfig } from "../../../packages/control-store/src/experiments.ts";
import { buildRustCommand } from "../../worker/src/rust-commands.ts";
import { parseEventLine } from "../../worker/src/parse-events.ts";

test("structured event parsing rejects human console text", () => {
  const event = parseEventLine(JSON.stringify({
    schemaVersion: 1,
    seq: 4,
    runId: "run-1",
    timestamp: "2026-09-02T00:00:00.000Z",
    type: "progress",
    step: "compile",
    completed: 4,
    total: 10,
    unit: "files"
  }), 1, "run-1");
  expect(event.type).toBe("progress");
  expect(() => parseEventLine("Processing line 4 of 10...", 2, "run-1")).toThrow("not valid JSON");
  expect(() => parseEventLine(JSON.stringify({ ...event, runId: "run-2" }), 3, "run-1")).toThrow("wrong runId");
});

test("Rust command construction returns an argv array for indexed inputs", () => {
  const root = process.cwd();
  const db = createDatabase(root, `/tmp/transit-lab-studio-command-test-${randomUUID()}.sqlite`);
  run(db, "INSERT INTO projects(id, name, created_at) VALUES ('project-local', 'Transit Lab', '2026-09-02T00:00:00Z')");
  run(db, "INSERT INTO networks(id, project_id, display_name, created_at, updated_at) VALUES ('demo', 'project-local', 'Demo', '2026-09-02T00:00:00Z', '2026-09-02T00:00:00Z')");
  run(db, `INSERT INTO snapshots(id, network_id, service_date, status, fingerprint, manifest_path, network_path, graph_path, created_at, updated_at)
    VALUES ('snapshot-1', 'demo', '2026-09-02', 'ready', 'fingerprint-1', 'data/demo/snapshot/manifest.json', 'data/demo/snapshot/network.json', 'data/demo/graph', '2026-09-02T00:00:00Z', '2026-09-02T00:00:00Z')`);
  run(db, "INSERT INTO artifacts(id, kind, fingerprint, uri, local_path, created_at) VALUES ('artifact-model', 'model-checkpoint', 'artifact-fingerprint', 'data/demo/model.json', 'data/demo/model.json', '2026-09-02T00:00:00Z')");
  run(db, "INSERT INTO model_versions(id, version, fingerprint, checkpoint_artifact_id, created_at) VALUES ('model-1', 'v1', 'model-fingerprint', 'artifact-model', '2026-09-02T00:00:00Z')");
  const command = buildRustCommand({ db, root, runId: "run-1", spec: { kind: "infer", modelId: "model-1", snapshotId: "snapshot-1" }, binary: "/opt/transit" });
  expect(command.argv).toEqual([
    "/opt/transit", "infer", "criticality", "--graph", "data/demo/graph", "--model", "data/demo/model.json", "--model-id", "model-1", "--output", "data/runs/run-1/inference-result.json"
  ]);
  expect(command.argv.some((part) => /[;&|`$()]/.test(part))).toBe(false);
  db.close();
});

test("Rust command construction supports an external data root", async () => {
  const root = process.cwd();
  const external = await mkdtemp(join(tmpdir(), "transit-lab-external-data-"));
  const previous = process.env.TRANSIT_LAB_DATA_ROOT;
  process.env.TRANSIT_LAB_DATA_ROOT = external;
  try {
    const db = createDatabase(root, ":memory:");
    run(db, "INSERT INTO projects(id, name, created_at) VALUES ('project-local', 'Transit Lab', '2026-09-02T00:00:00Z')");
    run(db, "INSERT INTO networks(id, project_id, display_name, created_at, updated_at) VALUES ('demo', 'project-local', 'Demo', '2026-09-02T00:00:00Z', '2026-09-02T00:00:00Z')");
    const feed = join(external, "raw/demo/gtfs.zip");
    await mkdir(join(external, "raw/demo"), { recursive: true });
    await writeFile(feed, "fixture");
    run(db, `INSERT INTO feed_revisions(id, network_id, sha256, local_path, created_at)
      VALUES (?, 'demo', ?, ?, '2026-09-02T00:00:00Z')`, ["feed-1", "a".repeat(64), feed]);
    const command = buildRustCommand({ db, root, runId: "run-1", spec: { kind: "compile-snapshot", feedRevisionId: "feed-1", serviceDate: "2026-09-02" }, binary: "/opt/transit" });
    expect(command.argv).toContain(feed);
    expect(command.argv.at(-1)).toBe(join(external, "runs/run-1/snapshot"));
    db.close();
  } finally {
    if (previous === undefined) delete process.env.TRANSIT_LAB_DATA_ROOT;
    else process.env.TRANSIT_LAB_DATA_ROOT = previous;
  }
});

test("resolved experiment configs are immutable and propagate the submitted seed", async () => {
  const root = await mkdtemp(join(tmpdir(), "transit-lab-experiment-config-"));
  await mkdir(join(root, "configs/models"), { recursive: true });
  await copyFile(
    join(process.cwd(), "configs/models/multitask-v1.yaml"),
    join(root, "configs/models/multitask-v1.yaml")
  );
  const first = materializeResolvedRunConfig(root, "run-1", {
    kind: "train",
    datasetId: "dataset-1",
    modelConfig: "configs/models/multitask-v1.yaml",
    seed: 42
  });
  const document = readResolvedRunConfig(first.path);
  expect(document.configFingerprint).toBe(first.configFingerprint);
  expect(document.modelConfig.pretraining.seed).toBe(42);
  expect(document.modelConfig.representation.seed).toBe(42);
  expect(document.modelConfig.criticality.seed).toBe(42);
  expect(() => materializeResolvedRunConfig(root, "run-1", {
    kind: "train",
    datasetId: "dataset-1",
    modelConfig: "configs/models/multitask-v1.yaml",
    seed: 43
  })).toThrow("immutable resolved config");
});

test("unsupported dataset and evaluation runs do not get fallback Rust commands", () => {
  const root = process.cwd();
  expect(() => buildRustCommand({
    root,
    runId: "run-1",
    spec: { kind: "build-dataset", snapshotIds: ["snapshot-1"] },
    binary: "transit"
  })).toThrow("build-dataset is not exposed");
  expect(() => buildRustCommand({
    root,
    runId: "run-1",
    spec: { kind: "evaluate", modelId: "model-1" },
    binary: "transit"
  })).toThrow("evaluate is not exposed");
});

test("artifact manifests include hashes and immutable file locations", async () => {
  const root = await mkdtemp(join(tmpdir(), "transit-lab-studio-artifact-"));
  const output = join(root, "output.json");
  await writeFile(output, "artifact\n");
  const file = await describeArtifactFile(root, output);
  const manifest = await createArtifactManifest({
    root,
    artifactId: "artifact-1",
    kind: "test-output",
    fingerprint: "fingerprint-1",
    files: [file]
  });
  expect(manifest.sha256).toHaveLength(64);
  expect(manifest.files[0].path).toBe("output.json");
  const manifestPath = join(root, "artifact-manifest.json");
  await writeArtifactManifest(manifestPath, manifest);
  await writeArtifactManifest(manifestPath, manifest);
  expect(JSON.parse(await Bun.file(manifestPath).text()).schemaVersion).toBe(1);
});

test("Studio preserves Rust-owned inference percentiles and structural scores", async () => {
  const root = process.cwd();
  const db = createDatabase(root, ":memory:");
  const timestamp = "2026-09-02T00:00:00.000Z";
  run(db, "INSERT INTO projects(id, name, created_at) VALUES (?, ?, ?)", ["project-local", "Transit Lab", timestamp]);
  run(db, "INSERT INTO networks(id, project_id, display_name, created_at, updated_at) VALUES (?, ?, ?, ?, ?)", ["demo", "project-local", "Demo", timestamp, timestamp]);
  run(db, `INSERT INTO snapshots(id, network_id, service_date, status, fingerprint, manifest_path, network_path, created_at, updated_at)
    VALUES (?, ?, ?, 'ready', ?, ?, ?, ?, ?)`, ["snapshot-1", "demo", "2026-09-02", "snapshot-fingerprint", "data/snapshot/manifest.json", "data/snapshot/network.json", timestamp, timestamp]);
  run(db, `INSERT INTO line_instances(id, snapshot_id, line_index, canonical_id, display_name, created_at, updated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?)`, ["snapshot-1:0", "snapshot-1", 0, "blue", "Blue", timestamp, timestamp]);
  run(db, `INSERT INTO model_versions(id, version, fingerprint, status, created_at)
    VALUES (?, ?, ?, 'ready', ?)`, ["model-1", "demo", "model-fingerprint", timestamp]);
  run(db, `INSERT INTO inference_sets(id, fingerprint, model_id, snapshot_id, status, config_json, created_at)
    VALUES (?, ?, ?, ?, 'ready', ?, ?)`, ["inference-1", "inference-fingerprint", "model-1", "snapshot-1", JSON.stringify({ metricNames: ["accessibility_auc_loss"] }), timestamp]);
  run(db, `INSERT INTO criticality_predictions(inference_id, line_instance_id, primary_score, uncertainty, values_json, created_at)
    VALUES (?, ?, ?, ?, ?, ?)`, ["inference-1", "snapshot-1:0", 0.25, 0.1, JSON.stringify({ accessibility_auc_loss: 0.25, percentiles: { accessibility_auc_loss: 0.8 }, structuralUniqueness: 0.7 }), timestamp]);

  const handle = createApiHandler({ db, root });
  const response = await handle(new Request("http://studio/api/criticality?inferenceId=inference-1"));
  expect(response.status).toBe(200);
  const result = await response.json();
  expect(result.predictions[0]).toMatchObject({
    metricPercentiles: [0.8],
    structuralUniqueness: 0.7,
    uncertainty: 0.1
  });
  db.close();
});

test("filesystem ingestion reuses explicit manifests and preserves camelCase inference fields", async () => {
  const root = await mkdtemp(join(tmpdir(), "transit-lab-studio-ingestion-"));

  const snapshotDirectory = join(root, "data/snapshots/demo/2026-09-02");
  const inferenceDirectory = join(root, "data/inference/inference-1");
  await mkdir(snapshotDirectory, { recursive: true });
  await mkdir(inferenceDirectory, { recursive: true });
  await writeFile(join(snapshotDirectory, "manifest.json"), JSON.stringify({
    snapshot_id: "snapshot-1",
    source_name: "Synthetic demo",
    geographical_scope: "Demo",
    descriptor: { service_date: "2026-09-02", compiler_version: "test" }
  }));
  await writeFile(join(snapshotDirectory, "network.json"), JSON.stringify({
    snapshot_id: "snapshot-1",
    stations: [{ name: "Central", latitude: 48.2, longitude: 16.3 }],
    lines: [{ index: 0, canonical_id: "blue", display_name: "Blue", mode: 1 }],
    patterns: []
  }));

  const resultPath = join(inferenceDirectory, "inference-result.json");
  await writeFile(resultPath, JSON.stringify({
    schemaVersion: 1,
    inferenceId: "inference-1",
    modelId: "model-1",
    snapshotId: "snapshot-1",
    metricNames: ["accessibility_auc_loss", "unreachable_share"],
    predictions: [{
      line: 0,
      lineName: "Blue",
      metrics: [0.25, 0.4],
      metricPercentiles: [0.8, 0.6],
      structuralUniqueness: 0.7
    }]
  }));
  const explicitManifest = await createArtifactManifest({
    root,
    artifactId: "artifact-inference-1",
    kind: "inference-result",
    fingerprint: "inference-artifact-fingerprint",
    files: [await describeArtifactFile(root, resultPath)]
  });
  await writeArtifactManifest(join(inferenceDirectory, "artifact-manifest.json"), explicitManifest);

  const db = createDatabase(root, ":memory:");
  run(db, `INSERT INTO model_versions(id, version, fingerprint, status, created_at)
    VALUES (?, ?, ?, 'ready', ?)`, ["model-1", "test", "model-fingerprint", "2026-09-02T00:00:00.000Z"]);
  await syncFilesystem(db, root);

  expect(findArtifact(db, "artifact-inference-1")).toMatchObject({
    id: "artifact-inference-1",
    kind: "inference-result",
    uri: "data/inference/inference-1/inference-result.json"
  });
  expect(one(db, "SELECT COUNT(*) AS count FROM artifacts WHERE kind = 'inference-result'").count).toBe(1);
  expect(one(db, "SELECT criticality_artifact_id FROM inference_sets WHERE id = ?", ["inference-1"]).criticality_artifact_id)
    .toBe("artifact-inference-1");

  const response = await createApiHandler({ db, root })(new Request("http://studio/api/criticality?inferenceId=inference-1"));
  expect(response.status).toBe(200);
  expect((await response.json()).predictions[0]).toMatchObject({
    metricPercentiles: [0.8, 0.6],
    structuralUniqueness: 0.7
  });
  db.close();
});
