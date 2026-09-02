import { expect, test } from "bun:test";
import {
  fingerprint,
  validateInferenceResult,
  normalizeRunSpec,
  profileWeights,
  stableStringify,
  validateArtifactManifest,
  validateDatasetManifest,
  validateExperimentSpec,
  validateRunEvent,
  validateRunSpec
} from "../src/index.ts";

test("run specs normalize to a stable fingerprint and reject shell-like input", () => {
  const first = normalizeRunSpec({
    kind: "infer",
    snapshotId: "snapshot-a",
    modelId: "model-a"
  });
  const second = normalizeRunSpec({
    modelId: "model-a",
    snapshotId: "snapshot-a",
    kind: "infer"
  });

  expect(first.spec).toEqual(second.spec);
  expect(first.fingerprint).toBe(second.fingerprint);
  expect(() => validateRunSpec({
    kind: "infer",
    snapshotId: "../../etc/passwd",
    modelId: "model-a"
  })).toThrow("safe identifier");
  expect(stableStringify({ z: 1, a: { d: 2, c: 3 } })).toBe('{"a":{"c":3,"d":2},"z":1}');
});

test("event validation covers replayable progress and rejects malformed metrics", () => {
  const event = {
    schemaVersion: 1,
    seq: 2,
    runId: "run-1",
    timestamp: "2026-09-02T00:00:00.000Z",
    type: "progress",
    step: "compile",
    completed: 2,
    total: 5,
    unit: "lines"
  };
  expect(validateRunEvent(event)).toEqual(event);
  expect(() => validateRunEvent({ ...event, value: Infinity, type: "metric", name: "loss", step: 1 })).toThrow("finite number");
  expect(() => validateRunEvent({ ...event, completed: 8 })).toThrow("cannot exceed total");
  expect(validateRunEvent({
    schemaVersion: 1,
    seq: 3,
    runId: "run-1",
    timestamp: "2026-09-02T00:00:00.000Z",
    type: "metric",
    phase: "metric-learning",
    epoch: 12,
    step: 1840,
    name: "validation_triplet_loss",
    value: 0.173,
    dimensions: { facet: "network-role", split: "validation" }
  }).type).toBe("metric");
  expect(validateRunEvent({
    schemaVersion: 1,
    seq: 4,
    runId: "run-1",
    timestamp: "2026-09-02T00:00:00.000Z",
    type: "checkpoint.created",
    phase: "criticality",
    path: "data/runs/run-1/model.json",
    epoch: 12,
    step: 1840
  }).type).toBe("checkpoint.created");
});

test("ExperimentSpec v1 makes runtime settings and seed explicit", () => {
  expect(validateExperimentSpec({
    datasetId: "dataset-1",
    modelConfig: "configs/models/multitask-v1.yaml",
    seed: 42,
    runtime: { device: "cuda:0", precision: "fp16", checkpointInterval: 5 }
  })).toEqual({
    schemaVersion: 1,
    datasetId: "dataset-1",
    modelConfig: "configs/models/multitask-v1.yaml",
    seed: 42,
    runtime: {
      device: "cuda:0",
      precision: "fp16",
      checkpointInterval: 5,
      metricInterval: 1,
      workerThreads: 1
    }
  });
  expect(() => validateExperimentSpec({
    datasetId: "dataset-1",
    modelConfig: "../../outside.yaml",
    seed: 42
  })).toThrow("repository-relative");
});

test("artifact and dataset manifests enforce lineage fields", () => {
  expect(validateArtifactManifest({
    schemaVersion: 1,
    artifactId: "artifact-1",
    kind: "graph",
    fingerprint: fingerprint("artifact", { id: 1 }),
    createdAt: "2026-09-02T00:00:00.000Z",
    inputs: []
  }).kind).toBe("graph");
  expect(validateDatasetManifest({
    schemaVersion: 1,
    datasetId: "dataset-1",
    fingerprint: "abc",
    featureSchema: "station-line-relational-v2",
    snapshotIds: ["snapshot-1"],
    split: { holdout: ["city-1"] },
    objectives: { masked: 3 }
  }).datasetId).toBe("dataset-1");
  expect(() => validateDatasetManifest({
    schemaVersion: 1,
    datasetId: "d",
    fingerprint: "f",
    featureSchema: "schema"
  })).toThrow("snapshotIds");
});

test("similarity weights are explicit and normalized", () => {
  expect(profileWeights("role")).toEqual({ role: 1, service: 0, geometry: 0, resilience: 0 });
  expect(profileWeights("network-role")).toEqual({ role: 1, service: 0, geometry: 0, resilience: 0 });
  expect(profileWeights("general", { role: 2, service: 1, geometry: 1, resilience: 0 })).toEqual({
    role: 0.5,
    service: 0.25,
    geometry: 0.25,
    resilience: 0
  });
  expect(() => profileWeights("general", { role: 0, service: 0, geometry: 0, resilience: 0 })).toThrow("positive sum");
});

test("versioned inference results carry named values and optional percentiles", () => {
  const result = {
    schemaVersion: 1,
    inferenceId: "inference-1",
    modelId: "model-1",
    snapshotId: "snapshot-1",
    metricNames: ["accessibility_auc_loss"],
    predictions: [{
      line: 2,
      metrics: [0.25],
      metricPercentiles: [0.9],
      structuralUniqueness: 0.7
    }]
  };
  expect(validateInferenceResult(result)).toEqual(result);
  expect(() => validateInferenceResult({ ...result, predictions: [{ ...result.predictions[0], metricPercentiles: [2] }] })).toThrow("between 0 and 1");
});
