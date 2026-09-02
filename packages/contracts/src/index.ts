/**
 * Versioned control-plane contracts shared by the Studio server, worker, and
 * client.
 *
 * This package intentionally has no runtime dependencies. Validation happens
 * at the boundary so the worker never turns browser input into a shell
 * command, and event payloads remain replayable by SSE clients.
 */

export const RUN_EVENT_SCHEMA_VERSION = 1;
export const ARTIFACT_MANIFEST_SCHEMA_VERSION = 1;
export const DATASET_MANIFEST_SCHEMA_VERSION = 1;
export const INFERENCE_RESULT_SCHEMA_VERSION = 1;
export const EXPERIMENT_SPEC_SCHEMA_VERSION = 1;
export const PUBLICATION_MANIFEST_SCHEMA_VERSION = 1;
export const DEFAULT_MODEL_CONFIG = "configs/models/multitask-v1.yaml";

export const RUNTIME_CONFIG_DEFAULTS = Object.freeze({
  device: "cpu",
  precision: "fp32",
  checkpointInterval: 1,
  metricInterval: 1,
  workerThreads: 1
});

export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };
export type ConfigValue = string | Record<string, JsonValue>;
export type RuntimeConfig = {
  device: string;
  precision: "fp32" | "fp16" | "bf16";
  checkpointInterval: number;
  metricInterval: number;
  workerThreads: number;
};

export type ExperimentSpec = {
  schemaVersion: typeof EXPERIMENT_SPEC_SCHEMA_VERSION;
  datasetId: string;
  modelConfig: ConfigValue;
  seed: number;
  runtime: RuntimeConfig;
};

export const RUN_KINDS = Object.freeze([
  "compile-snapshot",
  "simulate-criticality",
  "build-dataset",
  "train",
  "evaluate",
  "infer"
]);

export const RUN_STATUSES = Object.freeze([
  "queued",
  "claimed",
  "running",
  "succeeded",
  "failed",
  "cancelled",
  "orphaned"
]);

export const SIMILARITY_FACETS = Object.freeze([
  "general",
  "role",
  "service",
  "geometry",
  "resilience"
]);

const DATE_PATTERN = /^\d{4}-\d{2}-\d{2}$/;
const ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$/;

export function assertSafeId(value, field = "id") {
  if (typeof value !== "string" || !ID_PATTERN.test(value)) {
    throw new Error(`${field} must be a safe identifier`);
  }
  return value;
}

function requiredString(value, field, { safe = false } = {}) {
  if (typeof value !== "string" || value.trim() === "") {
    throw new Error(`${field} must be a non-empty string`);
  }
  return safe ? assertSafeId(value, field) : value.trim();
}

function optionalString(value, field) {
  if (value === undefined || value === null) return undefined;
  return requiredString(value, field);
}

function finiteNumber(value, field) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${field} must be a finite number`);
  }
  return value;
}

function nonNegativeInteger(value, field) {
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`${field} must be a non-negative integer`);
  }
  return value;
}

function optionalSafeId(value, field) {
  if (value === undefined || value === null || value === "") return undefined;
  return assertSafeId(requiredString(value, field), field);
}

function assertSha256(value, field) {
  if (typeof value !== "string" || !/^[a-f0-9]{64}$/i.test(value)) {
    throw new Error(`${field} must be a SHA-256 hex digest`);
  }
  return value.toLowerCase();
}

function validateRelativePath(value, field) {
  requiredString(value, field);
  if (value.startsWith("/") || value.split(/[\\/]+/).includes("..")) {
    throw new Error(`${field} must be a relative path`);
  }
  return value;
}

function validateConfigValue(value, field) {
  if (value === undefined || value === null) return undefined;
  if (typeof value !== "string" && (typeof value !== "object" || Array.isArray(value))) {
    throw new Error(`${field} must be a configuration string or object`);
  }
  if (typeof value === "string" && value.length > 16_384) {
    throw new Error(`${field} is too large`);
  }
  return value;
}

function validateConfigReference(value, field) {
  const config = requiredString(value, field);
  if (config.startsWith("/") || config.split(/[\\/]+/).includes("..")) {
    throw new Error(`${field} must be a repository-relative config path`);
  }
  if (!/\.(?:json|ya?ml)$/i.test(config)) {
    throw new Error(`${field} must point to a JSON or YAML config`);
  }
  return config;
}

function validateRuntimeConfig(value): RuntimeConfig {
  if (value !== undefined && (!value || typeof value !== "object" || Array.isArray(value))) {
    throw new Error("runtime must be an object");
  }
  const input = value || {};
  const device = input.device === undefined
    ? RUNTIME_CONFIG_DEFAULTS.device
    : requiredString(input.device, "runtime.device");
  const precision = (input.precision === undefined
    ? RUNTIME_CONFIG_DEFAULTS.precision
    : requiredString(input.precision, "runtime.precision")) as RuntimeConfig["precision"];
  if (!["fp32", "fp16", "bf16"].includes(precision)) {
    throw new Error("runtime.precision must be fp32, fp16, or bf16");
  }
  const positiveInteger = (candidate, field, fallback) => {
    if (candidate === undefined) return fallback;
    if (!Number.isInteger(candidate) || candidate < 1 || candidate > 100_000) {
      throw new Error(`${field} must be a positive integer`);
    }
    return candidate;
  };
  return {
    device,
    precision,
    checkpointInterval: positiveInteger(input.checkpointInterval, "runtime.checkpointInterval", RUNTIME_CONFIG_DEFAULTS.checkpointInterval),
    metricInterval: positiveInteger(input.metricInterval, "runtime.metricInterval", RUNTIME_CONFIG_DEFAULTS.metricInterval),
    workerThreads: positiveInteger(input.workerThreads, "runtime.workerThreads", RUNTIME_CONFIG_DEFAULTS.workerThreads)
  };
}

/** Validate and normalize a browser-submitted known run specification. */
export function validateRunSpec(input) {
  if (!input || typeof input !== "object" || Array.isArray(input)) {
    throw new Error("run spec must be an object");
  }
  const kind = requiredString(input.kind, "kind");
  if (!RUN_KINDS.includes(kind)) {
    throw new Error(`unsupported run kind: ${kind}`);
  }

  switch (kind) {
    case "compile-snapshot": {
      const serviceDate = requiredString(input.serviceDate, "serviceDate");
      if (!DATE_PATTERN.test(serviceDate) || Number.isNaN(Date.parse(`${serviceDate}T00:00:00Z`))) {
        throw new Error("serviceDate must use YYYY-MM-DD");
      }
      return {
        kind,
        feedRevisionId: assertSafeId(requiredString(input.feedRevisionId, "feedRevisionId"), "feedRevisionId"),
        serviceDate,
        compilerConfig: validateConfigValue(input.compilerConfig, "compilerConfig") ?? "default"
      };
    }
    case "simulate-criticality":
      return {
        kind,
        snapshotId: assertSafeId(requiredString(input.snapshotId, "snapshotId"), "snapshotId"),
        simulationConfig: validateConfigValue(input.simulationConfig, "simulationConfig") ?? "default"
      };
    case "build-dataset": {
      if (!Array.isArray(input.snapshotIds) || input.snapshotIds.length === 0 || input.snapshotIds.length > 10_000) {
        throw new Error("snapshotIds must contain between 1 and 10000 snapshots");
      }
      const snapshotIds = input.snapshotIds.map((value, index) =>
        assertSafeId(requiredString(value, `snapshotIds[${index}]`), `snapshotIds[${index}]`)
      );
      if (new Set(snapshotIds).size !== snapshotIds.length) {
        throw new Error("snapshotIds must not contain duplicates");
      }
      return {
        kind,
        snapshotIds,
        splitConfig: validateConfigValue(input.splitConfig, "splitConfig") ?? "system-level",
        featureSchema: optionalString(input.featureSchema, "featureSchema") ?? "station-line-relational-v2"
      };
    }
    case "train": {
      const seed = input.seed === undefined ? 7 : input.seed;
      if (!Number.isInteger(seed) || seed < 0 || seed > 2 ** 31 - 1) {
        throw new Error("seed must be a non-negative 32-bit integer");
      }
      return {
        kind,
        datasetId: assertSafeId(requiredString(input.datasetId, "datasetId"), "datasetId"),
        modelConfig: typeof input.modelConfig === "string"
          ? validateConfigReference(input.modelConfig, "modelConfig")
          : validateConfigValue(input.modelConfig, "modelConfig") ?? DEFAULT_MODEL_CONFIG,
        seed,
        runtime: validateRuntimeConfig(input.runtime)
      };
    }
    case "evaluate":
      return {
        kind,
        modelId: assertSafeId(requiredString(input.modelId, "modelId"), "modelId"),
        evaluationSuite: validateConfigValue(input.evaluationSuite, "evaluationSuite") ?? "default"
      };
    case "infer":
      return {
        kind,
        modelId: assertSafeId(requiredString(input.modelId, "modelId"), "modelId"),
        snapshotId: assertSafeId(requiredString(input.snapshotId, "snapshotId"), "snapshotId")
      };
    default:
      throw new Error(`unsupported run kind: ${kind}`);
  }
}

/** Validate the reproducible portion of a training experiment. */
export function validateExperimentSpec(input): ExperimentSpec {
  if (!input || typeof input !== "object" || Array.isArray(input)) {
    throw new Error("experiment spec must be an object");
  }
  const spec = validateRunSpec({ ...input, kind: "train" });
  return {
    schemaVersion: EXPERIMENT_SPEC_SCHEMA_VERSION,
    datasetId: spec.datasetId,
    modelConfig: spec.modelConfig,
    seed: spec.seed,
    runtime: spec.runtime
  };
}

export function stableStringify(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(",")}]`;
  return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stableStringify(value[key])}`).join(",")}}`;
}

export function sha256Hex(value) {
  const encoded = typeof value === "string" ? value : stableStringify(value);
  if (typeof Bun !== "undefined" && Bun.CryptoHasher) {
    const hasher = new Bun.CryptoHasher("sha256");
    hasher.update(encoded);
    return hasher.digest("hex");
  }
  throw new Error("sha256Hex requires the Bun runtime");
}

export function fingerprint(namespace, value) {
  return sha256Hex(`${namespace}:${stableStringify(value)}`);
}

export function normalizeRunSpec(input) {
  const spec = validateRunSpec(input);
  return { spec, fingerprint: fingerprint("run-spec-v1", spec) };
}

function validateEventBase(event) {
  if (!event || typeof event !== "object" || Array.isArray(event)) {
    throw new Error("run event must be an object");
  }
  if (event.schemaVersion !== RUN_EVENT_SCHEMA_VERSION) {
    throw new Error(`unsupported run event schema version: ${event.schemaVersion}`);
  }
  nonNegativeInteger(event.seq, "seq");
  assertSafeId(requiredString(event.runId, "runId"), "runId");
  requiredString(event.timestamp, "timestamp");
  requiredString(event.type, "type");
}

/** Validate the structured JSONL/SSE event protocol. */
export function validateRunEvent(event) {
  validateEventBase(event);
  switch (event.type) {
    case "run.queued":
    case "run.started":
    case "run.completed":
    case "run.failed":
    case "run.cancelled":
      if (event.type === "run.failed") {
        requiredString(event.code, "code");
        requiredString(event.message, "message");
      }
      break;
    case "step.started":
    case "step.completed":
      requiredString(event.step, "step");
      break;
    case "progress":
      requiredString(event.step, "step");
      nonNegativeInteger(event.completed, "completed");
      nonNegativeInteger(event.total, "total");
      requiredString(event.unit, "unit");
      if (event.total > 0 && event.completed > event.total) {
        throw new Error("progress completed cannot exceed total");
      }
      break;
    case "metric":
      requiredString(event.name, "name");
      nonNegativeInteger(event.step, "step");
      finiteNumber(event.value, "value");
      if (event.dimensions !== undefined && (!event.dimensions || typeof event.dimensions !== "object" || Array.isArray(event.dimensions))) {
        throw new Error("dimensions must be an object");
      }
      break;
    case "phase.started":
    case "phase.completed":
      requiredString(event.phase, "phase");
      if (event.total !== undefined) nonNegativeInteger(event.total, "total");
      if (event.durationMs !== undefined) finiteNumber(event.durationMs, "durationMs");
      break;
    case "epoch.started":
      requiredString(event.phase, "phase");
      nonNegativeInteger(event.epoch, "epoch");
      nonNegativeInteger(event.total, "total");
      break;
    case "learning-rate.changed":
      requiredString(event.phase, "phase");
      nonNegativeInteger(event.step, "step");
      finiteNumber(event.value, "value");
      break;
    case "checkpoint.created":
      requiredString(event.phase, "phase");
      requiredString(event.path, "path");
      if (event.epoch !== undefined) nonNegativeInteger(event.epoch, "epoch");
      if (event.step !== undefined) nonNegativeInteger(event.step, "step");
      break;
    case "heartbeat":
      requiredString(event.phase, "phase");
      nonNegativeInteger(event.step, "step");
      break;
    case "artifact.created":
      assertSafeId(requiredString(event.artifactId, "artifactId"), "artifactId");
      requiredString(event.artifactKind, "artifactKind");
      requiredString(event.uri, "uri");
      requiredString(event.sha256, "sha256");
      break;
    case "warning":
    case "error":
      requiredString(event.code, "code");
      requiredString(event.message, "message");
      break;
    default:
      throw new Error(`unsupported run event type: ${event.type}`);
  }
  return event;
}

export function validateArtifactManifest(manifest) {
  if (!manifest || typeof manifest !== "object" || Array.isArray(manifest)) {
    throw new Error("artifact manifest must be an object");
  }
  if (manifest.schemaVersion !== ARTIFACT_MANIFEST_SCHEMA_VERSION) {
    throw new Error("unsupported artifact manifest schema version");
  }
  assertSafeId(requiredString(manifest.artifactId, "artifactId"), "artifactId");
  requiredString(manifest.kind, "kind");
  requiredString(manifest.fingerprint, "fingerprint");
  requiredString(manifest.createdAt, "createdAt");
  if (!Array.isArray(manifest.inputs)) throw new Error("artifact inputs must be an array");
  for (const input of manifest.inputs) {
    if (!input || typeof input !== "object") throw new Error("artifact input must be an object");
    assertSafeId(requiredString(input.artifactId, "inputs.artifactId"), "inputs.artifactId");
    requiredString(input.fingerprint, "inputs.fingerprint");
  }
  if (manifest.sha256 !== undefined && manifest.sha256 !== null) {
    assertSha256(manifest.sha256, "sha256");
  }
  optionalSafeId(manifest.producingRunId, "producingRunId");
  if (manifest.gitCommit !== undefined && manifest.gitCommit !== null) {
    requiredString(manifest.gitCommit, "gitCommit");
  }
  if (manifest.configuration !== undefined &&
      (!manifest.configuration || typeof manifest.configuration !== "object" || Array.isArray(manifest.configuration))) {
    throw new Error("configuration must be an object");
  }
  if (manifest.files !== undefined) {
    if (!Array.isArray(manifest.files)) throw new Error("artifact files must be an array");
    for (const file of manifest.files) {
      if (!file || typeof file !== "object") throw new Error("artifact file must be an object");
      validateRelativePath(file.path, "files.path");
      if (file.sha256 !== undefined && file.sha256 !== null) assertSha256(file.sha256, "files.sha256");
    }
  }
  if (manifest.metadata !== undefined && (!manifest.metadata || typeof manifest.metadata !== "object" || Array.isArray(manifest.metadata))) {
    throw new Error("artifact metadata must be an object");
  }
  return manifest;
}

export function validateDatasetManifest(manifest) {
  if (!manifest || typeof manifest !== "object" || Array.isArray(manifest)) {
    throw new Error("dataset manifest must be an object");
  }
  if (manifest.schemaVersion !== DATASET_MANIFEST_SCHEMA_VERSION) {
    throw new Error("unsupported dataset manifest schema version");
  }
  assertSafeId(requiredString(manifest.datasetId, "datasetId"), "datasetId");
  requiredString(manifest.fingerprint, "fingerprint");
  requiredString(manifest.featureSchema, "featureSchema");
  if (!Array.isArray(manifest.snapshotIds) || manifest.snapshotIds.length === 0) {
    throw new Error("dataset snapshotIds must not be empty");
  }
  if (!manifest.split || typeof manifest.split !== "object" || Array.isArray(manifest.split)) {
    throw new Error("dataset split must be an object");
  }
  if (!manifest.objectives || typeof manifest.objectives !== "object" || Array.isArray(manifest.objectives)) {
    throw new Error("dataset objectives must be an object");
  }
  return manifest;
}

/** Validate the versioned result emitted by Rust inference commands. */
export function validateInferenceResult(result) {
  if (!result || typeof result !== "object" || Array.isArray(result)) {
    throw new Error("inference result must be an object");
  }
  if (result.schemaVersion !== INFERENCE_RESULT_SCHEMA_VERSION) {
    throw new Error(`unsupported inference result schema version: ${result.schemaVersion}`);
  }
  optionalSafeId(result.inferenceId, "inferenceId");
  assertSafeId(requiredString(result.modelId, "modelId"), "modelId");
  assertSafeId(requiredString(result.snapshotId, "snapshotId"), "snapshotId");
  if (!Array.isArray(result.metricNames) || result.metricNames.length === 0) {
    throw new Error("inference metricNames must not be empty");
  }
  if (result.metricNames.some((name) => typeof name !== "string" || !name.trim())) {
    throw new Error("inference metricNames must contain non-empty strings");
  }
  if (new Set(result.metricNames).size !== result.metricNames.length) {
    throw new Error("inference metricNames must be unique");
  }
  if (!Array.isArray(result.predictions)) throw new Error("inference predictions must be an array");
  for (const [index, prediction] of result.predictions.entries()) {
    if (!prediction || typeof prediction !== "object" || Array.isArray(prediction)) {
      throw new Error(`inference prediction ${index + 1} must be an object`);
    }
    const line = prediction.line ?? prediction.lineId;
    if ((typeof line !== "string" && typeof line !== "number") || String(line).trim() === "") {
      throw new Error(`inference prediction ${index + 1} has no line id`);
    }
    if (!Array.isArray(prediction.metrics) || prediction.metrics.length !== result.metricNames.length) {
      throw new Error(`inference prediction ${index + 1} metrics do not match metricNames`);
    }
    prediction.metrics.forEach((value, metricIndex) => finiteNumber(value, `predictions[${index}].metrics[${metricIndex}]`));
    if (prediction.metricPercentiles !== undefined) {
      if (!Array.isArray(prediction.metricPercentiles) || prediction.metricPercentiles.length !== result.metricNames.length) {
        throw new Error(`inference prediction ${index + 1} percentiles do not match metricNames`);
      }
      prediction.metricPercentiles.forEach((value, metricIndex) => {
        finiteNumber(value, `predictions[${index}].metricPercentiles[${metricIndex}]`);
        if (value < 0 || value > 1) throw new Error(`predictions[${index}].metricPercentiles[${metricIndex}] must be between 0 and 1`);
      });
    }
    finiteNumber(prediction.structuralUniqueness, `predictions[${index}].structuralUniqueness`);
  }
  return result;
}

export function validatePublicationManifest(manifest) {
  if (!manifest || typeof manifest !== "object" || Array.isArray(manifest)) {
    throw new Error("publication manifest must be an object");
  }
  if (manifest.schemaVersion !== PUBLICATION_MANIFEST_SCHEMA_VERSION) {
    throw new Error("unsupported publication manifest schema version");
  }
  assertSafeId(requiredString(manifest.publicationId, "publicationId"), "publicationId");
  assertSafeId(requiredString(manifest.slug, "slug"), "slug");
  requiredString(manifest.title, "title");
  requiredString(manifest.createdAt, "createdAt");
  for (const field of ["snapshotIds", "modelIds", "artifactIds"]) {
    if (!Array.isArray(manifest[field])) throw new Error(`publication ${field} must be an array`);
    for (const value of manifest[field]) assertSafeId(requiredString(value, `publication ${field}`), field);
  }
  if (!manifest.entries || typeof manifest.entries !== "object" || Array.isArray(manifest.entries)) {
    throw new Error("publication entries must be an object");
  }
  if (!Array.isArray(manifest.files)) throw new Error("publication files must be an array");
  for (const file of manifest.files) {
    if (!file || typeof file !== "object" || Array.isArray(file)) throw new Error("publication file must be an object");
    validateRelativePath(file.path, "publication files.path");
    assertSha256(file.sha256, "publication files.sha256");
    nonNegativeInteger(file.sizeBytes, "publication files.sizeBytes");
  }
  return manifest;
}

export function profileWeights(profile, overrides: Record<string, number | undefined> = {}) {
  // The CLI historically exposed this descriptive alias. Keep it equivalent
  // to the canonical role facet so API, CLI, and saved views cannot silently
  // fall back to the general mixture.
  if (profile === "network-role") profile = "role";
  if (profile === "role") return { role: 1, service: 0, geometry: 0, resilience: 0 };
  if (profile === "service") return { role: 0, service: 1, geometry: 0, resilience: 0 };
  if (profile === "geometry") return { role: 0, service: 0, geometry: 1, resilience: 0 };
  if (profile === "resilience") return { role: 0, service: 0, geometry: 0, resilience: 1 };
  const weights = {
    role: Number(overrides.role ?? 0.4),
    service: Number(overrides.service ?? 0.2),
    geometry: Number(overrides.geometry ?? 0.15),
    resilience: Number(overrides.resilience ?? 0.25)
  };
  for (const [name, value] of Object.entries(weights)) {
    if (!Number.isFinite(value) || value < 0) throw new Error(`${name} weight must be a non-negative number`);
  }
  const total = Object.values(weights).reduce((sum, value) => sum + value, 0);
  if (total <= 0) throw new Error("similarity weights must have a positive sum");
  return Object.fromEntries(Object.entries(weights).map(([name, value]) => [name, value / total]));
}
