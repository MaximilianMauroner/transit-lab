/**
 * Versioned control-plane contracts shared by the Studio server, worker, and
 * client.
 *
 * This package intentionally has no runtime dependencies. Validation happens
 * at the boundary so the worker never turns browser input into a shell
 * command, and event payloads remain replayable by SSE clients.
 */

export const RUN_EVENT_SCHEMA_VERSION = 2;
export const TRAINING_CHECKPOINT_SCHEMA_VERSION = 1;
export const TRAINING_CONTROL_SCHEMA_VERSION = 1;
export const ARTIFACT_MANIFEST_SCHEMA_VERSION = 1;
export const DATASET_MANIFEST_SCHEMA_VERSION = 1;
export const EVALUATION_RESULT_SCHEMA_VERSION = 1;
export const INFERENCE_RESULT_SCHEMA_VERSION = 1;
export const EXPERIMENT_SPEC_SCHEMA_VERSION = 1;
export const PUBLICATION_MANIFEST_SCHEMA_VERSION = 1;
export const BENCHMARK_RESULT_SCHEMA_VERSION = 1;
export const DEFAULT_MODEL_CONFIG = "configs/models/multitask-v1.yaml";

export const RUNTIME_CONFIG_DEFAULTS = Object.freeze({
  device: "cpu",
  precision: "fp32",
  checkpointInterval: 1,
  metricInterval: 1,
  workerThreads: 1
});

export const RUN_DESIRED_STATES = Object.freeze(["running", "paused", "cancelled"]);
export const RUN_OBSERVED_STATES = Object.freeze([
  "queued",
  "starting",
  "claimed",
  "running",
  "checkpointing",
  "paused",
  "succeeded",
  "failed",
  "cancelled",
  "interrupted",
  "orphaned"
]);

export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };
export type ConfigValue = string | Record<string, JsonValue>;
export type RuntimeConfig = {
  device: string;
  precision: "fp32" | "fp16" | "bf16";
  backend?: "reference" | "libtorch";
  checkpointInterval: number;
  metricInterval: number;
  workerThreads: number;
  checkpointEverySteps?: number;
  checkpointEverySeconds?: number;
  maxAttemptSeconds?: number;
  checkpointGraceSeconds?: number;
  gradientAccumulation?: number;
  rayonThreads?: number;
  allowedWindows?: TrainingWindow[];
};

export type TrainingWindow = {
  days: string[];
  start: string;
  end: string;
  timezone: string;
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
  "orphaned",
  "starting",
  "checkpointing",
  "paused",
  "interrupted"
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
  const segments = value.split(/[\\/]+/);
  if (/^(?:[\\/]|[A-Za-z]:)/.test(value) || segments.some((segment) => segment === "" || segment === "." || segment === "..")) {
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
  const runtime: RuntimeConfig = {
    device,
    precision,
    checkpointInterval: positiveInteger(input.checkpointInterval, "runtime.checkpointInterval", RUNTIME_CONFIG_DEFAULTS.checkpointInterval),
    metricInterval: positiveInteger(input.metricInterval, "runtime.metricInterval", RUNTIME_CONFIG_DEFAULTS.metricInterval),
    workerThreads: positiveInteger(input.workerThreads, "runtime.workerThreads", RUNTIME_CONFIG_DEFAULTS.workerThreads)
  };
  if (input.backend !== undefined) {
    if (input.backend !== "reference" && input.backend !== "libtorch") {
      throw new Error("runtime.backend must be reference or libtorch");
    }
    runtime.backend = input.backend;
  }
  const optionalPositive = [
    ["checkpointEverySteps", "runtime.checkpointEverySteps"],
    ["checkpointEverySeconds", "runtime.checkpointEverySeconds"],
    ["maxAttemptSeconds", "runtime.maxAttemptSeconds"],
    ["checkpointGraceSeconds", "runtime.checkpointGraceSeconds"],
    ["gradientAccumulation", "runtime.gradientAccumulation"],
    ["rayonThreads", "runtime.rayonThreads"]
  ];
  for (const [key, field] of optionalPositive) {
    if (input[key] !== undefined) runtime[key] = positiveInteger(input[key], field, 1);
  }
  if (input.allowedWindows !== undefined) {
    if (!Array.isArray(input.allowedWindows) || input.allowedWindows.length > 31) {
      throw new Error("runtime.allowedWindows must contain at most 31 windows");
    }
    const timePattern = /^([01]\d|2[0-3]):[0-5]\d$/;
    const dayNames = new Set(["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]);
    runtime.allowedWindows = input.allowedWindows.map((window, index) => {
      if (!window || typeof window !== "object" || Array.isArray(window)) {
        throw new Error(`runtime.allowedWindows[${index}] must be an object`);
      }
      if (!Array.isArray(window.days) || window.days.length === 0 ||
          window.days.some((day) => typeof day !== "string" || !dayNames.has(day.toLowerCase()))) {
        throw new Error(`runtime.allowedWindows[${index}].days is invalid`);
      }
      if (typeof window.start !== "string" || !timePattern.test(window.start) ||
          typeof window.end !== "string" || !timePattern.test(window.end)) {
        throw new Error(`runtime.allowedWindows[${index}] times must use HH:MM`);
      }
      const timezone = requiredString(window.timezone, `runtime.allowedWindows[${index}].timezone`);
      return {
        days: [...new Set(window.days.map((day) => day.toLowerCase()))],
        start: window.start,
        end: window.end,
        timezone
      };
    });
  }
  return runtime;
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
    case "evaluate": {
      const datasetId = optionalSafeId(input.datasetId, "datasetId");
      return {
        kind,
        modelId: assertSafeId(requiredString(input.modelId, "modelId"), "modelId"),
        ...(datasetId ? { datasetId } : {}),
        evaluationSuite: validateConfigValue(input.evaluationSuite, "evaluationSuite") ?? "default"
      };
    }
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
  if (![1, RUN_EVENT_SCHEMA_VERSION].includes(event.schemaVersion)) {
    throw new Error(`unsupported run event schema version: ${event.schemaVersion}`);
  }
  nonNegativeInteger(event.seq, "seq");
  assertSafeId(requiredString(event.runId, "runId"), "runId");
  requiredString(event.timestamp, "timestamp");
  requiredString(event.type, "type");
  if (event.attemptId !== undefined) assertSafeId(requiredString(event.attemptId, "attemptId"), "attemptId");
  if (event.attemptSeq !== undefined) nonNegativeInteger(event.attemptSeq, "attemptSeq");
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
    case "run.paused":
    case "run.resumed":
    case "run.time-sliced":
    case "run.recovered":
    case "attempt.started":
    case "attempt.ended":
    case "pause.requested":
      if (event.attemptId !== undefined) assertSafeId(requiredString(event.attemptId, "attemptId"), "attemptId");
      if (event.reason !== undefined) requiredString(event.reason, "reason");
      if (event.path !== undefined) requiredString(event.path, "path");
      if (event.exitCode !== undefined) finiteNumber(event.exitCode, "exitCode");
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
    case "checkpoint.started":
      requiredString(event.phase, "phase");
      nonNegativeInteger(event.step, "step");
      break;
    case "checkpoint.committed":
      requiredString(event.phase, "phase");
      nonNegativeInteger(event.step, "step");
      requiredString(event.path, "path");
      break;
    case "checkpoint.failed":
      requiredString(event.phase, "phase");
      nonNegativeInteger(event.step, "step");
      requiredString(event.code, "code");
      requiredString(event.message, "message");
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

function checkpointSha256(value, field) {
  assertSha256(value, field);
}

/** Match the Rust checkpoint manifest fingerprint over its ordered payload descriptors. */
export function trainingCheckpointFingerprint(files) {
  const ordered = files
    .map(({ path, sha256, sizeBytes }) => ({ path, sha256, sizeBytes }))
    .sort((left, right) => left.path < right.path ? -1 : left.path > right.path ? 1 : 0);
  return sha256Hex(JSON.stringify(ordered));
}

/** Validate the committed directory checkpoint manifest at the worker boundary. */
export function validateTrainingCheckpointManifest(manifest) {
  if (!manifest || typeof manifest !== "object" || Array.isArray(manifest)) {
    throw new Error("training checkpoint manifest must be an object");
  }
  if (manifest.schemaVersion !== TRAINING_CHECKPOINT_SCHEMA_VERSION) {
    throw new Error(`unsupported training checkpoint schema: ${manifest.schemaVersion}`);
  }
  assertSafeId(requiredString(manifest.runId, "runId"), "runId");
  if (manifest.attemptId !== undefined && manifest.attemptId !== null) {
    assertSafeId(requiredString(manifest.attemptId, "attemptId"), "attemptId");
  }
  nonNegativeInteger(manifest.globalStep, "globalStep");
  requiredString(manifest.phase, "phase");
  requiredString(manifest.datasetFingerprint, "datasetFingerprint");
  requiredString(manifest.configFingerprint, "configFingerprint");
  requiredString(manifest.codeCommit, "codeCommit");
  requiredString(manifest.backend, "backend");
  requiredString(manifest.backendVersion, "backendVersion");
  requiredString(manifest.deviceType, "deviceType");
  if (manifest.status !== "committed") throw new Error("training checkpoint is not committed");
  checkpointSha256(manifest.checkpointFingerprint, "checkpointFingerprint");
  if (!Array.isArray(manifest.files) || manifest.files.length === 0) {
    throw new Error("training checkpoint files must be a non-empty array");
  }
  const names = new Set();
  for (const file of manifest.files) {
    if (!file || typeof file !== "object" || Array.isArray(file)) throw new Error("checkpoint file must be an object");
    const path = requiredString(file.path, "files.path");
    if (!/^[A-Za-z0-9._-]+$/.test(path) || path === "manifest.json" || path === "." || path === ".." || names.has(path)) {
      throw new Error("checkpoint file paths must be unique payload names");
    }
    names.add(path);
    checkpointSha256(file.sha256, "files.sha256");
    nonNegativeInteger(file.sizeBytes, "files.sizeBytes");
  }
  if (trainingCheckpointFingerprint(manifest.files) !== String(manifest.checkpointFingerprint).toLowerCase()) {
    throw new Error("training checkpoint fingerprint does not match its payload descriptors");
  }
  return manifest;
}

export function validateTrainingControl(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("training control must be an object");
  }
  if (value.schemaVersion !== TRAINING_CONTROL_SCHEMA_VERSION) {
    throw new Error(`unsupported training control schema: ${value.schemaVersion}`);
  }
  if (!RUN_DESIRED_STATES.includes(value.desiredState)) {
    throw new Error(`unsupported training desired state: ${value.desiredState}`);
  }
  if (value.checkpointRequested !== undefined && typeof value.checkpointRequested !== "boolean") {
    throw new Error("checkpointRequested must be boolean");
  }
  if (value.requestedAt !== undefined && value.requestedAt !== null) requiredString(value.requestedAt, "requestedAt");
  if (value.reason !== undefined && value.reason !== null) requiredString(value.reason, "reason");
  return value;
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

/** Validate the ranking report emitted by the Rust evaluation command. */
export function validateEvaluationResult(result) {
  if (!result || typeof result !== "object" || Array.isArray(result)) {
    throw new Error("evaluation result must be an object");
  }
  if (result.schemaVersion !== EVALUATION_RESULT_SCHEMA_VERSION) {
    throw new Error(`unsupported evaluation result schema version: ${result.schemaVersion}`);
  }
  assertSafeId(requiredString(result.datasetId, "datasetId"), "datasetId");
  requiredString(result.datasetFingerprint, "datasetFingerprint");
  if (result.modelId !== undefined && result.modelId !== null) {
    assertSafeId(requiredString(result.modelId, "modelId"), "modelId");
  }
  if (result.modelPath !== undefined && result.modelPath !== null) requiredString(result.modelPath, "modelPath");
  const split = requiredString(result.split, "split");
  if (!["all", "train", "validation", "test"].includes(split)) {
    throw new Error("evaluation split must be all, train, validation, or test");
  }
  if (!Number.isInteger(result.topK) || result.topK < 1) throw new Error("evaluation topK must be a positive integer");
  if (!Number.isInteger(result.trainingExamples) || result.trainingExamples < 0) throw new Error("evaluation trainingExamples must be a non-negative integer");
  if (!Number.isInteger(result.fitExamples) || result.fitExamples < 0) throw new Error("evaluation fitExamples must be a non-negative integer");
  if (!Array.isArray(result.metrics) || result.metrics.length === 0) throw new Error("evaluation metrics must not be empty");
  const metricNames = new Set();
  for (const [index, metric] of result.metrics.entries()) {
    if (!metric || typeof metric !== "object" || Array.isArray(metric)) throw new Error(`evaluation metric ${index + 1} must be an object`);
    const baseline = requiredString(metric.baseline, `metrics[${index}].baseline`);
    if (metricNames.has(baseline)) throw new Error(`evaluation baseline ${baseline} is duplicated`);
    metricNames.add(baseline);
    if (!metric.values || typeof metric.values !== "object" || Array.isArray(metric.values)) throw new Error(`metrics[${index}].values must be an object`);
    const values = metric.values;
    for (const name of ["examples", "snapshots"]) {
      if (!Number.isInteger(values[name]) || values[name] < 0) throw new Error(`metrics[${index}].values.${name} must be a non-negative integer`);
    }
    for (const name of ["spearman", "pairwiseAccuracy", "topKOverlap"]) {
      if (values[name] !== null && values[name] !== undefined) finiteNumber(values[name], `metrics[${index}].values.${name}`);
    }
    if (values.spearman !== null && values.spearman !== undefined && (values.spearman < -1 || values.spearman > 1)) {
      throw new Error(`metrics[${index}].values.spearman must be between -1 and 1`);
    }
    for (const name of ["pairwiseAccuracy", "topKOverlap"]) {
      if (values[name] !== null && values[name] !== undefined && (values[name] < 0 || values[name] > 1)) {
        throw new Error(`metrics[${index}].values.${name} must be between 0 and 1`);
      }
    }
  }
  if (result.createdAt !== undefined && result.createdAt !== null) requiredString(result.createdAt, "createdAt");
  return result;
}

/** Validate a measured routing or training throughput artifact. */
export function validateBenchmarkResult(result) {
  if (!result || typeof result !== "object" || Array.isArray(result)) {
    throw new Error("benchmark result must be an object");
  }
  if (result.schemaVersion !== BENCHMARK_RESULT_SCHEMA_VERSION) {
    throw new Error(`unsupported benchmark result schema version: ${result.schemaVersion}`);
  }
  const benchmark = requiredString(result.benchmark, "benchmark");
  if (!["routing", "train-step", "threads"].includes(benchmark)) {
    throw new Error("benchmark must be routing, train-step, or threads");
  }
  const workload = requiredString(result.workload, "workload");
  if (!["routing", "train-step", "mixed"].includes(workload)) {
    throw new Error("benchmark workload must be routing, train-step, or mixed");
  }
  for (const field of ["snapshotId", "graphId"]) {
    if (result[field] !== undefined && result[field] !== null) {
      assertSafeId(requiredString(result[field], field), field);
    }
  }
  const nonNegativeInteger = (value, field) => {
    if (!Number.isInteger(value) || value < 0) throw new Error(`${field} must be a non-negative integer`);
  };
  for (const field of ["warmupUnits", "measuredUnits", "estimatedWorkUnits"]) {
    if (result[field] !== undefined && result[field] !== null) nonNegativeInteger(result[field], field);
  }
  for (const field of ["medianMilliseconds", "p95Milliseconds", "throughput"]) {
    if (result[field] !== undefined && result[field] !== null) {
      finiteNumber(result[field], field);
      if (result[field] < 0) throw new Error(`${field} must be non-negative`);
    }
  }
  if (result.peakResidentMemoryBytes !== undefined && result.peakResidentMemoryBytes !== null) {
    nonNegativeInteger(result.peakResidentMemoryBytes, "peakResidentMemoryBytes");
  }
  if (result.throughputUnit !== undefined && result.throughputUnit !== null) {
    requiredString(result.throughputUnit, "throughputUnit");
  }
  if (result.threadConfiguration !== undefined &&
      (!result.threadConfiguration || typeof result.threadConfiguration !== "object" || Array.isArray(result.threadConfiguration))) {
    throw new Error("threadConfiguration must be an object");
  }
  if (result.runtime !== undefined &&
      (!result.runtime || typeof result.runtime !== "object" || Array.isArray(result.runtime))) {
    throw new Error("benchmark runtime must be an object");
  }
  if (result.graphCounts !== undefined &&
      (!result.graphCounts || typeof result.graphCounts !== "object" || Array.isArray(result.graphCounts))) {
    throw new Error("graphCounts must be an object");
  }
  if (result.reports !== undefined) {
    if (!Array.isArray(result.reports)) throw new Error("benchmark reports must be an array");
    for (const [index, report] of result.reports.entries()) {
      if (!report || typeof report !== "object" || Array.isArray(report)) {
        throw new Error(`benchmark report ${index + 1} must be an object`);
      }
      if (report.workload !== undefined && !["routing", "train-step"].includes(report.workload)) {
        throw new Error(`benchmark report ${index + 1} has an unsupported workload`);
      }
      if (report.threads !== undefined) nonNegativeInteger(report.threads, `reports[${index}].threads`);
      if (report.throughput !== undefined) {
        finiteNumber(report.throughput, `reports[${index}].throughput`);
        if (report.throughput < 0) throw new Error(`reports[${index}].throughput must be non-negative`);
      }
    }
  }
  if (result.createdAt !== undefined && result.createdAt !== null) requiredString(result.createdAt, "createdAt");
  if (result.throughput === undefined && (!Array.isArray(result.reports) || result.reports.length === 0)) {
    throw new Error("benchmark result must contain throughput or reports");
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
