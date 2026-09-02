import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { relative, resolve, sep } from "node:path";
import {
  DEFAULT_MODEL_CONFIG,
  EXPERIMENT_SPEC_SCHEMA_VERSION,
  fingerprint,
  stableStringify,
  validateExperimentSpec,
  validateRunSpec
} from "../../contracts/src/index.ts";
import { dataRoot, repositoryRoot } from "./database.ts";

const CONFIG_DIRECTORY = "configs";

function inside(root, candidate) {
  const base = resolve(root);
  const path = resolve(candidate);
  return path === base || path.startsWith(`${base}${sep}`);
}

function cloneJson(value) {
  return JSON.parse(JSON.stringify(value));
}

function parseConfig(path, source) {
  try {
    const value = /\.ya?ml$/i.test(path) ? Bun.YAML.parse(source) : JSON.parse(source);
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      throw new Error("model config must decode to an object");
    }
    return cloneJson(value);
  } catch (error) {
    throw new Error(`could not decode model config ${path}: ${error instanceof Error ? error.message : String(error)}`);
  }
}

function setSeeds(value, seed) {
  if (Array.isArray(value)) return value.map((entry) => setSeeds(entry, seed));
  if (!value || typeof value !== "object") return value;
  return Object.fromEntries(Object.entries(value).map(([key, child]) => [
    key,
    key === "seed" ? seed : setSeeds(child, seed)
  ]));
}

function modelConfig(root, value) {
  if (typeof value !== "string") return { source: null, value: cloneJson(value) };
  const path = resolve(root, value || DEFAULT_MODEL_CONFIG);
  if (!inside(root, path) || relative(root, path).split(sep)[0] !== CONFIG_DIRECTORY) {
    throw new Error("modelConfig must resolve inside the repository configs directory");
  }
  return { source: value, value: parseConfig(value, readFileSync(path, "utf8")) };
}

function writeImmutable(path, document) {
  mkdirSync(resolve(path, ".."), { recursive: true });
  const encoded = `${JSON.stringify(document, null, 2)}\n`;
  try {
    writeFileSync(path, encoded, { flag: "wx" });
  } catch (error) {
    if (error?.code !== "EEXIST") throw error;
    const existing = readFileSync(path, "utf8");
    if (existing !== encoded) throw new Error(`refusing to overwrite immutable resolved config ${path}`);
  }
}

/**
 * Materialize the exact configuration a worker is allowed to execute.
 *
 * The returned fingerprint excludes the run ID so retries of the same
 * experiment can be compared by content, while the file itself remains
 * immutable and local to the run for auditability.
 */
export function materializeResolvedRunConfig(root = repositoryRoot(), runId, rawSpec) {
  const spec = validateRunSpec(rawSpec);
  let document;
  let configFingerprint;
  if (spec.kind === "train") {
    const experiment = validateExperimentSpec(spec);
    const resolved = modelConfig(root, experiment.modelConfig);
    const rustConfig = setSeeds(resolved.value, experiment.seed);
    const fingerprintInput = {
      schemaVersion: EXPERIMENT_SPEC_SCHEMA_VERSION,
      datasetId: experiment.datasetId,
      modelConfig: rustConfig,
      seed: experiment.seed,
      runtime: experiment.runtime
    };
    configFingerprint = fingerprint("resolved-experiment-config-v1", fingerprintInput);
    document = {
      schemaVersion: EXPERIMENT_SPEC_SCHEMA_VERSION,
      runId,
      kind: spec.kind,
      datasetId: experiment.datasetId,
      seed: experiment.seed,
      runtime: experiment.runtime,
      sourceConfig: resolved.source,
      configFingerprint,
      modelConfig: rustConfig
    };
  } else {
    configFingerprint = fingerprint("resolved-run-config-v1", spec);
    document = {
      schemaVersion: EXPERIMENT_SPEC_SCHEMA_VERSION,
      runId,
      kind: spec.kind,
      configFingerprint,
      spec: cloneJson(spec)
    };
  }
  const path = resolve(dataRoot(root), "runs", runId, "resolved-config.json");
  writeImmutable(path, document);
  return { path, configFingerprint, document, spec };
}

export function readResolvedRunConfig(path) {
  const document = JSON.parse(readFileSync(path, "utf8"));
  if (!document || document.schemaVersion !== EXPERIMENT_SPEC_SCHEMA_VERSION ||
      typeof document.configFingerprint !== "string") {
    throw new Error(`resolved config ${path} does not match ExperimentSpec v1`);
  }
  return document;
}

export function resolvedConfigFingerprint(path) {
  return readResolvedRunConfig(path).configFingerprint;
}

export function resolvedConfigJson(document) {
  return stableStringify(document);
}
