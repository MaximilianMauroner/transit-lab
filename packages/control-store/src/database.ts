import { Database } from "bun:sqlite";
import { randomUUID } from "node:crypto";
import { mkdirSync } from "node:fs";
import { dirname, relative, resolve, sep } from "node:path";
import {
  RUN_EVENT_SCHEMA_VERSION,
  assertSafeId,
  normalizeRunSpec,
  validateRunEvent
} from "../../contracts/src/index.ts";
import { materializeResolvedRunConfig } from "./experiments.ts";
import { writePublicationBundle } from "./publication-bundle.ts";
import { pushDatabaseSchema } from "./schema.ts";

export function repositoryRoot() {
  return resolve(process.env.TRANSIT_LAB_ROOT || resolve(import.meta.dir, "../../../"));
}

export function dataRoot(root = repositoryRoot()) {
  return resolve(root, process.env.TRANSIT_LAB_DATA_DIR || process.env.TRANSIT_LAB_DATA_ROOT || "data");
}

export function databasePath(root = repositoryRoot()) {
  const configured = process.env.TRANSIT_LAB_DB || "data/transit-lab.sqlite";
  return configured === ":memory:" ? configured : resolve(root, configured);
}

export function now() {
  return new Date().toISOString();
}

export function parseJson(value, fallback = {}) {
  if (value === null || value === undefined || value === "") return fallback;
  try {
    return JSON.parse(value);
  } catch {
    return fallback;
  }
}

export function json(value) {
  return JSON.stringify(value ?? {});
}

export function createDatabase(root = repositoryRoot(), path = databasePath(root)) {
  if (path !== ":memory:") mkdirSync(dirname(path), { recursive: true });
  const db = new Database(path, { create: true });
  pushDatabaseSchema(db);
  return db;
}

export function one(db, sql, params = []) {
  return db.query(sql).get(...params) ?? null;
}

export function all(db, sql, params = []) {
  return db.query(sql).all(...params);
}

export function run(db, sql, params = []) {
  return db.query(sql).run(...params);
}

export function appendRunEvent(db, runId, event) {
  validateRunEvent(event);
  if (event.runId !== runId) throw new Error("run event runId does not match the target run");
  const insert = db.transaction(() => {
    const next = one(db, "SELECT COALESCE(MAX(seq), -1) + 1 AS seq FROM run_events WHERE run_id = ?", [runId]);
    const sequence = Number(next?.seq ?? 0);
    const stored = { ...event, seq: sequence, schemaVersion: RUN_EVENT_SCHEMA_VERSION, runId };
    validateRunEvent(stored);
    run(db, "INSERT INTO run_events(run_id, seq, event_json, created_at) VALUES (?, ?, ?, ?)", [
      runId,
      sequence,
      json(stored),
      now()
    ]);
    return stored;
  });
  return insert.immediate();
}

export function appendRunEventType(db, runId, type, payload = {}) {
  return appendRunEvent(db, runId, {
    schemaVersion: RUN_EVENT_SCHEMA_VERSION,
    seq: 0,
    runId,
    timestamp: now(),
    type,
    ...payload
  });
}

export function addRunLog(db, runId, stream, line) {
  run(db, "INSERT INTO run_logs(run_id, stream, line, created_at) VALUES (?, ?, ?, ?)", [
    runId,
    stream,
    String(line),
    now()
  ]);
}

export function getRun(db, runId) {
  const row = one(db, "SELECT * FROM runs WHERE id = ?", [runId]);
  if (!row) return null;
  return hydrateRun(row, db);
}

export type HydratedRun = {
  id: string;
  projectId: string;
  kind: string;
  status: string;
  spec: Record<string, unknown>;
  fingerprint: string;
  configFingerprint: string;
  resolvedConfigPath: string;
  snapshotId: string | null;
  datasetId: string | null;
  modelId: string | null;
  progress: { completed: number; total: number; unit: string };
  currentStep: string;
  workerId: string | null;
  gitCommit: string;
  cancelRequested: boolean;
  error: { code: string; message: string } | null;
  startedAt: string | null;
  finishedAt: string | null;
  createdAt: string;
  updatedAt: string;
  steps?: Array<Record<string, unknown>>;
  logs?: Array<Record<string, unknown>>;
  events?: Array<Record<string, unknown>>;
  [key: string]: unknown;
};

export function hydrateRun(row, db?: unknown): HydratedRun {
  const run: HydratedRun = {
    id: row.id,
    projectId: row.project_id,
    kind: row.kind,
    status: row.status,
    spec: parseJson(row.spec_json, {}),
    fingerprint: row.fingerprint,
    configFingerprint: row.config_fingerprint || "",
    resolvedConfigPath: row.resolved_config_path || "",
    snapshotId: row.snapshot_id,
    datasetId: row.dataset_id,
    modelId: row.model_id,
    progress: {
      completed: Number(row.progress_completed || 0),
      total: Number(row.progress_total || 0),
      unit: row.progress_unit || ""
    },
    currentStep: row.current_step || "",
    workerId: row.worker_id,
    gitCommit: row.git_commit || "",
    cancelRequested: Boolean(row.cancel_requested),
    error: row.error_code ? { code: row.error_code, message: row.error_message || "" } : null,
    startedAt: row.started_at,
    finishedAt: row.finished_at,
    createdAt: row.created_at,
    updatedAt: row.updated_at
  };
  if (db) {
    run.steps = all(db, "SELECT * FROM run_steps WHERE run_id = ? ORDER BY id", [row.id]).map((step) => ({
      step: step.step,
      status: step.status,
      startedAt: step.started_at,
      finishedAt: step.finished_at,
      inputFingerprint: step.input_fingerprint,
      outputFingerprint: step.output_fingerprint,
      metrics: parseJson(step.metrics_json, {})
    }));
  }
  return run;
}

export function getRunEvents(db, runId, after = -1, limit = 500) {
  return all(db, "SELECT event_json FROM run_events WHERE run_id = ? AND seq > ? ORDER BY seq LIMIT ?", [
    runId,
    Number.isFinite(after) ? after : -1,
    Math.max(1, Math.min(2_000, limit))
  ]).map((row) => parseJson(row.event_json, null)).filter(Boolean);
}

export function getRunLogs(db, runId, limit = 2_000) {
  return all(db, "SELECT stream, line, created_at AS createdAt FROM run_logs WHERE run_id = ? ORDER BY id DESC LIMIT ?", [
    runId,
    Math.max(1, Math.min(10_000, limit))
  ]).reverse();
}

export function updateRun(db, runId, fields) {
  const allowed = {
    status: "status",
    progressCompleted: "progress_completed",
    progressTotal: "progress_total",
    progressUnit: "progress_unit",
    currentStep: "current_step",
    workerId: "worker_id",
    errorCode: "error_code",
    errorMessage: "error_message",
    startedAt: "started_at",
    finishedAt: "finished_at",
    cancelRequested: "cancel_requested",
    snapshotId: "snapshot_id",
    datasetId: "dataset_id",
    modelId: "model_id"
  };
  const entries = Object.entries(fields).filter(([key, value]) => allowed[key] && value !== undefined);
  if (!entries.length) return;
  const assignments = entries.map(([key]) => `${allowed[key]} = ?`).join(", ");
  const values = entries.map(([, value]) => value === true ? 1 : value === false ? 0 : value);
  values.push(now(), runId);
  run(db, `UPDATE runs SET ${assignments}, updated_at = ? WHERE id = ?`, values);
}

export function createRun(db, spec, projectId = "project-local", root = repositoryRoot()) {
  const normalized = normalizeRunSpec(spec);
  const id = `run_${randomUUID().replaceAll("-", "")}`;
  const resolved = materializeResolvedRunConfig(root, id, normalized.spec);
  const timestamp = now();
  run(db, `INSERT OR IGNORE INTO projects(id, name, description, created_at) VALUES (?, ?, ?, ?)`, [
    projectId,
    "Transit Lab",
    "Local-first GTFS representation, simulation, and evaluation workspace",
    timestamp
  ]);
  run(db, `INSERT INTO runs(
    id, project_id, kind, status, spec_json, fingerprint, snapshot_id, dataset_id, model_id,
    config_fingerprint, resolved_config_path, git_commit, created_at, updated_at
  ) VALUES (?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`, [
    id,
    projectId,
    normalized.spec.kind,
    json(normalized.spec),
    normalized.fingerprint,
    normalized.spec.snapshotId || null,
    normalized.spec.datasetId || null,
    normalized.spec.modelId || null,
    resolved.configFingerprint,
    relative(dataRoot(root), resolved.path).split(sep).join("/"),
    process.env.TRANSIT_LAB_GIT_COMMIT || "working-tree",
    timestamp,
    timestamp
  ]);
  appendRunEventType(db, id, "run.queued", { status: "queued", specFingerprint: normalized.fingerprint });
  return getRun(db, id);
}

export function listRuns(db, limit = 100) {
  return all(db, "SELECT * FROM runs ORDER BY created_at DESC, id DESC LIMIT ?", [
    Math.max(1, Math.min(500, Number(limit) || 100))
  ]).map((row) => hydrateRun(row));
}

export function requestRunCancellation(db, runId) {
  const current = one(db, "SELECT status FROM runs WHERE id = ?", [runId]);
  if (!current) return null;
  if (["queued", "claimed", "running"].includes(current.status)) {
    updateRun(db, runId, { cancelRequested: true });
    if (current.status === "queued") {
      updateRun(db, runId, { status: "cancelled", finishedAt: now() });
      appendRunEventType(db, runId, "run.cancelled");
    }
  }
  return getRun(db, runId);
}

/** Claim one queued run atomically for a single worker process. */
export function claimNextRun(db, workerId) {
  const transaction = db.transaction(() => {
    const candidate = one(db, "SELECT id FROM runs WHERE status = 'queued' AND cancel_requested = 0 ORDER BY created_at, id LIMIT 1");
    if (!candidate) return null;
    const timestamp = now();
    const result = run(db, "UPDATE runs SET status = 'claimed', worker_id = ?, updated_at = ? WHERE id = ? AND status = 'queued'", [
      workerId,
      timestamp,
      candidate.id
    ]);
    if (!result.changes) return null;
    return getRun(db, candidate.id);
  });
  return transaction();
}

export function upsertQualityCheck(db, { targetType, targetId, name, status, actualValue = null, thresholdValue = null, details = {} }) {
  run(db, `INSERT INTO quality_checks(target_type, target_id, name, status, actual_value, threshold_value, details_json, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(target_type, target_id, name) DO UPDATE SET
      status = excluded.status,
      actual_value = excluded.actual_value,
      threshold_value = excluded.threshold_value,
      details_json = excluded.details_json,
      created_at = excluded.created_at`, [
    targetType,
    targetId,
    name,
    status,
    actualValue,
    thresholdValue,
    json(details),
    now()
  ]);
}

export function publicationRow(row) {
  return {
    id: row.id,
    slug: row.slug,
    title: row.title,
    status: row.status,
    manifestPath: row.manifest_path,
    snapshotIds: parseJson(row.snapshot_ids_json, []),
    modelIds: parseJson(row.model_ids_json, []),
    artifactIds: parseJson(row.artifact_ids_json, []),
    metadata: parseJson(row.metadata_json, {}),
    createdAt: row.created_at,
    updatedAt: row.updated_at
  };
}

export function listPublications(db, status = "published") {
  return all(db, "SELECT * FROM publications WHERE status = ? ORDER BY updated_at DESC, id DESC", [status]).map(publicationRow);
}

export function getPublication(db, id) {
  const row = one(db, "SELECT * FROM publications WHERE id = ? OR slug = ? LIMIT 1", [id, id]);
  return row ? publicationRow(row) : null;
}

export function publishPublication(db, {
  id = `publication-${randomUUID().replaceAll("-", "")}`,
  slug = id,
  title,
  manifestPath = "",
  snapshotIds = [],
  modelIds = [],
  artifactIds = [],
  metadata = {}
}, root = repositoryRoot()) {
  if (!title || !String(title).trim()) throw new Error("publication title is required");
  if (!Array.isArray(snapshotIds) || snapshotIds.length === 0) throw new Error("publication needs at least one snapshot");
  for (const snapshotId of snapshotIds) {
    assertSafeId(String(snapshotId), "snapshotIds");
    if (!one(db, "SELECT id FROM snapshots WHERE id = ? AND status = 'ready'", [snapshotId])) {
      throw new Error(`snapshot ${snapshotId} is not ready for publication`);
    }
  }
  for (const modelId of modelIds) {
    assertSafeId(String(modelId), "modelIds");
    if (!one(db, "SELECT id FROM model_versions WHERE id = ? AND status = 'ready'", [modelId])) {
      throw new Error(`model ${modelId} is not ready for publication`);
    }
  }
  for (const artifactId of artifactIds) {
    assertSafeId(String(artifactId), "artifactIds");
    if (!one(db, "SELECT id FROM artifacts WHERE id = ? AND status = 'ready'", [artifactId])) {
      throw new Error(`artifact ${artifactId} is not ready for publication`);
    }
  }
  assertSafeId(String(id), "publication id");
  assertSafeId(String(slug), "publication slug");
  if (!metadata || typeof metadata !== "object" || Array.isArray(metadata)) {
    throw new Error("publication metadata must be an object");
  }
  const bundle = writePublicationBundle(db, root, {
    id,
    slug,
    title,
    snapshotIds,
    modelIds,
    artifactIds,
    metadata
  });
  const timestamp = now();
  run(db, `INSERT INTO publications(id, slug, title, status, manifest_path, snapshot_ids_json, model_ids_json, artifact_ids_json, metadata_json, created_at, updated_at)
    VALUES (?, ?, ?, 'published', ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET slug = excluded.slug, title = excluded.title, status = 'published',
      manifest_path = excluded.manifest_path, snapshot_ids_json = excluded.snapshot_ids_json,
      model_ids_json = excluded.model_ids_json, artifact_ids_json = excluded.artifact_ids_json,
      metadata_json = excluded.metadata_json, updated_at = excluded.updated_at`, [
    id,
    slug,
    String(title).trim(),
    bundle.manifestPath,
    json(snapshotIds),
    json(modelIds),
    json(artifactIds),
    json(metadata),
    timestamp,
    timestamp
  ]);
  return getPublication(db, id);
}

export function unpublishPublication(db, id) {
  run(db, "UPDATE publications SET status = 'withdrawn', updated_at = ? WHERE id = ?", [now(), id]);
  return getPublication(db, id);
}
