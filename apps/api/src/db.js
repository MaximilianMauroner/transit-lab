import { Database } from "bun:sqlite";
import { readFileSync } from "node:fs";
import { mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import {
  RUN_EVENT_SCHEMA_VERSION,
  validateRunEvent
} from "../../../packages/contracts/src/index.js";

export function repositoryRoot() {
  return resolve(import.meta.dir, "../../..");
}

export function dataRoot(root = repositoryRoot()) {
  return resolve(root, process.env.TRANSIT_LAB_DATA_ROOT || "data");
}

export function databasePath(root = repositoryRoot()) {
  return resolve(root, process.env.TRANSIT_LAB_DB || "data/transit-lab.sqlite");
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
  mkdirSync(dirname(path), { recursive: true });
  const db = new Database(path, { create: true });
  db.exec("PRAGMA foreign_keys = ON;");
  const migration = readFileSync(resolve(root, "migrations/001_initial.sql"), "utf8");
  db.exec(migration);
  db.query("INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)").run("001_initial", now());
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

export function hydrateRun(row, db) {
  const run = {
    id: row.id,
    projectId: row.project_id,
    kind: row.kind,
    status: row.status,
    spec: parseJson(row.spec_json, {}),
    fingerprint: row.fingerprint,
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
