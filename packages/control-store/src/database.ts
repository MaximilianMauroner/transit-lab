import { Database } from "bun:sqlite";
import { createHash, randomUUID } from "node:crypto";
import { mkdirSync, readFileSync, readdirSync, realpathSync, statSync } from "node:fs";
import { hostname } from "node:os";
import { dirname, relative, resolve, sep } from "node:path";
import {
  RUN_EVENT_SCHEMA_VERSION,
  assertSafeId,
  normalizeRunSpec,
  stableStringify,
  trainingCheckpointFingerprint,
  validateRunEvent,
  validateTrainingCheckpointManifest
} from "../../contracts/src/index.ts";
import { materializeResolvedRunConfig } from "./experiments.ts";
import { writePublicationBundle } from "./publication-bundle.ts";
import { isWithinAllowedWindow, nextAllowedWindowStart } from "./schedule.ts";
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
  desiredState: string;
  observedState: string;
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
  latestCheckpointId: string | null;
  currentAttemptId: string | null;
  parentRunId: string | null;
  resumeCheckpointId: string | null;
  checkpointRoot: string;
  controlFilePath: string;
  phase: string;
  globalStep: number;
  resumeNotBefore: string | null;
  totalComputeSeconds: number;
  pausedSeconds: number;
  pausedSince: string | null;
  schedule: Record<string, unknown>;
  lastHeartbeatAt: string | null;
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
    status: row.observed_state || row.status,
    desiredState: row.desired_state || (row.cancel_requested ? "cancelled" : "running"),
    observedState: row.observed_state || row.status,
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
    latestCheckpointId: row.latest_checkpoint_id || null,
    currentAttemptId: row.current_attempt_id || null,
    parentRunId: row.parent_run_id || null,
    resumeCheckpointId: row.resume_checkpoint_id || null,
    checkpointRoot: row.checkpoint_root || "",
    controlFilePath: row.control_file_path || "",
    phase: row.phase || "",
    globalStep: Number(row.global_step || 0),
    resumeNotBefore: row.resume_not_before || null,
    totalComputeSeconds: Number(row.total_compute_seconds || 0),
    pausedSeconds: Number(row.paused_seconds || 0),
    pausedSince: row.paused_since || null,
    schedule: parseJson(row.schedule_json, {}),
    lastHeartbeatAt: row.last_heartbeat_at || null,
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
    desiredState: "desired_state",
    observedState: "observed_state",
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
    latestCheckpointId: "latest_checkpoint_id",
    currentAttemptId: "current_attempt_id",
    parentRunId: "parent_run_id",
    resumeCheckpointId: "resume_checkpoint_id",
    checkpointRoot: "checkpoint_root",
    controlFilePath: "control_file_path",
    phase: "phase",
    globalStep: "global_step",
    resumeNotBefore: "resume_not_before",
    totalComputeSeconds: "total_compute_seconds",
    pausedSeconds: "paused_seconds",
    pausedSince: "paused_since",
    schedule: "schedule_json",
    lastHeartbeatAt: "last_heartbeat_at",
    snapshotId: "snapshot_id",
    datasetId: "dataset_id",
    modelId: "model_id"
  };
  const normalizedFields = { ...fields };
  if (normalizedFields.status !== undefined && normalizedFields.observedState === undefined) {
    normalizedFields.observedState = normalizedFields.status;
  }
  if (normalizedFields.observedState !== undefined && normalizedFields.status === undefined) {
    normalizedFields.status = normalizedFields.observedState;
  }
  const entries = Object.entries(normalizedFields).filter(([key, value]) => allowed[key] && value !== undefined);
  if (!entries.length) return;
  const assignments = entries.map(([key]) => `${allowed[key]} = ?`).join(", ");
  const values = entries.map(([key, value]) => {
    if (key === "schedule" && typeof value !== "string") return json(value);
    return value === true ? 1 : value === false ? 0 : value;
  });
  values.push(now(), runId);
  run(db, `UPDATE runs SET ${assignments}, updated_at = ? WHERE id = ?`, values);
}

export type RunCreationOptions = {
  parentRunId?: string | null;
  resumeCheckpointId?: string | null;
};

/**
 * A checkpoint may be resumed by its producing run or by a descendant fork,
 * but never by an unrelated run. Keep this check in the control store so all
 * callers (API, worker, and tests) share the same lineage boundary.
 */
export function findTrainingCheckpointForRun(db, runId, checkpointId) {
  if (!runId || !checkpointId) return null;
  return one(db, `WITH RECURSIVE lineage(id) AS (
      SELECT id FROM runs WHERE id = ?
      UNION
      SELECT parent_run_id FROM runs
      JOIN lineage ON runs.id = lineage.id
      WHERE parent_run_id IS NOT NULL
    )
    SELECT tc.*
    FROM training_checkpoints tc
    JOIN lineage ON lineage.id = tc.run_id
    WHERE tc.id = ? AND tc.status = 'committed'
    LIMIT 1`, [runId, checkpointId]);
}

export function createRun(
  db,
  spec,
  projectId = "project-local",
  root = repositoryRoot(),
  options: RunCreationOptions = {}
) {
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
  const runtime = normalized.spec.kind === "train" ? normalized.spec.runtime : {};
  const schedule = (runtime as any).allowedWindows !== undefined
    ? { allowedWindows: (runtime as any).allowedWindows || [] }
    : (runtime as any).schedule || (normalized.spec as any).schedule || {};
  const parentRunId = options.parentRunId || null;
  const resumeCheckpointId = options.resumeCheckpointId || null;
  if (parentRunId && !one(db, "SELECT id FROM runs WHERE id = ?", [parentRunId])) {
    throw new Error(`parent run ${parentRunId} is not indexed`);
  }
  if (resumeCheckpointId) {
    if (!parentRunId || !findTrainingCheckpointForRun(db, parentRunId, resumeCheckpointId)) {
      throw new Error(`resume checkpoint ${resumeCheckpointId} is not committed in the parent run lineage`);
    }
  }
  const checkpointRoot = `runs/${id}/checkpoints`;
  const controlFilePath = `runs/${id}/control.json`;
  run(db, `INSERT INTO runs(
    id, project_id, kind, status, desired_state, observed_state, spec_json, fingerprint, snapshot_id, dataset_id, model_id,
    config_fingerprint, resolved_config_path, git_commit, parent_run_id, resume_checkpoint_id,
    checkpoint_root, control_file_path, schedule_json, created_at, updated_at
  ) VALUES (?, ?, ?, 'queued', 'running', 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`, [
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
    parentRunId,
    resumeCheckpointId,
    checkpointRoot,
    controlFilePath,
    json(schedule),
    timestamp,
    timestamp
  ]);
  appendRunEventType(db, id, "run.queued", {
    status: "queued",
    specFingerprint: normalized.fingerprint,
    ...(parentRunId ? { parentRunId } : {}),
    ...(resumeCheckpointId ? { resumeCheckpointId } : {})
  });
  return getRun(db, id);
}

export function listRuns(db, limit = 100) {
  return all(db, "SELECT * FROM runs ORDER BY created_at DESC, id DESC LIMIT ?", [
    Math.max(1, Math.min(500, Number(limit) || 100))
  ]).map((row) => hydrateRun(row));
}

/**
 * Create a new logical run from a committed checkpoint. A fork gets a new
 * immutable experiment identity, while the worker receives an explicit source
 * checkpoint and the lineage remains visible in the run ledger.
 */
export function forkRun(db, runId, {
  checkpointId = null,
  spec = null
}: { checkpointId?: string | null; spec?: Record<string, unknown> | null } = {}, projectId = "project-local", root = repositoryRoot()) {
  const source = getRun(db, runId);
  if (!source) return null;
  if (source.kind !== "train") throw new Error("only training runs can be forked");
  const checkpoint = checkpointId
    ? one(db, "SELECT id, run_id, status FROM training_checkpoints WHERE id = ?", [checkpointId])
    : one(db, `SELECT id, run_id, status FROM training_checkpoints
      WHERE run_id = ? AND status = 'committed' ORDER BY global_step DESC, id DESC LIMIT 1`, [runId]);
  if (!checkpoint || checkpoint.run_id !== runId || checkpoint.status !== "committed") {
    throw new Error("fork requires a committed checkpoint from the source run");
  }
  const sourceSpec = source.spec as Record<string, unknown>;
  const requested = spec && typeof spec === "object" && !Array.isArray(spec) ? spec : {};
  const sourceRuntime = sourceSpec.runtime && typeof sourceSpec.runtime === "object" && !Array.isArray(sourceSpec.runtime)
    ? sourceSpec.runtime as Record<string, unknown>
    : {};
  const requestedRuntime = requested.runtime && typeof requested.runtime === "object" && !Array.isArray(requested.runtime)
    ? requested.runtime as Record<string, unknown>
    : null;
  const merged = {
    ...sourceSpec,
    ...requested,
    ...(requestedRuntime ? {
      runtime: { ...sourceRuntime, ...requestedRuntime }
    } : {})
  };
  return createRun(db, merged, projectId, root, {
    parentRunId: runId,
    resumeCheckpointId: checkpoint.id
  });
}

export function requestRunCancellation(db, runId) {
  const transaction = db.transaction(() => {
    const current = one(db, "SELECT * FROM runs WHERE id = ?", [runId]);
    if (!current) return false;
    const observed = current.observed_state || current.status;
    if (["succeeded", "failed", "cancelled"].includes(observed)) return true;
    const timestamp = now();
    const fields: Record<string, unknown> = { cancelRequested: true, desiredState: "cancelled" };
    if (["queued", "paused", "interrupted"].includes(observed)) {
      fields.status = "cancelled";
      fields.observedState = "cancelled";
      fields.finishedAt = timestamp;
      fields.pausedSince = null;
      const pausedSince = current.paused_since ? Date.parse(current.paused_since) : NaN;
      if (Number.isFinite(pausedSince)) {
        fields.pausedSeconds = Number(current.paused_seconds || 0) + Math.max(0, (Date.now() - pausedSince) / 1_000);
      }
      updateRun(db, runId, fields);
      appendRunEventType(db, runId, "run.cancelled");
    } else {
      updateRun(db, runId, fields);
    }
    return true;
  });
  if (!transaction.immediate()) return null;
  return getRun(db, runId);
}

/** Request a cooperative checkpoint-and-exit. Repeating the request is safe. */
export function requestRunPause(db, runId, reason = "user-request") {
  const transaction = db.transaction(() => {
    const current = one(db, "SELECT * FROM runs WHERE id = ?", [runId]);
    if (!current) return false;
    const observed = current.observed_state || current.status;
    if (["succeeded", "failed", "cancelled"].includes(observed)) return true;
    const changed = current.desired_state !== "paused";
    const fields: Record<string, unknown> = { desiredState: "paused" };
    if (observed === "queued" || observed === "interrupted" || observed === "paused") {
      fields.status = "paused";
      fields.observedState = "paused";
      if (!current.paused_since) fields.pausedSince = now();
    }
    updateRun(db, runId, fields);
    if (changed) appendRunEventType(db, runId, "pause.requested", { reason });
    return true;
  });
  if (!transaction.immediate()) return null;
  return getRun(db, runId);
}

/** Resume a paused or interrupted logical run. The worker will create a new attempt. */
export function requestRunResume(db, runId, resumeNotBefore = null) {
  const transaction = db.transaction(() => {
    const current = one(db, "SELECT * FROM runs WHERE id = ?", [runId]);
    if (!current) return false;
    const observed = current.observed_state || current.status;
    if (["succeeded", "failed", "cancelled"].includes(observed) || current.desired_state === "cancelled") return true;
    const wasPaused = current.desired_state === "paused" || ["paused", "interrupted"].includes(observed);
    const fields: Record<string, unknown> = {
      desiredState: "running",
      cancelRequested: false,
      resumeNotBefore,
      status: ["paused", "interrupted", "queued"].includes(observed) ? "queued" : current.status,
      observedState: ["paused", "interrupted"].includes(observed) ? "queued" : observed,
      pausedSince: null
    };
    if (current.paused_since) {
      const pausedSince = Date.parse(current.paused_since);
      if (Number.isFinite(pausedSince)) {
        fields.pausedSeconds = Number(current.paused_seconds || 0) + Math.max(0, (Date.now() - pausedSince) / 1_000);
      }
    }
    updateRun(db, runId, fields);
    if (wasPaused) appendRunEventType(db, runId, "run.resumed");
    return true;
  });
  if (!transaction.immediate()) return null;
  return getRun(db, runId);
}

/** Claim one queued run atomically for a single worker process. */
export function claimNextRun(db, workerId) {
  const transaction = db.transaction(() => {
    const candidates = all(db, `SELECT id, schedule_json, resume_not_before FROM runs
      WHERE (status = 'queued' OR observed_state = 'queued')
        AND desired_state = 'running' AND cancel_requested = 0
        AND (resume_not_before IS NULL OR resume_not_before <= ?)
      ORDER BY created_at, id LIMIT 100`, [now()]);
    const current = new Date();
    const candidate = candidates.find((row) => isWithinAllowedWindow(parseJson(row.schedule_json, {}), current));
    if (!candidate) {
      // Keep a scheduled queue quiet and make the next eligible time visible
      // to Studio without claiming the run.
      for (const row of candidates) {
        const next = nextAllowedWindowStart(parseJson(row.schedule_json, {}), current);
        if (next && (!row.resume_not_before || Date.parse(row.resume_not_before) < next.getTime())) {
          updateRun(db, row.id, { resumeNotBefore: next.toISOString() });
        }
      }
      return null;
    }
    const timestamp = now();
    const result = run(db, `UPDATE runs SET status = 'claimed', observed_state = 'claimed', worker_id = ?, updated_at = ?
      , last_heartbeat_at = ? WHERE id = ? AND (status = 'queued' OR observed_state = 'queued')`, [
      workerId,
      timestamp,
      timestamp,
      candidate.id
    ]);
    if (!result.changes) return null;
    return getRun(db, candidate.id);
  });
  return transaction.immediate();
}

/** Start a new continuous process attempt for a logical run. */
export function startRunAttempt(db, runId, workerId, {
  resumeCheckpointId = null,
  device = {},
  attemptId = `attempt-${randomUUID().replaceAll("-", "")}`
} = {}) {
  assertSafeId(attemptId, "attemptId");
  const transaction = db.transaction(() => {
    const current = one(db, "SELECT * FROM runs WHERE id = ?", [runId]);
    if (!current) return null;
    if (current.current_attempt_id) {
      const existing = one(db, "SELECT * FROM run_attempts WHERE id = ?", [current.current_attempt_id]);
      if (existing && ["starting", "running", "checkpointing"].includes(existing.status)) {
        if (existing.worker_id !== workerId) throw new Error(`run ${runId} already has an active attempt on worker ${existing.worker_id}`);
        return existing;
      }
    }
    if (resumeCheckpointId && !findTrainingCheckpointForRun(db, runId, resumeCheckpointId)) {
      throw new Error(`resume checkpoint ${resumeCheckpointId} is not committed in the run lineage`);
    }
    const ordinal = Number(one(db, "SELECT COALESCE(MAX(ordinal), 0) + 1 AS ordinal FROM run_attempts WHERE run_id = ?", [runId])?.ordinal || 1);
    const timestamp = now();
    run(db, `INSERT INTO run_attempts(
      id, run_id, ordinal, worker_id, resume_checkpoint_id, status, hostname, device_json,
      started_at, last_heartbeat_at
    ) VALUES (?, ?, ?, ?, ?, 'starting', ?, ?, ?, ?)`, [
      attemptId,
      runId,
      ordinal,
      workerId,
      resumeCheckpointId,
      hostname(),
      json(device),
      timestamp,
      timestamp
    ]);
    updateRun(db, runId, {
      status: "starting",
      currentAttemptId: attemptId,
      workerId,
      startedAt: current.started_at || timestamp,
      lastHeartbeatAt: timestamp
    });
    appendRunEventType(db, runId, "attempt.started", {
      attemptId,
      ordinal,
      resumeCheckpointId,
      device
    });
    return one(db, "SELECT * FROM run_attempts WHERE id = ?", [attemptId]);
  });
  return transaction.immediate();
}

export function updateRunAttempt(db, attemptId, fields) {
  const allowed = {
    status: "status",
    exitReason: "exit_reason",
    finishedAt: "finished_at",
    lastHeartbeatAt: "last_heartbeat_at",
    resumeCheckpointId: "resume_checkpoint_id",
    device: "device_json",
    computeSeconds: "compute_seconds"
  };
  const entries = Object.entries(fields).filter(([key, value]) => allowed[key] && value !== undefined);
  if (!entries.length) return;
  const values = entries.map(([key, value]) => key === "device" && typeof value !== "string" ? json(value) : value);
  values.push(attemptId);
  run(db, `UPDATE run_attempts SET ${entries.map(([key]) => `${allowed[key]} = ?`).join(", ")} WHERE id = ?`, values);
}

export function heartbeatRunAttempt(db, attemptId, phase = "", globalStep = null) {
  const transaction = db.transaction(() => {
    const attempt = one(db, "SELECT run_id, status FROM run_attempts WHERE id = ?", [attemptId]);
    if (!attempt || !["starting", "running", "checkpointing"].includes(attempt.status)) return false;
    const timestamp = now();
    const attemptResult = run(db, `UPDATE run_attempts SET last_heartbeat_at = ?
      WHERE id = ? AND status IN ('starting', 'running', 'checkpointing')`, [timestamp, attemptId]);
    if (!attemptResult.changes) return false;
    const fields = {
      lastHeartbeatAt: timestamp,
      phase: phase || undefined,
      globalStep: globalStep === null || globalStep === undefined ? undefined : Number(globalStep)
    };
    const allowed = { lastHeartbeatAt: "last_heartbeat_at", phase: "phase", globalStep: "global_step" };
    const entries = Object.entries(fields).filter(([, value]) => value !== undefined);
    const values = entries.map(([, value]) => value);
    values.push(attempt.run_id, attemptId);
    run(db, `UPDATE runs SET ${entries.map(([key]) => `${allowed[key]} = ?`).join(", ")}, updated_at = ?
      WHERE id = ? AND current_attempt_id = ? AND observed_state IN ('claimed', 'starting', 'running', 'checkpointing')`, [
      ...values.slice(0, -2),
      timestamp,
      attempt.run_id,
      attemptId
    ]);
    return true;
  });
  return transaction.immediate();
}

export function finishRunAttempt(db, attemptId, status, exitReason = null) {
  const finalStatuses = ["paused", "time-sliced", "cancelled", "succeeded", "failed", "interrupted"];
  if (!finalStatuses.includes(status)) throw new Error(`invalid terminal attempt status ${status}`);
  const transaction = db.transaction(() => {
    const attempt = one(db, "SELECT * FROM run_attempts WHERE id = ?", [attemptId]);
    if (!attempt) return null;
    if (finalStatuses.includes(attempt.status)) {
      if (attempt.status !== status || (attempt.exit_reason || null) !== (exitReason || null)) {
        throw new Error(`attempt ${attemptId} was already finalized as ${attempt.status}`);
      }
      return attempt;
    }
    const finishedAt = now();
    const startedAt = Date.parse(attempt.started_at);
    const finishedMillis = Date.parse(finishedAt);
    const computeSeconds = Number.isFinite(startedAt) && Number.isFinite(finishedMillis)
      ? Math.max(0, (finishedMillis - startedAt) / 1_000)
      : 0;
    run(db, `UPDATE run_attempts SET status = ?, exit_reason = ?, finished_at = ?,
      last_heartbeat_at = ?, compute_seconds = ? WHERE id = ?`, [
      status,
      exitReason,
      finishedAt,
      finishedAt,
      computeSeconds,
      attemptId
    ]);
    run(db, `UPDATE runs SET total_compute_seconds = COALESCE(total_compute_seconds, 0) + ?,
      updated_at = ? WHERE id = ?`, [computeSeconds, finishedAt, attempt.run_id]);
    appendRunEventType(db, attempt.run_id, "attempt.ended", {
      attemptId,
      status,
      reason: exitReason || status,
      computeSeconds
    });
    return one(db, "SELECT * FROM run_attempts WHERE id = ?", [attemptId]);
  });
  return transaction.immediate();
}

export function listRunAttempts(db, runId) {
  return all(db, "SELECT * FROM run_attempts WHERE run_id = ? ORDER BY ordinal", [runId]).map((row) => ({
    id: row.id,
    runId: row.run_id,
    ordinal: Number(row.ordinal),
    workerId: row.worker_id,
    resumeCheckpointId: row.resume_checkpoint_id,
    status: row.status,
    exitReason: row.exit_reason,
    hostname: row.hostname,
    device: parseJson(row.device_json, {}),
    startedAt: row.started_at,
    finishedAt: row.finished_at,
    lastHeartbeatAt: row.last_heartbeat_at
    ,computeSeconds: Number(row.compute_seconds || 0)
  }));
}

function checkpointRow(row) {
  return {
    id: row.id,
    runId: row.run_id,
    attemptId: row.attempt_id,
    phase: row.phase,
    globalStep: Number(row.global_step),
    localPath: row.local_path,
    sha256: row.sha256,
    configFingerprint: row.config_fingerprint,
    datasetFingerprint: row.dataset_fingerprint,
    gitCommit: row.git_commit,
    status: row.status,
    metrics: parseJson(row.metrics_json, {}),
    createdAt: row.created_at
  };
}

function checkpointPathInside(directory, candidate) {
  if (!(candidate === directory || candidate.startsWith(`${directory}${sep}`))) return false;
  try {
    const canonicalDirectory = realpathSync(directory);
    const canonicalCandidate = realpathSync(candidate);
    return canonicalCandidate === canonicalDirectory || canonicalCandidate.startsWith(`${canonicalDirectory}${sep}`);
  } catch {
    return false;
  }
}

export function listTrainingCheckpoints(db, runId) {
  return all(db, "SELECT * FROM training_checkpoints WHERE run_id = ? ORDER BY global_step, id", [runId]).map(checkpointRow);
}

export function benchmarkRow(row) {
  return {
    id: row.id,
    fingerprint: row.fingerprint,
    artifactId: row.artifact_id,
    runId: row.run_id,
    benchmark: row.benchmark,
    workload: row.workload,
    snapshotId: row.snapshot_id,
    graphId: row.graph_id,
    threadCount: row.thread_count === null || row.thread_count === undefined ? null : Number(row.thread_count),
    warmupUnits: Number(row.warmup_units || 0),
    measuredUnits: Number(row.measured_units || 0),
    estimatedWorkUnits: row.estimated_work_units === null || row.estimated_work_units === undefined
      ? null
      : Number(row.estimated_work_units),
    medianMilliseconds: row.median_milliseconds === null || row.median_milliseconds === undefined
      ? null
      : Number(row.median_milliseconds),
    p95Milliseconds: row.p95_milliseconds === null || row.p95_milliseconds === undefined
      ? null
      : Number(row.p95_milliseconds),
    throughput: Number(row.throughput || 0),
    throughputUnit: row.throughput_unit,
    peakResidentMemoryBytes: row.peak_resident_memory_bytes === null || row.peak_resident_memory_bytes === undefined
      ? null
      : Number(row.peak_resident_memory_bytes),
    graphCounts: parseJson(row.graph_counts_json, {}),
    runtime: parseJson(row.runtime_json, {}),
    threadConfiguration: parseJson(row.thread_configuration_json, {}),
    report: parseJson(row.report_json, {}),
    createdAt: row.created_at
  };
}

export function listBenchmarks(db, {
  workload = null,
  snapshotId = null,
  graphId = null,
  limit = 500
} = {}) {
  const clauses = [];
  const params = [];
  if (workload) {
    clauses.push("workload = ?");
    params.push(workload);
  }
  if (snapshotId) {
    clauses.push("snapshot_id = ?");
    params.push(snapshotId);
  }
  if (graphId) {
    clauses.push("graph_id = ?");
    params.push(graphId);
  }
  params.push(Math.max(1, Math.min(5_000, Number(limit) || 500)));
  const where = clauses.length ? ` WHERE ${clauses.join(" AND ")}` : "";
  return all(db, `SELECT * FROM benchmarks${where} ORDER BY created_at DESC, id DESC LIMIT ?`, params)
    .map(benchmarkRow);
}

/** Index a committed Rust checkpoint. Repeated event delivery is harmless. */
export function registerTrainingCheckpoint(db, {
  runId,
  attemptId = null,
  phase,
  globalStep,
  localPath,
  sha256,
  configFingerprint,
  datasetFingerprint,
  gitCommit = "working-tree",
  status = "committed",
  metrics = {}
}) {
  assertSafeId(runId, "runId");
  if (!Number.isSafeInteger(globalStep) || globalStep < 0) {
    throw new Error("checkpoint globalStep must be a non-negative safe integer");
  }
  if (attemptId !== null && attemptId !== undefined) assertSafeId(attemptId, "attemptId");
  if (typeof phase !== "string" || phase.trim() === "") throw new Error("checkpoint phase must be a non-empty string");
  if (typeof localPath !== "string" || localPath.trim() === "" ||
      /^(?:[A-Za-z]:[\\/]|[\\/])/.test(localPath) ||
      localPath.split(/[\\/]+/).some((segment) => !segment || segment === "." || segment === "..")) {
    throw new Error("checkpoint localPath must be a relative path");
  }
  if (typeof sha256 !== "string" || !/^[a-f0-9]{64}$/i.test(sha256)) {
    throw new Error("checkpoint sha256 must be a SHA-256 hex digest");
  }
  if (typeof configFingerprint !== "string" || configFingerprint.trim() === "") {
    throw new Error("checkpoint configFingerprint must be a non-empty string");
  }
  if (typeof datasetFingerprint !== "string" || datasetFingerprint.trim() === "") {
    throw new Error("checkpoint datasetFingerprint must be a non-empty string");
  }
  if (typeof gitCommit !== "string" || gitCommit.trim() === "") {
    throw new Error("checkpoint gitCommit must be a non-empty string");
  }
  if (status !== "committed") throw new Error("only committed checkpoints can be registered");
  if (!metrics || typeof metrics !== "object" || Array.isArray(metrics)) {
    throw new Error("checkpoint metrics must be an object");
  }
  if (attemptId !== null && attemptId !== undefined) {
    const attempt = one(db, "SELECT run_id FROM run_attempts WHERE id = ?", [attemptId]);
    if (!attempt || attempt.run_id !== runId) throw new Error(`checkpoint attempt ${attemptId} does not belong to run ${runId}`);
  }

  const id = `checkpoint-${runId}-${globalStep}`;
  const normalizedSha256 = sha256.toLowerCase();
  const metricsJson = stableStringify(metrics);
  const transaction = db.transaction(() => {
    const existing = one(db, "SELECT * FROM training_checkpoints WHERE run_id = ? AND global_step = ?", [runId, globalStep]);
    if (existing) {
      const conflicts = [];
      if (existing.id !== id) conflicts.push("id");
      if ((existing.attempt_id || null) !== (attemptId || null)) conflicts.push("attemptId");
      if (existing.phase !== phase) conflicts.push("phase");
      if (existing.local_path !== localPath) conflicts.push("localPath");
      if (String(existing.sha256).toLowerCase() !== normalizedSha256) conflicts.push("sha256");
      if (existing.config_fingerprint !== configFingerprint) conflicts.push("configFingerprint");
      if (existing.dataset_fingerprint !== datasetFingerprint) conflicts.push("datasetFingerprint");
      if (existing.git_commit !== gitCommit) conflicts.push("gitCommit");
      if (existing.status !== status) conflicts.push("status");
      if (stableStringify(parseJson(existing.metrics_json, {})) !== metricsJson) conflicts.push("metrics");
      if (conflicts.length) {
        throw new Error(`checkpoint ${runId} step ${globalStep} conflicts with committed metadata: ${conflicts.join(", ")}`);
      }
    } else {
      run(db, `INSERT INTO training_checkpoints(
        id, run_id, attempt_id, phase, global_step, local_path, sha256,
        config_fingerprint, dataset_fingerprint, git_commit, status, metrics_json, created_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`, [
        id,
        runId,
        attemptId || null,
        phase,
        globalStep,
        localPath,
        normalizedSha256,
        configFingerprint,
        datasetFingerprint,
        gitCommit,
        status,
        metricsJson,
        now()
      ]);
    }

    const latest = one(db, "SELECT id, global_step FROM training_checkpoints WHERE run_id = ? ORDER BY global_step DESC, id DESC LIMIT 1", [runId]);
    if (!latest || Number(latest.global_step) <= globalStep) {
      updateRun(db, runId, {
        latestCheckpointId: id,
        phase,
        globalStep
      });
    }
    return checkpointRow(one(db, "SELECT * FROM training_checkpoints WHERE run_id = ? AND global_step = ?", [runId, globalStep]));
  });
  return transaction.immediate();
}

/** Discover committed checkpoint manifests after a crash between filesystem and DB commits. */
export function reconcileTrainingCheckpoints(db, root, runId) {
  const record = one(db, "SELECT checkpoint_root FROM runs WHERE id = ?", [runId]);
  if (!record?.checkpoint_root) return [];
  const base = resolve(dataRoot(root), record.checkpoint_root);
  const dataBase = resolve(dataRoot(root));
  if (!(base === dataBase || base.startsWith(`${dataBase}${sep}`))) {
    throw new Error("checkpoint root escaped the data root");
  }
  let directories;
  try {
    directories = readdirSync(base, { withFileTypes: true });
  } catch (error) {
    if (error?.code === "ENOENT") return [];
    throw error;
  }
  const restored = [];
  for (const entry of directories) {
    if (!entry.isDirectory() || !/^step-\d+$/.test(entry.name)) continue;
    const directory = resolve(base, entry.name);
    try {
      const manifestPath = resolve(directory, "manifest.json");
      if (!checkpointPathInside(base, directory) || !checkpointPathInside(directory, manifestPath)) {
        throw new Error("checkpoint path escaped its root");
      }
      const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
      validateTrainingCheckpointManifest(manifest);
      if (manifest.runId !== runId) continue;
      for (const file of manifest.files) {
        const payloadPath = resolve(directory, file.path);
        if (!checkpointPathInside(directory, payloadPath)) {
          throw new Error("checkpoint payload path escaped its directory");
        }
        const payload = readFileSync(payloadPath);
        const digest = createHash("sha256").update(payload).digest("hex");
        if (payload.byteLength !== file.sizeBytes || digest !== file.sha256) throw new Error(`checkpoint payload mismatch: ${file.path}`);
      }
      if (trainingCheckpointFingerprint(manifest.files) !== String(manifest.checkpointFingerprint).toLowerCase()) {
        throw new Error("checkpoint fingerprint does not match its payload descriptors");
      }
      const relativePath = relative(dataBase, directory).split(sep).join("/");
      restored.push(registerTrainingCheckpoint(db, {
        runId,
        attemptId: manifest.attemptId || null,
        phase: manifest.phase,
        globalStep: manifest.globalStep,
        localPath: relativePath,
        sha256: manifest.checkpointFingerprint,
        configFingerprint: manifest.configFingerprint,
        datasetFingerprint: manifest.datasetFingerprint,
        gitCommit: manifest.codeCommit,
        status: manifest.status
      }));
    } catch {
      // An absent, temporary, or corrupt directory is not a resumable
      // checkpoint. Leave it for diagnostics and keep scanning older steps.
    }
  }
  return restored;
}

/** Mark abandoned process attempts and make their logical runs resumable. */
export function recoverInterruptedRuns(db, workerId = null, staleAfterSeconds = 30) {
  const threshold = Date.now() - Math.max(1, staleAfterSeconds) * 1_000;
  const staleAt = new Date(threshold).toISOString();
  const transaction = db.transaction(() => {
    const rows = all(db, `SELECT r.id, r.current_attempt_id, r.current_step, r.desired_state,
        r.observed_state, r.paused_since, r.latest_checkpoint_id,
        a.worker_id AS attempt_worker_id, a.status AS attempt_status,
        a.last_heartbeat_at AS attempt_heartbeat,
        COALESCE(a.last_heartbeat_at, r.last_heartbeat_at) AS effective_heartbeat
      FROM runs r
      LEFT JOIN run_attempts a ON a.id = r.current_attempt_id
      WHERE r.observed_state IN ('claimed', 'starting', 'running', 'checkpointing')
        AND (r.last_heartbeat_at IS NULL OR r.last_heartbeat_at < ?
          OR (a.id IS NOT NULL AND (a.last_heartbeat_at IS NULL OR a.last_heartbeat_at < ?)))`, [staleAt, staleAt]);
    const ownWorker = workerId ? one(db, "SELECT current_run_id, last_heartbeat_at FROM workers WHERE id = ?", [workerId]) : null;
    const recovered = [];
    for (const row of rows) {
      const ownHeartbeat = ownWorker?.current_run_id === row.id ? Date.parse(ownWorker.last_heartbeat_at) : NaN;
      if (workerId && row.attempt_worker_id === workerId && Number.isFinite(ownHeartbeat) && ownHeartbeat >= threshold) continue;

      const timestamp = now();
      if (row.current_attempt_id && ["starting", "running", "checkpointing"].includes(row.attempt_status)) {
        finishRunAttempt(db, row.current_attempt_id, "interrupted", "worker-lease-expired");
      }
      const nextState = row.desired_state === "paused" ? "paused" : row.desired_state === "cancelled" ? "cancelled" : "queued";
      const fields = {
        status: nextState,
        observedState: nextState,
        currentAttemptId: null,
        workerId: null,
        lastHeartbeatAt: timestamp,
        ...(nextState === "queued" ? {
          desiredState: "running",
          finishedAt: null,
          errorCode: "interrupted",
          errorMessage: "worker lease expired; run requeued from the latest committed checkpoint"
        } : {}),
        ...(nextState === "paused" ? {
          pausedSince: row.paused_since || timestamp,
          finishedAt: null
        } : {}),
        ...(nextState === "cancelled" ? {
          finishedAt: timestamp,
          pausedSince: null
        } : {})
      };
      updateRun(db, row.id, fields);
      if (row.current_step) {
        run(db, "UPDATE run_steps SET status = ?, finished_at = ? WHERE run_id = ? AND step = ?", [
          nextState,
          timestamp,
          row.id,
          row.current_step
        ]);
      }
      appendRunEventType(db, row.id, "run.recovered", {
        attemptId: row.current_attempt_id || undefined,
        previousState: row.observed_state,
        reason: "worker-lease-expired",
        ...(row.latest_checkpoint_id ? { checkpointId: row.latest_checkpoint_id } : {})
      });
      if (nextState === "queued") {
        appendRunEventType(db, row.id, "run.queued", {
          reason: "worker-recovery",
          ...(row.current_attempt_id ? { attemptId: row.current_attempt_id } : {}),
          ...(row.latest_checkpoint_id ? { checkpointId: row.latest_checkpoint_id } : {})
        });
      }
      recovered.push(row.id);
    }
    run(db, `UPDATE workers SET status = 'offline', current_run_id = NULL
      WHERE last_heartbeat_at < ? AND status <> 'offline'${workerId ? " AND id <> ?" : ""}`,
    workerId ? [staleAt, workerId] : [staleAt]);
    return recovered;
  });
  return transaction.immediate();
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
