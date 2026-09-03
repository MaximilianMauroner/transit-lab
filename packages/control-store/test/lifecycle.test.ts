import { expect, test } from "bun:test";
import { Database } from "bun:sqlite";
import {
  claimNextRun,
  createRun,
  finishRunAttempt,
  forkRun,
  getRun,
  listRunAttempts,
  listTrainingCheckpoints,
  registerTrainingCheckpoint,
  requestRunCancellation,
  requestRunPause,
  requestRunResume,
  startRunAttempt,
  updateRunAttempt,
  heartbeatRunAttempt
} from "../src/database.ts";
import { pushDatabaseSchema } from "../src/schema.ts";

function trainingSpec(runtime = {}) {
  return {
    kind: "train",
    datasetId: "dataset-test",
    modelConfig: {},
    seed: 7,
    runtime
  };
}

function database() {
  const db = new Database(":memory:");
  pushDatabaseSchema(db);
  return db;
}

test("pause, resume, and cancel requests are idempotent", () => {
  const db = database();
  const created = createRun(db, trainingSpec());
  const claimed = claimNextRun(db, "worker-test");
  expect(claimed?.id).toBe(created.id);
  const attempt = startRunAttempt(db, created.id, "worker-test", { attemptId: "attempt-test" });
  expect(attempt?.id).toBe("attempt-test");
  updateRunAttempt(db, attempt.id, { status: "running" });

  const paused = requestRunPause(db, created.id);
  requestRunPause(db, created.id);
  expect(paused?.desiredState).toBe("paused");
  expect(getRun(db, created.id)?.observedState).toBe("starting");
  expect((db.query("SELECT COUNT(*) AS count FROM run_events WHERE run_id = ? AND event_json LIKE '%pause.requested%'").get(created.id) as { count: number }).count).toBe(1);

  requestRunResume(db, created.id);
  requestRunResume(db, created.id);
  expect(getRun(db, created.id)?.desiredState).toBe("running");
  expect((db.query("SELECT COUNT(*) AS count FROM run_events WHERE run_id = ? AND event_json LIKE '%run.resumed%'").get(created.id) as { count: number }).count).toBe(1);

  requestRunCancellation(db, created.id);
  requestRunCancellation(db, created.id);
  expect(getRun(db, created.id)?.desiredState).toBe("cancelled");
  expect(getRun(db, created.id)?.cancelRequested).toBe(true);
  db.close();
});

test("scheduled runs are not claimed before their window", () => {
  const db = database();
  const now = new Date();
  const start = new Date(now.getTime() + 5 * 60_000);
  const end = new Date(start.getTime() + 60_000);
  const clock = (value: Date) => value.toISOString().slice(11, 16);
  const scheduled = createRun(db, trainingSpec({
    allowedWindows: [{
      days: ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"],
      start: clock(start),
      end: clock(end),
      timezone: "UTC"
    }]
  }));

  expect(claimNextRun(db, "worker-test")).toBeNull();
  expect(getRun(db, scheduled.id)?.resumeNotBefore).not.toBeNull();
  db.close();
});

test("forks preserve checkpoint lineage and use the selected checkpoint", () => {
  const db = database();
  const source = createRun(db, trainingSpec());
  const checkpoint = registerTrainingCheckpoint(db, {
    runId: source.id,
    phase: "pretraining",
    globalStep: 12,
    localPath: `runs/${source.id}/checkpoints/step-000000012`,
    sha256: "a".repeat(64),
    configFingerprint: source.configFingerprint,
    datasetFingerprint: "dataset-fingerprint",
    status: "committed"
  });
  const fork = forkRun(db, source.id, {
    checkpointId: checkpoint.id,
    spec: { runtime: { checkpointEverySteps: 3 } }
  });

  expect(fork).toMatchObject({
    parentRunId: source.id,
    resumeCheckpointId: checkpoint.id,
    spec: { runtime: { checkpointEverySteps: 3 } }
  });
  expect(listTrainingCheckpoints(db, source.id)).toHaveLength(1);
  expect(listRunAttempts(db, source.id)).toHaveLength(0);
  db.close();
});

test("attempt starts are not duplicated while an attempt is active", () => {
  const db = database();
  const run = createRun(db, trainingSpec());
  claimNextRun(db, "worker-test");
  const first = startRunAttempt(db, run.id, "worker-test", { attemptId: "attempt-one" });
  const second = startRunAttempt(db, run.id, "worker-test", { attemptId: "attempt-two" });
  expect(second?.id).toBe(first?.id);
  expect(listRunAttempts(db, run.id)).toHaveLength(1);
  db.close();
});

test("queued work is claimed by at most one worker", () => {
  const db = database();
  const created = createRun(db, trainingSpec());

  expect(claimNextRun(db, "worker-one")?.id).toBe(created.id);
  expect(claimNextRun(db, "worker-two")).toBeNull();
  expect(getRun(db, created.id)).toMatchObject({
    observedState: "claimed",
    workerId: "worker-one"
  });
  db.close();
});

test("checkpoint registration is idempotent but rejects conflicting immutable delivery", () => {
  const db = database();
  const logicalRun = createRun(db, trainingSpec());
  const attempt = startRunAttempt(db, logicalRun.id, "worker-test", { attemptId: "attempt-checkpoint" });
  const input = {
    runId: logicalRun.id,
    attemptId: attempt.id,
    phase: "pretraining",
    globalStep: 12,
    localPath: `runs/${logicalRun.id}/checkpoints/step-000000000012`,
    sha256: "a".repeat(64),
    configFingerprint: logicalRun.configFingerprint,
    datasetFingerprint: "dataset-fingerprint",
    gitCommit: "commit-a",
    metrics: { loss: 0.25, nested: { epoch: 1 } }
  };

  const first = registerTrainingCheckpoint(db, input);
  const repeated = registerTrainingCheckpoint(db, {
    ...input,
    metrics: { nested: { epoch: 1 }, loss: 0.25 }
  });
  expect(repeated).toEqual(first);
  expect(listTrainingCheckpoints(db, logicalRun.id)).toHaveLength(1);

  expect(() => registerTrainingCheckpoint(db, { ...input, sha256: "b".repeat(64) })).toThrow(/sha256/);
  expect(() => registerTrainingCheckpoint(db, { ...input, localPath: `${input.localPath}-other` })).toThrow(/localPath/);
  expect(() => registerTrainingCheckpoint(db, { ...input, phase: "criticality" })).toThrow(/phase/);
  expect(listTrainingCheckpoints(db, logicalRun.id)[0]).toMatchObject({
    id: first.id,
    sha256: "a".repeat(64),
    localPath: input.localPath
  });
  db.close();
});

test("checkpoint registration keeps the newest checkpoint as latest regardless of discovery order", () => {
  const db = database();
  const logicalRun = createRun(db, trainingSpec());
  const register = (globalStep) => registerTrainingCheckpoint(db, {
    runId: logicalRun.id,
    phase: "pretraining",
    globalStep,
    localPath: `runs/${logicalRun.id}/checkpoints/step-${String(globalStep).padStart(12, "0")}`,
    sha256: String(globalStep).padStart(64, "0"),
    configFingerprint: logicalRun.configFingerprint,
    datasetFingerprint: "dataset-fingerprint",
    gitCommit: "commit-a"
  });

  register(24);
  register(12);
  expect(getRun(db, logicalRun.id)?.latestCheckpointId).toBe(`checkpoint-${logicalRun.id}-24`);
  expect(getRun(db, logicalRun.id)?.globalStep).toBe(24);
  db.close();
});

test("attempt finalization accounts compute once and rejects conflicting delivery", () => {
  const db = database();
  const logicalRun = createRun(db, trainingSpec());
  const attempt = startRunAttempt(db, logicalRun.id, "worker-test", { attemptId: "attempt-finalize" });
  updateRunAttempt(db, attempt.id, { status: "running" });
  db.query("UPDATE run_attempts SET started_at = ? WHERE id = ?").run("2000-01-01T00:00:00.000Z", attempt.id);

  const first = finishRunAttempt(db, attempt.id, "succeeded", "completed");
  const repeated = finishRunAttempt(db, attempt.id, "succeeded", "completed");
  expect(first?.status).toBe("succeeded");
  expect(repeated?.status).toBe("succeeded");
  expect(listRunAttempts(db, logicalRun.id)[0].computeSeconds).toBeGreaterThan(0);
  expect(getRun(db, logicalRun.id)?.totalComputeSeconds).toBeGreaterThan(0);
  expect((db.query("SELECT COUNT(*) AS count FROM run_events WHERE run_id = ? AND event_json LIKE '%attempt.ended%'").get(logicalRun.id) as { count: number }).count).toBe(1);
  expect(() => finishRunAttempt(db, attempt.id, "failed", "different-result")).toThrow(/already finalized/);
  expect(heartbeatRunAttempt(db, attempt.id, "late", 99)).toBe(false);
  expect(getRun(db, logicalRun.id)?.phase).toBe("");
  db.close();
});

test("resume checkpoints are restricted to the run lineage", () => {
  const db = database();
  const source = createRun(db, trainingSpec());
  const unrelated = createRun(db, trainingSpec());
  const checkpoint = registerTrainingCheckpoint(db, {
    runId: source.id,
    phase: "pretraining",
    globalStep: 3,
    localPath: `runs/${source.id}/checkpoints/step-000000003`,
    sha256: "c".repeat(64),
    configFingerprint: source.configFingerprint,
    datasetFingerprint: "dataset-fingerprint"
  });

  expect(() => createRun(db, trainingSpec(), "project-local", undefined, {
    parentRunId: unrelated.id,
    resumeCheckpointId: checkpoint.id
  })).toThrow(/parent run lineage/);
  expect(() => startRunAttempt(db, unrelated.id, "worker-test", {
    resumeCheckpointId: checkpoint.id,
    attemptId: "attempt-unrelated"
  })).toThrow(/run lineage/);
  db.close();
});
