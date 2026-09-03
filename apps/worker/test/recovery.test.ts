import { expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { trainingCheckpointFingerprint } from "../../../packages/contracts/src/index.ts";
import {
  claimNextRun,
  createDatabase,
  createRun,
  getRun,
  listRunAttempts,
  listTrainingCheckpoints,
  reconcileTrainingCheckpoints,
  recoverInterruptedRuns,
  startRunAttempt,
  updateRun,
  updateRunAttempt
} from "../../../packages/control-store/src/database.ts";
import { findLatestValidTrainingCheckpoint } from "../src/rust-commands.ts";

function trainingSpec() {
  return {
    kind: "train",
    datasetId: "dataset-test",
    modelConfig: {},
    seed: 7,
    runtime: {}
  };
}

function checkpointManifest(runId, globalStep, file) {
  return {
    schemaVersion: 1,
    runId,
    attemptId: null,
    globalStep,
    phase: "pretraining",
    datasetFingerprint: "dataset-fingerprint",
    configFingerprint: "config-fingerprint",
    codeCommit: "working-tree",
    backend: "reference",
    backendVersion: "test",
    deviceType: "cpu",
    status: "committed",
    checkpointFingerprint: trainingCheckpointFingerprint([file]),
    files: [file]
  };
}

test("checkpoint reconciliation imports only committed, contained, hash-valid directories", async () => {
  const root = await mkdtemp(join(tmpdir(), "transit-lab-worker-recovery-"));
  const db = createDatabase(root, ":memory:");
  const created = createRun(db, trainingSpec(), "project-local", root);
  const checkpointRoot = join(root, "data", created.checkpointRoot);
  const validDirectory = join(checkpointRoot, "step-000000001");
  await mkdir(validDirectory, { recursive: true });
  const validBytes = Buffer.from("valid-checkpoint");
  const validFile = {
    path: "model.ot",
    sha256: createHash("sha256").update(validBytes).digest("hex"),
    sizeBytes: validBytes.byteLength
  };
  await writeFile(join(validDirectory, validFile.path), validBytes);
  await writeFile(
    join(validDirectory, "manifest.json"),
    JSON.stringify(checkpointManifest(created.id, 1, validFile))
  );

  const external = join(root, "outside.bin");
  await writeFile(external, "outside");
  const escapedDirectory = join(checkpointRoot, "step-000000002");
  await mkdir(escapedDirectory, { recursive: true });
  const escapedBytes = Buffer.from("outside");
  const escapedFile = {
    path: "model.ot",
    sha256: createHash("sha256").update(escapedBytes).digest("hex"),
    sizeBytes: escapedBytes.byteLength
  };
  await symlink(external, join(escapedDirectory, escapedFile.path));
  await writeFile(
    join(escapedDirectory, "manifest.json"),
    JSON.stringify(checkpointManifest(created.id, 2, escapedFile))
  );

  const restored = reconcileTrainingCheckpoints(db, root, created.id);
  expect(restored).toHaveLength(1);
  expect(listTrainingCheckpoints(db, created.id)).toHaveLength(1);
  expect(findLatestValidTrainingCheckpoint(db, root, created.id)?.row.global_step).toBe(1);

  await writeFile(join(validDirectory, validFile.path), "corrupted");
  expect(findLatestValidTrainingCheckpoint(db, root, created.id)).toBeNull();
  db.close();
});

test("stale attempts are interrupted once and their logical runs are requeued", () => {
  const root = process.cwd();
  const db = createDatabase(root, ":memory:");
  const created = createRun(db, trainingSpec(), "project-local", root);
  expect(claimNextRun(db, "worker-one")?.id).toBe(created.id);
  const attempt = startRunAttempt(db, created.id, "worker-one", { attemptId: "attempt-stale" });
  expect(attempt?.id).toBe("attempt-stale");
  updateRunAttempt(db, attempt.id, { status: "running" });
  updateRun(db, created.id, { lastHeartbeatAt: "2000-01-01T00:00:00.000Z" });

  expect(recoverInterruptedRuns(db, "worker-two", 1)).toEqual([created.id]);
  expect(getRun(db, created.id)).toMatchObject({
    observedState: "queued",
    desiredState: "running",
    currentAttemptId: null
  });
  expect(listRunAttempts(db, created.id)[0]).toMatchObject({
    status: "interrupted",
    exitReason: "worker-lease-expired",
    computeSeconds: expect.any(Number)
  });
  expect(listRunAttempts(db, created.id)[0].computeSeconds).toBeGreaterThan(0);
  expect(getRun(db, created.id)?.totalComputeSeconds).toBeGreaterThan(0);
  expect((db.query("SELECT COUNT(*) AS count FROM run_events WHERE run_id = ? AND event_json LIKE '%attempt.ended%'").get(created.id) as { count: number }).count).toBe(1);
  expect(recoverInterruptedRuns(db, "worker-two", 1)).toEqual([]);
  db.close();
});
