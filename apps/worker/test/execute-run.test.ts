import { expect, test } from "bun:test";
import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  claimNextRun,
  createDatabase,
  createRun,
  getRun,
  getRunEvents,
  listRunAttempts,
  requestRunPause
} from "../../../packages/control-store/src/database.ts";
import { syncFilesystem } from "../../../packages/control-store/src/inventory.ts";
import { executeRun } from "../src/execute-run.ts";

function closedStream() {
  return new ReadableStream({
    start(controller) {
      controller.close();
    }
  });
}

function fakeChild() {
  let resolveExit;
  const exited = new Promise((resolve) => {
    resolveExit = resolve;
  });
  return {
    exited,
    stdout: closedStream(),
    stderr: closedStream(),
    kill() {
      resolveExit(77);
    },
    unref() {},
    resolveExit
  };
}

async function waitFor(predicate, timeout = 2_000) {
  const deadline = Date.now() + timeout;
  while (!predicate() && Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  expect(predicate()).toBe(true);
}

async function simulationFixture() {
  const root = await mkdtemp(join(tmpdir(), "transit-lab-worker-execute-"));
  const snapshotDirectory = join(root, "data/snapshots/demo/2026-09-02");
  await mkdir(snapshotDirectory, { recursive: true });
  await writeFile(join(snapshotDirectory, "manifest.json"), JSON.stringify({
    snapshot_id: "snapshot-1",
    source_name: "Worker test",
    geographical_scope: "test",
    descriptor: { service_date: "2026-09-02", compiler_version: "test" }
  }));
  await writeFile(join(snapshotDirectory, "network.json"), JSON.stringify({
    snapshot_id: "snapshot-1",
    stations: [{ name: "Central", latitude: 48.2, longitude: 16.3 }],
    lines: [{ index: 0, canonical_id: "test", display_name: "Test", mode: 1 }],
    patterns: []
  }));

  const db = createDatabase(root, ":memory:");
  await syncFilesystem(db, root);
  const created = createRun(db, {
    kind: "simulate-criticality",
    snapshotId: "snapshot-1"
  }, "project-local", root);
  const claimed = claimNextRun(db, "worker-test");
  expect(claimed?.id).toBe(created.id);
  return { root, db, run: claimed };
}

function flagValue(argv, flag) {
  const index = argv.indexOf(flag);
  return index < 0 ? null : argv[index + 1];
}

test("executeRun tails live events once and releases a successful attempt", async () => {
  const fixture = await simulationFixture();
  const { root, db, run: runRecord } = fixture;
  const spawn = (argv, options) => {
    const child = fakeChild();
    const event = {
      schemaVersion: 2,
      seq: 0,
      runId: runRecord.id,
      timestamp: new Date().toISOString(),
      type: "progress",
      attemptId: options.env.TRANSIT_ATTEMPT_ID,
      attemptSeq: 0,
      step: "simulate-criticality",
      completed: 1,
      total: 1,
      unit: "query"
    };
    void (async () => {
      await writeFile(options.env.TRANSIT_EVENT_FILE, `${JSON.stringify(event)}\n`);
      await writeFile(
        join(root, flagValue(argv, "--output")),
        '{"snapshot":"snapshot-1","line":0}\n'
      );
      child.resolveExit(0);
    })();
    return child;
  };

  const result = await executeRun({
    db,
    root,
    run: runRecord,
    workerId: "worker-test",
    binary: "fake-transit",
    spawn: spawn as any
  });

  expect(result).toMatchObject({
    id: runRecord.id,
    status: "succeeded",
    observedState: "succeeded",
    currentAttemptId: null,
    workerId: null
  });
  expect(listRunAttempts(db, runRecord.id)).toMatchObject([{
    status: "succeeded",
    exitReason: "completed"
  }]);
  const events = getRunEvents(db, runRecord.id);
  expect(events.filter((event) => event.type === "progress")).toHaveLength(1);
  expect(new Set(events.map((event) => event.seq)).size).toBe(events.length);
  expect(events.at(-1)?.type).toBe("run.completed");
  db.close();
});

test("executeRun turns a cooperative pause into a resource-free paused run", async () => {
  const fixture = await simulationFixture();
  const { root, db, run: runRecord } = fixture;
  let child;
  const spawn = () => {
    child = fakeChild();
    return child;
  };
  const execution = executeRun({
    db,
    root,
    run: runRecord,
    workerId: "worker-test",
    binary: "fake-transit",
    spawn: spawn as any
  });

  await waitFor(() => getRun(db, runRecord.id)?.observedState === "running");
  requestRunPause(db, runRecord.id);
  child.resolveExit(75);
  const result = await execution;

  expect(result).toMatchObject({
    id: runRecord.id,
    status: "paused",
    observedState: "paused",
    desiredState: "paused",
    currentAttemptId: null,
    workerId: null
  });
  expect(listRunAttempts(db, runRecord.id)).toMatchObject([{
    status: "paused",
    exitReason: "cooperative-pause"
  }]);
  expect(getRun(db, runRecord.id)?.pausedSince).not.toBeNull();
  db.close();
});

test("executeRun requeues a deadline-sliced attempt for a later worker", async () => {
  const fixture = await simulationFixture();
  const { root, db, run: runRecord } = fixture;
  const spawn = () => {
    const child = fakeChild();
    queueMicrotask(() => child.resolveExit(76));
    return child;
  };

  const result = await executeRun({
    db,
    root,
    run: runRecord,
    workerId: "worker-test",
    binary: "fake-transit",
    spawn: spawn as any
  });

  expect(result).toMatchObject({
    id: runRecord.id,
    status: "queued",
    observedState: "queued",
    desiredState: "running",
    currentAttemptId: null,
    workerId: null
  });
  expect(listRunAttempts(db, runRecord.id)).toMatchObject([{
    status: "time-sliced",
    exitReason: "attempt-deadline"
  }]);
  expect(claimNextRun(db, "worker-two")?.id).toBe(runRecord.id);
  db.close();
});

test("executeRun retries child failures with a new attempt and then succeeds", async () => {
  const fixture = await simulationFixture();
  const { root, db, run: initialRun } = fixture;
  const exitCodes = [9, 0];
  const spawn = () => {
    const child = fakeChild();
    queueMicrotask(() => child.resolveExit(exitCodes.shift()));
    return child;
  };

  const first = await executeRun({
    db,
    root,
    run: initialRun,
    workerId: "worker-one",
    binary: "fake-transit",
    maxRetries: 1,
    retryBackoffSeconds: 0,
    spawn: spawn as any
  });
  expect(first).toMatchObject({ status: "queued", error: { code: "child-exit-retrying" } });

  const nextRun = claimNextRun(db, "worker-two");
  expect(nextRun?.id).toBe(initialRun.id);
  const second = await executeRun({
    db,
    root,
    run: nextRun,
    workerId: "worker-two",
    binary: "fake-transit",
    maxRetries: 1,
    retryBackoffSeconds: 0,
    spawn: spawn as any
  });

  expect(second).toMatchObject({ status: "succeeded", observedState: "succeeded" });
  expect(listRunAttempts(db, initialRun.id)).toMatchObject([
    { status: "failed", exitReason: "child-exit-9" },
    { status: "succeeded", exitReason: "completed" }
  ]);
  db.close();
});

test("executeRun fails closed when a child leaves an invalid artifact manifest", async () => {
  const fixture = await simulationFixture();
  const { root, db, run: runRecord } = fixture;
  const spawn = (argv) => {
    const child = fakeChild();
    const output = join(root, flagValue(argv, "--output"));
    void (async () => {
      await writeFile(output, '{"snapshot":"snapshot-1","line":0}\n');
      await writeFile(`${output}.artifact-manifest.json`, "{}\n");
      child.resolveExit(0);
    })();
    return child;
  };

  const result = await executeRun({
    db,
    root,
    run: runRecord,
    workerId: "worker-test",
    binary: "fake-transit",
    spawn: spawn as any
  });

  expect(result).toMatchObject({
    id: runRecord.id,
    status: "failed",
    observedState: "failed",
    error: { code: "worker-error" }
  });
  expect(listRunAttempts(db, runRecord.id)).toMatchObject([{
    status: "failed",
    exitReason: "worker-error"
  }]);
  db.close();
});
