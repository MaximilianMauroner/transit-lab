import { mkdir, readFile, readdir, rename, stat, writeFile } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import { fingerprint, validateArtifactManifest, validateRunEvent } from "../../../packages/contracts/src/index.ts";
import {
  addRunLog,
  appendRunEvent,
  appendRunEventType,
  dataRoot,
  finishRunAttempt,
  heartbeatRunAttempt,
  getRun,
  json,
  now,
  one,
  reconcileTrainingCheckpoints,
  run,
  startRunAttempt,
  updateRun,
  updateRunAttempt
} from "../../../packages/control-store/src/database.ts";
import { secondsUntilWindowEnd } from "../../../packages/control-store/src/schedule.ts";
import { syncFilesystem } from "../../../packages/control-store/src/inventory.ts";
import {
  createArtifactManifest,
  describeArtifactFile,
  repositoryRelative,
  writeArtifactManifest
} from "../../../packages/control-store/src/manifest.ts";
import { buildRustCommand, findLatestValidTrainingCheckpoint } from "./rust-commands.ts";
import { readStructuredEvents } from "./parse-events.ts";

const PAUSE_EXIT_CODE = 75;
const TIME_SLICE_EXIT_CODE = 76;
const CANCEL_EXIT_CODE = 77;
const DEFAULT_MAX_RETRIES = 3;
const MAX_RETRIES = 100;
const DEFAULT_RETRY_BACKOFF_SECONDS = 1;
const MAX_RETRY_BACKOFF_SECONDS = 60;

class UnexpectedChildExit extends Error {
  exitCode;

  constructor(exitCode) {
    super(`Rust command exited with status ${exitCode}`);
    this.name = "UnexpectedChildExit";
    this.exitCode = exitCode;
  }
}

function boundedInteger(value, fallback, { minimum = 0, maximum }) {
  const candidate = Number(value);
  if (!Number.isInteger(candidate) || candidate < minimum || candidate > maximum) return fallback;
  return candidate;
}

function retryOptions(runRecord, maxRetries, retryBackoffSeconds) {
  const runtime = runRecord.spec?.runtime || {};
  const configuredRetries = maxRetries ?? runtime.maxRetries ?? process.env.TRANSIT_LAB_MAX_RETRIES;
  const configuredBackoff = retryBackoffSeconds ?? runtime.retryBackoffSeconds ?? process.env.TRANSIT_LAB_RETRY_BACKOFF_SECONDS;
  return {
    maxRetries: boundedInteger(configuredRetries, DEFAULT_MAX_RETRIES, { maximum: MAX_RETRIES }),
    retryBackoffSeconds: boundedInteger(configuredBackoff, DEFAULT_RETRY_BACKOFF_SECONDS, {
      maximum: MAX_RETRY_BACKOFF_SECONDS
    })
  };
}

function retryableFailureCount(db, runId) {
  const row = one(db, `SELECT COUNT(*) AS count FROM run_attempts
    WHERE run_id = ? AND status = 'failed' AND exit_reason LIKE 'child-exit-%'`, [runId]);
  return Number(row?.count || 0);
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

async function consumeLines(stream, onLine) {
  if (!stream) return;
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let pending = "";
  while (true) {
    const chunk = await reader.read();
    if (chunk.done) break;
    pending += decoder.decode(chunk.value, { stream: true });
    const lines = pending.split(/\r?\n/);
    pending = lines.pop() || "";
    for (const line of lines) if (line) onLine(line);
  }
  pending += decoder.decode();
  if (pending) onLine(pending);
}

async function filesUnder(path) {
  let info;
  try { info = await stat(path); } catch { return []; }
  if (info.isFile()) return [path];
  if (!info.isDirectory()) return [];
  const entries = await readdir(path, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    if (entry.name === "artifact-manifest.json") continue;
    const child = join(path, entry.name);
    if (entry.isDirectory()) files.push(...await filesUnder(child));
    else if (entry.isFile()) files.push(child);
  }
  return files;
}

async function materializeOutputManifest({ root, runRecord, output }) {
  const artifactRoot = dataRoot(root);
  const outputInfo = await stat(output.path).catch(() => null);
  if (!outputInfo) return null;
  const existingPath = outputInfo.isDirectory()
    ? resolve(output.path, "artifact-manifest.json")
    : `${output.path}.artifact-manifest.json`;
  try {
    const existing = JSON.parse(await readFile(existingPath, "utf8"));
    validateArtifactManifest(existing);
    return { manifest: existing, manifestPath: existingPath };
  } catch (error) {
    if (error?.code !== "ENOENT") {
      // An invalid Rust manifest is a boundary failure; the worker must not
      // replace it with metadata inferred from stdout.
      if (error instanceof SyntaxError || /manifest|schema|artifact/i.test(error?.message || "")) throw error;
    }
  }
  const files = await filesUnder(output.path);
  if (!files.length) return null;
  const descriptors = [];
  for (const file of files) descriptors.push(await describeArtifactFile(artifactRoot, file));
  const artifactId = `artifact-${runRecord.id}-${output.kind}`;
  const manifest = await createArtifactManifest({
    root: artifactRoot,
    artifactId,
    kind: output.kind,
    fingerprint: fingerprint("artifact-output-v1", {
      runId: runRecord.id,
      kind: output.kind,
      files: descriptors
    }),
    files: descriptors,
    producingRunId: runRecord.id,
    configuration: {
      runSpec: runRecord.spec,
      configFingerprint: runRecord.configFingerprint || null,
      resolvedConfigPath: runRecord.resolvedConfigPath || null
    },
    metadata: { outputPath: repositoryRelative(artifactRoot, output.path) }
  });
  const manifestDirectory = outputInfo.isDirectory() ? output.path : dirname(output.path);
  const manifestPath = resolve(manifestDirectory, "artifact-manifest.json");
  await writeArtifactManifest(manifestPath, manifest);
  return { manifest, manifestPath };
}

function appendRustEvent(db, root, runId, event, attemptId = null) {
  validateRunEvent(event);
  if (event.attemptId && attemptId && event.attemptId !== attemptId) {
    throw new Error(`structured event attemptId ${event.attemptId} does not match ${attemptId}`);
  }
  const stored = appendRunEvent(db, runId, event);
  if (event.type === "progress") {
    updateRun(db, runId, {
      progressCompleted: event.completed,
      progressTotal: event.total,
      progressUnit: event.unit
    });
  }
  if (event.type === "metric") {
    updateRun(db, runId, {
      phase: event.phase,
      globalStep: event.step,
      progressCompleted: event.step,
      progressUnit: `${event.phase}-steps`
    });
  }
  if (event.type === "phase.started") {
    updateRun(db, runId, {
      phase: event.phase,
      progressCompleted: 0,
      progressTotal: event.total === undefined ? undefined : event.total,
      progressUnit: `${event.phase}-steps`
    });
  }
  if (event.type === "heartbeat") {
    const heartbeatAttemptId = event.attemptId || attemptId;
    if (heartbeatAttemptId) heartbeatRunAttempt(db, heartbeatAttemptId, event.phase, event.step);
  }
  if (event.type === "checkpoint.committed") {
    // The manifest is authoritative. Reconciliation validates payload hashes
    // and also covers the crash window between directory rename and SQLite.
    const current = getRun(db, runId);
    if (current) reconcileTrainingCheckpoints(db, root, runId);
  }
  return stored;
}

async function writeControlFile(path, desiredState, reason = null) {
  if (!path) return;
  const document = {
    schemaVersion: 1,
    desiredState: desiredState || "running",
    checkpointRequested: desiredState === "paused",
    requestedAt: now(),
    ...(reason ? { reason } : {})
  };
  const temporary = `${path}.tmp-${process.pid}`;
  await writeFile(temporary, `${JSON.stringify(document, null, 2)}\n`, { flag: "w" });
  await rename(temporary, path);
}

async function importStructuredEvents({ db, root, runId, path, cursor, artifacts, attemptId, final = false }) {
  let events;
  try {
    events = await readStructuredEvents(path, runId, { allowTrailingPartial: !final });
  } catch (error) {
    // A writer may be between append and newline. The final import will retry
    // and report a real malformed event instead of losing it.
    if (!final && error instanceof Error && /not valid JSON/.test(error.message)) return cursor;
    throw error;
  }
  for (const event of events.slice(cursor)) {
    if (event.type === "artifact.created") artifacts.add(event.artifactId);
    if ((event.type === "run.started" && event.step === undefined) ||
        event.type === "run.completed" || event.type === "run.failed" ||
        event.type === "run.paused" || event.type === "run.resumed" ||
        event.type === "run.time-sliced") continue;
    appendRustEvent(db, root, runId, event, attemptId);
  }
  return events.length;
}

function safeRunLog(db, runId, stream, line) {
  try {
    addRunLog(db, runId, stream, line);
  } catch {
    // The worker is already on an error path. A closed or unavailable control
    // store must not turn diagnostics into an unhandled exception.
  }
}

function waitForExit(promise, milliseconds) {
  if (!promise) return Promise.resolve(false);
  let timeout;
  return Promise.race([
    Promise.resolve(promise).then(() => true, () => true),
    new Promise((resolve) => {
      timeout = setTimeout(() => resolve(false), milliseconds);
    })
  ]).finally(() => clearTimeout(timeout));
}

async function terminateChild({ child, childExited, childExitPromise, stdoutPromise, stderrPromise }) {
  if (child && !childExited) {
    try { child.kill("SIGTERM"); } catch { /* process may have exited between the check and kill */ }
    const terminated = await waitForExit(childExitPromise, 2_000);
    if (!terminated) {
      try { child.kill("SIGKILL"); } catch { /* best effort; the worker still releases its references */ }
      await waitForExit(childExitPromise, 1_000);
      try { child.unref?.(); } catch { /* best effort */ }
    }
  }
  const streams = [stdoutPromise, stderrPromise].filter(Boolean);
  if (streams.length) {
    // A force-killed child should close its pipes. Bound this wait so a broken
    // pipe cannot prevent the worker from releasing the attempt and looping.
    await Promise.race([
      Promise.allSettled(streams),
      new Promise((resolve) => setTimeout(resolve, 2_000))
    ]);
  }
}

function refreshLatestCheckpoint(db, root, runId) {
  reconcileTrainingCheckpoints(db, root, runId);
  const latest = findLatestValidTrainingCheckpoint(db, root, runId, { includeResume: false });
  const current = getRun(db, runId);
  const latestId = latest?.row?.id || null;
  if (current && current.latestCheckpointId !== latestId) {
    updateRun(db, runId, { latestCheckpointId: latestId });
  }
  return latest;
}

export async function executeRun({
  db,
  root,
  run: runRecord,
  workerId,
  binary,
  maxRetries = undefined,
  retryBackoffSeconds = undefined,
  spawn = Bun.spawn
}) {
  let command;
  let monitorTimer;
  let eventTimer;
  let child;
  let attempt;
  let attemptFinished = false;
  let eventCursor = 0;
  let latestDesiredState = null;
  let childExitCode = null;
  let childExitPromise = null;
  let childExited = false;
  let stdoutPromise = null;
  let stderrPromise = null;
  let eventTail = Promise.resolve();
  let eventTailError = null;
  let controlTail = Promise.resolve();
  let controlWriteError = null;
  let cancellationKillRequested = false;
  const rustArtifactIds = new Set();
  const startedAt = Date.now();
  const retry = retryOptions(runRecord, maxRetries, retryBackoffSeconds);

  const finishAttempt = (status, reason) => {
    if (!attempt || attemptFinished) return;
    finishRunAttempt(db, attempt.id, status, reason);
    attemptFinished = true;
  };

  try {
    if (runRecord.cancelRequested || runRecord.desiredState === "cancelled") {
      updateRun(db, runRecord.id, {
        status: "cancelled",
        observedState: "cancelled",
        desiredState: "cancelled",
        currentAttemptId: null,
        workerId: null,
        finishedAt: now()
      });
      appendRunEventType(db, runRecord.id, "run.cancelled");
      return getRun(db, runRecord.id);
    }
    let currentForAttempt = getRun(db, runRecord.id) || runRecord;
    if (currentForAttempt.spec?.kind === "train") {
      refreshLatestCheckpoint(db, root, runRecord.id);
      currentForAttempt = getRun(db, runRecord.id) || currentForAttempt;
    }
    const runtime = currentForAttempt.spec?.runtime || {};
    const scheduledSeconds = secondsUntilWindowEnd(currentForAttempt.schedule, new Date());
    const configuredSeconds = Number(runtime.maxAttemptSeconds);
    const positiveLimits = [
      Number.isFinite(configuredSeconds) && configuredSeconds > 0 ? configuredSeconds : null,
      scheduledSeconds !== null ? Math.max(1, scheduledSeconds) : null
    ].filter((value) => value !== null);
    const effectiveMaxAttemptSeconds = positiveLimits.length ? Math.min(...positiveLimits) : undefined;
    const effectiveCheckpointGraceSeconds = scheduledSeconds !== null
      ? Number(runtime.checkpointGraceSeconds) > 0 ? Number(runtime.checkpointGraceSeconds) : 300
      : runtime.checkpointGraceSeconds;
    command = buildRustCommand({
      db,
      root,
      runId: runRecord.id,
      spec: currentForAttempt.spec,
      binary,
      maxAttemptSeconds: effectiveMaxAttemptSeconds,
      checkpointGraceSeconds: effectiveCheckpointGraceSeconds,
      forkFromCheckpoint: Boolean(currentForAttempt.resumeCheckpointId && !currentForAttempt.latestCheckpointId)
    });
    await mkdir(command.outputDirectory, { recursive: true });
    const currentBeforeAttempt = getRun(db, runRecord.id);
    attempt = startRunAttempt(db, runRecord.id, workerId, {
      resumeCheckpointId: currentBeforeAttempt?.latestCheckpointId
        || currentBeforeAttempt?.resumeCheckpointId
        || null,
      device: currentBeforeAttempt?.spec?.runtime || {}
    });
    if (!attempt) throw new Error(`run ${runRecord.id} disappeared before attempt start`);
    latestDesiredState = currentBeforeAttempt?.desiredState || "running";
    updateRunAttempt(db, attempt.id, { status: "running" });
    updateRun(db, runRecord.id, {
      status: "running",
      observedState: "running",
      currentStep: command.step,
      workerId,
      phase: command.step
    });
    run(db, `INSERT INTO run_steps(run_id, step, status, started_at, metrics_json)
      VALUES (?, ?, 'running', ?, '{}')
      ON CONFLICT(run_id, step) DO UPDATE SET status = 'running', started_at = excluded.started_at,
        finished_at = NULL, metrics_json = '{}'`, [runRecord.id, command.step, now()]);
    appendRunEventType(db, runRecord.id, "run.started", { step: command.step, attemptId: attempt.id });
    appendRunEventType(db, runRecord.id, "step.started", { step: command.step, attemptId: attempt.id });

    const attemptDirectory = resolve(command.outputDirectory, "attempts", attempt.id);
    await mkdir(attemptDirectory, { recursive: true });
    const attemptEventPath = resolve(attemptDirectory, "events.jsonl");
    if (command.controlFile) {
      await writeControlFile(command.controlFile, latestDesiredState);
    }
    const childEnvironment = Object.fromEntries(Object.entries({
      ...process.env,
      TRANSIT_RUN_ID: runRecord.id,
      TRANSIT_EVENT_FILE: attemptEventPath,
      TRANSIT_RESOLVED_CONFIG: runRecord.resolvedConfigPath
        ? resolve(dataRoot(root), runRecord.resolvedConfigPath)
        : undefined,
      TRANSIT_CONFIG_FINGERPRINT: runRecord.configFingerprint,
      TRANSIT_DATASET_FINGERPRINT: command.datasetFingerprint,
      TRANSIT_ATTEMPT_ID: attempt.id,
      TRANSIT_LAB_ROOT: root,
      TRANSIT_LAB_ARTIFACT_ROOT: dataRoot(root)
    }).filter(([, value]) => value !== undefined));
    child = spawn(command.argv, {
      cwd: root,
      env: childEnvironment,
      stdout: "pipe",
      stderr: "pipe"
    });
    childExitPromise = child.exited;
    stdoutPromise = consumeLines(child.stdout, (line) => addRunLog(db, runRecord.id, "stdout", line));
    stderrPromise = consumeLines(child.stderr, (line) => addRunLog(db, runRecord.id, "stderr", line));
    const importEvents = (final = false) => importStructuredEvents({
      db,
      root,
      runId: runRecord.id,
      path: attemptEventPath,
      cursor: eventCursor,
      artifacts: rustArtifactIds,
      attemptId: attempt.id,
      final
    }).then((next) => { eventCursor = next; });

    const scheduleEventImport = () => {
      eventTail = eventTail
        .then(() => importEvents())
        .catch((error) => {
          eventTailError ||= error;
          safeRunLog(db, runRecord.id, "stderr", `event tail: ${errorMessage(error)}`);
        });
    };
    // Import immediately so short-lived commands still expose their first
    // progress event before they exit, then serialize later polls. Concurrent
    // full-file reads can otherwise race the cursor and duplicate events.
    scheduleEventImport();
    eventTimer = setInterval(scheduleEventImport, 500);

    const monitor = () => {
      try {
        const current = getRun(db, runRecord.id);
        if (!current) return;
        const desired = current.cancelRequested ? "cancelled" : current.desiredState;
        if (desired !== latestDesiredState && command.controlFile) {
          latestDesiredState = desired;
          controlTail = controlTail
            .catch(() => {})
            .then(() => writeControlFile(
              command.controlFile,
              desired,
              desired === "paused" ? "pause-requested" : desired === "cancelled" ? "cancel-requested" : "resume-requested"
            ))
            .catch((error) => {
              controlWriteError ||= error;
              safeRunLog(db, runRecord.id, "stderr", `control file: ${errorMessage(error)}`);
            });
        }
        heartbeatRunAttempt(db, attempt.id, current.phase || command.step, current.globalStep);
        if (desired === "cancelled" && !cancellationKillRequested && Date.now() - startedAt > 30_000 && child?.kill) {
          cancellationKillRequested = true;
          child.kill("SIGTERM");
        }
      } catch (error) {
        safeRunLog(db, runRecord.id, "stderr", `worker monitor: ${errorMessage(error)}`);
      }
    };
    monitor();
    monitorTimer = setInterval(monitor, 500);

    const exitCode = await childExitPromise;
    childExitCode = exitCode;
    childExited = true;
    if (monitorTimer) clearInterval(monitorTimer);
    if (eventTimer) clearInterval(eventTimer);
    monitorTimer = null;
    eventTimer = null;
    const streamResults = await Promise.allSettled([stdoutPromise, stderrPromise].filter(Boolean));
    const streamFailure = streamResults.find((result) => result.status === "rejected");
    if (streamFailure?.status === "rejected") throw streamFailure.reason;
    await eventTail;
    await importEvents(true);
    if (eventTailError) throw eventTailError;
    await controlTail;
    if (controlWriteError) safeRunLog(db, runRecord.id, "stderr", `control file warning: ${errorMessage(controlWriteError)}`);

    const current = getRun(db, runRecord.id);
    const cancelled = current?.cancelRequested || current?.desiredState === "cancelled" || exitCode === CANCEL_EXIT_CODE;
    if (cancelled) {
      updateRun(db, runRecord.id, {
        status: "cancelled",
        observedState: "cancelled",
        desiredState: "cancelled",
        currentAttemptId: null,
        workerId: null,
        errorCode: null,
        errorMessage: null,
        finishedAt: now()
      });
      finishAttempt("cancelled", "cancel-requested");
      run(db, "UPDATE run_steps SET status = 'cancelled', finished_at = ? WHERE run_id = ? AND step = ?", [now(), runRecord.id, command.step]);
      appendRunEventType(db, runRecord.id, "run.cancelled", { attemptId: attempt.id });
      return getRun(db, runRecord.id);
    }
    if (exitCode === PAUSE_EXIT_CODE || current?.desiredState === "paused") {
      refreshLatestCheckpoint(db, root, runRecord.id);
      const paused = getRun(db, runRecord.id);
      updateRun(db, runRecord.id, {
        status: "paused",
        observedState: "paused",
        currentAttemptId: null,
        workerId: null,
        errorCode: null,
        errorMessage: null,
        pausedSince: paused?.pausedSince || now(),
        phase: paused?.phase || command.step,
        finishedAt: null
      });
      finishAttempt("paused", "cooperative-pause");
      run(db, "UPDATE run_steps SET status = 'paused', finished_at = ? WHERE run_id = ? AND step = ?", [now(), runRecord.id, command.step]);
      appendRunEventType(db, runRecord.id, "run.paused", { attemptId: attempt.id, checkpointId: paused?.latestCheckpointId || undefined });
      return getRun(db, runRecord.id);
    }
    if (exitCode === TIME_SLICE_EXIT_CODE) {
      refreshLatestCheckpoint(db, root, runRecord.id);
      const sliced = getRun(db, runRecord.id);
      updateRun(db, runRecord.id, {
        status: "queued",
        observedState: "queued",
        desiredState: "running",
        currentAttemptId: null,
        workerId: null,
        errorCode: null,
        errorMessage: null,
        resumeNotBefore: sliced?.resumeNotBefore || null,
        finishedAt: null
      });
      finishAttempt("time-sliced", "attempt-deadline");
      run(db, "UPDATE run_steps SET status = 'queued', finished_at = ? WHERE run_id = ? AND step = ?", [now(), runRecord.id, command.step]);
      appendRunEventType(db, runRecord.id, "run.time-sliced", { attemptId: attempt.id, checkpointId: sliced?.latestCheckpointId || undefined });
      return getRun(db, runRecord.id);
    }
    if (exitCode !== 0) throw new UnexpectedChildExit(exitCode);

    for (const output of command.outputs) {
      const artifact = await materializeOutputManifest({ root, runRecord, output });
      if (artifact && !rustArtifactIds.has(artifact.manifest.artifactId)) {
        appendRunEventType(db, runRecord.id, "artifact.created", {
          artifactId: artifact.manifest.artifactId,
          artifactKind: artifact.manifest.kind,
          uri: repositoryRelative(dataRoot(root), artifact.manifestPath),
          sha256: artifact.manifest.sha256
        });
      }
    }
    // Do inventory work before committing the run state so a failed index
    // update cannot leave a succeeded run with an attempt marked failed.
    await syncFilesystem(db, root);
    updateRun(db, runRecord.id, {
      status: "succeeded",
      observedState: "succeeded",
      desiredState: "running",
      currentAttemptId: null,
      workerId: null,
      currentStep: "",
      errorCode: null,
      errorMessage: null,
      finishedAt: now()
    });
    finishAttempt("succeeded", "completed");
    run(db, "UPDATE run_steps SET status = 'succeeded', finished_at = ? WHERE run_id = ? AND step = ?", [now(), runRecord.id, command.step]);
    appendRunEventType(db, runRecord.id, "step.completed", { step: command.step, attemptId: attempt.id });
    appendRunEventType(db, runRecord.id, "run.completed", { attemptId: attempt.id });
  } catch (error) {
    if (monitorTimer) clearInterval(monitorTimer);
    if (eventTimer) clearInterval(eventTimer);
    monitorTimer = null;
    eventTimer = null;
    await terminateChild({ child, childExited, childExitPromise, stdoutPromise, stderrPromise });
    await eventTail.catch(() => {});
    await controlTail.catch(() => {});

    const current = getRun(db, runRecord.id);
    const cancelled = current?.cancelRequested || current?.desiredState === "cancelled";
    let latestCheckpoint = null;
    let checkpointError = null;
    if (current?.spec?.kind === "train") {
      try {
        latestCheckpoint = refreshLatestCheckpoint(db, root, runRecord.id);
      } catch (checkpointFailure) {
        checkpointError = checkpointFailure;
        safeRunLog(db, runRecord.id, "stderr", `checkpoint recovery: ${errorMessage(checkpointFailure)}`);
      }
    }
    const paused = !cancelled && current?.desiredState === "paused";
    const childFailure = childExitCode !== null && childExitCode !== 0 &&
      ![PAUSE_EXIT_CODE, TIME_SLICE_EXIT_CODE, CANCEL_EXIT_CODE].includes(childExitCode);
    const failureMessage = errorMessage(error);

    if (cancelled) {
      updateRun(db, runRecord.id, {
        status: "cancelled",
        observedState: "cancelled",
        desiredState: "cancelled",
        currentAttemptId: null,
        workerId: null,
        errorCode: null,
        errorMessage: null,
        finishedAt: now()
      });
      finishAttempt("cancelled", "cancel-requested");
      if (command?.step) run(db, "UPDATE run_steps SET status = 'cancelled', finished_at = ? WHERE run_id = ? AND step = ?", [now(), runRecord.id, command.step]);
      appendRunEventType(db, runRecord.id, "run.cancelled", { attemptId: attempt?.id });
      return getRun(db, runRecord.id);
    }

    if (paused) {
      const pausedRun = getRun(db, runRecord.id);
      updateRun(db, runRecord.id, {
        status: "paused",
        observedState: "paused",
        currentAttemptId: null,
        workerId: null,
        errorCode: null,
        errorMessage: null,
        pausedSince: pausedRun?.pausedSince || now(),
        phase: pausedRun?.phase || command?.step || "",
        finishedAt: null
      });
      finishAttempt("paused", "cooperative-pause");
      if (command?.step) run(db, "UPDATE run_steps SET status = 'paused', finished_at = ? WHERE run_id = ? AND step = ?", [now(), runRecord.id, command.step]);
      appendRunEventType(db, runRecord.id, "run.paused", {
        attemptId: attempt?.id,
        checkpointId: pausedRun?.latestCheckpointId || latestCheckpoint?.row?.id || undefined
      });
      return getRun(db, runRecord.id);
    }

    if (childFailure && !checkpointError) {
      const usedRetries = retryableFailureCount(db, runRecord.id) + 1;
      finishAttempt("failed", `child-exit-${childExitCode}`);
      const retryCurrent = getRun(db, runRecord.id);
      if (usedRetries <= retry.maxRetries) {
        const resumeNotBefore = retry.retryBackoffSeconds > 0
          ? new Date(Date.now() + retry.retryBackoffSeconds * 1_000 * Math.min(2 ** (usedRetries - 1), 16)).toISOString()
          : null;
        updateRun(db, runRecord.id, {
          status: "queued",
          observedState: "queued",
          desiredState: "running",
          currentAttemptId: null,
          workerId: null,
          latestCheckpointId: latestCheckpoint?.row?.id || null,
          errorCode: "child-exit-retrying",
          errorMessage: `${failureMessage}; retry ${usedRetries}/${retry.maxRetries}`,
          resumeNotBefore,
          finishedAt: null
        });
        if (command?.step) run(db, "UPDATE run_steps SET status = 'queued', finished_at = ? WHERE run_id = ? AND step = ?", [now(), runRecord.id, command.step]);
        appendRunEventType(db, runRecord.id, "run.queued", {
          reason: "child-failure-retry",
          attemptId: attempt?.id,
          exitCode: childExitCode,
          retry: usedRetries,
          maxRetries: retry.maxRetries,
          ...(latestCheckpoint?.row?.id ? { checkpointId: latestCheckpoint.row.id } : {})
        });
        return getRun(db, runRecord.id);
      }
    } else if (childFailure && checkpointError) {
      finishAttempt("failed", "checkpoint-recovery-failed");
    } else if (attempt && !attemptFinished) {
      finishAttempt("failed", "worker-error");
    }

    const failed = getRun(db, runRecord.id);
    updateRun(db, runRecord.id, {
      status: "failed",
      observedState: "failed",
      desiredState: failed?.desiredState || "running",
      currentAttemptId: null,
      workerId: null,
      errorCode: childFailure ? "rust-run-failed" : "worker-error",
      errorMessage: failureMessage,
      finishedAt: now()
    });
    if (command?.step) {
      run(db, "UPDATE run_steps SET status = 'failed', finished_at = ? WHERE run_id = ? AND step = ?", [now(), runRecord.id, command.step]);
    }
    appendRunEventType(db, runRecord.id, "run.failed", {
      attemptId: attempt?.id,
      code: childFailure ? "rust-run-failed" : "worker-error",
      message: failureMessage
    });
  }
  return getRun(db, runRecord.id);
}
