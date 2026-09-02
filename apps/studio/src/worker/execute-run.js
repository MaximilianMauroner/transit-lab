import { mkdir, readFile, readdir, stat } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import { fingerprint, validateArtifactManifest, validateRunEvent } from "../shared/contracts/index.js";
import {
  addRunLog,
  appendRunEvent,
  appendRunEventType,
  getRun,
  json,
  now,
  run,
  updateRun
} from "../server/database/db.js";
import { syncFilesystem } from "../server/artifacts/inventory.js";
import {
  createArtifactManifest,
  describeArtifactFile,
  repositoryRelative,
  writeArtifactManifest
} from "../server/artifacts/manifest.js";
import { buildRustCommand } from "./rust-commands.js";
import { readStructuredEvents } from "./parse-events.js";

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
  for (const file of files) descriptors.push(await describeArtifactFile(root, file));
  const artifactId = `artifact-${runRecord.id}-${output.kind}`;
  const manifest = await createArtifactManifest({
    root,
    artifactId,
    kind: output.kind,
    fingerprint: fingerprint("artifact-output-v1", {
      runId: runRecord.id,
      kind: output.kind,
      files: descriptors
    }),
    files: descriptors,
    producingRunId: runRecord.id,
    configuration: runRecord.spec,
    metadata: { outputPath: repositoryRelative(root, output.path) }
  });
  const manifestDirectory = outputInfo.isDirectory() ? output.path : dirname(output.path);
  const manifestPath = resolve(manifestDirectory, "artifact-manifest.json");
  await writeArtifactManifest(manifestPath, manifest);
  return { manifest, manifestPath };
}

function appendRustEvent(db, runId, event) {
  validateRunEvent(event);
  const stored = appendRunEvent(db, runId, event);
  if (event.type === "progress") {
    updateRun(db, runId, {
      progressCompleted: event.completed,
      progressTotal: event.total,
      progressUnit: event.unit
    });
  }
  return stored;
}

export async function executeRun({ db, root, run: runRecord, workerId, binary }) {
  let command;
  let cancelTimer;
  let child;
  try {
    if (runRecord.cancelRequested) {
      updateRun(db, runRecord.id, { status: "cancelled", finishedAt: now() });
      appendRunEventType(db, runRecord.id, "run.cancelled");
      return getRun(db, runRecord.id);
    }
    command = buildRustCommand({ db, root, runId: runRecord.id, spec: runRecord.spec, binary });
    await mkdir(command.outputDirectory, { recursive: true });
    updateRun(db, runRecord.id, {
      status: "running",
      startedAt: now(),
      currentStep: command.step,
      workerId
    });
    run(db, `INSERT INTO run_steps(run_id, step, status, started_at, metrics_json)
      VALUES (?, ?, 'running', ?, '{}')
      ON CONFLICT(run_id, step) DO UPDATE SET status = 'running', started_at = excluded.started_at,
        finished_at = NULL, metrics_json = '{}'`, [runRecord.id, command.step, now()]);
    appendRunEventType(db, runRecord.id, "run.started", { step: command.step });
    appendRunEventType(db, runRecord.id, "step.started", { step: command.step });

    const eventPath = resolve(command.outputDirectory, "events.jsonl");
    const childEnvironment = Object.fromEntries(Object.entries({
      ...process.env,
      TRANSIT_RUN_ID: runRecord.id,
      TRANSIT_EVENT_FILE: eventPath,
      TRANSIT_LAB_ROOT: root
    }).filter(([, value]) => value !== undefined));
    child = Bun.spawn(command.argv, {
      cwd: root,
      env: childEnvironment,
      stdout: "pipe",
      stderr: "pipe"
    });
    const stdout = consumeLines(child.stdout, (line) => addRunLog(db, runRecord.id, "stdout", line));
    const stderr = consumeLines(child.stderr, (line) => addRunLog(db, runRecord.id, "stderr", line));
    cancelTimer = setInterval(() => {
      const current = getRun(db, runRecord.id);
      if (current?.cancelRequested && child?.kill) child.kill();
    }, 500);
    const exitCode = await child.exited;
    await Promise.all([stdout, stderr]);
    if (cancelTimer) clearInterval(cancelTimer);

    const structuredEvents = await readStructuredEvents(eventPath, runRecord.id);
    const rustArtifactIds = new Set();
    for (const event of structuredEvents) {
      if (event.type === "artifact.created") rustArtifactIds.add(event.artifactId);
      // The worker owns the control-plane lifecycle. Rust keeps these events
      // in its JSONL evidence, but importing them as well would duplicate the
      // same state transition in the Studio ledger.
      if ((event.type === "run.started" && event.step === undefined) ||
          event.type === "run.completed" || event.type === "run.failed") continue;
      appendRustEvent(db, runRecord.id, event);
    }
    if (exitCode !== 0) throw new Error(`Rust command exited with status ${exitCode}`);

    for (const output of command.outputs) {
      const artifact = await materializeOutputManifest({ root, runRecord, output });
      if (artifact) {
        if (!rustArtifactIds.has(artifact.manifest.artifactId)) {
          appendRunEventType(db, runRecord.id, "artifact.created", {
            artifactId: artifact.manifest.artifactId,
            artifactKind: artifact.manifest.kind,
            uri: repositoryRelative(root, artifact.manifestPath),
            sha256: artifact.manifest.sha256
          });
        }
      }
    }
    updateRun(db, runRecord.id, {
      status: "succeeded",
      currentStep: "",
      finishedAt: now()
    });
    run(db, "UPDATE run_steps SET status = 'succeeded', finished_at = ? WHERE run_id = ? AND step = ?", [now(), runRecord.id, command.step]);
    appendRunEventType(db, runRecord.id, "step.completed", { step: command.step });
    appendRunEventType(db, runRecord.id, "run.completed");
    await syncFilesystem(db, root);
  } catch (error) {
    if (cancelTimer) clearInterval(cancelTimer);
    const current = getRun(db, runRecord.id);
    const cancelled = current?.cancelRequested || (child && child.exitCode === null && current?.status === "running");
    updateRun(db, runRecord.id, {
      status: cancelled ? "cancelled" : "failed",
      errorCode: cancelled ? "cancelled" : "rust-run-failed",
      errorMessage: error instanceof Error ? error.message : String(error),
      finishedAt: now()
    });
    if (command?.step) {
      run(db, "UPDATE run_steps SET status = ?, finished_at = ? WHERE run_id = ? AND step = ?", [cancelled ? "cancelled" : "failed", now(), runRecord.id, command.step]);
    }
    appendRunEventType(db, runRecord.id, cancelled ? "run.cancelled" : "run.failed", cancelled
      ? {}
      : { code: "rust-run-failed", message: error instanceof Error ? error.message : String(error) });
  }
  return getRun(db, runRecord.id);
}
