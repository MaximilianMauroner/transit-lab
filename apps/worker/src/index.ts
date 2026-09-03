import {
  createDatabase,
  recoverInterruptedRuns,
  repositoryRoot,
  requestRunPause
} from "../../../packages/control-store/src/database.ts";
import { claimRun, heartbeat, registerWorker, workerId } from "./claim-run.ts";
import { executeRun } from "./execute-run.ts";

function pause(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

export async function runWorker({
  root = repositoryRoot(),
  once = false,
  pollMs = 500,
  maxRetries = undefined,
  retryBackoffSeconds = undefined
} = {}) {
  const db = createDatabase(root);
  const id = registerWorker(db);
  let stopping = false;
  let activeRunId = null;
  const stop = () => {
    stopping = true;
    if (activeRunId) {
      try { requestRunPause(db, activeRunId, "worker-shutdown"); } catch { /* shutdown may race DB close */ }
    }
  };
  process.once("SIGINT", stop);
  process.once("SIGTERM", stop);
  try {
    recoverInterruptedRuns(db, id, 30);
    while (!stopping) {
      const claimed = claimRun(db, id);
      if (!claimed) {
        heartbeat(db, id, "idle");
        if (once) break;
        await pause(pollMs);
        continue;
      }
      activeRunId = claimed.id;
      heartbeat(db, id, "running", claimed.id);
      try {
        await executeRun({
          db,
          root,
          run: claimed,
          workerId: id,
          binary: process.env.TRANSIT_LAB_BINARY,
          maxRetries,
          retryBackoffSeconds
        });
      } finally {
        // A rejected executeRun must not leave this worker advertising a
        // claimed run. The next worker can then recover the stale attempt.
        activeRunId = null;
        heartbeat(db, id, "idle");
      }
      if (once) break;
    }
  } finally {
    process.removeListener("SIGINT", stop);
    process.removeListener("SIGTERM", stop);
    heartbeat(db, id, "offline");
    db.close();
  }
}

if (import.meta.main) {
  await runWorker({ once: process.argv.includes("--once") });
}
