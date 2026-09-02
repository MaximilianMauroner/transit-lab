import { createDatabase, repositoryRoot } from "../../../packages/control-store/src/database.ts";
import { claimRun, heartbeat, registerWorker, workerId } from "./claim-run.ts";
import { executeRun } from "./execute-run.ts";

function pause(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

export async function runWorker({ root = repositoryRoot(), once = false, pollMs = 500 } = {}) {
  const db = createDatabase(root);
  const id = registerWorker(db);
  let stopping = false;
  const stop = () => { stopping = true; };
  process.once("SIGINT", stop);
  process.once("SIGTERM", stop);
  try {
    while (!stopping) {
      const claimed = claimRun(db, id);
      if (!claimed) {
        heartbeat(db, id, "idle");
        if (once) break;
        await pause(pollMs);
        continue;
      }
      heartbeat(db, id, "running", claimed.id);
      await executeRun({
        db,
        root,
        run: claimed,
        workerId: id,
        binary: process.env.TRANSIT_LAB_BINARY
      });
      heartbeat(db, id, "idle");
      if (once) break;
    }
  } finally {
    heartbeat(db, id, "offline");
    db.close();
  }
}

if (import.meta.main) {
  await runWorker({ once: process.argv.includes("--once") });
}
