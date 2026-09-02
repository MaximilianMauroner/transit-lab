import { hostname } from "node:os";
import { claimNextRun, json, now, run } from "../server/database/db.js";

export function workerId() {
  return process.env.TRANSIT_LAB_WORKER_ID || `studio-worker-${hostname()}-${process.pid}`;
}

export function registerWorker(db, id = workerId()) {
  run(db, `INSERT INTO workers(id, hostname, status, last_heartbeat_at, metadata_json)
    VALUES (?, ?, 'idle', ?, ?)
    ON CONFLICT(id) DO UPDATE SET hostname = excluded.hostname, status = 'idle', last_heartbeat_at = excluded.last_heartbeat_at`, [
    id,
    hostname(),
    now(),
    json({ pid: process.pid, application: "transit-lab-studio" })
  ]);
  return id;
}

export function heartbeat(db, id, status = "idle", currentRunId = null) {
  run(db, `UPDATE workers SET status = ?, current_run_id = ?, last_heartbeat_at = ? WHERE id = ?`, [
    status,
    currentRunId,
    now(),
    id
  ]);
}

export function claimRun(db, id) {
  return claimNextRun(db, id);
}
