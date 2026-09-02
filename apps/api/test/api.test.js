import { expect, test } from "bun:test";
import { createApiHandler } from "../src/server.js";
import {
  appendRunEventType,
  createDatabase,
  now,
  run,
  updateRun
} from "../src/db.js";

const ROOT = new URL("../../../", import.meta.url).pathname.replace(/\/$/, "");

function seedDatabase() {
  const db = createDatabase(ROOT, ":memory:");
  const timestamp = now();
  run(db, "INSERT INTO projects(id, name, description, created_at) VALUES (?, ?, ?, ?)", ["project-local", "Transit Lab", "test", timestamp]);
  run(db, "INSERT INTO networks(id, project_id, display_name, geographical_scope, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)", ["test-network", "project-local", "Test Network", "Test City", timestamp, timestamp]);
  run(db, `INSERT INTO feed_revisions(id, network_id, sha256, local_path, validation_status, created_at)
    VALUES (?, ?, ?, ?, ?, ?)`, ["feed-test", "test-network", "a".repeat(64), "data/raw/test/gtfs.zip", "valid", timestamp]);
  run(db, `INSERT INTO canonical_lines(id, network_id, canonical_name, mode, created_at)
    VALUES (?, ?, ?, ?, ?)`, ["canonical-line-1", "test-network", "1", "metro", timestamp]);
  run(db, `INSERT INTO canonical_lines(id, network_id, canonical_name, mode, created_at)
    VALUES (?, ?, ?, ?, ?)`, ["canonical-line-2", "test-network", "2", "tram", timestamp]);
  for (const [id, date] of [["snapshot-a", "2026-09-01"], ["snapshot-b", "2026-09-02"]]) {
    run(db, `INSERT INTO snapshots(id, network_id, service_date, status, fingerprint, manifest_path, network_path, counts_json, validation_json, created_at, updated_at)
      VALUES (?, ?, ?, 'ready', ?, ?, ?, ?, ?, ?, ?)`, [
      id,
      "test-network",
      date,
      `fingerprint-${id}`,
      `data/snapshots/${id}/manifest.json`,
      `data/snapshots/${id}/network.json`,
      JSON.stringify({ stations: 3, lines: 2, patterns: 2 }),
      JSON.stringify({ errors: [], warnings: [] }),
      timestamp,
      timestamp
    ]);
  }
  const feature = (overrides = {}) => JSON.stringify({
    station_count: 3,
    pattern_count: 1,
    route_length_metres: 1_000,
    end_to_end_distance_metres: 900,
    branching_factor: 1,
    service_span_seconds: 50_000,
    daily_trip_count: 20,
    median_headway_seconds: 600,
    peak_headway_seconds: 480,
    off_peak_headway_seconds: 900,
    transfer_station_count: 2,
    unique_station_fraction: 0.5,
    shared_segment_fraction: 0.2,
    ...overrides
  });
  for (const snapshotId of ["snapshot-a", "snapshot-b"]) {
    run(db, `INSERT INTO line_instances(id, snapshot_id, canonical_line_id, line_index, canonical_id, display_name, agency_key, mode, feature_json, geometry_json, created_at, updated_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '{}', ?, ?)`, [
      `${snapshotId}:0`, snapshotId, "canonical-line-1", 0, "line:1", "1", "agency", 1, feature(), timestamp, timestamp
    ]);
    run(db, `INSERT INTO line_instances(id, snapshot_id, canonical_line_id, line_index, canonical_id, display_name, agency_key, mode, feature_json, geometry_json, created_at, updated_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '{}', ?, ?)`, [
      `${snapshotId}:1`, snapshotId, "canonical-line-2", 1, "line:2", "2", "agency", 3, feature({ route_length_metres: 3_000, transfer_station_count: 0, daily_trip_count: 8 }), timestamp, timestamp
    ]);
  }
  run(db, "INSERT INTO model_versions(id, version, fingerprint, status, architecture_json, embedding_dimensions_json, supported_heads_json, evaluation_json, created_at) VALUES (?, ?, ?, 'ready', '{}', '{}', '[]', '{}', ?)", ["model-test", "test-v1", "model-fingerprint", timestamp]);
  run(db, "INSERT INTO model_aliases(alias, model_id, updated_at) VALUES (?, ?, ?)", ["candidate", "model-test", timestamp]);
  run(db, `INSERT INTO inference_sets(id, fingerprint, model_id, snapshot_id, status, config_json, created_at)
    VALUES (?, ?, ?, ?, 'ready', ?, ?)`, ["inference-test", "inference-fingerprint", "model-test", "snapshot-a", JSON.stringify({ metricNames: ["accessibility_auc_loss"] }), timestamp]);
  run(db, `INSERT INTO criticality_predictions(inference_id, line_instance_id, primary_score, uncertainty, values_json, created_at)
    VALUES (?, ?, ?, NULL, ?, ?)`, ["inference-test", "snapshot-a:0", 0.8, JSON.stringify({ accessibility_auc_loss: 0.8, unreachable_share: 0.2 }), timestamp]);
  run(db, `INSERT INTO criticality_labels(snapshot_id, line_index, values_json, created_at)
    VALUES (?, ?, ?, ?)`, ["snapshot-a", 0, JSON.stringify({ accessibility_auc_loss: 0.7, policy_fingerprint: "policy-1" }), timestamp]);
  return db;
}

async function call(handler, path, options = {}) {
  return handler(new Request(`http://transit.test${path}`, options));
}

test("catalog and line responses retain feeds, predictions, labels, and provenance", async () => {
  const db = seedDatabase();
  const handler = createApiHandler({ db, root: ROOT });
  const catalog = await (await call(handler, "/api/catalog")).json();
  expect(catalog.networks[0].feeds[0].id).toBe("feed-test");
  expect(catalog.snapshots).toHaveLength(2);
  const lines = await (await call(handler, "/api/snapshots/snapshot-a/lines")).json();
  expect(lines[0].criticality.primaryScore).toBe(0.8);
  expect(lines[0].label.policy_fingerprint).toBe("policy-1");
  const detail = await (await call(handler, "/api/lines/snapshot-a:0")).json();
  expect(detail.provenance.modelId).toBe("model-test");
  expect(detail.label.accessibility_auc_loss).toBe(0.7);
  db.close();
});

test("network-role is a role-only profile and similarity uses latest ready inference rows", async () => {
  const db = seedDatabase();
  const handler = createApiHandler({ db, root: ROOT });
  const response = await call(handler, "/api/similarity/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      querySnapshotId: "snapshot-a",
      queryLineId: "snapshot-a:0",
      candidateSnapshotId: "snapshot-b",
      profile: "network-role",
      topK: 1
    })
  });
  expect(response.status).toBe(200);
  const result = await response.json();
  expect(result.weights).toEqual({ role: 1, service: 0, geometry: 0, resilience: 0 });
  expect(result.matches).toHaveLength(1);
  expect(result.matches[0].comparison.sameMode).toBe(true);
  db.close();
});

test("queued runs emit sequence zero and SSE resumes after the requested event", async () => {
  const db = seedDatabase();
  const handler = createApiHandler({ db, root: ROOT });
  const created = await (await call(handler, "/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ kind: "simulate-criticality", snapshotId: "snapshot-a" })
  })).json();
  expect(created.status).toBe("queued");
  expect(created.events[0]).toMatchObject({ seq: 0, type: "run.queued" });
  updateRun(db, created.id, { status: "succeeded", finishedAt: now() });
  const initial = await (await call(handler, `/api/runs/${created.id}/events`)).text();
  expect(initial).toContain("id: 0");
  const resumed = await (await call(handler, `/api/runs/${created.id}/events`, { headers: { "Last-Event-ID": "0" } })).text();
  expect(resumed).not.toContain("id: 0");
  db.close();
});

test("run creation rejects missing resources and shell-shaped identifiers", async () => {
  const db = seedDatabase();
  const handler = createApiHandler({ db, root: ROOT });
  const response = await call(handler, "/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ kind: "infer", snapshotId: "snapshot-a", modelId: "../../etc/passwd" })
  });
  expect(response.status).toBe(422);
  expect((await response.json()).error).toContain("safe identifier");
  db.close();
});
