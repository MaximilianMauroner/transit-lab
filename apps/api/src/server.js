import { mkdir } from "node:fs/promises";
import { extname, join, relative, resolve, sep } from "node:path";
import {
  normalizeRunSpec,
  profileWeights,
  RUN_EVENT_SCHEMA_VERSION,
  validateRunEvent
} from "../../../packages/contracts/src/index.js";
import {
  all,
  appendRunEventType,
  createDatabase,
  dataRoot,
  databasePath,
  getRun,
  getRunEvents,
  getRunLogs,
  hydrateRun,
  json,
  now,
  one,
  repositoryRoot,
  run as sqlRun,
  updateRun
} from "./db.js";
import { syncFilesystem } from "./inventory.js";
import {
  datasetRow,
  embeddingPreview,
  formatLine,
  formatNetworkRow,
  formatSnapshot,
  getLineRow,
  getNetwork,
  getSnapshotRow,
  inferenceRow,
  lineRows,
  modelRow,
  networkPayload,
  snapshotArtifacts
} from "./query.js";
import { rankSimilarLines } from "./similarity.js";

const ROOT = repositoryRoot();
const WEB_ROOT = resolve(ROOT, "apps/web");
const CONTENT_TYPES = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml"
};
const TERMINAL_STATUSES = new Set(["succeeded", "failed", "cancelled", "orphaned"]);

function headers(contentType = "application/json; charset=utf-8") {
  return {
    "Cache-Control": "no-store",
    "Content-Type": contentType,
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type, Last-Event-ID",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS"
  };
}

function response(body, status = 200) {
  return Response.json(body, { status, headers: headers() });
}

function textResponse(body, status = 200, contentType = "text/plain; charset=utf-8") {
  return new Response(body, { status, headers: headers(contentType) });
}

function errorResponse(error, status = 400) {
  return response({ error: error instanceof Error ? error.message : String(error) }, status);
}

function pathId(value) {
  const decoded = decodeURIComponent(value || "");
  if (!decoded || decoded.includes("/") || decoded.includes("\\") || decoded === "." || decoded === "..") {
    throw new Error("invalid resource identifier");
  }
  return decoded;
}

function parseJsonBody(request) {
  return request.json();
}

function gitCommit(root) {
  try {
    const result = Bun.spawnSync(["git", "rev-parse", "--short", "HEAD"], { cwd: root });
    return new TextDecoder().decode(result.stdout).trim();
  } catch {
    return "unknown";
  }
}

function latestEvaluation(db) {
  const rows = all(db, `SELECT name, value, split, network_id AS networkId, dimensions_json, created_at AS createdAt
    FROM metric_points ORDER BY created_at DESC, id DESC LIMIT 100`);
  const seen = new Set();
  return rows.filter((row) => {
    const key = `${row.name}:${row.split || ""}:${row.networkId || ""}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  }).map((row) => ({
    name: row.name,
    value: Number(row.value),
    split: row.split,
    networkId: row.networkId,
    dimensions: JSON.parse(row.dimensions_json || "{}"),
    createdAt: row.createdAt
  }));
}

function readiness(db) {
  const inventory = {
    feeds: Number(one(db, "SELECT COUNT(*) AS count FROM feed_revisions")?.count || 0),
    snapshots: Number(one(db, "SELECT COUNT(*) AS count FROM snapshots")?.count || 0),
    graphs: Number(one(db, "SELECT COUNT(*) AS count FROM snapshots WHERE graph_path IS NOT NULL")?.count || 0),
    lines: Number(one(db, "SELECT COUNT(*) AS count FROM line_instances")?.count || 0),
    labels: Number(one(db, "SELECT COUNT(*) AS count FROM criticality_labels")?.count || 0),
    models: Number(one(db, "SELECT COUNT(*) AS count FROM model_versions")?.count || 0),
    datasets: Number(one(db, "SELECT COUNT(*) AS count FROM datasets")?.count || 0),
    evaluations: Number(one(db, "SELECT COUNT(*) AS count FROM metric_points")?.count || 0),
    qualityPasses: Number(one(db, "SELECT COUNT(*) AS count FROM quality_checks WHERE status = 'passed'")?.count || 0)
  };
  const networkSnapshotCounts = all(db, "SELECT network_id AS networkId, COUNT(*) AS count FROM snapshots GROUP BY network_id");
  const snapshotPairs = networkSnapshotCounts.reduce((sum, row) => {
    const count = Number(row.count);
    return sum + count * Math.max(0, count - 1) / 2;
  }, 0);
  const labelCoverage = inventory.lines ? inventory.labels / inventory.lines : 0;
  const gates = [
    { id: "feeds-valid", group: "Data", label: "Required GTFS files valid", passed: inventory.feeds > 0, detail: inventory.feeds ? `${inventory.feeds} feed revisions indexed` : "Register a feed revision" },
    { id: "compiled-graph", group: "Data", label: "Compiled graph exists", passed: inventory.graphs > 0, detail: inventory.graphs ? `${inventory.graphs} graph artifact(s)` : "Compile and build at least one graph" },
    { id: "snapshot-pairs", group: "Data", label: "At least two snapshots for identity consistency", passed: snapshotPairs > 0, detail: snapshotPairs ? `${snapshotPairs} possible same-network pair(s)` : "Add a second snapshot of a network" },
    { id: "city-splits", group: "Data", label: "Train, validation, and holdout cities assigned", passed: false, detail: "No frozen dataset split is registered" },
    { id: "positive-pairs", group: "Similarity", label: "Enough same-line positive pairs", passed: snapshotPairs > 0, detail: snapshotPairs ? "Pair generation can begin" : "Requires matched snapshots" },
    { id: "facet-triplets", group: "Similarity", label: "Enough facet-specific triplets", passed: false, detail: "No dataset manifest has triplet counts" },
    { id: "id-invariance", group: "Similarity", label: "ID-renaming invariance test passes", passed: inventory.qualityPasses > 0, detail: inventory.qualityPasses ? "Recorded quality checks include a pass" : "Run the invariance suite" },
    { id: "simulation-coverage", group: "Criticality", label: "Simulation label coverage meets threshold", passed: labelCoverage >= 0.8, detail: `${Math.round(labelCoverage * 100)}% of indexed line instances have labels` },
    { id: "simulation-failures", group: "Criticality", label: "Failed simulation rate below threshold", passed: false, detail: "No simulation run quality report is registered" },
    { id: "baselines", group: "Criticality", label: "Baseline metrics computed", passed: inventory.evaluations > 0, detail: inventory.evaluations ? `${inventory.evaluations} metric point(s)` : "Evaluate a model and baselines" },
    { id: "dataset-frozen", group: "Training", label: "Dataset manifest frozen", passed: inventory.datasets > 0, detail: inventory.datasets ? `${inventory.datasets} dataset version(s)` : "Build a versioned dataset" },
    { id: "smoke-training", group: "Training", label: "Smoke training run passes", passed: inventory.models > 0, detail: inventory.models ? `${inventory.models} model artifact(s)` : "Run a small training job" }
  ];
  const networks = all(db, "SELECT * FROM networks ORDER BY display_name").map((network) => {
    const snapshotCount = Number(one(db, "SELECT COUNT(*) AS count FROM snapshots WHERE network_id = ?", [network.id])?.count || 0);
    const lineCount = Number(one(db, "SELECT COUNT(*) AS count FROM line_instances li JOIN snapshots s ON s.id = li.snapshot_id WHERE s.network_id = ?", [network.id])?.count || 0);
    const labelCount = Number(one(db, "SELECT COUNT(*) AS count FROM criticality_labels cl JOIN snapshots s ON s.id = cl.snapshot_id WHERE s.network_id = ?", [network.id])?.count || 0);
    const inferenceCount = Number(one(db, "SELECT COUNT(*) AS count FROM inference_sets i JOIN snapshots s ON s.id = i.snapshot_id WHERE s.network_id = ?", [network.id])?.count || 0);
    const feedCount = Number(one(db, "SELECT COUNT(*) AS count FROM feed_revisions WHERE network_id = ?", [network.id])?.count || 0);
    const invalid = Number(one(db, "SELECT COUNT(*) AS count FROM snapshots WHERE network_id = ? AND validation_json LIKE '%\"errors\":[%' AND validation_json NOT LIKE '%\"errors\":[]%'")?.count || 0);
    return {
      id: network.id,
      displayName: network.display_name,
      feed: feedCount > 0,
      compile: snapshotCount > 0,
      snapshotPairs: snapshotCount > 1 ? snapshotCount * (snapshotCount - 1) / 2 : 0,
      labels: lineCount ? labelCount / lineCount : 0,
      infer: inferenceCount > 0,
      valid: invalid === 0
    };
  });
  return { inventory, gates, networks, snapshotPairs };
}

function overview(db) {
  const status = readiness(db);
  const activeRow = one(db, "SELECT * FROM runs WHERE status IN ('queued', 'claimed', 'running') ORDER BY created_at LIMIT 1");
  const latestRow = one(db, "SELECT * FROM runs ORDER BY created_at DESC LIMIT 1");
  const evaluations = latestEvaluation(db);
  return {
    project: one(db, "SELECT id, name, description FROM projects WHERE id = 'project-local'") || { id: "project-local", name: "Transit Lab" },
    readiness: {
      passed: status.gates.filter((gate) => gate.passed).length,
      total: status.gates.length,
      gates: status.gates
    },
    activeRun: activeRow ? hydrateRun(activeRow, db) : null,
    latestRun: latestRow ? hydrateRun(latestRow, db) : null,
    latestEvaluation: evaluations,
    corpus: {
      cities: status.networks.length,
      feeds: status.inventory.feeds,
      snapshots: status.inventory.snapshots,
      lineInstances: status.inventory.lines,
      snapshotPairs: status.snapshotPairs,
      labels: status.inventory.labels,
      models: status.inventory.models
    },
    cityReadiness: status.networks,
    provenance: catalog(db)
  };
}

function catalog(db) {
  // Keep the catalog self-contained for the global selectors and the Data
  // view. Returning only network labels here made the feed registry silently
  // disappear from the UI because it needs the feed revisions nested under
  // each network.
  const networks = all(db, "SELECT * FROM networks ORDER BY display_name").map((row) => formatNetworkRow(row, db));
  const snapshots = all(db, "SELECT * FROM snapshots ORDER BY service_date DESC, created_at DESC").map(formatSnapshot);
  const models = all(db, "SELECT * FROM model_versions ORDER BY created_at DESC").map((row) => modelRow(row, db));
  const dates = [...new Set(snapshots.map((snapshot) => snapshot.serviceDate))];
  return { networks, snapshots, models, dates, facets: ["general", "role", "service", "geometry", "resilience"] };
}

function checkRunReferences(db, spec) {
  if (spec.kind === "compile-snapshot") {
    if (!one(db, "SELECT id FROM feed_revisions WHERE id = ?", [spec.feedRevisionId])) throw new Error("feed revision does not exist");
  }
  if (spec.kind === "simulate-criticality" || spec.kind === "infer") {
    if (!one(db, "SELECT id FROM snapshots WHERE id = ?", [spec.snapshotId])) throw new Error("snapshot does not exist");
  }
  if (spec.kind === "build-dataset") {
    const missing = spec.snapshotIds.find((id) => !one(db, "SELECT id FROM snapshots WHERE id = ?", [id]));
    if (missing) throw new Error(`snapshot does not exist: ${missing}`);
  }
  if (spec.kind === "train") {
    if (!one(db, "SELECT id FROM datasets WHERE id = ?", [spec.datasetId])) throw new Error("dataset does not exist");
  }
  if (spec.kind === "evaluate" || spec.kind === "infer") {
    if (!one(db, "SELECT id FROM model_versions WHERE id = ?", [spec.modelId])) throw new Error("model does not exist");
  }
}

function createRun(db, spec) {
  const { spec: normalized, fingerprint } = normalizeRunSpec(spec);
  checkRunReferences(db, normalized);
  const id = `run-${crypto.randomUUID()}`;
  const snapshotId = normalized.snapshotId || null;
  const datasetId = normalized.datasetId || null;
  const modelId = normalized.modelId || null;
  const timestamp = now();
  sqlRun(db, `INSERT INTO runs(id, project_id, kind, status, spec_json, fingerprint, snapshot_id, dataset_id, model_id, git_commit, created_at, updated_at)
    VALUES (?, 'project-local', ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?)`, [
    id,
    normalized.kind,
    json(normalized),
    fingerprint,
    snapshotId,
    datasetId,
    modelId,
    gitCommit(ROOT),
    timestamp,
    timestamp
  ]);
  appendRunEventType(db, id, "run.queued");
  return {
    ...getRun(db, id),
    events: getRunEvents(db, id, -1, 50)
  };
}

async function runDetail(db, id) {
  const run = getRun(db, id);
  if (!run) return null;
  return {
    ...run,
    events: getRunEvents(db, id, -1, 500),
    logs: getRunLogs(db, id),
    artifacts: artifactForRun(db, id)
  };
}

function sseResponse(db, request, runId, after) {
  let timer;
  let closed = false;
  let cursor = Number.isFinite(after) ? after : -1;
  const stream = new ReadableStream({
    start(controller) {
      const close = () => {
        if (closed) return;
        closed = true;
        if (timer) clearInterval(timer);
        try { controller.close(); } catch { /* client disconnected */ }
      };
      const pump = () => {
        if (closed) return;
        const events = getRunEvents(db, runId, cursor, 500);
        for (const event of events) {
          try { validateRunEvent(event); } catch { continue; }
          cursor = event.seq;
          controller.enqueue(`id: ${event.seq}\ndata: ${JSON.stringify(event)}\n\n`);
        }
        const row = one(db, "SELECT status FROM runs WHERE id = ?", [runId]);
        if (!row || (TERMINAL_STATUSES.has(row.status) && events.length === 0)) close();
      };
      controller.enqueue(`: transit-lab run events v${RUN_EVENT_SCHEMA_VERSION}\n\n`);
      pump();
      timer = setInterval(pump, 400);
      request.signal?.addEventListener("abort", close, { once: true });
    },
    cancel() {
      closed = true;
      if (timer) clearInterval(timer);
    }
  });
  return new Response(stream, { headers: headers("text/event-stream; charset=utf-8") });
}

function artifactForRun(db, id) {
  return all(db, "SELECT * FROM artifacts WHERE producing_run_id = ? ORDER BY created_at, id", [id]).map((row) => ({
    id: row.id,
    kind: row.kind,
    fingerprint: row.fingerprint,
    uri: row.uri,
    localPath: row.local_path,
    sizeBytes: Number(row.size_bytes || 0),
    sha256: row.sha256,
    schemaVersion: row.schema_version,
    producingRunId: row.producing_run_id,
    gitCommit: row.git_commit,
    configuration: JSON.parse(row.configuration_json || "{}"),
    files: JSON.parse(row.files_json || "[]"),
    status: row.status,
    metadata: JSON.parse(row.metadata_json || "{}"),
    createdAt: row.created_at,
    supersededBy: row.superseded_by
  }));
}

async function apiRequest(request, url, db, root) {
  if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: headers() });
  const parts = url.pathname.split("/").filter(Boolean).slice(1);
  if (!parts.length) return errorResponse(new Error("API resource is required"), 404);
  const resource = parts[0];

  if (resource === "health") return response(healthPayload(db, root));
  if (resource === "catalog") return response(catalog(db));
  if (resource === "overview") return response(overview(db));
  if (resource === "networks") {
    if (parts.length === 1 && request.method === "GET") return response(all(db, "SELECT * FROM networks ORDER BY display_name").map((row) => formatNetworkRow(row, db)));
    if (parts.length === 3 && parts[2] === "snapshots" && request.method === "GET") {
      const network = getNetwork(db, pathId(parts[1]));
      if (!network) return errorResponse(new Error("network not found"), 404);
      return response(all(db, "SELECT * FROM snapshots WHERE network_id = ? ORDER BY service_date DESC", [network.id]).map(formatSnapshot));
    }
  }
  if (resource === "snapshots") {
    const snapshotId = pathId(parts[1]);
    const snapshot = getSnapshotRow(db, snapshotId);
    if (!snapshot) return errorResponse(new Error("snapshot not found"), 404);
    if (parts.length === 2 && request.method === "GET") {
      return response({ ...formatSnapshot(snapshot), artifacts: snapshotArtifacts(db, snapshotId) });
    }
    if (parts.length === 3 && parts[2] === "lines" && request.method === "GET") {
      return response(lineRows(db, snapshotId).map(formatLine));
    }
    if (parts.length === 3 && parts[2] === "network" && request.method === "GET") {
      const loaded = await import("./inventory.js").then(({ loadSnapshotNetwork }) => loadSnapshotNetwork(db, root, snapshotId));
      if (!loaded) return errorResponse(new Error("compiled network artifact is unavailable"), 404);
      return response(networkPayload(db, snapshot, loaded.network));
    }
  }
  if (resource === "lines" && parts[1] && request.method === "GET") {
    const line = getLineRow(db, pathId(parts[1]));
    if (!line) return errorResponse(new Error("line instance not found"), 404);
    const snapshot = getSnapshotRow(db, line.snapshot_id);
    const labels = one(db, "SELECT values_json, source_artifact_id FROM criticality_labels WHERE snapshot_id = ? AND line_index = ?", [line.snapshot_id, line.line_index]);
    return response({
      line: formatLine(line),
      snapshot: formatSnapshot(snapshot),
      label: labels ? { ...JSON.parse(labels.values_json), sourceArtifactId: labels.source_artifact_id } : null,
      provenance: {
        snapshotId: line.snapshot_id,
        modelId: one(db, "SELECT model_id FROM inference_sets WHERE snapshot_id = ? ORDER BY created_at DESC LIMIT 1", [line.snapshot_id])?.model_id || null,
        featureSchema: "compiled-network-line-features"
      }
    });
  }
  if (resource === "runs") {
    if (parts.length === 1 && request.method === "GET") {
      const limit = Math.max(1, Math.min(200, Number(url.searchParams.get("limit") || 50)));
      return response(all(db, "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", [limit]).map((row) => hydrateRun(row, db)));
    }
    if (parts.length === 1 && request.method === "POST") {
      try {
        return response(createRun(db, await parseJsonBody(request)), 201);
      } catch (error) {
        return errorResponse(error, 422);
      }
    }
      if (parts.length >= 2) {
      const runId = pathId(parts[1]);
      if (!getRun(db, runId)) return errorResponse(new Error("run not found"), 404);
      if (parts.length === 3 && parts[2] === "events" && request.method === "GET") {
        const headerValue = request.headers.get("last-event-id");
        const queryValue = url.searchParams.get("after");
        const cursorValue = headerValue ?? queryValue;
        const cursor = cursorValue === null ? -1 : Number(cursorValue);
        return sseResponse(db, request, runId, Number.isInteger(cursor) ? cursor : -1);
      }
      if (parts.length === 3 && parts[2] === "logs" && request.method === "GET") return response(getRunLogs(db, runId));
      if (parts.length === 3 && parts[2] === "cancel" && request.method === "POST") {
        const row = one(db, "SELECT status FROM runs WHERE id = ?", [runId]);
        if (!row || TERMINAL_STATUSES.has(row.status)) return response(getRun(db, runId));
        updateRun(db, runId, { cancelRequested: true });
        appendRunEventType(db, runId, "warning", { code: "cancel_requested", message: "Cancellation requested by the user." });
        return response(getRun(db, runId));
      }
      if (parts.length === 2 && request.method === "GET") return response(await runDetail(db, runId));
    }
  }
  if (resource === "datasets" && parts[1] && request.method === "GET") {
    const row = one(db, "SELECT * FROM datasets WHERE id = ?", [pathId(parts[1])]);
    return row ? response(datasetRow(row)) : errorResponse(new Error("dataset not found"), 404);
  }
  if (resource === "models") {
    if (parts.length === 1 && request.method === "GET") return response(all(db, "SELECT * FROM model_versions ORDER BY created_at DESC").map((row) => modelRow(row, db)));
    if (parts[1] && request.method === "GET") {
      const row = one(db, "SELECT * FROM model_versions WHERE id = ?", [pathId(parts[1])]);
      if (!row) return errorResponse(new Error("model not found"), 404);
      if (parts[2] === "evaluation") return response({ model: modelRow(row, db), metrics: all(db, "SELECT * FROM metric_points WHERE model_id = ? ORDER BY created_at DESC", [row.id]) });
      return response(modelRow(row, db));
    }
  }
  if (resource === "inference" && parts[1]) {
    const inference = one(db, "SELECT * FROM inference_sets WHERE id = ?", [pathId(parts[1])]);
    if (!inference) return errorResponse(new Error("inference set not found"), 404);
    if (parts[2] === "criticality") {
      const rows = all(db, `SELECT cp.*, li.display_name, li.line_index, li.snapshot_id
        FROM criticality_predictions cp JOIN line_instances li ON li.id = cp.line_instance_id
        WHERE cp.inference_id = ? ORDER BY cp.primary_score DESC`, [inference.id]);
      return response({ inference: inferenceRow(inference), predictions: rows.map((row) => ({ lineInstanceId: row.line_instance_id, lineIndex: Number(row.line_index), displayName: row.display_name, values: JSON.parse(row.values_json || "{}"), primaryScore: row.primary_score, uncertainty: row.uncertainty })) });
    }
    if (parts[2] === "embeddings") return response(embeddingPreview(db, inference.snapshot_id, url.searchParams.get("facet") || "general"));
    if (parts.length === 2) return response(inferenceRow(inference));
  }
  if (resource === "similarity" && parts[1] === "search" && request.method === "POST") {
    try {
      const body = await parseJsonBody(request);
      const result = rankSimilarLines(db, {
        querySnapshotId: pathId(body.querySnapshotId),
        queryLineId: body.queryLineId ? pathId(body.queryLineId) : undefined,
        queryLineIndex: body.queryLineIndex,
        candidateSnapshotId: pathId(body.candidateSnapshotId),
        profile: body.profile || "general",
        weights: body.weights || {},
        topK: body.topK || 10
      });
      return response(result);
    } catch (error) {
      return errorResponse(error, 422);
    }
  }
  if (resource === "embeddings" && request.method === "GET") {
    const snapshotId = pathId(url.searchParams.get("snapshotId"));
    return response(embeddingPreview(db, snapshotId, url.searchParams.get("facet") || "general"));
  }
  if (resource === "pipeline" && request.method === "GET") {
    const nodes = ["Feed", "Validate", "Compile canonical snapshot", "Extract features", "Simulate line removals", "Build dataset", "Train", "Evaluate", "Infer", "Similarity search"];
    return response({ nodes: nodes.map((label, index) => ({ id: `pipeline-${index}`, label, status: index < 3 ? "ready" : "pending" })), edges: nodes.slice(0, -1).map((_, index) => ({ source: `pipeline-${index}`, target: `pipeline-${index + 1}` })) });
  }
  if (resource === "annotation-tasks" && parts[1] === "next" && request.method === "GET") {
    return response({ task: null, message: "No human benchmark task has been created yet." });
  }
  if (resource === "annotations" && request.method === "POST") {
    try {
      const body = await parseJsonBody(request);
      const allowedFacets = new Set(["role", "service", "geometry", "resilience"]);
      if (!allowedFacets.has(body.facet) || !["a", "b", "tie"].includes(body.choice)) throw new Error("annotation facet or choice is invalid");
      sqlRun(db, `INSERT INTO annotations(anchor_line_instance_id, candidate_a_line_instance_id, candidate_b_line_instance_id, facet, choice, confidence, notes, annotator, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`, [body.anchorLineInstanceId, body.candidateALineInstanceId, body.candidateBLineInstanceId, body.facet, body.choice, body.confidence || null, body.notes || "", body.annotator || "local", now()]);
      return response({ ok: true }, 201);
    } catch (error) {
      return errorResponse(error, 422);
    }
  }
  return errorResponse(new Error("API route not found"), 404);
}

async function staticRequest(request, url, root) {
  let pathname;
  try { pathname = decodeURIComponent(url.pathname); } catch { return textResponse("Not found", 404); }
  if (pathname === "/") pathname = "/index.html";
  if (pathname.includes("..")) return textResponse("Not found", 404);
  const filePath = resolve(WEB_ROOT, pathname.slice(1));
  const webBase = resolve(WEB_ROOT);
  if (!(filePath === webBase || filePath.startsWith(`${webBase}${sep}`))) return textResponse("Not found", 404);
  const file = Bun.file(filePath);
  if (!(await file.exists())) return textResponse("Not found", 404);
  return new Response(file, { headers: headers(CONTENT_TYPES[extname(filePath)] || "application/octet-stream") });
}

function healthPayload(db, root) {
  return { ok: true, service: "transit-lab-control-plane", database: databasePath(root) };
}

export async function handleRequest(request, { db, root = ROOT } = {}) {
  const url = new URL(request.url);
  try {
    if (url.pathname === "/health") return response(healthPayload(db, root));
    if (url.pathname.startsWith("/api/")) return await apiRequest(request, url, db, root);
    return await staticRequest(request, url, root);
  } catch (error) {
    console.error(error);
    return errorResponse(error, 500);
  }
}

/** Build a request handler for tests, embedded servers, and the Bun listener. */
export function createApiHandler({ db, root = ROOT }) {
  return (request) => handleRequest(request, { db, root });
}

export async function startServer({ port = Number(process.env.PORT || 3000), root = ROOT, sync = true } = {}) {
  await mkdir(dataRoot(root), { recursive: true });
  const db = createDatabase(root);
  if (sync) await syncFilesystem(db, root);
  const server = Bun.serve({ port, fetch: (request) => handleRequest(request, { db, root }) });
  console.log(`Transit Lab control plane listening on http://localhost:${server.port}`);
  console.log(`Database: ${databasePath(root)}`);
  console.log("Run the worker separately with: bun run apps/worker/src/worker.js");
  return { server, db };
}

if (import.meta.main) startServer().catch((error) => { console.error(error); process.exit(1); });
