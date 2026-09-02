import {
  INFERENCE_RESULT_SCHEMA_VERSION,
  RUN_EVENT_SCHEMA_VERSION,
  SIMILARITY_FACETS,
  assertSafeId,
  validateRunSpec
} from "../../../packages/contracts/src/index.ts";
import {
  all,
  appendRunEventType,
  createRun,
  getRun,
  getRunEvents,
  getRunLogs,
  json,
  listRuns,
  now,
  one,
  parseJson,
  requestRunCancellation,
  run
} from "../../../packages/control-store/src/database.ts";
import {
  artifactDependencies,
  findArtifact,
  findSnapshot,
  inventorySummary,
  loadSnapshotNetwork,
  syncFilesystem
} from "../../../packages/control-store/src/inventory.ts";
import { loadSimilarityResult } from "../../../packages/control-store/src/similarity.ts";

const DEFAULT_METRIC_NAMES = [
  "accessibility_auc_loss",
  "unreachable_share",
  "mean_delay_reachable_seconds",
  "p95_delay_reachable_seconds",
  "mean_extra_transfers",
  "stations_losing_all_service_share"
];

export class ApiError extends Error {
  status: number;

  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

function sendJson(value, status = 200) {
  return Response.json(value, {
    status,
    headers: { "Cache-Control": "no-store" }
  });
}

function fail(status, message) {
  return sendJson({ error: message }, status);
}

function numeric(value, fallback = 0) {
  const result = Number(value);
  return Number.isFinite(result) ? result : fallback;
}

function snapshotRow(row) {
  return {
    id: row.id,
    networkId: row.network_id,
    feedRevisionId: row.feed_revision_id,
    serviceDate: row.service_date,
    serviceProfile: row.service_profile,
    status: row.status,
    fingerprint: row.fingerprint,
    compilerVersion: row.compiler_version,
    compilerCommit: row.compiler_commit,
    sourceName: row.source_name,
    geographicalScope: row.geographical_scope,
    counts: parseJson(row.counts_json),
    validation: parseJson(row.validation_json),
    manifestPath: row.manifest_path,
    networkPath: row.network_path,
    graphPath: row.graph_path,
    createdAt: row.created_at,
    updatedAt: row.updated_at
  };
}

function artifactRow(row) {
  return {
    id: row.id,
    kind: row.kind,
    fingerprint: row.fingerprint,
    uri: row.uri,
    localPath: row.local_path,
    sizeBytes: numeric(row.size_bytes),
    sha256: row.sha256,
    schemaVersion: row.schema_version,
    producingRunId: row.producing_run_id,
    gitCommit: row.git_commit,
    configuration: parseJson(row.configuration_json),
    files: parseJson(row.files_json, []),
    status: row.status,
    metadata: parseJson(row.metadata_json),
    createdAt: row.created_at,
    supersededBy: row.superseded_by
  };
}

function datasetRow(row) {
  return {
    id: row.id,
    fingerprint: row.fingerprint,
    status: row.status,
    manifestPath: row.manifest_path,
    featureSchema: row.feature_schema,
    snapshotIds: parseJson(row.snapshot_ids_json, []),
    split: parseJson(row.split_json),
    objectives: parseJson(row.objective_counts_json),
    quality: parseJson(row.quality_json),
    createdAt: row.created_at,
    updatedAt: row.updated_at
  };
}

function modelRow(row) {
  return {
    id: row.id,
    version: row.version,
    fingerprint: row.fingerprint,
    status: row.status,
    architecture: parseJson(row.architecture_json),
    datasetId: row.dataset_id,
    trainingRunId: row.training_run_id,
    checkpointArtifactId: row.checkpoint_artifact_id,
    embeddingDimensions: parseJson(row.embedding_dimensions_json),
    supportedHeads: parseJson(row.supported_heads_json, []),
    evaluation: parseJson(row.evaluation_json),
    createdAt: row.created_at
  };
}

function networkForClient(network) {
  return {
    snapshot_id: network.snapshot_id,
    manifest: network.manifest,
    stations: network.stations || [],
    lines: network.lines || [],
    patterns: (network.patterns || []).map((pattern) => ({
      index: pattern.index,
      signature: pattern.signature
    })),
    transit_edges: (network.transit_edges || []).map(({ departures_by_bin, median_runtime_by_bin, ...edge }) => edge),
    transfers: network.transfers || [],
    interchanges: network.interchanges || []
  };
}

function listSnapshots(db) {
  return all(db, "SELECT * FROM snapshots ORDER BY source_name, service_date, id").map(snapshotRow);
}

function listArtifacts(db, limit = 200) {
  return all(db, "SELECT * FROM artifacts ORDER BY created_at DESC, id DESC LIMIT ?", [
    Math.max(1, Math.min(1_000, numeric(limit, 200)))
  ]).map(artifactRow);
}

function getInference(db, inferenceId, snapshotId) {
  const row = inferenceId
    ? one(db, "SELECT * FROM inference_sets WHERE id = ?", [inferenceId])
    : one(db, "SELECT * FROM inference_sets WHERE snapshot_id = ? ORDER BY created_at DESC, id DESC LIMIT 1", [snapshotId]);
  if (!row) return null;
  const config = parseJson(row.config_json);
  const metricNames = Array.isArray(config.metricNames) && config.metricNames.length
    ? config.metricNames
    : DEFAULT_METRIC_NAMES;
  const rows = all(db, `SELECT cp.*, li.line_index, li.display_name, li.canonical_id, li.mode
    FROM criticality_predictions cp
    JOIN line_instances li ON li.id = cp.line_instance_id
    WHERE cp.inference_id = ? ORDER BY cp.primary_score DESC, li.display_name`, [row.id]);
  const predictions = rows.map((prediction) => {
    const values = parseJson(prediction.values_json);
    const percentiles = values.percentiles && typeof values.percentiles === "object" ? values.percentiles : null;
    return {
      line: Number(prediction.line_index),
      lineName: prediction.display_name,
      metrics: metricNames.map((name) => numeric(values[name] ?? values[name.replaceAll("_auc_loss", "_loss")])),
      metricPercentiles: percentiles
        ? metricNames.map((name) => numeric(percentiles[name], 0))
        : undefined,
      structuralUniqueness: numeric(values.structuralUniqueness ?? values.structural_uniqueness),
      uncertainty: prediction.uncertainty === null ? 0 : numeric(prediction.uncertainty)
    };
  });
  return {
    schemaVersion: INFERENCE_RESULT_SCHEMA_VERSION,
    inferenceId: row.id,
    modelId: row.model_id,
    snapshotId: row.snapshot_id,
    metricNames,
    predictions,
    status: row.status,
    config,
    createdAt: row.created_at
  };
}

function listEvaluations(db) {
  return all(db, `SELECT se.*, mv.version AS model_version, d.id AS dataset_id_value
    FROM similarity_evaluations se
    LEFT JOIN model_versions mv ON mv.id = se.model_id
    LEFT JOIN datasets d ON d.id = se.dataset_id
    ORDER BY se.created_at DESC, se.id DESC`).map((row) => ({
    id: row.id,
    modelId: row.model_id,
    modelVersion: row.model_version,
    datasetId: row.dataset_id_value,
    facet: row.facet,
    metricName: row.metric_name,
    value: numeric(row.value),
    split: row.split,
    createdAt: row.created_at
  }));
}

function listEmbeddings(db) {
  return listArtifacts(db, 500).filter((artifact) => /embedding|projection/i.test(artifact.kind));
}

function parseWeights(url) {
  return Object.fromEntries([
    ["role", url.searchParams.get("roleWeight")],
    ["service", url.searchParams.get("serviceWeight")],
    ["geometry", url.searchParams.get("geometryWeight")],
    ["resilience", url.searchParams.get("resilienceWeight")]
  ].filter(([, value]) => value !== null).map(([key, value]) => [key, Number(value)]));
}

function eventStream(request, db, runId, after) {
  const encoder = new TextEncoder();
  let cursor = Number.isInteger(after) ? after : -1;
  let interval = null;
  let closed = false;
  let controller;
  const close = () => {
    if (closed) return;
    closed = true;
    if (interval) clearInterval(interval);
    try { controller?.close(); } catch { /* client disconnected */ }
  };
  const stream = new ReadableStream({
    start(nextController) {
      controller = nextController;
      const pump = () => {
        if (closed) return;
        for (const event of getRunEvents(db, runId, cursor, 500)) {
          cursor = event.seq;
          controller.enqueue(encoder.encode(`id: ${event.seq}\ndata: ${JSON.stringify(event)}\n\n`));
        }
        const current = getRun(db, runId);
        if (current && ["succeeded", "failed", "cancelled", "orphaned"].includes(current.status) &&
            getRunEvents(db, runId, cursor, 1).length === 0) close();
      };
      pump();
      interval = setInterval(pump, 500);
      request.signal.addEventListener("abort", close, { once: true });
    },
    cancel: close
  });
  return new Response(stream, {
    headers: {
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
      "Content-Type": "text/event-stream; charset=utf-8",
      "X-Accel-Buffering": "no"
    }
  });
}

async function refreshInventory(db, root) {
  try {
    return await syncFilesystem(db, root);
  } catch (error) {
    throw new ApiError(500, `artifact inventory failed: ${error instanceof Error ? error.message : String(error)}`);
  }
}

export function createApiHandler({ db, root }) {
  return async function handleApi(request) {
    const url = new URL(request.url);
    const pathname = url.pathname;
    try {
      if (request.method === "GET" && pathname === "/api/health") {
        return sendJson({ ok: true, service: "transit-lab-control-api", eventSchemaVersion: RUN_EVENT_SCHEMA_VERSION });
      }
      if (request.method === "GET" && pathname === "/api/overview") {
        if (url.searchParams.get("refresh") === "1") await refreshInventory(db, root);
        return sendJson({
          projectId: "project-local",
          counts: inventorySummary(db),
          snapshots: listSnapshots(db).slice(0, 8),
          recentRuns: listRuns(db, 8)
        });
      }
      if (request.method === "POST" && pathname === "/api/inventory/refresh") {
        return sendJson(await refreshInventory(db, root));
      }
      if (request.method === "GET" && pathname === "/api/snapshots") {
        return sendJson(listSnapshots(db));
      }
      const snapshotMatch = pathname.match(/^\/api\/snapshots\/([^/]+)$/);
      if (request.method === "GET" && snapshotMatch) {
        const snapshotId = decodeURIComponent(snapshotMatch[1]);
        const snapshot = findSnapshot(db, snapshotId);
        if (!snapshot) throw new ApiError(404, "snapshot not found");
        return sendJson(snapshot);
      }
      const networkMatch = pathname.match(/^\/api\/snapshots\/([^/]+)\/network$/);
      if (request.method === "GET" && networkMatch) {
        const snapshotId = decodeURIComponent(networkMatch[1]);
        const loaded = await loadSnapshotNetwork(db, root, snapshotId);
        if (!loaded) throw new ApiError(404, "snapshot network not found");
        return sendJson(networkForClient(loaded.network));
      }
      if (request.method === "GET" && pathname === "/api/artifacts") {
        return sendJson(listArtifacts(db, numeric(url.searchParams.get("limit"), 200)));
      }
      const artifactMatch = pathname.match(/^\/api\/artifacts\/([^/]+)$/);
      if (request.method === "GET" && artifactMatch) {
        const artifact = findArtifact(db, decodeURIComponent(artifactMatch[1]));
        if (!artifact) throw new ApiError(404, "artifact not found");
        return sendJson({ ...artifact, inputs: artifactDependencies(db, artifact.id) });
      }
      if (request.method === "GET" && pathname === "/api/datasets") {
        return sendJson(all(db, "SELECT * FROM datasets ORDER BY updated_at DESC, id DESC").map(datasetRow));
      }
      if (request.method === "GET" && pathname === "/api/models") {
        return sendJson(all(db, "SELECT * FROM model_versions ORDER BY created_at DESC, id DESC").map(modelRow));
      }
      if (request.method === "GET" && pathname === "/api/inferences") {
        return sendJson(all(db, "SELECT * FROM inference_sets ORDER BY created_at DESC, id DESC").map((row) => ({
          id: row.id,
          fingerprint: row.fingerprint,
          modelId: row.model_id,
          snapshotId: row.snapshot_id,
          status: row.status,
          config: parseJson(row.config_json),
          createdAt: row.created_at
        })));
      }
      if (request.method === "GET" && pathname === "/api/criticality") {
        const result = getInference(db, url.searchParams.get("inferenceId"), url.searchParams.get("snapshotId"));
        if (!result) throw new ApiError(404, "criticality inference not found");
        return sendJson(result);
      }
      if (request.method === "GET" && pathname === "/api/embeddings") {
        return sendJson(listEmbeddings(db));
      }
      if (request.method === "GET" && pathname === "/api/evaluations") {
        return sendJson(listEvaluations(db));
      }
      if (request.method === "GET" && pathname === "/api/similarity") {
        const querySnapshotId = url.searchParams.get("querySnapshotId");
        const candidateSnapshotId = url.searchParams.get("candidateSnapshotId");
        if (!querySnapshotId || !candidateSnapshotId) throw new ApiError(400, "querySnapshotId and candidateSnapshotId are required");
        const profile = url.searchParams.get("profile") || "general";
        if (!SIMILARITY_FACETS.includes(profile) && profile !== "network-role") {
          throw new ApiError(400, `unsupported similarity profile: ${profile}`);
        }
        const result = await loadSimilarityResult({
          db,
          root,
          querySnapshotId,
          queryLineId: url.searchParams.get("queryLineId"),
          queryLineIndex: url.searchParams.get("queryLineIndex"),
          candidateSnapshotId,
          profile,
          weights: parseWeights(url),
          topK: numeric(url.searchParams.get("topK"), 10)
        });
        if (!result) throw new ApiError(404, "no Rust-produced similarity result matches this query");
        return sendJson(result);
      }
      if (request.method === "GET" && pathname === "/api/runs") {
        return sendJson(listRuns(db, numeric(url.searchParams.get("limit"), 100)));
      }
      if (request.method === "POST" && pathname === "/api/runs") {
        const body = await request.json().catch(() => null);
        if (!body || typeof body !== "object" || Array.isArray(body)) throw new ApiError(400, "request body must be a JSON object");
        const spec = body.spec || body;
        validateRunSpec(spec);
        return sendJson(createRun(db, spec, "project-local", root), 202);
      }
      const runEventsMatch = pathname.match(/^\/api\/runs\/([^/]+)\/events$/);
      if (request.method === "GET" && runEventsMatch) {
        const runId = decodeURIComponent(runEventsMatch[1]);
        if (!getRun(db, runId)) throw new ApiError(404, "run not found");
        const lastEventId = Number(url.searchParams.get("after") ?? request.headers.get("Last-Event-ID") ?? -1);
        return eventStream(request, db, runId, Number.isInteger(lastEventId) ? lastEventId : -1);
      }
      const runMatch = pathname.match(/^\/api\/runs\/([^/]+)$/);
      if (request.method === "GET" && runMatch) {
        const current = getRun(db, decodeURIComponent(runMatch[1]));
        if (!current) throw new ApiError(404, "run not found");
        current.logs = getRunLogs(db, current.id);
        current.events = getRunEvents(db, current.id);
        return sendJson(current);
      }
      const cancelMatch = pathname.match(/^\/api\/runs\/([^/]+)\/cancel$/);
      if (request.method === "POST" && cancelMatch) {
        const current = requestRunCancellation(db, decodeURIComponent(cancelMatch[1]));
        if (!current) throw new ApiError(404, "run not found");
        if (current.status !== "cancelled") appendRunEventType(db, current.id, "warning", { code: "cancel-requested", message: "Cancellation requested; the worker will stop at its next safe boundary." });
        return sendJson(getRun(db, current.id));
      }
      if (request.method === "GET" && pathname === "/api/views") {
        return sendJson(all(db, "SELECT id, name, spec_json, created_at, updated_at FROM saved_views ORDER BY updated_at DESC").map((row) => ({
          id: row.id,
          name: row.name,
          spec: parseJson(row.spec_json),
          createdAt: row.created_at,
          updatedAt: row.updated_at
        })));
      }
      if (request.method === "POST" && pathname === "/api/views") {
        const body = await request.json().catch(() => null);
        if (!body || typeof body !== "object" || Array.isArray(body) || typeof body.name !== "string" || !body.name.trim()) {
          throw new ApiError(400, "a view name and object spec are required");
        }
        if (!body.spec || typeof body.spec !== "object" || Array.isArray(body.spec)) throw new ApiError(400, "view spec must be an object");
        const id = body.id ? assertSafeId(body.id, "id") : `view-${crypto.randomUUID()}`;
        run(db, `INSERT INTO saved_views(id, name, spec_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?)
          ON CONFLICT(id) DO UPDATE SET name = excluded.name, spec_json = excluded.spec_json, updated_at = excluded.updated_at`, [id, body.name.trim(), json(body.spec), now(), now()]);
        return sendJson({ id, name: body.name.trim(), spec: body.spec }, 201);
      }
      return fail(404, "API route not found");
    } catch (error) {
      if (error instanceof ApiError) return fail(error.status, error.message);
      if (error instanceof SyntaxError) return fail(400, "request body must be valid JSON");
      return fail(400, error instanceof Error ? error.message : String(error));
    }
  };
}
