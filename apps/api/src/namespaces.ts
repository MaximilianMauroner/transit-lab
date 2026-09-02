import {
  getPublication,
  listPublications,
  publishPublication,
  unpublishPublication
} from "../../../packages/control-store/src/database.ts";
import { loadPublicationBundle } from "../../../packages/control-store/src/publication-bundle.ts";
import { createApiHandler } from "./routes.ts";

const PUBLIC_PREFIX = "/api/public";
const CONTROL_PREFIX = "/api/control";
const PUBLIC_ORIGINS = process.env.TRANSIT_LAB_PUBLIC_ORIGIN || "*";

function json(value, status = 200, publicResponse = false) {
  const headers = new Headers({
    "Cache-Control": "no-store",
    "Content-Type": "application/json; charset=utf-8"
  });
  if (publicResponse) headers.set("Access-Control-Allow-Origin", PUBLIC_ORIGINS);
  return new Response(JSON.stringify(value), { status, headers });
}

function error(message, status = 400, publicResponse = false) {
  return json({ error: message }, status, publicResponse);
}

function pathRequest(request, pathname) {
  const url = new URL(request.url);
  url.pathname = pathname;
  return new Request(url, request);
}

function bundleEntry(bundle, name) {
  const path = bundle.manifest.entries?.[name];
  return typeof path === "string" ? bundle.data[path] : undefined;
}

function bundleNetwork(bundle, snapshotId) {
  const path = bundle.manifest.entries?.networks?.[snapshotId];
  return typeof path === "string" ? bundle.data[path] : undefined;
}

function publicationSummary(bundle) {
  const manifest = bundle.manifest;
  return {
    id: manifest.publicationId,
    slug: manifest.slug,
    title: manifest.title,
    status: "published",
    snapshotIds: manifest.snapshotIds,
    modelIds: manifest.modelIds,
    createdAt: manifest.createdAt,
    metadata: manifest.metadata || {}
  };
}

function loadPublishedBundles(root, db) {
  return listPublications(db).map((publication) => {
    try {
      return loadPublicationBundle(root, publication);
    } catch (cause) {
      throw new Error(`published bundle ${publication.slug} is unavailable: ${cause instanceof Error ? cause.message : String(cause)}`);
    }
  });
}

function findBundleForSnapshot(bundles, snapshotId) {
  return bundles.find((bundle) => bundle.manifest.snapshotIds.includes(String(snapshotId)));
}

function findCriticality(bundles, snapshotId, inferenceId) {
  for (const bundle of bundles) {
    const values = bundleEntry(bundle, "criticality")?.results || {};
    for (const result of Object.values(values) as Array<Record<string, any>>) {
      if (inferenceId && result.inferenceId === inferenceId) return result;
      if (!inferenceId && result.snapshotId === snapshotId) return result;
    }
  }
  return null;
}

function similarityMatches(bundles, querySnapshotId, candidateSnapshotId, queryLineIndex, queryLineId, profile) {
  for (const bundle of bundles) {
    const values = bundleEntry(bundle, "similarity");
    if (!Array.isArray(values)) continue;
    for (const result of values) {
      const query = result.query || {};
      if (String(query.snapshotId ?? query.snapshot_id ?? query.snapshot) !== String(querySnapshotId)) continue;
      if (String(result.candidateSnapshotId ?? result.candidate_snapshot_id ?? result.candidateSnapshot ?? candidateSnapshotId) !== String(candidateSnapshotId)) continue;
      if (result.profile && result.profile !== profile && !(profile === "network-role" && result.profile === "role")) continue;
      const lineMatches = queryLineId
        ? String(query.lineInstanceId ?? query.lineId ?? "") === String(queryLineId)
        : queryLineIndex === null || queryLineIndex === undefined || Number(query.lineIndex ?? query.line) === Number(queryLineIndex);
      if (lineMatches) return { ...result, artifactId: result.artifactId || undefined };
    }
  }
  return null;
}

async function publicHandler(request, db, root) {
  const url = new URL(request.url);
  const suffix = url.pathname.slice(PUBLIC_PREFIX.length) || "/";
  if (request.method !== "GET") return error("public API is read-only", 405, true);

  let bundles;
  try {
    bundles = loadPublishedBundles(root, db);
  } catch (cause) {
    return error(cause instanceof Error ? cause.message : String(cause), 503, true);
  }

  if (suffix === "/catalog") {
    return json({
      schemaVersion: 1,
      publications: bundles.map(publicationSummary)
    }, 200, true);
  }

  const publicationMatch = suffix.match(/^\/publications\/([^/]+)$/);
  if (publicationMatch) {
    const identifier = decodeURIComponent(publicationMatch[1]);
    const bundle = bundles.find((candidate) => candidate.manifest.publicationId === identifier || candidate.manifest.slug === identifier);
    return bundle ? json(publicationSummary(bundle), 200, true) : error("publication not found", 404, true);
  }

  if (!bundles.length) return error("no published results are available", 404, true);

  if (suffix === "/overview") {
    const primary = bundleEntry(bundles[0], "overview") || { projectId: "project-local", counts: {} };
    const snapshots = bundles.flatMap((bundle) => bundleEntry(bundle, "snapshots") || []);
    return json({
      projectId: primary.projectId,
      publications: bundles.map(publicationSummary),
      snapshots,
      counts: {
        ...(primary.counts || {}),
        publications: bundles.length,
        snapshots: snapshots.length,
        models: new Set(bundles.flatMap((bundle) => bundle.manifest.modelIds)).size
      }
    }, 200, true);
  }

  if (suffix === "/snapshots") {
    const seen = new Set();
    const snapshots = bundles.flatMap((bundle) => bundleEntry(bundle, "snapshots") || [])
      .filter((snapshot) => !seen.has(snapshot.id) && seen.add(snapshot.id));
    return json(snapshots, 200, true);
  }

  const snapshotMatch = suffix.match(/^\/snapshots\/([^/]+)(\/network)?$/);
  if (snapshotMatch) {
    const snapshotId = decodeURIComponent(snapshotMatch[1]);
    const bundle = findBundleForSnapshot(bundles, snapshotId);
    if (!bundle) return error("snapshot is not published", 404, true);
    if (snapshotMatch[2]) {
      const network = bundleNetwork(bundle, snapshotId);
      return network ? json(network, 200, true) : error("published snapshot network is unavailable", 404, true);
    }
    const snapshot = (bundleEntry(bundle, "snapshots") || []).find((value) => value.id === snapshotId);
    return snapshot ? json(snapshot, 200, true) : error("snapshot is not published", 404, true);
  }

  if (suffix === "/criticality") {
    const result = findCriticality(bundles, url.searchParams.get("snapshotId"), url.searchParams.get("inferenceId"));
    if (!result) return error("criticality result is not published", 404, true);
    return json(result, 200, true);
  }

  if (suffix === "/embeddings") {
    return json(bundles.flatMap((bundle) => bundleEntry(bundle, "embeddings") || []), 200, true);
  }

  if (suffix === "/evaluations") {
    return json(bundles.flatMap((bundle) => bundleEntry(bundle, "evaluations") || []), 200, true);
  }

  if (suffix === "/similarity") {
    const querySnapshotId = url.searchParams.get("querySnapshotId");
    const candidateSnapshotId = url.searchParams.get("candidateSnapshotId");
    const profile = url.searchParams.get("profile") || "general";
    const result = similarityMatches(
      bundles,
      querySnapshotId,
      candidateSnapshotId,
      url.searchParams.get("queryLineIndex"),
      url.searchParams.get("queryLineId"),
      profile
    );
    return result ? json(result, 200, true) : error("no published similarity result matches this query", 404, true);
  }

  return error("public API route not found", 404, true);
}

async function controlPublication(request, db, root) {
  const url = new URL(request.url);
  const suffix = url.pathname.slice(CONTROL_PREFIX.length) || "/";
  if (suffix === "/publications" && request.method === "GET") return json(listPublications(db));
  if (suffix === "/publications" && request.method === "POST") {
    const body = await request.json().catch(() => null);
    if (!body || typeof body !== "object" || Array.isArray(body)) return error("publication body must be an object");
    try {
      return json(publishPublication(db, body, root), 201);
    } catch (cause) {
      return error(cause instanceof Error ? cause.message : String(cause));
    }
  }
  const match = suffix.match(/^\/publications\/([^/]+)$/);
  if (match && request.method === "DELETE") return json(unpublishPublication(db, decodeURIComponent(match[1])));
  return null;
}

export function createNamespacedApiHandler({ db, root, allowLegacy = true }) {
  const legacy = createApiHandler({ db, root });
  return async function handleApi(request) {
    const url = new URL(request.url);
    if (url.pathname.startsWith(PUBLIC_PREFIX)) return publicHandler(request, db, root);
    if (url.pathname.startsWith(CONTROL_PREFIX)) {
      const publicationResponse = await controlPublication(request, db, root);
      if (publicationResponse) return publicationResponse;
      const suffix = url.pathname.slice(CONTROL_PREFIX.length) || "/";
      return legacy(pathRequest(request, `/api${suffix}`));
    }
    if (allowLegacy && (url.pathname === "/api" || url.pathname.startsWith("/api/"))) return legacy(request);
    if (url.pathname === "/health") return legacy(pathRequest(request, "/api/health"));
    return error("API route not found", 404);
  };
}
