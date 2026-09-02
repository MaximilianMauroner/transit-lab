import { extname, resolve, sep } from "node:path";
import { createApiHandler } from "./api/routes.js";
import {
  createDatabase,
  repositoryRoot
} from "./database/db.js";
import { syncFilesystem } from "./artifacts/inventory.js";

const CONTENT_TYPES = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml"
};

function inside(base, candidate) {
  const root = resolve(base);
  const path = resolve(candidate);
  return path === root || path.startsWith(`${root}${sep}`);
}

function staticResponse(file, contentType) {
  return new Response(file, {
    headers: {
      "Cache-Control": "no-store",
      "Content-Type": contentType || "application/octet-stream"
    }
  });
}

async function serveStatic(request, publicRoot, clientRoot) {
  if (request.method !== "GET" && request.method !== "HEAD") return new Response("Method not allowed", { status: 405 });
  const url = new URL(request.url);
  let pathname;
  try {
    pathname = decodeURIComponent(url.pathname);
  } catch {
    return new Response("Not found", { status: 404 });
  }
  const isClientModule = pathname === "/client" || pathname.startsWith("/client/");
  const base = isClientModule ? clientRoot : publicRoot;
  const requested = isClientModule
    ? pathname.slice("/client".length).replace(/^\/+/, "")
    : pathname === "/" || ["/overview", "/data", "/runs", "/network", "/criticality", "/similarity", "/embeddings", "/evaluation"].includes(pathname)
      ? "index.html"
      : pathname.replace(/^\/+/, "");
  const path = resolve(base, requested || "index.html");
  if (!inside(base, path)) return new Response("Not found", { status: 404 });
  const file = Bun.file(path);
  if (!(await file.exists())) return new Response("Not found", { status: 404 });
  return staticResponse(file, CONTENT_TYPES[extname(path)]);
}

export async function createStudioServer({ root = repositoryRoot(), port = Number(process.env.PORT || 3000), refresh = true } = {}) {
  const db = createDatabase(root);
  if (refresh) {
    try {
      await syncFilesystem(db, root);
    } catch (error) {
      console.error(`Studio inventory refresh failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  const handleApi = createApiHandler({ db, root });
  const publicRoot = resolve(import.meta.dir, "../../public");
  const sourceClientRoot = resolve(import.meta.dir, "../client");
  const bundledClientRoot = resolve(publicRoot, "dist");
  const hasBundle = await Bun.file(resolve(bundledClientRoot, "app.js")).exists();
  const clientRoot = hasBundle
    ? bundledClientRoot
    : sourceClientRoot;
  const server = Bun.serve({
    port,
    async fetch(request) {
      const url = new URL(request.url);
      if (url.pathname.startsWith("/api/")) return handleApi(request);
      if (url.pathname === "/health") return handleApi(new Request(new URL("/api/health", url), request));
      return serveStatic(request, publicRoot, clientRoot);
    }
  });
  return { server, db, root };
}

if (import.meta.main) {
  const { server } = await createStudioServer();
  console.log(`Transit Lab Studio listening on http://localhost:${server.port}`);
}
