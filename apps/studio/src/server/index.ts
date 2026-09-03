import { extname, resolve, sep } from "node:path";
import { buildStudioClient } from "../client/build.ts";

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

async function serveStatic(request, publicRoot) {
  if (request.method !== "GET" && request.method !== "HEAD") return new Response("Method not allowed", { status: 405 });
  const url = new URL(request.url);
  let pathname;
  try { pathname = decodeURIComponent(url.pathname); } catch { return new Response("Not found", { status: 404 }); }
  const requested = pathname === "/" || ["/overview", "/data", "/runs", "/network", "/criticality", "/similarity", "/embeddings", "/evaluation"].includes(pathname)
    ? "index.html"
    : pathname.replace(/^\/+/, "");
  const path = resolve(publicRoot, requested);
  if (!inside(publicRoot, path)) return new Response("Not found", { status: 404 });
  const file = Bun.file(path);
  if (!(await file.exists())) return new Response("Not found", { status: 404 });
  return new Response(file, { headers: { "Cache-Control": "no-store", "Content-Type": CONTENT_TYPES[extname(path)] || "application/octet-stream" } });
}

async function proxyApi(request) {
  const url = new URL(request.url);
  const target = new URL(url.pathname + url.search, process.env.TRANSIT_LAB_CONTROL_API_URL || "http://127.0.0.1:3100");
  try {
    return await fetch(new Request(target, request));
  } catch {
    return Response.json({ error: "Control API unavailable" }, { status: 502 });
  }
}

export async function createStudioServer({ port = Number(process.env.STUDIO_PORT || process.env.PORT || 3000) } = {}) {
  const publicRoot = resolve(import.meta.dir, "../../public");
  const bundledClient = resolve(publicRoot, "dist/app.js");
  if (!(await Bun.file(bundledClient).exists()) || !(await Bun.file(resolve(publicRoot, "dist/styles.css")).exists())) {
    const result = await buildStudioClient(resolve(publicRoot, "dist"));
    if (!result.success) throw new Error(`Studio client build failed: ${result.logs.map((log) => log.message).join("\n")}`);
  }
  const server = Bun.serve({
    port,
    fetch(request) {
      const url = new URL(request.url);
      if (url.pathname.startsWith("/api/")) return proxyApi(request);
      return serveStatic(request, publicRoot);
    }
  });
  return { server };
}

if (import.meta.main) {
  const { server } = await createStudioServer();
  console.log(`Transit Lab Studio listening on http://localhost:${server.port}`);
}
