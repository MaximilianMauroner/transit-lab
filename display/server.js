import { readdir } from "node:fs/promises";
import { extname, join, resolve, sep } from "node:path";

const root = import.meta.dir;
const dataRoot = resolve(root, "..", "data");
const port = Number(process.env.PORT || 3001);

const contentTypes = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8"
};

function headers(contentType) {
  return {
    "Cache-Control": "no-store",
    "Content-Type": contentType
  };
}

function inside(base, candidate) {
  return candidate === base || candidate.startsWith(`${base}${sep}`);
}

async function findSnapshots(directory, prefix = "") {
  let entries;
  try {
    entries = await readdir(directory, { withFileTypes: true });
  } catch {
    return [];
  }

  const snapshots = [];
  for (const entry of entries) {
    const path = join(directory, entry.name);
    const relativePath = prefix ? join(prefix, entry.name) : entry.name;
    if (entry.isDirectory() && entry.name !== "raw" && entry.name !== "target") {
      snapshots.push(...await findSnapshots(path, relativePath));
    } else if (entry.isFile() && entry.name === "network.json") {
      const networkPath = relativePath.split(sep).join("/");
      try {
        const manifestPath = join(entry.parentPath || directory, "manifest.json");
        const manifest = await Bun.file(manifestPath).json().catch(() => ({}));
        snapshots.push({
          id: manifest.snapshot_id || networkPath,
          label: manifest.source_name || manifest.geographical_scope || networkPath,
          scope: manifest.geographical_scope || "Unknown scope",
          serviceDate: manifest.descriptor?.service_date || "Unknown service date",
          path: `/data/${networkPath}`
        });
      } catch {
        // An incomplete snapshot should not prevent the rest from appearing.
      }
    }
  }
  return snapshots;
}

function displayNetwork(network) {
  return {
    snapshot_id: network.snapshot_id,
    manifest: network.manifest,
    stations: network.stations,
    lines: network.lines,
    patterns: (network.patterns || []).map((pattern) => ({
      index: pattern.index,
      signature: pattern.signature
    })),
    transit_edges: (network.transit_edges || []).map(({ departures_by_bin, median_runtime_by_bin, ...edge }) => edge),
    transfers: network.transfers || [],
    interchanges: network.interchanges || []
  };
}

const server = Bun.serve({
  port,
  async fetch(request) {
    const url = new URL(request.url);

    if (url.pathname === "/health") {
      return Response.json({ ok: true, service: "transit-lab-display" });
    }

    if (url.pathname === "/api/snapshots") {
      const snapshots = await findSnapshots(dataRoot);
      snapshots.sort((left, right) => left.label.localeCompare(right.label));
      return Response.json(snapshots, { headers: headers("application/json; charset=utf-8") });
    }

    if (url.pathname === "/api/network") {
      const requestedPath = url.searchParams.get("path");
      if (!requestedPath || !requestedPath.startsWith("/data/")) {
        return new Response("A local snapshot path is required", { status: 400 });
      }
      const filePath = resolve(dataRoot, requestedPath.slice("/data/".length));
      if (!inside(dataRoot, filePath) || !filePath.endsWith("network.json")) {
        return new Response("Not found", { status: 404 });
      }
      const file = Bun.file(filePath);
      if (!(await file.exists())) return new Response("Not found", { status: 404 });
      try {
        return Response.json(displayNetwork(await file.json()), {
          headers: headers("application/json; charset=utf-8")
        });
      } catch {
        return new Response("Snapshot is not valid JSON", { status: 422 });
      }
    }

    let pathname;
    try {
      pathname = decodeURIComponent(url.pathname);
    } catch {
      return new Response("Not found", { status: 404 });
    }

    const base = pathname.startsWith("/data/") ? dataRoot : root;
    const requested = pathname.startsWith("/data/")
      ? pathname.slice("/data/".length)
      : pathname === "/" ? "index.html" : pathname.slice(1);
    const filePath = resolve(base, requested);
    if (!inside(base, filePath)) {
      return new Response("Not found", { status: 404 });
    }

    const file = Bun.file(filePath);
    if (!(await file.exists())) {
      return new Response("Not found", { status: 404 });
    }

    return new Response(file, {
      headers: headers(contentTypes[extname(filePath)] || "application/octet-stream")
    });
  }
});

console.log(`Transit Lab display listening on http://localhost:${server.port}`);
