import { createNamespacedApiHandler } from "./namespaces.ts";
import {
  createDatabase,
  repositoryRoot
} from "../../../packages/control-store/src/database.ts";
import { syncFilesystem } from "../../../packages/control-store/src/inventory.ts";

export async function createControlApiServer({
  root = repositoryRoot(),
  port = Number(process.env.CONTROL_API_PORT || process.env.PORT || 3100),
  refresh = true,
  allowLegacy = process.env.TRANSIT_LAB_ALLOW_LEGACY_API === "1"
} = {}) {
  const db = createDatabase(root);
  if (refresh) {
    try {
      await syncFilesystem(db, root);
    } catch (error) {
      console.error(`Control API inventory refresh failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  const handleApi = createNamespacedApiHandler({ db, root, allowLegacy });
  const server = Bun.serve({
    port,
    async fetch(request) {
      if (request.method === "OPTIONS" && new URL(request.url).pathname.startsWith("/api/public")) {
        return new Response(null, {
          status: 204,
          headers: {
            "Access-Control-Allow-Origin": process.env.TRANSIT_LAB_PUBLIC_ORIGIN || "*",
            "Access-Control-Allow-Headers": "Accept, Content-Type, Last-Event-ID",
            "Access-Control-Allow-Methods": "GET, OPTIONS"
          }
        });
      }
      return handleApi(request);
    }
  });
  return { server, db, root };
}

if (import.meta.main) {
  const { server } = await createControlApiServer();
  console.log(`Transit Lab Control API listening on http://localhost:${server.port}`);
}
