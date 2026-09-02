import { expect, test } from "bun:test";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createNamespacedApiHandler } from "../src/namespaces.ts";
import { createDatabase, run } from "../../../packages/control-store/src/database.ts";

test("control publications gate the read-only public API", async () => {
  const root = await mkdtemp(join(tmpdir(), "transit-lab-publication-test-"));
  const db = createDatabase(root, ":memory:");
  const timestamp = "2026-09-02T00:00:00.000Z";
  run(db, "INSERT INTO projects(id, name, created_at) VALUES (?, ?, ?)", ["project-local", "Transit Lab", timestamp]);
  run(db, "INSERT INTO networks(id, project_id, display_name, created_at, updated_at) VALUES (?, ?, ?, ?, ?)", ["demo", "project-local", "Demo", timestamp, timestamp]);
  run(db, `INSERT INTO snapshots(id, network_id, service_date, status, fingerprint, manifest_path, network_path, created_at, updated_at)
    VALUES (?, ?, ?, 'ready', ?, ?, ?, ?, ?)`, ["snapshot-1", "demo", "2026-09-02", "snapshot-fingerprint", "data/private/manifest.json", "data/private/network.json", timestamp, timestamp]);

  const handle = createNamespacedApiHandler({ db, root, allowLegacy: false });
  expect((await handle(new Request("http://api/api/public/catalog"))).status).toBe(200);
  expect((await handle(new Request("http://api/api/public/snapshots"))).status).toBe(404);
  expect((await handle(new Request("http://api/api/public/catalog", { method: "POST" }))).status).toBe(405);

  const publication = await handle(new Request("http://api/api/control/publications", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id: "publication-1", slug: "demo-release", title: "Demo release", snapshotIds: ["snapshot-1"] })
  }));
  expect(publication.status).toBe(201);

  const catalog = await handle(new Request("http://api/api/public/catalog"));
  expect((await catalog.json()).publications[0]).toMatchObject({ id: "publication-1", snapshotIds: ["snapshot-1"] });
  const snapshots = await handle(new Request("http://api/api/public/snapshots"));
  expect((await snapshots.json())[0]).not.toHaveProperty("manifestPath");
  db.close();
});
