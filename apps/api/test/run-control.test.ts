import { expect, test } from "bun:test";
import { createApiHandler } from "../src/routes.ts";
import { createDatabase, createRun } from "../../../packages/control-store/src/database.ts";

test("resume and fork reject malformed JSON without swallowing the body error", async () => {
  const root = process.cwd();
  const db = createDatabase(root, ":memory:");
  const created = createRun(db, {
    kind: "train",
    datasetId: "dataset-test",
    modelConfig: {},
    seed: 7,
    runtime: {}
  });
  const handle = createApiHandler({ db, root });

  const resume = await handle(new Request(`http://api/api/runs/${created.id}/resume`, {
    method: "POST",
    body: "{"
  }));
  const fork = await handle(new Request(`http://api/api/runs/${created.id}/fork`, {
    method: "POST",
    body: "{"
  }));

  expect(resume.status).toBe(400);
  expect(fork.status).toBe(400);
  expect((await resume.json()).error).toContain("valid JSON");
  expect((await fork.json()).error).toContain("valid JSON");
  db.close();
});
