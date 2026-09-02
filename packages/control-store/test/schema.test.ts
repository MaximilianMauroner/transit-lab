import { expect, test } from "bun:test";
import { Database } from "bun:sqlite";
import {
  CONTROL_STORE_SCHEMA_VERSION,
  pushDatabaseSchema
} from "../src/schema.ts";

test("control-store schema push is idempotent and has no migration ledger", () => {
  const db = new Database(":memory:");
  expect(pushDatabaseSchema(db)).toEqual({ version: CONTROL_STORE_SCHEMA_VERSION });
  expect(pushDatabaseSchema(db)).toEqual({ version: CONTROL_STORE_SCHEMA_VERSION });

  const tables = db.query("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
    .all()
    .map((row: { name: string }) => row.name);
  expect(tables).toContain("runs");
  expect(tables).toContain("publications");
  expect(tables).not.toContain("schema_migrations");

  const runColumns = db.query("PRAGMA table_info(runs)").all().map((row: { name: string }) => row.name);
  expect(runColumns).toContain("config_fingerprint");
  expect(runColumns).toContain("resolved_config_path");
  db.close();
});

test("schema push adds the compatibility columns to an older runs table", () => {
  const db = new Database(":memory:");
  db.exec(`CREATE TABLE runs (
    id TEXT PRIMARY KEY,
    project_id TEXT,
    kind TEXT,
    status TEXT,
    spec_json TEXT,
    fingerprint TEXT
  )`);

  pushDatabaseSchema(db);

  const columns = db.query("PRAGMA table_info(runs)").all().map((row: { name: string }) => row.name);
  expect(columns).toContain("config_fingerprint");
  expect(columns).toContain("resolved_config_path");
  db.close();
});
