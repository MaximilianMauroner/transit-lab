import { expect, test } from "bun:test";
import { parseEventLine } from "../src/parse-events.ts";

const base = {
  schemaVersion: 1,
  seq: 0,
  runId: "run-test",
  timestamp: "2026-09-02T00:00:00.000Z",
  type: "progress",
  step: "compile",
  completed: 1,
  total: 1,
  unit: "command"
};

test("structured event parser accepts versioned worker events", () => {
  expect(parseEventLine(JSON.stringify(base), 1, "run-test")).toEqual(base);
});

test("structured event parser rejects wrong run identities and console text", () => {
  expect(() => parseEventLine(JSON.stringify({ ...base, runId: "run-other" }), 1, "run-test")).toThrow("wrong runId");
  expect(() => parseEventLine("progress: 50%", 2, "run-test")).toThrow("not valid JSON");
});
