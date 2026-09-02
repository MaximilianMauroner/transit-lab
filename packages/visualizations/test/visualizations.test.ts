import { expect, test } from "bun:test";
import { modeColor, transitMode } from "../src/modes.ts";
import { stationsForVisibleLines } from "../src/renderer.ts";
import { validateNetwork } from "../src/network.ts";

test("maps standard GTFS route types to stable transit categories", () => {
  expect(transitMode(0).key).toBe("tram");
  expect(transitMode(1).key).toBe("metro");
  expect(transitMode(2).key).toBe("rail");
  expect(transitMode(3).key).toBe("bus");
  expect(transitMode(4).key).toBe("ferry");
  expect(modeColor(3)).toBe("#59e0c0");
});

test("groups extended GTFS route types with their base transit type", () => {
  expect(transitMode(100).key).toBe("rail");
  expect(transitMode(700).key).toBe("bus");
  expect(transitMode(900).key).toBe("tram");
  expect(transitMode(9999).key).toBe("other");
});

test("accepts the canonical snapshot shape and optional relation arrays", () => {
  const network = validateNetwork({
    snapshot_id: "snapshot",
    manifest: { source_name: "Austria", geographical_scope: "Vienna" },
    stations: [{ index: 0, name: "Central", latitude: 48.2, longitude: 16.3 }],
    lines: [{ index: 0, display_name: "U1" }]
  });

  expect(network.stations).toHaveLength(1);
  expect(network.lines[0].display_name).toBe("U1");
  expect(network.transit_edges).toEqual([]);
  expect(network.transfers).toEqual([]);
});

test("rejects a file that is not a compiled network", () => {
  expect(() => validateNetwork({ stations: [], routes: [] })).toThrow("network.json");
});

const network = {
  stations: [{ index: 0 }, { index: 1 }, { index: 2 }, { index: 3 }],
  lines: [],
  patterns: [
    { signature: { line: 0, stops: [0, 1] } },
    { signature: { line: 1, stops: [2, 3] } }
  ],
  transit_edges: [
    { line: 0, from: 0, to: 1 },
    { line: 1, from: 2, to: 3 }
  ]
};

test("only keeps stations served by visible lines", () => {
  expect([...stationsForVisibleLines(network, new Set([0]))]).toEqual([0, 1]);
  expect([...stationsForVisibleLines(network, new Set([1]))]).toEqual([2, 3]);
  expect([...stationsForVisibleLines(network, new Set())]).toEqual([]);
});
