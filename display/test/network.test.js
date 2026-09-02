import { expect, test } from "bun:test";

function validateNetwork(raw) {
  const network = raw?.network && raw.network.stations ? raw.network : raw;
  if (!network || !Array.isArray(network.stations) || !Array.isArray(network.lines)) {
    throw new Error("This file does not contain a Transit Lab network.json snapshot.");
  }
  return {
    ...network,
    transit_edges: Array.isArray(network.transit_edges) ? network.transit_edges : [],
    transfers: Array.isArray(network.transfers) ? network.transfers : [],
    interchanges: Array.isArray(network.interchanges) ? network.interchanges : []
  };
}

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
