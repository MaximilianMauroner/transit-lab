import { expect, test } from "bun:test";
import { stationsForVisibleLines } from "../src/renderer.js";

const network = {
  stations: [{ index: 0 }, { index: 1 }, { index: 2 }, { index: 3 }],
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
