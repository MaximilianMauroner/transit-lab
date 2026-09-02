import { expect, test } from "bun:test";
import { modeColor, transitMode } from "../src/modes.js";

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
