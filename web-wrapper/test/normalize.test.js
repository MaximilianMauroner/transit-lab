import { expect, test } from "bun:test";
import {
  formatMetricValue,
  metricValue,
  normalizePredictionFile,
  primaryMetricName
} from "../src/normalize.js";

test("normalizes the Rust prediction contract and line names", () => {
  const dataset = normalizePredictionFile({
    snapshot_id: "abc123",
    metric_names: ["accessibility_auc_loss", "mean_delay_reachable_seconds"],
    line_names: { "7": "Ring line" },
    predictions: [
      {
        line: 7,
        metrics: [0.125, 95],
        structural_uniqueness: 0.4
      }
    ]
  });

  expect(dataset.snapshotId).toBe("abc123");
  expect(dataset.predictions).toHaveLength(1);
  expect(dataset.predictions[0].label).toBe("Ring line");
  expect(metricValue(dataset.predictions[0], "accessibility_loss")).toBe(0.125);
  expect(primaryMetricName(dataset.metricNames)).toBe("accessibility_auc_loss");
  expect(formatMetricValue("accessibility_auc_loss", 0.125)).toBe("12.5%");
  expect(formatMetricValue("mean_delay_reachable_seconds", 95)).toBe("1m 35s");
});

test("accepts named metric objects and keeps generated values aligned", () => {
  const dataset = normalizePredictionFile({
    snapshotId: "object-metrics",
    predictions: [
      {
        lineId: "U1",
        metricValues: {
          accessibility_loss: 0.2,
          unreachable_share: 0.1
        },
        structuralUniqueness: 0.9
      }
    ]
  });

  expect(dataset.metricNames).toEqual(["accessibility_loss", "unreachable_share"]);
  expect(dataset.predictions[0].lineId).toBe("U1");
  expect(metricValue(dataset.predictions[0], "accessibility_auc_loss")).toBe(0.2);
});

test("rejects malformed prediction rows", () => {
  expect(() => normalizePredictionFile({ predictions: "not an array" })).toThrow(
    "predictions array"
  );
  expect(() => normalizePredictionFile({
    metric_names: ["accessibility_auc_loss"],
    predictions: [{ line: 1, metrics: ["bad"], structural_uniqueness: 0.2 }]
  })).toThrow("invalid value");
  expect(() => normalizePredictionFile({
    metric_names: ["accessibility_auc_loss"],
    predictions: [{ line: 1, metrics: [0.2], structural_uniqueness: Infinity }]
  })).toThrow("invalid structural_uniqueness");
});
