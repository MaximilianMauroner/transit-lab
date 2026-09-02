export const DEFAULT_METRIC_NAMES = [
  "accessibility_auc_loss",
  "unreachable_share",
  "mean_delay_reachable_seconds",
  "p95_delay_reachable_seconds",
  "mean_extra_transfers",
  "stations_losing_all_service_share"
];

const metricAliases = new Map([
  ["accessibility_loss", "accessibility_auc_loss"],
  ["accessibility_auc_loss", "accessibility_auc_loss"],
  ["unreachable", "unreachable_share"],
  ["unreachable_share", "unreachable_share"],
  ["mean_delay_seconds", "mean_delay_reachable_seconds"],
  ["mean_delay_reachable_seconds", "mean_delay_reachable_seconds"],
  ["p95_delay_seconds", "p95_delay_reachable_seconds"],
  ["p95_delay_reachable_seconds", "p95_delay_reachable_seconds"],
  ["extra_transfers", "mean_extra_transfers"],
  ["mean_extra_transfers", "mean_extra_transfers"],
  ["stations_losing_service_share", "stations_losing_all_service_share"],
  ["stations_losing_all_service_share", "stations_losing_all_service_share"]
]);

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function asMetricKey(name) {
  const snakeName = String(name)
    .trim()
    .replace(/([a-z0-9])([A-Z])/g, "$1_$2")
    .replace(/[\s-]+/g, "_")
    .toLowerCase();

  return metricAliases.get(snakeName) || snakeName;
}

function readMetricNames(input, firstMetrics) {
  const declared = input.metric_names ?? input.metricNames;
  if (declared !== undefined && !Array.isArray(declared)) {
    throw new Error("metric_names must be an array of strings");
  }

  const names = declared?.map((name) => String(name).trim()) ||
    (Array.isArray(firstMetrics) && firstMetrics.length === DEFAULT_METRIC_NAMES.length
      ? [...DEFAULT_METRIC_NAMES]
      : []);

  if (names.some((name) => !name)) {
    throw new Error("metric_names cannot contain blank names");
  }

  const keys = names.map(asMetricKey);
  if (new Set(keys).size !== keys.length) {
    throw new Error("metric_names must be unique");
  }

  return names;
}

function readLineName(input, lineId, index) {
  const candidates = [
    input.line_name,
    input.lineName,
    input.name,
    input.label
  ];
  const lineNames = input.__lineNames;
  if (lineNames && isRecord(lineNames)) {
    candidates.push(lineNames[String(lineId)]);
  }

  const name = candidates.find((candidate) => typeof candidate === "string" && candidate.trim());
  if (name) return name.trim();

  if (lineId === undefined || lineId === null || String(lineId).trim() === "") {
    return `Line ${index}`;
  }

  return `Line ${String(lineId).trim()}`;
}

function readLineId(input, index) {
  const value = input.line ?? input.line_id ?? input.lineId ?? input.id ?? index;
  if (typeof value !== "string" && typeof value !== "number") {
    throw new Error(`prediction ${index + 1} has no usable line id`);
  }

  const lineId = String(value).trim();
  if (!lineId) throw new Error(`prediction ${index + 1} has a blank line id`);
  return lineId;
}

function readMetrics(input, metricNames, index) {
  const rawMetrics = input.metrics ?? input.metric_values ?? input.metricValues;
  if (!Array.isArray(rawMetrics) && !isRecord(rawMetrics)) {
    throw new Error(`prediction ${index + 1} must have a metrics array or object`);
  }

  const values = metricNames.map((name, metricIndex) => {
    const rawValue = Array.isArray(rawMetrics)
      ? rawMetrics[metricIndex]
      : rawMetrics[name] ?? rawMetrics[asMetricKey(name)];

    if (typeof rawValue !== "number" || !Number.isFinite(rawValue)) {
      throw new Error(`prediction ${index + 1} has an invalid value for ${name}`);
    }
    return rawValue;
  });

  if (Array.isArray(rawMetrics) && rawMetrics.length !== metricNames.length) {
    throw new Error(
      `prediction ${index + 1} has ${rawMetrics.length} metrics, expected ${metricNames.length}`
    );
  }

  return values;
}

export function normalizePredictionFile(input) {
  if (Array.isArray(input)) input = { predictions: input };
  if (!isRecord(input)) {
    throw new Error("prediction JSON must be an object");
  }
  if (!Array.isArray(input.predictions)) {
    throw new Error("prediction JSON must contain a predictions array");
  }

  const firstMetrics = input.predictions[0]?.metrics;
  let metricNames = readMetricNames(input, firstMetrics);
  if (!metricNames.length && input.predictions.length) {
    const first = input.predictions[0];
    const firstObjectMetrics = first && (first.metrics ?? first.metric_values ?? first.metricValues);
    if (isRecord(firstObjectMetrics)) {
      metricNames = Object.keys(firstObjectMetrics);
    } else if (Array.isArray(firstObjectMetrics)) {
      metricNames = firstObjectMetrics.map((_, index) => `metric_${index + 1}`);
    }
  }

  const lineNames = input.line_names ?? input.lineNames;
  if (lineNames !== undefined && !isRecord(lineNames)) {
    throw new Error("line_names must be an object keyed by line id");
  }

  const predictions = input.predictions.map((rawPrediction, index) => {
    if (!isRecord(rawPrediction)) {
      throw new Error(`prediction ${index + 1} must be an object`);
    }

    const lineId = readLineId(rawPrediction, index);
    const metrics = readMetrics(
      { ...rawPrediction, __lineNames: lineNames },
      metricNames,
      index
    );
    const structuralUniqueness = rawPrediction.structural_uniqueness ??
      rawPrediction.structuralUniqueness;

    if (typeof structuralUniqueness !== "number" || !Number.isFinite(structuralUniqueness)) {
      throw new Error(`prediction ${index + 1} has an invalid structural_uniqueness value`);
    }

    const values = {};
    metricNames.forEach((name, metricIndex) => {
      values[name] = metrics[metricIndex];
      values[asMetricKey(name)] = metrics[metricIndex];
    });

    return {
      index,
      lineId,
      lineName: readLineName({ ...rawPrediction, __lineNames: lineNames }, lineId, index + 1),
      label: readLineName({ ...rawPrediction, __lineNames: lineNames }, lineId, index + 1),
      metrics,
      values,
      structuralUniqueness
    };
  });

  return {
    snapshotId: String(input.snapshot_id ?? input.snapshotId ?? "Unidentified snapshot"),
    metricNames,
    predictions
  };
}

export function metricValue(prediction, name) {
  const key = asMetricKey(name);
  return prediction?.values?.[key] ?? prediction?.values?.[name];
}

export function metricLabel(name) {
  return String(name)
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function metricKind(name) {
  const key = asMetricKey(name);
  if (key.includes("seconds") || key.includes("delay")) return "duration";
  if (key.includes("share") || key.includes("loss") || key.includes("uniqueness")) {
    return "percent";
  }
  return "number";
}

export function formatMetricValue(name, value) {
  if (value === undefined || value === null || !Number.isFinite(value)) return "—";

  switch (metricKind(name)) {
    case "percent":
      return `${(value * 100).toFixed(1)}%`;
    case "duration": {
      const sign = value < 0 ? "-" : "";
      const seconds = Math.round(Math.abs(value));
      if (seconds < 60) return `${sign}${seconds}s`;
      const minutes = Math.floor(seconds / 60);
      const remainder = seconds % 60;
      return `${sign}${minutes}m ${String(remainder).padStart(2, "0")}s`;
    }
    default:
      return Number.isInteger(value) ? String(value) : value.toFixed(2);
  }
}

export function primaryMetricName(metricNames) {
  return metricNames.find((name) => asMetricKey(name) === "accessibility_auc_loss") ||
    metricNames[0];
}
