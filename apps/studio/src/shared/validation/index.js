export {
  assertSafeId,
  normalizeRunSpec,
  validateArtifactManifest,
  validateDatasetManifest,
  validateInferenceResult,
  validateRunEvent,
  validateRunSpec
} from "../contracts/index.js";

export function jsonBody(request) {
  return request.json().catch(() => {
    throw new Error("request body must be valid JSON");
  });
}

export function boundedInteger(value, fallback, { min = 0, max = 2_000 } = {}) {
  const number = Number(value);
  if (!Number.isInteger(number)) return fallback;
  return Math.max(min, Math.min(max, number));
}
