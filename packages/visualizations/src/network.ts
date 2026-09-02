type NetworkRecord = Record<string, any>;

function isRecord(value: unknown): value is NetworkRecord {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/** Validate and normalize a Rust-compiled network before a renderer sees it. */
export function validateNetwork(raw: unknown): NetworkRecord {
  const candidate = isRecord(raw) && isRecord(raw.network) && Array.isArray(raw.network.stations)
    ? raw.network
    : raw;
  if (!isRecord(candidate) || !Array.isArray(candidate.stations) || !Array.isArray(candidate.lines)) {
    throw new Error("This file does not contain a Transit Lab network.json snapshot.");
  }
  if (!candidate.stations.every((station) => isRecord(station) &&
      Number.isFinite(Number(station.latitude)) && Number.isFinite(Number(station.longitude)))) {
    throw new Error("The snapshot has stations without usable latitude and longitude values.");
  }
  return {
    ...candidate,
    patterns: Array.isArray(candidate.patterns) ? candidate.patterns : [],
    transit_edges: Array.isArray(candidate.transit_edges) ? candidate.transit_edges : [],
    transfers: Array.isArray(candidate.transfers) ? candidate.transfers : [],
    interchanges: Array.isArray(candidate.interchanges) ? candidate.interchanges : []
  };
}
