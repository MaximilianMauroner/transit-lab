const modeDefinitions = [
  { key: "tram", label: "Tram", values: [0], color: "#f5bd57", rgb: [0.96, 0.74, 0.34] },
  { key: "metro", label: "Metro", values: [1], color: "#7899ff", rgb: [0.47, 0.60, 1.0] },
  { key: "rail", label: "Rail", values: [2], color: "#b783e5", rgb: [0.72, 0.51, 0.90] },
  { key: "bus", label: "Bus", values: [3], color: "#59e0c0", rgb: [0.35, 0.88, 0.76] },
  { key: "ferry", label: "Ferry", values: [4], color: "#55c9ed", rgb: [0.33, 0.79, 0.93] },
  { key: "cable-tram", label: "Cable tram", values: [5], color: "#ef934d", rgb: [0.84, 0.42, 0.18] },
  { key: "aerial", label: "Aerial lift", values: [6], color: "#ef709b", rgb: [0.85, 0.28, 0.50] },
  { key: "funicular", label: "Funicular", values: [7], color: "#a8c866", rgb: [0.60, 0.78, 0.31] },
  { key: "trolleybus", label: "Trolleybus", values: [11], color: "#ff8b82", rgb: [1.0, 0.43, 0.38] },
  { key: "monorail", label: "Monorail", values: [12], color: "#cf8df2", rgb: [0.81, 0.46, 0.95] },
  { key: "other", label: "Other", values: [], color: "#95a2af", rgb: [0.58, 0.64, 0.69] }
];

function extendedMode(value) {
  if (value >= 100 && value < 200) return "rail";
  if (value >= 200 && value < 300) return "bus";
  if (value >= 300 && value < 400) return "rail";
  if (value >= 400 && value < 500) return "metro";
  if (value >= 500 && value < 600) return "ferry";
  if (value >= 700 && value < 800) return "bus";
  if (value >= 800 && value < 900) return "trolleybus";
  if (value >= 900 && value < 1000) return "tram";
  if (value >= 1000 && value < 1100) return "monorail";
  return "other";
}

export const TRANSIT_MODES = modeDefinitions;

export function transitMode(routeType) {
  const value = Number(routeType);
  const exact = modeDefinitions.find((mode) => mode.values.includes(value));
  if (exact) return exact;
  const key = extendedMode(value);
  return modeDefinitions.find((mode) => mode.key === key) || modeDefinitions.at(-1);
}

export function modeColor(routeType) {
  return transitMode(routeType).color;
}

export function modeRgb(routeType) {
  return transitMode(routeType).rgb;
}
