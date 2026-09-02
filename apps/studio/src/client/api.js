export async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { Accept: "application/json", ...(options.body ? { "Content-Type": "application/json" } : {}) },
    ...options
  });
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("json") ? await response.json() : await response.text();
  if (!response.ok) {
    const message = typeof payload === "object" ? payload.error : payload;
    throw new Error(message || `Request failed with HTTP ${response.status}`);
  }
  return payload;
}

export function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

export function formatCount(value) {
  return new Intl.NumberFormat().format(Number(value || 0));
}

export function shortId(value, start = 8, end = 6) {
  const text = String(value ?? "");
  return text.length > start + end + 1 ? `${text.slice(0, start)}…${text.slice(-end)}` : text || "—";
}

export function formatDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
}

export function statusClass(status) {
  return `status status-${String(status || "unknown").toLowerCase()}`;
}
