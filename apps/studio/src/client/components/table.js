import { escapeHtml } from "../api.js";
import { empty } from "./ui.js";

export function table({ headers, rows, emptyTitle = "Nothing indexed yet", emptyMessage = "Run an inventory refresh after producing a Rust artifact." }) {
  if (!rows.length) return empty(emptyTitle, emptyMessage);
  return `<div class="table-wrap"><table class="data-table"><thead><tr>${headers.map((header) => `<th>${escapeHtml(header)}</th>`).join("")}</tr></thead><tbody>${rows.join("")}</tbody></table></div>`;
}
