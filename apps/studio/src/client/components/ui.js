import { escapeHtml, formatDate, statusClass } from "../api.js";

export function loading(message = "Loading indexed artifacts…") {
  return `<div class="card loading"><span class="spinner"></span>${escapeHtml(message)}</div>`;
}

export function empty(title, message) {
  return `<div class="empty"><strong>${escapeHtml(title)}</strong><p>${escapeHtml(message)}</p></div>`;
}

export function errorCard(error) {
  return `<div class="card empty"><strong class="danger">Could not load this view</strong><p>${escapeHtml(error?.message || error)}</p></div>`;
}

export function status(status) {
  return `<span class="${statusClass(status)}">${escapeHtml(status || "unknown")}</span>`;
}

export function sectionHeading(title, copy = "", action = "") {
  return `<div class="section-head"><div><h3>${escapeHtml(title)}</h3>${copy ? `<p>${escapeHtml(copy)}</p>` : ""}</div>${action}</div>`;
}

export function dateCell(value) {
  return `<span title="${escapeHtml(value || "")}">${escapeHtml(formatDate(value))}</span>`;
}
