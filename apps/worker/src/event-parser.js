import { readFile } from "node:fs/promises";
import { validateRunEvent } from "../../../packages/contracts/src/index.js";

export function parseEventLine(line, lineNumber, runId) {
  if (!line.trim()) return null;
  let event;
  try {
    event = JSON.parse(line);
  } catch {
    throw new Error(`structured event line ${lineNumber} is not valid JSON`);
  }
  validateRunEvent(event);
  if (event.runId !== runId) {
    throw new Error(`structured event line ${lineNumber} has the wrong runId`);
  }
  return event;
}

export async function readStructuredEvents(path, runId) {
  let content;
  try {
    content = await readFile(path, "utf8");
  } catch (error) {
    if (error?.code === "ENOENT") return [];
    throw error;
  }
  return content
    .split(/\r?\n/)
    .map((line, index) => parseEventLine(line, index + 1, runId))
    .filter(Boolean);
}
