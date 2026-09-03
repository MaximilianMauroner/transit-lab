import { readFile } from "node:fs/promises";
import { validateRunEvent } from "../../../packages/contracts/src/index.ts";

export function parseEventLine(line, lineNumber, runId) {
  if (!line.trim()) return null;
  let event;
  try {
    event = JSON.parse(line);
  } catch {
    throw new Error(`structured event line ${lineNumber} is not valid JSON`);
  }
  validateRunEvent(event);
  if (event.runId !== runId) throw new Error(`structured event line ${lineNumber} has the wrong runId`);
  return event;
}

export async function readStructuredEvents(path, runId, { allowTrailingPartial = false } = {}) {
  let content;
  try {
    content = await readFile(path, "utf8");
  } catch (error) {
    if (error?.code === "ENOENT") return [];
    throw error;
  }
  const lines = content.split(/\r?\n/);
  const endsWithNewline = /\r?\n$/.test(content);
  if (allowTrailingPartial && !endsWithNewline) lines.pop();
  return lines.map((line, index) => parseEventLine(line, index + 1, runId)).filter(Boolean);
}
