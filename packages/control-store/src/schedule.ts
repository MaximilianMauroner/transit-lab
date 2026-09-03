import type { TrainingWindow } from "../../contracts/src/index.ts";

const DAY_NAMES = ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"];
const formatterCache = new Map<string, Intl.DateTimeFormat>();

function formatter(timezone: string) {
  let value = formatterCache.get(timezone);
  if (!value) {
    // Constructing this formatter also validates an IANA timezone. The
    // contracts package checks that the field is present; this boundary
    // checks that a worker can actually interpret it.
    value = new Intl.DateTimeFormat("en-US", {
      timeZone: timezone,
      weekday: "long",
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      hourCycle: "h23"
    });
    formatterCache.set(timezone, value);
  }
  return value;
}

function localParts(instant: Date, timezone: string) {
  const parts = Object.fromEntries(formatter(timezone).formatToParts(instant)
    .filter((part) => part.type !== "literal")
    .map((part) => [part.type, part.value]));
  return {
    year: Number(parts.year),
    month: Number(parts.month),
    day: Number(parts.day),
    hour: Number(parts.hour),
    minute: Number(parts.minute),
    weekday: String(parts.weekday).toLowerCase()
  };
}

function civilDate(year: number, month: number, day: number) {
  return new Date(Date.UTC(year, month - 1, day));
}

function civilParts(date: Date) {
  return {
    year: date.getUTCFullYear(),
    month: date.getUTCMonth() + 1,
    day: date.getUTCDate()
  };
}

function shiftCivil(date: Date, days: number) {
  const shifted = new Date(date.getTime());
  shifted.setUTCDate(shifted.getUTCDate() + days);
  return shifted;
}

function localWeekday(date: Date) {
  return DAY_NAMES[date.getUTCDay()];
}

function localClock(value: string) {
  const [hour, minute] = value.split(":").map(Number);
  return hour * 60 + minute;
}

/** Convert a local civil date/time to an instant, including normal DST days. */
function instantForLocal(date: Date, clock: string, timezone: string) {
  const { year, month, day } = civilParts(date);
  const [hour, minute] = clock.split(":").map(Number);
  const desired = Date.UTC(year, month - 1, day, hour, minute);
  let guess = desired;
  for (let attempt = 0; attempt < 5; attempt += 1) {
    const actual = localParts(new Date(guess), timezone);
    const represented = Date.UTC(actual.year, actual.month - 1, actual.day, actual.hour, actual.minute);
    const correction = desired - represented;
    if (correction === 0) return new Date(guess);
    guess += correction;
  }
  // A local time in a DST spring-forward gap has no exact instant. Returning
  // the converged boundary keeps the interval deterministic; normal windows
  // never depend on a missing minute because their endpoints are user-owned.
  return new Date(guess);
}

function normalizeSchedule(schedule: unknown): TrainingWindow[] {
  if (!schedule || typeof schedule !== "object" || Array.isArray(schedule)) return [];
  const windows = (schedule as { allowedWindows?: unknown }).allowedWindows;
  if (!Array.isArray(windows)) return [];
  return windows.filter((window): window is TrainingWindow => Boolean(window) &&
    typeof window === "object" && !Array.isArray(window) &&
    Array.isArray((window as TrainingWindow).days) &&
    typeof (window as TrainingWindow).start === "string" &&
    typeof (window as TrainingWindow).end === "string" &&
    typeof (window as TrainingWindow).timezone === "string");
}

export type ScheduleWindowMatch = {
  window: TrainingWindow;
  start: Date;
  end: Date;
};

export type ScheduleState = {
  allowed: boolean;
  active: ScheduleWindowMatch | null;
  nextStart: Date | null;
};

/**
 * Resolve the current and next allowed execution windows. An empty schedule
 * means unrestricted execution. Overnight windows (22:00–08:00) are attached
 * to the day on which they start.
 */
export function scheduleState(schedule: unknown, at = new Date()): ScheduleState {
  const windows = normalizeSchedule(schedule);
  if (windows.length === 0) return { allowed: true, active: null, nextStart: null };

  const candidates: ScheduleWindowMatch[] = [];
  for (const window of windows) {
    // Two local weeks around now cover the current overnight interval and the
    // next occurrence even when the local timezone is far from UTC.
    const local = localParts(at, window.timezone);
    const currentDate = civilDate(local.year, local.month, local.day);
    const startMinutes = localClock(window.start);
    const endMinutes = localClock(window.end);
    for (let dayOffset = -2; dayOffset <= 9; dayOffset += 1) {
      const date = shiftCivil(currentDate, dayOffset);
      if (!window.days.map((day) => day.toLowerCase()).includes(localWeekday(date))) continue;
      const start = instantForLocal(date, window.start, window.timezone);
      const endDate = endMinutes <= startMinutes ? shiftCivil(date, 1) : date;
      const end = instantForLocal(endDate, window.end, window.timezone);
      if (!(end > start)) continue;
      candidates.push({ window, start, end });
    }
  }

  const active = candidates
    .filter((candidate) => candidate.start <= at && at < candidate.end)
    .sort((left, right) => right.start.getTime() - left.start.getTime())[0] || null;
  const nextStart = candidates
    .filter((candidate) => candidate.start > at)
    .sort((left, right) => left.start.getTime() - right.start.getTime())[0]?.start || null;
  return { allowed: Boolean(active), active, nextStart };
}

export function isWithinAllowedWindow(schedule: unknown, at = new Date()) {
  return scheduleState(schedule, at).allowed;
}

export function nextAllowedWindowStart(schedule: unknown, at = new Date()) {
  return scheduleState(schedule, at).nextStart;
}

export function secondsUntilWindowEnd(schedule: unknown, at = new Date()) {
  const active = scheduleState(schedule, at).active;
  if (!active) return null;
  return Math.max(0, Math.floor((active.end.getTime() - at.getTime()) / 1_000));
}

export function scheduleDescription(schedule: unknown, at = new Date()) {
  const state = scheduleState(schedule, at);
  return {
    allowed: state.allowed,
    activeWindowEnd: state.active?.end.toISOString() || null,
    nextWindowStart: state.nextStart?.toISOString() || null
  };
}
