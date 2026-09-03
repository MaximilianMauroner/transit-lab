import { expect, test } from "bun:test";
import {
  isWithinAllowedWindow,
  nextAllowedWindowStart,
  scheduleDescription,
  scheduleState,
  secondsUntilWindowEnd
} from "../src/schedule.ts";

const weekdaySchedule = {
  allowedWindows: [{
    days: ["monday", "tuesday", "wednesday", "thursday", "friday"],
    start: "09:00",
    end: "17:00",
    timezone: "Europe/Berlin"
  }]
};

test("an empty schedule is unrestricted", () => {
  const state = scheduleState({}, new Date("2026-09-02T12:00:00.000Z"));
  expect(state).toEqual({ allowed: true, active: null, nextStart: null });
  expect(isWithinAllowedWindow({}, new Date("2026-09-02T12:00:00.000Z"))).toBe(true);
});

test("weekday windows resolve in their declared timezone", () => {
  const before = new Date("2026-09-02T06:00:00.000Z"); // 08:00 in Berlin
  const active = new Date("2026-09-02T12:00:00.000Z"); // 14:00 in Berlin

  expect(isWithinAllowedWindow(weekdaySchedule, before)).toBe(false);
  expect(nextAllowedWindowStart(weekdaySchedule, before)?.toISOString()).toBe("2026-09-02T07:00:00.000Z");
  expect(isWithinAllowedWindow(weekdaySchedule, active)).toBe(true);
  expect(secondsUntilWindowEnd(weekdaySchedule, active)).toBe(3 * 60 * 60);
  expect(scheduleDescription(weekdaySchedule, active)).toMatchObject({
    allowed: true,
    activeWindowEnd: "2026-09-02T15:00:00.000Z"
  });
});

test("overnight windows remain active after midnight", () => {
  const schedule = {
    allowedWindows: [{
      days: ["monday"],
      start: "22:00",
      end: "08:00",
      timezone: "Europe/Rome"
    }]
  };

  expect(isWithinAllowedWindow(schedule, new Date("2026-09-07T21:00:00.000Z"))).toBe(true); // 23:00 local
  expect(isWithinAllowedWindow(schedule, new Date("2026-09-08T05:00:00.000Z"))).toBe(true); // 07:00 local
  expect(isWithinAllowedWindow(schedule, new Date("2026-09-08T07:00:00.000Z"))).toBe(false); // 09:00 local
  expect(nextAllowedWindowStart(schedule, new Date("2026-09-08T07:00:00.000Z"))?.toISOString()).toBe("2026-09-14T20:00:00.000Z");
});

test("invalid timezones fail at schedule resolution", () => {
  expect(() => scheduleState({
    allowedWindows: [{ days: ["monday"], start: "09:00", end: "10:00", timezone: "Not/AZone" }]
  }, new Date("2026-09-07T09:00:00.000Z"))).toThrow();
});
