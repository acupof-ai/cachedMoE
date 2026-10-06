"""Align host decode intervals with a supervisor's absolute cooling intervals.

The host receives `done` shortly after the engine finishes decoding. This is a
host boundary estimate, not a GPU timestamp. Raw wall time is always retained.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time


def interval_overlap(begin, end, intervals):
    if not all(math.isfinite(v) for v in (begin, end)) or end < begin:
        raise ValueError("invalid decode interval")
    clipped = []
    for a, b in intervals:
        if not all(math.isfinite(v) for v in (a, b)) or b < a:
            raise ValueError("invalid thermal pause interval")
        if b > begin and a < end:
            clipped.append((max(begin, a), min(end, b)))
    # Union the pauses before adding their lengths: overlapping receipt
    # intervals must never subtract cooling twice.
    total, cursor = 0.0, begin
    for a, b in sorted(clipped):
        total += max(0.0, b - max(cursor, a))
        cursor = max(cursor, b)
    return total


def pause_intervals(path, end_unix=None):
    rows = Path(path).read_text().splitlines()
    intervals, start, previous = [], None, None
    for index, line in enumerate(rows):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(rows) - 1:
                break  # the supervisor can be appending the current sample
            raise
        if "wall_time_s" not in row or "paused" not in row:
            raise ValueError("thermal log has no absolute timestamps/pause state")
        now = float(row["wall_time_s"])
        if not math.isfinite(now) or previous is not None and now < previous:
            raise ValueError("thermal sample clock moved backwards")
        previous = now
        if row["paused"] and start is None:
            start = now
        elif not row["paused"] and start is not None:
            intervals.append((start, now))
            start = None
    if start is not None:
        # A just-finished arm may be analysed while the guard is still paused.
        end = time.time() if end_unix is None else end_unix
        intervals.append((start, max(start, end)))
    return intervals


def decode_timing(events_path, clock_path=None, thermal_log=None):
    done = [json.loads(line) for line in Path(events_path).read_text().splitlines()
            if line.strip()]
    done = [ev for ev in done if ev.get("event") == "done"]
    raw = sum(float(ev.get("decode_ms", 0)) for ev in done)
    steps = sum(int(ev.get("decode_steps", 0)) for ev in done)
    result = dict(raw_decode_ms=raw, decode_steps=steps,
                  raw_ms_per_token=raw / steps if steps else None,
                  thermal_pause_ms=None, active_decode_ms=None, active_ms_per_token=None,
                  timing_alignment="unavailable")
    if not thermal_log:
        return result
    host_start = None
    if clock_path and Path(clock_path).exists():
        host_start = json.loads(Path(clock_path).read_text()).get("host_start_unix")
    if any("host_unix" not in ev and (host_start is None or "host_s" not in ev) for ev in done):
        return result
    pauses = pause_intervals(thermal_log)
    cooling = 0.0
    for ev in done:
        end = ev.get("host_unix", (host_start or 0) + ev.get("host_s", 0))
        seconds = float(ev.get("decode_ms", 0)) / 1000
        if seconds < 0:
            raise ValueError("negative decode time")
        cooling += interval_overlap(end - seconds, end, pauses) * 1000
    active = max(0.0, raw - cooling)
    result.update(thermal_pause_ms=cooling, active_decode_ms=active,
                  active_ms_per_token=active / steps if steps else None,
                  timing_alignment="host done receipt minus engine decode_ms",
                  thermal_log=str(Path(thermal_log).resolve()))
    return result
