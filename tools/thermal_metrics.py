"""Align engine decode intervals with absolute cooling intervals.

Prefer the engine's recorded decode end. Old logs use the host done receipt
as an explicitly labelled estimate. Neither boundary is a GPU timestamp.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
import json
import math
from pathlib import Path
import time

THERMAL_COVERAGE_TOLERANCE_S = 0.25


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


def _thermal_rows(path):
    rows = Path(path).read_text().splitlines()
    samples, previous, partial = [], None, False
    for index, line in enumerate(rows):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(rows) - 1:
                partial = True
                break  # the supervisor can be appending the current sample
            raise
        if "wall_time_s" not in row or "paused" not in row:
            raise ValueError("thermal log has no absolute timestamps/pause state")
        now = float(row["wall_time_s"])
        if not math.isfinite(now) or previous is not None and now < previous:
            raise ValueError("thermal sample clock moved backwards")
        previous = now
        samples.append((now, row["paused"]))
    return samples, partial


def _pause_intervals(samples, end_unix=None):
    intervals, start = [], None
    for now, paused in samples:
        if paused and start is None:
            start = now
        elif not paused and start is not None:
            intervals.append((start, now))
            start = None
    if start is not None:
        # A just-finished arm may be analysed while the guard is still paused.
        end = time.time() if end_unix is None else end_unix
        intervals.append((start, max(start, end)))
    return intervals


def pause_intervals(path, end_unix=None):
    samples, _ = _thermal_rows(path)
    return _pause_intervals(samples, end_unix)


def coverage_for_windows(timestamps, windows, tolerance_s=THERMAL_COVERAGE_TOLERANCE_S):
    """Check observed thermal coverage without demanding a sample after done.

    The supervisor samples every 50 ms. Allow at most 250 ms at the window
    boundaries or between samples, and retain those gaps in the receipt. A
    missing/truncated recording must not turn into zero measured cooling.
    """
    if not math.isfinite(tolerance_s) or not 0 <= tolerance_s <= THERMAL_COVERAGE_TOLERANCE_S:
        raise ValueError("thermal coverage tolerance must be within 0..0.25 s")
    first, last = (timestamps[0], timestamps[-1]) if timestamps else (None, None)
    checked = []
    for begin, end in windows:
        if not all(math.isfinite(v) for v in (begin, end)) or end < begin:
            raise ValueError("invalid decode interval")
        # Include the neighboring observations on either side. A long sample
        # gap that crosses a decode boundary is also unobserved work.
        left = max(0, bisect_right(timestamps, begin) - 1)
        right = min(len(timestamps) - 1, bisect_left(timestamps, end))
        gaps = [(timestamps[i], timestamps[i + 1]) for i in range(left, right)
                if timestamps[i + 1] > begin and timestamps[i] < end]
        start_gap = max(0.0, first - begin) if first is not None else None
        end_gap = max(0.0, end - last) if last is not None else None
        largest = max(gaps, key=lambda pair: pair[1] - pair[0], default=None)
        maximum_gap = largest[1] - largest[0] if largest else 0.0
        reasons = []
        if not timestamps:
            reasons.append("thermal log contains no complete samples")
        else:
            if start_gap > tolerance_s:
                reasons.append("thermal log starts after the decode window")
            if end_gap > tolerance_s:
                reasons.append("thermal log ends before the decode window")
            if maximum_gap > tolerance_s:
                reasons.append("thermal sampling gap crosses the decode window")
        checked.append(dict(begin_unix=begin, end_unix=end, covered=not reasons,
                            missing_start_s=start_gap, missing_end_s=end_gap,
                            maximum_sample_gap_s=maximum_gap,
                            maximum_gap_interval=list(largest) if largest else None,
                            errors=reasons))
    return dict(complete=bool(checked) and all(row["covered"] for row in checked),
                tolerance_s=tolerance_s, sample_count=len(timestamps),
                first_sample_unix=first, last_sample_unix=last, windows=checked)


def decode_window(event, host_start=None):
    """Use the decode end captured before checkpointing or done delivery."""
    host_end = event.get("host_unix")
    if host_end is None and host_start is not None and "host_s" in event:
        host_end = host_start + event["host_s"]
    marker = event.get("decode_finished_unix")
    end = marker if marker is not None else host_end
    if isinstance(end, bool) or end is None:
        raise ValueError("decode event has no absolute engine or host clock")
    end = float(end)
    seconds = float(event.get("decode_ms", 0)) / 1000
    if not math.isfinite(end) or not math.isfinite(seconds) or seconds < 0 or end < seconds:
        raise ValueError("invalid decode interval")
    delay_ms = None
    if marker is not None and host_end is not None:
        delay_ms = 1000 * (float(host_end) - end)
        if not math.isfinite(delay_ms) or delay_ms < -1000 * THERMAL_COVERAGE_TOLERANCE_S:
            raise ValueError("engine decode end is after the host receipt")
    return end - seconds, end, marker is not None, delay_ms


def decode_events_timing(done, host_start=None, thermal_log=None):
    """Shared accounting for already parsed completed turns."""
    raw = sum(float(ev.get("decode_ms", 0)) for ev in done)
    steps = sum(int(ev.get("decode_steps", 0)) for ev in done)
    result = dict(raw_decode_ms=raw, decode_steps=steps,
                  raw_ms_per_token=raw / steps if steps else None,
                  thermal_pause_ms=None, active_decode_ms=None, active_ms_per_token=None,
                  timing_alignment="unavailable", alignment_error=None,
                  thermal_coverage=None,
                  engine_decode_boundaries=False, host_receipt_delay_ms=None,
                  thermal_log=str(Path(thermal_log).resolve()) if thermal_log else None)
    if not thermal_log:
        return result
    try:
        boundaries = [decode_window(ev, host_start) for ev in done]
    except (TypeError, ValueError) as error:
        result["alignment_error"] = str(error)
        return result
    windows = [(begin, end) for begin, end, _, _ in boundaries]
    result["engine_decode_boundaries"] = bool(boundaries) and all(b[2] for b in boundaries)
    result["host_receipt_delay_ms"] = [b[3] for b in boundaries]
    try:
        samples, partial = _thermal_rows(thermal_log)
        coverage = coverage_for_windows([now for now, _ in samples], windows)
        coverage["partial_final_line"] = partial
    except (OSError, ValueError) as error:
        result["alignment_error"] = f"thermal log unavailable: {error}"
        return result
    result["thermal_coverage"] = coverage
    if not coverage["complete"]:
        reasons = {error for window in coverage["windows"] for error in window["errors"]}
        result["alignment_error"] = "; ".join(sorted(reasons)) or "no completed decode windows"
        return result
    pauses = _pause_intervals(samples)
    cooling = sum(interval_overlap(begin, end, pauses) for begin, end in windows) * 1000
    active = max(0.0, raw - cooling)
    result.update(thermal_pause_ms=cooling, active_decode_ms=active,
                  active_ms_per_token=active / steps if steps else None,
                  timing_alignment="engine decode end minus engine decode_ms"
                  if result["engine_decode_boundaries"] else
                  "host done receipt estimate minus engine decode_ms (legacy or mixed logs)")
    return result


def decode_timing(events_path, clock_path=None, thermal_log=None):
    done = [json.loads(line) for line in Path(events_path).read_text().splitlines()
            if line.strip()]
    done = [ev for ev in done if ev.get("event") == "done"]
    host_start = None
    if clock_path and Path(clock_path).exists():
        host_start = json.loads(Path(clock_path).read_text()).get("host_start_unix")
    return decode_events_timing(done, host_start, thermal_log)
