"""Reduce the owner power-profile experiment without using active-time rankings."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from thermal_metrics import decode_events_timing, decode_window

PROFILES = ("power-saver", "balanced", "performance")


def select_profile(costs):
    if set(costs) != set(PROFILES) or any(not math.isfinite(v) or v <= 0 for v in costs.values()):
        raise ValueError("three positive, complete profile costs are required")
    fastest = min(costs, key=costs.get)
    # The ordering is the owner's lower-power tie break, not the arm order's
    # observed energy consumption. We do not have whole-system energy counters.
    selected = next(p for p in PROFILES if costs[p] <= costs[fastest] * 1.03)
    return dict(fastest=fastest, selected=selected, costs=costs)


def thermal_summary(rows, turns):
    windows = [decode_window(t)[:2] for t in turns]
    total, paused, clock_weight, clock_time = 0., 0., 0., 0.
    pauses = 0
    peaks = {}
    previous_paused = False
    for left, right in zip(rows, rows[1:]):
        begin, end = left["wall_time_s"], right["wall_time_s"]
        dt = sum(max(0., min(end, b) - max(begin, a)) for a, b in windows)
        if not dt:
            continue
        total += dt
        current = bool(left["paused"])
        paused += dt if current else 0.
        pauses += int(current and not previous_paused)
        previous_paused = current
        for key, value in left.items():
            if key.startswith(("amdgpu:", "nvme:")):
                peaks[key] = max(peaks.get(key, value), value)
        clock = re.search(r"(\d+)Mhz\s*\*", left.get("pp_dpm_sclk", ""))
        if clock:
            clock_weight += int(clock.group(1)) * dt
            clock_time += dt
    return dict(decode_wall_s=total, decode_cooling_s=paused,
                pause_entries_in_decode=pauses, peak_c=peaks,
                mean_gpu_clock_mhz=clock_weight / clock_time if clock_time else None,
                clock_observed_s=clock_time)


def arm_report(turns, thermal_path):
    if len(turns) != 8 or any(t["generated"] != 512 for t in turns):
        raise ValueError("an arm must finish eight outputs of exactly 512 tokens")
    rows = [json.loads(line) for line in thermal_path.read_text().splitlines()]
    profile = turns[0]["power_profile"]
    arm_samples = [r for r in rows if r.get("arm") == profile
                   and r.get("phase") in ("generate", "between_turns")]
    if not arm_samples or any(r["power_profile"] != profile for r in arm_samples):
        raise ValueError("missing arm telemetry or unexpected profile change")
    whole_windows = [(arm_samples[0]["wall_time_s"], arm_samples[-1]["wall_time_s"])]
    whole_turn = dict(decode_ms=(whole_windows[0][1] - whole_windows[0][0]) * 1000,
                      decode_finished_unix=whole_windows[0][1])
    whole = thermal_summary(rows, [whole_turn])
    pieces = {}
    for name, subset in (("all", turns), ("first_two", turns[:2]), ("last_two", turns[-2:])):
        timing = decode_events_timing(subset, thermal_log=thermal_path)
        pieces[name] = dict(timing=timing, thermal=thermal_summary(rows, subset))
    verified = sum(t["speculation"]["verified"] for t in turns)
    accepted = sum(t["speculation"]["accepted"] for t in turns)
    cycles = sum(t["speculation"]["cycles"] for t in turns)
    steps = sum(t["decode_steps"] for t in turns)
    return dict(groups=pieces, whole_arm_thermal=whole, generated=sum(t["generated"] for t in turns),
                decode_steps=steps, cycles=cycles, verified=verified, accepted=accepted,
                acceptance=accepted / verified if verified else None,
                decode_hit_rate=sum(t["decode_hit_rate"] * t["decode_steps"] for t in turns) / steps,
                total_engine_ms_per_output=sum(t["total_ms"] for t in turns) / (512 * 8),
                request_wall_ms_per_output=sum(t["request_wall_ms"] for t in turns) / (512 * 8),
                repetition=[t["repetition"] for t in turns],
                stage_ms_per_cycle={key: sum(t["speculation"][key] for t in turns) / cycles
                                    for key in ("draft_ms", "verify_ms", "commit_ms", "cpu_ms")})


def report(out):
    arms = {}
    for profile in PROFILES:
        directory = out / profile
        arms[profile] = arm_report(json.loads((directory / "turns.json").read_text()),
                                  out / "thermal_measurement.jsonl")
    rankings = {group: select_profile({p: a["groups"][group]["timing"]["raw_ms_per_token"]
                                      for p, a in arms.items()})
                for group in ("all", "first_two", "last_two")}
    # The owner must choose if the short and sustained raw winners differ.
    needs_owner = rankings["first_two"]["fastest"] != rankings["last_two"]["fastest"]
    return dict(arms=arms, rankings=rankings, owner_decision_required=needs_owner,
                default_candidate=rankings["all"]["selected"],
                default_change_authorized=not needs_owner,
                definitions=dict(primary="engine decode wall ms / timed decode steps, including cooling",
                                 active="CPU-suspension subtraction estimate only; never used for selection",
                                 frequency="wall-time weighted reported current sclk within decode windows",
                                 cache="same engine, fixed arm order; KV resets, experts carry across arms",
                                 tie="within 3% of raw fastest, select lower-power profile"))
