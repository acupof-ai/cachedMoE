#!/usr/bin/env python3
"""Report existing mask-quality experiments without starting an engine.

Inputs are comparison roots, individual arm directories, or web_longtest
directories. Cooling-adjusted time uses host receipt boundaries; it is not a
GPU timestamp. A quality file maps arm names to explicit gate evidence:
off_nll, nll, chinese64_no_loop, long512_no_loop, long512_repeat_vs_off,
mmlu_correct, mmlu_n, decode_baseline, and longctx_baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from repetition_metrics import metrics
from thermal_metrics import decode_events_timing, interval_overlap, pause_intervals

STAGES = ("draft_ms", "verify_ms", "commit_ms", "cpu_ms")
QUALITY_BOOLEANS = ("chinese64_no_loop", "long512_no_loop",
                    "long512_repeat_vs_off", "decode_baseline", "longctx_baseline")


def read_json(path):
    return json.loads(Path(path).read_text())


def event_rows(path):
    """A final half-written line is pending evidence, not a completed turn."""
    rows = Path(path).read_text().splitlines()
    result, incomplete_line = [], False
    for index, line in enumerate(rows):
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(rows) - 1:
                raise
            incomplete_line = True
    return result, incomplete_line


def event_outputs(rows):
    outputs, ids, errors = [], [], []
    for row in rows:
        if row.get("event") == "token":
            token = row.get("id")
            if not isinstance(token, int) or isinstance(token, bool) or token < 0:
                raise ValueError("invalid generated token id")
            ids.append(token)
        elif row.get("event") == "done":
            outputs.append((row, ids))
            ids = []
        elif row.get("event") == "error":
            errors.append(row)
            ids = []
    return outputs, len(ids), errors


def timing_for_turn(turn, host_start, pauses):
    raw, steps = float(turn.get("decode_ms", 0)), int(turn.get("decode_steps", 0))
    if not math.isfinite(raw) or raw < 0 or steps < 0:
        raise ValueError("invalid decode counters")
    end = turn.get("host_unix")
    if end is None and host_start is not None and "host_s" in turn:
        end = host_start + turn["host_s"]
    cooling = None
    if pauses is not None and end is not None:
        cooling = 1000 * interval_overlap(float(end) - raw / 1000, float(end), pauses)
    active = None if cooling is None else max(0.0, raw - cooling)
    return dict(raw_decode_ms=raw, decode_steps=steps,
                raw_ms_per_token=raw / steps if steps else None,
                thermal_pause_ms=cooling, active_decode_ms=active,
                active_ms_per_token=active / steps if active is not None and steps else None)


def cycle_report(turns, timing, common_fraction=.60):
    counters = {}
    for turn in turns:
        for key, value in turn.get("speculation", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(value) or value < 0:
                    raise ValueError("invalid speculative counter")
                counters[key] = counters.get(key, 0) + value
    cycles = counters.get("cycles", 0)
    if not cycles:
        return dict(cycles=0, counters=counters, full_raw_cycle_ms=None,
                    full_active_cycle_ms=None, components_raw_ms=None,
                    uncovered_raw_cycle_ms=None, acceptance=None,
                    common_2_25=None, common_accepted_fraction=None)
    components = {key: counters[key] / cycles if key in counters else None for key in STAGES}
    raw_cycle = timing["raw_decode_ms"] / cycles
    active_ms = timing.get("active_decode_ms")
    active_cycle = None if active_ms is None else active_ms / cycles
    verified, accepted, emitted = (counters.get(k) for k in ("verified", "accepted", "tokens"))
    if accepted is not None and verified is not None and accepted > verified:
        raise ValueError("accepted count exceeds verified drafts")
    if accepted is not None and emitted is not None and emitted > accepted + cycles:
        raise ValueError("emitted count exceeds accepted drafts plus cycle bonuses")
    proposed_per_cycle = verified / cycles if verified is not None else None
    # A stop inside an accepted prefix emits no correction/bonus token.
    # Keep that observed terminal correction separate from the common-ratio
    # arithmetic; neither calculation predicts a different generated route.
    terminal_correction = ((accepted + cycles - emitted) / cycles
                           if accepted is not None and emitted is not None else None)
    common_outputs = 1 + common_fraction * proposed_per_cycle if proposed_per_cycle is not None else None
    corrected_outputs = (common_outputs - terminal_correction
                         if common_outputs is not None and terminal_correction is not None else None)
    def normalized(outputs):
        return dict(outputs_per_cycle=outputs,
                    raw_ms_per_output=raw_cycle / outputs if outputs is not None and outputs > 0 else None,
                    active_ms_per_output=active_cycle / outputs
                    if active_cycle is not None and outputs is not None and outputs > 0 else None)
    def per_cycle(key):
        return counters[key] / cycles if key in counters else None
    def per_output(key):
        return counters[key] / emitted if key in counters and emitted else None
    return dict(cycles=cycles, counters=counters, full_raw_cycle_ms=raw_cycle,
                full_active_cycle_ms=active_cycle, components_raw_ms=components,
                uncovered_raw_cycle_ms=raw_cycle - sum(components.values())
                if all(v is not None for v in components.values()) else None,
                active_stage_attribution="unavailable: cooling can occur inside any raw stage timer",
                acceptance=accepted / verified if verified and accepted is not None else None,
                accepted_per_cycle=per_cycle("accepted"),
                outputs_per_cycle=per_cycle("tokens"),
                verified_per_cycle=proposed_per_cycle,
                union_per_cycle=per_cycle("union_experts"),
                union_per_output=per_output("union_experts"),
                miss_bytes_per_cycle=per_cycle("miss_bytes"),
                miss_bytes_per_output=per_output("miss_bytes"),
                common_2_25=normalized(2.25),
                common_accepted_fraction=dict(fraction=common_fraction,
                    uncorrected=normalized(common_outputs),
                    observed_terminal_correction=terminal_correction,
                    terminal_adjusted=normalized(corrected_outputs),
                    scope="same aggregate accepted/proposed ratio; not independent per-position acceptance"))


def status_counters(status):
    result = {}
    patterns = {
        "store_failed_fills": ("store", r"failed fills (\d+)"),
        "io_failed_requests": ("io", r"io:\s*\d+ req \(\d+ done, (\d+) failed"),
        "p0_reserve_failures": ("planner", r"P0 failures reserve (\d+)"),
        "p0_submit_failures": ("planner", r"P0 failures reserve \d+ / submit (\d+)"),
        "p0_io_failures": ("planner", r"P0 failures reserve \d+ / submit \d+ / IO (\d+)"),
    }
    for name, (field, pattern) in patterns.items():
        match = re.search(pattern, status.get(field, ""))
        result[name] = int(match[1]) if match else None
    result["source_dropped"] = "DROPPED" in status.get("io", "") if status else None
    return result


def failure_report(before, after):
    start, finish = status_counters(before), status_counters(after)
    delta = {}
    for key, value in finish.items():
        if key == "source_dropped":
            delta[key] = value
        elif value is not None and start[key] is not None:
            delta[key] = value - start[key]
        else:
            delta[key] = None
    # Without a start snapshot, zero cumulative failures still proves zero
    # failures in this arm. Positive totals cannot be attributed to this arm.
    checked = delta if before else finish
    values = [v for k, v in checked.items() if k != "source_dropped"]
    if checked["source_dropped"] or any(v is not None and v > 0 for v in values):
        verdict = False
    elif all(v == 0 for v in values) and checked["source_dropped"] is False:
        verdict = True
    else:
        verdict = None
    return dict(before=start, after=finish, delta=delta, no_load_failures=verdict,
                scope="arm counter delta" if before else "end cumulative counters include startup")


def repetition_gate(candidate, baseline):
    """Compare matched outputs directly; a zero baseline has no epsilon floor."""
    if not baseline or len(candidate) != len(baseline):
        return dict(passed=None, reason="missing or unequal completed baseline turns", turns=[])
    rows = []
    for index, (turn, control) in enumerate(zip(candidate, baseline)):
        a, b = turn.get("repetition"), control.get("repetition")
        matched = all(turn.get(key) == control.get(key) for key in ("label", "seed"))
        if not a or not b or not matched:
            passed, ratio, threshold = None, None, None
        else:
            threshold = 1.5 * b["repeated_4gram_fraction"]
            value = a["repeated_4gram_fraction"]
            ratio = value / b["repeated_4gram_fraction"] if b["repeated_4gram_fraction"] else None
            passed = a["no_loop"] and value <= threshold
        rows.append(dict(turn=index, label=turn.get("label"), seed=turn.get("seed"),
                         matched=matched, candidate=a, baseline=b,
                         repeat4_limit=threshold, repeat4_ratio=ratio, passed=passed))
    passed = False if any(r["passed"] is False for r in rows) else (
        True if all(r["passed"] is True for r in rows) else None)
    return dict(passed=passed, turns=rows)


def quality_gates(evidence):
    evidence = evidence or {}
    nll, off_nll = evidence.get("nll"), evidence.get("off_nll")
    valid_nll = (isinstance(nll, (int, float)) and not isinstance(nll, bool) and
                 math.isfinite(nll) and nll >= 0)
    valid_off = (isinstance(off_nll, (int, float)) and not isinstance(off_nll, bool) and
                 math.isfinite(off_nll) and off_nll > 0)
    ratio = nll / off_nll if valid_nll and valid_off else None
    correct = evidence.get("mmlu_correct")
    valid_mmlu = isinstance(correct, int) and not isinstance(correct, bool) and 0 <= correct <= 57
    gates = dict(off_nll_exact=off_nll == .622784 if off_nll is not None else None,
                 nll_ratio_le_1_10=ratio <= 1.10 if ratio is not None else (
                     False if nll is not None and not valid_nll else None),
                 mmlu57=correct >= 48 if evidence.get("mmlu_n") == 57 and valid_mmlu else None)
    for name in QUALITY_BOOLEANS:
        value = evidence.get(name)
        gates[name] = value if isinstance(value, bool) else None
    verdict = False if False in gates.values() else (True if all(v is True for v in gates.values()) else None)
    return dict(gates=gates, passed=verdict, nll_ratio=ratio, evidence=evidence,
                missing=[key for key, value in gates.items() if value is None])


def c_decision(arm, baseline):
    timing, control = arm["timing"], baseline["timing"] if baseline else {}
    value, off = timing.get("active_ms_per_token"), control.get("active_ms_per_token")
    speedup = off / value if off and value and value > 0 else None
    quality = arm["quality"]
    if (not arm["complete"] or arm["performance_verified"] is not True or
            not baseline or not baseline["complete"] or baseline["performance_verified"] is not True):
        verdict = "PENDING"
    elif quality["passed"] is False or arm["repetition_vs_off"]["passed"] is False:
        verdict = "NO-GO: quality gate failed"
    elif speedup is not None and speedup < 1.10:
        verdict = "NO-GO: speedup below 10%"
    elif speedup is not None and speedup < 1.20:
        verdict = "NO-GO: required 20% speedup not reached"
    elif (speedup is not None and quality["passed"] is True and
          arm["repetition_vs_off"]["passed"] is True and arm["failures"]["no_load_failures"] is True):
        verdict = "GO: explicit option only; owner default unchanged"
    else:
        verdict = "PENDING"
    return dict(verdict=verdict, active_speedup_vs_off=speedup,
                raw_speedup_vs_off=control.get("raw_ms_per_token") / timing["raw_ms_per_token"]
                if control.get("raw_ms_per_token") and timing.get("raw_ms_per_token") else None,
                missing_quality=quality["missing"])


def load_arm(directory, name, record=None, *, comparison_root=None,
             thermal_log=None, expected_turns=None, common_fraction=.60):
    directory = Path(directory).resolve()
    record = record or {}
    paths = []
    turns_path = directory / "turns.json"
    document = read_json(turns_path) if turns_path.exists() else {}
    if not record and isinstance(document, dict):
        record = document.get("comparison", {})
    if turns_path.exists():
        paths.append(turns_path)
    stored = document if isinstance(document, list) else document.get("turns", [])
    events = directory / "events.jsonl"
    pending, errors, partial = 0, [], False
    if events.exists():
        paths.append(events)
        rows, partial = event_rows(events)
        outputs, pending, errors = event_outputs(rows)
        turns = []
        for index, (done, ids) in enumerate(outputs):
            row = dict(stored[index] if index < len(stored) else {}) | done
            row["repetition"] = metrics(ids) if ids else None
            row["token_count_matches_generated"] = len(ids) == row.get("generated", len(ids))
            turns.append(row)
    else:
        turns = []
        for stored_turn in stored:
            row = dict(stored_turn)
            if "token_ids" in row:
                row["repetition"] = metrics(row["token_ids"])
                row["token_count_matches_generated"] = len(row["token_ids"]) == row.get("generated")
            else:
                nested = row.get("repetition", {}).get("outputs", [])
                row["repetition"] = nested[0] if len(nested) == 1 else None
                row["token_count_matches_generated"] = None
            turns.append(row)
    clock = directory / "clock.json"
    if not clock.exists():
        clock = Path(record.get("clock", Path(comparison_root or directory.parent) / "clock.json"))
    host_start = read_json(clock).get("host_start_unix") if clock.exists() else None
    if clock.exists():
        paths.append(clock)
    thermal = thermal_log or record.get("decode_timing", {}).get("thermal_log")
    pauses = None
    if thermal:
        paths.append(Path(thermal))
        try:
            pauses = pause_intervals(thermal)
        except (OSError, ValueError):
            pass  # shared accounting below retains the alignment error
    turn_timing = [timing_for_turn(row, host_start, pauses) for row in turns]
    # The shared implementation also checks log coverage, including standalone
    # web turn files. An incomplete recording cannot validate either aggregate
    # speed or a per-turn adjusted figure.
    timing = decode_events_timing(turns, host_start, thermal)
    coverage = timing.get("thermal_coverage")
    for index, row in enumerate(turn_timing):
        if timing["active_decode_ms"] is None:
            row.update(thermal_pause_ms=None, active_decode_ms=None, active_ms_per_token=None)
        if timing.get("alignment_error"):
            row["alignment_error"] = timing["alignment_error"]
        if coverage and len(coverage["windows"]) == len(turn_timing):
            row["thermal_coverage"] = coverage["windows"][index]
    status_path = directory / "status.json"
    if not status_path.exists() and isinstance(document, list) and turns:
        status_path = directory / f"turn{len(turns)-1}_status.json"
    after = record.get("status_after") or (read_json(status_path) if status_path.exists() else {})
    if status_path.exists():
        paths.append(status_path)
    before = record.get("status_before", {})
    before_path = directory / "status_before.json"
    if not before and before_path.exists():
        before = read_json(before_path)
        paths.append(before_path)
    startup_path = directory / "startup.json"
    startup = read_json(startup_path) if startup_path.exists() else {}
    if startup_path.exists():
        paths.append(startup_path)
    ready = startup.get("ready", {}) or (document.get("server", {}) if isinstance(document, dict) else {})
    config_path = directory / "config.json"
    config = read_json(config_path) if config_path.exists() else {}
    if config_path.exists():
        paths.append(config_path)
    ready = ready or config.get("ready", {})
    command = startup.get("command") or config.get("cmd") or (
        document.get("command", []) if isinstance(document, dict) else [])
    def option(name):
        if not isinstance(command, list) or name not in command:
            return None
        index = command.index(name)
        return command[index + 1] if index + 1 < len(command) else None
    policy = record.get("policy", {}) or startup.get("policy", {})
    mode = policy.get("mode") or config.get("resident_only") or option("--resident-only")
    if mode is None:
        match = re.search(r"resident-only=(\w+)", after.get("route", ""))
        mode = match[1] if match else None
    dynamic = config.get("mask_cache") == "dynamic" or option("--mask-cache") == "dynamic"
    if not dynamic and "cache_fixed" in after and "cache_frozen" in after:
        dynamic = after["cache_fixed"] is False and after["cache_frozen"] is False
    spec = ready.get("speculation", {})
    k = policy.get("draft_tokens", spec.get("draft_tokens") if spec.get("enabled") else 0)
    powers = [record.get(key, {}).get("power_profile") for key in ("start_power", "end_power")]
    verified_power = all(p == "performance" for p in powers) if all(powers) else None
    if verified_power is None and config.get("power_profile"):
        verified_power = config["power_profile"] == "performance"
    expected = (expected_turns if expected_turns is not None else
                (3 if isinstance(document, list) else record.get("turns")))
    complete = bool(turns) and not pending and not errors and not partial
    if expected is not None:
        complete &= len(turns) == expected
    if comparison_root:
        complete &= bool(record)
    elif not isinstance(document, list):
        complete &= bool(after)
    no_loop = all(row.get("repetition") and row["repetition"]["no_loop"] for row in turns) if turns else None
    if any(row.get("token_count_matches_generated") is False for row in turns):
        complete = False
    summarized = [dict(index=i, label=row.get("label"), seed=row.get("seed"),
                       generated=row.get("generated"), decode_steps=row.get("decode_steps"),
                       timing=turn_timing[i], repetition=row.get("repetition"),
                       cycle=cycle_report([row], turn_timing[i], common_fraction))
                  for i, row in enumerate(turns)]
    steps = timing["decode_steps"]
    hit_rate = (sum(float(r.get("decode_hit_rate", 0)) * int(r.get("decode_steps", 0))
                    for r in turns) / steps if steps else None)
    return dict(name=name, directory=str(directory), complete=complete,
                expected_turns=expected, completed_turns=len(turns), pending_tokens=pending,
                partial_event_line=partial, errors=errors,
                engine_pid=record.get("pid"), session=record.get("session"),
                policy=policy, spec_k=k, resident_mode=mode, mask_dynamic=dynamic,
                ready=ready, performance_verified=verified_power,
                power_evidence=record.get("start_power", {}) | {"end": record.get("end_power", {})},
                timing=timing, cycle=cycle_report(turns, timing, common_fraction),
                failures=failure_report(before, after), no_loop=no_loop, turns=summarized,
                hit_rate=hit_rate,
                inputs=[str(p) for p in paths])


def discover(inputs, thermal_log=None, expected_turns=None, common_fraction=.60):
    arms, comparisons = [], []
    for value in inputs:
        alias, separator, raw_path = value.partition("=")
        path = Path(raw_path if separator else value).resolve()
        comparison_path = path / "comparison.json"
        if comparison_path.exists():
            comparison = read_json(comparison_path)
            comparisons.append(dict(path=str(comparison_path), **comparison))
            records = {r["name"]: r for r in comparison.get("results", [])}
            for name in comparison.get("order", records):
                arms.append(load_arm(path / name, f"{alias}/{name}" if separator else name,
                    records.get(name), comparison_root=path, thermal_log=thermal_log,
                    expected_turns=expected_turns, common_fraction=common_fraction))
        else:
            arms.append(load_arm(path, alias if separator else path.name,
                thermal_log=thermal_log, expected_turns=expected_turns,
                common_fraction=common_fraction))
    names = [arm["name"] for arm in arms]
    if len(set(names)) != len(names):
        raise ValueError("duplicate arm names; use ALIAS=PATH inputs")
    return arms, comparisons


def build_report(inputs, *, baseline_name="off", quality=None, thermal_log=None,
                 expected_turns=None, common_fraction=.60):
    arms, comparisons = discover(inputs, thermal_log, expected_turns, common_fraction)
    baseline = next((arm for arm in arms if arm["name"] == baseline_name), None)
    for arm in arms:
        arm["repetition_vs_off"] = repetition_gate(arm["turns"], baseline["turns"] if baseline else None)
        arm["quality"] = quality_gates((quality or {}).get(arm["name"]))
        if "tau" in arm["name"] or arm["policy"].get("tau") is not None:
            arm["C_decision"] = c_decision(arm, baseline)
    eligible = [arm for arm in arms if arm["spec_k"] and arm["resident_mode"] == "mask" and
                arm["mask_dynamic"] and arm["complete"] and
                arm["performance_verified"] is True and arm["no_loop"] is True and
                arm["repetition_vs_off"]["passed"] is True and
                arm["failures"]["no_load_failures"] is True and
                arm["timing"]["active_ms_per_token"] is not None]
    selected = min(eligible, key=lambda arm: arm["timing"]["active_ms_per_token"]) if eligible else None
    paths = {Path(p) for arm in arms for p in arm["inputs"]}
    paths.update(Path(c["path"]) for c in comparisons)
    return dict(schema=1, baseline=baseline_name, arms=arms,
                comparisons=[{key: c.get(key) for key in ("path", "engine_pid", "order",
                    "cache_carries_between_arms", "route_selection")} for c in comparisons],
                D_selection=dict(selected=selected["name"] if selected else None,
                    rule="owner: dynamic mask + speculation, minimum active ms/token among complete "
                         "performance arms passing loop/repetition/load gates",
                    selected_quality=selected["quality"] if selected else None,
                    scope="speed/repetition selection does not claim missing MMLU/long-context gates passed",
                    provisional_before_completed_D="mask+k2, ONECB on; GPU route determined independently"),
                inputs=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                        for p in sorted(paths) if p.exists()],
                limitations=["Active time subtracts cooling using host receipt boundaries, not GPU stamps.",
                    "Raw stage timers retain cooling; no unsupported per-stage subtraction is made.",
                    "Common acceptance costs are arithmetic comparisons; generated routes/unions can differ.",
                    "Missing quality evidence stays pending; C requires all gates and >=20% speedup.",
                    "Four-gram comparisons require matching labels/seeds and completed output counts.",
                    "MTP pins occupy 384 of the configured slots; hit-rate changes alone "
                    "do not isolate their causal cost."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="PATH or ALIAS=PATH; comparison root or arm/web directory")
    parser.add_argument("--out", type=Path, required=True,
                        help="new JSON receipt path; existing files are refused")
    parser.add_argument("--baseline", default="off")
    parser.add_argument("--quality", type=Path, help="explicit per-arm quality gate JSON")
    parser.add_argument("--thermal-log", type=Path)
    parser.add_argument("--expected-turns", type=int)
    parser.add_argument("--common-accept-fraction", type=float, default=.60)
    args = parser.parse_args()
    if not math.isfinite(args.common_accept_fraction) or not 0 <= args.common_accept_fraction <= 1:
        parser.error("common acceptance fraction must be in [0,1]")
    if args.expected_turns is not None and args.expected_turns < 1:
        parser.error("expected turns must be positive")
    if args.out.exists():
        parser.error("refusing to replace an existing result; use a fresh --out")
    quality = read_json(args.quality) if args.quality else None
    report = build_report(args.inputs, baseline_name=args.baseline, quality=quality,
        thermal_log=args.thermal_log, expected_turns=args.expected_turns,
        common_fraction=args.common_accept_fraction)
    if args.quality:
        report["inputs"].append(dict(path=str(args.quality.resolve()),
            sha256=hashlib.sha256(args.quality.read_bytes()).hexdigest()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(out=str(args.out.resolve()), arms=len(report["arms"]),
                         D_selected=report["D_selection"]["selected"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
