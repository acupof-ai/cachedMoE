#!/usr/bin/env python3
"""Compare DSpark controls in one engine and default session.

Each arm resets KV, changes policy at the completed request boundary, then
feeds the same script. Expert residency and IO counters carry between arms.
Cooling-adjusted host time is an estimate; raw wall time is always retained.
Run under bench/thermal_guard.py with the user web engine stopped.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import runtime_env

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import chat
from hitrate_bench import BenchServer, capture_status, round_stats
import provenance
from thermal_metrics import decode_timing


def parse_arm(name):
    match = re.fullmatch(r"k([1-5])_(gpu|onecb_cpu_route|serial|best)", name)
    if not match:
        raise ValueError("arms must be k<1..5>_gpu, _onecb_cpu_route, _serial or _best")
    k, variant = match.groups()
    return int(k), variant


def choose_route(results, forced="auto"):
    if forced != "auto":
        return forced == "gpu", {"rule": "explicit route override", "route": forced}
    controls = {row["name"]: row for row in results}
    wanted = ("k2_gpu", "k2_onecb_cpu_route")
    if not all(name in controls for name in wanted):
        raise ValueError("automatic route choice needs both completed k2 ONECB controls first")
    costs, raw_costs, no_loop = {}, {}, {}
    for name in wanted:
        row = controls[name]
        cost = row["decode_timing"]["active_ms_per_token"]
        if cost is None:
            raise ValueError("automatic route choice requires aligned thermal timing")
        if row["decode_timing"].get("engine_decode_boundaries") is not True:
            raise ValueError("automatic route choice requires recorded engine decode boundaries")
        raw = row["decode_timing"].get("raw_ms_per_token")
        if raw is None or raw <= 0:
            raise ValueError("automatic route choice requires measured raw decode timing")
        # Preserve both measured controls, even when one fails the loop gate.
        # A failed control cannot remove the paired cost threshold.
        costs[name] = cost
        raw_costs[name] = raw
        no_loop[name] = row["repetition"]["no_loop"] is True
    if not any(no_loop.values()):
        raise ValueError("both k2 route controls looped; stop before testing longer drafts")
    # Inside the documented ±3% floor, retain CPU routing. The report still
    # carries per-op evidence and common-acceptance cycle costs for review.
    use_gpu = (no_loop["k2_gpu"] and
        costs["k2_gpu"] < .97 * costs["k2_onecb_cpu_route"] and
        raw_costs["k2_gpu"] < .97 * raw_costs["k2_onecb_cpu_route"])
    return use_gpu, dict(rule="GPU must not loop and needs >3% lower paired raw and adjusted ms/token",
                         active_costs=costs, raw_costs=raw_costs, no_loop=no_loop,
                         route="gpu" if use_gpu else "cpu")


def spec_totals(events_path):
    totals = {}
    for line in Path(events_path).read_text().splitlines():
        event = json.loads(line)
        if event.get("event") == "done":
            for key, value in event.get("speculation", {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    totals[key] = totals.get(key, 0) + value
    return totals


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def startup_environment(inherited, requested):
    """Initialize the complete resource set and validate effective CLI aliases."""
    startup = dict(CACHEDMOE_DSPARK_ONECB="1", CACHEDMOE_BATCH_GPU_ROUTE="1",
                   CACHEDMOE_DSPARK_MEGA="0", CACHEDMOE_DSPARK_PROFILE="0",
                   CACHEDMOE_DSPARK_TRIM_TAIL="1", CACHEDMOE_SPEC_GPU_READOUT="1",
                   CACHEDMOE_MASK_DYNAMIC_LRU="1", CACHEDMOE_IO_ENGRAM_DEADLINE="0",
                   CACHEDMOE_SPEC_DIAGNOSTICS="")
    env = dict(inherited)
    runtime_env.apply_overrides(env, startup)
    overrides = {}
    for item in requested:
        if "=" not in item:
            raise ValueError("--env requires KEY=VALUE")
        key, value = item.split("=", 1)
        overrides[key] = value
    runtime_env.apply_overrides(env, overrides)
    for key, expected in startup.items():
        selected = runtime_env.resolve(key, env)
        if key != "CACHEDMOE_SPEC_DIAGNOSTICS" and selected.value != expected:
            raise ValueError(f"{key} must be {expected} for complete startup resources")
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shader-dir", required=True)
    parser.add_argument("--power-profile", choices=("performance", "power-saver"), required=True)
    parser.add_argument("--arms", default="k2_gpu,k2_onecb_cpu_route,k2_serial,k3_best,k5_best")
    parser.add_argument("--best-route", choices=("auto", "gpu", "cpu"), default="auto")
    parser.add_argument("--thermal-log", default=runtime_env.getenv("CACHEDMOE_THERMAL_LOG"))
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--serve-arg", action="append", default=[])
    args = parser.parse_args()
    arms = args.arms.split(",")
    try:
        policies = [(name, *parse_arm(name)) for name in arms]
    except ValueError as exc:
        parser.error(str(exc))
    if len(set(arms)) != len(arms):
        parser.error("each configuration runs once")
    if any(variant == "best" for _, _, variant in policies) and args.best_route == "auto":
        first_best = next(i for i, (_, _, variant) in enumerate(policies) if variant == "best")
        if not {"k2_gpu", "k2_onecb_cpu_route"}.issubset(set(arms[:first_best])):
            parser.error("both k2 ONECB route controls must precede automatic _best arms")
    if provenance.power_state()["power_profile"] != args.power_profile:
        parser.error("actual power profile differs from the requested profile")
    # Use the complete resource set at startup. The setter selects arms only
    # after initialization and never lazily builds a measured arm's pipelines.
    try:
        engine_env = startup_environment(os.environ, args.env)
    except ValueError as exc:
        parser.error(str(exc))
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    server_args = argparse.Namespace(
        exe=str(args.exe.resolve()), max_context=4096, cache_gb=0, cache_slots=5500,
        shader_dir=args.shader_dir, require_sources=2,
        env=[f"{key}={value}" for key, value in engine_env.items()
             if runtime_env.is_control(key) or os.environ.get(key) != value],
        serve_arg=["--resident-only", "mask", "--mask-cache", "dynamic",
                   "--gpu-prefill-min", "16", "--dspark", "--spec-k", "5",
                   "--spec-top-k", "4", "--allow-spec-switch", *args.serve_arg])
    script = json.loads(args.script.read_text())
    encoding = chat.load_encoding()
    server = BenchServer(server_args, str(out))
    pid = server.p.pid
    results, route_selection = [], None
    try:
        ready = server.ready.get("speculation", {})
        if not ready.get("enabled") or ready.get("draft_tokens") != 5:
            raise RuntimeError("startup did not initialize k=5 DSpark")
        for name, k, variant in policies:
            onecb, route = variant != "serial", variant == "gpu"
            if variant == "best":
                route, route_selection = choose_route(results, args.best_route)
            arm = out / name
            arm.mkdir()
            server.events.close()
            server.events = (arm / "events.jsonl").open("w", encoding="utf-8")
            server.send({"op": "reset", "session": "default"})
            if server.read_event().get("event") != "reset":
                raise RuntimeError("default session reset failed")
            request = dict(op="set_spec_config", session="default", draft_tokens=k,
                           onecb=onecb, gpu_route=route)
            server.send(request)
            policy = server.read_event()
            if policy.get("event") != "spec_config" or any(
                    policy.get(key) != request[key] for key in ("draft_tokens", "onecb", "gpu_route")):
                raise RuntimeError(policy)
            if policy.get("main_paths") != 1 or policy.get("accept_top_k") != 4:
                raise RuntimeError("verification policy changed")
            start_power = provenance.power_state()
            if start_power["power_profile"] != args.power_profile:
                raise RuntimeError("power profile changed between arms")
            provenance.write(arm, exe=args.exe, env=runtime_env, shader_dir=args.shader_dir)
            save_json(arm / "startup.json", dict(ready=server.ready, policy=policy,
                                                command=server.cmd, engine_pid=pid))
            status_before = capture_status(server, str(arm))
            save_json(arm / "status_before.json", status_before)
            options = argparse.Namespace(think=False, temp=1., top_p=.95,
                                         max_tokens=256, seed=None, system="")
            client = chat.Chat(server, encoding, options)
            print("ARM", name, "PID", pid, "POLICY", policy, flush=True)
            chat.run_script(client, server, script, arm / "transcript.md", arm / "turns.json")
            status_after = capture_status(server, str(arm))
            save_json(arm / "status_after.json", status_after)
            summary = round_stats(arm / "events.jsonl")
            timing = decode_timing(arm / "events.jsonl", out / "clock.json", args.thermal_log)
            end_power = provenance.power_state()
            if end_power["power_profile"] != args.power_profile:
                raise RuntimeError("power profile changed during an arm")
            record = dict(name=name, pid=pid, session="default", policy=policy,
                          arm_dir=str(arm), clock=str(out / "clock.json"),
                          start_power=start_power, end_power=end_power,
                          status_before=status_before, status_after=status_after, **summary)
            record["decode_timing"] = timing
            record["speculation"] = spec_totals(arm / "events.jsonl")
            results.append(record)
            doc = json.loads((arm / "turns.json").read_text())
            doc.update(status=status_after, repetition=summary["repetition"],
                       command=server.cmd, comparison=record)
            save_json(arm / "turns.json", doc)
            provenance.finish(arm)
            save_json(out / "comparison.json", dict(
                engine_pid=pid, session="default", cache_carries_between_arms=True,
                order=arms, route_selection=route_selection, results=results))
            print("ARM_DONE", name, "raw_ms/token", timing["raw_ms_per_token"],
                  "active_ms/token", timing["active_ms_per_token"],
                  "no_loop", summary["repetition"]["no_loop"], flush=True)
    finally:
        server.close()
        server.events.close()
        provenance.finish(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
