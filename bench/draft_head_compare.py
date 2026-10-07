#!/usr/bin/env python3
"""Isolate draft head math or cost; never infer a production speed winner.

Run inside thermal_guard with the user web stopped. Both arms retain the copy
and slab reservation: this isolates head speed, not the separate 5500→5400
cache cost. Exact target routing removes asynchronous expert availability from
the output check. Frozen-cache cost controls require the identical settled
slot map throughout both arms. Neither is a dynamic-mask production speed test.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import chat
import provenance
import runtime_defaults
import runtime_env
from hitrate_bench import BenchServer, bench_configuration, capture_status, round_stats
from thermal_metrics import decode_timing
from thermal_guard import ProfileMonitor, discover_sensors, sample, guarded_temperatures


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def cache_identity(state, expected=None):
    if state.get("event") != "benchmark_cache_state" or state.get("frozen") is not True:
        raise ValueError("missing frozen cache control")
    if any(state.get(key) != 0 for key in
           ("free", "filling", "queued_requests", "inflight_chunks", "fills_failed", "evictions")):
        raise ValueError("cache control has pending IO, failures or replacements")
    slots = state.get("slots", [])
    if not slots or len(slots) != state.get("resident"):
        raise ValueError("incomplete cache slot map")
    if any(not isinstance(row, list) or len(row) != 4 or
           any(type(x) is not int or x < 0 for x in row) or row[3] not in (0, 1)
           for row in slots):
        raise ValueError("invalid cache slot geometry")
    if type(state.get("fills_started")) is not int or state["fills_started"] < len(slots):
        raise ValueError("invalid cache fill count")
    if [row[0] for row in slots] != list(range(len(slots))):
        raise ValueError("cache map must cover every slot once")
    keys = [(row[1], row[2]) for row in slots]
    if len(set(keys)) != len(slots):
        raise ValueError("duplicate cached expert")
    digest = hashlib.sha256(json.dumps(slots, separators=(",", ":")).encode()).hexdigest()
    value = dict(slot_sha256=digest, resident=state["resident"], fills_started=state["fills_started"])
    if expected is not None and value != expected:
        raise ValueError("expert cache changed between controlled measurements")
    return value


def capture_cache(server, path, expected=None):
    server.send({"op": "benchmark_cache_state", "session": "default"})
    state = server.read_event()
    save(path, state)
    return cache_identity(state, expected)


def check_target_mode(done, frozen):
    expected_mode = "mask-spec" if frozen else "off-spec"
    if (done.get("decode_mode") != expected_mode or
            done.get("speculation", {}).get("cycles", 0) <= 0):
        raise ValueError("control did not run the requested speculative target mode")


def check_heat_file(path, model):
    config = json.loads((Path(model) / "config.json").read_text())
    config = config.get("text_config", config)
    layers, experts = config["num_hidden_layers"], config["n_routed_experts"]
    entries = re.findall(r"\{\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\}", path.read_text())
    keys = [(int(layer), int(expert)) for layer, expert, _ in entries]
    expected = {(layer, expert) for layer in range(layers) for expert in range(experts)}
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("heat control must list every main expert once; no static fallback")


class CheckedServer(BenchServer):
    def __init__(self, *args, **kwargs):
        self.expected = None
        self.ids = []
        self.difference = None
        super().__init__(*args, **kwargs)

    def start_turn(self, expected=None):
        self.expected, self.ids, self.difference = expected, [], None

    def read_event(self):
        event = super().read_event()
        if event.get("event") == "token":
            position = len(self.ids)
            token = event["id"]
            self.ids.append(token)
            if self.expected is not None and self.difference is None:
                reference = self.expected[position] if position < len(self.expected) else None
                if token != reference:
                    self.difference = dict(position=position, reference=reference, candidate=token)
                    self.send({"op": "cancel"})
        return event


def cold_start(policy, sensors, timeout_s=900):
    deadline = time.monotonic() + timeout_s
    announced = time.monotonic()
    with ProfileMonitor() as monitor:
        while True:
            values = sample(sensors, monitor)
            if values["ac"] != 1 or values["power_profile"] != runtime_defaults.DEFAULT_POWER_PROFILE:
                raise RuntimeError("AC or balanced profile changed")
            if policy.cold(guarded_temperatures(values)):
                return values
            if time.monotonic() > deadline:
                raise RuntimeError("head comparison did not meet cold-start gates")
            if time.monotonic() - announced > 30:
                print("COOLING", guarded_temperatures(values), flush=True)
                announced = time.monotonic()
            time.sleep(.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mirror", required=True)
    parser.add_argument("--shader-dir", required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--control", choices=("exact-target", "frozen-cache"),
                        default="exact-target")
    parser.add_argument("--heat-file", type=Path)
    parser.add_argument("--limit-turns", type=int, default=1)
    parser.add_argument("--limit-tokens", type=int, default=64)
    parser.add_argument("--trace", type=Path, help="optional dispatch trace; not a speed decision")
    args = parser.parse_args()
    out = args.out.resolve()
    for root in (args.model, args.mirror):
        if out.is_relative_to(Path(root).resolve()):
            raise ValueError("output must be outside checkpoint roots")
    script = json.loads(args.script.read_text())
    if len(script["turns"]) != 8:
        raise ValueError("require the original eight-turn workload")
    if not 1 <= args.limit_turns <= 8 or args.limit_tokens < 1:
        raise ValueError("invalid short-control limits")
    frozen = args.control == "frozen-cache"
    if frozen and (args.heat_file is None or not args.heat_file.is_file()):
        raise ValueError("frozen cost control requires an explicit recorded heat file")
    if frozen:
        check_heat_file(args.heat_file, args.model)
    script = dict(script, turns=[dict(turn, max_tokens=min(turn["max_tokens"], args.limit_tokens))
                                for turn in script["turns"][:args.limit_turns]])
    out.mkdir(exist_ok=False)
    overrides = {
        "CACHEDMOE_DSPARK_HEAD_FP8": "1", "CACHEDMOE_DSPARK_PROFILE": "0",
        "CACHEDMOE_MODEL_DIR": args.model, "CACHEDMOE_MODEL_MIRRORS": args.mirror,
        "CACHEDMOE_SPEC_DIAGNOSTICS": str(out / "diagnostics.jsonl"),
    }
    if frozen:
        overrides["CACHEDMOE_HEAT_FILE"] = str(args.heat_file.resolve())
    env = runtime_defaults.profile_environment("production", os.environ,
        mask_cache="fixed" if frozen else "dynamic", overrides=overrides)
    server_args = argparse.Namespace(
        exe=str(args.exe.resolve()), model=args.model, max_context=runtime_defaults.MAX_CONTEXT,
        cache_gb=0, cache_slots=runtime_defaults.CACHE_SLOTS, shader_dir=args.shader_dir,
        require_sources=2, profile=False,
        env=[f"{key}={value}" for key, value in env.items() if runtime_env.is_control(key)],
        serve_arg=["--mirror", args.mirror, "--resident-only", "mask" if frozen else "off",
                   "--mask-cache", "fixed" if frozen else "dynamic",
                   "--gpu-prefill-min", str(runtime_defaults.GPU_PREFILL_MIN), "--dspark", "--spec-k", "2",
                   "--spec-top-k", str(runtime_defaults.ACCEPT_TOP_K), "--allow-spec-switch",
                   "--kv-dir", str(out / "kv"), "--kv-max-gb", str(runtime_defaults.KV_DISK_GB)])
    config = bench_configuration(server_args, str(out))
    if args.trace:
        server_args.serve_arg += ["--trace", str(args.trace.resolve())]
    encoding = chat.load_encoding(config.model)
    policy = runtime_defaults.ThermalPolicy()
    sensors = discover_sensors(required_nvme=runtime_defaults.PRODUCTION_READ_SOURCES)
    server = CheckedServer(server_args, str(out), config=config)
    results, reference, verdict = [], [], "PENDING"
    common_cache = None
    save(out / "manifest.json", dict(pid=server.p.pid, command=server.cmd,
         script_sha256=hashlib.sha256(args.script.read_bytes()).hexdigest(),
         cache_reservation_fixed_across_arms=True, default_baseline_slots=5500, actual_slots=5400,
         target_bf16=True, k=2, gpu_route=0, onecb=1, acceptance_top_k=4,
         head_profiling=False, target_profiling=False, thermal_thresholds=policy.record()))
    save(out / "control.json", dict(kind=args.control, production_speed_test=False,
        target_experts_exact=not frozen, cache_identity_fixed=frozen,
        limit_turns=args.limit_turns, limit_tokens=args.limit_tokens,
        effective_script=script, heat_file=str(args.heat_file) if frozen else None,
        heat_sha256=hashlib.sha256(args.heat_file.read_bytes()).hexdigest() if frozen else None))
    try:
        for name, fp8 in (("native_bf16", False), ("row_fp8", True)):
            arm = out / name
            arm.mkdir()
            server.events.close()
            server.events = (arm / "events.jsonl").open("x")
            server.send({"op": "reset", "session": "default"})
            if server.read_event().get("event") != "reset":
                raise RuntimeError("default session reset failed")
            server.send({"op": "set_draft_head", "session": "default", "fp8": fp8})
            selected = server.read_event()
            if selected != {"event": "draft_head", "fp8": fp8, "target_bf16": True}:
                raise RuntimeError(selected)
            save(arm / "cold_start.json", cold_start(policy, sensors))
            save(arm / "status_before.json", capture_status(server, str(arm)))
            if frozen:
                common_cache = capture_cache(server, arm / "cache_before.json", common_cache)
            options = argparse.Namespace(think=False, temp=runtime_defaults.REQUEST_DEFAULTS["temperature"],
                top_p=runtime_defaults.REQUEST_DEFAULTS["top_p"], max_tokens=512, seed=None, system="")
            client = chat.Chat(server, encoding, options)
            differences, outputs = [], []
            print("ARM", name, "PID", server.p.pid, flush=True)
            for index, turn in enumerate(script["turns"]):
                server.start_turn(reference[index] if fp8 else None)
                chat.run_script(client, server, {"turns": [turn]}, arm / f"turn_{index:02}_transcript.md",
                                arm / f"turn_{index:02}_stats.json")
                outputs.append(server.ids.copy())
                events = [json.loads(line) for line in Path(server.events.name).read_text().splitlines()]
                check_target_mode([event for event in events if event.get("event") == "done"][-1], frozen)
                if frozen:
                    capture_cache(server, arm / f"turn_{index:02}_cache_after.json", common_cache)
                if fp8 and (server.difference or server.ids != reference[index]):
                    differences.append(dict(turn=index, first_difference=server.difference,
                                            expected_count=len(reference[index]), actual_count=len(server.ids)))
                    verdict = "CONTROLLED_FINAL_IDS_FAIL"
                    break
                if "dropped after" in (out / "serve.log").read_text():
                    raise RuntimeError("checkpoint read source dropped")
            if not fp8:
                reference = outputs
            save(arm / "output_ids.json", outputs)
            save(arm / "status_after.json", capture_status(server, str(arm)))
            result = dict(name=name, completed_turns=len(outputs), final_id_differences=differences,
                timing=decode_timing(arm / "events.jsonl", out / "clock.json",
                                     runtime_env.getenv("CACHEDMOE_THERMAL_LOG")),
                **round_stats(arm / "events.jsonl"))
            results.append(result)
            save(out / "comparison.json", dict(verdict=verdict, results=results, same_engine_pid=server.p.pid))
            print("ARM_DONE", name, result["completed_turns"], result["timing"], flush=True)
            if differences:
                break
        if verdict == "PENDING":
            verdict = "CONTROLLED_FINAL_IDS_PASS"
        save(out / "comparison.json", dict(verdict=verdict, results=results, same_engine_pid=server.p.pid,
            control=args.control, production_speed_test=False, frozen_cache_identity=common_cache,
            equal_memory_reservation=True, cache_cost_against_5500_not_measured_in_this_pair=True))
    finally:
        server.close()
        server.events.close()
        server.log.close()
        save(out / "shutdown.json", dict(engine_rc=server.p.returncode, private_kv_dir=str(out / "kv")))
        provenance.finish(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
