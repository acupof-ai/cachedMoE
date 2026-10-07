#!/usr/bin/env python3
"""Two head formats in one engine/session; stop immediately on changed final IDs.

Run inside thermal_guard with the user web stopped. Both arms retain the copy
and slab reservation: this isolates head speed, not the separate 5500→5400
cache cost. Native cache/IO residency carries into FP8, as in spec_compare.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
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
    args = parser.parse_args()
    out = args.out.resolve()
    for root in (args.model, args.mirror):
        if out.is_relative_to(Path(root).resolve()):
            raise ValueError("output must be outside checkpoint roots")
    script = json.loads(args.script.read_text())
    if len(script["turns"]) != 8:
        raise ValueError("require the original eight-turn workload")
    out.mkdir(exist_ok=False)
    env = runtime_defaults.profile_environment("production", os.environ, overrides={
        "CACHEDMOE_DSPARK_HEAD_FP8": "1", "CACHEDMOE_DSPARK_PROFILE": "0",
        "CACHEDMOE_MODEL_DIR": args.model, "CACHEDMOE_MODEL_MIRRORS": args.mirror,
        "CACHEDMOE_SPEC_DIAGNOSTICS": str(out / "diagnostics.jsonl"),
    })
    server_args = argparse.Namespace(
        exe=str(args.exe.resolve()), model=args.model, max_context=runtime_defaults.MAX_CONTEXT,
        cache_gb=0, cache_slots=runtime_defaults.CACHE_SLOTS, shader_dir=args.shader_dir,
        require_sources=2, profile=False,
        env=[f"{key}={value}" for key, value in env.items() if runtime_env.is_control(key)],
        serve_arg=["--mirror", args.mirror, "--resident-only", "mask", "--mask-cache", "dynamic",
                   "--gpu-prefill-min", str(runtime_defaults.GPU_PREFILL_MIN), "--dspark", "--spec-k", "2",
                   "--spec-top-k", str(runtime_defaults.ACCEPT_TOP_K), "--allow-spec-switch",
                   "--kv-dir", str(out / "kv"), "--kv-max-gb", str(runtime_defaults.KV_DISK_GB)])
    config = bench_configuration(server_args, str(out))
    encoding = chat.load_encoding(config.model)
    policy = runtime_defaults.ThermalPolicy()
    sensors = discover_sensors(required_nvme=runtime_defaults.PRODUCTION_READ_SOURCES)
    server = CheckedServer(server_args, str(out), config=config)
    results, reference, verdict = [], [], "PENDING"
    save(out / "manifest.json", dict(pid=server.p.pid, command=server.cmd,
         script_sha256=hashlib.sha256(args.script.read_bytes()).hexdigest(),
         cache_reservation_fixed_across_arms=True, default_baseline_slots=5500, actual_slots=5400,
         target_bf16=True, k=2, gpu_route=0, onecb=1, acceptance_top_k=4,
         head_profiling=False, target_profiling=False, thermal_thresholds=policy.record()))
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
                if fp8 and (server.difference or server.ids != reference[index]):
                    differences.append(dict(turn=index, first_difference=server.difference,
                                            expected_count=len(reference[index]), actual_count=len(server.ids)))
                    verdict = "QUALITY_NO_GO"
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
            native, fp8 = (r["timing"]["raw_ms_per_token"] for r in results)
            verdict = "SPEED_GO_OWNER_DECISION_PENDING" if fp8 < .97 * native else "SPEED_NO_GO"
        save(out / "comparison.json", dict(verdict=verdict, results=results, same_engine_pid=server.p.pid,
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
