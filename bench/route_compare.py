#!/usr/bin/env python3
"""Compare decode routes consecutively in one engine and session.

Run under a thermal supervisor after stopping the user web engine. KV resets
between arms; expert residency carries over. Report this order effect instead
of treating the second arm as an independent cold-cache measurement.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import chat
from hitrate_bench import BenchServer, capture_status, round_stats
import provenance
from thermal_metrics import decode_timing


def power_state():
    return provenance.power_state()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shader-dir", required=True)
    parser.add_argument("--power-profile", choices=("performance", "power-saver"), required=True)
    parser.add_argument("--arms", default="off,mask")
    parser.add_argument("--thermal-log", default=os.environ.get("DEEPMOE_THERMAL_LOG"))
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    arms = args.arms.split(",")
    policies = []
    for name in arms:
        if name in ("off", "mask"):
            policies.append((name, {"op": "set_decode_route", "mode": name}))
        elif name.startswith("tau"):
            tau = float(name[3:])
            if not 0 <= tau <= 1:
                parser.error("tau must be in [0,1]")
            policies.append((name, {"op": "set_decode_route", "mode": "mask", "tau": tau}))
        else:
            parser.error("arms must be off, mask, or tau<number>")
    if len(set(arms)) != len(arms):
        parser.error("each configuration runs once")
    assert power_state()["power_profile"] == args.power_profile
    server_args = argparse.Namespace(
        exe=str(args.exe.resolve()), max_context=4096, cache_gb=0, cache_slots=5500,
        shader_dir=args.shader_dir, require_sources=2, env=[],
        serve_arg=["--resident-only", "off", "--mask-cache", "dynamic",
                   "--gpu-prefill-min", "16", "--allow-route-switch"])
    script = json.loads(args.script.read_text())
    server = BenchServer(server_args, str(out))
    pid = server.p.pid
    enc = chat.load_encoding()
    results = []
    try:
        for name, request in policies:
            arm = out / name
            arm.mkdir()
            server.events.close()
            server.events = (arm / "events.jsonl").open("w", encoding="utf-8")
            server.send(request)
            policy = server.read_event()
            if policy.get("event") != "decode_route":
                raise RuntimeError(policy)
            assert policy["mode"] == request["mode"]
            assert policy["tau"] == request.get("tau")
            server.send({"op": "reset"})
            assert server.read_event()["event"] == "reset"
            begin = power_state()
            assert begin["power_profile"] == args.power_profile
            provenance.write(arm, exe=args.exe, env=os.environ, shader_dir=args.shader_dir)
            options = argparse.Namespace(think=False, temp=1., top_p=.95,
                                         max_tokens=256, seed=None, system="")
            client = chat.Chat(server, enc, options)
            print("ARM", name, "PID", pid, "POWER", begin, flush=True)
            chat.run_script(client, server, script, arm / "transcript.md", arm / "turns.json")
            status = capture_status(server, str(arm))
            summary = round_stats(arm / "events.jsonl")
            timing = decode_timing(arm / "events.jsonl", out / "clock.json", args.thermal_log)
            finish = power_state()
            assert finish["power_profile"] == args.power_profile
            record = dict(name=name, pid=pid, session="default", policy=policy,
                          start_power=begin, end_power=finish, **summary)
            record["decode_timing"] = timing
            results.append(record)
            doc = json.loads((arm / "turns.json").read_text())
            doc.update(status=status, repetition=summary["repetition"],
                       command=server.cmd, comparison=record)
            (arm / "turns.json").write_text(json.dumps(doc, indent=2) + "\n")
            provenance.finish(arm)
            (out / "comparison.json").write_text(json.dumps(dict(
                engine_pid=pid, cache_carries_between_arms=True,
                order=arms, results=results), indent=2) + "\n")
            print("ARM_DONE", name, "ms/token", summary["decode_ms"] / summary["decode_steps"],
                  "active_ms/token", timing["active_ms_per_token"],
                  "no_loop", summary["repetition"]["no_loop"], flush=True)
    finally:
        server.close()
        server.events.close()
        provenance.finish(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
