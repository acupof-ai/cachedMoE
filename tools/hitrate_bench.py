#!/usr/bin/env python3
"""Expert-cache hit-rate and stall bench over `deepmoe serve` (Track R1,
docs/p4_hitrate.md).

    .venv/Scripts/python.exe tools/hitrate_bench.py --script bench/results/hitrate/long_turns.json \
        --out bench/results/hitrate/auto [--cache-gb N] [--exe build/deepmoe.exe] \
        [--shader-dir build/shaders] [--serve-arg=--gpu-prefill-min --serve-arg=256] [--env K=V]

One serve process runs a chat.py-format script (the same renderer, the same KV
continuation), with `--profile` and `DEEPMOE_ROUTE_DUMP` on. The out directory gets

    events.jsonl    every protocol event, with the host's receive time
    profile.jsonl   the engine's design 13.1 record, one line per decode step
    route.bin       the engine's routing dump (runtime/engine.h), one record per step
    turns.json      chat.py's per-turn stats plus the step layout of each request
    transcript.md   the conversation

`tools/hitrate_sim.py OUT` turns them into the per-128-step curve and replays the
same routing through tools/cache_sim.py's LRU.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import chat  # noqa: E402


class BenchServer(chat.Server):
    """chat.Server with a free-form command line, extra environment and an event log."""

    def __init__(self, args, out_dir):
        cmd = [args.exe, "serve", "--model", chat.MODEL, "--max-context", str(args.max_context),
               "--engram-tables", os.path.join(REPO, "tests", "data", "l3"),
               "--profile", os.path.join(out_dir, "profile.jsonl")]
        if args.cache_gb:
            cmd += ["--cache-gb", str(args.cache_gb)]
        cmd += args.serve_arg
        env = dict(os.environ)
        env["DEEPMOE_ROUTE_DUMP"] = os.path.join(out_dir, "route.bin")
        if args.shader_dir:
            env["DEEPMOE_SHADER_DIR"] = args.shader_dir
        for kv in args.env:
            k, v = kv.split("=", 1)
            env[k] = v
        for f in ("profile.jsonl", "route.bin", "events.jsonl"):
            p = os.path.join(out_dir, f)
            if os.path.exists(p):
                os.remove(p)
        self.events = open(os.path.join(out_dir, "events.jsonl"), "w", encoding="utf-8", newline="\n")
        self.t0 = time.time()
        self.log = open(os.path.join(out_dir, "serve.log"), "wb")
        self.p = subprocess.Popen(cmd, cwd=REPO, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self.log, bufsize=0, env=env)
        self.cmd = cmd
        self.ready = self.read_event()
        if self.ready.get("event") != "ready":
            raise SystemExit(f"server did not start: {self.ready}")

    def read_event(self):
        ev = super().read_event()
        rec = dict(ev)
        rec["host_s"] = round(time.time() - self.t0, 4)
        self.events.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.events.flush()
        return ev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="")
    ap.add_argument("--requests", default="",
                    help='raw {"requests":[{"prompt_ids":[..],"max_tokens":N,..}]} instead of a chat script')
    ap.add_argument("--idle-s", type=float, default=0.0,
                    help="seconds to wait after ready before the first request (the P3 backfill's idle time)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exe", default=os.path.join(REPO, "build", "deepmoe.exe"))
    ap.add_argument("--shader-dir", default="")
    ap.add_argument("--cache-gb", type=int, default=0)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--serve-arg", action="append", default=[])
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--repeat", type=int, default=1, help="run the script this many times back to back")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    os.makedirs(args.out, exist_ok=True)
    enc = chat.load_encoding()
    server = BenchServer(args, args.out)
    print(f"ready: {json.dumps(server.ready)}", flush=True)
    if args.idle_s:
        time.sleep(args.idle_s)
    if args.requests:
        with open(args.requests, encoding="utf-8") as f:
            reqs = json.load(f)["requests"]
        stats = []
        try:
            for r in reqs:
                server.send(dict({"op": "generate", "stop_ids": [1]}, **r))
                while True:
                    ev = server.read_event()
                    if ev.get("event") == "done":
                        stats.append(ev)
                        print(json.dumps({k: ev[k] for k in ("prompt_tokens", "prefill_mode", "prefill_ms",
                                                            "decode_steps", "tok_s", "decode_hit_rate")}),
                              flush=True)
                        break
                    if ev.get("event") == "error":
                        raise SystemExit(ev["message"])
        finally:
            server.close()
            server.events.close()
        with open(os.path.join(args.out, "turns.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump({"server": server.ready, "turns": stats, "cmd": server.cmd, "env": args.env}, f, indent=1)
        return 0
    cargs = argparse.Namespace(think=False, temp=1.0, top_p=0.95, max_tokens=256, seed=None, system="")
    c = chat.Chat(server, enc, cargs)
    with open(args.script, encoding="utf-8") as f:
        script = json.load(f)
    if args.repeat > 1:
        turns = []
        for r in range(args.repeat):
            for t in script["turns"]:
                t2 = dict(t)
                if r and t is script["turns"][0]:
                    t2["reset"] = True
                turns.append(t2)
        script = dict(script, turns=turns)
    try:
        chat.run_script(c, server, script, os.path.join(args.out, "transcript.md"),
                        os.path.join(args.out, "turns.json"))
    finally:
        server.close()
        server.events.close()
    with open(os.path.join(args.out, "turns.json"), encoding="utf-8") as f:
        doc = json.load(f)
    doc["cmd"] = server.cmd
    doc["env"] = args.env
    with open(os.path.join(args.out, "turns.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
