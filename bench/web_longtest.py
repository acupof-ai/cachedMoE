#!/usr/bin/env python3
"""Long web generation using one engine; run under a thermal supervisor.

Stop the user web engine first. The temporary server uses a separate port and
transcript directory. No tracing, route dumps or cycle diagnostics are enabled.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plain", "spec5"), required=True)
    parser.add_argument("--mask-cache", choices=("dynamic", "fixed"), default="fixed",
                        help="keep the historical fixed-cache longtest reproducible")
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--mirror", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    script = json.loads(args.script.read_text())
    env = dict(os.environ)
    env["XDG_CACHE_HOME"] = str(out / "session_state")
    for key in ("DEEPMOE_SPEC_DIAGNOSTICS", "DEEPMOE_ROUTE_DUMP"):
        env.pop(key, None)
    env.update(DEEPMOE_MASK_DYNAMIC_LRU="0" if args.mask_cache == "fixed" else "1", DEEPMOE_DSPARK_PROFILE="0",
        DEEPMOE_DSPARK_ONECB="1" if args.mode == "spec5" else "0",
        DEEPMOE_BATCH_GPU_ROUTE="1" if args.mode == "spec5" else "0",
        DEEPMOE_DSPARK_MEGA="0", DEEPMOE_MGT_PAIR_DOT="0",
        DEEPMOE_MGT_ATTN_CM="0", DEEPMOE_MGT_FOLD_SCALE="0")
    command = [sys.executable, str(ROOT / "tools/web/server.py"), "--exe", str(args.exe.resolve()),
        "--resident-only", "mask", "--mask-cache", args.mask_cache,
        "--cache-slots", "5500", "--max-context", "4096",
        "--gpu-prefill-min", "16", "--mirror", str(args.mirror), "--no-kv-disk",
        "--port", str(args.port), "--log", str(out / "engine.log")]
    if args.mode == "spec5":
        command += ["--dspark", "--spec-k", "5", "--spec-top-k", "4"]
    provenance.write(out, exe=args.exe, env=env)
    base = f"http://127.0.0.1:{args.port}"

    def rpc(route, body=None):
        request = urllib.request.Request(base + route,
            data=None if body is None else json.dumps(body, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(request, timeout=120))

    server = None
    turns = []
    try:
        with (out / "web.log").open("w") as log:
            server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 150
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"web exited: {server.returncode}")
                try:
                    config = rpc("/api/config")
                    break
                except (OSError, urllib.error.URLError):
                    if time.monotonic() > deadline:
                        raise RuntimeError("web startup timed out")
                    time.sleep(.5)
            assert config["ready"]["sources"] == 2
            spec = config["ready"]["speculation"]
            assert spec["enabled"] == (args.mode == "spec5")
            assert spec["draft_tokens"] == 5 and spec["accept_top_k"] == 4
            (out / "config.json").write_text(json.dumps(config, indent=2))
            print("READY", args.mode, json.dumps(spec), flush=True)
            for index, turn in enumerate(script["turns"]):
                session = "longtest"
                if turn.get("reset"):
                    rpc("/api/reset", {"session": session})
                body = {"session": session, "text": turn["text"], "think": turn["think"],
                    "reasoning_effort": 75, "temperature": 1.0, "top_p": .95,
                    "seed": turn["seed"], "max_tokens": turn["max_tokens"]}
                print("TURN", index, turn["label"], flush=True)
                token_ids, token_times, token_costs = [], [], []
                done = None
                request = urllib.request.Request(base + "/api/chat",
                    data=json.dumps(body, ensure_ascii=False).encode(),
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=240) as response, \
                        (out / f"turn{index}_events.jsonl").open("w") as events:
                    for line in response:
                        if not line.startswith(b"data:"):
                            continue
                        event = json.loads(line[5:])
                        events.write(json.dumps(event, ensure_ascii=False) + "\n")
                        events.flush()
                        if event["event"] == "error":
                            raise RuntimeError(event)
                        if event["event"] == "token":
                            assert 0 <= event["id"] < 129280
                            assert all(math.isfinite(event[k]) for k in ("t_ms", "step_ms", "p", "margin"))
                            assert not token_times or event["t_ms"] >= token_times[-1]
                            token_ids.append(event["id"])
                            token_times.append(event["t_ms"])
                            token_costs.append(event["step_ms"])
                            if len(token_ids) % 128 == 0:
                                print("PROGRESS", index, len(token_ids), "tokens", flush=True)
                        if event["event"] == "done":
                            done = event
                            break
                assert done is not None and done["generated"] == len(token_ids)
                assert done["context"] == done["prompt_tokens"] + done["generated"] - 1
                assert done["finish"] in ("length", "stop")
                assert done["decode_nvme_mb"] == 0
                if args.mode == "spec5":
                    spec = done["speculation"]
                    assert spec["cycles"] and spec["gpu_readout_cycles"] == spec["cycles"]
                    assert spec["accepted"] <= spec["verified"] and spec["miss_bytes"] == 0
                    submits = done["per_token_ms"]["submits"] * done["decode_steps"]
                    assert abs(submits - spec["cycles"]) < 1e-4
                status = rpc("/api/status?session=" + session)
                assert status["cache_fixed"] and status["cache_frozen"]
                assert "failed fills 0" in status["store"] and "evictions 0" in status["store"]
                assert "P0 failures reserve 0 / submit 0 / IO 0" in status["planner"]
                done.update(label=turn["label"], token_ids=token_ids, token_times_ms=token_times,
                    token_costs_ms=token_costs, request=body)
                turns.append(done)
                (out / "turns.json").write_text(json.dumps(turns, ensure_ascii=False, indent=2))
                (out / f"turn{index}_status.json").write_text(json.dumps(status, indent=2))
                print("DONE", index, json.dumps({key: done[key] for key in
                    ("generated", "tok_s", "context", "prefill_ms", "reused_tokens", "speculation")}), flush=True)
            assert sum(turn["generated"] for turn in turns) >= 1024, "not enough tokens for long-run coverage"
    finally:
        if server is not None and server.poll() is None:
            server.send_signal(signal.SIGINT)
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.terminate()
                server.wait(timeout=10)
        provenance.finish(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
