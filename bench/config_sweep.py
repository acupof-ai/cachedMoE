"""Same dialogue, different cache configuration: what each optimisation is worth.

One serve process per configuration, a scripted multi-turn conversation on one
topic, the same prompts and the same generation length. Prints one JSON row per
configuration and writes the lot to a file.

    .venv\\Scripts\\python.exe bench\\config_sweep.py --tokens 24
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_EXE = os.path.join(REPO, "build", "deepmoe.exe")
DEFAULT_MODEL = os.environ.get("DEEPMOE_MODEL_DIR", r"D:\models\DeepSeek-V4.1-Flash")

TOPIC = [
    "Explain in detail how a Vulkan timeline semaphore orders submissions across "
    "queues, and what happens to a wait that is submitted before its signal.",
    "Continue: what does that mean for a command buffer that both waits on and "
    "signals the same timeline value, and how should a decoder loop use it?",
    "Continue: summarise the rules for host waits versus device waits on that "
    "timeline, and the failure modes of each.",
    "Given that, how should a runtime size its expert cache so that a long "
    "conversation keeps its working set resident?",
]


def run_config(exe, model, label, extra, tokens, turns, max_context):
    cmd = [exe, "serve", "--model", model, "--max-context", str(max_context)] + extra
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1)
    row = {"config": label, "args": extra, "turns": [], "ok": False}

    def send(o):
        p.stdin.write(json.dumps(o) + "\n")
        p.stdin.flush()

    def ev():
        line = p.stdout.readline()
        if not line:
            raise SystemExit("server closed")
        return json.loads(line)

    def until(*kinds):
        while True:
            e = ev()
            if e.get("event") in kinds:
                return e

    try:
        ready = until("ready", "error")
        if ready.get("event") == "error":
            row["error"] = ready["message"]
            return row
        row["cache_slots"] = ready.get("cache_slots")
        row["cache_gb"] = ready.get("cache_gb")
        t0 = time.time()
        for i, text in enumerate(TOPIC[:turns]):
            send({"op": "tokenize", "text": text})
            tok = until("tokens")
            send({"op": "generate", "prompt_ids": tok["ids"], "max_tokens": tokens,
                  "temperature": 0.0, "reuse": True})
            while True:
                e = ev()
                if e.get("event") == "done":
                    break
                if e.get("event") == "error":
                    row["error"] = e["message"]
                    return row
            row["turns"].append({k: e[k] for k in
                                 ("prompt_tokens", "reused_tokens", "generated", "prefill_mode",
                                  "ttft_ms", "tok_s", "decode_steps", "decode_hit_rate",
                                  "prefill_hit_rate", "decode_nvme_mb", "prefill_nvme_mb",
                                  "total_ms", "context") if k in e})
        row["wall_s"] = round(time.time() - t0, 1)
        steps = sum(t["decode_steps"] for t in row["turns"])
        dec_ms = sum(t["total_ms"] - t["ttft_ms"] for t in row["turns"])
        row["decode_tok_s"] = round(steps * 1e3 / dec_ms, 3) if dec_ms > 0 else 0.0
        row["decode_hit"] = round(sum(t["decode_hit_rate"] * t["decode_steps"] for t in row["turns"])
                                  / max(steps, 1), 4)
        row["nvme_mb"] = round(sum(t["decode_nvme_mb"] + t["prefill_nvme_mb"] for t in row["turns"]))
        row["nvme_mb_per_decode_token"] = round(row["nvme_mb"] / max(steps, 1), 1)
        row["ok"] = True
        return row
    except SystemExit as e:
        row["error"] = str(e)
        return row
    finally:
        try:
            send({"op": "quit"})
            p.wait(timeout=60)
        except Exception:
            p.kill()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=DEFAULT_EXE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--out", default=os.path.join(REPO, "bench", "results", "config_sweep.json"))
    args = ap.parse_args()

    configs = [
        # 16 slots is below one slab (112 slots), so the store refuses; the
        # smallest usable cache is ~112 slots and "1,000" is close enough to a
        # no-cache floor to be worth the run.
        ("cache-1000", ["--cache-slots", "1000"]),
        ("cache-2200", ["--cache-slots", "2200"]),
        ("cache-2200-reheat", ["--cache-slots", "2200", "--reheat"]),
        ("cache-4500", ["--cache-slots", "4500"]),
        ("cache-4500-reheat", ["--cache-slots", "4500", "--reheat"]),
    ]
    rows = []
    for label, extra in configs:
        r = run_config(args.exe, args.model, label, extra, args.tokens, args.turns,
                       args.max_context)
        rows.append(r)
        print(json.dumps({k: r.get(k) for k in
                          ("config", "cache_slots", "cache_gb", "decode_tok_s", "decode_hit",
                           "nvme_mb", "nvme_mb_per_decode_token", "wall_s", "error")}))
        sys.stdout.flush()
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1)
        print("wrote " + args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
