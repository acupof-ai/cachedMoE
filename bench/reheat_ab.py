"""Per-turn reheat A/B: does the turn boundary make the NEXT turn hit more?

docs/p4_hitrate.md section 7. Three turns on one topic, one serve process, with
--reheat on or off; the number that matters is turn C's decode hit rate, because
that is the turn the reheat pass had a chance to prepare for. Turn A also warms
the cache from nothing, so it is reported and not compared.

    .venv\\Scripts\\python.exe bench\\reheat_ab.py --cache-slots 2200 --reheat
    .venv\\Scripts\\python.exe bench\\reheat_ab.py --cache-slots 2200 --no-reheat --out ...\\off.json
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
    "timeline, with the failure modes of each.",
]


class Server:
    def __init__(self, args):
        cmd = [args.exe, "serve", "--model", args.model, "--cache-slots", str(args.cache_slots),
               "--max-context", str(args.max_context)]
        cmd += ["--reheat", "--reheat-decay", str(args.decay)] if args.reheat else []
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=None, text=True, encoding="utf-8", bufsize=1)

    def send(self, obj):
        self.p.stdin.write(json.dumps(obj) + "\n")
        self.p.stdin.flush()

    def event(self):
        line = self.p.stdout.readline()
        if not line:
            raise SystemExit("server closed the protocol")
        return json.loads(line)

    def until(self, *kinds):
        while True:
            ev = self.event()
            if ev.get("event") in kinds:
                return ev

    def close(self):
        try:
            self.send({"op": "quit"})
            self.p.wait(timeout=60)
        except Exception:
            self.p.kill()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=DEFAULT_EXE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--cache-slots", type=int, default=2200)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--decay", type=float, default=0.5)
    ap.add_argument("--reheat", action="store_true")
    ap.add_argument("--no-reheat", dest="reheat", action="store_false")
    ap.set_defaults(reheat=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    srv = Server(args)
    ready = srv.until("ready", "error")
    if ready.get("event") == "error":
        print("server: " + ready["message"], file=sys.stderr)
        return 1
    result = {"reheat": args.reheat, "decay": args.decay, "cache_slots": ready["cache_slots"],
              "turns": []}
    print(json.dumps({"mode": "reheat" if args.reheat else "off",
                      "cache_slots": ready["cache_slots"]}))
    try:
        for i, text in enumerate(TOPIC[:3]):
            srv.send({"op": "tokenize", "text": text})
            tok = srv.until("tokens")
            srv.send({"op": "generate", "prompt_ids": tok["ids"], "max_tokens": args.tokens,
                      "temperature": 0.0, "reuse": True})
            t0 = time.time()
            while True:
                ev = srv.event()
                if ev.get("event") == "done":
                    done = ev
                    break
                if ev.get("event") == "error":
                    print(f"turn {i}: {ev['message']}", file=sys.stderr)
                    return 1
            row = {k: done[k] for k in ("prompt_tokens", "reused_tokens", "generated",
                                        "prefill_mode", "ttft_ms", "tok_s", "decode_hit_rate",
                                        "prefill_hit_rate", "decode_nvme_mb", "prefill_nvme_mb",
                                        "decode_steps", "context", "total_ms", "reheat")
                   if k in done}
            row["turn"] = i + 1
            row["wall_s"] = round(time.time() - t0, 2)
            if i + 1 < 3 and args.reheat:
                srv.send({"op": "reheat"})
                rh = srv.until("reheat", "error")
                row["reheat_op"] = rh
            result["turns"].append(row)
            print(json.dumps(row))
    finally:
        srv.close()
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
