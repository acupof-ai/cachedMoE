"""Drive `deepmoe serve` through a two-topic conversation and watch the reheat pass.

docs/p4_hitrate.md section 7. This is the G1-style probe for the per-turn reheat:
one serve process, --cache-slots small enough that the cache is genuinely full,
two turns, then the reheat op, printed as JSON so the numbers can be pasted into
the doc. It does not generate a long reply -- the point is the turn boundary, not
throughput (that is tools/chat.py --script).

    .venv\\Scripts\\python.exe bench\\reheat_probe.py --cache-slots 3000 --tokens 24
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


class Server:
    def __init__(self, args):
        cmd = [args.exe, "serve", "--model", args.model, "--cache-slots", str(args.cache_slots),
               "--max-context", str(args.max_context), "--reheat",
               "--reheat-decay", str(args.decay)]
        env = dict(os.environ)
        if args.profile:
            cmd += ["--profile", args.profile]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=None, text=True, encoding="utf-8", bufsize=1, env=env)

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
    ap.add_argument("--cache-slots", type=int, default=3000)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--tokens", type=int, default=24, help="tokens per turn")
    ap.add_argument("--decay", type=float, default=0.5)
    ap.add_argument("--prompt-a", default="Explain how a Vulkan timeline semaphore orders "
                                          "submissions across queues.")
    ap.add_argument("--prompt-b", default="Now write a short poem about the sea at night.")
    ap.add_argument("--out", default=os.path.join(REPO, "bench", "results", "reheat_probe.json"))
    ap.add_argument("--profile", default="")
    args = ap.parse_args()

    srv = Server(args)
    ready = srv.until("ready", "error")
    if ready.get("event") == "error":
        print("server: " + ready["message"], file=sys.stderr)
        return 1
    out = {"ready": ready, "turns": []}
    print(json.dumps({"ready": {k: ready[k] for k in
                                ("cache_slots", "reheat", "reheat_decay", "max_context")}}))

    try:
        for label, text in (("A", args.prompt_a), ("B", args.prompt_b)):
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
                    print(f"turn {label}: {ev['message']}", file=sys.stderr)
                    return 1
            row = {
                "turn": label,
                "prompt_tokens": done["prompt_tokens"],
                "reused_tokens": done["reused_tokens"],
                "generated": done["generated"],
                "prefill_mode": done["prefill_mode"],
                "ttft_ms": done["ttft_ms"],
                "tok_s": done["tok_s"],
                "decode_hit_rate": done["decode_hit_rate"],
                "prefill_hit_rate": done["prefill_hit_rate"],
                "decode_nvme_mb": done["decode_nvme_mb"],
                "context": done["context"],
                "wall_s": round(time.time() - t0, 2),
                "reheat": done.get("reheat"),
            }
            out["turns"].append(row)
            print(json.dumps(row))

            # The same pass the turn boundary runs, on demand, so the probe can
            # see the order it built and how many slots it had to fill.
            srv.send({"op": "reheat"})
            rh = srv.until("reheat", "error")
            if rh.get("event") == "error":
                print("reheat: " + rh["message"], file=sys.stderr)
                return 1
            out["turns"][-1]["reheat_op"] = rh
            print(json.dumps({"reheat": rh}))
            srv.send({"op": "status"})
            st = srv.until("status")
            out["turns"][-1]["store_after"] = st["store"]
            print(json.dumps({"store": st["store"]}))
    finally:
        srv.close()

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
