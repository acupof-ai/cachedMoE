"""One config, same script as config_sweep, for an A/B on the binary alone.

    .venv\\Scripts\\python.exe bench\\one_config.py --slots 4500 --out bench\\results\\after_mask.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_sweep as cs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=cs.DEFAULT_EXE)
    ap.add_argument("--model", default=cs.DEFAULT_MODEL)
    ap.add_argument("--slots", type=int, default=4500)
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--reheat", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    label = f"cache-{args.slots}" + ("-reheat" if args.reheat else "")
    extra = ["--cache-slots", str(args.slots)] + (["--reheat"] if args.reheat else [])
    r = cs.run_config(args.exe, args.model, label, extra, args.tokens, args.turns,
                      args.max_context)
    print(json.dumps({k: r.get(k) for k in
                      ("config", "cache_slots", "decode_tok_s", "decode_hit", "nvme_mb",
                       "nvme_mb_per_decode_token", "wall_s", "error")}))
    for t in r.get("turns", []):
        print(json.dumps({k: t.get(k) for k in
                          ("prompt_tokens", "generated", "ttft_ms", "tok_s",
                           "decode_steps", "decode_hit_rate", "prefill_hit_rate")}))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=1)
        print("wrote " + args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
