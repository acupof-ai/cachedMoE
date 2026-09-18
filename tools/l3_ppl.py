#!/usr/bin/env python3
"""Track Y step 4: the >= 64-step teacher-forced PPL harness, end to end.

What it measures
----------------
`deepmoe run --state <64-step export> --teacher-force` forces the reference's
own greedy continuation through the engine and, at every one of the 64
positions, takes the negative log-probability the engine's head assigns to the
reference's next token. `exp(mean NLL)` is the perplexity Track Y's bars are
written against:

    <= 1.05 x off   GO for that mode as a default
    <= 1.30 x off   worth building verify-only on DSpark
    otherwise       NO-GO

docs/p4_resident_routing.md section 8.2 measured that ratio on the 8-step
`tests/data/l3` export and section 8.6 threw the result out: all three modes
diverge on the same two positions (steps 6 and 7), so the ratio was two
already-diverged positions' probability mass, and a x2.5 and a x3.4 could not be
told apart. Sixty-four positions is the fix.

The harness's own gate
----------------------
Before any ratio is read, `off` -- the engine with resident-only routing turned
off, i.e. the ordinary engine -- is compared against the REFERENCE's own mean
NLL on the same 64 targets (`reference_nll` in the export's index.json, the mean
of `max - logsumexp` over the exported positions). The two cannot be bit-equal:
the engine is FP4 weights on a GPU and the reference is fp32 on a CPU. But they
must be close, and how close is printed, because every ratio below is measured
against `off` and a broken `off` makes all of them meaningless.

Serial by construction
----------------------
One engine at a time: each mode is a separate process, run one after another,
because two engines on this machine fight over both GPU heaps and the NVMe
queue.

    .venv/Scripts/python.exe tools/l3_ppl.py --state traces/l3_64
    .venv/Scripts/python.exe tools/l3_ppl.py --state traces/l3_64 --json out.json
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = r"D:\models\DeepSeek-V4.1-Flash"


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _f(pattern: str, text: str, group: int = 1, cast=float):
    m = re.search(pattern, text)
    return cast(m.group(group)) if m else None


def parse_run(out: str) -> dict:
    """Everything the table needs out of one `deepmoe run` transcript."""
    r: dict = {}
    r["nll"] = _f(r"teacher-forced NLL ([0-9.]+) over (\d+) steps", out)
    r["nll_steps"] = _f(r"teacher-forced NLL [0-9.]+ over (\d+) steps", out, 1, int)
    r["ppl"] = _f(r"-> PPL ([0-9.]+)", out)
    m = re.search(r"(\d+)/(\d+) tokens match the fp32 reference", out)
    if m:
        r["top1"], r["top1_of"] = int(m.group(1)), int(m.group(2))
    r["tok_s"] = _f(r"wall [0-9.]+ s\s+([0-9.]+) tok/s", out)
    r["hit_rate"] = _f(r"tok/s\s+hit_rate ([0-9.]+)", out)
    r["requested"] = _f(r"experts requested (\d+)", out, 1, int)
    r["served"] = _f(r"served (\d+) \(([0-9.]+)\)", out, 1, int)
    r["served_frac"] = _f(r"served \d+ \(([0-9.]+)\)", out)
    r["mass_lost"] = _f(r"gate mass lost\s+([0-9.]+)", out)
    r["shared_only"] = _f(r"shared-expert-only layer-steps (\d+)", out, 1, int)
    r["stall1_p0"] = _f(r"stall1 P0 fetches (\d+)", out, 1, int)
    r["stall1_ms"] = _f(r"stall1 P0 fetches \d+\s+([0-9.]+) ms", out)
    m = re.search(r"^tokens\s+((?:\d+ )+)$", out, re.M)
    if m:
        r["tokens"] = [int(t) for t in m.group(1).split()]
    return r


def run_mode(exe: str, model: str, state: str, steps: int, mode: str,
             teacher_force: bool, log_dir: str | None) -> dict:
    cmd = [exe, "run", "--model", model, "--state", state, "--steps", str(steps),
           "--warm-cache", "--resident-only", mode]
    if teacher_force:
        cmd.append("--teacher-force")
    log(f"{mode:7s} {' '.join(cmd[1:])}")
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    out = (p.stdout or "") + (p.stderr or "")
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        with io.open(os.path.join(log_dir, f"{mode}.txt"), "w", encoding="utf-8") as f:
            f.write(out)
    if p.returncode != 0:
        print(out[-4000:], file=sys.stderr)
        raise SystemExit(f"`deepmoe run --resident-only {mode}` exited {p.returncode}")
    r = parse_run(out)
    r["mode"] = mode
    r["seconds"] = time.perf_counter() - t0
    log(f"{mode:7s} NLL {r['nll']} PPL {r['ppl']} top1 {r.get('top1')}/{r.get('top1_of')} "
        f"{r['tok_s']} tok/s ({r['seconds']:.0f} s)")
    return r


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", default=os.path.join("traces", "l3_64"),
                   help="the >= 64-step export (tools/oracle_l3_ppl.py)")
    p.add_argument("--model", default=os.environ.get("DEEPMOE_MODEL_DIR") or DEFAULT_MODEL)
    p.add_argument("--exe", default=os.path.join(REPO, "build", "deepmoe.exe"))
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--modes", default="off,all,stall1")
    p.add_argument("--free-run", default="",
                   help="comma separated modes to ALSO run free-running (no "
                        "--teacher-force); used to mint a candidate continuation")
    p.add_argument("--json", default=None, help="write the parsed table here")
    p.add_argument("--log-dir", default=None, help="keep each mode's transcript")
    args = p.parse_args()

    state = args.state if os.path.isabs(args.state) else os.path.join(REPO, args.state)
    idx = os.path.join(state, "index.json")
    if not os.path.exists(idx):
        # ctest treats this as a skip: the export is regenerable but large, so it
        # is not in the repository (traces/ is .gitignore'd).
        print(f"SKIP l3_ppl: {idx} is missing -- build it with (~28 min, CPU only)\n"
              f"  tools/oracle_l3_ppl.py greedy --out {args.state} --steps 64 "
              f"--model {args.model}")
        return 0
    if not os.path.exists(args.exe):
        print(f"SKIP l3_ppl: {args.exe} is missing -- build first")
        return 0
    with io.open(idx, encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("bootstrap"):
        print(f"SKIP l3_ppl: {idx} is still the one-step bootstrap "
              f"(run `oracle_l3_ppl.py greedy`)")
        return 0
    have = int(meta.get("steps_exported", 0))
    steps = min(args.steps, have)
    ref_nll = meta.get("reference_nll")
    log(f"state {state}: {meta['prefill_len']} prompt tokens, {have} exported steps, "
        f"running {steps}; reference NLL {ref_nll}")
    if steps < 64:
        log(f"WARNING: only {steps} steps -- section 8.6 asks for >= 64")

    rows = []
    for mode in [m for m in args.modes.split(",") if m]:
        rows.append(run_mode(args.exe, args.model, state, steps, mode, True, args.log_dir))
    for mode in [m for m in args.free_run.split(",") if m]:
        r = run_mode(args.exe, args.model, state, steps, mode, False, args.log_dir)
        r["mode"] = mode + " (free)"
        rows.append(r)

    off = next((r for r in rows if r["mode"] == "off"), None)

    # --- the harness's own gate ------------------------------------------- #
    print()
    if off and ref_nll is not None and off["nll"] is not None:
        d = off["nll"] - ref_nll
        print(f"gate  reference NLL {ref_nll:.6f} (PPL {math.exp(ref_nll):.4f})  vs  "
              f"off NLL {off['nll']:.6f} (PPL {off['ppl']:.4f})   delta {d:+.6f} "
              f"({off['ppl'] / math.exp(ref_nll):.4f}x)")
        print("      the engine is FP4 on a GPU and the reference fp32 on a CPU, so "
              "this is the harness's noise floor: every ratio below is only readable "
              "to the extent it is small.")

    # --- the table --------------------------------------------------------- #
    print()
    hdr = ("mode", "NLL", "PPL", "xoff", "top-1", "tok/s", "cache hit", "served",
           "mass lost", "shared-only", "verdict")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        ratio = (r["ppl"] / off["ppl"]) if (off and off["ppl"] and r["ppl"]) else None
        if r["mode"] == "off":
            verdict = "baseline"
        elif ratio is None:
            verdict = "-"
        elif ratio <= 1.05:
            verdict = "GO"
        elif ratio <= 1.30:
            verdict = "verify-only"
        else:
            verdict = "NO-GO"
        print("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["mode"],
            f"{r['nll']:.6f}" if r["nll"] is not None else "-",
            f"{r['ppl']:.4f}" if r["ppl"] is not None else "-",
            f"{ratio:.3f}x" if ratio is not None else "-",
            f"{r.get('top1')}/{r.get('top1_of')}",
            f"{r['tok_s']:.2f}" if r["tok_s"] is not None else "-",
            f"{r['hit_rate']:.4f}" if r.get("hit_rate") is not None else "-",
            f"{r['served']} ({r['served_frac']:.4f})" if r.get("served_frac") is not None else "-",
            f"{r['mass_lost']:.4f}" if r.get("mass_lost") is not None else "-",
            f"{r.get('shared_only')}" if r.get("shared_only") is not None else "-",
            verdict))

    if args.json:
        with io.open(args.json, "w", encoding="utf-8") as f:
            json.dump({"state": state, "steps": steps, "reference_nll": ref_nll,
                       "reference_ppl": math.exp(ref_nll) if ref_nll is not None else None,
                       "rows": rows}, f, indent=1)
        log(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
