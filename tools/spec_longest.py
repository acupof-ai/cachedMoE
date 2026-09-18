#!/usr/bin/env python3
"""M3's offline number: what `--accept longest` would buy over `--accept chain`.

The scheme
----------
The draft head gives a K = 16 candidate matrix per position, not a single
token. `chain` acceptance stops at the first position where the drafted token
is not the target's argmax. `longest` keeps going: at position i it asks
whether the TARGET's argmax is among the draft's top-16 candidates at i, and
accepts the longest run for which that is true.

Why it is approximate
---------------------
Row i + 1 of the verify batch was computed with the CHAIN's token at position
i in its KV, not the token `longest` substitutes there. So every position after
the first substitution is conditioned on a prefix that never happened: the
argmax it reports is not the argmax the model would produce for the accepted
sequence. `longest` therefore is NOT lossless and does NOT preserve design
§10.2's invariant -- it trades that for length. This script measures the length;
whether the trade is acceptable is a separate question and this says nothing
about it.

What is measured
----------------
On the same trajectories `tools/dspark_tree.py analyse` uses
(traces/dspark_tree, 5 prompts x {greedy, sampling}, the real speculative loop's
own drafts and verify rows):

  * E[a] for `chain` and for `longest`, and E[tokens](k) = 1 + E[min(a, k)];
  * the conditional the scheme lives or dies on: at a position PAST the first
    chain mismatch, how often is the target's argmax in the draft's top-16;
  * the same, split by position, and by how far the chain got.

    .venv/Scripts/python.exe tools/spec_longest.py
    .venv/Scripts/python.exe tools/spec_longest.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dspark_tree as dt   # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="spec_longest.py")
    p.add_argument("--traces", default=dt.DEFAULT_TRACES)
    p.add_argument("--model", default=dt.DEFAULT_MODEL)
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)

    if not os.path.isdir(args.traces):
        print(f"SKIP spec_longest: no trajectories at {args.traces}")
        return 0
    E, H, W = dt.load_tables(args.model)
    trajs = dt.mode_dirs(args.traces)
    t0 = time.time()

    a_chain: list[int] = []
    a_long: list[int] = []
    # off-chain hits: position -> [n_hit, n_seen]
    off = [[0, 0] for _ in range(5)]
    # by how far the chain got before it broke
    by_a = {}
    n_events = 0

    for tid, (prompt, mode, pdir, log) in enumerate(trajs):
        if mode != "greedy":
            continue                      # the scheme is a temperature-0 one
        n = log["prefill_len"]
        produced = log["produced"]
        rows = dt._traj_rows(pdir, log)
        for di, dmeta in enumerate(log["drafts"]):
            s = dmeta["start_pos"]
            first = s + 2 - n
            if len(produced) - first < 5 or any((s + 2 + i) not in rows for i in range(5)):
                continue                  # full lookahead only, as `analyse` does
            y = [int(t) for t in produced[first:first + 5]]
            d = dt.load_draft(pdir, di)
            idx, cl, lse_a, e_in, e_cand, h_cand = dt.lattice_inputs(
                d["B"], d["input_token"], E, H, args.K, "anchor")
            lat = dt.Lattice(idx, cl, e_in, e_cand, h_cand, lse_a)
            toks = lat.tokens(lat.path("eal"))
            cand = [set(int(t) for t in idx[i]) for i in range(5)]

            ac = 0
            while ac < 5 and int(toks[ac]) == y[ac]:
                ac += 1
            al = 0
            while al < 5 and y[al] in cand[al]:
                al += 1
            a_chain.append(ac)
            a_long.append(al)
            n_events += 1
            # every position the chain did NOT already take is an "off-chain"
            # question: is the target's own argmax one of the 16 candidates?
            for i in range(ac, 5):
                off[i][1] += 1
                off[i][0] += int(y[i] in cand[i])
            b = by_a.setdefault(ac, {"n": 0, "hit": 0})
            if ac < 5:
                b["n"] += 1
                b["hit"] += int(y[ac] in cand[ac])

    if not n_events:
        print("SKIP spec_longest: no greedy draft events with full lookahead")
        return 0

    ch = np.asarray(a_chain)
    lo = np.asarray(a_long)
    out = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "traces": args.traces,
        "K": args.K,
        "events": n_events,
        "chain": {"mean_a": round(float(ch.mean()), 4)},
        "longest": {"mean_a": round(float(lo.mean()), 4)},
        "off_chain_top16_hit": {},
        "by_chain_a": {},
    }
    for k in range(1, 6):
        out["chain"][f"E_tokens_k{k}"] = round(1.0 + float(np.minimum(ch, k).mean()), 4)
        out["longest"][f"E_tokens_k{k}"] = round(1.0 + float(np.minimum(lo, k).mean()), 4)
    hit = sum(o[0] for o in off)
    seen = sum(o[1] for o in off)
    out["off_chain_top16_hit"]["all"] = {
        "n": seen, "hit": hit, "rate": round(hit / seen, 4) if seen else None}
    for i in range(5):
        out["off_chain_top16_hit"][f"pos{i}"] = {
            "n": off[i][1], "hit": off[i][0],
            "rate": round(off[i][0] / off[i][1], 4) if off[i][1] else None}
    for a, b in sorted(by_a.items()):
        out["by_chain_a"][str(a)] = {
            "n": b["n"], "hit": b["hit"],
            "rate": round(b["hit"] / b["n"], 4) if b["n"] else None}

    print(f"{n_events} greedy draft events, K = {args.K}  ({time.time() - t0:.1f}s)")
    print(f"  chain    mean a {out['chain']['mean_a']:.4f}   "
          f"E[tokens](k=1..5) " +
          " / ".join(f"{out['chain'][f'E_tokens_k{k}']:.2f}" for k in range(1, 6)))
    print(f"  longest  mean a {out['longest']['mean_a']:.4f}   "
          f"E[tokens](k=1..5) " +
          " / ".join(f"{out['longest'][f'E_tokens_k{k}']:.2f}" for k in range(1, 6)))
    r = out["off_chain_top16_hit"]["all"]
    print(f"  off-chain: the target's argmax is in the draft's top-16 at "
          f"{r['hit']}/{r['n']} = {r['rate']:.4f} of the positions the chain did not take")
    for i in range(5):
        q = out["off_chain_top16_hit"][f"pos{i}"]
        if q["n"]:
            print(f"    position {i}: {q['hit']}/{q['n']} = {q['rate']:.4f}")
    print("  at the FIRST mismatch only, by how far the chain got:")
    for a, b in sorted(by_a.items()):
        if b["n"]:
            print(f"    chain a = {a}: {b['hit']}/{b['n']} = {b['hit'] / b['n']:.4f}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
