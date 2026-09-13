#!/usr/bin/env python3
"""Offline expert-cache simulator: pick the policy before writing it in C++.

Design reference: section 9.2 (measurement tools), section 9.3 (residency) and
section 9.4 (lookahead prefetch). P1 deliverable (section 15). Section 16:
"no Planner policy code before the P-1 and P1 measurements".

Section 3.1 is the reason this exists. At 200 GB/s of LPDDR and ~4.5 GB/s of
NVMe, decode time per token is dominated by expert misses:

    h = 0.30 -> 679 ms/token (1.5 tok/s)
    h = 0.60 -> 416 ms       (2.4 tok/s)
    h = 0.85 -> ~240 ms      (4.2 tok/s)
    h = 1.00 ->  65 ms       (15.3 tok/s, unreachable: 288 GB does not fit)

Ten points of hit rate are worth more than any kernel optimisation, so the
policy is chosen by simulation over a real trace, not by intuition.

What it sweeps:

  capacity   fraction of the 15,360 routed experts that fit (design section 5.2
             puts the real cache at ~29%): {20, 25, 30, 35}%
  policy     lru | lfu-decay | arc | score-aware | static-pin+lru (section 9.3)
  allocation one global pool vs per-layer quotas (decided by Q2)
  prefetch   lookahead depth d and width K, using the predictions the trace
             recorded (section 9.4)

What it reports, per configuration:

  hit rate (total and per layer), miss bytes per token, prefetch precision and
  recall, wasted prefetch bytes, and the implied tok/s from the section 3.1
  model using the measured NVMe bandwidth from bench/results/.

It also answers Q1 (static frequency coverage), Q2 (reuse-distance CDF, which
gives LRU's hit rate analytically at every capacity) and Q3 (per-layer Jaccard
overlap between consecutive tokens, which feeds the speculation schedule of
section 10.3).

The winning configuration is written back into design section 9 before P3
starts, and store/planner.cpp implements exactly it.

Usage:
    uv run python tools/cache_sim.py --trace traces/mixed.parquet \\
        --capacities 20,25,30,35 --policies lru,score-aware \\
        --nvme-gbps 4.5 --out reports/cache_sweep.json
"""

from __future__ import annotations

import argparse
import sys


# Design section 2.3 / 3.1. Kept in step with model/layout.h.
EXPERT_BYTES = 18_800_640
N_ROUTED_EXPERTS_TOTAL = 15_360
HOT_BYTES_PER_TOKEN = 8.5e9
ROUTED_BYTES_PER_TOKEN = 4.51e9


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cache_sim.py",
        description="simulate expert-cache policies over a recorded route trace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--trace", required=True, action="append",
                   help="a .parquet from tools/route_trace.py; repeatable")
    p.add_argument("--capacities", default="20,25,30,35",
                   help="cache capacities as a percentage of all 15,360 experts")
    p.add_argument("--policies", default="lru,lfu-decay,arc,score-aware,static-pin+lru",
                   help="comma-separated policies to compare (design section 9.3)")
    p.add_argument("--allocation", choices=["global", "per-layer"], default="global",
                   help="one pool or per-layer quotas; Q2 decides whether the "
                        "latter is worth it")
    p.add_argument("--prefetch-depths", default="0,2,3,4,6",
                   help="lookahead depth d; 0 disables prefetch (design section 9.4)")
    p.add_argument("--prefetch-widths", default="6,8,12,16",
                   help="lookahead width K (design section 9.4)")
    p.add_argument("--prefetch-precision-floor", type=float, default=0.35,
                   help="below this measured precision the online planner would "
                        "shrink K; simulate the same rule")
    p.add_argument("--nvme-gbps", type=float, default=4.5,
                   help="measured effective NVMe bandwidth from bench/results/; "
                        "used to turn miss bytes into the tok/s of section 3.1")
    p.add_argument("--lpddr-gbps", type=float, default=200.0,
                   help="measured effective LPDDR bandwidth from bench/results/")
    p.add_argument("--questions", default="q1,q2,q3",
                   help="which of the section 9.1 analyses to also emit")
    p.add_argument("--out", default=None, help="write the report as JSON here")
    p.add_argument("--plot-dir", default=None,
                   help="also write the hit-rate and reuse-distance curves here")
    return p


def analyse_q1(traces, args) -> dict:
    """Static expert frequency: does top-x% cover y% of routes?

    TODO(design section 9.1 Q1): per layer and overall, the coverage curve.
    If `noaux_tc` balances well the curve will be nearly diagonal, and a pinned
    expert set is not worth having.
    """
    raise NotImplementedError("cache_sim.analyse_q1 (design section 9.1)")


def analyse_q2(traces, args) -> dict:
    """Reuse-distance (stack-distance) distribution.

    TODO(design section 9.1 Q2): the CDF gives LRU's hit rate at every capacity
    in closed form, without simulating each one, and shows whether layers differ
    enough to justify per-layer quotas.
    """
    raise NotImplementedError("cache_sim.analyse_q2 (design section 9.1)")


def analyse_q3(traces, args) -> dict:
    """Jaccard overlap of consecutive tokens' expert sets, per layer.

    TODO(design section 9.1 Q3): this is what decides whether speculative
    decoding thins NVMe traffic or merely widens it, and it feeds T_nvme(k) in
    the schedule curve of section 10.3.
    """
    raise NotImplementedError("cache_sim.analyse_q3 (design section 9.1)")


def simulate(traces, capacity_pct: int, policy: str, depth: int, width: int,
             args) -> dict:
    """Replay the trace through one cache configuration.

    TODO(design section 9.3, 9.4): model Free/Filling/Resident, the chosen
    eviction policy, and (when depth > 0) the prefetch queue using the
    predictions recorded in the trace. Return hit rate, miss bytes per token,
    prefetch precision/recall, wasted bytes, and the implied tok/s:

        t = HOT_BYTES_PER_TOKEN / lpddr
          + hit_bytes / lpddr
          + miss_bytes / nvme
    """
    raise NotImplementedError("cache_sim.simulate (design section 9.3, 9.4)")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raise NotImplementedError("cache_sim.main (design section 9.2, P1)")


if __name__ == "__main__":
    sys.exit(main())
