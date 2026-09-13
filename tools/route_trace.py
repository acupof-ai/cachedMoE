#!/usr/bin/env python3
"""Record routing decisions over a corpus, so cache policy can be chosen from data.

Design reference: section 9.2 (measurement tools). This is a P1 deliverable
(section 15), and section 16 forbids writing any Planner policy code before it
has run.

Everything the expert cache does rests on five unknowns (section 9.1):

  Q1  static frequency: does top-x% of experts cover y% of routes? `noaux_tc`
      load balancing may leave almost no static skew, in which case a pinned
      expert set is worthless.
  Q2  reuse (stack) distance distribution, which gives LRU's hit rate
      analytically at any capacity, and says whether per-layer quotas beat one
      global pool.
  Q3  Jaccard overlap of consecutive tokens' expert sets, per layer. This sets
      how much speculative decoding actually saves on NVMe traffic, and feeds
      the schedule curve of section 10.3.
  Q4  lookahead recall: predicting layer L's top-6 from the layer L-d residual
      stream, over d = 1..8 and K = 6..16, for each of the two input
      approximations of section 9.4.
  Q5  T_layer / T_io, which sets the minimum useful prefetch depth d.

This script answers Q1-Q4 by running the CPU forward pass from P0 over >= 20K
tokens of mixed Chinese, English and code from several prompts, recording for
every (token, layer):

    token_index, layer, top6_ids, top16_ids, top16_scores, hidden_norm_input,
    and, for each d in --lookahead-depths, the predicted top-K set

Output is Parquet, consumed by tools/cache_sim.py.

Usage:
    uv run python tools/route_trace.py --model D:/models/deepmoe-v41 \\
        --corpus corpus/mixed.txt --tokens 20000 --out traces/mixed.parquet
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="route_trace.py",
        description="record (token, layer) -> expert routing for cache simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", required=True, help="repacked model directory")
    p.add_argument("--corpus", required=True, action="append",
                   help="text file to trace; repeatable. Design section 9.2 asks "
                        "for a Chinese/English/code mix over several prompts")
    p.add_argument("--tokens", type=int, default=20000,
                   help="minimum tokens to record (design section 9.2 asks for >= 20K)")
    p.add_argument("--out", required=True, help="output .parquet")
    p.add_argument("--lookahead-depths", default="1,2,3,4,6,8",
                   help="the d values to record predictions for (Q4)")
    p.add_argument("--lookahead-widths", default="6,8,10,12,16",
                   help="the K values to record (Q4)")
    p.add_argument("--lookahead-input", choices=["mean-hc", "pre-identity", "both"],
                   default="both",
                   help="which approximation of the layer input feeds the "
                        "predicted layer's gate (design section 9.4)")
    p.add_argument("--record-top16", action="store_true", default=True,
                   help="also record the top-16 ids and scores, which is what the "
                        "score-aware eviction of section 9.3 would use")
    p.add_argument("--engine", choices=["cpu", "deepmoe"], default="cpu",
                   help="cpu = the P0 fp32 forward pass; deepmoe = a running "
                        "engine with --profile, once P3 exists")
    p.add_argument("--seed", type=int, default=0)
    return p


def trace(args: argparse.Namespace) -> int:
    """Run the forward pass and write the Parquet trace.

    TODO(design section 9.2): needs the P0 CPU forward pass to exist first. The
    recording itself is cheap -- the cost is that a full fp32 forward is minutes
    per token, so 20K tokens is a long run and should checkpoint incrementally.

    Schema (one row per (token, layer)):
        token       uint32    position in the corpus
        layer       uint8     0..39
        top6        list<uint16>
        top6_w      list<float32>
        top16       list<uint16>
        top16_score list<float32>
        hidden_norm float32   ||ffn_norm input||, for sanity plots
        pred_d{N}_K{M} list<uint16>   one column per (d, K) pair
    """
    raise NotImplementedError("route_trace.trace (design section 9.2, P1)")


def main(argv: list[str] | None = None) -> int:
    return trace(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
