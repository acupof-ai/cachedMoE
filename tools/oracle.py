#!/usr/bin/env python3
"""fp32 reference forward pass for DeepSeek-V4.1-Flash -- the correctness anchor.

Design reference: section 12 (the four-level oracle). This is a P0 deliverable
(section 15).

There is no official CUDA environment to compare against, no GGUF, and no
llama.cpp build that understands V4.1. The oracle is therefore self-built, and
section 12 is blunt about what that costs: `oracle.py` is the trust anchor, and
its own correctness rests on a line-by-line review against `inference/model.py`
plus a dual-implementation agreement test on a small synthetic model. That
review is not optional and must be written down.

Levels this script serves:

  L0  decode  -- FP4 / FP8 / E8M0 -> fp32 must match numpy bit for bit.
                 The C++ side is cpu/dequant.cpp; tests/test_dequant.cpp pins it.
                 `--level l0` dumps the three tables for a cross-check.
  L2  layer   -- run one real layer in torch fp32 and compare against deepMoE's
                 per-layer output. Pass: cosine >= 0.999, max relative error
                 recorded and plotted.
  L3  e2e     -- full-model fp32 forward, minutes per token, >= 5 prompts x 64
                 tokens greedy. Pass: 100% token agreement, with any divergence
                 recorded together with the top1-top2 logit margin.

The details section 2.4 says are easy to get wrong, and which this script exists
to be right about:

  * the residual stream is [hc=4, 5120] fp32, and a sublayer's `pre` is consumed
    by the *next* sublayer, not its own
  * Sinkhorn runs 20 iterations in fp32 and must come out doubly stochastic
  * gate: sqrt(softplus(x . W)); select on score+bias, weight on the unbiased
    score, normalise, multiply by 1.5
  * expert: clamp(w1 x, max=10), clamp(w3 x, +-10), silu * up, fp32 accumulation
    across the 6 routed experts and the shared one
  * attention: RoPE on the last 64 dims only, fp8 quantisation into the window
    cache, inverse RoPE on the output before the grouped wo_a
  * rms_norm_eps = 1e-20, and the gate normalisation adds 1e-20

Usage:
    uv run python tools/oracle.py --model D:/models/DeepSeek-V4.1-Flash \\
        --level l2 --layer 5 --compare build/layer5.npy
    uv run python tools/oracle.py --model ... --level l3 \\
        --prompts tests/data/prompts.txt --max-tokens 64
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="oracle.py",
        description="fp32 reference forward pass (design section 12)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", required=True,
                   help="the original safetensors directory (the oracle decodes "
                        "weights itself rather than trusting repack.py)")
    p.add_argument("--level", choices=["l0", "l1", "l2", "l3"], required=True,
                   help="l0 decode tables, l1 single kernel, l2 single layer, l3 end to end")
    p.add_argument("--layer", type=int, default=None,
                   help="which layer for --level l2")
    p.add_argument("--kernel", default=None,
                   help="which kernel for --level l1 (mega_mhc, wq_b, gate, "
                        "moe_gateup, ... -- the names in gpu/shaders/)")
    p.add_argument("--compare", default=None,
                   help=".npy produced by deepMoE to compare against")
    p.add_argument("--prompts", default=None,
                   help="one prompt per line, for --level l3")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--prefill-mode", choices=["oracle", "bounded-replay"], default="oracle",
                   help="oracle runs the decoder over the whole prompt (exactly "
                        "model.py); bounded-replay is the production approximation "
                        "of design section 11.2, and the difference rate is a "
                        "reported number")
    p.add_argument("--tolerance-cos", type=float, default=0.999,
                   help="L2 pass threshold (design section 12)")
    p.add_argument("--tolerance-rel", type=float, default=1e-3,
                   help="L1 relative error threshold; 5e-3 for the int8 path")
    p.add_argument("--out", default=None, help="write the report as JSON here")
    p.add_argument("--seed", type=int, default=0)
    return p


def level0_tables(args: argparse.Namespace) -> int:
    """Dump the FP4 E2M1, FP8 E4M3 and E8M0 decode tables.

    TODO(design section 12 L0): build all three in numpy straight from the
    format definitions and compare against cpu/dequant.cpp. The packing
    convention to verify is the one in appendix A: low nibble = even element.
    """
    raise NotImplementedError("oracle.level0_tables (design section 12 L0)")


def level1_kernel(args: argparse.Namespace) -> int:
    """Compare one GPU kernel against an fp32 numpy implementation on random input.

    TODO(design section 12 L1): relative error <= 1e-3 for the fp16-carrying
    paths, <= 5e-3 for the int8 dot4 variant.
    """
    raise NotImplementedError("oracle.level1_kernel (design section 12 L1)")


def level2_layer(args: argparse.Namespace) -> int:
    """Run one real layer in torch fp32 and compare per-layer outputs.

    TODO(design section 12 L2): decode this layer's weights from the
    safetensors shards, run the mHC / attention / engram / MoE chain exactly as
    model.py does, and report cosine similarity plus max relative error against
    --compare.
    """
    raise NotImplementedError("oracle.level2_layer (design section 12 L2)")


def level3_end_to_end(args: argparse.Namespace) -> int:
    """Full-model fp32 greedy decode.

    TODO(design section 12 L3): minutes per token, >= 5 prompts x 64 tokens.
    Report per-token agreement and, at any divergence, the top1-top2 margin --
    a disagreement at a margin of 1e-6 is a different finding from one at 0.5.
    """
    raise NotImplementedError("oracle.level3_end_to_end (design section 12 L3)")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return {
        "l0": level0_tables,
        "l1": level1_kernel,
        "l2": level2_layer,
        "l3": level3_end_to_end,
    }[args.level](args)


if __name__ == "__main__":
    sys.exit(main())
