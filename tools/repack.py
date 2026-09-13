#!/usr/bin/env python3
"""Repack the DeepSeek-V4.1-Flash safetensors shards into deepMoE's NVMe layout.

Design reference: section 5.1 (NVMe file layout) and appendix A (tensor
inventory). This is a P0 deliverable (section 15).

The 510 GB checkpoint arrives as 48 safetensors shards whose tensor order has
nothing to do with how the runtime reads them. Repacking answers three needs
that mmap over the original shards cannot:

  1. **4 KiB alignment.** Every tensor and every routed-expert block starts on a
     sector boundary, so FILE_FLAG_NO_BUFFERING / O_DIRECT can read straight
     into a GPU-visible slab with no bounce buffer (section 9.6).
  2. **Arithmetic addressing.** Routed experts go into `experts.bin` in
     `[layer][expert]` order at a fixed 18,800,640 B stride, so the planner
     computes an offset instead of consulting a table, and a whole layer's
     7.22 GB is contiguous for the expert-major prefill stream (section 9.7).
  3. **Interleaved engram rows.** Each engram row becomes 264 B of value bytes
     followed by its 8 B of E8M0 scale, so one 4 KiB read returns whole rows
     *with* their scales (section 5.1).

Outputs, all in --dst:

  hot.bin            attention, shared experts, router, mHC, norms, engram wkv,
                     embed, head. Layer order, each tensor 4 KiB aligned.
                     ~10.5 GB, read sequentially at startup and pinned.
  experts.bin        40 x 384 routed experts, 18,800,640 B each, 288.8 GB.
  mtp.bin            the three DSpark blocks, 7.9 GB, pinned.
  engram.L1.bin      384,006,168 rows x 264 B interleaved.
  engram.L14.bin     384,016,682 rows x 264 B interleaved.
  manifest.json      version 1; every tensor's file/offset/bytes/shape/dtype
                     and its block-scale layout. model/manifest.h is the reader
                     and tests/test_model.cpp pins the schema.

The expert block layout is `[w1 rows | w3 rows | w2 rows | s1 | s3 | s2]`.
Design section 5.1 also asks for a `w1/w3 row-interleaved` variant so a fused
gate/up wave reads both rows in one go; --expert-layout selects it and P2 picks
a winner by measurement.

Usage:
    uv run python tools/repack.py --src D:/models/DeepSeek-V4.1-Flash \\
                                  --dst D:/models/deepmoe-v41 --verify
"""

from __future__ import annotations

import argparse
import sys


# Design section 2.3 / appendix A. model/layout.h holds the same numbers for
# the C++ side; the two must not drift.
EXPERT_BYTES = 18_800_640
EXPERT_MAT_WEIGHT_BYTES = 5_898_240
EXPERT_MAT_SCALE_BYTES = 368_640
ENGRAM_ROW_BYTES = 264
ALIGNMENT = 4096
N_LAYERS = 40
N_ROUTED_EXPERTS = 384
MANIFEST_VERSION = 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="repack.py",
        description="safetensors -> hot.bin / experts.bin / engram.*.bin / mtp.bin / manifest.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--src", required=True,
                   help="directory holding the 48 safetensors shards and config.json")
    p.add_argument("--dst", required=True,
                   help="output directory for the repacked blobs (needs ~510 GB)")
    p.add_argument("--verify", action="store_true",
                   help="cross-check shard sizes against model.safetensors.index.json "
                        "and record a sha256 per output file in the manifest")
    p.add_argument("--expert-layout", choices=["planar", "w13-interleaved"], default="planar",
                   help="planar = [w1|w3|w2|s1|s3|s2]; w13-interleaved puts matching "
                        "w1/w3 rows next to each other for the fused gate/up kernel "
                        "(design section 5.1, A/B in P2)")
    p.add_argument("--only", choices=["hot", "experts", "engram", "mtp", "manifest"],
                   action="append", default=None,
                   help="repack only these parts; repeatable. Default: all of them")
    p.add_argument("--skip-engram", action="store_true",
                   help="skip the 203 GB engram tables (they dominate the runtime "
                        "and are not needed until the engram kernel of section 7.10)")
    p.add_argument("--workers", type=int, default=4,
                   help="shard readers to run concurrently")
    p.add_argument("--dry-run", action="store_true",
                   help="plan and print the layout without writing anything")
    return p


def plan_layout(args: argparse.Namespace) -> dict:
    """Walk the shard headers and decide every output offset.

    TODO(design section 5.1): read each shard's safetensors header (an 8-byte
    little-endian length followed by that many bytes of JSON), classify every
    tensor into hot / experts / engram / mtp by name, and assign 4 KiB-aligned
    offsets. Nothing is copied here -- this pass is cheap and its output is the
    manifest, which --dry-run prints.
    """
    raise NotImplementedError("repack.plan_layout (design section 5.1)")


def write_hot(plan: dict, args: argparse.Namespace) -> None:
    """Copy the pinned tensors into hot.bin in layer order.

    TODO(design section 5.1): attention (wq_a, wq_b, wkv, wo_a, wo_b + scales),
    shared experts, router W/bias/bias_vl, hc_*_fn/base/scale, norms, engram
    wkv and q/k, embed and head. ~10.5 GB, ~2 s to read back sequentially.
    """
    raise NotImplementedError("repack.write_hot (design section 5.1)")


def write_experts(plan: dict, args: argparse.Namespace) -> None:
    """Copy the 15,360 routed experts into experts.bin.

    TODO(design section 5.1): for each (layer, expert), emit exactly
    EXPERT_BYTES so `offset = (layer * 384 + expert) * 18_800_640`. Honour
    --expert-layout. Assert the block size after every expert: a single wrong
    stride silently misaddresses everything downstream.
    """
    raise NotImplementedError("repack.write_experts (design section 5.1)")


def write_engram(plan: dict, args: argparse.Namespace) -> None:
    """Interleave the engram value and scale planes into 264 B rows.

    TODO(design section 5.1): the checkpoint stores `engram.embed.weight`
    [rows, 256] fp8 and `engram.embed.scale` [rows, 8] E8M0 as separate planes.
    The runtime wants them interleaved so one 4 KiB read covers whole rows with
    their scales. 384M rows per layer, 101 GB out per layer: stream it.
    """
    raise NotImplementedError("repack.write_engram (design section 5.1)")


def write_mtp(plan: dict, args: argparse.Namespace) -> None:
    """Copy the three DSpark blocks into mtp.bin.

    TODO(design section 5.1, 7.12): same structure as a main-model layer plus
    main_proj, the Markov head and the confidence head. 7.9 GB, all pinned, so
    the draft cycle never waits on NVMe.
    """
    raise NotImplementedError("repack.write_mtp (design section 5.1)")


def write_manifest(plan: dict, args: argparse.Namespace) -> None:
    """Emit manifest.json v1.

    TODO(design section 5.1): the schema model/manifest.h parses -- files,
    tensors (file/offset/bytes/dtype/shape/scale), the arithmetic `experts`
    section and the `engram` array. With --verify, a sha256 per output file.
    """
    raise NotImplementedError("repack.write_manifest (design section 5.1)")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    parts = set(args.only or ["hot", "experts", "engram", "mtp", "manifest"])
    if args.skip_engram:
        parts.discard("engram")

    plan = plan_layout(args)
    if args.dry_run:
        print(plan)
        return 0
    if "hot" in parts:
        write_hot(plan, args)
    if "experts" in parts:
        write_experts(plan, args)
    if "engram" in parts:
        write_engram(plan, args)
    if "mtp" in parts:
        write_mtp(plan, args)
    if "manifest" in parts:
        write_manifest(plan, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
