#!/usr/bin/env python3
"""L1 golden vectors for the two things `tools/oracle.py` does not cover yet.

Design reference: section 7.9 (the two MoE dispatches), section 12 (the L1
oracle), section 2.4 (the details the reference implementation gets right and a
reimplementation gets wrong). This is the P2 companion of `oracle.py --level
l1`, kept as a separate file because `oracle.py` belongs to the attention/
runtime track.

Two gaps:

1. **The shared expert is fp8, not FP4.** `oracle.py --level l1` only ever
   loads a routed expert, whose six tensors are FP4 E2M1 + a per-row UE8M0
   scale. `layers.L.ffn.shared_experts.{w1,w2,w3}.weight` are FP8 E4M3
   [2304, 5120] / [5120, 2304] with a [rows/32, K/32] UE8M0 *tile* scale --
   11,796,480 B of weights and 11,520 B of scales each, 35.4 MB per layer
   against a routed expert's 18.8 MB. `gpu/shaders/moe_gateup.slang` and
   `moe_down.slang` run it through the same two dispatches under
   `Fp8Slots = 1`, and this script is what says whether the answer is right.

2. **`h` is quantised before `w2`, and the L1 golden was written without it.**
   `inference/model.py` `Expert.forward` ends in `self.w2(x.to(dtype))`, and
   `linear()` for an fp4 or fp8 weight begins with
   `act_quant(x, 32, "ue8m0", e8m0)`. So `silu(gate) * up` makes an fp8 E4M3
   round trip with a power-of-two block-32 scale before it ever reaches w2.
   `oracle.py`'s `expert_ffn` is explicitly "in fp32 with no activation
   quantisation", so the `tests/data/l1_*.bin` vectors are the *unquantised*
   answer: a kernel that implements design section 7.9 v0.6 correctly must move
   AWAY from them. This script writes all three answers for the same x so the
   move can be measured rather than argued about:

       y_ref    fp32 everywhere              (what l1_*.bin holds)
       y_hq     h quantised, x in fp32       (design section 7.9 v0.6 minus the
                                              fp16 activation choice of section 6)
       y_full   x and h both quantised       (the reference implementation)
       y_hq16   x and h rounded to fp16, then h quantised -- exactly the
                arithmetic gpu/shaders/moe_*.slang perform under HQuant != 0,
                so the GPU has to match this one to the fp32 summation-order
                floor rather than to y_hq. The two differ by far more than the
                fp16 rounding suggests because `fast_round_scale` is a step
                function of the block amax: a block whose amax sits just either
                side of a power of two gets a scale that differs by 2x, and
                then all 32 of its values are quantised on a grid twice as
                coarse. That amplification is a property of the reference's
                quantiser, not of the kernel.

   The kernel keeps x in fp16 rather than fp8 (design section 6), so `y_hq` is
   the answer it should converge on and the `y_hq`-to-`y_full` distance is what
   that choice costs.

Usage
-----
    uv run python tools/oracle_shared.py --model D:/models/DeepSeek-V4.1-Flash \\
        --shared 0 --expert 0:0 --expert 39:383 --out tests/data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oracle  # noqa: E402  -- the manifest reader and the FP4 table live there

FP8_BLOCK = 32          # model.py fp8_block_size
FP8_MAX = 448.0
SWIGLU_LIMIT = oracle.SWIGLU_LIMIT

SHARED_MAGIC = b"DMS1"
SHARED_VERSION = 1
Q_MAGIC = b"DMQ1"
Q_VERSION = 1

SHARED_PARTS = ("w1.weight", "w2.weight", "w3.weight")


# --------------------------------------------------------------------------- #
# quantisation -- `act_quant(x, 32, scale_fmt="ue8m0", scale_dtype=e8m0)`
# followed by the dequantisation fp8_gemm folds into the accumulator.
# Copied from tools/dsref.py (validated against the reference by track B) rather
# than imported, so this file has no dependency on the reference model loader.
# --------------------------------------------------------------------------- #

def round_scale_pow2(amax: torch.Tensor, max_inv: float) -> torch.Tensor:
    """`fast_round_scale`: the smallest power of two >= amax*max_inv, done on the
    bit pattern because ceil(log2(x)) in floating point gets the exact powers of
    two wrong about as often as it gets them right."""
    v = (amax.float() * max_inv).contiguous()
    bits = v.view(torch.int32)
    exp = (bits >> 23) & 0xFF
    man = bits & ((1 << 23) - 1)
    e = exp - 127 + (man != 0).to(torch.int32)
    return torch.ldexp(torch.ones_like(v), e)


def act_quant_dequant(x: torch.Tensor, block: int = FP8_BLOCK) -> torch.Tensor:
    """amax per (row, group of `block` along the last dim) floored at 1e-4, a
    power-of-two scale, then the fp8 E4M3 round trip."""
    shape = x.shape
    n = shape[-1]
    assert n % block == 0, (n, block)
    xf = x.float().reshape(-1, n // block, block)
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    s = round_scale_pow2(amax, 1.0 / FP8_MAX)
    q = torch.clamp(xf / s, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()
    return (q * s).reshape(shape)


def fp8_e4m3_table() -> np.ndarray:
    b = torch.arange(256, dtype=torch.uint8)
    return b.view(torch.float8_e4m3fn).float().numpy().astype(np.float32)


def dequant_fp8_tiles(values: bytes, scales: bytes, rows: int, k: int,
                      table: np.ndarray) -> np.ndarray:
    """[rows, k] E4M3 + [rows/32, k/32] UE8M0 -> fp32 [rows, k].

    The tile scale is the only structural difference from the FP4 path, whose
    scale is per (row, 32 elements along K).
    """
    assert rows % 32 == 0 and k % 32 == 0
    v = table[np.frombuffer(values, dtype=np.uint8)].reshape(rows, k)
    e = np.frombuffer(scales, dtype=np.uint8).reshape(rows // 32, k // 32).astype(np.int32)
    s = np.ldexp(np.ones_like(e, dtype=np.float32), e - 127)
    return v * np.repeat(np.repeat(s, 32, axis=0), 32, axis=1)


# --------------------------------------------------------------------------- #
# the three answers
# --------------------------------------------------------------------------- #

def expert_ffn3(w1: np.ndarray, w2: np.ndarray, w3: np.ndarray,
                x: np.ndarray, limit: float = SWIGLU_LIMIT) -> dict[str, np.ndarray]:
    """y_ref / y_hq / y_full for one x, as described in the module docstring."""
    xt = torch.from_numpy(x).to(torch.float32)
    t1 = torch.from_numpy(w1).to(torch.float32)
    t2 = torch.from_numpy(w2).to(torch.float32)
    t3 = torch.from_numpy(w3).to(torch.float32)

    def run(xin: torch.Tensor, quantise_h: bool, fp16_h: bool = False) -> np.ndarray:
        gate = torch.clamp(t1 @ xin, max=limit)
        up = torch.clamp(t3 @ xin, min=-limit, max=limit)
        h = torch.nn.functional.silu(gate) * up
        if fp16_h:
            # design section 6: h travels in fp16 (kernel_p1.md section 3.1
            # measured fp32 h as 9% slower for a 16% error improvement).
            h = h.to(torch.float16).to(torch.float32)
        if quantise_h:
            # model.py: self.w2(x.to(dtype)) -> linear() -> act_quant(..., 32).
            h = act_quant_dequant(h.reshape(1, -1)).reshape(-1)
        return (t2 @ h).numpy()

    xq = act_quant_dequant(xt.reshape(1, -1)).reshape(-1)
    x16 = xt.to(torch.float16).to(torch.float32)
    return {
        "y_ref": run(xt, False),
        "y_hq": run(xt, True),
        "y_full": run(xq, True),
        "y_hq16": run(x16, True, fp16_h=True),
    }


# --------------------------------------------------------------------------- #
# int8 x -- docs/kernel_p2_moe.md section 3.6 / section 8 item 2
#
# `XMode = 6` has gpu/shaders/moe_xquant.slang quantise x to int8 with a
# per-block-32 scale once per token, and dispatch A then consumes it with
# dot4add_i8packed. The weight side of that dot product is exact -- 2*E2M1 is
# an int8 and the factor 2 rides in the UE8M0 block exponent -- so the *only*
# error is x's own quantisation, which makes this study a faithful model of the
# kernel rather than an approximation of it. Dispatch B is left on the fp16
# path (its activation is h, not x), which is what the kernel does too.
#
# Three quantisers are compared, because section 8 item 2 asks whether a
# cheaper or a more accurate one changes the verdict:
#   i8_blk32   what the kernel does: amax/127 per 32 elements along K
#   i8_row     one scale for the whole 5120-element row -- one fewer scalar
#              load per (column, block) in the kernel
#   i8_resid   int8 plus the fp16 rounding of what int8 threw away, i.e. a
#              second activation plane and a second FMA stream
# --------------------------------------------------------------------------- #

def quant_i8(x: torch.Tensor, block: int | None) -> torch.Tensor:
    """Symmetric int8 round trip, amax/127 per `block` elements (None = per row).
    Identical to moe_common.slang i8_block_scale + i8_pack4."""
    n = x.numel() if block is None else block
    xf = x.float().reshape(-1, n)
    s = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30) / 127.0
    q = torch.clamp(torch.floor(xf / s + 0.5), -127.0, 127.0)
    return (q * s).reshape(x.shape)


def x_quant_study(w1: np.ndarray, w2: np.ndarray, w3: np.ndarray,
                  x: np.ndarray, y_ref: np.ndarray,
                  limit: float = SWIGLU_LIMIT) -> dict[str, float]:
    t1 = torch.from_numpy(w1).to(torch.float32)
    t2 = torch.from_numpy(w2).to(torch.float32)
    t3 = torch.from_numpy(w3).to(torch.float32)
    # design section 6: the kernel's x is fp16 before anything else happens.
    x16 = torch.from_numpy(x).to(torch.float16).to(torch.float32)

    def run(xin: torch.Tensor) -> np.ndarray:
        gate = torch.clamp(t1 @ xin, max=limit)
        up = torch.clamp(t3 @ xin, min=-limit, max=limit)
        h = (torch.nn.functional.silu(gate) * up).to(torch.float16).to(torch.float32)
        return (t2 @ h).numpy()

    q_blk = quant_i8(x16, FP8_BLOCK)
    q_row = quant_i8(x16, None)
    variants = {
        "fp16 x (XMode 0/4)": x16,
        "i8_blk32 (XMode 6)": q_blk,
        "i8_row": q_row,
        # The residual is what int8 dropped, carried in fp16. It needs a second
        # activation plane and a second FMA stream in the kernel, so it costs
        # more than just staying on packed fp16 -- the number is here to show
        # that the error really is x's quantisation and nothing else.
        "i8_blk32 + fp16 residual": q_blk + (x16 - q_blk).to(torch.float16).to(torch.float32),
    }
    scale = float(np.abs(y_ref).max())
    out = {}
    for name, xin in variants.items():
        y = run(xin)
        d = float(np.abs(y - y_ref).max())
        cos = float(np.dot(y, y_ref) / (np.linalg.norm(y) * np.linalg.norm(y_ref)))
        out[name] = d / scale
        print(f"    x as {name:<26} cos {cos:.9f}  "
              f"max|d| {d:.4g} ({d / scale:.3e} of |y|max)")
    return out


def report_deltas(name: str, ys: dict[str, np.ndarray]) -> None:
    ref = ys["y_ref"]
    scale = float(np.abs(ref).max())
    for key in ("y_hq", "y_full", "y_hq16"):
        d = ys[key] - ref
        cos = float(np.dot(ys[key], ref) /
                    (np.linalg.norm(ys[key]) * np.linalg.norm(ref)))
        print(f"    {name} {key} vs y_ref: cos {cos:.9f}  "
              f"max|d| {np.abs(d).max():.4g} ({np.abs(d).max() / scale:.3e} of |y|max)")
    d = ys["y_full"] - ys["y_hq"]
    print(f"    {name} y_full vs y_hq:  max|d| {np.abs(d).max():.4g} "
          f"({np.abs(d).max() / scale:.3e} of |y|max)  -- the cost of fp8 x")
    d = ys["y_hq16"] - ys["y_hq"]
    print(f"    {name} y_hq16 vs y_hq:  max|d| {np.abs(d).max():.4g} "
          f"({np.abs(d).max() / scale:.3e} of |y|max)  -- fp16 x/h through the "
          f"step-function scale")


# --------------------------------------------------------------------------- #
# writers
# --------------------------------------------------------------------------- #

def write_vecs(path: str, magic: bytes, version: int, header: bytes,
               vectors: list[np.ndarray]) -> int:
    with open(path, "wb") as f:
        f.write(magic)
        f.write(struct.pack("<I", version))
        f.write(header)
        for v in vectors:
            f.write(v.astype("<f4").tobytes())
    return os.path.getsize(path)


def shared_expert(reader: "oracle.ManifestReader", layer: int, seed: int,
                  out_dir: str | None) -> dict:
    prefix = f"layers.{layer}.ffn.shared_experts"
    table = fp8_e4m3_table()
    mats: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}
    for part in SHARED_PARTS:
        mat = part.split(".")[0]
        name = f"{prefix}.{mat}.weight"
        t = reader.m["tensors"][name]
        rows, k = int(t["shape"][0]), int(t["shape"][1])
        if t["dtype"] != "fp8_e4m3":
            raise SystemExit(f"{name} is {t['dtype']}, expected fp8_e4m3")
        values = reader.tensor_bytes(name)
        sc = t["scale"]
        scales = read_absolute(reader, sc["file"], sc["offset"], sc["bytes"])
        mats[mat] = dequant_fp8_tiles(values, scales, rows, k, table)
        meta[mat] = {
            "shape": [rows, k],
            "bytes": len(values),
            "scale_shape": list(t["scale"]["shape"]),
            "scale_bytes": len(scales),
            "value_hash64": oracle.block_hash64(values),
            "scale_hash64": oracle.block_hash64(scales),
            "value_sha256": hashlib.sha256(values).hexdigest(),
            "scale_sha256": hashlib.sha256(scales).hexdigest(),
        }

    dim = mats["w1"].shape[1]
    inter = mats["w1"].shape[0]
    rng = np.random.default_rng(seed + (layer << 20) + 0xB0B)
    x = rng.standard_normal(dim, dtype=np.float32)
    ys = expert_ffn3(mats["w1"], mats["w2"], mats["w3"], x)

    rep = {
        "kind": "shared", "layer": layer, "dim": dim, "inter_dim": inter, "seed": seed,
        "parts": meta,
        "w1_absmax": float(np.abs(mats["w1"]).max()),
        "w2_absmax": float(np.abs(mats["w2"]).max()),
        "w3_absmax": float(np.abs(mats["w3"]).max()),
        "y_absmax": float(np.abs(ys["y_ref"]).max()),
        "y_l2": float(np.linalg.norm(ys["y_ref"])),
    }
    print(f"shared expert layer {layer}: w1 {meta['w1']['shape']} fp8_e4m3, "
          f"scale {meta['w1']['scale_shape']}, {meta['w1']['bytes']:,} + "
          f"{meta['w1']['scale_bytes']:,} B per matrix")
    print(f"    |w1|max {rep['w1_absmax']:.6g}  |w2|max {rep['w2_absmax']:.6g}  "
          f"|w3|max {rep['w3_absmax']:.6g}")
    print(f"    |y|max  {rep['y_absmax']:.6g}  ||y||2 {rep['y_l2']:.6g}")
    report_deltas(f"shared[{layer}]", ys)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"l1_shared_layer{layer}.bin")
        header = struct.pack("<IIIQf", layer, dim, inter, seed, SWIGLU_LIMIT)
        for mat in ("w1", "w2", "w3"):
            header += struct.pack("<QQ", meta[mat]["value_hash64"], meta[mat]["scale_hash64"])
        n = write_vecs(path, SHARED_MAGIC, SHARED_VERSION, header,
                       [x, ys["y_ref"], ys["y_hq"], ys["y_full"], ys["y_hq16"]])
        rep["golden"] = path
        rep["golden_bytes"] = n
        print(f"    -> {path} ({n:,} B)")
    return rep


def read_absolute(reader: "oracle.ManifestReader", file_index: int,
                  offset: int, nbytes: int) -> bytes:
    f = reader._file(file_index)
    f.seek(offset)
    buf = f.read(nbytes)
    if len(buf) != nbytes:
        raise SystemExit(f"short read of {nbytes} B at {offset}")
    return buf


def routed_expert_quantised(reader: "oracle.ManifestReader", layer: int, expert: int,
                            seed: int, out_dir: str | None) -> dict:
    """The same (layer, expert, x) as `oracle.py --level l1`, with the two extra
    answers section 7.9 v0.6 needs. The FP4 decode and the x seed are taken from
    oracle.py so `y_ref` here is bit-identical to the y in l1_*.bin."""
    table = oracle.fp4_e2m1_table()
    slot, offsets, sizes = reader.expert_slot(layer, expert)
    parts = {p: slot[offsets[p]:offsets[p] + sizes[p]] for p in oracle.EXPERT_PARTS}
    dim = 5120
    inter = len(parts["w1.weight"]) * 2 // dim
    w1 = oracle.dequant_fp4(parts["w1.weight"], parts["w1.scale"], inter, dim, table)
    w3 = oracle.dequant_fp4(parts["w3.weight"], parts["w3.scale"], inter, dim, table)
    w2 = oracle.dequant_fp4(parts["w2.weight"], parts["w2.scale"], dim, inter, table)

    rng = np.random.default_rng(seed + (layer << 20) + expert)
    x = rng.standard_normal(dim, dtype=np.float32)
    ys = expert_ffn3(w1, w2, w3, x)

    print(f"routed ({layer}, {expert}): |y|max {np.abs(ys['y_ref']).max():.6g}  "
          f"||y||2 {np.linalg.norm(ys['y_ref']):.6g}")
    report_deltas(f"routed({layer},{expert})", ys)
    xq = x_quant_study(w1, w2, w3, x, ys["y_ref"])

    rep = {"kind": "routed", "layer": layer, "expert": expert, "dim": dim,
           "inter_dim": inter, "seed": seed,
           "y_absmax": float(np.abs(ys["y_ref"]).max()),
           "x_quant_rel": xq}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"l1q_layer{layer}_expert{expert}.bin")
        header = struct.pack("<IIIIQf", layer, expert, dim, inter, seed, SWIGLU_LIMIT)
        n = write_vecs(path, Q_MAGIC, Q_VERSION, header,
                       [x, ys["y_ref"], ys["y_hq"], ys["y_full"], ys["y_hq16"]])
        rep["golden"] = path
        rep["golden_bytes"] = n
        print(f"    -> {path} ({n:,} B)")
    return rep


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="checkpoint directory")
    p.add_argument("--manifest", default=None, help="deepmoe_manifest.json (default: in --model)")
    p.add_argument("--shared", action="append", type=int, default=None,
                   help="layer whose fp8 shared expert to write; repeatable")
    p.add_argument("--expert", action="append", default=None,
                   help="LAYER:EXPERT for the quantised-h routed golden; repeatable")
    p.add_argument("--seed", type=int, default=0, help="must match oracle.py's --seed")
    p.add_argument("--out", default=None, help="directory for the .bin goldens")
    p.add_argument("--report", default=None, help="write a JSON report here")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reader = oracle.ManifestReader(args.model, args.manifest)
    reports = []
    try:
        for layer in (args.shared if args.shared is not None else [0]):
            reports.append(shared_expert(reader, layer, args.seed, args.out))
        for spec in (args.expert or ["0:0", "39:383"]):
            layer, expert = oracle.parse_expert(spec)
            reports.append(routed_expert_quantised(reader, layer, expert, args.seed, args.out))
    finally:
        reader.close()
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2)
        print(f"report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
