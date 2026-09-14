#!/usr/bin/env python3
"""fp32 reference forward pass for DeepSeek-V4.1-Flash -- the correctness anchor.

Design reference: section 12 (the four-level oracle). This is a P0 deliverable
(section 15).

There is no official CUDA environment to compare against, no GGUF, and no
llama.cpp build that understands V4.1. The oracle is therefore self-built, and
section 12 is blunt about what that costs: `oracle.py` is the trust anchor, and
its own correctness rests on a line-by-line review against `inference/model.py`
plus a dual-implementation agreement test. That review is not optional and must
be written down.

Levels
------
L0 decode   FP4 E2M1 / FP8 E4M3 / UE8M0 -> fp32, exported as golden vectors to
            tests/data/l0_dequant.bin so tests/test_dequant.cpp is pinned to the
            checkpoint's semantics rather than to our own reading of the spec.

            Where each table comes from:
              * FP8 E4M3 and UE8M0 are produced by torch itself
                (torch.float8_e4m3fn / torch.float8_e8m0fnu), which is exactly
                what `inference/kernel.py` casts through.
              * FP4 E2M1 has no CPU conversion in torch ("copy_kernel not
                implemented for Float4_e2m1fn_x2"), so the 16-entry table is
                built from the OCP E2M1 definition and cross-checked against
                `ml_dtypes.float4_e2m1fn` when that package is importable. The
                agreement is exact: {0, .5, 1, 1.5, 2, 3, 4, 6} with a sign bit,
                which also matches `fp4_max = 6.0` in inference/kernel.py.
              * The *nibble order* (low nibble = even element along K) is
                PyTorch's `float4_e2m1fn_x2` packing, which inference/kernel.py
                relies on for `B: [N, K//2] FP4, logical [N, K]`. Nothing below
                L2 can falsify the order -- permuting nibbles inside a byte
                leaves every per-32 block's value multiset unchanged -- so it is
                recorded here as an assumption and is what L2 will test first if
                layer outputs disagree.

L1 kernel   One routed expert's FFN, in fp32, for a seeded random x:
                y = w2( silu(clamp(w1 x, max=L)) * clamp(w3 x, -L, L) ),  L = 10
            The six tensors are loaded TWICE: once through deepmoe_manifest.json
            (open the shard, read [aligned_off, aligned_off+aligned_bytes),
            slice at `skew`) and once through the `safetensors` library. The two
            byte streams must be identical -- that is the test of the manifest's
            run/skew arithmetic -- and the resulting y is written to
            tests/data/l1_*.bin for the C++ integration test to reproduce from
            an ExpertStore slot filled by the real IoEngine.

L2 layer    one real layer vs deepMoE's per-layer output (cosine >= 0.999).
L3 e2e      full-model fp32 greedy decode, token-for-token agreement.

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

Usage
-----
    uv run python tools/oracle.py --model D:/models/DeepSeek-V4.1-Flash \\
        --level l0 --out tests/data
    uv run python tools/oracle.py --model D:/models/DeepSeek-V4.1-Flash \\
        --level l1 --expert 0:0 --expert 39:383 --out tests/data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time

import numpy as np
import torch

# Design section 2.1 / appendix A.
FP4_BLOCK = 32          # one UE8M0 scale per 32 elements along K
SWIGLU_LIMIT = 10.0     # ModelArgs.swiglu_limit in config.json
DEFAULT_MANIFEST = "deepmoe_manifest.json"

L0_MAGIC = b"DMQ0"
L0_VERSION = 1
L1_MAGIC = b"DML1"
L1_VERSION = 1

# model/manifest.h ExpertPart order.
EXPERT_PARTS = ("w1.weight", "w1.scale", "w2.weight", "w2.scale", "w3.weight", "w3.scale")


# --------------------------------------------------------------------------- #
# hashing shared with C++ (tests/test_integration.cpp implements the same)
# --------------------------------------------------------------------------- #

HASH_SEED = np.uint64(0xCBF29CE484222325)
HASH_STEP = np.uint64(0x9E3779B97F4A7C15)
HASH_M1 = np.uint64(0xBF58476D1CE4E5B9)
HASH_M2 = np.uint64(0x94D049BB133111EB)


def block_hash64(data: bytes) -> int:
    """A cheap 64-bit checksum of a byte range, position-sensitive but
    order-independent in its accumulation, so numpy can compute it on 17 MB in
    milliseconds and tests/test_integration.cpp can re-implement it in ten lines.

    Each little-endian 64-bit word is offset by its index times the golden-ratio
    constant, run through the SplitMix64 finaliser, and XOR-accumulated; the
    byte length is folded into the seed. A tail shorter than 8 bytes is
    zero-padded, which the length term keeps unambiguous. This is not a
    cryptographic hash -- the SHA-256 of every part is recorded alongside it --
    it exists so a C++ test can say "the bytes in this slot are the bytes the
    oracle read" without linking a crypto library.
    """
    a = np.frombuffer(data, dtype=np.uint8)
    pad = (-a.size) % 8
    if pad:
        a = np.concatenate([a, np.zeros(pad, dtype=np.uint8)])
    w = a.view("<u8")
    idx = np.arange(1, w.size + 1, dtype=np.uint64)
    x = w + HASH_STEP * idx
    x ^= x >> np.uint64(30)
    x *= HASH_M1
    x ^= x >> np.uint64(27)
    x *= HASH_M2
    x ^= x >> np.uint64(31)
    h = HASH_SEED ^ np.uint64(len(data))
    h ^= np.bitwise_xor.reduce(x) if w.size else np.uint64(0)
    return int(h)


# --------------------------------------------------------------------------- #
# L0: decode tables
# --------------------------------------------------------------------------- #

def fp4_e2m1_table() -> np.ndarray:
    """The 16 values of OCP FP4 E2M1: sign | 2-bit exponent (bias 1) | 1-bit mantissa.

    exp == 0 is subnormal: value = m * 2^-1 = m/2.
    exp >  0 is normal:    value = (1 + m/2) * 2^(exp-1).
    No infinities, no NaN -- every one of the 16 codes is a finite number.
    """
    out = np.zeros(16, dtype=np.float32)
    for code in range(16):
        sign = -1.0 if (code & 0x8) else 1.0
        exp = (code >> 1) & 0x3
        man = code & 0x1
        if exp == 0:
            mag = man * 0.5
        else:
            mag = (1.0 + man * 0.5) * (2.0 ** (exp - 1))
        out[code] = np.float32(sign * mag)
    # -0.0 must survive, so the sign is applied to the magnitude, not copysign'd
    # onto a zero that numpy might normalise.
    out[8] = np.float32(-0.0)
    return out


def fp8_e4m3_table() -> np.ndarray:
    """Produced by torch, which is what inference/kernel.py casts through."""
    b = torch.arange(256, dtype=torch.uint8)
    return b.view(torch.float8_e4m3fn).float().numpy().astype(np.float32)


def e8m0_table() -> np.ndarray:
    """UE8M0: value = 2^(e-127); 0xFF is NaN. Also straight from torch."""
    b = torch.arange(256, dtype=torch.uint8)
    return b.view(torch.float8_e8m0fnu).float().numpy().astype(np.float32)


def cross_check_fp4(table: np.ndarray) -> str:
    try:
        import ml_dtypes  # optional; an independent implementation of the same spec
    except ImportError:
        return "ml_dtypes not installed -- FP4 table not cross-checked"
    ref = np.arange(16, dtype=np.uint8).view(ml_dtypes.float4_e2m1fn).astype(np.float32)
    same = all(
        (np.signbit(a) == np.signbit(b)) and (a == b or (np.isnan(a) and np.isnan(b)))
        for a, b in zip(table, ref)
    )
    return f"ml_dtypes.float4_e2m1fn agrees: {same}"


def level0_tables(args: argparse.Namespace) -> int:
    fp4 = fp4_e2m1_table()
    fp8 = fp8_e4m3_table()
    e8m0 = e8m0_table()

    print("FP4 E2M1 (16 codes):")
    print("  " + " ".join(f"{v:g}" for v in fp4))
    print("  " + cross_check_fp4(fp4))
    print(f"FP8 E4M3: min {np.nanmin(fp8):g}  max {np.nanmax(fp8):g}  "
          f"NaN codes {np.flatnonzero(np.isnan(fp8)).tolist()}")
    print(f"UE8M0:    2^-127 = {e8m0[0]:g} .. 2^127 = {e8m0[254]:g}  "
          f"NaN codes {np.flatnonzero(np.isnan(e8m0)).tolist()}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, "l0_dequant.bin")
        with open(path, "wb") as f:
            f.write(L0_MAGIC)
            f.write(struct.pack("<IIII", L0_VERSION, fp4.size, fp8.size, e8m0.size))
            f.write(fp4.astype("<f4").tobytes())
            f.write(fp8.astype("<f4").tobytes())
            f.write(e8m0.astype("<f4").tobytes())
        print(f"\nwrote {path} ({os.path.getsize(path)} B)")
    return 0


# --------------------------------------------------------------------------- #
# L1: one routed expert's FFN
# --------------------------------------------------------------------------- #

class ManifestReader:
    """Reads tensors the way the runtime does: by aligned run plus skew."""

    def __init__(self, model_dir: str, manifest_path: str | None = None):
        self.dir = model_dir
        path = manifest_path or os.path.join(model_dir, DEFAULT_MANIFEST)
        if not os.path.exists(path):
            raise SystemExit(f"{path} is missing -- run tools/manifest.py first")
        with open(path, "rb") as f:
            self.m = json.load(f)
        if self.m["version"] != 2:
            raise SystemExit(f"manifest version {self.m['version']}, expected 2")
        self.align = self.m["alignment"]
        self._handles: dict[int, object] = {}

    def _file(self, index: int):
        h = self._handles.get(index)
        if h is None:
            h = open(os.path.join(self.dir, self.m["files"][index]["path"]), "rb", buffering=0)
            self._handles[index] = h
        return h

    def close(self):
        for h in self._handles.values():
            h.close()
        self._handles.clear()

    def read_run(self, run: dict) -> bytes:
        """The IoEngine's job: one sector-aligned read, no interpretation."""
        assert run["aligned_off"] % self.align == 0, "run offset is not sector aligned"
        assert run["aligned_bytes"] % self.align == 0, "run length is not a sector multiple"
        f = self._file(run["file"])
        f.seek(run["aligned_off"])
        buf = f.read(run["aligned_bytes"])
        if len(buf) != run["aligned_bytes"]:
            raise SystemExit(f"short read of {run['aligned_bytes']} B at {run['aligned_off']}")
        return buf

    def expert_slot(self, layer: int, expert: int) -> tuple[bytes, dict, dict]:
        """Fills one slot exactly as store::ExpertStore does: runs back to back.

        Returns (slot bytes, {part -> slot_offset}, {part -> bytes}).
        """
        table = next((L for L in self.m["experts"] if L["layer"] == layer), None)
        if table is None:
            raise SystemExit(f"manifest has no expert layer {layer}")
        runs = table["experts"][expert]
        slot = bytearray()
        offsets, sizes = {}, {}
        for run in runs:
            assert len(slot) == run["slot_offset"], "runs are not laid back to back"
            slot += self.read_run(run)
            for part in run["parts"]:
                assert part["slot_offset"] == run["slot_offset"] + part["skew"]
                offsets[part["tensor"]] = part["slot_offset"]
                sizes[part["tensor"]] = part["bytes"]
        return bytes(slot), offsets, sizes

    def tensor_bytes(self, name: str) -> bytes:
        t = self.m["tensors"][name]
        f = self._file(t["file"])
        a_off = (t["offset"] // self.align) * self.align
        a_end = -(-(t["offset"] + t["bytes"]) // self.align) * self.align
        f.seek(a_off)
        buf = f.read(a_end - a_off)
        skew = t["offset"] - a_off
        return buf[skew:skew + t["bytes"]]


def safetensors_expert(model_dir: str, prefix: str, expert: int) -> dict[str, bytes]:
    """The same six tensors, via the safetensors library -- the control group."""
    from safetensors import safe_open

    with open(os.path.join(model_dir, "model.safetensors.index.json"), "rb") as f:
        weight_map = json.load(f)["weight_map"]
    out: dict[str, bytes] = {}
    handles: dict[str, object] = {}
    try:
        for part in EXPERT_PARTS:
            mat, what = part.split(".")
            name = f"{prefix}.ffn.experts.{expert}.{mat}.{what}"
            shard = weight_map[name]
            h = handles.get(shard)
            if h is None:
                h = safe_open(os.path.join(model_dir, shard), framework="pt")
                h.__enter__()
                handles[shard] = h
            t = h.get_tensor(name)
            out[part] = t.view(torch.uint8).contiguous().numpy().tobytes()
    finally:
        for h in handles.values():
            h.__exit__(None, None, None)
    return out


def dequant_fp4(packed: bytes, scales: bytes, rows: int, k: int,
                table: np.ndarray) -> np.ndarray:
    """[rows, k//2] packed E2M1 + [rows, k//32] UE8M0 -> fp32 [rows, k].

    Low nibble = even element along K (PyTorch float4_e2m1fn_x2).
    """
    assert k % FP4_BLOCK == 0
    b = np.frombuffer(packed, dtype=np.uint8).reshape(rows, k // 2)
    lo = table[b & 0x0F]
    hi = table[b >> 4]
    vals = np.empty((rows, k), dtype=np.float32)
    vals[:, 0::2] = lo
    vals[:, 1::2] = hi
    e = np.frombuffer(scales, dtype=np.uint8).reshape(rows, k // FP4_BLOCK).astype(np.int32)
    # 2^(e-127) exactly, without a pow: ldexp on the mantissa 1.0.
    scale = np.ldexp(np.ones_like(e, dtype=np.float32), e - 127)
    return vals * np.repeat(scale, FP4_BLOCK, axis=1)


def expert_ffn(w1: np.ndarray, w2: np.ndarray, w3: np.ndarray,
               x: np.ndarray, limit: float = SWIGLU_LIMIT) -> np.ndarray:
    """inference/model.py Expert.forward, in fp32 with no activation quantisation.

        gate = clamp(w1 x, max=limit)
        up   = clamp(w3 x, -limit, limit)
        y    = w2( silu(gate) * up )

    `weights` (the routing weight) is None here: L1 tests one expert in
    isolation, and the multiply is linear so it adds nothing to the test.
    """
    xt = torch.from_numpy(x).to(torch.float32)
    gate = torch.from_numpy(w1).to(torch.float32) @ xt
    up = torch.from_numpy(w3).to(torch.float32) @ xt
    gate = torch.clamp(gate, max=limit)
    up = torch.clamp(up, min=-limit, max=limit)
    h = torch.nn.functional.silu(gate) * up
    return (torch.from_numpy(w2).to(torch.float32) @ h).numpy()


def level1_expert(reader: ManifestReader, layer: int, expert: int,
                  model_dir: str, seed: int, out_dir: str | None) -> dict:
    table = fp4_e2m1_table()
    prefix = f"layers.{layer}" if layer < 40 else f"mtp.{layer - 40}"

    t0 = time.perf_counter()
    slot, offsets, sizes = reader.expert_slot(layer, expert)
    via_manifest = {p: slot[offsets[p]:offsets[p] + sizes[p]] for p in EXPERT_PARTS}
    t_manifest = time.perf_counter() - t0

    t0 = time.perf_counter()
    via_safetensors = safetensors_expert(model_dir, prefix, expert)
    t_st = time.perf_counter() - t0

    agree = {p: via_manifest[p] == via_safetensors[p] for p in EXPERT_PARTS}
    if not all(agree.values()):
        bad = [p for p, ok in agree.items() if not ok]
        raise SystemExit(f"({layer}, {expert}): manifest and safetensors disagree on {bad}")

    # shapes, from the manifest's own tensor table where available
    dim = 5120
    inter = len(via_manifest["w1.weight"]) * 2 // dim
    assert inter * dim // 2 == len(via_manifest["w1.weight"])

    w1 = dequant_fp4(via_manifest["w1.weight"], via_manifest["w1.scale"], inter, dim, table)
    w3 = dequant_fp4(via_manifest["w3.weight"], via_manifest["w3.scale"], inter, dim, table)
    w2 = dequant_fp4(via_manifest["w2.weight"], via_manifest["w2.scale"], dim, inter, table)

    rng = np.random.default_rng(seed + (layer << 20) + expert)
    x = rng.standard_normal(dim, dtype=np.float32)
    y = expert_ffn(w1, w2, w3, x)

    hashes = {p: block_hash64(via_manifest[p]) for p in EXPERT_PARTS}
    sha = {p: hashlib.sha256(via_manifest[p]).hexdigest() for p in EXPERT_PARTS}

    report = {
        "layer": layer, "expert": expert, "prefix": prefix,
        "dim": dim, "inter_dim": inter, "seed": seed,
        "slot_bytes": len(slot),
        "runs": len(next(L for L in reader.m["experts"] if L["layer"] == layer)
                    ["experts"][expert]),
        "manifest_read_s": round(t_manifest, 3),
        "safetensors_read_s": round(t_st, 3),
        "bytes_agree": True,
        "w1_absmax": float(np.abs(w1).max()), "w2_absmax": float(np.abs(w2).max()),
        "w3_absmax": float(np.abs(w3).max()),
        "y_absmax": float(np.abs(y).max()), "y_l2": float(np.linalg.norm(y)),
        "y_mean": float(y.mean()),
        "part_hash64": {p: f"{hashes[p]:#018x}" for p in EXPERT_PARTS},
        "part_sha256": sha,
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"l1_layer{layer}_expert{expert}.bin")
        with open(path, "wb") as f:
            f.write(L1_MAGIC)
            f.write(struct.pack("<IIIIIQ", L1_VERSION, layer, expert, dim, inter, seed))
            f.write(struct.pack("<f", SWIGLU_LIMIT))
            f.write(struct.pack("<I", len(slot)))
            for p in EXPERT_PARTS:
                f.write(struct.pack("<QQQ", offsets[p], sizes[p], hashes[p]))
            f.write(x.astype("<f4").tobytes())
            f.write(y.astype("<f4").tobytes())
            for p in EXPERT_PARTS:
                f.write(sha[p].encode("ascii"))
        report["golden"] = path
        report["golden_bytes"] = os.path.getsize(path)
    return report


def parse_expert(spec: str) -> tuple[int, int]:
    try:
        layer, expert = spec.split(":")
        return int(layer), int(expert)
    except ValueError:
        raise SystemExit(f"--expert wants LAYER:EXPERT, got {spec!r}")


def level1_kernel(args: argparse.Namespace) -> int:
    specs = [parse_expert(s) for s in (args.expert or ["0:0", "39:383"])]
    reader = ManifestReader(args.model, args.manifest)
    reports = []
    try:
        for layer, expert in specs:
            r = level1_expert(reader, layer, expert, args.model, args.seed, args.out)
            reports.append(r)
            print(f"({layer}, {expert}) {r['runs']} runs, slot {r['slot_bytes']:,} B; "
                  f"manifest {r['manifest_read_s']}s vs safetensors {r['safetensors_read_s']}s; "
                  f"bytes agree")
            print(f"    |w1|max {r['w1_absmax']:.6g}  |w2|max {r['w2_absmax']:.6g}  "
                  f"|w3|max {r['w3_absmax']:.6g}")
            print(f"    |y|max  {r['y_absmax']:.6g}  ||y||2 {r['y_l2']:.6g}  "
                  f"mean {r['y_mean']:.6g}")
            if "golden" in r:
                print(f"    -> {r['golden']} ({r['golden_bytes']:,} B)")
    finally:
        reader.close()
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2)
        print(f"report -> {args.report}")
    return 0


# --------------------------------------------------------------------------- #
# L2: per-stage golden tensors for one token, straight out of inference/model.py
# --------------------------------------------------------------------------- #
#
# What L2 is for
# --------------
# L1 proved one expert FFN. L2 has to prove the other ten dispatches of design
# section 7.14, one at a time, which means golden data at every boundary between
# them -- not just a per-layer output. So this exports, for a single token
# position, the input and output of every stage in the section 7.14 order, taken
# from `inference/model.py` running unmodified behind tools/dsref.py's CPU
# kernel shims.
#
# How the capture works, and why it is not a re-implementation
# -----------------------------------------------------------
# Nothing here recomputes any stage. Every tensor below is either
#   * an `nn.Module` forward hook's input or output (wq_a, q_norm, wq_b, wkv,
#     kv_norm, attn_norm, ffn_norm, wo_b, the compressor's norm, the indexer's
#     wq_b / wk / k_norm / weights_proj), or
#   * an argument or result of one of the four module-level functions the
#     reference routes through (`act_quant`, `fp4_act_quant`, `sparse_attn`,
#     plus `Block.hc_mixes` / `Block.hc_post`).
#
# RoPE is the one thing with no capture point of its own: `apply_rotary_emb`
# rewrites its argument in place and returns nothing anyone stores. It is
# captured by difference instead -- a hook clones its output (the pre-RoPE
# value) and also keeps the live tensor; reading that same tensor again after
# `Attention.forward` returns gives the post-RoPE value, because `unflatten`
# and `x[..., -64:]` are views of the very storage the hook saw. The same trick
# gives the post-inverse-RoPE attention output.
#
# Precision
# ---------
# `generate.py` runs under `torch.set_default_dtype(torch.bfloat16)`, so the
# residual stream, every Linear output and the attention inputs are **bf16**,
# while the mHC coefficients, the gate and the expert accumulation are fp32.
# This export keeps exactly that and records the storage dtype per tensor: a
# tensor written as "bf16" held a bf16 value in the reference, and deepMoE's
# fp32 residual stream is therefore *more* precise than what produced it. That
# is the source of most of the tolerance in tests/test_gpu_attn.cpp, and it is
# why the exported `block_in` is the right thing to feed a kernel: it is the
# reference's own input, bit for bit.

L2_MAGIC = b"DML2"
L2_VERSION = 1

# design section 2.4 / route_trace.md section 11.3: the two fp4 activation
# formats are NOT the same. Compressed KV is block 16 with an E4M3 scale;
# everything the indexer quantises is block 32 with an E8M0 scale.
L2_CMP_KV_BLOCK = 16
L2_INDEX_BLOCK = 32

L2_PROMPT = (
    "The expert cache holds about thirty percent of the routed experts, so the "
    "decode loop is bounded by NVMe rather than by memory bandwidth.\n"
    "这台机器的真实瓶颈是 NVMe，而不是 LPDDR 带宽。\n"
    "def hit_rate(hits: int, total: int) -> float:\n"
    "    return hits / total if total else 0.0\n"
)

_L2_DTYPES = {"f32": (torch.float32, 4), "bf16": (torch.bfloat16, 2),
              "i32": (torch.int32, 4), "u8": (torch.uint8, 1)}


class L2Writer:
    """One flat binary per (layer, step) plus a JSON directory.

    A self-describing container would need a parser on the C++ side; a JSON
    index plus raw concatenated tensors needs only core/json.h, which already
    exists, and stays readable with `python -c` when a test disagrees.
    """

    def __init__(self, out_dir: str):
        self.dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.steps: list[dict] = []

    def write(self, layer: int, step: str, tensors: dict) -> dict:
        name = f"l2_L{layer:02d}_{step}.bin"
        entries, blob = [], bytearray()
        for key, (kind, t) in tensors.items():
            dtype, width = _L2_DTYPES[kind]
            a = t.detach().contiguous()
            if a.dtype != dtype:
                a = a.to(dtype)
            raw = a.view(torch.uint8).numpy().tobytes() if kind == "bf16" \
                else a.numpy().tobytes()
            assert len(raw) == a.numel() * width, (key, len(raw), a.numel())
            entries.append({"name": key, "dtype": kind, "shape": list(a.shape),
                            "offset": len(blob), "bytes": len(raw)})
            blob += raw
        with open(os.path.join(self.dir, name), "wb") as f:
            f.write(L2_MAGIC)
            f.write(struct.pack("<III", L2_VERSION, layer, len(entries)))
            f.write(bytes(blob))
        rec = {"layer": layer, "step": step, "file": name,
               "data_offset": len(L2_MAGIC) + 12, "bytes": len(blob),
               "tensors": entries}
        self.steps.append(rec)
        return rec

    def finish(self, meta: dict) -> str:
        meta = dict(meta)
        meta["steps"] = self.steps
        path = os.path.join(self.dir, "index.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1, ensure_ascii=False)
        return path


def _e8m0_code(scale: torch.Tensor) -> torch.Tensor:
    """A power-of-two fp32 scale -> its UE8M0 byte. `2^e` has code `e + 127`,
    which is just the fp32 exponent field, so no log is involved."""
    bits = scale.float().contiguous().view(torch.int32)
    return ((bits >> 23) & 0xFF).to(torch.uint8)


def _pack_fp4(vals: torch.Tensor, mags: torch.Tensor) -> torch.Tensor:
    """E2M1 values -> packed nibbles, low nibble = the EVEN element along K.

    The inverse of cpu/dequant.cpp and of dsref.dequant_fp4_bf16, and the one
    place this file commits to the nibble order design section 12 records as an
    assumption. `mags` is the eight magnitudes in code order.
    """
    v = vals.float().reshape(-1)
    code = torch.searchsorted(mags, v.abs().contiguous())
    code = torch.where(v < 0, code + 8, code).to(torch.uint8)
    code = code.reshape(*vals.shape[:-1], vals.shape[-1])
    lo, hi = code[..., 0::2].int(), code[..., 1::2].int()
    return (lo | (hi << 4)).to(torch.uint8)


def _l2_quantise_fp8(dsref_mod, x: torch.Tensor, block: int = 32):
    """`act_quant(..., scale_fmt="ue8m0", inplace=True)` with its bytes exposed.

    Returns (dequantised value, fp8 e4m3 bytes, ue8m0 scale bytes). This is what
    the window KV cache write of design section 7.4 has to produce; the caller
    asserts the dequantised value equals the one the reference computed, so the
    byte layout is pinned to the reference and not to this function.
    """
    xf = x.float().reshape(-1, x.shape[-1] // block, block)
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    s = dsref_mod._round_scale_pow2(amax, 1.0 / 448.0)
    q = torch.clamp(xf / s, -448.0, 448.0).to(torch.float8_e4m3fn)
    value = (q.float() * s).reshape(x.shape)
    return (value,
            q.reshape(x.shape).view(torch.uint8).contiguous(),
            _e8m0_code(s.reshape(*x.shape[:-1], -1)).contiguous())


def _l2_quantise_fp4(dsref_mod, x: torch.Tensor, block: int, e4m3_scale: bool):
    """`fp4_act_quant` with its bytes exposed (route_trace.md section 11.3).

    e4m3_scale=True is the compressed KV (block 16, amax floor 6 * 2^-9, the
    scale itself cast to fp8); False is the indexer (block 32, amax floor
    6 * 2^-126, scale rounded up to a power of two).
    """
    xf = x.float().reshape(-1, x.shape[-1] // block, block)
    amax = xf.abs().amax(dim=-1, keepdim=True)
    if e4m3_scale:
        amax = amax.clamp_min(6.0 * 2.0 ** -9)
        s = (amax / 6.0).to(torch.float8_e4m3fn)
        scale_bytes = s.view(torch.uint8).reshape(*x.shape[:-1], -1).contiguous()
        s = s.float()
    else:
        amax = amax.clamp_min(6.0 * 2.0 ** -126)
        s = dsref_mod._round_scale_pow2(amax, 1.0 / 6.0)
        scale_bytes = _e8m0_code(s.reshape(*x.shape[:-1], -1)).contiguous()
    q = dsref_mod._round_to_fp4(torch.clamp(xf / s, -6.0, 6.0))
    value = (q * s).reshape(x.shape)
    packed = _pack_fp4(q.reshape(x.shape), dsref_mod._FP4_MAGS)
    return value, packed, scale_bytes


class L2Capture:
    """Records one block's per-stage tensors without changing any arithmetic.

    `enable(pos)` selects the token position to keep; every capture stores the
    whole batched tensor and slices at the end, because the reference computes a
    prefill chunk in one shot and slicing early would not save anything.
    """

    def __init__(self, ref):
        self.ref = ref
        self.on = False
        self.t: dict = {}
        self.live: dict = {}
        self.mixes: list = []
        self.posts: list = []
        self.q8: list = []
        self.q4: list = []
        self.sparse: dict | None = None
        self._handles: list = []
        self._install()

    # -- module-level interception ---------------------------------------

    def _install(self):
        ref = self.ref
        cap = self

        orig_act = ref.act_quant
        def act_quant(x, *a, **k):
            if cap.on and len(a) >= 4 and a[3] is True:
                pre = x.detach().clone()
                out = orig_act(x, *a, **k)
                cap.q8.append((pre, out.detach().clone()))
                return out
            return orig_act(x, *a, **k)
        ref.act_quant = act_quant

        orig_fp4 = ref.fp4_act_quant
        def fp4_act_quant(x, block_size=32, inplace=False, **k):
            if cap.on and inplace:
                pre = x.detach().clone()
                out = orig_fp4(x, block_size, inplace, **k)
                cap.q4.append((block_size, k.get("scale_dtype", torch.float8_e8m0fnu),
                               pre, out.detach().clone()))
                return out
            return orig_fp4(x, block_size, inplace, **k)
        ref.fp4_act_quant = fp4_act_quant

        orig_sparse = ref.sparse_attn
        def sparse_attn(q, kv, sink, topk_idxs, scale):
            out = orig_sparse(q, kv, sink, topk_idxs, scale)
            if cap.on:
                cap.sparse = {"q": q.detach().clone(), "kv": kv.detach().clone(),
                              "sink": sink.detach().clone(),
                              "idx": topk_idxs.detach().clone(),
                              "o_pre_inverse": out.detach().clone()}
                cap.live["o"] = out
            return out
        ref.sparse_attn = sparse_attn

        orig_mixes = ref.Block.hc_mixes
        def hc_mixes(self_b, x, fn, sc, base):
            r = orig_mixes(self_b, x, fn, sc, base)
            if cap.on:
                cap.mixes.append(tuple(v.detach().clone() for v in r))
            return r
        ref.Block.hc_mixes = hc_mixes

        orig_post = ref.Block.hc_post
        def hc_post(self_b, x, residual, post, comb):
            r = orig_post(self_b, x, residual, post, comb)
            if cap.on:
                cap.posts.append(r.detach().clone())
            return r
        ref.Block.hc_post = hc_post

    # -- per-block hooks --------------------------------------------------

    def attach(self, block):
        b = block.b
        def hook(name, keep_input=False):
            def fn(_m, inp, out):
                if not self.on:
                    return
                o = out[0] if isinstance(out, tuple) else out
                self.t[name] = o.detach().clone()
                self.live[name] = o
                if keep_input:
                    self.t[name + ".in"] = inp[0].detach().clone()
            return fn
        pairs = [("attn_norm", b.attn_norm, True), ("ffn_norm", b.ffn_norm, True),
                 ("wq_a", b.attn.wq_a, False), ("q_norm", b.attn.q_norm, False),
                 ("wq_b", b.attn.wq_b, False), ("wkv", b.attn.wkv, False),
                 ("kv_norm", b.attn.kv_norm, False), ("wo_b", b.attn.wo_b, True)]
        if b.attn.compressor is not None:
            pairs += [("cmp_wkv", b.attn.compressor.wkv, False),
                      ("cmp_norm", b.attn.compressor.norm, False)]
        if b.attn.indexer is not None:
            ix = b.attn.indexer
            pairs += [("idx_q", ix.wq_b, False), ("idx_w", ix.weights_proj, False)]
            if ix.owns_k:
                pairs += [("idx_k", ix.k_norm, False)]
        for name, mod, ki in pairs:
            self._handles.append(mod.register_forward_hook(hook(name, ki)))

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def reset(self, on: bool):
        self.on = on
        self.t, self.live = {}, {}
        self.mixes, self.posts, self.q8, self.q4 = [], [], [], []
        self.sparse = None


def _l2_collect(dsref_mod, cap: L2Capture, block, margs, pos: int,
                start_pos: int, n_win: int, extra: dict) -> dict:
    """Turn one captured block into the named-tensor dict the exporter writes.

    `pos` is the index of the token of interest inside the chunk that was run,
    `start_pos` its absolute position, `n_win` how many of the KV rows
    `sparse_attn` saw are window rows. That last one is not a constant: prefill
    attends over the chunk itself (seqlen rows, seeded into the ring afterwards)
    while decode attends over the whole 128-slot ring (design section 7.5).
    """
    b = block.b
    nh, hd = margs.n_heads, margs.head_dim
    t = cap.t
    out: dict = {}

    def put(name, kind, v):
        out[name] = (kind, v)

    def tok(v):
        """[b, s, ...] -> the one token, as a contiguous tensor."""
        return v[0, pos].contiguous()

    for k, v in extra.items():
        out[k] = v

    # --- 1. mega-mHC (attention half): design section 7.2 ------------------
    attn_pre, attn_post, attn_comb = cap.mixes[0]
    put("attn_pre", "f32", tok(attn_pre))
    put("attn_post", "f32", tok(attn_post))
    put("attn_comb", "f32", tok(attn_comb))
    put("attn_hc_pre_out", "bf16", tok(t["attn_norm.in"]))
    put("attn_norm_out", "bf16", tok(t["attn_norm"]))

    # --- 2/3. Q path: design section 7.3 -----------------------------------
    put("wq_a_out", "bf16", tok(t["wq_a"]))
    put("qr", "bf16", tok(t["q_norm"]))
    put("q_pre_rope", "bf16", tok(t["wq_b"]).reshape(nh, hd))
    put("q", "bf16", cap.live["wq_b"][0, pos].reshape(nh, hd).contiguous())

    # --- 4. KV path + the fp8 cache write: design section 7.4 --------------
    put("wkv_out", "bf16", tok(t["wkv"]))
    put("kv_pre_rope", "bf16", tok(t["kv_norm"]))
    kv_pre_quant, kv_post_quant = cap.q8[0]
    put("kv_pre_quant", "bf16", tok(kv_pre_quant))
    put("kv", "bf16", tok(kv_post_quant))
    value, qbytes, sbytes = _l2_quantise_fp8(dsref_mod, tok(kv_pre_quant), 32)
    assert torch.equal(value.bfloat16(), tok(kv_post_quant).bfloat16()), \
        "the fp8 window-cache quantisation here is not the reference's"
    put("kv_fp8", "u8", qbytes)
    put("kv_scale_e8m0", "u8", sbytes)

    # --- compressor / indexer on a source layer: design section 7.4 --------
    if b.attn.compressor is not None and "cmp_norm" in t:
        latent = t["cmp_norm"]
        lp = min(pos // b.attn.compress_ratio, latent.size(1) - 1) if start_pos == 0 else 0
        put("latent_pre_rope", "bf16", latent[0, lp].contiguous())
        cmp = [e for e in cap.q4 if e[0] == L2_CMP_KV_BLOCK]
        if cmp:
            _, _, pre, post = cmp[0]
            put("latent_pre_quant", "bf16", pre[0, lp].contiguous())
            put("latent", "bf16", post[0, lp].contiguous())
            v, pk, sc = _l2_quantise_fp4(dsref_mod, pre[0, lp].contiguous(),
                                         L2_CMP_KV_BLOCK, True)
            assert torch.equal(v.bfloat16(), post[0, lp].bfloat16()), \
                "compressed-KV fp4 quantisation disagrees with the reference"
            put("latent_fp4", "u8", pk)
            put("latent_scale_e4m3", "u8", sc)
    if b.attn.indexer is not None:
        ix = b.attn.indexer
        idxq = [e for e in cap.q4 if e[0] == L2_INDEX_BLOCK and e[2].ndim == 4]
        idxk = [e for e in cap.q4 if e[0] == L2_INDEX_BLOCK and e[2].ndim == 3]
        put("index_q_pre_rope", "bf16",
            t["idx_q"][0, pos].reshape(ix.n_heads, ix.index_head_dim).contiguous())
        if idxq:
            _, _, pre, post = idxq[0]
            put("index_q_pre_quant", "bf16", pre[0, pos].contiguous())
            put("index_q", "bf16", post[0, pos].contiguous())
        if idxk and "idx_k" in t:
            _, _, pre, post = idxk[0]
            kp = pre.size(1) - 1
            put("index_k_pre_rope", "bf16", t["idx_k"][0, kp].contiguous())
            put("index_k_pre_quant", "bf16", pre[0, kp].contiguous())
            put("index_k", "bf16", post[0, kp].contiguous())
            v, pk, sc = _l2_quantise_fp4(dsref_mod, pre[0, kp].contiguous(),
                                         L2_INDEX_BLOCK, False)
            assert torch.equal(v.bfloat16(), post[0, kp].bfloat16()), \
                "indexer fp4 quantisation disagrees with the reference"
            put("index_k_fp4", "u8", pk)
            put("index_k_scale_e8m0", "u8", sc)
        put("index_weights", "bf16", t["idx_w"][0, pos].contiguous())

    # --- 5. sparse attention: design section 7.5 ---------------------------
    sp = cap.sparse
    kv_all = sp["kv"][0]
    idx = sp["idx"][0, pos]
    put("attn_sink", "f32", sp["sink"])
    put("win_kv", "bf16", kv_all[:n_win].contiguous())
    if kv_all.size(0) > n_win:
        put("cmp_kv", "bf16", kv_all[n_win:].contiguous())
    put("topk_idxs", "i32", idx.contiguous())
    put("attn_out", "bf16", sp["o_pre_inverse"][0, pos].reshape(nh, hd).contiguous())
    put("attn_out_irope", "bf16", cap.live["o"][0, pos].reshape(nh, hd).contiguous())

    # --- 6/7. output projection: design section 7.6 ------------------------
    put("wo_a_out", "bf16", tok(t["wo_b.in"]))
    put("wo_b_out", "bf16", tok(t["wo_b"]))
    put("attn_block_out", "bf16", cap.posts[0][0, pos].contiguous())

    # --- 8. mega-mHC (ffn half) + hc_post: design sections 7.2, 7.7 --------
    ffn_pre, ffn_post, ffn_comb = cap.mixes[1]
    put("ffn_pre", "f32", tok(ffn_pre))
    put("ffn_post", "f32", tok(ffn_post))
    put("ffn_comb", "f32", tok(ffn_comb))
    put("ffn_hc_pre_out", "bf16", tok(t["ffn_norm.in"]))
    put("ffn_norm_out", "bf16", tok(t["ffn_norm"]))
    put("block_out", "bf16", cap.posts[1][0, pos].contiguous())
    return out


_L2_ATTN_BUFFERS = (("window_kv_cache", ()), ("compress_kv_cache", ()),
                    ("compressor", ("kv_state", "score_state")),
                    ("indexer", ("k_cache",)))


def _l2_save_attn_state(block) -> dict:
    """Clone every KV buffer one Attention accumulated during prefill.

    The decode pass rebuilds each Block from the shards (5.1 GB of attention
    weights, ~0.5 s a layer) rather than holding forty of them in memory; these
    buffers are all that has to survive, and they are ~200 KiB a layer.
    """
    a, out = block.b.attn, {}
    for name, subs in _L2_ATTN_BUFFERS:
        holder = getattr(a, name, None)
        if holder is None:
            continue
        if not subs:
            out[name] = holder.clone()
            continue
        for s in subs:
            v = getattr(holder, s, None)
            if v is not None:
                out[f"{name}.{s}"] = v.clone()
    return out


def _l2_load_attn_state(block, state: dict):
    a = block.b.attn
    for key, v in state.items():
        if "." in key:
            holder, sub = key.split(".", 1)
            getattr(getattr(a, holder), sub).copy_(v)
        else:
            getattr(a, key).copy_(v)


def level2_layer(args: argparse.Namespace) -> int:
    """Export per-stage golden tensors for a prefill chunk and one decode step.

    The loop is layer-major and builds each Block exactly once: prefill the
    whole prompt through layer L, then immediately run the decode step for
    position `prefill_len` through the same module, so the KV caches the decode
    step reads are the ones the prefill just wrote and the 5 GB of attention
    weights are touched once. The two passes carry independent residual streams
    and independent `SharedAttentionRuntime` snapshots.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import dsref  # noqa: E402  (needs sys.path first)

    torch.set_grad_enabled(False)
    inference_dir = os.path.join(args.model, "inference")
    ref = dsref.load_reference(inference_dir)
    store = dsref.WeightStore(args.model, args.manifest)
    tokenizer = dsref.TokenizerAdapter(os.path.join(args.model, "tokenizer.json"))

    ids = tokenizer.encode(L2_PROMPT)[: args.l2_tokens + 1]
    n_prefill = min(args.l2_tokens, len(ids) - 1)
    if n_prefill < 8:
        raise SystemExit(f"the L2 prompt tokenises to only {len(ids)} tokens")
    decode_pos = n_prefill
    # freqs_cis is precomputed for max_seq_len rows and the compressed caches are
    # max_seq_len // ratio deep, so this only has to cover prefill + one step.
    max_seq_len = max(128, n_prefill + 8)
    margs = dsref.build_args(ref, inference_dir, max_seq_len=max_seq_len)
    layout_e = ref.EngramLayout.from_args(margs)

    want = sorted({int(s) for s in args.l2_layers.split(",") if s.strip() != ""})
    for L in want:
        if not 0 <= L < margs.n_layers:
            raise SystemExit(f"--l2-layers: layer {L} is outside 0..{margs.n_layers - 1}")

    print(f"L2: {n_prefill} prefill tokens + one decode step at position {decode_pos}; "
          f"exporting layers {want}")

    ngram = None
    os.makedirs(args.out, exist_ok=True)
    if args.l2_engram:
        sys.path.insert(0, inference_dir)
        import engram as eng  # noqa: E402
        cached = dsref.CachedTokenMap.build(tokenizer, os.path.join(args.out, "token_map.npz"))
        orig_build = eng.build_compressed_token_map
        eng.build_compressed_token_map = lambda _t: (cached.lookup, cached.size)
        try:
            ngram = ref.NgramHashState(margs, layout_e, tokenizer)
        finally:
            eng.build_compressed_token_map = orig_build

    embed = store.tensor("embed.weight")
    pre_ids = torch.tensor(ids[:n_prefill], dtype=torch.long).unsqueeze(0)
    dec_ids = torch.tensor([ids[decode_pos]], dtype=torch.long).unsqueeze(0)
    h_pre = embed[pre_ids[0]].unsqueeze(1).repeat(1, margs.hc_mult, 1).unsqueeze(0)
    h_dec = embed[dec_ids[0]].unsqueeze(1).repeat(1, margs.hc_mult, 1).unsqueeze(0)
    del embed
    h_pre = h_pre.to(torch.bfloat16)
    h_dec = h_dec.to(torch.bfloat16)
    pre_mix_pre = ref.make_identity_pre_mix(h_pre, margs.hc_mult)
    pre_mix_dec = ref.make_identity_pre_mix(h_dec, margs.hc_mult)

    hashes_pre = ngram(pre_ids, 0, None) if ngram is not None else None
    hashes_dec = ngram(dec_ids, decode_pos, None) if ngram is not None else None

    shared = ref.shared_attn
    shared.compress_kv = shared.index_k = shared.topk_idxs = shared.candidates = None
    cap = L2Capture(ref)
    writer = L2Writer(args.out)
    steps = [s for s in args.l2_steps.split(",") if s.strip()]
    kv_state: dict[int, dict] = {}

    def run_layer(L, block, h, pre_mix, hashes, start_pos, do, step_name):
        """One Block on one chunk, with the MoE run expert-major by the driver."""
        pos = (h.size(1) - 1) if start_pos == 0 else 0
        cap.reset(do)
        extra: dict = {"pre_mix_in": ("f32", pre_mix[0, pos].contiguous())}
        if do:
            extra["block_in"] = ("bf16", h[0, pos].contiguous())
        if block.b.engram is not None and hashes is not None:
            hi = layout_e.layer_ids.index(L)
            h_after = block.b.engram(h, hashes[:, :, hi, :], None)
            if do:
                extra["engram_out"] = ("bf16", h_after[0, pos].contiguous())
            h = h_after
        if do:
            extra["attn_resid_in"] = ("bf16", h[0, pos].contiguous())

        ffn_in, resid, fpre, fpost, fcomb = block.forward_attn(h, start_pos, pre_mix)

        moe = block.b.ffn
        flat = ffn_in.view(-1, margs.dim)
        weights, indices = moe.route(flat)
        y = torch.zeros(flat.size(0), margs.dim, dtype=torch.float32)
        used = sorted(set(indices.reshape(-1).tolist()))
        for e, w1, w2, w3 in store.expert_stream(L, used, margs.dim, margs.moe_inter_dim):
            r, s = torch.where(indices == e)
            y[r] += dsref.expert_ffn(flat[r], w1, w2, w3, weights[r, s, None],
                                     margs.swiglu_limit).float()
            del w1, w2, w3
        shared_out = moe.shared_experts(flat).float()
        y += shared_out
        h_out = block.finish_ffn(y.unsqueeze(0), resid, fpost, fcomb)

        if do:
            extra["gate_scores"] = ("f32", moe.gate_scores(flat)[pos].contiguous())
            extra["gate_bias"] = ("f32", moe.gate.bias.float().contiguous())
            extra["gate_top6_ids"] = ("i32", indices[pos].int().contiguous())
            extra["gate_top6_weights"] = ("f32", weights[pos].float().contiguous())
            extra["moe_shared_out"] = ("f32", shared_out[pos].contiguous())
            extra["moe_out"] = ("f32", y[pos].contiguous())
            n_win = h.size(1) if start_pos == 0 else margs.window_size
            tensors = _l2_collect(dsref, cap, block, margs, pos, start_pos, n_win, extra)
            rec = writer.write(L, step_name, tensors)
            print(f"    layer {L:2d} {step_name:10s} -> {rec['file']} "
                  f"({rec['bytes'] / 1024:.0f} KiB, {len(rec['tensors'])} tensors)")
        return h_out, fpre, len(used)

    t_start = time.perf_counter()

    # --- pass 1: prefill ---------------------------------------------------
    # A full pass first, not prefill-then-decode interleaved per layer, because
    # `shared_attn` is a process-wide singleton that model.py deliberately never
    # resets ("every source writes before its consumers read"). A decode step
    # therefore starts with whatever the *last* source layer of the previous
    # forward published -- which for a ratio-2 source whose group is incomplete
    # at this position is layer 20's index keys, not its own. Reproducing that
    # is the whole point of an oracle, so the decode pass has to begin where a
    # complete prefill pass ended.
    for L in range(margs.n_layers):
        t_layer = time.perf_counter()
        block = dsref.make_block(ref, margs, L, layout_e, store, args.l2_engram_threads)
        cap.attach(block)
        do = (L in want) and ("prefill" in steps)
        h_pre, pre_mix_pre, n_used = run_layer(L, block, h_pre, pre_mix_pre, hashes_pre,
                                               0, do, f"prefill{n_prefill - 1}")
        kv_state[L] = _l2_save_attn_state(block)
        cap.detach()
        del block
        print(f"    prefill layer {L:2d}  {n_used:3d} experts  "
              f"{time.perf_counter() - t_layer:5.1f}s  |h| "
              f"{h_pre.float().norm().item():.4e}", flush=True)
    # --- pass 2: one decode step at `decode_pos` ---------------------------
    for L in range(margs.n_layers):
        t_layer = time.perf_counter()
        block = dsref.make_block(ref, margs, L, layout_e, store, args.l2_engram_threads)
        _l2_load_attn_state(block, kv_state[L])
        cap.attach(block)
        do = (L in want) and ("decode" in steps)
        h_dec, pre_mix_dec, n_used = run_layer(L, block, h_dec, pre_mix_dec, hashes_dec,
                                               decode_pos, do, f"decode{decode_pos}")
        cap.detach()
        del block
        print(f"    decode  layer {L:2d}  {n_used:3d} experts  "
              f"{time.perf_counter() - t_layer:5.1f}s  |h| "
              f"{h_dec.float().norm().item():.4e}", flush=True)

    meta = {
        "version": L2_VERSION,
        "model": "DeepSeek-V4.1-Flash",
        "generator": "tools/oracle.py --level l2",
        "prompt": L2_PROMPT,
        "prompt_ids": [int(i) for i in ids[: n_prefill + 1]],
        "prefill_len": n_prefill,
        "decode_pos": decode_pos,
        "engram": bool(args.l2_engram),
        "stream_dtype": "bf16",
        "notes": (
            "Every tensor is one token of the reference's own computation, taken "
            "from inference/model.py behind tools/dsref.py's CPU kernel shims. "
            "dtype is the dtype the reference actually held: the residual stream "
            "and every Linear output are bf16 because generate.py sets the default "
            "dtype to bf16, while the mHC coefficients, the gate and the MoE "
            "accumulation are fp32. Stage order follows design section 7.14."),
        "layers": want,
        "config": {"dim": margs.dim, "hc_mult": margs.hc_mult, "n_heads": margs.n_heads,
                   "head_dim": margs.head_dim, "rope_head_dim": margs.rope_head_dim,
                   "q_lora_rank": margs.q_lora_rank, "o_lora_rank": margs.o_lora_rank,
                   "o_groups": margs.o_groups, "window_size": margs.window_size,
                   "n_routed_experts": margs.n_routed_experts,
                   "n_activated_experts": margs.n_activated_experts,
                   "moe_inter_dim": margs.moe_inter_dim,
                   "norm_eps": margs.norm_eps, "swiglu_limit": margs.swiglu_limit,
                   "route_scale": margs.route_scale,
                   "index_n_heads": margs.index_n_heads,
                   "index_head_dim": margs.index_head_dim,
                   "index_topk": margs.index_topk,
                   "compress_ratios": list(margs.compress_ratios),
                   "kv_source_layers": list(margs.kv_source_layers),
                   "index_source_layers": list(margs.index_source_layers),
                   "max_seq_len": max_seq_len},
        "seconds": round(time.perf_counter() - t_start, 1),
    }
    path = writer.finish(meta)
    total = sum(s["bytes"] for s in writer.steps)
    print(f"\nwrote {len(writer.steps)} steps, {total / 1e6:.2f} MB -> {path} "
          f"in {meta['seconds']}s")
    store.close()
    return 0


def level3_end_to_end(args: argparse.Namespace) -> int:
    """Full-model fp32 greedy decode.

    TODO(design section 12 L3): minutes per token, >= 5 prompts x 64 tokens.
    Report per-token agreement and, at any divergence, the top1-top2 margin --
    a disagreement at a margin of 1e-6 is a different finding from one at 0.5.
    """
    raise NotImplementedError("oracle.level3_end_to_end (design section 12 L3)")


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="oracle.py",
        description="fp32 reference forward pass (design section 12)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", required=True,
                   help="the safetensors directory; L1 reads it both through "
                        "deepmoe_manifest.json and through the safetensors library")
    p.add_argument("--manifest", default=None,
                   help=f"manifest path (default: <model>/{DEFAULT_MANIFEST})")
    p.add_argument("--level", choices=["l0", "l1", "l2", "l3"], required=True,
                   help="l0 decode tables, l1 one expert FFN, l2 one layer, l3 end to end")
    p.add_argument("--expert", action="append", default=None, metavar="LAYER:EXPERT",
                   help="which expert for --level l1; repeatable. "
                        "Default: 0:0 and 39:383")
    p.add_argument("--layer", type=int, default=None, help="which layer for --level l2")
    # --- L2 ---------------------------------------------------------------
    p.add_argument("--l2-layers", default="0,1,2,13,14,20,39",
                   help="which layers to export golden tensors for. The default "
                        "covers every structural variant: 0 (window only), "
                        "1 (engram, window only), 2 (kv+index source, ratio 2), "
                        "13 (plain reuse), 14 (engram + kv/index source), "
                        "20 (first decoder, ratio 1, candidate source), 39 (last). "
                        "Every layer is still COMPUTED -- only the export is "
                        "restricted, since layer L's input is layer L-1's output")
    p.add_argument("--l2-tokens", type=int, default=64,
                   help="prefill length; the decode step runs at this position")
    p.add_argument("--l2-steps", default="decode",
                   help="comma separated subset of {prefill, decode}. decode is the "
                        "one design section 7 targets; prefill adds the last prompt "
                        "position, which is the same arithmetic with a different "
                        "KV shape and roughly doubles the export size")
    p.add_argument("--l2-engram", action="store_true", default=True,
                   help="run the layer 1 / 14 n-gram lookups (default on: off is "
                        "not the model)")
    p.add_argument("--no-l2-engram", dest="l2_engram", action="store_false")
    p.add_argument("--l2-engram-threads", type=int, default=32)
    p.add_argument("--compare", default=None, help=".npy produced by deepMoE to compare against")
    p.add_argument("--prompts", default=None, help="one prompt per line, for --level l3")
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
    p.add_argument("--out", default="tests/data",
                   help="directory for the golden vectors; empty string to skip writing")
    p.add_argument("--report", default=None, help="write the run report as JSON here")
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.set_num_threads(min(16, os.cpu_count() or 4))
    return {
        "l0": level0_tables,
        "l1": level1_kernel,
        "l2": level2_layer,
        "l3": level3_end_to_end,
    }[args.level](args)


if __name__ == "__main__":
    sys.exit(main())
