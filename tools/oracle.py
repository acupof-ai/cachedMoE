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
# L2 / L3
# --------------------------------------------------------------------------- #

def level2_layer(args: argparse.Namespace) -> int:
    """Run one real layer in torch fp32 and compare per-layer outputs.

    TODO(design section 12 L2): decode this layer's weights through the manifest,
    run the mHC / attention / engram / MoE chain exactly as model.py does, and
    report cosine similarity plus max relative error against --compare. This is
    also where the FP4 nibble ORDER assumed by L1 becomes falsifiable.
    """
    raise NotImplementedError("oracle.level2_layer (design section 12 L2)")


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
