#!/usr/bin/env python3
"""CPU reference runner for DeepSeek-V4.1-Flash, built on `inference/model.py` itself.

Design reference: section 2.4 (the details that are easy to get wrong), section 12
(the oracle), section 9.2 (what `tools/route_trace.py` needs). This module is the
engine under `tools/route_trace.py`; it is not a CLI.

The trust strategy
------------------
Design section 12 is blunt that `oracle.py` is a hand port and therefore the weakest
link in the chain. This module takes the opposite approach wherever it can: instead
of re-deriving the model, it **imports `inference/model.py` unchanged** and supplies
CPU implementations of the six tilelang kernels it cannot run here:

    act_quant  fp4_act_quant  fp8_gemm  fp4_gemm  hc_split_sinkhorn  sparse_attn

So mHC/Sinkhorn, the gate, the experts, attention with its compressor / indexer /
candidate pre-filter, RoPE and YaRN, and the Engram hashing are all the reference's
own code, running on its own `inference/config.json`. What is ours, and therefore
what has to be reviewed, is:

  1. the six kernel shims (below, each with the tilelang source it mirrors quoted),
  2. the manifest-backed weight loader (`WeightStore`), which is the same
     `deepmoe_manifest.json` arithmetic `tools/oracle.py` L1 already proved
     byte-identical to the `safetensors` library,
  3. `StreamingMoE` / `StreamingBlock`, which do not change any math but split
     `Block.forward` into phases so the driver can stream one layer of expert
     weights at a time (see `tools/route_trace.py` for why that is mandatory), and
  4. the Engram row reader, which fetches the 24 rows a token needs instead of
     materialising a 98 GB embedding table.

Numerics
--------
`generate.py` runs with `torch.set_default_dtype(torch.bfloat16)`, so the residual
stream, every `Linear` output and the attention inputs are **bf16**, while mHC
coefficients, the gate and the expert accumulation are fp32. We keep exactly that.

Weights are dequantised once, into bf16, and this is lossless, not an approximation:

  * FP4 E2M1 has magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6} -- one mantissa bit -- and
    UE8M0 scales are exact powers of two, so `value * scale` always fits bf16's
    eight mantissa bits.
  * FP8 E4M3 has three mantissa bits, and its block scales are UE8M0 as well, so
    the same argument holds.

A bf16 x bf16 matmul on this CPU accumulates in fp32 and rounds the *output* to
bf16, which is precisely what `fp8_gemm` / `fp4_gemm` do (fp32 accumulator,
`out_dtype = BF16`). The activation-side quantisation those kernels perform is not
skipped either -- `act_quant_dequant` below reproduces it bit for bit, including
`fast_round_scale`'s power-of-two rounding -- so the only deliberate difference from
the reference is fp32 accumulation *order*. `tools/route_trace.py --verify` measures
what that is worth.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# constants mirroring inference/model.py's module globals
# ---------------------------------------------------------------------------

FP8_BLOCK = 32          # model.py: fp8_block_size
FP4_BLOCK = 32          # model.py: fp4_block_size
FP8_MAX = 448.0
FP4_MAX = 6.0
DEFAULT_MANIFEST = "deepmoe_manifest.json"

# tools/oracle.py EXPERT_PARTS, in the same order.
EXPERT_PARTS = ("w1.weight", "w1.scale", "w2.weight", "w2.scale", "w3.weight", "w3.scale")


# ---------------------------------------------------------------------------
# 1. quantisation, as the tilelang kernels define it
# ---------------------------------------------------------------------------

def _round_scale_pow2(amax: torch.Tensor, max_inv: float) -> torch.Tensor:
    """`fast_round_scale` from inference/kernel.py, exactly.

        fast_log2_ceil(x) = (exp(x) - 127) + (mantissa(x) != 0)
        fast_pow2(e)      = reinterpret((e + 127) << 23)
        fast_round_scale(amax, inv) = fast_pow2(fast_log2_ceil(amax * inv))

    i.e. the smallest power of two >= amax/max. Doing it on the fp32 bit pattern
    rather than through `log2` matters: `ceil(log2(x))` in floating point gets the
    exact powers of two wrong about as often as it gets them right.
    """
    v = (amax.float() * max_inv).contiguous()
    bits = v.view(torch.int32)
    exp = (bits >> 23) & 0xFF
    man = bits & ((1 << 23) - 1)
    e = exp - 127 + (man != 0).to(torch.int32)
    return torch.ldexp(torch.ones_like(v), e)


def act_quant_dequant(x: torch.Tensor, block_size: int = FP8_BLOCK) -> torch.Tensor:
    """`act_quant(x, block_size, scale_fmt="ue8m0", scale_dtype=e8m0)` followed by the
    dequantisation `fp8_gemm` applies to the accumulator -- the round trip an
    activation makes on its way through any quantised `Linear`.

    From act_quant_kernel: absmax per (row, group of `block_size` along the last
    dim), floored at 1e-4; `round_scale=True` because `scale_fmt` is set; then
    `fp8(clamp(x / s, -448, 448)) * s`.
    """
    shape = x.shape
    n = shape[-1]
    assert n % block_size == 0, (n, block_size)
    xf = x.float().reshape(-1, n // block_size, block_size)
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    s = _round_scale_pow2(amax, 1.0 / FP8_MAX)
    q = torch.clamp(xf / s, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()
    return (q * s).reshape(shape).to(x.dtype)


# E2M1 magnitudes and the midpoints between them. Ties round to even *mantissa*,
# which alternates: 0.25 -> 0, 0.75 -> 1.0, 1.25 -> 1.0, 1.75 -> 2.0, 2.5 -> 2.0,
# 3.5 -> 4.0, 5.0 -> 4.0. So a tie at an odd boundary index takes the upper value.
_FP4_MAGS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
_FP4_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)


def _round_to_fp4(v: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even onto the 16 E2M1 values; `v` must already be clamped."""
    mag = v.abs()
    i = torch.searchsorted(_FP4_BOUNDS, mag.reshape(-1).contiguous()).reshape(mag.shape)
    # exact tie on an odd boundary index rounds up
    tie = (i < _FP4_BOUNDS.numel()) & (mag == _FP4_BOUNDS[i.clamp_max(_FP4_BOUNDS.numel() - 1)])
    i = i + (tie & (i % 2 == 1)).to(i.dtype)
    return torch.copysign(_FP4_MAGS[i.clamp_max(7)], v)


def fp4_quant_dequant(x: torch.Tensor, block_size: int = FP4_BLOCK,
                      scale_dtype: torch.dtype = torch.float8_e8m0fnu) -> torch.Tensor:
    """`fp4_act_quant(..., inplace=True)` from inference/kernel.py.

    Two scale formats, and the floors differ with them (fp4_quant_kernel):
      * UE8M0 (the indexer's q/k): amax floored at 6 * 2^-126, scale rounded up to
        a power of two.
      * E4M3 (the compressed KV, block 16): amax floored at 6 * 2^-9 so an all-zero
        group still gets a nonzero scale, and the scale itself is cast to fp8.
    """
    shape = x.shape
    n = shape[-1]
    assert n % block_size == 0, (n, block_size)
    xf = x.float().reshape(-1, n // block_size, block_size)
    amax = xf.abs().amax(dim=-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        amax = amax.clamp_min(FP4_MAX * 2.0 ** -9)
        s = (amax / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp_min(FP4_MAX * 2.0 ** -126)
        s = _round_scale_pow2(amax, 1.0 / FP4_MAX)
    q = _round_to_fp4(torch.clamp(xf / s, -FP4_MAX, FP4_MAX))
    return (q * s).reshape(shape).to(x.dtype)


# ---------------------------------------------------------------------------
# 2. weight decode
# ---------------------------------------------------------------------------

# The sixteen E2M1 values, in code order (OCP: sign | 2-bit exponent | 1-bit mantissa;
# exponent 0 is subnormal). This is the table convert.py's FP4_TABLE spells out and
# the one tools/oracle.py L0 cross-checks against `ml_dtypes.float4_e2m1fn`.
_FP4_TABLE_BF16 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.bfloat16)


def dequant_fp4_bf16(packed, scales, rows: int, k: int) -> torch.Tensor:
    """[rows, k/2] packed E2M1 + [rows, k/32] UE8M0 -> bf16 [rows, k], losslessly.

    Low nibble = even element along K (PyTorch `float4_e2m1fn_x2`; design section 12
    records the nibble order as the one assumption nothing below L2 can falsify).

    Exact, not approximate: an E2M1 value has one mantissa bit and a UE8M0 scale is
    an exact power of two, so the product always lands on a representable bf16. The
    fp32 intermediate is there because torch has no bf16 broadcast multiply that
    beats it, not because the bf16 result needs rounding -- and `tools/route_trace.py
    --verify` checks the whole thing bit for bit against `dequant_fp4_fp32_ref`.

    torch rather than numpy because this runs 15,360 times per pass over the corpus
    (384 experts x 40 layers x 3 matrices) and torch parallelises the gather and the
    multiply across cores; the numpy version measured 7x slower.
    """
    assert k % FP4_BLOCK == 0
    b = _as_u8(packed).reshape(rows, k // 2)
    vals = torch.empty(rows, k, dtype=torch.bfloat16)
    vals[:, 0::2] = _FP4_TABLE_BF16[(b & 0x0F).long()]
    vals[:, 1::2] = _FP4_TABLE_BF16[(b >> 4).long()]
    s = _as_u8(scales).view(torch.float8_e8m0fnu).reshape(rows, k // FP4_BLOCK).float()
    return (vals.view(rows, k // FP4_BLOCK, FP4_BLOCK).float()
            * s.unsqueeze(-1)).view(rows, k).bfloat16()


def _as_u8(buf) -> torch.Tensor:
    """A uint8 view of bytes / bytearray / memoryview without copying when possible."""
    if isinstance(buf, torch.Tensor):
        return buf
    if isinstance(buf, (bytes, memoryview)):
        buf = bytearray(buf)
    return torch.frombuffer(buf, dtype=torch.uint8)


def dequant_fp4_fp32_ref(packed: bytes, scales: bytes, rows: int, k: int) -> torch.Tensor:
    """The obvious float implementation, kept as the test oracle for the fast path.
    Identical to `tools/oracle.py:dequant_fp4`."""
    table = np.zeros(16, dtype=np.float32)
    for code in range(16):
        sign = -1.0 if (code & 0x8) else 1.0
        exp, man = (code >> 1) & 0x3, code & 0x1
        mag = man * 0.5 if exp == 0 else (1.0 + man * 0.5) * (2.0 ** (exp - 1))
        table[code] = np.float32(sign * mag)
    b = np.frombuffer(packed, dtype=np.uint8).reshape(rows, k // 2)
    vals = np.empty((rows, k), dtype=np.float32)
    vals[:, 0::2] = table[b & 0x0F]
    vals[:, 1::2] = table[b >> 4]
    e = np.frombuffer(scales, dtype=np.uint8).reshape(rows, k // FP4_BLOCK).astype(np.int32)
    scale = np.ldexp(np.ones_like(e, dtype=np.float32), e - 127)
    return torch.from_numpy(vals * np.repeat(scale, FP4_BLOCK, axis=1))


def _fp8_rows_to_bf16(value_bytes: bytes, scale_bytes: bytes, rows: int, cols: int,
                      scale_cols: int) -> torch.Tensor:
    """FP8 E4M3 + per-32 UE8M0 along the last dim -> bf16, used for engram rows."""
    v = torch.frombuffer(bytearray(value_bytes), dtype=torch.uint8).view(torch.float8_e4m3fn)
    s = torch.frombuffer(bytearray(scale_bytes), dtype=torch.uint8).view(torch.float8_e8m0fnu)
    v = v.reshape(rows, cols).float().unflatten(-1, (scale_cols, -1))
    s = s.reshape(rows, scale_cols).float().unsqueeze(-1)
    return (v * s).flatten(-2).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# 3. the six CPU kernel shims
# ---------------------------------------------------------------------------

def _k_act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    """inference/kernel.py `act_quant`. Only the two call shapes model.py uses are
    supported: `inplace=True` (the window KV cache) and the (y, s) form consumed by
    `fp8_gemm`, which we serve pre-dequantised because our weights are already bf16."""
    assert scale_fmt == "ue8m0" and scale_dtype == torch.float8_e8m0fnu, (scale_fmt, scale_dtype)
    y = act_quant_dequant(x, block_size)
    if inplace:
        x.copy_(y)
        return x
    # `_k_fp8_gemm` below ignores the scale and multiplies the dequantised value, so
    # handing back (dequantised x, ones) keeps fp8_gemm's contract without a second
    # quantisation. Nothing else in model.py reads these scales.
    s = x.new_ones(*x.shape[:-1], x.size(-1) // block_size, dtype=torch.float32)
    return y, s


def _k_fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    """inference/kernel.py `fp4_act_quant`. model.py only ever calls it inplace."""
    assert inplace, "model.py only uses fp4_act_quant(..., inplace=True)"
    x.copy_(fp4_quant_dequant(x, block_size, scale_dtype))
    return x


def _k_fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    """`C[M,N] = A[M,K] @ B[N,K]^T`. `a` arrives already dequantised from
    `_k_act_quant`, and `b` has been dequantised to bf16 at load time (lossless: see
    the module docstring), so this is one bf16 matmul with an fp32 accumulator --
    the same thing fp8_gemm_kernel computes, modulo accumulation order."""
    return F.linear(a, b)


def _k_fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    """Same argument as `_k_fp8_gemm`; only reached if a routed expert is ever run
    through model.py's own `Expert` (the streaming path below bypasses it)."""
    return F.linear(a, b)


def _k_hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """inference/kernel.py `hc_split_sinkhorn_kernel`, transcribed.

        pre[j]     = sigmoid(mixes[j]      * scale[0] + base[j]) + eps
        post[j]    = 2 * sigmoid(mixes[j+hc] * scale[1] + base[j+hc])
        comb[j,k]  = mixes[j*hc + k + 2hc] * scale[2] + base[j*hc + k + 2hc]
        comb       = softmax(comb, dim=-1) + eps
        comb      /= comb.sum(-2) + eps
        repeat sinkhorn_iters - 1 times:  /= sum(-1) + eps ;  /= sum(-2) + eps

    Twenty iterations leave `comb` doubly stochastic (design section 2.4)."""
    hc = hc_mult
    m = mixes.float()
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2.0 * torch.sigmoid(m[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (m[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


SPARSE_ATTN_QUERY_CHUNK = 64


def _k_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """inference/kernel.py `sparse_attn_kernel`, in torch.

    q [b,m,h,d] bf16, kv [b,n,d] bf16, attn_sink [h] fp32, topk_idxs [b,m,topk] int32
    with -1 for "this slot holds nothing".

    Two details from the kernel that a naive port loses:
      * the running max starts at a *finite* -1e30, not -inf, so a query whose whole
        index row is -1 comes out all zero instead of NaN;
      * the probabilities are rounded to bf16 (`acc_s_cast`) before the P.V gemm.

    Chunked over queries because the gathered KV is [chunk, topk, 512] and a 2048
    token prompt with topk = 128 + 512 would otherwise want gigabytes.
    """
    b, m, h, d = q.shape
    topk = topk_idxs.size(-1)
    out = torch.empty_like(q)
    sink = attn_sink.float().view(1, 1, h)
    for lo in range(0, m, SPARSE_ATTN_QUERY_CHUNK):
        hi = min(lo + SPARSE_ATTN_QUERY_CHUNK, m)
        idx = topk_idxs[:, lo:hi].long()                        # [b, mc, topk]
        valid = idx >= 0
        safe = idx.clamp_min(0)
        gathered = torch.stack(
            [kv[bi].index_select(0, safe[bi].reshape(-1)).reshape(hi - lo, topk, d)
             for bi in range(b)]).float()
        # bf16 inputs upcast exactly, so an fp32 matmul == the kernel's bf16 gemm
        # with an fp32 accumulator.
        scores = torch.einsum("bmhd,bmkd->bmhk", q[:, lo:hi].float(), gathered)
        scores = scores * softmax_scale
        scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
        mx = scores.amax(dim=-1).clamp_min(-1e30)               # [b, mc, h]
        p = torch.exp(scores - mx.unsqueeze(-1))
        p = torch.where(valid.unsqueeze(2), p, torch.zeros((), dtype=p.dtype))
        p = p.bfloat16().float()                                # acc_s_cast
        denom = p.sum(dim=-1) + torch.exp(sink - mx)
        o = torch.einsum("bmhk,bmkd->bmhd", p, gathered) / denom.unsqueeze(-1)
        out[:, lo:hi] = o.to(q.dtype)
    return out


def _make_kernel_module() -> types.ModuleType:
    mod = types.ModuleType("kernel")
    mod.act_quant = _k_act_quant
    mod.fp4_act_quant = _k_fp4_act_quant
    mod.fp8_gemm = _k_fp8_gemm
    mod.fp4_gemm = _k_fp4_gemm
    mod.hc_split_sinkhorn = _k_hc_split_sinkhorn
    mod.sparse_attn = _k_sparse_attn
    return mod


def _make_image_processor_module() -> types.ModuleType:
    """image_processor.py needs Pillow and does image preprocessing we never touch;
    model.py imports four constants from it (image_processor.py lines 22-23)."""
    mod = types.ModuleType("image_processor")
    mod.TEXT = -1
    mod.IMAGE_START, mod.IMAGE, mod.IMAGE_NEW_LINE, mod.IMAGE_END = range(4)
    return mod


_REF_LOCK = threading.Lock()
_REF: types.ModuleType | None = None


def load_reference(inference_dir: str) -> types.ModuleType:
    """Import `inference/model.py` with CPU kernels behind it. Idempotent."""
    global _REF
    with _REF_LOCK:
        if _REF is not None:
            return _REF
        inference_dir = os.path.abspath(inference_dir)
        if not os.path.isfile(os.path.join(inference_dir, "model.py")):
            raise SystemExit(f"{inference_dir} has no model.py")
        sys.modules.setdefault("kernel", _make_kernel_module())
        sys.modules.setdefault("image_processor", _make_image_processor_module())
        sys.path.insert(0, inference_dir)
        import model as ref  # noqa: E402  (needs the shims installed first)
        # model.py resolves `linear` through its own module globals, so patching it
        # here reroutes every Linear in the reference. See `_linear` for why.
        ref._deepmoe_orig_linear = ref.linear
        ref.linear = _linear
        _REF = ref
        return ref


def _linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None):
    """Replaces model.py's `linear`.

    The reference dispatches on the *storage* dtype of the weight (fp4 / fp8 / other).
    We store every weight dequantised, so that dispatch would silently skip the
    activation quantisation. `WeightStore` therefore tags each weight that came from
    a quantised `Linear` with `.act_block`, and this function keeps the round trip.
    """
    assert bias is None
    block = getattr(weight, "act_block", None)
    if block is not None:
        x = act_quant_dequant(x, block)
    return F.linear(x, weight)


# ---------------------------------------------------------------------------
# 4. manifest-backed weight store
# ---------------------------------------------------------------------------

class WeightStore:
    """Reads tensors out of the original safetensors shards through
    `deepmoe_manifest.json`, the same run/skew arithmetic `tools/oracle.py` L1 proved
    byte-identical to the `safetensors` library (design section 12).

    Everything comes back as bf16 or fp32 -- never as a quantised dtype -- because
    the CPU has no fp8 or fp4 arithmetic and the dequantisation is lossless anyway.
    """

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
        self._lock = threading.Lock()
        self._expert_index = {L["layer"]: L for L in self.m["experts"]}
        self._engram_index = {E["layer"]: E for E in self.m["engram"]}

    # -- raw io ------------------------------------------------------------

    def _file(self, index: int):
        h = self._handles.get(index)
        if h is None:
            h = open(os.path.join(self.dir, self.m["files"][index]["path"]), "rb",
                     buffering=0)
            self._handles[index] = h
        return h

    def close(self):
        for h in self._handles.values():
            h.close()
        self._handles.clear()

    def _read(self, file_index: int, offset: int, nbytes: int) -> bytes:
        with self._lock:
            f = self._file(file_index)
            f.seek(offset)
            buf = f.read(nbytes)
        if len(buf) != nbytes:
            raise SystemExit(f"short read of {nbytes} B at {offset} in file {file_index}")
        return buf

    def read_region(self, file_index: int, offset: int, nbytes: int) -> bytes:
        """A sector-aligned read plus skew, the way storage::IoEngine does it."""
        a_off = (offset // self.align) * self.align
        a_end = -(-(offset + nbytes) // self.align) * self.align
        file_bytes = self.m["files"][file_index]["bytes"]
        buf = self._read(file_index, a_off, min(a_end, file_bytes) - a_off)
        skew = offset - a_off
        return buf[skew:skew + nbytes]

    # -- named tensors -----------------------------------------------------

    def has(self, name: str) -> bool:
        return name in self.m["tensors"]

    def tensor(self, name: str, *, want: str = "auto") -> torch.Tensor:
        """Decode one named tensor.

        `want` picks the output dtype: "auto" gives bf16 for bf16/fp8 sources and
        fp32 for f32 sources, which is what the corresponding `nn.Parameter` in
        model.py holds. "f32" forces fp32 (the Compressor's promoted weights).
        """
        t = self.m["tensors"].get(name)
        if t is None:
            raise KeyError(name)
        raw = self.read_region(t["file"], t["offset"], t["bytes"])
        dtype, shape = t["dtype"], t["shape"]
        if dtype == "bf16":
            out = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(shape)
        elif dtype == "f32":
            out = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(shape)
        elif dtype == "fp8_e4m3":
            sc = t["scale"]
            sraw = self.read_region(sc["file"], sc["offset"], sc["bytes"])
            v = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
            v = v.view(torch.float8_e4m3fn).reshape(shape).float()
            s = torch.frombuffer(bytearray(sraw), dtype=torch.uint8)
            s = s.view(torch.float8_e8m0fnu).reshape(sc["shape"]).float()
            bo, bi = sc["block"]
            s = s.repeat_interleave(bo, 0).repeat_interleave(bi, 1)
            out = (v * s[:shape[0], :shape[1]]).to(torch.bfloat16)
        else:
            raise SystemExit(f"{name}: unhandled manifest dtype {dtype!r}")
        if want == "f32":
            out = out.float()
        return out.contiguous()

    # -- routed experts ----------------------------------------------------

    def expert_slot(self, layer: int, expert: int):
        """One expert's slot, filled exactly as store::ExpertStore fills one.

        Returns (slot bytearray, {part -> uint8 view}). The views alias the slot, so
        an 18.8 MB expert is copied once on the way out of the file and never again.
        """
        table = self._expert_index.get(layer)
        if table is None:
            raise SystemExit(f"manifest has no expert layer {layer}")
        slot = bytearray()
        spans: list[tuple[str, int, int]] = []
        for run in table["experts"][expert]:
            assert len(slot) == run["slot_offset"], "runs are not laid back to back"
            end = min(run["aligned_off"] + run["aligned_bytes"],
                      self.m["files"][run["file"]]["bytes"])
            slot += self._read(run["file"], run["aligned_off"], end - run["aligned_off"])
            for part in run["parts"]:
                spans.append((part["tensor"], part["slot_offset"], part["bytes"]))
        base = torch.frombuffer(slot, dtype=torch.uint8)
        return slot, {name: base[off:off + n] for name, off, n in spans}

    def expert_bytes(self, layer: int, expert: int) -> dict[str, bytes]:
        """The same six parts as plain `bytes`, for tests and for comparing against
        the `safetensors` library."""
        _slot, views = self.expert_slot(layer, expert)
        return {k: v.numpy().tobytes() for k, v in views.items()}

    @staticmethod
    def expert_weights_from_slot(views: dict, dim: int, inter: int):
        w1 = dequant_fp4_bf16(views["w1.weight"], views["w1.scale"], inter, dim)
        w3 = dequant_fp4_bf16(views["w3.weight"], views["w3.scale"], inter, dim)
        w2 = dequant_fp4_bf16(views["w2.weight"], views["w2.scale"], dim, inter)
        return w1, w2, w3

    def expert_weights(self, layer: int, expert: int, dim: int, inter: int):
        """-> (w1, w2, w3) bf16, ready for `F.linear`."""
        _slot, views = self.expert_slot(layer, expert)
        return self.expert_weights_from_slot(views, dim, inter)

    def expert_stream(self, layer: int, experts, dim: int, inter: int, depth: int = 2):
        """Yield (expert_id, w1, w2, w3) with the next expert's bytes read while the
        current one is being used.

        A slot is 18.8 MB and reads at ~1.2 GB/s through Python's file object (~16 ms)
        while dequantising it costs ~35 ms, so one reader thread hides the I/O
        completely. This is the offline shadow of design section 9.7's expert-major
        prefill stream, and of section 9.6's P1 queue.
        """
        import queue
        q: "queue.Queue" = queue.Queue(maxsize=depth)
        experts = list(experts)
        stop = threading.Event()

        def reader():
            try:
                for e in experts:
                    if stop.is_set():
                        break
                    q.put((e, self.expert_slot(layer, e)))
            except BaseException as exc:            # noqa: BLE001
                q.put(("error", exc))
            q.put(None)

        th = threading.Thread(target=reader, daemon=True)
        th.start()
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                e, payload = item
                if e == "error":
                    raise payload
                slot, views = payload
                yield (e, *self.expert_weights_from_slot(views, dim, inter))
                del slot, views
        finally:
            stop.set()
            try:
                while not q.empty():
                    q.get_nowait()
            except Exception:                       # noqa: BLE001
                pass
            th.join(timeout=5.0)

    # -- engram rows -------------------------------------------------------

    def engram_rows(self, layer: int, row_ids: np.ndarray, threads: int = 32) -> torch.Tensor:
        """Fetch `row_ids` from the layer's n-gram table -> bf16 [n, head_dim].

        The table is ~98 GB per layer, so this is the only sane way to run Engram on
        a machine with 63 GB of RAM: 24 rows per token per layer, 256 B of value and
        8 B of scale each. Duplicate ids are read once. Issued from a thread pool
        because a single-threaded 4 KiB random read stream gets nowhere near the
        83,700 IOPS section 9.2.1 measured at QD 48.
        """
        spec = self._engram_index.get(layer)
        if spec is None:
            raise SystemExit(f"manifest has no engram layer {layer}")
        flat = np.asarray(row_ids, dtype=np.int64).reshape(-1)
        uniq, inverse = np.unique(flat, return_inverse=True)
        vb, sb = spec["value"]["row_bytes"], spec["scale"]["row_bytes"]
        vbase, sbase = spec["value"]["offset"], spec["scale"]["offset"]
        vfile, sfile = spec["value"]["file"], spec["scale"]["file"]
        vals = bytearray(len(uniq) * vb)
        scs = bytearray(len(uniq) * sb)

        def fetch(i: int):
            r = int(uniq[i])
            vals[i * vb:(i + 1) * vb] = self._read(vfile, vbase + r * vb, vb)
            scs[i * sb:(i + 1) * sb] = self._read(sfile, sbase + r * sb, sb)

        if threads > 1 and len(uniq) > threads:
            with ThreadPoolExecutor(max_workers=threads) as pool:
                list(pool.map(fetch, range(len(uniq))))
        else:
            for i in range(len(uniq)):
                fetch(i)
        rows = _fp8_rows_to_bf16(bytes(vals), bytes(scs), len(uniq), vb, sb)
        return rows[torch.from_numpy(inverse)].reshape(*np.shape(row_ids), vb)


# ---------------------------------------------------------------------------
# 5. streaming MoE / Block
# ---------------------------------------------------------------------------

class StreamingMoE(torch.nn.Module):
    """`model.MoE` with the 384 routed experts taken out.

    The reference builds every `Expert` up front, which on the real shapes is 6.8 GB
    of parameters per layer and makes layer streaming impossible. This keeps the
    reference's `Gate` and the shared expert verbatim, records what the gate decided,
    and leaves the routed half to the driver, which runs it expert-major (design
    section 9.7's ordering, for the same reason: one pass over the layer's weights).
    """

    def __init__(self, layer_id: int, args, ref):
        super().__init__()
        n_routed, n_act = args.get_moe_config(layer_id)
        self.layer_id = layer_id
        self.dim = args.dim
        self.inter_dim = args.moe_inter_dim
        self.n_routed_experts = n_routed
        self.n_activated_experts = n_act
        self.swiglu_limit = args.swiglu_limit
        self.gate = ref.Gate(layer_id, args)
        self.shared_experts = ref.Expert(args.dim, args.moe_inter_dim,
                                         swiglu_limit=args.swiglu_limit)

    def route(self, x: torch.Tensor, image_mask=None):
        """-> (weights [n, topk] fp32, indices [n, topk] int64), the reference's own
        `Gate.forward`: sqrt(softplus(x.W)), select on score+bias, weight on the
        unbiased score, normalise with a 1e-20 floor, scale by route_scale."""
        return self.gate(x, image_mask)

    def gate_scores(self, x: torch.Tensor) -> torch.Tensor:
        """The raw, unbiased scores for all experts -- what design section 9.3's
        score-aware eviction and section 9.4's lookahead both need."""
        g = self.gate
        scores = F.linear(x.float(), g.weight.float()) / g.gate_temp
        assert g.score_func == "sqrtsoftplus", g.score_func
        return F.softplus(scores).sqrt()


class StreamingBlock(torch.nn.Module):
    """`model.Block` split at the FFN so a layer's experts can be streamed.

    Nothing here changes the math. `Block.forward` is:

        residual = x
        attn_pre, attn_post, attn_comb = hc_mixes(x, hc_attn_*)
        x = attn(attn_norm(hc_pre(x, pre_mix)), start_pos)
        x = hc_post(x, residual, attn_post, attn_comb)
        residual = x
        ffn_pre, ffn_post, ffn_comb = hc_mixes(x, hc_ffn_*)
        x = ffn(ffn_norm(hc_pre(x, attn_pre)))
        x = hc_post(x, residual, ffn_post, ffn_comb)
        return x, ffn_pre

    `forward_attn` runs the first half and stops with the FFN's input in hand;
    `finish_ffn` applies `hc_post` once the driver has accumulated the routed and
    shared expert outputs. Note the coefficient handoff design section 2.4 warns
    about: attention consumes the *previous* layer's `ffn_pre`, and this layer's FFN
    consumes the `attn_pre` its own attention produced.
    """

    def __init__(self, block):
        super().__init__()
        self.b = block

    def forward_attn(self, x, start_pos, pre_mix):
        b = self.b
        residual = x
        attn_pre, attn_post, attn_comb = b.hc_mixes(x, b.hc_attn_fn, b.hc_attn_scale,
                                                    b.hc_attn_base)
        y = b.hc_pre(x, pre_mix)
        y = b.attn_norm(y)
        y = b.attn(y, start_pos)
        x = b.hc_post(y, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = b.hc_mixes(x, b.hc_ffn_fn, b.hc_ffn_scale,
                                                 b.hc_ffn_base)
        ffn_in = b.ffn_norm(b.hc_pre(x, attn_pre))
        return ffn_in, residual, ffn_pre, ffn_post, ffn_comb

    def finish_ffn(self, ffn_out, residual, ffn_post, ffn_comb):
        return self.b.hc_post(ffn_out.to(residual.dtype), residual, ffn_post, ffn_comb)


def expert_ffn(x: torch.Tensor, w1, w2, w3, weights: torch.Tensor | None,
               limit: float, input_is_quantised: bool = False) -> torch.Tensor:
    """`model.Expert.forward` for a dequantised expert.

        gate = clamp(w1 x, max=limit); up = clamp(w3 x, +-limit)
        y    = w2( (silu(gate) * up) * routing_weight )

    `x` is bf16 and gets the same fp8 activation round trip the fp4 kernel path
    applies; the elementwise part runs in fp32 and the result is cast back to bf16
    before `w2`, exactly as the reference does (`self.w2(x.to(dtype))`).

    `input_is_quantised` skips the first round trip for a caller that already did it.
    `act_quant` is per (row, 32-element group) and has nothing to do with the weight,
    so quantising a layer's FFN input once instead of once per routed expert is not
    an approximation -- it produces the identical bytes six times less often.
    """
    xq = x if input_is_quantised else act_quant_dequant(x, FP8_BLOCK)
    gate = F.linear(xq, w1).float()
    up = F.linear(xq, w3).float()
    if limit > 0:
        up = torch.clamp(up, min=-limit, max=limit)
        gate = torch.clamp(gate, max=limit)
    h = F.silu(gate) * up
    if weights is not None:
        h = weights * h
    h = act_quant_dequant(h.to(x.dtype), FP8_BLOCK)
    return F.linear(h, w2)


# ---------------------------------------------------------------------------
# 6. model args and weight binding
# ---------------------------------------------------------------------------

def build_args(ref, inference_dir: str, max_seq_len: int, max_batch_size: int = 1,
               temperature: float = 0.0):
    """`ModelArgs` straight from `inference/config.json`, with the vision path off.

    Vision is design section 1.2's explicit non-goal, and switching it off also
    removes `bias_vl` from the gate -- which is correct for a text-only trace, since
    `Gate.forward` only consults `bias_vl` when `image_mask` is not None anyway.
    """
    with open(os.path.join(inference_dir, "config.json"), "rb") as f:
        cfg = json.load(f)
    cfg = {k: v for k, v in cfg.items() if not k.startswith("vision_")}
    cfg["vision_n_layers"] = 0
    for key in ("compress_ratios", "kv_source_layers", "index_source_layers",
                "engram_layer_ids", "engram_num_embeddings", "dspark_target_layer_ids"):
        if key in cfg:
            cfg[key] = tuple(cfg[key])
    cfg.pop("image_token_id", None)
    known = {f.name for f in ref.ModelArgs.__dataclass_fields__.values()}
    unknown = set(cfg) - known
    if unknown:
        raise SystemExit(f"inference/config.json has keys ModelArgs does not: {sorted(unknown)}")
    return ref.ModelArgs(max_batch_size=max_batch_size, max_seq_len=max_seq_len,
                         temperature=temperature, **cfg)


# Weights whose checkpoint dtype is fp8 but whose model.py `Linear` is declared
# bf16, so they must NOT get the activation round trip. convert.py dequantises
# wo_a to bf16 for exactly this reason (model.py: "convert.py dequantizes it to
# bf16; an fp8 grouped GEMM would halve the memory").
_NO_ACT_QUANT_SUFFIXES = ("attn.wo_a.weight",)

# Modules model.py promotes to fp32 (Compressor with compress_ratio > 1 does its
# softmax pooling in fp32, so both its projections are fp32 parameters).
_F32_SUFFIXES = ("compressor.wkv.weight", "compressor.wgate.weight")


def bind_layer_weights(block, store: WeightStore, prefix: str, verbose: bool = False) -> int:
    """Copy every parameter of one Block out of the shards.

    Two wrinkles from model.py's `Linear`:

      * a quantised `Linear` carries a separate `scale` parameter, which the
        manifest folds into the weight entry instead of exposing by name. We
        dequantise the weight and leave the now-meaningless `scale` alone -- `_linear`
        never reads it.
      * the resulting weight is bf16 where the module declared fp8, so the parameter
        changes dtype. That is the point: there is no fp8 arithmetic on this CPU.

    Raises if the module wants a parameter the checkpoint does not have. A silently
    uninitialised weight would poison the whole trace and the layer-streaming
    structure makes that easy to miss.
    """
    bound = 0
    for name, param in list(block.named_parameters(recurse=True)):
        full = f"{prefix}.{name}"
        if name.endswith(".scale"):
            base = full[: -len(".scale")] + ".weight"
            if store.has(base) and store.m["tensors"][base]["dtype"] == "fp8_e4m3":
                continue  # folded into the dequantised weight
        if not store.has(full):
            raise SystemExit(f"{full} is not in the manifest (module expects it)")
        want = "f32" if full.endswith(_F32_SUFFIXES) else "auto"
        t = store.tensor(full, want=want)
        if tuple(t.shape) != tuple(param.shape):
            raise SystemExit(f"{full}: manifest {list(t.shape)} vs module {list(param.shape)}")
        if param.dtype in (torch.float32, torch.bfloat16):
            t = t.to(param.dtype)
        param.data = t
        src = store.m["tensors"][full]
        if src["dtype"] == "fp8_e4m3" and not full.endswith(_NO_ACT_QUANT_SUFFIXES):
            param.act_block = FP8_BLOCK
        bound += 1
        if verbose:
            print(f"    {full:52s} {list(t.shape)} {t.dtype}"
                  f"{'  +act_quant' if hasattr(param, 'act_block') else ''}")
    return bound


def make_block(ref, args, layer_id: int, engram_layout, store: WeightStore,
               engram_reader=None):
    """Build one Block with `StreamingMoE` in place of `MoE`, and bind its weights."""
    orig_moe = ref.MoE
    ref.MoE = lambda lid, a, _ref=ref: StreamingMoE(lid, a, _ref)
    try:
        with ref.set_dtype(torch.bfloat16):
            block = ref.Block(layer_id, args, engram_layout)
    finally:
        ref.MoE = orig_moe
    prefix = f"layers.{layer_id}" if layer_id < args.n_layers else f"mtp.{layer_id - args.n_layers}"
    if block.engram is not None:
        # The 98 GB row table is not a parameter we can materialise; swap the
        # embedding for the manifest-backed reader before binding.
        idx = engram_layout.layer_ids.index(layer_id)
        block.engram.embed = EngramRowEmbedding(store, layer_id,
                                                engram_layout.num_embeddings[idx])
    bind_layer_weights(block, store, prefix)
    return StreamingBlock(block)


class EngramRowEmbedding(torch.nn.Module):
    """Stands in for `ParallelEngramEmbedding`, reading rows from NVMe on demand.

    `ParallelEngramEmbedding.forward` dequantises with a per-32 UE8M0 scale and
    returns bf16; `WeightStore.engram_rows` does the same arithmetic on the rows it
    actually fetched. The returned `row_ids` are also what design section 9.5's
    EngramPrefetcher would issue, so `tools/route_trace.py` writes them out.
    """

    def __init__(self, store: WeightStore, layer_id: int, num_embeddings: int):
        super().__init__()
        self.store = store
        self.layer_id = layer_id
        self.num_embeddings = num_embeddings
        self.last_rows: np.ndarray | None = None

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        ids = indices.detach().cpu().numpy().astype(np.int64)
        if ids.max(initial=0) >= self.num_embeddings or ids.min(initial=0) < 0:
            raise SystemExit(f"engram row id out of range for layer {self.layer_id}")
        self.last_rows = ids
        return self.store.engram_rows(self.layer_id, ids)


# ---------------------------------------------------------------------------
# 7. tokenizer adapter
# ---------------------------------------------------------------------------

class TokenizerAdapter:
    """The slice of the HuggingFace tokenizer API that `inference/engram.py` uses.

    `build_compressed_token_map` wants `len(tokenizer)` and `.backend_tokenizer`;
    everything it does with the backend is `decode` and `id_to_token`. Wrapping the
    raw `tokenizers.Tokenizer` avoids a `transformers` dependency, and the resulting
    map is checked against `engram_compressed_vocab_size` by `NgramHashState`, which
    is a real test that the wrapper behaves like the real thing.
    """

    def __init__(self, tokenizer_json: str):
        from tokenizers import Tokenizer
        self.backend_tokenizer = Tokenizer.from_file(tokenizer_json)
        self._n = self.backend_tokenizer.get_vocab_size(with_added_tokens=True)

    def __len__(self) -> int:
        return self._n

    def encode(self, text: str) -> list[int]:
        return self.backend_tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids) -> str:
        return self.backend_tokenizer.decode(list(ids), skip_special_tokens=False)


@dataclass
class CachedTokenMap:
    """`build_compressed_token_map` decodes all 129,280 tokens one at a time, which
    costs ~20 s. Cache it: the map is a pure function of tokenizer.json."""
    lookup: list[int]
    size: int

    @staticmethod
    def build(tokenizer, cache_path: str | None) -> "CachedTokenMap":
        if cache_path and os.path.exists(cache_path):
            d = np.load(cache_path)
            return CachedTokenMap(d["lookup"].tolist(), int(d["size"]))
        from engram import build_compressed_token_map
        lookup, size = build_compressed_token_map(tokenizer)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            np.savez(cache_path, lookup=np.array(lookup, dtype=np.int64), size=size)
        return CachedTokenMap(lookup, size)
