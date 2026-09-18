#!/usr/bin/env python3
"""2-bit routed-expert quantisation: schemes, packing, metrics (Track F5).

Design reference: section 2.2 (the 288.8 GB of routed experts are the NVMe-bound
part), section 3.1 (the hit-rate curve and the stall model), section 6 (the
standing rule is "use the checkpoint's own precision, do not re-quantise" -- this
module exists to measure what breaking that rule would cost), section 12 (the
oracle ladder these numbers have to be read against).

Ground truth
------------
The FP4 E2M1 + UE8M0/32 weights *are* the model. Everything here dequantises one
expert exactly (`tools/oracle.py:dequant_fp4`, `tools/dsref.py:dequant_fp4_bf16`
-- both lossless into fp32/bf16) and then re-quantises that fp32 array to two
bits. So "error" always means "error against the checkpoint", never against some
hypothetical fp16 original that nobody has.

Layout vocabulary
-----------------
Every expert matrix is stored `[rows, K]` with the quantisation block running
along K, 32 elements per UE8M0 scale byte:

    w1 [2304, 5120]   gate    K = dim  = 5120
    w3 [2304, 5120]   up      K = dim  = 5120
    w2 [5120, 2304]   down    K = inter = 2304

K is the reduction axis in all three, which is why the block runs along it: one
scale covers 32 values that are summed together, so the scale factors out of the
dot product. Every scheme below keeps that property -- a 2-bit kernel is then the
FP4 kernel with a different unpack step and a 4-entry LUT.

Scheme families
---------------
(a) `int2`      fixed symmetric codebook {-1.5, -0.5, +0.5, +1.5} x block scale.
(b) `lloyd`     4-level Lloyd-Max codebook fitted per row (or per tensor) on the
                block-normalised values; the block scale stays UE8M0.
(c) `mixed`     the top x% of rows stay FP4 E2M1, the rest go to (a) or (b).
(d) `actw`      (a)/(b) with the fit minimising E||(W - Wq) x||^2 rather than
                ||W - Wq||_F^2, using a per-input-channel second moment from the
                routing trace's real expert inputs.

Scales are UE8M0 (a power of two, one byte, exactly what the FP4 path already
stores and what the GPU unpack already knows how to apply) or FP8 E4M3 for the
"is a power of two costing us anything?" control.
"""

from __future__ import annotations

import numpy as np

FP4_BLOCK = 32
DIM = 5120
INTER = 2304

# The 8 FP4 E2M1 magnitudes, so a "keep this row in FP4" path can re-quantise a
# row exactly (it is already an FP4 value, so this is a no-op -- it is here so the
# mixed scheme's reconstruction is written the same way as the others).
FP4_MAGS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)

# The fixed symmetric 2-bit codebook. code -> (2*code - 3) / 2.
INT2_LEVELS = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)


# The 15 distinct E2M1 values, *unnormalised*. Used as a codebook they make the
# fit the identity: the checkpoint's own UE8M0 scale is one of the power-of-two
# candidates `fit_block_scale` tries, so `fp4_exact` must come back at zero error.
FP4_LEVELS = np.concatenate([-FP4_MAGS[:0:-1], FP4_MAGS])


def uniform_levels(n: int) -> np.ndarray:
    """The symmetric uniform codebook with `n` levels, normalised so the outermost
    level is 1.0 (which is what `fit_block_scale`'s `amax / max(levels)` assumes).

    n = 4 reproduces INT2_LEVELS up to that normalisation; n = 8 / 16 are the
    3-bit / 4-bit controls the report needs to say *where* the cliff is, not just
    that 2 bits is past it.
    """
    lv = np.arange(n, dtype=np.float32) - (n - 1) / 2.0
    return lv / np.abs(lv).max()


def code_entropy_bits(codes: np.ndarray, n_levels: int) -> float:
    """Shannon entropy of the code stream, bits per weight.

    This is the honest floor for any *lossless* recoding of a quantised tensor:
    if the checkpoint's own FP4 codes carry more than 2 bits of entropy, then no
    2-bit format can be a recoding of them -- it has to throw information away.
    """
    h = np.bincount(np.asarray(codes).ravel(), minlength=n_levels).astype(np.float64)
    p = h / max(h.sum(), 1.0)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


# --------------------------------------------------------------------------- #
# scale formats
# --------------------------------------------------------------------------- #

def ue8m0_round_down(x: np.ndarray) -> np.ndarray:
    """Largest power of two <= x, as UE8M0 would store it. x > 0."""
    e = np.floor(np.log2(np.maximum(x, 1e-38)))
    return np.ldexp(np.ones_like(x), e.astype(np.int32))


def ue8m0_code(s: np.ndarray) -> np.ndarray:
    """s (an exact power of two) -> the UE8M0 byte, value = 2^(code-127)."""
    e = np.rint(np.log2(np.maximum(s, 2.0 ** -127))).astype(np.int32)
    return np.clip(e + 127, 0, 254).astype(np.uint8)


def ue8m0_value(code: np.ndarray) -> np.ndarray:
    return np.ldexp(np.ones(code.shape, dtype=np.float32),
                    code.astype(np.int32) - 127)


_E4M3_TABLE: np.ndarray | None = None


def e4m3_table() -> np.ndarray:
    """The 256 FP8 E4M3 values, straight from torch (same source as
    tools/oracle.py:fp8_e4m3_table, so L0's table is the anchor)."""
    global _E4M3_TABLE
    if _E4M3_TABLE is None:
        import torch
        b = torch.arange(256, dtype=torch.uint8)
        _E4M3_TABLE = b.view(torch.float8_e4m3fn).float().numpy().astype(np.float32)
    return _E4M3_TABLE


def e4m3_round(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Round positive x to the nearest representable E4M3 value. -> (value, code)."""
    tab = e4m3_table()
    pos = np.where(np.isfinite(tab) & (tab > 0))[0]
    vals = tab[pos]
    order = np.argsort(vals)
    vals, pos = vals[order], pos[order]
    idx = np.searchsorted(vals, x.ravel())
    idx = np.clip(idx, 1, len(vals) - 1)
    lo, hi = vals[idx - 1], vals[idx]
    take_hi = (hi - x.ravel()) < (x.ravel() - lo)
    chosen = np.where(take_hi, idx, idx - 1)
    return vals[chosen].reshape(x.shape), pos[chosen].reshape(x.shape).astype(np.uint8)


# --------------------------------------------------------------------------- #
# the core fit: block scale + 4 levels
# --------------------------------------------------------------------------- #

def _blocks(w: np.ndarray, block: int) -> np.ndarray:
    r, k = w.shape
    assert k % block == 0, f"K={k} is not a multiple of block {block}"
    return w.reshape(r, k // block, block)


def _assign(vb: np.ndarray, s: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Nearest-level assignment of vb/s. vb [r, nb, B], s [r, nb, 1]."""
    u = vb / s
    # levels are sorted, so len(levels)-1 midpoint comparisons give the code
    # without ever materialising an [..., n_levels] distance tensor.
    mid = (levels[:-1] + levels[1:]) / 2.0
    c = np.zeros(u.shape, dtype=np.uint8)
    for j in range(len(mid)):
        c += (u >= mid[j])
    return c


def fit_block_scale(w: np.ndarray, block: int, levels: np.ndarray,
                    scale_fmt: str = "ue8m0",
                    colw: np.ndarray | None = None,
                    lo: float = -3.0, hi: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Per-block scale minimising the (optionally column-weighted) SSE.

    Returns (codes uint8 [r, nb, B], scale float32 [r, nb, 1]).

    The scale candidates are the ones a UE8M0 byte can hold: `amax / max(levels)`
    rounded down to a power of two, times 2^k for k in [lo, hi]. The optimum is
    *below* amax for a coarse codebook -- clipping the two or three extreme values
    in a block buys more than it costs -- which is why the range runs down to
    amax/8 as well as up. It is an exhaustive search over that range, so there is
    no "it converged to a local minimum" caveat to write down.
    `scale_fmt="e4m3"` searches the same range on a 4x finer grid and rounds the
    winner to E4M3, which is the control for "is the power-of-two scale costing
    us anything?".

    `colw` (shape [K]) is the per-input-channel weight for the activation-aware
    objective: E[x_c^2]. The reconstruction is unchanged; only which code and
    which scale get picked changes.
    """
    vb = _blocks(w.astype(np.float32), block)
    amax = np.abs(vb).max(axis=2, keepdims=True)
    base = ue8m0_round_down(np.maximum(amax / float(levels.max()), 2.0 ** -126))
    wb = None if colw is None else _blocks(np.broadcast_to(
        colw.astype(np.float32), w.shape), block)

    steps = np.arange(lo, hi + 1.0, dtype=np.float32)
    if scale_fmt == "e4m3":
        steps = np.arange(lo, hi + 0.25, 0.25, dtype=np.float32)

    best_sse = None
    best_s = None
    best_c = None
    for st in steps:
        s = base * np.float32(2.0 ** st)
        if scale_fmt == "e4m3":
            s, _ = e4m3_round(s)
        c = _assign(vb, s, levels)
        rec = levels[c] * s
        d = (rec - vb)
        d *= d
        if wb is not None:
            d *= wb
        sse = d.sum(axis=2, keepdims=True)
        if best_sse is None:
            best_sse, best_s, best_c = sse, s, c
        else:
            take = sse < best_sse
            best_sse = np.where(take, sse, best_sse)
            best_s = np.where(take, s, best_s)
            best_c = np.where(take, c, best_c)
    return best_c, best_s.astype(np.float32)


def lloyd_levels(w: np.ndarray, block: int, scale: np.ndarray,
                 per_row: bool, iters: int = 12,
                 colw: np.ndarray | None = None,
                 init: np.ndarray | None = None) -> np.ndarray:
    """4-level Lloyd-Max codebook on the block-normalised values u = w / scale.

    Returns [r, 4] (per_row) or [1, 4] (per tensor), sorted ascending. The block
    scale is held fixed, so the codebook is a pure LUT in the kernel: one 4-entry
    fp16 table per row (8 bytes / 5120 weights = 0.0125 bit/weight) or per tensor.
    """
    vb = _blocks(w.astype(np.float32), block)
    u = (vb / scale).reshape(w.shape[0], -1)
    if colw is not None:
        cw = np.broadcast_to(colw.astype(np.float32), w.shape)
    else:
        cw = None
    if not per_row:
        u = u.reshape(1, -1)
        cw = None if cw is None else cw.reshape(1, -1)
    base = INT2_LEVELS if init is None else init
    nlev = len(base)
    lev = np.repeat(np.asarray(base, np.float32)[None, :], u.shape[0], axis=0)
    for _ in range(iters):
        mid = (lev[:, :-1] + lev[:, 1:]) / 2.0
        c = np.zeros(u.shape, dtype=np.uint8)
        for j in range(nlev - 1):
            c += (u >= mid[:, j:j + 1])
        new = lev.copy()
        for j in range(nlev):
            m = (c == j)
            if cw is None:
                n = m.sum(axis=1)
                s = np.where(m, u, 0.0).sum(axis=1)
            else:
                n = np.where(m, cw, 0.0).sum(axis=1)
                s = np.where(m, u * cw, 0.0).sum(axis=1)
            new[:, j] = np.where(n > 0, s / np.maximum(n, 1e-12), lev[:, j])
        new = np.sort(new, axis=1)
        if np.allclose(new, lev, atol=1e-6):
            lev = new
            break
        lev = new
    return lev.astype(np.float32)


def assign_with_levels(w: np.ndarray, block: int, scale: np.ndarray,
                       levels: np.ndarray) -> np.ndarray:
    """Nearest-level codes for a per-row (or per-tensor) codebook."""
    vb = _blocks(w.astype(np.float32), block)
    u = (vb / scale).reshape(w.shape[0], -1)
    lv = (levels if levels.shape[0] == w.shape[0]
          else np.broadcast_to(levels, (w.shape[0], levels.shape[1])))
    mid = (lv[:, :-1] + lv[:, 1:]) / 2.0
    c = np.zeros(u.shape, dtype=np.uint8)
    for j in range(mid.shape[1]):
        c += (u >= mid[:, j:j + 1])
    return c.reshape(vb.shape)


def reconstruct(codes: np.ndarray, scale: np.ndarray, levels: np.ndarray,
                shape: tuple[int, int]) -> np.ndarray:
    """codes [r, nb, B] + scale [r, nb, 1] + levels [r|1, L] -> W_hat [r, K]."""
    r = shape[0]
    lv = (levels if levels.shape[0] == r
          else np.broadcast_to(levels, (r, levels.shape[1])))
    v = np.take_along_axis(lv[:, None, None, :],
                           codes[..., None].astype(np.int64), axis=3)[..., 0]
    return (v * scale).reshape(shape).astype(np.float32)


# --------------------------------------------------------------------------- #
# the schemes
# --------------------------------------------------------------------------- #

class Quantised:
    """One quantised tensor: what gets stored, what it dequantises to, and how
    many bits per weight that really costs."""

    def __init__(self, name: str, shape, codes, scale_code, levels,
                 fp4_rows: np.ndarray | None, w_hat: np.ndarray,
                 block: int, scale_fmt: str, per_row_lut: bool):
        self.name = name
        self.shape = shape
        self.codes = codes                  # uint8 [r, nb, B] 2-bit codes
        self.scale_code = scale_code        # uint8 [r, nb]
        self.levels = levels                # f32 [r|1, 4]
        self.fp4_rows = fp4_rows            # int32 row ids kept in FP4, or None
        self.w_hat = w_hat                  # f32 [r, K] reconstruction
        self.block = block
        self.scale_fmt = scale_fmt
        self.per_row_lut = per_row_lut

    def bits_per_weight(self) -> float:
        r, k = self.shape
        n = r * k
        n_fp4 = 0 if self.fp4_rows is None else len(self.fp4_rows) * k
        n_2b = n - n_fp4
        pay = getattr(self, "payload_bits", 2.0)
        bits = pay * n_2b + 4.0 * n_fp4           # payload
        bits += 8.0 * (n / self.block)            # one scale byte per block, both paths
        nl = getattr(self, "n_levels", 4)
        if self.per_row_lut:
            bits += 16.0 * nl * r                 # n_levels x fp16 per row
        else:
            bits += 16.0 * nl
        if self.fp4_rows is not None:
            bits += float(r)                      # a one-bit-per-row "is FP4" bitmap
        return bits / n


def quantise_tensor(w: np.ndarray, scheme: str, block: int = 32,
                    scale_fmt: str = "ue8m0", per_row_lut: bool = True,
                    fp4_frac: float = 0.0, colw: np.ndarray | None = None,
                    n_levels: int = 4, name: str = "",
                    fit_cache: dict | None = None) -> Quantised:
    """`w` is the exact FP4 dequantisation, fp32 [rows, K].

    scheme:
      "int2"   fixed {-1.5,-0.5,0.5,1.5}
      "lloyd"  4-level Lloyd-Max, per row if per_row_lut else per tensor
      plus `fp4_frac > 0` for the mixed variant and `colw` for activation-aware.
    """
    w = np.ascontiguousarray(w, dtype=np.float32)
    r, k = w.shape
    base_levels = FP4_LEVELS if n_levels == 15 else uniform_levels(n_levels)

    def fit():
        """The uniform-codebook fit, memoised per (tensor, block, scale format,
        objective, level count) so the dozen scheme variants in the grid share it
        instead of re-running the scale search a dozen times."""
        key = (name, block, scale_fmt, colw is None, n_levels)
        if fit_cache is not None and key in fit_cache:
            return fit_cache[key]
        v = fit_block_scale(w, block, base_levels, scale_fmt, colw)
        if fit_cache is not None:
            fit_cache[key] = v
        return v

    fp4_rows = None
    if fp4_frac > 0.0:
        # Saliency: the rows a 2-bit fit hurts most, measured by running the cheap
        # fixed-codebook fit once and ranking rows by their (optionally
        # activation-weighted) residual energy relative to the row's own energy.
        c0, s0 = fit()
        rec0 = reconstruct(c0, s0, base_levels[None, :], (r, k))
        d = (rec0 - w) ** 2
        if colw is not None:
            d = d * colw[None, :]
        n_keep = max(1, int(round(fp4_frac * r)))
        fp4_rows = np.sort(np.argsort(d.sum(axis=1))[::-1][:n_keep]).astype(np.int32)

    codes, scale = fit()
    if scheme == "lloyd":
        levels = lloyd_levels(w, block, scale, per_row_lut, colw=colw,
                              init=base_levels)
        codes = assign_with_levels(w, block, scale, levels)
        # one more scale pass against the fitted levels would couple the two fits;
        # measured gain is under 1e-3 relative and it costs a second sweep, so the
        # scale stays the one the uniform fit chose (recorded here, not hidden).
    elif scheme == "int2":
        levels = base_levels[None, :].copy()
    else:
        raise SystemExit(f"unknown scheme {scheme!r}")

    w_hat = reconstruct(codes, scale, levels, (r, k))
    if fp4_rows is not None:
        w_hat[fp4_rows] = w[fp4_rows]        # already FP4 values: bit-exact

    if scale_fmt == "ue8m0":
        scode = ue8m0_code(scale[..., 0])
    else:
        _v, scode = e4m3_round(scale[..., 0])
    q = Quantised(name, (r, k), codes, scode, levels, fp4_rows, w_hat,
                  block, scale_fmt, per_row_lut and scheme == "lloyd")
    q.n_levels = n_levels
    q.payload_bits = float(np.log2(n_levels))
    q.entropy_bits = code_entropy_bits(codes, n_levels)
    return q


# --------------------------------------------------------------------------- #
# packing (the on-disk format the doc proposes)
# --------------------------------------------------------------------------- #

def pack_codes(codes: np.ndarray) -> np.ndarray:  # noqa: D401
    """[r, nb, B] 2-bit codes -> [r, K/4] uint8, element j in bits 2*(j%4).

    Element 0 in the *lowest* bits, matching the FP4 path's "low nibble = even
    element along K" convention (design section 12); a kernel that already knows
    how to walk a K-contiguous FP4 row walks this one the same way.
    """
    assert codes.max(initial=0) < 4, (
        "pack_codes is the 2-bit format; the 3-bit and 4-bit rows in the scheme "
        "table are error controls, not proposed formats, and are measured "
        "through their reconstruction only")
    r = codes.shape[0]
    flat = codes.reshape(r, -1)
    assert flat.shape[1] % 4 == 0
    q = flat.reshape(r, -1, 4).astype(np.uint8)
    return (q[..., 0] | (q[..., 1] << 2) | (q[..., 2] << 4) | (q[..., 3] << 6))


def unpack_codes(packed: np.ndarray, k: int) -> np.ndarray:
    """The dequant reference's first step; inverse of pack_codes."""
    r = packed.shape[0]
    out = np.empty((r, k), dtype=np.uint8)
    out[:, 0::4] = packed & 0x3
    out[:, 1::4] = (packed >> 2) & 0x3
    out[:, 2::4] = (packed >> 4) & 0x3
    out[:, 3::4] = (packed >> 6) & 0x3
    return out


def dequant2_reference(packed: np.ndarray, scale_code: np.ndarray,
                       lut: np.ndarray, rows: int, k: int,
                       block: int = 32) -> np.ndarray:
    """The reference 2-bit dequantiser, written the way `cpu/dequant.cpp` would be:

        value = lut[row][code] * 2^(scale_code - 127)

    `lut` is [rows, 4] or [1, 4] fp32. This is the function `tests/data/quant2/`'s
    golden bytes pin.
    """
    codes = unpack_codes(packed.reshape(rows, k // 4), k)
    lv = lut if lut.shape[0] == rows else np.broadcast_to(lut, (rows, 4))
    v = np.take_along_axis(lv, codes.astype(np.int64), axis=1)
    s = ue8m0_value(scale_code.reshape(rows, k // block))
    return (v.reshape(rows, k // block, block) * s[..., None]).reshape(rows, k)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    """||a - b|| / ||b||, fp64 accumulation."""
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    den = np.linalg.norm(b.ravel())
    return float(np.linalg.norm((a - b).ravel()) / den) if den else 0.0


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64, copy=False).ravel()
    b = b.astype(np.float64, copy=False).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 1.0
