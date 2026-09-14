#!/usr/bin/env python3
"""DSpark tree sampling over the draft matrix, and top-K acceptance on the verify matrix.

The scheme (docs/p3_dspark.md section 3, Track K2)
--------------------------------------------------
Speculation stays single-pass and single-path: one draft forward, one verify
forward of M = k + 1 tokens. Only the CPU-side sampling and comparison change.

1. **Draft = a matrix.** `DSparkBlock.forward_head` computes `logits = head(norm(x))`
   for all five block positions *before* its sampling loop, and the draft input is
   `[last token, noise x 4]` with one shared attention row for every query
   (`get_dspark_topk_idxs` -- no causal mask inside the block). So the base logits
   `B [5, V]` do not depend on which tokens are sampled. The only token-dependent
   term is the Markov bias `bias(prev) = H . E[prev]` (rank 256).
2. **Tree sampling (CPU).** Top-K candidates per position from B. Between
   consecutive positions the bias restricted to candidates is a K x K matrix of
   256-d dot products of gathered rows. A 5-level lattice search picks ONE path:
   `viterbi` (max joint log-prob), `eal` (max expected accepted length under the
   draft, q0(1 + q1(1 + q2(1 + q3(1 + q4))))), `chain` (sequential argmax inside
   the lattice), or ancestral `sample` (temperature 1, truncated to the lattice).
3. **Verify** is unchanged: `[last token, path[:k]]`, M = k + 1.
4. **Compare on the verify matrix's top-K (CPU).** Greedy: longest prefix with
   `path[j] == argmax[j]` plus the bonus token. Sampling: speculative-sampling
   acceptance against the verify rows' top-Kv distribution (see `accept_sampling`
   for the truncation rule).

Bit-exactness contract with cpu/dspark_tree.cpp
-----------------------------------------------
Every float value here is produced by an operation sequence the C++ side repeats
exactly, so the golden data can be compared with `==`:

  * `dot16`: 256- or 5376-d float32 dot products accumulated in 16 lanes (lane l
    takes elements i = l mod 16, rows in order), then the lanes summed 0..15. This
    is one AVX-512 register per row, and numpy float32 elementwise ops are IEEE.
  * `dm_exp` / `dm_log`: float64 exp and log from +, -, *, /, floor, frexp, ldexp
    only (a fixed-length Taylor / atanh series), so no libm is involved.
  * every logsumexp / cumulative sum runs sequentially in index order.

No FMA: numpy never fuses, and the C++ file is compiled with contraction off.

CLI
---
    dspark_tree.py golden  --traces traces/dspark_tree --out tests/data/dspark
    dspark_tree.py analyse --traces traces/dspark_tree --model D:\\models\\DeepSeek-V4.1-Flash
    dspark_tree.py bench   (Python CPU time of steps 2 and 4)
    dspark_tree.py lossless (empirical losslessness of accept_sampling)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time

import numpy as np

sys.dont_write_bytecode = True

P_BLOCK = 5
LANES = 16
LN2 = 0.6931471805599453
SQRT_HALF = 0.7071067811865476
EXP_TERMS = 18
LOG_TERMS = 14
EXP_LO = -745.0

OBJECTIVES = ("viterbi", "eal", "chain")


# --------------------------------------------------------------------------- #
# 1. deterministic arithmetic (mirrored in cpu/dspark_tree.cpp)
# --------------------------------------------------------------------------- #

def dm_exp(x) -> np.ndarray:
    """exp(x) for float64 x from +,-,*,/,floor,ldexp only.

    k = floor(x / ln2 + 0.5), r = x - k ln2 (|r| <= 0.35), exp(r) by an 18-term Horner
    Taylor series, then ldexp(., k). 0 below -745 (and for -inf)."""
    x = np.asarray(x, dtype=np.float64)
    lo = ~(x >= EXP_LO)                       # also catches -inf / nan
    xs = np.where(lo, 0.0, x)
    k = np.floor(xs / LN2 + 0.5)
    r = xs - k * LN2
    p = np.ones_like(xs)
    for n in range(EXP_TERMS, 0, -1):
        p = 1.0 + (p * r) / float(n)
    out = np.ldexp(p, k.astype(np.int64))
    return np.where(lo, 0.0, out)


def dm_log(y) -> np.ndarray:
    """log(y) for float64 y > 0: y = m 2^e with m in [sqrt(1/2), sqrt(2)),
    log m = 2 atanh(f), f = (m-1)/(m+1), a 14-term odd series. -inf for y <= 0."""
    y = np.asarray(y, dtype=np.float64)
    bad = ~(y > 0.0)
    ys = np.where(bad, 1.0, y)
    m, e = np.frexp(ys)
    small = m < SQRT_HALF
    m = np.where(small, m * 2.0, m)
    e = np.where(small, e - 1, e).astype(np.float64)
    f = (m - 1.0) / (m + 1.0)
    f2 = f * f
    s = np.full_like(ys, 2.0 / float(2 * LOG_TERMS + 1))
    for n in range(LOG_TERMS - 1, -1, -1):
        s = 2.0 / float(2 * n + 1) + f2 * s
    out = f * s + e * LN2
    return np.where(bad, -np.inf, out)


def dot16(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """float32 dot over the last axis (length a multiple of 16), lane order.

    Broadcasts over leading axes: `dot16(E[:, None, :], H[None, :, :])` is the K x K
    bias matrix."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    n = a.shape[-1]
    assert n == b.shape[-1] and n % LANES == 0, (a.shape, b.shape)
    rows = n // LANES
    a = a.reshape(*a.shape[:-1], rows, LANES)
    b = b.reshape(*b.shape[:-1], rows, LANES)
    lead = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    acc = np.zeros(lead + (LANES,), dtype=np.float32)
    for r in range(rows):
        acc = acc + a[..., r, :] * b[..., r, :]
    s = np.zeros(lead, dtype=np.float32)
    for l in range(LANES):
        s = s + acc[..., l]
    return s


def lse_seq(x: np.ndarray, extra: np.ndarray | None = None) -> np.ndarray:
    """logsumexp over the last axis (plus an optional extra term per row), in index
    order: m = max, s = sum_t dm_exp(x_t - m) [+ dm_exp(extra - m)], m + dm_log(s)."""
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=-1)
    if extra is not None:
        extra = np.asarray(extra, dtype=np.float64)
        m = np.maximum(m, extra)
    fin = np.isfinite(m)
    ms = np.where(fin, m, 0.0)
    s = np.zeros_like(ms)
    for t in range(x.shape[-1]):
        s = s + dm_exp(x[..., t] - ms)
    if extra is not None:
        s = s + dm_exp(extra - ms)
    return np.where(fin, ms + dm_log(s), m)


def tail_logmass(cand_logit: np.ndarray, lse_base: np.ndarray) -> np.ndarray:
    """log sum_{v not in top-K} exp(B[v]) from the full-vocab logsumexp and the K
    candidates: lse + log(1 - sum_c exp(l_c - lse)); -inf when nothing is left."""
    lb = np.asarray(lse_base, dtype=np.float64)
    s = np.zeros_like(lb)
    cl = np.asarray(cand_logit, dtype=np.float64)
    for c in range(cl.shape[-1]):
        s = s + dm_exp(cl[..., c] - lb)
    rest = 1.0 - s
    return np.where(rest > 0.0, lb + dm_log(np.where(rest > 0.0, rest, 1.0)), -np.inf)


# --------------------------------------------------------------------------- #
# 2. the lattice
# --------------------------------------------------------------------------- #

class Lattice:
    """Normalised draft log-probabilities over the top-K candidates of each position.

    Inputs (all float32 unless noted; P = 5 positions):
      cand_ids   i32 [P, K]        top-K of the base logits, best first
      cand_logit     [P, K]
      e_in           [256]         markov embed of the draft input token
      e_cand         [P-1, K, 256] markov embed of the candidates at positions 0..P-2
      h_cand         [P, K, 256]   markov head rows of the candidates
      lse_base       [P] or None   full-vocab logsumexp of B; None = normalise over K
      tail_h         [P, 256] or None  bias proxy for the tail (None = tail bias 0)

    logq[i][p][c] = B_i[c] + E[prev_p].H[c] - lse_i(p), where prev_p is the input
    token for i = 0 (p = 0 only) and candidate p of position i-1 otherwise, and
    lse_i(p) runs over the K candidates plus the tail term
    tail_i + E[prev_p].tail_h_i when lse_base is given.
    """

    def __init__(self, cand_ids, cand_logit, e_in, e_cand, h_cand,
                 lse_base=None, tail_h=None):
        self.cand_ids = np.asarray(cand_ids, dtype=np.int32)
        self.P, self.K = self.cand_ids.shape
        cl = np.asarray(cand_logit, dtype=np.float32)
        self.e_in = np.asarray(e_in, dtype=np.float32)
        self.e_cand = np.asarray(e_cand, dtype=np.float32)
        self.h_cand = np.asarray(h_cand, dtype=np.float32)
        self.logq = []                       # [P] arrays: [1, K] then [K, K]
        self.bias = []
        tails = tail_logmass(cl, lse_base) if lse_base is not None else None
        for i in range(self.P):
            prev = self.e_in[None, :] if i == 0 else self.e_cand[i - 1]       # [Pp, 256]
            b = dot16(prev[:, None, :], self.h_cand[i][None, :, :])           # [Pp, K] f32
            score = cl[i].astype(np.float64)[None, :] + b.astype(np.float64)
            extra = None
            if tails is not None:
                tb = (dot16(prev, np.broadcast_to(tail_h[i], prev.shape)).astype(np.float64)
                      if tail_h is not None else np.zeros(prev.shape[0]))
                extra = np.float64(tails[i]) + tb
            lse = lse_seq(score, extra)
            self.bias.append(b)
            self.logq.append(score - lse[:, None])

    # -- deterministic paths ------------------------------------------------
    def path(self, objective: str) -> list[int]:
        """-> candidate index per position."""
        if objective == "viterbi":
            return self._viterbi()
        if objective == "eal":
            return self._eal()
        if objective == "chain":
            return self._chain()
        raise ValueError(objective)

    def _viterbi(self) -> list[int]:
        P, K = self.P, self.K
        v = self.logq[0][0].copy()
        back = []
        for i in range(1, P):
            # cand[p, c] = v[p] + logq[i][p][c]; strict > so the first max wins
            tot = v[:, None] + self.logq[i]
            arg = np.zeros(K, dtype=np.int64)
            best = tot[0].copy()
            for p in range(1, K):
                upd = tot[p] > best
                best = np.where(upd, tot[p], best)
                arg = np.where(upd, p, arg)
            back.append(arg)
            v = best
        c = int(_first_argmax(v))
        path = [c]
        for i in range(P - 1, 0, -1):
            c = int(back[i - 1][c])
            path.append(c)
        return path[::-1]

    def _eal(self) -> list[int]:
        """argmax of sum_j prod_{i<=j} q_i -- the expected accepted length if the
        draft's own conditionals were the acceptance probabilities."""
        P, K = self.P, self.K
        g = np.zeros(K, dtype=np.float64)          # G_{i+1}(c)
        choice = [None] * P
        for i in range(P - 1, -1, -1):
            val = dm_exp(self.logq[i]) * (1.0 + g[None, :])          # [Pp, K]
            arg = np.array([_first_argmax(row) for row in val], dtype=np.int64)
            choice[i] = arg
            g = val[np.arange(val.shape[0]), arg]
        path, prev = [], 0
        for i in range(P):
            c = int(choice[i][prev])
            path.append(c)
            prev = c
        return path

    def eal_value(self, path: list[int]) -> float:
        tot, run, prev = 0.0, 1.0, 0
        for i, c in enumerate(path):
            run *= float(dm_exp(self.logq[i][prev][c]))
            tot += run
            prev = c
        return tot

    def _chain(self) -> list[int]:
        path, prev = [], 0
        for i in range(self.P):
            c = int(_first_argmax(self.logq[i][prev]))
            path.append(c)
            prev = c
        return path

    # -- ancestral sampling ---------------------------------------------------
    def sample(self, u: np.ndarray) -> list[int]:
        """One path from the lattice's own conditionals, u [P] uniforms in [0, 1)."""
        path, prev = [], 0
        for i in range(self.P):
            w = dm_exp(self.logq[i][prev])
            path.append(_sample_index(w, float(u[i])))
            prev = path[-1]
        return path

    def q_of(self, i: int, prev: int, c: int) -> float:
        return float(dm_exp(self.logq[i][prev][c]))

    def tokens(self, path: list[int]) -> list[int]:
        return [int(self.cand_ids[i][c]) for i, c in enumerate(path)]

    def prev_embed(self, path: list[int]) -> np.ndarray:
        """markov embeds the confidence head pairs with each position: [P, 256]."""
        rows = [self.e_in] + [self.e_cand[i][path[i]] for i in range(self.P - 1)]
        return np.stack(rows)


def _first_argmax(v: np.ndarray) -> int:
    best, arg = v[0], 0
    for i in range(1, v.shape[0]):
        if v[i] > best:
            best, arg = v[i], i
    return arg


def _sample_index(w: np.ndarray, u: float) -> int:
    total = 0.0
    for x in w:
        total = total + float(x)
    t = u * total
    cum = 0.0
    last = 0
    for i, x in enumerate(w):
        if x > 0.0:
            last = i
        cum = cum + float(x)
        if cum > t:
            return i
    return last


def confidence(x: np.ndarray, prev_e: np.ndarray, w: np.ndarray, bias: float = 0.0) -> np.ndarray:
    """DSparkConfidenceHead: proj(cat(x_i, E[prev_i])) -> raw score per position.
    x [P, 5120], prev_e [P, 256], w [5376]. float32, lane order."""
    h = np.concatenate([np.asarray(x, np.float32), np.asarray(prev_e, np.float32)], axis=-1)
    return dot16(h, np.broadcast_to(np.asarray(w, np.float32), h.shape)) + np.float32(bias)


def sigmoid(c) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(c, dtype=np.float64)))


def k_from_confidence(conf: np.ndarray, theta: float, k_max: int = P_BLOCK) -> int:
    """Verify length from the confidence head: the longest prefix whose cumulative
    logistic acceptance estimate stays >= theta. A prefix rule, so under the
    sampling coupling the decision for position j depends only on positions < j."""
    run, k = 1.0, 0
    for j in range(k_max):
        run *= float(sigmoid(conf[j]))
        if run < theta:
            break
        k = j + 1
    return k


# --------------------------------------------------------------------------- #
# 3. acceptance on the verify matrix
# --------------------------------------------------------------------------- #

def accept_greedy(path_tokens: list[int], argmax: list[int], k: int) -> tuple[int, list[int]]:
    """-> (a, emitted tokens). argmax[j] is the main model's top-1 on row j."""
    a = 0
    while a < k and path_tokens[a] == argmax[a]:
        a += 1
    return a, list(path_tokens[:a]) + [int(argmax[a])]


def accept_sampling(lat: Lattice, path: list[int], k: int, ver_ids: np.ndarray,
                    ver_logit: np.ndarray, u_acc: np.ndarray, u_res: np.ndarray):
    """Speculative-sampling acceptance with top-K truncation. -> (a, emitted tokens).

    Truncation rule (the scheme's definition of the target distribution):
      * target p~_j = softmax over the verify row's top-Kv logits (zero elsewhere);
      * proposal q_j(.|prev) = the lattice's conditionals normalised over its K
        candidates (the sampler proposes nothing else, so this is exact for q).
    With these two, the procedure is exactly lossless with respect to p~: each
    emitted token is distributed as top-Kv sampling from the main model. The gap to
    plain temperature-1 sampling is the tail mass beyond Kv (measured, doc 3.4).

    Accept position j iff u_acc[j] * q(x) < p~(x). On rejection, sample from
    max(0, p~ - q) over the verify row's top-Kv ids; after k acceptances sample the
    bonus token from p~_k. All sums sequential; u in [0, 1)."""
    Kv = ver_ids.shape[1]
    out = []
    for j in range(k):
        pl = ver_logit[j].astype(np.float64)
        pv = dm_exp(pl - lse_seq(pl))
        prev = 0 if j == 0 else path[j - 1]
        qrow = dm_exp(lat.logq[j][prev])
        c = path[j]
        x = int(lat.cand_ids[j][c])
        qx = float(qrow[c])
        hit = np.nonzero(ver_ids[j] == x)[0]
        px = float(pv[hit[0]]) if hit.size else 0.0
        if float(u_acc[j]) * qx < px:
            out.append(x)
            continue
        r = np.zeros(Kv, dtype=np.float64)
        cids = lat.cand_ids[j]
        for v in range(Kv):
            m = np.nonzero(cids == ver_ids[j][v])[0]
            qv = float(qrow[m[0]]) if m.size else 0.0
            r[v] = max(0.0, float(pv[v]) - qv)
        out.append(_residual_token(ver_ids[j], r, pv, float(u_res[j])))
        return j, out
    pl = ver_logit[k].astype(np.float64)
    pv = dm_exp(pl - lse_seq(pl))
    out.append(int(ver_ids[k][_sample_index(pv, float(u_res[k]))]))
    return k, out


def _residual_token(ids, r, pv, u) -> int:
    total = 0.0
    for x in r:
        total = total + float(x)
    if not total > 0.0:
        return int(ids[_first_argmax(pv)])
    return int(ids[_sample_index(r, u)])


# --------------------------------------------------------------------------- #
# 4. golden container (read by tests/test_dspark_tree.cpp)
# --------------------------------------------------------------------------- #

GOLDEN_MAGIC = b"DMTR"
GOLDEN_VERSION = 1


def lattice_inputs(B: np.ndarray, input_token: int, E: np.ndarray, H: np.ndarray, K: int):
    """Gather what the GPU would hand the CPU for one draft: top-K ids/logits per
    position, lse of each base row, and the E/H rows of the candidates."""
    B = np.asarray(B, dtype=np.float32)
    idx = np.argsort(-B, axis=-1, kind="stable")[:, :K].astype(np.int32)
    cl = np.take_along_axis(B, idx, axis=-1)
    lse = _lse_f64(B).astype(np.float32)
    e_in = E[input_token]
    e_cand = E[idx[:-1]]
    h_cand = H[idx]
    return idx, cl, lse, e_in, e_cand, h_cand


def _lse_f64(B: np.ndarray) -> np.ndarray:
    b = B.astype(np.float64)
    m = b.max(axis=-1, keepdims=True)
    return (m + np.log(np.exp(b - m).sum(axis=-1, keepdims=True)))[..., 0]


# --------------------------------------------------------------------------- #
# 5. CLI
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    raise SystemExit("CLI not yet written")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(main())
