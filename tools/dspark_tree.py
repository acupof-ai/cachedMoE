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


# --------------------------------------------------------------------------- #
# 2. the lattice
# --------------------------------------------------------------------------- #

class Lattice:
    """Normalised draft log-probabilities over the K candidates of each position.

    Inputs (all float32 unless noted; P = 5 positions):
      cand_ids   i32 [P, K]        candidates, the anchor row's top-1 first
      cand_logit     [P, K]        BASE logits B_i at the candidates (no Markov bias)
      e_in           [256]         markov embed of the draft input token
      e_cand         [P-1, K, 256] markov embed of the candidates at positions 0..P-2
      h_cand         [P, K, 256]   markov head rows of the candidates
      lse_anchor     [P] or None   full-vocab logsumexp of the ANCHOR row
                                   B_i + bias(anchor_i); None = normalise over K only

    score[i][p][c] = B_i[c] + E[prev_p].H[c], prev_p = the input token at i = 0
    (p = 0 only), candidate p of position i-1 otherwise.
    logq[i][p][c]  = score[i][p][c] - lse_i(p).

    The anchor of position i is the prev the GPU biased that position's full row
    with: the input token at i = 0, candidate 0 of position i-1 (the greedy chain)
    after. With lse_anchor given, the tail mass outside the K candidates is known
    exactly for the anchor, tail_i = log(exp(lse_anchor_i) - sum_c exp(score[i][anchor][c])),
    and is used for every prev: lse_i(p) = logsumexp(score[i][p][:], tail_i).
    """

    def __init__(self, cand_ids, cand_logit, e_in, e_cand, h_cand, lse_anchor=None):
        self.cand_ids = np.asarray(cand_ids, dtype=np.int32)
        self.P, self.K = self.cand_ids.shape
        cl = np.asarray(cand_logit, dtype=np.float32)
        self.e_in = np.asarray(e_in, dtype=np.float32)
        self.e_cand = np.asarray(e_cand, dtype=np.float32)
        self.h_cand = np.asarray(h_cand, dtype=np.float32)
        self.logq = []                       # [P] arrays: [1, K] then [K, K]
        self.bias = []
        self.tail = []
        for i in range(self.P):
            prev = self.e_in[None, :] if i == 0 else self.e_cand[i - 1]       # [Pp, 256]
            b = dot16(prev[:, None, :], self.h_cand[i][None, :, :])           # [Pp, K] f32
            score = cl[i].astype(np.float64)[None, :] + b.astype(np.float64)
            extra = None
            if lse_anchor is not None:
                la = np.float64(lse_anchor[i])
                s = np.float64(0.0)
                for c in range(self.K):
                    s = s + dm_exp(score[0, c] - la)
                rest = 1.0 - s
                t = la + dm_log(rest) if rest > 0.0 else np.float64(-np.inf)
                extra = np.full(score.shape[0], t)
                self.tail.append(float(t))
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
    return 1.0 / (1.0 + dm_exp(-np.asarray(c, dtype=np.float64)))


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


CAND_RULES = ("anchor", "base", "union2", "union4")


def _topk(row: np.ndarray, K: int) -> np.ndarray:
    part = np.argpartition(-row, K)[:K]
    order = np.lexsort((part, -row[part]))
    return part[order].astype(np.int32)


def lattice_inputs(B: np.ndarray, input_token: int, E: np.ndarray, H: np.ndarray, K: int,
                   rule: str = "anchor"):
    """What the GPU hands the CPU for one draft, for a candidate rule.

    anchor : C_i = top-K of B_i + bias(anchor_i), anchor_0 = input token and
             anchor_i = C_{i-1}[0]. This is the reference's own greedy chain
             (5 Markov-head GEMVs, the same bytes the chain scheme reads).
    base   : C_i = top-K of B_i alone (no Markov GEMV at all).
    unionM : C_i = the rows biased by the top-M candidates of position i-1, merged
             round-robin best-first until K distinct ids (one GEMM with M columns
             per position: the same 66 MB read, M x the compute).
    Returns (cand_ids [P, K], base logits at the candidates [P, K], lse of the anchor
    row [P], e_in, e_cand [P-1, K, 256], h_cand [P, K, 256])."""
    B = np.asarray(B, dtype=np.float32)
    P = B.shape[0]
    idx = np.zeros((P, K), dtype=np.int32)
    lse = np.zeros(P, dtype=np.float32)
    m = 1
    if rule.startswith("union"):
        m = int(rule[5:])
    for i in range(P):
        anchor = input_token if i == 0 else int(idx[i - 1][0])
        if rule == "base":
            idx[i] = _topk(B[i], K)
            row = B[i] + H @ E[anchor]
            lse[i] = _lse_f64(row[None])[0]
            continue
        prevs = [anchor] if i == 0 else [int(t) for t in idx[i - 1][:m]]
        lists = []
        for pi, p in enumerate(prevs):
            row = B[i] + H @ E[p]
            if pi == 0:
                lse[i] = _lse_f64(row[None])[0]
            lists.append(_topk(row, K))
        if len(lists) == 1:
            idx[i] = lists[0]
        else:
            seen, merged, ptr = set(), [], [0] * len(lists)
            while len(merged) < K:
                for li, lst in enumerate(lists):
                    while ptr[li] < K and int(lst[ptr[li]]) in seen:
                        ptr[li] += 1
                    if ptr[li] < K and len(merged) < K:
                        t = int(lst[ptr[li]])
                        seen.add(t)
                        merged.append(t)
                        ptr[li] += 1
            idx[i] = np.asarray(merged, dtype=np.int32)
    cl = np.take_along_axis(B, idx, axis=-1)
    return idx, cl, lse, E[input_token], E[idx[:-1]], H[idx]


def exact_logq(B: np.ndarray, input_token: int, idx: np.ndarray, E: np.ndarray, H: np.ndarray):
    """Full-vocab normalisation per (position, prev): [P] arrays of log q over the
    candidates, float64. Offline reference for the lattice's tail approximation."""
    P, K = idx.shape
    out = []
    for i in range(P):
        prevs = [input_token] if i == 0 else [int(t) for t in idx[i - 1]]
        rows = []
        for p in prevs:
            row = (B[i] + H @ E[p]).astype(np.float64)
            lse = _lse_f64(row[None])[0]
            rows.append(row[idx[i]] - lse)
        out.append(np.stack(rows))
    return out


def _lse_f64(B: np.ndarray) -> np.ndarray:
    b = B.astype(np.float64)
    m = b.max(axis=-1, keepdims=True)
    return (m + np.log(np.exp(b - m).sum(axis=-1, keepdims=True)))[..., 0]


# --------------------------------------------------------------------------- #
# 5. traces (written by tools/oracle_dspark.py --tree)
# --------------------------------------------------------------------------- #

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRACES = os.path.join(REPO, "traces", "dspark_tree")
DEFAULT_MODEL = r"D:\models\DeepSeek-V4.1-Flash"
KS = (4, 8, 16, 32)
KV = 32


def load_tables(model_dir: str):
    """-> E [V, 256], H [V, 256], W [5376] float32 (bf16 on disk, promoted exactly)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import dsref                                                    # noqa: E402
    store = dsref.WeightStore(model_dir, None)
    try:
        E = store.tensor("mtp.2.markov_head.embed.weight").float().numpy().copy()
        H = store.tensor("mtp.2.markov_head.head.weight").float().numpy().copy()
        W = store.tensor("mtp.2.confidence_head.proj.weight").float().numpy()[0].copy()
    finally:
        store.close()
    return E, H, W


def load_draft(pdir: str, i: int) -> dict:
    with np.load(os.path.join(pdir, f"d{i:04d}.npz")) as z:
        d = {"B": z["B"], "x": z["x"], "lse_base": z["lse_base"]}
        d.update(json.loads(bytes(z["meta"]).decode()))
    return d


def load_verify(pdir: str, c: int, prefix: str = "v") -> dict:
    with np.load(os.path.join(pdir, f"{prefix}{c:03d}.npz")) as z:
        return {k: z[k] for k in z.files}


def mode_dirs(tdir: str):
    """-> [(prompt, mode, pdir, log)] for every trajectory with at least one cycle."""
    out = []
    if not os.path.isdir(tdir):
        return out
    for prompt in sorted(os.listdir(tdir)):
        for mode in ("greedy", "sampling"):
            pdir = os.path.join(tdir, prompt, mode)
            lp = os.path.join(pdir, "log.json")
            if os.path.exists(lp):
                with open(lp, encoding="utf-8") as f:
                    log = json.load(f)
                if log["cycles"]:
                    out.append((prompt, mode, pdir, log))
    return out


# --------------------------------------------------------------------------- #
# 6. golden data for tests/test_dspark_tree.cpp
# --------------------------------------------------------------------------- #

def _fnv1a(b: bytes) -> int:
    h = 0xcbf29ce484222325
    for byte in b:
        h ^= byte
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


def _bf16_bytes(a: np.ndarray) -> bytes:
    import torch
    t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    b = t.to(torch.bfloat16)
    assert torch.equal(b.float(), t), "value is not bf16-exact"
    return b.view(torch.int16).numpy().tobytes()


def _pad_rows(first: np.ndarray, rest: list, K: int, dtype: str) -> np.ndarray:
    """logq / bias as one [P, K, K] array; position 0's single prev row is padded
    with zeros (the C++ Lattice stores it the same way)."""
    z = np.zeros((K, K), dtype=first.dtype)
    z[0] = first[0]
    return np.stack([z] + list(rest)).astype(dtype)


def golden_case(d: dict, ver: dict, E, H, W, u_path, u_acc, u_res) -> bytes:
    """One case: the top-32 lattice inputs, the verify rows, the uniforms, and every
    expected output for K in KS x {tail, no tail}."""
    Kf = 32
    idx, cl, lse, e_in, e_cand, h_cand = lattice_inputs(d["B"], d["input_token"], E, H, Kf)
    out = bytearray()
    out += struct.pack("<I", int(d["input_token"]))
    out += idx.astype("<i4").tobytes() + cl.astype("<f4").tobytes() + lse.astype("<f4").tobytes()
    out += _bf16_bytes(e_in) + _bf16_bytes(e_cand) + _bf16_bytes(h_cand) + _bf16_bytes(d["x"])
    vids = ver["top_ids"][:, :KV].astype("<i4")
    vlog = ver["top_logits"][:, :KV].astype("<f4")
    out += vids.tobytes() + vlog.tobytes() + ver["argmax"].astype("<i4").tobytes()
    out += np.asarray(u_path, "<f8").tobytes() + np.asarray(u_acc, "<f8").tobytes()
    out += np.asarray(u_res, "<f8").tobytes()
    diag = None
    for K in KS:
        for tail in (True, False):
            lat = Lattice(idx[:, :K], cl[:, :K], e_in, e_cand[:, :K], h_cand[:, :K],
                          lse if tail else None)
            lq = _pad_rows(lat.logq[0], lat.logq[1:], K, "<f8")
            bs = _pad_rows(lat.bias[0], lat.bias[1:], K, "<f4")
            out += struct.pack("<QQ", _fnv1a(lq.tobytes()), _fnv1a(bs.tobytes()))
            paths = [lat.path(o) for o in OBJECTIVES]
            for pth in paths:
                out += struct.pack("<5I", *pth)
            out += struct.pack("<d", lat.eal_value(paths[1]))
            for pth in paths:
                cf = confidence(d["x"], lat.prev_embed(pth), W).astype("<f4")
                out += cf.tobytes()
                out += struct.pack("<I", k_from_confidence(cf, 0.5))
            sp = lat.sample(np.asarray(u_path))
            out += struct.pack("<5I", *sp)
            a, em = accept_sampling(lat, sp, 5, vids, vlog, np.asarray(u_acc), np.asarray(u_res))
            out += struct.pack("<II", a, len(em)) + np.asarray(em + [0] * (6 - len(em)), "<i4").tobytes()
            a, em = accept_greedy(lat.tokens(paths[1]), [int(v) for v in ver["argmax"]], 5)
            out += struct.pack("<II", a, len(em)) + np.asarray(em + [0] * (6 - len(em)), "<i4").tobytes()
            if K == 16 and tail:
                diag = lq
    out += diag.tobytes()
    return bytes(out)


def cmd_golden(args) -> int:
    E, H, W = load_tables(args.model)
    cases, picked = [], []
    rng = np.random.default_rng(7)
    by_mode = {"greedy": [], "sampling": []}
    for prompt, mode, pdir, log in mode_dirs(args.traces):
        by_mode[mode].append((prompt, pdir, log))
    order = []
    for i in range(args.cases):
        for mode in ("sampling", "greedy"):
            if i < len(by_mode[mode]):
                order.append((mode, *by_mode[mode][i]))
    for mode, prompt, pdir, log in order[: args.cases]:
        rec = log["cycles"][0]
        d = load_draft(pdir, rec["draft"])
        ver = load_verify(pdir, rec["cycle"])
        if rec.get("u_path"):
            u = (rec["u_path"], rec["u_acc"], rec["u_res"])
        else:
            u = (rng.random(5).tolist(), rng.random(5).tolist(), rng.random(6).tolist())
        cases.append(golden_case(d, ver, E, H, W, *u))
        picked.append({"prompt": prompt, "mode": mode, "cycle": rec["cycle"], "draft": rec["draft"]})
    blob = bytearray(b"DMTR")
    blob += struct.pack("<IIII", GOLDEN_VERSION, len(cases), 32, KV)
    blob += _bf16_bytes(W)
    for c in cases:
        blob += c
    path = os.path.join(args.out, "tree_golden.bin")
    with open(path, "wb") as f:
        f.write(blob)
    with open(os.path.join(args.out, "tree_golden.json"), "w", encoding="utf-8") as f:
        json.dump({"version": GOLDEN_VERSION, "cases": picked, "Ks": list(KS), "Kv": KV,
                   "objectives": list(OBJECTIVES), "theta": 0.5,
                   "layout": "see tests/test_dspark_tree.cpp::read_case"}, f, indent=1)
    print(f"{len(cases)} cases -> {path} ({len(blob) / 1024:.0f} KiB): {picked}")
    return 0


# --------------------------------------------------------------------------- #
# 7. CPU cost and losslessness
# --------------------------------------------------------------------------- #

def cmd_bench(args) -> int:
    E, H, W = load_tables(args.model)
    rows, items = [], []
    for prompt, mode, pdir, log in mode_dirs(args.traces):
        for rec in log["cycles"][: args.per_mode]:
            items.append((load_draft(pdir, rec["draft"]), load_verify(pdir, rec["cycle"])))
    rng = np.random.default_rng(3)
    for K in KS:
        acc = dict.fromkeys(("gather", "lat_tail", "eal", "conf", "acc_g", "lat", "sample", "acc_s"), 0.0)
        n = 0
        for _ in range(args.repeat):
            for d, ver in items:
                t0 = time.perf_counter()
                idx, cl, lse, e_in, e_cand, h_cand = lattice_inputs(d["B"], d["input_token"], E, H, 32)
                idx, cl, e_cand, h_cand = idx[:, :K], cl[:, :K], e_cand[:, :K], h_cand[:, :K]
                t1 = time.perf_counter()
                lat = Lattice(idx, cl, e_in, e_cand, h_cand, lse)
                t2 = time.perf_counter()
                pth = lat.path("eal")
                t3 = time.perf_counter()
                confidence(d["x"], lat.prev_embed(pth), W)
                t4 = time.perf_counter()
                accept_greedy(lat.tokens(pth), [int(v) for v in ver["argmax"]], 5)
                t5 = time.perf_counter()
                lat2 = Lattice(idx, cl, e_in, e_cand, h_cand, None)
                t6 = time.perf_counter()
                sp = lat2.sample(rng.random(5))
                t7 = time.perf_counter()
                accept_sampling(lat2, sp, 5, ver["top_ids"][:, :KV], ver["top_logits"][:, :KV],
                                rng.random(5), rng.random(6))
                t8 = time.perf_counter()
                for key, dt_ in zip(acc, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4,
                                          t6 - t5, t7 - t6, t8 - t7)):
                    acc[key] += dt_
                n += 1
        row = {"K": K, "n": n}
        row.update({k + "_ms": round(v / n * 1e3, 3) for k, v in acc.items()})
        rows.append(row)
        print(row, flush=True)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1)
    return 0


def cmd_lossless(args) -> int:
    """Draw N (path, accept) pairs at one cycle and compare the first emitted token's
    frequencies with direct top-Kv sampling from the verify row."""
    E, H, W = load_tables(args.model)
    got = [g for g in mode_dirs(args.traces) if g[1] == "sampling"] or mode_dirs(args.traces)
    prompt, mode, pdir, log = got[0]
    rec = log["cycles"][args.cycle]
    d = load_draft(pdir, rec["draft"])
    ver = load_verify(pdir, rec["cycle"])
    rng = np.random.default_rng(11)
    res = {}
    vids, vlog = ver["top_ids"][:, :KV], ver["top_logits"][:, :KV]
    l64 = vlog[0].astype(np.float64)
    pv = dm_exp(l64 - lse_seq(l64))
    ids = [int(v) for v in vids[0]]
    for K in (4, 16):
        idx, cl, lse, e_in, e_cand, h_cand = lattice_inputs(d["B"], d["input_token"], E, H, K)
        lat = Lattice(idx, cl, e_in, e_cand, h_cand, None)
        counts: dict[int, int] = {}
        acc0 = 0
        for _ in range(args.n):
            sp = lat.sample(rng.random(5))
            a, em = accept_sampling(lat, sp, 1, vids, vlog, rng.random(5), rng.random(6))
            counts[em[0]] = counts.get(em[0], 0) + 1
            acc0 += a
        freq = np.array([counts.get(t, 0) / args.n for t in ids])
        outside = sum(c for t, c in counts.items() if t not in ids)
        tv = 0.5 * float(np.abs(freq - pv).sum())
        exp = pv * args.n
        obs = freq * args.n
        big = exp >= 5
        chi2 = float((((obs[big] - exp[big]) ** 2) / exp[big]).sum())
        dof = int(big.sum()) - 1
        pooled_e, pooled_o = float(exp[~big].sum()), float(obs[~big].sum())
        if pooled_e >= 5:
            chi2 += (pooled_o - pooled_e) ** 2 / pooled_e
            dof += 1
        noise = float(np.sqrt(pv * (1 - pv) / (2 * np.pi * args.n)).sum())
        q0 = dm_exp(lat.logq[0][0])
        draft_tv = 0.5 * float(sum(abs(float(pv[i]) - (float(q0[list(idx[0]).index(t)])
                                                         if t in list(idx[0]) else 0.0))
                                   for i, t in enumerate(ids))
                               + sum(float(q0[c]) for c, t in enumerate(idx[0]) if int(t) not in ids))
        res[K] = {"n": args.n, "accept_rate_pos0": acc0 / args.n,
                  "expected_accept_rate": 1.0 - draft_tv,
                  "tv_vs_topKv": round(tv, 5), "tv_noise_floor": round(noise, 5),
                  "chi2": round(chi2, 2), "dof": dof, "tokens_outside_topKv": outside}
        print(K, res[K], flush=True)
    tail = 1.0 - float(np.exp(np.logaddexp.reduce(vlog[0].astype(np.float64)) - ver["lse"][0]))
    out = {"prompt": prompt, "mode": mode, "cycle": args.cycle,
           "tail_mass_beyond_Kv_row0": tail, "results": res}
    print(out)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    return 0


# --------------------------------------------------------------------------- #
# 8. CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dspark_tree.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--traces", default=DEFAULT_TRACES)
        sp.add_argument("--model", default=DEFAULT_MODEL)
        sp.add_argument("--json", default=None)

    g = sub.add_parser("golden")
    common(g)
    g.add_argument("--out", default=os.path.join(REPO, "tests", "data", "dspark"))
    g.add_argument("--cases", type=int, default=2)
    b = sub.add_parser("bench")
    common(b)
    b.add_argument("--per-mode", type=int, default=2)
    b.add_argument("--repeat", type=int, default=3)
    lo = sub.add_parser("lossless")
    common(lo)
    lo.add_argument("--n", type=int, default=200000)
    lo.add_argument("--cycle", type=int, default=0)
    a = sub.add_parser("analyse")
    common(a)
    a.add_argument("--out-stats", default=os.path.join(REPO, "tests", "data", "dspark",
                                                       "tree_stats.json"))
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "golden":
        return cmd_golden(args)
    if args.cmd == "bench":
        return cmd_bench(args)
    if args.cmd == "lossless":
        return cmd_lossless(args)
    if args.cmd == "analyse":
        return cmd_analyse(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
