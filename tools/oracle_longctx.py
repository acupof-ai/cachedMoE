#!/usr/bin/env python3
"""Long-context correctness data for DeepSeek-V4.1-Flash (Track M).

Design reference: section 7.4 (compressor / indexer), 7.5 (sparse attention),
11 (KV cache), 12 (the oracle). Report: docs/p3_longctx.md.

Why this exists
---------------
Everything L2 / L3 validated so far ran a 64-token prompt, and at 64 tokens three
of the long-context mechanisms are degenerate:

  * the indexer keeps min(index_topk = 512, n) positions, and n <= 65, so it keeps
    all of them -- top-k selection is never tested;
  * the 128-slot window ring never wraps, so "slot = pos % 128" and "slot = pos"
    are indistinguishable;
  * a ratio-2 compressor completes a group on alternate steps, and L2's single
    decode step at position 64 is a non-completing one, so pooling has no
    reference output (design section 12, "ratio-2 pooling unverified").

This tool builds two real-text prompts (~4K and ~17K tokens), runs them through
`inference/model.py` UNMODIFIED behind `tools/dsref.py`'s CPU kernel shims --
layer-streaming prefill, then N greedy decode steps -- and exports:

  traces/longctx/<name>/            (large, regenerable; gitignored)
      index.json + l3_prefill.bin + l3_stepNN.bin
                                    the L3 container of tools/oracle.py, loadable
                                    as-is by runtime::DecodeState::load
      l2/index.json + l2_Lnn_decodeP.bin
                                    oracle.py's L2 per-stage tensors at the probe
                                    layers for EVERY decode step
      routing.npz                   the gate's top-6 for every prompt token x layer
      stats_raw.json                per (step, layer) attention / indexer stats
  tests/data/longctx/               (small, committed)
      prompts.json                  the two prompts: text, token ids, spans
      <name>/index.json + l3s_*.bin the same L3 container, minus the big KV
      <name>/l2/...                 L2 stage tensors at the probe layers for the
                                    first two decode steps (one completes a
                                    ratio-2 group, one does not), minus the KV
      stats.json                    the aggregated numbers docs/p3_longctx.md quotes

The one deliberate deviation from "run model.py as is"
------------------------------------------------------
`Indexer.forward` computes its prefill score as

    index_score = torch.einsum("bshd,btd->bsht", q, index_k)      # [1, n, 32, n/r]
    index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)

which is 17 GB of bf16 per copy at n = 17K and ratio 1, and the statement holds
two copies. `--index-chunk ROWS` swaps `model.torch` for a proxy whose `einsum`
returns a lazy object for exactly that equation: `relu_`, `* weights` and
`sum(dim=2)` are then applied ROWS queries at a time, with the same torch ops on
the same dtypes, and every other attribute raises. The 4K run uses
`--index-verify`, which computes both and asserts they are bit-identical on all
eight index layers of the real prompt; the result is in stats_raw.json.

Usage
-----
    .venv/Scripts/python.exe tools/oracle_longctx.py prompts
    .venv/Scripts/python.exe tools/oracle_longctx.py run --name ctx4k  --index-chunk 1024 --index-verify
    .venv/Scripts/python.exe tools/oracle_longctx.py run --name ctx16k --index-chunk 1024
    .venv/Scripts/python.exe tools/oracle_longctx.py stats
"""

from __future__ import annotations

import argparse
import ctypes
import datetime
import json
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True          # never leave a __pycache__ under D:\models

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import corpus as corpus_mod  # noqa: E402
import dsref  # noqa: E402
import oracle  # noqa: E402

DEFAULT_MODEL = r"D:\models\DeepSeek-V4.1-Flash"
DEFAULT_TRACES = os.path.join(REPO, "traces", "longctx")
DEFAULT_DATA = os.path.join(REPO, "tests", "data", "longctx")
PROBE_LAYERS = (0, 2, 13, 14, 20, 39)

# --------------------------------------------------------------------------- #
# 1. prompts
# --------------------------------------------------------------------------- #

# Token counts. Neither is a multiple of 128, so the prefill leaves the window
# ring wrapped at a non-zero cutoff (slot = pos % 128 is tested, not slot = pos
# - n). 4133 is odd, so the ratio-2 compressors carry a one-token partial group
# out of prefill and decode step 0 completes it; 17010 is even, so step 1
# completes one. 17010 > 16384 = candidate_topk_blocks x candidate_block_size,
# so layer 20's candidate-block pre-filter actually drops blocks for layers
# 24-36 (at exactly 16384 it would drop at most one).
PROMPT_TOKENS = {"ctx4k": 4133, "ctx16k": 17010}
BOS_ID = 0            # <｜begin▁of▁sentence｜>, what encoding.py prepends

# Sources are read from a pinned git revision, not the working tree: other tracks
# edit docs/ concurrently, and prompts.json's ids are the truth anyway.
SOURCE_REV = "e41e54c"

HEAD = '''# ledger.py -- definitions that the rest of this bundle refers back to
LEDGER_SALT = "kestrel-4471-amber"
QUORUM_WINDOW_MS = 7919


def ledger_key(user_id: int) -> str:
    """Stable ledger row key for one user; the salt never changes."""
    return f"{LEDGER_SALT}:{user_id % QUORUM_WINDOW_MS}"


# 术语约定：下文所说的「潮汐缓存」指按访问时间淘汰、容量上限为 4471 个槽的 expert 缓存。
# The files below are unrelated reference material bundled with ledger.py.
'''

MIDDLE = '''

# ---- ledger.py, continued ----
def ledger_audit(user_ids):
    # every key shares the salt defined at the top of ledger.py
    return sorted({ledger_key(u) for u in user_ids})
'''

TAIL = '''

# ---- back to ledger.py ----
# Regression test: the salt must equal the literal defined at the very top of this file.
def test_salt_is_stable():
    assert LEDGER_SALT == "'''

EXPECTED_CONTINUATION = 'kestrel-4471-amber"'


def _git_show(path: str) -> str:
    out = subprocess.run(["git", "-C", REPO, "show", f"{SOURCE_REV}:{path}"],
                         capture_output=True)
    if out.returncode != 0:
        raise SystemExit(f"git show {SOURCE_REV}:{path} failed: {out.stderr.decode()[:200]}")
    return out.stdout.decode("utf-8", errors="replace")


def _file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def prompt_sources(model_dir: str) -> dict[str, list[tuple[str, callable]]]:
    import sysconfig
    stdlib = sysconfig.get_paths()["stdlib"]
    en = [("model:README.md", lambda: _file(os.path.join(model_dir, "README.md")))]
    for p in ("docs/p2_decode.md", "docs/p2_attention.md", "docs/kernel_p1.md"):
        en.append((f"git:{SOURCE_REV}:{p}", lambda p=p: _git_show(p)))
    zh = [(f"git:{SOURCE_REV}:{p}", lambda p=p: _git_show(p))
          for p in ("docs/design.md", "docs/route_trace.md")]
    code = [(f"model:inference/{n}", lambda n=n: _file(os.path.join(model_dir, "inference", n)))
            for n in ("model.py", "engram.py", "kernel.py")]
    for n in ("argparse.py", "json/encoder.py", "dataclasses.py", "textwrap.py"):
        if os.path.isfile(os.path.join(stdlib, n)):
            code.append((f"cpython-stdlib:{n}", lambda n=n: _file(os.path.join(stdlib, n))))
    return {"en": en, "zh": zh, "code": code}


def _find(ids: list[int], pat: list[int], start: int = 0) -> int:
    for i in range(start, len(ids) - len(pat) + 1):
        if ids[i:i + len(pat)] == pat:
            return i
    return -1


def build_prompts(model_dir: str, names=None) -> dict:
    tok = dsref.TokenizerAdapter(os.path.join(model_dir, "tokenizer.json"))
    srcs = prompt_sources(model_dir)
    # Segments of ~SEG tokens of contiguous paragraphs from one document, rotating
    # en -> zh -> code and, within a kind, across that kind's documents: every
    # stretch of the prompt mixes all three, and each segment is still coherent
    # text rather than a shuffle of unrelated paragraphs.
    SEG = 450
    docs = {}
    for kind, lst in srcs.items():
        docs[kind] = []
        for name, read in lst:
            bl = []
            for b in corpus_mod._blocks(read()):
                if len(b) > 6000:           # one giant table / code block would dominate
                    b = b[:6000]
                bl.append(b)
            docs[kind].append({"name": name, "blocks": bl, "cur": 0})
    order = []
    rot = {k: 0 for k in docs}
    total = 0
    limit = max(PROMPT_TOKENS.values()) + 2000
    while total < limit:
        progressed = False
        for k in ("en", "zh", "code"):
            live = [d for d in docs[k] if d["cur"] < len(d["blocks"])]
            if not live:
                continue
            d = live[rot[k] % len(live)]
            rot[k] += 1
            parts, n = [], 0
            while d["cur"] < len(d["blocks"]) and n < SEG:
                b = d["blocks"][d["cur"]]
                d["cur"] += 1
                parts.append(b)
                n += len(tok.encode(b))
            order.append((k, d["name"], "\n\n".join(parts)))
            total += n
            progressed = True
        if not progressed:
            break

    head_ids = [BOS_ID] + tok.encode(HEAD)
    mid_ids = tok.encode(MIDDLE)
    tail_ids = tok.encode(TAIL)
    salt_ids = tok.encode('"kestrel-4471-amber"')[1:-1]
    out = {"source_rev": SOURCE_REV, "bos_id": BOS_ID,
           "expected_continuation": EXPECTED_CONTINUATION,
           "expected_ids": tok.encode(EXPECTED_CONTINUATION), "prompts": {}}
    for name, n in PROMPT_TOKENS.items():
        if names and name not in names:
            continue
        budget = n - len(head_ids) - len(mid_ids) - len(tail_ids)
        filler, used, kinds = [], [], {"en": 0, "zh": 0, "code": 0}
        mid_at = None
        for kind, src, text in order:
            ids = tok.encode("\n\n" + text)
            take = ids[: budget - len(filler)]
            filler += take
            kinds[kind] += len(take)
            used.append(src)
            if mid_at is None and len(filler) >= budget // 2:
                mid_at = len(filler)
            if len(filler) >= budget:
                break
        if len(filler) < budget:
            raise SystemExit(f"{name}: sources only provide {len(filler)} of {budget} filler tokens")
        ids = head_ids + filler[:mid_at] + mid_ids + filler[mid_at:] + tail_ids
        assert len(ids) == n, (len(ids), n)
        s = _find(ids, salt_ids)
        assert s > 0, "salt literal not found in the head"
        m = len(head_ids) + mid_at
        out["prompts"][name] = {
            "n_tokens": n,
            "spans": {"salt_definition": [s, s + len(salt_ids)],
                      "head": [0, len(head_ids)],
                      "middle_usage": [m, m + len(mid_ids)],
                      "tail": [n - len(tail_ids), n]},
            "filler_tokens_by_kind": kinds,
            "sources": sorted(set(used)),
            "ids": [int(i) for i in ids],
            "text": tok.decode(ids),
        }
        print(f"{name}: {n} tokens, salt at {s}..{s + len(salt_ids)}, middle usage at {m}, "
              f"filler by kind {kinds}, {len(set(used))} sources")
    return out


def cmd_prompts(args) -> int:
    d = build_prompts(args.model)
    os.makedirs(args.data, exist_ok=True)
    path = os.path.join(args.data, "prompts.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    print(f"-> {path} ({os.path.getsize(path) / 1024:.0f} KiB)")
    return 0


# --------------------------------------------------------------------------- #
# 2. environment: other tracks, memory, time
# --------------------------------------------------------------------------- #

def now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str):
    print(f"[{now()}] {msg}", flush=True)


class _MemStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def _memstatus() -> _MemStatus:
    m = _MemStatus()
    m.dwLength = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m


def avail_gb() -> float:
    return _memstatus().ullAvailPhys / 2**30


def commit_avail_gb() -> float:
    """Commit limit minus commit charge (ullAvailPageFile is exactly that). The limit
    is shared with Track I's expert cache, which sizes itself from it."""
    return _memstatus().ullAvailPageFile / 2**30


COMMIT_HEADROOM_GB = 20.0


HEAVY_PY = ("oracle", "route_trace", "cache_sim", "dsref", "prefill", "dspark")


def other_processes() -> tuple[list[str], list[str]]:
    """-> (heavy python oracles, GPU / benchmark processes), excluding ourselves."""
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -match "
          "'^(python|deepmoe|bench|vulkaninfo)' } | ForEach-Object { [string]$_.ProcessId + "
          "[char]9 + $_.Name + [char]9 + $_.CommandLine }")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True,
                           timeout=60)
        lines = r.stdout.decode("utf-8", errors="replace").splitlines()
    except Exception as exc:                          # noqa: BLE001
        return [], [f"(process scan failed: {exc})"]
    me = os.getpid()
    heavy, gpu = [], []
    for ln in lines:
        parts = ln.split("\t", 2)
        if len(parts) < 2:
            continue
        pid, name = parts[0], parts[1]
        cmd = parts[2] if len(parts) > 2 else ""
        if pid.strip() == str(me) or "oracle_longctx" in cmd:
            continue
        if name.lower().startswith("python"):
            if any(h in cmd for h in HEAVY_PY):
                heavy.append(f"{pid} {cmd[:160]}")
        else:
            gpu.append(f"{pid} {name} {cmd[:120]}")
    return heavy, gpu


def wait_quiet(need_gb: float, poll_s: int, max_wait_h: float) -> dict:
    t0 = time.time()
    while True:
        heavy, gpu = other_processes()
        av = avail_gb()
        cm = commit_avail_gb()
        cm_need = need_gb + COMMIT_HEADROOM_GB
        if not heavy and av >= need_gb and cm >= cm_need:
            log(f"quiet enough: available {av:.1f} GiB (need {need_gb:.0f}), commit available "
                f"{cm:.1f} GiB (need {cm_need:.0f}); gpu/bench processes: {gpu or 'none'}")
            return {"available_gib": round(av, 1), "commit_available_gib": round(cm, 1),
                    "gpu_processes": gpu, "waited_s": round(time.time() - t0)}
        if time.time() - t0 > max_wait_h * 3600:
            raise SystemExit(f"gave up waiting after {max_wait_h} h: heavy={heavy} "
                             f"avail={av:.1f} commit={cm:.1f}")
        log(f"waiting: heavy python {heavy or 'none'}; available {av:.1f} GiB "
            f"(need {need_gb:.0f}); commit available {cm:.1f} GiB (need {cm_need:.0f}); "
            f"gpu/bench {gpu or 'none'}")
        time.sleep(poll_s)


class _NoEngramTable(torch.nn.Module):
    """Replaces model.ParallelEngramEmbedding at construction time.

    The reference's constructor does `torch.empty(384M, 256, fp8)` -- ~98 GB that
    is never touched, costs no physical memory, but is charged in full against
    the Windows commit limit, twice per pass (layers 1 and 14). dsref.make_block
    swaps the embedding for its manifest-backed row reader immediately after
    `Block(...)` returns, so nothing ever reads this object; it only has to carry
    the two attributes a reader might look at."""

    def __init__(self, num_embeddings: int, dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.dim = dim

    def forward(self, indices):                       # pragma: no cover
        raise RuntimeError("the engram table stub was read; dsref.make_block did not swap it")


def estimate(n: int, steps: int) -> dict:
    """Wall-time model, from docs/route_trace.md section 10 and docs/p2_decode.md:
    a prefill layer loads + dequantises all 384 experts (~51 ms each through
    Python) and computes n x 6 expert rows; a decode step rebuilds forty Blocks
    (~0.5 s each) and loads ~240 experts. The indexer's prefill score is n^2 x 32
    x 128 FLOP at ratio 1 (five layers) and a quarter of it at ratio 2 (three)."""
    load = 40 * 384 * 0.051
    moe = 40 * n * 0.8e-3                    # ~0.8 ms per token-layer incl. attention
    idx_flop = (5 * n * n + 3 * n * n / 2) * 32 * 128 * 2
    idx = idx_flop / 60e9                    # bf16 CPU matmul, ~60 GFLOPS, x2 for relu/mul/sum
    engram = 2 * n * 24 / 10_000
    prefill = load + moe + idx + engram + 40 * 0.5
    decode = steps * (40 * 0.6 + 240 * 0.055 + 8)
    return {"prefill_s": round(prefill), "decode_s": round(decode),
            "total_min": round((prefill + decode) / 60, 1),
            "parts_s": {"expert_load": round(load), "moe_attn": round(moe),
                        "index_score": round(idx), "engram": round(engram)}}


# --------------------------------------------------------------------------- #
# 3. the one shim: row-chunked prefill index score
# --------------------------------------------------------------------------- #

INDEX_EQ = "bshd,btd->bsht"


class _LazyIndexScore:
    """Stands in for `torch.einsum(INDEX_EQ, q, index_k)` inside Indexer.forward.

    Supports exactly the three operations the reference applies to it, in order:
    `.relu_()`, `* weights.unsqueeze(-1)`, `.sum(dim=2)`. Anything else raises,
    so a model.py that used the score differently could not be silently
    miscomputed."""

    def __init__(self, shim, q, k):
        self.__dict__.update(shim=shim, q=q, k=k, w=None, relu=False)

    def __getattr__(self, name):
        raise AttributeError(f"_LazyIndexScore does not support .{name}; "
                             f"Indexer.forward changed -- run without --index-chunk")

    def relu_(self):
        assert not self.relu and self.w is None
        self.__dict__["relu"] = True
        return self

    def __mul__(self, other):
        assert self.relu and self.w is None
        self.__dict__["w"] = other
        return self

    def sum(self, dim):
        assert dim == 2 and self.relu and self.w is not None, (dim, self.relu)
        q, k, w, rows = self.q, self.k, self.w, self.shim.rows
        s = q.size(1)
        out = None
        for lo in range(0, s, rows):
            hi = min(s, lo + rows)
            sc = torch.einsum(INDEX_EQ, q[:, lo:hi], k)
            sc.relu_()
            part = (sc * w[:, lo:hi]).sum(dim=2)
            if out is None:
                out = torch.empty(q.size(0), s, part.size(-1), dtype=part.dtype)
            out[:, lo:hi] = part
            del sc, part
        if self.shim.verify:
            full = torch.einsum(INDEX_EQ, q, k)
            full.relu_()
            full = (full * w).sum(dim=2)
            eq = bool(torch.equal(full, out))
            diff = int((full != out).sum().item())
            self.shim.checks.append({"layer": self.shim.layer, "rows": int(s),
                                     "keys": int(k.size(1)), "bit_identical": eq,
                                     "differing": diff})
            log(f"    index-chunk verify L{self.shim.layer}: {s}x{k.size(1)} "
                f"bit_identical={eq} differing={diff}")
            return full
        return out


class IndexChunkShim:
    def __init__(self, ref, rows: int, verify: bool):
        self.rows, self.verify, self.checks, self.layer = rows, verify, [], None
        real = torch
        shim = self

        class _TorchProxy:
            def __getattr__(self, name):
                return getattr(real, name)

            @staticmethod
            def einsum(eq, *ops):
                if eq == INDEX_EQ and len(ops) == 2 and ops[0].dim() == 4 \
                        and ops[0].size(1) > shim.rows:
                    return _LazyIndexScore(shim, ops[0], ops[1])
                return real.einsum(eq, *ops)

        ref.torch = _TorchProxy()


# --------------------------------------------------------------------------- #
# 4. attention statistics (analysis only -- nothing here feeds the model)
# --------------------------------------------------------------------------- #

DIST_EDGES = (128, 512, 2048, 8192)
DIST_LABELS = ("<128", "128-512", "512-2K", "2K-8K", ">=8K")


def attn_row_stats(q, kv, sink, idx, scale, pos, offset, ratio, win, prefill, span):
    """One query row's attention mass, the way sparse_attn_kernel distributes it.

    q [h, d], kv [n, d], sink [h], idx [k] (-1 = empty). fp32 throughout: these
    are statistics, not reference values."""
    idx = idx.long()
    valid = idx >= 0
    g = kv.index_select(0, idx.clamp_min(0)).float()
    s = (q.float() @ g.T) * scale
    s = s.masked_fill(~valid, float("-inf"))
    mx = s.amax(-1).clamp_min(-1e30)
    p = torch.exp(s - mx[:, None])
    p = torch.where(valid, p, torch.zeros((), dtype=p.dtype))
    sk = torch.exp(sink.float() - mx)
    denom = p.sum(-1) + sk
    pn = p / denom[:, None]
    sink_m = sk / denom
    is_win = valid & (idx < offset)
    is_cmp = valid & (idx >= offset)
    dist = torch.zeros_like(idx)
    tokpos = torch.zeros_like(idx)
    if prefill:
        dist[is_win] = pos - idx[is_win]
        tokpos[is_win] = idx[is_win]
    else:
        dist[is_win] = (pos - idx[is_win]) % win
        tokpos[is_win] = pos - dist[is_win]
    j = idx - offset
    first = j * max(ratio, 1)
    last = first + max(ratio, 1) - 1
    dist[is_cmp] = pos - last[is_cmp]
    b = torch.bucketize(dist, torch.tensor(DIST_EDGES), right=True)
    rec = {
        "sink_mass_mean": float(sink_m.mean()), "sink_mass_max": float(sink_m.max()),
        "sink_mass_min": float(sink_m.min()),
        "win_mass": float(pn[:, is_win].sum(-1).mean()),
        "cmp_mass": float(pn[:, is_cmp].sum(-1).mean()) if bool(is_cmp.any()) else 0.0,
        "n_win": int(is_win.sum()), "n_cmp": int(is_cmp.sum()),
        "mass_by_dist": [float(pn[:, valid & (b == i)].sum(-1).mean()) for i in range(5)],
        "cmp_count_by_dist": [int((is_cmp & (b == i)).sum()) for i in range(5)],
    }
    if bool(is_cmp.any()):
        cp = pn[:, is_cmp]
        srt = cp.sort(-1, descending=True).values
        tot = cp.sum(-1).clamp_min(1e-30)
        rec["cmp_top64_share"] = float((srt[:, :64].sum(-1) / tot).mean())
        rec["cmp_top16_share"] = float((srt[:, :16].sum(-1) / tot).mean())
    if span is not None:
        a, e = span
        hit_cmp = is_cmp & (first <= e - 1) & (last >= a)
        hit_win = is_win & (tokpos >= a) & (tokpos < e)
        hit = hit_cmp | hit_win
        rec["salt_rows_selected"] = int(hit_cmp.sum())
        rec["salt_mass_mean"] = float(pn[:, hit].sum(-1).mean()) if bool(hit.any()) else 0.0
        rec["salt_mass_max_head"] = float(pn[:, hit].sum(-1).max()) if bool(hit.any()) else 0.0
    if not prefill:
        rec["sink_mass_per_head"] = [round(float(v), 5) for v in sink_m]
    return rec


class AttnProbe:
    """Wraps model.sparse_attn (outermost) and computes attn_row_stats for the
    rows the driver asked for. It never changes the tensor model.py gets back."""

    def __init__(self, ref, span):
        self.span = span
        self.ctx = None
        self.records: list[dict] = []
        inner = ref.sparse_attn
        probe = self

        def sparse_attn(q, kv, sink, idx, scale):
            out = inner(q, kv, sink, idx, scale)
            c = probe.ctx
            if c is not None:
                for r in c["rows"]:
                    pos = c["pos0"] + r
                    rec = attn_row_stats(q[0, r], kv[0], sink, idx[0, r], scale, pos,
                                         c["offset"], c["ratio"], c["win"],
                                         c["prefill"], probe.span)
                    rec.update(layer=c["layer"], step=c["step"], pos=int(pos))
                    probe.records.append(rec)
            return out

        ref.sparse_attn = sparse_attn


# --------------------------------------------------------------------------- #
# 5. the run
# --------------------------------------------------------------------------- #

SMALL_L2_DROP_ALWAYS = ("win_kv", "cmp_kv", "block_in", "attn_resid_in", "engram_out",
                        "attn_block_out", "gate_bias", "wo_a_out", "moe_shared_out",
                        # the FFN half does not depend on context length except
                        # through attn_out, which is kept; the full set is in traces/
                        "wo_b_out", "ffn_pre", "ffn_post", "ffn_comb", "ffn_hc_pre_out",
                        "ffn_norm_out", "block_out", "gate_scores", "moe_out")
# per-layer names dropped from the small L3 step records (kept in traces/)
SMALL_L3_DROP_SUFFIX = (".gate_scores",)
SMALL_L2_DROP_NONCOMPLETING = ("q_pre_rope", "attn_out_irope")


def _pooling_extras(block, before: dict | None, cap) -> dict:
    """The ratio-2 pooling reference: carried state before and after the step,
    the two fp32 projections that fill a slot, and the pooled value that enters
    compressor.norm when the group completes. Absent pieces are simply absent."""
    cmp = block.b.attn.compressor
    out: dict = {}
    if cmp is None:
        return out
    if before is not None:
        out["cmp_state_kv_before"] = ("f32", before["kv"])
        out["cmp_state_score_before"] = ("f32", before["score"])
    if getattr(cmp, "kv_state", None) is not None:
        out["cmp_state_kv_after"] = ("f32", cmp.kv_state[0].float().clone())
        out["cmp_state_score_after"] = ("f32", cmp.score_state[0].float().clone())
    t = cap.t
    if "cmp_wkv" in t:
        out["cmp_wkv_out"] = ("f32", t["cmp_wkv"][0, 0].float().contiguous())
    if "cmp_wgate" in t:
        out["cmp_wgate_out"] = ("f32", t["cmp_wgate"][0, 0].float().contiguous())
    if "cmp_norm.in" in t:
        out["cmp_pooled"] = ("bf16", t["cmp_norm.in"][0, 0].contiguous())
    return out


def _attach_extra_hooks(cap, block) -> list:
    cmp = block.b.attn.compressor
    handles = []
    if cmp is None:
        return handles

    def mk(name, want_input):
        def fn(_m, inp, out):
            if not cap.on:
                return
            if want_input:
                cap.t[name] = inp[0].detach().clone()
            else:
                o = out[0] if isinstance(out, tuple) else out
                cap.t[name] = o.detach().clone()
        return fn

    if getattr(cmp, "wgate", None) is not None:
        handles.append(cmp.wgate.register_forward_hook(mk("cmp_wgate", False)))
    handles.append(cmp.norm.register_forward_hook(mk("cmp_norm.in", True)))
    return handles


def run_layer_decode(store, margs, layout_e, L, block, h, pre_mix, hashes, pos):
    """oracle.py level2's run_layer and _l3_run_layer in one: the same arithmetic,
    with both exports' captures. One token."""
    extra: dict = {"pre_mix_in": ("f32", pre_mix[0, 0].contiguous()),
                   "block_in": ("bf16", h[0, 0].contiguous())}
    if block.b.engram is not None:
        hi = layout_e.layer_ids.index(L)
        h = block.b.engram(h, hashes[:, :, hi, :], None)
        extra["engram_out"] = ("bf16", h[0, 0].contiguous())
    extra["attn_resid_in"] = ("bf16", h[0, 0].contiguous())
    ffn_in, resid, fpre, fpost, fcomb = block.forward_attn(h, pos, pre_mix)
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
    scores = moe.gate_scores(flat)[0].float().contiguous()
    extra["gate_scores"] = ("f32", scores)
    extra["gate_bias"] = ("f32", moe.gate.bias.float().contiguous())
    extra["gate_top6_ids"] = ("i32", indices[0].int().contiguous())
    extra["gate_top6_weights"] = ("f32", weights[0].float().contiguous())
    extra["moe_shared_out"] = ("f32", shared_out[0].contiguous())
    extra["moe_out"] = ("f32", y[0].contiguous())
    gcap = {"ffn_in": flat[0].clone(), "gate_scores": scores,
            "gate_ids": indices[0].int().clone(), "gate_weights": weights[0].float().clone()}
    return h_out, fpre, extra, gcap


def index_replay(ref, block, cap, pos: int, used_keys: torch.Tensor, ref_idx: torch.Tensor,
                 margs) -> dict:
    """Recompute this decode step's index score from the captured q / weights and
    (a) the keys the reference actually scored against, (b) the layer's own
    keys. (a) must reproduce the reference's top-k -- that is a check on our
    reading of Indexer.forward; (b) is the counterfactual for design section 7.4."""
    ix = block.b.attn.indexer
    ratio = ix.compress_ratio
    T = (pos + 1) // ratio
    idxq = [e for e in cap.q4 if e[0] == oracle.L2_INDEX_BLOCK and e[2].ndim == 4]
    q = idxq[0][3]                                            # [1, 1, 32, 128] post-quant
    w = cap.t["idx_w"] * (ix.softmax_scale * ix.n_heads ** -0.5)

    def topk_of(keys):
        sc = torch.einsum(INDEX_EQ, q, keys[:1, :T])
        sc = (sc.relu_() * w.unsqueeze(-1)).sum(dim=2)
        if ix.uses_candidates:
            sc = sc.masked_fill(~ref.shared_attn.candidates, float("-inf"))
        k = min(ix.index_topk, T)
        ids = sc.topk(k, dim=-1, sorted=False).indices.sort(dim=-1).values[0, 0]
        srt = sc[0, 0].float().sort(descending=True).values
        gap = float(srt[k - 1] - srt[k]) if T > k and torch.isfinite(srt[k]) else None
        thr = float(srt[k - 1])
        near = int(((srt - thr).abs() <= 1e-3 * max(abs(thr), 1e-6)).sum()) if T > k else 0
        return set(ids.tolist()), gap, near, thr

    ref_rows = set((ref_idx[ref_idx >= margs.window_size] - margs.window_size).tolist())
    got_used, gap, near, thr = topk_of(used_keys)
    rec = {"T": T, "k": len(ref_rows), "replay_match": len(got_used & ref_rows),
           "threshold_score": thr, "gap_k_k1": gap, "near_threshold": near}
    own = ix.k_cache if ix.owns_k else used_keys
    if own is not used_keys:
        got_own, gap_o, _n, _t = topk_of(own)
        rec["own_keys_overlap"] = len(got_own & ref_rows)
    if ix.is_candidate_source and ref.shared_attn.candidates is not None:
        cand = ref.shared_attn.candidates[0, 0]
        rec["candidate_positions_dropped"] = int((~cand).sum())
    return rec


def _save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def cmd_run(args) -> int:
    with open(os.path.join(args.data, "prompts.json"), encoding="utf-8") as f:
        pj = json.load(f)
    P = pj["prompts"][args.name]
    ids = P["ids"]
    N = len(ids)
    steps = args.steps
    span = tuple(P["spans"]["salt_definition"])
    tdir = os.path.join(args.traces, args.name)
    ddir = os.path.join(args.data, args.name)
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(ddir, exist_ok=True)
    ckpt_pre = os.path.join(tdir, "ckpt_prefill.pt")
    ckpt_dec = os.path.join(tdir, "ckpt_decode.pt")

    est = estimate(N, steps)
    log(f"{args.name}: N={N}, {steps} steps; estimate {est}")
    if args.estimate_only:
        return 0
    need = args.need_gb if args.need_gb else (14 + 4 * N / 4096 if args.index_chunk
                                             else 14 + 2.2 * (N / 4096) ** 2)
    env_start = wait_quiet(need, args.poll_s, args.max_wait_h)

    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    inference_dir = os.path.join(args.model, "inference")
    ref = dsref.load_reference(inference_dir)
    ref.ParallelEngramEmbedding = _NoEngramTable
    store = dsref.WeightStore(args.model, None)
    tokenizer = dsref.TokenizerAdapter(os.path.join(args.model, "tokenizer.json"))
    max_seq_len = N + steps + 8
    margs = dsref.build_args(ref, inference_dir, max_seq_len=max_seq_len)
    layout_e = ref.EngramLayout.from_args(margs)
    sys.path.insert(0, inference_dir)
    import engram as eng  # noqa: E402
    cached = dsref.CachedTokenMap.build(tokenizer, os.path.join(args.traces, "token_map.npz"))
    orig_build = eng.build_compressed_token_map
    eng.build_compressed_token_map = lambda _t: (cached.lookup, cached.size)
    try:
        ngram = ref.NgramHashState(margs, layout_e, tokenizer)
    finally:
        eng.build_compressed_token_map = orig_build

    shim = IndexChunkShim(ref, args.index_chunk, args.index_verify) if args.index_chunk else None
    cap = oracle.L2Capture(ref)
    l3cap = oracle._L3Sparse(ref)
    probe = AttnProbe(ref, span)
    shared = ref.shared_attn
    shared.compress_kv = shared.index_k = shared.topk_idxs = shared.candidates = None
    W = margs.window_size
    probes = [int(x) for x in args.probe_layers.split(",")]
    small_steps = {int(x) for x in args.small_steps.split(",") if x.strip()}
    run_log = {"env_start": env_start, "started": now(), "phases": []}

    # ------------------------------------------------------------------ prefill
    pre_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    hashes = ngram(pre_ids, 0, None)
    if os.path.exists(ckpt_dec):
        log(f"prefill already complete ({ckpt_dec} exists)")
        blob_pre = None
    else:
        if os.path.exists(ckpt_pre):
            blob_pre = torch.load(ckpt_pre, weights_only=False)
            log(f"resuming prefill at layer {blob_pre['next_layer']}")
            h, pre_mix = blob_pre["h"], blob_pre["pre_mix"]
            probe.records = blob_pre["attn"]
            if shim is not None:
                shim.checks = blob_pre["index_checks"]
            for k in ("compress_kv", "index_k", "topk_idxs", "candidates"):
                setattr(shared, k, blob_pre["shared"][k])
        else:
            embed = store.tensor("embed.weight")
            h = embed[pre_ids[0]].unsqueeze(1).repeat(1, margs.hc_mult, 1).unsqueeze(0)
            h = h.to(torch.bfloat16)
            del embed
            pre_mix = ref.make_identity_pre_mix(h, margs.hc_mult)
            blob_pre = {"next_layer": 0, "kv_state": {}, "prefill_tensors": {},
                        "small_prefill": {}, "routing": {}, "timings": [], "attn": [],
                        "index_checks": []}
        sample_rows = sorted(set(list(range(0, N, max(1, N // 32))) + [N - 1]))
        t_pass = time.perf_counter()
        run_log["phases"].append({"phase": "prefill", "start": now(),
                                  "from_layer": blob_pre["next_layer"]})
        for L in range(blob_pre["next_layer"], margs.n_layers):
            t0 = time.perf_counter()
            block = dsref.make_block(ref, margs, L, layout_e, store, args.engram_threads)
            attn = block.b.attn
            if shim is not None:
                shim.layer = L
            probe.ctx = {"rows": sample_rows, "pos0": 0, "offset": N,
                         "ratio": attn.compress_ratio, "win": W, "prefill": True,
                         "layer": L, "step": -1}
            h, pre_mix, n_used, indices = oracle._l3_run_layer(
                dsref, store, margs, layout_e, L, block, h, pre_mix, hashes, 0)
            probe.ctx = None
            kv = oracle._l2_save_attn_state(block)
            blob_pre["kv_state"][L] = kv
            pt = blob_pre["prefill_tensors"]
            pt[f"L{L:02d}.win_kv"] = ("bf16", attn.window_kv_cache[0].clone())
            pt.update(oracle._l3_prefill_state(block, L))
            pt[f"L{L:02d}.gate_bias"] = ("f32", block.b.ffn.gate.bias.float().clone())
            last = ("i32", l3cap.idx[0, -1].clone().contiguous())
            pt[f"L{L:02d}.topk_idxs_last"] = last
            sp = blob_pre["small_prefill"]
            sp[f"L{L:02d}.topk_idxs_last"] = last
            if attn.compressor is not None and getattr(attn.compressor, "kv_state", None) is not None:
                sp[f"L{L:02d}.cmp_state_kv"] = pt[f"L{L:02d}.cmp_state_kv"]
                sp[f"L{L:02d}.cmp_state_score"] = pt[f"L{L:02d}.cmp_state_score"]
            sp[f"L{L:02d}.gate_ids_last"] = ("i32", indices[-1].int().clone())
            blob_pre["routing"][L] = indices.numpy().astype(np.uint16)
            del block
            dt = time.perf_counter() - t0
            blob_pre["timings"].append({"layer": L, "seconds": round(dt, 1), "experts": n_used})
            log(f"  prefill L{L:02d} {n_used:3d} experts {dt:6.1f}s |h| "
                f"{h.float().norm().item():.4e} avail {avail_gb():.1f} GiB")
            blob_pre["next_layer"] = L + 1
            if (L + 1) % args.checkpoint_every == 0 and L + 1 < margs.n_layers:
                blob_pre.update(h=h, pre_mix=pre_mix, attn=probe.records,
                                shared={k: getattr(shared, k) for k in
                                        ("compress_kv", "index_k", "topk_idxs", "candidates")},
                                index_checks=(shim.checks if shim else []))
                t1 = time.perf_counter()
                _save(blob_pre, ckpt_pre)
                log(f"  checkpoint -> {ckpt_pre} ({time.perf_counter() - t1:.1f}s)")
        run_log["phases"][-1].update(end=now(), seconds=round(time.perf_counter() - t_pass))

        norm_w = store.tensor("norm.weight")
        head_w = store.tensor("head.weight")
        logits, normed = oracle._l3_collapse_and_head(h[:, -1:], pre_mix[:, -1:], norm_w,
                                                      head_w, margs.norm_eps)
        tok = int(logits.argmax().item())
        pt = blob_pre["prefill_tensors"]
        lrec = oracle._l3_logit_record(logits, args.topk)
        pt.update(lrec)
        pt["collapse_in"] = ("bf16", h[0, -1].contiguous())
        pt["pre_mix_final"] = ("f32", pre_mix[0, -1].contiguous())
        pt["norm_out"] = ("bf16", normed.contiguous())
        pt["argmax"] = ("i32", torch.tensor([tok], dtype=torch.int32))
        sp = blob_pre["small_prefill"]
        sp.update(lrec)
        sp["argmax"] = ("i32", torch.tensor([tok], dtype=torch.int32))
        writer = oracle.L3Writer(tdir)
        rec = writer.write("prefill", pt)
        srec = _write_small(ddir, "prefill", sp)
        np.savez_compressed(os.path.join(tdir, "routing.npz"),
                            **{f"L{L:02d}": v for L, v in blob_pre["routing"].items()})
        log(f"prefill -> {rec['file']} ({rec['bytes'] / 1e6:.1f} MB), small "
            f"{srec['bytes'] / 1e3:.0f} KB, next token {tok}")
        blob_dec = {"next_step": 0, "kv_state": blob_pre["kv_state"], "produced": [tok],
                    "index_k": shared.index_k, "publisher": _last_publisher(margs),
                    "l3_steps": list(writer.steps), "small_steps": [srec],
                    "l2_full": [], "l2_small": [], "attn_prefill": probe.records,
                    "attn_decode": [], "index": [], "index_checks": (shim.checks if shim else []),
                    "prefill_timings": blob_pre["timings"], "run_log": run_log}
        del h, pre_mix, blob_pre, logits
        _save(blob_dec, ckpt_dec)
        if os.path.exists(ckpt_pre):
            os.remove(ckpt_pre)

    # ------------------------------------------------------------------ decode
    blob = torch.load(ckpt_dec, weights_only=False)
    run_log = blob["run_log"]
    kv_state = blob["kv_state"]
    produced = blob["produced"]
    for i in range(blob["next_step"]):
        ngram(torch.tensor([[produced[i]]], dtype=torch.long), N + i, None)
    norm_w = store.tensor("norm.weight")
    head_w = store.tensor("head.weight")
    embed = store.tensor("embed.weight")
    l2_full = oracle.L2Writer(os.path.join(tdir, "l2"))
    l2_full.steps = blob["l2_full"]
    l2_small = oracle.L2Writer(os.path.join(ddir, "l2"))
    l2_small.steps = blob["l2_small"]
    writer = oracle.L3Writer(tdir)
    writer.steps = blob["l3_steps"]
    shared.index_k = blob["index_k"]
    publisher = blob["publisher"]
    probe.records = []
    if blob["next_step"] < steps:
        run_log["phases"].append({"phase": "decode", "start": now(),
                                  "from_step": blob["next_step"]})
    for s in range(blob["next_step"], steps):
        pos = N + s
        in_tok = produced[-1]
        t_step = time.perf_counter()
        t = torch.tensor([[in_tok]], dtype=torch.long)
        h = embed[t[0]].unsqueeze(1).repeat(1, margs.hc_mult, 1).unsqueeze(0).to(torch.bfloat16)
        hsh = ngram(t, pos, None)
        pre_mix = ref.make_identity_pre_mix(h, margs.hc_mult)
        shared.compress_kv = shared.topk_idxs = shared.candidates = None
        step_t: dict = {}
        small_t: dict = {}
        for L in range(margs.n_layers):
            block = dsref.make_block(ref, margs, L, layout_e, store, args.engram_threads)
            oracle._l2_load_attn_state(block, kv_state[L])
            attn = block.b.attn
            is_src = attn.compressor is not None
            is_idx = attn.indexer is not None
            want_l2 = L in probes
            cap_on = want_l2 or is_src or is_idx
            extra_handles = []
            if cap_on:
                cap.attach(block)
                extra_handles = _attach_extra_hooks(cap, block)
            cap.reset(cap_on)
            before = None
            if is_src and getattr(attn.compressor, "kv_state", None) is not None:
                before = {"kv": attn.compressor.kv_state[0].float().clone(),
                          "score": attn.compressor.score_state[0].float().clone()}
            probe.ctx = {"rows": [0], "pos0": pos, "offset": W, "ratio": attn.compress_ratio,
                         "win": W, "prefill": False, "layer": L, "step": s}
            h, pre_mix, extra, gcap = run_layer_decode(store, margs, layout_e, L, block, h,
                                                       pre_mix, hsh, pos)
            probe.ctx = None
            ref_idx = l3cap.idx[0, -1].clone()
            # which layer's key cache did this layer's indexer score against?
            published = bool(is_idx and attn.indexer.owns_k and
                             shared.index_k is attn.indexer.k_cache)
            used_src = None
            if is_idx:
                used_src = L if published else publisher
                irec = index_replay(ref, block, cap, pos, shared.index_k, ref_idx, margs)
                irec.update(step=s, layer=L, pos=pos, ratio=attn.compress_ratio,
                            keys_from_layer=used_src, published=published)
                blob["index"].append(irec)
                if irec["replay_match"] != irec["k"]:
                    log(f"    ! L{L:02d} index replay {irec['replay_match']}/{irec['k']}")
            if published:
                publisher = L

            kv_state[L] = oracle._l2_save_attn_state(block)
            step_t.update(oracle._l3_kv_tensors(l3cap, L, W))
            for k in ("ffn_in", "gate_scores", "gate_ids", "gate_weights"):
                step_t[f"L{L:02d}.{k}"] = ("bf16" if k == "ffn_in" else
                                           "i32" if k == "gate_ids" else "f32",
                                           gcap[k].contiguous())
            small_t[f"L{L:02d}.topk_idxs"] = ("i32", ref_idx.contiguous())
            for k in ("gate_ids", "gate_weights"):
                small_t[f"L{L:02d}.{k}"] = step_t[f"L{L:02d}.{k}"]
            pool = _pooling_extras(block, before, cap) if cap_on else {}
            if is_src:
                ratio = attn.compress_ratio
                if (pos + 1) % ratio == 0:
                    row = pos // ratio
                    newrow = ("bf16", attn.compress_kv_cache[0, row:row + 1].clone())
                    small_t[f"L{L:02d}.cmp_row_new"] = newrow
                    step_t[f"L{L:02d}.cmp_row_new"] = newrow
                    small_t[f"L{L:02d}.cmp_row_index"] = ("i32", torch.tensor([row],
                                                                        dtype=torch.int32))
                    if attn.indexer is not None and attn.indexer.owns_k:
                        small_t[f"L{L:02d}.index_k_row_new"] = (
                            "bf16", attn.indexer.k_cache[0, row:row + 1].clone())
                    if "cmp_norm" in cap.t:
                        pool["latent_pre_rope"] = ("bf16", cap.t["cmp_norm"][0, 0].contiguous())
                for k, v in pool.items():
                    small_t[f"L{L:02d}.{k}"] = v
            if want_l2:
                tensors = oracle._l2_collect(dsref, cap, block, margs, 0, pos, W, extra)
                tensors.update(pool)
                if is_idx:
                    tensors["index_keys_from_layer"] = ("i32", torch.tensor(
                        [used_src], dtype=torch.int32))
                l2_full.write(L, f"decode{pos}", tensors)
                if s in small_steps:
                    # keep the RoPE pairs (q_pre_rope / q, attn_out / attn_out_irope)
                    # on the step that completes a ratio-2 group only
                    drop = set(SMALL_L2_DROP_ALWAYS)
                    if (pos + 1) % 2 != 0:
                        drop |= set(SMALL_L2_DROP_NONCOMPLETING)
                    l2_small.write(L, f"decode{pos}",
                                   {k: v for k, v in tensors.items() if k not in drop})
            for hd in extra_handles:
                hd.remove()
            cap.detach()
            del block
        logits, normed = oracle._l3_collapse_and_head(h, pre_mix, norm_w, head_w, margs.norm_eps)
        nxt = int(logits.argmax().item())
        lrec = oracle._l3_logit_record(logits, args.topk)
        step_t.update(lrec)
        step_t["collapse_in"] = ("bf16", h[0, -1].contiguous())
        step_t["pre_mix_final"] = ("f32", pre_mix[0, -1].contiguous())
        step_t["norm_out"] = ("bf16", normed.contiguous())
        for tgt in (step_t, small_t):
            tgt["in_token"] = ("i32", torch.tensor([in_tok], dtype=torch.int32))
            tgt["argmax"] = ("i32", torch.tensor([nxt], dtype=torch.int32))
        small_t.update(lrec)
        rec = writer.write(f"step{s:02d}", step_t)
        srec = _write_small(ddir, f"step{s:02d}", small_t)
        produced.append(nxt)
        blob["attn_decode"].extend(probe.records)
        probe.records = []
        dt = time.perf_counter() - t_step
        margin = float(lrec["top_logits"][1][0] - lrec["top_logits"][1][1])
        log(f"  step {s} @pos {pos}: {in_tok} -> {nxt} {tokenizer.decode([nxt])!r} "
            f"margin {margin:.3f} ({dt:.1f}s, {rec['bytes'] / 1e6:.1f} MB, small "
            f"{srec['bytes'] / 1e3:.0f} KB)")
        blob.update(next_step=s + 1, kv_state=kv_state, produced=produced,
                    index_k=shared.index_k, publisher=publisher,
                    l3_steps=list(writer.steps), l2_full=list(l2_full.steps),
                    l2_small=list(l2_small.steps))
        blob["small_steps"].append(srec)
        blob.setdefault("step_seconds", []).append(round(dt, 1))
        _save(blob, ckpt_dec)
    if run_log["phases"] and "end" not in run_log["phases"][-1]:
        run_log["phases"][-1]["end"] = now()
    run_log["finished"] = now()

    # ------------------------------------------------------------------ indexes
    text = tokenizer.decode(produced)
    meta = {
        "version": oracle.L3_VERSION, "model": "DeepSeek-V4.1-Flash",
        "generator": f"tools/oracle_longctx.py run --name {args.name}",
        "prompt": f"tests/data/longctx/prompts.json#{args.name}",
        "prompt_ids": [int(i) for i in ids], "prefill_len": N, "decode_pos": N,
        "steps_exported": steps, "top_k": args.topk,
        "greedy_tokens": [int(t) for t in produced], "text": text,
        "expected_continuation": pj["expected_continuation"],
        "spans": P["spans"],
        "engram": oracle._l3_engram_tables(ngram, layout_e, margs, tdir),
        "notes": ("Same container and tensor names as tools/oracle.py --level l3. "
                  "Extensions: prefill 'Lnn.topk_idxs_last' (the full window+compressed "
                  "index row the last prompt position attended to); step 'Lnn.cmp_row_new' "
                  "(the compressed row a kv source wrote this step, when its group "
                  "completed). cmp_cache / index_k are the whole max_seq_len//ratio "
                  "buffers, of which N//ratio rows are filled after prefill."),
        "config": {"dim": margs.dim, "hc_mult": margs.hc_mult, "n_heads": margs.n_heads,
                   "head_dim": margs.head_dim, "window_size": W,
                   "vocab_size": margs.vocab_size, "n_layers": margs.n_layers,
                   "norm_eps": margs.norm_eps, "index_topk": margs.index_topk,
                   "candidate_topk_blocks": margs.candidate_topk_blocks,
                   "candidate_block_size": margs.candidate_block_size,
                   "compress_ratios": list(margs.compress_ratios),
                   "kv_source_layers": list(margs.kv_source_layers),
                   "index_source_layers": list(margs.index_source_layers),
                   "max_seq_len": max_seq_len},
        "run_log": run_log,
    }
    path = writer.finish(meta)
    small_meta = dict(meta)
    small_meta.pop("engram")      # constants, identical to tests/data/l3's; not duplicated
    small_meta["generator"] += " (small: no KV buffers; see traces/longctx)"
    small_meta["steps"] = blob["small_steps"]
    small_meta["notes"] = ("The committed half of the long-context export. Records are "
                           "the L3 container, minus every KV buffer: prefill has the "
                           "logits, each layer's last-position index row and top-6, and "
                           "the ratio-2 carried state; each step has every layer's "
                           "top-k index row, gate scores/ids/weights, the kv sources' "
                           "pooling tensors (cmp_state_*_before/after, cmp_wkv_out, "
                           "cmp_wgate_out, cmp_pooled, latent_pre_rope, cmp_row_new, "
                           "index_k_row_new) and the logits. The prefill record cannot "
                           "seed a KvStore -- load traces/longctx/<name> for that.")
    with open(os.path.join(ddir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(small_meta, f, indent=1, ensure_ascii=False)
    l2meta = {"version": oracle.L2_VERSION, "model": "DeepSeek-V4.1-Flash",
              "generator": f"tools/oracle_longctx.py run --name {args.name}",
              "prompt_ids": [int(i) for i in ids], "prefill_len": N, "decode_pos": N,
              "engram": True, "stream_dtype": "bf16", "layers": probes,
              "notes": (
                  "oracle.py L2 per-stage tensors for decode steps at long context. "
                  "Extra names on kv sources: cmp_state_{kv,score}_{before,after}, "
                  "cmp_wkv_out, cmp_wgate_out, cmp_pooled (compressor.norm input, only "
                  "on a completing step). On index sources: index_keys_from_layer "
                  "(whose key cache the indexer scored against)."),
              "config": meta["config"]}
    l2_full.finish(dict(l2meta))
    sm = dict(l2meta)
    sm["notes"] += (" Small copy: steps " + args.small_steps + " only; dropped "
                    + ", ".join(SMALL_L2_DROP_ALWAYS) + " always and "
                    + ", ".join(SMALL_L2_DROP_NONCOMPLETING)
                    + " on the step that does not complete a ratio-2 group.")
    l2_small.finish(sm)
    raw = {"name": args.name, "N": N, "produced": produced, "text": text,
           "attn_prefill": blob["attn_prefill"], "attn_decode": blob["attn_decode"],
           "index": blob["index"], "index_checks": blob["index_checks"],
           "prefill_timings": blob["prefill_timings"],
           "step_seconds": blob.get("step_seconds", []), "run_log": run_log,
           "estimate": est}
    with open(os.path.join(tdir, "stats_raw.json"), "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=1)
    log(f"done: {path}; tokens {produced} -> {text!r}")
    store.close()
    return 0


def _last_publisher(margs) -> int:
    """The last layer in a forward pass whose indexer owns (and therefore always,
    at ratio 1, publishes) index keys. model.py's shared_attn is never reset, so
    this layer's keys are what the next pass's non-publishing indexers read."""
    owners = [L for L in margs.index_source_layers if L in margs.kv_source_layers]
    return max(owners)


def _write_small(ddir: str, name: str, tensors: dict) -> dict:
    w = oracle.L3Writer(ddir)
    rec = w.write(name, tensors)
    # the small container uses its own file prefix so the two directories can
    # never be confused for one another
    src = os.path.join(ddir, rec["file"])
    dst = os.path.join(ddir, "l3s_" + rec["file"][3:])
    os.replace(src, dst)
    rec["file"] = os.path.basename(dst)
    return rec


# --------------------------------------------------------------------------- #
# 6. stats
# --------------------------------------------------------------------------- #

def _read_l3(dirpath: str) -> tuple[dict, list[dict]]:
    with open(os.path.join(dirpath, "index.json"), encoding="utf-8") as f:
        meta = json.load(f)
    recs = []
    for r in meta["steps"]:
        with open(os.path.join(dirpath, r["file"]), "rb") as f:
            blob = f.read()
        base = r["data_offset"]
        t = {}
        for e in r["tensors"]:
            raw = blob[base + e["offset"]: base + e["offset"] + e["bytes"]]
            if e["dtype"] == "i32":
                a = np.frombuffer(raw, dtype=np.int32)
            elif e["dtype"] == "f32":
                a = np.frombuffer(raw, dtype=np.float32)
            elif e["dtype"] == "bf16":
                a = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).float().numpy()
            else:
                a = np.frombuffer(raw, dtype=np.uint8)
            t[e["name"]] = a.reshape(e["shape"])
        recs.append(t)
    return meta, recs


def _jaccard_adjacent(top6: np.ndarray) -> np.ndarray:
    a, b = top6[:-1], top6[1:]
    inter = (a[:, :, None] == b[:, None, :]).any(-1).sum(-1)
    return inter / (12 - inter)


def _union_frac(top6: np.ndarray, k: int, n_experts: int = 384) -> float:
    n = top6.shape[0] - k + 1
    if n <= 0:
        return float("nan")
    oh = np.zeros((top6.shape[0], n_experts), dtype=np.int32)
    np.put_along_axis(oh, top6.astype(np.int64), 1, axis=1)
    c = np.cumsum(np.vstack([np.zeros((1, n_experts), np.int32), oh]), axis=0)
    win = (c[k:] - c[:-k]) > 0
    return float(win.sum(1).mean() / (6 * k))


def tie_analysis(tdir: str, ddir: str, N: int, steps: int, cfg: dict,
                 index_recs: list[dict]) -> list[dict]:
    """Exact score structure at the top-k boundary, on the probe index layers.

    Rebuilds the key cache each probe indexer scored against (prefill index_k plus
    the rows published per step), recomputes the score in bf16 exactly as
    Indexer.forward does from the exported index_q / index_weights, and asks: how
    many positions share the k-th score bit for bit, and which of them did
    torch.topk keep? A runtime's radix select must make the same choice or its
    top-k set differs from the reference without any arithmetic being wrong."""
    W, K = cfg["window_size"], cfg["index_topk"]
    with open(os.path.join(tdir, "index.json"), encoding="utf-8") as f:
        m3 = json.load(f)
    pre = _load_record(tdir, m3["steps"][0])
    with open(os.path.join(ddir, "index.json"), encoding="utf-8") as f:
        ms = json.load(f)
    small = [_load_record(ddir, r) for r in ms["steps"]]
    with open(os.path.join(tdir, "l2", "index.json"), encoding="utf-8") as f:
        m2 = json.load(f)
    caches = {L: pre[f"L{L:02d}.index_k"][1].clone() for L in cfg["kv_source_layers"]
              if f"L{L:02d}.index_k" in pre}
    keys_from = {(r["layer"], r["step"]): r["keys_from_layer"] for r in index_recs}
    l2 = {(r["layer"], r["step"]): r for r in m2["steps"]}
    out = []
    for s in range(steps):
        pos = N + s
        # the caches as they stand AFTER step s: sources publish before scoring,
        # and a non-publishing indexer reads layer 20's cache of the previous
        # pass, which lacks only row `pos` -- never inside [:T] for ratio 2
        st = small[1 + s]
        for L in caches:
            k = st.get(f"L{L:02d}.index_k_row_new")
            if k is not None:
                row = int(st[f"L{L:02d}.cmp_row_index"][1][0])
                caches[L][row] = k[1][0]
        for L in cfg["index_source_layers"]:
            rec = l2.get((L, f"decode{pos}"))
            if rec is None:
                continue
            t = _load_record(os.path.join(tdir, "l2"), rec)
            ratio = cfg["compress_ratios"][L]
            T = (pos + 1) // ratio
            src = keys_from[(L, s)]
            q = t["index_q"][1].reshape(1, 1, 32, -1)
            w = (t["index_weights"][1] * (128 ** -0.5 * 32 ** -0.5)).reshape(1, 1, 32)
            sc = torch.einsum(INDEX_EQ, q, caches[src][:T].unsqueeze(0))
            sc = (sc.relu_() * w.unsqueeze(-1)).sum(dim=2)[0, 0]
            idx = t["topk_idxs"][1]
            ref_rows = (idx[idx >= W] - W).long()
            k = min(K, T)
            thr = sc.sort(descending=True).values[k - 1]
            above = (sc > thr)
            eq = (sc == thr)
            sel = torch.zeros(T, dtype=torch.bool)
            sel[ref_rows] = True
            eq_ids = torch.nonzero(eq).flatten()
            chosen = torch.nonzero(eq & sel).flatten()
            rec_o = {"layer": L, "step": s, "pos": pos, "T": T, "keys_from_layer": src,
                     "above_all_selected": bool((above & ~sel).sum() == 0),
                     "n_above": int(above.sum()), "n_tied_at_threshold": int(eq.sum()),
                     "tied_selected": int(chosen.numel()),
                     "selected_below_threshold": int((sel & (sc < thr)).sum()),
                     "threshold": float(thr)}
            if chosen.numel() and chosen.numel() < eq_ids.numel():
                lo = set(eq_ids[:chosen.numel()].tolist())
                hi = set(eq_ids[-chosen.numel():].tolist())
                c = set(chosen.tolist())
                rec_o["tie_pick"] = ("lowest-index" if c == lo else
                                     "highest-index" if c == hi else "neither")
            out.append(rec_o)
    return out


def cmd_stats(args) -> int:
    out = {"generated": now(), "runs": {}}
    for name in PROMPT_TOKENS:
        tdir = os.path.join(args.traces, name)
        ddir = os.path.join(args.data, name)
        rawp = os.path.join(tdir, "stats_raw.json")
        if not os.path.exists(rawp):
            print(f"{name}: no stats_raw.json, skipped")
            continue
        with open(rawp, encoding="utf-8") as f:
            raw = json.load(f)
        meta, recs = _read_l3(ddir)
        cfg = meta["config"]
        N, W = raw["N"], cfg["window_size"]
        ratios = cfg["compress_ratios"]
        steps = meta["steps_exported"]
        r: dict = {"N": N, "steps": steps, "text": raw["text"],
                   "expected": meta["expected_continuation"],
                   "retrieved": raw["text"].startswith(meta["expected_continuation"]),
                   "produced": raw["produced"], "run_log": raw["run_log"],
                   "prefill_seconds_total": round(sum(t["seconds"] for t in raw["prefill_timings"])),
                   "step_seconds": raw["step_seconds"], "estimate": raw["estimate"],
                   "index_chunk_checks": raw["index_checks"]}
        # --- margins
        margins = []
        for i, t in enumerate(recs):
            tl = t["top_logits"]
            margins.append({"record": meta["steps"][i]["step"], "argmax": int(t["argmax"][0]),
                            "top1": float(tl[0]), "margin": float(tl[0] - tl[1]),
                            "top2_id": int(t["top_ids"][1])})
        r["margins"] = margins
        # --- indexer top-k overlap between consecutive decode steps
        idx_layers = cfg["index_source_layers"]
        ov = {}
        for L in idx_layers:
            rows = []
            for s in range(steps):
                v = recs[1 + s][f"L{L:02d}.topk_idxs"]
                rows.append(set((v[v >= W] - W).tolist()))
            pairs = []
            for s in range(steps - 1):
                a, b = rows[s], rows[s + 1]
                pairs.append({"from_step": s, "overlap": len(a & b) / max(1, len(a)),
                              "new_rows": len(b - a)})
            last = recs[0][f"L{L:02d}.topk_idxs_last"]
            prefill_rows = set((last[last >= N] - N).tolist())
            ov[L] = {"ratio": ratios[L], "k": len(rows[0]),
                     "pairs": pairs,
                     "mean_overlap": float(np.mean([p["overlap"] for p in pairs])),
                     "min_overlap": float(np.min([p["overlap"] for p in pairs])),
                     "overlap_prefill_last_vs_step0": len(prefill_rows & rows[0]) / max(1, len(rows[0])),
                     "union_over_steps": len(set().union(*rows))}
        r["topk_overlap"] = ov
        # --- replay / keys-from-layer / own-key counterfactual
        # the first 4K run counted the 128 window entries into "k"; k is min(512, T)
        for rec in raw["index"]:
            rec["k"] = min(cfg["index_topk"], rec["T"])
        r["index_decode"] = raw["index"]
        # --- attention mass
        ad = raw["attn_decode"]
        by_layer = {}
        for L in range(cfg["n_layers"]):
            rs = [a for a in ad if a["layer"] == L]
            if not rs:
                continue
            by_layer[L] = {k: float(np.mean([a[k] for a in rs])) for k in
                           ("sink_mass_mean", "sink_mass_max", "win_mass", "cmp_mass")}
            by_layer[L]["mass_by_dist"] = np.mean([a["mass_by_dist"] for a in rs], 0).tolist()
            by_layer[L]["cmp_count_by_dist"] = np.mean([a["cmp_count_by_dist"] for a in rs], 0).tolist()
            if "cmp_top64_share" in rs[0]:
                by_layer[L]["cmp_top64_share"] = float(np.mean([a["cmp_top64_share"] for a in rs]))
                by_layer[L]["cmp_top16_share"] = float(np.mean([a["cmp_top16_share"] for a in rs]))
            by_layer[L]["salt_mass_mean"] = float(np.mean([a.get("salt_mass_mean", 0) for a in rs]))
            by_layer[L]["salt_mass_max_head"] = float(np.max([a.get("salt_mass_max_head", 0) for a in rs]))
            by_layer[L]["salt_rows_selected_steps"] = int(sum(1 for a in rs if a.get("salt_rows_selected", 0)))
        r["attn_decode_by_layer"] = by_layer
        # ratio-2 layers, split by whether their kv source scored with its own
        # keys this step or with layer 20's (design section 7.4 item 5)
        src_idx = {}
        cur = None
        for L in range(cfg["n_layers"]):
            if L in idx_layers:
                cur = L
            src_idx[L] = cur
        kf = {(x["layer"], x["step"]): x["keys_from_layer"] for x in raw["index"]}
        split = {}
        for L in range(cfg["n_layers"]):
            if ratios[L] != 2:
                continue
            for tag in ("own", "layer20"):
                rs = [a for a in ad if a["layer"] == L and
                      ((kf[(src_idx[L], a["step"])] == src_idx[L]) == (tag == "own"))]
                if not rs:
                    continue
                split.setdefault(L, {})[tag] = {
                    "steps": len(rs),
                    "cmp_mass": float(np.mean([a["cmp_mass"] for a in rs])),
                    "win_mass": float(np.mean([a["win_mass"] for a in rs])),
                    "cmp_top16_share": float(np.mean([a.get("cmp_top16_share", 0) for a in rs])),
                    "salt_mass_mean": float(np.mean([a.get("salt_mass_mean", 0) for a in rs])),
                    "salt_rows_selected": float(np.mean([a.get("salt_rows_selected", 0) for a in rs])),
                    "cmp_count_by_dist": np.mean([a["cmp_count_by_dist"] for a in rs], 0).tolist(),
                    "mass_by_dist": np.mean([a["mass_by_dist"] for a in rs], 0).tolist()}
        r["attn_ratio2_by_key_source"] = split
        r["ties"] = tie_analysis(tdir, ddir, N, steps, cfg, raw["index"])
        ap = raw["attn_prefill"]
        curve = {}
        for a in ap:
            curve.setdefault(a["pos"], []).append(a["sink_mass_mean"])
        r["prefill_sink_mass_vs_pos"] = {int(p): float(np.mean(v)) for p, v in sorted(curve.items())}
        # --- routing
        rt = np.load(os.path.join(tdir, "routing.npz"))
        jac, early, late, uf = [], [], [], {k: [] for k in (2, 3, 4, 5)}
        for L in range(cfg["n_layers"]):
            t6 = rt[f"L{L:02d}"]
            j = _jaccard_adjacent(t6)
            jac.append(float(j.mean()))
            early.append(float(j[:2047].mean()))
            late.append(float(j[2047:].mean()))
            for k in uf:
                uf[k].append(_union_frac(t6, k))
        dec = []
        for L in range(cfg["n_layers"]):
            seq = [recs[0][f"L{L:02d}.gate_ids_last"]] + [recs[1 + s][f"L{L:02d}.gate_ids"]
                                                          for s in range(steps)]
            dec.append(float(_jaccard_adjacent(np.stack(seq)).mean()))
        r["routing"] = {"jaccard_mean": float(np.mean(jac)), "jaccard_per_layer": jac,
                        "jaccard_first_2048": float(np.mean(early)),
                        "jaccard_after_2048": float(np.mean(late)),
                        "jaccard_decode_mean": float(np.mean(dec)),
                        "union_frac": {k: float(np.mean(v)) for k, v in uf.items()},
                        "q3_reference_27k": {"jaccard_mean": 0.234, "union_frac_5": 0.631}}
        # --- KV bytes per decode step
        kvb = []
        src_of = {}
        cur = None
        for L in range(cfg["n_layers"]):
            if L in cfg["kv_source_layers"]:
                cur = L
            src_of[L] = cur
        for s in range(steps):
            pos = N + s
            t = recs[1 + s]
            win_rows = 0
            cmp_rows = 0
            uniq = set()
            for L in range(cfg["n_layers"]):
                v = t[f"L{L:02d}.topk_idxs"]
                win_rows += int(((v >= 0) & (v < W)).sum())
                c = v[v >= W] - W
                cmp_rows += int(c.size)
                uniq |= {(src_of[L], int(x)) for x in c}
            idx_keys = sum((pos + 1) // ratios[L] for L in idx_layers)
            kvb.append({"step": s, "pos": pos,
                        "window_B": win_rows * 528, "cmp_attended_B": cmp_rows * 288,
                        "cmp_unique_B": len(uniq) * 288, "index_keys_B": idx_keys * 68,
                        "window_rows": win_rows, "cmp_rows": cmp_rows,
                        "cmp_unique_rows": len(uniq), "index_keys": idx_keys})
        tot = {k: float(np.mean([x[k] for x in kvb])) for k in kvb[0] if k.endswith("_B")}
        tot["total_attended_B"] = tot["window_B"] + tot["cmp_attended_B"] + tot["index_keys_B"]
        tot["total_unique_B"] = tot["window_B"] + tot["cmp_unique_B"] + tot["index_keys_B"]
        r["kv_bytes_per_step"] = {"steps": kvb, "mean": tot}
        out["runs"][name] = r
    path = os.path.join(args.data, "stats.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"-> {path}")
    return 0


_KINDS = {"f32": torch.float32, "bf16": torch.bfloat16, "i32": torch.int32, "u8": torch.uint8}


def _load_record(dirpath: str, rec: dict) -> dict:
    with open(os.path.join(dirpath, rec["file"]), "rb") as f:
        blob = f.read()
    base = rec["data_offset"]
    out = {}
    for e in rec["tensors"]:
        raw = bytearray(blob[base + e["offset"]: base + e["offset"] + e["bytes"]])
        t = torch.frombuffer(raw, dtype=_KINDS[e["dtype"]]).reshape(e["shape"]).clone() \
            if e["bytes"] else torch.empty(e["shape"], dtype=_KINDS[e["dtype"]])
        out[e["name"]] = (e["dtype"], t)
    return out


def cmd_prune(args) -> int:
    """Rewrite the committed copy with the current SMALL_* drop lists, so an export
    made before a list changed does not need a rerun. traces/ is untouched."""
    for name in PROMPT_TOKENS:
        ddir = os.path.join(args.data, name)
        if not os.path.exists(os.path.join(ddir, "index.json")):
            continue
        l2dir = os.path.join(ddir, "l2")
        with open(os.path.join(l2dir, "index.json"), encoding="utf-8") as f:
            m2 = json.load(f)
        w = oracle.L2Writer(l2dir)
        for rec in m2["steps"]:
            t = _load_record(l2dir, rec)
            w.write(rec["layer"], rec["step"],
                    {k: v for k, v in t.items() if k not in SMALL_L2_DROP_ALWAYS})
        m2.pop("steps")
        m2["notes"] = m2["notes"].split(" Small copy:")[0] + (
            " Small copy: steps 0,1 only; dropped " + ", ".join(SMALL_L2_DROP_ALWAYS)
            + " always and " + ", ".join(SMALL_L2_DROP_NONCOMPLETING)
            + " on the step that does not complete a ratio-2 group.")
        w.finish(m2)
        with open(os.path.join(ddir, "index.json"), encoding="utf-8") as f:
            m3 = json.load(f)
        steps = []
        for rec in m3["steps"]:
            t = _load_record(ddir, rec)
            steps.append(_write_small(ddir, rec["step"],
                                      {k: v for k, v in t.items()
                                       if not k.endswith(SMALL_L3_DROP_SUFFIX)}))
        m3["steps"] = steps
        with open(os.path.join(ddir, "index.json"), "w", encoding="utf-8") as f:
            json.dump(m3, f, indent=1, ensure_ascii=False)
        size = sum(os.path.getsize(os.path.join(dp, fn))
                   for dp, _d, fs in os.walk(ddir) for fn in fs)
        print(f"{name}: small copy now {size / 1e6:.2f} MB")
    return 0


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="oracle_longctx.py", description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--model", default=DEFAULT_MODEL)
        sp.add_argument("--traces", default=DEFAULT_TRACES)
        sp.add_argument("--data", default=DEFAULT_DATA)

    sp = sub.add_parser("prompts", help="build tests/data/longctx/prompts.json")
    common(sp)
    sp = sub.add_parser("run", help="prefill + greedy decode + export, resumable")
    common(sp)
    sp.add_argument("--name", required=True, choices=sorted(PROMPT_TOKENS))
    sp.add_argument("--steps", type=int, default=8)
    sp.add_argument("--topk", type=int, default=64)
    sp.add_argument("--probe-layers", default=",".join(map(str, PROBE_LAYERS)))
    sp.add_argument("--small-steps", default="0,1",
                    help="decode steps whose probe-layer L2 tensors also go to tests/data")
    sp.add_argument("--index-chunk", type=int, default=0,
                    help="rows per chunk for the prefill index score (0 = model.py as is)")
    sp.add_argument("--index-verify", action="store_true",
                    help="compute the index score both ways and assert bit identity")
    sp.add_argument("--checkpoint-every", type=int, default=4)
    sp.add_argument("--threads", type=int, default=16)
    sp.add_argument("--engram-threads", type=int, default=32)
    sp.add_argument("--need-gb", type=float, default=0.0)
    sp.add_argument("--poll-s", type=int, default=180)
    sp.add_argument("--max-wait-h", type=float, default=8.0)
    sp.add_argument("--estimate-only", action="store_true")
    sp = sub.add_parser("stats", help="aggregate stats_raw.json + the small exports")
    common(sp)
    sp = sub.add_parser("prune", help="apply the current small-copy drop lists in place")
    common(sp)
    args = p.parse_args(argv)
    return {"prompts": cmd_prompts, "run": cmd_run, "stats": cmd_stats,
            "prune": cmd_prune}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
