#!/usr/bin/env python3
"""The three L2 tensors `tools/oracle.py` does not export, for the §7.4
compressor and indexer kernels.

Why a separate script
---------------------
`tools/oracle.py --level l2` exports one token of every stage of design §7.14,
which is everything the Q path, the KV path, sparse attention and the output
projection need. It does NOT export:

  * `index_k_all` -- the whole index-key cache the indexer scores against.
    `oracle.py` exports only the ONE key this step produced (`index_k`), which
    pins the key derivation but says nothing about the scoring, because the
    other 64 keys came out of the prefill.
  * `index_score` -- the per-position score, and therefore the only thing that
    can falsify a top-k.
  * the ratio-2 compressor's carried state. At decode position 64 a ratio-2
    source's group is INCOMPLETE ((64 + 1) % 2 != 0), so `Compressor.forward`
    returns None and `oracle.py` has nothing to export for layers 2 and 14.
    What it does do is write `kv_state[0]` / `score_state[0]`; those two, plus
    the raw `wkv` and `wgate` outputs that produced them, are what a decode
    compressor kernel has to reproduce.

`tools/oracle.py` belongs to another track, so this reaches in from outside:
it imports the module, wraps `L2Capture.attach` and `_l2_collect`, and calls
`level2_layer` unchanged. Everything `oracle.py` normally writes is written
too -- the output is a superset -- so `tests/data/l2x/` loads with the same
`tests/l2_golden.h` reader as `tests/data/l2/`.

About `index_score`
-------------------
It is not a capture point: `Indexer.forward` computes it in a local and returns
only the indices. This script recomputes it **with torch, from the reference's
own captured tensors** -- the post-fp4 `index_q`, the live `shared_attn.index_k`
slice and the `weights_proj` output -- in the same op order `Indexer.forward`
uses. That is weaker than a hook (a wrong reading of the formula would be
reproduced here too) and is labelled as such in the index's `notes`, but it is
the reference's numbers going in, so it still catches everything a GPU kernel
can get wrong about the arithmetic.

Usage (about 5 minutes, same cost as the oracle's own L2 run):
    uv run python tools/oracle_l2_extra.py --model D:/models/DeepSeek-V4.1-Flash \\
        --out tests/data/l2x --l2-layers 2,14,20
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oracle  # noqa: E402


# --- extra capture ----------------------------------------------------------

_orig_attach = oracle.L2Capture.attach


def attach(self, block):
    """`oracle.L2Capture.attach` plus a hook on the compressor's gate.

    `wgate` exists only on a ratio > 1 source and is the half of the pooling
    `oracle.py` has no use for, because at a position where the group does not
    complete there is no latent to export.
    """
    _orig_attach(self, block)
    a = block.b.attn
    if a.compressor is not None and getattr(a.compressor, "wgate", None) is not None:
        def fn(_m, _inp, out):
            if self.on:
                self.t["cmp_wgate"] = out.detach().clone()
        self._handles.append(a.compressor.wgate.register_forward_hook(fn))


_orig_collect = oracle._l2_collect


def collect(dsref_mod, cap, block, margs, pos, start_pos, n_win, extra):
    out = _orig_collect(dsref_mod, cap, block, margs, pos, start_pos, n_win, extra)
    b = block.b
    a = b.attn
    shared = cap.ref.shared_attn
    t = cap.t

    def put(name, kind, v):
        out[name] = (kind, v)

    # --- the compressor's decode-time state (design §7.4) -------------------
    # `Compressor.forward` for ratio > 1 writes slot `start_pos % ratio` of both
    # states every step and pools only when the group completes. Exporting the
    # state after the step gives a kernel both halves of the check: the raw
    # projections it must produce, and the buffer they must land in.
    if a.compressor is not None:
        if "cmp_wkv" in t:
            put("cmp_wkv_out", "f32", t["cmp_wkv"][0, pos].float().contiguous())
        if "cmp_wgate" in t:
            put("cmp_wgate_out", "f32", t["cmp_wgate"][0, pos].float().contiguous())
        c = a.compressor
        if getattr(c, "kv_state", None) is not None:
            put("cmp_state_kv", "f32", c.kv_state[0].float().contiguous())
            # -inf does not survive a round trip through a float32 comparison in
            # the C++ reader any better than it does here, but it is what the
            # reference holds for a slot no token has written yet, and softmax
            # over it is the thing being reproduced.
            put("cmp_state_score", "f32", c.score_state[0].float().contiguous())
        put("cmp_ratio", "i32", torch.tensor([c.compress_ratio], dtype=torch.int32))

    # --- the indexer's keys and scores (design §7.4) ------------------------
    if a.indexer is not None and shared.index_k is not None:
        ix = a.indexer
        ratio = ix.compress_ratio
        end_pos = start_pos + (pos + 1 if start_pos == 0 else 1)
        n = end_pos // ratio
        index_k = shared.index_k[:1, :n]
        put("index_k_all", "bf16", index_k[0].contiguous())
        idxq = [e for e in cap.q4 if e[0] == oracle.L2_INDEX_BLOCK and e[2].ndim == 4]
        if idxq and "idx_w" in t:
            q = idxq[0][3][:1, pos: pos + 1]                     # [1, 1, h, d]
            w = t["idx_w"][:1, pos: pos + 1] * (ix.softmax_scale * ix.n_heads ** -0.5)
            score = torch.einsum("bshd,btd->bsht", q, index_k)
            score = (score.relu() * w.unsqueeze(-1)).sum(dim=2)   # [1, 1, n]
            put("index_score", "f32", score[0, 0].float().contiguous())
            put("index_weights_scaled", "f32", w[0, 0].float().contiguous())
        put("index_compress_len", "i32", torch.tensor([n], dtype=torch.int32))
        put("index_topk_n", "i32", torch.tensor([min(ix.index_topk, n)], dtype=torch.int32))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--out", default="tests/data/l2x")
    p.add_argument("--l2-layers", default="2,14,20",
                   help="source layers only: every extra tensor here lives on one")
    p.add_argument("--l2-tokens", type=int, default=64)
    p.add_argument("--l2-steps", default="decode")
    p.add_argument("--l2-engram", action="store_true", default=True)
    p.add_argument("--no-l2-engram", dest="l2_engram", action="store_false")
    p.add_argument("--l2-engram-threads", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    oracle.L2Capture.attach = attach
    oracle._l2_collect = collect
    rc = oracle.level2_layer(args)
    print(f"wrote {args.out} (a superset of oracle.py --level l2: the same tensors "
          f"plus cmp_* and index_*)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
