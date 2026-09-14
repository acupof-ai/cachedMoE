#!/usr/bin/env python3
"""DSpark draft-path oracle and speculative-decoding verify loop (design sections 7.12, 9.5, 10).

What this is for
----------------
Design section 10 says the verify loop has to be written by us: `inference/generate.py`
never calls `Transformer.forward_spec`, so the *only* thing the reference pins down
is one draft cycle's arithmetic. Everything around it -- how a verify batch of
M = k + 1 tokens is pushed through the main model, what gets rolled back when a
draft is rejected, how long the accepted prefix is in practice, and how many
distinct routed experts a verify batch touches per layer (section 9.5 / 10.3's
`union_frac`) -- is ours to establish, and this file establishes it.

Three outputs, in one run:

  1. **Golden draft-path tensors** at the L3 prompt's first decode position
     (position 64), per stage, for `tests/test_gpu_dspark.cpp`: the concatenated
     37/38/39 block-input means, `main_proj`/`main_norm`, the draft attention's
     KV and window state, the five position logits (top-64 + whole-vector stats),
     the Markov head's per-position bias, the confidence head's acceptance
     probabilities and the sampled draft tokens.
  2. **The verify loop** of section 10.2/10.3, run for real on the trajectory.
  3. **The statistics** section 10.3's scheduling curve needs: the acceptance
     length distribution, tokens per verify, and the per-layer routed-expert
     union of a verify batch for every prefix M = 1..6.

Trust strategy
--------------
Same as `tools/dsref.py`: run `inference/model.py` itself. `DSparkBlock`,
`DSparkAttention`, `DSparkMarkovHead`, `DSparkConfidenceHead` and
`Transformer.forward_spec` are the reference's own code, unmodified, behind
`dsref`'s six CPU kernel shims. Three things are ours and are therefore called
out explicitly:

  * `make_mtp_block` -- the same `StreamingMoE` substitution `dsref.make_block`
    does for a backbone layer, because 128 mtp experts a stage is 7.2 GB;
  * `ChunkedHead` -- `ParallelHead` with the fp32 promotion done a chunk at a
    time, so the 1.32 GB bf16 head never exists as a 2.6 GB fp32 copy. bf16 ->
    fp32 is exact and `F.linear` over a row slice is the same arithmetic, so
    this is a memory trick and not an approximation (the same one
    `oracle.py::_l3_collapse_and_head` already uses);
  * `install_batched_decode` -- the real piece of new semantics, below.

`install_batched_decode`: why the reference cannot verify a batch
-----------------------------------------------------------------
The reference's decode path is hardwired to seqlen == 1 in exactly four places:

    Attention._window_kv:  self.window_kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
    Compressor.forward:    slot = start_pos % ratio; should_compress = (start_pos+1) % ratio == 0
    Attention._compress_kv / Indexer.forward:
                           freqs = self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
    Indexer.forward:       compress_lens = end_pos // ratio          (a scalar, no per-query mask)
    get_window_topk_idxs:  one row, "the whole ring"

A verify batch is M > 1 tokens at start_pos > 0, so all five have to be
generalised. Each generalisation below is written so that

  * at seqlen == 1 it computes *exactly* what the reference computes, and
  * at start_pos == 0 the reference's own prefill branch still runs untouched,

and `--crosscheck` proves the result: it re-runs the same verify batch as a
plain prefill of the whole sequence (the reference's own seqlen > 1 path) and
compares the logits. That is the anchor for this file, the way the L2 export's
"assert the requantised bytes dequantise back to the reference's value" is the
anchor for `oracle.py`.

Rollback
--------
Design section 10.2 says the main model's window KV and the Compressor's partial
group state must be rolled back when a draft is rejected. This file takes the
brute-force-but-exact route: snapshot every layer's attention state before the
verify forward, and if the accepted prefix is shorter than the batch, restore
the snapshot and re-run the main model on the accepted tokens only. Because
attention is causal and the compressor only pools completed groups, positions
p..p+a compute the same values in both forwards -- so the re-run also serves as
a free per-cycle check on `install_batched_decode`, and the check is asserted.
A runtime cannot afford the re-run; section "rollback" of docs/p3_dspark.md
spells out the surgical version for Track I.

Usage
-----
    .venv/Scripts/python.exe tools/oracle_dspark.py --model D:\\models\\DeepSeek-V4.1-Flash \\
        --out tests/data/dspark --steps 32 --crosscheck
"""

from __future__ import annotations

import argparse
import io
import json
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import dsref                      # noqa: E402
import oracle                     # noqa: E402  (L3Writer / L2_PROMPT / state helpers)

DSPARK_VERSION = 1

# The five positions a draft cycle emits and the noise id that fills 2..5 come
# from config.json (dspark_block_size = 5, dspark_noise_token_id = 128799); they
# are read off ModelArgs rather than hardcoded, and asserted where it matters.

_DTYPES = {"f32": (torch.float32, 4), "bf16": (torch.bfloat16, 2),
           "i32": (torch.int32, 4), "i64": (torch.int64, 8), "u8": (torch.uint8, 1)}


# --------------------------------------------------------------------------- #
# 1. the M > 1 decode path
# --------------------------------------------------------------------------- #

def _window_topk_idxs_m(win: int, bsz: int, seqlen: int, start_pos: int) -> torch.Tensor:
    """Which ring slots each of `seqlen` queries at start_pos.. may attend to.

    The reference's decode branch is

        oldest = start_pos % win + 1
        idxs = cat([arange(oldest, win), arange(oldest)])
        idxs = where(idxs > start_pos, -1, idxs)

    -- one row, every slot, with the "ring still filling" slots knocked out by
    comparing a slot index against an absolute position (valid precisely because
    slot j holds position j until the ring wraps).

    Generalising means answering the same question per query. After a batch of
    `seqlen` writes the newest position is q = start_pos + seqlen - 1, so slot s
    holds the largest position <= q congruent to s:

        p(s) = q - ((q - s) mod win)          (< 0 => never written)

    and slot s is visible to the query at position P iff 0 <= p(s) <= P. The
    p(s) > P case is new and is the whole point: with M tokens in flight, the
    slots of the *later* tokens in the same batch are already written when the
    earlier queries run, and causality says they must not be visible.

    At seqlen == 1 this returns the same *set* as the reference for every query
    (the order within a row is irrelevant -- `sparse_attn` treats slots
    independently, and model.py's own docstring says so).
    """
    q = start_pos + seqlen - 1
    s = torch.arange(win)
    p = q - ((q - s) % win)                                   # [win]
    pos = start_pos + torch.arange(seqlen).unsqueeze(1)       # [seqlen, 1]
    ok = (p >= 0) & (p.unsqueeze(0) <= pos)
    idxs = torch.where(ok, s.unsqueeze(0).expand(seqlen, win), -1)
    return idxs.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


def _completed_group_positions(start_pos: int, seqlen: int, ratio: int) -> list[int]:
    """Absolute positions of the *first token* of every compression group that
    completes inside this batch.

    A group completes at the position p with (p + 1) % ratio == 0 and stands for
    the first token of its group, p + 1 - ratio -- which is exactly the
    reference's `self.freqs_cis[start_pos + 1 - ratio]` when seqlen == 1.
    """
    return [start_pos + j + 1 - ratio
            for j in range(seqlen) if (start_pos + j + 1) % ratio == 0]


def install_batched_decode(ref) -> None:
    """Replace the five seqlen == 1 assumptions with their seqlen == M forms.

    Idempotent. Every replacement delegates to the reference for start_pos == 0.
    """
    if getattr(ref, "_deepmoe_batched_decode", False):
        return
    ref._deepmoe_batched_decode = True

    # -- Attention._window_kv ------------------------------------------------
    def _window_kv(self, x, freqs_cis, start_pos):
        bsz, seqlen, _ = x.size()
        win = self.window_size
        kv = self.kv_norm(self.wkv(x))
        ref.apply_rotary_emb(kv[..., -self.rope_head_dim:], freqs_cis)
        ref.act_quant(kv, ref.fp8_block_size, ref.scale_fmt, ref.scale_dtype, True)
        if start_pos == 0:                       # the reference's prefill branch, verbatim
            if seqlen <= win:
                self.window_kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                self.window_kv_cache[:bsz, cutoff:win], self.window_kv_cache[:bsz, :cutoff] = \
                    kv[:, -win:].split([win - cutoff, cutoff], dim=1)
            window_kv = kv
            return window_kv, ref.get_window_topk_idxs(win, bsz, seqlen, start_pos)
        # decode, generalised: one ring slot per token of the batch
        for j in range(seqlen):
            self.window_kv_cache[:bsz, (start_pos + j) % win] = kv[:, j]
        return self.window_kv_cache[:bsz], _window_topk_idxs_m(win, bsz, seqlen, start_pos)

    ref.Attention._window_kv = _window_kv

    # -- Compressor.forward --------------------------------------------------
    orig_compressor_forward = ref.Compressor.forward

    def compressor_forward(self, x, start_pos):
        ratio = self.compress_ratio
        if ratio == 1 or start_pos == 0 or x.size(1) == 1:
            return orig_compressor_forward(self, x, start_pos)
        bsz, seqlen, _ = x.size()
        dtype = x.dtype
        xf = x.float()
        kv, score = self.wkv(xf), self.wgate(xf)
        pooled = []
        for j in range(seqlen):
            slot = (start_pos + j) % ratio
            self.kv_state[:bsz, slot] = kv[:, j]
            self.score_state[:bsz, slot] = score[:, j]
            if (start_pos + j + 1) % ratio == 0:
                pooled.append((self.kv_state[:bsz]
                               * self.score_state[:bsz].softmax(dim=1)).sum(dim=1))
        if not pooled:
            return None
        return self.norm(torch.stack(pooled, dim=1).to(dtype))

    ref.Compressor.forward = compressor_forward

    # -- Attention._compress_kv (only the latent's RoPE frequencies change) ---
    def _compress_kv(self, x, qr, start_pos, offset):
        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        compress_len = (start_pos + seqlen) // ratio
        latent = None
        if self.is_kv_source:
            latent = self.compressor(x, start_pos)
            ref.shared_attn.compress_kv = self.compress_kv_cache
        idxs = self._compress_topk_idxs(x, qr, latent, start_pos, offset, compress_len)
        if latent is not None:
            freqs = (self.freqs_cis[: seqlen - seqlen % ratio: ratio] if start_pos == 0
                     else self.freqs_cis[_completed_group_positions(start_pos, seqlen, ratio)])
            ref.apply_rotary_emb(latent[..., -self.rope_head_dim:], freqs)
            ref.fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)
            self.compress_kv_cache[:bsz, start_pos // ratio:
                                   start_pos // ratio + latent.size(1)] = latent
        return ref.shared_attn.compress_kv[:bsz, :compress_len], idxs

    ref.Attention._compress_kv = _compress_kv

    # -- Indexer.forward -----------------------------------------------------
    def indexer_forward(self, x, qr, latent, start_pos, offset):
        assert self.freqs_cis is not None
        bsz, seqlen, _ = x.size()
        ratio, rd, end_pos = self.compress_ratio, self.rope_head_dim, start_pos + seqlen

        if self.owns_k and latent is not None:
            freqs = (self.freqs_cis[: seqlen - seqlen % ratio: ratio] if start_pos == 0
                     else self.freqs_cis[_completed_group_positions(start_pos, seqlen, ratio)])
            k = self.k_norm(self.wk(latent))
            ref.apply_rotary_emb(k[..., -rd:], freqs)
            ref.fp4_act_quant(k, ref.fp4_block_size, True)
            self.k_cache[:bsz, start_pos // ratio: start_pos // ratio + k.size(1)] = k
            ref.shared_attn.index_k = self.k_cache

        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.index_head_dim))
        ref.apply_rotary_emb(q[..., -rd:], self.freqs_cis[start_pos:end_pos])
        ref.fp4_act_quant(q, ref.fp4_block_size, True)

        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
        emu = getattr(ref, "_dm_emulate_decode", None)
        if emu is not None and start_pos > 0 and seqlen > 1:
            # Track K2's batch-boundary isolation (docs/p3_dspark.md section 6):
            # score every query against the key cache it would have seen had the
            # batch been decoded one token at a time. A key source publishes for
            # query j only if query j's position completes one of its groups;
            # otherwise query j keeps whatever was published before -- ultimately
            # the previous forward's last publisher (layer 20). `emu` is that
            # per-query source list, seeded by `forward_emulated`.
            if self.owns_k:
                for j in range(seqlen):
                    if (start_pos + j + 1) % ratio == 0:
                        emu[j] = self.k_cache
            T = end_pos // ratio
            groups: dict = {}
            for j, src in enumerate(emu):
                groups.setdefault(id(src), (src, []))[1].append(j)
            index_score = None
            for src, js in groups.values():
                sc = torch.einsum("bshd,btd->bsht", q[:, js], src[:bsz, :T])
                sc = (sc.relu_() * weights[:, js].unsqueeze(-1)).sum(dim=2)
                if index_score is None:
                    index_score = sc.new_empty(bsz, seqlen, T)
                index_score[:, js] = sc
        else:
            index_k = ref.shared_attn.index_k[:bsz, : end_pos // ratio]
            index_score = torch.einsum("bshd,btd->bsht", q, index_k)
            index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)

        # per-query visibility: query at position P sees (P + 1) // ratio groups.
        # start_pos == 0 gives (i + 1) // ratio, the reference's prefill form; a
        # single decode query gives end_pos // ratio, the reference's scalar.
        compress_lens = ((start_pos + torch.arange(1, seqlen + 1)) // ratio).unsqueeze(-1)
        index_score.masked_fill_(
            torch.arange(index_score.size(-1)) >= compress_lens, -torch.inf)

        if self.is_candidate_source:
            ref.shared_attn.candidates = ref.select_candidate_blocks(
                index_score, compress_lens, self.candidate_topk_blocks,
                self.candidate_block_size)
        elif self.uses_candidates:
            index_score = index_score.masked_fill(~ref.shared_attn.candidates, -torch.inf)

        topk = min(self.index_topk, end_pos // ratio)
        idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        return torch.where(idxs < compress_lens, idxs + offset, -1).int()

    ref.Indexer.forward = indexer_forward

    # -- DSparkAttention: the same ring write, for a multi-position main_x ----
    def dspark_attention_forward(self, x, start_pos, main_x):
        assert self.compress_ratio == 0
        bsz, seqlen, _ = main_x.size()
        win, rd = self.window_size, self.rope_head_dim

        main_freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]
        main_kv = self.kv_norm(self.wkv(main_x))
        ref.apply_rotary_emb(main_kv[..., -rd:], main_freqs_cis)
        ref.act_quant(main_kv, ref.fp8_block_size, ref.scale_fmt, ref.scale_dtype, True)

        if start_pos == 0:                       # the reference's prefill branch, verbatim
            if seqlen <= win:
                self.window_kv_cache[:bsz, :seqlen] = main_kv
            else:
                cutoff = seqlen % win
                self.window_kv_cache[:bsz, cutoff:win], self.window_kv_cache[:bsz, :cutoff] = \
                    main_kv[:, -win:].split([win - cutoff, cutoff], dim=1)
            return x

        bsz, block_size, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos + seqlen: start_pos + seqlen + block_size]

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        ref.apply_rotary_emb(q[..., -rd:], freqs_cis)
        kv = self.kv_norm(self.wkv(x))
        ref.apply_rotary_emb(kv[..., -rd:], freqs_cis)
        ref.act_quant(kv, ref.fp8_block_size, ref.scale_fmt, ref.scale_dtype, True)

        # the reference: matrix = cat(arange(min(win, start_pos + 1)), win + arange(block))
        # -- "every window slot that has ever been written, plus the five in-block
        # positions". With `seqlen` main positions committed this cycle the count
        # becomes start_pos + seqlen; at seqlen == 1 it is the reference's value.
        n_win = min(win, start_pos + seqlen)
        matrix = torch.cat([torch.arange(n_win), win + torch.arange(block_size)])
        topk_idxs = matrix.int().view(1, 1, -1).expand(bsz, block_size, -1).contiguous()

        for j in range(seqlen):
            self.window_kv_cache[:bsz, (start_pos + j) % win] = main_kv[:, j]
        kv = torch.cat([self.window_kv_cache[:bsz], kv], dim=1)
        o = ref.sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        ref.apply_rotary_emb(o[..., -rd:], freqs_cis, True)

        o = o.view(bsz, block_size, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return self.wo_b(o.flatten(2))

    ref.DSparkAttention.forward = dspark_attention_forward


# --------------------------------------------------------------------------- #
# 2. building the mtp stages
# --------------------------------------------------------------------------- #

class ChunkedHead(torch.nn.Module):
    """`ParallelHead` with the bf16 -> fp32 promotion done a chunk of rows at a
    time. Identical arithmetic (`F.linear(x.float(), w.float())` is a sum over K
    that does not depend on how the N rows are grouped, and bf16 -> fp32 is
    exact); it exists only so the 1.32 GB head never becomes a 2.6 GB fp32 copy.
    """

    def __init__(self, weight: torch.Tensor, chunk: int = 16384):
        super().__init__()
        self.w = weight
        self.chunk = chunk
        self.vocab_size = weight.size(0)

    def forward(self, x: torch.Tensor, full_logits: bool = False) -> torch.Tensor:
        if not full_logits:
            x = x[:, -1]
        q = x.float()
        parts = [F.linear(q, self.w[i:i + self.chunk].float())
                 for i in range(0, self.w.size(0), self.chunk)]
        return torch.cat(parts, dim=-1)


def make_mtp_block(ref, args, stage: int, store: dsref.WeightStore):
    """One `DSparkBlock` with `StreamingMoE` in place of `MoE`, weights bound.

    The same substitution `dsref.make_block` performs for a backbone layer and
    for the same reason: 128 routed experts a stage is 2.4 GB of fp4 that the
    driver streams instead of materialising.
    """
    layer_id = args.n_layers + stage
    orig_moe = ref.MoE
    ref.MoE = lambda lid, a, _ref=ref: dsref.StreamingMoE(lid, a, _ref)
    try:
        with ref.set_dtype(torch.bfloat16):
            block = ref.DSparkBlock(layer_id, args)
    finally:
        ref.MoE = orig_moe
    dsref.bind_layer_weights(block, store, f"mtp.{stage}")
    return block


def mtp_forward_attn(b, x, start_pos, pre_mix, main_x):
    """`Block.forward`'s first half with `main_x` threaded through to the draft
    attention -- `dsref.StreamingBlock.forward_attn` with `*attn_args`, which it
    does not have because a backbone layer has none."""
    residual = x
    attn_pre, attn_post, attn_comb = b.hc_mixes(x, b.hc_attn_fn, b.hc_attn_scale, b.hc_attn_base)
    y = b.hc_pre(x, pre_mix)
    y = b.attn_norm(y)
    y = b.attn(y, start_pos, main_x)
    x = b.hc_post(y, residual, attn_post, attn_comb)

    residual = x
    ffn_pre, ffn_post, ffn_comb = b.hc_mixes(x, b.hc_ffn_fn, b.hc_ffn_scale, b.hc_ffn_base)
    ffn_in = b.ffn_norm(b.hc_pre(x, attn_pre))
    return ffn_in, residual, ffn_pre, ffn_post, ffn_comb


# --------------------------------------------------------------------------- #
# 3. the main model, layer-streamed
# --------------------------------------------------------------------------- #

def collapse_and_head_all(h, pre_mix, norm_w, head_w, norm_eps: float, chunk: int = 16384):
    """`Transformer.forward`'s tail with every position kept.

    `oracle.py::_l3_collapse_and_head` keeps only the last position because plain
    decode needs only that; a verify batch needs all M, and the accepted prefix
    is decided by comparing all M argmaxes against the drafts.
    """
    x = torch.sum(pre_mix.unsqueeze(-1) * h.float(), dim=2).to(h.dtype)   # hc_pre
    xf = x[0].float()
    var = xf.square().mean(-1, keepdim=True)
    normed = (norm_w.float() * (xf * torch.rsqrt(var + norm_eps))).to(h.dtype)
    q = normed.float()
    parts = [F.linear(q, head_w[i:i + chunk].float())
             for i in range(0, head_w.size(0), chunk)]
    return torch.cat(parts, dim=-1), normed


class MainRunner:
    """The forty backbone layers, one `Block` in memory at a time.

    Carries the ~200 KiB of per-layer attention state between passes exactly the
    way `oracle.py --level l3` does, and additionally records
    `h.mean(dim=2)` at the three DSpark target layers *before* the block runs
    (design section 2.4's correction, model.py line 1266).
    """

    def __init__(self, ref, store, margs, layout_e, ngram, embed, norm_w, head_w,
                 engram_threads: int = 32, verbose: bool = True):
        self.ref, self.store, self.margs = ref, store, margs
        self.layout_e, self.ngram = layout_e, ngram
        self.embed, self.norm_w, self.head_w = embed, norm_w, head_w
        self.engram_threads = engram_threads
        self.verbose = verbose
        self.kv_state: dict[int, dict] = {}
        self.target_layers = tuple(margs.dspark_target_layer_ids)

    # -- state ------------------------------------------------------------
    def snapshot(self) -> dict:
        return {L: {k: v.clone() for k, v in st.items()} for L, st in self.kv_state.items()}

    def restore(self, snap: dict) -> None:
        self.kv_state = {L: {k: v.clone() for k, v in st.items()} for L, st in snap.items()}

    def reset(self) -> None:
        self.kv_state = {}
        sa = self.ref.shared_attn
        sa.compress_kv = sa.index_k = sa.topk_idxs = sa.candidates = None

    # -- one forward ------------------------------------------------------
    def forward(self, tokens: list[int], start_pos: int, want_all_logits: bool = True):
        """M tokens at `start_pos`. Returns (logits [M, V] or [1, V],
        main_hidden [1, M, 15360], routing {layer: indices [M, topk]},
        n_used {layer: int})."""
        margs = self.margs
        t = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        h = self.embed[t[0]].unsqueeze(1).repeat(1, margs.hc_mult, 1).unsqueeze(0).to(torch.bfloat16)
        hashes = self.ngram(t, start_pos, None)
        pre_mix = self.ref.make_identity_pre_mix(h, margs.hc_mult)
        mains, routing, n_used = [], {}, {}
        for L in range(margs.n_layers):
            block = dsref.make_block(self.ref, margs, L, self.layout_e, self.store,
                                     self.engram_threads)
            if L in self.kv_state:
                oracle._l2_load_attn_state(block, self.kv_state[L])
            if L in self.target_layers:
                # model.py: `if i in self.target_layer_ids: main_hiddens.append(h.mean(dim=2))`
                # -- after the engram (layers 1/14 only, so not here) and before the block.
                mains.append(h.mean(dim=2).clone())
            h, pre_mix, used, idx = oracle._l3_run_layer(
                dsref, self.store, margs, self.layout_e, L, block, h, pre_mix,
                hashes, start_pos)
            self.kv_state[L] = oracle._l2_save_attn_state(block)
            routing[L] = idx.clone()
            n_used[L] = used
            del block
        main_hidden = torch.cat(mains, dim=-1)
        if want_all_logits:
            logits, normed = collapse_and_head_all(h, pre_mix, self.norm_w, self.head_w,
                                                   margs.norm_eps)
        else:
            logits, normed = oracle._l3_collapse_and_head(h, pre_mix, self.norm_w,
                                                          self.head_w, margs.norm_eps)
            logits = logits.unsqueeze(0)
            normed = normed.unsqueeze(0)
        return logits, main_hidden, routing, n_used, normed


# --------------------------------------------------------------------------- #
# 4. the draft path
# --------------------------------------------------------------------------- #

class DraftCapture:
    """Per-stage golden tensors for the draft block, recorded without changing
    any arithmetic.

    Same construction as `oracle.py::L2Capture` and for the same reason (design
    section 12: "no tensor is re-derived -- every export is a hook's input or
    output, or an argument the reference passed to one of the module-level
    functions"), with two differences:

      * module hooks know their stage at attach time, so names are prefixed
        rather than overwritten as the three stages run;
      * `wkv` / `kv_norm` fire twice a stage -- once for `main_x`, once for the
        five draft positions -- so those are kept as lists, in call order.

    RoPE has no capture point of its own: `apply_rotary_emb` rewrites its
    argument in place and returns nothing anyone stores. Post-RoPE values are
    therefore taken from where the reference hands them on -- `sparse_attn`'s
    `q` and `kv` arguments -- and the inverse rotation from the live output
    tensor, which `Attention.forward` rotates after `sparse_attn` returns.
    """

    def __init__(self, ref):
        self.ref = ref
        self.on = False
        self.t: dict = {}
        self.live: dict = {}
        self.q8: list = []            # (pre, post) of every inplace act_quant
        self.sparse: list = []
        self.mixes: list = []
        self.posts: list = []
        self._handles: list = []
        self._install()

    def _install(self):
        ref, cap = self.ref, self

        orig_act = ref.act_quant

        def act_quant(x, *a, **k):
            if cap.on and len(a) >= 4 and a[3] is True:
                pre = x.detach().clone()
                out = orig_act(x, *a, **k)
                cap.q8.append((pre, out.detach().clone()))
                return out
            return orig_act(x, *a, **k)
        ref.act_quant = act_quant
        self._orig_act = orig_act

        orig_sparse = ref.sparse_attn

        def sparse_attn(q, kv, sink, topk_idxs, scale):
            out = orig_sparse(q, kv, sink, topk_idxs, scale)
            if cap.on:
                cap.sparse.append({"q": q.detach().clone(), "kv": kv.detach().clone(),
                                   "sink": sink.detach().clone(),
                                   "idx": topk_idxs.detach().clone(),
                                   "o_pre_inv": out.detach().clone()})
                cap.live.setdefault("o", []).append(out)
            return out
        ref.sparse_attn = sparse_attn
        self._orig_sparse = orig_sparse

        orig_mixes = ref.Block.hc_mixes

        def hc_mixes(self_b, x, fn, sc, base):
            r = orig_mixes(self_b, x, fn, sc, base)
            if cap.on:
                cap.mixes.append(tuple(v.detach().clone() for v in r))
            return r
        ref.Block.hc_mixes = hc_mixes
        self._orig_mixes = orig_mixes

        orig_post = ref.Block.hc_post

        def hc_post(self_b, x, residual, post, comb):
            r = orig_post(self_b, x, residual, post, comb)
            if cap.on:
                cap.posts.append(r.detach().clone())
            return r
        ref.Block.hc_post = hc_post
        self._orig_post = orig_post

    def uninstall(self):
        self.ref.act_quant = self._orig_act
        self.ref.sparse_attn = self._orig_sparse
        self.ref.Block.hc_mixes = self._orig_mixes
        self.ref.Block.hc_post = self._orig_post

    def attach(self, block, stage: int):
        p = f"s{stage}."

        def hook(name, keep_input=False, multi=False):
            def fn(_m, inp, out):
                if not self.on:
                    return
                o = out[0] if isinstance(out, tuple) else out
                v = o.detach().clone()
                if multi:
                    self.t.setdefault(name, []).append(v)
                else:
                    self.t[name] = v
                if keep_input:
                    key = name + ".in"
                    iv = inp[0].detach().clone()
                    if multi:
                        self.t.setdefault(key, []).append(iv)
                    else:
                        self.t[key] = iv
            return fn

        b = block
        pairs = [(p + "attn_norm", b.attn_norm, True, False),
                 (p + "ffn_norm", b.ffn_norm, True, False),
                 (p + "wq_a", b.attn.wq_a, True, False),
                 (p + "q_norm", b.attn.q_norm, False, False),
                 (p + "wq_b", b.attn.wq_b, False, False),
                 (p + "wkv", b.attn.wkv, True, True),
                 (p + "kv_norm", b.attn.kv_norm, False, True),
                 (p + "wo_b", b.attn.wo_b, True, False)]
        if getattr(b, "main_proj", None) is not None:
            pairs += [(p + "main_proj", b.main_proj, True, False),
                      (p + "main_norm", b.main_norm, False, False)]
        if getattr(b, "norm", None) is not None:
            pairs += [(p + "final_norm", b.norm, True, False)]
        if getattr(b, "markov_head", None) is not None:
            pairs += [(p + "markov_embed", b.markov_head.embed, False, True)]
        if getattr(b, "confidence_head", None) is not None:
            pairs += [(p + "conf_proj", b.confidence_head.proj, True, False)]
        for name, mod, ki, multi in pairs:
            self._handles.append(mod.register_forward_hook(hook(name, ki, multi)))

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def reset(self, on: bool):
        self.on = on
        self.t, self.live = {}, {}
        self.q8, self.sparse, self.mixes, self.posts = [], [], [], []


class DraftRunner:
    """The three mtp stages, resident.

    Unlike the backbone these stay in memory: the non-expert half of a stage is
    ~250 MB in bf16, and `forward_spec` runs once a cycle. The 128 routed
    experts a stage are still streamed (`StreamingMoE`), which is also what
    design section 7.12's "mtp expert all pinned, no I/O wait" costs on the real
    machine -- 2.4 GB a stage resident there, streamed here.
    """

    def __init__(self, ref, store, margs, embed_mod, head_mod, verbose: bool = True):
        self.ref, self.store, self.margs = ref, store, margs
        self.verbose = verbose
        self.blocks = []
        for stage in range(margs.n_mtp_layers):
            t0 = time.perf_counter()
            b = make_mtp_block(ref, margs, stage, store)
            b.embed = embed_mod
            b.head = head_mod
            self.blocks.append(b)
            if verbose:
                print(f"  mtp stage {stage} built in {time.perf_counter() - t0:.1f}s", flush=True)
        assert margs.dspark_block_size == 5, margs.dspark_block_size
        self.block_size = margs.dspark_block_size

    # -- the reference's forward_spec, with the expert half streamed ---------
    def _run(self, input_id: int, main_hidden: torch.Tensor, start_pos: int,
             capture=None):
        """`Transformer.forward_spec`, with `StreamingMoE`'s routed half driven here.

            h, main_x = mtp[0].forward_embed(main_hidden, input_ids)
            pre_mix   = make_identity_pre_mix(h, hc_mult)
            for layer in mtp: h, pre_mix = layer(h, start_pos, pre_mix, main_x)
            return mtp[-1].forward_head(h, pre_mix, input_ids)
        """
        ref, margs = self.ref, self.margs
        ids = torch.tensor([input_id], dtype=torch.long)
        h, main_x = self.blocks[0].forward_embed(main_hidden, ids)
        if capture is not None:
            capture["main_x"] = main_x.detach().clone()
            capture["draft_embed"] = h.detach().clone()
        pre_mix = ref.make_identity_pre_mix(h, margs.hc_mult)
        routing = {}
        for stage, b in enumerate(self.blocks):
            if start_pos == 0:
                # `DSparkBlock.forward`'s own start_pos == 0 branch: seed the ring
                # and hand the (unused) draft stream straight through.
                h, pre_mix = b(h, 0, pre_mix, main_x)
                continue
            ffn_in, resid, fpre, fpost, fcomb = mtp_forward_attn(b, h, start_pos, pre_mix, main_x)
            moe = b.ffn
            flat = ffn_in.view(-1, margs.dim)
            weights, indices = moe.route(flat)
            routing[stage] = indices.clone()
            y = torch.zeros(flat.size(0), margs.dim, dtype=torch.float32)
            used = sorted(set(indices.reshape(-1).tolist()))
            layer_id = margs.n_layers + stage
            for e, w1, w2, w3 in self.store.expert_stream(layer_id, used, margs.dim,
                                                          margs.moe_inter_dim):
                r, s = torch.where(indices == e)
                y[r] += dsref.expert_ffn(flat[r], w1, w2, w3, weights[r, s, None],
                                         margs.swiglu_limit).float()
                del w1, w2, w3
            y += moe.shared_experts(flat).float()
            h = b.hc_post(y.unsqueeze(0).to(resid.dtype), resid, fpost, fcomb)
            pre_mix = fpre
            if capture is not None:
                capture[f"s{stage}.ffn_in"] = ffn_in.detach().clone()
                capture[f"s{stage}.route_w"] = weights.detach().clone()
                capture[f"s{stage}.route_i"] = indices.detach().clone()
                capture[f"s{stage}.block_out"] = h.detach().clone()
        if start_pos == 0:
            return None
        out = self.blocks[-1].forward_head(h, pre_mix, ids)
        return out, routing

    def seed(self, main_hidden_all: torch.Tensor, input_id: int) -> None:
        """`forward_spec(..., start_pos=0)`: fill each stage's 128-slot window KV
        ring from the prompt's `main_x`. Returns nothing, like the reference."""
        self._run(input_id, main_hidden_all, 0)

    def draft(self, input_id: int, main_hidden: torch.Tensor, start_pos: int,
              capture=None):
        """-> (output_ids [1, 6], logits [1, 5, V], confidence [1, 5], routing)."""
        (output_ids, logits, confidence), routing = self._run(
            input_id, main_hidden, start_pos, capture)
        return output_ids, logits, confidence, routing


# --------------------------------------------------------------------------- #
# 5. export container
# --------------------------------------------------------------------------- #

class DSparkWriter:
    """`oracle.py`'s L2/L3 container: magic + version + count, a flat blob, and a
    JSON index beside it. Same reader on the C++ side."""

    MAGIC = b"DMDK"

    def __init__(self, out_dir: str):
        self.dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.records: list[dict] = []

    def write(self, name: str, tensors: dict) -> dict:
        fname = f"dspark_{name}.bin"
        entries, blob = [], bytearray()
        seen: dict[bytes, int] = {}
        for key, (kind, t) in tensors.items():
            dtype, width = _DTYPES[kind]
            a = t.detach().contiguous()
            if a.dtype != dtype:
                a = a.to(dtype)
            raw = a.view(torch.uint8).numpy().tobytes() if kind == "bf16" else a.numpy().tobytes()
            assert len(raw) == a.numel() * width, (key, len(raw), a.numel())
            at = seen.get(raw)
            if at is None:
                at = seen[raw] = len(blob)
                blob += raw
            entries.append({"name": key, "dtype": kind, "shape": list(a.shape),
                            "offset": at, "bytes": len(raw)})
        with open(os.path.join(self.dir, fname), "wb") as f:
            f.write(self.MAGIC)
            f.write(struct.pack("<II", DSPARK_VERSION, len(entries)))
            f.write(bytes(blob))
        # `layer` / `step` rather than a name of its own: `tests/l2_golden.h`'s
        # `load_l2` already parses exactly this index shape, so the C++ side
        # needs no second reader.
        rec = {"layer": len(self.records), "step": name, "file": fname,
               "data_offset": len(self.MAGIC) + 8,
               "bytes": len(blob), "tensors": entries}
        self.records.append(rec)
        return rec

    def finish(self, meta: dict) -> str:
        meta = dict(meta)
        meta["steps"] = self.records
        path = os.path.join(self.dir, "index.json")
        with io.open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1, ensure_ascii=False)
        return path


def golden_stage_tensors(cap: "DraftCapture", capture: dict, n_stages: int,
                         full_stage: int = 0) -> dict:
    """Everything `tests/test_gpu_dspark.cpp` needs to drive one stage at a time.

    Every entry is a hook's input or output, or an argument the reference handed
    to `sparse_attn` / `act_quant` -- nothing is re-derived here.

    The three stages are structurally identical, so the two large attention
    tensors (`sparse_attn`'s q and its output) are exported for `full_stage`
    only; the whole record has to stay under the 5 MB `tests/data` budget and
    validating the same kernel three times buys nothing.
    """
    out: dict = {}

    def put(key, kind, t):
        out[key] = (kind, t)

    for s in range(n_stages):
        p = f"s{s}."
        t = cap.t
        put(p + "hc_pre_attn", "bf16", t[p + "attn_norm.in"][0])
        put(p + "attn_norm", "bf16", t[p + "attn_norm"][0])
        put(p + "wq_a", "bf16", t[p + "wq_a"][0])
        put(p + "q_norm", "bf16", t[p + "q_norm"][0])
        put(p + "wq_b_prerope", "bf16", t[p + "wq_b"][0])
        # wkv / kv_norm fire twice a stage: [0] is main_x, [1] is the five drafts
        put(p + "wkv_main", "bf16", t[p + "wkv"][0][0])
        put(p + "kv_norm_main", "bf16", t[p + "kv_norm"][0][0])
        put(p + "wkv_draft", "bf16", t[p + "wkv"][1][0])
        put(p + "kv_norm_draft", "bf16", t[p + "kv_norm"][1][0])
        # act_quant(..., inplace=True) x2 a stage: main_kv then the draft kv,
        # each captured before and after, both post-RoPE
        pre_m, post_m = cap.q8[2 * s]
        pre_d, post_d = cap.q8[2 * s + 1]
        put(p + "main_kv_prequant", "bf16", pre_m[0])
        put(p + "main_kv", "bf16", post_m[0])
        put(p + "draft_kv_prequant", "bf16", pre_d[0])
        put(p + "draft_kv", "bf16", post_d[0])
        sp = cap.sparse[s]
        put(p + "sparse_kv", "bf16", sp["kv"][0])
        put(p + "sparse_idx", "i32", sp["idx"][0])
        put(p + "attn_sink", "f32", sp["sink"])
        if s == full_stage:
            put(p + "sparse_q", "bf16", sp["q"][0])
            put(p + "attn_out_inv", "bf16", cap.live["o"][s][0])
        put(p + "wo_a_out", "bf16", t[p + "wo_b.in"][0])
        put(p + "wo_b", "bf16", t[p + "wo_b"][0])
        put(p + "hc_pre_ffn", "bf16", t[p + "ffn_norm.in"][0])
        put(p + "ffn_norm", "bf16", t[p + "ffn_norm"][0])
        put(p + "route_w", "f32", capture[f"s{s}.route_w"])
        put(p + "route_i", "i32", capture[f"s{s}.route_i"].int())
        put(p + "block_out", "bf16", capture[f"s{s}.block_out"][0])
        amix, fmix = cap.mixes[2 * s], cap.mixes[2 * s + 1]
        for tag, mix in (("attn", amix), ("ffn", fmix)):
            put(f"{p}hc_{tag}_pre", "f32", mix[0][0])
            put(f"{p}hc_{tag}_post", "f32", mix[1][0])
            put(f"{p}hc_{tag}_comb", "f32", mix[2][0])
    # stage 0's main_proj / main_norm and stage 2's head-side tensors
    put("main_proj_in", "bf16", cap.t["s0.main_proj.in"][0])
    put("main_proj_out", "bf16", cap.t["s0.main_proj"][0])
    put("main_norm_out", "bf16", cap.t["s0.main_norm"][0])
    last = n_stages - 1
    put("head_norm_in", "bf16", cap.t[f"s{last}.final_norm.in"][0])
    put("head_norm_out", "bf16", cap.t[f"s{last}.final_norm"][0])
    put("markov_embed", "bf16", torch.cat(cap.t[f"s{last}.markov_embed"], dim=0).reshape(-1))
    put("conf_proj_in", "f32", cap.t[f"s{last}.conf_proj.in"][0])
    put("conf_proj_out", "f32", cap.t[f"s{last}.conf_proj"][0])
    return out


def logit_record(logits: torch.Tensor, prefix: str, top_k: int = 64) -> dict:
    """top-k (id, logit) plus max / logsumexp / min of the whole vector, per row.
    Same shape of evidence `oracle.py::_l3_logit_record` exports, for M rows."""
    lg = logits.float()
    if lg.dim() == 1:
        lg = lg.unsqueeze(0)
    vals, idx = torch.topk(lg, top_k, dim=-1)
    lse = torch.logsumexp(lg, dim=-1)
    stats = torch.stack([lg.max(dim=-1).values, lse, lg.min(dim=-1).values], dim=-1)
    return {f"{prefix}.top_ids": ("i32", idx.int()),
            f"{prefix}.top_logits": ("f32", vals),
            f"{prefix}.stats": ("f32", stats)}


# --------------------------------------------------------------------------- #
# 6. statistics
# --------------------------------------------------------------------------- #

def prefix_unions(routing: dict) -> dict:
    """For each layer, the number of distinct routed experts the first m tokens
    of a verify batch touch, m = 1..M. This is section 9.5 / 10.3's
    `union_frac`: union(m) / (m * topk)."""
    out = {}
    for L, idx in routing.items():
        seen, sizes = set(), []
        for m in range(idx.size(0)):
            seen |= set(int(v) for v in idx[m].reshape(-1).tolist())
            sizes.append(len(seen))
        out[int(L)] = sizes
    return out


def conf_to_prob(conf: list[float]) -> list[float]:
    """The confidence head's output is NOT a probability.

    `DSparkConfidenceHead.forward` is `self.proj(hidden.float()).squeeze(-1)` --
    a bare fp32 linear with no activation. Design section 10.3 uses `c_j` as a
    conditional acceptance probability, so something has to map the score onto
    [0, 1]; the only squashing function consistent with a single-logit head is
    the logistic one, and that is what this does. It is an assumption, flagged
    as such, and `stats.confidence_vs_accept` in the export is the evidence for
    or against it: it is the empirical acceptance rate bucketed by raw score.
    """
    return [1.0 / (1.0 + pow(2.718281828459045, -c)) for c in conf]


def expected_accepted(conf: list[float], k: int) -> float:
    """`E[accepted](k) = sum_{i<=k} prod_{j<=i} c_j` (design section 10.3)."""
    tot, run = 0.0, 1.0
    for i in range(k):
        run *= conf[i]
        tot += run
    return tot


def schedule_k(conf: list[float], union_frac: list[float], t_draft_ms: float,
               hit_rate: float, k_max: int = 5) -> tuple[int, list[dict]]:
    """Maximise `E[accepted](k) / T_cycle(k)` with section 13.4's measured terms.

        T_cycle(k) = T_nonmoe + 40 * T_moe(M) + T_nvme(k) + T_draft
        T_nonmoe   = 48.5 ms                       (section 7.15.2, M=1; flat in M by assumption)
        T_moe(M)   = interpolated 0.683 .. 1.310   (section 7.9.2, M = 1..6)
        T_nvme(k)  = (1-h) * M * union_frac[M] * 112.8 MB * 40 / 4.5 GB/s
    """
    t_nonmoe = 48.5
    moe_m1, moe_m6 = 0.683, 1.310
    rows = []
    best, best_k = -1.0, 0
    for k in range(0, k_max + 1):
        m = k + 1
        t_moe = 40.0 * (moe_m1 + (moe_m6 - moe_m1) * (m - 1) / 5.0)
        uf = union_frac[m - 1] if m - 1 < len(union_frac) else union_frac[-1]
        miss_mb = (1.0 - hit_rate) * m * uf * 112.8 * 40.0
        t_nvme = miss_mb / 4500.0 * 1000.0
        t_cycle = t_nonmoe + t_moe + t_nvme + (t_draft_ms if k > 0 else 0.0)
        ea = 1.0 + expected_accepted(conf, k)      # +1 for the corrected token
        rows.append({"k": k, "t_cycle_ms": round(t_cycle, 2),
                     "t_moe_ms": round(t_moe, 2), "t_nvme_ms": round(t_nvme, 2),
                     "E_tokens": round(ea, 4), "tok_per_s": round(1000.0 * ea / t_cycle, 3)})
        if ea / t_cycle > best:
            best, best_k = ea / t_cycle, k
    return best_k, rows


# --------------------------------------------------------------------------- #
# 7. driver
# --------------------------------------------------------------------------- #

def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return float(F.cosine_similarity(a, b, dim=0))


def run(args: argparse.Namespace) -> int:
    torch.set_grad_enabled(False)
    if args.threads:
        torch.set_num_threads(args.threads)
    t_start = time.perf_counter()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] oracle_dspark start "
          f"(torch threads {torch.get_num_threads()})", flush=True)

    inference_dir = os.path.join(args.model, "inference")
    ref = dsref.load_reference(inference_dir)
    install_batched_decode(ref)
    store = dsref.WeightStore(args.model, args.manifest)
    tokenizer = dsref.TokenizerAdapter(os.path.join(args.model, "tokenizer.json"))

    ids = tokenizer.encode(oracle.L2_PROMPT)[: args.tokens]
    n_prefill = len(ids)
    max_seq_len = max(256, n_prefill + args.steps + 32)
    margs = dsref.build_args(ref, inference_dir, max_seq_len=max_seq_len)
    assert margs.dspark_block_size == 5 and margs.n_mtp_layers == 3
    assert tuple(margs.dspark_target_layer_ids) == (37, 38, 39)
    layout_e = ref.EngramLayout.from_args(margs)
    os.makedirs(args.out, exist_ok=True)

    sys.path.insert(0, inference_dir)
    import engram as eng                                          # noqa: E402
    cached = dsref.CachedTokenMap.build(tokenizer, os.path.join(args.out, "token_map.npz"))
    orig_build = eng.build_compressed_token_map
    eng.build_compressed_token_map = lambda _t: (cached.lookup, cached.size)
    try:
        ngram = ref.NgramHashState(margs, layout_e, tokenizer)
    finally:
        eng.build_compressed_token_map = orig_build

    embed_w = store.tensor("embed.weight")
    norm_w = store.tensor("norm.weight")
    head_w = store.tensor("head.weight")
    print(f"prompt {n_prefill} tokens, head {list(head_w.shape)} {head_w.dtype}, "
          f"out -> {args.out}", flush=True)

    with ref.set_dtype(torch.bfloat16):
        embed_mod = ref.ParallelEmbedding(margs.vocab_size, margs.dim)
    embed_mod.weight.data = embed_w
    head_mod = ChunkedHead(head_w)

    writer = DSparkWriter(args.out)
    main = MainRunner(ref, store, margs, layout_e, ngram, embed_w, norm_w, head_w,
                      args.engram_threads)
    main.reset()

    # --- pass 0: prefill ---------------------------------------------------
    t0 = time.perf_counter()
    logits, main_hidden_pre, _routing, _used, _n = main.forward(ids, 0, want_all_logits=False)
    tok0 = int(logits[0].argmax().item())
    print(f"prefill done in {time.perf_counter() - t0:.1f}s, next token {tok0}", flush=True)

    # --- the draft stages, and the mtp window seeded from the prompt --------
    draft = DraftRunner(ref, store, margs, embed_mod, head_mod)
    draft.seed(main_hidden_pre, tok0)

    # --- one plain decode step at position 64 (the reference's own order) ---
    pos0 = n_prefill
    t0 = time.perf_counter()
    logits1, mh1, _r, _u, _n = main.forward([tok0], pos0, want_all_logits=True)
    tok1 = int(logits1[0].argmax().item())
    print(f"decode @pos {pos0}: {tok0} -> {tok1} ({time.perf_counter() - t0:.1f}s)", flush=True)

    # --- GOLDEN: the draft cycle at position 64 ----------------------------
    cap = DraftCapture(ref)
    for s, b in enumerate(draft.blocks):
        cap.attach(b, s)
    cap.reset(True)
    capture: dict = {}
    t0 = time.perf_counter()
    out_ids, dlogits, conf, droute = draft.draft(tok1, mh1, pos0, capture)
    t_draft_golden = time.perf_counter() - t0
    drafts0 = [int(v) for v in out_ids[0, 1:].tolist()]
    golden_conf = [float(v) for v in conf[0].tolist()]
    print(f"draft @pos {pos0}: input {tok1} -> {drafts0}  conf "
          f"{[round(float(c), 4) for c in conf[0].tolist()]} "
          f"({t_draft_golden:.1f}s)", flush=True)

    golden = golden_stage_tensors(cap, capture, margs.n_mtp_layers)
    cap.reset(False)
    cap.detach()
    cap.uninstall()
    golden["main_hidden"] = ("bf16", mh1[0])                       # [1, 15360]
    golden["draft_embed"] = ("bf16", capture["draft_embed"][0])    # [5, 4, 5120]
    golden["draft_input_ids"] = ("i32", torch.tensor(
        [tok1] + [margs.dspark_noise_token_id] * (margs.dspark_block_size - 1),
        dtype=torch.int32))
    golden["draft_ids"] = ("i32", out_ids[0].int())
    golden["confidence"] = ("f32", conf[0].float())
    golden.update(logit_record(dlogits[0], "draft_logits", args.topk))
    golden.update(logit_record(logits1[0], "verify_logits_pos64", args.topk))
    golden["main_argmax_pos64"] = ("i32", torch.tensor([tok1], dtype=torch.int32))
    rec = writer.write("golden_pos64", golden)
    print(f"  golden -> {rec['file']} ({rec['bytes'] / 1e6:.2f} MB)", flush=True)

    # --- the verify loop ---------------------------------------------------
    K = margs.dspark_block_size
    pos = pos0 + 1                    # position of `cur`
    cur = tok1
    pending = drafts0                 # drafts for positions pos+1 .. pos+K
    cur_mh = mh1                      # main_hidden at `pos - 1`... refreshed below
    produced = [tok0, tok1]
    cycles: list[dict] = []
    thin_exports = 0
    budget = args.max_seconds

    while len(produced) - 2 < args.steps and time.perf_counter() - t_start < budget:
        cyc_t0 = time.perf_counter()
        batch = [cur] + pending[:K]
        M = len(batch)
        snap = main.snapshot()
        t0 = time.perf_counter()
        vlogits, vmh, vroute, vused, _n = main.forward(batch, pos, want_all_logits=True)
        t_verify = time.perf_counter() - t0
        preds = [int(vlogits[i].argmax().item()) for i in range(M)]
        a = 0
        while a < M - 1 and preds[a] == batch[a + 1]:
            a += 1
        new_tokens = batch[1:a + 1] + [preds[a]]
        unions = prefix_unions(vroute)

        # cross-check the batched decode path against the reference's own
        # seqlen > 1 branch, once
        crosscheck = None
        if args.crosscheck and not cycles:
            crosscheck = _crosscheck_prefill(ref, main, ids, produced, batch, pos,
                                             vlogits, margs)

        # rollback + commit
        t_commit = 0.0
        commit_cos = None
        divergence = None
        if a < M - 1:
            main.restore(snap)
            t0 = time.perf_counter()
            clogits, cmh, _cr, _cu, _cn = main.forward(batch[:a + 1], pos,
                                                       want_all_logits=False)
            t_commit = time.perf_counter() - t0
            commit_cos = cos(clogits[0], vlogits[a])
            ctok = int(clogits[0].argmax().item())
            if ctok != preds[a]:
                # NOT an assert: the reference's decode path is itself batch-
                # boundary dependent. A ratio-2 KV source publishes
                # `shared_attn.index_k` only when a compression group completes
                # inside THIS forward; otherwise the layers under it index against
                # whatever the previous forward left there. An M = 6 verify batch
                # always completes a group, an M = 1 re-run at an even position
                # does not, so the two compute different top-k lists for layers
                # 2..19. See docs/p3_dspark.md §7.1. The committed state is the
                # re-run's, so its token is the one the trajectory continues with.
                cm = torch.topk(clogits[0].float(), 2).values
                vm = torch.topk(vlogits[a].float(), 2).values
                divergence = {"verify_token": preds[a], "commit_token": ctok,
                              "verify_margin": round(float(vm[0] - vm[1]), 5),
                              "commit_margin": round(float(cm[0] - cm[1]), 5)}
                print(f"  DIVERGENCE at pos {pos + a}: verify {preds[a]} "
                      f"(margin {divergence['verify_margin']}) vs commit {ctok} "
                      f"(margin {divergence['commit_margin']})", flush=True)
                preds = preds[:a] + [ctok] + preds[a + 1:]
                new_tokens = batch[1:a + 1] + [ctok]
            mh_use = cmh
        else:
            mh_use = vmh
        del snap

        # draft from the last accepted position
        t0 = time.perf_counter()
        want_cap = thin_exports < args.thin_exports and len(cycles) in (1, 3)
        thin: dict = {} if want_cap else None
        out_ids, dlogits, conf, _dr = draft.draft(preds[a], mh_use[:, -1:], pos + a, thin)
        t_draft = time.perf_counter() - t0
        next_drafts = [int(v) for v in out_ids[0, 1:].tolist()]
        confs = [float(v) for v in conf[0].tolist()]

        produced.extend(new_tokens)
        rec = {
            "cycle": len(cycles), "pos": pos, "M": M, "k": M - 1,
            "batch": batch, "preds": preds, "accepted": a,
            "tokens_out": len(new_tokens), "new_tokens": new_tokens,
            "confidence": [round(c, 6) for c in confs],
            "next_drafts": next_drafts,
            "union": unions,
            "n_used_M": {int(L): int(v) for L, v in vused.items()},
            "t_verify_s": round(t_verify, 2), "t_commit_s": round(t_commit, 2),
            "t_draft_s": round(t_draft, 2),
            "t_cycle_s": round(time.perf_counter() - cyc_t0, 2),
        }
        vtop = torch.topk(vlogits.float(), 2, dim=-1).values
        rec["verify_margins"] = [round(float(vtop[i, 0] - vtop[i, 1]), 5) for i in range(M)]
        if commit_cos is not None:
            rec["commit_logit_cos"] = round(commit_cos, 9)
        if divergence is not None:
            rec["divergence"] = divergence
        if crosscheck is not None:
            rec["crosscheck"] = crosscheck
        cycles.append(rec)
        print(f"cycle {rec['cycle']:2d} @pos {pos:3d} M={M} accepted={a} "
              f"-> {len(new_tokens)} tok  ({rec['t_cycle_s']:.1f}s: "
              f"v {t_verify:.0f}/c {t_commit:.0f}/d {t_draft:.0f})  "
              f"conf {[round(c, 3) for c in confs]}", flush=True)

        if want_cap:
            thin_t = {"main_hidden": ("bf16", mh_use[0, -1]),
                      "main_x": ("bf16", thin["main_x"][0]),
                      "draft_ids": ("i32", out_ids[0].int()),
                      "confidence": ("f32", conf[0].float())}
            thin_t.update(logit_record(dlogits[0], "draft_logits", args.topk))
            thin_t.update(logit_record(vlogits, "verify_logits", args.topk))
            thin_t["verify_batch"] = ("i32", torch.tensor(batch, dtype=torch.int32))
            thin_t["verify_preds"] = ("i32", torch.tensor(preds, dtype=torch.int32))
            r2 = writer.write(f"thin_pos{pos:04d}", thin_t)
            print(f"  thin -> {r2['file']} ({r2['bytes'] / 1e6:.2f} MB)", flush=True)
            thin_exports += 1

        with io.open(os.path.join(args.out, "cycles_partial.json"), "w", encoding="utf-8") as f:
            json.dump({"golden_confidence": golden_conf, "produced": produced,
                       "cycles": cycles}, f, ensure_ascii=False)

        pos = pos + a + 1
        cur = preds[a]
        pending = next_drafts

    # --- statistics --------------------------------------------------------
    stats = summarise(cycles, margs, t_draft_golden,
                      golden_conf)
    meta = {
        "version": DSPARK_VERSION,
        "model": "DeepSeek-V4.1-Flash",
        "generator": "tools/oracle_dspark.py",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "prompt": oracle.L2_PROMPT,
        "prompt_ids": [int(i) for i in ids],
        "prefill_len": n_prefill,
        "golden_pos": pos0,
        "golden_input_token": tok1,
        "golden_drafts": drafts0,
        "golden_confidence": golden_conf,
        "greedy_tokens": produced,
        "text": tokenizer.decode(produced),
        "top_k": args.topk,
        "config": {"dim": margs.dim, "hc_mult": margs.hc_mult,
                   "dspark_block_size": margs.dspark_block_size,
                   "dspark_noise_token_id": margs.dspark_noise_token_id,
                   "dspark_target_layer_ids": list(margs.dspark_target_layer_ids),
                   "dspark_markov_rank": margs.dspark_markov_rank,
                   "n_mtp_layers": margs.n_mtp_layers,
                   "mtp_routed_experts": margs.dspark_n_routed_experts,
                   "mtp_activated_experts": margs.dspark_n_activated_experts,
                   "window_size": margs.window_size, "vocab_size": margs.vocab_size,
                   "n_layers": margs.n_layers, "max_seq_len": max_seq_len},
        "cycles": cycles,
        "stats": stats,
        "seconds": round(time.perf_counter() - t_start, 1),
        "notes": (
            "Greedy speculative decoding built on inference/model.py's own DSpark "
            "modules behind tools/dsref.py's CPU kernel shims. The main model runs "
            "the verify batch through install_batched_decode's seqlen>1 decode path; "
            "'crosscheck' on cycle 0 compares that against the reference's own "
            "prefill branch over the same token sequence. 'commit_logit_cos' is the "
            "per-cycle check that re-running the accepted prefix alone reproduces the "
            "verify batch's logits at that position."),
    }
    path = writer.finish(meta)
    with io.open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "cycles": cycles}, f, indent=1, ensure_ascii=False)
    total = sum(r["bytes"] for r in writer.records)
    print(f"\nwrote {len(writer.records)} records, {total / 1e6:.2f} MB -> {path} "
          f"in {meta['seconds']}s", flush=True)
    print(json.dumps(stats, indent=1, ensure_ascii=False))
    store.close()
    return 0


def _crosscheck_prefill(ref, main: MainRunner, prompt_ids, produced, batch, pos,
                        vlogits, margs) -> dict:
    """Run the whole sequence as one prefill -- the reference's own seqlen > 1
    branch -- and compare the last M logits against the batched decode path.

    This is the anchor for `install_batched_decode`: the prefill branch is
    untouched reference code, and a prompt shorter than the 128-slot window
    makes the two paths semantically identical (every query sees its whole
    causal history; the compressor pools the same groups).
    """
    # `produced` ends with batch[0] (the last confirmed token), so the whole
    # causal sequence the verify batch implies is prompt + produced + drafts.
    seq = list(prompt_ids) + list(produced) + batch[1:]
    assert len(seq) == pos + len(batch), (len(seq), pos, len(batch))
    print(f"  crosscheck: prefilling {len(seq)} tokens through the reference's "
          f"seqlen>1 branch ...", flush=True)
    t0 = time.perf_counter()
    saved = main.snapshot()
    sa = ref.shared_attn
    saved_shared = (sa.compress_kv, sa.index_k, sa.topk_idxs, sa.candidates)
    main.reset()
    try:
        logits, _mh, _r, _u, _n = main.forward(seq, 0, want_all_logits=True)
        M = len(batch)
        tail = logits[-M:]
        out = {"seconds": round(time.perf_counter() - t0, 1),
               "n_tokens": len(seq),
               "cos": [round(cos(tail[i], vlogits[i]), 9) for i in range(M)],
               "argmax_match": [int(tail[i].argmax()) == int(vlogits[i].argmax())
                                for i in range(M)],
               "prefill_argmax": [int(tail[i].argmax()) for i in range(M)]}
    finally:
        main.restore(saved)
        sa.compress_kv, sa.index_k, sa.topk_idxs, sa.candidates = saved_shared
    print(f"  crosscheck: cos {out['cos']} match {out['argmax_match']} "
          f"({out['seconds']}s)", flush=True)
    return out


def summarise(cycles: list[dict], margs, t_draft_golden: float,
              golden_conf: list[float] | None = None) -> dict:
    if not cycles:
        return {}
    K = margs.dspark_block_size
    acc = [c["accepted"] for c in cycles]
    dist = {i: acc.count(i) for i in range(K + 1)}
    n = len(cycles)
    tokens = sum(c["tokens_out"] for c in cycles)
    # union_frac[m] = union(m) / (m * topk), averaged over layers and cycles
    topk = margs.n_activated_experts
    per_m = [[] for _ in range(K + 1)]
    per_layer = {}
    for c in cycles:
        for L, sizes in c["union"].items():
            L = int(L)
            per_layer.setdefault(L, [[] for _ in range(K + 1)])
            for m, s in enumerate(sizes):
                per_m[m].append(s / ((m + 1) * topk))
                per_layer[L][m].append(s / ((m + 1) * topk))
    union_frac = [round(sum(v) / len(v), 5) if v else None for v in per_m]
    abs_m = [[] for _ in range(K + 1)]
    for c in cycles:
        for L, sizes in c["union"].items():
            for m, s in enumerate(sizes):
                abs_m[m].append(s)
    union_abs = [round(sum(v) / len(v), 3) if v else None for v in abs_m]
    layer_frac = {L: [round(sum(v) / len(v), 5) if v else None for v in rows]
                  for L, rows in sorted(per_layer.items())}
    # E[min(a, k)] from the observed distribution -- the acceptance curve for any k
    e_acc = {}
    for k in range(K + 1):
        e_acc[k] = round(sum(min(a_, k) for a_ in acc) / n, 4)

    # The shadow scheduler of section 10.3: what k it would have picked each
    # cycle from that cycle's confidence head, with the measured union_frac and
    # the section 13.4 time model. It does not steer the run -- the trajectory is
    # greedy and therefore identical for any k -- so this is a free read-out.
    uf = [v if v is not None else 1.0 for v in union_frac]
    t_draft_ms = 11.0                       # design section 7.12's estimate
    picks, sched_rows = [], None
    # ALIGNMENT: a cycle record's `confidence` belongs to its `next_drafts`,
    # i.e. to the drafts verified by the NEXT cycle. The drafts cycle 0 verifies
    # came from the golden draft at position 64.
    conf_of = [golden_conf] + [c["confidence"] for c in cycles[:-1]]
    for ci, c in enumerate(cycles):
        if conf_of[ci] is None:
            continue
        k, rows = schedule_k(conf_to_prob(conf_of[ci]), uf, t_draft_ms, 0.920, K)
        picks.append(k)
        if sched_rows is None:
            sched_rows = rows

    # Is the confidence head predictive at all? For each draft position i, the
    # raw score paired with whether that position was in fact accepted.
    conf_vs_accept = []
    for i in range(K):
        # position i is only tested if 0..i-1 were accepted (conditional rate)
        pairs = [(conf_of[ci][i], 1 if c["accepted"] > i else 0)
                 for ci, c in enumerate(cycles)
                 if conf_of[ci] is not None and c["k"] > i and c["accepted"] >= i]
        if not pairs:
            continue
        acc_rate = sum(p[1] for p in pairs) / len(pairs)
        mean_hit = [p[0] for p in pairs if p[1]] or float("nan")
        mean_miss = [p[0] for p in pairs if not p[1]] or float("nan")
        conf_vs_accept.append({
            "pos": i, "n": len(pairs),
            "accept_rate": round(acc_rate, 4),
            "mean_score": round(sum(p[0] for p in pairs) / len(pairs), 5),
            # None, not NaN: NaN is not JSON and core/json.h (rightly) rejects it
            "mean_score_accepted": (round(sum(mean_hit) / len(mean_hit), 5)
                                    if mean_hit == mean_hit else None),
            "mean_score_rejected": (round(sum(mean_miss) / len(mean_miss), 5)
                                    if mean_miss == mean_miss else None),
            "mean_sigmoid": round(sum(conf_to_prob([p[0]])[0] for p in pairs) / len(pairs), 5),
        })
    return {
        "cycles": n,
        "tokens": tokens,
        "tokens_per_verify": round(tokens / n, 4),
        "mean_accepted": round(sum(acc) / n, 4),
        "accept_dist": dist,
        "accept_dist_frac": {i: round(dist[i] / n, 4) for i in dist},
        "E_accepted_vs_k": e_acc,
        "E_tokens_vs_k": {k: round(v + 1.0, 4) for k, v in e_acc.items()},
        "union_frac_by_M": union_frac,
        "union_experts_by_M_per_layer": union_abs,
        "union_frac_by_layer": layer_frac,
        "scheduler_k_picks": picks,
        "scheduler_k_hist": {k: picks.count(k) for k in range(K + 1)},
        "scheduler_curve_cycle0": sched_rows,
        "confidence_vs_accept": conf_vs_accept,
        "mean_confidence_by_pos": [
            round(sum(cc[i] for cc in conf_of if cc is not None)
                  / max(1, sum(1 for cc in conf_of if cc is not None)), 5) for i in range(K)],
        "t_draft_golden_s": round(t_draft_golden, 2),
        "mean_t_verify_s": round(sum(c["t_verify_s"] for c in cycles) / n, 2),
        "mean_t_draft_s": round(sum(c["t_draft_s"] for c in cycles) / n, 2),
    }


# --------------------------------------------------------------------------- #
# 8. Track K2: tree-sampled speculation on five normal prompts
# --------------------------------------------------------------------------- #
#
# Track K's loop above verified a single greedy chain. The owner's scheme is
# different (tools/dspark_tree.py's docstring): the draft forward is a matrix,
# the CPU samples one path from a top-K lattice over it, and the verify matrix is
# compared through its top-K. This driver generates the data every scheme needs:
#
#   * one trajectory per (prompt, mode), mode in {greedy, sampling}, driven by
#     the real speculative loop with tree K = 16 (greedy: `eal` path; sampling:
#     ancestral path + lossless top-32 acceptance), k = 5, M = 6;
#   * a draft event at EVERY accepted position (the reference's M = 1 order:
#     forward_spec(token at p+1, main_hidden[p], start_pos = p)), saving the base
#     logits B [5, V] and the confidence head's hidden x [5, 5120];
#   * the verify rows' top-256 / logsumexp / argmax for every trajectory position.
#
# From these, `dspark_tree.py analyse` evaluates every other scheme offline:
# greedy acceptance is "path prefix == the greedy trajectory", and sampling
# acceptance uses the exact coupling P(accept_j | output y_j) = min(1, q(y_j)/p(y_j))
# along the sampled trajectory (docs/p3_dspark.md section 3.3).
#
# No rerun on rejection: the state the verify forward leaves at rejected positions
# is overwritten before it can be read (window slots by the next batch's writes,
# ratio-2 group slots before their group pools, compressed rows before their group
# is visible), which is exactly what a runtime does.

TREE_SOURCE_REV = "0170339"
TREE_PROMPTS = (
    ("en_prose", "model:README.md", "**Architecture.** DeepSeek-V4.1-Flash adopts"),
    ("zh_prose", f"git:{TREE_SOURCE_REV}:docs/design.md", "deepMoE 是一个针对 Windows Strix Halo"),
    ("python", f"git:{TREE_SOURCE_REV}:tools/dsref.py", "        import queue\n"),
    ("cpp", f"git:{TREE_SOURCE_REV}:cpu/gate.cpp", "Result<GateResult> gate_topk("),
    ("markdown", f"git:{TREE_SOURCE_REV}:docs/p2_decode.md", "1. **The MoE's 30 ms of host work"),
)
TREE_BOS = 0
TREE_K = 16
TREE_VERIFY_TOPK = 32           # Kv: the target distribution of sampling mode
TREE_TRACE_TOPK = 256


def _tree_source_text(model_dir: str, source: str) -> str:
    import subprocess
    kind, _, rest = source.partition(":")
    if kind == "model":
        with open(os.path.join(model_dir, rest), encoding="utf-8") as f:
            return f.read()
    rev, _, path = rest.partition(":")
    repo = os.path.dirname(_HERE)
    out = subprocess.run(["git", "-C", repo, "show", f"{rev}:{path}"], capture_output=True)
    if out.returncode != 0:
        raise SystemExit(f"git show {rev}:{path} failed")
    return out.stdout.decode("utf-8")


def build_tree_prompts(model_dir: str, tokenizer, n_tokens: int) -> list[dict]:
    out = []
    for name, source, locator in TREE_PROMPTS:
        text = _tree_source_text(model_dir, source)
        at = text.find(locator)
        if at < 0:
            raise SystemExit(f"{name}: locator not found in {source}")
        body = tokenizer.encode(text[at: at + 4000])[:n_tokens]
        ids = [TREE_BOS] + body
        out.append({"name": name, "source": source, "locator": locator,
                    "ids": [int(i) for i in ids], "text": tokenizer.decode(body)})
    return out


class TreeDraft:
    """`DraftRunner` plus the two tensors the matrix view needs: the head's output
    before the Markov loop adds anything to it (the base logits B), and the
    confidence head's hidden x (`hc_pre` output = the final norm's input)."""

    def __init__(self, draft: DraftRunner, head_mod: ChunkedHead):
        self.draft = draft
        self._grab: dict = {}
        last = draft.blocks[-1]
        # temperature 0 so the reference's own output_ids are the greedy chain
        for b in draft.blocks:
            b.temperature = 0.0
        head_mod.register_forward_hook(
            lambda _m, _i, o: self._grab.__setitem__("B", o.detach().float().clone()))
        last.norm.register_forward_hook(
            lambda _m, i, _o: self._grab.__setitem__("x", i[0].detach().float().clone()))
        mh = last.markov_head
        self.E = mh.embed.weight.detach().float().numpy().copy()          # [V, 256]
        self.H = mh.head.weight.detach().float().numpy().copy()           # [V, 256]
        proj = last.confidence_head.proj
        assert proj.bias is None
        self.W = proj.weight.detach().float().numpy()[0].copy()           # [5376]

    def run(self, input_token: int, main_hidden: torch.Tensor, start_pos: int) -> dict:
        self._grab.clear()
        t0 = time.perf_counter()
        out_ids, _logits, conf, _r = self.draft.draft(input_token, main_hidden, start_pos)
        B = self._grab["B"][0]                                            # [5, V] fp32
        x = self._grab["x"][0]                                            # [5, 5120]
        lse = torch.logsumexp(B.double(), dim=-1).float()
        return {"start_pos": int(start_pos), "input_token": int(input_token),
                "ref_chain": [int(v) for v in out_ids[0, 1:].tolist()],
                "ref_conf": [float(v) for v in conf[0].tolist()],
                "B": B.numpy(), "x": x.numpy(), "lse_base": lse.numpy(),
                "t_s": time.perf_counter() - t0}


def _full_snapshot(ref, main: MainRunner, draft: DraftRunner, ngram) -> dict:
    sa = ref.shared_attn
    return {"kv": main.snapshot(),
            "shared": {k: (getattr(sa, k).clone() if getattr(sa, k) is not None else None)
                       for k in ("compress_kv", "index_k", "topk_idxs", "candidates")},
            "mtp": [b.attn.window_kv_cache.clone() for b in draft.blocks],
            "ngram": ngram.cache.clone()}


def _full_restore(ref, main: MainRunner, draft: DraftRunner, ngram, snap: dict) -> None:
    main.restore(snap["kv"])
    sa = ref.shared_attn
    for k, v in snap["shared"].items():
        setattr(sa, k, v.clone() if v is not None else None)
    for b, w in zip(draft.blocks, snap["mtp"]):
        b.attn.window_kv_cache.copy_(w)
    ngram.cache.copy_(snap["ngram"])


def _atomic_save(obj, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_json(obj, path: str) -> None:
    tmp = path + ".tmp"
    with io.open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def forward_emulated(ref, main: MainRunner, tokens: list[int], pos: int):
    """A verify forward whose indexer reproduces one-token-at-a-time decoding."""
    ref._dm_emulate_decode = [ref.shared_attn.index_k] * len(tokens)
    try:
        return main.forward(tokens, pos, want_all_logits=True)
    finally:
        ref._dm_emulate_decode = None


def verify_rows(logits: torch.Tensor, batch: list[int]) -> dict:
    lg = logits.float()
    vals, idx = torch.topk(lg, TREE_TRACE_TOPK, dim=-1)
    lse = torch.logsumexp(lg.double(), dim=-1)
    tok = [float(lg[j, batch[j + 1]]) for j in range(len(batch) - 1)]
    return {"top_ids": idx.int().numpy(), "top_logits": vals.numpy(),
            "lse": lse.numpy(), "argmax": idx[:, 0].int().numpy(),
            "tok_logit": np.asarray(tok, dtype=np.float32)}


def _save_draft(d: dict, path: str) -> None:
    meta = {k: v for k, v in d.items() if k not in ("B", "x", "lse_base")}
    np.savez(path, B=d["B"], x=d["x"], lse_base=d["lse_base"],
             meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8))


def run_tree(args: argparse.Namespace) -> int:
    import dspark_tree as dt
    import oracle_longctx as olc

    log = olc.log
    # the venv launcher is a second process with our own command line; do not wait on it
    _scan = olc.other_processes

    def other_processes():
        heavy, gpu = _scan()
        return [h for h in heavy if "oracle_dspark" not in h], gpu
    olc.other_processes = other_processes
    tdir = args.tree_traces
    os.makedirs(tdir, exist_ok=True)
    run_log_path = os.path.join(tdir, "run_log.json")
    run_log = {"runs": []}
    if os.path.exists(run_log_path):
        with io.open(run_log_path, encoding="utf-8") as f:
            run_log = json.load(f)
    env = olc.wait_quiet(args.need_gb, args.poll_s, args.max_wait_h)
    this_run = {"start": olc.now(), "env_start": env, "phases": []}
    run_log["runs"].append(this_run)
    _atomic_json(run_log, run_log_path)
    t_start = time.perf_counter()

    torch.set_grad_enabled(False)
    if args.threads:
        torch.set_num_threads(args.threads)
    inference_dir = os.path.join(args.model, "inference")
    ref = dsref.load_reference(inference_dir)
    ref.ParallelEngramEmbedding = olc._NoEngramTable        # the 98 GB commit placeholder
    install_batched_decode(ref)
    store = dsref.WeightStore(args.model, args.manifest)
    tokenizer = dsref.TokenizerAdapter(os.path.join(args.model, "tokenizer.json"))
    prompts = build_tree_prompts(args.model, tokenizer, args.tree_tokens)
    _atomic_json(prompts, os.path.join(tdir, "prompts.json"))
    margs = dsref.build_args(ref, inference_dir, max_seq_len=256)
    assert margs.dspark_block_size == 5 and margs.n_mtp_layers == 3
    layout_e = ref.EngramLayout.from_args(margs)
    sys.path.insert(0, inference_dir)
    import engram as eng                                          # noqa: E402
    cached = dsref.CachedTokenMap.build(tokenizer, os.path.join(tdir, "token_map.npz"))
    orig_build = eng.build_compressed_token_map
    eng.build_compressed_token_map = lambda _t: (cached.lookup, cached.size)
    try:
        ngram = ref.NgramHashState(margs, layout_e, tokenizer)
    finally:
        eng.build_compressed_token_map = orig_build

    embed_w = store.tensor("embed.weight")
    norm_w = store.tensor("norm.weight")
    head_w = store.tensor("head.weight")
    with ref.set_dtype(torch.bfloat16):
        embed_mod = ref.ParallelEmbedding(margs.vocab_size, margs.dim)
    embed_mod.weight.data = embed_w
    head_mod = ChunkedHead(head_w)
    main = MainRunner(ref, store, margs, layout_e, ngram, embed_w, norm_w, head_w,
                      args.engram_threads, verbose=False)
    draft = DraftRunner(ref, store, margs, embed_mod, head_mod)
    td = TreeDraft(draft, head_mod)
    E, H = td.E, td.H
    K = TREE_K

    def out_of_time() -> bool:
        return time.perf_counter() - t_start > args.max_seconds

    def lattice_for(dd: dict, with_tail: bool):
        idx, cl, lse, e_in, e_cand, h_cand = dt.lattice_inputs(dd["B"], dd["input_token"], E, H, K)
        return dt.Lattice(idx, cl, e_in, e_cand, h_cand, lse if with_tail else None)

    def run_mode(pi: int, P: dict, mode: str) -> None:
        pdir = os.path.join(tdir, P["name"], mode)
        os.makedirs(pdir, exist_ok=True)
        lpath = os.path.join(pdir, "log.json")
        ckpt = os.path.join(pdir, "ckpt.pt")
        state = None
        if os.path.exists(lpath):
            with io.open(lpath, encoding="utf-8") as f:
                state = json.load(f)
            if state.get("done") or len(state["cycles"]) >= args.tree_cycles:
                log(f"{P['name']}/{mode}: already has {len(state['cycles'])} cycles")
                return
        rng = np.random.default_rng(1000 + 17 * pi + (1 if mode == "sampling" else 0))
        tgen = torch.Generator().manual_seed(5000 + 17 * pi)
        n = len(P["ids"])
        if state is not None and os.path.exists(ckpt):
            blob = torch.load(ckpt, weights_only=False)
            _full_restore(ref, main, draft, ngram, blob["snap"])
            rng.bit_generator.state = blob["rng"]
            log(f"{P['name']}/{mode}: resumed at cycle {len(state['cycles'])}")
        else:
            ppath = os.path.join(tdir, P["name"], "prefill.pt")
            if os.path.exists(ppath):
                blob = torch.load(ppath, weights_only=False)
                _full_restore(ref, main, draft, ngram, blob["snap"])
                tok0, mh_last, (ids32, lg32) = blob["tok0"], blob["mh_last"], blob["rows"]
                log(f"{P['name']}: prefill restored")
            else:
                main.reset()
                t0 = time.perf_counter()
                logits, mh_pre, _r, _u, _n = main.forward(P["ids"], 0, want_all_logits=False)
                tok0 = int(logits[0].argmax().item())
                v32, i32 = torch.topk(logits[0].float(), TREE_VERIFY_TOPK)
                ids32, lg32 = i32.int().numpy(), v32.numpy()
                draft.seed(mh_pre, tok0)
                mh_last = mh_pre[:, -1:].clone()
                t_pre = time.perf_counter() - t0
                _atomic_save({"snap": _full_snapshot(ref, main, draft, ngram), "tok0": tok0,
                              "mh_last": mh_last, "rows": (ids32, lg32),
                              "t_prefill_s": t_pre}, ppath)
                log(f"{P['name']}: prefill {n} tokens in {t_pre:.0f}s, argmax {tok0}")
            if mode == "sampling":
                # position n itself is a plain top-32 sample from the prefill's last row
                l64 = lg32.astype(np.float64)
                pv = dt.dm_exp(l64 - dt.lse_seq(l64))
                tok0 = int(ids32[dt._sample_index(pv, float(rng.random()))])
            state = {"prompt": P, "mode": mode, "K": K, "k": 5, "prefill_len": n,
                     "first_token": tok0, "produced": [tok0], "cycles": [], "drafts": []}
            dd = td.run(tok0, mh_last, n - 1)
            _save_draft(dd, os.path.join(pdir, "d0000.npz"))
            state["drafts"].append({k: v for k, v in dd.items() if k not in ("B", "x", "lse_base")})
            state["pending_draft"] = 0
            state["pos"] = n
            state["cur"] = tok0

        while len(state["cycles"]) < args.tree_cycles and not out_of_time():
            cyc_t0 = time.perf_counter()
            pos, cur = state["pos"], state["cur"]
            dpath = os.path.join(pdir, f"d{state['pending_draft']:04d}.npz")
            with np.load(dpath) as z:
                dd = {"B": z["B"], "x": z["x"], "lse_base": z["lse_base"],
                      **json.loads(bytes(z["meta"]).decode())}
            t1 = time.perf_counter()
            if mode == "greedy":
                lat = lattice_for(dd, with_tail=True)
                path = lat.path("eal")
                u_path = u_acc = u_res = None
            else:
                idx32, cl32, lse32, e_in, e_cand32, h_cand32 = dt.lattice_inputs(
                    dd["B"], dd["input_token"], E, H, 32)
                lat = dt.Lattice(idx32[:, :K], cl32[:, :K], e_in, e_cand32[:, :K],
                                 h_cand32[:, :K], None)
                u_path, u_acc, u_res = rng.random(5), rng.random(5), rng.random(6)
                path = lat.sample(u_path)
            t_tree = time.perf_counter() - t1
            ptoks = lat.tokens(path)
            conf = dt.confidence(dd["x"], lat.prev_embed(path), td.W)
            batch = [cur] + ptoks
            snap_path = None
            if mode == "greedy" and len(state["cycles"]) < args.bb_cycles:
                snap_path = os.path.join(pdir, f"bb_snap{len(state['cycles']):03d}.pt")
                _atomic_save(_full_snapshot(ref, main, draft, ngram), snap_path)
            t0 = time.perf_counter()
            vlogits, vmh, vroute, _vu, _n = main.forward(batch, pos, want_all_logits=True)
            t_verify = time.perf_counter() - t0
            rows = verify_rows(vlogits, batch)
            t1 = time.perf_counter()
            if mode == "greedy":
                a, emitted = dt.accept_greedy(ptoks, [int(v) for v in rows["argmax"]], 5)
            else:
                # exactly lossless against plain temperature-1 sampling: candidate
                # logits, row logsumexp, a C_j-masked sample and a full-row sample
                lg = vlogits.float()
                gum = -torch.log(torch.empty_like(lg, dtype=torch.float64).exponential_(
                    generator=tgen)).float()
                cand_rows = torch.stack([lg[j, torch.from_numpy(idx32[j]).long()]
                                         for j in range(5)]).numpy()
                lse_rows = torch.logsumexp(lg.double(), dim=-1).float().numpy()
                masked = {}
                for Kx in dt.KS:
                    ms = []
                    for j in range(5):
                        z = lg[j] + gum[j]
                        z[torch.from_numpy(idx32[j][:Kx]).long()] = -float("inf")
                        ms.append(int(z.argmax()))
                    masked[Kx] = ms
                full = [int((lg[j] + gum[j]).argmax()) for j in range(len(batch))]
                a, emitted = dt.accept_sampling_exact(lat, path, 5, cand_rows, lse_rows,
                                                      np.asarray(masked[K]), np.asarray(full),
                                                      u_acc, u_res)
                rows["cand_logit32"] = cand_rows
                rows["masked_samples"] = np.asarray([masked[Kx] for Kx in dt.KS], dtype=np.int32)
                rows["full_samples"] = np.asarray(full, dtype=np.int32)
                rows["emit_logit"] = np.asarray([float(lg[j, emitted[j]]) for j in range(a + 1)],
                                                dtype=np.float32)
            t_accept = time.perf_counter() - t1
            ci = len(state["cycles"])
            np.savez(os.path.join(pdir, f"v{ci:03d}.npz"), **rows)
            # drafts at every accepted position pos + j, j = 0..a
            new_drafts = []
            t_d = 0.0
            for j in range(a + 1):
                dj = td.run(emitted[j], vmh[:, j:j + 1], pos + j)
                t_d += dj["t_s"]
                di = len(state["drafts"])
                _save_draft(dj, os.path.join(pdir, f"d{di:04d}.npz"))
                state["drafts"].append({k: v for k, v in dj.items()
                                        if k not in ("B", "x", "lse_base")})
                new_drafts.append(di)
            rec = {"cycle": ci, "pos": pos, "M": len(batch), "batch": batch,
                   "draft": state["pending_draft"], "path": path, "path_tokens": ptoks,
                   "path_conf": [float(c) for c in conf], "accepted": int(a),
                   "emitted": [int(t) for t in emitted],
                   "u_path": None if u_path is None else u_path.tolist(),
                   "u_acc": None if u_acc is None else u_acc.tolist(),
                   "u_res": None if u_res is None else u_res.tolist(),
                   "union": prefix_unions(vroute),
                   "verify_margins": [float(rows["top_logits"][j, 0] - rows["top_logits"][j, 1])
                                      for j in range(len(batch))],
                   "new_drafts": new_drafts, "bb_snapshot": snap_path is not None,
                   "t_verify_s": round(t_verify, 2), "t_drafts_s": round(t_d, 2),
                   "t_tree_py_ms": round(t_tree * 1e3, 3), "t_accept_py_ms": round(t_accept * 1e3, 3),
                   "t_cycle_s": round(time.perf_counter() - cyc_t0, 2), "time": olc.now()}
            state["cycles"].append(rec)
            state["produced"].extend(int(t) for t in emitted)
            state["pos"] = pos + a + 1
            state["cur"] = int(emitted[a])
            state["pending_draft"] = new_drafts[-1]
            state["text"] = tokenizer.decode(state["produced"])
            _atomic_save({"snap": _full_snapshot(ref, main, draft, ngram),
                          "rng": rng.bit_generator.state}, ckpt)
            _atomic_json(state, lpath)
            log(f"{P['name']}/{mode} cycle {ci} @pos {pos}: path {ptoks} a={a} -> {emitted} "
                f"(verify {t_verify:.0f}s, drafts {t_d:.0f}s, tree {t_tree * 1e3:.1f} ms)")
        if len(state["cycles"]) >= args.tree_cycles:
            state["done"] = True
            _atomic_json(state, lpath)

    modes = [m for m in args.tree_modes.split(",") if m]
    for mode in modes:
        this_run["phases"].append({"phase": mode, "start": olc.now()})
        _atomic_json(run_log, run_log_path)
        for pi, P in enumerate(prompts):
            if out_of_time():
                break
            run_mode(pi, P, mode)
        this_run["phases"][-1]["end"] = olc.now()
        _atomic_json(run_log, run_log_path)

    # --- phase 3: batch-boundary isolation from the saved snapshots ------------
    if args.bb_cycles > 0 and not out_of_time():
        this_run["phases"].append({"phase": "batch_boundary", "start": olc.now()})
        _atomic_json(run_log, run_log_path)
        for pi, P in enumerate(prompts):
            pdir = os.path.join(tdir, P["name"], "greedy")
            lpath = os.path.join(pdir, "log.json")
            if not os.path.exists(lpath):
                continue
            with io.open(lpath, encoding="utf-8") as f:
                st = json.load(f)
            for rec in st["cycles"]:
                if out_of_time():
                    break
                sp = os.path.join(pdir, f"bb_snap{rec['cycle']:03d}.pt")
                ep = os.path.join(pdir, f"e{rec['cycle']:03d}.npz")
                if not os.path.exists(sp) or os.path.exists(ep):
                    continue
                _full_restore(ref, main, draft, ngram, torch.load(sp, weights_only=False))
                t0 = time.perf_counter()
                elog, _mh, _r, _u, _n = forward_emulated(ref, main, rec["batch"], rec["pos"])
                erows = verify_rows(elog, rec["batch"])
                extra = {}
                if pi == 0 and rec["cycle"] == 0 and args.seq_check:
                    _full_restore(ref, main, draft, ngram, torch.load(sp, weights_only=False))
                    seq = []
                    for j, tok in enumerate(rec["batch"]):
                        lg1, _m1, _r1, _u1, _n1 = main.forward([tok], rec["pos"] + j, True)
                        seq.append(lg1[0].float())
                    srows = verify_rows(torch.stack(seq), rec["batch"])
                    extra = {f"seq_{k}": v for k, v in srows.items()}
                    # cosines need full rows, so the plain verify runs once more; its
                    # top-256 must equal the trajectory's v file (determinism check)
                    _full_restore(ref, main, draft, ngram, torch.load(sp, weights_only=False))
                    plog, _m2, _r2, _u2, _n2 = main.forward(rec["batch"], rec["pos"], True)
                    prow = verify_rows(plog, rec["batch"])
                    with np.load(os.path.join(pdir, f"v{rec['cycle']:03d}.npz")) as z:
                        extra["plain_rerun_identical"] = np.asarray(
                            [bool(np.array_equal(z["top_logits"], prow["top_logits"]))])
                    extra["cos_plain_seq"] = np.asarray(
                        [cos(plog[j], seq[j]) for j in range(len(seq))])
                    extra["cos_emul_seq"] = np.asarray(
                        [cos(elog[j], seq[j]) for j in range(len(seq))])
                    extra["cos_plain_emul"] = np.asarray(
                        [cos(plog[j], elog[j]) for j in range(len(seq))])
                np.savez(ep, **{f"emul_{k}": v for k, v in erows.items()}, **extra)
                log(f"batch-boundary {P['name']} cycle {rec['cycle']}: emulated verify "
                    f"{time.perf_counter() - t0:.0f}s, argmax plain "
                    f"{[int(v) for v in np.load(os.path.join(pdir, 'v%03d.npz' % rec['cycle']))['argmax']]}"
                    f" emul {[int(v) for v in erows['argmax']]}"
                    + (f" cos(plain,seq) {extra['cos_plain_seq'].round(6).tolist()} "
                       f"cos(emul,seq) {extra['cos_emul_seq'].round(6).tolist()}" if extra else ""))
        this_run["phases"][-1]["end"] = olc.now()

    this_run["end"] = olc.now()
    this_run["seconds"] = round(time.perf_counter() - t_start, 1)
    _atomic_json(run_log, run_log_path)
    store.close()
    log(f"tree run finished in {this_run['seconds']}s")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="oracle_dspark.py",
        description="DSpark draft oracle + speculative verify loop (design sections 7.12 / 10)")
    p.add_argument("--model", required=True, help="model directory (read only)")
    p.add_argument("--manifest", default=None)
    p.add_argument("--out", default="tests/data/dspark")
    p.add_argument("--tokens", type=int, default=64, help="prompt tokens (L3 uses 64)")
    p.add_argument("--steps", type=int, default=32, help="accepted tokens to generate")
    p.add_argument("--topk", type=int, default=64, help="logit rows to export")
    p.add_argument("--thin-exports", type=int, default=2)
    p.add_argument("--engram-threads", type=int, default=32)
    p.add_argument("--threads", type=int, default=0, help="torch intra-op threads (0 = default)")
    p.add_argument("--max-seconds", type=float, default=6600.0)
    p.add_argument("--resummarise", action="store_true",
                   help="recompute stats from an existing --out index.json (no model run)")
    p.add_argument("--crosscheck", action="store_true",
                   help="validate install_batched_decode against a full prefill (one cycle)")
    p.add_argument("--tree", action="store_true",
                   help="Track K2: tree-sampled trajectories on five normal prompts")
    p.add_argument("--tree-traces", default=os.path.join(os.path.dirname(_HERE), "traces",
                                                         "dspark_tree"))
    p.add_argument("--tree-tokens", type=int, default=56, help="prompt tokens after BOS")
    p.add_argument("--tree-cycles", type=int, default=6, help="verify cycles per prompt and mode")
    p.add_argument("--tree-modes", default="greedy,sampling")
    p.add_argument("--bb-cycles", type=int, default=2,
                   help="greedy cycles per prompt re-verified with decode-emulated indexing")
    p.add_argument("--seq-check", type=int, default=1,
                   help="also decode prompt 0's first verify batch one token at a time")
    p.add_argument("--need-gb", type=float, default=8.0)
    p.add_argument("--poll-s", type=int, default=120)
    p.add_argument("--max-wait-h", type=float, default=6.0)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.resummarise:
        return resummarise(args)
    if args.tree:
        return run_tree(args)
    return run(args)


def resummarise(args) -> int:
    """Re-derive `stats` from the cycles an earlier run recorded, and rewrite
    index.json / stats.json in place. Used when the statistics change but the
    ~40 minutes of model time behind them do not need to be spent again."""
    import types
    path = os.path.join(args.out, "index.json")
    with io.open(path, encoding="utf-8") as f:
        meta = json.load(f)
    cfg = meta["config"]
    margs = types.SimpleNamespace(dspark_block_size=cfg["dspark_block_size"],
                                  n_activated_experts=6)
    gconf = meta.get("golden_confidence")
    old = meta.get("stats", {})
    stats = summarise(meta["cycles"], margs, old.get("t_draft_golden_s", 0.0), gconf)
    meta["stats"] = stats
    if "records" in meta and "steps" not in meta:
        meta["steps"] = meta.pop("records")
        for i, r in enumerate(meta["steps"]):
            r.setdefault("layer", i)
            r.setdefault("step", r.pop("record", f"r{i}"))
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    with io.open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "cycles": meta["cycles"]}, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in stats.items() if k != "union_frac_by_layer"},
                     indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
