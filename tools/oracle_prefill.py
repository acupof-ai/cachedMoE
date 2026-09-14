#!/usr/bin/env python3
"""Per-stage golden tensors of the `start_pos == 0` forward -- the PREFILL
branch of every module -- for docs/p3_prefill.md (Track L).

What exists already and why it is not enough
--------------------------------------------
`tools/oracle.py --level l2` exports the DECODE step at position 64: one query,
a 128-slot ring, the compressed caches the prefill left. `--level l3` exports
the prefill's END STATE (window rings, compressed caches, index keys) and the
logits. Neither has a single tensor of the prefill computation itself: the
64-query attention with its causal band, the compressor pooling all 32 groups
at once, the indexer's per-query visibility mask, or 64 tokens routed through
the gate in one batch. Those are what the batched prefill kernels compute, so
this script runs `inference/model.py`'s prefill branch -- unmodified, behind
`tools/dsref.py`'s CPU kernel shims, exactly as the L2/L3 exporters do -- and
captures them.

It imports `tools/oracle.py` for its capture hooks and its container writer and
never edits it. The output loads with `tests/l2_golden.h`.

Keeping it small
----------------
A 64-token prefill of one layer holds ~10 MB of interesting tensors. Seven
layers of that would be 70 MB, so the export is shaped by what a test can use:

  full, all 64 positions   the tensors an ATTENTION or MoE kernel needs as input
                           on the layers where that kernel is validated
                           (`attn_norm_out` on 0/2/20, `ffn_norm_out` on
                           0/20/39), plus everything small: the KV row per
                           position, the mHC coefficients, the gate's ids and
                           weights, the whole top-k index matrix, the
                           compressed latents / cache / index keys, and the
                           recomputed index scores.
  a few positions          everything wide -- the residual stream in and out,
                           q, the attention output, wo_a / wo_b, moe_out -- at
                           positions {0, 63} (q and the attention output at
                           {31, 63}; position 0's attention is its own KV row).
  a sketch per position    for those wide tensors, (L2 norm, sum, signed sum)
                           of EVERY position's row, so a kernel's output can be
                           checked at all 64 rows without storing them. The
                           signed sum uses `sketch_sign(i)` below, which
                           tests/test_gpu_prefill.cpp reimplements bit for bit.

and a `routing` record with every one of the forty layers' top-6 ids and
weights for all 64 tokens (126 KB), which is what "the routing sets are equal
for every token at every layer" is checked against.

`index_score` is recomputed with torch from the reference's own captured
post-fp4 queries, its live `shared_attn.index_k` and its `weights_proj` output,
in `Indexer.forward`'s op order, and then masked exactly as the prefill branch
masks it -- the same caveat `tools/oracle_l2_extra.py` records (a misread
formula would be reproduced on both sides).

Usage (about the cost of `oracle.py --level l3`'s prefill pass, ~5 minutes):
    .venv/Scripts/python.exe tools/oracle_prefill.py \
        --model D:/models/DeepSeek-V4.1-Flash --out tests/data/prefill
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oracle  # noqa: E402

EXPORT_LAYERS = (0, 1, 2, 13, 14, 20, 39)
ATTN_FULL_LAYERS = (0, 2, 20)      # attn_norm_out for every position
FFN_FULL_LAYERS = (0, 20, 39)      # ffn_norm_out for every position
ROW_POSITIONS = (0, 63)            # residual stream, projections
ATTN_POSITIONS = (31, 63)          # q and the attention output
N_TOKENS = 64


def sketch_sign(n: int) -> torch.Tensor:
    """+1/-1 per element index: bit 16 of a Knuth multiplicative hash.

    tests/test_gpu_prefill.cpp computes `((i * 2654435761u + 12345u) >> 16) & 1`
    in uint32 arithmetic; so does this.
    """
    i = np.arange(n, dtype=np.uint64)
    u = (i * np.uint64(2654435761) + np.uint64(12345)) & np.uint64(0xFFFFFFFF)
    bit = (u >> np.uint64(16)) & np.uint64(1)
    return torch.from_numpy(np.where(bit == 1, -1.0, 1.0).astype(np.float64))


def sketch(rows: torch.Tensor) -> torch.Tensor:
    """[n, ...] -> [n, 3] of (L2 norm, sum, signed sum), in float64 then f32."""
    flat = rows.reshape(rows.size(0), -1).double()
    s = sketch_sign(flat.size(1))
    return torch.stack([flat.norm(dim=1), flat.sum(dim=1), (flat * s).sum(dim=1)],
                       dim=1).float().contiguous()


def at(t: torch.Tensor, positions) -> torch.Tensor:
    """[1, n, ...] -> [len(positions), ...]"""
    return t[0, list(positions)].contiguous()


def run(args) -> int:
    import dsref  # noqa: E402  (oracle put tools/ on sys.path)

    torch.set_grad_enabled(False)
    torch.set_num_threads(min(16, os.cpu_count() or 4))
    inference_dir = os.path.join(args.model, "inference")
    ref = dsref.load_reference(inference_dir)
    store = dsref.WeightStore(args.model, args.manifest)
    tokenizer = dsref.TokenizerAdapter(os.path.join(args.model, "tokenizer.json"))

    ids = tokenizer.encode(oracle.L2_PROMPT)[:N_TOKENS]
    if len(ids) != N_TOKENS:
        raise SystemExit(f"the L2 prompt tokenises to {len(ids)} < {N_TOKENS} tokens")
    l3_index = os.path.join(os.path.dirname(os.path.abspath(args.out)), "l3", "index.json")
    if os.path.exists(l3_index):
        with open(l3_index, encoding="utf-8") as f:
            l3_ids = json.load(f)["prompt_ids"]
        if list(l3_ids) != list(ids):
            raise SystemExit("prompt ids differ from tests/data/l3's; the two exports "
                             "must describe the same prefill")

    max_seq_len = 128
    margs = dsref.build_args(ref, inference_dir, max_seq_len=max_seq_len)
    layout_e = ref.EngramLayout.from_args(margs)
    os.makedirs(args.out, exist_ok=True)

    sys.path.insert(0, inference_dir)
    import engram as eng  # noqa: E402
    cached = dsref.CachedTokenMap.build(tokenizer, args.token_map)
    orig_build = eng.build_compressed_token_map
    eng.build_compressed_token_map = lambda _t: (cached.lookup, cached.size)
    try:
        ngram = ref.NgramHashState(margs, layout_e, tokenizer)
    finally:
        eng.build_compressed_token_map = orig_build

    embed = store.tensor("embed.weight")
    t_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    h = embed[t_ids[0]].unsqueeze(1).repeat(1, margs.hc_mult, 1).unsqueeze(0).to(torch.bfloat16)
    del embed
    hashes = ngram(t_ids, 0, None)
    pre_mix = ref.make_identity_pre_mix(h, margs.hc_mult)

    shared = ref.shared_attn
    shared.compress_kv = shared.index_k = shared.topk_idxs = shared.candidates = None
    cap = oracle.L2Capture(ref)
    writer = oracle.L2Writer(args.out)
    routing: dict = {}
    t_start = time.perf_counter()
    N = N_TOKENS

    for L in range(margs.n_layers):
        t0 = time.perf_counter()
        block = dsref.make_block(ref, margs, L, layout_e, store, args.engram_threads)
        do = L in EXPORT_LAYERS
        b = block.b
        if do:
            cap.attach(block)
        cap.reset(do)
        out: dict = {}

        def put(name, kind, v):
            out[name] = (kind, v)

        if do:
            put("block_in", "bf16", at(h, ROW_POSITIONS))
            put("sk.block_in", "f32", sketch(h[0]))
            put("pre_mix_in", "f32", pre_mix[0].float().contiguous())
        if b.engram is not None:
            hi = layout_e.layer_ids.index(L)
            if do:
                put("engram_hashes", "i32", hashes[0, :, hi, :].int().contiguous())
            h = b.engram(h, hashes[:, :, hi, :], None)
            if do:
                put("engram_out", "bf16", at(h, ROW_POSITIONS))
                put("sk.engram_out", "f32", sketch(h[0]))

        ffn_in, resid, fpre, fpost, fcomb = block.forward_attn(h, 0, pre_mix)

        moe = b.ffn
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
        routing[f"L{L:02d}.gate_ids"] = ("i32", indices.int().contiguous())
        routing[f"L{L:02d}.gate_weights"] = ("f32", weights.float().contiguous())

        if do:
            t = cap.t
            a = b.attn
            nh, hd = margs.n_heads, margs.head_dim
            # --- mHC (attention half) ------------------------------------
            ap, apo, ac = cap.mixes[0]
            put("attn_pre", "f32", ap[0].contiguous())
            put("attn_post", "f32", apo[0].contiguous())
            put("attn_comb", "f32", ac[0].reshape(N, -1).contiguous())
            put("attn_hc_pre_out", "bf16", at(t["attn_norm.in"], ROW_POSITIONS))
            if L in ATTN_FULL_LAYERS:
                put("attn_norm_out", "bf16", t["attn_norm"][0].contiguous())
            else:
                put("attn_norm_out.rows", "bf16", at(t["attn_norm"], ROW_POSITIONS))
            put("sk.attn_norm_out", "f32", sketch(t["attn_norm"][0]))
            # --- Q path ---------------------------------------------------
            put("wq_a_out", "bf16", at(t["wq_a"], ROW_POSITIONS))
            put("qr", "bf16", at(t["q_norm"], ROW_POSITIONS))
            put("sk.qr", "f32", sketch(t["q_norm"][0]))
            q_post = cap.live["wq_b"][0].reshape(N, nh, hd)
            put("q", "bf16", q_post[list(ATTN_POSITIONS)].contiguous())
            put("sk.q", "f32", sketch(q_post))
            # --- KV path --------------------------------------------------
            put("wkv_out", "bf16", at(t["wkv"], ROW_POSITIONS))
            kv_pre, kv_post = cap.q8[0]
            put("kv_pre_quant", "bf16", at(kv_pre, ROW_POSITIONS))
            put("kv", "bf16", kv_post[0].contiguous())
            # --- compressor / indexer ------------------------------------
            if a.compressor is not None:
                ratio = a.compressor.compress_ratio
                G = N // ratio
                put("cmp_ratio", "i32", torch.tensor([ratio], dtype=torch.int32))
                put("cmp_wkv_out", "f32", at(t["cmp_wkv"], ROW_POSITIONS).float())
                put("sk.cmp_wkv_out", "f32", sketch(t["cmp_wkv"][0].float()))
                put("cmp_latent_pre_rope", "bf16", t["cmp_norm"][0].contiguous())
                put("cmp_cache", "bf16", a.compress_kv_cache[0, :G].contiguous())
                c = a.compressor
                if getattr(c, "kv_state", None) is not None:
                    put("cmp_state_kv", "f32", c.kv_state[0].float().contiguous())
                    put("cmp_state_score", "f32", c.score_state[0].float().contiguous())
            if a.indexer is not None and shared.index_k is not None:
                ix = a.indexer
                ratio = ix.compress_ratio
                G = N // ratio
                if ix.owns_k:
                    put("index_k_pre_rope", "bf16", t["idx_k"][0].contiguous())
                    put("index_k", "bf16", ix.k_cache[0, :G].contiguous())
                put("index_k_all", "bf16", shared.index_k[0, :G].contiguous())
                idxq = [e for e in cap.q4 if e[0] == oracle.L2_INDEX_BLOCK and e[2].ndim == 4]
                if idxq and "idx_w" in t:
                    qi = idxq[0][3]                                     # [1, N, h, d]
                    put("index_q_pre_rope", "bf16",
                        at(t["idx_q"], ATTN_POSITIONS).reshape(len(ATTN_POSITIONS),
                                                               ix.n_heads, ix.index_head_dim))
                    put("index_q", "bf16", at(qi, ATTN_POSITIONS))
                    put("sk.index_q", "f32", sketch(qi[0]))
                    w = t["idx_w"] * (ix.softmax_scale * ix.n_heads ** -0.5)
                    put("index_weights", "bf16", w[0].contiguous())
                    index_k = shared.index_k[:1, :G]
                    score = torch.einsum("bshd,btd->bsht", qi, index_k)
                    score = (score.relu_() * w.unsqueeze(-1)).sum(dim=2)   # [1, N, G]
                    lens = (torch.arange(1, N + 1) // ratio).unsqueeze(-1)
                    score.masked_fill_(torch.arange(G) >= lens, -torch.inf)
                    put("index_score", "f32", score[0].float().contiguous())
            # --- sparse attention ----------------------------------------
            sp = cap.sparse
            put("topk_idxs", "i32", sp["idx"][0].contiguous())
            put("attn_sink", "f32", sp["sink"].float().contiguous())
            n_kv_rows = sp["kv"].size(1)
            put("attn_n_kv_rows", "i32", torch.tensor([n_kv_rows], dtype=torch.int32))
            o_post = cap.live["o"][0].reshape(N, nh, hd)
            put("attn_out_irope", "bf16", o_post[list(ATTN_POSITIONS)].contiguous())
            put("sk.attn_out_irope", "f32", sketch(o_post))
            # --- output projection + hc_post -----------------------------
            put("wo_a_out", "bf16", at(t["wo_b.in"], ROW_POSITIONS))
            put("sk.wo_a_out", "f32", sketch(t["wo_b.in"][0]))
            put("wo_b_out", "bf16", at(t["wo_b"], ROW_POSITIONS))
            put("sk.wo_b_out", "f32", sketch(t["wo_b"][0]))
            put("attn_block_out", "bf16", at(cap.posts[0], ROW_POSITIONS))
            put("sk.attn_block_out", "f32", sketch(cap.posts[0][0]))
            # --- mHC (ffn half), gate, MoE, block out --------------------
            fp, fpo, fc = cap.mixes[1]
            put("ffn_pre", "f32", fp[0].contiguous())
            put("ffn_post", "f32", fpo[0].contiguous())
            put("ffn_comb", "f32", fc[0].reshape(N, -1).contiguous())
            put("ffn_hc_pre_out", "bf16", at(t["ffn_norm.in"], ROW_POSITIONS))
            if L in FFN_FULL_LAYERS:
                put("ffn_norm_out", "bf16", t["ffn_norm"][0].contiguous())
            else:
                put("ffn_norm_out.rows", "bf16", at(t["ffn_norm"], ROW_POSITIONS))
            put("sk.ffn_norm_out", "f32", sketch(t["ffn_norm"][0]))
            put("gate_scores", "f32", moe.gate_scores(flat)[list(ROW_POSITIONS)].float())
            put("gate_bias", "f32", moe.gate.bias.float().contiguous())
            put("gate_ids", "i32", indices.int().contiguous())
            put("gate_weights", "f32", weights.float().contiguous())
            put("moe_shared_out", "f32", shared_out[list(ROW_POSITIONS)].contiguous())
            put("sk.moe_shared_out", "f32", sketch(shared_out))
            put("moe_out", "f32", y[list(ROW_POSITIONS)].contiguous())
            put("sk.moe_out", "f32", sketch(y))
            put("block_out", "bf16", at(h_out, ROW_POSITIONS))
            put("sk.block_out", "f32", sketch(h_out[0]))
            put("positions_rows", "i32", torch.tensor(ROW_POSITIONS, dtype=torch.int32))
            put("positions_attn", "i32", torch.tensor(ATTN_POSITIONS, dtype=torch.int32))
            rec = writer.write(L, f"prefill{N}", out)
            print(f"    layer {L:2d} -> {rec['file']} ({rec['bytes'] / 1024:.0f} KiB, "
                  f"{len(rec['tensors'])} tensors)", flush=True)
            cap.detach()

        h, pre_mix = h_out, fpre
        del block
        print(f"    prefill layer {L:2d}  {len(used):3d} experts  "
              f"{time.perf_counter() - t0:5.1f}s  |h| {h.float().norm().item():.4e}",
              flush=True)

    rec = writer.write(99, "routing", routing)
    print(f"    routing -> {rec['file']} ({rec['bytes'] / 1024:.0f} KiB)")
    meta = {
        "version": oracle.L2_VERSION,
        "model": "DeepSeek-V4.1-Flash",
        "generator": "tools/oracle_prefill.py",
        "prompt": oracle.L2_PROMPT,
        "prompt_ids": [int(i) for i in ids],
        "prefill_len": N,
        "decode_pos": N,
        "positions_rows": list(ROW_POSITIONS),
        "positions_attn": list(ATTN_POSITIONS),
        "sketch": "per row: [L2 norm, sum, sum(v[i] * s(i))], s(i) = -1 if "
                  "((i * 2654435761 + 12345) mod 2^32 >> 16) & 1 else +1",
        "notes": (
            "The start_pos == 0 branch of inference/model.py, behind tools/dsref.py's "
            "CPU kernel shims, 64 tokens, all forty layers computed and seven exported. "
            "'attn_norm_out' / 'ffn_norm_out' hold every position on the layers named in "
            "attn_full_layers / ffn_full_layers and only positions_rows elsewhere "
            "('.rows' suffix). index_score is recomputed from the reference's captured "
            "tensors (see the docstring). 'routing' (layer 99) holds every layer's "
            "top-6 ids and weights for all 64 tokens."),
        "attn_full_layers": list(ATTN_FULL_LAYERS),
        "ffn_full_layers": list(FFN_FULL_LAYERS),
        "config": {"dim": margs.dim, "hc_mult": margs.hc_mult, "n_heads": margs.n_heads,
                   "head_dim": margs.head_dim, "rope_head_dim": margs.rope_head_dim,
                   "q_lora_rank": margs.q_lora_rank, "o_lora_rank": margs.o_lora_rank,
                   "o_groups": margs.o_groups, "window_size": margs.window_size,
                   "n_routed_experts": margs.n_routed_experts,
                   "n_activated_experts": margs.n_activated_experts,
                   "moe_inter_dim": margs.moe_inter_dim,
                   "index_n_heads": margs.index_n_heads,
                   "index_head_dim": margs.index_head_dim,
                   "index_topk": margs.index_topk, "max_seq_len": max_seq_len},
        "seconds": round(time.perf_counter() - t_start, 1),
    }
    path = writer.finish(meta)
    total = sum(s["bytes"] for s in writer.steps)
    print(f"\nwrote {len(writer.steps)} records, {total / 1e6:.2f} MB -> {path} "
          f"in {meta['seconds']}s")
    store.close()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--out", default="tests/data/prefill")
    p.add_argument("--token-map", default="tests/data/l2x/token_map.npz",
                   help="cache of the engram compressed-vocabulary map (a pure "
                        "function of tokenizer.json); read if present")
    p.add_argument("--engram-threads", type=int, default=32)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
