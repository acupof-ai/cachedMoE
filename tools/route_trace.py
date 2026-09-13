#!/usr/bin/env python3
"""Record routing decisions over a corpus, so cache policy can be chosen from data.

Design reference: section 9.2 (measurement tools). P1 deliverable (section 15), and
section 16 forbids writing any Planner policy code before it has run.

Everything the expert cache does rests on five unknowns (section 9.1):

  Q1  static frequency: does top-x% of experts cover y% of routes? `noaux_tc` load
      balancing may leave almost no static skew, in which case a pinned expert set
      is worthless.
  Q2  reuse (stack) distance distribution, which gives LRU's hit rate analytically
      at any capacity, and says whether per-layer quotas beat one global pool.
  Q3  Jaccard overlap of consecutive tokens' expert sets, per layer. This sets how
      much speculative decoding actually saves on NVMe traffic, and feeds the
      schedule curve of section 10.3.
  Q4  lookahead recall: predicting layer L's top-6 from the layer L-d residual
      stream, over d = 1..8 and K = 6..16, for each of the two input approximations
      of section 9.4.
  Q5  T_layer / T_io, which sets the minimum useful prefetch depth d. (Measured, not
      traced: section 9.2.1 has T_io already.)

This script answers Q1-Q4. `tools/cache_sim.py` consumes what it writes.

Why prefill, not decode
-----------------------
Under teacher forcing every position's routing decision is the one decode would make
for the same text -- the model is causal and nothing downstream of position p feeds
it. But a decode-shaped trace streams 13 GB of weights *per token*, while a prefill
pass streams 510 GB **once for the whole corpus**. So the corpus is processed layer
by layer: load layer L's weights, push every prompt's hidden states through it, free,
move on. 20K tokens of [hc=4, 5120] bf16 is 0.8 GB; one layer of routed experts is
7.2 GB. Both fit; 40 layers of experts at once would not.

Within a layer the FFN runs **expert-major** for the same reason -- one pass over the
layer's 384 experts, each dequantised once and applied to whichever tokens routed to
it. That is also design section 9.7's prefill ordering, so the shape of this script
is the shape of the runtime's prefill path.

The decoder (layers 20-39) runs over *all* positions, i.e. section 11.2's "oracle"
mode rather than the production bounded replay. Bounded replay is a prefill
approximation; for a trace we want the routing every position would see while being
decoded, which is what the full decoder forward gives.

Usage
-----
    uv run python tools/route_trace.py --model D:/models/DeepSeek-V4.1-Flash \\
        --verify --out traces/smoke

    uv run python tools/route_trace.py --model D:/models/DeepSeek-V4.1-Flash \\
        --tokens 20000 --out traces/mixed        # resumable; re-run to continue
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import corpus as corpus_mod          # noqa: E402
import dsref                         # noqa: E402

# The corpus is deliberately multilingual, so a cp1252 console would mangle every
# progress line that quotes it.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

STATE_FILE = "state.pt"
META_FILE = "meta.json"
TOPK_RECORD = 16


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="route_trace.py",
        description="record (token, layer) -> expert routing for cache simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", required=True,
                   help="the safetensors directory, with deepmoe_manifest.json in it")
    p.add_argument("--manifest", default=None,
                   help="manifest path (default: <model>/deepmoe_manifest.json)")
    p.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   help="this repository, one of the corpus sources")
    p.add_argument("--out", required=True,
                   help="output directory: one Parquet per layer, plus state for resume")
    p.add_argument("--corpus", action="append", default=None, metavar="PATH:KIND",
                   help="extra corpus file, KIND in {en, zh, code}; repeatable. "
                        "Without any, the defaults in tools/corpus.py are used")
    p.add_argument("--tokens", type=int, default=20000,
                   help="token budget for the corpus (design section 9.2 asks for >= 20K)")
    p.add_argument("--min-prompts", type=int, default=40)
    p.add_argument("--min-len", type=int, default=128, help="shortest prompt, in tokens")
    p.add_argument("--max-len", type=int, default=2048, help="longest prompt, in tokens")
    p.add_argument("--layers", default=None,
                   help="restrict to these layers, e.g. '0-3' or '0,1,20'. Development "
                        "only: the residual stream is wrong for any layer whose "
                        "predecessors were skipped")
    p.add_argument("--prompts", type=int, default=None,
                   help="cap the number of prompts (development slices)")
    p.add_argument("--lookahead-depths", default="1,2,3,4,5,6,7,8",
                   help="the d values to record predictions for (Q4)")
    p.add_argument("--lookahead-input", choices=["mean-hc", "pre-identity", "both"],
                   default="both",
                   help="which approximation of the layer input feeds the predicted "
                        "layer's gate (design section 9.4)")
    p.add_argument("--stream-dtype", choices=["bf16", "fp32"], default="bf16",
                   help="bf16 is what generate.py's default dtype makes the residual "
                        "stream; fp32 is available for the ablation --verify reports")
    p.add_argument("--engram", choices=["on", "off"], default="on",
                   help="off skips the layer 1 / 14 n-gram lookups. They change the "
                        "residual stream, so off is not faithful -- it exists to "
                        "separate NVMe row-read time from compute time")
    p.add_argument("--engram-threads", type=int, default=32,
                   help="concurrent 4 KiB row reads; section 9.2.1 measured 83.7K IOPS "
                        "at QD 48")
    p.add_argument("--threads", type=int, default=None, help="torch intra-op threads")
    p.add_argument("--checkpoint-every", type=int, default=1,
                   help="write the resumable state every N layers (0 disables)")
    p.add_argument("--verify", action="store_true",
                   help="run the faithfulness checks and exit without tracing")
    p.add_argument("--verify-tokens", type=int, default=64)
    p.add_argument("--compression", default="zstd")
    p.add_argument("--seed", type=int, default=0)
    return p


def parse_layers(spec: str | None, n_layers: int) -> list[int]:
    if not spec:
        return list(range(n_layers))
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


# --------------------------------------------------------------------------- #
# per-prompt attention state that outlives a layer
# --------------------------------------------------------------------------- #

class CrossLayerState:
    """What `model.SharedAttentionRuntime` carries between layers, kept per prompt.

    In the reference, layers run back to back for one sequence and a single slot each
    is enough. Here the loop is transposed -- all prompts, then the next layer -- so
    each prompt needs its own copy of whatever a source layer published:

        compress_kv   written by kv_source_layers (2, 8, 14, 20), read until the next
        index_k       written by the same layers' indexers
        topk_idxs     written by index_source_layers (2, 8, 14, 20, 24, 28, 32, 36)
        candidates    written once by candidate_source_layer (20), read by 24..36

    Only the most recent of each is ever read, so one slot per prompt suffices --
    except `candidates`, which has to survive from layer 20 to layer 36.
    """

    __slots__ = ("compress_kv", "index_k", "topk_idxs", "candidates")

    def __init__(self):
        self.compress_kv = None
        self.index_k = None
        self.topk_idxs = None
        self.candidates = None

    def install(self, shared):
        shared.compress_kv = self.compress_kv
        shared.index_k = self.index_k
        shared.topk_idxs = self.topk_idxs
        shared.candidates = self.candidates

    def capture(self, shared, seqlen: int, ratio: int):
        """Snapshot anything this layer *republished*.

        A source layer points `shared.*` at its own module buffer, which the next
        prompt is about to overwrite, so that has to be cloned -- but only the live
        prefix. A consumer layer leaves `shared.*` pointing at the clone we installed,
        and identity tells the two apart, which keeps layers 21-39 from re-cloning
        layer 20's [s, s] candidate mask forty times.
        """
        live = max(1, seqlen // max(ratio, 1))
        if shared.compress_kv is not self.compress_kv:
            ck = shared.compress_kv
            self.compress_kv = None if ck is None else ck[:, :live].clone()
        if shared.index_k is not self.index_k:
            ik = shared.index_k
            self.index_k = None if ik is None else ik[:, :live].clone()
        if shared.topk_idxs is not self.topk_idxs:
            ti = shared.topk_idxs
            self.topk_idxs = None if ti is None else ti.clone()
        if shared.candidates is not self.candidates:
            ca = shared.candidates
            self.candidates = None if ca is None else ca.clone()

    def state_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__slots__}

    @staticmethod
    def from_state(d: dict) -> "CrossLayerState":
        s = CrossLayerState()
        for k in s.__slots__:
            setattr(s, k, d.get(k))
        return s


# --------------------------------------------------------------------------- #
# the tracer
# --------------------------------------------------------------------------- #

class Tracer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.inference_dir = os.path.join(args.model, "inference")
        self.ref = dsref.load_reference(self.inference_dir)
        self.store = dsref.WeightStore(args.model, args.manifest)
        self.tokenizer = dsref.TokenizerAdapter(os.path.join(args.model, "tokenizer.json"))
        self.stream_dtype = torch.bfloat16 if args.stream_dtype == "bf16" else torch.float32
        self.depths = [int(d) for d in args.lookahead_depths.split(",") if d.strip()]
        self.approxes = (["mean", "pre0"] if args.lookahead_input == "both"
                         else ["mean"] if args.lookahead_input == "mean-hc" else ["pre0"])
        self.model_args = None
        self.layout = None
        self.ngram = None

    # -- setup ------------------------------------------------------------

    def prepare(self, max_seq_len: int):
        ref = self.ref
        self.model_args = dsref.build_args(ref, self.inference_dir, max_seq_len=max_seq_len)
        self.layout = ref.EngramLayout.from_args(self.model_args)
        if self.args.engram == "on":
            cache = os.path.join(self.args.out, "token_map.npz")
            self.ngram = make_ngram_hash(ref, self.model_args, self.layout,
                                         self.tokenizer, cache)

    def build_corpus(self) -> list[corpus_mod.Prompt]:
        a = self.args
        sources = corpus_mod.default_sources(a.repo, a.model)
        for spec in a.corpus or []:
            path, _, kind = spec.rpartition(":")
            if not path:
                path, kind = spec, "en"
            sources.append(corpus_mod.Source(path, kind, "user supplied"))
        prompts = corpus_mod.build(sources, self.tokenizer, a.tokens, a.min_prompts,
                                   a.min_len, a.max_len, a.seed)
        if a.prompts:
            prompts = prompts[:a.prompts]
        return prompts

    # -- one layer --------------------------------------------------------

    def run_layer(self, layer_id: int, block, H, pre_mix, offsets, states, snapshots):
        """Push the whole corpus through one layer. Returns the trace arrays."""
        ref = self.ref
        args = self.model_args
        n_tok = H.size(0)
        dim = args.dim
        ratio = args.compress_ratios[layer_id]

        engram_rows = None
        if block.b.engram is not None and self.args.engram == "on":
            engram_rows = np.empty((n_tok, self.layout.n_heads *
                                    (self.layout.max_ngram_size - 1)), dtype=np.int64)

        ffn_in = torch.empty(n_tok, dim, dtype=self.stream_dtype)
        ffn_post = torch.empty(n_tok, args.hc_mult, dtype=torch.float32)
        ffn_comb = torch.empty(n_tok, args.hc_mult, args.hc_mult, dtype=torch.float32)
        ffn_pre = torch.empty(n_tok, args.hc_mult, dtype=torch.float32)

        t0 = time.perf_counter()
        for pid, (lo, hi) in enumerate(offsets):
            s = hi - lo
            x = H[lo:hi].unsqueeze(0)                       # [1, s, hc, dim]
            if engram_rows is not None:
                hashes = self.hashes[pid][:, :, self.layout.layer_ids.index(layer_id), :]
                x = block.b.engram(x, hashes, None)
                engram_rows[lo:hi] = block.b.engram.embed.last_rows[0]
            states[pid].install(ref.shared_attn)
            fi, resid, fpre, fpost, fcomb = block.forward_attn(x, 0, pre_mix[lo:hi].unsqueeze(0))
            states[pid].capture(ref.shared_attn, s, ratio)
            H[lo:hi] = resid[0]
            ffn_in[lo:hi] = fi[0]
            ffn_pre[lo:hi] = fpre[0]
            ffn_post[lo:hi] = fpost[0]
            ffn_comb[lo:hi] = fcomb[0]
        t_attn = time.perf_counter() - t0

        # -- gate, for every token at once -------------------------------
        t0 = time.perf_counter()
        moe = block.b.ffn
        weights, indices = moe.route(ffn_in)                 # the reference's own Gate
        scores = moe.gate_scores(ffn_in)                     # raw, unbiased
        bias = moe.gate.bias.float()
        biased = scores + bias
        top16 = biased.topk(TOPK_RECORD, dim=-1)
        top16_ids = top16.indices
        top16_scores = scores.gather(1, top16_ids)
        # The gate selects by score+bias, so the first six of our top-16 must be
        # exactly what it chose. If this ever fires, the trace is not the model.
        if not torch.equal(top16_ids[:, :moe.n_activated_experts].sort(-1).values,
                           indices.sort(-1).values):
            raise SystemExit(f"layer {layer_id}: recomputed top-6 disagrees with Gate")
        raw_top6 = scores.topk(moe.n_activated_experts, dim=-1).indices
        bias_applied = (raw_top6.sort(-1).values != indices.sort(-1).values).any(-1)
        t_gate = time.perf_counter() - t0

        # -- lookahead predictions (section 9.4) -------------------------
        t0 = time.perf_counter()
        preds: dict[str, np.ndarray | None] = {}
        for d in self.depths:
            src = snapshots.get(layer_id - d)
            for ai, approx in enumerate(self.approxes):
                key = f"pred_d{d}_{approx}"
                if src is None:
                    preds[key] = None
                    continue
                a = src[:, 0 if approx == "mean" else 1]
                pred_in = block.b.ffn_norm(a.to(self.stream_dtype))
                ps = moe.gate_scores(pred_in) + bias
                preds[key] = ps.topk(TOPK_RECORD, dim=-1).indices.numpy().astype(np.int16)
        t_pred = time.perf_counter() - t0

        # -- experts, expert-major (section 9.7) -------------------------
        t0 = time.perf_counter()
        y = torch.zeros(n_tok, dim, dtype=torch.float32)
        idx_np = indices.numpy()
        order = np.argsort(idx_np.reshape(-1), kind="stable")
        flat_expert = idx_np.reshape(-1)[order]
        rows = (order // moe.n_activated_experts).astype(np.int64)
        slot = (order % moe.n_activated_experts).astype(np.int64)
        starts = np.searchsorted(flat_expert, np.arange(moe.n_routed_experts + 1))
        used = [e for e in range(moe.n_routed_experts) if starts[e] != starts[e + 1]]
        # act_quant is a property of the activation alone, so quantise the whole
        # layer's FFN input once rather than once per (token, routed expert).
        ffn_in_q = dsref.act_quant_dequant(ffn_in, dsref.FP8_BLOCK)
        for e, w1, w2, w3 in self.store.expert_stream(layer_id, used, dim,
                                                      args.moe_inter_dim):
            a, b = starts[e], starts[e + 1]
            r = torch.from_numpy(rows[a:b])
            rw = weights[r, torch.from_numpy(slot[a:b])].unsqueeze(-1)
            y[r] += dsref.expert_ffn(ffn_in_q[r], w1, w2, w3, rw, args.swiglu_limit,
                                     input_is_quantised=True).float()
            del w1, w2, w3
        y += moe.shared_experts(ffn_in).float()
        del ffn_in_q
        t_expert = time.perf_counter() - t0
        n_used = len(used)

        # -- close the block ---------------------------------------------
        for lo, hi in offsets:
            H[lo:hi] = block.finish_ffn(y[lo:hi].unsqueeze(0), H[lo:hi].unsqueeze(0),
                                        ffn_post[lo:hi].unsqueeze(0),
                                        ffn_comb[lo:hi].unsqueeze(0))[0]
        pre_mix.copy_(ffn_pre)

        return {
            "top6_ids": indices.numpy().astype(np.uint16),
            "top6_weights": weights.numpy().astype(np.float32),
            "top16_ids": top16_ids.numpy().astype(np.uint16),
            "top16_scores": top16_scores.numpy().astype(np.float32),
            "gate_bias_applied": bias_applied.numpy(),
            "preds": preds,
            "engram_rows": engram_rows,
            "experts_used": n_used,
            "timing": {"attn": t_attn, "gate": t_gate, "lookahead": t_pred,
                       "expert": t_expert},
        }

    def snapshot(self, H: torch.Tensor) -> torch.Tensor:
        """The two section 9.4 approximations of a layer's input, from the stream.

        `hc_pre(x, pre_identity)` with the one-hot initial mix is exactly copy 0 of
        the stream (`make_identity_pre_mix` puts all the weight on index 0), so both
        approximations are a [n, 5120] projection and the snapshot is [n, 2, 5120]
        rather than a whole [n, 4, 5120] stream -- which is what makes keeping eight
        of them affordable.
        """
        return torch.stack([H.float().mean(dim=1), H[:, 0].float()], dim=1).to(torch.bfloat16)


def make_ngram_hash(ref, args, layout, tokenizer, cache_path: str | None):
    """`NgramHashState` with the compressed token map cached.

    Its own constructor asserts the map's size equals `engram_compressed_vocab_size`
    (99,092), which is a real check on the tokenizer wrapper: every hash multiplier
    derives from that number, so a wrong normalizer would rehash the whole table and
    the assert would catch it.
    """
    import engram as eng
    cached = dsref.CachedTokenMap.build(tokenizer, cache_path)
    orig = eng.build_compressed_token_map
    eng.build_compressed_token_map = lambda _tok: (cached.lookup, cached.size)
    try:
        return ref.NgramHashState(args, layout, tokenizer)
    finally:
        eng.build_compressed_token_map = orig


# --------------------------------------------------------------------------- #
# parquet
# --------------------------------------------------------------------------- #

def _fsl(a: np.ndarray, pa, value_type):
    a = np.ascontiguousarray(a)
    n, w = a.shape
    return pa.FixedSizeListArray.from_arrays(pa.array(a.reshape(-1), type=value_type), w)


def write_layer_parquet(path: str, layer: int, prompt_ids: np.ndarray, pos: np.ndarray,
                        rec: dict, compression: str):
    import pyarrow as pa
    import pyarrow.parquet as pq

    n = len(pos)
    cols = {
        "prompt_id": pa.array(prompt_ids, type=pa.uint16()),
        "pos": pa.array(pos, type=pa.uint32()),
        "layer": pa.array(np.full(n, layer, dtype=np.uint8), type=pa.uint8()),
        "top6_ids": _fsl(rec["top6_ids"], pa, pa.uint16()),
        "top6_weights": _fsl(rec["top6_weights"], pa, pa.float32()),
        "top16_ids": _fsl(rec["top16_ids"], pa, pa.uint16()),
        "top16_scores": _fsl(rec["top16_scores"], pa, pa.float32()),
        "gate_bias_applied": pa.array(rec["gate_bias_applied"], type=pa.bool_()),
    }
    for key, val in rec["preds"].items():
        # Signed, with -1 for "layer L - d does not exist", rather than a null
        # fixed-size list: a fully-null FixedSizeListArray does not survive a Parquet
        # round trip (its child array is empty and the reader rejects it).
        filled = (np.full((n, TOPK_RECORD), -1, dtype=np.int16) if val is None
                  else val.astype(np.int16))
        cols[key] = _fsl(filled, pa, pa.int16())
    table = pa.table(cols)
    pq.write_table(table, path, compression=compression)
    return os.path.getsize(path)


def write_sidecars(out_dir: str, n_layers: int, store, compression: str,
                   engram_layers: list[int], engram_acc: dict):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows_layer, rows_expert, rows_bias = [], [], []
    for L in range(n_layers):
        name = f"layers.{L}.ffn.gate.bias"
        if not store.has(name):
            continue
        b = store.tensor(name).float().numpy()
        rows_layer.extend([L] * len(b))
        rows_expert.extend(range(len(b)))
        rows_bias.extend(b.tolist())
    pq.write_table(pa.table({
        "layer": pa.array(rows_layer, type=pa.uint8()),
        "expert": pa.array(rows_expert, type=pa.uint16()),
        "bias": pa.array(rows_bias, type=pa.float32()),
    }), os.path.join(out_dir, "gate_bias.parquet"), compression=compression)

    if engram_acc:
        layers, pids, poss, rows = [], [], [], []
        for L in engram_layers:
            e = engram_acc.get(L)
            if e is None:
                continue
            layers.append(np.full(len(e["pos"]), L, dtype=np.uint8))
            pids.append(e["prompt_id"])
            poss.append(e["pos"])
            rows.append(e["rows"])
        if rows:
            pq.write_table(pa.table({
                "layer": pa.array(np.concatenate(layers), type=pa.uint8()),
                "prompt_id": pa.array(np.concatenate(pids), type=pa.uint16()),
                "pos": pa.array(np.concatenate(poss), type=pa.uint32()),
                "rows": _fsl(np.concatenate(rows), pa, pa.int64()),
            }), os.path.join(out_dir, "engram_rows.parquet"), compression=compression)


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #

def trace(args: argparse.Namespace) -> int:
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    os.makedirs(args.out, exist_ok=True)

    t = Tracer(args)
    state_path = os.path.join(args.out, STATE_FILE)
    resume = os.path.exists(state_path)

    if resume:
        print(f"resuming from {state_path}")
        blob = torch.load(state_path, weights_only=False)
        prompts = [corpus_mod.Prompt(**p) for p in blob["prompts"]]
        start_layer = blob["next_layer"]
    else:
        prompts = None
        start_layer = None

    max_len = max(len(p.ids) for p in prompts) if prompts else args.max_len
    t.prepare(max_seq_len=max(max_len, 8))
    ref, margs = t.ref, t.model_args

    if prompts is None:
        t0 = time.perf_counter()
        prompts = t.build_corpus()
        print(f"corpus built in {time.perf_counter() - t0:.1f}s")
        summary = corpus_mod.summarise(prompts)
        print(json.dumps({k: v for k, v in summary.items() if k != "sources"}, indent=2))
        max_len = max(len(p.ids) for p in prompts)
        t.prepare(max_seq_len=max_len)
        margs = t.model_args

    offsets, cur = [], 0
    for p in prompts:
        offsets.append((cur, cur + len(p.ids)))
        cur += len(p.ids)
    n_tok = cur
    prompt_ids = np.concatenate([np.full(len(p.ids), i, dtype=np.uint16)
                                 for i, p in enumerate(prompts)])
    pos = np.concatenate([np.arange(len(p.ids), dtype=np.uint32) for p in prompts])

    layers = parse_layers(args.layers, margs.n_layers)

    # -- engram hashes, once for the whole corpus ------------------------
    t.hashes = []
    if args.engram == "on":
        t0 = time.perf_counter()
        for p in prompts:
            ids = torch.tensor(p.ids, dtype=torch.long).unsqueeze(0)
            t.hashes.append(t.ngram(ids, 0, None))
        print(f"engram n-gram hashes for {len(prompts)} prompts in "
              f"{time.perf_counter() - t0:.1f}s")

    # -- state -----------------------------------------------------------
    if resume:
        H = blob["H"]
        pre_mix = blob["pre_mix"]
        states = [CrossLayerState.from_state(d) for d in blob["states"]]
        snapshots = blob["snapshots"]
        engram_acc = blob["engram_acc"]
        timings = blob["timings"]
    else:
        embed = t.store.tensor("embed.weight")
        ids = torch.tensor(np.concatenate([p.ids for p in prompts]), dtype=torch.long)
        H = embed[ids].unsqueeze(1).repeat(1, margs.hc_mult, 1).to(t.stream_dtype)
        del embed
        pre_mix = torch.zeros(n_tok, margs.hc_mult, dtype=torch.float32)
        pre_mix[:, 0] = 1.0                       # model.make_identity_pre_mix
        states = [CrossLayerState() for _ in prompts]
        snapshots = {}
        engram_acc = {}
        timings = {}
        start_layer = layers[0]

    todo = [L for L in layers if L >= start_layer]
    if not todo:
        print(f"nothing to do: state says layer {start_layer}, requested {layers}")
        return 0
    print(f"{n_tok} tokens, {len(prompts)} prompts, layers {todo[0]}..{todo[-1]}")

    total_t0 = time.perf_counter()
    for L in todo:
        t0 = time.perf_counter()
        block = dsref.make_block(ref, margs, L, t.layout, t.store)
        t_load = time.perf_counter() - t0

        rec = t.run_layer(L, block, H, pre_mix, offsets, states, snapshots)
        path = os.path.join(args.out, f"route_layer{L:02d}.parquet")
        nbytes = write_layer_parquet(path, L, prompt_ids, pos, rec, args.compression)

        if rec["engram_rows"] is not None:
            engram_acc[L] = {"prompt_id": prompt_ids, "pos": pos,
                             "rows": rec["engram_rows"]}

        snapshots[L] = t.snapshot(H)
        for old in [k for k in snapshots if k < L - max(t.depths)]:
            del snapshots[old]
        del block

        t_ckpt = 0.0
        if args.checkpoint_every and ((L + 1) % args.checkpoint_every == 0 or L == todo[-1]):
            tc = time.perf_counter()
            tmp = state_path + ".tmp"
            torch.save({
                "H": H, "pre_mix": pre_mix, "snapshots": snapshots,
                "states": [s.state_dict() for s in states],
                "prompts": [vars(p) for p in prompts],
                "engram_acc": engram_acc, "timings": timings,
                "next_layer": L + 1,
            }, tmp)
            os.replace(tmp, state_path)
            t_ckpt = time.perf_counter() - tc

        d = rec["timing"]
        dt = time.perf_counter() - t0
        timings[str(L)] = {"total": round(dt, 2), "load": round(t_load, 2),
                           **{k: round(v, 2) for k, v in d.items()},
                           "checkpoint": round(t_ckpt, 2),
                           "experts_used": rec["experts_used"],
                           "parquet_bytes": nbytes}
        print(f"layer {L:2d}  {dt:7.1f}s  load {t_load:5.1f}  attn {d['attn']:6.1f}  "
              f"gate {d['gate']:5.1f}  look {d['lookahead']:5.1f}  "
              f"expert {d['expert']:7.1f}  ckpt {t_ckpt:5.1f}  "
              f"experts {rec['experts_used']:3d}  parquet {nbytes / 1e6:6.1f} MB",
              flush=True)

    write_sidecars(args.out, margs.n_layers, t.store, args.compression,
                   list(t.layout.layer_ids), engram_acc)
    meta = {
        "model": args.model,
        "corpus": corpus_mod.summarise(prompts),
        "layers": todo,
        "tokens": n_tok,
        "lookahead_depths": t.depths,
        "lookahead_inputs": t.approxes,
        "stream_dtype": args.stream_dtype,
        "engram": args.engram,
        "timings": timings,
        "wall_s": round(time.perf_counter() - total_t0, 1),
    }
    with open(os.path.join(args.out, META_FILE), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\ndone in {meta['wall_s']}s -> {args.out}")
    t.store.close()
    return 0


# --------------------------------------------------------------------------- #
# faithfulness checks (design section 12's discipline, applied to this port)
# --------------------------------------------------------------------------- #

VERIFY_PROMPT = (
    "The expert cache holds about thirty percent of the routed experts, so the "
    "decode loop is bounded by NVMe rather than by memory bandwidth.\n"
    "这台机器的真实瓶颈是 NVMe，而不是 LPDDR 带宽。\n"
    "def hit_rate(hits: int, total: int) -> float:\n"
    "    return hits / total if total else 0.0\n"
)


def verify(args: argparse.Namespace) -> int:
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    os.makedirs(args.out, exist_ok=True)
    report: dict = {}
    t = Tracer(args)
    ids = t.tokenizer.encode(VERIFY_PROMPT)[:args.verify_tokens]
    n = len(ids)
    print(f"verify prompt: {n} tokens\n")

    t.prepare(max_seq_len=max(n, 8))
    ref, margs = t.ref, t.model_args

    # 1. tokenizer / engram hashing -----------------------------------
    if t.ngram is not None:
        report["compressed_vocab_size"] = int(t.ngram.token_map.max().item()) + 1
        report["compressed_vocab_expected"] = margs.engram_compressed_vocab_size
        print(f"[1] engram compressed token map: {report['compressed_vocab_size']} "
              f"classes; config says {margs.engram_compressed_vocab_size}. "
              f"NgramHashState asserts these are equal on construction, so the "
              f"tokenizer wrapper and the NFKC/strip-accents/lowercase normalizer "
              f"agree with training -- every hash multiplier derives from this number")
    else:
        print("[1] engram disabled (--engram off): n-gram hashing not checked")

    # 2. manifest vs safetensors, on a real expert ---------------------
    t0 = time.perf_counter()
    parts = t.store.expert_bytes(0, 0)
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from oracle import safetensors_expert
        ctrl = safetensors_expert(args.model, "layers.0", 0)
        agree = all(parts[p] == ctrl[p] for p in dsref.EXPERT_PARTS)
    except Exception as exc:                      # noqa: BLE001
        agree = f"skipped: {exc}"
    report["manifest_vs_safetensors"] = agree
    print(f"[2] layer 0 expert 0, manifest run/skew vs safetensors library: {agree} "
          f"({time.perf_counter() - t0:.1f}s)")

    # 3. fp4 fast dequant vs the float reference -----------------------
    fast = dsref.dequant_fp4_bf16(parts["w1.weight"], parts["w1.scale"], 2304, 5120)
    slow = dsref.dequant_fp4_fp32_ref(parts["w1.weight"], parts["w1.scale"], 2304, 5120)
    exact = bool(torch.equal(fast.float(), slow.bfloat16().float()))
    report["fp4_fast_dequant_exact"] = exact
    print(f"[3] FP4 exponent-add dequant == float reference, bit for bit: {exact}")

    # 4. run the stack --------------------------------------------------
    x0 = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    hashes = t.ngram(x0, 0, None) if args.engram == "on" else None
    embed = t.store.tensor("embed.weight")
    h = embed[x0[0]].unsqueeze(1).repeat(1, margs.hc_mult, 1).to(t.stream_dtype).unsqueeze(0)
    del embed
    pre_mix = ref.make_identity_pre_mix(h, margs.hc_mult)
    shared = ref.shared_attn
    shared.compress_kv = shared.index_k = shared.topk_idxs = shared.candidates = None

    layer0 = {}
    sink_checks = []
    t0 = time.perf_counter()
    for L in range(margs.n_layers):
        block = dsref.make_block(ref, margs, L, t.layout, t.store)
        if block.b.engram is not None and hashes is not None:
            h = block.b.engram(h, hashes[:, :, t.layout.layer_ids.index(L), :], None)
        ffn_in, resid, fpre, fpost, fcomb = block.forward_attn(h, 0, pre_mix)
        moe = block.b.ffn
        flat = ffn_in.view(-1, margs.dim)
        weights, indices = moe.route(flat)
        if L == 0:
            layer0 = {"indices": indices.clone(), "weights": weights.clone(),
                      "scores": moe.gate_scores(flat).clone(),
                      "bias": moe.gate.bias.float().clone()}
            _, _, comb = block.b.hc_mixes(h, block.b.hc_attn_fn, block.b.hc_attn_scale,
                                          block.b.hc_attn_base)
            sink_checks = [float((comb.sum(-1) - 1).abs().max()),
                           float((comb.sum(-2) - 1).abs().max())]
        y = torch.zeros(flat.size(0), margs.dim, dtype=torch.float32)
        used = sorted(set(indices.reshape(-1).tolist()))
        for e in used:
            r, s = torch.where(indices == e)
            w1, w2, w3 = t.store.expert_weights(L, e, margs.dim, margs.moe_inter_dim)
            y[r] += dsref.expert_ffn(flat[r], w1, w2, w3, weights[r, s, None],
                                     margs.swiglu_limit).float()
        y += moe.shared_experts(flat).float()
        h = block.finish_ffn(y.unsqueeze(0), resid, fpost, fcomb)
        pre_mix = fpre
        del block
        print(f"    layer {L:2d}  {len(used):3d} distinct experts  "
              f"|h| {h.float().norm().item():.3e}", flush=True)
    t_fwd = time.perf_counter() - t0

    # 5. Sinkhorn -------------------------------------------------------
    report["sinkhorn_row_err"], report["sinkhorn_col_err"] = sink_checks
    print(f"\n[4] layer 0 Sinkhorn comb, 20 iterations: |colsum - 1| "
          f"{sink_checks[1]:.2e}, |rowsum - 1| {sink_checks[0]:.2e}. The columns are "
          f"exact and the rows are not because hc_split_sinkhorn_kernel ends its "
          f"loop on a *column* normalise -- design section 2.4 says 'doubly "
          f"stochastic', but the reference kernel only converges there, it does not "
          f"land there")

    # 6. gate semantics -------------------------------------------------
    sc, bias = layer0["scores"], layer0["bias"]
    biased_top6 = (sc + bias).topk(margs.n_activated_experts, -1).indices
    raw_top6 = sc.topk(margs.n_activated_experts, -1).indices
    agree_gate = bool(torch.equal(biased_top6.sort(-1).values,
                                  layer0["indices"].sort(-1).values))
    changed = float((raw_top6.sort(-1).values != layer0["indices"].sort(-1).values)
                    .any(-1).float().mean())
    wsum = layer0["weights"].sum(-1)
    report["gate_selection_matches"] = agree_gate
    report["gate_bias_changes_top6_frac"] = round(changed, 4)
    report["route_weight_sum"] = [round(float(wsum.min()), 4), round(float(wsum.max()), 4)]
    print(f"[5] layer 0 gate: our top-6 from score+bias == Gate.forward: {agree_gate}; "
          f"the noaux_tc bias changes the top-6 for {changed:.1%} of tokens; "
          f"routing weights sum to {float(wsum.min()):.4f}..{float(wsum.max()):.4f} "
          f"(route_scale = {margs.route_scale})")

    # 7. teacher-forced next-token predictions ---------------------------
    # Transformer.forward's tail: collapse the hc copies with the final pre_mix,
    # RMSNorm, then the head.
    collapsed = (pre_mix.unsqueeze(-1) * h.float()).sum(dim=2)[0]     # hc_pre
    norm_w = t.store.tensor("norm.weight").float()
    head = t.store.tensor("head.weight")
    preds = []
    for p in _verify_positions(n):
        logits = chunked_logits(block_norm(collapsed[p], norm_w, margs.norm_eps), head)
        top = logits.topk(5)
        preds.append({
            "pos": p,
            "context": t.tokenizer.decode(ids[max(0, p - 12):p + 1]),
            "gold_next": t.tokenizer.decode([ids[p + 1]]) if p + 1 < n else None,
            "top5": [[int(i), t.tokenizer.decode([int(i)]), round(float(v), 3)]
                     for v, i in zip(top.values, top.indices)],
            "margin": round(float(top.values[0] - top.values[1]), 3),
        })
    del head
    report["greedy"] = preds
    print("\n[6] teacher-forced greedy next-token predictions")
    for pr in preds:
        best = pr["top5"][0]
        print(f"      pos {pr['pos']:3d}  ...{pr['context']!r}")
        print(f"               gold {pr['gold_next']!r}   argmax {best[1]!r} "
              f"(logit {best[2]}, margin {pr['margin']})")
        print(f"               top5 {[x[1] for x in pr['top5']]}")
    hits = sum(1 for pr in preds if pr["gold_next"] is not None
               and pr["top5"][0][1] == pr["gold_next"])
    report["greedy_gold_top1"] = f"{hits}/{sum(1 for p in preds if p['gold_next'])}"
    print(f"      argmax == the corpus' own next token: {report['greedy_gold_top1']}")
    report["forward_seconds"] = round(t_fwd, 1)
    print(f"\nfull {margs.n_layers}-layer forward over {n} tokens: {t_fwd:.1f}s")

    with open(os.path.join(args.out, "verify.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    t.store.close()
    return 0


def block_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """model.RMSNorm.forward for a single vector."""
    xf = x.float()
    return (weight.float() * xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps))


def chunked_logits(x: torch.Tensor, head: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    """`ParallelHead.forward` in fp32, a vocab slice at a time so the fp32 copy of the
    1.32 GB head never exists all at once."""
    out = torch.empty(head.size(0), dtype=torch.float32)
    for lo in range(0, head.size(0), chunk):
        hi = min(lo + chunk, head.size(0))
        out[lo:hi] = head[lo:hi].float() @ x.float()
    return out


def _verify_positions(n: int) -> list[int]:
    """A handful of positions spread over the prompt, plus the last one.

    A free-running greedy continuation would need a decode loop with live KV caches
    across all 40 layers, i.e. another 510 GB read per generated token -- the exact
    thing this file is built to avoid. Teacher-forced argmax at several positions
    tests the same property (the stack produces sane, confident logits on English,
    Chinese and code) for the price of the single pass we already made.
    """
    if n <= 6:
        return list(range(n - 1))
    return sorted({n // 4, n // 2, 3 * n // 4, n - 2, n - 1})


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return verify(args) if args.verify else trace(args)


if __name__ == "__main__":
    sys.exit(main())
