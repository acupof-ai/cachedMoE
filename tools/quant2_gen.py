#!/usr/bin/env python3
"""Track F5: free-running generation under a re-quantised expert bank.

Teacher forcing (`tools/quant2_l3.py`) answers "does the next-token distribution
move", which is the right question for a *bounded* error but not for a visible
one: a model can score a corpus reasonably and still produce text nobody would
ship, because teacher forcing re-anchors it on the real token at every step and
never lets an error compound. This tool removes the anchor. It prefills a short
prompt and then greedily decodes N tokens, feeding the model its own output,
with the routed experts re-quantised exactly as `quant2_l3.py` does them.

The point is a sample a human can read.

    python tools/quant2_gen.py --variant fp4       --tokens 64
    python tools/quant2_gen.py --variant int2_b32  --tokens 64

The decode loop is `tools/oracle.py:level3`'s arithmetic with the export writer
taken out -- `_l3_run_layer` per layer, `_l3_collapse_and_head` for the logits --
and one structural change: the forty Blocks are built once and kept rather than
rebuilt per layer per step, so the KV ring stays where prefill left it and the
save/reload dance goes away. Nothing about routing, attention or the shared
expert changes.

Why there is a cache
--------------------
Decode reads 6 experts x 40 layers per token. Re-quantising one is ~0.24 s, so a
step is ~58 s and 64 steps would be an hour *per prompt per scheme* -- and it is
the same few thousand experts over and over, because that reuse is the entire
premise of the expert cache this project is built around. So a fitted expert is
kept, and kept in the **proposed on-disk format** of `docs/p4_quant2.md` section
7: 2-bit codes packed 4 to a byte, the UE8M0-style block scale, and the per-row
codebook, unpacked back to bf16 on every use.

That is deliberate. It means these samples are not generated from a float
shortcut -- every weight the model sees has been through the packed
representation a real implementation would store, so the format and the
dequantisation are exercised by the same run that produces the text. The
round trip is asserted exact against the direct fit on the first expert of every
run.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import quant2_common as q2                                     # noqa: E402
from quant2_l1 import SCHEMES                                  # noqa: E402
from quant2_l3 import torch_uniform_quantise                   # noqa: E402

MODEL = r"D:\models\DeepSeek-V4.1-Flash"

PROMPTS = {
    "en_tech": "The expert cache holds about thirty percent of the routed "
               "experts, so the decode loop is bounded by",
    "en_story": "The old lighthouse keeper had one rule for the winter months:",
    "code": "def hit_rate(hits: int, total: int) -> float:\n    return",
}


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# the packed representation (docs/p4_quant2.md section 7)
# --------------------------------------------------------------------------- #

def fit_codes(w, block: int, n_levels: int, steps):
    """The same fit as `quant2_l3.torch_uniform_quantise`, returning the *codes*
    and the block scale instead of the reconstruction.

    Kept as a separate function rather than a flag on that one so the value path
    used by the teacher-forced rung stays exactly as it was measured.
    """
    import torch
    r, k = w.shape
    vb = w.float().reshape(r, k // block, block)
    amax = vb.abs().amax(dim=2, keepdim=True)
    base = torch.exp2(torch.floor(torch.log2(amax.clamp_min(1e-30))))
    vn = vb * base.reciprocal()
    top = float(n_levels - 1)
    delta = 2.0 / top

    best_sse = best_code = best_pow = None
    for st in steps:
        u = vn * (2.0 ** -st)
        c = torch.clamp(torch.floor((u + 1.0) * (top / 2.0) + 0.5), 0.0, top)
        d = c * delta - 1.0 - u
        sse = d.mul_(d).sum(dim=2, keepdim=True) * (4.0 ** st)
        if best_sse is None:
            best_sse, best_code = sse, c
            best_pow = torch.full_like(sse, 2.0 ** st)
        else:
            take = sse < best_sse
            best_sse = torch.where(take, sse, best_sse)
            best_code = torch.where(take, c, best_code)
            best_pow = torch.where(take, torch.full_like(sse, 2.0 ** st), best_pow)
    return best_code.to(torch.uint8).reshape(r, k), (base * best_pow).squeeze(-1)


def pack(codes, bits: int):
    """`codes` [r, k] uint8 -> [r, k * bits / 8] uint8, element j in the LOWEST
    bits of its byte, which is the convention the FP4 unpack already uses."""
    import torch
    per = 8 // bits
    r, k = codes.shape
    v = codes.reshape(r, k // per, per)
    out = torch.zeros((r, k // per), dtype=torch.uint8)
    for i in range(per):
        out |= v[:, :, i] << (bits * i)
    return out


def unpack_dequant(packed, scale, bits: int, n_levels: int, block: int, k: int):
    """The reference dequantiser, in torch: the whole point of the format is that
    this is a shift, a mask, a table lookup and a multiply."""
    import torch
    per = 8 // bits
    mask = (1 << bits) - 1
    r = packed.shape[0]
    parts = [(packed >> (bits * i)) & mask for i in range(per)]
    codes = torch.stack(parts, dim=-1).reshape(r, k)
    # the codebook is the symmetric uniform grid, so the lookup is arithmetic
    vals = codes.float() * (2.0 / (n_levels - 1)) - 1.0
    return (vals.reshape(r, k // block, block)
            * scale.unsqueeze(-1)).reshape(r, k).to(torch.bfloat16)


class ExpertCache:
    """LRU over (layer, expert) holding the packed form, ~10 MB an expert at 2
    bits against the 71 MB the bf16 weights would take -- which is what makes a
    cache big enough to span a decode step's 240 experts affordable at all."""

    def __init__(self, capacity: int, block: int, n_levels: int, steps):
        self.cap = capacity
        self.block = block
        self.n_levels = n_levels
        self.bits = 2 if n_levels <= 4 else 4
        self.steps = steps
        self.d: OrderedDict = OrderedDict()
        self.hits = self.misses = 0
        self.fit_s = self.unpack_s = 0.0
        self.checked = False

    def get(self, key, mats):
        """`mats` is a callable returning (w1, w2, w3) bf16, only called on a miss."""
        import torch
        if key in self.d:
            self.d.move_to_end(key)
            self.hits += 1
            entry = self.d[key]
        else:
            self.misses += 1
            t0 = time.perf_counter()
            entry = []
            for w in mats():
                codes, scale = fit_codes(w, self.block, self.n_levels, self.steps)
                entry.append((pack(codes, self.bits), scale, w.shape[1]))
                if not self.checked:
                    # the round trip, once per run, against the direct fit
                    back = unpack_dequant(entry[-1][0], scale, self.bits,
                                          self.n_levels, self.block, w.shape[1])
                    direct = torch_uniform_quantise(w, self.block, self.n_levels,
                                                    self.steps)
                    assert torch.equal(back, direct), (
                        "the packed round trip does not reproduce the direct fit")
                    self.checked = True
            self.fit_s += time.perf_counter() - t0
            if self.cap:
                self.d[key] = entry
                while len(self.d) > self.cap:
                    self.d.popitem(last=False)
        t0 = time.perf_counter()
        out = tuple(unpack_dequant(p, s, self.bits, self.n_levels, self.block, k)
                    for p, s, k in entry)
        self.unpack_s += time.perf_counter() - t0
        return out

    def stats(self) -> dict:
        n = self.hits + self.misses
        return {"loads": n, "hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / max(n, 1), 4),
                "entries": len(self.d), "capacity": self.cap,
                "fit_seconds": round(self.fit_s, 1),
                "unpack_seconds": round(self.unpack_s, 1)}


def wrap_expert_stream(store, cache: "ExpertCache | None"):
    """Route `expert_stream` -- the one call that knows both the layer and the
    expert id -- through the cache. Returns the original for restoration."""
    orig = store.expert_stream

    def patched(layer, experts, dim, inter, depth: int = 2):
        for e in list(experts):
            def mats(_l=layer, _e=e):
                return store.expert_weights(_l, _e, dim, inter)
            yield (e, *cache.get((layer, e), mats))

    if cache is not None:
        store.expert_stream = patched
    return orig


# --------------------------------------------------------------------------- #

def generate(args, ref, dsref, oracle, store, margs, layout_e, ngram, tokenizer,
             ids: list[int], n_steps: int, cache) -> dict:
    import torch

    shared = ref.shared_attn
    shared.compress_kv = shared.index_k = shared.topk_idxs = shared.candidates = None
    embed = store.tensor("embed.weight")
    norm_w = store.tensor("norm.weight")
    head_w = store.tensor("head.weight")

    def embed_of(tokens, pos0):
        t = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        h = embed[t[0]].unsqueeze(1).repeat(1, margs.hc_mult, 1).unsqueeze(0).to(torch.bfloat16)
        return h, ngram(t, pos0, None)

    # The forty Blocks are built ONCE and kept. `tools/oracle.py:level3` rebuilds
    # them per layer per step and carries the attention state across in a dict,
    # because it is written to run inside the memory a layer-streaming runtime
    # would have; here that would mean re-reading 5.1 GB of attention weights on
    # every one of the 64 steps, 326 GB of I/O to avoid holding 5.1 GB on a 64 GB
    # machine. Prefilling and then decoding through the same Block object is the
    # reference-correct pattern either way -- it is exactly what
    # `oracle.level2_layer` does, and it makes the save/reload of the KV ring
    # unnecessary because the ring simply stays where prefill wrote it.
    t0 = time.perf_counter()
    blocks = [dsref.make_block(ref, margs, L, layout_e, store, args.engram_threads)
              for L in range(margs.n_layers)]
    log(f"  {margs.n_layers} blocks built in {time.perf_counter() - t0:.0f}s")

    t0 = time.perf_counter()
    h, hashes = embed_of(ids, 0)
    pre_mix = ref.make_identity_pre_mix(h, margs.hc_mult)
    for L in range(margs.n_layers):
        h, pre_mix, _n, _ = oracle._l3_run_layer(dsref, store, margs, layout_e, L,
                                                 blocks[L], h, pre_mix, hashes, 0)
    logits, _ = oracle._l3_collapse_and_head(h, pre_mix, norm_w, head_w, margs.norm_eps)
    tok = int(logits.argmax().item())
    log(f"  prefill {len(ids)} tokens in {time.perf_counter() - t0:.0f}s -> {tok}")

    produced = [tok]
    steps_log = []
    for s in range(n_steps):
        pos = len(ids) + s
        ts = time.perf_counter()
        h, hashes = embed_of([produced[-1]], pos)
        pre_mix = ref.make_identity_pre_mix(h, margs.hc_mult)
        for L in range(margs.n_layers):
            h, pre_mix, _n, _ = oracle._l3_run_layer(dsref, store, margs, layout_e, L,
                                                     blocks[L], h, pre_mix, hashes, pos)
        logits, _ = oracle._l3_collapse_and_head(h, pre_mix, norm_w, head_w,
                                                 margs.norm_eps)
        nxt = int(logits.argmax().item())
        produced.append(nxt)
        steps_log.append({"pos": pos, "token": nxt,
                          "logit": round(float(logits.max()), 3),
                          "seconds": round(time.perf_counter() - ts, 1)})
        if (s + 1) % 8 == 0 or s == 0:
            log(f"  step {s + 1}/{n_steps} @{pos} -> {nxt} "
                f"({time.perf_counter() - ts:.0f}s/step"
                + (f", cache {cache.stats()['hit_rate']:.2f}" if cache else "") + ")")
    del blocks
    out_ids = produced[:-1] if len(produced) > n_steps else produced
    return {"prompt_ids": ids, "prompt_text": tokenizer.decode(ids),
            "generated_ids": out_ids,
            "generated_text": tokenizer.decode(out_ids),
            "full_text": tokenizer.decode(list(ids) + list(out_ids)),
            "steps": steps_log,
            "seconds": round(time.perf_counter() - t0, 1)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", default="fp4")
    p.add_argument("--prompts", default="en_tech,en_story,code")
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--traces", default=r"C:\Users\Asus\code\deepmoe\traces")
    p.add_argument("--steps", default="-2,-1,0,1")
    p.add_argument("--cache-experts", type=int, default=700,
                   help="LRU entries, packed form (~10 MB each at 2 bits)")
    p.add_argument("--threads", type=int, default=10)
    p.add_argument("--engram-threads", type=int, default=16)
    p.add_argument("--out", default="bench/results/quant2")
    a = p.parse_args(argv)

    import torch
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    import dsref
    import oracle
    from oracle_longctx import _NoEngramTable

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    t_all = time.perf_counter()
    inference_dir = os.path.join(a.model, "inference")
    ref = dsref.load_reference(inference_dir)
    ref.ParallelEngramEmbedding = _NoEngramTable
    store = dsref.WeightStore(a.model, None)
    tokenizer = dsref.TokenizerAdapter(os.path.join(a.model, "tokenizer.json"))

    names = [s for s in a.prompts.split(",") if s]
    prompts = {n: tokenizer.encode(PROMPTS[n]) for n in names}
    max_seq_len = max(len(v) for v in prompts.values()) + a.tokens + 8
    margs = dsref.build_args(ref, inference_dir, max_seq_len=max_seq_len)
    layout_e = ref.EngramLayout.from_args(margs)
    sys.path.insert(0, inference_dir)
    import engram as eng                                        # noqa: E402
    cached = dsref.CachedTokenMap.build(
        tokenizer, os.path.join(a.traces, "longctx", "token_map.npz"))
    orig_build = eng.build_compressed_token_map
    eng.build_compressed_token_map = lambda _t: (cached.lookup, cached.size)
    try:
        ngram = ref.NgramHashState(margs, layout_e, tokenizer)
    finally:
        eng.build_compressed_token_map = orig_build

    cache = None
    if a.variant != "fp4":
        cfg = SCHEMES[a.variant]
        if cfg.get("scheme") != "int2" or cfg.get("fp4_frac") or cfg.get("act_aware"):
            raise SystemExit(f"--variant {a.variant}: fixed-codebook schemes only")
        cache = ExpertCache(a.cache_experts, cfg.get("block", 32),
                            cfg.get("n_levels", 4),
                            [float(s) for s in a.steps.split(",")])
        wrap_expert_stream(store, cache)
    log(f"variant={a.variant} prompts={names} tokens={a.tokens} "
        f"threads={a.threads} cache={a.cache_experts if cache else 0}")

    runs = {}
    for name, ids in prompts.items():
        log(f"{name}: {len(ids)} prompt tokens")
        runs[name] = generate(a, ref, dsref, oracle, store, margs, layout_e,
                              ngram, tokenizer, ids, a.tokens, cache)
        log(f"{name}: {runs[name]['seconds']}s\n---\n"
            f"{runs[name]['full_text']}\n---")

    blob = {"generator": "tools/quant2_gen.py", "variant": a.variant,
            "started": started, "ended": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seconds": round(time.perf_counter() - t_all, 1),
            "tokens_per_prompt": a.tokens, "scale_steps": a.steps,
            "cache": cache.stats() if cache else None,
            "note": "greedy, free-running; weights go through the packed 2-bit "
                    "format of docs/p4_quant2.md section 7 on every use",
            "runs": runs}
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"gen_{a.variant}.json")
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=1, ensure_ascii=False)
    log(f"-> {path}")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
