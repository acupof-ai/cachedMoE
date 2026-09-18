#!/usr/bin/env python3
"""Track Y step 4: a >= 64-step teacher-forced L3 oracle set.

Why this exists
---------------
`tools/oracle.py --level l3` exports **8** greedy decode steps, and
docs/p4_resident_routing.md section 8.6 says in as many words that those eight
steps are the only thing left standing between Track Y and a verdict: `off`,
`all` and `stall1` all diverge on the same two positions (steps 6 and 7), so the
PPL *ratio* the bars are written against is two already-diverged positions'
probability mass and nothing else. A x2.5 and a x3.4 cannot be told apart on it.

What this exports
-----------------
The same container `runtime/decode_state.h` already loads, for a ~64-token
natural-text prompt plus **64** reference-greedy continuation tokens:

  prefill   the whole attention state the prompt leaves behind (every layer's
            window KV ring; on each kv_source_layer the compressed-KV cache,
            the indexer key cache and the compressor's carried group state) --
            byte for byte what `oracle.py --level l3` writes, because the engine
            seeds from it unchanged -- plus the logits at the last prompt
            position.
  stepNN    ONLY the logit evidence: the reference's top-64 (id, logit), the
            three whole-vector statistics, the input token, the argmax and the
            log-probability of that argmax. No per-step compressed KV and no
            top-k list: the engine has had its own section 7.4 kernels since
            Track Q (`Engine::produce_ced()`), so the per-step tensors are never
            read, and dropping them takes a 64-step export from ~45 MB to 6 MB.

How the continuation is produced, and what was tried first
---------------------------------------------------------
`phase greedy` prefills the prompt and then runs the reference's own decode
loop one step at a time -- the same loop `oracle.py --level l3` has always run,
so every step's attention state is built incrementally exactly the way the
engine builds its own. 64 steps at ~6 s each plus a ~180 s prefill is about
10 minutes.

The obvious cheaper route was tried first and is kept here as `phase prefill` +
`phase steps`: teacher forcing means ONE prefill over prompt+continuation yields
every next-token distribution at once, so the 64 steps would cost a single
~227 s pass (Track F5's `tools/quant2_l3.py` makes the same argument). It needs
the continuation up front, and the continuation is by definition the reference's
own greedy path -- so the ENGINE free-ran a candidate in 21 s on the GPU and the
pass was to prove it: position 63's argmax must be c[0], position 64+s's argmax
must be c[s+1]. Measured: 5 of 64.

Then the same pass was run on the continuation `phase greedy` had produced --
the reference's OWN incrementally decoded greedy path -- and it diverged at the
same index, on the same token (` either` against the decode loop's ` about`).
That is the real result, and it is why the shortcut is dead rather than merely
inconvenient:

    the reference's one-pass prefill at position j and its incremental decode
    at position j are NOT the same distribution.

The compressor's carried tail group and the indexer's top-k over the whole
compressed cache are not per-position causal in a single prefill, so a one-pass
argmax at an INTERIOR position neither defines the greedy path nor is the right
target for a teacher-forced NLL -- the engine decodes incrementally, and the
reference it is scored against has to as well. For the record the engine was the
more faithful of the two: free-running, it stayed on the reference's decode-loop
greedy path for **12** tokens against the one-pass argmax's 5. The two phases are
kept as the probe that measured this.

Usage
-----
    .venv/Scripts/python.exe tools/oracle_l3_ppl.py greedy
        --model D:/models/DeepSeek-V4.1-Flash --out traces/l3_64 --steps 64

`index.json` is rewritten after every step, so a run stopped early leaves a
shorter but complete and loadable export behind. `tools/l3_ppl.py` then measures
the engine on it.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import oracle  # noqa: E402

# A natural-text prompt, 64 tokens exactly under tokenizer.json. Expository
# prose on purpose: its greedy continuation is 64 tokens of coherent narration
# (reference PPL 1.82), which is what a PPL ruler needs -- and it is not the L2
# prompt's mixture of English, Chinese and Python, whose greedy continuation ran
# into `<|end of sentence|>` after nine tokens and is why `tests/data/l3` stops
# at eight steps.
PPL_PROMPT = (
    "The Rhine has always been two things at once: a border and a road. For most of its "
    "length it divides one polity from another, and yet the barges that move down it each "
    "morning tie those same polities together more tightly than any treaty. Historians of "
    "the river have tended to choose one of these two"
)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _read_json(path: str) -> dict:
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, doc: dict):
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# the reference, set up exactly the way oracle.py --level l3 sets it up
# --------------------------------------------------------------------------- #

class Reference:
    """`inference/model.py` behind `tools/dsref.py`'s CPU kernel shims.

    Everything here is lifted from `oracle.level3_end_to_end`; it is a class
    only so the two phases can each build it and the engram token map can be
    cached on disk between them.
    """

    def __init__(self, model: str, manifest: str, out: str, max_seq_len: int,
                 engram_threads: int):
        import dsref
        torch.set_grad_enabled(False)
        self.dsref = dsref
        self.inference_dir = os.path.join(model, "inference")
        self.ref = dsref.load_reference(self.inference_dir)
        self.store = dsref.WeightStore(model, manifest)
        self.tokenizer = dsref.TokenizerAdapter(os.path.join(model, "tokenizer.json"))
        self.margs = dsref.build_args(self.ref, self.inference_dir, max_seq_len=max_seq_len)
        self.layout_e = self.ref.EngramLayout.from_args(self.margs)
        self.engram_threads = engram_threads
        os.makedirs(out, exist_ok=True)

        # The engram writes into the residual stream at layers 1 and 14, so a
        # run without it is a different model; its constructor reserves ~98 GB
        # of commit it never touches, so the compressed token map is built once
        # and cached next to the export (oracle.py does the same).
        sys.path.insert(0, self.inference_dir)
        import engram as eng  # noqa: E402
        cached = dsref.CachedTokenMap.build(self.tokenizer,
                                            os.path.join(out, "token_map.npz"))
        orig = eng.build_compressed_token_map
        eng.build_compressed_token_map = lambda _t: (cached.lookup, cached.size)
        try:
            self.ngram = self.ref.NgramHashState(self.margs, self.layout_e, self.tokenizer)
        finally:
            eng.build_compressed_token_map = orig

        self.embed = self.store.tensor("embed.weight")
        self.norm_w = self.store.tensor("norm.weight")
        self.head_w = self.store.tensor("head.weight")

    def embed_of(self, tokens: list[int], pos0: int):
        t = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        h = self.embed[t[0]].unsqueeze(1).repeat(1, self.margs.hc_mult, 1) \
                            .unsqueeze(0).to(torch.bfloat16)
        return t, h, self.ngram(t, pos0, None)

    def reset_shared(self):
        s = self.ref.shared_attn
        s.compress_kv = s.index_k = s.topk_idxs = s.candidates = None

    def prefill(self, ids: list[int], per_layer_state: bool):
        """One pass over the forty layers for the whole prompt at position 0.

        Returns (h, pre_mix, kv_state, prefill_tensors). `per_layer_state` also
        collects the window ring / compressed cache / index keys / group state
        the engine seeds from -- phase `steps` does not need them.
        """
        self.reset_shared()
        _t, h, hashes = self.embed_of(ids, 0)
        pre_mix = self.ref.make_identity_pre_mix(h, self.margs.hc_mult)
        tensors: dict = {}
        kv_state: dict[int, dict] = {}
        for L in range(self.margs.n_layers):
            t0 = time.perf_counter()
            block = self.dsref.make_block(self.ref, self.margs, L, self.layout_e,
                                          self.store, self.engram_threads)
            h, pre_mix, n_used, _ = oracle._l3_run_layer(
                self.dsref, self.store, self.margs, self.layout_e, L, block,
                h, pre_mix, hashes, 0)
            if per_layer_state:
                kv_state[L] = oracle._l2_save_attn_state(block)
                tensors[f"L{L:02d}.win_kv"] = (
                    "bf16", block.b.attn.window_kv_cache[0].clone())
                tensors.update(oracle._l3_prefill_state(block, L))
                tensors[f"L{L:02d}.gate_bias"] = (
                    "f32", block.b.ffn.gate.bias.float().clone())
            del block
            print(f"    layer {L:2d}  {n_used:3d} experts  "
                  f"{time.perf_counter() - t0:5.1f}s  |h| {h.float().norm().item():.4e}",
                  flush=True)
        return h, pre_mix, kv_state, tensors

    def logits_at(self, h, pre_mix, positions: list[int], chunk: int = 16384):
        """`Transformer.forward`'s tail at several positions at once.

        `oracle._l3_collapse_and_head` keeps only the last position because a
        greedy decode has only one; teacher forcing wants all of them, and the
        1.3 GB bf16 head must be promoted to fp32 ONCE for the batch rather than
        once per position.
        """
        x = torch.sum(pre_mix.unsqueeze(-1) * h.float(), dim=2).to(h.dtype)  # hc_pre
        xp = x[0, positions]                                                # [P, dim]
        xf = xp.float()
        var = xf.square().mean(-1, keepdim=True)
        normed = (self.norm_w.float() * (xf * torch.rsqrt(var + self.margs.norm_eps))) \
            .to(h.dtype)
        q = normed.float()
        parts = [torch.nn.functional.linear(q, self.head_w[i:i + chunk].float())
                 for i in range(0, self.head_w.size(0), chunk)]
        return torch.cat(parts, dim=-1), normed                             # [P, vocab]


def _logit_record(row: torch.Tensor, top_k: int) -> dict:
    """`oracle._l3_logit_record` plus the two numbers this harness is for.

    `ref_logprob` is log p of the reference's OWN argmax at this position: the
    per-step term of the reference's mean NLL, which is the gate the engine's
    `off` mode has to reproduce before any ratio measured against it means
    anything. It is `max - logsumexp` and is stored rather than recomputed only
    so a reader of the export never has to know that.
    """
    rec = oracle._l3_logit_record(row, top_k)
    lse = float(torch.logsumexp(row.float(), dim=-1))
    rec["ref_logprob"] = ("f32", torch.tensor([float(row.max()) - lse]))
    rec["argmax"] = ("i32", torch.tensor([int(row.argmax())], dtype=torch.int32))
    return rec


# --------------------------------------------------------------------------- #
# phase 1: the prompt's state
# --------------------------------------------------------------------------- #

def phase_prefill(args) -> int:
    max_seq_len = max(128, args.prompt_tokens + args.candidate + 8)
    r = Reference(args.model, args.manifest, args.out, max_seq_len, args.engram_threads)
    ids = r.tokenizer.encode(PPL_PROMPT)[: args.prompt_tokens]
    if len(ids) < 32:
        raise SystemExit(f"the prompt tokenises to only {len(ids)} tokens")
    log(f"prefill: {len(ids)} prompt tokens, max_seq_len {max_seq_len}, out -> {args.out}")

    t0 = time.perf_counter()
    h, pre_mix, _kv, tensors = r.prefill(ids, per_layer_state=True)
    logits, normed = r.logits_at(h, pre_mix, [len(ids) - 1])
    tok = int(logits[0].argmax())
    tensors.update(_logit_record(logits[0], args.topk))
    tensors["collapse_in"] = ("bf16", h[0, -1].contiguous())
    tensors["pre_mix_final"] = ("f32", pre_mix[0, -1].contiguous())
    tensors["norm_out"] = ("bf16", normed[0].contiguous())

    w = oracle.L3Writer(args.out)
    rec = w.write("prefill", tensors)
    log(f"prefill -> {rec['file']} ({rec['bytes'] / 1e6:.2f} MB), next token {tok}")

    # `DecodeState::load` refuses an export with steps_exported == 0, and the
    # engine has to load THIS export to free-run the candidate continuation, so
    # phase 1 leaves a one-step bootstrap behind. `phase steps` overwrites it.
    boot = _logit_record(logits[0], args.topk)
    boot["in_token"] = ("i32", torch.tensor([tok], dtype=torch.int32))
    w.write("step00", boot)

    meta = _meta(r, ids, [tok, tok], 1, args, max_seq_len)
    meta["bootstrap"] = True
    meta["notes"] = ("BOOTSTRAP. Only the prefill record is real; step00 is a copy of "
                     "the prefill logits so that DecodeState::load accepts the export "
                     "and the engine can free-run a candidate continuation from the "
                     "prompt state. Run `oracle_l3_ppl.py steps` to replace it.")
    path = w.finish(meta)
    log(f"wrote {path} in {time.perf_counter() - t0:.1f}s; "
        f"engram tables -> {args.out}/engram_token_map.bin")
    _write_json(os.path.join(args.out, "prompt_ids.json"),
                {"prompt_ids": [int(i) for i in ids], "first_token": tok,
                 "prompt": PPL_PROMPT})
    return 0


def _meta(r: Reference, ids: list[int], greedy: list[int], steps: int, args,
          max_seq_len: int) -> dict:
    m = r.margs
    return {
        "version": oracle.L3_VERSION,
        "model": "DeepSeek-V4.1-Flash",
        "generator": "tools/oracle_l3_ppl.py",
        "prompt": PPL_PROMPT,
        "prompt_ids": [int(i) for i in ids],
        "prefill_len": len(ids),
        "decode_pos": len(ids),
        "steps_exported": steps,
        "top_k": args.topk,
        "greedy_tokens": [int(t) for t in greedy],
        "text": r.tokenizer.decode(greedy),
        "engram": oracle._l3_engram_tables(r.ngram, r.layout_e, m, args.out),
        "config": {"dim": m.dim, "hc_mult": m.hc_mult, "n_heads": m.n_heads,
                   "head_dim": m.head_dim, "window_size": m.window_size,
                   "vocab_size": m.vocab_size, "n_layers": m.n_layers,
                   "norm_eps": m.norm_eps,
                   "compress_ratios": list(m.compress_ratios),
                   "kv_source_layers": list(m.kv_source_layers),
                   "index_source_layers": list(m.index_source_layers),
                   "max_seq_len": max_seq_len},
    }


# --------------------------------------------------------------------------- #
# phase 2: one teacher-forced pass over prompt + candidate
# --------------------------------------------------------------------------- #

def phase_steps(args) -> int:
    idx_path = os.path.join(args.out, "index.json")
    if not os.path.exists(idx_path):
        raise SystemExit(f"{idx_path} is missing -- run `phase prefill` first")
    old = _read_json(idx_path)
    prefill_rec = next(s for s in old["steps"] if s["step"] == "prefill")
    ids = [int(i) for i in old["prompt_ids"]]

    cand = _read_json(args.continuation)
    cont = [int(t) for t in (cand["tokens"] if isinstance(cand, dict) else cand)]
    n = args.steps
    if len(cont) < n:
        raise SystemExit(f"--continuation has {len(cont)} tokens, need {n}")
    cont = cont[:n]

    seq = ids + cont
    max_seq_len = max(128, len(seq) + 8)
    r = Reference(args.model, args.manifest, args.out, max_seq_len, args.engram_threads)
    log(f"steps: one teacher-forced pass over {len(seq)} tokens "
        f"({len(ids)} prompt + {n} candidate continuation)")

    t0 = time.perf_counter()
    h, pre_mix, _kv, _t = r.prefill(seq, per_layer_state=False)
    # Position 63 predicts c[0]; position 64+s predicts c[s+1]. That is 1 + n
    # distributions, and the last one (position len(seq)-1) is the reference's
    # own continuation of the whole forced sequence -- greedy_tokens[n], the
    # target of the last decode step.
    positions = list(range(len(ids) - 1, len(seq)))
    logits, _normed = r.logits_at(h, pre_mix, positions)
    log(f"pass done in {time.perf_counter() - t0:.1f}s; "
        f"{logits.size(0)} next-token distributions")

    argmax = [int(logits[i].argmax()) for i in range(logits.size(0))]
    chain = [cont[i] for i in range(n)]            # what the argmaxes must be
    agree = 0
    while agree < n and argmax[agree] == chain[agree]:
        agree += 1
    greedy = argmax[: agree + 1] if agree < n else argmax[:]    # [c0..c_{n-1}, c_n]
    log(f"greedy-chain proof: the candidate is the reference's own greedy path for "
        f"{agree}/{n} tokens")
    if agree < n:
        log(f"  first divergence at index {agree}: candidate {chain[agree]} "
            f"({r.tokenizer.decode([chain[agree]])!r}), reference {argmax[agree]} "
            f"({r.tokenizer.decode([argmax[agree]])!r})")
        _write_json(os.path.join(args.out, "repair.json"),
                    {"tokens": chain[:agree] + [argmax[agree]], "diverged_at": agree})
        if not args.allow_short:
            raise SystemExit(
                f"only {agree} of {n} candidate tokens are on the reference's greedy "
                f"path; repair.json holds the corrected prefix -- re-run the engine "
                f"from it and repeat, or pass --allow-short to export {agree} steps")

    steps_out = len(greedy) - 1
    # Rebuild the index: the prefill record is reused byte for byte (its file is
    # untouched), the bootstrap step00 is replaced.
    w = oracle.L3Writer(args.out)
    w.steps.append(prefill_rec)
    nll = 0.0
    for s in range(steps_out):
        row = logits[s + 1]                       # position len(ids)+s
        t = _logit_record(row, args.topk)
        t["in_token"] = ("i32", torch.tensor([greedy[s]], dtype=torch.int32))
        w.write(f"step{s:02d}", t)
        nll -= float(t["ref_logprob"][1][0])
    ref_nll = nll / steps_out

    meta = _meta(r, ids, greedy, steps_out, args, max_seq_len)
    meta["reference_nll"] = ref_nll
    meta["reference_ppl"] = math.exp(ref_nll)
    meta["candidate_agree"] = agree
    meta["notes"] = (
        "Teacher-forced. One reference prefill over prompt + continuation gives every "
        "next-token distribution at once, so the 64 steps cost one pass and not 64 "
        "decode passes. 'prefill' is the state the 64-token prompt leaves behind, "
        "written by `phase prefill` and byte for byte what `oracle.py --level l3` "
        "writes. Each 'stepNN' holds ONLY logit evidence for position prefill_len+s: "
        "the top-64 (id, logit), the whole-vector max/logsumexp/min, the input token "
        "greedy_tokens[s], the argmax greedy_tokens[s+1] and its log-probability. The "
        "per-step compressed KV and top-k list are deliberately absent -- the engine "
        "produces them itself (Engine::produce_ced) and loading them would cost 40 MB. "
        "greedy_tokens IS the argmax chain this pass measured: argmax at position "
        "prefill_len-1+s equals greedy_tokens[s] for every s, which is what makes the "
        "forced sequence the reference's own greedy continuation. reference_nll is the "
        "mean of -ref_logprob over the exported steps -- the number the engine's `off` "
        "mode has to reproduce.")
    path = w.finish(meta)
    total = sum(s["bytes"] for s in w.steps)
    log(f"wrote {len(w.steps)} records, {total / 1e6:.2f} MB -> {path}")
    log(f"reference NLL {ref_nll:.6f} over {steps_out} steps -> PPL "
        f"{math.exp(ref_nll):.4f}")
    log("continuation: " + repr(r.tokenizer.decode(greedy)))
    return 0


# --------------------------------------------------------------------------- #
# phase greedy: the reference's own decode loop, one step at a time
# --------------------------------------------------------------------------- #

def phase_greedy(args) -> int:
    """Prefill the prompt, then N reference greedy decode steps, exporting each.

    Why not one teacher-forced pass
    -------------------------------
    A single prefill over prompt+continuation yields every next-token
    distribution at once and would cost one pass instead of N. It was tried, and
    it is wrong here for a reason that outlives the chicken-and-egg: run on the
    continuation THIS phase produced, the one-pass argmax leaves the decode
    loop's own greedy path at index 5 of 64 (docs/p4_resident_routing.md section
    9.1). A one-pass prefill's interior positions are not the distribution an
    incremental decode produces at those positions, so they are neither the
    greedy path nor the right target for a teacher-forced NLL against an engine
    that decodes one token at a time. This loop is the same one `oracle.py
    --level l3` has always run, so every step's state is built incrementally
    exactly the way the engine builds its own.

    Every step is written as it is produced and `index.json` is rewritten after
    each one, so a run that is stopped early leaves a shorter but complete and
    loadable export behind.
    """
    max_seq_len = max(128, args.prompt_tokens + args.steps + 8)
    r = Reference(args.model, args.manifest, args.out, max_seq_len, args.engram_threads)
    ids = r.tokenizer.encode(PPL_PROMPT)[: args.prompt_tokens]
    if len(ids) < 32:
        raise SystemExit(f"the prompt tokenises to only {len(ids)} tokens")
    log(f"greedy: {len(ids)} prompt tokens + {args.steps} decode steps, "
        f"max_seq_len {max_seq_len}, out -> {args.out}")

    t0 = time.perf_counter()
    h, pre_mix, kv_state, tensors = r.prefill(ids, per_layer_state=True)
    logits, normed = r.logits_at(h, pre_mix, [len(ids) - 1])
    tok = int(logits[0].argmax())
    tensors.update(_logit_record(logits[0], args.topk))
    tensors["collapse_in"] = ("bf16", h[0, -1].contiguous())
    tensors["pre_mix_final"] = ("f32", pre_mix[0, -1].contiguous())
    tensors["norm_out"] = ("bf16", normed[0].contiguous())
    w = oracle.L3Writer(args.out)
    rec = w.write("prefill", tensors)
    log(f"prefill {time.perf_counter() - t0:.1f}s -> {rec['file']} "
        f"({rec['bytes'] / 1e6:.2f} MB), next token {tok}")

    greedy = [tok]
    nll = 0.0
    for s in range(args.steps):
        pos = len(ids) + s
        in_tok = greedy[-1]
        ts = time.perf_counter()
        _dt, h, hashes = r.embed_of([in_tok], pos)
        pre_mix = r.ref.make_identity_pre_mix(h, r.margs.hc_mult)
        for L in range(r.margs.n_layers):
            block = r.dsref.make_block(r.ref, r.margs, L, r.layout_e, r.store,
                                       r.engram_threads)
            oracle._l2_load_attn_state(block, kv_state[L])
            h, pre_mix, _n, _idx = oracle._l3_run_layer(
                r.dsref, r.store, r.margs, r.layout_e, L, block, h, pre_mix, hashes, pos)
            kv_state[L] = oracle._l2_save_attn_state(block)
            del block
        logits, _n = r.logits_at(h, pre_mix, [0])
        nxt = int(logits[0].argmax())
        t = _logit_record(logits[0], args.topk)
        t["in_token"] = ("i32", torch.tensor([in_tok], dtype=torch.int32))
        w.write(f"step{s:02d}", t)
        nll -= float(t["ref_logprob"][1][0])
        greedy.append(nxt)
        # Rewrite the index every step: a run killed at step k leaves a k-step
        # export that loads.
        meta = _meta(r, ids, greedy, s + 1, args, max_seq_len)
        meta["reference_nll"] = nll / (s + 1)
        meta["reference_ppl"] = math.exp(nll / (s + 1))
        meta["notes"] = _GREEDY_NOTES
        w.finish(meta)
        print(f"  step {s:2d} @pos {pos}: {in_tok} -> {nxt}  "
              f"({time.perf_counter() - ts:5.1f}s, logp {float(t['ref_logprob'][1][0]):+.4f})",
              flush=True)

    total = sum(x["bytes"] for x in w.steps)
    log(f"wrote {len(w.steps)} records, {total / 1e6:.2f} MB -> {args.out}/index.json "
        f"in {time.perf_counter() - t0:.1f}s")
    log(f"reference NLL {nll / args.steps:.6f} over {args.steps} steps -> PPL "
        f"{math.exp(nll / args.steps):.4f}")
    log("continuation: " + repr(r.tokenizer.decode(greedy)))
    _write_json(os.path.join(args.out, "prompt_ids.json"),
                {"prompt_ids": [int(i) for i in ids], "prompt": PPL_PROMPT,
                 "greedy_tokens": [int(t) for t in greedy]})
    return 0


_GREEDY_NOTES = (
    "Greedy decode straight out of inference/model.py behind tools/dsref.py's CPU "
    "kernel shims -- the same loop `oracle.py --level l3` runs, so every step's "
    "attention state is built incrementally exactly as the engine's is. 'prefill' is "
    "the whole attention state the prompt leaves behind (every layer's window KV ring; "
    "on each kv_source_layer the compressed-KV cache, the indexer key cache and the "
    "compressor's carried group state) plus the logits at the last prompt position, "
    "whose argmax is greedy_tokens[0] and the input to step00. Each 'stepNN' holds "
    "ONLY logit evidence: the top-64 (id, logit), the whole-vector max/logsumexp/min, "
    "the input token greedy_tokens[NN], the argmax greedy_tokens[NN+1] and its "
    "log-probability. The per-step compressed KV and top-k list are deliberately "
    "absent -- the engine produces them itself (Engine::produce_ced) and exporting "
    "them would cost 40 MB for 64 steps. reference_nll is the mean of -ref_logprob "
    "over the exported steps: the number the engine's `off` mode has to reproduce "
    "before any ratio measured against `off` means anything.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=["prefill", "steps", "greedy"])
    p.add_argument("--model", default=r"D:\models\DeepSeek-V4.1-Flash")
    p.add_argument("--manifest", default=None)
    p.add_argument("--out", default="traces/l3_64")
    p.add_argument("--prompt-tokens", type=int, default=64)
    p.add_argument("--candidate", type=int, default=80,
                   help="how far the engine will free-run (prefill: sizes max_seq_len)")
    p.add_argument("--steps", type=int, default=64, help="continuation tokens to export")
    p.add_argument("--topk", type=int, default=64)
    p.add_argument("--continuation", default=None,
                   help="JSON list (or {tokens: [...]}) of the candidate continuation")
    p.add_argument("--allow-short", action="store_true",
                   help="export the proven prefix instead of failing when the "
                        "candidate leaves the reference's greedy path early")
    p.add_argument("--engram-threads", type=int, default=32)
    args = p.parse_args()
    if args.phase == "steps" and not args.continuation:
        raise SystemExit("phase steps needs --continuation")
    return {"prefill": phase_prefill, "steps": phase_steps,
            "greedy": phase_greedy}[args.phase](args)


if __name__ == "__main__":
    raise SystemExit(main())
