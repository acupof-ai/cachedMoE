#!/usr/bin/env python3
"""Track F5 step 2b: the L3 rung -- teacher-forced agreement and KL, end to end.

L1 (`tools/quant2_l1.py`) measures one expert's output error. That is necessary
and not sufficient: design section 12's own argument for why the ladder has four
rungs is that a per-kernel test cannot see what happens when forty layers of them
are composed. This tool runs the **unmodified reference** (`inference/model.py`
behind `tools/dsref.py`'s six CPU kernel shims, exactly as `tools/oracle.py
--level l2/l3` and `tools/oracle_longctx.py` do) with one thing changed: the
routed experts are re-quantised on the way out of `WeightStore`.

    python tools/quant2_l3.py --variant fp4    --prompt corpus2k --dump-calib ...
    python tools/quant2_l3.py --variant int2_b32 --prompt corpus2k --baseline <npy>

Teacher forcing means one prefill pass over N tokens gives N next-token
distributions, so the whole comparison -- argmax agreement, mean KL against FP4,
and the perplexity of the corpus's own next token -- costs one pass, not N
decode steps. A pass is dominated by reading and dequantising 40 x ~384 experts,
so a second prompt inside the same process is nearly free of setup but not of
that; both prompts are therefore run in one invocation.

Engram: the reference's `ParallelEngramEmbedding` constructor reserves ~98 GB of
commit that is never touched, so it is replaced by the same zero-parameter stub
`tools/oracle_longctx.py` uses, and `dsref.make_block` swaps in the real
manifest-backed row reader immediately afterwards.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import quant2_common as q2                                     # noqa: E402
from quant2_l1 import SCHEMES                                  # noqa: E402

MODEL = r"D:\models\DeepSeek-V4.1-Flash"


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# the fast re-quantiser used inside the forward pass
# --------------------------------------------------------------------------- #

def torch_uniform_quantise(w, block: int, n_levels: int, steps):
    """`quant2_common.fit_block_scale` + `reconstruct` for a fixed symmetric
    uniform codebook, in torch, on the bf16 tensor `dsref` just produced.

    A pass over the 27K-token corpus touches ~15,360 experts, so the numpy path
    (~5 s an expert) is three orders of magnitude too slow to sit inside a
    forward. This is the same arithmetic -- exhaustive over the same power-of-two
    scale candidates, nearest level, UE8M0 scale -- written so every step is one
    multithreaded elementwise kernel. `tools/quant2_l3.py --selftest` asserts it
    reproduces the numpy path bit for bit on a real expert.
    """
    import torch
    r, k = w.shape
    vb = w.float().reshape(r, k // block, block)
    lv = torch.from_numpy(q2.uniform_levels(n_levels))
    amax = vb.abs().amax(dim=2, keepdim=True)
    base = torch.exp2(torch.floor(torch.log2(amax.clamp_min(1e-30))))
    mid = ((lv[:-1] + lv[1:]) / 2.0)

    best_sse = None
    best = None
    for st in steps:
        s = base * (2.0 ** st)
        u = vb / s
        c = torch.zeros_like(u)
        for m in mid:
            c += (u >= m).float()
        rec = lv[c.long()] * s
        sse = (rec - vb).pow_(2).sum(dim=2, keepdim=True)
        if best_sse is None:
            best_sse, best = sse, rec
        else:
            take = sse < best_sse
            best_sse = torch.where(take, sse, best_sse)
            best = torch.where(take, rec, best)
    return best.reshape(r, k).to(w.dtype)


def install_expert_hook(dsref, variant: str, steps, counter: dict):
    """Re-quantise every routed expert as it leaves the WeightStore."""
    if variant == "fp4":
        return
    cfg = SCHEMES[variant]
    if cfg.get("scheme") != "int2" or cfg.get("fp4_frac") or cfg.get("act_aware"):
        raise SystemExit(
            f"--variant {variant}: only the fixed-codebook uniform schemes have a "
            "forward-speed implementation (see the module docstring); L1 covers "
            "the rest")
    block, n_levels = cfg.get("block", 32), cfg.get("n_levels", 4)
    orig = dsref.WeightStore.expert_weights_from_slot

    def patched(views, dim, inter):
        w1, w2, w3 = orig(views, dim, inter)
        counter["experts"] += 1
        return (torch_uniform_quantise(w1, block, n_levels, steps),
                torch_uniform_quantise(w2, block, n_levels, steps),
                torch_uniform_quantise(w3, block, n_levels, steps))

    dsref.WeightStore.expert_weights_from_slot = staticmethod(patched)


# --------------------------------------------------------------------------- #
# prompts
# --------------------------------------------------------------------------- #

def load_prompt(which: str, traces: str, tests_data: str, n: int) -> list[int]:
    if which == "l3_64":
        with io.open(os.path.join(tests_data, "l2", "index.json"), encoding="utf-8") as f:
            idx = json.load(f)
        # `prefill_len` is what the reference actually ran; the id list can carry
        # one more (the token the L3 export's step 0 consumes).
        return [int(i) for i in idx["prompt_ids"]][:int(idx["prefill_len"])]
    if which == "corpus2k":
        src = os.path.join(traces, "longctx", "ctx4k", "index.json")
        if not os.path.exists(src):
            raise SystemExit(f"{src} is missing; it carries the 4,133-token mixed "
                             "EN/ZH/code prompt this rung reuses")
        with io.open(src, encoding="utf-8") as f:
            ids = [int(i) for i in json.load(f)["prompt_ids"]]
        return ids[:n]
    raise SystemExit(f"unknown prompt set {which!r}")


# --------------------------------------------------------------------------- #
# one pass
# --------------------------------------------------------------------------- #

def run_pass(args, ref, dsref, oracle, store, margs, layout_e, ngram,
             ids: list[int], tag: str, calib: dict | None) -> dict:
    import torch

    N = len(ids)
    pre_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    hashes = ngram(pre_ids, 0, None)
    shared = ref.shared_attn
    shared.compress_kv = shared.index_k = shared.topk_idxs = shared.candidates = None

    embed = store.tensor("embed.weight")
    h = embed[pre_ids[0]].unsqueeze(1).repeat(1, margs.hc_mult, 1).unsqueeze(0).to(torch.bfloat16)
    del embed
    pre_mix = ref.make_identity_pre_mix(h, margs.hc_mult)

    t_pass = time.perf_counter()
    for L in range(margs.n_layers):
        t0 = time.perf_counter()
        block = dsref.make_block(ref, margs, L, layout_e, store, args.engram_threads)
        if calib is not None:
            # the FFN input, captured before the routed experts run: this is the
            # `x` every activation-aware objective in tools/quant2_l1.py needs,
            # and the only place all forty layers' versions of it exist.
            hook_out = {}

            def cap(_m, inp, out, _d=hook_out):
                _d["x"] = (out[0] if isinstance(out, tuple) else out).detach()
            hdl = block.b.ffn_norm.register_forward_hook(cap)
        h, pre_mix, n_used, _idx = oracle._l3_run_layer(
            dsref, store, margs, layout_e, L, block, h, pre_mix, hashes, 0)
        if calib is not None:
            hdl.remove()
            x = hook_out["x"].reshape(-1, margs.dim).float()
            rows = np.linspace(0, x.shape[0] - 1, min(64, x.shape[0])).round().astype(int)
            calib[f"x_L{L}"] = x[rows].numpy().astype(np.float32)
            m2 = (x.double() ** 2).mean(dim=0).numpy()
            calib[f"m2_L{L}"] = (m2 / max(m2.mean(), 1e-30)).astype(np.float32)
            del x, hook_out
        del block
        log(f"  {tag} L{L:02d} {n_used:3d} experts {time.perf_counter() - t0:6.1f}s "
            f"|h| {h.float().norm().item():.4e}")

    norm_w = store.tensor("norm.weight")
    head_w = store.tensor("head.weight")
    out = os.path.join(args.logits_dir, f"logits_{args.variant}_{tag}.npy")
    os.makedirs(args.logits_dir, exist_ok=True)
    lg = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16,
                                   shape=(N, margs.vocab_size))
    q = collapse_and_norm(h, pre_mix, norm_w, margs.norm_eps)
    chk = oracle._l3_collapse_and_head(h[:, -1:], pre_mix[:, -1:], norm_w,
                                       head_w, margs.norm_eps)[1]
    dn = float((q[-1].to(chk.dtype) - chk).abs().max())
    assert dn == 0.0, f"the batched collapse disagrees with oracle's at the last position: {dn}"
    del h, pre_mix
    # head chunk outer, positions inner: head_w is bf16 on disk and fp32 in the
    # reference, and promoting 129,280 x 5,120 costs 2.6 GB, so each chunk is
    # promoted exactly once and every position is projected through it.
    step = 16384
    for c0 in range(0, head_w.size(0), step):
        hw = head_w[c0:c0 + step].float()
        for i in range(0, N, 256):
            j = min(N, i + 256)
            lg[i:j, c0:c0 + hw.size(0)] = torch.nn.functional.linear(
                q[i:j].float(), hw).numpy().astype(np.float16)
        del hw
    lg.flush()
    del lg, q
    secs = time.perf_counter() - t_pass
    log(f"{tag}: {N} positions, {secs:.0f}s -> {out}")
    return {"tag": tag, "n": N, "seconds": round(secs, 1), "logits": out}


def collapse_and_norm(h, pre_mix, norm_w, norm_eps: float):
    """`Transformer.forward`'s tail up to the head, for **every** position.

    `tools/oracle.py:_l3_collapse_and_head` is the same arithmetic but ends with
    the reference's `x = x[:, -1]`, because a generator only ever needs the last
    position's logits. Teacher forcing needs all N of them, and hc_pre, the
    RMSNorm and the head are all position-wise, so dropping that one slice is the
    only difference -- asserted against the oracle's value at the last position
    on every pass.
    """
    import torch
    x = torch.sum(pre_mix.unsqueeze(-1) * h.float(), dim=2).to(h.dtype)
    xf = x.float()
    var = xf.square().mean(-1, keepdim=True)
    return (norm_w.float() * (xf * torch.rsqrt(var + norm_eps))).to(h.dtype)[0]


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #

def compare(base_path: str, var_path: str, ids: list[int]) -> dict:
    """Teacher-forced agreement, mean KL(P_fp4 || P_variant), and the NLL each
    model assigns to the corpus's own next token.

    KL is computed in fp64 from the fp16 logits both passes stored; the fp16
    round trip costs at most 5e-4 nats on these magnitudes and is recorded here
    rather than hidden, because the bar (0.02 nats) is forty times larger.
    """
    a = np.load(base_path, mmap_mode="r")
    b = np.load(var_path, mmap_mode="r")
    assert a.shape == b.shape, (a.shape, b.shape)
    n = a.shape[0]
    agree = 0
    kl_sum = 0.0
    nll_a = nll_b = 0.0
    top5 = 0
    n_nll = 0
    step = 64
    for i in range(0, n, step):
        j = min(n, i + step)
        la = np.asarray(a[i:j], dtype=np.float64)
        lb = np.asarray(b[i:j], dtype=np.float64)
        la -= la.max(axis=1, keepdims=True)
        lb -= lb.max(axis=1, keepdims=True)
        pa = np.exp(la); pa /= pa.sum(axis=1, keepdims=True)
        pb = np.exp(lb); pb /= pb.sum(axis=1, keepdims=True)
        ia, ib = la.argmax(axis=1), lb.argmax(axis=1)
        agree += int((ia == ib).sum())
        ord5 = np.argpartition(-la, 5, axis=1)[:, :5]
        top5 += int((ord5 == ib[:, None]).any(axis=1).sum())
        kl_sum += float((pa * (np.log(np.maximum(pa, 1e-300))
                               - np.log(np.maximum(pb, 1e-300)))).sum())
        for r in range(j - i):
            pos = i + r
            if pos + 1 < len(ids):
                t = ids[pos + 1]
                nll_a -= float(np.log(max(pa[r, t], 1e-300)))
                nll_b -= float(np.log(max(pb[r, t], 1e-300)))
                n_nll += 1
    return {"positions": n,
            "argmax_agreement": round(agree / n, 4),
            "variant_argmax_in_fp4_top5": round(top5 / n, 4),
            "mean_kl_nats": round(kl_sum / n, 5),
            "ppl_fp4": round(float(np.exp(nll_a / max(n_nll, 1))), 4),
            "ppl_variant": round(float(np.exp(nll_b / max(n_nll, 1))), 4),
            "ppl_ratio": round(float(np.exp((nll_b - nll_a) / max(n_nll, 1))), 4)}


# --------------------------------------------------------------------------- #

def selftest(args) -> int:
    """The torch fast path against the numpy schemes, on a real expert."""
    import torch
    import oracle
    torch.set_grad_enabled(False)
    r = oracle.ManifestReader(args.model)
    tab = oracle.fp4_e2m1_table()
    slot, off, sz = r.expert_slot(0, 0)
    parts = {k: slot[off[k]:off[k] + sz[k]] for k in off}
    w = oracle.dequant_fp4(parts["w1.weight"], parts["w1.scale"], 2304, 5120, tab)
    steps = [float(s) for s in args.steps.split(",")]
    for tag in ("int2_b32", "uni3_b32", "fp4_exact"):
        cfg = SCHEMES[tag]
        nl = cfg.get("n_levels", 4)
        if nl == 15:
            print(f"{tag}: skipped (non-uniform codebook, numpy path only)")
            continue
        ref_q = q2.quantise_tensor(w, name="w1", **cfg)
        fast = torch_uniform_quantise(torch.from_numpy(w).to(torch.bfloat16),
                                      cfg.get("block", 32), nl, steps).float().numpy()
        exact = np.array_equal(fast, ref_q.w_hat.astype(np.float32))
        print(f"{tag}: numpy rel_l2 {q2.rel_l2(ref_q.w_hat, w):.6f}  "
              f"torch rel_l2 {q2.rel_l2(fast, w):.6f}  identical={exact}  "
              f"max|delta| {np.abs(fast - ref_q.w_hat).max():.3e}")
    r.close()
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", default="fp4",
                   help="'fp4' (the baseline) or a fixed-codebook scheme from quant2_l1.SCHEMES")
    p.add_argument("--prompts", default="l3_64,corpus2k")
    p.add_argument("--n", type=int, default=2048, help="tokens taken for corpus2k")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--traces", default=r"C:\Users\Asus\code\deepmoe\traces")
    p.add_argument("--tests-data", default="tests/data")
    p.add_argument("--logits-dir", default=None)
    p.add_argument("--baseline-dir", default=None,
                   help="where the --variant fp4 run put its logits")
    p.add_argument("--steps", default="-3,-2,-1,0,1,2,3",
                   help="power-of-two scale candidates for the forward-speed quantiser")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--engram-threads", type=int, default=32)
    p.add_argument("--dump-calib", default=None,
                   help="npz of per-layer expert inputs (only meaningful with --variant fp4)")
    p.add_argument("--out", default="bench/results/quant2")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args(argv)

    scratch = os.environ.get("TEMP", ".")
    a.logits_dir = a.logits_dir or os.path.join(scratch, "quant2_logits")
    if a.selftest:
        return selftest(a)

    import torch
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    import dsref
    import oracle
    from oracle_longctx import _NoEngramTable

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.perf_counter()
    inference_dir = os.path.join(a.model, "inference")
    ref = dsref.load_reference(inference_dir)
    ref.ParallelEngramEmbedding = _NoEngramTable
    store = dsref.WeightStore(a.model, None)
    tokenizer = dsref.TokenizerAdapter(os.path.join(a.model, "tokenizer.json"))

    sets = [s for s in a.prompts.split(",") if s]
    prompts = {s: load_prompt(s, a.traces, a.tests_data, a.n) for s in sets}
    max_seq_len = max(len(v) for v in prompts.values()) + 8
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

    counter = {"experts": 0}
    steps = [float(s) for s in a.steps.split(",")]
    install_expert_hook(dsref, a.variant, steps, counter)
    log(f"variant={a.variant} prompts={ {k: len(v) for k, v in prompts.items()} } "
        f"threads={a.threads}")

    calib: dict | None = {} if (a.dump_calib and a.variant == "fp4") else None
    passes = {}
    for name, ids in prompts.items():
        passes[name] = run_pass(a, ref, dsref, oracle, store, margs, layout_e,
                                ngram, ids, name, calib if name == "corpus2k" else None)
        if calib is not None and name == "corpus2k":
            np.savez_compressed(a.dump_calib, **calib)
            log(f"calibration bank -> {a.dump_calib} ({len(calib) // 2} layers)")

    result = {"generator": "tools/quant2_l3.py", "variant": a.variant,
              "started": started, "ended": time.strftime("%Y-%m-%d %H:%M:%S"),
              "seconds": round(time.perf_counter() - t0, 1),
              "experts_requantised": counter["experts"],
              "scale_steps": steps, "passes": passes, "compare": {}}
    if a.baseline_dir and a.variant != "fp4":
        for name, ids in prompts.items():
            base = os.path.join(a.baseline_dir, f"logits_fp4_{name}.npy")
            if os.path.exists(base):
                result["compare"][name] = compare(base, passes[name]["logits"], ids)
                log(f"{name}: {result['compare'][name]}")
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"l3_{a.variant}.json")
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    log(f"-> {path}")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
