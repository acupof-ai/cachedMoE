#!/usr/bin/env python3
"""Track F5 step 1-2a: the L1 rung of the 2-bit ladder.

For each sampled routed expert, dequantise the checkpoint's FP4 E2M1 + UE8M0/32
weights exactly, re-quantise them to 2 bits under each scheme, and measure

  * weight space: relative L2 and cosine, per matrix (w1 / w3 = gate / up,
    w2 = down), which is what a kernel LUT has to live with;
  * function space: the expert's own FFN output for real (or realistically
    shaped) inputs, through `tools/dsref.py:expert_ffn` so the fp8 activation
    round trip the FP4 kernel performs is in the loop too. This is the number
    that matters -- design section 12's L1 judges a kernel by its output, not by
    its weights.

Usage
-----
    python tools/quant2_l1.py screen  --experts 24  --out bench/results/quant2
    python tools/quant2_l1.py sweep   --experts 216 --schemes int2_b32,lloyd_row_b32 ...

`screen` runs the whole scheme grid on a small expert sample to pick finalists;
`sweep` runs the named schemes on >= 200 experts across all 40 layers.

CPU cost is logged (start/end wall clock) because F1/F3/F4 read machine load.
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

import quant2_common as q2                                   # noqa: E402

MODEL = r"D:\models\DeepSeek-V4.1-Flash"
DIM, INTER = 5120, 2304
N_LAYERS, N_EXPERTS = 40, 384


# --------------------------------------------------------------------------- #
# scheme grid
# --------------------------------------------------------------------------- #

SCHEMES: dict[str, dict] = {
    # (a) fixed symmetric int2, block size and scale format sweep
    "int2_b16":       dict(scheme="int2",  block=16, scale_fmt="ue8m0"),
    "int2_b32":       dict(scheme="int2",  block=32, scale_fmt="ue8m0"),
    "int2_b64":       dict(scheme="int2",  block=64, scale_fmt="ue8m0"),
    "int2_b32_e4m3":  dict(scheme="int2",  block=32, scale_fmt="e4m3"),
    # (b) Lloyd-Max codebook
    "lloyd_row_b32":  dict(scheme="lloyd", block=32, scale_fmt="ue8m0", per_row_lut=True),
    "lloyd_ten_b32":  dict(scheme="lloyd", block=32, scale_fmt="ue8m0", per_row_lut=False),
    "lloyd_row_b16":  dict(scheme="lloyd", block=16, scale_fmt="ue8m0", per_row_lut=True),
    # (c) mixed: the worst x% of rows stay FP4
    "mix05_lloyd_b32": dict(scheme="lloyd", block=32, scale_fmt="ue8m0",
                            per_row_lut=True, fp4_frac=0.05),
    "mix12_lloyd_b32": dict(scheme="lloyd", block=32, scale_fmt="ue8m0",
                            per_row_lut=True, fp4_frac=0.125),
    "mix25_lloyd_b32": dict(scheme="lloyd", block=32, scale_fmt="ue8m0",
                            per_row_lut=True, fp4_frac=0.25),
    # (d) activation-aware: same formats, objective weighted by E[x_c^2]
    "actw_lloyd_b32":  dict(scheme="lloyd", block=32, scale_fmt="ue8m0",
                            per_row_lut=True, act_aware=True),
    "actw_int2_b32":   dict(scheme="int2",  block=32, scale_fmt="ue8m0",
                            act_aware=True),
    "actw_mix05_b32":  dict(scheme="lloyd", block=32, scale_fmt="ue8m0",
                            per_row_lut=True, fp4_frac=0.05, act_aware=True),
    # controls: where is the cliff? 3 bits and a 4-bit re-quantisation of the
    # checkpoint's own FP4 (which must come out at ~0 error, and so is also the
    # harness's self-test).
    "uni3_b32":        dict(scheme="int2",  block=32, scale_fmt="ue8m0", n_levels=8),
    "lloyd3_row_b32":  dict(scheme="lloyd", block=32, scale_fmt="ue8m0",
                            per_row_lut=True, n_levels=8),
    "uni4_b32":        dict(scheme="int2",  block=32, scale_fmt="ue8m0", n_levels=16),
    # The harness self-test: 15 levels = the E2M1 magnitudes themselves. A block
    # scale of amax/6 makes this the identity, so y_rel_l2 must be 0.
    "fp4_exact":       dict(scheme="int2",  block=32, scale_fmt="ue8m0", n_levels=15),
    "lloyd4_row_b32":  dict(scheme="lloyd", block=32, scale_fmt="ue8m0",
                            per_row_lut=True, n_levels=16),
}

SCREEN = ["int2_b16", "int2_b32", "int2_b64", "int2_b32_e4m3",
          "lloyd_row_b32", "lloyd_ten_b32", "lloyd_row_b16",
          "mix05_lloyd_b32", "mix12_lloyd_b32", "mix25_lloyd_b32",
          "actw_lloyd_b32", "actw_int2_b32", "actw_mix05_b32",
          "uni3_b32", "lloyd3_row_b32", "uni4_b32", "lloyd4_row_b32", "fp4_exact"]


# --------------------------------------------------------------------------- #
# activations
# --------------------------------------------------------------------------- #

def load_l2_container(index_path: str, want: str) -> dict[int, list[np.ndarray]]:
    """Pull one tensor name out of every (layer, step) record of an L2/L3 export.

    The container is `tools/oracle.py:L2Writer`'s: a 16-byte header then the
    tensors back to back at the offsets `index.json` records.
    """
    base = os.path.dirname(index_path)
    with io.open(index_path, encoding="utf-8") as f:
        idx = json.load(f)
    out: dict[int, list[np.ndarray]] = {}
    for step in idx.get("steps", []):
        path = os.path.join(base, step["file"])
        if not os.path.exists(path):
            path = os.path.join(base, "l2", step["file"])
            if not os.path.exists(path):
                continue
        ent = next((t for t in step["tensors"] if t["name"] == want), None)
        if ent is None:
            continue
        with open(path, "rb") as f:
            f.seek(step["data_offset"] + ent["offset"])
            raw = f.read(ent["bytes"])
        if ent["dtype"] == "bf16":
            u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
            a = u.view(np.float32)
        elif ent["dtype"] == "f32":
            a = np.frombuffer(raw, dtype=np.float32)
        else:
            continue
        out.setdefault(int(step["layer"]), []).append(
            np.ascontiguousarray(a.reshape(ent["shape"]), dtype=np.float32))
    return out


class Activations:
    """Per-layer expert inputs `x` (= `ffn_norm_out`).

    Real captures come from the L2/L3 exports that already exist in the tree
    (`tests/data/l2`, and the long-context probes under `traces/longctx`), which
    cover eight of the forty layers. For every other layer the sample is built
    from the layer's own `ffn_norm.weight`: an RMSNorm output is
    `(x / rms(x)) * w`, so a unit-RMS Gaussian times `w` has the right
    per-channel scale, which is the only property of `x` a per-channel
    quantisation objective can use. `--calib` overrides both with a bank dumped
    by `tools/quant2_l3.py --dump-calib` (real inputs for all forty layers).
    """

    def __init__(self, store_tensor, traces: str, tests_data: str,
                 calib: str | None = None, seed: int = 20260917):
        self.store_tensor = store_tensor
        self.rng = np.random.default_rng(seed)
        self.real: dict[int, np.ndarray] = {}
        self.colw: dict[int, np.ndarray] = {}
        self._norm: dict[int, np.ndarray] = {}
        if calib and os.path.exists(calib):
            z = np.load(calib)
            for k in z.files:
                if k.startswith("x_L"):
                    self.real[int(k[3:])] = z[k].astype(np.float32)
                elif k.startswith("m2_L"):
                    self.colw[int(k[4:])] = z[k].astype(np.float32)
            return
        sources = [os.path.join(tests_data, "l2", "index.json")]
        for name in ("ctx4k", "ctx16k"):
            sources.append(os.path.join(traces, "longctx", name, "index.json"))
        for src in sources:
            if not os.path.exists(src):
                continue
            try:
                got = load_l2_container(src, "ffn_norm_out")
            except Exception as exc:                          # noqa: BLE001
                print(f"  (skipping {src}: {exc})")
                continue
            for L, arrs in got.items():
                a = np.stack([v.reshape(-1) for v in arrs])
                self.real[L] = (a if L not in self.real
                                else np.concatenate([self.real[L], a]))

    def norm_w(self, layer: int) -> np.ndarray:
        if layer not in self._norm:
            t = self.store_tensor(f"layers.{layer}.ffn_norm.weight")
            self._norm[layer] = np.asarray(t, dtype=np.float32).reshape(-1)
        return self._norm[layer]

    def x(self, layer: int, n: int = 8) -> tuple[np.ndarray, str]:
        if layer in self.real and len(self.real[layer]) >= 2:
            a = self.real[layer]
            take = a if len(a) <= n else a[
                np.linspace(0, len(a) - 1, n).round().astype(int)]
            return np.ascontiguousarray(take, dtype=np.float32), "real"
        w = self.norm_w(layer)
        g = self.rng.standard_normal((n, DIM)).astype(np.float32)
        g /= np.sqrt((g ** 2).mean(axis=1, keepdims=True))
        return g * w[None, :], "synthetic"

    def col_moment(self, layer: int) -> np.ndarray:
        """E[x_c^2] over the calibration sample -- the activation-aware weight."""
        if layer in self.colw:
            return self.colw[layer]
        xs, _ = self.x(layer, n=32)
        m = (xs.astype(np.float64) ** 2).mean(axis=0)
        m = m / max(m.mean(), 1e-30)
        self.colw[layer] = m.astype(np.float32)
        return self.colw[layer]


# --------------------------------------------------------------------------- #
# one expert
# --------------------------------------------------------------------------- #

def expert_fp32(reader, layer: int, expert: int, table: np.ndarray):
    """The three matrices, dequantised exactly from the checkpoint."""
    import oracle
    slot, offsets, sizes = reader.expert_slot(layer, expert)
    parts = {k: slot[offsets[k]:offsets[k] + sizes[k]] for k in offsets}
    w1 = oracle.dequant_fp4(parts["w1.weight"], parts["w1.scale"], INTER, DIM, table)
    w3 = oracle.dequant_fp4(parts["w3.weight"], parts["w3.scale"], INTER, DIM, table)
    w2 = oracle.dequant_fp4(parts["w2.weight"], parts["w2.scale"], DIM, INTER, table)
    return w1, w2, w3


def fp4_codes(w: np.ndarray) -> np.ndarray:
    """The 16 FP4 E2M1 codes a value maps to, recovered from the dequantised
    array: sign bit plus the index of |w| / 2^floor(log2 blockscale) among the
    eight magnitudes. Used only for the entropy figure, so the per-block scale is
    recovered as the block amax over 6.0 -- exact, because every value in the
    block is (magnitude x that scale)."""
    vb = w.reshape(w.shape[0], -1, q2.FP4_BLOCK)
    s = np.abs(vb).max(axis=2, keepdims=True) / 6.0
    s = np.where(s > 0, s, 1.0)
    mag = np.abs(vb) / s
    idx = np.abs(mag[..., None] - q2.FP4_MAGS).argmin(axis=-1)
    return (idx + 8 * (vb < 0)).astype(np.uint8)


def ffn_out(x: np.ndarray, w1, w2, w3, limit: float = 10.0) -> np.ndarray:
    """dsref.expert_ffn, including the fp8 activation round trip, in bf16."""
    import torch
    import dsref
    xt = torch.from_numpy(np.ascontiguousarray(x)).to(torch.bfloat16)
    y = dsref.expert_ffn(xt,
                         torch.from_numpy(w1).to(torch.bfloat16),
                         torch.from_numpy(w2).to(torch.bfloat16),
                         torch.from_numpy(w3).to(torch.bfloat16),
                         None, limit)
    return y.float().numpy()


def eval_expert(reader, acts: Activations, layer: int, expert: int,
                schemes: list[str], table: np.ndarray,
                n_x: int = 8) -> dict:
    w = dict(zip(("w1", "w2", "w3"), expert_fp32(reader, layer, expert, table)))
    xs, x_kind = acts.x(layer, n_x)
    y_ref = ffn_out(xs, w["w1"], w["w2"], w["w3"])
    colw_in = acts.col_moment(layer)

    rec = {"layer": layer, "expert": expert, "x_kind": x_kind,
           "y_ref_norm": float(np.linalg.norm(y_ref)),
           "fp4_code_entropy_bits": {
               mat: round(q2.code_entropy_bits(
                   fp4_codes(w[mat]), 16), 4) for mat in ("w1", "w2", "w3")},
           "schemes": {}}
    fit_cache: dict = {}
    for tag in schemes:
        cfg = dict(SCHEMES[tag])
        act_aware = cfg.pop("act_aware", False)
        t0 = time.perf_counter()
        qs, bits, wm = {}, [], {}
        for mat in ("w1", "w2", "w3"):
            # w1 / w3 read the block's input x; w2 reads h = silu(gate)*up, whose
            # per-channel scale is a property of this expert, not of the layer, so
            # the activation-aware objective is applied to the two that share the
            # layer's input and left flat for the down matrix (recorded, not hidden).
            colw = colw_in if (act_aware and mat in ("w1", "w3")) else None
            q = q2.quantise_tensor(w[mat], colw=colw, name=mat,
                                   fit_cache=fit_cache, **cfg)
            qs[mat] = q
            bits.append(q.bits_per_weight())
            wm[mat] = {"rel_l2": round(q2.rel_l2(q.w_hat, w[mat]), 6),
                       "cos": round(q2.cosine(q.w_hat, w[mat]), 8),
                       "code_entropy_bits": round(q.entropy_bits, 4)}
        y_q = ffn_out(xs, qs["w1"].w_hat, qs["w2"].w_hat, qs["w3"].w_hat)
        rec["schemes"][tag] = {
            "bits_per_weight": round(float(np.mean(bits)), 4),
            "weights": wm,
            "y_rel_l2": round(q2.rel_l2(y_q, y_ref), 6),
            "y_cos": round(q2.cosine(y_q, y_ref), 8),
            "seconds": round(time.perf_counter() - t0, 2),
        }
        del qs, y_q
    del w
    return rec


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def pick_experts(n: int, seed: int = 4471) -> list[tuple[int, int]]:
    """`n` (layer, expert) pairs spread over all forty layers.

    Deterministic: every layer gets the same count, and the expert ids inside a
    layer come from a seeded permutation, so two runs compare like with like.
    """
    rng = np.random.default_rng(seed)
    base, extra = divmod(n, N_LAYERS)
    out = []
    for L in range(N_LAYERS):
        k = base + (1 if L < extra else 0)
        if k == 0:
            continue
        ids = rng.permutation(N_EXPERTS)[:k]
        out += [(L, int(e)) for e in sorted(ids)]
    if base == 0:                      # fewer experts than layers: spread them out
        out = [out[i] for i in np.linspace(0, len(out) - 1, n).round().astype(int)]
    return out[:n]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["screen", "sweep"])
    p.add_argument("--model", default=MODEL)
    p.add_argument("--traces", default=r"C:\Users\Asus\code\deepmoe\traces")
    p.add_argument("--tests-data", default="tests/data")
    p.add_argument("--calib", default=None,
                   help="npz from tools/quant2_l3.py --dump-calib (real inputs, 40 layers)")
    p.add_argument("--experts", type=int, default=24)
    p.add_argument("--schemes", default=None, help="comma-separated; default = the grid")
    p.add_argument("--n-x", type=int, default=8)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--out", default="bench/results/quant2")
    p.add_argument("--tag", default=None)
    a = p.parse_args(argv)

    if a.threads:
        import torch
        torch.set_num_threads(a.threads)
    import torch
    torch.set_grad_enabled(False)

    import oracle
    import dsref
    schemes = ([s for s in a.schemes.split(",") if s] if a.schemes
               else (SCREEN if a.mode == "screen" else list(SCHEMES)))
    for s in schemes:
        if s not in SCHEMES:
            raise SystemExit(f"unknown scheme {s!r}; have {sorted(SCHEMES)}")

    reader = oracle.ManifestReader(a.model)
    store = dsref.WeightStore(a.model, None)
    table = oracle.fp4_e2m1_table()
    acts = Activations(lambda n: store.tensor(n).float().numpy(),
                       a.traces, a.tests_data, a.calib)
    print(f"real activation captures for layers {sorted(acts.real)}", flush=True)

    pairs = pick_experts(a.experts)
    os.makedirs(a.out, exist_ok=True)
    tag = a.tag or a.mode
    path = os.path.join(a.out, f"l1_{tag}.json")

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.perf_counter()
    records = []
    for i, (L, e) in enumerate(pairs):
        rec = eval_expert(reader, acts, L, e, schemes, table, a.n_x)
        records.append(rec)
        best = min(rec["schemes"], key=lambda k: rec["schemes"][k]["y_rel_l2"])
        print(f"[{i + 1:3d}/{len(pairs)}] L{L:02d} e{e:03d} {rec['x_kind']:9s} "
              f"best {best} y_rel_l2 {rec['schemes'][best]['y_rel_l2']:.4f} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)

    summary = {}
    for s in schemes:
        ys = np.array([r["schemes"][s]["y_rel_l2"] for r in records])
        cs = np.array([r["schemes"][s]["y_cos"] for r in records])
        summary[s] = {
            "bits_per_weight": records[0]["schemes"][s]["bits_per_weight"],
            "y_rel_l2_mean": round(float(ys.mean()), 6),
            "y_rel_l2_p90": round(float(np.percentile(ys, 90)), 6),
            "y_rel_l2_max": round(float(ys.max()), 6),
            "y_cos_mean": round(float(cs.mean()), 8),
            "y_cos_min": round(float(cs.min()), 8),
        }
        for mat in ("w1", "w2", "w3"):
            v = np.array([r["schemes"][s]["weights"][mat]["rel_l2"] for r in records])
            summary[s][f"{mat}_rel_l2_mean"] = round(float(v.mean()), 6)

    blob = {"generator": "tools/quant2_l1.py", "mode": a.mode, "model": a.model,
            "started": started, "ended": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seconds": round(time.perf_counter() - t0, 1),
            "n_experts": len(records), "n_x": a.n_x,
            "calib": a.calib, "real_layers": sorted(acts.real),
            "schemes": {s: SCHEMES[s] for s in schemes},
            "summary": summary, "records": records}
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=1)
    print(f"\n-> {path}  ({time.perf_counter() - t0:.0f}s)")
    print(f"{'scheme':20s} {'bits':>6s} {'y_relL2':>9s} {'p90':>9s} {'max':>9s} {'cos_min':>10s}")
    for s, v in sorted(summary.items(), key=lambda kv: kv[1]["y_rel_l2_mean"]):
        print(f"{s:20s} {v['bits_per_weight']:6.3f} {v['y_rel_l2_mean']:9.5f} "
              f"{v['y_rel_l2_p90']:9.5f} {v['y_rel_l2_max']:9.5f} {v['y_cos_min']:10.6f}")
    reader.close()
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
