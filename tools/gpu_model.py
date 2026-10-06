"""A per-dispatch model of the decode step: what each stage SHOULD cost, next to
what the trace says it DID cost.

    # constants (once per machine / driver): bench/probes/model_probe.cpp
    build/model_probe --json bench/results/linux/gpu_model/constants.json
    # a hot-step trace with dispatch geometry (runtime/trace.cpp's DMGEOM01 block)
    python tools/gpu_model.py --capture            # or --trace FILE

The model is a roofline with a per-workgroup cap, every term measured by the
probe rather than taken from a data sheet:

    t = t0 + max( cold_bytes / min(W * cold_per_wg, cold_peak),     weights, DRAM
                  hot_bytes  / min(W * hot_per_wg,  hot_peak),      KV / activations, L2/MALL
                  flops      / fma_peak )                           arithmetic

  t0            the shortest timed dispatch (launch + drain, with the trace's
                own timestamps around it -- the same instrument as the trace)
  cold_per_wg   one workgroup streaming cold bytes (~29 GB/s): why a stage with
                few workgroups cannot reach the DRAM ceiling however it is coded
  cold_peak     the best cold stream at any W (~232 GB/s)
  hot_*         the same for bytes that are already in L2 / the 32 MB MALL

W is the stage's workgroup count from the trace. What the model does NOT
contain is a stage's serial chain -- barriers and dependent loads inside one
workgroup -- so a stage far above its model is either a serial-chain problem
(few workgroups, many barriers: gate.topk before 2026-09-28) or a stage whose
bytes or FLOPs the table below undercounts. Either way the GAP is the most a
rewrite of that stage could win, ranked by what it costs per token.

`--n-kv` sets the sparse-attention list length (window + compressed picks);
the hot step runs the 64-token L3 prompt, so its lists are short.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import perf_report as pr  # noqa: E402
from trace_timeline import Trace  # noqa: E402

CONSTANTS = ROOT / "bench" / "results" / "linux" / "gpu_model" / "constants.json"

HEADS, HEAD_DIM = 64, 512


def load_constants(path: Path) -> dict:
    c = json.loads(path.read_text())
    cold, hot = c["stream_cold"], c["stream_hot"]
    big = lambda rows: [r for r in rows if r["bytes"] >= 60e6]
    one = lambda rows, lo, hi: [r for r in rows if r["groups"] == 1 and lo <= r["bytes"] <= hi]
    rate = lambda r: r["bytes"] / (r["us"] * 1e-6)
    return {
        "t0_us": min(r["us"] for r in cold + hot),
        "cold_per_wg": max(rate(r) for r in one(cold, 1 << 20, 1 << 30)),
        "cold_peak": max(rate(r) for r in big(cold)),
        "hot_per_wg": max(rate(r) for r in one(hot, 1 << 20, 16 << 20)),
        "hot_peak": max(rate(r) for r in hot if r["bytes"] <= 16 << 20),
        "fma": c["fma_tflops"] * 1e12,
        "barrier_ns": c.get("barrier_ns"),
        "dispatch_chain_us": c.get("dispatch_barrier_us"),
    }


def hot_bytes_flops(name: str, n_kv: int) -> tuple[float, float]:
    """Non-weight bytes (already in cache) and FLOPs of one call of `name`."""
    kv_row = HEAD_DIM + HEAD_DIM // 32            # fp8 + UE8M0 per position
    if name == "sparse_attn.score":
        return n_kv * kv_row + HEADS * HEAD_DIM * 2 + HEADS * n_kv * 4, 2.0 * HEADS * n_kv * HEAD_DIM
    if name == "sparse_attn.combine":
        return n_kv * kv_row + HEADS * n_kv * 4 + HEADS * HEAD_DIM * 4, 2.0 * HEADS * n_kv * HEAD_DIM
    # decode_attn_cm: G and P are fp16 planes, S and O fp32; the two GEMMs are
    # fp16 tiles, counted at the fp32 rate (an upper bound on their time).
    g16, e = n_kv * HEAD_DIM * 2, (n_kv + 15) // 16 * 16
    if name == "attn_cm.gather":
        return n_kv * kv_row + HEADS * HEAD_DIM * 2 + g16 + HEADS * HEAD_DIM * 2, 0.0
    if name == "attn_cm.score":
        return g16 + HEADS * HEAD_DIM * 2 + HEADS * e * 4, 2.0 * HEADS * e * HEAD_DIM
    if name == "attn_cm.softmax":
        return HEADS * e * (4 + 2), 0.0
    if name == "attn_cm.pv":
        return g16 + HEADS * e * 2 + 4 * HEADS * HEAD_DIM * 4, 2.0 * HEADS * e * HEAD_DIM
    if name == "attn_cm.finish":
        return 5 * HEADS * HEAD_DIM * 4, 0.0
    return 0.0, 0.0


def model_us(k: dict, cold: float, hot: float, flops: float, groups: int, disp: int) -> tuple[float, str]:
    w = max(groups, 1)
    t_cold = cold / min(w * k["cold_per_wg"], k["cold_peak"]) * 1e6
    t_hot = hot / min(w * k["hot_per_wg"], k["hot_peak"]) * 1e6
    t_fma = flops / k["fma"] * 1e6
    body, bound = max((t_cold, "DRAM"), (t_hot, "cache"), (t_fma, "FMA"))
    if body < 0.05:
        bound = "launch"
    return k["t0_us"] * max(disp, 1) + body, bound


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", type=Path)
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--constants", type=Path, default=CONSTANTS)
    ap.add_argument("--model", default=pr.DEFAULT_MODEL)
    ap.add_argument("--n-kv", type=int, default=128)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    k = load_constants(a.constants)
    trace = a.trace
    if a.capture:
        trace = Path(tempfile.mkdtemp()) / "hot.bin"
        pr.capture(a.model, a.warm, trace, {})
    if not trace:
        ap.error("--trace or --capture")
    man = pr.load_manifest(a.model)
    h = pr.hot_step(trace, man, k["cold_peak"] / 1e9)
    t = Trace(trace.read_bytes())
    geom = {t.names.get(key, str(key)): g for key, g in t.geom.items()}
    if not geom:
        sys.exit("this trace has no DMGEOM01 block (it predates runtime/trace.cpp's geometry); re-capture")

    rows = []
    for r in h["rows"]:
        name, n = r["stage"], r["n"]
        groups, disp = geom.get(name, (1, 1))
        cold = r["mb"] * 1e6 / n
        hot, flops = hot_bytes_flops(name, a.n_kv)
        m, bound = model_us(k, cold, hot, flops, groups, disp)
        meas = r["ms"] * 1e3 / n
        rows.append({"stage": name, "cls": r["cls"], "n": n, "groups": groups, "dispatches": disp,
                     "meas_us": meas, "model_us": m, "bound": bound,
                     "gap_ms_tok": (meas - m) * n / 1e3, "cold_MB": cold / 1e6})
    rows.sort(key=lambda x: -x["gap_ms_tok"])

    tot_meas = sum(x["meas_us"] * x["n"] for x in rows) / 1e3
    tot_model = sum(x["model_us"] * x["n"] for x in rows) / 1e3
    print(f"== constants ({a.constants.name}): t0 {k['t0_us']:.2f} us, cold {k['cold_per_wg'] / 1e9:.0f} GB/s a "
          f"workgroup up to {k['cold_peak'] / 1e9:.0f}, hot {k['hot_per_wg'] / 1e9:.0f} up to "
          f"{k['hot_peak'] / 1e9:.0f}, fp32 {k['fma'] / 1e12:.1f} TFLOP/s")
    print(f"== hot step ({trace.name}, median of {h['used']} passes): kernels {tot_meas:.2f} ms measured vs "
          f"{tot_model:.2f} modelled; + {h['gaps']:.2f} ms between dispatches (submit boundaries and "
          f"barriers) = span {h['span']:.2f}")
    print(f"   {'stage':22s} {'W':>6s} {'d':>2s} {'calls':>5s} {'MB/call':>8s} {'meas us':>8s} {'model':>7s} "
          f"{'bound':>6s} {'x':>5s} {'gap ms/tok':>10s}")
    for x in rows[:a.top]:
        print(f"   {x['stage']:22s} {x['groups']:6d} {x['dispatches']:2d} {x['n']:5d} {x['cold_MB']:8.2f} "
              f"{x['meas_us']:8.1f} {x['model_us']:7.1f} {x['bound']:>6s} {x['meas_us'] / x['model_us']:5.1f} "
              f"{x['gap_ms_tok']:10.2f}")
    rest = rows[a.top:]
    if rest:
        print(f"   ({len(rest)} more: {sum(x['gap_ms_tok'] for x in rest):.2f} ms gap)")
    by_bound = defaultdict(float)
    for x in rows:
        by_bound[x["bound"]] += x["gap_ms_tok"]
    print("   gap by what bounds the model: " + ", ".join(f"{b} {v:.2f} ms" for b, v in
                                                        sorted(by_bound.items(), key=lambda kv: -kv[1])))
    if a.json:
        a.json.write_text(json.dumps({"constants": k, "trace": str(trace), "rows": rows,
                                      "kernels_ms": tot_meas, "model_ms": tot_model,
                                      "gaps_ms": h["gaps"], "span_ms": h["span"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
