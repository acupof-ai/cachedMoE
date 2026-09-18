#!/usr/bin/env python3
"""Track F5: is the 2-bit error concentrated anywhere a cheaper scheme could reach?

Every mitigation on the table -- keep the worst x% of rows at FP4, spend the
bits where the activations are large, use a finer block -- is a bet that the
damage is *concentrated*: that some small, identifiable subset of the weights
carries most of the error and can be bought back cheaply. This tool measures
whether that bet has anything behind it, at the two granularities a format could
actually exploit:

  * **rows** (the unit a mixed-precision scheme switches on, and the unit that
    owns a per-row LUT), and
  * **blocks of 32** (the unit that owns a scale).

For each, it sorts by squared error and reports what share of the total error the
worst 5 / 12.5 / 25 / 50% carry. A concentrated tensor puts most of its error in
the first few percent; a uniform one tracks the diagonal, and then a mixed scheme
that protects x% of rows can only ever remove about x% of the error -- which is
the arithmetic that decides whether `mix05` / `mix12` / `mix25` in
`tools/quant2_l1.py` were ever going to work.

It also reports the code histogram and Shannon entropy of the checkpoint's own
FP4 code stream, which is the same question one level up: if the codes were
concentrated, a 2-bit *lossless* recoding would exist and none of this would be
necessary.

    python tools/quant2_concentration.py --experts 0:0,10:77,20:191,30:300,39:383
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
from quant2_l1 import SCHEMES, expert_fp32, fp4_codes          # noqa: E402

MODEL = r"D:\models\DeepSeek-V4.1-Flash"
FRACTIONS = (0.05, 0.125, 0.25, 0.5)


def shares(err: np.ndarray) -> dict[str, float]:
    """Share of the total squared error held by the worst `f` of the units.

    The diagonal -- `share == f` -- is what perfect uniformity looks like, and is
    the null hypothesis every mixed-precision scheme is betting against.
    """
    e = np.sort(err.astype(np.float64))[::-1]
    tot = max(e.sum(), 1e-300)
    return {f"worst_{f:g}": round(float(e[:max(1, int(f * len(e)))].sum() / tot), 4)
            for f in FRACTIONS}


def analyse_matrix(w: np.ndarray, name: str, scheme: str, block: int) -> dict:
    q = q2.quantise_tensor(w, name=name, **SCHEMES[scheme])
    d2 = (q.w_hat.astype(np.float64) - w) ** 2

    row_err = d2.sum(axis=1)
    row_energy = (w.astype(np.float64) ** 2).sum(axis=1)
    row_rel = np.sqrt(row_err / np.maximum(row_energy, 1e-300))

    nb = w.shape[1] // block
    blk_err = d2.reshape(w.shape[0], nb, block).sum(axis=2).reshape(-1)

    return {"matrix": name, "shape": list(w.shape),
            "rel_l2": round(q2.rel_l2(q.w_hat, w), 6),
            "rows": {"n": int(w.shape[0]), **shares(row_err),
                     "rel_l2_mean": round(float(row_rel.mean()), 5),
                     "rel_l2_sd": round(float(row_rel.std()), 5),
                     "rel_l2_p1": round(float(np.percentile(row_rel, 1)), 5),
                     "rel_l2_p99": round(float(np.percentile(row_rel, 99)), 5),
                     "rel_l2_max": round(float(row_rel.max()), 5)},
            "blocks32": {"n": int(blk_err.size), **shares(blk_err)}}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--experts", default="0:0,10:77,20:191,30:300,39:383")
    p.add_argument("--scheme", default="int2_b32")
    p.add_argument("--out", default="bench/results/quant2")
    a = p.parse_args(argv)

    import oracle
    reader = oracle.ManifestReader(a.model)
    table = oracle.fp4_e2m1_table()
    block = SCHEMES[a.scheme].get("block", 32)

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.perf_counter()
    hist = np.zeros(16, dtype=np.int64)
    records = []
    for spec in a.experts.split(","):
        L, E = (int(x) for x in spec.split(":"))
        mats = dict(zip(("w1", "w2", "w3"), expert_fp32(reader, L, E, table)))
        for w in mats.values():
            hist += np.bincount(fp4_codes(w).ravel(), minlength=16)
        rec = {"layer": L, "expert": E,
               "matrices": [analyse_matrix(mats[m], m, a.scheme, block)
                            for m in ("w1", "w2", "w3")]}
        records.append(rec)
        print(f"L{L:02d} e{E:03d}: " + "  ".join(
            f"{m['matrix']} rel_l2 {m['rel_l2']:.4f} worst25%rows "
            f"{m['rows']['worst_0.25']:.4f}" for m in rec["matrices"]), flush=True)

    p_code = hist / max(hist.sum(), 1)
    nz = p_code[p_code > 0]
    entropy = float(-(nz * np.log2(nz)).sum())
    mags = q2.FP4_MAGS
    codes = [{"code": i, "value": float((-1 if i >= 8 else 1) * mags[i % 8]),
              "count": int(hist[i]), "share": round(float(p_code[i]), 6)}
             for i in range(16)]

    blob = {"generator": "tools/quant2_concentration.py", "scheme": a.scheme,
            "started": started, "ended": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seconds": round(time.perf_counter() - t0, 1),
            "fractions": list(FRACTIONS),
            "note": "share == fraction means perfectly uniform: the worst x% of "
                    "units carry x% of the error, and protecting them buys x%",
            "fp4_code_stream": {
                "weights": int(hist.sum()), "entropy_bits": round(entropy, 4),
                "codes_used": int((hist > 0).sum()),
                "entropy_ceiling_bits": round(float(np.log2((hist > 0).sum())), 4),
                "top4_share": round(float(np.sort(p_code)[::-1][:4].sum()), 4),
                "histogram": codes},
            "records": records}
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "concentration.json")
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=1)

    print(f"\nFP4 code stream: {entropy:.4f} bits/weight over {hist.sum():,} weights, "
          f"{int((hist > 0).sum())} codes used (ceiling "
          f"{np.log2((hist > 0).sum()):.4f}); top 4 codes = "
          f"{100 * np.sort(p_code)[::-1][:4].sum():.2f}% of mass")
    print(f"\n{a.scheme}: share of squared error held by the worst fraction "
          f"(uniform would equal the fraction)")
    print(f"{'unit':10s}" + "".join(f"{f:>10.3g}" for f in FRACTIONS))
    for unit in ("rows", "blocks32"):
        agg = {f: float(np.mean([m[unit][f"worst_{f:g}"]
                                 for r in records for m in r["matrices"]]))
               for f in FRACTIONS}
        print(f"{unit:10s}" + "".join(f"{agg[f]:10.4f}" for f in FRACTIONS))
    rs = [m["rows"] for r in records for m in r["matrices"]]
    print(f"\nper-row rel L2 across all matrices: mean "
          f"{np.mean([x['rel_l2_mean'] for x in rs]):.4f}, "
          f"sd {np.mean([x['rel_l2_sd'] for x in rs]):.4f}, "
          f"p1 {np.mean([x['rel_l2_p1'] for x in rs]):.4f}, "
          f"p99 {np.mean([x['rel_l2_p99'] for x in rs]):.4f}")
    print(f"-> {path}")
    reader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
