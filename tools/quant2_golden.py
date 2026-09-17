#!/usr/bin/env python3
"""Track F5: golden packed bytes + dequant reference for `tests/data/quant2/`.

If the 2-bit format is ever implemented, the first thing that needs pinning is
the same thing design section 12's L0 pins for FP4: the *bytes*. This writes, for
a few real routed experts, a slice of each matrix in the proposed 2-bit format
together with the fp32 values a correct dequantiser must produce, so a C++
`tests/test_dequant2.cpp` can be written against it without re-deriving anything.

The format (proposed; docs/p4_quant2.md section 5 argues for it)
----------------------------------------------------------------
Per tensor `[rows, K]`, K contiguous, exactly the FP4 path's layout with a
narrower payload:

    codes   [rows, K/4] u8   4 codes per byte, element j in bits 2*(j % 4),
                            element 0 in the LOWEST bits -- the same
                            low-nibble-first convention the FP4 unpack uses
    scales  [rows, K/32] u8 UE8M0, value = 2^(code - 127); unchanged from FP4
    lut     [rows, 4] f32   the per-row codebook, in ascending order

    value(r, j) = lut[r][code(r, j)] * 2^(scale[r][j / 32] - 127)

Only the payload and the LUT differ from FP4; the scale plane is bit-identical
in shape and meaning, which is the point -- the kernel change is the unpack, not
the addressing.

    python tools/quant2_golden.py --out tests/data/quant2
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import quant2_common as q2                                     # noqa: E402
from quant2_l1 import SCHEMES                                  # noqa: E402

MAGIC = 0x51_32_47_31                                          # "Q2G1"
VERSION = 1
MODEL = r"D:\models\DeepSeek-V4.1-Flash"
DIM, INTER = 5120, 2304

# A slice, not a whole expert: a full 2-bit expert is 9.95 MB and the directory
# budget is 5 MB. 64 rows is enough to exercise every code, both nibble halves of
# a byte, and a per-row LUT that actually varies row to row.
ROWS = 64


def write_expert(reader, layer: int, expert: int, scheme: str, out_dir: str,
                 table: np.ndarray) -> dict:
    import oracle
    slot, off, sz = reader.expert_slot(layer, expert)
    parts = {k: slot[off[k]:off[k] + sz[k]] for k in off}
    mats = {
        "w1": oracle.dequant_fp4(parts["w1.weight"], parts["w1.scale"], INTER, DIM, table),
        "w2": oracle.dequant_fp4(parts["w2.weight"], parts["w2.scale"], DIM, INTER, table),
        "w3": oracle.dequant_fp4(parts["w3.weight"], parts["w3.scale"], INTER, DIM, table),
    }
    cfg = dict(SCHEMES[scheme])
    cfg.pop("act_aware", None)
    block = cfg.get("block", 32)

    body = bytearray()
    entries = []
    for mat in ("w1", "w2", "w3"):
        w = np.ascontiguousarray(mats[mat][:ROWS], dtype=np.float32)
        k = w.shape[1]
        q = q2.quantise_tensor(w, name=mat, **cfg)
        packed = q2.pack_codes(q.codes)
        scode = q.scale_code.reshape(ROWS, k // block)
        lut = (q.levels if q.levels.shape[0] == ROWS
               else np.broadcast_to(q.levels, (ROWS, q.levels.shape[1]))).astype(np.float32)

        # The reference dequantiser, run on the bytes that are about to be
        # written -- not on the arrays they came from. This is what makes the
        # file a test of the format rather than of this script.
        deq = q2.dequant2_reference(packed, scode, lut, ROWS, k, block)
        assert np.array_equal(deq, q.w_hat), f"{mat}: packing round trip is not exact"

        for tag, arr in (("codes", packed), ("scales", scode),
                         ("lut", lut), ("deq", deq.astype(np.float32))):
            if tag == "deq":
                arr = arr[:2, :256]          # a small directly-comparable window
            raw = np.ascontiguousarray(arr).tobytes()
            entries.append({
                "tensor": mat, "part": tag,
                "dtype": {"codes": "u8", "scales": "u8",
                          "lut": "f32", "deq": "f32"}[tag],
                "shape": list(arr.shape), "offset": len(body), "bytes": len(raw),
                "hash64": oracle.block_hash64(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            })
            body += raw
        entries.append({
            "tensor": mat, "part": "deq_full_stats", "dtype": "meta",
            "rel_l2_vs_fp4": round(q2.rel_l2(q.w_hat, w), 6),
            "cos_vs_fp4": round(q2.cosine(q.w_hat, w), 8),
            "hash64": oracle.block_hash64(
                np.ascontiguousarray(deq, dtype=np.float32).tobytes()),
            "bits_per_weight": round(q.bits_per_weight(), 4),
        })
        del w, q, deq

    name = f"q2_{scheme}_L{layer:02d}_E{expert:03d}.bin"
    path = os.path.join(out_dir, name)
    with open(path, "wb") as f:
        f.write(struct.pack("<IIIIIII", MAGIC, VERSION, layer, expert, ROWS,
                            block, cfg.get("n_levels", 4)))
        f.write(body)
    return {"file": name, "layer": layer, "expert": expert, "scheme": scheme,
            "rows": ROWS, "block": block, "n_levels": cfg.get("n_levels", 4),
            "data_offset": 28, "bytes": len(body), "entries": entries}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--out", default="tests/data/quant2")
    p.add_argument("--experts", default="0:0,20:191,39:383")
    p.add_argument("--schemes", default="int2_b32,lloyd_row_b32")
    a = p.parse_args(argv)

    import oracle
    reader = oracle.ManifestReader(a.model)
    table = oracle.fp4_e2m1_table()
    os.makedirs(a.out, exist_ok=True)
    recs = []
    for spec in a.experts.split(","):
        L, E = (int(x) for x in spec.split(":"))
        for scheme in a.schemes.split(","):
            t0 = time.perf_counter()
            r = write_expert(reader, L, E, scheme, a.out, table)
            print(f"{r['file']}  {r['bytes'] / 1e3:.1f} kB  "
                  f"({time.perf_counter() - t0:.1f}s)")
            recs.append(r)
    idx = {"version": VERSION, "generator": "tools/quant2_golden.py",
           "model": "DeepSeek-V4.1-Flash", "rows_per_tensor": ROWS,
           "written": time.strftime("%Y-%m-%d %H:%M:%S"),
           "format": {
               "codes": "[rows, K/4] u8, 4 codes per byte, element j in bits "
                        "2*(j%4), element 0 in the lowest bits",
               "scales": "[rows, K/block] u8 UE8M0, value = 2^(code-127) -- "
                         "identical in shape and meaning to the FP4 path's",
               "lut": "[rows, n_levels] f32, ascending",
               "dequant": "value(r, j) = lut[r][code(r, j)] * 2^(scale[r][j/block]-127)",
               "deq": "the first 2 rows x 256 columns of the reference "
                      "dequantisation, for a direct value comparison",
               "hash64": "tools/oracle.py:block_hash64, the same one "
                         "tests/test_integration.cpp re-implements"},
           "files": recs}
    with io.open(os.path.join(a.out, "index.json"), "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=1)
    total = sum(r["bytes"] for r in recs)
    print(f"-> {a.out}  {len(recs)} files, {total / 1e6:.2f} MB of payload")
    reader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
