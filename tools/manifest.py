#!/usr/bin/env python3
"""Build deepmoe_manifest.json: a pure address book over the ORIGINAL safetensors shards.

Design reference: section 5.1 (NVMe layout) and appendix A (tensor inventory).
This is a P0 deliverable (section 15). It replaces the old tools/repack.py.

Why there is no repack
----------------------
The checkpoint is 510.3 GB and D: has ~196 GB free, so a second copy laid out
for the runtime simply does not fit. Re-downloading after a mistake costs an
hour. So the runtime reads the 48 shards in place and this script writes the
only new file: an address book.

What the shards actually look like (measured, not assumed)
----------------------------------------------------------
* A shard's data section starts at ``8 + header_len`` and header_len differs per
  shard, so absolute tensor offsets are multiples of 8 but never of 4096.
  FILE_FLAG_NO_BUFFERING needs 4 KiB alignment on offset, length and
  destination pointer, so the manifest -- not the I/O layer -- absorbs the
  misalignment.
* Every routed-expert tensor's *length* is a 4096 multiple (w1/w2/w3 weight =
  5,898,240 B packed FP4; each .scale = 368,640 B UE8M0).
* Within a shard an expert's three .scale tensors are adjacent and its three
  .weight tensors are adjacent, and no expert -- indeed no whole MoE layer --
  spans two shards. So one expert is exactly TWO runs.

Runs and skews
--------------
A *run* is a maximal group of tensors that are byte-contiguous in one shard::

    aligned_off   = floor(off / 4096) * 4096
    aligned_bytes = ceil((off + len) / 4096) * 4096 - aligned_off
    skew          = off - aligned_off

The IoEngine reads ``[aligned_off, aligned_off + aligned_bytes)`` into the slot
at ``slot_base + run.slot_offset``; the part itself then lives at
``slot_base + run.slot_offset + skew``. The alignment discipline lives here,
the I/O layer stays a dumb aligned-block mover.

Output schema (version 2)
-------------------------
``files``    array; index into it is the file id used everywhere else.
             Each entry carries ``path``, ``bytes`` and ``data_start``.
``tensors``  every non-expert tensor: file/offset/bytes/dtype/shape plus the
             linked ``scale`` sub-object (model/manifest.h ScaleEntry).
``experts``  per layer, per expert, the list of runs with their parts.
             Layers 0..39 have 384 routed experts, logical layers 40..42 are
             the three DSpark (mtp) blocks with 128 each.
``engram``   the two n-gram tables as value/scale planes plus row geometry, so
             the runtime can build a row's two runs arithmetically (384M rows
             cannot be enumerated).

Usage
-----
    uv run python tools/manifest.py --src D:/models/DeepSeek-V4.1-Flash
    uv run python tools/manifest.py --src ... --verify --workers 8
    uv run python tools/manifest.py --src ... --out build/manifest.json --dry-run
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
import re
import struct
import sys
import time

# Design section 2.3 / appendix A. model/layout.h holds the same numbers for the
# C++ side; tests/test_core.cpp and tests/test_model.cpp keep the two in step.
ALIGNMENT = 4096
MANIFEST_VERSION = 2
N_LAYERS = 40
N_MTP_BLOCKS = 3
N_ROUTED_EXPERTS = 384
N_DSPARK_EXPERTS = 128
EXPERT_MAT_WEIGHT_BYTES = 5_898_240
EXPERT_MAT_SCALE_BYTES = 368_640
EXPERT_PAYLOAD_BYTES = 3 * (EXPERT_MAT_WEIGHT_BYTES + EXPERT_MAT_SCALE_BYTES)  # 18,800,640

# The six parts of one routed expert, in the order model/manifest.h enumerates
# them. The manifest stores the name; the C++ side maps it back to the enum.
EXPERT_PARTS = ("w1.weight", "w1.scale", "w2.weight", "w2.scale", "w3.weight", "w3.scale")

DEFAULT_MANIFEST_NAME = "deepmoe_manifest.json"

# safetensors dtype string -> the name model/manifest.h's quant_from_string reads.
# "I8" is how the checkpoint labels a packed-FP4 plane: [out, in/2] bytes, two
# E2M1 nibbles each, low nibble = even element (appendix A).
DTYPE_MAP = {
    "I8": "fp4_e2m1",
    "F8_E4M3": "fp8_e4m3",
    "F8_E8M0": "e8m0",
    "BF16": "bf16",
    "F32": "f32",
    "F16": "f16",
    "F64": "f64",
    "I32": "i32",
    "I64": "i64",
    "U8": "u8",
    "BOOL": "bool",
}

EXPERT_RE = re.compile(r"^(layers|mtp)\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$")
ENGRAM_RE = re.compile(r"^layers\.(\d+)\.engram\.embed\.(weight|scale)$")


# --------------------------------------------------------------------------- #
# shard headers
# --------------------------------------------------------------------------- #

class Shard:
    __slots__ = ("index", "name", "path", "size", "data_start", "header", "sha256")

    def __init__(self, index: int, path: str):
        self.index = index
        self.path = path
        self.name = os.path.basename(path)
        self.size = os.path.getsize(path)
        with open(path, "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            raw = f.read(header_len)
        if len(raw) != header_len:
            raise ValueError(f"{self.name}: header is truncated ({len(raw)} of {header_len} B)")
        self.data_start = 8 + header_len
        self.header = json.loads(raw)
        self.sha256 = ""


def read_shards(src: str, workers: int) -> list[Shard]:
    names = sorted(n for n in os.listdir(src) if re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", n))
    if not names:
        raise SystemExit(f"no safetensors shards under {src}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda p: Shard(p[0], os.path.join(src, p[1])), enumerate(names)))


def sha256_file(path: str, block: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# tensor table
# --------------------------------------------------------------------------- #

class Tensor:
    __slots__ = ("name", "file", "offset", "bytes", "dtype", "shape")

    def __init__(self, name: str, file: int, offset: int, nbytes: int, dtype: str, shape: list[int]):
        self.name = name
        self.file = file
        self.offset = offset
        self.bytes = nbytes
        self.dtype = dtype
        self.shape = shape

    @property
    def end(self) -> int:
        return self.offset + self.bytes

    def logical_shape(self) -> list[int]:
        """FP4 is stored two elements per byte, so the stored plane is [out, K/2].
        The manifest records the LOGICAL [out, K] shape; `bytes` still describes
        what is on disk."""
        if self.dtype == "fp4_e2m1" and len(self.shape) == 2:
            return [self.shape[0], self.shape[1] * 2]
        return list(self.shape)


def collect_tensors(shards: list[Shard]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for sh in shards:
        for name, meta in sh.header.items():
            if name == "__metadata__":
                continue
            begin, end = meta["data_offsets"]
            dtype = DTYPE_MAP.get(meta["dtype"])
            if dtype is None:
                raise SystemExit(f"{name}: unhandled safetensors dtype {meta['dtype']!r}")
            if name in out:
                raise SystemExit(f"{name} appears in two shards")
            out[name] = Tensor(name, sh.index, sh.data_start + begin, end - begin,
                               dtype, list(meta["shape"]))
    return out


def verify_against_index(src: str, shards: list[Shard], tensors: dict[str, Tensor]) -> dict:
    """Cross-check the headers against model.safetensors.index.json and the file sizes."""
    report: dict = {"problems": []}
    idx_path = os.path.join(src, "model.safetensors.index.json")
    by_name = {sh.name: sh for sh in shards}

    # 1. every tensor's byte range sits inside its shard, and the shard ends
    #    exactly where the last tensor does.
    for sh in shards:
        hi = sh.data_start
        for name, meta in sh.header.items():
            if name == "__metadata__":
                continue
            b, e = meta["data_offsets"]
            if sh.data_start + e > sh.size:
                report["problems"].append(f"{sh.name}: {name} ends past the file")
            hi = max(hi, sh.data_start + e)
        if hi != sh.size:
            report["problems"].append(
                f"{sh.name}: file is {sh.size} B but the header covers only {hi} B")

    # 2. index.json agrees on which shard holds what, and on the payload total.
    if os.path.exists(idx_path):
        with open(idx_path, "rb") as f:
            index = json.load(f)
        wmap = index.get("weight_map", {})
        total = int(index.get("metadata", {}).get("total_size", 0))
        for name, shard_name in wmap.items():
            t = tensors.get(name)
            if t is None:
                report["problems"].append(f"index.json names {name} but no shard has it")
            elif shards[t.file].name != shard_name:
                report["problems"].append(
                    f"{name}: index.json says {shard_name}, header says {shards[t.file].name}")
            elif shard_name not in by_name:
                report["problems"].append(f"index.json references missing shard {shard_name}")
        missing = set(tensors) - set(wmap)
        if missing:
            report["problems"].append(f"{len(missing)} tensors are not in index.json, e.g. "
                                      f"{sorted(missing)[:3]}")
        payload = sum(t.bytes for t in tensors.values())
        report["payload_bytes"] = payload
        report["index_total_size"] = total
        if total and payload != total:
            report["problems"].append(
                f"payload {payload} != index.json total_size {total} (delta {payload - total})")
    else:
        report["problems"].append("model.safetensors.index.json is missing")
    report["shard_bytes"] = sum(sh.size for sh in shards)
    return report


# --------------------------------------------------------------------------- #
# runs
# --------------------------------------------------------------------------- #

def align_down(v: int) -> int:
    return (v // ALIGNMENT) * ALIGNMENT


def align_up(v: int) -> int:
    return -(-v // ALIGNMENT) * ALIGNMENT


def build_runs(parts: list[tuple[str, Tensor]], slot_base: int = 0) -> tuple[list[dict], int]:
    """Group byte-contiguous tensors of one shard into aligned runs.

    `parts` is [(part_name, tensor)]. Returns (runs, total_slot_bytes). Runs are
    ordered by file offset and laid back-to-back in the slot, so the slot is the
    concatenation of the aligned blocks the I/O layer actually reads.
    """
    ordered = sorted(parts, key=lambda p: (p[1].file, p[1].offset))
    groups: list[list[tuple[str, Tensor]]] = []
    for part_name, t in ordered:
        if groups and groups[-1][-1][1].file == t.file and groups[-1][-1][1].end == t.offset:
            groups[-1].append((part_name, t))
        else:
            groups.append([(part_name, t)])

    runs = []
    cursor = slot_base
    for group in groups:
        first, last = group[0][1], group[-1][1]
        a_off = align_down(first.offset)
        a_bytes = align_up(last.end) - a_off
        run = {
            "file": first.file,
            "aligned_off": a_off,
            "aligned_bytes": a_bytes,
            "slot_offset": cursor,
            "parts": [
                {
                    "tensor": part_name,
                    "skew": t.offset - a_off,
                    "bytes": t.bytes,
                    "slot_offset": cursor + (t.offset - a_off),
                }
                for part_name, t in group
            ],
        }
        runs.append(run)
        cursor += a_bytes
    return runs, cursor - slot_base


# --------------------------------------------------------------------------- #
# manifest assembly
# --------------------------------------------------------------------------- #

def scale_entry(tensors: dict[str, Tensor], weight: Tensor) -> dict | None:
    """The linked .scale plane, shaped the way model/manifest.h's ScaleEntry wants."""
    s = tensors.get(weight.name.rsplit(".", 1)[0] + ".scale") if weight.name.endswith(".weight") \
        else tensors.get(weight.name + ".scale")
    if s is None:
        return None
    wshape = weight.logical_shape()
    sshape = s.logical_shape()
    block_m = block_k = 0
    if len(wshape) == 2 and len(sshape) == 2 and sshape[0] and sshape[1]:
        block_m = wshape[0] // sshape[0]
        block_k = wshape[1] // sshape[1]
    return {
        "file": s.file,
        "offset": s.offset,
        "bytes": s.bytes,
        "dtype": s.dtype,
        "shape": sshape,
        "block": [block_m, block_k],
    }


def build_manifest(src: str, shards: list[Shard], tensors: dict[str, Tensor]) -> tuple[dict, dict]:
    stats: dict = {}

    # --- classify -----------------------------------------------------------
    experts: dict[tuple[str, int], dict[int, dict[str, Tensor]]] = collections.defaultdict(
        lambda: collections.defaultdict(dict))
    plain: dict[str, Tensor] = {}
    for name, t in tensors.items():
        m = EXPERT_RE.match(name)
        if m:
            kind, layer, eid, mat, what = m.group(1), int(m.group(2)), int(m.group(3)), \
                m.group(4), m.group(5)
            experts[(kind, layer)][eid][f"{mat}.{what}"] = t
        else:
            plain[name] = t

    # --- tensors ------------------------------------------------------------
    scale_names = {n for n in plain if n.endswith(".scale")}
    tensor_table: dict[str, dict] = {}
    for name, t in sorted(plain.items()):
        if name in scale_names:
            continue  # folded into its weight's entry
        entry = {
            "file": t.file,
            "offset": t.offset,
            "bytes": t.bytes,
            "dtype": t.dtype,
            "shape": t.logical_shape(),
        }
        sc = scale_entry(tensors, t)
        if sc is not None:
            entry["scale"] = sc
        tensor_table[name] = entry
    orphan_scales = [n for n in scale_names
                     if n.rsplit(".", 1)[0] + ".weight" not in plain and n[:-6] not in plain]

    # --- experts ------------------------------------------------------------
    # Logical layer numbering (core/types.h ExpertKey): 0..39 are the main MoE
    # layers, 40..42 the three DSpark blocks.
    layer_tables: list[dict] = []
    run_hist: collections.Counter = collections.Counter()
    slot_bytes_max = 0
    run_bytes_hist: collections.Counter = collections.Counter()
    cross_shard_experts: list[str] = []
    cross_shard_layers: list[str] = []

    def logical_layer(kind: str, layer: int) -> int:
        return layer if kind == "layers" else N_LAYERS + layer

    for (kind, layer) in sorted(experts, key=lambda k: logical_layer(*k)):
        per_expert = experts[(kind, layer)]
        expected = N_ROUTED_EXPERTS if kind == "layers" else N_DSPARK_EXPERTS
        if len(per_expert) != expected:
            raise SystemExit(f"{kind}.{layer}: {len(per_expert)} experts, expected {expected}")
        files_in_layer = set()
        entries = []
        for eid in range(expected):
            parts = per_expert.get(eid)
            if parts is None or len(parts) != len(EXPERT_PARTS):
                raise SystemExit(f"{kind}.{layer}.experts.{eid}: got {sorted(parts or [])}")
            payload = sum(t.bytes for t in parts.values())
            if payload != EXPERT_PAYLOAD_BYTES:
                raise SystemExit(f"{kind}.{layer}.experts.{eid}: payload {payload} B, "
                                 f"expected {EXPERT_PAYLOAD_BYTES}")
            runs, slot = build_runs(list(parts.items()))
            run_hist[len(runs)] += 1
            slot_bytes_max = max(slot_bytes_max, slot)
            for r in runs:
                run_bytes_hist[r["aligned_bytes"]] += 1
                files_in_layer.add(r["file"])
            if len({t.file for t in parts.values()}) > 1:
                cross_shard_experts.append(f"{kind}.{layer}.experts.{eid}")
            entries.append(runs)
        if len(files_in_layer) > 1:
            cross_shard_layers.append(
                f"{kind}.{layer} -> {sorted(shards[i].name for i in files_in_layer)}")
        layer_tables.append({
            "layer": logical_layer(kind, layer),
            "name": f"{kind}.{layer}",
            "experts": entries,
        })

    if slot_bytes_max % ALIGNMENT != 0:
        raise SystemExit(f"slot bytes {slot_bytes_max} is not a 4 KiB multiple")

    # --- engram -------------------------------------------------------------
    engram = []
    for name, t in sorted(tensors.items()):
        m = ENGRAM_RE.match(name)
        if not m or m.group(2) != "weight":
            continue
        layer = int(m.group(1))
        s = tensors[f"layers.{layer}.engram.embed.scale"]
        rows, head_dim = t.shape[0], t.shape[1]
        if s.shape[0] != rows:
            raise SystemExit(f"engram layer {layer}: value has {rows} rows, scale {s.shape[0]}")
        engram.append({
            "layer": layer,
            "rows": rows,
            "value": {"file": t.file, "offset": t.offset, "bytes": t.bytes,
                      "row_bytes": head_dim, "dtype": t.dtype, "tensor": name},
            "scale": {"file": s.file, "offset": s.offset, "bytes": s.bytes,
                      "row_bytes": s.shape[1], "dtype": s.dtype, "tensor": s.name},
        })

    manifest = {
        "version": MANIFEST_VERSION,
        "model": os.path.basename(os.path.normpath(src)),
        "generator": "tools/manifest.py",
        "alignment": ALIGNMENT,
        "expert_slot_bytes": slot_bytes_max,
        "expert_parts": list(EXPERT_PARTS),
        "files": [
            {"path": sh.name, "bytes": sh.size, "data_start": sh.data_start, "sha256": sh.sha256}
            for sh in shards
        ],
        "tensors": tensor_table,
        "experts": layer_tables,
        "engram": engram,
    }

    stats.update(
        run_hist=dict(sorted(run_hist.items())),
        run_bytes_hist=dict(sorted(run_bytes_hist.items())),
        slot_bytes=slot_bytes_max,
        expert_count=sum(run_hist.values()),
        cross_shard_experts=cross_shard_experts,
        cross_shard_layers=cross_shard_layers,
        tensor_entries=len(tensor_table),
        orphan_scales=orphan_scales,
        engram_rows=[e["rows"] for e in engram],
    )
    return manifest, stats


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def print_summary(manifest: dict, stats: dict, verify: dict, shards: list[Shard]) -> None:
    p = print
    p("")
    p(f"shards                 {len(shards)}  ({verify['shard_bytes']:,} B on disk)")
    p(f"tensor payload         {verify.get('payload_bytes', 0):,} B  "
      f"(index.json total_size {verify.get('index_total_size', 0):,})")
    starts = sorted({sh.data_start % ALIGNMENT for sh in shards})
    p(f"data_start % 4096      {starts}")
    p(f"non-expert tensors     {stats['tensor_entries']} entries "
      f"({len(manifest['tensors'])} with scales folded in)")
    if stats["orphan_scales"]:
        p(f"  !! orphan scales     {stats['orphan_scales'][:5]}")

    p("")
    p(f"routed + dspark experts {stats['expert_count']}")
    p("  runs per expert:")
    for n, count in stats["run_hist"].items():
        p(f"    {n} run(s): {count}")
    p("  aligned run sizes:")
    for nbytes, count in stats["run_bytes_hist"].items():
        p(f"    {nbytes:>12,} B  x {count}")
    p(f"  expert payload        {EXPERT_PAYLOAD_BYTES:,} B")
    p(f"  kExpertSlotBytes      {stats['slot_bytes']:,} B "
      f"= {stats['slot_bytes'] // ALIGNMENT} x 4 KiB "
      f"(+{stats['slot_bytes'] - EXPERT_PAYLOAD_BYTES} B of skew padding)")
    p(f"  experts spanning >1 shard  {len(stats['cross_shard_experts'])}")
    p(f"  layers spanning >1 shard   {len(stats['cross_shard_layers'])}")
    for line in stats["cross_shard_layers"]:
        p(f"    {line}")

    p("")
    for e in manifest["engram"]:
        p(f"engram layer {e['layer']:<3}       {e['rows']:,} rows x "
          f"{e['value']['row_bytes']} B value + {e['scale']['row_bytes']} B scale "
          f"(2 runs per row)")

    if verify["problems"]:
        p("")
        p(f"!! {len(verify['problems'])} consistency problem(s):")
        for line in verify["problems"][:20]:
            p(f"   {line}")
    else:
        p("")
        p("consistency            ok (headers, file sizes and index.json agree)")


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="manifest.py",
        description="safetensors shards -> deepmoe_manifest.json (schema v2, direct-read)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--src", required=True,
                   help="directory holding the 48 safetensors shards, config.json and "
                        "model.safetensors.index.json")
    p.add_argument("--out", default=None,
                   help=f"manifest path (default: <src>/{DEFAULT_MANIFEST_NAME})")
    p.add_argument("--verify", action="store_true",
                   help="also sha256 every shard in parallel and record it "
                        "(reads 510 GB, tens of minutes)")
    p.add_argument("--workers", type=int, default=8,
                   help="parallel shard readers / hashers")
    p.add_argument("--indent", type=int, default=None,
                   help="pretty-print the JSON with this indent (default: compact)")
    p.add_argument("--dry-run", action="store_true",
                   help="scan and report without writing the manifest")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    t0 = time.perf_counter()

    shards = read_shards(args.src, args.workers)
    t_headers = time.perf_counter()
    tensors = collect_tensors(shards)
    verify = verify_against_index(args.src, shards, tensors)

    if args.verify:
        print(f"hashing {len(shards)} shards with {args.workers} workers ...", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for sh, digest in zip(shards, pool.map(lambda s: sha256_file(s.path), shards)):
                sh.sha256 = digest
                print(f"  {sh.name}  {digest}", flush=True)

    manifest, stats = build_manifest(args.src, shards, tensors)
    t_build = time.perf_counter()

    out = args.out or os.path.join(args.src, DEFAULT_MANIFEST_NAME)
    if not args.dry_run:
        tmp = out + ".tmp"
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(manifest, f, separators=(",", ":") if args.indent is None else None,
                      indent=args.indent)
        os.replace(tmp, out)

    print_summary(manifest, stats, verify, shards)
    t1 = time.perf_counter()
    print("")
    if args.dry_run:
        print(f"manifest             (dry run, not written) -> {out}")
    else:
        print(f"manifest             {out}  ({os.path.getsize(out):,} B)")
    print(f"timing               headers {t_headers - t0:.2f} s, build {t_build - t_headers:.2f} s, "
          f"total {t1 - t0:.2f} s")
    return 1 if verify["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
