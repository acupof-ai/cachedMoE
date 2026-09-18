#!/usr/bin/env python3
"""Read a --trace file: one layer phase by phase, and the token by phase class.

The per-token profiler (core/profiler.h) says a token spent 36 ms in attention.
This says which of layer 20's 29 dispatches took it, and how much of the layer
is barrier tails rather than kernels.

    # one layer, dispatch by dispatch
    python tools/trace_timeline.py trace.bin --layer 20

    # the whole token, rolled up by phase class
    python tools/trace_timeline.py trace.bin --summary

    # a token other than the first one in the file
    python tools/trace_timeline.py trace.bin --token 3 --layer 20

Both stamps are bottom-of-pipe and dispatches are barrier-separated, so a
record's `begin` is "everything before this dispatch has completed" and `end`
is "this dispatch has completed". Therefore

    busy = end - begin           the dispatch itself
    gap  = begin - previous end  the barrier in front of it

and the sum of the gaps is what a persistent-dispatch decode
(docs/plan_p5.md §3(a)) would be attacking.

FORMAT (version 1) -- the writer is runtime/trace.{h,cpp}, which has the same
table in its header comment. If one of these changes without the other,
tests/test_trace.cpp's `record_layout_is_32_bytes_at_fixed_offsets` fails on
the C++ side and `--self-test` below fails on this one.

  header, 32 bytes:   "DMTRACE1" u32 version u32 record_bytes f64 period_ns
                      u32 record_count u32 name_count
  records, 32 bytes:  u32 token, u32 seq, u16 layer, u16 stage, u8 cls,
                      u8 flags, u16 submit, u64 begin_ns, u64 end_ns
  names:              u8 cls, u16 stage, u8 len, char[len]
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

MAGIC = b"DMTRACE1"
HEADER_FMT = "<8sIIdII"
HEADER_SIZE = 32
RECORD_FMT = "<IIHHBBHQQ"
RECORD_SIZE = 32

NO_LAYER = 0xFFFF
CLS = {0: "attention", 1: "ced", 2: "moe", 3: "engram", 4: "tail", 5: "other"}
FLAG_NO_BEGIN, FLAG_NO_END, FLAG_WRAPPED = 1, 2, 4

assert struct.calcsize(HEADER_FMT) == HEADER_SIZE
assert struct.calcsize(RECORD_FMT) == RECORD_SIZE


class Record:
    __slots__ = ("token", "seq", "layer", "stage", "cls", "flags", "submit",
                 "begin_ns", "end_ns")

    def __init__(self, tup):
        (self.token, self.seq, self.layer, self.stage, self.cls, self.flags,
         self.submit, self.begin_ns, self.end_ns) = tup

    @property
    def busy_ns(self) -> int:
        return self.end_ns - self.begin_ns if self.end_ns >= self.begin_ns else 0

    @property
    def timed(self) -> bool:
        return not (self.flags & (FLAG_NO_BEGIN | FLAG_NO_END))


class Trace:
    def __init__(self, blob: bytes):
        magic, self.version, self.record_bytes, self.period_ns, n_rec, n_name = \
            struct.unpack_from(HEADER_FMT, blob, 0)
        if magic != MAGIC:
            raise SystemExit(f"not a deepMoE trace: magic {magic!r}")
        if self.version != 1:
            raise SystemExit(f"trace version {self.version}, this reader knows 1")
        if self.record_bytes != RECORD_SIZE:
            raise SystemExit(
                f"trace has {self.record_bytes}-byte records, this reader reads {RECORD_SIZE}. "
                "runtime/trace.h and this file have drifted apart.")
        need = HEADER_SIZE + n_rec * self.record_bytes
        if len(blob) < need:
            raise SystemExit(f"header claims {n_rec} records ({need} B), file is {len(blob)} B")
        self.records = [Record(struct.unpack_from(RECORD_FMT, blob, HEADER_SIZE + i * RECORD_SIZE))
                        for i in range(n_rec)]
        self.names = {}
        off = need
        for _ in range(n_name):
            cls, stage, ln = struct.unpack_from("<BHB", blob, off)
            off += 4
            self.names[(cls, stage)] = blob[off:off + ln].decode("utf-8", "replace")
            off += ln

    def name(self, r: Record) -> str:
        return self.names.get((r.cls, r.stage), f"{CLS.get(r.cls, '?')}#{r.stage}")

    def tokens(self):
        seen, out = set(), []
        for r in self.records:
            if r.token not in seen:
                seen.add(r.token)
                out.append(r.token)
        return out

    def passes(self, token: int):
        """The records of `token`, split into one list per forward pass.

        `Tracer::token_begin` keys on the decode POSITION, and the warm-up of
        `deepmoe run --warm N` decodes the same position N times. So one token
        id can carry N+1 passes, and reading them as one token gives nonsense
        (every pass's `seq` starts at 0 again, so sorting by seq interleaves
        them: the gaps come out as the whole inter-pass wait). Split on the
        seq reset, in file order, which is record order.
        """
        out, cur = [], []
        for r in self.records:
            if r.token != token:
                continue
            if r.seq == 0 and cur:
                out.append(cur)
                cur = []
            cur.append(r)
        if cur:
            out.append(cur)
        return out


def us(ns: int) -> float:
    return ns / 1000.0


def print_layer(t: Trace, token: int, layer: int, recs=None) -> int:
    src = t.records if recs is None else recs
    rows = [r for r in src if r.token == token and r.layer == layer]
    if not rows:
        print(f"token {token} has no layer {layer}; "
              f"layers present: {sorted({r.layer for r in src if r.token == token})}")
        return 1
    rows.sort(key=lambda r: r.seq)
    timed = [r for r in rows if r.timed]
    t0 = min(r.begin_ns for r in timed) if timed else 0

    print(f"token {token}, layer {layer}: {len(rows)} dispatches, "
          f"span {us(max(r.end_ns for r in timed) - t0):.1f} us"
          if timed else f"token {token}, layer {layer}: {len(rows)} dispatches, untimed")
    print(f"  {'#':>3} {'phase':<10} {'stage':<22} {'start':>10} {'end':>10} "
          f"{'busy':>9} {'gap':>8} {'sub':>4}")
    prev_end = None
    busy_total = gap_total = 0
    for i, r in enumerate(rows):
        if not r.timed:
            print(f"  {i:>3} {CLS.get(r.cls, '?'):<10} {t.name(r):<22} "
                  f"{'--':>10} {'--':>10} {'--':>9} {'--':>8} {r.submit:>4}   "
                  f"(no {'begin' if r.flags & FLAG_NO_BEGIN else ''}"
                  f"{'/' if r.flags & FLAG_NO_BEGIN and r.flags & FLAG_NO_END else ''}"
                  f"{'end' if r.flags & FLAG_NO_END else ''} stamp)")
            continue
        gap = (r.begin_ns - prev_end) if prev_end is not None else 0
        busy_total += r.busy_ns
        gap_total += max(gap, 0)
        wrap = " W" if r.flags & FLAG_WRAPPED else ""
        print(f"  {i:>3} {CLS.get(r.cls, '?'):<10} {t.name(r):<22} "
              f"{us(r.begin_ns - t0):>8.1f}us {us(r.end_ns - t0):>8.1f}us "
              f"{us(r.busy_ns):>7.1f}us {us(gap):>6.1f}us {r.submit:>4}{wrap}")
        prev_end = r.end_ns
    span = (max(r.end_ns for r in timed) - t0) if timed else 0
    print(f"\n  busy {us(busy_total):.1f} us in {len(timed)} dispatches, "
          f"gaps {us(gap_total):.1f} us ({100.0 * gap_total / span if span else 0:.1f}% of the layer), "
          f"span {us(span):.1f} us")
    return 0


def print_summary(t: Trace, token: int, recs=None) -> int:
    rows = list(t.records if recs is None else recs)
    rows = [r for r in rows if r.token == token]
    if not rows:
        print(f"no records for token {token}; tokens present: {t.tokens()[:16]}")
        return 1
    rows.sort(key=lambda r: r.seq)
    timed = [r for r in rows if r.timed]
    if not timed:
        print(f"token {token}: {len(rows)} dispatches, none timed "
              "(the query pool gave out no slots -- is timestampValidBits 0?)")
        return 1
    t0 = min(r.begin_ns for r in timed)
    span = max(r.end_ns for r in timed) - t0

    by_cls = {}
    prev_end = None
    gap_by_cls = {}
    for r in rows:
        if not r.timed:
            continue
        b = by_cls.setdefault(r.cls, [0, 0])   # [busy_ns, count]
        b[0] += r.busy_ns
        b[1] += 1
        if prev_end is not None:
            gap_by_cls[r.cls] = gap_by_cls.get(r.cls, 0) + max(r.begin_ns - prev_end, 0)
        prev_end = r.end_ns

    layers = sorted({r.layer for r in rows if r.layer != NO_LAYER})
    submits = max(r.submit for r in rows) + 1
    untimed = len(rows) - len(timed)

    print(f"token {token}: {len(rows)} dispatches over {len(layers)} layers, "
          f"{submits} submits, GPU span {us(span) / 1000.0:.2f} ms")
    if untimed:
        print(f"  WARNING: {untimed} dispatches lost a stamp -- the query pool is too small; "
              "runtime/trace.h::suggested_pool sizes it")
    print(f"  {'phase':<12} {'dispatches':>10} {'busy':>11} {'% span':>8} "
          f"{'gaps after':>11} {'busy/disp':>10}")
    total_busy = total_gap = 0
    for cls in sorted(by_cls, key=lambda c: -by_cls[c][0]):
        busy, n = by_cls[cls]
        gap = gap_by_cls.get(cls, 0)
        total_busy += busy
        total_gap += gap
        print(f"  {CLS.get(cls, '?'):<12} {n:>10} {us(busy) / 1000.0:>9.3f}ms "
              f"{100.0 * busy / span:>7.1f}% {us(gap):>9.1f}us {us(busy) / n:>8.1f}us")
    print(f"  {'-' * 66}")
    print(f"  {'total':<12} {len(timed):>10} {us(total_busy) / 1000.0:>9.3f}ms "
          f"{100.0 * total_busy / span:>7.1f}% {us(total_gap):>9.1f}us")
    print(f"\n  barriers and submit boundaries: {us(total_gap) / 1000.0:.3f} ms "
          f"= {100.0 * total_gap / span:.1f}% of the GPU span, over "
          f"{len(timed) - 1} boundaries "
          f"({us(total_gap) / max(len(timed) - 1, 1):.2f} us each)")
    print(f"  host time not on the GPU: {us(span - total_busy - total_gap) / 1000.0:.3f} ms "
          "(the span minus every busy and every gap; a submit the host was late to fill "
          "shows up here)")
    return 0


def self_test() -> int:
    """Round-trips a synthetic file through this reader, so a format drift is
    caught without a GPU and without the C++ side."""
    recs = []
    for i in range(6):
        recs.append(struct.pack(RECORD_FMT, 7, i, i // 2, i % 3, i % 2, 0, i // 2,
                                2000 * i, 2000 * i + 1000))
    names = b""
    for cls in (0, 1):
        for stage in range(3):
            nm = f"s{cls}{stage}".encode()
            names += struct.pack("<BHB", cls, stage, len(nm)) + nm
    blob = struct.pack(HEADER_FMT, MAGIC, 1, RECORD_SIZE, 10.0, len(recs), 6) \
        + b"".join(recs) + names
    t = Trace(blob)
    assert len(t.records) == 6, len(t.records)
    assert t.tokens() == [7]
    assert t.records[3].busy_ns == 1000
    assert t.name(t.records[0]) == "s00", t.name(t.records[0])
    assert t.names[(1, 2)] == "s12"
    rc = print_layer(t, 7, 1)
    rc |= print_summary(t, 7)
    print("\nself-test: OK")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", nargs="?", type=Path)
    ap.add_argument("--layer", type=int, default=None, help="print this layer dispatch by dispatch")
    ap.add_argument("--token", type=int, default=None, help="default: the first token in the file")
    ap.add_argument("--pass", dest="which_pass", type=int, default=None,
                    help="which forward pass of that token (--warm decodes one position "
                         "several times); default: the last one, which is the warmest")
    ap.add_argument("--summary", action="store_true", help="roll the token up by phase class")
    ap.add_argument("--self-test", action="store_true",
                    help="parse a synthetic trace; no file and no GPU needed")
    a = ap.parse_args()

    if a.self_test:
        return self_test()
    if a.trace is None:
        ap.error("a trace file is required (or --self-test)")
    t = Trace(a.trace.read_bytes())
    toks = t.tokens()
    if not toks:
        print("the trace has no records")
        return 1
    token = a.token if a.token is not None else toks[0]
    print(f"{a.trace}: {len(t.records)} dispatches, {len(toks)} token(s) "
          f"{toks[0]}..{toks[-1]}, timestampPeriod {t.period_ns:g} ns")
    passes = t.passes(token)
    if len(passes) > 1:
        idx = a.which_pass if a.which_pass is not None else len(passes) - 1
        if not (0 <= idx < len(passes)):
            print(f"token {token} has {len(passes)} passes, 0..{len(passes) - 1}")
            return 1
        print(f"  token {token} carries {len(passes)} forward passes "
              f"(--warm re-decodes one position); showing pass {idx}"
              f"{' (the warmest)' if a.which_pass is None else ''}. "
              f"--pass N selects another.")
        recs = passes[idx]
    else:
        recs = None
    rc = 0
    if a.layer is not None:
        rc |= print_layer(t, token, a.layer, recs)
        print()
    if a.summary or a.layer is None:
        rc |= print_summary(t, token, recs)
    return rc


if __name__ == "__main__":
    sys.exit(main())
