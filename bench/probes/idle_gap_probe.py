#!/usr/bin/env python3
"""Does the drive pay for being idle? One routed expert's read after an idle gap.

Decode reads the drive in bursts: a layer's misses (18.8 MB each, as 1 MiB
chunks at QD 8) and then nothing while the GPU computes. If the link (ASPM L1.2)
or the drive (APST) sleeps in the gaps, every burst starts with a wake-up. This
reads one random routed expert the way the engine does (its two extents from
deepmoe_manifest.json, scales 1.1 MB + weights 17.7 MB, cut into 1 MiB chunks
shared by 8 threads) with O_DIRECT after sleeping `gap` ms, and reports the time
per burst by gap. No GPU, no engine, read-only.

    python bench/probes/idle_gap_probe.py ~/models/DeepSeek-V4.1-Flash [--trials 40] [--poke-ms 5]

--poke-ms N fills each gap with a 4 KiB read every N ms (IoEngine's keep-alive);
--spin busy-waits through the gap instead of sleeping (no I/O: is it the CPU?);
--chunk-kb / --qd change how the extents are cut (the engine's P0 default: 1024 x 8);
--contiguous reads one 18.8 MB extent at a random offset instead of a real expert.
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
import random
import statistics
import threading
import time
from pathlib import Path

CHUNK = 1 << 20
EXPERT = 18_808_832
QD = 8


def burst(fd: int, extents: list[tuple[int, int]], bufs: list[mmap.mmap]) -> float:
    """Read every (offset, bytes) extent of `fd` in chunks shared by len(bufs) threads."""
    chunk = len(bufs[0])
    todo = [(off + i, min(chunk, n - i)) for off, n in extents for i in range(0, n, chunk)]
    lock = threading.Lock()

    def worker(buf: mmap.mmap) -> None:
        while True:
            with lock:
                if not todo:
                    return
                off, n = todo.pop(0)
            os.preadv(fd, [memoryview(buf)[:n]], off)

    t0 = time.perf_counter()
    ts = [threading.Thread(target=worker, args=(b,)) for b in bufs]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return (time.perf_counter() - t0) * 1e3


def idle(gap_ms: float, poke_ms: float, spin: bool, fd: int, size: int, buf: mmap.mmap,
         rng: random.Random) -> None:
    """Sleep `gap_ms`, reading 4 KiB every `poke_ms` if that is > 0; or spin."""
    end = time.perf_counter() + gap_ms / 1e3
    nxt = time.perf_counter() + poke_ms / 1e3
    while poke_ms > 0 and nxt < end:
        if spin:
            while time.perf_counter() < nxt:
                pass
        else:
            time.sleep(poke_ms / 1e3)
        os.preadv(fd, [memoryview(buf)[:4096]], rng.randrange(0, size // 4096 - 1) * 4096)
        nxt += poke_ms / 1e3
    while spin and time.perf_counter() < end:
        pass
    time.sleep(max(0.0, end - time.perf_counter()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--gaps", default="0,0.3,1,3,10,30,100")
    ap.add_argument("--poke-ms", type=float, default=0.0)
    ap.add_argument("--spin", action="store_true")
    ap.add_argument("--chunk-kb", type=int, default=CHUNK >> 10)
    ap.add_argument("--qd", type=int, default=QD)
    ap.add_argument("--contiguous", action="store_true")
    a = ap.parse_args()
    man = json.loads((a.model / "deepmoe_manifest.json").read_text())
    fds = [os.open(a.model / f["path"], os.O_RDONLY | os.O_DIRECT) for f in man["files"]]
    sizes = [os.fstat(fd).st_size for fd in fds]
    experts = [[(r["file"], r["aligned_off"], r["aligned_bytes"]) for r in e]
               for layer in man["experts"] if layer["layer"] < 40 for e in layer["experts"]]
    bufs = [mmap.mmap(-1, a.chunk_kb << 10) for _ in range(a.qd)]
    rng = random.Random(1)
    gaps = [float(g) for g in a.gaps.split(",")]
    res: dict[float, list[float]] = {g: [] for g in gaps}
    for _ in range(a.trials):          # interleaved, so drift hits every gap alike
        for g in gaps:
            runs = rng.choice(experts)
            k = runs[0][0]
            ext = [(rng.randrange(0, (sizes[k] - EXPERT) // 4096) * 4096, EXPERT)] if a.contiguous \
                else [(off, n) for _, off, n in runs]
            idle(g, a.poke_ms, a.spin, fds[k], sizes[k], bufs[0], rng)
            res[g].append(burst(fds[k], ext, bufs))
    print(f"one {'18.8 MB extent' if a.contiguous else 'routed expert (2 extents, 18.8 MB)'} as {a.chunk_kb} KiB x QD {a.qd}, O_DIRECT, {a.trials} trials a gap, "
          f"{'spinning' if a.spin else 'sleeping'} through the gap"
          f"{', a 4 KiB poke every %g ms' % a.poke_ms if a.poke_ms > 0 else ''}")
    print(f"{'gap ms':>7} {'median ms':>10} {'p90 ms':>8} {'GB/s':>6}")
    for g in gaps:
        v = sorted(res[g])
        med, p90 = statistics.median(v), v[int(0.9 * (len(v) - 1))]
        print(f"{g:7.1f} {med:10.2f} {p90:8.2f} {EXPERT / med / 1e6:6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
