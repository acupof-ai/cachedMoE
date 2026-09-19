#!/usr/bin/env python3
"""Mutation gate: break the code on purpose, require the suite to notice.

A green suite proves nothing by itself. `fleet-mi300x` ran this audit on its
own tests and found that two of its seven caught NOTHING -- both had
re-implemented the production formula in Python and compared the mirror with
itself. That is not a hypothetical failure mode here either: docs/STATUS.md
§3 item 21b records an engine field (`RouteDecision.near_ids`) that was
identically zero for the whole life of the decode path while every test stayed
green, and item 39 records an enum (`XLayout`) that never took effect while the
code that read it passed its tests.

Each row below edits ONE expression in code a test claims to cover, rebuilds,
runs that one suite, and requires the expected outcome:

    caught      the suite must fail -- the normal expectation
    redundant   the suite must still pass because a second check covers the
                same mistake; a companion row removes both and must be caught

    python tests/mutate.py                    # everything that can run here
    python tests/mutate.py --list             # just print the table
    python tests/mutate.py --only dequant     # one suite's rows

HOW IT EDITS. In place, in the working tree, with the original bytes held in
memory and restored in a `finally` and in an atexit hook and on SIGINT. Before
it starts it refuses to run if `git status --porcelain` shows any of the target
files already modified, so a crash can never be confused with your own edit --
and if it ever does leave one behind, `git checkout -- <file>` is the whole
recovery. `--copy` does the same work in a temp clone instead, which is safer
and much slower (a fresh CMake configure is ~5 minutes on this machine).

COST. One incremental build per row. Keep the list short and the edits in .cpp
files rather than headers where possible: a header edit rebuilds everything
that includes it.
"""
from __future__ import annotations

import argparse
import atexit
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"

# (suite, [(file, find, replace)], label, expectation, needs_model)
MUTATIONS = [
    # --- cpu/: the FP4 / FP8 decode tables -------------------------------
    ("dequant", [("cpu/dequant.h",
                  "constexpr float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};",
                  "constexpr float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 4.0f, 3.0f, 6.0f};")],
     "swap two FP4 E2M1 magnitudes", "caught", False),
    ("dequant", [("cpu/dequant.h",
                  "const int   exp  = static_cast<int>(e) - 7;",
                  "const int   exp  = static_cast<int>(e) - 8;")],
     "shift the FP8 E4M3 exponent bias by one", "caught", False),

    # --- model/: the manifest's run / skew address book (design §5.1) ----
    ("manifest", [("model/manifest.cpp",
                   "r.skew          = static_cast<uint32_t>(off - r.aligned_off);",
                   "r.skew          = 0;")],
     "drop the run's skew (the tensor's offset inside its aligned run)", "caught", False),

    # --- storage/: the IoEngine's priority and its unbuffered min_bytes --
    ("io", [("storage/io_engine.cpp",
             "std::min<uint64_t>(ch.bytes, fsize > ch.off ? fsize - ch.off : 0));",
             "std::max<uint64_t>(ch.bytes, fsize > ch.off ? fsize - ch.off : 0));")],
     "let min_bytes run past the end of the file", "caught", False),
    ("io", [("storage/io_engine.cpp",
             "bg >= kBackgroundOpsWhileBusy)",
             "bg > kBackgroundOpsWhileBusy)")],
     "loosen the background-op throttle by one", "caught", False),

    # --- store/: the planner's global LRU (design §9.3) ------------------
    ("planner", [("store/planner.cpp",
                  "return c[a].last_use_token < c[b].last_use_token;",
                  "return c[a].last_use_token > c[b].last_use_token;")],
     "evict the most recently used expert instead of the least", "caught", False),
    ("planner", [("store/planner.cpp",
                  "return c[a].slot < c[b].slot;",
                  "return c[a].slot > c[b].slot;")],
     "reverse the LRU's deterministic tie-break", "caught", False),

    # --- text/: the BPE merge order --------------------------------------
    ("tokenizer", [("text/tokenizer.cpp",
                    "return a.rank != b.rank ? a.rank > b.rank : a.pos > b.pos;",
                    "return a.rank != b.rank ? a.rank > b.rank : a.pos < b.pos;")],
     "break merge ties by the rightmost position instead of the leftmost",
     "caught", True),

    # --- runtime/: nucleus sampling (design §7.11, Track P) --------------
    ("sampling", [("runtime/sampling.cpp",
                   "if (top_p < 1.0f && cum >= double(top_p)) break;",
                   "if (top_p < 1.0f && cum >= double(top_p) * 1.02) break;")],
     "widen the nucleus by 2% of the mass", "caught", False),
    ("sampling", [("runtime/sampling.cpp",
                   "n.exact = (top_p < 1.0f && cum >= double(top_p)) || whole_vocab;",
                   "n.exact = true;")],
     "claim the nucleus is exact when the top set cannot prove it", "caught", False),

    # --- runtime/: the window ring's replay bookkeeping (Track R2) -------
    ("kvstore", [("runtime/kvstore.h",
                  ": int64_t(s) + int64_t((n - 1 - s) / window) * window;",
                  ": int64_t(s) + int64_t((n - s) / window) * window;")],
     "shift the ring's slot->position map by one generation", "caught", False),
    ("kvstore", [("runtime/kvstore.h",
                  "return (window == 0 || n <= s) ? -1",
                  "return (window == 0 || n < s) ? -1")],
     "call a ring slot the sequence never reached occupied", "caught", False),

    # --- runtime/: the engram hash constants (design §7.10) --------------
    ("engram_tables", [("runtime/engram_tables.cpp",
                        "lt.multipliers[i] = m[i] * 2 + 1;",
                        "lt.multipliers[i] = m[i] * 2;")],
     "make the engram multipliers even (the reference forces them odd)",
     "caught", True),
    ("engram_tables", [("runtime/engram_tables.cpp",
                        "numpy_rng_integers(10007ull * lt.layer",
                        "numpy_rng_integers(10009ull * lt.layer")],
     "change the engram per-layer seed", "caught", True),

    # --- H1a: the auto cache-size cap and the probe's back-off plan ------
    ("cache_cap", [("runtime/engine.cpp",
                    "return budget_bytes > cap_bytes ? cap_bytes : budget_bytes;",
                    "return budget_bytes;")],
     "ignore the auto slot cap (the 5,100-slot default that dies on submit)",
     "caught", False),
    ("cache_cap", [("runtime/engine.cpp",
                    "n = n > step + floor_slots ? n - step : floor_slots;",
                    "n = n;")],
     "make the cache back-off plan never decrease", "caught", False),

    # --- H1b: a .pkv restore is a degradation, never fatal ---------------
    ("kvdisk", [("runtime/session.cpp",
                 "st.cold_fallback = true;\n"
                 "    st.cold_reason   = r.error().str();\n"
                 "    if (cold) cold();",
                 "st.cold_fallback = false;\n"
                 "    st.cold_reason   = r.error().str();")],
     "let a refused .pkv restore propagate instead of falling back cold",
     "caught", False),
]

# Original bytes, not text: reading a file as text translates CRLF to LF on
# Windows, and writing it back would leave every target "modified" in git's
# eyes even though nothing changed. Byte in, byte out.
_restore: dict[Path, bytes] = {}


def _restore_all() -> None:
    for path, raw in list(_restore.items()):
        try:
            path.write_bytes(raw)
        except OSError:
            print(f"!! could not restore {path}; run: git checkout -- {path}", file=sys.stderr)
    _restore.clear()


atexit.register(_restore_all)
for _sig in (signal.SIGINT, signal.SIGTERM):
    try:
        signal.signal(_sig, lambda *_: (_restore_all(), sys.exit(130)))
    except (ValueError, OSError):
        pass


def run(cmd: list[str], cwd: Path, env: dict | None = None) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def build(root: Path, build_dir: Path) -> tuple[bool, str]:
    rc, out = run(["cmake", "--build", str(build_dir)], root)
    # The zig cache races when two builds share it; one retry is the documented
    # remedy (docs/build.md).
    if rc != 0 and ("zig" in out.lower() or "cache" in out.lower()):
        rc, out = run(["cmake", "--build", str(build_dir)], root)
    return rc == 0, out


def test_binary(build_dir: Path) -> Path:
    exe = build_dir / "tests" / ("deepmoe_tests.exe" if os.name == "nt" else "deepmoe_tests")
    return exe


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the table and exit")
    ap.add_argument("--only", default="", help="only rows whose suite or label contains this")
    ap.add_argument("--copy", action="store_true",
                    help="work in a temp clone (safe, ~5 min extra for the CMake configure)")
    ap.add_argument("--build-dir", default=str(BUILD))
    a = ap.parse_args()

    rows = [m for m in MUTATIONS
            if not a.only or a.only in m[0] or a.only in m[2]]

    if a.list:
        w = max(len(r[2]) for r in rows)
        for suite, edits, label, expect, needs in rows:
            print(f"  {label:<{w}}  {suite:<16} {expect:<10} "
                  f"{'needs DEEPMOE_MODEL_DIR' if needs else ''}")
        print(f"\n{len(rows)} mutations")
        return 0

    have_model = bool(os.environ.get("DEEPMOE_MODEL_DIR"))
    root, build_dir = ROOT, Path(a.build_dir)

    if a.copy:
        tmp = Path(tempfile.mkdtemp(prefix="deepmoe-mutate-"))
        print(f"cloning the tree into {tmp} (this takes a while)")
        shutil.copytree(ROOT, tmp / "src",
                        ignore=shutil.ignore_patterns("build", ".git", ".zig-cache", "traces"))
        root = tmp / "src"
        build_dir = root / "build"
        rc, out = run(["cmake", "-S", ".", "-B", "build", "-G", "Ninja",
                       "-DCMAKE_TOOLCHAIN_FILE=cmake/zig-toolchain.cmake",
                       "-DCMAKE_BUILD_TYPE=Release"], root)
        if rc != 0:
            print(out[-2000:], file=sys.stderr)
            return 1

    # Refuse to start on a tree that already has one of the targets modified:
    # a crash must never be mistaken for the operator's own edit.
    targets = sorted({e[0] for _, edits, _, _, _ in rows for e in edits})
    if not a.copy:
        run(["git", "status", "--porcelain"], ROOT)          # refresh the stat cache
        rc, out = run(["git", "diff", "--name-only", "--"] + targets, ROOT)
        dirty = [l for l in out.splitlines() if l.strip() and "warning:" not in l]
        if dirty:
            print("refusing to run: these target files are already modified.\n"
                  "Commit or stash them first, or use --copy.\n  "
                  + "\n  ".join(dirty), file=sys.stderr)
            return 2

    # A clean baseline first: if the tree does not build and pass before any
    # mutation, every "caught" below is meaningless.
    print("baseline build...", flush=True)
    ok, out = build(root, build_dir)
    if not ok:
        print(out[-3000:], file=sys.stderr)
        print("baseline build failed; nothing below would mean anything", file=sys.stderr)
        return 1

    results = []
    for suite, edits, label, expect, needs in rows:
        if needs and not have_model:
            results.append((label, suite, "skipped", "no DEEPMOE_MODEL_DIR", True))
            continue

        applied = True
        for rel, old, new in edits:
            f = root / rel
            raw = f.read_bytes()
            _restore.setdefault(f, raw)
            # The patterns are written with LF; match against both line endings
            # so the same table works on a CRLF checkout.
            needle = old.encode()
            repl   = new.encode()
            if raw.count(needle) != 1:
                needle = old.replace(chr(10), chr(13) + chr(10)).encode()
                repl   = new.replace(chr(10), chr(13) + chr(10)).encode()
            if raw.count(needle) != 1:
                applied = False
                break
            f.write_bytes(raw.replace(needle, repl, 1))

        if not applied:
            _restore_all()
            results.append((label, suite, "BAD", "the pattern is gone or not unique", False))
            continue

        t0 = time.time()
        built, berr = build(root, build_dir)
        if not built:
            # A mutation that does not compile is not a test of the tests.
            _restore_all()
            build(root, build_dir)
            results.append((label, suite, "BAD", "the mutation did not compile", False))
            continue
        rc, _ = run([str(test_binary(build_dir)), suite + "."], root)
        _restore_all()

        got = "caught" if rc != 0 else "survived"
        ok_row = (got == "caught") if expect == "caught" else (got == "survived")
        results.append((label, suite, got, f"{time.time() - t0:.0f}s", ok_row))

    # Put the tree and the build back the way they were.
    _restore_all()
    print("restoring the baseline build...", flush=True)
    build(root, build_dir)

    w = max(len(r[0]) for r in results)
    print()
    for label, suite, got, note, ok_row in results:
        mark = "OK  " if ok_row else ("skip" if got == "skipped" else "BAD ")
        print(f"  {mark} {label:<{w}}  {suite:<16} {got:<9} {note}")
    bad = [r[0] for r in results if not r[4] and r[2] != "skipped"]
    skipped = [r[0] for r in results if r[2] == "skipped"]
    ran = len(results) - len(skipped)
    print(f"\n{ran - len(bad)}/{ran} mutations behaved as expected"
          + (f" -- unexpected: {', '.join(bad)}" if bad else "")
          + (f"\n{len(skipped)} skipped (set DEEPMOE_MODEL_DIR to run them)" if skipped else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
