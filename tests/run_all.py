#!/usr/bin/env python3
"""Every gate that runs without a GPU, in one command, with one exit code.

This is what to run before buying machine time, before a commit, and before
believing any number in docs/STATUS.md. Four kinds of gate, in the order a
regression usually trips them:

  1. the CPU ctest suites -- found by asking ctest itself for the `unit` label,
     so a new suite registered in tests/CMakeLists.txt is picked up without
     editing a list here;
  2. the CPU suites that need only the checkpoint's METADATA (tokenizer.json,
     config.json) and no GPU: `suite.tokenizer`, `suite.engram_tables`. They
     run when DEEPMOE_MODEL_DIR is set and are reported as skipped otherwise;
  3. the Python tools' own self-tests -- today `tools/trace_timeline.py
     --self-test`, which parses a synthetic trace and so catches a drift
     between the reader and runtime/trace.h without a GPU;
  4. --mutate: break the code on purpose and require the suite to notice
     (tests/mutate.py). A green suite proves nothing on its own.

Nothing here starts an engine process or touches the GPU, so it is safe to run
while someone else owns the device.

    python tests/run_all.py                  # gates 1-3
    python tests/run_all.py --mutate         # and the mutation audit (slow)
    python tests/run_all.py --build          # build first
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# Suites that need the checkpoint's metadata but no GPU. ctest labels them
# `needs-model`, which is correct -- but they are CPU gates and belong here.
METADATA_SUITES = ["suite.tokenizer", "suite.engram_tables"]


def run(cmd: list[str], cwd: Path = ROOT) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    except FileNotFoundError as e:
        return 127, str(e)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def ctest_unit_suites(build: Path) -> list[str]:
    """Ask ctest which tests carry the `unit` label. Asking beats a hard-coded
    list: tests/CMakeLists.txt has forgotten a label at least once
    (`suite.kvdisk` had none at all until Track W), and a list here would have
    hidden that instead of surfacing it."""
    rc, out = run(["ctest", "--test-dir", str(build), "-L", "unit", "-N"])
    if rc != 0:
        return []
    names = re.findall(r"^\s*Test\s+#\d+:\s+(\S+)\s*$", out, re.M)
    return [n for n in names if n != "deepmoe_tests"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-dir", default=str(ROOT / "build"))
    ap.add_argument("--build", action="store_true", help="cmake --build first")
    ap.add_argument("--mutate", action="store_true",
                    help="also run tests/mutate.py (one incremental build per mutation)")
    a = ap.parse_args()

    build = Path(a.build_dir)
    rows: list[tuple[str, str, str, bool]] = []

    if a.build:
        t0 = time.time()
        rc, out = run(["cmake", "--build", str(build)])
        if rc != 0 and "cache" in out.lower():          # the zig cache race
            rc, out = run(["cmake", "--build", str(build)])
        rows.append(("build", "ok" if rc == 0 else out.strip().splitlines()[-1][:70],
                     f"{time.time() - t0:.0f}s", rc == 0))
        if rc != 0:
            print(out[-3000:], file=sys.stderr)

    if not build.exists():
        print(f"no build directory at {build}. Configure it first:\n"
              "  cmake -S . -B build -G Ninja "
              "-DCMAKE_TOOLCHAIN_FILE=cmake/zig-toolchain.cmake -DCMAKE_BUILD_TYPE=Release",
              file=sys.stderr)
        return 1

    # --- 1. the CPU ctest suites ------------------------------------------
    suites = ctest_unit_suites(build)
    if not suites:
        rows.append(("ctest -L unit", "ctest listed no unit suites", "", False))
    for name in suites:
        t0 = time.time()
        rc, out = run(["ctest", "--test-dir", str(build), "-R", f"^{re.escape(name)}$",
                       "--output-on-failure"])
        m = re.search(r"(\d+) case\(s\) run, (\d+) failed", out)
        detail = (f"{m.group(1)} cases, {m.group(2)} failed" if m
                  else ("passed" if rc == 0 else "FAILED"))
        rows.append((name, detail, f"{time.time() - t0:.1f}s", rc == 0))

    # --- 2. the metadata-only suites --------------------------------------
    have_model = bool(os.environ.get("DEEPMOE_MODEL_DIR"))
    for name in METADATA_SUITES:
        if not have_model:
            rows.append((name, "skipped: set DEEPMOE_MODEL_DIR", "", True))
            continue
        t0 = time.time()
        rc, out = run(["ctest", "--test-dir", str(build), "-R", f"^{re.escape(name)}$",
                       "--output-on-failure"])
        skipped = "SKIP " in out
        m = re.search(r"(\d+) case\(s\) run, (\d+) failed", out)
        detail = ("self-skipped (the suite said SKIP)" if skipped else
                  (f"{m.group(1)} cases, {m.group(2)} failed" if m else "passed"))
        rows.append((name, detail, f"{time.time() - t0:.1f}s", rc == 0))

    # --- 3. the Python tools' self-tests ----------------------------------
    for label, cmd in (
        ("tools/trace_timeline.py", [PY, str(ROOT / "tools" / "trace_timeline.py"), "--self-test"]),
        ("tests/mutate.py --list", [PY, str(ROOT / "tests" / "mutate.py"), "--list"]),
    ):
        t0 = time.time()
        rc, out = run(cmd)
        tail = [l for l in out.splitlines() if l.strip()]
        rows.append((label, tail[-1].strip()[:70] if tail else "", f"{time.time() - t0:.1f}s",
                     rc == 0))

    # --- 4. the mutation gate ---------------------------------------------
    if a.mutate:
        t0 = time.time()
        rc, out = run([PY, str(ROOT / "tests" / "mutate.py"), "--build-dir", str(build)])
        tail = [l for l in out.splitlines() if "behaved as expected" in l]
        rows.append(("mutation gate", tail[-1].strip() if tail else "no verdict",
                     f"{time.time() - t0:.0f}s", rc == 0))
        if rc != 0:
            print(out[-4000:], file=sys.stderr)

    width = max(len(r[0]) for r in rows)
    print()
    for name, detail, secs, ok in rows:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}"
              + (f"   ({secs})" if secs else ""))
    failed = [r[0] for r in rows if not r[3]]
    print(f"\n{len(rows) - len(failed)}/{len(rows)} gates passed"
          + (f" -- failed: {', '.join(failed)}" if failed else ""))
    if not a.mutate:
        print("(add --mutate to require the suite to catch injected regressions)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
