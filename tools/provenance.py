"""What exactly a measurement ran: written by the machine when the run starts.

Every number that goes into docs/STATUS.md has to be traceable to the code,
binary, shaders and switches that produced it, without anyone copying those by
hand. `capture()` collects them; tools/hitrate_bench.py writes the result to
<run>/provenance.json before the first request, and tools/perf_report.py
--record copies it into the ledger next to the numbers.

    python tools/provenance.py [--exe build/deepmoe]     # print it for the current tree
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return ""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _file_sha(p: Path) -> str | None:
    try:
        return _sha(p.read_bytes())
    except OSError:
        return None


def _read(p: str) -> str | None:
    try:
        return Path(p).read_text().strip()
    except OSError:
        return None


def capture(exe: str | os.PathLike | None = None, env: dict[str, str] | None = None,
            shader_dir: str | os.PathLike | None = None) -> dict:
    """`env` is the environment the engine runs with (default: this process's)."""
    env = dict(os.environ if env is None else env)
    exe = Path(exe) if exe else ROOT / "build" / ("deepmoe.exe" if os.name == "nt" else "deepmoe")
    sdir = Path(shader_dir or env.get("DEEPMOE_SHADER_DIR") or exe.parent / "shaders")
    spv = sorted(sdir.glob("*.spv")) if sdir.is_dir() else []
    h = hashlib.sha256()
    for p in spv:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    dirty = [l[3:] for l in _git("status", "--porcelain", "--untracked-files=no").splitlines()]
    out = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "commit": _git("rev-parse", "--short=10", "HEAD").strip() or None,
        "dirty": dirty,
        "diff_sha": _sha(_git("diff", "HEAD").encode()) if dirty else None,
        "exe": str(exe),
        "exe_sha": _file_sha(exe),
        "shaders": {"dir": str(sdir), "count": len(spv), "sha": h.hexdigest()[:16] if spv else None},
        "env": {k: v for k, v in sorted(env.items()) if k.startswith("DEEPMOE_")},
    }
    if sys.platform.startswith("linux"):
        out["gpu_dpm"] = next((v for v in (_read(p) for p in sorted(
            str(x) for x in Path("/sys/class/drm").glob("card*/device/power_dpm_force_performance_level")))
            if v), None)
        out["gamemode_lib"] = any(Path(d, "libgamemode.so.0").exists()
                                  for d in ("/usr/lib", "/usr/lib64", "/usr/lib/x86_64-linux-gnu"))
    return out


def write(run_dir: str | os.PathLike, **kw) -> dict:
    p = capture(**kw)
    Path(run_dir, "provenance.json").write_text(json.dumps(p, indent=1) + "\n")
    return p


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe")
    a = ap.parse_args()
    print(json.dumps(capture(a.exe), indent=1))
