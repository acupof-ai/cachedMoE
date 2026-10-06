"""What exactly a measurement ran: written by the machine when the run starts.

Every number that goes into docs/STATUS.md has to be traceable to the code,
binary, shaders and switches that produced it, without anyone copying those by
hand. `capture()` collects them; tools/hitrate_bench.py writes the result to
<run>/provenance.json before the first request, and tools/perf_report.py
--record copies it into the ledger next to the numbers.

    python tools/provenance.py [--exe build/cachedmoe]     # print it for the current tree
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


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import runtime_env
import runtime_defaults
from process_names import is_gpu_process

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


def power_state() -> dict:
    """Read the actual platform mode and clocks, independent of requested flags."""
    state = dict(power_profile=None, platform_profile=None, ac_online=None,
                 gpu_dpm=None, pp_dpm_sclk=None, pp_dpm_mclk=None,
                 gpu_busy_percent=None, sampled_unix=time.time())
    if not sys.platform.startswith("linux"):
        return state
    try:
        state["power_profile"] = subprocess.check_output(
            ["powerprofilesctl", "get"], text=True, timeout=2).strip()
    except (OSError, subprocess.SubprocessError):
        pass
    state["platform_profile"] = _read("/sys/firmware/acpi/platform_profile")
    ac = _read("/sys/class/power_supply/AC0/online")
    state["ac_online"] = int(ac) if ac in ("0", "1") else None
    devices = sorted(Path("/sys/class/drm").glob("card*/device/power_dpm_force_performance_level"))
    if devices:
        device = devices[0].parent
        state["gpu_dpm"] = _read(str(device / "power_dpm_force_performance_level"))
        for name in ("pp_dpm_sclk", "pp_dpm_mclk", "gpu_busy_percent"):
            state[name] = _read(str(device / name))
    return state


def disk_snapshot() -> dict:
    """Sectors read / written so far on every whole disk (/proc/diskstats)."""
    out = {"t": time.time()}
    try:
        for line in Path("/proc/diskstats").read_text().splitlines():
            f = line.split()
            name = f[2]
            if (name.startswith("nvme") and "p" not in name[4:]) or \
               (name.startswith("sd") and name[2:].isalpha()):
                out[name] = [int(f[5]), int(f[9])]
    except OSError:
        pass
    return out


def disk_delta(a: dict, b: dict) -> dict:
    """MB read / written per disk between two snapshots."""
    d = {"seconds": round(b["t"] - a["t"], 1)}
    for k, v in a.items():
        if k != "t" and k in b:
            d[k] = {"read_MB": round((b[k][0] - v[0]) * 512 / 1e6, 1),
                    "write_MB": round((b[k][1] - v[1]) * 512 / 1e6, 1)}
    return d


def finish(run_dir: str | os.PathLike) -> dict | None:
    """At the end of a run: add what the disks did while it ran. Writes by
    anything else on the model's drive (a Steam download, a backup) show up
    here -- the engine itself writes only its few-MB logs."""
    pf = Path(run_dir, "provenance.json")
    if not pf.exists():
        return None
    p = json.loads(pf.read_text())
    if "disks_start" not in p:
        return None
    p["disks_during_run"] = disk_delta(p["disks_start"], disk_snapshot())
    pf.write_text(json.dumps(p, indent=1) + "\n")
    return p["disks_during_run"]


def idle_check(seconds: float = 1.0) -> dict:
    """How quiet the machine was when the run started: GPU busy % sampled for
    `seconds`, the load average, and the busiest other processes. A browser at
    a few % GPU shares the same LPDDR5X and moves every kernel by a few %."""
    busy = []
    nodes = sorted(Path("/sys/class/drm").glob("card*/device/gpu_busy_percent"))
    t_end = time.time() + seconds
    while nodes and time.time() < t_end:
        v = _read(str(nodes[0]))
        if v is not None:
            busy.append(int(v))
        time.sleep(0.05)
    top, gpu_processes = [], []
    try:
        ps = subprocess.run(["ps", "-eo", "pcpu,comm", "--sort=-pcpu", "--no-headers"],
                            capture_output=True, text=True).stdout.splitlines()
        me = {"ps", "python", "python3"}
        for line in ps:
            pc, _, comm = line.strip().partition(" ")
            if is_gpu_process(comm.strip()):
                gpu_processes.append(f"{comm.strip()} {pc}%")
        for line in ps:
            pc, _, comm = line.strip().partition(" ")
            if comm.strip() in me:
                continue
            if float(pc) < 2.0 or len(top) >= 4:
                break
            top.append(f"{comm.strip()} {pc}%")
    except (OSError, ValueError):
        pass
    return {"gpu_busy_mean": round(sum(busy) / len(busy), 1) if busy else None,
            "gpu_busy_max": max(busy) if busy else None,
            "loadavg": (_read("/proc/loadavg") or "").split()[:3], "top_cpu": top,
            "gpu_processes": gpu_processes}


def capture(exe: str | os.PathLike | None = None, env: dict[str, str] | None = None,
            shader_dir: str | os.PathLike | None = None,
            launch_config: runtime_defaults.LaunchConfig | None = None) -> dict:
    """`env` is the environment the engine runs with (default: this process's)."""
    config = launch_config or runtime_defaults.resolve_launch(exe, env, shader_dir=shader_dir)
    env = config.environment
    exe = Path(config.exe)
    # An empty native shader directory is not an instruction to hash Python's
    # working directory. Keep it observable without inventing resolved files.
    sdir = Path(config.shader_dir) if config.shader_dir else None
    spv = sorted(sdir.glob("*.spv")) if sdir is not None and sdir.is_dir() else []
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
        "launch": config.receipt(),
        "exe": str(exe),
        "exe_resolved": str(exe.resolve()),
        "exe_sha": _file_sha(exe),
        "shaders": {"dir": str(sdir) if sdir is not None else "", "count": len(spv),
                    "sha": h.hexdigest()[:16] if spv else None},
        "env": runtime_env.raw_controls(dict(sorted(env.items()))),
        "env_effective": {
            key: {"value": selected.value, "source": selected.source,
                  "selected_key": selected.selected_key}
            for key, selected in runtime_env.effective_controls(
                dict(sorted(env.items())), warn=True).items()
        },
    }
    if sys.platform.startswith("linux"):
        out["power"] = power_state()
        out["gpu_dpm"] = out["power"]["gpu_dpm"]
        out["idle"] = idle_check()
        out["disks_start"] = disk_snapshot()
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
