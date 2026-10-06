#!/usr/bin/env python3
"""Run serial GPU jobs on AC, pausing at 80 C and resuming at 72 C.

Only sensors that reached the pause threshold hold a pause open. Cooling is
reported in wall time, but the experiment budget counts active time. Raw
samples and absolute pause intervals are kept next to the supplied JSON plan.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time


import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import runtime_env
from process_names import GPU_COMM_PATTERN

class ThermalLatch:
    def __init__(self, pause=80.0, resume=72.0):
        if not math.isfinite(pause) or not math.isfinite(resume) or resume >= pause:
            raise ValueError("resume temperature must be below pause temperature")
        self.pause, self.resume = pause, resume
        self.hot = set()

    def update(self, temperatures):
        if not temperatures or any(not math.isfinite(v) for v in temperatures.values()):
            raise RuntimeError("temperature sensor unavailable or invalid")
        missing = self.hot.difference(temperatures)
        if missing:
            raise RuntimeError(f"missing latched temperature sensors: {sorted(missing)}")
        self.hot.update(k for k, v in temperatures.items() if v >= self.pause)
        self.hot = {k for k in self.hot if temperatures[k] > self.resume}
        return bool(self.hot)


class RunBudget:
    """Use monotonic time for limits; wall timestamps are for event alignment."""
    def __init__(self, now, active_limit, wall_limit):
        if active_limit <= 0 or wall_limit < active_limit:
            raise ValueError("wall budget must be at least the positive active budget")
        self.begin, self.active_limit, self.wall_limit = now, active_limit, wall_limit
        self.pause_start = None
        self.paused_s = 0.0

    def set_paused(self, paused, now):
        if paused and self.pause_start is None:
            self.pause_start = now
        elif not paused and self.pause_start is not None:
            self.paused_s += now - self.pause_start
            self.pause_start = None

    def elapsed(self, now):
        wall = now - self.begin
        paused = self.paused_s + (now - self.pause_start if self.pause_start is not None else 0)
        return wall, paused, max(0.0, wall - paused)

    def check(self, now):
        wall, _, active = self.elapsed(now)
        if active > self.active_limit:
            raise RuntimeError("experiment exceeded its active-time budget")
        if wall > self.wall_limit:
            raise RuntimeError("experiment exceeded its wall-time safety limit")


def profile():
    return subprocess.check_output(["powerprofilesctl", "get"], text=True, timeout=2).strip()


class ProfileMonitor:
    """Keep the slow CLI out of the 50 ms temperature/stop loop."""
    def __init__(self):
        self.value = profile()
        self.sample_unix = time.time()
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.wait(1):
            try:
                self.value = profile()
                self.sample_unix = time.time()
            except Exception as error:
                self.error = error
                return

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *unused):
        self.stop.set()
        self.thread.join(timeout=3)

    def state(self):
        if self.error:
            raise RuntimeError(f"power profile monitor failed: {self.error}")
        if time.time() - self.sample_unix > 4:
            raise RuntimeError("power profile sample is stale")
        return dict(power_profile=self.value, power_profile_sample_unix=self.sample_unix)


def discover_sensors():
    sensors = {}
    for name in Path("/sys/class/hwmon").glob("hwmon*/name"):
        kind = name.read_text().strip()
        node = name.parent / "temp1_input"
        if node.exists() and kind in ("amdgpu", "nvme", "k10temp"):
            sensors[f"{kind}:{name.parent.name}"] = node
    if not any(k.startswith("amdgpu:") for k in sensors):
        raise RuntimeError("GPU temperature sensor unavailable")
    return sensors


def sample(sensors, monitor):
    values = {k: int(p.read_text()) / 1000 for k, p in sensors.items()}
    values.update(wall_time_s=time.time(),
                  ac=int(Path("/sys/class/power_supply/AC0/online").read_text()),
                  **monitor.state())
    platform = Path("/sys/firmware/acpi/platform_profile")
    if platform.exists():
        values["platform_profile"] = platform.read_text().strip()
    devices = sorted(Path("/sys/class/drm").glob("card*/device/power_dpm_force_performance_level"))
    if devices:
        for name in ("gpu_busy_percent", "pp_dpm_sclk", "pp_dpm_mclk",
                     "power_dpm_force_performance_level"):
            node = devices[0].parent / name
            if node.exists():
                values[name] = node.read_text().strip()
    return values


def guarded_temperatures(values):
    # CPU is recorded; the established thresholds govern GPU and disks.
    return {k: v for k, v in values.items() if k.startswith(("amdgpu:", "nvme:"))}


def assert_idle():
    result = subprocess.run(["pgrep", "-x", GPU_COMM_PATTERN],
                            capture_output=True, text=True)
    if result.returncode != 1:
        raise RuntimeError(f"another GPU engine/test is present: {result.stdout.strip()}")


def group_exists(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def terminate(child):
    if not child:
        return
    # The wrapper can exit before its engine. Clean up the owned group even
    # when poll() already reports a wrapper exit; never leave an orphan engine.
    try:
        os.killpg(child.pid, signal.SIGTERM)
        os.killpg(child.pid, signal.SIGCONT)
    except ProcessLookupError:
        child.wait()
        return
    deadline = time.monotonic() + 3
    while group_exists(child.pid) and time.monotonic() < deadline:
        child.poll()
        time.sleep(0.05)
    if group_exists(child.pid):
        os.killpg(child.pid, signal.SIGKILL)
    child.wait()


def run_job(job, base, env, sensors):
    name = job["name"]
    log_path = base / f"{name}.log"
    thermal_path = base / f"{name}_thermal.jsonl"
    if log_path.exists() or thermal_path.exists():
        raise RuntimeError(f"refusing to repeat existing configuration: {name}")
    target = job.get("profile", "performance")
    subprocess.run(["powerprofilesctl", "set", target], check=True)
    assert_idle()
    with ProfileMonitor() as monitor:
        deadline = time.monotonic() + 120
        while any(v > 60 for k, v in sample(sensors, monitor).items() if k.startswith("amdgpu:")):
            if time.monotonic() >= deadline:
                raise RuntimeError("GPU did not cool below 60 C")
            time.sleep(0.2)
        start = sample(sensors, monitor)
        if start["ac"] != 1 or start["power_profile"] != target:
            raise RuntimeError("experiment requires AC and the requested power profile")
        peak = {k: start[k] for k in sensors}
        latch = ThermalLatch()
        active_limit = job.get("active_timeout_s", job.get("timeout_s", 600))
        wall_limit = job.get("wall_timeout_s", min(14400, max(active_limit * 4, active_limit + 1800)))
        budget = RunBudget(time.monotonic(), active_limit, wall_limit)
        child = None
        paused = False
        pause_start = None
        intervals = []
        transitions = []
        failure, rc = None, 99
        print("START", name, flush=True)
        try:
            with log_path.open("w") as log, thermal_path.open("w") as thermal:
                child_env = dict(env)
                runtime_env.apply_overrides(child_env, job.get("env", {}))
                runtime_env.set_value(child_env, "CACHEDMOE_THERMAL_LOG", str(thermal_path))
                child = subprocess.Popen(job["command"], env=child_env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                while child.poll() is None:
                    values = sample(sensors, monitor)
                    now = time.monotonic()
                    for sensor in sensors:
                        peak[sensor] = max(peak[sensor], values[sensor])
                    if values["ac"] != 1 or values["power_profile"] != target:
                        raise RuntimeError("AC or requested power profile changed during the job")
                    should_pause = latch.update(guarded_temperatures(values))
                    if should_pause != paused:
                        os.killpg(child.pid, signal.SIGSTOP if should_pause else signal.SIGCONT)
                        paused = should_pause
                        budget.set_paused(paused, now)
                        transitions.append(dict(wall_time_s=values["wall_time_s"], paused=paused,
                                                sensors=guarded_temperatures(values),
                                                latched_sensors=sorted(latch.hot)))
                        if paused:
                            pause_start = values["wall_time_s"]
                        else:
                            intervals.append([pause_start, values["wall_time_s"]])
                            pause_start = None
                        print("PAUSE" if paused else "RESUME", name, sorted(latch.hot), flush=True)
                    wall, cooling, active = budget.elapsed(now)
                    values.update(elapsed_s=wall, active_elapsed_s=active, thermal_paused_s=cooling,
                                  paused=paused, latched_sensors=sorted(latch.hot))
                    thermal.write(json.dumps(values) + "\n")
                    thermal.flush()
                    budget.check(now)
                    time.sleep(0.05)
                rc = child.wait()
            if re.search(r"(?<!\d)0 case\(s\) run", log_path.read_text()):
                rc, failure = 99, "zero test cases executed"
        except Exception as error:
            failure = str(error)
        finally:
            terminate(child)
        end, end_sample_error = None, None
        try:
            end = sample(sensors, monitor)
        except Exception as error:
            # A persistent sensor/profile error also affects this read. Keep
            # the original failure and its receipt after cleaning up the job;
            # an unavailable final read must not look like a successful run.
            end_sample_error = str(error)
            if failure is None:
                failure = f"failed to collect final thermal sample: {error}"
            if rc == 0:
                rc = 99
        finished_unix = time.time()
        if pause_start is not None:
            pause_end = end["wall_time_s"] if end is not None else finished_unix
            intervals.append([pause_start, pause_end])
        wall, cooling, active = budget.elapsed(time.monotonic())
        result = dict(name=name, command=job["command"], env=job.get("env", {}), profile=target,
                      process_group=child.pid if child else None,
                      rc=rc, failure=failure, start=start, end=end, peak=peak,
                      end_sample_error=end_sample_error, finished_unix=finished_unix,
                      elapsed_s=wall, active_elapsed_s=active, thermal_paused_s=cooling,
                      active_timeout_s=active_limit, wall_timeout_s=wall_limit,
                      thermal_pauses=sum(t["paused"] for t in transitions),
                      thermal_pause_intervals=intervals, thermal_transitions=transitions,
                      thermal_log=str(thermal_path), thermal_policy="per-sensor latched 80/72 C",
                      sampling_period_s=0.05,
                      ac_changed=end["ac"] != 1 if end is not None else None,
                      time=datetime.datetime.now().astimezone().isoformat())
        print("DONE", json.dumps(result), flush=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    base = args.plan.resolve().parent
    os.chdir(plan["cwd"])
    env = dict(os.environ)
    runtime_env.setdefault(env, "CACHEDMOE_MODEL_DIR", str(Path.home() / "models/DeepSeek-V4.1-Flash"))
    runtime_env.setdefault(env, "CACHEDMOE_LONGCTX_DIR", str(Path.cwd() / "traces/longctx"))
    runtime_env.apply_overrides(env, dict(CACHEDMOE_MIRROR_AUTO="0", CACHEDMOE_DSPARK_MEGA="0"))
    runtime_env.apply_overrides(env, plan.get("env", {}))
    runtime_env.clear(env, "CACHEDMOE_MODEL_MIRRORS")
    sensors = discover_sensors()
    results, original = [], profile()

    def interrupt(signum, frame):
        raise RuntimeError(f"interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        for job in plan["jobs"]:
            result = run_job(job, base, env, sensors)
            results.append(result)
            (base / "check_results.json").write_text(json.dumps(results, indent=2) + "\n")
            if result["rc"]:
                return result["rc"]
    finally:
        receipt = dict(original=original, final=None, requested_default="performance")
        try:
            receipt["final"] = profile()
        except Exception as error:
            receipt["final_read_error"] = str(error)
        (base / "profile_receipt.json").write_text(json.dumps(receipt) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
