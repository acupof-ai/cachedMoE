#!/usr/bin/env python3
"""Launch the web engine with the measured mask defaults and thermal guarding.

The HTTP process stays responsive while only its verified engine child is
paused. SIGINT requests a normal web shutdown and KV drain; shutdown remains
thermally guarded. Existing transcripts and the KV directory are preserved.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
from thermal_guard import (ProfileMonitor, ThermalLatch, assert_idle,
                           discover_sensors, guarded_temperatures, profile, sample)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--spec-k", type=int, choices=(2, 3, 5), default=2)
    parser.add_argument("--gpu-route", type=int, choices=(0, 1), default=0)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--dry-run", action="store_true", help="print configuration without starting anything")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be in 1..65535")
    args.repo = args.repo.resolve()
    args.state_dir = (args.state_dir or args.repo / "build/web_mask").resolve()
    return args


def launch_configuration(args, inherited=None):
    env = dict(os.environ if inherited is None else inherited)
    for key in ("DEEPMOE_SPEC_DIAGNOSTICS", "DEEPMOE_DSPARK_MEGA_DIAG",
                "DEEPMOE_MASK_WAIT_TAU", "DEEPMOE_MASK_WAIT_BUDGET",
                "DEEPMOE_THERMAL_LOG", "DEEPMOE_IO_ENGRAM_DEADLINE",
                "DEEPMOE_MODEL_MIRRORS"):
        env.pop(key, None)
    env.update(DEEPMOE_MODEL_DIR=str(Path.home() / "models/DeepSeek-V4.1-Flash"),
               DEEPMOE_SHADER_DIR=str(args.repo / "build/shaders"),
               DEEPMOE_MIRROR_AUTO="0",
               DEEPMOE_MASK_DYNAMIC_LRU="1", DEEPMOE_DSPARK_PROFILE="0",
               DEEPMOE_DSPARK_ONECB="1", DEEPMOE_BATCH_GPU_ROUTE=str(args.gpu_route),
               DEEPMOE_DSPARK_MEGA="0", DEEPMOE_DSPARK_TRIM_TAIL="1",
               DEEPMOE_SPEC_GPU_READOUT="1", DEEPMOE_MGT_PAIR_DOT="0",
               DEEPMOE_MGT_ATTN_CM="0", DEEPMOE_MGT_FOLD_SCALE="0")
    command = [sys.executable, str(args.repo / "tools/web/server.py"),
               "--exe", str(args.repo / "build/deepmoe"),
               "--resident-only", "mask", "--mask-cache", "dynamic",
               "--cache-slots", "5500", "--max-context", "1048576",
               "--gpu-prefill-min", "16", "--mirror",
               "/mnt/deepmoe2/models/DeepSeek-V4.1-Flash",
               "--kv-dir", str(args.state_dir / "kv"), "--kv-max-gb", "4",
               "--dspark", "--spec-k", str(args.spec_k), "--spec-top-k", "4",
               "--port", str(args.port), "--log", str(args.state_dir / "engine.log")]
    return command, env


def process_details(pid):
    """Identify a live process using kernel state, never a saved PID file."""
    root = Path(f"/proc/{pid}")
    try:
        fields = (root / "stat").read_text().rsplit(") ", 1)[1].split()
        return dict(pid=pid, state=fields[0], ppid=int(fields[1]), start_ticks=int(fields[19]),
                    comm=(root / "comm").read_text().strip(), exe=str((root / "exe").readlink()))
    except (FileNotFoundError, ProcessLookupError):
        return None


@dataclass
class EngineChild:
    pid: int
    parent: int
    start_ticks: int
    exe: str
    pidfd: int

    def live(self):
        poller = select.poll()
        poller.register(self.pidfd, select.POLLIN)
        return not poller.poll(0)

    def matches(self):
        details = process_details(self.pid)
        return bool(details and details["state"] != "Z" and
                    details["ppid"] == self.parent and details["start_ticks"] == self.start_ticks and
                    details["comm"] == "deepmoe" and details["exe"] == self.exe)

    def send(self, signum):
        if not self.live():
            return False
        # pidfd binds the signal to the discovered process even if its numeric
        # PID is later reused. Check ownership before each thermal transition.
        if not self.matches():
            raise RuntimeError("engine child identity changed; refusing to signal a saved PID")
        try:
            signal.pidfd_send_signal(self.pidfd, signum)
            return True
        except ProcessLookupError:
            return False

    def close(self):
        os.close(self.pidfd)


def find_engine(server_pid, expected_exe):
    children = Path(f"/proc/{server_pid}/task/{server_pid}/children")
    try:
        pids = [int(value) for value in children.read_text().split()]
    except FileNotFoundError:
        return None
    candidates = []
    for pid in pids:
        details = process_details(pid)
        if details and details["ppid"] == server_pid and details["state"] != "Z" and \
                details["comm"] == "deepmoe" and details["exe"] == str(expected_exe):
            candidates.append(details)
    if len(candidates) > 1:
        raise RuntimeError("web server has more than one engine child")
    if not candidates:
        return None
    details = candidates[0]
    try:
        fd = os.pidfd_open(details["pid"])
    except ProcessLookupError:
        return None
    child = EngineChild(details["pid"], server_pid, details["start_ticks"], details["exe"], fd)
    if not child.matches():
        child.close()
        return None
    return child


def thermal_transition(child, values, latch, paused, force_hold=False):
    should_pause = latch.update(guarded_temperatures(values)) or force_hold
    if child and should_pause != paused:
        if child.send(signal.SIGSTOP if should_pause else signal.SIGCONT):
            paused = should_pause
    return paused


def power_valid(values):
    return (values["ac"] == 1 and values["power_profile"] == "performance" and
            values.get("platform_profile", "performance") == "performance")


def write_state(path, state):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def bound_signal(child, signum):
    """Cleanup can outlive the parent; the held pidfd still pins its engine."""
    try:
        signal.pidfd_send_signal(child.pidfd, signum)
        return True
    except ProcessLookupError:
        return False


def shutdown(server, child, paused, latch, sensors, monitor, log, thermal):
    """Let Serve.close send quit and wait for KV writes, retaining thermal control."""
    clean = True
    if server.poll() is None:
        server.send_signal(signal.SIGINT)
    deadline = time.monotonic() + 180
    error_logged = False
    while server.poll() is None and time.monotonic() < deadline:
        try:
            values = sample(sensors, monitor)
            paused = thermal_transition(child, values, latch, paused, not power_valid(values))
            values.update(paused=paused, shutdown=True, latched_sensors=sorted(latch.hot))
            thermal.write(json.dumps(values) + "\n")
            thermal.flush()
        except Exception as error:
            if child and child.live():
                # The bound pidfd remains safe if the parent exits between
                # sample and ownership check. Hold that original engine.
                bound_signal(child, signal.SIGSTOP)
                paused = True
            if not error_logged:
                print(f"shutdown held: {error}", file=log, flush=True)
                error_logged = True
            clean = False
        time.sleep(0.05)
    if server.poll() is None:
        print("ERROR: web shutdown exceeded 180 s; KV drain is not confirmed", file=log, flush=True)
        clean = False
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
    if child and child.live():
        # A web exit should reap its engine. An orphan requires an explicit
        # failure receipt; never silently treat forced termination as KV drain.
        deadline = time.monotonic() + 2
        while child.live() and time.monotonic() < deadline:
            time.sleep(0.05)
        if child.live():
            print("ERROR: engine survived web exit; forced cleanup, KV drain not confirmed",
                  file=log, flush=True)
            clean = False
            # Ownership may have changed to init after the parent's exit. The
            # held pidfd still identifies the original engine, unlike a PID.
            bound_signal(child, signal.SIGTERM)
            bound_signal(child, signal.SIGCONT)
            deadline = time.monotonic() + 5
            while child.live() and time.monotonic() < deadline:
                time.sleep(0.05)
            if child.live():
                bound_signal(child, signal.SIGKILL)
                deadline = time.monotonic() + 5
                while child.live() and time.monotonic() < deadline:
                    time.sleep(0.05)
                if child.live():
                    raise RuntimeError("owned engine did not exit after SIGKILL")
    if server.returncode != 0:
        print(f"ERROR: web process exited with {server.returncode}; KV drain is not confirmed",
              file=log, flush=True)
        clean = False
    return clean


def main(argv=None):
    args = arguments(argv)
    command, env = launch_configuration(args)
    if args.dry_run:
        print(json.dumps(dict(command=command, env={k: v for k, v in env.items()
                         if k.startswith("DEEPMOE_")}, power_profile="performance"), indent=2))
        return 0
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("this guarded launcher requires Linux pidfd support")
    assert_idle()
    if not (args.repo / "build/deepmoe").is_file():
        raise RuntimeError("build/deepmoe is unavailable")
    if not Path("/mnt/deepmoe2/models/DeepSeek-V4.1-Flash/deepmoe_manifest.json").is_file():
        raise RuntimeError("the requested second checkpoint read source is unavailable")
    args.state_dir.mkdir(parents=True, exist_ok=True)
    original = profile()
    subprocess.run(["powerprofilesctl", "set", "performance"], check=True)
    sensors = discover_sensors()
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    server, child, paused = None, None, False
    latch, failure, clean = ThermalLatch(), None, False
    state = dict(guard_pid=os.getpid(), server_pid=None, engine_pid=None, paused=False,
                 profile="performance", original_profile=original, command=command,
                 spec_k=args.spec_k, gpu_route=bool(args.gpu_route), stopped=False)
    with (args.state_dir / "web.log").open("a") as log, \
            (args.state_dir / "thermal.jsonl").open("a") as thermal, ProfileMonitor() as monitor:
        try:
            values = sample(sensors, monitor)
            if not power_valid(values):
                raise RuntimeError("web engine requires AC and performance profile")
            assert_idle()
            server = subprocess.Popen(command, cwd=args.repo, env=env, stdin=subprocess.DEVNULL,
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state["server_pid"] = server.pid
            startup_deadline = time.monotonic() + 180
            next_state = 0
            while server.poll() is None and not stopping:
                if child is None:
                    child = find_engine(server.pid, args.repo / "build/deepmoe")
                    if child:
                        state.update(engine_pid=child.pid, engine_start_ticks=child.start_ticks)
                    elif time.monotonic() > startup_deadline:
                        raise RuntimeError("web engine child did not start within 180 s")
                elif not child.live():
                    raise RuntimeError("web engine exited while HTTP process was still running")
                values = sample(sensors, monitor)
                if not power_valid(values):
                    paused = thermal_transition(child, values, latch, paused, force_hold=True)
                    raise RuntimeError("AC or performance profile changed; stopping web engine")
                before = paused
                paused = thermal_transition(child, values, latch, paused)
                values.update(paused=paused, latched_sensors=sorted(latch.hot), engine_pid=state["engine_pid"])
                thermal.write(json.dumps(values) + "\n")
                thermal.flush()
                if paused != before:
                    print("web thermal:", "paused" if paused else "resumed", values,
                          file=log, flush=True)
                if time.monotonic() >= next_state or paused != before:
                    state.update(paused=paused, temperatures=guarded_temperatures(values),
                                 power_profile=values["power_profile"],
                                 platform_profile=values.get("platform_profile"),
                                 gpu_dpm=values.get("power_dpm_force_performance_level"),
                                 sampled_unix=values["wall_time_s"])
                    write_state(args.state_dir / "state.json", state)
                    next_state = time.monotonic() + 1
                time.sleep(0.05)
        except Exception as error:
            failure = str(error)
            print(f"ERROR: guarded web launcher: {failure}", file=log, flush=True)
        finally:
            if server:
                if child is None:
                    child = find_engine(server.pid, args.repo / "build/deepmoe")
                clean = shutdown(server, child, paused, latch, sensors, monitor, log, thermal)
                if not clean and failure is None:
                    failure = "web shutdown failed; see web.log"
            if child:
                child.close()
            state.update(stopped=True, paused=False, failure=failure,
                         shutdown_graceful=clean,
                         kv_drain_status="web quit requested; no forced cleanup" if clean else "unconfirmed",
                         server_returncode=server.returncode if server else None,
                         stopped_unix=time.time())
            write_state(args.state_dir / "state.json", state)
    return 0 if clean and failure is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
