#!/usr/bin/env python3
"""Launch the web engine with the measured mask defaults and thermal guarding.

The HTTP process stays responsive while only its verified engine child is
paused. SIGINT requests a normal web shutdown and KV drain; shutdown remains
thermally guarded. Existing transcripts and the KV directory are preserved.
"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import errno
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import runtime_env
import runtime_defaults
from process_names import is_engine_comm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
from thermal_guard import (ProfileMonitor, ThermalLatch, assert_idle,
                           discover_sensors, guarded_temperatures, profile, sample)


def libc_pidfd_call(name, argtypes, values):
    """Keep pidfd binding on Python builds that omit the Linux APIs."""
    if sys.platform != "linux":
        raise OSError(errno.ENOSYS, "guarded launcher requires Linux pidfd support")
    library = ctypes.CDLL(None, use_errno=True)
    try:
        function = getattr(library, name)
    except AttributeError as error:
        raise OSError(errno.ENOSYS, f"{name} is unavailable in libc; pidfd support is required") from error
    function.argtypes = argtypes
    function.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = function(*values)
    if result < 0:
        code = ctypes.get_errno() or errno.EIO
        raise OSError(code, os.strerror(code), name)
    return result


def pidfd_open(pid):
    builtin = getattr(os, "pidfd_open", None)
    if callable(builtin):
        return builtin(pid)
    return libc_pidfd_call("pidfd_open", [ctypes.c_int, ctypes.c_uint], (pid, 0))


def pidfd_send_signal(fd, signum):
    builtin = getattr(signal, "pidfd_send_signal", None)
    if callable(builtin):
        return builtin(fd, signum)
    libc_pidfd_call("pidfd_send_signal",
                   [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint],
                   (fd, signum, None, 0))


def require_pidfd_support():
    # Check the actual kernel/libc combination before launching children or
    # changing power policy. Signal 0 probes only this process's bound pidfd.
    fd = pidfd_open(os.getpid())
    try:
        pidfd_send_signal(fd, 0)
    finally:
        os.close(fd)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--spec-k", type=int, choices=(2, 3, 5), default=runtime_defaults.profile_default("production", "spec_k"))
    parser.add_argument("--gpu-route", type=int, choices=(0, 1), default=runtime_defaults.profile_default("production", "gpu_route"))
    parser.add_argument("--port", type=int, default=runtime_defaults.WEB_PORT)
    parser.add_argument("--dry-run", action="store_true", help="print configuration without starting anything")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be in 1..65535")
    args.repo = args.repo.resolve()
    args.state_dir = (args.state_dir or args.repo / "build/web_mask").resolve()
    return args


def launch_configuration(args, inherited=None):
    # This is a controlled production launch. Shell benchmark switches must
    # not change the selected policy, including through a legacy alias.
    source = os.environ if inherited is None else inherited
    env = runtime_defaults.profile_environment("production", source, gpu_route=args.gpu_route)
    env.update(CACHEDMOE_MODEL_DIR=runtime_defaults.model_fallback(windows=False),
               CACHEDMOE_SHADER_DIR=runtime_defaults.resolve_launch(
                   runtime_defaults.executable(args.repo, windows=False), env).shader_dir)
    command = [sys.executable, str(args.repo / "tools/web/server.py"),
               "--exe", runtime_defaults.executable(args.repo, windows=False),
               "--resident-only", "mask", "--mask-cache", "dynamic",
               "--cache-slots", str(runtime_defaults.CACHE_SLOTS),
               "--max-context", str(runtime_defaults.MAX_CONTEXT),
               "--gpu-prefill-min", str(runtime_defaults.GPU_PREFILL_MIN), "--mirror",
               runtime_defaults.mirror_directory(),
               "--kv-dir", str(args.state_dir / "kv"), "--kv-max-gb", str(runtime_defaults.KV_DISK_GB),
               "--dspark", "--spec-k", str(args.spec_k), "--spec-top-k", str(runtime_defaults.ACCEPT_TOP_K),
               "--port", str(args.port), "--log", str(args.state_dir / "engine.log")]
    return command, env


def process_details(pid):
    """Identify a live process using kernel state, never a saved PID file."""
    root = Path(f"/proc/{pid}")
    try:
        fields = (root / "stat").read_text().rsplit(") ", 1)[1].split()
        details = dict(pid=pid, state=fields[0], ppid=int(fields[1]),
                       process_group=int(fields[2]), session=int(fields[3]),
                       start_ticks=int(fields[19]), comm=(root / "comm").read_text().strip())
        # A fast startup failure can already be a zombie, but its unreaped
        # stat still identifies the session that this Popen created.
        details["exe"] = None if fields[0] == "Z" else str((root / "exe").readlink())
        return details
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


@dataclass
class EngineChild:
    pid: int
    parent: int
    start_ticks: int
    exe: str
    pidfd: int

    def __post_init__(self):
        self.exe = str(Path(self.exe).resolve())

    def live(self):
        poller = select.poll()
        poller.register(self.pidfd, select.POLLIN)
        return not poller.poll(0)

    def matches(self):
        details = process_details(self.pid)
        return bool(details and details["state"] != "Z" and
                    details["ppid"] == self.parent and details["start_ticks"] == self.start_ticks and
                    is_engine_comm(details["comm"]) and details["exe"] == self.exe)

    def send(self, signum):
        if not self.live():
            return False
        # pidfd binds the signal to the discovered process even if its numeric
        # PID is later reused. Check ownership before each thermal transition.
        if not self.matches():
            raise RuntimeError("engine child identity changed; refusing to signal a saved PID")
        try:
            pidfd_send_signal(self.pidfd, signum)
            return True
        except ProcessLookupError:
            return False

    def close(self):
        os.close(self.pidfd)


def find_engine(server_pid, expected_exe):
    expected_exe = str(Path(expected_exe).resolve())
    children = Path(f"/proc/{server_pid}/task/{server_pid}/children")
    try:
        pids = [int(value) for value in children.read_text().split()]
    except FileNotFoundError:
        return None
    candidates = []
    for pid in pids:
        details = process_details(pid)
        if details and details["ppid"] == server_pid and details["state"] != "Z" and \
                is_engine_comm(details["comm"]) and details["exe"] == expected_exe:
            candidates.append(details)
    if len(candidates) > 1:
        raise RuntimeError("web server has more than one engine child")
    if not candidates:
        return None
    details = candidates[0]
    try:
        fd = pidfd_open(details["pid"])
    except ProcessLookupError:
        return None
    child = EngineChild(details["pid"], server_pid, details["start_ticks"], details["exe"], fd)
    if not child.matches():
        child.close()
        return None
    return child


def session_members(session):
    """Find current members of this launch's new session/process group."""
    members = []
    for entry in Path("/proc").glob("[0-9]*/stat"):
        details = process_details(int(entry.parent.name))
        if details and details["session"] == session and details["process_group"] == session:
            members.append(details)
    return members


@dataclass
class OwnedProcess:
    pid: int
    start_ticks: int
    session: int
    pidfd: int
    expected_engine: str
    observations: list

    def live(self):
        poller = select.poll()
        poller.register(self.pidfd, select.POLLIN)
        return not poller.poll(0)

    def observe(self, details):
        if (not details or details["state"] == "Z" or
                details["start_ticks"] != self.start_ticks or
                details["session"] != self.session or details["process_group"] != self.session):
            return False
        identity = {key: details[key] for key in ("comm", "exe", "ppid")}
        if not identity["comm"] or not identity["exe"]:
            return False
        if not self.observations or any(self.observations[-1][key] != value
                                        for key, value in identity.items()):
            self.observations.append(dict(observed_unix=time.time(), **identity))
        return True

    def is_engine(self):
        return bool(self.observations and is_engine_comm(self.observations[-1]["comm"]) and
                    self.observations[-1]["exe"] == self.expected_engine)

    def send(self, signum):
        if not self.live():
            return False
        if not self.observe(process_details(self.pid)) or not self.is_engine():
            raise RuntimeError("owned child is not the verified engine; refusing thermal signal")
        return bound_signal(self, signum)

    def close(self):
        os.close(self.pidfd)


class OwnedSession:
    """Bind this launch's children before or after exec, including orphans.

    Normal thermal signals still require the engine's comm and executable.
    Shutdown also tracks pre-exec children: aborting Serve's constructor must
    not leave a child that execs the engine after its parent has exited.
    """
    def __init__(self, server, expected_engine):
        leader = process_details(server.pid)
        if (not leader or leader["ppid"] != os.getpid() or
                leader["session"] != server.pid or leader["process_group"] != server.pid):
            raise RuntimeError("web Popen did not create the expected owned session")
        self.server = server
        self.session = server.pid
        self.start_ticks = leader["start_ticks"]
        self.expected_engine = str(Path(expected_engine).resolve())
        self.children = {}
        self.closed = False

    def discover(self):
        if self.closed:
            return
        exited_before_scan = self.server.poll() is not None
        members = session_members(self.session)
        leader = process_details(self.session)
        if leader and leader["start_ticks"] != self.start_ticks:
            self.closed = True
            raise RuntimeError("owned session leader PID was reused; refusing discovery")
        for details in members:
            if (details["pid"] == self.session or details["state"] == "Z" or
                    details["session"] != self.session or details["process_group"] != self.session or
                    details["start_ticks"] < self.start_ticks):
                continue
            key = (details["pid"], details["start_ticks"])
            child = self.children.get(key)
            if child is None:
                try:
                    fd = pidfd_open(details["pid"])
                except ProcessLookupError:
                    continue
                child = OwnedProcess(details["pid"], details["start_ticks"], self.session,
                                     fd, self.expected_engine, [])
                # PID/exec can change between the scan and pidfd_open. Bind
                # only the same birth in our session; record the actual image.
                if not child.observe(process_details(child.pid)):
                    child.close()
                    continue
                self.children[key] = child
            else:
                child.observe(details)
        if exited_before_scan and not self.live_children():
            # No member can fork after the last owned process exits. Do not
            # later adopt an unrelated session if its numeric ID is reused.
            self.closed = True

    def live_children(self):
        return [child for child in self.children.values() if child.live()]

    def engine(self):
        return next((child for child in self.live_children() if child.is_engine()), None)

    def receipt(self):
        return dict(session=self.session, server_start_ticks=self.start_ticks,
                    children=[dict(pid=child.pid, start_ticks=child.start_ticks,
                                   session=child.session, live=child.live(),
                                   observations=child.observations)
                              for child in self.children.values()])

    def close(self):
        for child in self.children.values():
            child.close()


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
        pidfd_send_signal(child.pidfd, signum)
        return True
    except ProcessLookupError:
        return False


def shutdown(server, child, paused, latch, sensors, monitor, log, thermal, owned=None):
    """Let Serve.close send quit and wait for KV writes, retaining thermal control."""
    clean = True
    if owned:
        owned.discover()
    if server.poll() is None:
        server.send_signal(signal.SIGINT)
    deadline = time.monotonic() + 180
    error_logged = False
    while server.poll() is None and time.monotonic() < deadline:
        try:
            if owned:
                owned.discover()
                if child is None:
                    child = owned.engine()
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
    def survivors():
        remaining = []
        if owned:
            # The HTTP process can already be gone, so use the original
            # session rather than /proc/<parent>/children. This also captures
            # a still-pre-exec child that appears during shutdown.
            owned.discover()
            remaining = owned.live_children()
        if child and child.live() and all(process.pid != child.pid for process in remaining):
            remaining.append(child)
        return remaining

    deadline = time.monotonic() + 2
    while survivors() and time.monotonic() < deadline:
        time.sleep(0.05)
    if survivors():
        print("ERROR: owned child survived web exit; forced cleanup, KV drain not confirmed",
              file=log, flush=True)
        clean = False
        for signum in (signal.SIGTERM, signal.SIGKILL):
            signalled = set()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                remaining = survivors()
                if not remaining:
                    break
                for process in remaining:
                    if process.pidfd not in signalled:
                        # Descriptors bind identities verified in our session,
                        # including pre-exec children. No saved PID is used.
                        bound_signal(process, signum)
                        if signum == signal.SIGTERM:
                            bound_signal(process, signal.SIGCONT)
                        signalled.add(process.pidfd)
                time.sleep(0.05)
        if survivors():
            raise RuntimeError("owned child did not exit after SIGKILL")
    if server.returncode != 0:
        print(f"ERROR: web process exited with {server.returncode}; KV drain is not confirmed",
              file=log, flush=True)
        clean = False
    return clean


def main(argv=None):
    args = arguments(argv)
    expected_exe = Path(runtime_defaults.executable(args.repo, windows=False)).resolve()
    command, env = launch_configuration(args)
    if args.dry_run:
        print(json.dumps(dict(command=command, env=runtime_env.raw_controls(env),
                              power_profile="performance"), indent=2))
        return 0
    require_pidfd_support()
    assert_idle()
    if not expected_exe.is_file():
        raise RuntimeError(f"engine is unavailable: {expected_exe}")
    if not Path(runtime_defaults.mirror_directory(), "deepmoe_manifest.json").is_file():
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
    server, child, owned, paused = None, None, None, False
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
            owned = OwnedSession(server, expected_exe)
            state["server_pid"] = server.pid
            startup_deadline = time.monotonic() + 180
            next_state = 0
            while server.poll() is None and not stopping:
                if child is None:
                    child = find_engine(server.pid, expected_exe)
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
            shutdown_error = None
            try:
                if server:
                    if owned is None and child is None:
                        child = find_engine(server.pid, expected_exe)
                    clean = shutdown(server, child, paused, latch, sensors, monitor, log, thermal, owned)
                    if not clean and failure is None:
                        failure = "web shutdown failed; see web.log"
            except Exception as error:
                shutdown_error = str(error)
                clean = False
                if failure is None:
                    failure = f"web shutdown failed: {error}"
                print(f"ERROR: web cleanup failed; KV drain is unconfirmed: {error}", file=log, flush=True)
            state["shutdown_owned_processes"] = owned.receipt() if owned else None
            if child:
                child.close()
            if owned:
                owned.close()
            state.update(stopped=True, paused=False, failure=failure,
                         shutdown_error=shutdown_error,
                         shutdown_graceful=clean,
                         kv_drain_status="web quit requested; no forced cleanup" if clean else "unconfirmed",
                         server_returncode=server.returncode if server else None,
                         stopped_unix=time.time())
            write_state(args.state_dir / "state.json", state)
    return 0 if clean and failure is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
