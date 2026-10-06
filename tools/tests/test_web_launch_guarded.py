"""Launcher defaults and engine-only safety checks; never launch a GPU engine."""
import io
import errno
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/web"))
import launch_guarded as launch


class ThermalConfiguration(unittest.TestCase):
    def test_selected_thresholds_reach_server_without_environment_overrides(self):
        args = launch.arguments(["--gpu-pause-c", "83", "--gpu-resume-c", "75"])
        command, env = launch.launch_configuration(args, {})
        actual = json.loads(command[command.index("--thermal-policy-json") + 1])
        self.assertEqual(actual, args.thermal_policy.record())
        self.assertEqual(actual["gpu_pause_c"], 83)
        self.assertEqual(actual["nvme_pause_c"], 80)
        self.assertEqual(actual["nvme_start_c"], 65)


class PortablePidfd(unittest.TestCase):
    def test_builtins_are_preferred(self):
        with patch.object(launch.os, "pidfd_open", return_value=99, create=True) as opening, \
                patch.object(launch.signal, "pidfd_send_signal", create=True) as sending, \
                patch.object(launch.ctypes, "CDLL") as load:
            self.assertEqual(launch.pidfd_open(123), 99)
            launch.pidfd_send_signal(99, signal.SIGSTOP)
            opening.assert_called_once_with(123)
            sending.assert_called_once_with(99, signal.SIGSTOP)
            load.assert_not_called()

    def test_missing_builtins_use_typed_libc_calls(self):
        library = SimpleNamespace(pidfd_open=Mock(return_value=99),
                                  pidfd_send_signal=Mock(return_value=0))
        with patch.object(launch.os, "pidfd_open", None, create=True), \
                patch.object(launch.signal, "pidfd_send_signal", None, create=True), \
                patch.object(launch.ctypes, "CDLL", return_value=library) as load, \
                patch.object(launch.os, "kill") as unbound:
            self.assertEqual(launch.pidfd_open(123), 99)
            self.assertIsNone(launch.pidfd_send_signal(99, signal.SIGSTOP))
            library.pidfd_open.assert_called_once_with(123, 0)
            library.pidfd_send_signal.assert_called_once_with(99, signal.SIGSTOP, None, 0)
            self.assertEqual(library.pidfd_open.argtypes, [launch.ctypes.c_int, launch.ctypes.c_uint])
            self.assertEqual(library.pidfd_send_signal.argtypes,
                             [launch.ctypes.c_int, launch.ctypes.c_int,
                              launch.ctypes.c_void_p, launch.ctypes.c_uint])
            self.assertIs(library.pidfd_open.restype, launch.ctypes.c_int)
            self.assertIs(library.pidfd_send_signal.restype, launch.ctypes.c_int)
            self.assertEqual(load.call_count, 2)
            load.assert_called_with(None, use_errno=True)
            unbound.assert_not_called()

    def test_libc_errno_keeps_lookup_and_permission_error_semantics(self):
        for name, code, expected in (("pidfd_open", errno.ESRCH, ProcessLookupError),
                                     ("pidfd_send_signal", errno.EPERM, PermissionError)):
            def failure(*unused):
                launch.ctypes.set_errno(code)
                return -1
            library = SimpleNamespace(**{name: Mock(side_effect=failure)})
            with self.subTest(name=name), \
                    patch.object(launch.os, "pidfd_open", None, create=True), \
                    patch.object(launch.signal, "pidfd_send_signal", None, create=True), \
                    patch.object(launch.ctypes, "CDLL", return_value=library), \
                    patch.object(launch.os, "kill") as unbound:
                with self.assertRaises(expected) as error:
                    if name == "pidfd_open":
                        launch.pidfd_open(123)
                    else:
                        launch.pidfd_send_signal(99, signal.SIGTERM)
                self.assertEqual(error.exception.errno, code)
                self.assertEqual(error.exception.filename, name)
                unbound.assert_not_called()

    def test_unavailable_libc_symbol_fails_closed(self):
        with patch.object(launch.os, "pidfd_open", None, create=True), \
                patch.object(launch.ctypes, "CDLL", return_value=SimpleNamespace()), \
                patch.object(launch.os, "kill") as unbound:
            with self.assertRaisesRegex(OSError, "unavailable in libc") as error:
                launch.pidfd_open(123)
            self.assertEqual(error.exception.errno, errno.ENOSYS)
            unbound.assert_not_called()

    def test_real_libc_fallback_binds_only_this_cpu_test_process(self):
        with patch.object(launch.os, "pidfd_open", None, create=True), \
                patch.object(launch.signal, "pidfd_send_signal", None, create=True):
            launch.require_pidfd_support()

    def test_preflight_failure_precedes_power_changes_and_child_launch(self):
        with patch.object(launch, "require_pidfd_support", side_effect=OSError(errno.ENOSYS, "pidfd")), \
                patch.object(launch.subprocess, "Popen") as start, \
                patch.object(launch.subprocess, "run") as change:
            with self.assertRaises(OSError):
                launch.main([])
            start.assert_not_called()
            change.assert_not_called()


class GuardedWeb(unittest.TestCase):
    def test_default_command_and_environment_are_explicit(self):
        args = launch.arguments(["--repo", "/tmp/deepmoe-test"])
        command, env = launch.launch_configuration(args, {
            "CACHEDMOE_SPEC_DIAGNOSTICS": "stale.jsonl", "CACHEDMOE_MASK_WAIT_TAU": ".20",
            "CACHEDMOE_MASK_WAIT_BUDGET": "8,20", "CACHEDMOE_DSPARK_MEGA_DIAG": "1",
            "CACHEDMOE_MODEL_MIRRORS": "/tmp/old-mirror", "KEEP_ME": "yes"})
        for flag, expected in (("--resident-only", "mask"), ("--mask-cache", "dynamic"),
                               ("--cache-slots", "5500"), ("--max-context", "1048576"),
                               ("--gpu-prefill-min", "16"), ("--spec-k", "2"),
                               ("--spec-top-k", "4"), ("--kv-max-gb", "4")):
            self.assertEqual(command[command.index(flag) + 1], expected)
        self.assertIn("--dspark", command)
        self.assertEqual(command[command.index("--exe") + 1], "/tmp/deepmoe-test/build/cachedmoe")
        self.assertEqual(command[command.index("--kv-dir") + 1], "/tmp/deepmoe-test/build/web_mask/kv")
        self.assertEqual(env["CACHEDMOE_DSPARK_ONECB"], "1")
        self.assertEqual(env["CACHEDMOE_BATCH_GPU_ROUTE"], "0")
        self.assertEqual(env["CACHEDMOE_DSPARK_PROFILE"], "0")
        self.assertEqual(env["KEEP_ME"], "yes")
        for key in ("CACHEDMOE_SPEC_DIAGNOSTICS", "CACHEDMOE_MASK_WAIT_TAU",
                    "CACHEDMOE_MASK_WAIT_BUDGET", "CACHEDMOE_DSPARK_MEGA_DIAG", "CACHEDMOE_MODEL_MIRRORS"):
            self.assertNotIn(key, env)

    def test_selected_d_configuration_and_existing_state_dir(self):
        args = launch.arguments(["--spec-k", "5", "--gpu-route", "1", "--state-dir", "/tmp/existing"])
        command, env = launch.launch_configuration(args, {})
        self.assertEqual(command[command.index("--spec-k") + 1], "5")
        self.assertEqual(command[command.index("--kv-dir") + 1], "/tmp/existing/kv")
        self.assertEqual(env["CACHEDMOE_BATCH_GPU_ROUTE"], "1")

    def test_unsupported_spec_size_is_rejected(self):
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            launch.arguments(["--spec-k", "16"])

    def test_dry_run_never_changes_power_or_launches_a_process(self):
        with patch.object(launch.subprocess, "Popen") as start, \
                patch.object(launch.subprocess, "run") as setting, \
                patch("sys.stdout", io.StringIO()) as output:
            self.assertEqual(launch.main(["--dry-run"]), 0)
            start.assert_not_called()
            setting.assert_not_called()
            self.assertEqual(json.loads(output.getvalue())["power_profile"], "performance")

    def test_temperature_signals_only_bound_engine(self):
        signals = []

        class Child:
            def send(self, signum):
                signals.append(signum)
                return True

        latch = launch.ThermalLatch(80, 72)
        values = {"amdgpu:fake": 80, "nvme:fake": 74.85, "k10temp:fake": 90}
        paused = launch.thermal_transition(Child(), values, latch, False)
        self.assertTrue(paused)
        values["amdgpu:fake"] = 72
        paused = launch.thermal_transition(Child(), values, latch, paused)
        self.assertFalse(paused)
        self.assertEqual(signals, [signal.SIGSTOP, signal.SIGCONT])

    def test_ac_or_profile_change_holds_a_cool_engine(self):
        with patch.object(launch.EngineChild, "send", return_value=True) as signal_engine:
            child = launch.EngineChild(123, 100, 456, "/fake/deepmoe", 999)
            latch = launch.ThermalLatch(80, 72)
            for invalid in ({"ac": 0, "power_profile": "performance"},
                            {"ac": 1, "power_profile": "power-saver"},
                            {"ac": 1, "power_profile": "performance", "platform_profile": "balanced"}):
                values = {"amdgpu:fake": 42, "nvme:fake": 50} | invalid
                self.assertFalse(launch.power_valid(values))
                self.assertTrue(launch.thermal_transition(child, values, latch, False,
                                                         not launch.power_valid(values)))
            self.assertEqual(signal_engine.call_count, 3)

    def test_reused_pid_or_wrong_comm_is_never_signalled(self):
        child = launch.EngineChild(123, 100, 456, "/fake/deepmoe", 999)
        details = dict(pid=123, ppid=100, state="S", start_ticks=456,
                       comm="deepmoe", exe="/fake/deepmoe")
        for changed in (dict(start_ticks=457), dict(comm="python"), dict(ppid=1),
                        dict(exe="/another/deepmoe")):
            with patch.object(child, "live", return_value=True), \
                    patch.object(launch, "process_details", return_value=details | changed), \
                    patch.object(launch, "pidfd_send_signal") as send:
                with self.assertRaisesRegex(RuntimeError, "identity changed"):
                    child.send(signal.SIGSTOP)
                send.assert_not_called()

    def test_original_engine_uses_pidfd(self):
        child = launch.EngineChild(123, 100, 456, "/fake/deepmoe", 999)
        with patch.object(child, "live", return_value=True), \
                patch.object(child, "matches", return_value=True), \
                patch.object(launch, "pidfd_send_signal") as send:
            self.assertTrue(child.send(signal.SIGSTOP))
            send.assert_called_once_with(999, signal.SIGSTOP)

    def test_atomic_state_updates_preserve_existing_kv_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "kv").mkdir()
            snapshot = root / "kv/session.pkv"
            snapshot.write_bytes(b"keep this")
            launch.write_state(root / "state.json", dict(paused=True))
            self.assertTrue(json.loads((root / "state.json").read_text())["paused"])
            self.assertEqual(snapshot.read_bytes(), b"keep this")
            self.assertFalse((root / "state.json.tmp").exists())

    def test_shutdown_requests_sigint_and_keeps_thermal_control(self):
        server_signals, engine_signals = [], []

        class Server:
            returncode = None
            calls = 0
            def poll(self):
                self.calls += 1
                if self.calls >= 4:
                    self.returncode = 0
                return self.returncode
            def send_signal(self, signum):
                server_signals.append(signum)

        server = Server()

        class Child:
            def send(self, signum):
                engine_signals.append(signum)
                return True
            def live(self):
                return server.returncode is None

        values = [{"amdgpu:fake": temp, "nvme:fake": 74.85,
                   "ac": 1, "power_profile": "performance"} for temp in (80, 72)]
        with patch.object(launch, "sample", side_effect=values), \
                patch.object(launch.time, "sleep"):
            clean = launch.shutdown(server, Child(), False, launch.ThermalLatch(80, 72), {}, None,
                                    io.StringIO(), io.StringIO())
        self.assertTrue(clean)
        self.assertEqual(server_signals, [signal.SIGINT])
        self.assertEqual(engine_signals, [signal.SIGSTOP, signal.SIGCONT])

    def test_failed_http_shutdown_is_not_claimed_clean(self):
        class Server:
            returncode = 1
            def poll(self):
                return 1
        log = io.StringIO()
        clean = launch.shutdown(Server(), None, False, launch.ThermalLatch(80, 72), {}, None,
                                log, io.StringIO())
        self.assertFalse(clean)
        self.assertIn("KV drain is not confirmed", log.getvalue())


class StartupShutdownOwnership(unittest.TestCase):
    @staticmethod
    def details(pid, *, comm="deepmoe", exe="/fake/deepmoe", parent=100, birth=401,
                session=100, group=100):
        return dict(pid=pid, state="S", ppid=parent, start_ticks=birth, comm=comm, exe=exe,
                    session=session, process_group=group)

    def shutdown_fixture(self, preexec=False, fork_during_cleanup=False, ignore_term=False):
        # Model the fork/exec race entirely in Python objects. All proc reads,
        # pidfds, clocks and signals are mocked; no child process is started.
        children, signals, opens = {}, [], []
        leader = self.details(100, comm="python", exe="/fake/python",
                              parent=os.getpid(), birth=400)
        if preexec:
            children[101] = self.details(101, comm="python", exe="/fake/python")

        class Server:
            pid, returncode = 100, None
            def poll(self):
                return self.returncode
            def send_signal(self, signum):
                signals.append(("http", signum))
                children[101] = StartupShutdownOwnership.details(101, parent=1)
                self.returncode = 130

        server = Server()

        def details(pid):
            if pid == 100:
                return leader if server.returncode is None else None
            return children.get(pid)

        def bind(pid):
            opens.append(pid)
            return pid + 1000

        def send(fd, signum):
            pid = fd - 1000
            signals.append((pid, signum))
            if signum in (signal.SIGTERM, signal.SIGKILL):
                if ignore_term and signum == signal.SIGTERM:
                    return
                children.pop(pid, None)
                if fork_during_cleanup and pid == 101:
                    children[102] = self.details(102, parent=1, birth=402)

        tick = 0
        def monotonic():
            nonlocal tick
            tick += .5
            return tick

        with patch.object(launch, "process_details", side_effect=details), \
                patch.object(launch, "session_members", side_effect=lambda _: list(children.values())), \
                patch.object(launch, "pidfd_open", side_effect=bind), \
                patch.object(launch.os, "close"), \
                patch.object(launch.OwnedProcess, "live", lambda process: process.pid in children), \
                patch.object(launch, "pidfd_send_signal", side_effect=send), \
                patch.object(launch.time, "monotonic", side_effect=monotonic), \
                patch.object(launch.time, "sleep"):
            owned = launch.OwnedSession(server, "/fake/deepmoe")
            log = io.StringIO()
            clean = launch.shutdown(server, None, False, launch.ThermalLatch(80, 72), {}, None,
                                    log, io.StringIO(), owned)
            receipt = owned.receipt()
            owned.close()
        return clean, signals, opens, receipt, children, log.getvalue()

    def test_engine_appearing_after_final_discovery_is_bound_and_cleaned(self):
        clean, signals, opens, receipt, children, log = self.shutdown_fixture()
        self.assertFalse(clean)
        self.assertEqual(opens, [101])
        self.assertEqual(signals[0], ("http", signal.SIGINT))
        self.assertIn((101, signal.SIGTERM), signals)
        self.assertNotIn(("http", signal.SIGSTOP), signals)
        self.assertFalse(children)
        self.assertFalse(receipt["children"][0]["live"])
        self.assertEqual(receipt["children"][0]["start_ticks"], 401)
        self.assertIn("KV drain not confirmed", log)

    def test_preexec_child_stays_bound_across_exec_and_parent_exit(self):
        clean, signals, opens, receipt, children, _ = self.shutdown_fixture(preexec=True)
        self.assertFalse(clean)
        self.assertEqual(opens, [101])
        self.assertFalse(children)
        observations = receipt["children"][0]["observations"]
        self.assertEqual([row["comm"] for row in observations], ["python", "deepmoe"])
        self.assertEqual(observations[-1]["ppid"], 1)
        self.assertIn((101, signal.SIGTERM), signals)

    def test_new_owned_child_during_cleanup_is_also_bound_and_reaped(self):
        clean, signals, opens, receipt, children, _ = self.shutdown_fixture(fork_during_cleanup=True)
        self.assertFalse(clean)
        self.assertEqual(opens, [101, 102])
        self.assertFalse(children)
        self.assertIn((102, signal.SIGTERM), signals)
        self.assertEqual(len(receipt["children"]), 2)
        self.assertTrue(all(not row["live"] for row in receipt["children"]))

    def test_owned_child_ignoring_term_is_killed_without_claiming_kv_drain(self):
        clean, signals, _, receipt, children, log = self.shutdown_fixture(ignore_term=True)
        self.assertFalse(clean)
        self.assertFalse(children)
        self.assertIn((101, signal.SIGKILL), signals)
        self.assertFalse(receipt["children"][0]["live"])
        self.assertIn("KV drain not confirmed", log)

    def test_other_session_group_or_older_birth_is_never_bound(self):
        leader = self.details(100, parent=os.getpid(), birth=400)
        server = type("Server", (), dict(pid=100, poll=lambda _: None))()
        wrong = [self.details(101, session=200), self.details(102, group=200),
                 self.details(103, birth=399)]
        with patch.object(launch, "process_details", return_value=leader), \
                patch.object(launch, "session_members", return_value=wrong), \
                patch.object(launch, "pidfd_open") as bind:
            owned = launch.OwnedSession(server, "/fake/deepmoe")
            owned.discover()
            self.assertFalse(owned.children)
            bind.assert_not_called()

    def test_birth_change_between_scan_and_pidfd_bind_is_rejected(self):
        leader = self.details(100, parent=os.getpid(), birth=400)
        candidate = self.details(101)
        reused = candidate | dict(start_ticks=402)
        server = type("Server", (), dict(pid=100, poll=lambda _: None))()
        with patch.object(launch, "process_details", side_effect=lambda pid: leader if pid == 100 else reused), \
                patch.object(launch, "session_members", return_value=[candidate]), \
                patch.object(launch, "pidfd_open", return_value=999), \
                patch.object(launch.os, "close") as close, \
                patch.object(launch, "pidfd_send_signal") as send:
            owned = launch.OwnedSession(server, "/fake/deepmoe")
            owned.discover()
            self.assertFalse(owned.children)
            close.assert_called_once_with(999)
            send.assert_not_called()

    def test_reused_session_leader_cannot_adopt_new_children(self):
        leader = self.details(100, parent=os.getpid(), birth=400)
        server = type("Server", (), dict(pid=100, poll=lambda _: 130))()
        with patch.object(launch, "process_details", return_value=leader) as read, \
                patch.object(launch, "session_members", return_value=[self.details(101)]), \
                patch.object(launch, "pidfd_open") as bind:
            owned = launch.OwnedSession(server, "/fake/deepmoe")
            read.return_value = leader | dict(start_ticks=500)
            with self.assertRaisesRegex(RuntimeError, "leader PID was reused"):
                owned.discover()
            bind.assert_not_called()

    def test_cleanup_exception_still_writes_failed_state_receipt(self):
        class Monitor:
            def __enter__(self):
                return self
            def __exit__(self, *unused):
                pass

        class Owned:
            def receipt(self):
                return dict(children=[dict(pid=101, live=True)])
            def close(self):
                pass

        server = type("Server", (), dict(pid=100, returncode=1, poll=lambda _: None))()
        values = dict(ac=1, power_profile="performance", wall_time_s=1, **{"amdgpu:fake": 42})
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(launch, "require_pidfd_support"), \
                patch.object(launch.Path, "is_file", return_value=True), \
                patch.object(launch, "assert_idle"), \
                patch.object(launch, "profile", return_value="performance"), \
                patch.object(launch.subprocess, "run"), \
                patch.object(launch.subprocess, "Popen", return_value=server), \
                patch.object(launch.signal, "signal"), \
                patch.object(launch, "discover_sensors", return_value={}), \
                patch.object(launch, "ProfileMonitor", Monitor), \
                patch.object(launch, "OwnedSession", return_value=Owned()), \
                patch.object(launch, "find_engine", return_value=None), \
                patch.object(launch, "sample", side_effect=[values, RuntimeError("sensor failed")]), \
                patch.object(launch, "shutdown", side_effect=RuntimeError("cleanup did not finish")):
            self.assertEqual(launch.main(["--state-dir", tmp]), 1)
            receipt = json.loads((Path(tmp) / "state.json").read_text())
            self.assertEqual(receipt["failure"], "sensor failed")
            self.assertEqual(receipt["shutdown_error"], "cleanup did not finish")
            self.assertFalse(receipt["shutdown_graceful"])
            self.assertEqual(receipt["kv_drain_status"], "unconfirmed")
            self.assertTrue(receipt["shutdown_owned_processes"]["children"][0]["live"])


@unittest.skipUnless(sys.platform == "linux", "Linux comm and pidfd fixture")
class ExecutableAliases(unittest.TestCase):
    def test_real_cpu_executable_symlink_comm_and_pidfd(self):
        # The temporary engine-named image is a copy of sleep, never the GPU
        # executable. The legacy symlink changes comm but not /proc/exe.
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "cachedmoe"
            shutil.copy2(shutil.which("sleep"), exe)
            alias = Path(tmp) / "deepmoe"
            alias.symlink_to(exe.name)
            for selected in (exe, alias):
                with self.subTest(selected=selected.name):
                    process = subprocess.Popen([str(selected), "10"])
                    child = None
                    try:
                        deadline = time.monotonic() + 1
                        while child is None and time.monotonic() < deadline:
                            child = launch.find_engine(os.getpid(), selected)
                            time.sleep(.01)
                        self.assertIsNotNone(child)
                        details = launch.process_details(process.pid)
                        self.assertEqual(details["comm"], selected.name)
                        self.assertEqual(details["exe"], str(exe.resolve()))
                        self.assertEqual(child.exe, str(exe.resolve()))
                        self.assertTrue(child.send(signal.SIGSTOP))
                        deadline = time.monotonic() + 1
                        while time.monotonic() < deadline:
                            if launch.process_details(process.pid)["state"] == "T":
                                break
                            time.sleep(.01)
                        self.assertEqual(launch.process_details(process.pid)["state"], "T")
                        self.assertTrue(child.send(signal.SIGCONT))
                    finally:
                        process.kill()
                        process.wait(timeout=2)
                        if child:
                            child.close()

    def test_owned_session_normalizes_alias_without_relaxing_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "cachedmoe"
            exe.touch()
            alias = Path(tmp) / "deepmoe"
            alias.symlink_to(exe.name)
            leader = StartupShutdownOwnership.details(100, parent=os.getpid(), birth=400)
            server = SimpleNamespace(pid=100)
            with patch.object(launch, "process_details", return_value=leader):
                owned = launch.OwnedSession(server, alias)
            self.assertEqual(owned.expected_engine, str(exe.resolve()))
            for comm in ("cachedmoe", "deepmoe"):
                details = StartupShutdownOwnership.details(101, comm=comm, exe=str(exe))
                process = launch.OwnedProcess(101, 401, 100, 999, owned.expected_engine, [])
                self.assertTrue(process.observe(details))
                self.assertTrue(process.is_engine())
                process.observations[-1]["exe"] = "/wrong/cachedmoe"
                self.assertFalse(process.is_engine())
                self.assertFalse(process.observe(details | dict(session=101)))
                self.assertFalse(process.observe(details | dict(start_ticks=402)))


if __name__ == "__main__":
    unittest.main()
