"""Launcher defaults and engine-only safety checks; never launch a GPU engine."""
import io
import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/web"))
import launch_guarded as launch


class GuardedWeb(unittest.TestCase):
    def test_default_command_and_environment_are_explicit(self):
        args = launch.arguments(["--repo", "/tmp/deepmoe-test"])
        command, env = launch.launch_configuration(args, {
            "DEEPMOE_SPEC_DIAGNOSTICS": "stale.jsonl", "DEEPMOE_MASK_WAIT_TAU": ".20",
            "DEEPMOE_MASK_WAIT_BUDGET": "8,20", "DEEPMOE_DSPARK_MEGA_DIAG": "1",
            "DEEPMOE_MODEL_MIRRORS": "/tmp/old-mirror", "KEEP_ME": "yes"})
        for flag, expected in (("--resident-only", "mask"), ("--mask-cache", "dynamic"),
                               ("--cache-slots", "5500"), ("--max-context", "1048576"),
                               ("--gpu-prefill-min", "16"), ("--spec-k", "2"),
                               ("--spec-top-k", "4"), ("--kv-max-gb", "4")):
            self.assertEqual(command[command.index(flag) + 1], expected)
        self.assertIn("--dspark", command)
        self.assertEqual(command[command.index("--kv-dir") + 1], "/tmp/deepmoe-test/build/web_mask/kv")
        self.assertEqual(env["DEEPMOE_DSPARK_ONECB"], "1")
        self.assertEqual(env["DEEPMOE_BATCH_GPU_ROUTE"], "0")
        self.assertEqual(env["DEEPMOE_DSPARK_PROFILE"], "0")
        self.assertEqual(env["KEEP_ME"], "yes")
        for key in ("DEEPMOE_SPEC_DIAGNOSTICS", "DEEPMOE_MASK_WAIT_TAU",
                    "DEEPMOE_MASK_WAIT_BUDGET", "DEEPMOE_DSPARK_MEGA_DIAG", "DEEPMOE_MODEL_MIRRORS"):
            self.assertNotIn(key, env)

    def test_selected_d_configuration_and_existing_state_dir(self):
        args = launch.arguments(["--spec-k", "5", "--gpu-route", "1", "--state-dir", "/tmp/existing"])
        command, env = launch.launch_configuration(args, {})
        self.assertEqual(command[command.index("--spec-k") + 1], "5")
        self.assertEqual(command[command.index("--kv-dir") + 1], "/tmp/existing/kv")
        self.assertEqual(env["DEEPMOE_BATCH_GPU_ROUTE"], "1")

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

        latch = launch.ThermalLatch()
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
            latch = launch.ThermalLatch()
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
                    patch.object(launch.signal, "pidfd_send_signal") as send:
                with self.assertRaisesRegex(RuntimeError, "identity changed"):
                    child.send(signal.SIGSTOP)
                send.assert_not_called()

    def test_original_engine_uses_pidfd(self):
        child = launch.EngineChild(123, 100, 456, "/fake/deepmoe", 999)
        with patch.object(child, "live", return_value=True), \
                patch.object(child, "matches", return_value=True), \
                patch.object(launch.signal, "pidfd_send_signal") as send:
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
            clean = launch.shutdown(server, Child(), False, launch.ThermalLatch(), {}, None,
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
        clean = launch.shutdown(Server(), None, False, launch.ThermalLatch(), {}, None,
                                log, io.StringIO())
        self.assertFalse(clean)
        self.assertIn("KV drain is not confirmed", log.getvalue())


if __name__ == "__main__":
    unittest.main()
