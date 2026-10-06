"""Controller/cleanup error gates. Every power/process action is mocked; no GPU."""
import json
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import power_profile_compare as compare


class FailureReceipts(unittest.TestCase):
    def test_controller_failure_remains_cause_after_http_eof(self):
        controller = compare.ThermalController.__new__(compare.ThermalController)
        controller.error = FileNotFoundError("nvme composite sensor removed")
        with self.assertRaises(RuntimeError) as result:
            controller.check()
        self.assertIs(result.exception.__cause__, controller.error)

    def cleanup_case(self, *, primary, shutdown_error=None, clean=True):
        server = SimpleNamespace(returncode=0)
        child = SimpleNamespace(close=Mock())
        controller = SimpleNamespace(child=child, paused=False, error=primary,
                                     phase="generate", stop=Mock(), thread=Mock(), latch=Mock())
        owned = SimpleNamespace(close=Mock())
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(compare.subprocess, "run") as power, \
                patch.object(compare.time, "sleep"), \
                patch.object(compare, "ProfileMonitor"), \
                patch.object(compare, "discover_sensors", return_value={}), \
                patch.object(compare.web_guard, "shutdown", side_effect=shutdown_error, return_value=clean), \
                patch.object(compare, "abort_owned") as abort:
            out = Path(tmp)
            if primary is None and (not clean or shutdown_error):
                with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                    compare.finish(server, controller, owned, out, "balanced", False, primary)
            else:
                compare.finish(server, controller, owned, out, "balanced", False, primary)
            receipt = json.loads((out / "completion.json").read_text())
            self.assertEqual(power.call_args.args[0], ["powerprofilesctl", "set", "balanced"])
            child.close.assert_called_once()
            owned.close.assert_called_once()
            self.assertEqual(abort.call_count, int(shutdown_error is not None))
            return receipt

    def test_primary_failure_not_replaced_by_failed_drain(self):
        receipt = self.cleanup_case(primary=FileNotFoundError("NVMe disappeared"), clean=False)
        self.assertEqual(receipt["experiment_error"]["type"], "FileNotFoundError")
        self.assertFalse(receipt["graceful_shutdown"])

    def test_cleanup_exception_is_separate_and_power_is_restored(self):
        receipt = self.cleanup_case(primary=RuntimeError("original experiment failed"),
                                    shutdown_error=OSError("shutdown sensor disappeared"))
        self.assertEqual(receipt["experiment_error"]["message"], "original experiment failed")
        self.assertEqual(receipt["cleanup_error"]["type"], "OSError")
        self.assertFalse(receipt["graceful_shutdown"])

    def test_emergency_cleanup_signals_only_owned_cpu_session(self):
        # Stand-in parent/child both run Python sleep; neither can load Vulkan.
        command = [sys.executable, "-u", "-c",
                   "import subprocess,sys,time; "
                   "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']); "
                   "print(p.pid,flush=True);time.sleep(30)"]
        parent = subprocess.Popen(command, stdout=subprocess.PIPE, text=True, start_new_session=True)
        owned = compare.web_guard.OwnedSession(parent, sys.executable)
        try:
            child_pid = int(parent.stdout.readline())
            owned.discover()
            self.assertIn(child_pid, [p.pid for p in owned.live_children()])
            compare.abort_owned(parent, None, owned)
            self.assertIsNotNone(parent.poll())
            self.assertEqual(owned.live_children(), [])
        finally:
            compare.abort_owned(parent, None, owned)
            owned.close()
            parent.stdout.close()

    def test_source_drop_invalidates_arm_even_if_temperature_is_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "engine.log"
            log.write_text("[INF] engine: mirror holds 48 of 48 shards\n")
            compare.validate_storage_health(log)
            log.write_text("[WRN] IoEngine: source 1 '/mnt/mirror' dropped after 3 consecutive I/O errors\n")
            with self.assertRaisesRegex(RuntimeError, "dual-drive arm is invalid"):
                compare.validate_storage_health(log)

    def test_cleanup_only_failure_cannot_return_success(self):
        self.cleanup_case(primary=None, clean=False)


if __name__ == "__main__":
    unittest.main()
