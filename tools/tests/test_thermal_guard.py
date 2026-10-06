"""Temperature transitions, cooling budgets, and decode accounting; no GPU."""
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "tools")]
import thermal_guard as guard
from thermal_metrics import decode_timing, interval_overlap, pause_intervals
import provenance


class ThermalPolicy(unittest.TestCase):
    def test_thresholds_and_hysteresis(self):
        latch = guard.ThermalLatch()
        self.assertFalse(latch.update({"gpu": 79.99}))
        self.assertTrue(latch.update({"gpu": 80}))
        self.assertTrue(latch.update({"gpu": 72.01}))
        self.assertFalse(latch.update({"gpu": 72}))

    def test_untriggered_disk_does_not_extend_gpu_pause(self):
        latch = guard.ThermalLatch()
        self.assertTrue(latch.update({"gpu": 81, "nvme": 74.85}))
        self.assertFalse(latch.update({"gpu": 72, "nvme": 74.85}))

    def test_second_hot_sensor_must_also_cool(self):
        latch = guard.ThermalLatch()
        latch.update({"gpu": 81, "nvme": 78})
        self.assertTrue(latch.update({"gpu": 72, "nvme": 80}))
        self.assertTrue(latch.update({"gpu": 71, "nvme": 73}))
        self.assertFalse(latch.update({"gpu": 71, "nvme": 72}))

    def test_missing_hot_or_invalid_sensor_fails_closed(self):
        latch = guard.ThermalLatch()
        latch.update({"gpu": 81})
        for sensors in ({"nvme": 50}, {"gpu": float("nan")}, {}):
            with self.assertRaises(RuntimeError):
                latch.update(sensors)

    def test_budget_excludes_open_and_closed_pauses(self):
        budget = guard.RunBudget(100, 10, 60)
        budget.set_paused(True, 105)
        self.assertEqual(budget.elapsed(140), (40, 35, 5))
        budget.check(140)
        budget.set_paused(False, 145)
        self.assertEqual(budget.elapsed(149), (49, 40, 9))
        budget.check(149)
        with self.assertRaisesRegex(RuntimeError, "active-time"):
            budget.check(151)

    def test_safety_wall_budget_still_bounds_cooling(self):
        budget = guard.RunBudget(0, 10, 30)
        budget.set_paused(True, 1)
        with self.assertRaisesRegex(RuntimeError, "wall-time"):
            budget.check(31)

    def test_owned_cpu_process_is_paused_resumed_and_accounted(self):
        # A normal Python child exercises real process-group signals. No
        # engine, GPU command, or system power setting is touched.
        count = 0

        def sample(sensors, monitor):
            nonlocal count
            count += 1
            temperature = 81 if 4 <= count <= 6 else 42
            return {"amdgpu:fake": temperature, "nvme:fake": 74.85,
                    "ac": 1, "power_profile": "performance", "wall_time_s": time.time()}

        class Monitor:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(guard, "sample", sample), \
                patch.object(guard, "ProfileMonitor", Monitor), \
                patch.object(guard, "assert_idle"), \
                patch.object(guard.subprocess, "run"):
            job = dict(name="cpu_signal_check", profile="performance", active_timeout_s=1,
                       wall_timeout_s=2, command=[sys.executable, "-c",
                           "import time; time.sleep(.4); print('10 case(s) run, 0 failed')"])
            result = guard.run_job(job, Path(tmp), {}, {"amdgpu:fake": None, "nvme:fake": None})
            self.assertEqual(result["rc"], 0)
            self.assertEqual(result["thermal_pauses"], 1)
            self.assertGreater(result["thermal_paused_s"], .09)
            self.assertLess(result["active_elapsed_s"], result["elapsed_s"])
            self.assertEqual(len(result["thermal_pause_intervals"]), 1)
            self.assertEqual(len(result["thermal_transitions"]), 2)
            self.assertFalse(guard.group_exists(result["process_group"]))


class ThermalAccounting(unittest.TestCase):
    def test_overlap_unions_pauses_and_clips_boundaries(self):
        self.assertEqual(interval_overlap(10, 20, [(5, 11), (12, 14), (13, 16), (19, 30)]), 6)
        self.assertEqual(interval_overlap(10, 20, [(1, 2), (25, 30)]), 0)
        with self.assertRaises(ValueError):
            interval_overlap(20, 10, [])

    def test_done_uses_shared_clock_without_subtracting_prefill(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events, clock, thermal = root / "events.jsonl", root / "clock.json", root / "thermal.jsonl"
            # Decode interval [110,120], next [125,130]; prefill [100,110]
            # also paused, but it must not be subtracted from decode.
            events.write_text(json.dumps(dict(event="done", host_s=20, decode_ms=10000,
                                              decode_steps=10)) + "\n" +
                              json.dumps(dict(event="done", host_unix=130, decode_ms=5000,
                                              decode_steps=5)) + "\n")
            clock.write_text(json.dumps(dict(host_start_unix=100)))
            samples = [(101, True), (105, False), (113, True), (117, False),
                       (127, True), (129, False)]
            thermal.write_text("".join(json.dumps(dict(wall_time_s=t, paused=p)) + "\n"
                                       for t, p in samples))
            result = decode_timing(events, clock, thermal)
            self.assertEqual(result["raw_decode_ms"], 15000)
            self.assertEqual(result["thermal_pause_ms"], 6000)
            self.assertEqual(result["active_ms_per_token"], 600)
            self.assertEqual(result["raw_ms_per_token"], 1000)

    def test_missing_clock_cannot_claim_adjusted_speed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = root / "events.jsonl"
            events.write_text(json.dumps(dict(event="done", host_s=20, decode_ms=1000,
                                              decode_steps=10)) + "\n")
            thermal = root / "thermal.jsonl"
            thermal.write_text(json.dumps(dict(wall_time_s=10, paused=False)))
            self.assertIsNone(decode_timing(events, thermal_log=thermal)["active_ms_per_token"])

    def test_open_pause_closes_at_analysis_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "thermal.jsonl"
            path.write_text('{"wall_time_s":10,"paused":true}\n{"wall_time_s":')
            self.assertEqual(pause_intervals(path, end_unix=15), [(10, 15)])

    def test_old_log_is_rejected_instead_of_fabricating_a_clock(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "thermal.jsonl"
            path.write_text('{"elapsed_s":10,"amdgpu":80}\n')
            with self.assertRaises(ValueError):
                pause_intervals(path)

    def test_power_state_keeps_actual_mode_and_ac(self):
        with patch.object(provenance.subprocess, "check_output", return_value="performance\n"), \
                patch.object(provenance, "_read", side_effect=lambda p: {
                    "/sys/firmware/acpi/platform_profile": "performance",
                    "/sys/class/power_supply/AC0/online": "1"}.get(p)), \
                patch.object(provenance.Path, "glob", return_value=[]):
            state = provenance.power_state()
            self.assertEqual(state["power_profile"], "performance")
            self.assertEqual(state["platform_profile"], "performance")
            self.assertEqual(state["ac_online"], 1)


if __name__ == "__main__":
    unittest.main()
