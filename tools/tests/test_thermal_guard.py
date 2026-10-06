"""Temperature transitions, cooling budgets, and decode accounting; no GPU."""
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "tools")]
import thermal_guard as guard
from thermal_metrics import decode_timing, interval_overlap, pause_intervals
import provenance
from hitrate_bench import BenchServer


class ThermalPolicy(unittest.TestCase):
    def test_thresholds_and_hysteresis(self):
        latch = guard.ThermalLatch(80, 72)
        self.assertFalse(latch.update({"gpu": 79.99}))
        self.assertTrue(latch.update({"gpu": 80}))
        self.assertTrue(latch.update({"gpu": 72.01}))
        self.assertFalse(latch.update({"gpu": 72}))

    def test_untriggered_disk_does_not_extend_gpu_pause(self):
        latch = guard.ThermalLatch(80, 72)
        self.assertTrue(latch.update({"gpu": 81, "nvme": 74.85}))
        self.assertFalse(latch.update({"gpu": 72, "nvme": 74.85}))

    def test_second_hot_sensor_must_also_cool(self):
        latch = guard.ThermalLatch(80, 72)
        latch.update({"gpu": 81, "nvme": 78})
        self.assertTrue(latch.update({"gpu": 72, "nvme": 80}))
        self.assertTrue(latch.update({"gpu": 71, "nvme": 73}))
        self.assertFalse(latch.update({"gpu": 71, "nvme": 72}))

    def test_missing_hot_or_invalid_sensor_fails_closed(self):
        latch = guard.ThermalLatch(80, 72)
        latch.update({"gpu": 81})
        for sensors in ({"nvme": 50}, {"gpu": float("nan")}, {}):
            with self.assertRaises(RuntimeError):
                latch.update(sensors)

    def test_current_per_device_policy_and_cold_start(self):
        policy = guard.runtime_defaults.ThermalPolicy()
        latch = guard.ThermalLatch(policy=policy)
        self.assertFalse(latch.update({"amdgpu:fake": 84, "nvme:fake": 79}))
        self.assertTrue(latch.update({"amdgpu:fake": 85, "nvme:fake": 79}))
        self.assertTrue(latch.update({"amdgpu:fake": 77, "nvme:fake": 80}))
        self.assertTrue(latch.update({"amdgpu:fake": 76, "nvme:fake": 72.01}))
        self.assertFalse(latch.update({"amdgpu:fake": 77, "nvme:fake": 72}))
        self.assertTrue(policy.cold({"amdgpu:fake": 60, "nvme:fake": 65}))
        self.assertFalse(policy.cold({"amdgpu:fake": 60.01, "nvme:fake": 65}))
        self.assertFalse(policy.cold({"amdgpu:fake": 60, "nvme:fake": 65.01}))
        with self.assertRaises(RuntimeError):
            policy.cold({"nvme:fake": 40})

    def test_custom_policy_and_invalid_thresholds(self):
        parser = guard.argparse.ArgumentParser()
        guard.add_thermal_arguments(parser)
        policy = guard.thermal_policy(parser.parse_args(["--gpu-pause-c", "82", "--gpu-resume-c", "74"]))
        latch = guard.ThermalLatch(policy=policy)
        self.assertTrue(latch.update({"amdgpu:x": 82, "nvme:x": 79}))
        self.assertFalse(latch.update({"amdgpu:x": 74, "nvme:x": 79}))
        self.assertEqual(policy.record()["gpu_pause_c"], 82)
        for values in ({"gpu_pause_c": float("nan")}, {"nvme_pause_c": 72},
                       {"nvme_start_c": 73}, {"gpu_resume_c": True}):
            with self.assertRaises(ValueError):
                guard.runtime_defaults.ThermalPolicy(**values)

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
            temperature = 86 if 4 <= count <= 6 else 42
            return {"amdgpu:fake": temperature, "nvme:fake": 50 if count <= 2 else 74.85,
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
    def test_engine_end_excludes_pause_after_decode_and_delayed_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events, thermal = root / "events.jsonl", root / "thermal.jsonl"
            turn = dict(event="done", host_unix=120, decode_finished_unix=110,
                        decode_ms=10000, decode_steps=10)
            events.write_text(json.dumps(turn) + "\n")
            thermal.write_text("".join(
                json.dumps(dict(wall_time_s=100 + n / 20,
                                paused=110 <= 100 + n / 20 < 119)) + "\n"
                for n in range(401)))
            result = decode_timing(events, thermal_log=thermal)
            self.assertEqual(result["active_decode_ms"], 10000)
            self.assertEqual(result["thermal_pause_ms"], 0)
            self.assertTrue(result["engine_decode_boundaries"])
            self.assertEqual(result["host_receipt_delay_ms"], [10000])
            self.assertIn("engine decode end", result["timing_alignment"])
            # Old logs remain explicitly estimated; they cannot retrospectively
            # recover the engine boundary from the delayed receipt alone.
            turn.pop("decode_finished_unix")
            events.write_text(json.dumps(turn) + "\n")
            legacy = decode_timing(events, thermal_log=thermal)
            self.assertEqual(legacy["active_decode_ms"], 1000)
            self.assertFalse(legacy["engine_decode_boundaries"])
            self.assertIn("estimate", legacy["timing_alignment"])

    def test_invalid_engine_clock_does_not_fall_back_to_host(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for marker in (True, float("nan"), 120.5):
                with self.subTest(marker=marker):
                    events, thermal = self.coverage_fixture(root, [(110, False), (120, False)])
                    turn = json.loads(events.read_text())
                    turn["decode_finished_unix"] = marker
                    events.write_text(json.dumps(turn) + "\n")
                    result = decode_timing(events, thermal_log=thermal)
                    self.assertIsNone(result["active_decode_ms"])
                    self.assertIsNotNone(result["alignment_error"])

    def test_bench_clock_origin_matches_logged_event(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(provenance, "write"):
            root = Path(tmp)
            # This stand-in only speaks a ready/quit protocol. It cannot
            # initialise Vulkan or read any model data.
            executable = root / "protocol_stub.py"
            executable.write_text(f"#!{sys.executable}\n" +
                                  "import json,sys\n" +
                                  "print(json.dumps({'event':'ready'}),flush=True)\n" +
                                  "for line in sys.stdin:\n" +
                                  " if json.loads(line).get('op')=='quit': break\n")
            executable.chmod(0o700)
            args = SimpleNamespace(exe=str(executable), require_sources=0, max_context=64,
                                   cache_gb=0, shader_dir=None, env=[], serve_arg=[])
            server = BenchServer(args, root)
            try:
                clock = json.loads((root / "clock.json").read_text())
                event = json.loads((root / "events.jsonl").read_text())
                self.assertEqual(clock["host_start_unix"], server.t0)
                self.assertEqual(event["host_s"], round(event["host_unix"] - server.t0, 4))
                self.assertGreaterEqual(event["host_unix"], server.t0)
            finally:
                server.close()
                server.p.stdin.close()
                server.p.stdout.close()
                server.events.close()
                server.log.close()

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
            samples = [(100 + n / 20,
                        101 <= 100 + n / 20 < 105 or
                        113 <= 100 + n / 20 < 117 or
                        127 <= 100 + n / 20 < 129)
                       for n in range(621)]
            thermal.write_text("".join(json.dumps(dict(wall_time_s=t, paused=p)) + "\n"
                                       for t, p in samples))
            result = decode_timing(events, clock, thermal)
            self.assertEqual(result["raw_decode_ms"], 15000)
            self.assertEqual(result["thermal_pause_ms"], 6000)
            self.assertEqual(result["active_ms_per_token"], 600)
            self.assertEqual(result["raw_ms_per_token"], 1000)
            self.assertTrue(result["thermal_coverage"]["complete"])

    def coverage_fixture(self, root, samples):
        events, thermal = root / "events.jsonl", root / "thermal.jsonl"
        events.write_text(json.dumps(dict(event="done", host_unix=120, decode_ms=10000,
                                          decode_steps=10)) + "\n")
        thermal.write_text("".join(json.dumps(dict(wall_time_s=t, paused=p)) + "\n"
                                   for t, p in samples))
        return events, thermal

    def test_empty_old_or_truncated_log_cannot_claim_active_speed(self):
        fixtures = [([], "no complete samples"),
                    ([(1, False)], "ends before"),
                    ([(110 + n / 20, False) for n in range(81)], "ends before"),
                    ([(115 + n / 20, False) for n in range(101)], "starts after")]
        with tempfile.TemporaryDirectory() as tmp:
            for samples, expected in fixtures:
                with self.subTest(expected=expected, sample_count=len(samples)):
                    events, thermal = self.coverage_fixture(Path(tmp), samples)
                    result = decode_timing(events, thermal_log=thermal)
                    self.assertEqual(result["raw_decode_ms"], 10000)
                    self.assertEqual(result["raw_ms_per_token"], 1000)
                    self.assertIsNone(result["active_ms_per_token"])
                    self.assertFalse(result["thermal_coverage"]["complete"])
                    self.assertIn(expected, result["alignment_error"])

    def test_interior_sampling_gap_is_reported_not_filled(self):
        samples = [(110 + n / 20, False) for n in range(201)
                   if not 114 < 110 + n / 20 < 116]
        with tempfile.TemporaryDirectory() as tmp:
            events, thermal = self.coverage_fixture(Path(tmp), samples)
            result = decode_timing(events, thermal_log=thermal)
        self.assertIsNone(result["active_decode_ms"])
        self.assertIn("sampling gap", result["alignment_error"])
        self.assertEqual(result["thermal_coverage"]["windows"][0]["maximum_sample_gap_s"], 2)
        self.assertEqual(result["thermal_coverage"]["windows"][0]["maximum_gap_interval"], [114, 116])

    def test_live_log_may_end_one_sample_before_done_with_explicit_tolerance(self):
        samples = [(110 + n / 20, 113 <= 110 + n / 20 < 117) for n in range(200)]
        with tempfile.TemporaryDirectory() as tmp:
            events, thermal = self.coverage_fixture(Path(tmp), samples)
            # The next write can be incomplete while the caller reads the log.
            with thermal.open("a") as stream:
                stream.write('{"wall_time_s":')
            result = decode_timing(events, thermal_log=thermal)
        self.assertEqual(result["active_decode_ms"], 6000)
        self.assertEqual(result["active_ms_per_token"], 600)
        coverage = result["thermal_coverage"]
        self.assertTrue(coverage["complete"])
        self.assertTrue(coverage["partial_final_line"])
        self.assertEqual(coverage["tolerance_s"], .25)
        self.assertAlmostEqual(coverage["windows"][0]["missing_end_s"], .05)

    def test_invalid_legacy_record_preserves_raw_and_explains_alignment_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            events, thermal = self.coverage_fixture(Path(tmp), [])
            thermal.write_text('{"elapsed_s":10,"amdgpu":80}\n')
            result = decode_timing(events, thermal_log=thermal)
        self.assertEqual(result["raw_ms_per_token"], 1000)
        self.assertIsNone(result["active_ms_per_token"])
        self.assertIn("no absolute timestamps", result["alignment_error"])

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


class ThermalFailureReceipts(unittest.TestCase):
    class Monitor:
        def __enter__(self):
            return self

        def __exit__(self, *unused):
            pass

    @staticmethod
    def reading(temperature=42, timestamp=100):
        return {"amdgpu:fake": temperature, "ac": 1,
                "power_profile": "performance", "wall_time_s": timestamp}

    def run_mock_job(self, root, samples, child):
        # Every external action is mocked. These cases cannot spawn a process,
        # change the power profile, or signal a live engine.
        with patch.object(guard, "ProfileMonitor", self.Monitor), \
                patch.object(guard, "assert_idle"), \
                patch.object(guard.subprocess, "run"), \
                patch.object(guard.subprocess, "Popen", return_value=child), \
                patch.object(guard.os, "killpg"), \
                patch.object(guard.time, "sleep"), \
                patch.object(guard.time, "time", return_value=120), \
                patch.object(guard, "sample", side_effect=samples), \
                patch.object(guard, "terminate") as cleanup:
            job = dict(name="failure_receipt", command=["mock-no-process"],
                       active_timeout_s=10, wall_timeout_s=20)
            result = guard.run_job(job, root, {}, {"amdgpu:fake": None})
            cleanup.assert_called_once_with(child)
            return result

    def test_persistent_read_failure_preserves_primary_error_and_open_pause(self):
        child = SimpleNamespace(pid=999999, poll=lambda: None)
        samples = [self.reading(), self.reading(), self.reading(86, 101),
                   RuntimeError("monitor failed during the job"),
                   RuntimeError("monitor still unavailable")]
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_mock_job(Path(tmp), samples, child)
        self.assertEqual(result["rc"], 99)
        self.assertEqual(result["failure"], "monitor failed during the job")
        self.assertEqual(result["end_sample_error"], "monitor still unavailable")
        self.assertIsNone(result["end"])
        self.assertIsNone(result["ac_changed"])
        self.assertEqual(result["thermal_pause_intervals"], [[101, 120]])
        self.assertEqual(result["thermal_pauses"], 1)

    def test_missing_final_read_cannot_return_success_or_invent_ac_state(self):
        child = SimpleNamespace(pid=999999, poll=lambda: 0, wait=lambda: 0)
        samples = [self.reading(), self.reading(), OSError("sensor disappeared")]
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_mock_job(Path(tmp), samples, child)
        self.assertEqual(result["rc"], 99)
        self.assertIn("sensor disappeared", result["failure"])
        self.assertEqual(result["end_sample_error"], "sensor disappeared")
        self.assertIsNone(result["end"])
        self.assertIsNone(result["ac_changed"])
        self.assertEqual(result["thermal_pause_intervals"], [])

    def test_persistent_profile_failure_still_writes_main_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = root / "plan.json"
            plan.write_text(json.dumps(dict(cwd=str(root), jobs=[dict(name="mock")])) + "\n")
            result = dict(rc=99, failure="profile monitor failed", end=None)
            with patch.object(sys, "argv", ["thermal_guard.py", "--plan", str(plan)]), \
                    patch.object(guard.os, "chdir"), \
                    patch.object(guard.signal, "signal"), \
                    patch.object(guard, "discover_sensors", return_value={}), \
                    patch.object(guard, "profile", side_effect=["performance", OSError("CLI unavailable")]), \
                    patch.object(guard, "run_job", return_value=result):
                self.assertEqual(guard.main(), 99)
            self.assertEqual(json.loads((root / "check_results.json").read_text()), [result])
            receipt = json.loads((root / "profile_receipt.json").read_text())
            self.assertEqual(receipt["original"], "performance")
            self.assertIsNone(receipt["final"])
            self.assertEqual(receipt["final_read_error"], "CLI unavailable")


if __name__ == "__main__":
    unittest.main()
