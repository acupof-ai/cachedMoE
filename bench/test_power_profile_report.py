"""CPU gates for profile selection and thermal accounting."""
import math
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from power_profile_report import PROFILES, report, select_profile, thermal_summary


class PowerProfileReportTests(unittest.TestCase):
    def test_lower_power_tie_uses_raw_three_percent_boundary(self):
        selected = select_profile({"power-saver": 103, "balanced": 102, "performance": 100})
        self.assertEqual(selected["selected"], "power-saver")
        selected = select_profile({"power-saver": 103.001, "balanced": 102, "performance": 100})
        self.assertEqual(selected["selected"], "balanced")

    def test_incomplete_or_invalid_arm_cannot_choose_default(self):
        for costs in ({"balanced": 1}, {"power-saver": math.nan, "balanced": 1, "performance": 2},
                      {"power-saver": 0, "balanced": 1, "performance": 2}):
            with self.assertRaises(ValueError):
                select_profile(costs)

    def test_pause_and_clock_intersections_exclude_prefill(self):
        rows = [dict(wall_time_s=t, paused=paused, pp_dpm_sclk=f"0: {clock}Mhz *",
                     **{"amdgpu:test": 50 + t, "nvme:test": 40 + t})
                for t, paused, clock in ((0, False, 1000), (1, True, 500),
                                        (2, False, 2000), (3, False, 2000))]
        turn = dict(decode_ms=2000, decode_finished_unix=3)
        result = thermal_summary(rows, [turn])
        self.assertEqual(result["decode_cooling_s"], 1)
        self.assertEqual(result["pause_entries_in_decode"], 1)
        self.assertEqual(result["mean_gpu_clock_mhz"], 1250)
        self.assertEqual(result["peak_c"]["amdgpu:test"], 52)

    def test_short_and_sustained_disagreement_requires_owner(self):
        raw = ((102, 85, 110), (100, 100, 100), (110, 90, 95))
        reductions = [dict(groups={name: dict(timing=dict(raw_ms_per_token=value,
                                                           active_ms_per_token=1))
                                  for name, value in zip(("all", "first_two", "last_two"), costs)})
                      for costs in raw]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for mode in PROFILES:
                (out / mode).mkdir()
                (out / mode / "turns.json").write_text(json.dumps([]))
            with patch("power_profile_report.arm_report", side_effect=reductions):
                result = report(out)
        self.assertEqual(result["default_candidate"], "power-saver")
        self.assertTrue(result["owner_decision_required"])
        self.assertFalse(result["default_change_authorized"])


if __name__ == "__main__":
    unittest.main()
