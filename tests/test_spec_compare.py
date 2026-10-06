"""CPU checks for the adaptive arm choice; this module starts no engine."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "spec_compare", Path(__file__).resolve().parents[1] / "bench/spec_compare.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def row(name, cost, valid=True):
    return dict(name=name, decode_timing=dict(active_ms_per_token=cost, raw_ms_per_token=cost,
                                             engine_decode_boundaries=True),
                repetition=dict(no_loop=valid))


class SpecCompareTests(unittest.TestCase):
    def test_arm_validation(self):
        self.assertEqual(bench.parse_arm("k5_onecb_cpu_route"), (5, "onecb_cpu_route"))
        for invalid in ("k0_gpu", "k6_gpu", "k2_tree", "k2_gpujunk"):
            with self.assertRaises(ValueError):
                bench.parse_arm(invalid)

    def test_route_choice_respects_noise_and_quality(self):
        controls = [row("k2_gpu", 97.1), row("k2_onecb_cpu_route", 100)]
        self.assertFalse(bench.choose_route(controls)[0])
        controls[0] = row("k2_gpu", 96.9)
        self.assertTrue(bench.choose_route(controls)[0])
        controls[0] = row("k2_gpu", 90, False)
        self.assertFalse(bench.choose_route(controls)[0])
        controls[1] = row("k2_onecb_cpu_route", 100, False)
        with self.assertRaises(ValueError):
            bench.choose_route(controls)

    def test_missing_thermal_alignment_cannot_select_default(self):
        with self.assertRaises(ValueError):
            bench.choose_route([row("k2_gpu", None), row("k2_onecb_cpu_route", 100)])
        self.assertFalse(bench.choose_route([], "cpu")[0])

    def test_host_estimate_cannot_choose_route(self):
        controls = [row("k2_gpu", 90), row("k2_onecb_cpu_route", 100)]
        controls[0]["decode_timing"]["engine_decode_boundaries"] = False
        with self.assertRaisesRegex(ValueError, "engine decode boundaries"):
            bench.choose_route(controls)

    def test_gpu_cannot_win_only_by_subtracting_cpu_suspension(self):
        controls = [row("k2_gpu", 90), row("k2_onecb_cpu_route", 100)]
        controls[0]["decode_timing"]["raw_ms_per_token"] = 110
        use_gpu, receipt = bench.choose_route(controls)
        self.assertFalse(use_gpu)
        self.assertEqual(receipt["raw_costs"]["k2_gpu"], 110)

    def test_spec_totals_excludes_non_done_events(self):
        with tempfile.TemporaryDirectory() as folder:
            events = Path(folder) / "events.jsonl"
            events.write_text(
                '{"event":"token","speculation":{"cycles":99}}\n'
                '{"event":"done","speculation":{"cycles":2,"accepted":3,"verify_ms":4.5}}\n'
                '{"event":"done","speculation":{"cycles":1,"accepted":2,"verify_ms":2.0}}\n')
            self.assertEqual(bench.spec_totals(events),
                             dict(cycles=3, accepted=5, verify_ms=6.5))


if __name__ == "__main__":
    unittest.main()
