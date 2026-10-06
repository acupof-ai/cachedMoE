"""Current/legacy process names and aggregate aliases; no GPU executable."""
import importlib.util
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "bench")]
import process_names as names
import provenance
import thermal_guard


class ProcessNames(unittest.TestCase):
    def test_current_legacy_windows_and_truncated_probe_names(self):
        for name in ("cachedmoe", "deepmoe", "cachedmoe_tests", "deepmoe_tests",
                     "cachedmoe_tests.exe", "deepmoe_tests.exe", "CachedMoE.exe",
                     "dspark_grid_probe", "dspark_grid_pro", "cachedmoe-teste"):
            with self.subTest(name=name):
                self.assertTrue(names.is_gpu_process(name))
        for name in ("python", "cachedmoe_other", "mydeepmoe", "deepmoe_manifest.json"):
            self.assertFalse(names.is_gpu_process(name))
        self.assertTrue(names.is_engine_comm("cachedmoe"))
        self.assertTrue(names.is_engine_comm("deepmoe"))
        self.assertFalse(names.is_engine_comm("cachedmoe_tests"))

    def test_idle_pattern_covers_every_declared_executable_comm(self):
        for executable in names.GPU_EXECUTABLE_NAMES:
            self.assertIsNotNone(re.fullmatch(names.GPU_COMM_PATTERN, executable[:15]))
        self.assertIsNone(re.fullmatch(names.GPU_COMM_PATTERN, "cachedmoe_other"))
        self.assertIsNone(re.fullmatch(names.GPU_COMM_PATTERN, "mydeepmoe"))

    def test_current_cmake_targets_and_legacy_aliases_are_all_covered(self):
        cmake = "\n".join((ROOT / path).read_text() for path in
                           ("CMakeLists.txt", "tests/CMakeLists.txt"))
        targets = set(re.findall(r"add_executable\s*\(\s*(\w+)", cmake))
        for group in re.findall(r"foreach\s*\(_probe\s+([^)]*)\)", cmake):
            targets.update(group.split())
        for current, legacy in re.findall(
                r"(?m)^cachedmoe_legacy_executable\s*\(\s*(\w+)\s+(\w+)", cmake):
            targets.update((current, legacy))
        self.assertIn("dspark_bench", targets)
        self.assertIn("dspark_grid_probe", targets)
        for target in sorted(targets):
            with self.subTest(target=target):
                self.assertTrue(names.is_gpu_process(target))
                self.assertIsNotNone(re.fullmatch(names.GPU_COMM_PATTERN, target[:15]))

    def test_thermal_idle_uses_shared_pattern_and_fails_closed(self):
        for rc in (0, 1, 2):
            with self.subTest(rc=rc), patch.object(thermal_guard.subprocess, "run",
                    return_value=SimpleNamespace(returncode=rc, stdout="123")) as run:
                if rc == 1:
                    thermal_guard.assert_idle()
                else:
                    with self.assertRaisesRegex(RuntimeError, "another GPU engine/test"):
                        thermal_guard.assert_idle()
                self.assertEqual(run.call_args.args[0], ["pgrep", "-x", names.GPU_COMM_PATTERN])

    def test_provenance_records_low_cpu_gpu_jobs_without_losing_top_cpu(self):
        rows = "10.0 other\n0.0 cachedmoe\n0.0 deepmoe_tests\n0.0 dspark_grid_pro\n"
        with patch.object(provenance.Path, "glob", return_value=[]), \
                patch.object(provenance.subprocess, "run", return_value=SimpleNamespace(stdout=rows)), \
                patch.object(provenance, "_read", return_value=None):
            idle = provenance.idle_check(0)
        self.assertEqual(idle["top_cpu"], ["other 10.0%"])
        self.assertEqual(idle["gpu_processes"], ["cachedmoe 0.0%", "deepmoe_tests 0.0%",
                                                  "dspark_grid_pro 0.0%"])

    def test_ctest_both_aggregate_aliases_are_excluded(self):
        spec = importlib.util.spec_from_file_location("cachedmoe_run_all_fixture", ROOT / "tests/run_all.py")
        run_all = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_all)
        output = "  Test #1: cachedmoe_tests\n  Test #2: deepmoe_tests\n  Test #3: suite.cpu\n"
        with patch.object(run_all, "run", return_value=(0, output)):
            self.assertEqual(run_all.ctest_unit_suites(ROOT / "build"), ["suite.cpu"])


if __name__ == "__main__":
    unittest.main()
