#!/usr/bin/env python3
"""CPU regressions for runtime aliases and the actual Python control callers."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import runtime_env
import hitrate_bench
import perf_ledger
import provenance
from bench import spec_compare, web_longtest
from tools.web import launch_guarded


class RuntimeEnvironment(unittest.TestCase):
    def setUp(self):
        with runtime_env._warning_lock:
            runtime_env._warned.clear()

    def test_language_neutral_presence_vectors(self):
        # These mirror rename_prepared/env_cases.json without depending on raw
        # evidence files or a particular worktree remaining on this machine.
        vectors = [
            ({}, None, "absent", None),
            ({"DEEPMOE_X": "1"}, "1", "legacy", "deprecated"),
            ({"CACHEDMOE_X": "1"}, "1", "canonical", None),
            ({"CACHEDMOE_X": "1", "DEEPMOE_X": "1"}, "1", "canonical", None),
            ({"CACHEDMOE_X": "0", "DEEPMOE_X": "1"}, "0", "canonical", "conflict"),
            ({"CACHEDMOE_X": "", "DEEPMOE_X": "1"}, "", "canonical", "conflict"),
            ({"DEEPMOE_X": ""}, "", "legacy", "deprecated"),
            ({"CACHEDMOE_X": "", "DEEPMOE_X": ""}, "", "canonical", None),
            ({"CACHEDMOE_X": " ", "DEEPMOE_X": "1"}, " ", "canonical", "conflict"),
            ({"CACHEDMOE_X": "10", "DEEPMOE_X": "1"}, "10", "canonical", "conflict"),
        ]
        for env, value, source, warning in vectors:
            with self.subTest(env=env):
                self.setUp()
                stderr, stdout = io.StringIO(), io.StringIO()
                with redirect_stderr(stderr), redirect_stdout(stdout):
                    selected = runtime_env.resolve("CACHEDMOE_X", env)
                    self.assertEqual(runtime_env.getenv("DEEPMOE_X", "default", env),
                                     value if source != "absent" else "default")
                self.assertEqual((selected.value, selected.source), (value, source))
                self.assertEqual(selected.present, source != "absent")
                self.assertEqual(stdout.getvalue(), "")
                self.assertEqual(len(stderr.getvalue().splitlines()), int(warning is not None))
                if warning:
                    self.assertIn(warning, stderr.getvalue())

    def test_warning_once_is_thread_safe_and_has_no_values(self):
        env = {"CACHEDMOE_PRIVATE": "private-new-value", "DEEPMOE_PRIVATE": "private-old-value"}
        barrier = threading.Barrier(12)
        def read():
            barrier.wait()
            for _ in range(30):
                runtime_env.resolve("PRIVATE", env)
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            threads = [threading.Thread(target=read) for _ in range(12)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        self.assertEqual(len(stderr.getvalue().splitlines()), 1)
        self.assertNotIn("private-new-value", stderr.getvalue())
        self.assertNotIn("private-old-value", stderr.getvalue())
        self.assertIn("CACHEDMOE_PRIVATE", stderr.getvalue())
        self.assertIn("DEEPMOE_PRIVATE", stderr.getvalue())

    def test_warning_reasons_have_independent_once_state(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            runtime_env.resolve("X", {"DEEPMOE_X": "legacy"})
            runtime_env.resolve("X", {"CACHEDMOE_X": "new", "DEEPMOE_X": "legacy"})
            runtime_env.resolve("X", {"DEEPMOE_X": "changed"})
        self.assertEqual(len(stderr.getvalue().splitlines()), 2)

    def test_import_spellings_share_identity_and_warning_state(self):
        self.assertIs(runtime_env, importlib.import_module("tools.runtime_env"))
        for order in ("runtime_env, tools.runtime_env", "tools.runtime_env, runtime_env"):
            code = (f"import sys; sys.path[:0] = {[str(ROOT), str(ROOT / 'tools')]!r}; "
                    f"import {order}; import importlib; "
                    "a=importlib.import_module('runtime_env'); "
                    "b=importlib.import_module('tools.runtime_env'); assert a is b; "
                    "a.resolve('X', {'DEEPMOE_X':'private'}); "
                    "b.resolve('X', {'DEEPMOE_X':'private'})")
            child = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
            self.assertEqual(child.returncode, 0, child.stderr)
            self.assertEqual(len(child.stderr.splitlines()), 1)
            self.assertEqual(child.stdout, "")

    def test_default_requires_both_aliases_absent(self):
        for existing in ({"DEEPMOE_X": "0"}, {"DEEPMOE_X": ""}, {"CACHEDMOE_X": ""}):
            with self.subTest(existing=existing), redirect_stderr(io.StringIO()):
                env = dict(existing)
                runtime_env.setdefault(env, "X", "1")
                self.assertEqual(env, existing)
        env = {}
        self.assertEqual(runtime_env.setdefault(env, "X", "1"), "1")
        self.assertEqual(env, {"CACHEDMOE_X": "1"})

    def test_clear_and_force_remove_both_names(self):
        env = {"CACHEDMOE_X": "a", "DEEPMOE_X": "b", "HOME": "/keep"}
        runtime_env.set_value(env, "X", "")
        self.assertEqual(env, {"CACHEDMOE_X": "", "HOME": "/keep"})
        runtime_env.clear(env, "DEEPMOE_X")
        self.assertEqual(env, {"HOME": "/keep"})

    def test_explicit_layer_replaces_default_but_new_wins_within_layer(self):
        env = {"CACHEDMOE_X": "default"}
        runtime_env.apply_overrides(env, {"DEEPMOE_X": "explicit"})
        self.assertEqual(env, {"DEEPMOE_X": "explicit"})
        runtime_env.apply_overrides(env, {"DEEPMOE_X": "old", "CACHEDMOE_X": ""})
        self.assertEqual(runtime_env.resolve("X", env, warn=False).value, "")

    def test_canonical_subprocess_reset_preserves_effective_empty_and_external(self):
        original = {"DEEPMOE_X": "old", "CACHEDMOE_X": "", "DEEPMOE_Y": "10", "RADV_DEBUG": "llvm"}
        with redirect_stderr(io.StringIO()):
            env = runtime_env.canonicalized(original)
        self.assertEqual(env, {"CACHEDMOE_X": "", "CACHEDMOE_Y": "10", "RADV_DEBUG": "llvm"})
        self.assertEqual(original["DEEPMOE_X"], "old")

    def test_model_reader_preserves_new_empty_and_old_fallback(self):
        for controls, expected in (({"DEEPMOE_MODEL_DIR": "/old"}, "/old"),
                                   ({"DEEPMOE_MODEL_DIR": "/old", "CACHEDMOE_MODEL_DIR": ""}, "")):
            env = {k: v for k, v in os.environ.items() if not runtime_env.is_control(k)}
            env.update(controls)
            child = subprocess.run([sys.executable, "-c",
                                    f"import sys; sys.path.insert(0, {str(ROOT / 'tools')!r}); "
                                    "import runtime_defaults; print(repr(runtime_defaults.resolve_launch().model))"],
                                   env=env, capture_output=True, text=True)
            self.assertEqual(child.returncode, 0, child.stderr)
            self.assertEqual(child.stdout.strip(), repr(expected))

    def test_actual_auto_tune_legacy_cli_and_shell_defaults(self):
        self.assertEqual(hitrate_bench.auto_tune_controls(["DEEPMOE_BACKFILL=0"], "", {}),
                         ["DEEPMOE_BACKFILL=0"])
        self.assertEqual(hitrate_bench.auto_tune_controls([], "", {"DEEPMOE_BACKFILL": ""}), [])
        self.assertEqual(hitrate_bench.auto_tune_controls([], "", {}), ["CACHEDMOE_BACKFILL=1"])

    def test_actual_bench_server_legacy_cli_replaces_generated_control(self):
        args = argparse.Namespace(exe="unused-engine", max_context=4096, cache_slots=5500,
                                  cache_gb=0, serve_arg=[], shader_dir="/generated/shaders",
                                  env=["DEEPMOE_ROUTE_DUMP=/explicit/route",
                                       "DEEPMOE_SHADER_DIR=/explicit/shaders"], require_sources=0)
        inherited = {"CACHEDMOE_ROUTE_DUMP": "/shell/route",
                     "CACHEDMOE_SHADER_DIR": "/shell/shaders", "KEEP_ME": "yes"}
        with tempfile.TemporaryDirectory() as folder, \
             mock.patch.dict(os.environ, inherited, clear=True), \
             mock.patch.object(hitrate_bench.provenance, "write"), \
             mock.patch.object(hitrate_bench.subprocess, "Popen") as popen, \
             mock.patch.object(hitrate_bench.BenchServer, "read_event",
                               return_value={"event": "ready"}):
            server = hitrate_bench.BenchServer(args, folder)
            try:
                env = popen.call_args.kwargs["env"]
                self.assertEqual(env["DEEPMOE_ROUTE_DUMP"], "/explicit/route")
                self.assertEqual(env["DEEPMOE_SHADER_DIR"], "/explicit/shaders")
                self.assertNotIn("CACHEDMOE_ROUTE_DUMP", env)
                self.assertNotIn("CACHEDMOE_SHADER_DIR", env)
                self.assertEqual(env["KEEP_ME"], "yes")
            finally:
                server.events.close()
                server.log.close()

    def test_actual_spec_startup_validates_effective_aliases(self):
        with self.assertRaisesRegex(ValueError, "complete startup resources"):
            spec_compare.startup_environment({}, ["DEEPMOE_DSPARK_ONECB=0"])
        with redirect_stderr(io.StringIO()):
            env = spec_compare.startup_environment({}, ["DEEPMOE_DSPARK_ONECB=0",
                                                        "CACHEDMOE_DSPARK_ONECB=1",
                                                        "DEEPMOE_SPEC_DIAGNOSTICS=trace"])
        self.assertEqual(runtime_env.resolve("DSPARK_ONECB", env, warn=False).value, "1")
        self.assertEqual(runtime_env.resolve("SPEC_DIAGNOSTICS", env, warn=False).value, "trace")
        self.assertNotIn("CACHEDMOE_SPEC_DIAGNOSTICS", env)

    def test_actual_guarded_launch_purges_all_shell_controls(self):
        args = launch_guarded.arguments(["--repo", str(ROOT)])
        inherited = {"DEEPMOE_MASK_WAIT_TAU": ".2", "CACHEDMOE_MASK_WAIT_TAU": ".1",
                     "CACHEDMOE_UNEXPECTED_SWITCH": "1", "DEEPMOE_UNEXPECTED_SWITCH": "2",
                     "HOME": "/keep"}
        _, env = launch_guarded.launch_configuration(args, inherited)
        self.assertFalse(any(k.startswith("DEEPMOE_") for k in env))
        self.assertNotIn("CACHEDMOE_MASK_WAIT_TAU", env)
        self.assertNotIn("CACHEDMOE_UNEXPECTED_SWITCH", env)
        self.assertEqual(env["CACHEDMOE_DSPARK_ONECB"], "1")
        self.assertEqual(env["HOME"], "/keep")

    def test_actual_longtest_retains_intended_controls_and_clears_diagnostics(self):
        args = argparse.Namespace(mask_cache="dynamic", onecb=1, gpu_route=0)
        with redirect_stderr(io.StringIO()):
            env = web_longtest.run_environment(args, Path("/run"), False,
                {"DEEPMOE_MASK_WAIT_TAU": ".2", "DEEPMOE_ROUTE_DUMP": "old",
                 "CACHEDMOE_SPEC_DIAGNOSTICS": "stale", "CACHEDMOE_DSPARK_ONECB": "1"})
        self.assertEqual(env["CACHEDMOE_MASK_WAIT_TAU"], ".2")
        self.assertEqual(env["CACHEDMOE_DSPARK_ONECB"], "0")
        self.assertNotIn("CACHEDMOE_SPEC_DIAGNOSTICS", env)
        self.assertNotIn("CACHEDMOE_ROUTE_DUMP", env)
        self.assertFalse(any(k.startswith("DEEPMOE_") for k in env))

    def test_provenance_retains_original_aliases_empty_and_selected_source(self):
        env = {"DEEPMOE_X": "old", "CACHEDMOE_X": "", "DEEPMOE_Y": "1", "HOME": "/keep"}
        with tempfile.TemporaryDirectory() as folder, redirect_stderr(io.StringIO()), \
             mock.patch.object(provenance, "_git", return_value=""), \
             mock.patch.object(provenance, "power_state", return_value={"gpu_dpm": "auto"}), \
             mock.patch.object(provenance, "idle_check", return_value={}), \
             mock.patch.object(provenance, "disk_snapshot", return_value={}):
            captured = provenance.capture(Path(folder) / "unused", env, folder)
        self.assertEqual(captured["env"], {k: v for k, v in sorted(env.items()) if k != "HOME"})
        self.assertEqual(captured["env_effective"]["CACHEDMOE_X"],
                         {"value": "", "source": "canonical", "selected_key": "CACHEDMOE_X"})
        self.assertEqual(captured["env_effective"]["CACHEDMOE_Y"]["selected_key"], "DEEPMOE_Y")

    def test_ledger_old_only_order_and_mixed_metadata_exclusions(self):
        old = {"DEEPMOE_Z": "0", "DEEPMOE_MODEL_DIR": "/model", "DEEPMOE_A": "1"}
        self.assertEqual(perf_ledger.switches({"env": old}), "Z=0 A=1")
        mixed = old | {"CACHEDMOE_Z": "", "CACHEDMOE_SHADER_DIR": "/shaders",
                       "CACHEDMOE_ROUTE_DUMP": "/route", "DEEPMOE_LONGCTX_DIR": "/trace"}
        self.assertEqual(perf_ledger.switches({"env": mixed}), "Z= A=1")
        self.assertEqual(perf_ledger.switches({"env": {"DEEPMOE_MODEL_DIR": "/model"}}), "–")


if __name__ == "__main__":
    unittest.main()
