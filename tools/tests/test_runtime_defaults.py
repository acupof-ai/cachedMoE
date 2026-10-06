"""CPU checks of final launch configuration and the web's API-sourced controls."""
import argparse
from contextlib import redirect_stderr
import importlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tools/web"), str(ROOT)]
import runtime_defaults as defaults
import runtime_env
import chat
import hitrate_bench
import ms_bench
import provenance
import server
from bench import spec_compare, web_longtest
from tools.web import launch_guarded


class LaunchConfiguration(unittest.TestCase):
    def test_changed_bench_entrypoints_work_without_pythonpath(self):
        # Importing these in this test process could hide a missing bootstrap:
        # each real script starts independently from an unrelated directory.
        scripts = ("config_sweep", "d2_abab", "one_config", "reheat_ab", "reheat_probe",
                   "route_compare", "spec_compare", "thermal_guard", "web_longtest")
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        with tempfile.TemporaryDirectory() as folder:
            for script in scripts:
                with self.subTest(script=script):
                    child = subprocess.run([sys.executable, str(ROOT / "bench" / (script + ".py")),
                                            "--help"], cwd=folder, env=env,
                                           capture_output=True, text=True, timeout=5)
                    self.assertEqual(child.returncode, 0, child.stderr)
                    self.assertIn("usage:", child.stdout)

    def test_import_spellings_and_api_copies_have_one_authority(self):
        self.assertIs(defaults, importlib.import_module("tools.runtime_defaults"))
        for order in ("tools.runtime_defaults, runtime_defaults", "runtime_defaults, tools.runtime_defaults"):
            code = (f"import sys; sys.path[:0]={[str(ROOT), str(ROOT / 'tools')]!r}; "
                    f"import {order}; "
                    "assert sys.modules['runtime_defaults'] is sys.modules['tools.runtime_defaults']")
            child = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
            self.assertEqual(child.returncode, 0, child.stderr)
        policy = defaults.request_policy()
        policy["defaults"]["reasoning_effort"] = 1
        policy["constraints"]["reasoning_effort"]["max"] = 2
        self.assertEqual(defaults.request_policy()["defaults"]["reasoning_effort"], 75)
        self.assertEqual(defaults.request_policy()["constraints"]["reasoning_effort"]["max"], 100)
        with self.assertRaises(TypeError):
            defaults.REQUEST_CONSTRAINTS["reasoning_effort"]["max"] = 2

    def test_launch_snapshot_is_independent_of_shell_and_read_only(self):
        source = {"CACHEDMOE_MODEL_DIR": "/selected"}
        config = defaults.resolve_launch("engine", source)
        source["CACHEDMOE_MODEL_DIR"] = "/later"
        self.assertEqual(config.environment["CACHEDMOE_MODEL_DIR"], "/selected")
        with self.assertRaises(TypeError):
            config.environment["CACHEDMOE_MODEL_DIR"] = "/later"

    def test_model_presence_empty_and_deliberate_or_fallback(self):
        cases = [({}, defaults.model_fallback(), "default"),
                 ({"DEEPMOE_MODEL_DIR": "/old"}, "/old", "legacy"),
                 ({"CACHEDMOE_MODEL_DIR": "", "DEEPMOE_MODEL_DIR": "/old"}, "", "canonical")]
        for env, expected, source in cases:
            with self.subTest(env=env), redirect_stderr(io.StringIO()):
                config = defaults.resolve_launch("engine", env)
                self.assertEqual((config.model, config.model_source), (expected, source))
        self.assertEqual(defaults.model_directory({"CACHEDMOE_MODEL_DIR": ""}, empty_fallback=True),
                         defaults.model_fallback())
        self.assertEqual(defaults.resolve_launch("engine", {}, model="").model, "")

    def test_bench_env_and_native_last_cli_model_override_reach_encoder_and_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            model = folder / "actual-model"
            (model / "encoding").mkdir(parents=True)
            (model / "encoding/encoding.py").write_text("MODEL_SENTINEL = 'actual'\n")
            args = argparse.Namespace(exe="unused-cpu-fixture", max_context=4096, cache_slots=5500,
                cache_gb=0, require_sources=0, shader_dir="/generated/shaders", model="/explicit-model",
                env=["DEEPMOE_MODEL_DIR=/environment-model", "DEEPMOE_SHADER_DIR=/effective/shaders"],
                serve_arg=["--model", "/earlier-model", "--model", str(model)])
            with patch.dict(os.environ, {"CACHEDMOE_MODEL_DIR": "/shell-model"}, clear=True), \
                    redirect_stderr(io.StringIO()):
                config = hitrate_bench.bench_configuration(args, temporary)
            self.assertEqual(config.model, str(model))
            self.assertEqual(chat.load_encoding(config.model).MODEL_SENTINEL, "actual")
            with patch.object(provenance, "_git", return_value=""), \
                    patch.object(provenance, "power_state", return_value={"gpu_dpm": "auto"}), \
                    patch.object(provenance, "idle_check", return_value={}), \
                    patch.object(provenance, "disk_snapshot", return_value={}), \
                    patch.object(hitrate_bench.subprocess, "Popen") as popen, \
                    patch.object(hitrate_bench.BenchServer, "read_event", return_value={"event": "ready"}), \
                    redirect_stderr(io.StringIO()):
                engine = hitrate_bench.BenchServer(args, temporary, config=config)
                try:
                    cmd = popen.call_args.args[0]
                    selected = defaults.cli_value(cmd, "--model")
                    recorded = json.loads((folder / "provenance.json").read_text())
                    self.assertEqual(selected, str(model))
                    self.assertEqual(recorded["launch"]["model"], selected)
                    self.assertEqual(recorded["launch"]["shader_dir"], "/effective/shaders")
                    self.assertEqual(recorded["exe"], args.exe)
                    self.assertEqual(popen.call_args.kwargs["env"], config.environment)
                    self.assertEqual(recorded["env"]["DEEPMOE_MODEL_DIR"], "/environment-model")
                finally:
                    engine.events.close()
                    engine.log.close()

    def test_environment_model_without_cli_override_and_ms_renderer_agree(self):
        args = argparse.Namespace(exe="cpu-fixture", env=["DEEPMOE_MODEL_DIR=/explicit-old"],
                                  serve_arg=[], shader_dir="", route_dump=False)
        with redirect_stderr(io.StringIO()):
            config = hitrate_bench.bench_configuration(args, "/out", {"CACHEDMOE_MODEL_DIR": "/shell"})
            multi = ms_bench.launch_configuration(args, "/out", {"CACHEDMOE_MODEL_DIR": "/shell"})
        self.assertEqual(config.model, "/explicit-old")
        self.assertEqual(multi.model, config.model)
        args.env += ["CACHEDMOE_MODEL_DIR="]
        with redirect_stderr(io.StringIO()):
            self.assertEqual(hitrate_bench.bench_configuration(args, "/out", {}).model, "")

    def test_renderer_modules_are_keyed_by_selected_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            modules = []
            for index in (1, 2):
                model = Path(folder) / str(index)
                (model / "encoding").mkdir(parents=True)
                source = model / "encoding/encoding.py"
                source.write_text(f"SENTINEL={index}\n")
                module = defaults.load_encoding(str(model))
                modules.append(module)
                self.assertEqual(module.SENTINEL, index)
                self.assertIs(module, server.load_encoding(str(model)))
                self.assertFalse((source.parent / "__pycache__").exists())
            self.assertIsNot(modules[0], modules[1])

    def test_failed_renderer_import_is_removed_and_can_be_retried(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "encoding/encoding.py"
            source.parent.mkdir()
            source.write_text("raise ValueError('fixture import failure')\n")
            before = set(sys.modules)
            with self.assertRaisesRegex(ValueError, "fixture import failure"):
                defaults.load_encoding(folder)
            self.assertEqual(set(sys.modules), before)
            source.write_text("SENTINEL='recovered'\n")
            self.assertEqual(defaults.load_encoding(folder).SENTINEL, "recovered")
            self.assertFalse((source.parent / "__pycache__").exists())

    def test_actual_cpu_child_observes_the_final_argv_and_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            exe = Path(folder) / "cpu-protocol-fixture"
            exe.write_text(f"#!{sys.executable}\n" + """
import json, os, sys
print(json.dumps(dict(event='ready', observed_argv=sys.argv[1:],
 observed_model_env=os.environ.get('CACHEDMOE_MODEL_DIR'))), flush=True)
for line in sys.stdin:
    if json.loads(line).get('op') == 'quit':
        break
""")
            exe.chmod(0o700)
            args = argparse.Namespace(exe=str(exe), max_context=4096, cache_slots=0,
                cache_gb=0, require_sources=0, shader_dir="", env=["CACHEDMOE_MODEL_DIR=/effective-env"],
                serve_arg=["--model", "/effective-cli"])
            with patch.object(provenance, "_git", return_value=""), \
                    patch.object(provenance, "power_state", return_value={"gpu_dpm": "auto"}), \
                    patch.object(provenance, "idle_check", return_value={}), \
                    patch.object(provenance, "disk_snapshot", return_value={}):
                child = hitrate_bench.BenchServer(args, folder)
                try:
                    self.assertEqual(defaults.cli_value(child.ready["observed_argv"], "--model"),
                                     "/effective-cli")
                    self.assertEqual(child.ready["observed_model_env"], "/effective-env")
                    self.assertEqual(child.config.model, "/effective-cli")
                finally:
                    child.close()
                    child.events.close()
                    child.log.close()
                    child.p.stdin.close()
                    child.p.stdout.close()
                self.assertEqual(child.p.returncode, 0)

    def test_named_profiles_preserve_reset_and_inheritance_differences(self):
        inherited = {"CACHEDMOE_MASK_WAIT_TAU": ".2", "DEEPMOE_SHADOW": "1",
                     "RADV_DEBUG": "llvm", "CACHEDMOE_SPEC_DIAGNOSTICS": "old"}
        args = launch_guarded.arguments(["--repo", str(ROOT)])
        _, production = launch_guarded.launch_configuration(args, inherited)
        self.assertNotIn("CACHEDMOE_MASK_WAIT_TAU", production)
        self.assertNotIn("DEEPMOE_SHADOW", production)
        self.assertEqual(production["CACHEDMOE_BATCH_GPU_ROUTE"], "0")
        self.assertEqual(production["RADV_DEBUG"], "llvm")
        args = SimpleNamespace(mask_cache="dynamic", onecb=1, gpu_route=0)
        with redirect_stderr(io.StringIO()):
            longtest = web_longtest.run_environment(args, Path("/out"), True, inherited)
        self.assertEqual(longtest["CACHEDMOE_MASK_WAIT_TAU"], ".2")
        self.assertNotIn("CACHEDMOE_SPEC_DIAGNOSTICS", longtest)
        self.assertNotIn("CACHEDMOE_DSPARK_TRIM_TAIL", longtest)
        resources = spec_compare.startup_environment({}, [])
        self.assertEqual(resources["CACHEDMOE_DSPARK_TRIM_TAIL"], "1")
        self.assertEqual(resources["CACHEDMOE_BATCH_GPU_ROUTE"], "1")


class WebPolicy(unittest.TestCase):
    def test_api_actual_model_defaults_constraints_and_live_power(self):
        bridge = SimpleNamespace(serve=SimpleNamespace(
            config=defaults.resolve_launch("engine", {}, model="/chosen/model"), max_context=64,
            cmd=["engine"], ready={}), args=SimpleNamespace(resident_only="off", mask_cache="dynamic"),
            prefill_ms_per_token=24)
        with patch.object(server.subprocess, "check_output", side_effect=["performance", "power-saver"]), \
                patch.object(server.provenance, "_read", return_value="performance"):
            first, second = server.web_configuration(bridge), server.web_configuration(bridge)
        self.assertEqual(first["model"], "/chosen/model")
        self.assertEqual(first["request_defaults"], defaults.request_policy()["defaults"])
        self.assertEqual(first["request_constraints"], defaults.request_policy()["constraints"])
        self.assertEqual(first["effort_presets"], defaults.request_policy()["effort_presets"])
        self.assertEqual((first["power_profile"], second["power_profile"]), ("performance", "power-saver"))

    def test_effort_remains_strict_and_request_token_ui_max_is_not_http_cap(self):
        for value in (None, True, 1.0, "75", 0, 101):
            with self.subTest(value=value), self.assertRaises(ValueError):
                server.reasoning_effort(value)
        self.assertEqual(server.reasoning_effort(75), 75)
        handler = server.Handler.__new__(server.Handler)
        handler.path = "/api/chat"
        handler.bridge = SimpleNamespace(serve=SimpleNamespace(ready={}))
        handler._json_body = Mock(return_value={"text": "q", "max_tokens": 9000})
        handler._chat = Mock()
        handler._send = Mock()
        handler.do_POST()
        handler._chat.assert_called_once()
        handler._send.assert_not_called()

    def test_verbose_is_captured_once_and_regular_logs_do_not_read_environment(self):
        handler = server.Handler.__new__(server.Handler)
        handler.address_string = lambda: "cpu-fixture"
        handler.verbose = True
        with patch.object(server.runtime_env, "getenv", side_effect=AssertionError("hot env read")), \
                redirect_stderr(io.StringIO()) as text:
            handler.log_message("%s", "message")
        self.assertIn("message", text.getvalue())

    def test_actual_web_startup_uses_one_model_snapshot_and_truthy_verbose(self):
        fake = SimpleNamespace(ready={"load_s": 0, "cache_gb": 0, "cache_slots": 0},
                               max_context=64, close=Mock())
        root = SimpleNamespace(path="/unused-state", source="explicit", both_exist=False)
        bridge = SimpleNamespace(state_root=root, chat_dir="/unused-state/web_chat")
        http = Mock()
        observed = []
        def started(args, config):
            fake.config = config
            self.assertEqual(config.model, "/startup-model")
            # A later ambient change cannot change this startup snapshot.
            os.environ["CACHEDMOE_MODEL_DIR"] = "/later-model"
            os.environ["CACHEDMOE_WEB_VERBOSE"] = ""
            return fake
        def serving():
            observed.append(server.Handler.verbose)
            raise KeyboardInterrupt
        http.serve_forever.side_effect = serving
        with patch.dict(os.environ, {"CACHEDMOE_MODEL_DIR": "/startup-model",
                                     "CACHEDMOE_WEB_VERBOSE": "0"}, clear=True), \
                patch.object(sys, "argv", ["web", "--max-context", "64", "--log", ""]), \
                patch.object(server, "load_encoding", return_value=Mock()) as encoder, \
                patch.object(server, "Serve", side_effect=started), \
                patch.object(server, "Bridge", return_value=bridge), \
                patch.object(server, "GpuMon"), \
                patch.object(server, "ThreadingHTTPServer", return_value=http), \
                patch.object(server.Handler, "bridge", None), \
                patch.object(server.Handler, "gpumon", None), \
                patch.object(server.Handler, "verbose", False), \
                patch.object(sys, "stdout", io.StringIO()):
            self.assertEqual(server.main(), 0)
        encoder.assert_called_once_with("/startup-model")
        fake.close.assert_called_once()
        self.assertEqual(observed, [True])

    @unittest.skipUnless(shutil.which("node"), "node is required for the actual UI helpers")
    def test_ui_consumes_changed_api_values_and_captures_queue_settings(self):
        html = (ROOT / "tools/web/index.html").read_text()
        script = re.search(r"<script>(.*?)</script>", html, re.S).group(1)
        helpers = script[script.index("function configureRequestPolicy"):script.index("async function loadConfig")]
        helpers += script[script.index("function chatRequest"):script.index("async function send")]
        helpers += script[script.index("function effortValue"):script.index('$("think").addEventListener')]
        setup = """
const assert = require('assert');
const controls={};
for(const id of ['temp','topp','maxtok','effort_custom','send','think','seed','sess','effort_custom_label'])
  controls[id]={value:'',disabled:true,checked:true};
for(const id of ['effort','decode_mode'])controls[id]={options:[],value:'',disabled:false,
 replaceChildren(){this.options=[]},add(option){this.options.push(option)}};
function $(id){return controls[id]}
class Option{constructor(label,value){this.label=label;this.value=value}}
let CFG={request_defaults:{temperature:.7,top_p:.8,max_tokens:2048,reasoning_effort:63},
 request_constraints:{temperature:{min:0,max:3,step:.1},top_p:{min:0,max:1,step:.01},
 max_tokens:{min:1,max:12000,step:1},reasoning_effort:{min:2,max:90,step:1}},
 effort_presets:[{label:'Custom preset',value:63}]};
"""
        assertions = """
configureRequestPolicy(CFG);
assert.equal(controls.temp.value,.7);assert.equal(controls.temp.max,3);
assert.equal(controls.maxtok.value,2048);assert.equal(controls.maxtok.max,12000);
assert.equal(effortValue(),63);assert.equal(controls.effort.options[0].value,63);
controls.decode_mode.value='off-plain';controls.sess.value='tab';
const queued=chatRequest('q',true,effortValue());
controls.temp.value=1.9;controls.effort.value='custom';controls.effort_custom.value=88;
controls.decode_mode.value='mask-spec';
assert.equal(queued.temperature,.7);assert.equal(queued.reasoning_effort,63);
assert.equal(queued.decode_mode,'off-plain');
"""
        child = subprocess.run([shutil.which("node"), "-"], input=setup + helpers + assertions,
                               text=True, capture_output=True, timeout=5)
        self.assertEqual(child.returncode, 0, child.stderr)
        for name in ("temp", "topp", "maxtok", "effort_custom"):
            tag = re.search(r'<input[^>]*id="' + name + r'"[^>]*>', html).group()
            self.assertNotRegex(tag, r'\b(?:min|max|step|value)="')


if __name__ == "__main__":
    unittest.main()
