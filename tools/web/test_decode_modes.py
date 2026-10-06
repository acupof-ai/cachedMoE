"""Per-turn web modes. Mock protocol only; no engine or GPU is started."""
from pathlib import Path
import queue
import re
import shutil
import subprocess
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import server


def ready(default="mask-spec", available=None):
    return {
        "decode_modes": {
            "available": list(server.DECODE_MODES) if available is None else available,
            "default": default,
        },
        "speculation": {"enabled": True, "draft_tokens": 2, "accept_top_k": 4},
    }


def bridge_without_worker(policy):
    bridge = server.Bridge.__new__(server.Bridge)
    bridge.serve = Mock()
    bridge.serve.ready = policy
    bridge.serve.max_context = 64
    bridge.jobs = []
    bridge.jobs_lock = threading.Lock()
    bridge.jobs_cv = threading.Condition(bridge.jobs_lock)
    bridge.prefill_ms_per_token = 100
    return bridge


class DecodeModes(unittest.TestCase):
    def test_config_separates_resident_resources_and_selectable_modes(self):
        bridge = SimpleNamespace(
            serve=SimpleNamespace(config=SimpleNamespace(model="/selected/model"), ready=ready(), max_context=64, cmd=["mock"]),
            args=SimpleNamespace(resident_only="mask", mask_cache="dynamic"),
            prefill_ms_per_token=100,
        )
        with patch.object(server.subprocess, "check_output", return_value="performance"), \
                patch.object(server.provenance, "_read", return_value="performance"):
            config = server.web_configuration(bridge)
        self.assertEqual(config["decode_modes"], ready()["decode_modes"])
        self.assertTrue(config["speculation_resources_resident"])
        self.assertNotIn("speculation_enabled", config)

    def test_explicit_modes_and_default_are_captured_at_submission(self):
        bridge = bridge_without_worker(ready())
        body = {"text": "question", "decode_mode": "off-plain"}
        job = bridge.submit("tab", body)
        body["decode_mode"] = "mask-spec"
        self.assertEqual(job.body["decode_mode"], "off-plain")
        inherited = bridge.submit("other-tab", {"text": "next"})
        bridge.serve.ready["decode_modes"]["default"] = "mask-plain"
        self.assertEqual(inherited.body["decode_mode"], "mask-spec")

    def test_invalid_or_unavailable_mode_never_enters_queue(self):
        bridge = bridge_without_worker(ready(available=["off-plain", "mask-plain"]))
        for mode in (None, True, 1, [], {}, "", "off", "MASK-PLAIN", "mask-spec"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                bridge.submit("tab", {"text": "q", "decode_mode": mode})
        self.assertEqual(bridge.jobs, [])

    def test_nonselectable_startup_and_legacy_preserve_inheritance(self):
        for policy in ({}, ready(default="mask-spec", available=[])):
            bridge = bridge_without_worker(policy)
            job = bridge.submit("tab", {"text": "q"})
            self.assertNotIn("decode_mode", job.body)
        self.assertEqual(server.decode_mode_capabilities(ready(available=[])),
                         {"available": [], "default": "mask-spec"})

    def test_http_rejects_mode_before_sse_or_queue(self):
        for mode in (None, False, 2, "unknown", "mask-spec"):
            handler = server.Handler.__new__(server.Handler)
            handler.path = "/api/chat"
            handler.bridge = bridge_without_worker(ready(available=["off-plain"]))
            handler._json_body = Mock(return_value={"text": "q", "decode_mode": mode})
            handler._send = Mock()
            handler._chat = Mock()
            handler.do_POST()
            self.assertEqual(handler._send.call_args.args[0], 400)
            handler._chat.assert_not_called()
            self.assertEqual(handler.bridge.jobs, [])

    def test_generate_passes_mode_and_history_saves_actual_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            for mode in server.DECODE_MODES:
                with self.subTest(mode=mode):
                    bridge = bridge_without_worker(ready())
                    enc = Mock()
                    enc.parse_message_from_completion_text.return_value = {
                        "role": "assistant", "content": "answer", "reasoning_content": ""}
                    path = str(Path(tmp) / (mode + ".json"))
                    state = server.ChatState(enc, path=path)
                    state.prompt_ids = Mock(return_value=([1, 2], "prompt"))
                    bridge.state = lambda session: state
                    events = queue.Queue()
                    events.put({"event": "token", "id": 3, "text": "answer"})
                    # Actual response wins over the requested mode, even if different.
                    events.put({"event": "done", "decode_mode": "off-plain",
                                "speculation_enabled": False, "generated": 1,
                                "finish": "length"})
                    bridge.serve.begin_turn.return_value = events
                    bridge.serve.detokenize.return_value = "answer"
                    job = bridge.submit("tab", {"text": "q", "decode_mode": mode})
                    bridge._run(job)
                    self.assertEqual(bridge.serve.send.call_args.args[0]["decode_mode"], mode)
                    bridge.serve.end_turn.assert_called_once()
                    done = list(job.out.queue)[-1]
                    self.assertEqual(done["decode_mode"], "off-plain")
                    self.assertIs(done["speculation_enabled"], False)
                    restored = server.ChatState(enc, path=path)
                    stat = restored.history()[-1]["stat"]
                    self.assertEqual(stat["decode_mode"], "off-plain")
                    self.assertIs(stat["speculation_enabled"], False)

    def test_legacy_generate_omits_mode_and_does_not_invent_actual_policy(self):
        bridge = bridge_without_worker({})
        state = server.ChatState(Mock())
        state.prompt_ids = Mock(return_value=([1], "prompt"))
        state.commit = Mock(return_value={"content": ""})
        bridge.state = lambda session: state
        events = queue.Queue()
        events.put({"event": "done", "finish": "length"})
        bridge.serve.begin_turn.return_value = events
        job = bridge.submit("tab", {"text": "q"})
        bridge._run(job)
        self.assertNotIn("decode_mode", bridge.serve.send.call_args.args[0])
        stat = state.commit.return_value["stat"]
        self.assertIsNone(stat["decode_mode"])
        self.assertIsNone(stat["speculation_enabled"])


@unittest.skipUnless(shutil.which("node"), "Node required for browser JavaScript checks")
class BrowserModes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = Path(__file__).with_name("index.html").read_text()
        cls.script = re.search(r"<script>(.*?)</script>", cls.html, re.S).group(1)
        modes = cls.script[cls.script.index("function decodeModeLabel"):
                           cls.script.index("async function loadConfig")]
        request = cls.script[cls.script.index("function chatRequest"):
                             cls.script.index("async function send")]
        cls.helpers = modes + request

    def javascript(self, assertions):
        setup = """
const assert = require('assert');
const controls = {sess:{value:'tab'}, temp:{value:'1'}, topp:{value:'.95'},
  maxtok:{value:'32'}, seed:{value:''}, decode_mode:{value:'', disabled:true,
    options:[], replaceChildren(){this.options=[]}, add(option){this.options.push(option)}}};
function $(name){return controls[name]}
class Option {constructor(text,value){this.text=text; this.value=value}}
"""
        result = subprocess.run([shutil.which("node"), "-"],
                                input=setup + self.helpers + assertions,
                                text=True, capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_full_page_javascript_syntax(self):
        result = subprocess.run([shutil.which("node"), "--check"], input=self.script,
                                text=True, capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_ready_default_and_nonselectable_startup(self):
        self.javascript("""
configureDecodeModes({available:['mask-spec','mask-plain','off-plain'],default:'mask-spec'});
assert.equal(controls.decode_mode.value,'mask-spec');
assert.equal(controls.decode_mode.disabled,false);
configureDecodeModes({available:[],default:'mask-spec'});
assert.equal(controls.decode_mode.value,'mask-spec');
assert.equal(controls.decode_mode.disabled,true);
assert.equal('decode_mode' in chatRequest('q',false,75),false);
configureDecodeModes({});
assert.equal(controls.decode_mode.disabled,true);
""")

    def test_request_captures_mode_before_control_changes(self):
        self.javascript("""
configureDecodeModes({available:['mask-plain','off-plain'],default:'mask-plain'});
const pending = chatRequest('question',true,100);
controls.decode_mode.value='off-plain';
assert.equal(pending.decode_mode,'mask-plain');
assert.equal(chatRequest('next',false,75).decode_mode,'off-plain');
assert.equal(pending.reasoning_effort,100);
""")

    def test_actual_mode_display_and_exact_history_notice(self):
        self.javascript("""
assert.equal(decodeModeSummary({decode_mode:'off-plain',speculation_enabled:false}),
             'Off · plain / speculation off');
assert.equal(decodeModeSummary({decode_mode:'mask-spec',speculation_enabled:true}),
             'Mask + speculation / speculation on');
assert.equal(decodeModeSummary({}), 'Engine default / speculation unknown');
""")
        self.assertIn("Start a new chat for an exact history", self.html)
        self.assertIn("Resident DSpark", self.html)


if __name__ == "__main__":
    unittest.main()
