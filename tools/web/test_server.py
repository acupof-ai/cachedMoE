"""Web configuration checks. No engine process or GPU is started."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock

import server


class WebSettings(unittest.TestCase):
    def test_effort_changes_prompt_and_survives_restart(self):
        enc = Mock()
        enc.encode_messages.side_effect = lambda messages, **settings: json.dumps([messages, settings])
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "session.json")
            state = server.ChatState(enc, path=path)
            default = state.render("question")
            state.settings({"think": True, "reasoning_effort": 100})
            maximum = state.render("question")
            self.assertNotEqual(default, maximum)
            self.assertEqual(json.loads(maximum)[1]["reasoning_effort"], 100)
            state.save()
            restored = server.ChatState(enc, path=path)
            self.assertEqual(restored.render("question"), maximum)
            restored.reset()
            self.assertEqual(restored.reasoning_effort, 100)

    def test_old_transcript_uses_native_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.json"
            path.write_text(json.dumps({"messages": [], "think": True}))
            state = server.ChatState(Mock(), path=str(path))
            self.assertEqual(state.reasoning_effort, 75)

    def test_invalid_effort_rejected_before_queue_or_tokenize(self):
        for route in ("/api/chat", "/api/preview"):
            for invalid in (0, 101, True, 75.0, "100", None):
                with self.subTest(route=route, value=invalid):
                    handler = Mock(path=route)
                    handler._json_body.return_value = {"reasoning_effort": invalid}
                    server.Handler.do_POST(handler)
                    self.assertEqual(handler._send.call_args.args[0], 400)
                    handler._chat.assert_not_called()
                    handler._preview.assert_not_called()

    def test_1m_context_passes_to_engine(self):
        args = SimpleNamespace(exe="deepmoe", max_context=1 << 20, cache_slots=5500,
            cache_gb=0, gpu_prefill_min=16, gpu_prefill_speedup=None, kv_dir="",
            no_kv_disk=True, kv_max_gb=0, max_parked=0, resident_only="mask",
            dspark=True, spec_k=5, spec_top_k=4, mirror=[], no_mirror_auto=True,
            mask_cache="dynamic")
        command = server.Serve.command(args)
        self.assertEqual(command[command.index("--max-context") + 1], "1048576")
        self.assertEqual(command[command.index("--mask-cache") + 1], "dynamic")
        args.mask_cache = "fixed"
        command = server.Serve.command(args)
        self.assertEqual(command[command.index("--mask-cache") + 1], "fixed")
        # Keep the web preflight bound consistent with the compiled engine cap.
        self.assertEqual(server.K_MAX_INDEX_POSITIONS, 1 << 20)

    def test_speculation_is_explicit_and_passes_length(self):
        args = SimpleNamespace(exe="deepmoe", max_context=4096, cache_slots=5500,
            cache_gb=0, gpu_prefill_min=16, gpu_prefill_speedup=None, kv_dir="",
            no_kv_disk=True, kv_max_gb=0, max_parked=0, resident_only="mask",
            dspark=False, spec_k=5, spec_top_k=4, mirror=["/mirror"], no_mirror_auto=True,
            mask_cache="dynamic")
        self.assertNotIn("--dspark", server.Serve.command(args))
        args.dspark = True
        command = server.Serve.command(args)
        self.assertIn("--dspark", command)
        self.assertEqual(command[command.index("--spec-k") + 1], "5")
        self.assertEqual(command[command.index("--spec-top-k") + 1], "4")
        self.assertEqual(command[command.index("--mirror") + 1], "/mirror")


if __name__ == "__main__":
    unittest.main()
