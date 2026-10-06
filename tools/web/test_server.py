"""Web configuration checks. No engine process or GPU is started."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest.mock import Mock, call, mock_open, patch

import server


class WebSettings(unittest.TestCase):
    @staticmethod
    def bare_serve():
        serve = server.Serve.__new__(server.Serve)
        serve.io_lock = threading.Lock()
        serve.closing = False
        serve.dead = False
        serve.p = Mock()
        serve.p.poll.return_value = None
        serve.p.wait.return_value = 0
        return serve

    def test_shutdown_requires_successful_engine_exit(self):
        serve = self.bare_serve()
        serve._write_unlocked = Mock()
        serve.close()
        self.assertEqual(serve._write_unlocked.call_args_list,
                         [call({"op": "cancel"}), call({"op": "quit"})])
        serve.p.kill.assert_not_called()
        serve.p.wait.return_value = 1
        with self.assertRaisesRegex(RuntimeError, "KV drain"):
            serve.close()

    def test_shutdown_timeout_reports_unconfirmed_drain(self):
        serve = self.bare_serve()
        serve.p.wait.side_effect = [server.subprocess.TimeoutExpired("cachedmoe", 120), -9]
        with self.assertRaisesRegex(RuntimeError, "KV drain is unconfirmed"):
            serve.close()
        serve.p.kill.assert_called_once()

    def test_constructor_initializes_closing_before_launch(self):
        serve = server.Serve.__new__(server.Serve)
        args = SimpleNamespace(log=None, max_context=64)

        def start(*unused, **kwargs):
            self.assertFalse(serve.closing)
            self.assertIsNotNone(serve.io_lock)
            return Mock()

        with patch.object(serve, "command", return_value=["mock-no-process"]), \
                patch.object(serve, "_read_ready", return_value=dict(max_context=64)), \
                patch.object(server.subprocess, "Popen", side_effect=start), \
                patch.object(server.threading, "Thread"):
            serve.__init__(args)

    def test_normal_send_repeated_close_and_late_send(self):
        serve = self.bare_serve()
        serve.send({"op": "generate", "max_tokens": 2})
        serve.close()
        serve.close()
        with self.assertRaisesRegex(RuntimeError, "closing"):
            serve.send({"op": "generate", "max_tokens": 2048})
        operations = [json.loads(call.args[0])["op"] for call in serve.p.stdin.write.call_args_list]
        self.assertEqual(operations, ["generate", "cancel", "quit"])
        self.assertEqual(serve.p.wait.call_count, 2)
        serve.p.kill.assert_not_called()

    def test_worker_cannot_generate_between_shutdown_cancel_and_quit(self):
        serve = self.bare_serve()
        cancel_written, worker_attempting = threading.Event(), threading.Event()
        operations, locked, worker_errors, close_errors = [], [], [], []

        def write(obj):
            operations.append(obj["op"])
            locked.append(serve.io_lock.locked())
            if obj["op"] == "cancel":
                cancel_written.set()
                if not worker_attempting.wait(1):
                    raise RuntimeError("CPU worker did not reach send")

        def worker():
            if not cancel_written.wait(1):
                worker_errors.append("cancel hook timed out")
                return
            worker_attempting.set()
            try:
                serve.send({"op": "generate", "max_tokens": 2048})
            except RuntimeError as error:
                worker_errors.append(str(error))

        def closer():
            try:
                serve.close()
            except Exception as error:
                close_errors.append(str(error))

        serve._write_unlocked = write
        # Daemon threads keep a locking regression from hanging the CPU gate.
        writer = threading.Thread(target=worker, daemon=True)
        closing = threading.Thread(target=closer, daemon=True)
        writer.start()
        closing.start()
        closing.join(timeout=2)
        writer.join(timeout=2)
        self.assertFalse(closing.is_alive(), "close deadlocked")
        self.assertFalse(writer.is_alive(), "send deadlocked")
        self.assertEqual(close_errors, [])
        self.assertEqual(operations, ["cancel", "quit"])
        self.assertEqual(locked, [True, True])
        self.assertEqual(worker_errors, ["cachedmoe serve is closing"])
        serve.p.kill.assert_not_called()

    def test_broken_shutdown_pipe_keeps_failed_drain_status(self):
        serve = self.bare_serve()
        serve.p.stdin.write.side_effect = BrokenPipeError("mock closed pipe")
        serve.p.wait.return_value = -9
        with self.assertRaisesRegex(RuntimeError, "KV drain is unconfirmed"):
            serve.close()
        self.assertTrue(serve.closing)
        serve.p.kill.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "KV drain"):
            serve.close()
        self.assertEqual(serve.p.stdin.write.call_count, 1)

    def test_dead_engine_still_rejects_regular_send(self):
        serve = self.bare_serve()
        serve.dead = True
        with self.assertRaisesRegex(RuntimeError, "gone"):
            serve.send({"op": "status"})
        serve.p.stdin.write.assert_not_called()

    def test_config_reports_active_speculation_and_current_power(self):
        bridge = SimpleNamespace(
            args=SimpleNamespace(resident_only="mask", mask_cache="dynamic"),
            serve=SimpleNamespace(max_context=1 << 20, cmd=["cachedmoe", "serve"],
                ready={"speculation": {"enabled": True, "draft_tokens": 2,
                                       "accept_top_k": 4}}),
            prefill_ms_per_token=100)
        with patch.object(server.subprocess, "check_output", return_value="performance\n"), \
                patch("builtins.open", mock_open(read_data="performance\n")):
            config = server.web_configuration(bridge)
        self.assertEqual(config["resident_only"], "mask")
        self.assertEqual(config["mask_cache"], "dynamic")
        self.assertEqual(config["spec_k"], 2)
        self.assertEqual(config["power_profile"], "performance")
        self.assertEqual(config["platform_profile"], "performance")
        bridge.serve.ready["speculation"]["enabled"] = False
        with patch.object(server.subprocess, "check_output", side_effect=OSError), \
                patch("builtins.open", side_effect=OSError):
            plain = server.web_configuration(bridge)
        self.assertEqual(plain["spec_k"], 0)
        self.assertIsNone(plain["power_profile"])

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

    def test_partial_thinking_stays_reasoning_after_reload(self):
        enc = Mock()
        enc.parse_message_from_completion_text.side_effect = ValueError("missing </think>")
        state = server.ChatState(enc)
        state.think = True
        message = state.commit("q", "prompt", [2], [3], "Still working")
        self.assertEqual(message["reasoning_content"], "Still working")
        self.assertEqual(message["content"], "")
        self.assertEqual(state.history()[-1], message)

    def test_legacy_cancelled_thinking_is_reclassified_only_with_exact_prefix(self):
        enc = Mock()
        enc.encode_messages.return_value = "prompt<think>"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.json"
            data = {"think": True, "ctx_text": "prompt<think>Partial reasoning",
                    "messages": [{"role": "user", "content": "q"},
                        {"role": "assistant", "content": "Partial reasoning", "reasoning_content": "",
                         "stat": {"finish": "cancel"}}]}
            path.write_text(json.dumps(data))
            state = server.ChatState(enc, path=str(path))
            self.assertEqual(state.history()[-1]["content"], "")
            self.assertEqual(state.history()[-1]["reasoning_content"], "Partial reasoning")
            data["ctx_text"] = "unmatched prefix"
            path.write_text(json.dumps(data))
            unchanged = server.ChatState(enc, path=str(path))
            self.assertEqual(unchanged.history()[-1]["content"], "Partial reasoning")

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
        args = SimpleNamespace(exe="cachedmoe", max_context=1 << 20, cache_slots=5500,
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
        args = SimpleNamespace(exe="cachedmoe", max_context=4096, cache_slots=5500,
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
