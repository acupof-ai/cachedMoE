#!/usr/bin/env python3
"""CPU contract checks for the engine timing marker and delayed done delivery."""
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DecodeBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = (ROOT / "runtime/session.cpp").read_text()
        cls.serve = (ROOT / "cli/serve.cpp").read_text()

    def test_finish_clock_precedes_reheat_and_single_turn_return(self):
        finish = self.session.split("void turn_finish(", 1)[1].split(
            "}  // namespace", 1)[0]
        self.assertLess(finish.index("finish_decode_clock(ts)"),
                        finish.index("e.reheat("))
        single = self.session.split("Result<GenerateStats> Session::generate(", 1)[1].split(
            "// --- Track MS", 1)[0]
        self.assertLess(single.index("turn_finish(e, ts, opt_)"),
                        single.rindex("return ts.st"))
        serve = self.serve.split("auto st = pool.live().generate(", 1)[1]
        self.assertLess(serve.index("pool.checkpoint_active()"),
                        serve.index("st->json_fields()"))
        # The protocol emits the captured stats; it does not manufacture a
        # timestamp after the checkpoint and mistake it for the decode end.
        self.assertNotIn("system_clock::now()", serve)

    def test_each_stream_freezes_zero_step_boundary_without_late_overwrite(self):
        stopped = self.session.split("if (!turn_emit(e, ts[i], i, on_token))", 1)[1].split(
            "continue;", 1)[0]
        self.assertIn("finish_decode_clock(ts[i])", stopped)
        finish = self.session.split("void turn_finish(", 1)[1].split(
            "}  // namespace", 1)[0]
        self.assertIn("if (!st.decode_finished_unix) finish_decode_clock(ts)", finish)
        self.assertNotIn("if (st.decode_ms == 0.0)", finish)

    def test_wall_and_monotonic_clocks_are_taken_at_the_same_finish_site(self):
        clock = self.session.split("void finish_decode_clock(", 1)[1].split(
            "void turn_finish(", 1)[0]
        self.assertIn("const TimePoint finished = Clock::now()", clock)
        self.assertIn("std::chrono::system_clock::now()", clock)
        self.assertIn("finished - ts.td", clock)
        self.assertIn("finished_unix.time_since_epoch()", clock)
        self.assertNotIn("checkpoint", clock.split("const TimePoint", 1)[1])
        # At current epoch magnitudes, %.9g would round the marker by seconds.
        self.assertIn('std::format("{:.17g}", *decode_finished_unix)', self.session)

    def test_mock_checkpoint_and_delivery_delay_cannot_move_saved_marker(self):
        now = 1791251234.1234567
        stats = {"decode_ms": 125.25, "decode_steps": 3,
                 "decode_finished_unix": now}
        frozen = stats.copy()
        # A checkpoint can wait for disk queue space, and SIGSTOP can defer
        # writing/reading done. Both happen after the stats clock was frozen.
        now += 12.0
        emitted = json.loads(json.dumps({"event": "done", **stats}))
        received = {**emitted, "host_unix": now + 4.0}
        self.assertEqual(received["decode_finished_unix"], frozen["decode_finished_unix"])
        self.assertEqual(received["decode_ms"], frozen["decode_ms"])
        self.assertGreater(received["host_unix"], received["decode_finished_unix"])


if __name__ == "__main__":
    unittest.main()
