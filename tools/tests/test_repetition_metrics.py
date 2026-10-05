import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repetition_metrics import from_events, metrics


class RepetitionTests(unittest.TestCase):
    def test_run_boundary(self):
        self.assertTrue(metrics([7] * 3)["no_loop"])
        self.assertFalse(metrics([7] * 4)["no_loop"])
        self.assertEqual(metrics([7, 7, 8, 8, 8])["longest_same_token_run"], 3)

    def test_period_boundaries_and_embedded_loop(self):
        self.assertTrue(metrics([1, 2] * 7)["no_loop"])
        self.assertFalse(metrics([1, 2] * 8)["no_loop"])
        self.assertFalse(metrics(list(range(100)) + list(range(8)) * 4 + [99])["no_loop"])
        self.assertTrue(metrics(list(range(9)) * 4)["no_loop"])
        self.assertTrue(metrics(list(range(8)) * 3)["no_loop"])

    def test_ngrams_and_empty(self):
        self.assertEqual(metrics([])["distinct_2"], 0)
        self.assertEqual(metrics([1, 2, 1, 2, 1])["distinct_2"], .5)
        self.assertAlmostEqual(metrics([1, 2, 3, 4] * 2)["repeated_4gram_fraction"], .2)

    def test_events_never_cross_outputs_or_count_prompts(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "events.jsonl"
            events = [{"event": "tokens", "ids": [8] * 30}]
            for _ in range(2):
                events += [{"event": "token", "id": 7}] * 3 + [{"event": "done"}]
            p.write_text("\n".join(map(json.dumps, events)))
            result = from_events(p)
            self.assertTrue(result["no_loop"])
            self.assertEqual(result["tokens"], 6)
            self.assertEqual(len(result["outputs"]), 2)
            p.write_text('{"event":"token","id":true}')
            with self.assertRaisesRegex(ValueError, "invalid token"):
                from_events(p)


if __name__ == "__main__":
    unittest.main()
