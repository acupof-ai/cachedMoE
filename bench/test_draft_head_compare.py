#!/usr/bin/env python3
"""Stop on the first changed final ID, including an overlong candidate."""
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from draft_head_compare import BenchServer, CheckedServer, cache_identity, check_target_mode, check_heat_file


class CacheControl(unittest.TestCase):
    def test_explicit_heat_must_cover_nested_model_geometry(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(dict(text_config=dict(num_hidden_layers=1, n_routed_experts=2))))
            heat = root / "heat.h"
            heat.write_text("{0,0,10},\n{0,1,9},\n")
            check_heat_file(heat, root)
            heat.write_text("{0,0,10},\n{0,0,9},\n")
            with self.assertRaises(ValueError):
                check_heat_file(heat, root)

    def state(self):
        return dict(event="benchmark_cache_state", frozen=True, resident=2, free=0,
                    filling=0, queued_requests=0, inflight_chunks=0, fills_failed=0,
                    evictions=0, fills_started=2, slots=[[0, 0, 1, 0], [1, 40, 0, 1]])

    def test_same_counts_do_not_hide_changed_experts(self):
        state = self.state()
        expected = cache_identity(state)
        state["slots"][0][2] = 2
        with self.assertRaises(ValueError):
            cache_identity(state, expected)

    def test_same_counts_do_not_hide_pending_reads_or_replacements(self):
        for key in ("free", "filling", "queued_requests", "inflight_chunks", "fills_failed", "evictions"):
            with self.subTest(key=key):
                state = self.state()
                state[key] = 1
                with self.assertRaises(ValueError):
                    cache_identity(state)

    def test_rejects_bad_geometry_and_incomplete_map(self):
        for rows in ([], [[0, 0, 1]], [[0, 0, 1, 0], [0, 40, 0, 1]],
                     [[0, 0, 1, 0], [1, 0, 1, 0]], [[0, 0, 1, 2], [1, 40, 0, 1]]):
            with self.subTest(rows=rows):
                state = self.state()
                state["slots"] = rows
                with self.assertRaises(ValueError):
                    cache_identity(state)

    def test_settled_identical_cache_passes(self):
        state = self.state()
        expected = cache_identity(state)
        self.assertEqual(cache_identity(state, expected), expected)
        state["fills_started"] += 1
        with self.assertRaises(ValueError):
            cache_identity(state, expected)

    def test_exact_control_rejects_mask_or_plain_mode(self):
        check_target_mode(dict(decode_mode="off-spec", speculation=dict(cycles=2)), False)
        for mode, cycles in (("mask-spec", 2), ("off-plain", 0), ("off-spec", 0)):
            with self.assertRaises(ValueError):
                check_target_mode(dict(decode_mode=mode, speculation=dict(cycles=cycles)), False)


class FinalIdStop(unittest.TestCase):
    def stream(self, expected, tokens):
        server = CheckedServer.__new__(CheckedServer)
        server.start_turn(expected)
        requests = []
        server.send = requests.append
        with patch.object(BenchServer, "read_event", side_effect=[
                {"event": "token", "id": token} for token in tokens]):
            for token in tokens:
                server.read_event()
        return server, requests

    def test_equal_ids_do_not_cancel(self):
        server, requests = self.stream([1, 2, 3], [1, 2, 3])
        self.assertIsNone(server.difference)
        self.assertEqual(requests, [])

    def test_difference_cancels_once_even_if_cycle_has_more_tokens(self):
        server, requests = self.stream([1, 2, 3], [1, 9, 4])
        self.assertEqual(server.difference["position"], 1)
        self.assertEqual(requests, [{"op": "cancel"}])
        self.assertEqual(server.ids, [1, 9, 4])

    def test_extra_token_is_a_difference(self):
        server, requests = self.stream([1], [1, 2])
        self.assertIsNone(server.difference["reference"])
        self.assertEqual(len(requests), 1)


if __name__ == "__main__":
    unittest.main()
