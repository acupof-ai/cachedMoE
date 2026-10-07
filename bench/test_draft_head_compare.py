#!/usr/bin/env python3
"""Stop on the first changed final ID, including an overlong candidate."""
import unittest
from unittest.mock import patch
from draft_head_compare import BenchServer, CheckedServer


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
