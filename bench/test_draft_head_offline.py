import unittest
import copy

import numpy as np

from draft_head_offline import (accepted_prefix, candidate_metrics, chain, frequency_subset,
                                gain, quantize_rows, top_ids, validate_capture)


class DraftHeadOfflineTest(unittest.TestCase):
    def test_subset_counts_and_declared_zero_frequency_tie(self):
        self.assertEqual(frequency_subset([7, 7, 5, 5, 5], 8, 4).tolist(), [5, 7, 0, 1])
        with self.assertRaises(ValueError):
            frequency_subset([8], 8, 4)

    def test_top_ties_are_token_ids_and_nan_is_rejected(self):
        ids = np.array([9, 5, 3, 7])
        self.assertEqual(top_ids(np.ones(4), ids).tolist(), [3, 5, 7, 9])
        with self.assertRaises(ValueError):
            top_ids(np.array([0, 1, 2, np.nan]), ids)

    def test_markov_second_row_uses_new_proposal(self):
        base = np.zeros((1, 2, 4), dtype=np.float32)
        embed = np.array([[1], [0], [-1], [0]], dtype=np.float32)
        markov = np.array([[0], [0], [10], [-10]], dtype=np.float32)
        paths, _ = chain(base, embed, markov, [0], np.arange(4))
        self.assertEqual(paths.tolist(), [[2, 3]])

    def test_rejection_stops_prefix_even_if_later_row_matches(self):
        self.assertEqual(accepted_prefix([1, 9, 3], [[1], [2], [3]]), 1)
        self.assertEqual(accepted_prefix([1, 2], [[7], [2]]), 0)

    def test_halved_gain_is_decision_basis(self):
        result = gain(8.4, 1000, 500)
        self.assertAlmostEqual(result["halved_saved_ms"], 2.1)
        self.assertTrue(result["gain_gate_pass"])
        self.assertFalse(gain(7.9, 1000, 500)["gain_gate_pass"])

    def test_invalid_cost_cannot_pass_gain_gate(self):
        for head_ms, source, derived in ((float("nan"), 100, 50), (float("inf"), 100, 50),
                                         (0, 100, 50), (8, 0, 0), (8, 100, 101)):
            with self.subTest(head_ms=head_ms), self.assertRaises(ValueError):
                gain(head_ms, source, derived)

    def test_fp8_row_scales_zero_rows_and_source_immutability(self):
        import torch
        weights = torch.tensor([[0, 0, 0], [448, -224, 112], [4.48, -2.24, 1.12]],
                               dtype=torch.bfloat16)
        before = weights.clone()
        encoded, scales, decoded = quantize_rows(weights)
        self.assertEqual(encoded.dtype, torch.float8_e4m3fn)
        self.assertEqual(scales.dtype, torch.float32)
        self.assertTrue(torch.equal(weights, before))
        self.assertTrue(torch.isfinite(decoded).all())
        self.assertEqual(scales[0].item(), 1)
        self.assertTrue(torch.equal(decoded[0], torch.zeros(3)))
        torch.testing.assert_close(scales[1:], weights[1:].float().abs().amax(dim=1) / 448)
        torch.testing.assert_close(decoded[1:], weights[1:].float())

    def test_fp8_rounds_halfway_to_even_and_rejects_nonfinite_rows(self):
        import torch
        _, _, decoded = quantize_rows(torch.tensor([[448., 1.0625, 1.1875, -1.0625]]))
        self.assertEqual(decoded.tolist(), [[448., 1., 1.25, -1.]])
        for bad in (float("nan"), float("inf")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                quantize_rows(torch.tensor([[bad, 1.]]))

    def test_unused_draft_row_does_not_change_agreement(self):
        cycles = [dict(k=1, accepted=1, proposal=[1], target_top4=[1, 2, 3, 4] * 2,
                       target_greedy=[1, 1])]
        baseline = np.array([[1, 2]])
        candidate = np.array([[1, 9]])
        tops = np.array([[[1, 2, 3, 4], [2, 3, 4, 5]]])
        different = tops.copy()
        different[0, 1] = [9, 8, 7, 6]
        result = candidate_metrics(candidate, different, baseline, tops, cycles, [2, 3, 4, 5])
        for key in ("top1_agreement", "top4_set_agreement", "top4_rows_identical"):
            self.assertEqual(result[key], 1)
        self.assertEqual(result["recorded_greedy_targets_outside_subset"], 1)

    @staticmethod
    def capture():
        meta = dict(schema=1, sampling="greedy", outputs=64, first_output=0,
                    block_rows=5, dim=2, vocab=8, slot_bytes=18808832)
        cycles, blocks = [], 0
        for generated in range(1, 64):
            k = min(2, 63 - generated)
            cycles.append(dict(k=k, accepted=0, position=10 + generated, root=0,
                               proposal=[7] * k, target_greedy=[0] * (k + 1),
                               target_top4=[0, 1, 2, 3] * (k + 1), emitted=[0],
                               hidden_index=blocks if k else -1))
            blocks += bool(k)
        return meta, cycles, np.zeros((blocks, 5, 2), dtype=np.float32)

    def test_complete_capture_accepts_native_tail_k1_and_k0(self):
        validate_capture(*self.capture())

    def test_incomplete_or_inconsistent_capture_is_rejected(self):
        mutations = (
            lambda m, c, h: c.pop(),
            lambda m, c, h: c[1].update(root=1),
            lambda m, c, h: c[1].update(position=10),
            lambda m, c, h: c[0].update(hidden_index=-1),
            lambda m, c, h: c[0].update(target_greedy=[1, 0, 0]),
            lambda m, c, h: c[0].update(target_top4=[0, 0, 2, 3] * 3),
            lambda m, c, h: c[0].update(proposal=[0, 7]),
            lambda m, c, h: c[0].update(proposal=[8, 7]),
            lambda m, c, h: h.fill(float("nan")),
            lambda m, c, h: m.update(outputs=63),
        )
        for mutate in mutations:
            data = copy.deepcopy(self.capture())
            mutate(*data)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                validate_capture(*data)


if __name__ == "__main__":
    unittest.main()
