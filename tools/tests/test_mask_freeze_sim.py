"""Policy edges and trace validation, CPU only."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mask_freeze_sim import Controller, native_run, weighted_run


class FreezePolicy(unittest.TestCase):
    def ready(self, ctl):
        for _ in range(31):
            self.assertIsNone(ctl.observe(np.ones(40), 0, True))
        self.assertEqual(ctl.observe(np.ones(40), 0, True), "freeze")

    def test_warm_minimum_and_full_cache(self):
        ctl = Controller(5500)
        for _ in range(40):
            self.assertIsNone(ctl.observe(np.ones(40), 0, False))
        self.assertEqual(ctl.observe(np.ones(40), 0, True), "freeze")

    def test_churn_gate_is_strict_and_rolling(self):
        ctl = Controller(5500)
        for _ in range(31):
            ctl.observe(np.ones(40), 4, True)
        self.assertIsNone(ctl.observe(np.ones(40), 0, True))
        self.assertFalse(ctl.frozen)
        for _ in range(2):
            result = ctl.observe(np.ones(40), 0, True)
        self.assertEqual(result, "freeze")

    def test_hard_floor_bypasses_frozen_dwell(self):
        ctl = Controller(5500)
        self.ready(ctl)
        self.assertEqual(ctl.observe(np.full(40, .79), 0, True), "hard_floor")
        self.assertFalse(ctl.frozen)
        self.assertEqual(ctl.age, 0)
        self.assertFalse(ctl.churn)

    def test_soft_floor_respects_dwell_and_hysteresis(self):
        ctl = Controller(5500)
        self.ready(ctl)
        for _ in range(15):
            self.assertIsNone(ctl.observe(np.full(40, .85), 0, True))
        self.assertEqual(ctl.observe(np.full(40, .85), 0, True), "mean_mass")
        for _ in range(40):
            self.assertIsNone(ctl.observe(np.full(40, .94), 0, True))
        self.assertFalse(ctl.frozen)

    def test_worst_layer_has_its_own_ewma(self):
        ctl = Controller(5500)
        self.ready(ctl)
        mass = np.ones(40)
        mass[7] = .1
        for _ in range(15):
            ctl.observe(mass, 0, True)
        self.assertEqual(ctl.observe(mass, 0, True), "layer_mass")
        self.assertGreater(ctl.ewma.mean(), .95)
        ctl.reset()
        for layer in range(40):
            mass = np.ones(40)
            mass[layer] = .1
            ctl.observe(mass, 4, True)  # churn prevents entry
        self.assertGreater(ctl.ewma.min(), .85)

    def test_repeat_four_and_trigram_force_thaw(self):
        for tokens in ([8]*4, [1,2,3]*3):
            ctl = Controller(5500)
            self.ready(ctl)
            for token in tokens[:-1]:
                self.assertIsNone(ctl.observe(np.ones(40), 0, True, token))
            self.assertEqual(ctl.observe(np.ones(40), 0, True, tokens[-1]), "repeat")

    def test_turn_reset_discards_prior_history(self):
        ctl = Controller(5500)
        self.ready(ctl)
        for token in [8]*3:
            ctl.observe(np.ones(40), 0, True, token)
        ctl.reset()
        self.assertFalse(ctl.frozen)
        self.assertFalse(ctl.tokens)
        self.assertEqual(ctl.age, 0)

    def test_invalid_mass(self):
        ctl = Controller(5500)
        for mass in [np.full(40, float('nan')), np.full(40, -1), np.ones(39), np.full(40, 2)]:
            with self.assertRaises(ValueError):
                ctl.observe(mass, 0, True)

    def test_gate_mass_is_weighted_not_hit_count(self):
        ids = np.empty((2, 40, 6), dtype=np.int32)
        ids[0] = np.arange(6)
        ids[1] = [0, 6, 7, 8, 9, 10]
        weights = np.full((2, 40, 6), .25, dtype=np.float64)
        weights[1] = [1.45, .01, .01, .01, .01, .01]
        result = weighted_run((np.zeros(2), ids, weights, {}), 500, .95, .90)
        self.assertAlmostEqual(result['mean_pre_admission_mass'], (1.45 / 1.5) / 2)

    def test_native_never_invents_gate_weights_and_rejects_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            heat = root / 'heat.inc'
            heat.write_text('0, 0, 1,\n')
            route = root / 'route.bin'
            record = struct.pack('<II', 0, 12) + np.zeros((40,6), dtype='<u2').tobytes() + bytes(40)
            route.write_bytes(record)
            (root / 'turns.json').write_text(json.dumps(dict(turns=[dict(prefill_mode='gpu', prefill_tokens=12, decode_steps=1)])))
            self.assertIsNone(native_run(root, 5500, heat)['weighted_mass'])
            route.write_bytes(record + b'\0')
            with self.assertRaises(ValueError):
                native_run(root, 5500, heat)
            route.write_bytes(record * 2)
            with self.assertRaises(ValueError):
                native_run(root, 5500, heat)


if __name__ == '__main__':
    unittest.main()
