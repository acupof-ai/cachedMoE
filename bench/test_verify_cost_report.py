#!/usr/bin/env python3
import unittest
from verify_cost_report import summarize


class VerifyAccounting(unittest.TestCase):
    def inputs(self):
        rows = [dict(schema=1)]
        for position in (29, 32):
            rows.append(dict(k=2, position=position, emitted=3,
                             verify_host_ms=dict(record_layers_tail=180, fence_wait=165,
                                                 engram_issue_land=23)))
        events = [dict(event='done', decode_steps=6,
                       speculation=dict(cycles=2, draft_ms=52, verify_ms=400,
                                        commit_ms=2, cpu_ms=1))]
        return rows, events

    def test_nested_timers_are_not_added_to_cycle_cost(self):
        result = summarize(*self.inputs())
        self.assertEqual(result['accounted_cycle_ms'], 227.5)
        self.assertFalse(result['host_timers_are_additive'])
        self.assertEqual(result['mean_verify_host_ms']['fence_wait'], 165)

    def test_native_events_limit_excludes_candidate_tail(self):
        rows, events = self.inputs()
        rows.append(dict(k=2))
        self.assertEqual(summarize(rows, events)['cycles'], 2)

    def test_missing_or_reordered_cycles_rejected(self):
        rows, events = self.inputs()
        rows[-1]['position'] = 33
        with self.assertRaises(ValueError):
            summarize(rows, events)
        with self.assertRaises(ValueError):
            summarize(rows[:2], events)

    def test_mismatched_emitted_counts_rejected(self):
        rows, events = self.inputs()
        rows[1]['emitted'] = 2
        with self.assertRaises(ValueError):
            summarize(rows, events)


if __name__ == '__main__':
    unittest.main()
