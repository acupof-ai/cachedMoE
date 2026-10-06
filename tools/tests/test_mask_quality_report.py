import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
from mask_quality_report import (build_report, c_decision, cycle_report,
                                 failure_report, quality_gates, repetition_gate)
from repetition_metrics import metrics


ZERO_STATUS = dict(store="expert store; failed fills 0",
                   planner="P0 failures reserve 0 / submit 0 / IO 0",
                   io="io: 20 req (20 done, 0 failed, 0 cancelled)")


def save(path, value):
    Path(path).write_text(json.dumps(value))


class QualityReportTests(unittest.TestCase):
    def test_complete_cycle_includes_uncovered_time(self):
        turn = dict(speculation=dict(cycles=2, verified=4, accepted=2, tokens=4,
            draft_ms=100, verify_ms=600, commit_ms=50, cpu_ms=50,
            union_experts=800, miss_bytes=1000))
        report = cycle_report([turn], dict(raw_decode_ms=1000, active_decode_ms=800))
        self.assertEqual(report["full_raw_cycle_ms"], 500)
        self.assertEqual(report["full_active_cycle_ms"], 400)
        self.assertEqual(report["uncovered_raw_cycle_ms"], 100)
        self.assertEqual(report["acceptance"], .5)
        self.assertEqual(report["outputs_per_cycle"], 2)
        self.assertEqual(report["verified_per_cycle"], 2)
        self.assertEqual(report["union_per_cycle"], 400)
        self.assertAlmostEqual(report["common_2_25"]["active_ms_per_output"], 400 / 2.25)
        self.assertAlmostEqual(report["common_accepted_fraction"]["uncorrected"]["outputs_per_cycle"], 2.2)

    def test_terminal_stop_and_missing_stage_are_explicit(self):
        row = dict(speculation=dict(cycles=2, verified=4, accepted=2, tokens=3,
                                   draft_ms=100))
        result = cycle_report([row], dict(raw_decode_ms=1000, active_decode_ms=None))
        self.assertIsNone(result["full_active_cycle_ms"])
        self.assertIsNone(result["components_raw_ms"]["verify_ms"])
        self.assertIsNone(result["uncovered_raw_cycle_ms"])
        normalized = result["common_accepted_fraction"]
        self.assertEqual(normalized["observed_terminal_correction"], .5)
        self.assertAlmostEqual(normalized["terminal_adjusted"]["outputs_per_cycle"], 1.7)

    def test_zero_off_repeat_has_no_epsilon_floor(self):
        off = [dict(label="same", seed=1, repetition=metrics(list(range(8))))]
        repeated = [dict(label="same", seed=1, repetition=metrics([1, 2, 3, 4] * 2))]
        result = repetition_gate(repeated, off)
        self.assertFalse(result["passed"])
        self.assertEqual(result["turns"][0]["repeat4_limit"], 0)
        self.assertIsNone(result["turns"][0]["repeat4_ratio"])
        self.assertTrue(repetition_gate(off, off)["passed"])
        mismatch = [dict(label="different", seed=1, repetition=metrics(list(range(8))))]
        self.assertIsNone(repetition_gate(mismatch, off)["passed"])

    def test_missing_quality_never_becomes_GO(self):
        arm = dict(complete=True, performance_verified=True,
            timing=dict(active_ms_per_token=80, raw_ms_per_token=100),
            quality=quality_gates({}), repetition_vs_off=dict(passed=True),
            failures=dict(no_load_failures=True))
        baseline = dict(complete=True, performance_verified=True,
                        timing=dict(active_ms_per_token=120, raw_ms_per_token=150))
        self.assertEqual(c_decision(arm, baseline)["verdict"], "PENDING")
        evidence = dict(off_nll=.622784, nll=.623711, mmlu_n=57, mmlu_correct=48,
            chinese64_no_loop=True, long512_no_loop=True, long512_repeat_vs_off=True,
            decode_baseline=True, longctx_baseline=True)
        arm["quality"] = quality_gates(evidence)
        self.assertTrue(c_decision(arm, baseline)["verdict"].startswith("GO:"))
        arm["quality"] = quality_gates(evidence | dict(mmlu_correct=47))
        self.assertTrue(c_decision(arm, baseline)["verdict"].startswith("NO-GO:"))
        self.assertFalse(quality_gates(evidence | dict(nll=-1))["passed"])
        arm["quality"] = quality_gates(evidence)
        arm["timing"]["active_ms_per_token"] = 105
        self.assertIn("required 20%", c_decision(arm, baseline)["verdict"])
        baseline["complete"] = False
        self.assertEqual(c_decision(arm, baseline)["verdict"], "PENDING")

    def test_failure_counter_deltas_do_not_blame_previous_arm(self):
        old = ZERO_STATUS | dict(store="failed fills 2")
        self.assertTrue(failure_report(old, old)["no_load_failures"])
        changed = ZERO_STATUS | dict(store="failed fills 3")
        self.assertFalse(failure_report(old, changed)["no_load_failures"])
        self.assertIsNone(failure_report({}, {})["no_load_failures"])
        self.assertFalse(failure_report({}, old)["no_load_failures"])

    def test_same_engine_thermal_intervals_exclude_prefill(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            thermal = root / "thermal.jsonl"
            thermal.write_text("".join(json.dumps(dict(wall_time_s=t, paused=p)) + "\n"
                for t, p in [(101, True), (105, False), (113, True), (117, False),
                             (127, True), (129, False)]))
            save(root / "clock.json", dict(host_start_unix=100))
            arm = root / "off"
            arm.mkdir()
            turns = [dict(event="done", generated=11, decode_steps=10, decode_ms=10000,
                          host_s=20, label="first", seed=1),
                     dict(event="done", generated=6, decode_steps=5, decode_ms=5000,
                          host_unix=130, label="second", seed=2)]
            events = []
            for index, turn in enumerate(turns):
                events.extend(dict(event="token", id=i + index * 100) for i in range(turn["generated"]))
                events.append(turn)
            (arm / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events))
            save(arm / "turns.json", dict(turns=turns))
            save(arm / "status.json", ZERO_STATUS)
            record = dict(name="off", pid=77, session="default", turns=2,
                policy=dict(mode="off"), start_power=dict(power_profile="performance"),
                end_power=dict(power_profile="performance"),
                decode_timing=dict(thermal_log=str(thermal)))
            save(root / "comparison.json", dict(engine_pid=77, order=["off", "missing"], results=[record]))
            report = build_report([str(root)], expected_turns=2)
            cell = report["arms"][0]
            self.assertTrue(cell["complete"])
            self.assertEqual(cell["timing"]["raw_decode_ms"], 15000)
            self.assertEqual(cell["timing"]["thermal_pause_ms"], 6000)
            self.assertEqual(cell["timing"]["active_ms_per_token"], 600)
            self.assertEqual([r["timing"]["thermal_pause_ms"] for r in cell["turns"]], [4000, 2000])
            self.assertTrue(cell["repetition_vs_off"]["passed"])
            self.assertFalse(report["arms"][1]["complete"])
            self.assertIsNone(report["D_selection"]["selected"])

    def test_standalone_web_turns_and_cumulative_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            turns = [dict(label=str(i), seed=i, generated=8, decode_steps=7,
                          decode_ms=700, token_ids=list(range(8))) for i in range(3)]
            save(root / "turns.json", turns)
            save(root / "config.json", dict(power_profile="performance",
                ready=dict(speculation=dict(enabled=True, draft_tokens=2))))
            save(root / "turn2_status.json", ZERO_STATUS)
            report = build_report(["off=" + str(root)])
            arm = report["arms"][0]
            self.assertTrue(arm["complete"])
            self.assertEqual(arm["spec_k"], 2)
            self.assertEqual(arm["timing"]["raw_ms_per_token"], 100)
            self.assertIsNone(arm["timing"]["active_ms_per_token"])
            self.assertTrue(arm["failures"]["no_load_failures"])

    def test_D_selects_mask_speculation_and_preserves_missing_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            thermal = root / "thermal.jsonl"
            thermal.write_text(json.dumps(dict(wall_time_s=1, paused=False)) + "\n")
            records = []
            for index, (name, k, mode, cost) in enumerate([
                    ("off", 0, "off", 1000), ("k2", 2, "mask", 700),
                    ("k3", 3, "mask", 600), ("exact_spec", 2, "off", 100)]):
                arm = root / name
                arm.mkdir()
                turn = dict(event="done", label="matched", seed=1, generated=8,
                            decode_steps=7, decode_ms=cost, host_unix=10 + index)
                if k:
                    turn["speculation"] = dict(cycles=3, verified=k * 3, accepted=4,
                        tokens=7, draft_ms=cost / 4, verify_ms=cost / 2,
                        commit_ms=cost / 16, cpu_ms=cost / 16)
                events = [dict(event="token", id=i) for i in range(8)] + [turn]
                (arm / "events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in events))
                save(arm / "turns.json", dict(turns=[turn]))
                save(arm / "startup.json", dict(command=["--resident-only", mode,
                                                        "--mask-cache", "dynamic"]))
                status = ZERO_STATUS | dict(cache_fixed=False, cache_frozen=False)
                records.append(dict(name=name, turns=1, policy=dict(draft_tokens=k),
                    start_power=dict(power_profile="performance"),
                    end_power=dict(power_profile="performance"),
                    status_before=status, status_after=status,
                    decode_timing=dict(thermal_log=str(thermal))))
            save(root / "comparison.json", dict(order=[r["name"] for r in records], results=records))
            report = build_report([str(root)])
            self.assertEqual(report["D_selection"]["selected"], "k3")
            self.assertIsNone(report["D_selection"]["selected_quality"]["passed"])

    def test_cli_refuses_existing_output_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "summary.json"
            target.write_text("keep this receipt")
            process = subprocess.run([sys.executable, str(ROOT / "bench/mask_quality_report.py"),
                "/does/not/exist", "--out", str(target)], capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("refusing to replace", process.stderr)
            self.assertEqual(target.read_text(), "keep this receipt")


if __name__ == "__main__":
    unittest.main()
