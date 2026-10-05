#!/usr/bin/env python3
"""Summarise existing long-test results. No GPU work or inferred speedups."""
import argparse
from collections import Counter
import json
from pathlib import Path


def summarize(directory):
    turns = json.loads((directory / "turns.json").read_text())
    rows = []
    for turn in turns:
        ids, costs = turn["token_ids"], turn["token_costs_ms"][1:]
        periods = [p for p in range(1, 33) if len(ids) >= 128 and
            all(ids[-128:][i] == ids[-128:][i % p] for i in range(128))]
        grams = Counter(tuple(ids[i:i+4]) for i in range(len(ids)-3))
        quarters = []
        for q in range(4):
            values = costs[len(costs)*q//4:len(costs)*(q+1)//4]
            quarters.append(1000*len(values)/sum(values) if values and sum(values) else None)
        spec = turn["speculation"]
        cycles = spec["cycles"]
        row = {key: turn[key] for key in ("label", "generated", "decode_steps", "tok_s",
            "decode_ms", "prefill_ms", "prefill_tokens", "reused_tokens", "context", "finish", "total_ms")}
        row.update(decode_hit_rate=turn["decode_hit_rate"], decode_nvme_mb=turn["decode_nvme_mb"],
            effective_quarter_tok_s=quarters, tail128_exact_period=min(periods) if periods else None,
            repeated_fourgram_fraction=1-len(grams)/sum(grams.values()) if grams else 0,
            content_chars=len(turn.get("content", "")), reasoning_chars=len(turn.get("reasoning", "")),
            text_tail=(turn.get("content") or turn.get("reasoning", ""))[-320:],
            speculation=spec, target_submits=round(turn["per_token_ms"]["submits"]*turn["decode_steps"]),
            target_forwards=cycles if cycles else turn["decode_steps"])
        if cycles:
            row.update(accepted_per_verified=spec["accepted"]/spec["verified"] if spec["verified"] else 0,
                outputs_per_cycle=spec["tokens"]/cycles,
                cycle_ms=sum(spec[k] for k in ("draft_ms", "verify_ms", "commit_ms", "cpu_ms"))/cycles,
                cycle_components_ms={k:spec[k]/cycles for k in ("draft_ms", "verify_ms", "commit_ms", "cpu_ms")})
        rows.append(row)
    steps = sum(row["decode_steps"] for row in rows)
    decode_ms = sum(row["decode_ms"] for row in rows)
    totals = {key:sum(turn["speculation"][key] for turn in turns) for key in turns[0]["speculation"]}
    total_ms = sum(turn["total_ms"] for turn in turns)
    generated = sum(row["generated"] for row in rows)
    return {"generated":sum(row["generated"] for row in rows), "decode_steps":steps,
        "decode_ms":decode_ms, "weighted_tok_s":1000*steps/decode_ms,
        "turn_total_ms":total_ms, "including_prefill_tok_s":1000*generated/total_ms,
        "prefill_ms":sum(row["prefill_ms"] for row in rows), "turns":rows, "speculation":totals}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    cells = {mode:summarize(root/mode) for mode in ("plain", "spec5")}
    checks = json.loads((root/"check_results.json").read_text())
    assert all(check["rc"] == 0 and check["mirror_holds"] == 48 and
        not check["thermal_pauses"] and not check["ac_changed"] for check in checks)
    result = {"cells":cells, "thermal_checks":checks,
        "observed_tok_s_ratio":cells["spec5"]["weighted_tok_s"]/cells["plain"]["weighted_tok_s"],
        "observed_including_prefill_ratio":cells["spec5"]["including_prefill_tok_s"]/cells["plain"]["including_prefill_tok_s"],
        "limits":["Same prompts/settings, different generated histories and resident routes.",
            "Plain has no draft pins; spec5 pins 384 MTP experts within 5500 total slots.",
            "No per-op trace or cycle logging; draft GPU timestamps off.",
            "Quarter rates derive from per-output costs; speculative bursts share a cycle's cost.",
            "Tail period and repeated ngrams detect simple loops, not semantic correctness.",
            "Long generation here does not establish 4K/17K input quality or a full MMLU score."]}
    (root/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"plain_tok_s":cells["plain"]["weighted_tok_s"],
        "spec5_tok_s":cells["spec5"]["weighted_tok_s"],
        "ratio":result["observed_tok_s_ratio"]}))


if __name__ == "__main__":
    main()
