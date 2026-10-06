#!/usr/bin/env python3
"""Owner TODO 4.9: draft-only FP8/subset screening on a captured native path.

CPU top-K arithmetic is approximate; native BF16 agreement gates its use.
Target top-4 rows belong to the original path. Counterfactual acceptance is a
fixed-trajectory estimate, not a proof for contexts changed by a new proposal.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


def frequency_subset(tokens, vocab, size):
    counts = Counter(int(t) for t in tokens)
    if not 0 < size <= vocab or any(t < 0 or t >= vocab for t in counts):
        raise ValueError("invalid corpus IDs or subset size")
    # Unseen tokens have equal zero frequency; lowest ID is the declared tie rule.
    return np.array(sorted(range(vocab), key=lambda t: (-counts[t], t))[:size], dtype=np.int64)


def top_ids(scores, ids, count=4):
    if not np.isfinite(scores).all() or len(scores) < count:
        raise ValueError("nonfinite logits or insufficient candidates")
    return ids[np.lexsort((ids, -scores))[:count]]


def chain(base, embed, markov, roots, candidates):
    """Markov bias follows each candidate's own previous top-1, including FP8."""
    paths, tops = [], []
    for cycle, root in enumerate(roots):
        previous, path, rows = int(root), [], []
        for position in range(base.shape[1]):
            scores = base[cycle, position, candidates] + markov[candidates] @ embed[previous]
            top = top_ids(scores, candidates)
            previous = int(top[0])
            path.append(previous)
            rows.append(top.tolist())
        paths.append(path)
        tops.append(rows)
    return np.array(paths), np.array(tops)


def accepted_prefix(path, target_top4):
    accepted = 0
    for token, ids in zip(path, target_top4):
        if int(token) not in ids:
            break
        accepted += 1
    return accepted


def validate_capture(meta, cycles, hidden):
    """Reject partial or misaligned native data before reading any head weights."""
    if (meta.get("schema") != 1 or meta.get("sampling") != "greedy" or
            meta.get("outputs") != 64 or not cycles):
        raise ValueError("require a complete 64-output greedy capture, schema 1")
    if (meta["dim"] <= 0 or meta["block_rows"] < 2 or meta["vocab"] < 4 or
            meta["slot_bytes"] <= 0 or
            hidden.ndim != 3 or hidden.shape[1:] != (meta["block_rows"], meta["dim"]) or
            not np.isfinite(hidden).all()):
        raise ValueError("invalid hidden geometry or nonfinite activation")
    generated, previous = 1, meta["first_output"]
    position, hidden_count = cycles[0]["position"], 0
    for c in cycles:
        k, accepted = c["k"], c["accepted"]
        if (k != min(2, 63 - generated) or not 0 <= accepted <= k or
                c["position"] != position or c["root"] != previous or
                len(c["proposal"]) != k or len(c["target_greedy"]) != k + 1 or
                len(c["target_top4"]) != (k + 1) * 4 or
                len(c["emitted"]) != accepted + 1):
            raise ValueError("incomplete or inconsistent native cycle")
        all_ids = [c["root"]] + c["proposal"] + c["target_greedy"] + c["target_top4"] + c["emitted"]
        if any(not isinstance(i, int) or not 0 <= i < meta["vocab"] for i in all_ids):
            raise ValueError("native token ID outside vocabulary")
        target = np.array(c["target_top4"]).reshape(-1, 4)
        if any(len(set(row)) != 4 or row[0] != greedy
               for row, greedy in zip(target, c["target_greedy"])):
            raise ValueError("target rank rows disagree with recorded greedy IDs")
        if (accepted_prefix(c["proposal"], target) != accepted or
                c["emitted"] != c["proposal"][:accepted] + [c["target_greedy"][accepted]]):
            raise ValueError("native acceptance or emitted prefix is inconsistent")
        if c["hidden_index"] != (hidden_count if k else -1):
            raise ValueError("native hidden block is missing or misaligned")
        hidden_count += bool(k)
        generated += len(c["emitted"])
        position += len(c["emitted"])
        previous = c["emitted"][-1]
    if generated != meta["outputs"] or hidden_count != len(hidden):
        raise ValueError("incomplete native capture")


def quantize_rows(weights):
    """CPU reference: E4M3FN RNE, independent float32 amax/448 row scales."""
    import torch
    weights = weights.float()
    if weights.ndim != 2 or not torch.isfinite(weights).all():
        raise ValueError("require finite two-dimensional head weights")
    amax = weights.abs().amax(dim=1)
    scale = torch.where(amax > 0, amax / 448, torch.ones_like(amax))
    quantized = (weights / scale[:, None]).to(torch.float8_e4m3fn)
    return quantized, scale, quantized.float() * scale[:, None]


def candidate_metrics(paths, tops, baseline_paths, baseline_tops, cycles, candidates):
    verified = sum(c["k"] for c in cycles)
    accepted = sum(accepted_prefix(p[:c["k"]], np.array(c["target_top4"]).reshape(-1, 4))
                   for p, c in zip(paths, cycles))
    baseline_accepted = sum(c["accepted"] for c in cycles)
    subset = set(int(i) for i in candidates)
    targets = [i for c in cycles for i in c["target_greedy"][:c["k"]]]
    # Changed accepted IDs already prove output identity is at risk on this path.
    changed_accepted = sum(int(p[j]) != c["proposal"][j]
                           for p, c in zip(paths, cycles) for j in range(c["accepted"]))
    valid = np.arange(paths.shape[1])[None, :] < np.array([c["k"] for c in cycles])[:, None]
    return dict(top1_agreement=float(np.mean((paths == baseline_paths)[valid])),
                top4_set_agreement=float(np.mean(np.all(np.sort(tops, axis=-1) ==
                                                        np.sort(baseline_tops, axis=-1), axis=-1)[valid])),
                top4_rows_identical=float(np.mean(np.all(tops == baseline_tops, axis=-1)[valid])),
                verified=verified, accepted_fixed_trajectory=accepted,
                acceptance_fixed_trajectory=accepted / verified,
                baseline_native_acceptance=baseline_accepted / verified,
                acceptance_drop_pp=100 * (baseline_accepted - accepted) / verified,
                target_greedy_match=float(np.mean([int(p[j]) == c["target_greedy"][j]
                                                   for p, c in zip(paths, cycles)
                                                   for j in range(min(c["k"], len(c["target_greedy"])))])),
                recorded_greedy_targets_outside_subset=sum(t not in subset for t in targets),
                recorded_greedy_target_count=len(targets),
                changed_native_accepted_ids=changed_accepted)


def gain(head_ms, source_bytes, candidate_bytes):
    if not math.isfinite(head_ms) or head_ms <= 0 or source_bytes <= 0 or not 0 <= candidate_bytes <= source_bytes:
        raise ValueError("invalid bandwidth estimate")
    saved = head_ms * (1 - candidate_bytes / source_bytes)
    return dict(source_bytes=source_bytes, candidate_read_bytes=candidate_bytes,
                measured_effective_gbps=source_bytes / (head_ms * 1e6),
                predicted_saved_ms=saved, halved_saved_ms=saved / 2,
                gain_gate_pass=saved / 2 >= 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mirror", type=Path, action="append", default=[], help="read-only checkpoint roots")
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--corpus-ids", type=Path, required=True, help="independent corpus, uint32 LE")
    parser.add_argument("--head-ms", type=float, required=True, help="unprofiled head cost, ms/cycle")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be positive")
    gain(args.head_ms, 1, 1)  # Validate the supplied cost before loading tensors.
    # No implicit GPU libraries or oracle model import; WeightStore only reads.
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from dsref import WeightStore
    torch.set_num_threads(args.threads)
    if args.out.exists() or any(args.out.resolve().is_relative_to(p.resolve())
                                for p in [args.model] + args.mirror):
        raise ValueError("new output directory must be outside the checkpoint")
    lines = [json.loads(l) for l in (args.capture / "cycles.jsonl").read_text().splitlines()]
    if not lines:
        raise ValueError("empty native capture")
    meta = lines[0]
    cycles = [c for c in lines[1:] if c["k"] > 0]
    if (args.capture / "hidden.f32").stat().st_size % 4:
        raise ValueError("hidden file is not aligned to float32 elements")
    hidden = np.fromfile(args.capture / "hidden.f32", dtype="<f4").reshape(-1, meta["block_rows"], meta["dim"])
    validate_capture(meta, lines[1:], hidden)
    if args.corpus_ids.stat().st_size % 4:
        raise ValueError("frequency corpus is not aligned to uint32 elements")
    corpus = np.fromfile(args.corpus_ids, dtype="<u4")
    if not len(corpus) or np.any(corpus >= meta["vocab"]):
        raise ValueError("empty frequency corpus or ID outside vocabulary")
    x = np.stack([hidden[c["hidden_index"], :2] for c in cycles])
    roots = [c["root"] for c in cycles]
    config_path = args.model / "config.json"
    text_config = json.loads(config_path.read_text())["text_config"]
    final_stage = text_config["num_nextn_predict_layers"] - 1
    if final_stage < 0:
        raise ValueError("checkpoint has no draft layers")
    markov_prefix = f"mtp.{final_stage}.markov_head."
    with_store = WeightStore(str(args.model), None)
    try:
        head = with_store.tensor("head.weight")
        embed = with_store.tensor(markov_prefix + "embed.weight").float().numpy()
        markov = with_store.tensor(markov_prefix + "head.weight").float().numpy()
    finally:
        with_store.close()
    vocab, dim = head.shape
    if (vocab, dim) != (meta["vocab"], meta["dim"]):
        raise ValueError("head and capture geometry mismatch")
    if (head.dtype != torch.bfloat16 or
            embed.shape != (vocab, text_config["dspark_markov_rank"]) or markov.shape != embed.shape):
        raise ValueError("unexpected head dtype or Markov geometry")
    # mgt1_head stages the RMSNorm input as BF16 in LDS, even though the
    # captured scratch is float32. Preserve that existing activation rounding.
    flat = torch.from_numpy(x.reshape(-1, dim)).to(torch.bfloat16).float()
    base = np.empty((len(flat), vocab), dtype=np.float32)
    fp8_base = np.empty_like(base)
    original_hash, fp8_hash, scale_hash = (hashlib.sha256() for _ in range(3))
    for begin in range(0, vocab, 512):
        weights = head[begin:begin + 512].float()
        original_hash.update(head[begin:begin + 512].view(torch.uint8).numpy().tobytes())
        quantized, scale, dequantized = quantize_rows(weights)
        fp8_hash.update(quantized.view(torch.uint8).numpy().tobytes())
        scale_hash.update(scale.numpy().tobytes())
        base[:, begin:begin + len(weights)] = (flat @ weights.T).numpy()
        fp8_base[:, begin:begin + len(weights)] = (flat @ dequantized.T).numpy()
    base = base.reshape(len(cycles), 2, vocab)
    fp8_base = fp8_base.reshape(len(cycles), 2, vocab)
    full = np.arange(vocab)
    paths, tops = chain(base, embed, markov, roots, full)
    native = np.array([c["proposal"] + [-1] * (2 - c["k"]) for c in cycles])
    valid = native >= 0
    native_agreement = float(np.mean(paths[valid] == native[valid]))
    source_bytes = vocab * dim * 2
    args.out.mkdir(parents=True)
    results = {}
    for name, scores, ids in [("fp8", fp8_base, full)] + [
            (f"subset_{size}", base, frequency_subset(corpus, vocab, size))
            for size in (16384, 32768, 65536)]:
        candidate_paths, candidate_tops = chain(scores, embed, markov, roots, ids)
        metrics = candidate_metrics(candidate_paths, candidate_tops, paths, tops, cycles, ids)
        read_bytes = vocab * (dim + 4) if name == "fp8" else len(ids) * (dim * 2 + 4)
        estimate = gain(args.head_ms, source_bytes, read_bytes)
        eligible = native_agreement == 1 and estimate["gain_gate_pass"] and metrics["acceptance_drop_pp"] <= 3
        results[name] = dict(metrics=metrics, gain=estimate,
                             derived_copy_bytes=read_bytes,
                             slot_equivalents=read_bytes / meta["slot_bytes"],
                             gpu_screen_eligible=eligible,
                             final_output_identity_unproven=True)
        np.save(args.out / (name + "_path.npy"), candidate_paths)
        if name != "fp8":
            np.save(args.out / (name + "_vocab.npy"), ids)
    result = dict(native_bf16_top1_agreement=native_agreement, cycles=len(cycles),
                  capture=dict(path=str(args.capture),
                               cycles_sha256=hashlib.sha256((args.capture / "cycles.jsonl").read_bytes()).hexdigest(),
                               hidden_sha256=hashlib.sha256((args.capture / "hidden.f32").read_bytes()).hexdigest()),
                  checkpoint_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                  markov_tensor_prefix=markov_prefix,
                  head_ms=args.head_ms, head_raw_sha256=original_hash.hexdigest(),
                  fp8_encoded_sha256=fp8_hash.hexdigest(), fp8_scale_sha256=scale_hash.hexdigest(),
                  quantization="per-row amax/448, E4M3FN RNE, float32 scales, target weights unchanged",
                  corpus=dict(path=str(args.corpus_ids), sha256=hashlib.sha256(args.corpus_ids.read_bytes()).hexdigest(),
                              tokens=len(corpus), unique=len(np.unique(corpus)),
                              unseen_tie="token ID ascending; corpus may be unrepresentative"),
                  agreement_scope="only actually verified draft positions; k=0 and padding excluded",
                  acceptance_scope="counterfactual on native verify rows; changed contexts need GPU validation",
                  candidates=results)
    (args.out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
