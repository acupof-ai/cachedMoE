#!/usr/bin/env python3
"""Paired zero-shot four-choice logit MMLU sample over serve, serial GPU arms.

Input JSON: {"rows": [{"question", "choices", "answer", "subject", ...}]}.
Exact GPU prefill of all but the final prompt token, then extend by that token
through single-position forward so the answer distribution exercises miss mask.
Expert cache stays warm between questions; KV resets. This is a zero-shot sample, not the canonical full 5-shot score.
"""
import argparse
import json
import math
import re
from pathlib import Path
import time

import chat
import hitrate_bench
import provenance


def request(server, ids, n):
    server.send({"op": "generate", "prompt_ids": ids, "max_tokens": n,
                 "temperature": 0, "seed": 42, "stop_ids": [1], "reuse": True})
    pieces = []
    emitted_ids = []
    while True:
        ev = server.read_event()
        if ev['event'] == 'error':
            raise RuntimeError(ev)
        if ev['event'] == 'token':
            pieces.append(ev['text'])
            emitted_ids.append(ev['id'])
        if ev['event'] == 'done':
            return ''.join(pieces), dict(ev, emitted_ids=emitted_ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sample', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--modes', default='off,mask')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--require-sources', type=int, default=0)
    ap.add_argument('--generation', action='store_true',
                    help='generate Answer: X, so speculative acceptance participates in the answer')
    ap.add_argument('--max-tokens', type=int, default=16)
    ap.add_argument('--spec-k', type=int, default=2)
    ap.add_argument('--spec-top-k', type=int, default=4)
    ap.add_argument('--cache-slots', type=int, default=0)
    ap.add_argument('--exe', default=str(Path(chat.REPO) / 'build/cachedmoe'),
                    help='engine from the isolated worktree under evaluation')
    args = ap.parse_args()
    sample = json.loads(Path(args.sample).read_text())
    rows = sample['rows'][:args.limit or None]
    enc = chat.load_encoding()
    summaries = []
    for mode in args.modes.split(','):
        speculative = mode.endswith('-spec')
        resident_mode = mode.removesuffix('-spec')
        if speculative and not args.generation:
            ap.error('use --generation for speculation; scoring the first answer token bypasses it')
        out = Path(args.out) / mode
        out.mkdir(parents=True, exist_ok=True)
        opts = argparse.Namespace(exe=args.exe,
            max_context=4096, cache_gb=0, cache_slots=args.cache_slots, shader_dir='', env=[], require_sources=args.require_sources,
            serve_arg=['--resident-only', resident_mode, '--gpu-prefill-min', '1',
                       '--gpu-prefill-speedup', '0'])
        if speculative:
            opts.serve_arg += ['--dspark', '--spec-k', str(args.spec_k), '--spec-top-k', str(args.spec_top_k)]
        server = hitrate_bench.BenchServer(opts, str(out))
        server.send({'op': 'score_tokens', 'token_ids': [0]})
        if server.read_event().get('event') != 'error':
            server.close()
            raise RuntimeError('score_tokens returned stale prefill logits')
        choice_ids = [server.tokenize(c) for c in 'ABCD']
        if any(len(ids) != 1 for ids in choice_ids):
            server.close()
            raise RuntimeError('each answer letter must tokenize to exactly one token')
        choice_ids = [ids[0] for ids in choice_ids]
        correct = 0
        invalid = cycles = accepted = verified = 0
        started = time.monotonic()
        try:
            with (out / 'answers.jsonl').open('w') as answers:
                for i, row in enumerate(rows):
                    server.send({'op': 'reset'})
                    ev = server.read_event()
                    if ev['event'] != 'reset':
                        raise RuntimeError(ev)
                    instruction = ("Reply exactly in the format 'Answer: X', where X is A, B, C, or D. Do not add an explanation."
                                   if args.generation else "Answer with only one letter: A, B, C, or D.")
                    q = (f"The following is a multiple choice question about {row['subject'].replace('_', ' ')}.\n" +
                         instruction + '\n\n' + row['question'] + '\n' +
                         '\n'.join(f'{c}. {v}' for c, v in zip('ABCD', row['choices'])))
                    prompt = enc.encode_messages([{'role': 'user', 'content': q}],
                                                  thinking_mode='chat', drop_thinking=True)
                    ids = server.tokenize(prompt)
                    _, prefix = request(server, ids[:-1], 1)
                    if prefix.get('prefill_mode') != 'gpu':
                        raise RuntimeError('MMLU prefix was not prefetched exactly on the GPU')
                    text, done = request(server, ids, args.max_tokens if args.generation else 1)
                    if done.get('reused_tokens') != len(ids) - 1 or done.get('prefill_tokens') != 1:
                        raise RuntimeError('final prompt token did not use the single-position path')
                    if args.generation:
                        match = re.match(r'\s*Answer:\s*([ABCD])\b', text, re.IGNORECASE)
                        pred = ord(match[1].upper()) - ord('A') if match else None
                        invalid += pred is None
                        correct += pred == row['answer']
                        spec = done.get('speculation', {})
                        cycles += spec.get('cycles', 0)
                        accepted += spec.get('accepted', 0)
                        verified += spec.get('verified', 0)
                        rec = dict(row, prediction=pred, output=text, correct=pred == row['answer'],
                                   prompt_tokens=len(ids), prefix=prefix, done=done)
                        answers.write(json.dumps(rec, ensure_ascii=False) + '\n'); answers.flush()
                        status = hitrate_bench.capture_status(server, str(out))
                        if status.get('event') != 'status' or (args.require_sources and 'DROPPED' in status.get('io', '')):
                            raise RuntimeError('source check failed during generated MMLU')
                        print(f'{mode} {i+1}/{len(rows)} accuracy={correct/(i+1):.3f} invalid={invalid} cycles={cycles}', flush=True)
                        continue
                    scored_ids = choice_ids + [done['emitted_ids'][0]]
                    server.send({'op': 'score_tokens', 'token_ids': scored_ids})
                    scores = server.read_event()
                    if scores.get('event') != 'scores' or scores.get('token_ids') != scored_ids:
                        raise RuntimeError(scores)
                    values = scores['logits'][:4]
                    if scores['logits'][-1] + 1e-5 < max(values):
                        raise RuntimeError('scores do not match the greedy sample: stale logits')
                    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
                        raise RuntimeError('non-finite answer score')
                    pred = max(range(4), key=values.__getitem__)
                    correct += pred == row['answer']
                    rec = dict(row, prediction=pred, output=text, correct=pred == row['answer'],
                               prompt_tokens=len(ids), prefix=prefix, done=done,
                               choice_ids=choice_ids, choice_logits=values)
                    answers.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    answers.flush()
                    if done.get('reused_tokens') != len(ids) - 1 or done.get('prefill_tokens') != 1:
                        raise RuntimeError('final prompt token did not use the single-position path')
                    status = hitrate_bench.capture_status(server, str(out))
                    if status.get('event') != 'status' or (args.require_sources and 'DROPPED' in status.get('io', '')):
                        raise RuntimeError('source check failed during MMLU')
                    print(f'{mode} {i+1}/{len(rows)} accuracy={correct/(i+1):.3f}', flush=True)
        finally:
            server.close()
            server.events.close()
            server.log.close()
        if speculative and not cycles:
            raise RuntimeError('no speculative verification participated in the generated sample')
        summary = dict(mode=mode, n=len(rows), correct=correct, accuracy=correct / len(rows), invalid=invalid,
                       speculative_cycles=cycles, accepted=accepted, verified=verified,
                       seconds=time.monotonic()-started,
                       sources=server.ready.get('sources'), cache_slots=server.ready.get('cache_slots'),
                       protocol=('zero-shot generated Answer: X; invalid format counts as incorrect; exact prefix + single-position final prompt token'
                                 if args.generation else 'zero-shot argmax of A/B/C/D logits; exact prefix + single-position final prompt token'),
                       sample_source=sample.get('source'), sample_sha256=sample.get('test_sha256'))
        provenance.finish(str(out))
        (out / 'summary.json').write_text(json.dumps(summary, indent=2))
        summaries.append(summary)
        Path(args.out, 'summary.json').write_text(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
