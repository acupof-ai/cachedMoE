#!/usr/bin/env python3
"""Track MS: system throughput for N concurrent conversations (docs/p4_multistream.md).

One `deepmoe serve` process, N decode streams inside it, one chat script per stream.
Turn i of every script is sent as ONE `generate_multi` request, so the turns decode
together -- layer-interleaved, so one stream's NVMe stall is another's GPU compute.

    .venv/Scripts/python.exe tools/ms_bench.py --out bench/results/ms/two \
        --script bench/results/hitrate/y_turns.json \
        --script bench/results/hitrate/long_turns.json \
        --cache-slots 5100 --warm-cache

With one --script it is the single-stream baseline over the same code path
(`generate_multi` with one request falls through to `decode_step`).

The out directory gets events.jsonl, profile.jsonl, serve.log, turns.json and
summary.json; `--sched pingpong` runs design D2 (whole token of A, then of B) as
the control.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import chat  # noqa: E402


class Server:
    """chat.Server's protocol, with a free-form command line and an event log."""

    def __init__(self, args, out_dir, n_streams):
        cmd = [args.exe, "serve", "--model", chat.MODEL,
               "--max-context", str(args.max_context),
               "--engram-tables", os.path.join(REPO, "tests", "data", "l3"),
               "--profile", os.path.join(out_dir, "profile.jsonl"),
               "--streams", str(n_streams), "--no-kv-disk"]
        if args.cache_slots:
            cmd += ["--cache-slots", str(args.cache_slots)]
        elif args.cache_gb:
            cmd += ["--cache-gb", str(args.cache_gb)]
        if args.warm_cache:
            cmd += ["--warm-cache"]
        if args.sched in ("pipeline", "interleave", "pingpong"):
            cmd += ["--ms-sched", args.sched]
        cmd += args.serve_arg
        env = dict(os.environ)
        for kv in args.env:
            k, v = kv.split("=", 1)
            env[k] = v
        if args.route_dump:
            env["DEEPMOE_ROUTE_DUMP"] = os.path.join(out_dir, "route.bin")
        for f in ("profile.jsonl", "route.bin", "events.jsonl"):
            p = os.path.join(out_dir, f)
            if os.path.exists(p):
                os.remove(p)
        self.events = open(os.path.join(out_dir, "events.jsonl"), "w",
                           encoding="utf-8", newline="\n")
        self.log = open(os.path.join(out_dir, "serve.log"), "wb")
        self.t0 = time.time()
        self.p = subprocess.Popen(cmd, cwd=REPO, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=self.log, bufsize=0, env=env)
        self.cmd = cmd
        self.ready = self.read_event()
        if self.ready.get("event") != "ready":
            raise SystemExit(f"server did not start: {self.ready}")

    def send(self, obj):
        self.p.stdin.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
        self.p.stdin.flush()

    def read_event(self):
        line = self.p.stdout.readline()
        if not line:
            raise SystemExit("server exited (see serve.log)")
        ev = json.loads(line.decode("utf-8"))
        rec = dict(ev)
        rec["host_s"] = round(time.time() - self.t0, 4)
        self.events.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.events.flush()
        return ev

    def tokenize(self, text):
        self.send({"op": "tokenize", "text": text})
        while True:
            ev = self.read_event()
            if ev.get("event") == "tokens":
                return ev["ids"]
            if ev.get("event") == "error":
                raise SystemExit(ev["message"])

    def detokenize(self, ids):
        self.send({"op": "detokenize", "ids": list(ids)})
        while True:
            ev = self.read_event()
            if ev.get("event") == "text":
                return ev["text"]
            if ev.get("event") == "error":
                raise SystemExit(ev["message"])

    def close(self):
        try:
            self.send({"op": "quit"})
            self.p.wait(timeout=180)
        except Exception:
            self.p.kill()


class Conversation:
    """One script's rolling chat state: the same rendering chat.py uses."""

    def __init__(self, server, enc, script, name):
        self.s, self.enc, self.script, self.name = server, enc, script, name
        self.messages = []
        self.ctx_ids, self.ctx_text = [], ""
        self.turn_stats = []

    def prompt_for(self, turn, max_tokens=0, no_stop=False):
        self.messages.append({"role": "user", "content": turn["user"]})
        full = self.enc.encode_messages(self.messages, thinking_mode="chat", drop_thinking=True)
        if self.ctx_ids and full.startswith(self.ctx_text):
            ids = self.ctx_ids + self.s.tokenize(full[len(self.ctx_text):])
        else:
            ids = self.s.tokenize(full)
        self.pending_prompt = (ids, full)
        req = {"prompt_ids": ids,
               "max_tokens": max_tokens or int(turn.get("max_tokens", 256)),
               "temperature": float(turn.get("temperature", 1.0)),
               "top_p": float(turn.get("top_p", 0.95)),
               "seed": int(turn.get("seed", 0)),
               "stop_ids": [] if no_stop else [1]}
        return req

    def finish(self, ids):
        """Fold the reply back in, exactly as chat.py does after a turn."""
        text = self.s.detokenize(ids)
        self.messages.append({"role": "assistant", "content": text})
        prompt_ids, prompt_text = self.pending_prompt
        self.ctx_ids = prompt_ids + list(ids)[:-1] if ids else prompt_ids
        self.ctx_text = prompt_text + text
        return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", action="append", required=True,
                    help="one chat script per stream; repeat the flag")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exe", default=os.path.join(REPO, "build", "deepmoe.exe"))
    ap.add_argument("--cache-gb", type=int, default=0)
    ap.add_argument("--cache-slots", type=int, default=0)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--turns", type=int, default=0, help="only the first N turns of each script")
    ap.add_argument("--warm-cache", action="store_true")
    ap.add_argument("--route-dump", action="store_true")
    ap.add_argument("--sched", default="", choices=["", "pipeline", "interleave", "pingpong", "serial"],
                    help="interleave = D1 (layer-interleaved); pingpong = D2 (whole token "
                         "of A, then of B); serial = the baseline, one conversation at a "
                         "time over the same N streams, so only the SCHEDULE differs")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="override every turn's max_tokens, so the streams generate the "
                         "same number of tokens and the aggregate is not a long tail")
    ap.add_argument("--no-stop", action="store_true",
                    help="drop the stop id, so a turn always runs to max_tokens")
    ap.add_argument("--serve-arg", action="append", default=[])
    ap.add_argument("--env", action="append", default=[])
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    os.makedirs(args.out, exist_ok=True)
    scripts = [json.load(open(p, encoding="utf-8")) for p in args.script]
    n = len(scripts)
    enc = chat.load_encoding()
    server = Server(args, args.out, n)
    print("ready: " + json.dumps(server.ready), flush=True)
    convs = [Conversation(server, enc, sc, os.path.basename(p))
             for sc, p in zip(scripts, args.script)]
    n_turns = min(len(sc["turns"]) for sc in scripts)
    if args.turns:
        n_turns = min(n_turns, args.turns)

    rounds = []
    t_all = time.time()
    try:
        for ti in range(n_turns):
            reqs = []
            for si, c in enumerate(convs):
                r = c.prompt_for(c.script["turns"][ti], args.max_tokens, args.no_stop)
                r["stream"] = si
                reqs.append(r)
            t0 = time.time()
            toks = [[] for _ in range(n)]
            dones = [None] * n
            # `serial` is the baseline: the same N streams, the same KV stores
            # and the same shared expert cache, but one conversation decoding
            # at a time. Only the schedule differs from `interleave`.
            batches = [[r] for r in reqs] if args.sched == "serial" else [reqs]
            multi = {"decode_ms": 0.0, "decode_steps": 0, "rounds": 0, "full_rounds": 0}
            for batch in batches:
                server.send({"op": "generate_multi", "requests": batch})
                got = None
                while got is None:
                    ev = server.read_event()
                    e = ev.get("event")
                    if e == "token":
                        toks[int(ev.get("stream", 0))].append(ev["id"])
                    elif e == "done":
                        dones[int(ev.get("stream", 0))] = ev
                    elif e == "done_multi":
                        got = ev
                    elif e == "error":
                        raise SystemExit(ev["message"])
                multi["decode_ms"] += got["decode_ms"]
                multi["decode_steps"] += got["decode_steps"]
                multi["rounds"] += got["rounds"]
                multi["full_rounds"] += got["full_rounds"] if len(batch) == n else 0
            multi["aggregate_tok_s"] = (multi["decode_steps"] * 1e3 / multi["decode_ms"]
                                        if multi["decode_ms"] else 0.0)
            wall = time.time() - t0
            texts = [c.finish(toks[si]) for si, c in enumerate(convs)]
            row = {"turn": ti, "wall_s": wall, "multi": multi,
                   "streams": [dict(d, label=convs[i].script["turns"][ti].get("label", ""),
                                    script=convs[i].name) for i, d in enumerate(dones)]}
            rounds.append(row)
            agg = multi["aggregate_tok_s"]
            per = "  ".join(
                f"s{i}={d['tok_s']:.3f} hit {d['decode_hit_rate']:.4f} "
                f"stall {d['per_token_ms']['nvme_stall']:.0f} "
                f"gpu {d['per_token_ms']['attn'] + d['per_token_ms']['moe_gpu']:.0f}"
                for i, d in enumerate(dones))
            print(f"turn {ti}: wall {wall:6.1f}s  aggregate {agg:.3f} tok/s  "
                  f"rounds {multi['rounds']} (full {multi['full_rounds']})  {per}", flush=True)
            for si, t in enumerate(texts):
                open(os.path.join(args.out, f"transcript_{si}.md"), "a",
                     encoding="utf-8", newline="\n").write(
                         f"\n### turn {ti}\n**User:** {convs[si].script['turns'][ti]['user']}\n\n"
                         f"**Assistant:** {t}\n")
        server.send({"op": "status"})
        status = {}
        while True:
            ev = server.read_event()
            if ev.get("event") in ("status", "error"):
                status = ev
                break
    finally:
        server.close()
        server.events.close()

    steps = sum(d["decode_steps"] for r in rounds for d in r["streams"])
    ms = sum(r["multi"]["decode_ms"] for r in rounds)
    summary = {
        "cmd": server.cmd,
        "scripts": args.script,
        "streams": n,
        "sched": args.sched or "pipeline",
        "turns": n_turns,
        "decode_steps": steps,
        "decode_ms": ms,
        "aggregate_tok_s": steps * 1e3 / ms if ms else 0.0,
        "wall_s": time.time() - t_all,
        "per_stream": [],
        "status": status,
    }
    for si in range(n):
        st = [r["streams"][si] for r in rounds]
        stp = sum(d["decode_steps"] for d in st)
        dms = sum(d["decode_ms"] for d in st)
        if not stp:
            summary["per_stream"].append({"script": convs[si].name, "decode_steps": 0})
            continue
        w = lambda k: sum(d[k] * d["decode_steps"] for d in st) / stp
        wp = lambda k: sum(d["per_token_ms"][k] * d["decode_steps"] for d in st) / stp
        summary["per_stream"].append({
            "script": convs[si].name,
            "decode_steps": stp,
            "decode_ms": dms,
            "tok_s": stp * 1e3 / dms if dms else 0.0,
            "hit_rate": w("decode_hit_rate"),
            "nvme_mb_per_token": sum(d["decode_nvme_mb"] for d in st) / stp,
            "per_token_ms": {k: wp(k) for k in
                             ("attn", "moe_gpu", "moe_host", "nvme_stall", "engram",
                              "tail", "other", "fence_wait", "submits")},
        })
    with open(os.path.join(args.out, "turns.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"server": server.ready, "rounds": rounds}, f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: summary[k] for k in
                      ("streams", "sched", "decode_steps", "aggregate_tok_s", "per_stream")},
                     ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
