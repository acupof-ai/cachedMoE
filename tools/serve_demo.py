#!/usr/bin/env python3
"""Scripted `deepmoe serve` session exercising Track R2's serve UX (docs/p4_kv_ux.md §10):
KV continuation, rollback to a diverging prompt's common prefix, cancel mid-generation,
THREE named sessions with LRU parking (--max-parked 2, so coming back to the first one
has to restore it by replay), and the engram tables derived at startup.

Every step is timed and the run ends with a table the doc quotes: wall time, what the
turn reused, what it re-prefilled, and what the window replay cost.

    .venv/Scripts/python.exe tools/serve_demo.py [--exe build/deepmoe.exe] [--cache-gb 16]
        [--max-parked 2] [--log build/serve_demo.log] [--out events.jsonl]
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ.get("DEEPMOE_MODEL_DIR", r"D:\models\DeepSeek-V4.1-Flash")


class Server:
    def __init__(self, args):
        cmd = [args.exe, "serve", "--model", MODEL, "--max-context", "8192",
               "--max-parked", str(args.max_parked)]
        if args.cache_gb:
            cmd += ["--cache-gb", str(args.cache_gb)]
        if args.no_kv_disk:
            cmd += ["--no-kv-disk"]
        elif args.kv_dir:
            # A private, empty directory, so the run does not inherit a previous
            # one's parked sessions and the LRU spill is this run's.
            if os.path.isdir(args.kv_dir):
                for f in os.listdir(args.kv_dir):
                    os.remove(os.path.join(args.kv_dir, f))
            cmd += ["--kv-dir", args.kv_dir]
        self.log = open(args.log, "ab")
        self.out = open(args.out, "w", encoding="utf-8") if args.out else None
        self.p = subprocess.Popen(cmd, cwd=REPO, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self.log, bufsize=0)
        self.t0 = time.time()

    def send(self, obj):
        self.note(">>", obj)
        self.p.stdin.write((json.dumps(obj) + "\n").encode("utf-8"))
        self.p.stdin.flush()

    def note(self, tag, obj):
        line = f"[{time.time() - self.t0:7.1f}s] {tag} {json.dumps(obj, ensure_ascii=False)}"
        if self.out:
            self.out.write(line + "\n")
            self.out.flush()

    def event(self):
        line = self.p.stdout.readline()
        if not line:
            raise SystemExit("server exited")
        ev = json.loads(line.decode("utf-8"))
        if ev.get("event") != "token":
            self.note("<<", ev)
        return ev

    def until_done(self, on_token=None):
        toks = []
        while True:
            ev = self.event()
            if ev["event"] == "token":
                toks.append(ev["id"])
                if on_token:
                    on_token(len(toks))
            elif ev["event"] in ("done", "error"):
                return toks, ev


ROWS = []


def brief(tag, done, toks, text="", wall=0.0, switch=None):
    keys = ["finish", "prompt_tokens", "reused_tokens", "rollback_dropped", "replay_steps", "replay_ms",
            "prefill_tokens", "prefill_ms", "generated", "tok_s", "context"]
    print(f"{tag:30s} wall={wall:6.2f}s " + " ".join(f"{k}={done.get(k)}" for k in keys)
          + (f" | {text!r}" if text else ""), flush=True)
    ROWS.append(dict(tag=tag, wall=wall, switch=switch or {}, **{k: done.get(k) for k in keys}))


def table():
    """The table docs/p4_kv_ux.md §10 quotes."""
    head = ("step", "wall s", "prompt", "reused", "dropped", "replay", "replay ms",
            "prefill", "gen", "tok/s", "finish")
    print("\n| " + " | ".join(head) + " |", flush=True)
    print("|" + "|".join("---" for _ in head) + "|", flush=True)
    for r in ROWS:
        sw = r["switch"]
        replay = r["replay_steps"]
        rms = r["replay_ms"]
        if sw:
            # A session switch replays the window before the turn's own work.
            replay = f"{replay}+{sw.get('replay_steps')}"
            rms = f"{rms:.0f}+{sw.get('replay_ms', 0):.0f}" if isinstance(rms, float) else rms
        print("| {} | {:.2f} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["tag"], r["wall"], r["prompt_tokens"], r["reused_tokens"], r["rollback_dropped"],
            replay, rms if isinstance(rms, str) else f"{rms:.0f}", r["prefill_tokens"],
            r["generated"], f'{r["tok_s"]:.2f}' if isinstance(r["tok_s"], float) else r["tok_s"],
            r["finish"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=os.path.join(REPO, "build", "deepmoe.exe"))
    ap.add_argument("--cache-gb", type=int, default=16)
    ap.add_argument("--max-parked", type=int, default=2,
                    help="parked sessions kept in memory; 2 makes the third one evict the first")
    ap.add_argument("--kv-dir", default=os.path.join(REPO, "build", "serve_demo_kv"),
                    help="SSD prefix-cache directory, emptied before the run")
    ap.add_argument("--no-kv-disk", action="store_true",
                    help="do not fall back to the SSD prefix cache, so parking is measured alone")
    ap.add_argument("--log", default=os.path.join(REPO, "build", "serve_demo.log"))
    ap.add_argument("--out", default=os.path.join(REPO, "build", "serve_demo_events.txt"))
    args = ap.parse_args()
    s = Server(args)
    ready = s.event()
    print("ready:", json.dumps(ready), flush=True)

    def ids(text):
        s.send({"op": "tokenize", "text": text})
        return s.event()["ids"]

    def detok(v):
        s.send({"op": "detokenize", "ids": v})
        return s.event()["text"]

    base = ids("<｜begin▁of▁sentence｜>The three primary colors of light are")
    gen = {"op": "generate", "max_tokens": 12, "temperature": 0, "stop_ids": [1]}

    # A turn: switching session emits its own `session` event (the window replay
    # the switch paid for) before the `done` of the turn itself.
    def turn(tag, prompt, session, max_tokens=12, on_token=None):
        t0 = time.time()
        s.send(dict(gen, prompt_ids=prompt, max_tokens=max_tokens, session=session))
        switch = None
        toks = []
        while True:
            ev = s.event()
            if ev["event"] == "session":
                switch = ev
            elif ev["event"] == "token":
                toks.append(ev["id"])
                if on_token:
                    on_token(len(toks))
            elif ev["event"] in ("done", "error"):
                break
        wall = time.time() - t0
        brief(tag, ev, toks, detok(toks), wall, switch)
        return toks, ev

    t1, d1 = turn("1 alice first turn", base, "alice")
    t2, d2 = turn("2 alice continuation", base + t1 + ids(" In summary,"), "alice")

    edit = base + t1[:4] + ids(" Actually, let me rephrase:")
    t3, d3 = turn("3 alice diverging edit", edit, "alice")

    # cancel after 5 streamed tokens, from a second thread as a client would
    fired = {"n": False}

    def on_tok(n):
        if n == 5 and not fired["n"]:
            fired["n"] = True
            threading.Thread(target=lambda: s.send({"op": "cancel"})).start()
    t4, d4 = turn("4 alice cancel after 5", edit + t3, "alice", max_tokens=200, on_token=on_tok)

    # Session two: alice is parked (non-SWA state packed, window dropped).
    bob = ids("<｜begin▁of▁sentence｜>def fibonacci(n):")
    t5, d5 = turn("5 bob new session", bob, "bob")

    # Session three: with --max-parked 2 this evicts the least recently used
    # parked session, which is what `evicted` in the sessions event counts.
    carol = ids("<｜begin▁of▁sentence｜>The capital of France is")
    t6, d6 = turn("6 carol new session", carol, "carol")

    s.send({"op": "sessions"})
    print("sessions:", json.dumps(s.event()), flush=True)

    # Back to each of the first two: the window is rebuilt by replay, not by
    # re-prefilling the prompt (reused_tokens is the whole history).
    alice_hist = edit + t3 + t4[:-1]
    t7, d7 = turn("7 alice back from parking", alice_hist + ids(" Finally"), "alice")
    t8, d8 = turn("8 bob back from parking", bob + t5 + ids("    return"), "bob")
    t9, d9 = turn("9 carol still parked", carol + t6 + ids(" and"), "carol")

    s.send({"op": "sessions"})
    print("sessions:", json.dumps(s.event()), flush=True)
    s.send({"op": "status"})
    print("status:", json.dumps(s.event()), flush=True)
    table()
    s.send({"op": "quit"})
    s.p.wait(timeout=120)


if __name__ == "__main__":
    sys.exit(main())
