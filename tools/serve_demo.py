#!/usr/bin/env python3
"""Scripted `deepmoe serve` session exercising Track R2's serve UX (docs/p4_kv_ux.md §4):
KV continuation, rollback to a diverging prompt's common prefix, cancel mid-generation,
two named sessions with parking, and the engram tables derived at startup.

    .venv/Scripts/python.exe tools/serve_demo.py [--exe build/deepmoe.exe] [--cache-gb 16]
        [--log build/serve_demo.log] [--out events.jsonl]
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
        cmd = [args.exe, "serve", "--model", MODEL, "--max-context", "8192"]
        if args.cache_gb:
            cmd += ["--cache-gb", str(args.cache_gb)]
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


def brief(tag, done, toks, text=""):
    keys = ["finish", "prompt_tokens", "reused_tokens", "rollback_dropped", "replay_steps", "replay_ms",
            "prefill_tokens", "prefill_ms", "generated", "tok_s", "context"]
    print(f"{tag:28s} " + " ".join(f"{k}={done.get(k)}" for k in keys) + (f" | {text!r}" if text else ""),
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=os.path.join(REPO, "build", "deepmoe.exe"))
    ap.add_argument("--cache-gb", type=int, default=16)
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

    s.send(dict(gen, prompt_ids=base, session="alice"))
    t1, d1 = s.until_done()
    brief("1 alice: first turn", d1, t1, detok(t1))

    cont = base + t1 + ids(" In summary,")
    s.send(dict(gen, prompt_ids=cont, session="alice"))
    t2, d2 = s.until_done()
    brief("2 alice: continuation", d2, t2, detok(t2))

    edit = base + t1[:4] + ids(" Actually, let me rephrase:")
    s.send(dict(gen, prompt_ids=edit, session="alice"))
    t3, d3 = s.until_done()
    brief("3 alice: diverging edit", d3, t3, detok(t3))

    # cancel after 5 streamed tokens, from a second thread as a client would
    s.send(dict(gen, prompt_ids=edit + t3, max_tokens=200, session="alice"))
    fired = {"n": False}

    def on_tok(n):
        if n == 5 and not fired["n"]:
            fired["n"] = True
            threading.Thread(target=lambda: s.send({"op": "cancel"})).start()
    t4, d4 = s.until_done(on_tok)
    brief("4 alice: cancel after 5", d4, t4, detok(t4))

    s.send(dict(gen, prompt_ids=ids("<｜begin▁of▁sentence｜>def fibonacci(n):"), session="bob"))
    t5, d5 = s.until_done()
    brief("5 bob: new session", d5, t5, detok(t5))

    s.send({"op": "sessions"})
    print("sessions:", json.dumps(s.event()), flush=True)

    alice_hist = edit + t3 + t4[:-1]
    s.send(dict(gen, prompt_ids=alice_hist + ids(" Finally"), session="alice"))
    t6, d6 = s.until_done()
    brief("6 alice: back from parking", d6, t6, detok(t6))

    s.send({"op": "status"})
    print("status:", json.dumps(s.event()), flush=True)
    s.send({"op": "quit"})
    s.p.wait(timeout=120)


if __name__ == "__main__":
    sys.exit(main())
