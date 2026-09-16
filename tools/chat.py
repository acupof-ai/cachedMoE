#!/usr/bin/env python3
"""Terminal chat with DeepSeek-V4.1-Flash over `deepmoe serve` (Track P,
docs/p3_chat.md).

    .venv/Scripts/python.exe tools/chat.py [--think] [--temp 1.0] [--top-p 0.95]
        [--max-tokens 1024] [--system TEXT] [--cache-gb N] [--max-context N]
        [--gpu-prefill-min N] [--check-topk] [--log FILE]
    .venv/Scripts/python.exe tools/chat.py --script turns.json --transcript out.md --stats out.json

Prompts are rendered by the checkpoint's own encoding/encoding.py, imported
read-only (bytecode writing off), and tokenised by the server's C++ tokenizer.
The server keeps a KV context per named session: a turn whose token ids extend
the previous turn's (prompt + reply) only prefills the new tokens, and one that
diverges rolls the server's KV back to the common prefix (Track R2,
docs/p4_kv_ux.md) -- which is what makes the official thinking-mode default,
drop_thinking=True (earlier turns' reasoning removed from the prompt), cheap: the
divergence is at the previous reasoning, so the turn re-prefills only what
follows it plus a <= 128-token window replay. `/drop` toggles it.

Ctrl+C while a reply streams sends {"op":"cancel"}; the server stops between
tokens and the partial reply is kept as the assistant message.

Commands: /reset  /think  /drop  /temp X  /top_p X  /greedy  /max N  /seed N
          /system TEXT  /session NAME  /sessions  /stats  /quit
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time

sys.dont_write_bytecode = True
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ.get("DEEPMOE_MODEL_DIR", r"D:\models\DeepSeek-V4.1-Flash")

DIM, RESET_C, CYAN, YELLOW = "\033[2m", "\033[0m", "\033[36m", "\033[33m"


def load_encoding():
    sys.path.insert(0, os.path.join(MODEL, "encoding"))
    import encoding  # noqa: E402
    return encoding


class Server:
    def __init__(self, args):
        exe = args.exe
        cmd = [exe, "serve", "--model", MODEL, "--max-context", str(args.max_context)]
        if args.cache_gb:
            cmd += ["--cache-gb", str(args.cache_gb)]
        if args.gpu_prefill_min:
            cmd += ["--gpu-prefill-min", str(args.gpu_prefill_min)]
        if args.check_topk:
            cmd += ["--check-topk"]
        if getattr(args, "kv_dir", ""):
            cmd += ["--kv-dir", args.kv_dir]
            if getattr(args, "kv_max_gb", 0):
                cmd += ["--kv-max-gb", str(args.kv_max_gb)]
        self.log = open(args.log, "ab") if args.log else subprocess.DEVNULL
        self.p = subprocess.Popen(cmd, cwd=REPO, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self.log, bufsize=0)
        self.ready = self.read_event()
        if self.ready.get("event") != "ready":
            raise SystemExit(f"server did not start: {self.ready}")

    def send(self, obj):
        self.p.stdin.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
        self.p.stdin.flush()

    def read_event(self):
        # `cancel` acknowledgements and `session` switches are informational and
        # may arrive between any two replies.
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise SystemExit("server exited (see --log)")
            ev = json.loads(line.decode("utf-8"))
            if ev.get("event") == "session":
                self.last_switch = ev
                continue
            if ev.get("event") != "cancel":
                return ev

    def tokenize(self, text):
        self.send({"op": "tokenize", "text": text})
        ev = self.read_event()
        return ev["ids"]

    def close(self):
        try:
            self.send({"op": "quit"})
            self.p.wait(timeout=60)
        except Exception:
            self.p.kill()


def kind_is(ev, *kinds):
    return ev.get("event") in kinds


class Chat:
    def __init__(self, server, enc, args, out=sys.stdout):
        self.s, self.enc, self.out = server, enc, out
        self.think = args.think
        self.drop_thinking = True
        self.session = "default"
        self.saved = {}       # parked chat state per session name
        self.temp, self.top_p = args.temp, args.top_p
        self.max_tokens = args.max_tokens
        self.seed = args.seed
        self.system = args.system
        self.color = out.isatty() if hasattr(out, "isatty") else False
        self.reset()
        self.turn_stats = []

    def reset(self):
        self.messages = [{"role": "system", "content": self.system}] if self.system else []
        self.ctx_ids = []     # prompt + reply ids of the last turn: what the server's KV holds (+1)
        self.ctx_text = ""    # the text those ids render to

    def mode(self):
        return "thinking" if self.think else "chat"

    def render(self):
        return self.enc.encode_messages(self.messages, thinking_mode=self.mode(),
                                        drop_thinking=self.drop_thinking)

    def turn(self, user_text, echo=True):
        self.messages.append({"role": "user", "content": user_text})
        full = self.render()
        if self.ctx_ids and full.startswith(self.ctx_text):
            ids = self.ctx_ids + self.s.tokenize(full[len(self.ctx_text):])
        else:
            ids = self.s.tokenize(full)
        seed = self.seed if self.seed is not None else random.randrange(1 << 62)
        req = {"op": "generate", "prompt_ids": ids, "max_tokens": self.max_tokens,
               "temperature": self.temp, "top_p": self.top_p, "seed": seed, "stop_ids": [1],
               "session": self.session}
        self.s.send(req)
        gen, text = [], []
        in_think = self.think
        if echo and in_think and self.color:
            self.out.write(DIM)
        while True:
            try:
                ev = self.s.read_event()
            except KeyboardInterrupt:
                self.s.send({"op": "cancel"})
                continue
            if kind_is(ev, "cancel", "session"):
                continue
            kind = ev.get("event")
            if kind == "prefill":
                if echo and ev["total"] > 32:
                    tag = f"[prefill {ev['done']}/{ev['total']}]"
                    self.out.write(f"\r{YELLOW}{tag}{RESET_C}" if self.color else f"\r{tag}")
                    self.out.flush()
                continue
            if kind == "token":
                if echo and len(gen) == 0:
                    self.out.write("\r" + " " * 30 + "\r")
                gen.append(ev["id"])
                piece = ev["text"]
                text.append(piece)
                if echo:
                    if in_think and "</think>" in "".join(text[-3:]):
                        in_think = False
                        if self.color:
                            self.out.write(piece + RESET_C)
                            self.out.flush()
                            continue
                    self.out.write(piece)
                    self.out.flush()
                continue
            if kind == "done":
                stats = ev
                break
            if kind == "error":
                self.messages.pop()
                raise RuntimeError(ev["message"])
        if echo and self.color:
            self.out.write(RESET_C)
        completion = self.s_detok(gen)
        # the assistant message, parsed the official way when it is well formed
        try:
            comp = completion if completion.endswith("<\uff5cend\u2581of\u2581sentence\uff5c>") else \
                completion + "<\uff5cend\u2581of\u2581sentence\uff5c>"
            msg = self.enc.parse_message_from_completion_text(comp, thinking_mode=self.mode())
        except Exception:
            body = completion.replace("<\uff5cend\u2581of\u2581sentence\uff5c>", "")
            reasoning, content = "", body
            if self.think and "</think>" in body:
                reasoning, content = body.split("</think>", 1)
            msg = {"role": "assistant", "content": content, "reasoning_content": reasoning}
        self.messages.append(msg)
        self.ctx_ids = ids + gen
        self.ctx_text = full + completion
        stats["seed"] = seed
        stats["mode"] = self.mode()
        stats["temperature"] = self.temp
        stats["top_p"] = self.top_p
        self.turn_stats.append(stats)
        if echo:
            self.out.write("\n" + self.stats_line(stats) + "\n")
        return msg, completion, stats

    def s_detok(self, ids):
        self.s.send({"op": "detokenize", "ids": ids})
        return self.s.read_event()["text"]

    def stats_line(self, st):
        s = (f"[TTFT {st['ttft_ms'] / 1e3:.1f} s | prefill {st['prefill_tokens']} tok"
             f" ({st['prefill_mode']}, reused {st['reused_tokens']}) {st['prefill_tok_s']:.2f} tok/s"
             f" hit {st['prefill_hit_rate']:.2f} | decode {st['generated']} tok {st['tok_s']:.2f} tok/s"
             f" hit {st['decode_hit_rate']:.3f} | {st['finish']} | ctx {st['context']}"
             + (f" | rollback -{st['rollback_dropped']} replay {st['replay_steps']}"
                f" ({st['replay_ms'] / 1e3:.1f} s)" if st.get('rollback_dropped') else "") + "]")
        return f"{CYAN}{s}{RESET_C}" if self.color else s

    def command(self, line):
        parts = line.split(maxsplit=1)
        c, arg = parts[0], (parts[1] if len(parts) > 1 else "")
        if c == "/reset":
            self.s.send({"op": "reset", "session": self.session})
            self.s.read_event()
            self.reset()
            return "context cleared (expert cache stays warm)"
        if c == "/think":
            self.think = not self.think
            return f"thinking mode {'on' if self.think else 'off'} (the next turn re-renders the prompt)"
        if c == "/drop":
            self.drop_thinking = not self.drop_thinking
            return f"drop_thinking {self.drop_thinking}"
        if c == "/temp":
            self.temp = float(arg)
            return f"temperature {self.temp}"
        if c == "/top_p":
            self.top_p = float(arg)
            return f"top_p {self.top_p}"
        if c == "/greedy":
            self.temp = 0.0
            return "greedy"
        if c == "/max":
            self.max_tokens = int(arg)
            return f"max_tokens {self.max_tokens}"
        if c == "/seed":
            self.seed = int(arg) if arg else None
            return f"seed {self.seed}"
        if c == "/system":
            self.system = arg
            self.reset()
            return "system prompt set; context cleared"
        if c == "/session":
            name = arg or "default"
            if name != self.session:
                self.saved[self.session] = (self.messages, self.ctx_ids, self.ctx_text)
                self.messages, self.ctx_ids, self.ctx_text = self.saved.pop(
                    name, ([{"role": "system", "content": self.system}] if self.system else [], [], ""))
                self.session = name
            return f"session '{name}' (the server parks the others' KV and replays <= 128 tokens on return)"
        if c == "/sessions":
            self.s.send({"op": "sessions"})
            return json.dumps(self.s.read_event(), indent=1)
        if c == "/stats":
            return json.dumps(self.turn_stats[-1] if self.turn_stats else {}, indent=1)
        return ("commands: /reset /think /drop /temp X /top_p X /greedy /max N /seed N /system TEXT "
                "/session NAME /sessions /stats /quit")


def run_script(chat, server, script, transcript_path, stats_path):
    turns = script["turns"]
    md = ["# deepMoE chat transcript", "",
          f"server: {json.dumps(server.ready)}", ""]
    all_stats = []
    for i, t in enumerate(turns):
        if t.get("reset"):
            chat.command("/reset")
            md.append("---\n*(context reset)*\n")
        if "think" in t:
            chat.think = bool(t["think"])
        if "temperature" in t:
            chat.temp = float(t["temperature"])
        if "top_p" in t:
            chat.top_p = float(t["top_p"])
        if "seed" in t:
            chat.seed = t["seed"]
        if "max_tokens" in t:
            chat.max_tokens = int(t["max_tokens"])
        print(f"\n>>> [{i}] {t['user']}", flush=True)
        t0 = time.time()
        msg, completion, st = chat.turn(t["user"], echo=True)
        st["wall_s"] = time.time() - t0
        st["label"] = t.get("label", "")
        all_stats.append(st)
        md.append(f"### Turn {i}{' — ' + t['label'] if t.get('label') else ''}")
        md.append(f"*mode {chat.mode()}, temperature {chat.temp}, top_p {chat.top_p}, seed {st['seed']}*\n")
        md.append("**User:** " + t["user"] + "\n")
        if msg.get("reasoning_content"):
            md.append("**Reasoning:**\n\n```text\n" + msg["reasoning_content"].strip() + "\n```\n")
        md.append("**Assistant:**\n\n" + (msg.get("content") or "").strip() + "\n")
        md.append("`" + chat.stats_line(st) + "`\n")
        if transcript_path:
            with open(transcript_path, "w", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(md) + "\n")
        if stats_path:
            with open(stats_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump({"server": server.ready, "turns": all_stats}, f, ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=os.path.join(REPO, "build", "deepmoe.exe"))
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--system", default="")
    ap.add_argument("--cache-gb", type=int, default=0)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--gpu-prefill-min", type=int, default=0)
    ap.add_argument("--check-topk", action="store_true")
    ap.add_argument("--kv-dir", default="", help="directory for the SSD parked-session/prefix KV cache")
    ap.add_argument("--kv-max-gb", type=int, default=0, help="disk cache budget in GiB (0 = server default)")
    ap.add_argument("--log", default=os.path.join(REPO, "build", "serve.log"))
    ap.add_argument("--script")
    ap.add_argument("--transcript")
    ap.add_argument("--stats")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if os.name == "nt":
        os.system("")   # enable ANSI escapes on the Windows console

    enc = load_encoding()
    print(f"starting deepmoe serve (log: {args.log}) ...", flush=True)
    server = Server(args)
    r = server.ready
    print(f"ready in {r['load_s']:.1f} s: expert cache {r['cache_gb']:.1f} GiB ({r['cache_slots']} slots), "
          f"context {r['max_context']}", flush=True)
    chat = Chat(server, enc, args)
    try:
        if args.script:
            with open(args.script, encoding="utf-8") as f:
                run_script(chat, server, json.load(f), args.transcript, args.stats)
            return 0
        print("type a message; /help for commands", flush=True)
        while True:
            try:
                line = input(f"\n{'[think] ' if chat.think else ''}>>> ")
            except EOFError:
                break
            line = line.strip()
            if not line:
                continue
            if line in ("/quit", "/exit"):
                break
            if line.startswith("/"):
                try:
                    print(chat.command(line))
                except Exception as e:
                    print(f"error: {e}")
                continue
            try:
                chat.turn(line)
            except RuntimeError as e:
                print(f"\nserver error: {e}")
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
