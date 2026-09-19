#!/usr/bin/env python3
"""A local web chat UI for DeepSeek-V4.1-Flash over `deepmoe serve`.

    .venv\\Scripts\\python.exe tools\\web\\server.py [--port 8080] [--max-context 65536]
        [--cache-gb N | --cache-slots N] [--kv-dir DIR] [--kv-max-gb N]
        [--exe build\\deepmoe.exe] [--think] [--log build\\web_serve.log]

Then open http://127.0.0.1:8080 .

What this is
------------
`deepmoe serve` (docs/p3_chat.md §1) is one long-running engine behind
line-delimited JSON on stdin/stdout.  This process launches it as a child, keeps
the chat state that `tools/chat.py` keeps (the same `encoding/encoding.py`
renderer imported read-only from the checkpoint, so a reply here and a reply in
the CLI are the same reply for the same seed), and bridges a browser to it:
`fetch()` + a chunked `text/event-stream` body carries `prefill` / `token` /
`done` through to the page as they arrive.

Concurrency
-----------
serve's main loop is serial: it pops one request at a time off its inbox and
`Session::generate` runs to completion before the next is popped.  Track MS's
multi-stream scheduler (docs/p4_multistream.md) is reachable only through
`generate_multi`, which wants every turn of the round handed to it up front --
an offline batch shape, not "whoever hits send next".  So turns are QUEUED here,
one in flight, and a waiting tab is told its place in the queue.  The engine's
named sessions (docs/p4_kv_ux.md §7) still do the real work: each browser tab is
a named session, serve keeps one live in the KV store and parks the rest, and a
tab coming back replays <= 128 tokens instead of re-prefilling.

The context ceiling
-------------------
`Engine::max_context()` is `min(KvStoreConfig::max_context, kMaxIndexPositions)`
and `kMaxIndexPositions` is `65535 * kIdxScoreTile` = **524,280**
(runtime/decode_layer.h): `indexer.score` covers 8 compressed positions per
workgroup and a dispatch may have at most 65,535 workgroups.  It is a ceiling in
TOKENS and not only in compressed positions because the checkpoint's last KV
source, layer 20, has `compress_ratio == 1` -- it keeps one compressed row per
token, so `n_cmp == positions` on that plane.  `Session::generate` refuses a
prompt with `prompt.size() >= max_context()`, and so does this server, before
anything is spent on it.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import random
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
MODEL = os.environ.get("DEEPMOE_MODEL_DIR", r"D:\models\DeepSeek-V4.1-Flash")

# runtime/decode_layer.h: 65535 workgroups * gpu::kIdxScoreTile (8).
K_MAX_INDEX_POSITIONS = 65535 * 8          # 524,280
# The live bf16 KV a position costs: (head_dim + index_dim) * 2 B per compressed
# row, summed over the four kv sources at their ratios (2,2,2,1) -- 2,560 + 640.
# docs/design.md §11's 894 B/token is the PACKED (.pkv) form of the same state.
KV_BYTES_PER_TOKEN = 3200
# Prefill on the decode path, measured (docs/p4_kv_ux.md §8, p4_prefill_speed.md
# §1.2: 4,133 tokens in 101.6 s). Only a seed: the real rate is measured live
# from the `prefill` events and fed back to the UI.
PREFILL_MS_PER_TOKEN = 24.0

EOS_TEXT = "<\uff5cend\u2581of\u2581sentence\uff5c>"
STOP_IDS = [1]


def load_encoding():
    """The checkpoint's own prompt renderer, exactly as tools/chat.py imports it."""
    sys.path.insert(0, os.path.join(MODEL, "encoding"))
    import encoding  # noqa: E402
    return encoding


# --------------------------------------------------------------------------- serve

class Serve:
    """The `deepmoe serve` child: one writer, one reader thread, one turn in flight."""

    def __init__(self, args):
        cmd = [args.exe, "serve", "--model", MODEL, "--max-context", str(args.max_context)]
        if args.cache_slots:
            cmd += ["--cache-slots", str(args.cache_slots)]
        elif args.cache_gb:
            cmd += ["--cache-gb", str(args.cache_gb)]
        if args.gpu_prefill_min:
            cmd += ["--gpu-prefill-min", str(args.gpu_prefill_min)]
        if args.kv_dir:
            cmd += ["--kv-dir", args.kv_dir]
        if args.no_kv_disk:
            cmd += ["--no-kv-disk"]
        if args.kv_max_gb:
            cmd += ["--kv-max-gb", str(args.kv_max_gb)]
        if args.max_parked:
            cmd += ["--max-parked", str(args.max_parked)]
        # Track D4: the second read source. Repeatable, off unless asked for, and
        # the engine drops a mirror that fails its health probe rather than
        # dying on it (docs/p4_e_drive_diag.md §5.2), so a passthrough here
        # cannot take the web UI down with the drive.
        for d in args.mirror:
            cmd += ["--mirror", d]
        self.cmd = cmd
        self.log = open(args.log, "ab") if args.log else subprocess.DEVNULL
        self.p = subprocess.Popen(cmd, cwd=REPO, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self.log, bufsize=0)
        self.io_lock = threading.Lock()          # one writer
        self.rpc_lock = threading.Lock()         # one non-generate round trip at a time
        self.rpc_q: queue.Queue = queue.Queue()
        self.stream_q: queue.Queue | None = None  # the in-flight turn's event sink
        self.stream_lock = threading.Lock()
        self.dead = False
        self.ready = self._read_ready()
        self.max_context = int(self.ready.get("max_context", args.max_context))
        threading.Thread(target=self._reader, daemon=True).start()

    # -- plumbing ----------------------------------------------------------
    def _readline(self):
        line = self.p.stdout.readline()
        if not line:
            self.dead = True
            raise RuntimeError("deepmoe serve exited (see the --log file)")
        return json.loads(line.decode("utf-8"))

    def _read_ready(self):
        ev = self._readline()
        if ev.get("event") != "ready":
            raise SystemExit(f"server did not start: {ev}")
        return ev

    def _reader(self):
        """Every line from serve. Turn events go to the in-flight turn, the rest to rpc."""
        turn_events = ("token", "prefill", "done", "session", "cancel")
        while True:
            try:
                ev = self._readline()
            except Exception as e:
                with self.stream_lock:
                    if self.stream_q:
                        self.stream_q.put({"event": "error", "message": str(e)})
                self.rpc_q.put({"event": "error", "message": str(e)})
                return
            kind = ev.get("event")
            if kind in turn_events:
                with self.stream_lock:
                    q = self.stream_q
                if q is not None:
                    q.put(ev)
                    continue
                # `cancel`/`session` can land between turns; nothing waits on them.
                if kind in ("cancel", "session"):
                    continue
            if kind == "error":
                with self.stream_lock:
                    q = self.stream_q
                if q is not None:
                    q.put(ev)
                    continue
            self.rpc_q.put(ev)

    def send(self, obj):
        if self.dead:
            raise RuntimeError("deepmoe serve is gone")
        with self.io_lock:
            self.p.stdin.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
            self.p.stdin.flush()

    def rpc(self, obj, timeout=600):
        """A request with exactly one reply (tokenize / detokenize / status / ...)."""
        with self.rpc_lock:
            self.send(obj)
            ev = self.rpc_q.get(timeout=timeout)
            if ev.get("event") == "error":
                raise RuntimeError(ev.get("message", "serve error"))
            return ev

    # -- ops ---------------------------------------------------------------
    def tokenize(self, text):
        return self.rpc({"op": "tokenize", "text": text})["ids"]

    def detokenize(self, ids):
        return self.rpc({"op": "detokenize", "ids": list(ids)})["text"]

    def status(self, session):
        return self.rpc({"op": "status", "session": session})

    def sessions(self):
        return self.rpc({"op": "sessions"})

    def reset(self, session):
        return self.rpc({"op": "reset", "session": session})

    def cancel(self):
        # Cancels every generate serve has READ so far. Only one is ever in
        # flight here (the rest are still in this process's queue), so this
        # cancels exactly the running turn.
        self.send({"op": "cancel"})

    def begin_turn(self):
        q: queue.Queue = queue.Queue()
        with self.stream_lock:
            self.stream_q = q
        return q

    def end_turn(self):
        with self.stream_lock:
            self.stream_q = None

    def close(self):
        try:
            self.send({"op": "quit"})
            self.p.wait(timeout=120)
        except Exception:
            try:
                self.p.kill()
            except Exception:
                pass


# --------------------------------------------------------------------------- chat state

class ChatState:
    """One named session's conversation -- the same fields tools/chat.py keeps."""

    def __init__(self, enc, system=""):
        self.enc = enc
        self.system = system
        self.think = False
        self.drop_thinking = True
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.messages = [{"role": "system", "content": self.system}] if self.system else []
        self.ctx_ids = []    # prompt + reply ids of the last turn: what serve's KV holds (+1)
        self.ctx_text = ""   # the text those ids render to

    def mode(self):
        return "thinking" if self.think else "chat"

    def render(self, user_text):
        msgs = self.messages + [{"role": "user", "content": user_text}]
        return self.enc.encode_messages(msgs, thinking_mode=self.mode(),
                                        drop_thinking=self.drop_thinking)

    def prompt_ids(self, serve, user_text):
        """The turn's prompt ids, reusing the ids of the prefix already rendered."""
        full = self.render(user_text)
        if self.ctx_ids and full.startswith(self.ctx_text):
            ids = self.ctx_ids + serve.tokenize(full[len(self.ctx_text):])
        else:
            ids = serve.tokenize(full)
        return ids, full

    def reused(self, ids):
        """How many leading ids serve's KV already holds (what it will not prefill)."""
        n = 0
        for a, b in zip(self.ctx_ids, ids):
            if a != b:
                break
            n += 1
        return n

    def commit(self, user_text, full, ids, gen, completion):
        self.messages.append({"role": "user", "content": user_text})
        try:
            comp = completion if completion.endswith(EOS_TEXT) else completion + EOS_TEXT
            msg = self.enc.parse_message_from_completion_text(comp, thinking_mode=self.mode())
        except Exception:
            body = completion.replace(EOS_TEXT, "")
            reasoning, content = "", body
            if self.think and "</think>" in body:
                reasoning, content = body.split("</think>", 1)
            msg = {"role": "assistant", "content": content, "reasoning_content": reasoning}
        self.messages.append(msg)
        self.ctx_ids = ids + gen
        self.ctx_text = full + completion
        return msg


# --------------------------------------------------------------------------- the turn queue

class Job:
    def __init__(self, session, body):
        self.session = session
        self.body = body
        self.out: queue.Queue = queue.Queue()
        self.cancelled = False
        self.started = False


class Bridge:
    def __init__(self, serve, enc, args):
        self.serve = serve
        self.enc = enc
        self.args = args
        self.states: dict[str, ChatState] = {}
        self.states_lock = threading.Lock()
        self.jobs: list[Job] = []                  # [0] is running once started
        self.jobs_lock = threading.Lock()
        self.jobs_cv = threading.Condition(self.jobs_lock)
        # serve's main loop pops one request at a time, so while a turn runs it
        # cannot answer `tokenize` or `status` either -- a preview issued mid-turn
        # would block for the whole turn. Callers check this and say so instead.
        self.busy = False
        self.prefill_ms_per_token = PREFILL_MS_PER_TOKEN
        threading.Thread(target=self._worker, daemon=True).start()

    def state(self, session):
        with self.states_lock:
            st = self.states.get(session)
            if st is None:
                st = ChatState(self.enc, self.args.system)
                st.think = self.args.think
                self.states[session] = st
            return st

    # -- queue -------------------------------------------------------------
    def submit(self, session, body):
        job = Job(session, body)
        with self.jobs_cv:
            self.jobs.append(job)
            pos = len(self.jobs) - 1
            self.jobs_cv.notify_all()
        if pos > 0:
            job.out.put({"event": "queued", "position": pos})
        return job

    def _broadcast_positions(self):
        for i, j in enumerate(self.jobs[1:], start=1):
            j.out.put({"event": "queued", "position": i})

    def cancel(self, session):
        """Cancel this session's running turn, or drop its queued ones."""
        hit = False
        with self.jobs_cv:
            for j in list(self.jobs):
                if j.session != session:
                    continue
                j.cancelled = True
                hit = True
                if not j.started:
                    self.jobs.remove(j)
                    j.out.put({"event": "done", "finish": "cancel", "queued_drop": True})
                    j.out.put(None)
            running = self.jobs[0] if self.jobs and self.jobs[0].started else None
            self._broadcast_positions()
        if running is not None and running.session == session:
            self.serve.cancel()
        return hit

    def _worker(self):
        while True:
            with self.jobs_cv:
                while not self.jobs:
                    self.jobs_cv.wait()
                job = self.jobs[0]
                job.started = True
                self.busy = True
                self._broadcast_positions()
            try:
                self._run(job)
            except Exception as e:
                traceback.print_exc()
                job.out.put({"event": "error", "message": str(e)})
            finally:
                job.out.put(None)
                with self.jobs_cv:
                    if self.jobs and self.jobs[0] is job:
                        self.jobs.pop(0)
                    self.busy = bool(self.jobs)
                    self._broadcast_positions()

    # -- one turn ----------------------------------------------------------
    def _run(self, job):
        serve, st = self.serve, self.state(job.session)
        b = job.body
        user_text = b.get("text", "")
        if not user_text.strip():
            job.out.put({"event": "error", "message": "空消息"})
            return
        with st.lock:
            st.think = bool(b.get("think", st.think))
            ids, full = st.prompt_ids(serve, user_text)
        if len(ids) >= serve.max_context:
            job.out.put({"event": "error", "message":
                         f"提示词 {len(ids)} token，超过引擎上限 {serve.max_context}"})
            return
        reused = st.reused(ids)
        to_prefill = max(0, len(ids) - reused)
        job.out.put({"event": "start", "prompt_tokens": len(ids), "reused": reused,
                     "to_prefill": to_prefill,
                     "eta_s": to_prefill * self.prefill_ms_per_token / 1e3,
                     "max_context": serve.max_context})

        seed = b.get("seed")
        seed = int(seed) if seed not in (None, "") else random.randrange(1 << 62)
        req = {"op": "generate", "prompt_ids": ids,
               "max_tokens": int(b.get("max_tokens", 1024)),
               "temperature": float(b.get("temperature", 1.0)),
               "top_p": float(b.get("top_p", 0.95)),
               "seed": seed, "stop_ids": STOP_IDS, "session": job.session}

        q = serve.begin_turn()
        try:
            serve.send(req)
            if job.cancelled:
                serve.cancel()
            gen, text = [], []
            t_pf0, pf_done0 = None, 0
            while True:
                ev = q.get()
                kind = ev.get("event")
                if kind == "prefill":
                    now = time.time()
                    if t_pf0 is None:
                        t_pf0, pf_done0 = now, ev["done"]
                    elif ev["done"] > pf_done0:
                        rate = (now - t_pf0) * 1e3 / (ev["done"] - pf_done0)
                        if 1.0 < rate < 5000.0:
                            self.prefill_ms_per_token += 0.2 * (rate - self.prefill_ms_per_token)
                    left = max(0, ev["total"] - ev["done"])
                    ev["eta_s"] = left * self.prefill_ms_per_token / 1e3
                    ev["ms_per_token"] = self.prefill_ms_per_token
                    job.out.put(ev)
                    continue
                if kind == "token":
                    gen.append(ev["id"])
                    text.append(ev["text"])
                    job.out.put(ev)
                    continue
                if kind in ("session", "cancel"):
                    continue
                if kind == "error":
                    job.out.put(ev)
                    with st.lock:
                        st.reset()
                    return
                if kind == "done":
                    completion = serve.detokenize(gen) if gen else ""
                    with st.lock:
                        msg = st.commit(user_text, full, ids, gen, completion)
                    ev["seed"] = seed
                    ev["mode"] = st.mode()
                    ev["content"] = msg.get("content") or ""
                    ev["reasoning"] = msg.get("reasoning_content") or ""
                    job.out.put(ev)
                    return
        finally:
            serve.end_turn()


# --------------------------------------------------------------------------- http

def sse(obj):
    return ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    bridge: Bridge = None        # set on the class before serve_forever
    protocol_version = "HTTP/1.1"
    server_version = "deepmoe-web"

    def log_message(self, fmt, *a):
        if os.environ.get("DEEPMOE_WEB_VERBOSE"):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % a))

    # -- helpers -----------------------------------------------------------
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    # -- routes ------------------------------------------------------------
    def do_GET(self):
        u = urlparse(self.path)
        try:
            if u.path in ("/", "/index.html"):
                with open(os.path.join(HERE, "index.html"), "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if u.path == "/api/config":
                b = self.bridge
                return self._send(200, {
                    "max_context": b.serve.max_context,
                    "ceiling": K_MAX_INDEX_POSITIONS,
                    "ready": b.serve.ready,
                    "prefill_ms_per_token": b.prefill_ms_per_token,
                    "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
                    "model": MODEL,
                    "cmd": " ".join(b.serve.cmd),
                })
            if u.path == "/api/status":
                q = parse_qs(u.query)
                s = (q.get("session") or ["default"])[0]
                b = self.bridge
                with b.jobs_lock:
                    busy, depth = b.busy, len(b.jobs)
                # A `status` op would sit in serve's inbox until the turn ends.
                ev = {"busy": True} if busy else b.serve.status(s)
                ev["prefill_ms_per_token"] = b.prefill_ms_per_token
                ev["queue"] = depth
                return self._send(200, ev)
            if u.path == "/api/sessions":
                return self._send(200, self.bridge.serve.sessions())
            return self._send(404, {"error": "not found"})
        except Exception as e:
            return self._send(500, {"error": str(e)})

    def do_POST(self):
        u = urlparse(self.path)
        try:
            body = self._json_body()
            if u.path == "/api/preview":
                return self._preview(body)
            if u.path == "/api/chat":
                return self._chat(body)
            if u.path == "/api/cancel":
                hit = self.bridge.cancel(body.get("session", "default"))
                return self._send(200, {"cancelled": hit})
            if u.path == "/api/reset":
                s = body.get("session", "default")
                self.bridge.serve.reset(s)
                stt = self.bridge.state(s)
                with stt.lock:
                    stt.reset()
                return self._send(200, {"ok": True, "session": s})
            return self._send(404, {"error": "not found"})
        except Exception as e:
            traceback.print_exc()
            return self._send(500, {"error": str(e)})

    def _preview(self, body):
        """Token count + prefill ETA for a draft, before a single token is spent."""
        b = self.bridge
        s = body.get("session", "default")
        with b.jobs_lock:
            if b.busy:
                # serve is inside a turn; a `tokenize` would block until it ends.
                return self._send(200, {"pending": True, "over": False,
                                        "ms_per_token": b.prefill_ms_per_token,
                                        "message": "生成中，发送前再统计"})
        st = b.state(s)
        with st.lock:
            st.think = bool(body.get("think", st.think))
            ids, _full = st.prompt_ids(b.serve, body.get("text", ""))
            reused = st.reused(ids)
        to_prefill = max(0, len(ids) - reused)
        over = len(ids) >= b.serve.max_context
        return self._send(200, {
            "prompt_tokens": len(ids), "reused": reused, "to_prefill": to_prefill,
            "ms_per_token": b.prefill_ms_per_token,
            "eta_s": to_prefill * b.prefill_ms_per_token / 1e3,
            "max_context": b.serve.max_context, "ceiling": K_MAX_INDEX_POSITIONS,
            "over": over,
            "message": (f"提示词 {len(ids)} token，超过本次启动的上限 {b.serve.max_context}"
                        if over else ""),
        })

    def _chat(self, body):
        job = self.bridge.submit(body.get("session", "default"), body)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def chunk(data: bytes):
            self.wfile.write(b"%X\r\n" % len(data) + data + b"\r\n")
            self.wfile.flush()

        try:
            while True:
                try:
                    ev = job.out.get(timeout=20)
                except queue.Empty:
                    chunk(b": ping\n\n")       # keep the connection warm through a long prefill
                    continue
                if ev is None:
                    break
                chunk(sse(ev))
            chunk(b"")
        except (BrokenPipeError, ConnectionResetError, OSError):
            # The tab went away mid-reply; stop the turn rather than burn the GPU.
            job.cancelled = True
            try:
                self.bridge.cancel(job.session)
            except Exception:
                pass


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="local web chat UI for `deepmoe serve`")
    ap.add_argument("--exe", default=os.path.join(REPO, "build", "deepmoe.exe"))
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--max-context", type=int, default=65536)
    ap.add_argument("--cache-gb", type=int, default=0)
    ap.add_argument("--cache-slots", type=int, default=0)
    ap.add_argument("--gpu-prefill-min", type=int, default=0)
    ap.add_argument("--kv-dir", default="", help="SSD .pkv prefix/parked KV cache directory")
    ap.add_argument("--kv-max-gb", type=int, default=0)
    ap.add_argument("--no-kv-disk", action="store_true")
    ap.add_argument("--max-parked", type=int, default=0)
    ap.add_argument("--mirror", action="append", default=[],
                    help="a second read source holding the same checkpoint "
                         "(repeatable); passed through to `deepmoe serve`")
    ap.add_argument("--system", default="")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--log", default=os.path.join(REPO, "build", "web_serve.log"))
    args = ap.parse_args()

    if args.max_context > K_MAX_INDEX_POSITIONS:
        raise SystemExit(
            f"--max-context {args.max_context} is past the engine ceiling "
            f"{K_MAX_INDEX_POSITIONS} (runtime/decode_layer.h kMaxIndexPositions = "
            f"65535 workgroups * 8 positions); the engine would clamp it silently.")
    kv_gb = args.max_context * KV_BYTES_PER_TOKEN / (1 << 30)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    enc = load_encoding()
    print(f"starting deepmoe serve (log: {args.log}) ...", flush=True)
    serve = Serve(args)
    r = serve.ready
    print(f"ready in {r['load_s']:.1f} s | expert cache {r['cache_gb']:.1f} GiB "
          f"({r['cache_slots']} slots) | max_context {serve.max_context} "
          f"(live KV ~{kv_gb:.2f} GiB at full) | kv disk {r.get('kv_disk_dir', '')} "
          f"| read sources {r.get('sources', 1)}", flush=True)

    bridge = Bridge(serve, enc, args)
    Handler.bridge = bridge
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    print(f"open http://{args.host}:{args.port}/", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nstopping deepmoe serve ...", flush=True)
        serve.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
