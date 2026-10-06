#!/usr/bin/env python3
"""Owner §4.8: one web engine, three power modes, eight 512-token turns each.

The user web engine must be stopped first. This experiment owns a separate
HTTP port and disk state. Only the bound engine child receives thermal signals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools/web"))
import runtime_defaults
import launch_guarded as web_guard
from repetition_metrics import metrics
from thermal_guard import ProfileMonitor, ThermalLatch, assert_idle, discover_sensors, sample
from power_profile_report import PROFILES, report


def save(path, value):
    web_guard.write_state(path, value)


class ThermalController:
    def __init__(self, server, out, expected_engine):
        self.server, self.out = server, out
        self.expected_engine = expected_engine
        self.child = None
        self.sensors = discover_sensors()
        self.latch = ThermalLatch()
        self.paused = False
        self.stop = threading.Event()
        self.error = None
        self.label = "startup"
        self.phase = "startup"
        self.latest = None
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        try:
            with (self.out / "thermal.jsonl").open("x") as log, ProfileMonitor() as monitor:
                while not self.stop.is_set():
                    if self.child is None:
                        self.child = web_guard.find_engine(self.server.pid, self.expected_engine)
                    values = sample(self.sensors, monitor)
                    if values["ac"] != 1 or values["power_profile"] not in PROFILES:
                        raise RuntimeError("AC or supported power profile lost")
                    if self.phase == "generate" and values["power_profile"] != self.label:
                        raise RuntimeError("power profile changed during a turn")
                    self.paused = web_guard.thermal_transition(
                        self.child, values, self.latch, self.paused)
                    values.update(paused=self.paused, latched_sensors=sorted(self.latch.hot),
                                  arm=self.label, phase=self.phase,
                                  engine_pid=self.child.pid if self.child else None)
                    log.write(json.dumps(values) + "\n")
                    log.flush()
                    self.latest = values
                    self.stop.wait(.05)
        except Exception as error:
            self.error = error
            if self.child and self.child.live():
                web_guard.bound_signal(self.child, signal.SIGSTOP)
                self.paused = True
            # Wake a blocked HTTP reader. The owned shutdown below handles any
            # surviving engine; an incomplete arm is never used for selection.
            if self.server.poll() is None:
                self.server.terminate()

    def check(self):
        if self.error:
            raise RuntimeError("thermal controller failed") from self.error
        if self.server.poll() is not None:
            raise RuntimeError(f"temporary web server exited: {self.server.returncode}")

    def cool_start(self, target):
        self.phase, self.label = "cooldown", target
        subprocess.run(["powerprofilesctl", "set", target], check=True)
        deadline = time.monotonic() + 900
        announced = time.monotonic()
        while True:
            self.check()
            values = self.latest
            if (values and values["power_profile"] == target and not self.paused
                    and all(v <= 60 for v in web_guard.guarded_temperatures(values).values())):
                return dict(values)
            if time.monotonic() > deadline:
                raise RuntimeError("all GPU/NVMe sensors did not cool to <=60 C within 900 s")
            if time.monotonic() - announced >= 30:
                print("COOLING", target, web_guard.guarded_temperatures(values or {}), flush=True)
                announced = time.monotonic()
            time.sleep(.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-repo", type=Path, required=True)
    parser.add_argument("--script", type=Path, default=ROOT / "bench/power_profile_prompts.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    assert_idle()
    with socket.socket() as port_probe:
        if port_probe.connect_ex(("127.0.0.1", args.port)) == 0:
            raise RuntimeError("temporary HTTP port is already occupied")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    turns = json.loads(args.script.read_text())["turns"]
    if len(turns) != 8 or any(t["max_tokens"] != 512 for t in turns):
        raise ValueError("fixed workload must contain eight 512-token turns")
    original_profile = subprocess.check_output(["powerprofilesctl", "get"], text=True).strip()
    launch = argparse.Namespace(repo=args.launch_repo.resolve(), state_dir=out,
                               spec_k=runtime_defaults.PRODUCTION_DRAFT_TOKENS,
                               gpu_route=0, port=args.port)
    command, env = web_guard.launch_configuration(launch)
    # Preserve production policy. Only the private state, port and log differ.
    config = dict(command=command, environment={k: v for k, v in env.items()
                  if k.startswith(("CACHEDMOE_", "DEEPMOE_"))},
                  source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  launch_repo_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.launch_repo, text=True).strip(),
                  executable_sha256=hashlib.sha256(Path(command[command.index("--exe") + 1]).read_bytes()).hexdigest(),
                  workload_sha256=hashlib.sha256(args.script.read_bytes()).hexdigest(),
                  order=list(PROFILES), original_profile=original_profile)
    save(out / "manifest.json", config)
    (out / "workload.json").write_bytes(args.script.read_bytes())
    base = f"http://127.0.0.1:{args.port}"

    def rpc(path, body=None):
        request = urllib.request.Request(base + path,
            None if body is None else json.dumps(body, ensure_ascii=False).encode(),
            {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)

    server = controller = owned = None
    complete = False
    try:
        subprocess.run(["powerprofilesctl", "set", PROFILES[0]], check=True)
        with (out / "web.log").open("x") as log:
            server = subprocess.Popen(command, cwd=args.launch_repo, env=env,
                                      stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
            expected = command[command.index("--exe") + 1]
            owned = web_guard.OwnedSession(server, expected)
            controller = ThermalController(server, out, expected)
            controller.thread.start()
            deadline = time.monotonic() + 180
            while True:
                controller.check()
                try:
                    ready = rpc("/api/config")
                    break
                except urllib.error.URLError:
                    if time.monotonic() > deadline:
                        raise RuntimeError("temporary web startup timed out")
                    time.sleep(.5)
            if (ready["ready"]["sources"] != 2 or ready["ready"]["cache_slots"] != 5500
                    or not ready["ready"]["kv_disk"] or ready["max_context"] != runtime_defaults.MAX_CONTEXT
                    or ready["mask_cache"] != "dynamic" or ready["spec_k"] != 2):
                raise RuntimeError("live policy differs from owner experiment")
            save(out / "config.json", ready)
            session = "power-profile-comparison"
            engine_pid = controller.child.pid
            for mode in PROFILES:
                arm = out / mode
                arm.mkdir()
                rpc("/api/reset", dict(session=session))
                cold = controller.cool_start(mode)
                save(arm / "cold_start.json", cold)
                save(arm / "status_before.json", rpc("/api/status?session=" + session))
                print("ARM", mode, "engine", engine_pid, "cold", web_guard.guarded_temperatures(cold), flush=True)
                outputs = []
                for index, turn in enumerate(turns):
                    controller.check()
                    if index and turn.get("reset"):
                        rpc("/api/reset", dict(session=session))
                    body = dict(session=session, text=turn["text"], think=turn["think"],
                                reasoning_effort=runtime_defaults.REQUEST_DEFAULTS["reasoning_effort"],
                                temperature=runtime_defaults.REQUEST_DEFAULTS["temperature"],
                                top_p=runtime_defaults.REQUEST_DEFAULTS["top_p"], seed=turn["seed"],
                                max_tokens=512, decode_mode="mask-spec")
                    controller.phase = "generate"
                    begin = time.time()
                    request = urllib.request.Request(base + "/api/chat",
                        json.dumps(body, ensure_ascii=False).encode(), {"Content-Type": "application/json"})
                    token_ids, done = [], None
                    with urllib.request.urlopen(request, timeout=1800) as response, \
                            (arm / f"turn{index}_events.jsonl").open("x") as events:
                        for line in response:
                            controller.check()
                            if not line.startswith(b"data:"):
                                continue
                            event = json.loads(line[5:])
                            event["host_unix"] = time.time()
                            events.write(json.dumps(event, ensure_ascii=False) + "\n")
                            events.flush()
                            if event["event"] == "error":
                                raise RuntimeError(event)
                            if event["event"] == "token":
                                token_ids.append(event["id"])
                                if len(token_ids) % 128 == 0:
                                    print("PROGRESS", mode, index + 1, len(token_ids), flush=True)
                            if event["event"] == "done":
                                done = event
                    controller.phase = "between_turns"
                    request_wall_ms = (time.time() - begin) * 1000
                    if not done or done["generated"] != len(token_ids) or len(token_ids) != 512:
                        raise RuntimeError("turn did not produce all 512 outputs; incomplete arm")
                    if controller.child.pid != engine_pid:
                        raise RuntimeError("engine identity changed between arms")
                    status = rpc("/api/status?session=" + session)
                    if (status["cache_fixed"] or status["cache_frozen"]
                            or "failed fills 0" not in status["store"]
                            or "P0 failures reserve 0 / submit 0 / IO 0" not in status["planner"]):
                        raise RuntimeError("cache/IO policy or loading-failure gate failed")
                    done.update(label=turn["label"], token_ids=token_ids, request=body,
                                request_start_unix=begin, request_wall_ms=request_wall_ms, power_profile=mode,
                                repetition=metrics(token_ids), engine_pid=engine_pid)
                    outputs.append(done)
                    save(arm / "turns.json", outputs)
                    save(arm / f"turn{index}_status.json", status)
                    print("DONE", mode, index + 1, round(done["decode_ms"] / done["decode_steps"], 3), "raw ms/token", flush=True)
                save(arm / "status_after.json", status)
                print("ARM_COMPLETE", mode, flush=True)
            # Ensure the thermal tail covers the final engine boundary.
            time.sleep(.2)
            data = (out / "thermal.jsonl").read_bytes()
            (out / "thermal_measurement.jsonl").write_bytes(data[:data.rfind(b"\n") + 1])
            save(out / "report.json", report(out))
            complete = True
    finally:
        if server:
            # Existing shutdown logic requires its production profile; use it
            # only after measurement, and record the unmeasured transition.
            if controller:
                controller.phase = "shutdown"
            subprocess.run(["powerprofilesctl", "set", "performance"], check=True)
            time.sleep(1.2)
            if controller:
                controller.stop.set()
                controller.thread.join(timeout=3)
            with (out / "shutdown.log").open("w") as log, \
                    (out / "shutdown_thermal.jsonl").open("w") as thermal, ProfileMonitor() as monitor:
                clean = web_guard.shutdown(server, controller.child if controller else None,
                    controller.paused if controller else False, controller.latch if controller else ThermalLatch(),
                    discover_sensors(), monitor, log, thermal, owned)
            if controller and controller.child:
                controller.child.close()
            if owned:
                owned.close()
            save(out / "completion.json", dict(all_arms_complete=complete, graceful_shutdown=clean,
                                               server_rc=server.returncode, completed_unix=time.time()))
            if not clean:
                raise RuntimeError("temporary web shutdown did not drain cleanly")
        subprocess.run(["powerprofilesctl", "set", original_profile], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
