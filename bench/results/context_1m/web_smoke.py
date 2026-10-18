import json, time, urllib.request
from pathlib import Path
out=Path(__file__).resolve().parent
base="http://127.0.0.1:8080"
def get(path):
    with urllib.request.urlopen(base+path,timeout=5) as r:return json.load(r)
def post(path,body):
    req=urllib.request.Request(base+path,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    return urllib.request.urlopen(req,timeout=120)
limit=time.monotonic()+75
while True:
    try:
        cfg=get("/api/config");break
    except (OSError,TimeoutError):
        if time.monotonic()>limit:raise
        time.sleep(1)
assert cfg["max_context"]==cfg["ceiling"]==1048576,cfg
assert cfg["ready"]["speculation"]["enabled"]
assert cfg["ready"]["speculation"]["draft_tokens"]==5
print("READY: native 1M, k5/top4",flush=True)
body=dict(session="ctx1m-smoke-20261005",text="Ignore padding. Reply exactly OK.\n<padding>"+" filler"*4140+"</padding>\nReply exactly OK.",think=False,reasoning_effort=75,max_tokens=12,temperature=0,top_p=.95,seed=1048576)
with post("/api/preview",body) as r:preview=json.load(r)
assert not preview["over"] and preview["prompt_tokens"]>4096,preview
assert preview["max_context"]==1048576
print("PREVIEW:",json.dumps(preview),flush=True)
events=[];t0=time.monotonic()
with post("/api/chat",body) as r:
    for line in r:
        if line.startswith(b"data:"):
            ev=json.loads(line[5:]);events.append(ev)
            if ev["event"] in ("done","error"):print(json.dumps(ev),flush=True)
            assert ev["event"]!="error",ev
assert any(ev["event"]=="done" for ev in events),events[-3:]
status=get("/api/status?session="+body["session"])
assert status["cache_fixed"] and status["cache_frozen"],status
(out/"web_smoke.json").write_text(json.dumps(dict(config=cfg,preview=preview,events=events,status=status,elapsed_s=time.monotonic()-t0),indent=2)+"\n")
print("WEB SMOKE PASS",flush=True)
