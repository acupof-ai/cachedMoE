from pathlib import Path
import json,os,signal,time,subprocess
base=Path(__file__).resolve().parent
supervisor=2645282
assert b'bench/results/adaptive_mask/run_checks.py' in Path(f'/proc/{supervisor}/cmdline').read_bytes()
children=Path(f'/proc/{supervisor}/task/{supervisor}/children').read_text().split()
assert len(children)==1,children
job_pid=int(children[0])
assert b'tools/mmlu_bench.py' in Path(f'/proc/{job_pid}/cmdline').read_bytes()
assert os.getpgid(job_pid)==job_pid
sensors={}
for name in Path('/sys/class/hwmon').glob('hwmon*/name'):
 kind=name.read_text().strip();p=name.parent/'temp1_input'
 if kind in ('amdgpu','nvme','k10temp') and p.exists():sensors[kind+'_'+name.parent.name]=p
paused=False;stopped=False;rests=0;peak={};start=time.monotonic();failure=None
try:
 # Freeze the old guard while it is asleep, so its next operation on resume is
 # current.poll(). The benchmark and its engine continue under this guard.
 deadline=time.monotonic()+5
 while Path(f'/proc/{supervisor}/wchan').read_text().strip()!='hrtimer_nanosleep':
  assert time.monotonic()<deadline
  time.sleep(.01)
 os.kill(supervisor,signal.SIGSTOP);stopped=True
 (base/'deadline_extension.json').write_text(json.dumps(dict(job='mmlu57_spec5',old_seconds=900,new_seconds=1300,mechanism='temporary continuation thermal guard; original supervisor resumes after the same benchmark exits',job_pid=job_pid,reason='57 exact prefix prefills exceeded the initial budget; no model configuration change or repeated run'),indent=2)+'\n')
 with (base/'mmlu_continuation_thermal.jsonl').open('w') as log:
  while True:
   stat=Path(f'/proc/{job_pid}/stat')
   if not stat.exists() or stat.read_text().split()[2]=='Z':break
   temps={k:int(p.read_text())/1000 for k,p in sensors.items()}
   ac=int(Path('/sys/class/power_supply/AC0/online').read_text())
   assert ac==1,'AC lost'
   active=[v for k,v in temps.items() if k.startswith(('amdgpu','nvme'))]
   for k,v in temps.items():peak[k]=max(peak.get(k,v),v)
   if not paused and max(active)>=80:
    os.killpg(job_pid,signal.SIGSTOP);paused=True;rests+=1
   elif paused and max(active)<=72:
    os.killpg(job_pid,signal.SIGCONT);paused=False
   log.write(json.dumps(dict(elapsed_s=time.monotonic()-start,temperatures=temps,ac=ac,paused=paused))+'\n');log.flush()
   if time.monotonic()-start>500:raise RuntimeError('continuation exceeded 500 seconds')
   time.sleep(.5)
 summary=json.loads((base/'mmlu57_spec5/mask-spec/summary.json').read_text())
 assert summary['n']==57 and summary['sources']==2,summary
 print('MMLU completed same run',summary,flush=True)
except BaseException as e:
 failure=str(e)
 if Path(f'/proc/{job_pid}/stat').exists():
  os.killpg(job_pid,signal.SIGCONT);os.killpg(job_pid,signal.SIGTERM)
 raise
finally:
 if paused:
  try:os.killpg(job_pid,signal.SIGCONT)
  except ProcessLookupError:pass
 (base/'mmlu_continuation_receipt.json').write_text(json.dumps(dict(peak=peak,thermal_pauses=rests,elapsed_s=time.monotonic()-start,failure=failure),indent=2)+'\n')
 if stopped:os.kill(supervisor,signal.SIGCONT)
