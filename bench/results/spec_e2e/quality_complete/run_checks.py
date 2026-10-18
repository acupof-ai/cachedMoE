from pathlib import Path
import os,sys,json,subprocess,signal,time,datetime
base=Path(__file__).resolve().parent
plan=json.loads((base/'jobs.json').read_text());os.chdir(plan['cwd'])
env=dict(os.environ);env.update(DEEPMOE_MODEL_DIR='/home/chenkailun/models/DeepSeek-V4.1-Flash',DEEPMOE_LONGCTX_DIR=str(Path.cwd()/'traces/longctx'),DEEPMOE_MIRROR_AUTO='0',DEEPMOE_DSPARK_MEGA='0');env.pop('DEEPMOE_MODEL_MIRRORS',None)
sensors={}
for name in Path('/sys/class/hwmon').glob('hwmon*/name'):
 kind=name.read_text().strip();temp=name.parent/'temp1_input'
 if temp.exists() and kind in ['amdgpu','nvme','k10temp']:sensors[kind]=temp
ac=Path('/sys/class/power_supply/AC0/online')
def read():return {k:int(p.read_text())/1000 for k,p in sensors.items()}|{'ac':int(ac.read_text())}
def profile():return subprocess.check_output(['powerprofilesctl','get'],text=True).strip()
current=None
original=profile()
def stop(sig,frame):
 if current and current.poll() is None:
  os.killpg(current.pid,signal.SIGCONT);os.killpg(current.pid,signal.SIGTERM)
 raise SystemExit(128+sig)
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
results=[]
try:
 subprocess.run(['powerprofilesctl','set','power-saver'],check=True)
 assert profile()=='power-saver'
 for job in plan['jobs']:
  target_profile=job.get('profile','power-saver')
  subprocess.run(['powerprofilesctl','set',target_profile],check=True)
  assert profile()==target_profile
  name=job['name'];assert not (base/(name+'.log')).exists(),name
  assert subprocess.run(['pgrep','-x','deepmoe'],stdout=subprocess.DEVNULL).returncode==1
  assert subprocess.run(['pgrep','-x','deepmoe_tests'],stdout=subprocess.DEVNULL).returncode==1
  deadline=time.monotonic()+120
  while read()['amdgpu']>60:
   if time.monotonic()>deadline:raise RuntimeError('GPU did not cool below 60 C')
   time.sleep(1)
  start=read();assert start['ac']==1,start
  print('START',name,flush=True);t0=time.monotonic();paused=False;rests=0;power_changed=False;high=dict(start);failure=None;rc=99
  try:
   with (base/(name+'.log')).open('w') as log,(base/(name+'_thermal.jsonl')).open('w') as therm:
    job_env=env|job.get('env',{})
    current=subprocess.Popen(job['command'],env=job_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    while current.poll() is None:
     s=read();s['elapsed_s']=time.monotonic()-t0;therm.write(json.dumps(s)+'\n');therm.flush()
     for k in sensors:high[k]=max(high[k],s[k])
     power_changed |= s['ac']!=1
     if not paused and (s['amdgpu']>=80 or s.get('nvme',0)>=80):
      os.killpg(current.pid,signal.SIGSTOP);paused=True;rests+=1
      if job.get('stop_on_pause'):raise RuntimeError('stopped on first thermal pause')
     elif paused and s['amdgpu']<=72 and s.get('nvme',0)<=72:os.killpg(current.pid,signal.SIGCONT);paused=False
     if time.monotonic()-t0>job['timeout_s']:raise RuntimeError(name+' exceeded smallest-run time budget')
     time.sleep(.5)
    rc=current.wait()
    if '0 case(s) run' in (base/(name+'.log')).read_text():rc=99;failure='zero cases executed'
  except Exception as e:failure=str(e);rc=99
  finally:
   if current and current.poll() is None:
    os.killpg(current.pid,signal.SIGCONT);os.killpg(current.pid,signal.SIGTERM)
    try:current.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(current.pid,signal.SIGKILL);current.wait()
  result=dict(name=name,command=job['command'],env=job.get('env',{}),profile=profile(),rc=rc,failure=failure,elapsed_s=time.monotonic()-t0,start=start,end=read(),peak={k:high[k] for k in sensors},thermal_pauses=rests,ac_changed=bool(power_changed),time=datetime.datetime.now().astimezone().isoformat())
  results.append(result);(base/'check_results.json').write_text(json.dumps(results,indent=2)+'\n')
  print('DONE',json.dumps(result),flush=True)
  print('\n'.join((base/(name+'.log')).read_text().splitlines()[-5:]),flush=True)
  if rc:raise SystemExit(rc)
finally:
 if current and current.poll() is None:
  os.killpg(current.pid,signal.SIGCONT);os.killpg(current.pid,signal.SIGTERM)
  try:current.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(current.pid,signal.SIGKILL);current.wait()
 if profile() in ['power-saver','balanced']:subprocess.run(['powerprofilesctl','set',original],check=True)
 (base/'profile_receipt.json').write_text(json.dumps(dict(original=original,final=profile()))+'\n')
