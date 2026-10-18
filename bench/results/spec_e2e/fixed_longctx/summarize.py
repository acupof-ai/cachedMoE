from pathlib import Path
import json
base=Path(__file__).resolve().parent
root=base/'needle4k17k'
d=json.loads((root/'turns.json').read_text())
events=[json.loads(l) for l in (root/'events.jsonl').read_text().splitlines()]
rows=[];tokens=[];text=''
for r in events:
 if r.get('event')=='token':tokens.append(r['id']);text+=r['text']
 if r.get('event')=='done':
  n=r['session'];ref=json.loads((Path('tests/data/longctx')/n/'index.json').read_text())['greedy_tokens'];m=0
  for a,b in zip(tokens,ref):
   if a!=b:break
   m+=1
  rows.append({'session':n,'prompt_tokens':r['prompt_tokens'],'emitted_ids':tokens,'reference_ids':ref,'matching_greedy_prefix':m,'output':text,'needle_recovered':'kestrel-4471-amber' in text,'prefill_ms':r['prefill_ms'],'speculation':r['speculation']});tokens=[];text=''
c=[json.loads(l) for l in (base/'cycles.jsonl').read_text().splitlines()];c=[r for r in c if 'verify_host_ms' in r]
q={'protocol':'independent raw prompt sessions at 4133/17010 tokens; exact prefill; fixed mask + DSpark k2 top4; T0 seed42; 9 outputs per prompt; 4900 total slots to leave workspace for 17K prefill','sources':d['server']['sources'],'cache_slots':d['server']['cache_slots'],'cache_fixed':d['status']['cache_fixed'],'cache_frozen':d['status']['cache_frozen'],'store':d['status']['store'],'rows':rows,'all_cycles_fixed_frozen':all(r['cache_fixed'] and r['cache_frozen'] for r in c),'load_failures':{k:sum(r['load_failures'][k] for r in c) for k in c[0]['load_failures']},'both_needles_recovered':len(rows)==2 and all(r['needle_recovered'] for r in rows),'limitations':'two retrieval prompts and nine-token free generations; does not establish lossless long-context quality or teacher-forced bitwise identity'}
assert len(rows)==2 and q['cache_fixed'] and q['cache_frozen'] and q['sources']==2
(base/'quality.json').write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n');print(json.dumps(q,ensure_ascii=False,indent=2))
