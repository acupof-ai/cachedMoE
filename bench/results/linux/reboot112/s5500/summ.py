import json,glob,os,re
for d in sorted(glob.glob(os.path.dirname(__file__)+'/[0-9]_*/')):
    try: t=json.load(open(d+'turns.json'))
    except Exception as e: print(d,'no turns',e); continue
    T=[x for x in t['turns'] if x.get('event')=='done']
    g=sum(x['decode_steps'] for x in T); ms=sum(x['decode_ms'] for x in T)
    hit=sum(x['decode_hit_rate']*x['decode_steps'] for x in T)/g
    st=sum(x['per_token_ms']['nvme_stall']*x['sampled_steps'] for x in T)/sum(x['sampled_steps'] for x in T)
    io=t['status'].get('io','') if isinstance(t['status'],dict) else ''
    eff=re.search(r'eff ([\d.]+) GB/s',io); src=[l.strip() for l in io.split('\n') if 'source' in l or 'src' in l]
    print(f"{os.path.basename(d[:-1]):8s} turns {len(T)} tok/s {g/ms*1000:.4f} hit {hit:.4f} stall {st:.1f} ms eff {eff.group(1) if eff else '-'}")
    for l in src[:4]: print('   ',l)
