#!/usr/bin/env python3
"""Reuse the tested shader arithmetic inside one persistent entry point.
Only bindings, push-constant access and workgroup scheduling change. LDS is
allocated once and overlaid between phases, not summed across shader files.
"""
import re, sys
from pathlib import Path
src=Path(sys.argv[1]); dest=Path(sys.argv[2])
def expand(name):
    s=(src/name).read_text()
    s=re.sub(r'#include "([^"]+)"',lambda m:expand(m[1]),s)
    return re.sub(r'//[^\n]*','',s)
base={'M':5,'LanesPerRow':32,'RowsPerWgUnused':8,'SubgroupSize':32,'Stage':0,
      'ActQuant':1,'RowsPerLane':1,'HeadsPerWg':1,'WaveReduce':1,'KSplit':1,'Fp8Arith':0,
      'DecodeMode':3,'HPrecision':0,'XMode':0,'HQuant':3,'Fp8Slots':1,'StaticM':0}
files=[('gemv','dspark_gemv.slang','DsparkGemvPush'),('attn','dspark_attn.slang','DsparkAttnPush'),
       ('mhc','mgt1_mhc.slang','MhcPush'),('gate','mgt1_gate.slang','GatePush'),
       ('head','mgt1_head.slang','Mgt1HeadPush'),('markov','dspark_head.slang','DsparkHeadPush'),
       ('up','moe_gateup.slang','GateUpPush'),('hq','moe_hquant.slang','HQuantPush'),
       ('down','moe_down.slang','DownPush'),('xq','moe_xact.slang','XActPush')]
resources={
 'ExpertAddr':('uint64_t',0),'Ids':('uint',1),'SlotList':('uint',2),
 'RouteW':('float',3),'X':('uint4',4),'XU':('uint',4),'H16':('half',5),
 'H32':('float',5),'HU':('uint',5),'H':('uint',5),'Y':('float',6)}
entries={
 'gemv':['stage_gemv','stage_rmsnorm','stage_ropequant','stage_woa'],
 'attn':['stage_score','stage_combine'],'mhc':['stage_post','stage_mix','stage_final'],
 'gate':['stage_scores','stage_topk'],'head':['stage_gemv'],
 'markov':['stage_markov','stage_argmax','stage_conf'],
 'up':['main'],'hq':['main'],'down':['main'],'xq':['main']}
def phase_layout(s, arrays, roots):
    assert base['WaveReduce']==1 and base['LanesPerRow']==base['SubgroupSize']==32
    funcs={}
    for match in re.finditer(r'^\s*(?:\w+\s+)+(\w+)\s*\([^;{}]*\)\s*\{',s,re.M):
        at=match.end();end=at;depth=1
        while depth:depth+=(s[end]=='{')-(s[end]=='}');end+=1
        funcs[match[1]]=s[at:end-1]
    def uses(name, seen):
        if name in seen:return set()
        seen.add(name);body=funcs[name]
        found={a for a in arrays if re.search(r'\b'+a+r'\[',body)}
        # All fused stages use WaveReduce=1. The other branch of row_reduce
        # uses gRed; its compile-time dead access must not prolong its lifetime.
        if name=='row_reduce':found.discard('gRed')
        for call in re.findall(r'\b(\w+)\s*\(',body):
            if call in funcs:found|=uses(call,seen)
        return found
    phases=[uses(root,set()) for root in roots]
    active=set().union(*phases);layout={}
    for name in sorted(active,key=lambda n:(-arrays[n][1],n)):
        conflicts=[other for other in layout if any(name in p and other in p for p in phases)]
        offset=0;size=arrays[name][1]
        while True:
            overlaps=[layout[n]+arrays[n][1] for n in conflicts
                      if offset<layout[n]+arrays[n][1] and layout[n]<offset+size]
            if not overlaps:break
            offset=max(overlaps)
        layout[name]=offset
    words=max((layout[n]+arrays[n][1] for n in active),default=1)
    return {name:(typ,layout.get(name,0)) for name,(typ,size) in arrays.items()},words
chunks=[]; max_words=0
for ns,file,push in files:
    s=expand(file); vals=base.copy()
    # Immutable checkpoint weights are at least eight-byte aligned. Keep
    # their loads cacheable; only GPU-written activations need device visibility.
    immutable16='uint4 immutable16(uint64_t a) { return loadAligned<8>((uint4*)a); }\n'
    if ns=='gemv':
        s=s.replace('load16(wbase + off)', 'immutable16(wbase + off)').replace('load16(wbase + off + 16)', 'immutable16(wbase + off + 16)')
    if ns in ('head','gate','markov'):
        s=re.sub(r'load16\((wrow|w|Ptr\[kMkW\])',r'immutable16(\1',s)
    if ns in ('up','down'):
        s=s.replace('return *(uint4*)addr;', 'return loadAligned<8>((uint4*)addr);')
    if ns=='down':
        # The draft consumes BF16-rounded output. Round in the producer's
        # final store instead of another global read/write/grid barrier.
        old='Y[oi] = ((pc.flags & 1u) != 0) ? (Y[oi] + s) : s;'
        assert old in s
        s=s.replace(old,'Y[oi] = gemv::bf16_round(((pc.flags & 1u) != 0) ? (Y[oi] + s) : s);')
    s=immutable16+s
    if ns in ('up','hq','down','xq'): vals['M']=6
    def constant(m):
        key=m[1];v=vals[key]
        if key=='ActQuant' and ns=='gemv': return 'static uint ActQuant;'
        return f'static const uint {key} = {v};'
    s=re.sub(r'\[vk::constant_id\(\d+\)\]\s*const uint\s+(\w+)\s*=\s*[^;]+;',constant,s)
    env=vals.copy();lds={}
    for line in s.splitlines():
        m=re.match(r'\s*static const uint\s+(\w+)\s*=\s*(.*);',line)
        if m:
            try:
                ex=re.sub(r'(\d+)u\b',r'\1',m[2])
                env[m[1]]=eval(ex,{'__builtins__':{}},env)
            except Exception: pass
        m=re.match(r'\s*groupshared\s+(float|uint|int)\s+(\w+)\[([^]]+)\];',line)
        if m:
            ex=re.sub(r'(\d+)u\b',r'\1',m[3])
            try: size=int(eval(ex,{'__builtins__':{}},env))
            except Exception:
                # All conditional allocations in the fixed XMode=0 path.
                size=1 if m[2] in ('sX','sXS','sHq','sI8') else 256
            t,name=m[1],m[2];lds[name]=(t,size)
    lds,words=phase_layout(s,lds,entries[ns]);max_words=max(max_words,words)
    # The largest simultaneously live set determines the allocation, across
    # both shader files and entry-point phases. Scalar types share raw bits.
    s=re.sub(r'^\s*groupshared[^;]+;', '',s,flags=re.M)
    for name,(t,off) in lds.items():
        arr={'float':'MegaF','uint':'MegaURef','int':'MegaIRef'}[t]
        pieces=[];at=0
        for match in re.finditer(r'\b'+name+r'\[',s):
            if match.start()<at:continue
            end=match.end();depth=1
            while depth:
                depth += (s[end]=='[')-(s[end]==']');end+=1
            index=s[match.end():end-1]
            pieces.extend([s[at:match.start()],f'{arr}[{off} + ({index})]']);at=end
        pieces.append(s[at:]);s=''.join(pieces)
    for marker,cast,typ in [('MegaURef','asuint','uint'),('MegaIRef','asint','int')]:
        pieces=[];at=0
        for match in re.finditer(r'\b'+marker+r'\[',s):
            if match.start()<at:continue
            end=match.end();depth=1
            while depth:depth+=(s[end]=='[')-(s[end]==']');end+=1
            index=s[match.end():end-1]
            assign=re.match(r'\s*=\s*(?!=)',s[end:])
            if assign:
                rhs_start=end+assign.end();semi=s.index(';',rhs_start)
                replacement=f'MegaF[{index}] = asfloat(({typ})({s[rhs_start:semi]}));'
                pieces.extend([s[at:match.start()],replacement]);at=semi+1
            else:
                pieces.extend([s[at:match.start()],f'{cast}(MegaF[{index}])']);at=end
        pieces.append(s[at:]);s=''.join(pieces)
        # A store's RHS can itself read the same array; the pass above skips
        # that RHS after consuming its assignment.
        while re.search(r'\b'+marker+r'\[',s):
            match=re.search(r'\b'+marker+r'\[',s);end=match.end();depth=1
            while depth:depth+=(s[end]=='[')-(s[end]==']');end+=1
            s=s[:match.start()]+f'{cast}(MegaF[{s[match.end():end-1]}])'+s[end:]
    s=re.sub(r'^\s*\[\[vk::binding\([^\n]+\n','',s,flags=re.M)
    s=re.sub(r'\[\[vk::push_constant\]\]\s*\w+\s+pc;', '',s)
    macros=[f'#define pc Plan.Load<{push}>(opidx * 336 + 16)']
    if ns in ('up','hq','down','xq'):
        for name,(typ,slot) in resources.items():
            if re.search(r'\b'+name+r'\[',s): macros.append(f'#define {name} (({typ}*)Ptr[{slot}])')
        s=s.replace('pc.list_count','(*(uint*)Ptr[7])')
    if ns=='xq':
        # Original xact computes a single column; cover all five here.
        s=s.replace('pc.k / kBlockK','5 * pc.k / kBlockK')
    s=re.sub(r'\[shader\("compute"\)\]\s*\[numthreads\(256,\s*1,\s*1\)\]', '',s)
    s=s.replace('void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex)', 'void task(uint3 gid, uint tid)')
    # Stage-specific entry points are called directly for the non-MoE files.
    if ns not in ('up','hq','down','xq'):
        s=s[:s.rfind('void task(')]
    # Schedule metadata is descriptor-backed, so the driver can use scalar
    # loads instead of carrying push fields and tensor addresses in VGPRs.
    pieces=[];at=0
    for match in re.finditer(r'\bPtr\[',s):
        end=match.end();depth=1
        while depth:depth += (s[end]=='[')-(s[end]==']');end+=1
        pieces.extend([s[at:match.start()],f'plan_ptr({s[match.end():end-1]})']);at=end
    pieces.append(s[at:]);s=''.join(pieces)
    macros=[m.replace('Ptr[','plan_ptr(').replace('])', '))') for m in macros]
    undefs=['#undef '+re.match(r'#define (\w+)',m)[1] for m in macros]
    if '#define kLiveM' in s: undefs.append('#undef kLiveM')
    chunks.append('\n'.join([f'namespace {ns} {{',*macros,s,*undefs,'}']))
header='// Generated by tools/fuse_dspark.py; do not edit. Arithmetic comes from the original shaders.\n'
header+='static uint opidx;\n'
header+=f'groupshared float MegaF[{max_words}];\n'
dest.write_text(header+'\n'.join(chunks))
print('mega LDS words:',max_words, 'bytes=',max_words*4)
