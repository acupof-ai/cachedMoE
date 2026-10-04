#!/usr/bin/env python3
"""Give the fused kernel's BDA intermediates explicit Vulkan memory visibility.
Standalone dispatches acquire visibility at the API barrier. A persistent kernel
must make physical-buffer stores available and loads visible inside the shader.
"""
import sys,struct
from pathlib import Path
f=Path(sys.argv[1]); words=list(struct.unpack('<%dI'%(f.stat().st_size//4),f.read_bytes()))
head=words[:5]; inst=[];i=5
while i<len(words):
 n,op=words[i]>>16,words[i]&65535;inst.append([op,words[i+1:i+n]]);i+=n
ptrtypes={a[0] for op,a in inst if op==32 and a[1]==5349}
types={a[0] for op,a in inst if 19<=op<=39}
value_types={a[1]:a[0] for op,a in inst if len(a)>1 and a[0] in types and not 19<=op<=39}
uint64_type=next(a[0] for op,a in inst if op==21 and a[1:]==[64,0])
# Descriptor-backed schedule loads need no physical-buffer memory operands.
# The only physical uint64 loads read immutable expert address tables. The
# generator marks immutable checkpoint data with alignment 1 or 8; mutable
# intermediates retain their natural alignment 2 or 4.
uint_type=next(a[0] for op,a in inst if op==21 and a[1:]==[32,0])
uint1=next(a[1] for op,a in inst if op==43 and a[0]==uint_type and a[2:]==[1])
consts={a[1]:a[2] for op,a in inst if op==43 and a[0]==uint_type and len(a)==3}
new=[];semantic_ids={}
for op,a in inst:
 if op in (224,225):
  sid=a[-1];scope=a[-2]
  if consts.get(scope)==1:
   value=consts[sid]|0x6000
   if value not in semantic_ids:
    semantic_ids[value]=head[3];head[3]+=1
   a[-1]=semantic_ids[value]
 if op==14:a[1]=3 # Vulkan memory model
 if op==71 and a[1]==23:continue # Coherent is expressed by operands below
 if op in (61,62):
  pointer=a[2] if op==61 else a[0]
  pos=3 if op==61 else 2
  mask=a[pos] if len(a)>pos else 0
  alignment=a[pos+1] if mask & 2 else 0
  readonly=(op==61 and (a[0]==uint64_type or alignment in (1,8)))
  if value_types.get(pointer) in ptrtypes and not readonly:
   pos=3 if op==61 else 2
   mask=a[pos] if len(a)>pos else 0
   if mask & ~3:raise RuntimeError('unexpected existing memory operands')
   tail=a[pos+1:] if len(a)>pos else []
   a=a[:pos]+[mask|0x20|(0x10 if op==61 else 0x8)]+tail+[uint1]
 new.append([op,a])
# Capabilities must precede every non-capability instruction.
at=next(i for i,(op,a) in enumerate(new) if op!=17)
new[at:at]=[[17,[5345]],[17,[5346]]]
# Constants belong before the functions.
at=next(i for i,(op,a) in enumerate(new) if op==54)
new[at:at]=[[43,[uint_type,id,value]] for value,id in semantic_ids.items()]
out=head
for op,a in new:out.extend([((len(a)+1)<<16)|op,*a])
f.write_bytes(struct.pack('<%dI'%len(out),*out))
