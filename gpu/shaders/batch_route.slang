// Finite routing dispatch, followed by normal MoE dispatches. No spin waits.
#include "dspark_plan_layout.h"
[[vk::binding(0,0)]] StructuredBuffer<uint64_t> A;
[vk::constant_id(4)] const uint Copy=0;
// These are the fixed kernel ABI dimensions checked by model/layout.h.
static const uint MaxColumns = DM_ROUTE_MAX_COLUMNS;
static const uint GateRecords = DM_ROUTE_GATE_RECORDS;
static const uint AddressWords = DM_ROUTE_ADDRESS_WORDS;
static const uint InterBlocks = DM_ROUTE_INTER_BLOCKS;
struct RoutePush{uint layer,m,topk,slots,experts,up_groups,hq_groups,copy_count;uint64_t src,dst;};
[[vk::push_constant]] RoutePush pc;
[shader("compute")]
[numthreads(256,1,1)]
void main(uint3 gid:SV_GroupID,uint tid:SV_GroupIndex){
    if(Copy==2){
        const uint i=gid.x*256+tid;if(i>=pc.copy_count)return;
        const uint D=pc.experts,base=(i/D)*4*D+i%D;float* x=(float*)pc.src;
        precise float sum=x[base]+x[base+D];sum=sum+x[base+2*D];sum=sum+x[base+3*D];
        const uint bits=asuint(sum*.25f);
        const uint rounded=((bits&0x7f800000u)==0x7f800000u && (bits&0x007fffffu)!=0)
            ? ((bits>>16)|0x40u) : ((bits+0x7fffu+((bits>>16)&1u))>>16);
        ((float*)pc.dst)[i]=asfloat(rounded<<16);return;
    }
    if(Copy!=0){uint i=gid.x*256+tid;if(i<pc.copy_count)((uint*)pc.dst)[i]=((uint*)pc.src)[i];return;}
    // At most 36 routes. Serial first-occurrence union keeps expert and sum
    // order identical to the CPU reference; parallelising it is a separate
    // numeric/performance experiment, not part of this refactor.
    if(tid!=0)return;
    uint* raw=(uint*)A[0];float* raw_w=(float*)A[1];uint64_t* table=(uint64_t*)A[2];
    uint* ids=(uint*)A[3];uint* list=(uint*)A[4];float* weights=(float*)A[5];
    uint* saved=(uint*)A[6];uint* indirect=(uint*)A[7];
    for(uint i=0;i<MaxColumns*pc.slots;++i)weights[i]=0;
    for(uint i=0;i<pc.m*GateRecords;++i){saved[i]=raw[i];saved[MaxColumns*GateRecords+i]=asuint(raw_w[i]);}
    uint count=0;
    for(uint m=0;m<pc.m;++m)for(uint j=0;j<pc.topk;++j){
        uint e=raw[m*GateRecords+j];float w=raw_w[m*GateRecords+j];
        if(w==0 || e>=pc.experts-1 || table[e*AddressWords]==0)continue;
        uint u=0;for(;u<count&&ids[u]!=e;++u){}
        if(u==count){ids[u]=e;list[u]=u;++count;}
        weights[m*pc.slots+u]+=w;
    }
    ids[count]=(pc.experts-1)|0x80000000u;list[count]=count;
    for(uint m=0;m<pc.m;++m)weights[m*pc.slots+count]=1;
    list[pc.slots]=count+1;
    indirect[0]=pc.up_groups;indirect[1]=count+1;indirect[2]=1;
    indirect[3]=(MaxColumns*(count+1)*InterBlocks+255)/256;indirect[4]=1;indirect[5]=1;
}
