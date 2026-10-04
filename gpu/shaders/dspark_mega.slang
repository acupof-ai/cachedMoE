// Entire five-position DSpark proposal, one vkCmdDispatch. Intermediates
// reuse GPU allocations; only the initial ring/embedding input and final
// readback cross the host boundary. Fixed resident grid, bounded waits.
[[vk::binding(0,0)]] ByteAddressBuffer Plan;
uint64_t plan_ptr(uint slot);
#include "dspark_mega_ops.slang"
uint64_t plan_ptr(uint slot) {return Plan.Load<uint64_t>(opidx*336+80+slot*8);}
uint plan_push(uint i) {return Plan.Load(opidx*336+16+i*4);}
[[vk::binding(1,0)]] globallycoherent RWStructuredBuffer<uint> State;
struct MegaPush { uint count; uint groups; uint spin_limit; };
[[vk::push_constant]] MegaPush pc;
groupshared uint grid_failed;
// Epoch barriers use atomics for device participants. Unlike the old host
// flag probe, neither participant is a CPU writer with an incoherent cache.
bool grid_barrier(uint phase,uint gid,uint tid) {
    DeviceMemoryBarrierWithGroupSync();
    if(tid==0) {
        uint old;InterlockedAdd(State[0],1,old);
        if(old==pc.groups-1) {
            InterlockedExchange(State[0],0);
            DeviceMemoryBarrier();
            InterlockedExchange(State[1],phase+1);
        }
        uint n=0,epoch;
        do {InterlockedAdd(State[1],0,epoch);++n;}
        while(epoch<phase+1 && n<pc.spin_limit);
        grid_failed=(epoch<phase+1)?1:0;
        if(grid_failed!=0) InterlockedAdd(State[2],1);
        State[4+gid]=phase+1;
    }
    GroupMemoryBarrierWithGroupSync();
    if(grid_failed!=0) return false;
    DeviceMemoryBarrierWithGroupSync();
    return true;
}
void union_routes(uint tid) {
    if(tid!=0)return;
    
    uint* source=(uint*)plan_ptr(0);float* weights=(float*)plan_ptr(1);
    uint* ids=(uint*)plan_ptr(2);uint* list=(uint*)plan_ptr(3);float* route=(float*)plan_ptr(4);
    for(uint i=0;i<5*16;++i)route[i]=0;
    uint count=0;
    for(uint m=0;m<5;++m)for(uint j=0;j<3;++j) {
        uint e=source[m*16+j],s=0;
        for(;s<count && ids[s]!=e;++s){}
        if(s==count){ids[s]=e;list[s]=s;++count;}
        route[m*16+s]+=weights[m*16+j];
    }
    ids[count]=128u|0x80000000u;list[count]=count;
    for(uint m=0;m<5;++m)route[m*16+count]=1;
    *(uint*)plan_ptr(5)=count+1;
}
void bf_round_copy(uint gid,uint tid) {
    
    uint i=gid*256+tid;if(i>=plan_push(0))return;
    float v=gemv::bf16_round(*(float*)(plan_ptr(0)+uint64_t(i)*4));
    if(plan_push(1)!=0) *(uint16_t*)(plan_ptr(1)+uint64_t(i)*2)=(uint16_t)gemv::bf16_encode(v);
    else *(float*)(plan_ptr(1)+uint64_t(i)*4)=v;
}
void capture(uint gid,uint tid) {
    uint count=plan_push(0),i=gid*256+tid;

    if(i<count) *(uint*)(plan_ptr(1)+uint64_t(i)*4)=*(uint*)(plan_ptr(0)+uint64_t(i)*4);
}
void capture_routes(uint gid,uint tid) {
    
    uint i=gid*256+tid;if(i>=plan_push(0))return;uint ix=(i/3)*plan_push(1)+i%3;
    float v=plan_push(2)==1?float(((uint*)plan_ptr(0))[ix]):((float*)plan_ptr(0))[ix];
    ((float*)plan_ptr(1))[i]=v;
}
[shader("compute")]
[numthreads(256,1,1)]
void main(uint3 gid:SV_GroupID,uint tid:SV_GroupIndex) {
    for(uint op=0;op<pc.count;++op) {
        opidx=WaveReadLaneFirst(op);uint kind=Plan.Load(opidx*336),gx=Plan.Load(opidx*336+4),gy=Plan.Load(opidx*336+8);
        if(kind==17)gy=*(uint*)plan_ptr(7);
        // Separate virtual-task loops keep each stage's private temporaries
        // out of other stages' loop backedges.
        switch(kind) {
            case 0:
                for(uint task=gid.x;task<gx*gy;task+=pc.groups) {
                    uint3 v=uint3(task%gx,task/gx,0);
                    gemv::ActQuant=1;gemv::stage_gemv(v.x,tid);
                    GroupMemoryBarrierWithGroupSync();
                }
                break;
            case 1:
                for(uint task=gid.x;task<gx*gy;task+=pc.groups) {
                    uint3 v=uint3(task%gx,task/gx,0);
                    gemv::stage_rmsnorm(v.x,tid);
                    GroupMemoryBarrierWithGroupSync();
                }
                break;
            case 2:
                for(uint task=gid.x;task<gx*gy;task+=pc.groups) {
                    uint3 v=uint3(task%gx,task/gx,0);
                    gemv::stage_ropequant(v.x,tid);
                    GroupMemoryBarrierWithGroupSync();
                }
                break;
            case 3:
                for(uint task=gid.x;task<gx*gy;task+=pc.groups) {
                    uint3 v=uint3(task%gx,task/gx,0);
                    gemv::ActQuant=0;gemv::stage_woa(v.x,tid);
                    GroupMemoryBarrierWithGroupSync();
                }
                break;
            case 4:
                for(uint task=gid.x;task<gx*gy;task+=pc.groups) {
                    uint3 v=uint3(task%gx,task/gx,0);
                    attn::stage_score(v.x,tid);
                    GroupMemoryBarrierWithGroupSync();
                }
                break;
            case 5:
                for(uint task=gid.x;task<gx*gy;task+=pc.groups) {
                    uint3 v=uint3(task%gx,task/gx,0);
                    attn::stage_combine(v.x,tid);
                    GroupMemoryBarrierWithGroupSync();
                }
                break;
            case 6:
                for(uint task=gid.x;task<gx*gy;task+=pc.groups) {
                    uint3 v=uint3(task%gx,task/gx,0);
                    mhc::stage_post(v.x,v.y,tid);
                    GroupMemoryBarrierWithGroupSync();
                }
                break;
            case 7:
                for(uint task=gid.x;task<gx*gy;task+=pc.groups) {
                    uint3 v=uint3(task%gx,task/gx,0);
                    mhc::stage_mix(v.x,v.y,tid);
                    GroupMemoryBarrierWithGroupSync();
                }
                break;
            case 8:
                for(uint task=gid.x;task<gx*gy;task+=pc.groups) {
                    uint3 v=uint3(task%gx,task/gx,0);
                    mhc::stage_final(v.x,v.y,tid);
                    GroupMemoryBarrierWithGroupSync();
                }
                break;
            case 9:
                for(uint task=gid.x;task<gx*gy;task+=pc.groups) {
                    uint3 v=uint3(task%gx,task/gx,0);
                    gate::stage_scores(v.x,v.y,tid);
                    GroupMemoryBarrierWithGroupSync();
                }
                break;
            case 10:
                for(uint task=gid.x;task<gx*gy;task+=pc.groups) {
                    uint3 v=uint3(task%gx,task/gx,0);
                    gate::stage_topk(v.y,tid);
                    GroupMemoryBarrierWithGroupSync();
                }
                break;
            case 11:
                for(uint task=gid.x;task<gx*gy;task+=pc.groups) {
                    uint3 v=uint3(task%gx,task/gx,0);
                    head::stage_gemv(v.x,tid);
                    GroupMemoryBarrierWithGroupSync();
                }
                break;
            case 12:
                for(uint task=gid.x;task<gx*gy;task+=pc.groups) {
                    uint3 v=uint3(task%gx,task/gx,0);
                    markov::stage_markov(v.x,tid);
                    GroupMemoryBarrierWithGroupSync();
                }
                break;
            case 13:
                for(uint task=gid.x;task<gx*gy;task+=pc.groups) {
                    uint3 v=uint3(task%gx,task/gx,0);
                    markov::stage_argmax(tid);
                    GroupMemoryBarrierWithGroupSync();
                }
                break;
            case 14:
                for(uint task=gid.x;task<gx*gy;task+=pc.groups) {
                    uint3 v=uint3(task%gx,task/gx,0);
                    markov::stage_conf(tid);
                    GroupMemoryBarrierWithGroupSync();
                }
                break;
            case 15:
                for(uint task=gid.x;task<gx*gy;task+=pc.groups) {
                    uint3 v=uint3(task%gx,task/gx,0);
                    union_routes(tid);
                    GroupMemoryBarrierWithGroupSync();
                }
                break;
            case 16:
                for(uint task=gid.x;task<gx*gy;task+=pc.groups) {
                    uint3 v=uint3(task%gx,task/gx,0);
                    xq::task(v,tid);
                    GroupMemoryBarrierWithGroupSync();
                }
                break;
            case 17:
                for(uint task=gid.x;task<gx*gy;task+=pc.groups) {
                    uint3 v=uint3(task%gx,task/gx,0);
                    up::task(v,tid);
                    GroupMemoryBarrierWithGroupSync();
                }
                break;
            case 18:
                for(uint task=gid.x;task<gx*gy;task+=pc.groups) {
                    uint3 v=uint3(task%gx,task/gx,0);
                    hq::task(v,tid);
                    GroupMemoryBarrierWithGroupSync();
                }
                break;
            case 19:
                for(uint task=gid.x;task<gx*gy;task+=pc.groups) {
                    uint3 v=uint3(task%gx,task/gx,0);
                    down::task(v,tid);
                    GroupMemoryBarrierWithGroupSync();
                }
                break;
            case 20:
                for(uint task=gid.x;task<gx*gy;task+=pc.groups) {
                    uint3 v=uint3(task%gx,task/gx,0);
                    bf_round_copy(v.x,tid);
                    GroupMemoryBarrierWithGroupSync();
                }
                break;
            case 21:
                for(uint task=gid.x;task<gx*gy;task+=pc.groups) {
                    uint3 v=uint3(task%gx,task/gx,0);
                    capture(v.x,tid);
                    GroupMemoryBarrierWithGroupSync();
                }
                break;
            case 26:
                for(uint task=gid.x;task<gx*gy;task+=pc.groups) {
                    uint3 v=uint3(task%gx,task/gx,0);
                    {uint i=v.x*256+tid;
                    if(i<plan_push(0)*512){uint dest=((plan_push(1)+i/512)%128)*512+i%512;
                        ((uint16_t*)plan_ptr(1))[dest]=(uint16_t)gemv::bf16_encode(((float*)plan_ptr(0))[i]);}}
                    GroupMemoryBarrierWithGroupSync();
                }
                break;
            case 24:
                for(uint task=gid.x;task<gx*gy;task+=pc.groups) {
                    uint3 v=uint3(task%gx,task/gx,0);
                    {uint n=plan_push(0),i=v.x*256+tid;
                    if(i<n){uint* h=(uint*)plan_ptr(0);uint b=(h[6*16*2304/2+i/4]>>(8*(i%4)))&255;
                        ((float*)plan_ptr(1))[i]=hq::fp8_e4m3_decode(b)*asfloat(h[6*16*2304*3/4+i/32]);}}
                    GroupMemoryBarrierWithGroupSync();
                }
                break;
            case 23:
                for(uint task=gid.x;task<gx*gy;task+=pc.groups) {
                    uint3 v=uint3(task%gx,task/gx,0);
                    {uint n=plan_push(0),i=v.x*256+tid;
                    if(i<n)((float*)plan_ptr(1))[i]=f16tof32(uint(((uint16_t*)plan_ptr(0))[i]));}
                    GroupMemoryBarrierWithGroupSync();
                }
                break;
            case 22:
                for(uint task=gid.x;task<gx*gy;task+=pc.groups) {
                    uint3 v=uint3(task%gx,task/gx,0);
                    capture_routes(v.x,tid);
                    GroupMemoryBarrierWithGroupSync();
                }
                break;
            case 25:
                for(uint task=gid.x;task<gx*gy;task+=pc.groups) {
                    uint i=task*256+tid;
                    if(i<plan_push(0))((float*)plan_ptr(1))[i]=gemv::bf16_decode(uint(((uint16_t*)plan_ptr(0))[i]));
                    GroupMemoryBarrierWithGroupSync();
                }
                break;
        }
        if(!grid_barrier(op,gid.x,tid))return;
    }
}
