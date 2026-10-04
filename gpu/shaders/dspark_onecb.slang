// Only GPU-only routing/rounding helpers. Every dispatch has its own pipeline;
// the original GEMV, attention, mHC, head and Markov pipelines are unchanged.
[[vk::binding(0,0)]] ByteAddressBuffer Plan;
uint64_t plan_ptr(uint slot);
#include "dspark_mega_ops.slang"
uint64_t plan_ptr(uint slot){return Plan.Load<uint64_t>(opidx*336+80+slot*8);}
uint plan_push(uint i){return Plan.Load(opidx*336+16+i*4);}
[[vk::constant_id(4)]] const uint Helper=15;
struct OnePush{uint index;};
[[vk::push_constant]] OnePush cb;
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
void main(uint3 gid:SV_GroupID,uint tid:SV_GroupIndex){
    opidx=cb.index;
    switch(Helper){
        case 15:union_routes(tid);break;
        case 16:xq::task(gid,tid);break;
        case 17:up::task(gid,tid);break;
        case 18:hq::task(gid,tid);break;
        case 19:down::task(gid,tid);break;
        case 20:bf_round_copy(gid.x,tid);break;
        case 21:capture(gid.x,tid);break;
        case 22:capture_routes(gid.x,tid);break;
        case 23:{uint i=gid.x*256+tid;if(i<plan_push(0))((float*)plan_ptr(1))[i]=f16tof32(uint(((uint16_t*)plan_ptr(0))[i]));break;}
        case 24:{uint i=gid.x*256+tid;if(i<plan_push(0)){uint* h=(uint*)plan_ptr(0);uint b=(h[6*16*2304/2+i/4]>>(8*(i%4)))&255;
            ((float*)plan_ptr(1))[i]=hq::fp8_e4m3_decode(b)*asfloat(h[6*16*2304*3/4+i/32]);}break;}
        case 25:{uint i=gid.x*256+tid;if(i<plan_push(0))((float*)plan_ptr(1))[i]=gemv::bf16_decode(uint(((uint16_t*)plan_ptr(0))[i]));break;}
        case 26:{uint i=gid.x*256+tid;if(i<plan_push(0)*512){uint dest=((plan_push(1)+i/512)%128)*512+i%512;
            ((uint16_t*)plan_ptr(1))[dest]=(uint16_t)gemv::bf16_encode(((float*)plan_ptr(0))[i]);}break;}
    }
}
