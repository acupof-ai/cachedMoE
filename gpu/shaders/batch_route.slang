// Finite routing dispatch, followed by normal MoE dispatches. No spin waits.
[[vk::binding(0,0)]] StructuredBuffer<uint64_t> A;
[vk::constant_id(4)] const uint Copy=0;
struct RoutePush{uint layer,m,topk,slots,experts,up_groups,hq_groups,copy_count;uint64_t src,dst;};
[[vk::push_constant]] RoutePush pc;
[shader("compute")]
[numthreads(256,1,1)]
void main(uint3 gid:SV_GroupID,uint tid:SV_GroupIndex){
    if(Copy!=0){uint i=gid.x*256+tid;if(i<pc.copy_count)((uint*)pc.dst)[i]=((uint*)pc.src)[i];return;}
    if(tid!=0)return;
    uint* raw=(uint*)A[0];float* raw_w=(float*)A[1];uint64_t* table=(uint64_t*)A[2];
    uint* ids=(uint*)A[3];uint* list=(uint*)A[4];float* weights=(float*)A[5];
    uint* saved=(uint*)A[6];uint* indirect=(uint*)A[7];
    for(uint i=0;i<6*pc.slots;++i)weights[i]=0;
    for(uint i=0;i<pc.m*16;++i){saved[i]=raw[i];saved[6*16+i]=asuint(raw_w[i]);}
    uint count=0;
    for(uint m=0;m<pc.m;++m)for(uint j=0;j<pc.topk;++j){
        uint e=raw[m*16+j];float w=raw_w[m*16+j];
        if(w==0 || e>=pc.experts-1 || table[e*6]==0)continue;
        uint u=0;for(;u<count&&ids[u]!=e;++u){}
        if(u==count){ids[u]=e;list[u]=u;++count;}
        weights[m*pc.slots+u]+=w;
    }
    ids[count]=(pc.experts-1)|0x80000000u;list[count]=count;
    for(uint m=0;m<pc.m;++m)weights[m*pc.slots+count]=1;
    list[pc.slots]=count+1;
    indirect[0]=pc.up_groups;indirect[1]=count+1;indirect[2]=1;
    indirect[3]=(6*(count+1)*72+255)/256;indirect[4]=1;indirect[5]=1;
}
