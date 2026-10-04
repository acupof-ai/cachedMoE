// Bounded device-only grid barrier. No CPU participant.
[[vk::binding(0,0)]] globallycoherent RWStructuredBuffer<uint> C;
[[vk::binding(1,0)]] globallycoherent RWStructuredBuffer<uint> Data;
struct Push { uint groups; uint rounds; uint spin_limit; };
[[vk::push_constant]] Push pc;
groupshared uint failed;
[shader("compute")]
[numthreads(256,1,1)]
void main(uint3 gid: SV_GroupID, uint tid: SV_GroupIndex) {
    for(uint phase=0;phase<pc.rounds;++phase) {
        Data[gid.x*256+tid] = phase+1;
        DeviceMemoryBarrierWithGroupSync();
        if(tid==0) {
            uint old; InterlockedAdd(C[0],1,old);
            if(old==pc.groups-1) {
                InterlockedExchange(C[0],0);
                DeviceMemoryBarrier();
                InterlockedExchange(C[1],phase+1);
            }
            uint spins=0, epoch;
            do { InterlockedAdd(C[1],0,epoch); ++spins; }
            while(epoch<phase+1 && spins<pc.spin_limit);
            failed=(epoch<phase+1)?1:0;
            if(failed) InterlockedAdd(C[2],1);
        }
        GroupMemoryBarrierWithGroupSync();
        if(failed) return;
        DeviceMemoryBarrierWithGroupSync();
        if(tid==0) {
            for(uint g=0;g<pc.groups;++g)
                if(Data[g*256] < phase+1) InterlockedAdd(C[3],1);
            C[4+gid.x]=phase+1;
        }
        // The next phase writes larger values: payload checks use >=.
    }
}
