#include <cstring>
#include "gpu/vulkan/dspark_mega.h"
#include "gpu/vulkan/dspark_kernels.h"
#include "runtime/rope.h"
#include "bench/probes/probe_common.h"
using namespace cachedmoe;
int main(int argc,char**argv) {
    probe::Gpu g; auto r=g.create(); if(!r){std::fprintf(stderr,"%s\n",r.error().str().c_str());return 1;}
    if(probe::has_flag(argc,argv,"--rope")||probe::has_flag(argc,argv,"--pipeline-only")) {
        gpu::DsparkRunner serial;gpu::DsparkMegaRunner mega;
        auto r1=serial.create(g.device,g.alloc,g.shader_dir),r2=mega.create(g.device,g.alloc,g.shader_dir);
        if(!r1||!r2){std::fprintf(stderr,"create: %s\n",(!r1?r1.error():r2.error()).str().c_str());return 1;}
        if(probe::has_flag(argc,argv,"--pipeline-only"))return 0;
        auto x=g.alloc.allocate(5*512*4,true,true),y=g.alloc.allocate(5*512*4,true,true),z=g.alloc.allocate(5*512*4,true,true),tab=g.alloc.allocate(5*64*4,true,true);
        if(!x||!y||!z||!tab){std::fprintf(stderr,"rope buffers unavailable\n");return 1;}
        for(uint32_t i=0;i<5*512;++i)static_cast<float*>(x->host_ptr)[i]=float(int(i%67)-31)/8;
        for(uint32_t m=0;m<5;++m){auto t=runtime::rope_table(runtime::rope_for_layer(0,64),65+m);std::memcpy(static_cast<float*>(tab->host_ptr)+m*64,t.data(),256);}
        gpu::DsparkGemvPush push{};push.m=5;push.rows=1;push.k=512;push.x_stride=512;push.y_stride=512;push.rope_dim=64;push.quant=1;
        auto*slots=serial.slots(gpu::DsparkStage::RopeQuant);slots[0]=x->dev_addr;slots[1]=tab->dev_addr;slots[2]=y->dev_addr;
        if(!serial.dispatch_now(gpu::DsparkStage::RopeQuant,&push,sizeof push,1))return 1;
        gpu::DsparkMegaOp op;op.kind=2;std::memcpy(op.push.data(),&push,sizeof push);op.ptr[0]=x->dev_addr;op.ptr[1]=tab->dev_addr;op.ptr[2]=z->dev_addr;
        auto r=mega.run(std::span(&op,1));if(!r){std::fprintf(stderr,"%s\n",r.error().str().c_str());return 1;}
        uint32_t bad=0;for(uint32_t i=0;i<5*512;++i)if(static_cast<float*>(y->host_ptr)[i]!=static_cast<float*>(z->host_ptr)[i]){if(bad<12)std::printf("i=%u x=%.6f serial=%.6f mega=%.6f\n",i,static_cast<float*>(x->host_ptr)[i],static_cast<float*>(y->host_ptr)[i],static_cast<float*>(z->host_ptr)[i]);++bad;}
        std::printf("rope bad=%u / 2560\n",bad);
        g.alloc.free(*tab);g.alloc.free(*z);g.alloc.free(*y);g.alloc.free(*x);return bad?2:0;
    }
    uint32_t groups=uint32_t(probe::arg_u64(argc,argv,"--groups",20));
    if(groups==0||groups>40){std::fprintf(stderr,"use 1..40 resident workgroups\n");return 1;}
    gpu::Pipeline pipe;gpu::PipelineLayoutSpec spec{2,12}; gpu::PipelineSpec ps;ps.subgroup_size=32;
    r=pipe.create(g.device,g.shader_dir+"/dspark_grid.spv",spec,ps);if(!r){std::fprintf(stderr,"%s\n",r.error().str().c_str());return 1;}
    auto c=g.alloc.allocate((4+groups)*4,true,false),d=g.alloc.allocate(groups*256*4,true,false);if(!c||!d)return 1;
    std::memset(c->host_ptr,0,c->bytes);std::memset(d->host_ptr,0,d->bytes);
    gpu::DescriptorPool desc;r=desc.create(g.device,1,2);if(!r)return 1;
    auto set=desc.allocate(pipe,{{0,0,c->bytes,c->buffer},{1,0,d->bytes,d->buffer}});if(!set)return 1;
    uint32_t push[3]{groups,128,200000};
    r=g.cmd.begin();if(!r)return 1;(void)g.cmd.bind(pipe,*set);(void)g.cmd.push(pipe,push,sizeof push);(void)g.cmd.dispatch(groups);(void)g.cmd.end();
    auto seconds=probe::best_seconds(g,g.cmd,1);if(!seconds){std::fprintf(stderr,"%s\n",seconds.error().str().c_str());return 1;}
    auto*out=static_cast<uint32_t*>(c->host_ptr);uint32_t phases=128;for(uint32_t i=0;i<groups;++i)phases=std::min(phases,out[4+i]);
    std::printf("groups=%u phases=%u timeouts=%u stale=%u ms=%.6f\n",groups,phases,out[2],out[3],*seconds*1000);
    bool ok=phases==128&&out[2]==0&&out[3]==0;g.alloc.free(*d);g.alloc.free(*c);return ok?0:2;
}
