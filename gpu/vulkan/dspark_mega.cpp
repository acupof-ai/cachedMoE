#include "gpu/vulkan/dspark_mega.h"
#include <cstring>
#include <format>
namespace cachedmoe::gpu {
DsparkMegaRunner::~DsparkMegaRunner() {
    pool_.destroy();descriptors_.destroy();pipe_.destroy();
    if(alloc_){if(plan_.valid())alloc_->free(plan_);if(state_.valid())alloc_->free(state_);}
}
Result<void> DsparkMegaRunner::create(Device& device,MemoryAllocator& alloc,const std::string& dir) {
#if defined(CACHEDMOE_ENABLE_VULKAN)
    if(!device.caps().vulkan_memory_model)return fail(Err::Unavailable,"DSpark mega requires Vulkan device memory model");
    if(device.caps().driver_id!=VK_DRIVER_ID_MESA_RADV||device.caps().device_name.find("8060S")==std::string::npos)
        return fail(Err::Unavailable,"DSpark persistent grid is verified only on RADV Strix Halo 8060S");
    device_=&device;alloc_=&alloc;
    PipelineSpec ps;ps.subgroup_size=32;
    if(auto r=pipe_.create(device,dir+"/dspark_mega.spv",{2,12},ps);!r)return r;
    auto p=alloc.allocate(128*sizeof(DsparkMegaOp),true,true);if(!p)return std::unexpected(p.error());plan_=*p;
    auto s=alloc.allocate(128*4,true,false);if(!s)return std::unexpected(s.error());state_=*s;
    if(auto r=descriptors_.create(device,1,2);!r)return r;
    auto set=descriptors_.allocate(pipe_,{{0,0,plan_.bytes,plan_.buffer},{1,0,state_.bytes,state_.buffer}});
    if(!set)return std::unexpected(set.error());set_=*set;
    if(auto r=pool_.create(device);!r)return r;
    ready_=true;return {};
#else
    return fail(Err::Unavailable,"no Vulkan");
#endif
}
Result<void> DsparkMegaRunner::run(std::span<const DsparkMegaOp> ops,uint32_t groups) {
#if defined(CACHEDMOE_ENABLE_VULKAN)
    if(!valid()||ops.empty()||ops.size()>128||groups==0||groups>120)
        return fail(Err::InvalidArgument,"DSpark mega needs 1..128 phases and 1..120 resident workgroups");
    // Hardware-specific opt-in: Vulkan has no cooperative grid launch.
    if(device_->caps().device_name.find("8060S")==std::string::npos)
        return fail(Err::Unavailable,"DSpark persistent grid is verified only on RADV Strix Halo 8060S");
    std::memcpy(plan_.host_ptr,ops.data(),ops.size_bytes());std::memset(state_.host_ptr,0,state_.bytes);
    auto cb=pool_.acquire();if(!cb)return std::unexpected(cb.error());auto cmd=*cb;
    uint32_t push[]{uint32_t(ops.size()),groups,200000};
    if(auto r=cmd.begin();!r)return r;if(auto r=cmd.bind(pipe_,set_);!r)return r;
    if(auto r=cmd.push(pipe_,push,sizeof push);!r)return r;if(auto r=cmd.dispatch(groups);!r)return r;
    if(auto r=cmd.end();!r)return r;if(auto r=submit_and_wait(*device_,cmd);!r)return r;
    auto*s=static_cast<uint32_t*>(state_.host_ptr);
    if(s[2])return fail(Err::Unavailable,std::format("DSpark mega grid timeout: {} workgroups, epoch {} / {}",s[2],s[1],ops.size()));
    for(uint32_t g=0;g<groups;++g)if(s[4+g]!=ops.size())return fail(Err::Internal,"DSpark mega incomplete schedule");
    return {};
#else
    return fail(Err::Unavailable,"no Vulkan");
#endif
}
}
