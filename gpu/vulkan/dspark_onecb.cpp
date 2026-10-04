#include "gpu/vulkan/dspark_onecb.h"
#include <cstring>
namespace deepmoe::gpu {
DsparkOneCbRunner::~DsparkOneCbRunner(){
    pool_.destroy();descriptors_.destroy();
    if(alloc_){if(plan_.valid())alloc_->free(plan_);if(args_.valid())alloc_->free(args_);}
}
Result<void> DsparkOneCbRunner::create(Device& device,MemoryAllocator& alloc,const std::string& dir){
#if defined(DEEPMOE_ENABLE_VULKAN)
    device_=&device;alloc_=&alloc;
    for(uint32_t k=15;k<=26;++k){PipelineSpec ps;ps.subgroup_size=32;ps.extra={k};
        if(auto r=helpers_[k-15].create(device,dir+"/dspark_onecb.spv",{1,64},ps);!r)return r;}
    // minStorageBufferOffsetAlignment <= 256 is checked, rather than assumed.
    VkPhysicalDeviceProperties props{};vkGetPhysicalDeviceProperties(device.physical(),&props);
    stride_=std::max<uint64_t>(256,props.limits.minStorageBufferOffsetAlignment);
    auto p=alloc.allocate(128*sizeof(DsparkMegaOp),true,true);if(!p)return std::unexpected(p.error());plan_=*p;
    auto a=alloc.allocate(128*stride_,true,true);if(!a)return std::unexpected(a.error());args_=*a;
    if(auto r=descriptors_.create(device,129,129);!r)return r;
    auto set=descriptors_.allocate(helpers_[0],{{0,0,plan_.bytes,plan_.buffer}});
    if(!set)return std::unexpected(set.error());helper_set_=*set;
    // All original runners use the same layout: storage buffer 0, 64 B push.
    for(uint32_t i=0;i<128;++i){auto s=descriptors_.allocate(helpers_[0],{{0,i*stride_,256,args_.buffer}});
        if(!s)return std::unexpected(s.error());arg_sets_[i]=*s;}
    if(auto r=pool_.create(device);!r)return r;auto cb=pool_.acquire();if(!cb)return std::unexpected(cb.error());cmd_=*cb;
    if(device.caps().timestamp_valid_bits){if(auto r=queries_.create(device,256);!r)return r;}
    ready_=true;return {};
#else
    return fail(Err::Unavailable,"no Vulkan");
#endif
}
Result<void> DsparkOneCbRunner::run(std::span<const DsparkMegaOp> ops,DsparkRunner& ds,MgtRunner& mgt,
                                   std::span<const std::string> labels,bool profile){
#if defined(DEEPMOE_ENABLE_VULKAN)
    if(!ready_||ops.empty()||ops.size()>128||(!labels.empty()&&labels.size()!=ops.size()))
        return fail(Err::InvalidArgument,"DSpark one-CB schedule needs 1..128 phases and matching labels");
    if(profile&&!device_->caps().timestamp_valid_bits)return fail(Err::Unavailable,"GPU timestamps unavailable");
    std::memcpy(plan_.host_ptr,ops.data(),ops.size_bytes());
    for(uint32_t i=0;i<ops.size();++i)std::memcpy(static_cast<std::byte*>(args_.host_ptr)+i*stride_,ops[i].ptr.data(),256);
    pool_.reset();timing_ms.clear();
#define CB_TRY(e) do{auto r=(e);if(!r)return r;}while(0)
    CB_TRY(cmd_.begin());if(profile)CB_TRY(cmd_.reset_queries(queries_,0,uint32_t(ops.size())*2));
    for(uint32_t i=0;i<ops.size();++i){const auto& o=ops[i];
        if(profile)CB_TRY(cmd_.write_timestamp(queries_,2*i,false));
        if(o.kind<=5||o.kind==12||o.kind==13||o.kind==14){
            auto st=o.kind<=5?DsparkStage(o.kind):o.kind==12?DsparkStage::MarkovBias:o.kind==13?DsparkStage::AddBiasArgmax:DsparkStage::Confidence;
            CB_TRY(ds.record_bound(cmd_,st,o.push.data(),64,o.gx,arg_sets_[i]));
        }else if(o.kind<=11){
            CB_TRY(mgt.record_bound(cmd_,o.reserved&255,MgtStage(o.reserved>>8),o.push.data(),64,o.gx,o.gy,arg_sets_[i]));
        }else if(o.kind<=26){const auto& pipe=helpers_[o.kind-15];
            CB_TRY(cmd_.bind(pipe,helper_set_));CB_TRY(cmd_.push(pipe,&i,4));CB_TRY(cmd_.dispatch(o.gx,o.gy));
        }else return fail(Err::InvalidArgument,"unknown DSpark one-CB operation");
        if(profile)CB_TRY(cmd_.write_timestamp(queries_,2*i+1,true));
        if(i+1<ops.size())CB_TRY(cmd_.barrier());
    }
    CB_TRY(cmd_.end());CB_TRY(submit_and_wait(*device_,cmd_));
    if(profile){auto ticks=queries_.read_range(0,uint32_t(ops.size())*2);if(!ticks)return std::unexpected(ticks.error());
        const auto bits=device_->caps().timestamp_valid_bits;const uint64_t mask=bits>=64?~0ull:(1ull<<bits)-1;
        for(uint32_t i=0;i<ops.size();++i){uint64_t dt=((*ticks)[2*i+1]-(*ticks)[2*i])&mask;
            timing_ms[labels.empty()?std::to_string(ops[i].kind):labels[i]]+=double(dt)*device_->caps().timestamp_period_ns*1e-6;}}
    return {};
#undef CB_TRY
#else
    return fail(Err::Unavailable,"no Vulkan");
#endif
}
}
