#pragma once
#include <array>
#include <cstddef>
#include <span>
#include "gpu/vulkan/cmdbuf.h"
#include "gpu/vulkan/descriptor.h"
#include "gpu/vulkan/memory.h"
namespace deepmoe::gpu {
// Immutable schedule: every phase sees its own arguments and addresses.
struct DsparkMegaOp {
    uint32_t kind=0, gx=1, gy=1, reserved=0;
    std::array<uint32_t,16> push{};
    std::array<uint64_t,32> ptr{};
};
static_assert(sizeof(DsparkMegaOp)==336);
static_assert(offsetof(DsparkMegaOp,push)==16 && offsetof(DsparkMegaOp,ptr)==80);
class DsparkMegaRunner {
public:
    ~DsparkMegaRunner();
    Result<void> create(Device&,MemoryAllocator&,const std::string&);
    Result<void> run(std::span<const DsparkMegaOp>,uint32_t groups=120);
    bool valid() const {return ready_;}
private:
    Device* device_=nullptr; MemoryAllocator* alloc_=nullptr;
    bool ready_=false;
    Pipeline pipe_;DescriptorPool descriptors_;CommandPool pool_;
    GpuBuffer plan_,state_;
#if defined(DEEPMOE_ENABLE_VULKAN)
    VkDescriptorSet set_=VK_NULL_HANDLE;
#endif
};
}
