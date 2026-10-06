#pragma once
#include <map>
#include "gpu/vulkan/dspark_mega.h"
#include "gpu/vulkan/dspark_kernels.h"
#include "gpu/vulkan/decode_kernels.h"
namespace deepmoe::gpu {
// Independent pipelines, immutable arguments, one queue submission. No grid
// synchronisation or persistent kernel; the device schedules each dispatch.
class DsparkOneCbRunner {
  public:
    ~DsparkOneCbRunner();
    Result<void> create(Device &, MemoryAllocator &, const std::string &);
    Result<void> run(std::span<const DsparkMegaOp>, DsparkRunner &, MgtRunner &,
                     std::span<const std::string> labels = {}, bool profile = false);
    bool valid() const { return ready_; }
    std::map<std::string, double> timing_ms;

  private:
    Device *device_ = nullptr;
    MemoryAllocator *alloc_ = nullptr;
    bool ready_ = false;
    std::array<Pipeline, 12> helpers_;
    DescriptorPool descriptors_;
    CommandPool pool_;
    CommandBuffer cmd_;
    QueryPool queries_;
    GpuBuffer plan_, args_;
    uint64_t stride_ = 256;
#if defined(CACHEDMOE_ENABLE_VULKAN)
    VkDescriptorSet helper_set_ = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, 128> arg_sets_{};
#endif
};
} // namespace deepmoe::gpu
