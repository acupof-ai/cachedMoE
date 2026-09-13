// Vulkan instance / physical device / compute queue, plus the capability set
// design §1.1 depends on.
//
// Required device extensions (all present on gfx1151 / Adrenalin 32.0.31041):
//   VK_EXT_external_memory_host      path B: import host memory, 4 KiB aligned
//   VK_KHR_timeline_semaphore        host-signal "expert resident" (§7.1, §7.8)
//   VK_KHR_buffer_device_address     the expert pointer table (§5.3)
//   VK_KHR_shader_integer_dot_product int8 dot4 GEMV alternative (§6)
//   VK_EXT_subgroup_size_control     Wave32 vs Wave64 A/B (§7.1)
// Optional:
//   VK_KHR_cooperative_matrix        prefill GEMM (§7.13)
//
// Ownership/threading: Device owns the VkInstance, VkDevice and the command
// pools it hands out. Vulkan queue submission is not thread-safe, so the
// runtime keeps exactly one GPU submit thread (docs/architecture.md); the
// Device itself is only mutated during setup and teardown.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/status.h"

#if defined(DEEPMOE_ENABLE_VULKAN)
#include <vulkan/vulkan.h>
#endif

namespace deepmoe::gpu {

// Reported even when Vulkan is compiled out, so `deepmoe info` and the design
// docs share one vocabulary.
struct HeapInfo {
    uint64_t bytes        = 0;
    bool     device_local = false;
    uint32_t index        = 0;
};

struct MemoryTypeInfo {
    uint32_t index        = 0;
    uint32_t heap_index   = 0;
    bool device_local     = false;
    bool host_visible     = false;
    bool host_coherent    = false;
    bool host_cached      = false;
};

struct DeviceCaps {
    std::string device_name;
    uint32_t    api_major = 0, api_minor = 0, api_patch = 0;
    uint32_t    driver_version = 0;
    uint32_t    vendor_id = 0, device_id = 0;

    // design §1.1: these two limits are why the slab pool exists.
    uint64_t max_memory_allocation_size = 0;   // 2 GiB on gfx1151
    uint64_t max_buffer_size            = 0;
    uint64_t max_storage_buffer_range   = 0;   // 4 GiB
    uint32_t max_compute_shared_memory  = 0;   // 32 KiB: the attention tile budget
    uint32_t max_compute_workgroup_invocations = 0;

    uint32_t subgroup_size = 0;                // 64 by default, 32..64 controllable
    uint32_t min_subgroup_size = 0, max_subgroup_size = 0;

    // VK_EXT_external_memory_host: the import alignment must divide 4 KiB for
    // the zero-copy NVMe path of design §9.6 to work.
    uint64_t min_imported_host_pointer_alignment = 0;

    bool external_memory_host   = false;
    bool timeline_semaphore     = false;
    bool buffer_device_address  = false;
    bool integer_dot_product    = false;
    bool subgroup_size_control  = false;
    bool cooperative_matrix     = false;
    bool shader_float16         = false;
    bool shader_int8            = false;
    bool storage_buffer_8bit    = false;

    std::vector<HeapInfo>       heaps;
    std::vector<MemoryTypeInfo> memory_types;

    std::string to_string() const;
    // Everything design §7 and §9 assume is present; returns FailedPrecondition
    // listing what is missing.
    Result<void> check_required() const;
};

struct DeviceOptions {
    bool     enable_validation = false;
    uint32_t preferred_subgroup_size = 32;   // design §7.1 prefers Wave32 for GEMV
    int32_t  physical_device_index = -1;     // -1 = first discrete/integrated GPU
};

class Device {
public:
    Device() = default;
    ~Device();

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    // Creates the instance, picks a physical device and creates one compute
    // queue. Unavailable when no Vulkan loader or no suitable device is found,
    // so the CPU-only paths keep working on a machine without a GPU.
    Result<void> create(const DeviceOptions& opts = {});
    void destroy();
    bool valid() const { return valid_; }

    const DeviceCaps& caps() const { return caps_; }
    uint32_t compute_queue_family() const { return compute_family_; }

    // Enumerates devices without creating one; this is what `deepmoe info`
    // prints and what tools/envcheck reports.
    static Result<std::vector<DeviceCaps>> enumerate(bool enable_validation = false);

#if defined(DEEPMOE_ENABLE_VULKAN)
    VkInstance       instance() const { return instance_; }
    VkPhysicalDevice physical() const { return physical_; }
    VkDevice         handle()   const { return device_; }
    VkQueue          compute_queue() const { return compute_queue_; }
#endif

private:
    bool     valid_ = false;
    DeviceCaps caps_{};
    uint32_t compute_family_ = 0;
#if defined(DEEPMOE_ENABLE_VULKAN)
    VkInstance       instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice         device_   = VK_NULL_HANDLE;
    VkQueue          compute_queue_ = VK_NULL_HANDLE;
#endif
};

}  // namespace deepmoe::gpu
