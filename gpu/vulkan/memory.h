// GPU memory allocation for the two unified-memory paths of design §3.3.
//
//   Path A  DeviceLocalHostVisible: allocate from a DEVICE_LOCAL|HOST_VISIBLE
//           memory type (heap 1, 74 GiB on this machine) and vkMapMemory it.
//           The CPU writes NVMe data straight into GPU memory. CPU *reads* of
//           that mapping are uncached and very slow, so nothing on the CPU side
//           may compute over it.
//   Path B  ExternalMemoryHost: allocate ordinary host memory and import it
//           with VK_EXT_external_memory_host. Capacity is system RAM
//           (~110 GB), and whether the GPU reads it at full rate is the
//           question P-1 answers.
//
// Both paths hand the ExpertStore the same thing: a SlabMemory with a host
// pointer and a device address. That is the whole point of the abstraction --
// store/slab.h never learns which path it got.
//
// Ownership/threading: a MemoryAllocator borrows the Device and owns every
// VkDeviceMemory and VkBuffer it creates. Allocation happens at startup from
// one thread; the resulting mappings are then written by the IoEngine
// completion threads and read by the GPU.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "core/config.h"
#include "core/status.h"
#include "core/types.h"
#include "gpu/vulkan/device.h"
#include "store/slab.h"

namespace deepmoe::gpu {

struct GpuBuffer {
    uint64_t      bytes    = 0;
    void*         host_ptr = nullptr;            // null when not mapped
    DeviceAddress dev_addr = kNoDeviceAddress;
#if defined(DEEPMOE_ENABLE_VULKAN)
    VkBuffer       buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint32_t       memory_type = 0;
#endif
};

class MemoryAllocator {
public:
    MemoryAllocator() = default;
    ~MemoryAllocator();

    MemoryAllocator(const MemoryAllocator&) = delete;
    MemoryAllocator& operator=(const MemoryAllocator&) = delete;

    // `path` may be Auto, in which case the allocator picks A when a
    // DEVICE_LOCAL|HOST_VISIBLE type exists and B otherwise.
    Result<void> init(Device& device, MemoryPath path);
    void         shutdown();
    MemoryPath   path() const { return path_; }

    // Which memory type index the chosen path resolves to, and how big its
    // heap is. `deepmoe info` prints this; P-1 records it per VGM setting.
    Result<MemoryTypeInfo> chosen_memory_type() const;

    // A general device-local buffer (KV cache, pointer table, activations).
    // `host_visible` maps it; `device_address` enables buffer_device_address.
    // TODO(design §7): implement alongside the first kernel that needs it.
    Result<GpuBuffer> allocate(uint64_t bytes, bool host_visible, bool device_address);
    void              free(GpuBuffer& buf);

    // Host-coherent scratch for the router readback of design §7.8 and the
    // expert pointer table of §5.3.
    // TODO(design §5.3, §7.8).
    Result<GpuBuffer> allocate_host_coherent(uint64_t bytes);

    // Path B: wraps an existing 4 KiB-aligned host allocation.
    // minImportedHostPointerAlignment must divide the pointer and the size.
    // TODO(design §3.3 path B).
    Result<GpuBuffer> import_host_memory(void* host_ptr, uint64_t bytes);

    // A SlabBacking that routes through this allocator, so store/slab.h works
    // unchanged on either path.
    // TODO(design §5.3): implement once allocate()/import_host_memory() exist.
    Result<std::unique_ptr<store::SlabBacking>> make_slab_backing();

private:
    Device*    device_ = nullptr;
    MemoryPath path_   = MemoryPath::Auto;
    int32_t    memory_type_index_ = -1;
    std::vector<GpuBuffer> owned_;
};

}  // namespace deepmoe::gpu
