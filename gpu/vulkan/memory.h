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
// Measured answer (docs/kernel_p1.md, bench/results/bw_matrix.csv): both paths
// read at the same rate on gfx1151, so the choice is made on capacity, not
// bandwidth.
//
// Ownership/threading: a MemoryAllocator borrows the Device and owns every
// VkDeviceMemory, VkBuffer and host allocation it creates. Allocation happens
// at startup from one thread; the resulting mappings are then written by the
// IoEngine completion threads and read by the GPU.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
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
    uint32_t      memory_type = 0;
    // Path B bookkeeping: the host allocation the buffer was imported from, and
    // whether it came back as large pages. `host_alloc` is null for path A and
    // for an import of memory the caller owns.
    void*         host_alloc  = nullptr;
    bool          large_pages = false;
    bool          imported    = false;
#if defined(DEEPMOE_ENABLE_VULKAN)
    VkBuffer       buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
#endif
    bool valid() const { return bytes != 0; }
};

// What a host allocation for path B cost and whether large pages were granted.
// bench/bw_matrix reports this; design §3.3 asks for "large pages if the
// privilege allows, report if not".
struct HostAllocInfo {
    void*    ptr         = nullptr;
    uint64_t bytes       = 0;
    bool     large_pages = false;
    std::string note;         // why large pages were refused, when they were
};

// Allocates page-aligned host memory (VirtualAlloc on Windows, aligned new
// elsewhere). `try_large_pages` attempts MEM_LARGE_PAGES after enabling
// SeLockMemoryPrivilege and silently falls back, recording why in `note`.
Result<HostAllocInfo> alloc_host_pages(uint64_t bytes, bool try_large_pages);
void                  free_host_pages(const HostAllocInfo& info);

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

    // A general buffer. `host_visible` maps it; `device_address` enables
    // buffer_device_address. The memory type is picked by flags, not by the
    // configured path, so the pointer table and activations are allocated the
    // same way whichever expert-cache path is in use.
    Result<GpuBuffer> allocate(uint64_t bytes, bool host_visible, bool device_address);
    void              free(GpuBuffer& buf);

    // Host-coherent scratch for the router readback of design §7.8 and the
    // expert pointer table of §5.3.
    Result<GpuBuffer> allocate_host_coherent(uint64_t bytes);

    // Allocate from one specific memory type. bench/bw_matrix uses it to
    // measure the DEVICE_LOCAL-only ceiling against the path A type.
    Result<GpuBuffer> allocate_from_type(uint64_t bytes, uint32_t memory_type_index,
                                         bool map, bool device_address);

    // Path B: wraps an existing host allocation. `host_ptr` and `bytes` must
    // both be multiples of minImportedHostPointerAlignment (4096 here).
    Result<GpuBuffer> import_host_memory(void* host_ptr, uint64_t bytes,
                                         bool device_address = true);

    // Path B, allocation included: VirtualAlloc (optionally large pages) plus
    // the import. The returned buffer owns the host allocation.
    Result<GpuBuffer> allocate_imported(uint64_t bytes, bool device_address = true,
                                        bool try_large_pages = false);

    // One slab of the expert cache, on whichever path init() selected.
    Result<GpuBuffer> allocate_slab(uint64_t bytes);

    // A SlabBacking that routes through this allocator, so store/slab.h works
    // unchanged on either path.
    Result<std::unique_ptr<store::SlabBacking>> make_slab_backing();

    // Memory types that are DEVICE_LOCAL and NOT host-visible: the "pure VRAM"
    // ceiling of design §3.3. Empty on a UMA part that exposes none.
    std::vector<uint32_t> device_local_only_types() const;

    uint64_t allocated_bytes() const { return allocated_bytes_; }

private:
    // `require` / `prefer` are VkMemoryPropertyFlags, spelled as uint32_t so the
    // header still compiles with DEEPMOE_ENABLE_VULKAN off.
    Result<int32_t> pick_type(uint32_t type_bits, uint32_t require, uint32_t prefer) const;

    Device*    device_ = nullptr;
    MemoryPath path_   = MemoryPath::Auto;
    int32_t    memory_type_index_ = -1;
    uint64_t   allocated_bytes_ = 0;
    std::vector<GpuBuffer> owned_;
};

}  // namespace deepmoe::gpu
