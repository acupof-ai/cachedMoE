#include "gpu/vulkan/memory.h"

#include <algorithm>
#include <format>

#include "core/log.h"

namespace deepmoe::gpu {

MemoryAllocator::~MemoryAllocator() { shutdown(); }

Result<void> MemoryAllocator::init(Device& device, MemoryPath path) {
    if (!device.valid()) return fail(Err::FailedPrecondition, "device is not created");
    device_ = &device;
    const DeviceCaps& c = device.caps();

    if (path == MemoryPath::Auto) {
        // design §3.3: prefer path A when the driver exposes a
        // DEVICE_LOCAL|HOST_VISIBLE type, since it needs no import and the
        // heap is the BIOS VGM carve-out the GPU reads fastest.
        const bool has_a = std::any_of(c.memory_types.begin(), c.memory_types.end(),
                                       [](const MemoryTypeInfo& t) { return t.device_local && t.host_visible; });
        path = has_a ? MemoryPath::DeviceLocalHostVisible : MemoryPath::ExternalMemoryHost;
    }
    path_ = path;

    if (path_ == MemoryPath::DeviceLocalHostVisible) {
        // Prefer coherent-but-uncached: the CPU only ever writes here.
        int32_t best = -1;
        for (const MemoryTypeInfo& t : c.memory_types) {
            if (!t.device_local || !t.host_visible) continue;
            if (best < 0) best = static_cast<int32_t>(t.index);
            if (t.host_coherent && !t.host_cached) { best = static_cast<int32_t>(t.index); break; }
        }
        if (best < 0)
            return fail(Err::Unavailable, "no DEVICE_LOCAL|HOST_VISIBLE memory type for path A");
        memory_type_index_ = best;
    } else if (path_ == MemoryPath::ExternalMemoryHost) {
        if (!c.external_memory_host)
            return fail(Err::Unavailable, "VK_EXT_external_memory_host is not supported (path B)");
        if (c.min_imported_host_pointer_alignment > kPageSize)
            return fail(Err::FailedPrecondition,
                        std::format("minImportedHostPointerAlignment {} exceeds the 4 KiB layout",
                                    c.min_imported_host_pointer_alignment));
    }
    log_info("gpu memory: path {} (memory type {})",
             path_ == MemoryPath::DeviceLocalHostVisible ? "A device-local/host-visible"
                                                         : "B external_memory_host",
             memory_type_index_);
    return {};
}

void MemoryAllocator::shutdown() {
    for (GpuBuffer& b : owned_) free(b);
    owned_.clear();
    device_ = nullptr;
    memory_type_index_ = -1;
}

Result<MemoryTypeInfo> MemoryAllocator::chosen_memory_type() const {
    if (!device_) return fail(Err::FailedPrecondition, "allocator is not initialised");
    if (memory_type_index_ < 0) return fail(Err::FailedPrecondition, "no memory type selected");
    for (const MemoryTypeInfo& t : device_->caps().memory_types)
        if (static_cast<int32_t>(t.index) == memory_type_index_) return t;
    return fail(Err::Internal, "selected memory type vanished");
}

// TODO(design §7): vkCreateBuffer with STORAGE_BUFFER|TRANSFER_DST|
// SHADER_DEVICE_ADDRESS, vkAllocateMemory from memory_type_index_, bind, map,
// vkGetBufferDeviceAddress. Landing this is the first task of P2.
Result<GpuBuffer> MemoryAllocator::allocate(uint64_t, bool, bool) {
    return unimplemented("gpu::MemoryAllocator::allocate (design §7, P2)");
}

void MemoryAllocator::free(GpuBuffer& buf) {
#if defined(DEEPMOE_ENABLE_VULKAN)
    if (!device_ || !device_->valid()) { buf = GpuBuffer{}; return; }
    if (buf.host_ptr && buf.memory) vkUnmapMemory(device_->handle(), buf.memory);
    if (buf.buffer) vkDestroyBuffer(device_->handle(), buf.buffer, nullptr);
    if (buf.memory) vkFreeMemory(device_->handle(), buf.memory, nullptr);
#endif
    buf = GpuBuffer{};
}

// TODO(design §5.3, §7.8): host-coherent scratch for the pointer table and the
// router readback.
Result<GpuBuffer> MemoryAllocator::allocate_host_coherent(uint64_t) {
    return unimplemented("gpu::MemoryAllocator::allocate_host_coherent (design §5.3, §7.8)");
}

// TODO(design §3.3 path B): VkImportMemoryHostPointerInfoEXT with
// VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT.
Result<GpuBuffer> MemoryAllocator::import_host_memory(void*, uint64_t) {
    return unimplemented("gpu::MemoryAllocator::import_host_memory (design §3.3 path B)");
}

// TODO(design §5.3): a SlabBacking whose allocate() calls allocate() or
// import_host_memory() depending on path_. store/slab.h is already written
// against the interface, so this is the only glue that is missing.
Result<std::unique_ptr<store::SlabBacking>> MemoryAllocator::make_slab_backing() {
    return unimplemented("gpu::MemoryAllocator::make_slab_backing (design §5.3, P2)");
}

}  // namespace deepmoe::gpu
