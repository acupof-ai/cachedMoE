#include "gpu/vulkan/memory.h"

#include <algorithm>
#include <cstring>
#include <format>

#include "core/align.h"
#include "core/log.h"

// The one place outside storage/windows that needs a platform header
// (docs/architecture.md §1.2). design §3.3 asks path B to be measured on
// VirtualAlloc memory and on large pages when SeLockMemoryPrivilege allows,
// and neither is reachable through the C++ allocator. The rest of gpu/ is
// portable; everything below the guard has an aligned-new fallback.
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace deepmoe::gpu {

// --- host page allocation (path B) ------------------------------------------

#if defined(_WIN32)
namespace {

// SeLockMemoryPrivilege is off by default and is not grantable at runtime: the
// account needs "Lock pages in memory" in secpol.msc and a re-login. Trying is
// free, so we try and report.
bool enable_lock_memory_privilege(std::string& why) {
    HANDLE token = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &token)) {
        why = std::format("OpenProcessToken failed ({})", GetLastError());
        return false;
    }
    TOKEN_PRIVILEGES tp{};
    tp.PrivilegeCount = 1;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    bool ok = LookupPrivilegeValueW(nullptr, L"SeLockMemoryPrivilege", &tp.Privileges[0].Luid) != 0;
    if (ok) {
        AdjustTokenPrivileges(token, FALSE, &tp, sizeof(tp), nullptr, nullptr);
        const DWORD e = GetLastError();
        if (e == ERROR_NOT_ALL_ASSIGNED) {
            why = "SeLockMemoryPrivilege is not held by this account "
                  "(secpol.msc -> Lock pages in memory, then re-login)";
            ok = false;
        }
    } else {
        why = std::format("LookupPrivilegeValue failed ({})", GetLastError());
    }
    CloseHandle(token);
    return ok;
}

}  // namespace
#endif

Result<HostAllocInfo> alloc_host_pages(uint64_t bytes, bool try_large_pages) {
    if (bytes == 0) return fail(Err::InvalidArgument, "alloc_host_pages(0)");
    HostAllocInfo info;
    info.bytes = align_up(bytes, kPageSize);

#if defined(_WIN32)
    if (try_large_pages) {
        std::string why;
        const SIZE_T lp = GetLargePageMinimum();
        if (lp == 0) {
            info.note = "the system reports no large page size";
        } else if (!enable_lock_memory_privilege(why)) {
            info.note = why;
        } else {
            const uint64_t rounded = align_up(bytes, static_cast<uint64_t>(lp));
            void* p = VirtualAlloc(nullptr, static_cast<SIZE_T>(rounded),
                                   MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES, PAGE_READWRITE);
            if (p) {
                info.ptr = p;
                info.bytes = rounded;
                info.large_pages = true;
                info.note = std::format("{} MiB large pages", lp >> 20);
                return info;
            }
            info.note = std::format("VirtualAlloc(MEM_LARGE_PAGES) failed ({})", GetLastError());
        }
    }
    void* p = VirtualAlloc(nullptr, static_cast<SIZE_T>(info.bytes),
                           MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!p)
        return fail(Err::ResourceExhausted,
                    std::format("VirtualAlloc({} B) failed", info.bytes), GetLastError());
    info.ptr = p;
    return info;
#else
    (void)try_large_pages;
    void* p = ::operator new(static_cast<size_t>(info.bytes),
                             std::align_val_t{static_cast<size_t>(kPageSize)}, std::nothrow);
    if (!p) return fail(Err::ResourceExhausted, std::format("aligned new({} B) failed", info.bytes));
    info.ptr = p;
    info.note = "aligned new (large pages are Windows-only here)";
    return info;
#endif
}

void free_host_pages(const HostAllocInfo& info) {
    if (!info.ptr) return;
#if defined(_WIN32)
    VirtualFree(info.ptr, 0, MEM_RELEASE);
#else
    ::operator delete(info.ptr, std::align_val_t{static_cast<size_t>(kPageSize)});
#endif
}

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
    allocated_bytes_ = 0;
}

Result<MemoryTypeInfo> MemoryAllocator::chosen_memory_type() const {
    if (!device_) return fail(Err::FailedPrecondition, "allocator is not initialised");
    if (memory_type_index_ < 0) return fail(Err::FailedPrecondition, "no memory type selected");
    for (const MemoryTypeInfo& t : device_->caps().memory_types)
        if (static_cast<int32_t>(t.index) == memory_type_index_) return t;
    return fail(Err::Internal, "selected memory type vanished");
}

std::vector<uint32_t> MemoryAllocator::device_local_only_types() const {
    std::vector<uint32_t> out;
    if (!device_) return out;
    for (const MemoryTypeInfo& t : device_->caps().memory_types)
        if (t.device_local && !t.host_visible) out.push_back(t.index);
    return out;
}

#if !defined(DEEPMOE_ENABLE_VULKAN)

Result<int32_t> MemoryAllocator::pick_type(uint32_t, uint32_t, uint32_t) const {
    return fail(Err::Unavailable, "built without DEEPMOE_ENABLE_VULKAN");
}
Result<GpuBuffer> MemoryAllocator::allocate(uint64_t, bool, bool) {
    return fail(Err::Unavailable, "built without DEEPMOE_ENABLE_VULKAN");
}
Result<GpuBuffer> MemoryAllocator::allocate_from_type(uint64_t, uint32_t, bool, bool) {
    return fail(Err::Unavailable, "built without DEEPMOE_ENABLE_VULKAN");
}
Result<GpuBuffer> MemoryAllocator::allocate_host_coherent(uint64_t) {
    return fail(Err::Unavailable, "built without DEEPMOE_ENABLE_VULKAN");
}
Result<GpuBuffer> MemoryAllocator::import_host_memory(void*, uint64_t, bool) {
    return fail(Err::Unavailable, "built without DEEPMOE_ENABLE_VULKAN");
}
Result<GpuBuffer> MemoryAllocator::allocate_imported(uint64_t, bool, bool) {
    return fail(Err::Unavailable, "built without DEEPMOE_ENABLE_VULKAN");
}
Result<GpuBuffer> MemoryAllocator::allocate_slab(uint64_t) {
    return fail(Err::Unavailable, "built without DEEPMOE_ENABLE_VULKAN");
}
Result<std::unique_ptr<store::SlabBacking>> MemoryAllocator::make_slab_backing() {
    return fail(Err::Unavailable, "built without DEEPMOE_ENABLE_VULKAN");
}
void MemoryAllocator::free(GpuBuffer& buf) {
    if (buf.host_alloc) free_host_pages(HostAllocInfo{buf.host_alloc, buf.bytes, buf.large_pages, {}});
    buf = GpuBuffer{};
}

#else

namespace {

constexpr VkBufferUsageFlags kSlabUsage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT  | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

PFN_vkGetMemoryHostPointerPropertiesEXT host_ptr_props_fn(VkDevice d) {
    static PFN_vkGetMemoryHostPointerPropertiesEXT fn = nullptr;
    static VkDevice cached = VK_NULL_HANDLE;
    if (cached != d) {
        cached = d;
        fn = reinterpret_cast<PFN_vkGetMemoryHostPointerPropertiesEXT>(
            vkGetDeviceProcAddr(d, "vkGetMemoryHostPointerPropertiesEXT"));
    }
    return fn;
}

}  // namespace

Result<int32_t> MemoryAllocator::pick_type(uint32_t type_bits, uint32_t require,
                                           uint32_t prefer) const {
    if (!device_) return fail(Err::FailedPrecondition, "allocator is not initialised");
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(device_->physical(), &mp);
    int32_t fallback = -1;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if (!(type_bits & (1u << i))) continue;
        const VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        if ((f & require) != require) continue;
        if ((f & prefer) == prefer) return static_cast<int32_t>(i);
        if (fallback < 0) fallback = static_cast<int32_t>(i);
    }
    if (fallback >= 0) return fallback;
    return fail(Err::Unavailable,
                std::format("no memory type satisfies flags {:#x} within mask {:#x}", require, type_bits));
}

Result<GpuBuffer> MemoryAllocator::allocate_from_type(uint64_t bytes, uint32_t memory_type_index,
                                                      bool map, bool device_address) {
    if (!device_ || !device_->valid()) return fail(Err::FailedPrecondition, "allocator is not initialised");
    if (bytes == 0) return fail(Err::InvalidArgument, "allocate(0)");
    const DeviceCaps& c = device_->caps();
    if (c.max_memory_allocation_size && bytes > c.max_memory_allocation_size)
        return fail(Err::InvalidArgument,
                    std::format("{} B exceeds maxMemoryAllocationSize {} B (design §5.3: use more slabs)",
                                bytes, c.max_memory_allocation_size));
    VkDevice d = device_->handle();

    GpuBuffer out;
    out.bytes = bytes;
    out.memory_type = memory_type_index;

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes;
    bci.usage = kSlabUsage;
    if (!device_address) bci.usage &= ~VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(d, &bci, nullptr, &out.buffer) != VK_SUCCESS)
        return fail(Err::ResourceExhausted, std::format("vkCreateBuffer({} B) failed", bytes));

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(d, out.buffer, &req);
    if (!(req.memoryTypeBits & (1u << memory_type_index))) {
        vkDestroyBuffer(d, out.buffer, nullptr);
        return fail(Err::InvalidArgument,
                    std::format("memory type {} is not allowed for this buffer (mask {:#x})",
                                memory_type_index, req.memoryTypeBits));
    }

    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    if (device_address) mai.pNext = &flags;
    mai.allocationSize  = std::max<uint64_t>(req.size, bytes);
    mai.memoryTypeIndex = memory_type_index;
    if (VkResult r = vkAllocateMemory(d, &mai, nullptr, &out.memory); r != VK_SUCCESS) {
        vkDestroyBuffer(d, out.buffer, nullptr);
        return fail(Err::ResourceExhausted,
                    std::format("vkAllocateMemory({} B, type {}) failed ({})",
                                mai.allocationSize, memory_type_index, static_cast<int>(r)));
    }
    if (vkBindBufferMemory(d, out.buffer, out.memory, 0) != VK_SUCCESS) {
        vkFreeMemory(d, out.memory, nullptr);
        vkDestroyBuffer(d, out.buffer, nullptr);
        return fail(Err::Internal, "vkBindBufferMemory failed");
    }
    if (map) {
        if (vkMapMemory(d, out.memory, 0, VK_WHOLE_SIZE, 0, &out.host_ptr) != VK_SUCCESS) {
            vkFreeMemory(d, out.memory, nullptr);
            vkDestroyBuffer(d, out.buffer, nullptr);
            return fail(Err::Internal, "vkMapMemory failed");
        }
    }
    if (device_address) {
        VkBufferDeviceAddressInfo ai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        ai.buffer = out.buffer;
        out.dev_addr = vkGetBufferDeviceAddress(d, &ai);
    }
    allocated_bytes_ += bytes;
    return out;
}

Result<GpuBuffer> MemoryAllocator::allocate(uint64_t bytes, bool host_visible, bool device_address) {
    if (!device_) return fail(Err::FailedPrecondition, "allocator is not initialised");
    // Ask the driver which types this buffer shape allows before choosing.
    VkBufferCreateInfo probe{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    probe.size = bytes;
    probe.usage = kSlabUsage;
    if (!device_address) probe.usage &= ~VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VkBuffer tmp = VK_NULL_HANDLE;
    if (vkCreateBuffer(device_->handle(), &probe, nullptr, &tmp) != VK_SUCCESS)
        return fail(Err::ResourceExhausted, std::format("vkCreateBuffer({} B) failed", bytes));
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device_->handle(), tmp, &req);
    vkDestroyBuffer(device_->handle(), tmp, nullptr);

    const uint32_t require = host_visible
        ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    const uint32_t prefer = host_visible
        ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    auto type = pick_type(req.memoryTypeBits, require, prefer);
    if (!type) return std::unexpected(type.error());
    return allocate_from_type(bytes, static_cast<uint32_t>(*type), host_visible, device_address);
}

Result<GpuBuffer> MemoryAllocator::allocate_host_coherent(uint64_t bytes) {
    return allocate(bytes, /*host_visible=*/true, /*device_address=*/true);
}

Result<GpuBuffer> MemoryAllocator::import_host_memory(void* host_ptr, uint64_t bytes,
                                                      bool device_address) {
    if (!device_ || !device_->valid()) return fail(Err::FailedPrecondition, "allocator is not initialised");
    const DeviceCaps& c = device_->caps();
    if (!c.external_memory_host)
        return fail(Err::Unavailable, "VK_EXT_external_memory_host is not supported");
    const uint64_t align = std::max<uint64_t>(c.min_imported_host_pointer_alignment, 1);
    if (!is_aligned(host_ptr, align) || !is_aligned(bytes, align))
        return fail(Err::InvalidArgument,
                    std::format("import needs pointer and size aligned to {} B", align));
    if (c.max_memory_allocation_size && bytes > c.max_memory_allocation_size)
        return fail(Err::InvalidArgument,
                    std::format("{} B exceeds maxMemoryAllocationSize {} B", bytes,
                                c.max_memory_allocation_size));

    VkDevice d = device_->handle();
    auto props_fn = host_ptr_props_fn(d);
    if (!props_fn) return fail(Err::Unavailable, "vkGetMemoryHostPointerPropertiesEXT is missing");

    VkMemoryHostPointerPropertiesEXT hp{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    if (props_fn(d, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, host_ptr, &hp) != VK_SUCCESS)
        return fail(Err::Internal, "vkGetMemoryHostPointerPropertiesEXT failed");
    if (hp.memoryTypeBits == 0)
        return fail(Err::Unavailable, "the driver accepts no memory type for this host pointer");

    GpuBuffer out;
    out.bytes    = bytes;
    out.host_ptr = host_ptr;
    out.imported = true;

    VkExternalMemoryBufferCreateInfo emb{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    emb.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &emb};
    bci.size  = bytes;
    bci.usage = kSlabUsage;
    if (!device_address) bci.usage &= ~VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    if (vkCreateBuffer(d, &bci, nullptr, &out.buffer) != VK_SUCCESS)
        return fail(Err::ResourceExhausted, std::format("vkCreateBuffer({} B, external) failed", bytes));

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(d, out.buffer, &req);
    auto type = pick_type(req.memoryTypeBits & hp.memoryTypeBits, 0,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (!type) { vkDestroyBuffer(d, out.buffer, nullptr); return std::unexpected(type.error()); }
    out.memory_type = static_cast<uint32_t>(*type);

    VkImportMemoryHostPointerInfoEXT imp{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
    imp.handleType   = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    imp.pHostPointer = host_ptr;
    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, &imp};
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                             device_address ? static_cast<void*>(&flags) : static_cast<void*>(&imp)};
    // The imported allocation must be a whole number of import-alignment units
    // and must cover the buffer's requirement.
    mai.allocationSize  = align_up(std::max<uint64_t>(req.size, bytes), align);
    mai.memoryTypeIndex = out.memory_type;
    if (VkResult r = vkAllocateMemory(d, &mai, nullptr, &out.memory); r != VK_SUCCESS) {
        vkDestroyBuffer(d, out.buffer, nullptr);
        return fail(Err::ResourceExhausted,
                    std::format("vkAllocateMemory(import {} B, type {}) failed ({})",
                                mai.allocationSize, out.memory_type, static_cast<int>(r)));
    }
    if (vkBindBufferMemory(d, out.buffer, out.memory, 0) != VK_SUCCESS) {
        vkFreeMemory(d, out.memory, nullptr);
        vkDestroyBuffer(d, out.buffer, nullptr);
        return fail(Err::Internal, "vkBindBufferMemory failed on the imported allocation");
    }
    if (device_address) {
        VkBufferDeviceAddressInfo ai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        ai.buffer = out.buffer;
        out.dev_addr = vkGetBufferDeviceAddress(d, &ai);
    }
    allocated_bytes_ += bytes;
    return out;
}

Result<GpuBuffer> MemoryAllocator::allocate_imported(uint64_t bytes, bool device_address,
                                                     bool try_large_pages) {
    auto host = alloc_host_pages(bytes, try_large_pages);
    if (!host) return std::unexpected(host.error());
    auto buf = import_host_memory(host->ptr, host->bytes, device_address);
    if (!buf) { free_host_pages(*host); return std::unexpected(buf.error()); }
    buf->host_alloc  = host->ptr;
    buf->large_pages = host->large_pages;
    return buf;
}

Result<GpuBuffer> MemoryAllocator::allocate_slab(uint64_t bytes) {
    if (path_ == MemoryPath::ExternalMemoryHost)
        return allocate_imported(bytes, /*device_address=*/true, /*try_large_pages=*/false);
    if (memory_type_index_ < 0) return fail(Err::FailedPrecondition, "no memory type selected");
    return allocate_from_type(bytes, static_cast<uint32_t>(memory_type_index_),
                              /*map=*/true, /*device_address=*/true);
}

void MemoryAllocator::free(GpuBuffer& buf) {
    if (device_ && device_->valid()) {
        VkDevice d = device_->handle();
        // An imported allocation must not be unmapped: the mapping is the
        // caller's VirtualAlloc, not a vkMapMemory result.
        if (buf.host_ptr && buf.memory && !buf.imported) vkUnmapMemory(d, buf.memory);
        if (buf.buffer) vkDestroyBuffer(d, buf.buffer, nullptr);
        if (buf.memory) vkFreeMemory(d, buf.memory, nullptr);
    }
    if (buf.host_alloc)
        free_host_pages(HostAllocInfo{buf.host_alloc, buf.bytes, buf.large_pages, {}});
    if (allocated_bytes_ >= buf.bytes) allocated_bytes_ -= buf.bytes;
    buf = GpuBuffer{};
}

// --- SlabBacking over the allocator -----------------------------------------

namespace {

// design §5.3 / architecture.md §1.3: the interface inversion. store/ defines
// SlabBacking; gpu/ implements it, so the ExpertStore never learns which of the
// two paths of design §3.3 it is running on.
class VulkanSlabBacking final : public store::SlabBacking {
public:
    explicit VulkanSlabBacking(MemoryAllocator& alloc) : alloc_(alloc) {
        name_ = alloc.path() == MemoryPath::ExternalMemoryHost ? "vulkan-B-external-host"
                                                               : "vulkan-A-device-host-visible";
    }
    ~VulkanSlabBacking() override {
        for (GpuBuffer& b : bufs_) alloc_.free(b);
    }

    Result<store::SlabMemory> allocate(uint64_t bytes) override {
        auto b = alloc_.allocate_slab(bytes);
        if (!b) return std::unexpected(b.error());
        if (!b->host_ptr) {
            alloc_.free(*b);
            return fail(Err::Internal, "slab is not host-writable; the IoEngine cannot fill it");
        }
        bufs_.push_back(*b);
        return store::SlabMemory{b->host_ptr, b->dev_addr, b->bytes};
    }

    void release(const store::SlabMemory& mem) override {
        for (auto it = bufs_.begin(); it != bufs_.end(); ++it) {
            if (it->host_ptr == mem.host_ptr) {
                alloc_.free(*it);
                bufs_.erase(it);
                return;
            }
        }
    }

    const char* name() const override { return name_; }

private:
    MemoryAllocator&       alloc_;
    std::vector<GpuBuffer> bufs_;
    const char*            name_ = "vulkan";
};

}  // namespace

Result<std::unique_ptr<store::SlabBacking>> MemoryAllocator::make_slab_backing() {
    if (!device_ || !device_->valid()) return fail(Err::FailedPrecondition, "allocator is not initialised");
    return std::unique_ptr<store::SlabBacking>(new VulkanSlabBacking(*this));
}

#endif  // DEEPMOE_ENABLE_VULKAN

}  // namespace deepmoe::gpu
