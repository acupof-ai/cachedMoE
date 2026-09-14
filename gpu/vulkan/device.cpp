#include "gpu/vulkan/device.h"

#include <algorithm>
#include <cstring>
#include <format>

#include "core/log.h"

namespace deepmoe::gpu {

std::string DeviceCaps::to_string() const {
    std::string s = std::format("{}  api {}.{}.{}  driver {}\n",
                                device_name, api_major, api_minor, api_patch, driver_version);
    s += std::format("  maxMemoryAllocationSize {} MiB   maxBufferSize {} MiB   maxStorageBufferRange {} MiB\n",
                     max_memory_allocation_size >> 20, max_buffer_size >> 20,
                     max_storage_buffer_range >> 20);
    s += std::format("  maxComputeSharedMemorySize {} KiB   subgroup {} ({}..{})\n",
                     max_compute_shared_memory >> 10, subgroup_size,
                     min_subgroup_size, max_subgroup_size);
    if (min_imported_host_pointer_alignment)
        s += std::format("  minImportedHostPointerAlignment {} B\n", min_imported_host_pointer_alignment);
    for (const HeapInfo& h : heaps)
        s += std::format("  heap[{}] {:.1f} GiB {}\n", h.index, h.bytes / 1073741824.0,
                         h.device_local ? "DEVICE_LOCAL" : "host");
    for (const MemoryTypeInfo& t : memory_types)
        if (t.device_local && t.host_visible)
            s += std::format("  type[{}] DEVICE_LOCAL|HOST_VISIBLE{}{} (heap {})\n", t.index,
                             t.host_coherent ? "|COHERENT" : "", t.host_cached ? "|CACHED" : "",
                             t.heap_index);
    auto yn = [](bool b) { return b ? "yes" : "NO"; };
    s += std::format("  external_memory_host {}   timeline_semaphore {}   buffer_device_address {}\n",
                     yn(external_memory_host), yn(timeline_semaphore), yn(buffer_device_address));
    s += std::format("  integer_dot_product {}   subgroup_size_control {}   cooperative_matrix {}\n",
                     yn(integer_dot_product), yn(subgroup_size_control), yn(cooperative_matrix));
    s += std::format("  shaderFloat16 {}   shaderInt8 {}   shaderInt16 {}   shaderInt64 {}\n",
                     yn(shader_float16), yn(shader_int8), yn(shader_int16), yn(shader_int64));
    s += std::format("  8bit_storage {}   16bit_storage {}   synchronization2 {}\n",
                     yn(storage_buffer_8bit), yn(storage_buffer_16bit), yn(synchronization2));
    s += std::format("  timestampPeriod {:.1f} ns   timestampValidBits {}\n",
                     timestamp_period_ns, timestamp_valid_bits);
    return s;
}

Result<void> DeviceCaps::check_required() const {
    std::string missing;
    // design §1.1 / §5.3 / §7.1: without these the whole memory and dispatch
    // model of deepMoE has no implementation.
    if (!external_memory_host)  missing += "  VK_EXT_external_memory_host (design §3.3 path B)\n";
    if (!timeline_semaphore)    missing += "  VK_KHR_timeline_semaphore (design §7.1 expert readiness)\n";
    if (!buffer_device_address) missing += "  VK_KHR_buffer_device_address (design §5.3 pointer table)\n";
    if (!subgroup_size_control) missing += "  VK_EXT_subgroup_size_control (design §7.1 Wave32/64 A/B)\n";
    if (!shader_float16)        missing += "  shaderFloat16 (design §6 activation precision)\n";
    if (!storage_buffer_8bit)   missing += "  8-bit storage (design §6 FP4/FP8 weights)\n";
    if (!storage_buffer_16bit)  missing += "  16-bit storage (design §6 fp16 activation buffers)\n";
    if (!shader_int16)          missing += "  shaderInt16 (design §6 fp16 activation buffers)\n";
    if (!shader_int64)          missing += "  shaderInt64 (design §5.3 buffer_device_address pointer table)\n";
    if (!synchronization2)      missing += "  synchronization2 (design §7.1 barriers and GPU timestamps)\n";
    if (max_memory_allocation_size == 0) missing += "  maxMemoryAllocationSize is unknown\n";
    if (!missing.empty())
        return fail(Err::FailedPrecondition, "device is missing required capabilities:\n" + missing);
    return {};
}

#if !defined(DEEPMOE_ENABLE_VULKAN)

Device::~Device() = default;
void Device::destroy() { valid_ = false; }

Result<void> Device::create(const DeviceOptions&) {
    return fail(Err::Unavailable, "built without DEEPMOE_ENABLE_VULKAN");
}
Result<std::vector<DeviceCaps>> Device::enumerate(bool) {
    return fail(Err::Unavailable, "built without DEEPMOE_ENABLE_VULKAN");
}

#else

namespace {

bool has_ext(const std::vector<VkExtensionProperties>& list, const char* name) {
    return std::any_of(list.begin(), list.end(),
                       [&](const VkExtensionProperties& e) { return std::strcmp(e.extensionName, name) == 0; });
}

Result<VkInstance> create_instance(bool validation) {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "deepmoe";
    app.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    const char* layers[] = {"VK_LAYER_KHRONOS_validation"};
    if (validation) {
        ici.enabledLayerCount = 1;
        ici.ppEnabledLayerNames = layers;
    }
    VkInstance inst = VK_NULL_HANDLE;
    const VkResult r = vkCreateInstance(&ici, nullptr, &inst);
    if (r == VK_ERROR_LAYER_NOT_PRESENT && validation) {
        ici.enabledLayerCount = 0;
        log_warn("vulkan: validation layer not present, continuing without it");
        if (vkCreateInstance(&ici, nullptr, &inst) == VK_SUCCESS) return inst;
    }
    if (r != VK_SUCCESS)
        return fail(Err::Unavailable, std::format("vkCreateInstance failed ({})", static_cast<int>(r)));
    return inst;
}

DeviceCaps query_caps(VkPhysicalDevice pd) {
    DeviceCaps c;

    VkPhysicalDeviceSubgroupSizeControlProperties sgc{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
    VkPhysicalDeviceExternalMemoryHostPropertiesEXT emh{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT, &sgc};
    VkPhysicalDeviceMaintenance4Properties m4{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES, &emh};
    VkPhysicalDeviceVulkan11Properties v11{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES, &m4};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &v11};
    vkGetPhysicalDeviceProperties2(pd, &p2);

    const VkPhysicalDeviceProperties& p = p2.properties;
    c.device_name    = p.deviceName;
    c.api_major      = VK_API_VERSION_MAJOR(p.apiVersion);
    c.api_minor      = VK_API_VERSION_MINOR(p.apiVersion);
    c.api_patch      = VK_API_VERSION_PATCH(p.apiVersion);
    c.driver_version = p.driverVersion;
    c.vendor_id      = p.vendorID;
    c.device_id      = p.deviceID;
    c.max_storage_buffer_range = p.limits.maxStorageBufferRange;
    c.max_compute_shared_memory = p.limits.maxComputeSharedMemorySize;
    c.max_compute_workgroup_invocations = p.limits.maxComputeWorkGroupInvocations;
    c.timestamp_period_ns = p.limits.timestampPeriod;
    // Report the timestamp width of the family a compute queue would come from,
    // so `deepmoe info` says the same thing the benchmarks see.
    {
        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qp(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qp.data());
        for (uint32_t i = 0; i < nq; ++i)
            if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                c.timestamp_valid_bits = qp[i].timestampValidBits;
                break;
            }
    }
    c.max_buffer_size = m4.maxBufferSize;
    c.subgroup_size     = v11.subgroupSize;
    c.min_subgroup_size = sgc.minSubgroupSize;
    c.max_subgroup_size = sgc.maxSubgroupSize;
    c.min_imported_host_pointer_alignment = emh.minImportedHostPointerAlignment;

    // maxMemoryAllocationSize comes from the 1.1 maintenance3 limits.
    VkPhysicalDeviceMaintenance3Properties m3{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES};
    VkPhysicalDeviceProperties2 p2b{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &m3};
    vkGetPhysicalDeviceProperties2(pd, &p2b);
    c.max_memory_allocation_size = m3.maxMemoryAllocationSize;

    VkPhysicalDeviceShaderFloat16Int8Features f16i8{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    VkPhysicalDeviceSynchronization2Features sync2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES, &f16i8};
    VkPhysicalDevice16BitStorageFeatures s16{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES, &sync2};
    VkPhysicalDevice8BitStorageFeatures s8{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES, &s16};
    VkPhysicalDeviceTimelineSemaphoreFeatures ts{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES, &s8};
    VkPhysicalDeviceBufferDeviceAddressFeatures bda{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES, &ts};
    VkPhysicalDeviceShaderIntegerDotProductFeatures idp{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES, &bda};
    VkPhysicalDeviceFeatures2 feat{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &idp};
    vkGetPhysicalDeviceFeatures2(pd, &feat);
    c.shader_float16       = f16i8.shaderFloat16 == VK_TRUE;
    c.shader_int8          = f16i8.shaderInt8 == VK_TRUE;
    c.storage_buffer_8bit  = s8.storageBuffer8BitAccess == VK_TRUE;
    c.storage_buffer_16bit = s16.storageBuffer16BitAccess == VK_TRUE &&
                             s16.uniformAndStorageBuffer16BitAccess == VK_TRUE;
    c.shader_int16         = feat.features.shaderInt16 == VK_TRUE;
    c.shader_int64         = feat.features.shaderInt64 == VK_TRUE;
    c.synchronization2     = sync2.synchronization2 == VK_TRUE;
    c.timeline_semaphore   = ts.timelineSemaphore == VK_TRUE;
    c.buffer_device_address = bda.bufferDeviceAddress == VK_TRUE;
    c.integer_dot_product  = idp.shaderIntegerDotProduct == VK_TRUE;

    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &ne, nullptr);
    std::vector<VkExtensionProperties> ex(ne);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &ne, ex.data());
    c.external_memory_host  = has_ext(ex, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
    c.subgroup_size_control = has_ext(ex, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
    c.cooperative_matrix    = has_ext(ex, "VK_KHR_cooperative_matrix");
    if (!c.timeline_semaphore)    c.timeline_semaphore    = has_ext(ex, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
    if (!c.buffer_device_address) c.buffer_device_address = has_ext(ex, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    if (!c.integer_dot_product)   c.integer_dot_product   = has_ext(ex, "VK_KHR_shader_integer_dot_product");

    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryHeapCount; ++i)
        c.heaps.push_back(HeapInfo{mp.memoryHeaps[i].size,
                                   (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0, i});
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        const VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        c.memory_types.push_back(MemoryTypeInfo{
            i, mp.memoryTypes[i].heapIndex,
            (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0,
            (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0,
            (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0,
            (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0});
    }
    return c;
}

Result<uint32_t> pick_compute_family(VkPhysicalDevice pd) {
    uint32_t n = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, nullptr);
    std::vector<VkQueueFamilyProperties> q(n);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, q.data());
    // Prefer a compute-only family: on RDNA it is an async compute engine that
    // does not contend with graphics work the desktop compositor submits.
    for (uint32_t i = 0; i < n; ++i)
        if ((q[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(q[i].queueFlags & VK_QUEUE_GRAPHICS_BIT))
            return i;
    for (uint32_t i = 0; i < n; ++i)
        if (q[i].queueFlags & VK_QUEUE_COMPUTE_BIT) return i;
    return fail(Err::Unavailable, "no compute queue family");
}

}  // namespace

Device::~Device() { destroy(); }

void Device::destroy() {
    if (device_) { vkDestroyDevice(device_, nullptr); device_ = VK_NULL_HANDLE; }
    if (instance_) { vkDestroyInstance(instance_, nullptr); instance_ = VK_NULL_HANDLE; }
    physical_ = VK_NULL_HANDLE;
    compute_queue_ = VK_NULL_HANDLE;
    valid_ = false;
}

Result<std::vector<DeviceCaps>> Device::enumerate(bool validation) {
    auto inst = create_instance(validation);
    if (!inst) return std::unexpected(inst.error());
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(*inst, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(*inst, &n, devs.data());
    std::vector<DeviceCaps> out;
    out.reserve(n);
    for (VkPhysicalDevice d : devs) out.push_back(query_caps(d));
    vkDestroyInstance(*inst, nullptr);
    if (out.empty()) return fail(Err::Unavailable, "no Vulkan physical devices");
    return out;
}

Result<void> Device::create(const DeviceOptions& opts) {
    destroy();
    auto inst = create_instance(opts.enable_validation);
    if (!inst) return std::unexpected(inst.error());
    instance_ = *inst;

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(instance_, &n, nullptr);
    if (n == 0) { destroy(); return fail(Err::Unavailable, "no Vulkan physical devices"); }
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(instance_, &n, devs.data());

    uint32_t pick = 0;
    if (opts.physical_device_index >= 0) {
        if (static_cast<uint32_t>(opts.physical_device_index) >= n) {
            destroy();
            return fail(Err::OutOfRange, std::format("physical device {} of {}", opts.physical_device_index, n));
        }
        pick = static_cast<uint32_t>(opts.physical_device_index);
    }
    physical_ = devs[pick];
    caps_ = query_caps(physical_);

    auto fam = pick_compute_family(physical_);
    if (!fam) { destroy(); return std::unexpected(fam.error()); }
    compute_family_ = *fam;
    {
        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_, &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qp(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_, &nq, qp.data());
        if (compute_family_ < nq) caps_.timestamp_valid_bits = qp[compute_family_].timestampValidBits;
    }

    std::vector<const char*> exts;
    if (caps_.external_memory_host)  exts.push_back(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
    if (caps_.subgroup_size_control) exts.push_back(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
    if (caps_.cooperative_matrix)    exts.push_back("VK_KHR_cooperative_matrix");

    // Everything promoted into a VkPhysicalDeviceVulkanNNFeatures struct has to
    // be requested *there* and nowhere else: mixing the promoted struct with the
    // original extension struct is VUID-VkDeviceCreateInfo-pNext-02830.
    // 1.1: 16-bit storage for the fp16 activation buffers of design §6.
    VkPhysicalDeviceVulkan11Features v11f{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    v11f.storageBuffer16BitAccess           = caps_.storage_buffer_16bit ? VK_TRUE : VK_FALSE;
    v11f.uniformAndStorageBuffer16BitAccess = caps_.storage_buffer_16bit ? VK_TRUE : VK_FALSE;
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &v11f};
    v12.timelineSemaphore       = caps_.timeline_semaphore ? VK_TRUE : VK_FALSE;
    v12.bufferDeviceAddress     = caps_.buffer_device_address ? VK_TRUE : VK_FALSE;
    v12.storageBuffer8BitAccess = caps_.storage_buffer_8bit ? VK_TRUE : VK_FALSE;
    v12.shaderFloat16           = caps_.shader_float16 ? VK_TRUE : VK_FALSE;
    v12.shaderInt8              = caps_.shader_int8 ? VK_TRUE : VK_FALSE;
    VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &v12};
    v13.subgroupSizeControl  = caps_.subgroup_size_control ? VK_TRUE : VK_FALSE;
    v13.computeFullSubgroups = caps_.subgroup_size_control ? VK_TRUE : VK_FALSE;
    v13.shaderIntegerDotProduct = caps_.integer_dot_product ? VK_TRUE : VK_FALSE;
    v13.synchronization2     = caps_.synchronization2 ? VK_TRUE : VK_FALSE;
    VkPhysicalDeviceFeatures2 feat{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v13};
    // SPIR-V PhysicalStorageBuffer64 addressing (the expert pointer table of
    // design §5.3) needs Int64; `half` in a storage buffer needs Int16.
    feat.features.shaderInt64 = caps_.shader_int64 ? VK_TRUE : VK_FALSE;
    feat.features.shaderInt16 = caps_.shader_int16 ? VK_TRUE : VK_FALSE;

    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = compute_family_;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &feat};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = static_cast<uint32_t>(exts.size());
    dci.ppEnabledExtensionNames = exts.data();

    const VkResult r = vkCreateDevice(physical_, &dci, nullptr, &device_);
    if (r != VK_SUCCESS) {
        destroy();
        return fail(Err::Unavailable, std::format("vkCreateDevice failed ({})", static_cast<int>(r)));
    }
    vkGetDeviceQueue(device_, compute_family_, 0, &compute_queue_);
    valid_ = true;
    log_info("vulkan: {} (compute family {})", caps_.device_name, compute_family_);
    return {};
}

#endif  // DEEPMOE_ENABLE_VULKAN

}  // namespace deepmoe::gpu
