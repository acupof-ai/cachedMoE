// C++20 smoke test: enumerate Vulkan device, heaps, and check the UMA-relevant bits.
#include <vulkan/vulkan.h>
#include <cstdio>
#include <vector>
#include <string_view>
#include <span>
#include <format>
#include <immintrin.h>

static const char* heapFlags(VkMemoryHeapFlags f) {
    return (f & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? "DEVICE_LOCAL" : "host";
}

int main() {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "deepmoe-smoke";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst{};
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) { std::puts("vkCreateInstance failed"); return 1; }

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(inst, &n, devs.data());
    for (VkPhysicalDevice d : std::span(devs)) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(d, &p);
        std::puts(std::format("device: {}  api {}.{}.{}", p.deviceName,
                              VK_API_VERSION_MAJOR(p.apiVersion), VK_API_VERSION_MINOR(p.apiVersion),
                              VK_API_VERSION_PATCH(p.apiVersion)).c_str());
        std::puts(std::format("  maxStorageBufferRange = {} MiB", p.limits.maxStorageBufferRange >> 20).c_str());

        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(d, &mp);
        for (uint32_t i = 0; i < mp.memoryHeapCount; ++i)
            std::puts(std::format("  heap[{}] {:.1f} GiB {}", i, mp.memoryHeaps[i].size / 1073741824.0,
                                  heapFlags(mp.memoryHeaps[i].flags)).c_str());
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            auto f = mp.memoryTypes[i].propertyFlags;
            if ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
                std::puts(std::format("  type[{}] DEVICE_LOCAL|HOST_VISIBLE{} (heap {})", i,
                                      (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? "|CACHED" : "",
                                      mp.memoryTypes[i].heapIndex).c_str());
        }

        uint32_t ne = 0;
        vkEnumerateDeviceExtensionProperties(d, nullptr, &ne, nullptr);
        std::vector<VkExtensionProperties> ex(ne);
        vkEnumerateDeviceExtensionProperties(d, nullptr, &ne, ex.data());
        for (std::string_view want : {"VK_EXT_external_memory_host", "VK_KHR_shader_integer_dot_product",
                                      "VK_KHR_cooperative_matrix", "VK_KHR_timeline_semaphore",
                                      "VK_KHR_buffer_device_address"}) {
            bool ok = false;
            for (auto& e : ex) if (want == e.extensionName) ok = true;
            std::puts(std::format("  {:45} {}", want, ok ? "yes" : "NO").c_str());
        }
    }
    // AVX-512 sanity: compile + run a VNNI dot product
    __m512i a = _mm512_set1_epi8(2), b = _mm512_set1_epi8(3), acc = _mm512_setzero_si512();
    acc = _mm512_dpbusd_epi32(acc, a, b);
    std::puts(std::format("avx512-vnni dpbusd lane0 = {} (expect 24)", _mm512_cvtsi512_si32(acc)).c_str());
    vkDestroyInstance(inst, nullptr);
    return 0;
}
