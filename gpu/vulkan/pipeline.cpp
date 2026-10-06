#include "core/env.h"
#include "gpu/vulkan/pipeline.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>

namespace deepmoe::gpu {

Result<std::vector<uint32_t>> load_spirv(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return fail(Err::NotFound, std::format("cannot open SPIR-V '{}'", path));
    std::vector<uint32_t> words;
    uint32_t chunk[4096];
    size_t n;
    while ((n = std::fread(chunk, sizeof(uint32_t), 4096, f)) > 0)
        words.insert(words.end(), chunk, chunk + n);
    const bool bad = std::ferror(f) != 0;
    std::fclose(f);
    if (bad) return fail(Err::Io, std::format("read error on '{}'", path));
    if (words.empty() || words[0] != 0x07230203u)
        return fail(Err::Corrupt, std::format("'{}' is not SPIR-V (magic {:#x})", path,
                                              words.empty() ? 0u : words[0]));
    return words;
}

Pipeline::~Pipeline() { destroy(); }

#if !defined(CACHEDMOE_ENABLE_VULKAN)

Pipeline::Pipeline(Pipeline&& o) noexcept { device_ = o.device_; name_ = std::move(o.name_); valid_ = o.valid_; o.valid_ = false; }
Pipeline& Pipeline::operator=(Pipeline&& o) noexcept {
    if (this != &o) { destroy(); device_ = o.device_; name_ = std::move(o.name_); valid_ = o.valid_; o.valid_ = false; }
    return *this;
}
void Pipeline::destroy() { valid_ = false; }

Result<void> Pipeline::create(Device&, const std::string& spv_path,
                              const PipelineLayoutSpec&, const PipelineSpec&) {
    // Still validate the file, so a shader-only CI job catches a broken .spv
    // without a GPU.
    if (auto w = load_spirv(spv_path); !w) return std::unexpected(w.error());
    return fail(Err::Unavailable, "built without CACHEDMOE_ENABLE_VULKAN");
}

#else

namespace {

// CACHEDMOE_PIPELINE_STATS: the driver's own account of one pipeline -- its
// statistics as one line of `<dir>/index.tsv`, and statistics plus every
// internal representation (NIR, ACO IR, ISA on RADV) in `<dir>/<name>.txt`.
// `tag` is the .spv's base name and the specialisation constants, which is
// what tells two pipelines of one shader apart.
void write_pipeline_stats(VkDevice d, VkPipeline pipe, const std::string& tag) {
    const char* dir = ::deepmoe::environment::get("CACHEDMOE_PIPELINE_STATS");
    if (!dir || !*dir) return;
    auto props = reinterpret_cast<PFN_vkGetPipelineExecutablePropertiesKHR>(
        vkGetDeviceProcAddr(d, "vkGetPipelineExecutablePropertiesKHR"));
    auto stats = reinterpret_cast<PFN_vkGetPipelineExecutableStatisticsKHR>(
        vkGetDeviceProcAddr(d, "vkGetPipelineExecutableStatisticsKHR"));
    auto irs = reinterpret_cast<PFN_vkGetPipelineExecutableInternalRepresentationsKHR>(
        vkGetDeviceProcAddr(d, "vkGetPipelineExecutableInternalRepresentationsKHR"));
    if (!props || !stats || !irs) return;
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);

    VkPipelineInfoKHR pi{VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR};
    pi.pipeline = pipe;
    uint32_t ne = 0;
    props(d, &pi, &ne, nullptr);
    std::vector<VkPipelineExecutablePropertiesKHR> ex(ne, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});
    props(d, &pi, &ne, ex.data());

    std::string line = tag, body;
    for (uint32_t e = 0; e < ne; ++e) {
        VkPipelineExecutableInfoKHR ei{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
        ei.pipeline = pipe;
        ei.executableIndex = e;
        body += std::format("== executable {}: {} ({}), subgroup {}\n", e, ex[e].name,
                            ex[e].description, ex[e].subgroupSize);
        uint32_t ns = 0;
        stats(d, &ei, &ns, nullptr);
        std::vector<VkPipelineExecutableStatisticKHR> st(ns, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
        stats(d, &ei, &ns, st.data());
        for (const auto& x : st) {
            std::string v;
            switch (x.format) {
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR: v = x.value.b32 ? "1" : "0"; break;
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:  v = std::to_string(x.value.i64); break;
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR: v = std::to_string(x.value.u64); break;
                default: v = std::format("{:g}", x.value.f64); break;
            }
            body += std::format("  {:<28s} {}\n", x.name, v);
            line += std::format("\t{}={}", x.name, v);
        }
        uint32_t nr = 0;
        irs(d, &ei, &nr, nullptr);
        std::vector<VkPipelineExecutableInternalRepresentationKHR> rep(
            nr, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR});
        irs(d, &ei, &nr, rep.data());
        std::vector<std::vector<char>> text(nr);
        for (uint32_t i = 0; i < nr; ++i) {
            text[i].resize(rep[i].dataSize + 1, 0);
            rep[i].pData = text[i].data();
        }
        irs(d, &ei, &nr, rep.data());
        for (uint32_t i = 0; i < nr; ++i)
            body += std::format("\n== {} ({})\n{}\n", rep[i].name, rep[i].description,
                                rep[i].isText ? text[i].data() : "(binary)");
    }
    std::string fname = tag;
    for (char& ch : fname) if (ch == ' ' || ch == '/' || ch == '\\') ch = '_';
    if (std::FILE* f = std::fopen((std::filesystem::path(dir) / (fname + ".txt")).string().c_str(), "w")) {
        std::fputs(body.c_str(), f);
        std::fclose(f);
    }
    if (std::FILE* f = std::fopen((std::filesystem::path(dir) / "index.tsv").string().c_str(), "a")) {
        std::fprintf(f, "%s\n", line.c_str());
        std::fclose(f);
    }
}

}  // namespace

Pipeline::Pipeline(Pipeline&& o) noexcept
    : device_(o.device_), name_(std::move(o.name_)), valid_(o.valid_),
      module_(o.module_), set_layout_(o.set_layout_), layout_(o.layout_), pipeline_(o.pipeline_) {
    o.module_ = VK_NULL_HANDLE; o.set_layout_ = VK_NULL_HANDLE;
    o.layout_ = VK_NULL_HANDLE; o.pipeline_ = VK_NULL_HANDLE; o.valid_ = false;
}

Pipeline& Pipeline::operator=(Pipeline&& o) noexcept {
    if (this != &o) {
        destroy();
        device_ = o.device_; name_ = std::move(o.name_); valid_ = o.valid_;
        module_ = o.module_; set_layout_ = o.set_layout_; layout_ = o.layout_; pipeline_ = o.pipeline_;
        o.module_ = VK_NULL_HANDLE; o.set_layout_ = VK_NULL_HANDLE;
        o.layout_ = VK_NULL_HANDLE; o.pipeline_ = VK_NULL_HANDLE; o.valid_ = false;
    }
    return *this;
}

void Pipeline::destroy() {
    if (device_ && device_->valid()) {
        VkDevice d = device_->handle();
        if (pipeline_)   vkDestroyPipeline(d, pipeline_, nullptr);
        if (layout_)     vkDestroyPipelineLayout(d, layout_, nullptr);
        if (set_layout_) vkDestroyDescriptorSetLayout(d, set_layout_, nullptr);
        if (module_)     vkDestroyShaderModule(d, module_, nullptr);
    }
    pipeline_ = VK_NULL_HANDLE; layout_ = VK_NULL_HANDLE;
    set_layout_ = VK_NULL_HANDLE; module_ = VK_NULL_HANDLE;
    device_ = nullptr;
    valid_ = false;
}

Result<void> Pipeline::create(Device& device, const std::string& spv_path,
                              const PipelineLayoutSpec& lspec, const PipelineSpec& spec) {
    destroy();
    if (!device.valid()) return fail(Err::FailedPrecondition, "device is not created");
    auto words = load_spirv(spv_path);
    if (!words) return std::unexpected(words.error());
    device_ = &device;
    name_ = spv_path;

    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = words->size() * sizeof(uint32_t);
    smci.pCode    = words->data();
    if (vkCreateShaderModule(device.handle(), &smci, nullptr, &module_) != VK_SUCCESS) {
        destroy();
        return fail(Err::Internal, std::format("vkCreateShaderModule failed for '{}'", spv_path));
    }

    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.reserve(lspec.storage_buffers);
    for (uint32_t i = 0; i < lspec.storage_buffers; ++i)
        bindings.push_back(VkDescriptorSetLayoutBinding{i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                                        VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = static_cast<uint32_t>(bindings.size());
    dslci.pBindings    = bindings.data();
    if (vkCreateDescriptorSetLayout(device.handle(), &dslci, nullptr, &set_layout_) != VK_SUCCESS) {
        destroy();
        return fail(Err::Internal, "vkCreateDescriptorSetLayout failed");
    }

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, lspec.push_constant_size};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &set_layout_;
    plci.pushConstantRangeCount = lspec.push_constant_size ? 1u : 0u;
    plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(device.handle(), &plci, nullptr, &layout_) != VK_SUCCESS) {
        destroy();
        return fail(Err::Internal, "vkCreatePipelineLayout failed");
    }

    // Specialisation constants: ids 0..3 are the fixed knobs of design §7.1,
    // anything in `extra` continues from 4.
    std::vector<uint32_t> data;
    std::vector<VkSpecializationMapEntry> entries;
    auto push = [&](uint32_t id, uint32_t v) {
        entries.push_back(VkSpecializationMapEntry{id, static_cast<uint32_t>(data.size() * sizeof(uint32_t)),
                                                   sizeof(uint32_t)});
        data.push_back(v);
    };
    push(0, spec.m);
    push(1, spec.lanes_per_row);
    push(2, spec.rows_per_wg);
    push(3, spec.subgroup_size);
    for (uint32_t i = 0; i < spec.extra.size(); ++i) push(4 + i, spec.extra[i]);

    VkSpecializationInfo si{};
    si.mapEntryCount = static_cast<uint32_t>(entries.size());
    si.pMapEntries   = entries.data();
    si.dataSize      = data.size() * sizeof(uint32_t);
    si.pData         = data.data();

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo sgs{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    sgs.requiredSubgroupSize = spec.subgroup_size;

    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = module_;
    stage.pName  = "main";
    stage.pSpecializationInfo = &si;
    if (spec.subgroup_size && device.caps().subgroup_size_control) stage.pNext = &sgs;

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage  = stage;
    cpci.layout = layout_;
    if (device.caps().pipeline_stats)
        cpci.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR |
                      VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;
    const VkResult r = vkCreateComputePipelines(device.handle(), VK_NULL_HANDLE, 1, &cpci,
                                                nullptr, &pipeline_);
    if (r != VK_SUCCESS) {
        destroy();
        return fail(Err::Internal, std::format("vkCreateComputePipelines failed for '{}' ({})",
                                               spv_path, static_cast<int>(r)));
    }
    if (device.caps().pipeline_stats) {
        std::string tag = std::filesystem::path(spv_path).stem().string();
        tag += std::format(" m{} l{} r{} s{}", spec.m, spec.lanes_per_row, spec.rows_per_wg,
                           spec.subgroup_size);
        for (uint32_t v : spec.extra) tag += std::format(" {}", v);
        write_pipeline_stats(device.handle(), pipeline_, tag);
    }
    valid_ = true;
    return {};
}

#endif  // CACHEDMOE_ENABLE_VULKAN

}  // namespace deepmoe::gpu
