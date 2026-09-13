#include "gpu/vulkan/pipeline.h"

#include <cstdio>
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

#if !defined(DEEPMOE_ENABLE_VULKAN)

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
    return fail(Err::Unavailable, "built without DEEPMOE_ENABLE_VULKAN");
}

#else

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
    const VkResult r = vkCreateComputePipelines(device.handle(), VK_NULL_HANDLE, 1, &cpci,
                                                nullptr, &pipeline_);
    if (r != VK_SUCCESS) {
        destroy();
        return fail(Err::Internal, std::format("vkCreateComputePipelines failed for '{}' ({})",
                                               spv_path, static_cast<int>(r)));
    }
    valid_ = true;
    return {};
}

#endif  // DEEPMOE_ENABLE_VULKAN

}  // namespace deepmoe::gpu
