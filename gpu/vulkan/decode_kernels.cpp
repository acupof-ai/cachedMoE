#include "gpu/vulkan/decode_kernels.h"

#include <cstring>
#include <format>

namespace deepmoe::gpu {

const char* decode_stage_name(DecodeStage s) {
    switch (s) {
        case DecodeStage::EngramGemv: return "engram.gemv";
        case DecodeStage::EngramGate: return "engram.gate";
        case DecodeStage::Argmax:     return "head.argmax";
        case DecodeStage::SampleTopK: return "sample_topk";
        case DecodeStage::Count:      break;
    }
    return "?";
}

#if !defined(DEEPMOE_ENABLE_VULKAN)

Result<void> DecodeRunner::create(Device&, MemoryAllocator&, const std::string&, const AttnSpec&) {
    return fail(Err::Unavailable, "built without DEEPMOE_ENABLE_VULKAN");
}
void DecodeRunner::destroy() {}
uint64_t* DecodeRunner::slots(DecodeStage) { return nullptr; }
Result<void> DecodeRunner::record(CommandBuffer&, DecodeStage, const void*, uint32_t, uint32_t) {
    return fail(Err::Unavailable, "no vulkan");
}
Result<void> DecodeRunner::dispatch_now(DecodeStage, const void*, uint32_t, uint32_t) {
    return fail(Err::Unavailable, "no vulkan");
}
Result<void> DecodeRunner::make(DecodeStage, const std::string&, uint32_t) {
    return fail(Err::Unavailable, "no vulkan");
}

#else

namespace {
// The same 64 B range attn_kernels.cpp uses, so a stage can be pushed either
// struct without the layout caring.
constexpr uint32_t kPushBytes = 64;

struct StageDef { DecodeStage stage; const char* spv; uint32_t stage_const; };

constexpr StageDef kStages[] = {
    {DecodeStage::EngramGemv, "engram", 0},
    {DecodeStage::EngramGate, "engram", 1},
    {DecodeStage::Argmax,     "head",   1},
    {DecodeStage::SampleTopK, "sample_topk", 0},
};
static_assert(sizeof(kStages) / sizeof(kStages[0]) ==
              static_cast<size_t>(DecodeStage::Count));
}  // namespace

Result<void> DecodeRunner::make(DecodeStage s, const std::string& spv, uint32_t stage_const) {
    PipelineSpec ps;
    ps.m             = 1;
    ps.lanes_per_row = spec_.lanes_per_row;
    ps.rows_per_wg   = 256 / spec_.lanes_per_row;
    ps.subgroup_size = spec_.subgroup_size;
    // act_quant is on (engram's wkv is an fp8 Linear) and RowsPerLane is 1:
    // neither kernel row-blocks, and head.slang's argmax stage ignores both.
    ps.extra = {stage_const, 1u, 1u};
    PipelineLayoutSpec la;
    la.storage_buffers    = 1;
    la.push_constant_size = kPushBytes;
    return pipes_[static_cast<uint32_t>(s)].create(*device_, spv, la, ps);
}

Result<void> DecodeRunner::create(Device& device, MemoryAllocator& alloc,
                                  const std::string& shader_dir, const AttnSpec& spec) {
    destroy();
    if (!device.valid()) return fail(Err::FailedPrecondition, "device is not created");
    if (spec.lanes_per_row != 16 && spec.lanes_per_row != 32 && spec.lanes_per_row != 64)
        return fail(Err::InvalidArgument, "lanes_per_row must be 16, 32 or 64 (design §7.1 rule 3)");
    device_ = &device;
    alloc_  = &alloc;
    spec_   = spec;
    spec_.rows_per_lane = 1;

    const uint32_t n = static_cast<uint32_t>(DecodeStage::Count);
    auto t = alloc.allocate(uint64_t(n) * kAttnStageStride, true, false);
    if (!t) { destroy(); return std::unexpected(t.error()); }
    table_ = *t;
    std::memset(table_.host_ptr, 0, static_cast<size_t>(table_.bytes));

    if (auto r = descriptors_.create(device, n, n); !r) { destroy(); return r; }
    for (const StageDef& d : kStages) {
        if (auto r = make(d.stage, shader_dir + "/" + d.spv + ".spv", d.stage_const); !r) {
            destroy();
            return fail(r.error().code,
                        std::format("{}: {}", decode_stage_name(d.stage), r.error().message));
        }
        const uint32_t i = static_cast<uint32_t>(d.stage);
        std::vector<BufferBinding> b(1);
        b[0] = {0, uint64_t(i) * kAttnStageStride, kAttnStageStride, table_.buffer};
        auto set = descriptors_.allocate(pipes_[i], b);
        if (!set) { destroy(); return std::unexpected(set.error()); }
        sets_[i] = *set;
    }
    if (auto r = pool_.create(device); !r) { destroy(); return r; }
    return {};
}

void DecodeRunner::destroy() {
    pool_.destroy();
    descriptors_.destroy();
    for (Pipeline& p : pipes_) p.destroy();
    if (alloc_ && table_.valid()) alloc_->free(table_);
    table_ = GpuBuffer{};
    for (VkDescriptorSet& s : sets_) s = VK_NULL_HANDLE;
    device_ = nullptr;
    alloc_ = nullptr;
}

uint64_t* DecodeRunner::slots(DecodeStage s) {
    if (!table_.host_ptr) return nullptr;
    return reinterpret_cast<uint64_t*>(static_cast<std::byte*>(table_.host_ptr) +
                                       uint64_t(static_cast<uint32_t>(s)) * kAttnStageStride);
}

Result<void> DecodeRunner::record(CommandBuffer& cmd, DecodeStage s, const void* push,
                                  uint32_t push_bytes, uint32_t groups) {
    const uint32_t i = static_cast<uint32_t>(s);
    if (!pipes_[i].valid()) return fail(Err::FailedPrecondition, "pipeline is not created");
    if (push_bytes > kPushBytes)
        return fail(Err::InvalidArgument, "push constants exceed the shared 64 B range");
    if (groups == 0) return fail(Err::InvalidArgument, "zero workgroups");
    if (auto r = cmd.bind(pipes_[i], sets_[i]); !r) return r;
    if (push_bytes) {
        if (auto r = cmd.push(pipes_[i], push, push_bytes); !r) return r;
    }
    return cmd.dispatch(groups);
}

Result<void> DecodeRunner::dispatch_now(DecodeStage s, const void* push, uint32_t push_bytes,
                                        uint32_t groups) {
    auto cb = pool_.acquire();
    if (!cb) return std::unexpected(cb.error());
    CommandBuffer cmd = *cb;
    if (auto r = cmd.begin(); !r) return r;
    if (auto r = record(cmd, s, push, push_bytes, groups); !r) return r;
    if (auto r = cmd.end(); !r) return r;
    return submit_and_wait(*device_, cmd);
}

#endif  // DEEPMOE_ENABLE_VULKAN

}  // namespace deepmoe::gpu
