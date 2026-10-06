#pragma once

#include "core/namespace.h"
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <map>
#include <span>
#include <vector>
#include "core/status.h"
#include "core/config.h"
#include "model/manifest.h"
#include "runtime/moe_bridge.h"
#include "gpu/vulkan/dspark_kernels.h"
#include "gpu/vulkan/decode_kernels.h"

namespace cachedmoe::runtime {
// Real checkpoint draft chain. It owns only draft scratch/KV; weights and the
// pinned MTP experts belong to the engine. One chain produces a [5,vocab] matrix.
class DsparkRuntime {
  public:
    struct Output {
        std::array<uint32_t, 5> tokens{};
        std::array<float, 5> confidence{};
        std::array<uint32_t, 45> expert_ids{}; // three MTP stages x five rows x top-3
        bool expert_ids_valid = false;
        std::vector<float> logits;
        std::map<std::string, double> timing_ms;
        std::map<std::string, double> gpu_timing_ms; // timestamp durations, opt-in
        uint32_t gpu_submissions = 0;
        uint32_t gpu_dispatches = 0, gpu_phases = 0; // measured host wall per dispatch/transfer
    };
    DsparkRuntime();
    ~DsparkRuntime();
    Result<void> create(gpu::Device &, gpu::MemoryAllocator &, const store::PinnedStore &,
                        store::ExpertStore &, store::Planner &, const TextConfig &,
                        const GpuExecutionConfig & = {});
    void reset();
    Result<void> set_profile(bool enabled);
    void set_onecb(bool enabled);
    void set_mega(bool enabled, uint32_t groups = 120);
    uint32_t next_position() const;
    // Committed main positions only. Input is [n,15360], rounded hc means.
    Result<void> append(uint32_t p0, std::span<const float> hidden);
    // All five positions still traverse the bidirectional MTP backbone.
    // Only the independent output head / sequential Markov suffix is truncated.
    Result<Output> draft(uint32_t main_position, uint32_t next_token, uint32_t output_rows = 5,
                         bool read_logits = true);
    Result<void> seed_window(uint32_t next_position,
                             const std::array<std::span<const float>, 3> &rings);
    std::function<void(std::string_view, std::span<const float>)> probe;
    static std::vector<std::string> tensors(const Manifest &);

  private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
} // namespace cachedmoe::runtime
