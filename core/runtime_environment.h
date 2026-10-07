// The sole registry and parser authority for production environment options.
// A snapshot owns presence (including empty strings), alias provenance and
// typed overrides. Engine init shares one snapshot with all lazy resources;
// standalone factories capture their own setup epoch. Telemetry is not here.
#pragma once

#include "core/namespace.h"
#include "core/env.h"
#include "core/status.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace cachedmoe::configuration {

// Defaults whose historical public constants remain aliases in engine.h.
inline constexpr uint32_t kAutoSlotCap = 5000;
inline constexpr uint32_t kMesaRadvDriverId = 3;
inline constexpr uint32_t kCacheBackoffSlots = 200;
inline constexpr double kGpuWaitSeconds = 900;
inline constexpr double kFenceSpinMicroseconds = 0;
inline constexpr uint32_t kMaskWaitExperts = 8;
inline constexpr double kMaskWaitMilliseconds = 20;
inline constexpr uint32_t kMirrorProbeMilliseconds = 1000;
inline constexpr uint32_t kSourceErrorBudget = 3;
inline constexpr int kMirrorHotCelsius = 80;
inline constexpr uint32_t kResidentQueueSteps = 2;
inline constexpr uint32_t kResidentQueueExperts = 24;

// Key names live only in this table. Compatibility aliases are resolved by
// core/env.h before the bytes are copied; consumer code never re-reads them.
#define CACHEDMOE_RUNTIME_KEYS(X)                                                                  \
    X(ATTN_CM)                                                                                     \
    X(ATTN_KSPLIT)                                                                                 \
    X(BACKFILL)                                                                                    \
    X(BATCH_ENGRAM_EARLY)                                                                          \
    X(BATCH_GPU_ROUTE)                                                                             \
    X(CACHE_BACKOFF_SLOTS)                                                                         \
    X(CACHE_SLOT_CAP)                                                                              \
    X(CPU_AFFINITY)                                                                                \
    X(DSPARK_MEGA)                                                                                 \
    X(DSPARK_MEGA_DIAG)                                                                            \
    X(DSPARK_HEAD_FP8)                                                                             \
    X(DSPARK_ONECB)                                                                                \
    X(DSPARK_PROFILE)                                                                              \
    X(DSPARK_TRIM_TAIL)                                                                            \
    X(EVICT_PATH)                                                                                  \
    X(FENCE_SPIN_US)                                                                               \
    X(GAMEMODE)                                                                                    \
    X(GATE_PROBE)                                                                                  \
    X(GPU_WAIT_S)                                                                                  \
    X(HEAT_FILE)                                                                                   \
    X(IO_BG_CAP_BUSY)                                                                              \
    X(IO_BG_QD)                                                                                    \
    X(IO_BG_THROTTLE_P2)                                                                           \
    X(IO_ENGRAM_DEADLINE)                                                                          \
    X(IO_ENGRAM_QD)                                                                                \
    X(IO_P0_CHUNK_MB)                                                                              \
    X(IO_P0_INFLIGHT_MB)                                                                           \
    X(IO_P0_QD)                                                                                    \
    X(IO_SUBMIT_THREADS)                                                                           \
    X(KV_DIR)                                                                                      \
    X(MASK_DYNAMIC_LRU)                                                                            \
    X(MASK_WAIT_BUDGET)                                                                            \
    X(MASK_WAIT_TAU)                                                                               \
    X(MGT_ATTN_CM)                                                                                 \
    X(MGT_FOLD_SCALE)                                                                              \
    X(MGT_PAIR_DOT)                                                                                \
    X(MIRROR_AUTO)                                                                                 \
    X(MIRROR_CLASSES)                                                                              \
    X(MIRROR_ERROR_BUDGET)                                                                         \
    X(MIRROR_HEALTH)                                                                               \
    X(MIRROR_HOT_C)                                                                                \
    X(MIRROR_KEEPALIVE_MS)                                                                         \
    X(MIRROR_PROBE_MS)                                                                             \
    X(MIRROR_PROBE_WARMUP_MS)                                                                      \
    X(MIRROR_STATIC_SPLIT)                                                                         \
    X(MIRROR_STRIPE)                                                                               \
    X(MIRROR_WEIGHTS)                                                                              \
    X(MODEL_DIR)                                                                                   \
    X(MODEL_MIRRORS)                                                                               \
    X(MOE_DEC)                                                                                     \
    X(MOE_HQUANT)                                                                                  \
    X(MOE_L)                                                                                       \
    X(MOE_LB)                                                                                      \
    X(MOE_OVERLAP)                                                                                 \
    X(MOE_R)                                                                                       \
    X(MOE_RB)                                                                                      \
    X(MOE_STATIC_M1)                                                                               \
    X(MOE_UNION_L)                                                                                 \
    X(MOE_UNION_LB)                                                                                \
    X(MOE_UNION_R)                                                                                 \
    X(MOE_UNION_RB)                                                                                \
    X(MOE_UNION_XMODE)                                                                             \
    X(MOE_WC_READ)                                                                                 \
    X(MOE_XMODE)                                                                                   \
    X(MOE_XMODE_B)                                                                                 \
    X(MS_EAGER_MOE)                                                                                \
    X(PATH_A_CAP)                                                                                  \
    X(PERF_COUNTERS)                                                                               \
    X(PF_CREATE_TRACE)                                                                             \
    X(PF_LDS)                                                                                      \
    X(PF_OPS_JSON)                                                                                 \
    X(PF_READ_AHEAD)                                                                               \
    X(PIPELINE_STATS)                                                                              \
    X(PREFAULT)                                                                                    \
    X(PREFILL_HANDOFF)                                                                             \
    X(RESIDENT_QUEUE_EXPERTS)                                                                      \
    X(RESIDENT_QUEUE_STEPS)                                                                        \
    X(ROUTE_DUMP)                                                                                  \
    X(ROUTE_RESIDENT_ONLY)                                                                         \
    X(RUN_IO_REPORT)                                                                               \
    X(SE_CHECK)                                                                                    \
    X(SHADER_DIR)                                                                                  \
    X(SHARED_EARLY)                                                                                \
    X(SHARED_EARLY_MS)                                                                             \
    X(SPEC_DIAGNOSTICS)                                                                            \
    X(SPEC_GPU_READOUT)                                                                            \
    X(VERIFY_DRAFT)                                                                                \
    X(VERIFY_FIRST)

enum class Key : size_t {
#define CACHEDMOE_KEY_ENUM(name) name,
    CACHEDMOE_RUNTIME_KEYS(CACHEDMOE_KEY_ENUM)
#undef CACHEDMOE_KEY_ENUM
        Count
};
inline constexpr std::array<std::string_view, static_cast<size_t>(Key::Count)> kKeyNames{{
#define CACHEDMOE_KEY_NAME(name) "CACHEDMOE_" #name,
    CACHEDMOE_RUNTIME_KEYS(CACHEDMOE_KEY_NAME)
#undef CACHEDMOE_KEY_NAME
}};
#undef CACHEDMOE_RUNTIME_KEYS

struct RawValue {
    std::optional<std::string> text;
    environment::Source source = environment::Source::Absent;
    bool present() const { return text.has_value(); }
    bool nonempty() const { return text && !text->empty(); }
    const char *c_str() const { return text ? text->c_str() : nullptr; }
};

class EnvironmentSnapshot {
  public:
    static EnvironmentSnapshot capture() {
        EnvironmentSnapshot result;
        for (size_t i = 0; i < kKeyNames.size(); ++i) {
            const auto raw = environment::lookup(kKeyNames[i].data());
            if (raw.present())
                result.values_[i] = {std::string(raw.value), raw.source};
        }
        return result;
    }
    const RawValue &operator[](Key key) const { return values_[static_cast<size_t>(key)]; }
    // Also supports deterministic CPU fixtures without mutating process env.
    void set(Key key, std::optional<std::string> value,
             environment::Source source = environment::Source::Canonical) {
        values_[static_cast<size_t>(key)] = {std::move(value), source};
        if (!values_[static_cast<size_t>(key)].present())
            values_[static_cast<size_t>(key)].source = environment::Source::Absent;
    }

  private:
    std::array<RawValue, static_cast<size_t>(Key::Count)> values_{};
};

namespace parse {

inline std::optional<bool> exact_one(const RawValue &raw) {
    return raw.present() ? std::optional<bool>(*raw.text == "1") : std::nullopt;
}
inline std::optional<bool> first_one(const RawValue &raw) {
    return raw.present() ? std::optional<bool>(*raw.c_str() == '1') : std::nullopt;
}
inline std::optional<bool> not_zero(const RawValue &raw) {
    return raw.present() ? std::optional<bool>(*raw.c_str() != '0') : std::nullopt;
}
inline std::optional<bool> nonempty_flag(const RawValue &raw) {
    return raw.nonempty() ? std::optional<bool>(*raw.c_str() != '0') : std::nullopt;
}
inline std::optional<bool> atoi_flag(const RawValue &raw) {
    return raw.nonempty() ? std::optional<bool>(std::atoi(raw.c_str()) != 0) : std::nullopt;
}
inline std::optional<uint32_t> prefix_u32(const RawValue &raw) {
    if (!raw.nonempty())
        return std::nullopt;
    char *end = nullptr;
    const auto value = std::strtoul(raw.c_str(), &end, 10);
    return end != raw.c_str() ? std::optional<uint32_t>(static_cast<uint32_t>(value))
                              : std::nullopt;
}
inline uint32_t unchecked_u32(const RawValue &raw, uint32_t fallback) {
    return raw.nonempty() ? static_cast<uint32_t>(std::strtoul(raw.c_str(), nullptr, 10))
                          : fallback;
}
inline std::optional<double> positive(const RawValue &raw, double fallback) {
    if (!raw.present())
        return std::nullopt;
    const auto value = std::atof(raw.c_str());
    return value > 0 ? value : fallback;
}
inline std::vector<std::string> semicolon_list(std::string_view value) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i <= value.size()) {
        const size_t j = value.find(';', i);
        std::string part(value.substr(i, j == std::string_view::npos ? j : j - i));
        while (!part.empty() && (part.back() == ' ' || part.back() == '"'))
            part.pop_back();
        while (!part.empty() && (part.front() == ' ' || part.front() == '"'))
            part.erase(part.begin());
        if (!part.empty())
            out.push_back(std::move(part));
        if (j == std::string_view::npos)
            break;
        i = j + 1;
    }
    return out;
}

} // namespace parse

struct GpuOverrides {
    std::optional<bool> batch_gpu_route, batch_engram_early, spec_gpu_readout;
    std::optional<bool> draft_head_fp8, draft_onecb, draft_mega, draft_profile, draft_diagnostics, draft_trim_tail;
    std::optional<bool> mgt_pair_dot, mgt_fold_scale, mgt_attn_cm;
};
struct DecodeOverrides {
    std::optional<double> gpu_wait_seconds, fence_spin_microseconds;
    // A present empty variable clears the optional platform override.
    bool shared_early_present = false, eager_moe_present = false;
    std::optional<bool> shared_early, eager_moe, engram_deadline;
    std::optional<bool> shared_early_multistream, shared_early_check, dynamic_mask_lru;
};
struct IoOverrides {
    std::optional<uint32_t> bg_cap_busy, bg_qd, p0_qd, p0_inflight_mb, p0_chunk_mb;
    std::optional<uint32_t> engram_qd, submit_threads;
    bool throttle_engram = false;
    uint32_t source_error_budget = kSourceErrorBudget;
    int64_t keepalive_ms = 0;
    double static_split = 0;
    bool stripe = true;
    std::optional<uint32_t> route_classes;
    int mirror_hot_c = kMirrorHotCelsius;
};
struct MoeOverrides {
    std::optional<uint32_t> lanes, rows, xmode, xmode_b, decode, lanes_b, rows_b, hquant;
    std::optional<uint32_t> union_lanes, union_rows, union_xmode, union_lanes_b, union_rows_b;
    bool wc_read = true, static_m1 = true;
};
enum class ResidentPolicy : uint8_t { Off = 0, All = 1, Stall1 = 2, Verify = 3, Mask = 4 };
enum class EvictPath : uint8_t { None, PreferA, PreferB };
struct MaskWait {
    bool enabled = false;
    double tau = -1;
    uint32_t experts = kMaskWaitExperts;
    double milliseconds = kMaskWaitMilliseconds;
    std::optional<Status> error;
};

struct RuntimeEnvironment {
    EnvironmentSnapshot raw;
    GpuOverrides gpu;
    DecodeOverrides decode;
    IoOverrides io;
    MoeOverrides moe;
    MaskWait mask_wait;
    std::optional<bool> attn_ksplit, attn_cm, backfill;
    std::optional<std::string> model_dir, kv_dir, prefill_ops_json;
    std::optional<std::vector<std::string>> model_mirrors;
    std::optional<std::vector<double>> mirror_weights;
    std::string cpu_affinity = "auto", shader_dir, heat_file, route_dump, spec_diagnostics;
    uint32_t cache_slot_cap = kAutoSlotCap, cache_backoff_slots = kCacheBackoffSlots;
    bool cache_slot_cap_nonempty = false;
    bool mirror_auto = true, mirror_health = true, path_a_cap = false;
    uint32_t mirror_probe_ms = kMirrorProbeMilliseconds;
    uint32_t mirror_probe_warmup_ms = kMirrorProbeMilliseconds;
    bool overlap = true, gate_probe = false, prefill_handoff = true;
    bool prefill_lds = true, prefill_read_ahead = true, prefill_create_trace = false;
    bool prefault = true, gamemode = true, perf_counters = false, run_io_report = false;
    std::string pipeline_stats_dir;
    ResidentPolicy resident_only = ResidentPolicy::Off;
    ResidentPolicy verify_first = ResidentPolicy::Off, verify_draft = ResidentPolicy::All;
    bool resident_only_valid = true, verify_first_valid = true, verify_draft_valid = true;
    uint32_t resident_queue_steps = kResidentQueueSteps,
             resident_queue_experts = kResidentQueueExperts;
    bool resident_queue_steps_valid = true;
    EvictPath evict_path = EvictPath::None;

    explicit RuntimeEnvironment(EnvironmentSnapshot snapshot) : raw(std::move(snapshot)) {
        resolve();
    }
    static std::shared_ptr<const RuntimeEnvironment> capture() {
        return std::make_shared<const RuntimeEnvironment>(EnvironmentSnapshot::capture());
    }

  private:
    void resolve();
    void resolve_gpu_decode();
    void resolve_paths_cache();
    void resolve_resident();
    void resolve_io();
    void resolve_moe();
    void resolve_mask_wait();
};

inline void RuntimeEnvironment::resolve() {
    resolve_gpu_decode();
    resolve_paths_cache();
    resolve_resident();
    resolve_io();
    resolve_moe();
    resolve_mask_wait();
}

inline void RuntimeEnvironment::resolve_gpu_decode() {
    const auto &r = raw;
    gpu.batch_gpu_route = parse::exact_one(r[Key::BATCH_GPU_ROUTE]);
    gpu.batch_engram_early = parse::not_zero(r[Key::BATCH_ENGRAM_EARLY]);
    gpu.spec_gpu_readout = parse::not_zero(r[Key::SPEC_GPU_READOUT]);
    gpu.draft_head_fp8 = parse::exact_one(r[Key::DSPARK_HEAD_FP8]);
    gpu.draft_onecb = parse::exact_one(r[Key::DSPARK_ONECB]);
    gpu.draft_mega = parse::exact_one(r[Key::DSPARK_MEGA]);
    gpu.draft_profile = parse::exact_one(r[Key::DSPARK_PROFILE]);
    if (r[Key::DSPARK_MEGA_DIAG].present())
        gpu.draft_diagnostics = true;
    gpu.draft_trim_tail = parse::not_zero(r[Key::DSPARK_TRIM_TAIL]);
    gpu.mgt_pair_dot = parse::first_one(r[Key::MGT_PAIR_DOT]);
    gpu.mgt_fold_scale = parse::not_zero(r[Key::MGT_FOLD_SCALE]);
    gpu.mgt_attn_cm = parse::not_zero(r[Key::MGT_ATTN_CM]);

    decode.gpu_wait_seconds = parse::positive(r[Key::GPU_WAIT_S], kGpuWaitSeconds);
    decode.fence_spin_microseconds = parse::positive(r[Key::FENCE_SPIN_US], kFenceSpinMicroseconds);
    decode.shared_early_present = r[Key::SHARED_EARLY].present();
    decode.shared_early = parse::nonempty_flag(r[Key::SHARED_EARLY]);
    decode.eager_moe_present = r[Key::MS_EAGER_MOE].present();
    decode.eager_moe = parse::nonempty_flag(r[Key::MS_EAGER_MOE]);
    if (r[Key::SHARED_EARLY_MS].present())
        decode.shared_early_multistream =
            parse::nonempty_flag(r[Key::SHARED_EARLY_MS]).value_or(false);
    if (r[Key::SE_CHECK].present())
        decode.shared_early_check = true;
    if (r[Key::MASK_DYNAMIC_LRU].present())
        decode.dynamic_mask_lru = *r[Key::MASK_DYNAMIC_LRU].text != "0";
    decode.engram_deadline = parse::exact_one(r[Key::IO_ENGRAM_DEADLINE]);
    attn_ksplit = parse::atoi_flag(r[Key::ATTN_KSPLIT]);
    attn_cm = parse::atoi_flag(r[Key::ATTN_CM]);
    backfill = parse::nonempty_flag(r[Key::BACKFILL]);
}

inline void RuntimeEnvironment::resolve_paths_cache() {
    const auto &r = raw;
    model_dir = r[Key::MODEL_DIR].text;
    kv_dir = r[Key::KV_DIR].text;
    prefill_ops_json = r[Key::PF_OPS_JSON].text;
    if (r[Key::MODEL_MIRRORS].present())
        model_mirrors = parse::semicolon_list(*r[Key::MODEL_MIRRORS].text);
    if (r[Key::MIRROR_WEIGHTS].nonempty()) {
        mirror_weights.emplace();
        for (const auto &item : parse::semicolon_list(*r[Key::MIRROR_WEIGHTS].text))
            mirror_weights->push_back(std::strtod(item.c_str(), nullptr));
    }
    if (r[Key::CPU_AFFINITY].present())
        cpu_affinity = *r[Key::CPU_AFFINITY].text;
    if (r[Key::SHADER_DIR].present())
        shader_dir = *r[Key::SHADER_DIR].text;
#if defined(CACHEDMOE_SHADER_DIR)
    else
        shader_dir = CACHEDMOE_SHADER_DIR;
#else
    else
        shader_dir = "build/shaders";
#endif
    const auto path = [&](Key key) { return r[key].nonempty() ? *r[key].text : std::string{}; };
    heat_file = path(Key::HEAT_FILE);
    route_dump = path(Key::ROUTE_DUMP);
    spec_diagnostics = path(Key::SPEC_DIAGNOSTICS);
    pipeline_stats_dir = path(Key::PIPELINE_STATS);
    cache_slot_cap_nonempty = r[Key::CACHE_SLOT_CAP].nonempty();
    if (cache_slot_cap_nonempty) {
        const auto value = std::atoll(r[Key::CACHE_SLOT_CAP].c_str());
        if (value >= 0)
            cache_slot_cap = static_cast<uint32_t>(value);
    }
    if (r[Key::CACHE_BACKOFF_SLOTS].nonempty()) {
        const auto value = std::atoll(r[Key::CACHE_BACKOFF_SLOTS].c_str());
        if (value > 0)
            cache_backoff_slots = static_cast<uint32_t>(value);
    }
    mirror_auto = parse::not_zero(r[Key::MIRROR_AUTO]).value_or(true);
    mirror_probe_ms = parse::unchecked_u32(r[Key::MIRROR_PROBE_MS], kMirrorProbeMilliseconds);
    mirror_probe_warmup_ms =
        parse::unchecked_u32(r[Key::MIRROR_PROBE_WARMUP_MS], kMirrorProbeMilliseconds);
    if (r[Key::MIRROR_HEALTH].nonempty())
        mirror_health = std::strtol(r[Key::MIRROR_HEALTH].c_str(), nullptr, 10) != 0;
    if (r[Key::PATH_A_CAP].present())
        path_a_cap = *r[Key::PATH_A_CAP].text == "on" || *r[Key::PATH_A_CAP].text == "1";
    if (r[Key::EVICT_PATH].nonempty()) {
        const char value = *r[Key::EVICT_PATH].c_str();
        if (value == 'a' || value == 'A')
            evict_path = EvictPath::PreferA;
        else if (value == 'b' || value == 'B')
            evict_path = EvictPath::PreferB;
    }
    overlap = parse::not_zero(r[Key::MOE_OVERLAP]).value_or(true);
    gate_probe = parse::nonempty_flag(r[Key::GATE_PROBE]).value_or(false);
    prefill_handoff = parse::not_zero(r[Key::PREFILL_HANDOFF]).value_or(true);
    prefill_lds = parse::not_zero(r[Key::PF_LDS]).value_or(true);
    prefill_read_ahead = parse::not_zero(r[Key::PF_READ_AHEAD]).value_or(true);
    prefill_create_trace = r[Key::PF_CREATE_TRACE].present();
    prefault = parse::not_zero(r[Key::PREFAULT]).value_or(true);
    gamemode = parse::not_zero(r[Key::GAMEMODE]).value_or(true);
    perf_counters = parse::nonempty_flag(r[Key::PERF_COUNTERS]).value_or(false);
    run_io_report = parse::exact_one(r[Key::RUN_IO_REPORT]).value_or(false);
}

inline void RuntimeEnvironment::resolve_resident() {
    const auto &r = raw;
    const auto policy = [&](Key key, ResidentPolicy fallback, bool &valid) {
        const auto &value = r[key];
        if (!value.nonempty())
            return fallback;
        const auto &s = *value.text;
        if (s == "off")
            return ResidentPolicy::Off;
        if (s == "exact" && key != Key::ROUTE_RESIDENT_ONLY)
            return ResidentPolicy::Off;
        if (s == "0" && key == Key::ROUTE_RESIDENT_ONLY)
            return ResidentPolicy::Off;
        if (s == "all")
            return ResidentPolicy::All;
        if (s == "stall1")
            return ResidentPolicy::Stall1;
        if (key == Key::ROUTE_RESIDENT_ONLY && s == "verify")
            return ResidentPolicy::Verify;
        if (key == Key::ROUTE_RESIDENT_ONLY && s == "mask")
            return ResidentPolicy::Mask;
        valid = false;
        return fallback;
    };
    resident_only = policy(Key::ROUTE_RESIDENT_ONLY, ResidentPolicy::Off, resident_only_valid);
    verify_first = policy(Key::VERIFY_FIRST, ResidentPolicy::Off, verify_first_valid);
    verify_draft = policy(Key::VERIFY_DRAFT, ResidentPolicy::All, verify_draft_valid);
    // Historical VERIFY_DRAFT=0 was invalid and retained All.
    if (r[Key::VERIFY_DRAFT].nonempty() && *r[Key::VERIFY_DRAFT].text == "0") {
        verify_draft = ResidentPolicy::All;
        verify_draft_valid = false;
    }
    if (r[Key::RESIDENT_QUEUE_STEPS].nonempty()) {
        const auto value = std::atoi(r[Key::RESIDENT_QUEUE_STEPS].c_str());
        resident_queue_steps_valid = value >= 1 && value <= 1024;
        if (resident_queue_steps_valid)
            resident_queue_steps = static_cast<uint32_t>(value);
    }
    if (r[Key::RESIDENT_QUEUE_EXPERTS].nonempty()) {
        const auto value = std::atoi(r[Key::RESIDENT_QUEUE_EXPERTS].c_str());
        if (value >= 1 && value <= 4096)
            resident_queue_experts = static_cast<uint32_t>(value);
    }
}

inline void RuntimeEnvironment::resolve_io() {
    const auto &r = raw;
    io.bg_cap_busy = parse::prefix_u32(r[Key::IO_BG_CAP_BUSY]);
    io.bg_qd = parse::prefix_u32(r[Key::IO_BG_QD]);
    io.p0_qd = parse::prefix_u32(r[Key::IO_P0_QD]);
    io.p0_inflight_mb = parse::prefix_u32(r[Key::IO_P0_INFLIGHT_MB]);
    io.p0_chunk_mb = parse::prefix_u32(r[Key::IO_P0_CHUNK_MB]);
    io.engram_qd = parse::prefix_u32(r[Key::IO_ENGRAM_QD]);
    io.submit_threads = parse::prefix_u32(r[Key::IO_SUBMIT_THREADS]);
    io.throttle_engram = parse::nonempty_flag(r[Key::IO_BG_THROTTLE_P2]).value_or(false);
    io.source_error_budget = parse::unchecked_u32(r[Key::MIRROR_ERROR_BUDGET], kSourceErrorBudget);
    if (r[Key::MIRROR_KEEPALIVE_MS].nonempty()) {
        const char *value = r[Key::MIRROR_KEEPALIVE_MS].c_str();
        if (*value != 'o' && *value != 'O')
            io.keepalive_ms = std::strtoll(value, nullptr, 10);
        if (io.keepalive_ms < 0)
            io.keepalive_ms = 0;
    }
    if (r[Key::MIRROR_STATIC_SPLIT].nonempty()) {
        const auto value = std::strtod(r[Key::MIRROR_STATIC_SPLIT].c_str(), nullptr);
        if (value > 0 && value < 1)
            io.static_split = value;
    }
    io.stripe = parse::not_zero(r[Key::MIRROR_STRIPE]).value_or(true);
    if (r[Key::MIRROR_CLASSES].nonempty()) {
        uint32_t mask = 0;
        for (const auto c : *r[Key::MIRROR_CLASSES].text)
            if (c >= '0' && c <= '3')
                mask |= 1u << static_cast<uint32_t>(c - '0');
        if (mask)
            io.route_classes = mask;
    }
    if (r[Key::MIRROR_HOT_C].nonempty())
        io.mirror_hot_c = std::atoi(r[Key::MIRROR_HOT_C].c_str());
}

inline void RuntimeEnvironment::resolve_moe() {
    const auto &r = raw;
    moe.lanes = parse::prefix_u32(r[Key::MOE_L]);
    moe.rows = parse::prefix_u32(r[Key::MOE_R]);
    moe.xmode = parse::prefix_u32(r[Key::MOE_XMODE]);
    moe.xmode_b = parse::prefix_u32(r[Key::MOE_XMODE_B]);
    moe.decode = parse::prefix_u32(r[Key::MOE_DEC]);
    moe.lanes_b = parse::prefix_u32(r[Key::MOE_LB]);
    moe.rows_b = parse::prefix_u32(r[Key::MOE_RB]);
    if (r[Key::MOE_HQUANT].nonempty())
        moe.hquant = static_cast<uint32_t>(std::atoi(r[Key::MOE_HQUANT].c_str()));
    moe.union_lanes = parse::prefix_u32(r[Key::MOE_UNION_L]);
    moe.union_rows = parse::prefix_u32(r[Key::MOE_UNION_R]);
    moe.union_xmode = parse::prefix_u32(r[Key::MOE_UNION_XMODE]);
    moe.union_lanes_b = parse::prefix_u32(r[Key::MOE_UNION_LB]);
    moe.union_rows_b = parse::prefix_u32(r[Key::MOE_UNION_RB]);
    moe.wc_read = parse::not_zero(r[Key::MOE_WC_READ]).value_or(true);
    moe.static_m1 = parse::not_zero(r[Key::MOE_STATIC_M1]).value_or(true);
}

inline void RuntimeEnvironment::resolve_mask_wait() {
    const auto &r = raw;
    // An inactive BUDGET remains irrelevant, even if its bytes are malformed.
    if (r[Key::MASK_WAIT_TAU].present()) {
        mask_wait.enabled = true;
        const char *value = r[Key::MASK_WAIT_TAU].c_str();
        char *end = nullptr;
        mask_wait.tau = std::strtod(value, &end);
        if (end == value || *end || !std::isfinite(mask_wait.tau) || mask_wait.tau < 0 ||
            mask_wait.tau > 1) {
            mask_wait.error =
                Status{Err::InvalidArgument, "CACHEDMOE_MASK_WAIT_TAU must be in [0,1]"};
            return;
        }
        if (r[Key::MASK_WAIT_BUDGET].present()) {
            const char *budget = r[Key::MASK_WAIT_BUDGET].c_str();
            char *middle = nullptr;
            const auto n = std::strtoul(budget, &middle, 10);
            if (middle == budget || *middle != ',' || n > UINT32_MAX) {
                mask_wait.error = Status{
                    Err::InvalidArgument,
                    "CACHEDMOE_MASK_WAIT_BUDGET takes experts,milliseconds (0 means unlimited)"};
                return;
            }
            char *tail = nullptr;
            const auto ms = std::strtod(middle + 1, &tail);
            if (tail == middle + 1 || *tail || !std::isfinite(ms) || ms < 0) {
                mask_wait.error = Status{Err::InvalidArgument, "invalid weighted mask time budget"};
                return;
            }
            mask_wait.experts = static_cast<uint32_t>(n);
            mask_wait.milliseconds = ms;
        }
    }
}

} // namespace cachedmoe::configuration
