#include "runtime/engine.h"

#include <format>

#include "core/log.h"
#include "model/layout.h"
#include "storage/backend.h"

namespace deepmoe::runtime {
namespace {

std::string join_path(const std::string& dir, const std::string& name) {
    if (dir.empty()) return name;
    const char last = dir.back();
    if (last == '/' || last == '\\') return dir + name;
#if defined(_WIN32)
    return dir + "\\" + name;
#else
    return dir + "/" + name;
#endif
}

}  // namespace

Engine::~Engine() { shutdown(); }

Result<void> Engine::open_model_files() {
    const std::string& dir = cfg_.model_dir;

    auto open_one = [&](storage::File& f, const char* logical, bool required) -> Result<void> {
        // Honour the manifest's path when it has one, else the default name.
        std::string name = logical;
        if (const FileEntry* e = manifest_.file(logical); e && !e->path.empty()) name = e->path;
        storage::FileFlags flags = storage::FileFlags::Overlapped | storage::FileFlags::Random;
        if (cfg_.io.unbuffered) flags = flags | storage::FileFlags::Unbuffered;
        auto r = storage::File::open(join_path(dir, name), flags);
        if (!r) {
            if (required) return std::unexpected(r.error());
            log_warn("engine: optional blob '{}' not opened: {}", name, r.error().str());
            return {};
        }
        f = *std::move(r);
        log_debug("engine: {} = {} ({})", logical, f.path(), human_bytes(f.size()));
        return {};
    };

    if (auto r = open_one(experts_, "experts", true); !r) return r;
    if (auto r = open_one(hot_, "hot", false); !r) return r;
    if (auto r = open_one(mtp_, "mtp", false); !r) return r;
    if (auto r = open_one(engram_l1_, "engramL1", false); !r) return r;
    if (auto r = open_one(engram_l14_, "engramL14", false); !r) return r;
    return {};
}

Result<void> Engine::init(const RuntimeConfig& cfg) {
    shutdown();
    cfg_ = cfg;

    if (!cfg_.profile_jsonl.empty()) {
        if (auto r = profiler_.open_jsonl(cfg_.profile_jsonl); !r) return r;
        log_info("engine: profiler -> {}", cfg_.profile_jsonl);
    }

    if (cfg_.model_dir.empty())
        return fail(Err::InvalidArgument, "RuntimeConfig::model_dir is empty");

    // config.json travels with the repacked model so a run is self-describing.
    auto mc = V41Config::load(join_path(cfg_.model_dir, "config.json"));
    if (!mc) return std::unexpected(mc.error());
    model_cfg_ = *std::move(mc);
    if (auto r = model_cfg_.validate_against_layout(); !r) return r;
    log_info("engine: {}", model_cfg_.summary());

    auto mf = Manifest::load(join_path(cfg_.model_dir, layout::kManifestFile));
    if (!mf) return std::unexpected(mf.error());
    manifest_ = *std::move(mf);
    if (auto r = manifest_.validate(); !r) return r;
    log_info("engine: manifest v{} '{}', {} tensors, {}", manifest_.version(), manifest_.model(),
             manifest_.tensors().size(), human_bytes(manifest_.total_bytes()));

    if (auto r = open_model_files(); !r) return r;

    auto backend = storage::make_default_backend(cfg_.io);
    if (!backend) return std::unexpected(backend.error());
    if (auto r = io_.start(std::move(*backend), cfg_.io, &profiler_); !r) return r;

    // Path A/B decide where the slabs live; without a GPU the host backing is
    // the honest choice and the only one the tests need (design §3.3).
    std::unique_ptr<store::SlabBacking> backing = std::make_unique<store::HostSlabBacking>();
    if (auto r = store_.init(std::move(backing), cfg_.cache,
                             layout::kTotalLogicalLayers, layout::kRoutedExperts); !r)
        return r;

    if (auto r = planner_.init(store_, io_, experts_, cfg_.cache, cfg_.prefetch, &profiler_); !r)
        return r;

    token_ = 0;
    ready_ = true;
    return {};
}

// TODO(design §7, §15 P2): create the device, pick the memory path, allocate
// the slab pool through gpu/vulkan/memory.h, build the pipelines from
// gpu/shaders/*.spv, create the timeline and the KV cache. Every piece has an
// interface already; what is missing is the kernel bodies.
Result<void> Engine::init_gpu() {
    if (!ready_) return fail(Err::FailedPrecondition, "call init() first");
    if (auto r = device_.create(); !r) return r;
    if (auto r = device_.caps().check_required(); !r) return r;
    if (auto r = timeline_.create(device_, 0); !r) return r;
    // The remaining bring-up (memory path, slab re-backing, pipelines, KV)
    // depends on kernels that do not exist yet.
    return unimplemented("runtime::Engine::init_gpu (design §7, P2)");
}

void Engine::shutdown() {
    io_.stop();
    timeline_.destroy();
    device_.destroy();
    kv_.reset();
    profiler_.close();
    ready_ = false;
    gpu_ready_ = false;
}

// TODO(design §11, §9.7): encoder over the full prompt, decoder bounded replay
// over the last 128 tokens, expert-major streaming above the length threshold.
Result<void> Engine::prefill(std::span<const uint32_t>) {
    return unimplemented("runtime::Engine::prefill (design §11, P5)");
}

// TODO(design §7.14, §7.8): the per-token command buffer, the per-layer routing
// resolution and the timeline host-signal.
Result<SampleResult> Engine::decode_step() {
    return unimplemented("runtime::Engine::decode_step (design §7.14, P2/P3)");
}

// TODO(design §10): the DSpark draft/verify cycle around decode_step.
Result<GenerateResult> Engine::generate(std::span<const uint32_t>, const GenerateOptions&) {
    return unimplemented("runtime::Engine::generate (design §10, P4)");
}

std::string Engine::status() const {
    std::string s;
    s += std::format("model     {}\n", ready_ ? model_cfg_.summary() : std::string("(not loaded)"));
    s += std::format("experts   {} ({})\n", experts_.path(), human_bytes(experts_.size()));
    s += std::format("io        {}\n", io_.running() ? io_.backend_caps().name : "stopped");
    s += "          " + io_.stats().to_string();
    s += std::format("store     {}\n", store_.stats().to_string());
    s += std::format("planner   {}\n", planner_.stats().to_string());
    s += std::format("gpu       {}\n", device_.valid() ? device_.caps().device_name
                                                       : std::string("(not created)"));
    return s;
}

}  // namespace deepmoe::runtime
