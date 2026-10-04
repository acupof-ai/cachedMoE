#include <thread>
#include <chrono>
#include "runtime/engine.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <format>
#include <future>
#include <set>

#include "core/align.h"
#include "cpu/dequant.h"
#include "core/wc_read.h"
#include "core/log.h"
#include "gpu/vulkan/moe_kernels.h"   // gpu::default_shader_dir
#include "gpu/vulkan/prefill_kernels.h"
#include "runtime/engram_tables.h"
#include "model/layout.h"
#include "storage/backend.h"
#if defined(__linux__)
#include <dirent.h>
#include <sched.h>
#include <cstdio>
#endif

namespace deepmoe::runtime {

// See Engine::ms_eager_moe (Track LX): 0 on RADV, set by init_gpu.
namespace { std::atomic<int> g_ms_eager_default{1}; std::atomic<int> g_shared_early_default{0}; }

#if defined(__linux__)
namespace {

// Parses a cpulist ("8-15,24-31") into a cpu_set_t. False if nothing parsed.
bool parse_cpulist(const std::string& s, cpu_set_t& set) {
    CPU_ZERO(&set);
    bool any = false;
    size_t i = 0;
    while (i < s.size()) {
        char* end = nullptr;
        const long a = std::strtol(s.c_str() + i, &end, 10);
        if (end == s.c_str() + i) break;
        long b = a;
        i = static_cast<size_t>(end - s.c_str());
        if (i < s.size() && s[i] == '-') {
            b = std::strtol(s.c_str() + i + 1, &end, 10);
            i = static_cast<size_t>(end - s.c_str());
        }
        for (long c = a; c <= b && c < CPU_SETSIZE; ++c) { CPU_SET(static_cast<int>(c), &set); any = true; }
        if (i < s.size() && s[i] == ',') ++i; else break;
    }
    return any;
}

std::string read_line(const std::string& path) {
    std::string out;
    if (std::FILE* f = std::fopen(path.c_str(), "r")) {
        char buf[512];
        if (std::fgets(buf, sizeof buf, f)) out = buf;
        std::fclose(f);
    }
    while (!out.empty() && (out.back() == '\n' || out.back() == ' ')) out.pop_back();
    return out;
}

// Track LX (docs/STATUS.md §7 0h): the fence that ends every layer's attention
// chain is signalled by the amdgpu interrupt, and the host thread it wakes then
// reads the gate's ids out of host-visible memory. Strix Halo has two CCDs; a
// waiter on the OTHER CCD than the one the interrupt lands on pays a cross-CCD
// wake-up and cache-line transfer every layer. Measured on the dev machine
// (amdgpu IRQ effective on CPU 29 = CCD1): the hot step is 78-79.5 ms pinned to
// CCD1 against 90-100 ms pinned to CCD0, and an unpinned process lands on
// either, which is the 81-vs-93 ms bimodality the hot-step runs showed.
//
// So pin the calling thread -- before IoEngine starts, so its threads inherit
// it -- to the L3 domain of the CPU that services the amdgpu interrupt.
// DEEPMOE_CPU_AFFINITY: unset/"auto" = this; "off" = leave the scheduler alone;
// anything else is a cpulist used verbatim.
void pin_to_gpu_irq_ccd() {
    const char* env = std::getenv("DEEPMOE_CPU_AFFINITY");
    const std::string mode = env ? env : "auto";
    if (mode == "off") return;
    cpu_set_t set;
    std::string list;
    if (mode != "auto") {
        list = mode;
    } else {
        // Find the amdgpu interrupt and the CPU it is effectively delivered to.
        int cpu = -1;
        if (std::FILE* f = std::fopen("/proc/interrupts", "r")) {
            char line[4096];
            while (std::fgets(line, sizeof line, f)) {
                if (!std::strstr(line, "amdgpu")) continue;
                const int irq = std::atoi(line);
                const std::string eff =
                    read_line(std::format("/proc/irq/{}/effective_affinity_list", irq));
                if (!eff.empty()) cpu = std::atoi(eff.c_str());
                break;
            }
            std::fclose(f);
        }
        if (cpu < 0) {
            log_info("engine: cpu affinity auto: no amdgpu interrupt found; not pinning");
            return;
        }
        list = read_line(std::format("/sys/devices/system/cpu/cpu{}/cache/index3/shared_cpu_list", cpu));
        if (list.empty()) return;
        log_info("engine: cpu affinity auto: amdgpu interrupt on cpu {}, pinning to its L3 domain {}",
                 cpu, list);
    }
    if (!parse_cpulist(list, set)) {
        log_warn("engine: DEEPMOE_CPU_AFFINITY='{}' is not a cpulist; not pinning", list);
        return;
    }
    if (sched_setaffinity(0, sizeof set, &set) != 0)
        log_warn("engine: sched_setaffinity({}) failed (errno {}); not pinning", list, errno);
}

}  // namespace
#endif


// --- H1a: the hard cap on the auto-sized cache (see runtime/engine.h) -------
uint32_t auto_slot_cap() {
    static const uint32_t v = [] {
        const char* e = std::getenv("DEEPMOE_CACHE_SLOT_CAP");
        if (!e || !*e) return kAutoSlotCap;
        const long long x = std::atoll(e);
        return x < 0 ? kAutoSlotCap : static_cast<uint32_t>(x);   // 0 = no cap, deliberately
    }();
    return v;
}

uint64_t cap_auto_budget(uint64_t budget_bytes, uint64_t slot_bytes, uint32_t slot_cap) {
    if (slot_cap == 0 || slot_bytes == 0) return budget_bytes;
    const uint64_t cap_bytes = uint64_t(slot_cap) * slot_bytes;
    return budget_bytes > cap_bytes ? cap_bytes : budget_bytes;
}

uint32_t cache_backoff_step() {
    static const uint32_t v = [] {
        const char* e = std::getenv("DEEPMOE_CACHE_BACKOFF_SLOTS");
        const long long x = (e && *e) ? std::atoll(e) : 0;
        return x > 0 ? static_cast<uint32_t>(x) : kCacheBackoffSlots;
    }();
    return v;
}

std::vector<uint32_t> cache_backoff_slots(uint32_t start, uint32_t step, uint32_t attempts,
                                          uint32_t floor_slots) {
    std::vector<uint32_t> out;
    if (start == 0 || attempts == 0) return out;
    if (step == 0) step = kCacheBackoffSlots;
    uint32_t n = start;
    for (uint32_t i = 0; i < attempts; ++i) {
        out.push_back(n);
        if (n <= floor_slots) break;               // the floor is tried once, then we stop
        n = n > step + floor_slots ? n - step : floor_slots;
    }
    return out;
}

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

double ms_since(TimePoint t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

double ms_between(TimePoint a, TimePoint b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

float bf16_to_f32(uint16_t h) {
    const uint32_t b = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

// design §3.3: fill path A, then path B.
//
// Path A is the DEVICE_LOCAL|HOST_VISIBLE heap -- 74 GiB on this machine, and
// what `bench/results/heap_capacity_idle.csv` measured as the larger of the
// two. When it runs out the pool keeps going on path B (imported host memory,
// ~26 GiB) rather than stopping, which is the difference between a 100 GiB
// cache and a 74 GiB one. The ExpertStore never learns which it got: both
// hand back a (host_ptr, device_address) pair, which is the whole point of
// store::SlabBacking.
class DualPathBacking final : public store::SlabBacking {
public:
    // `a_cap` (0 = none) stops path A before the driver refuses. RADV never
    // refuses: it over-commits the VRAM heap and TTM migrates the excess to GTT
    // under the running decode (Linux, 2026-09-28: 41 slabs / 71.8 GiB landed on
    // a 64 GiB heap, VRAM 59 -> 41 GB and GTT 24 GB within 9 s, path B unused).
    DualPathBacking(std::unique_ptr<store::SlabBacking> a,
                    std::unique_ptr<store::SlabBacking> b, uint64_t a_cap = 0)
        : a_(std::move(a)), b_(std::move(b)), a_cap_(a_cap) {}

    Result<store::SlabMemory> allocate(uint64_t bytes) override {
        if (a_ && !a_done_ && a_cap_ && a_bytes_ + bytes > a_cap_) {
            log_info("slab pool: path A capped at {} after {} slabs ({} of heap budget "
                     "headroom), continuing on path B",
                     human_bytes(a_bytes_), a_slabs_, human_bytes(a_cap_));
            a_done_ = true;
        }
        if (a_ && !a_done_) {
            auto m = a_->allocate(bytes);
            if (m) { owner_.push_back({m->host_ptr, false}); ++a_slabs_; a_bytes_ += bytes; return m; }
            log_info("slab pool: path A full after {} slabs ({}), continuing on path B",
                     a_slabs_, m.error().str());
            a_done_ = true;
        }
        if (!b_) return fail(Err::ResourceExhausted, "path A is full and there is no path B");
        auto m = b_->allocate(bytes);
        if (!m) return m;
        owner_.push_back({m->host_ptr, true});
        ++b_slabs_;
        return m;
    }

    void release(const store::SlabMemory& mem) override {
        for (auto it = owner_.begin(); it != owner_.end(); ++it) {
            if (it->first != mem.host_ptr) continue;
            (it->second ? b_ : a_)->release(mem);
            owner_.erase(it);
            return;
        }
    }
    const char* name() const override { return "path A+B"; }

    uint32_t a_slabs() const { return a_slabs_; }
    uint32_t b_slabs() const { return b_slabs_; }

private:
    std::unique_ptr<store::SlabBacking> a_, b_;
    std::vector<std::pair<void*, bool>> owner_;
    uint64_t a_cap_ = 0, a_bytes_ = 0;
    uint32_t a_slabs_ = 0, b_slabs_ = 0;
    bool     a_done_ = false;
};

// Track Y's four routing modes, for logs and for the end-of-run report.
const char* resident_only_name(Engine::ResidentOnly m) {
    switch (m) {
        case Engine::ResidentOnly::Off:    return "off";
        case Engine::ResidentOnly::All:    return "all";
        case Engine::ResidentOnly::Stall1: return "stall1";
        case Engine::ResidentOnly::Verify: return "verify";
        case Engine::ResidentOnly::Mask: return "mask";
    }
    return "?";
}

// VK_EXT_memory_budget's view of the HOST heap: what the OS says this process
// may still put there, given everything every other process already has. Zero
// when the extension is not there. Path B's imports are charged to this heap,
// and on this driver an import past the budget does not fail cleanly -- it
// returns VK_ERROR_INVALID_EXTERNAL_HANDLE and the device is lost on the next
// submit -- so a cache that sizes itself has to ask first.
// VK_EXT_memory_budget headroom (budget - usage) of heap `heap_index`, or 0
// when the extension is missing. The usage is the whole machine's, not this
// process's, on RADV.
uint64_t heap_headroom(const gpu::Device& d, uint32_t heap_index, uint64_t* budget_out,
                       uint64_t* usage_out) {
#if defined(DEEPMOE_ENABLE_VULKAN)
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(d.physical(), nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> ext(n);
    vkEnumerateDeviceExtensionProperties(d.physical(), nullptr, &n, ext.data());
    bool have = false;
    for (const VkExtensionProperties& e : ext)
        have |= std::strcmp(e.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0;
    if (!have) return 0;
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    props.pNext = &budget;
    vkGetPhysicalDeviceMemoryProperties2(d.physical(), &props);
    if (heap_index >= props.memoryProperties.memoryHeapCount) return 0;
    if (budget_out) *budget_out = budget.heapBudget[heap_index];
    if (usage_out) *usage_out = budget.heapUsage[heap_index];
    return budget.heapBudget[heap_index] > budget.heapUsage[heap_index]
               ? budget.heapBudget[heap_index] - budget.heapUsage[heap_index] : 0;
#else
    (void)d; (void)heap_index; (void)budget_out; (void)usage_out;
    return 0;
#endif
}

uint64_t host_heap_headroom(const gpu::Device& d, uint64_t* budget_out, uint64_t* usage_out) {
#if defined(DEEPMOE_ENABLE_VULKAN)
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(d.physical(), nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> ext(n);
    vkEnumerateDeviceExtensionProperties(d.physical(), nullptr, &n, ext.data());
    bool have = false;
    for (const VkExtensionProperties& e : ext)
        have |= std::strcmp(e.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0;
    if (!have) return 0;
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    props.pNext = &budget;
    vkGetPhysicalDeviceMemoryProperties2(d.physical(), &props);
    for (uint32_t i = 0; i < props.memoryProperties.memoryHeapCount; ++i) {
        if (props.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) continue;
        if (budget_out) *budget_out = budget.heapBudget[i];
        if (usage_out) *usage_out = budget.heapUsage[i];
        return budget.heapBudget[i] > budget.heapUsage[i]
                   ? budget.heapBudget[i] - budget.heapUsage[i] : 0;
    }
#else
    (void)d; (void)budget_out; (void)usage_out;
#endif
    return 0;
}

}  // namespace

Engine::~Engine() { shutdown(); }

Engine::Engine() {
    streams_.push_back(std::make_unique<Stream>());
    cur_ = streams_[0].get();
}

namespace {

// ';'-separated list, the way PATH is written on this platform. Empty entries
// are dropped so a trailing ';' is not an error.
std::vector<std::string> split_semis(const std::string& v) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i <= v.size()) {
        const size_t j = v.find(';', i);
        std::string part = v.substr(i, j == std::string::npos ? std::string::npos : j - i);
        while (!part.empty() && (part.back() == ' ' || part.back() == '"')) part.pop_back();
        while (!part.empty() && (part.front() == ' ' || part.front() == '"')) part.erase(part.begin());
        if (!part.empty()) out.push_back(std::move(part));
        if (j == std::string::npos) break;
        i = j + 1;
    }
    return out;
}

}  // namespace

Result<void> Engine::open_model_files() {
    // design §5.1 (v0.5): there are no repacked blobs. Every file the runtime
    // reads is an original safetensors shard named by the manifest.
    if (auto r = shards_.open_all(cfg_.model_dir, manifest_, cfg_.io.unbuffered); !r) return r;
    log_info("engine: {} shards open, {}", shards_.size(), human_bytes(shards_.total_bytes()));

    // Track D2 (docs/p4_dual_source.md): the second read source. --mirror wins
    // over the environment so a bench cell can turn it on or off without
    // touching the shell it inherited.
    std::vector<std::string> mirrors = cfg_.model_mirrors;
    if (mirrors.empty())
        if (const char* e = std::getenv("DEEPMOE_MODEL_MIRRORS"); e && *e)
            mirrors = split_semis(e);
    for (const std::string& dir : mirrors) {
        if (auto r = shards_.open_mirror(dir, manifest_, cfg_.io.unbuffered); !r) {
            // A mirror is an optimisation, never a correctness input: if the
            // drive is unplugged the run reads from the primary and says so.
            log_warn("engine: mirror '{}' unusable ({}); reading from the primary only",
                     dir, r.error().message);
            continue;
        }
        const auto& m = shards_.mirror(shards_.mirror_count() - 1);
        log_info("engine: mirror '{}' holds {} of {} shards, {}",
                 m.root, m.n_present, shards_.size(), human_bytes(m.bytes));
    }
    return {};
}

// The mirror-side health gate (Track D4, docs/p4_e_drive_diag.md §5.2).
//
// `open_mirror` succeeding means the files are there at the right length. It
// does NOT mean the drive can serve them: Track DX watched E: open all 48
// handles and then stop answering seventeen seconds later, and because the
// pinned load runs through the router (P3 is a routed class), that turned an
// optional mirror into `gpu init: io: pinned load of 'norm.weight': overlapped
// read failed` -- a fatal error from an optimisation. D2 §2's rule is that a
// mirror is never a correctness input, and this is the missing half of it.
//
// So: read the shape that failed. A few small tensors -- the pinned-weight
// size class, not 4 MiB expert chunks -- from the mirror, with every shard
// handle on both drives already open. Deterministic choice (sorted by name,
// one per shard) so two cells of an A/B read the same bytes.
//
// The probe opens its own handle rather than using the ShardSet's, for the
// reason probe_source_gbps does: a Win32 handle belongs to one completion port
// for life (storage/backend.h), and the ShardSet's are about to be the
// backend's.
Result<void> Engine::probe_mirror_health(uint32_t source) const {
    if (source == 0 || source > shards_.mirror_count())
        return fail(Err::OutOfRange, "no such mirror");
    const auto& m = shards_.mirror(source - 1);

    std::vector<std::string> names;
    for (const auto& [name, t] : manifest_.tensors()) {
        if (t.file >= m.present.size() || !m.present[t.file]) continue;
        if (t.bytes == 0 || t.bytes > (256ull << 10)) continue;   // pinned-weight sized
        names.push_back(name);
    }
    if (names.empty())
        return fail(Err::NotFound,
                    std::format("mirror '{}' holds no small tensor to probe with", m.root));
    std::sort(names.begin(), names.end());

    constexpr size_t kProbes = 8;
    std::set<uint32_t> shards_seen;
    size_t done = 0;
    for (const std::string& name : names) {
        if (done >= kProbes) break;
        auto rd = manifest_.tensor_read(name);
        if (!rd) continue;
        if (!shards_seen.insert(rd->file).second) continue;   // one per shard
        const std::string path = store::ShardSet::join(m.root, manifest_.files()[rd->file].path);
        storage::FileFlags flags =
            storage::FileFlags::Overlapped | storage::FileFlags::Random;
        if (cfg_.io.unbuffered) flags = flags | storage::FileFlags::Unbuffered;
        auto f = storage::File::open(path, flags);
        if (!f)
            return fail(f.error().code,
                        std::format("health probe: open '{}': {}", path, f.error().message));
        AlignedBuffer buf(rd->aligned_bytes, kPageSize);
        auto n = f->read_at(rd->aligned_off, MutBytes(buf.data(), rd->aligned_bytes));
        if (!n)
            return fail(n.error().code,
                        std::format("health probe: read of '{}' ({} B at {}) from '{}': {}",
                                    name, rd->aligned_bytes, rd->aligned_off, m.root,
                                    n.error().message));
        if (*n < uint64_t(rd->skew) + rd->bytes)
            return fail(Err::Io,
                        std::format("health probe: short read of '{}' from '{}': {} of {} B",
                                    name, m.root, *n, rd->aligned_bytes));
        ++done;
    }
    if (done == 0)
        return fail(Err::NotFound, std::format("mirror '{}': nothing probed", m.root));
    log_info("engine: mirror '{}' health probe: {} pinned-sized reads, all served", m.root, done);
    return {};
}

// Measures each source's 4 MiB random-read rate and hands the ratio to the
// router. The probe reads a mirrored shard through its own handle and closes
// it, so it never touches the handles the IoEngine is about to own.
Result<void> Engine::configure_io_sources() {
    if (shards_.mirror_count() == 0) return {};

    // Which shard to probe: one that every source holds, so the measurement is
    // of the same bytes on each drive.
    uint32_t probe_idx = UINT32_MAX;
    for (uint32_t i = 0; i < shards_.size(); ++i) {
        bool all = true;
        for (size_t m = 1; m <= shards_.mirror_count(); ++m)
            all &= shards_.at(i, static_cast<uint32_t>(m)) != nullptr;
        if (all && shards_.at(i)->size() >= (64ull << 20)) { probe_idx = i; break; }
    }

    std::vector<std::string> roots{cfg_.model_dir};
    std::vector<double>      weights{0.0};
    for (size_t m = 1; m <= shards_.mirror_count(); ++m) {
        roots.push_back(shards_.mirror(m - 1).root);
        weights.push_back(0.0);
    }

    // DEEPMOE_MIRROR_WEIGHTS=4.6;1.0 skips the probe: two seconds of startup is
    // two seconds, and an A/B that repeats a cell wants the same weights each
    // time rather than a fresh measurement's noise.
    bool probed = false;
    if (const char* e = std::getenv("DEEPMOE_MIRROR_WEIGHTS"); e && *e) {
        const auto parts = split_semis(e);
        for (size_t i = 0; i < parts.size() && i < weights.size(); ++i)
            weights[i] = std::strtod(parts[i].c_str(), nullptr);
    } else if (probe_idx != UINT32_MAX) {
        uint32_t ms = 1000;
        if (const char* e2 = std::getenv("DEEPMOE_MIRROR_PROBE_MS"); e2 && *e2)
            ms = static_cast<uint32_t>(std::strtoul(e2, nullptr, 10));
        // Track D5: warm up before measuring. A USB4 NVMe enclosure that has
        // been idle answers its first read in ~1 s; an unwarmed 1 s window
        // therefore measured E: at 0.03 GB/s against its real 3.77, the router
        // gave it 0.0% of the bytes and the second source bought nothing
        // (docs/p4_dual_source.md §9.2). The same 1 s window on a warm drive
        // reads 3.74 -- so the number was not noisy, it was the wake-up.
        uint32_t warmup = 1000;
        if (const char* e3 = std::getenv("DEEPMOE_MIRROR_PROBE_WARMUP_MS"); e3 && *e3)
            warmup = static_cast<uint32_t>(std::strtoul(e3, nullptr, 10));
        if (ms) {
            const std::string name = manifest_.files()[probe_idx].path;
            for (size_t i = 0; i < roots.size(); ++i) {
                auto g = storage::IoEngine::probe_source_gbps(
                    store::ShardSet::join(roots[i], name), ms, 8, warmup);
                if (!g) { log_warn("engine: probe of '{}' failed: {}", roots[i], g.error().message); continue; }
                weights[i] = *g;
                log_info("engine: source '{}' probes at {:.2f} GB/s "
                         "(4 MiB, QD 8, random, {} ms after a {} ms warmup)",
                         roots[i], *g, ms, warmup);
            }
            probed = true;
        }
    }
    for (double& w : weights) if (!(w > 0.0)) w = 1.0;
    (void)probed;

    io_.set_sources(roots, weights);

    // Track D4: the gate runs after set_sources so a mirror that fails it still
    // appears in status.json -- marked DROPPED, with the reason in the log --
    // rather than vanishing as if it had never been asked for.
    // DEEPMOE_MIRROR_HEALTH=0 skips it (for a deliberate negative test); the
    // default is on, because the whole point is that "it opened" is not enough.
    bool gate = true;
    if (const char* e = std::getenv("DEEPMOE_MIRROR_HEALTH"); e && *e)
        gate = std::strtol(e, nullptr, 10) != 0;
    if (gate) {
        for (size_t m = 1; m <= shards_.mirror_count(); ++m) {
            const uint32_t s = static_cast<uint32_t>(m);
            if (auto r = probe_mirror_health(s); !r) {
                log_warn("engine: mirror '{}' failed its health probe ({}); "
                         "dropped, reading from the primary only",
                         shards_.mirror(m - 1).root, r.error().message);
                io_.drop_source(s);
            }
        }
    }

    for (uint32_t i = 0; i < shards_.size(); ++i) {
        const storage::File* prim = shards_.at(i);
        if (!prim) continue;
        for (size_t m = 1; m <= shards_.mirror_count(); ++m) {
            const storage::File* alt = shards_.at(i, static_cast<uint32_t>(m));
            if (!alt) continue;
            if (auto r = io_.add_mirror(prim, static_cast<uint32_t>(m), alt); !r)
                log_warn("engine: shard {} not mirrored on source {}: {}", i, m, r.error().message);
        }
    }
    return {};
}

Result<void> Engine::init(const RuntimeConfig& cfg) {
    shutdown();
    cfg_ = cfg;
#if defined(__linux__)
    pin_to_gpu_irq_ccd();
#endif

    if (!cfg_.profile_jsonl.empty()) {
        if (auto r = profiler_.open_jsonl(cfg_.profile_jsonl); !r) return r;
        log_info("engine: profiler -> {}", cfg_.profile_jsonl);
    }

    // ADDITIVE (Track W): the per-dispatch trace. Opened before anything
    // else touches the device so the query pool below can be sized for it.
    if (!cfg_.trace_file.empty()) {
        if (auto r = tracer_.open(cfg_.trace_file); !r) return r;
        tracer_.bind_stamp(&Engine::trace_stamp, this);
        log_info("engine: per-dispatch trace -> {}", cfg_.trace_file);
    }

    if (cfg_.model_dir.empty())
        return fail(Err::InvalidArgument, "RuntimeConfig::model_dir is empty");

    // config.json ships with the checkpoint, so a run is self-describing.
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

    // The IO shape serve runs with (Track Q2's depth, Linux's 1 MiB x 8 P0,
    // the DEEPMOE_IO_* knobs), shared with the benches that stand in for it.
    storage::IoEngine::runtime_shape(cfg_.io);
    auto backend = storage::make_default_backend(cfg_.io);
    if (!backend) return std::unexpected(backend.error());
    if (auto r = io_.start(std::move(*backend), cfg_.io, &profiler_); !r) return r;
    // The router has to be in place before the pinned load's first request.
    if (auto r = configure_io_sources(); !r) return r;

    // Without a GPU the host backing is the honest choice and the only one the
    // storage tests need (design §3.3). init_gpu() re-backs the store.
    //
    // `budget_bytes == 0` means "size it from the machine", which only
    // init_gpu() can do -- and it would be absurd to commit gigabytes of host
    // memory here only to hand them straight back. A single slot is enough to
    // keep the store, the planner and the pointer table well-formed for a
    // caller that never brings up a GPU.
    CacheConfig boot = cfg_.cache;
    if (boot.budget_bytes == 0) {
        boot.slots_per_slab = 1;
        boot.budget_bytes   = layout::kExpertSlotBytes;
    }
    std::unique_ptr<store::SlabBacking> backing = std::make_unique<store::HostSlabBacking>();
    if (auto r = store_.init(std::move(backing), boot,
                             layout::kTotalLogicalLayers, layout::kRoutedExperts); !r)
        return r;

    if (auto r = planner_.init(store_, io_, manifest_, shards_, boot, cfg_.prefetch,
                               &profiler_); !r)
        return r;

    clock_ = 0;
    ready_ = true;
    return {};
}

// --- GPU bring-up -----------------------------------------------------------

Result<void> Engine::load_pinned() {
    // design §9.3's pinned set: attention, shared experts, router, mHC, norms,
    // engram wkv, embed and head, read once and never evicted. §9.3's 17.7 GB
    // also counts the three DSpark (mtp) blocks; decode does not touch them, so
    // what is loaded here is the forty backbone layers plus the three globals
    // and it measures 9.17 GiB (docs/p2_decode.md §2.1).
    std::vector<std::string> names = store::pinned_global_tensors(manifest_);
    for (uint32_t L = 0; L < model_cfg_.text.num_hidden_layers; ++L) {
        auto per = store::pinned_layer_tensors(manifest_, L);
        names.insert(names.end(), per.begin(), per.end());
    }
    if (cfg_.speculation.enabled) {
        auto mtp = DsparkRuntime::tensors(manifest_);
        names.insert(names.end(), mtp.begin(), mtp.end());
    }
    const uint64_t want = store::pinned_bytes(manifest_, names);

    // Both halves at once, before anything is allocated: design §5.2 / §9.2.2
    // measured the Windows commit limit as the binding constraint, and a
    // failure 14 GB into a 17.7 GB load is a worse diagnostic than a sentence.
    if (auto r = store::check_commit_available(want + cache_budget_,
                                               "pinned weights + expert cache"); !r)
        return r;

    store::PinnedConfig pc;
    pc.region_bytes = 1ull << 30;   // under the 2 GiB maxMemoryAllocationSize of §1.1
    auto pb = alloc_a_.make_slab_backing();
    if (!pb) return std::unexpected(pb.error());
    if (auto r = pinned_.init(std::move(*pb), pc); !r) return r;

    const TimePoint t0 = Clock::now();
    if (auto r = pinned_.load(manifest_, shards_, io_, names); !r) return r;
    const double s = ms_since(t0) / 1000.0;
    log_info("engine: pinned {} tensors, {} in {:.1f}s ({:.2f} GB/s), {} regions",
             pinned_.tensor_count(), human_bytes(pinned_.bytes_loaded()), s,
             s > 0 ? pinned_.bytes_loaded() / s / 1e9 : 0.0, pinned_.region_count());
    return {};
}

Result<void> Engine::build_expert_cache() {
    CacheConfig cache = cfg_.cache;
    cache.budget_bytes = cache_budget_;
    // Path-A headroom, held across the pool build and handed back after it
    // (docs/p4_hitrate.md, Track F4 §3). The slab pool fills path A until
    // `vkAllocateMemory` refuses, so on this machine a cache of 3,600 slots or
    // more takes *every* path-A slab -- and then the GPU prefill, which is
    // allocated later and only from path A, cannot get its workspace:
    //
    //   [WRN] session: GPU prefill failed (resource-exhausted: prefill buffers
    //   (1203765248 B): vkAllocateMemory(1203765248 B, type 2) failed (-2));
    //   falling back to the decode path
    //
    // at `--cache-slots 4500`, and again for an allocation of 10,584,064 B --
    // path A was empty, not merely short. The fallback is silent in every
    // metric except the clock: a 4,133-token prompt took 1,070 s of decode-path
    // prefill at 3.86 tok/s. `kPathAOther` already reserves this in the
    // *auto* budget, but `--cache-slots` / `--cache-gb` bypass that arithmetic
    // entirely, so the reserve has to be taken from the heap, not from a number.
    //
    // 4 GiB, not 2. A 2 GiB reserve is enough for a 512-token prompt at a
    // 4,096-position context (`ho_512_*`: the GPU prefill ran, 43.7 s against
    // 151 s on the decode path), but the same reserve with `--max-context 8192`
    // still lost the 1.12 GiB workspace to the same refusal, so roughly another
    // gigabyte of path A goes to whatever the larger context sizes up front.
    // 4 GiB covers both measured cases and costs ~214 slots (~0.4 points of hit
    // on the 8-turn curve) against an order of magnitude of TTFT.
    //
    // It is held in `maxMemoryAllocationSize` pieces: one 4 GiB allocation is
    // refused outright ("4294967296 B exceeds maxMemoryAllocationSize
    // 2147483648 B"), which is the same 2 GiB cap that made slabs exist
    // (design §1.1 / §5.3).
    constexpr uint64_t kPathAReserve = 4ull << 30;
    constexpr uint64_t kReservePiece = 2ull << 30;
    std::vector<gpu::GpuBuffer> reserve;
    for (uint64_t held = 0; held < kPathAReserve; held += kReservePiece) {
        auto r = alloc_a_.allocate_slab(std::min(kReservePiece, kPathAReserve - held));
        if (!r) {
            log_warn("engine: could only hold {} of the {} path-A reserve for the GPU "
                     "prefill ({}); the cache will take what it can and the prefill may "
                     "fall back to the decode path",
                     human_bytes(held), human_bytes(kPathAReserve), r.error().str());
            break;
        }
        reserve.push_back(*r);
    }
    struct ReserveGuard {
        gpu::MemoryAllocator& a; std::vector<gpu::GpuBuffer>& v;
        ~ReserveGuard() { for (gpu::GpuBuffer& b : v) a.free(b); }
    } reserve_guard{alloc_a_, reserve};

    auto a = alloc_a_.make_slab_backing();
    if (!a) return std::unexpected(a.error());
    std::unique_ptr<store::SlabBacking> b;
    if (auto rb = alloc_b_.make_slab_backing(); rb) b = std::move(*rb);
    // Path A cap (DEEPMOE_PATH_A_CAP=on), measured with the prefill reserve
    // above still held, so the reserve is what the post-cache allocations (KV,
    // prefill workspace, scratch) get back. DEFAULT OFF, and NO-GO on Linux
    // (STATUS §7 0h, 2026-09-28): RADV over-commits path A and TTM moves ~24 GB
    // of it to GTT, which costs nothing measurable (hot step 79.3 ms), while
    // capping sends 14 slabs to path B -- imported host pages, amdgpu userptr --
    // and the same step became 482 ms: submit 0.2 -> 356 ms (~8.7 ms per
    // vkQueueSubmit), moe gpu 32 -> 64 ms, engram 2.8 -> 8.9 ms. The Windows
    // driver refuses at the heap edge by itself, so it never needed this.
    uint64_t a_cap = 0;
    {
        bool want_cap = false;
        if (const char* e = std::getenv("DEEPMOE_PATH_A_CAP"))
            want_cap = std::strcmp(e, "on") == 0 || std::strcmp(e, "1") == 0;
        if (want_cap && b) {
            if (auto t = alloc_a_.chosen_memory_type(); t) {
                uint64_t hb = 0, hu = 0;
                a_cap = heap_headroom(device_, t->heap_index, &hb, &hu);
                log_info("engine: path A cap {} (heap budget {}, {} in use incl. the {} reserve)",
                         human_bytes(a_cap), human_bytes(hb), human_bytes(hu),
                         human_bytes(kPathAReserve));
                if (hb == 0) a_cap = 0;          // no VK_EXT_memory_budget: no cap
                else if (a_cap == 0) a_cap = 1;  // budget exhausted: go straight to path B
            }
        }
    }
    auto dual = std::make_unique<DualPathBacking>(std::move(*a), std::move(b), a_cap);
    DualPathBacking* raw = dual.get();

    if (auto r = store_.init(std::move(dual), cache, layout::kTotalLogicalLayers,
                             layout::kRoutedExperts); !r)
        return r;
    if (auto r = planner_.init(store_, io_, manifest_, shards_, cache, cfg_.prefetch,
                               &profiler_); !r)
        return r;
    // Track K1b: the store allocates through SlabBacking and cannot see which
    // path a slab came from, so tell it where path B starts. Everything from
    // this slab up is imported host memory, which the MoE kernel reads slower
    // (STATUS §3 row 30).
    store_.set_path_b_first_slab(raw->a_slabs());
    // Track K1b: which memory path evict_lru drains first. DEFAULT OFF -- a
    // single global LRU over both paths, which is what every measurement before
    // K1b used and what every measurement after it still uses.
    //
    // Both directions were measured and both are NO-GO (plan_p5.md §3(i),
    // STATUS §3 row 55):
    //
    //   `a`  the original hypothesis -- path B is 28% slower for the MoE kernel
    //        to READ, so free path-A slots and let new experts land there. The
    //        4-turn chat lost 5.4%: path-A hit share moved 1.5 points while
    //        nvme_stall rose 3.9 ms, because writing an admission into
    //        DEVICE_LOCAL|HOST_VISIBLE costs 704 us a 4 MiB ReadFile against
    //        70 us into ordinary host memory (Track Q2).
    //   `b`  the mirror. It won the 4-turn chat by 7.0% for exactly that reason
    //        -- stall -6.8 ms against +1.8 ms of read -- and then lost the
    //        8-turn script by 20.5%, because it also freezes path A: fills into
    //        path A stop at 3,400, i.e. the slab pool's path-A slots are filled
    //        once at cold start and never evicted again. That is a first-touch
    //        pin of two thirds of the cache, the shape §3 rows 24/35 already
    //        measured as negative, and a conversation that changes topic pays
    //        for it (hit 0.914 -> 0.879).
    //
    // The knob stays so the next attempt starts from the measurement rather
    // than from the hypothesis. What it would take is a placement preference
    // that is NOT also a pin -- e.g. taking the path-B victim only while its
    // last_use is within a bounded slack of the global LRU victim.
    if (const char* e = std::getenv("DEEPMOE_EVICT_PATH"); e && *e) {
        if (*e == 'a' || *e == 'A')
            store_.set_evict_path(store::ExpertStore::EvictPath::PreferA);
        else if (*e == 'b' || *e == 'B')
            store_.set_evict_path(store::ExpertStore::EvictPath::PreferB);
        log_info("engine: evict path preference '{}'", e);
    }
    log_info("engine: expert cache {} slots, {} ({} slabs on path A, {} on path B)",
             store_.slot_count(), human_bytes(store_.capacity_bytes()),
             raw->a_slabs(), raw->b_slabs());
    return {};
}

// H1a. One empty command buffer, submitted and waited on. Nothing in it
// matters: what matters is that a submit makes the process's allocations
// resident, which is the step that fails when the expert cache is too big for
// this machine. The alternative -- discovering it on the first real decode --
// is what killed the 5,100-slot run on 2026-09-19 (STATUS §3 row 59).
Result<void> Engine::probe_submit() {
    gpu::CommandPool pool;
    if (auto r = pool.create(device_); !r) return r;
    auto cb = pool.acquire();
    if (!cb) return std::unexpected(cb.error());
    if (auto r = cb->begin(); !r) return r;
    if (auto r = cb->barrier(); !r) return r;
    if (auto r = cb->end(); !r) return r;
    return gpu::submit_and_wait(device_, *cb);
}

Result<void> Engine::build_expert_cache_probed(bool backoff) {
    const uint64_t slot_bytes = layout::kExpertSlotBytes;
    const uint32_t want_slots = static_cast<uint32_t>(budget_slots(cache_budget_, slot_bytes));
    const uint32_t step       = cache_backoff_step();
    const std::vector<uint32_t> plan =
        backoff ? cache_backoff_slots(want_slots, step, kCacheBackoffTries)
                : std::vector<uint32_t>{want_slots};
    if (plan.empty()) return fail(Err::InvalidArgument, "expert cache budget is zero slots");

    std::string tried;
    for (size_t i = 0; i < plan.size(); ++i) {
        cache_budget_ = uint64_t(plan[i]) * slot_bytes;
        if (i) log_warn("engine: retrying the expert cache at {} slots ({} of {})", plan[i],
                        i + 1, plan.size());
        if (auto r = build_expert_cache(); !r) return r;
        auto probe = probe_submit();
        if (probe) {
            if (i) log_info("engine: cache probe passed at {} slots after {} refusal(s); the "
                            "budget-derived {} was not usable on this machine",
                            store_.slot_count(), i, want_slots);
            else log_info("engine: cache probe passed at {} slots", store_.slot_count());
            return {};
        }
        const std::string why = probe.error().str();
        if (!tried.empty()) tried += ", ";
        tried += std::format("{} -> refused", plan[i]);
        // A lost device cannot be reused: every queue, pipeline and allocation
        // made from it is gone, and recreating it here would mean tearing down
        // and rebuilding the allocators, the pinned set (17.7 GiB, ~10 s) and
        // every runner. That is a far bigger change than this fix, so the
        // honest move is to stop and name the number to pass.
        if (why.find("VK_ERROR_DEVICE_LOST") != std::string::npos) {
            return fail(Err::ResourceExhausted,
                        std::format(
                            "the expert cache probe LOST THE DEVICE at {} slots ({}). A lost "
                            "device cannot be reused in this process, so there is nothing to back "
                            "off to from here. Start again with --cache-slots {}, and keep going "
                            "down in steps of {} if it happens again "
                            "(docs/p4_hitrate.md §4, docs/STATUS.md §1 'cache capacity').",
                            plan[i], why, plan[i] > step ? plan[i] - step : 1, step));
        }
        log_warn("engine: expert cache probe refused at {} slots ({})", plan[i], why);
        planner_.stop_backfill();
        store_.reset();
        if (!backoff)
            return fail(Err::ResourceExhausted,
                        std::format(
                            "the expert cache probe was refused at {} slots ({}). This size was "
                            "asked for explicitly (--cache-slots / --cache-gb), so it is not "
                            "silently reduced. Retry with --cache-slots {} (100 fewer), or drop "
                            "the flag and let `auto` probe for a size that works.",
                            plan[i], why, plan[i] > 100 ? plan[i] - 100 : 1));
    }
    return fail(Err::ResourceExhausted,
                std::format("the expert cache probe was refused at every size tried ({}). Pass "
                            "--cache-slots with something smaller than {}.",
                            tried, plan.back()));
}

Result<void> Engine::resolve_weights() {
    // Resolved once per layer, not per token: the whole point of the pinned set
    // is that these addresses never move.
    weights_.clear();
    weights_.reserve(model_cfg_.text.num_hidden_layers);
    layer_hot_bytes_.assign(model_cfg_.text.num_hidden_layers, 0);
    for (uint32_t L = 0; L < model_cfg_.text.num_hidden_layers; ++L) {
        auto w = LayerWeights::from_pinned(pinned_, L);
        if (!w) return std::unexpected(w.error());
        weights_.push_back(*w);
        // Everything this layer pins, which is everything a token reads from
        // it -- minus the compressor and indexer tensors on a source layer,
        // which are pinned but not dispatched until design 7.4's kernels land.
        layer_hot_bytes_[L] =
            store::pinned_bytes(manifest_, store::pinned_layer_tensors(manifest_, L));
    }
    auto n = pinned_.require("norm.weight");
    if (!n) return std::unexpected(n.error());
    norm_w_ = (*n)->data;
    auto h = pinned_.require("head.weight");
    if (!h) return std::unexpected(h.error());
    head_w_ = (*h)->data;
    auto e = pinned_.require("embed.weight");
    if (!e) return std::unexpected(e.error());
    embed_ = *e;
    return {};
}

Result<void> Engine::init_gpu() {
    if (!ready_) return fail(Err::FailedPrecondition, "call init() first");
    if (gpu_ready_) return {};

    if (auto r = device_.create(); !r) return r;
    if (auto r = device_.caps().check_required(); !r) return r;
    if (device_.caps().driver_id == VK_DRIVER_ID_MESA_RADV) {
        g_ms_eager_default.store(0);
        g_shared_early_default.store(1);
        set_attn_ksplit_default(true);
        set_attn_cm_default(true);
    }
    if (auto r = cur_->timeline_.create(device_, 0); !r) return r;
    if (auto r = alloc_a_.init(device_, MemoryPath::DeviceLocalHostVisible); !r) return r;
    // Path B is optional: it only widens the expert cache. A machine that
    // cannot import host memory still runs, with a path-A-sized cache.
    if (auto r = alloc_b_.init(device_, MemoryPath::ExternalMemoryHost); !r)
        log_warn("engine: path B unavailable ({}); the expert cache is path A only",
                 r.error().str());

    // init() backed the store with HOST memory sized by `--cache-gb` -- 100 GiB
    // of commit for `--cache-gb 100` -- and build_expert_cache re-backs it on
    // the GPU paths anyway. Let it go first, or the commit check below counts
    // the cache twice and refuses (docs/p4_hitrate.md §2).
    planner_.stop_backfill();
    store_.reset();
    if (cfg_.engram_scales_resident && !engram_scales_) {
        std::vector<int64_t> layers;
        for (const EngramEntry& e : manifest_.engram()) layers.push_back(e.layer);
        const auto t0 = std::chrono::steady_clock::now();
        auto planes = EngramTables::load_scales(manifest_, shards_, layers);
        if (!planes) return std::unexpected(planes.error());
        engram_scales_ = *std::move(planes);
        uint64_t bytes = 0;
        for (const auto& [L, v] : engram_scales_->by_layer) bytes += v.size();
        log_info("engram scale planes resident: {} layers, {} in {:.1f} s", layers.size(), human_bytes(bytes),
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    cache_budget_ = cfg_.cache.budget_bytes;
    if (cache_budget_ == 0) {
        // "As much as the machine will give". On Windows that is the commit
        // charge (design §5.2 / §9.2.2), minus the pinned set and a margin for
        // everything else the process does. The pool itself stops early and
        // says so if an allocation refuses before the budget is spent, so an
        // over-estimate costs a log line rather than a failure; the ceiling is
        // there so a machine with a very large pagefile does not spend a minute
        // allocating memory a decode step will never touch.
        //
        // Both paths, each by what bounds IT (docs/p2_decode.md §10):
        //   path A  the DEVICE_LOCAL|HOST_VISIBLE heap (74 GiB here), which the
        //           pinned set, the KV store and the activations share. It
        //           charges commit but not physical memory.
        //   path B  imported host pages, which charge BOTH. Bounded by what is
        //           physically free now, less a floor for the OS and whatever
        //           else the machine is running.
        // and the sum by the commit that is actually available. P2 step 2
        // capped the whole thing at path A's heap size, which with the pinned
        // set also living there left ~64 GiB of experts and never touched B.
        constexpr uint64_t kCommitMargin = 8ull << 30;
        constexpr uint64_t kPhysFloor    = 12ull << 30;
        // What else lives on path A and is allocated AFTER the cache: the KV
        // store (16 MB at a 4,096-position context, but KvStoreConfig grows it
        // with --max-context, and the parked-session pool holds up to `max_parked`
        // of them), the decode scratch, the logits buffer and every pipeline's
        // runner: 1 GiB, the P2 figure for a 4K-context decode-only run. The
        // GPU prefill's workspace, which was the other 2 GiB of the old 3, is
        // priced by the prefill itself below (RADV) -- it grows with the prompt.
        constexpr uint64_t kPathAOther   = 1ull << 30;
        const uint64_t avail_commit = store::available_commit_bytes();
        const uint64_t avail_phys   = store::available_physical_bytes();
        std::vector<std::string> pnames = store::pinned_global_tensors(manifest_);
        for (uint32_t L = 0; L < model_cfg_.text.num_hidden_layers; ++L) {
            auto per = store::pinned_layer_tensors(manifest_, L);
            pnames.insert(pnames.end(), per.begin(), per.end());
        }
        if (cfg_.speculation.enabled) {
            auto mtp = DsparkRuntime::tensors(manifest_);
            pnames.insert(pnames.end(), mtp.begin(), mtp.end());
        }
        const uint64_t pinned = store::pinned_bytes(manifest_, pnames);
        uint64_t heap_a = 74ull << 30;
        if (auto t = alloc_a_.chosen_memory_type(); t)
            for (const gpu::HeapInfo& h : device_.caps().heaps)
                if (h.index == t->heap_index) heap_a = h.bytes;
        const uint64_t a_cache = heap_a > pinned + kPathAOther ? heap_a - pinned - kPathAOther : 0;
        const bool     have_b  = alloc_b_.path() == MemoryPath::ExternalMemoryHost;
        uint64_t b_cache = (have_b && avail_phys > kPhysFloor) ? avail_phys - kPhysFloor : 0;
        // Path B is also bounded by the HOST heap the driver reports (37.2 GiB
        // here), which is what imported pages are charged to. Physical memory
        // alone is not the limit: with 50 GB free, an import past ~33 GiB
        // failed with VK_ERROR_INVALID_EXTERNAL_HANDLE and the device was lost
        // on the next submit (docs/p2_decode.md §12.2). bench/heap_capacity's
        // mixed run stopped path B at 26 GiB; 10 GiB of the heap is left for
        // everything else host-heap-backed.
        //
        // On RADV (Linux) the same heap is the GTT, and what it bounds is not an
        // import but where TTM moves the path-A over-commit (path B is never
        // used there, STATUS §7 0h). The 8-turn chat at 4,450 slots (78 GiB)
        // peaked at 30.6 of 31.2 GiB GTT with 26 GB of RAM still free, and ran
        // 5.31 tok/s against auto's 4.90 at 4,100 (hit 0.9045 vs 0.8965), so
        // there the margin is the ~2 GiB of non-cache GTT plus 3 of slack.
        const uint64_t kHostHeapMargin =
            device_.caps().driver_id == VK_DRIVER_ID_MESA_RADV ? (5ull << 30) : (10ull << 30);
        for (const gpu::HeapInfo& h : device_.caps().heaps)
            if (!h.device_local && h.bytes > kHostHeapMargin)
                b_cache = std::min(b_cache, h.bytes - kHostHeapMargin);
        // And by what the OS will actually grant now -- the heap is shared with
        // every other Vulkan process on the machine -- less a 4 GiB margin. On
        // this driver the budget is 35.4 GiB and the usage reads 0 whatever is
        // running, so it is logged and applied but is not what keeps an import
        // from failing.
        uint64_t hb = 0, hu = 0;
        const uint64_t headroom = host_heap_headroom(device_, &hb, &hu);
        if (hb) {
            constexpr uint64_t kBudgetMargin = 4ull << 30;
            b_cache = std::min(b_cache, headroom > kBudgetMargin ? headroom - kBudgetMargin : 0);
            log_info("engine: host heap budget {}, {} reported in use", human_bytes(hb),
                     human_bytes(hu));
        }
        // What does: a fixed ceiling on path B. Imports failed -- and took the
        // device with them -- at 33 GiB on an idle machine with 50 GB free, and
        // at 15.8 GiB while another track's GPU test held host-heap memory. The
        // 16 GiB that was under both put the auto-sized cache at ~80 GiB
        // (4,500 slots), and the hit-rate curve measured since then says that
        // is 1.5-2 points of hit and ~1.5 tok/s below what this machine can
        // hold: `--cache-slots 5,500` (96.3 GiB) ran the 8-turn script at hit
        // 0.9431 / 6.05 tok/s against 4,500's 0.9234, and 5,600 (98.1 GiB) is
        // where path B's 20th slab refuses (docs/p4_hitrate.md).
        //
        // So the ceiling is 30 GiB, which puts auto at ~92 GiB / ~5,240 slots:
        // above the 16 GiB that the contended run survived only because the
        // cache was small, and ~6 GiB under the measured refusal. The other
        // three bounds above are what protect the contended case -- imported
        // pages charge physical memory, so `avail_phys - kPhysFloor` shrinks
        // this machine's path B whenever another track is holding host memory --
        // and the slab pool stops and logs rather than failing if the OS
        // refuses early. `--cache-slots` / `--cache-gb` still ask for more
        // explicitly and own the risk.
        constexpr uint64_t kPathBAutoCeiling = 30ull << 30;
        b_cache = std::min(b_cache, kPathBAutoCeiling);
        uint64_t want = a_cache + b_cache;
        // RADV with the UMA carve-out at its 512 MB minimum (the x box since
        // 2026-09-28, STATUS §7 0h): both Vulkan heaps are the GTT -- RADV
        // splits ttm.pages_limit 2:1 into a "device-local" 75 GiB and a host
        // 37.5 GiB -- so the cache is bounded by their SUM, not by heap A. The
        // 8-turn chat at 5,500 slots (96.3 GiB) peaked at 107 of 112 GiB GTT and
        // ran 6.19 tok/s against auto's 5.78 at 5,000. The slab pool rounds down
        // to whole 100-slot slabs, so 4 GiB of slack (5,499 slots of budget)
        // landed on 5,400; 3 GiB (5,553) lands on the measured 5,500.
        const bool radv = device_.caps().driver_id == VK_DRIVER_ID_MESA_RADV;
        if (radv) {
            uint64_t heaps = 0;
            for (const gpu::HeapInfo& h : device_.caps().heaps) heaps += h.bytes;
            constexpr uint64_t kGttSlack = 3ull << 30;
            // The GPU prefill's workspace grows with the prompt (0.6 MB a
            // token: 10 GiB at 17K), and a submission that needs more GTT than
            // is left loses the device (STATUS §6 17). Hold the room the
            // longest prompt this engine admits would take, priced by the
            // prefill itself, up to kPrefillReserveTokens -- past that a prompt
            // has to fit what the cache leaves, and gpu_prefill refuses it
            // cleanly instead (a 65K reservation would be a third of the GTT).
            constexpr uint32_t kPrefillReserveTokens = 16384;
            gpu::PrefillConfig pc;
            pc.max_tokens = std::min(cfg_.max_context, kPrefillReserveTokens);
            pc.transit_segments = cfg_.prefill_transit_segments;
            const uint64_t prefill_ws = gpu::Prefill{}.workspace_bytes(model_cfg_.text, pc);
            const uint64_t other = pinned + kPathAOther + kGttSlack + prefill_ws;
            want = heaps > other ? heaps - other : 0;
            log_info("engine: RADV: GTT {} less pinned {}, {} fixed, {} prefill workspace at {} tokens",
                     human_bytes(heaps), human_bytes(pinned), human_bytes(kPathAOther + kGttSlack),
                     human_bytes(prefill_ws), pc.max_tokens);
        }
        if (avail_commit) {
            const uint64_t commit_cap =
                avail_commit > pinned + kCommitMargin ? avail_commit - pinned - kCommitMargin : 0;
            want = std::min(want, commit_cap);
        }
        cache_budget_ = std::max<uint64_t>(want, 8ull << 30);
        const uint64_t avail = avail_commit;
        // H1a: the arithmetic above is bounded by heaps, commit and physical
        // memory -- none of which can see that this machine loses the device
        // above 5,000 slots. Cap it with the MEASURED number and log both, so a
        // startup log always says whether the cap bit (engine.h, kAutoSlotCap).
        const uint64_t slot_bytes  = layout::kExpertSlotBytes;
        // kAutoSlotCap is the Windows driver losing the device above 5,000
        // slots; RADV ran 5,500 clean, so there only an explicit
        // DEEPMOE_CACHE_SLOT_CAP applies.
        const char*    cap_env     = std::getenv("DEEPMOE_CACHE_SLOT_CAP");
        const uint32_t slot_cap    = (radv && !(cap_env && *cap_env)) ? 0 : auto_slot_cap();
        applied_slot_cap_          = slot_cap;
        const uint64_t from_budget = cache_budget_;
        cache_budget_ = cap_auto_budget(cache_budget_, slot_bytes, slot_cap);
        log_info("engine: cache budget auto -> {} (path A {} after {} pinned, path B {} of "
                 "{} physical free; {} of commit available)",
                 human_bytes(cache_budget_), human_bytes(a_cache), human_bytes(pinned),
                 human_bytes(b_cache), human_bytes(avail_phys), human_bytes(avail));
        if (slot_cap == 0 && radv) {
            log_info("engine: RADV: no auto slot cap, budget from the GTT heaps -> {} slots ({})",
                     budget_slots(from_budget, slot_bytes), human_bytes(from_budget));
        } else if (slot_cap == 0) {
            log_warn("engine: auto slot cap DISABLED (DEEPMOE_CACHE_SLOT_CAP=0); the budget-derived "
                     "{} slots ({}) are what this run will try to allocate -- above 5,000 slots "
                     "this machine has lost the device on the first submit (STATUS §3 row 59)",
                     budget_slots(from_budget, slot_bytes), human_bytes(from_budget));
        } else if (cache_budget_ < from_budget) {
            log_info("engine: auto slot cap {} applied: budget-derived {} slots ({}) -> {} slots "
                     "({}). --cache-slots / --cache-gb bypass this cap and own the risk; "
                     "DEEPMOE_CACHE_SLOT_CAP changes it",
                     slot_cap, budget_slots(from_budget, slot_bytes), human_bytes(from_budget),
                     budget_slots(cache_budget_, slot_bytes), human_bytes(cache_budget_));
        } else {
            log_info("engine: auto slot cap {} not reached: budget-derived {} slots ({})",
                     slot_cap, budget_slots(cache_budget_, slot_bytes), human_bytes(cache_budget_));
        }
    }

    if (auto r = load_pinned(); !r) return r;
    // H1a: auto probes rather than computes -- build, submit once, and back the
    // slot count off if the submit is refused. An explicit --cache-slots /
    // --cache-gb is probed too, but is never silently reduced.
    if (auto r = build_expert_cache_probed(/*backoff=*/cfg_.cache.budget_bytes == 0); !r) return r;
    if (auto r = resolve_weights(); !r) return r;

    const TextConfig& c = model_cfg_.text;
    build_ced_plan();
    // Track MS: the per-sequence half, one stream to start with. Everything a
    // second stream needs is in `create_stream`, so `set_streams(n)` is the
    // only thing a multi-stream caller has to say.
    if (auto r = set_streams(1); !r) return r;

    if (tracer_.enabled()) {
        tracer_.set_period_ns(device_.caps().timestamp_period_ns);
        tracer_.set_valid_bits(device_.caps().timestamp_valid_bits);
        cur_->layer_.set_tracer(&tracer_);
    }

    if (const char* e = std::getenv("DEEPMOE_ROUTE_DUMP"); e && *e && !route_dump_) {
        route_dump_ = std::fopen(e, "ab");
        if (route_dump_) log_info("engine: routing dump -> {}", e);
        else log_warn("engine: cannot open the routing dump '{}'", e);
    }
    if(const char* e=std::getenv("DEEPMOE_SPEC_DIAGNOSTICS");e && *e && !spec_diagnostics_) {
        spec_diagnostics_=std::fopen(e,"ab");
        if(!spec_diagnostics_)return fail(Err::Io,"cannot open speculative diagnostics");
    }
    // docs/p4_hitrate.md §4: on unless DEEPMOE_MOE_OVERLAP=0 (the A/B switch).
    if (const char* e = std::getenv("DEEPMOE_MOE_OVERLAP"); e && *e == '0') overlap_ = false;
    if (const char* e = std::getenv("DEEPMOE_GATE_PROBE"); e && *e && *e != '0') gate_probe_ = true;
    // Track Y (docs/p4_resident_routing.md): off | all | stall1 | verify.
    // Anything else is off.
    if (const char* e = std::getenv("DEEPMOE_ROUTE_RESIDENT_ONLY"); e && *e) {
        const std::string_view v{e};
        if (v == "all") resident_only_ = ResidentOnly::All;
        else if (v == "stall1") resident_only_ = ResidentOnly::Stall1;
        else if (v == "verify") resident_only_ = ResidentOnly::Verify;
        else if (v == "mask") resident_only_ = ResidentOnly::Mask;
        else if (v != "off" && v != "0" && v != "")
            log_warn("DEEPMOE_ROUTE_RESIDENT_ONLY={}: expected off|all|stall1|verify|mask, "
                     "using off", v);
        if (resident_only_ != ResidentOnly::Off)
            log_info("route: resident-only={} -- {}", resident_only_name(resident_only_),
                     resident_only_ == ResidentOnly::Mask
                         ? "initial cache only; no LRU eviction; skip miss computation without waiting or renormalising"
                     : resident_only_ == ResidentOnly::All
                         ? "a layer's non-resident experts are dropped and renormalised, "
                           "never waited for"
                     : resident_only_ == ResidentOnly::Verify
                         ? "one decode step in five routes exactly (the DSpark block's "
                           "first position); the other four drop and renormalise and "
                           "never wait"
                         : "a layer's non-resident experts are dropped and renormalised, "
                           "except the single highest-weight one, which is fetched at P0");
    }
    set_resident_only(resident_only_);
    // `verify`'s two halves, for the sweep of docs/p4_resident_routing.md §10:
    // what the block's first position does (`exact` = off, the default, or
    // `stall1`) and what its four draft positions do (`all`, the default --
    // never wait -- or `stall1`, one P0 fetch a layer). Anything else keeps the
    // default.
    if (const char* e = std::getenv("DEEPMOE_VERIFY_FIRST"); e && *e) {
        const std::string_view v{e};
        if (v == "stall1") verify_first_ = ResidentOnly::Stall1;
        else if (v == "all") verify_first_ = ResidentOnly::All;
        else if (v != "exact" && v != "off")
            log_warn("DEEPMOE_VERIFY_FIRST={}: expected exact|stall1|all, keeping exact", v);
    }
    if (const char* e = std::getenv("DEEPMOE_VERIFY_DRAFT"); e && *e) {
        const std::string_view v{e};
        if (v == "stall1") verify_draft_ = ResidentOnly::Stall1;
        else if (v == "exact" || v == "off") verify_draft_ = ResidentOnly::Off;
        else if (v != "all")
            log_warn("DEEPMOE_VERIFY_DRAFT={}: expected all|stall1|exact, keeping all", v);
    }
    // Track Y step 3: the background miss window, in decode steps.
    if (const char* e = std::getenv("DEEPMOE_RESIDENT_QUEUE_STEPS"); e && *e) {
        const int v = std::atoi(e);
        if (v >= 1 && v <= 1024) rr_queue_steps_ = static_cast<uint32_t>(v);
        else log_warn("DEEPMOE_RESIDENT_QUEUE_STEPS={}: expected 1..1024, keeping {}", e,
                      rr_queue_steps_);
    }
    if (const char* e = std::getenv("DEEPMOE_RESIDENT_QUEUE_EXPERTS"); e && *e) {
        const int v = std::atoi(e);
        if (v >= 1 && v <= 4096) rr_outstanding_cap_ = static_cast<uint32_t>(v);
    }
    if (const char* e = std::getenv("DEEPMOE_PREFILL_HANDOFF"); e && *e == '0') handoff_ = false;
    if (cfg_.speculation.enabled) {
        if (cfg_.speculation.max_draft < 1 || cfg_.speculation.max_draft > 5 ||
            cfg_.speculation.accept_topk < 1 || cfg_.speculation.accept_topk > model_cfg_.text.vocab_size)
            return fail(Err::InvalidArgument, "DSpark requires draft length 1..5 and valid acceptance top-K");
        if(cfg_.speculation.min_confidence && !std::isfinite(*cfg_.speculation.min_confidence))
            return fail(Err::InvalidArgument,"DSpark confidence threshold must be finite");
        if (store_.slot_count() < 400) return fail(Err::ResourceExhausted, "DSpark needs 384 pinned MTP slots plus main-model working slots");
        std::array<uint16_t,128> ids;
        std::array<float,128> weights; weights.fill(1.0f);
        for (uint32_t i=0;i<128;++i) ids[i]=uint16_t(i);
        for (uint32_t st=0;st<3;++st) {
            store::RouteDecision route{40+st,ids,weights,{},{}};
            auto plan=planner_.plan_layer(route,++clock_);
            if (!plan) return std::unexpected(plan.error());
            if (auto r=planner_.wait_layer(*plan,std::chrono::seconds(180));!r) return r;
            for (auto id:ids) if (auto r=planner_.pin({uint16_t(40+st),id});!r) return r;
        }
        dspark_=std::make_unique<DsparkRuntime>();
        if (auto r=dspark_->create(device_,alloc_a_,pinned_,store_,planner_,model_cfg_.text);!r) return r;
        log_info("engine: DSpark ready, 3 MTP stages / 384 pinned experts, draft {}, acceptance top-{}",cfg_.speculation.max_draft,cfg_.speculation.accept_topk);
    }
    gpu_ready_ = true;
    log_info("engine: gpu ready on {}", device_.caps().device_name);
    return {};
}

// model.py's `shared_attn`, written down. A source publishes its cache and
// every layer under it until the next source reads that same buffer; the
// reader's own `compress_ratio` still decides how much of it it may see.
void Engine::build_ced_plan() {
    const TextConfig& c = model_cfg_.text;
    ced_.assign(c.num_hidden_layers, CedPlan{});
    uint32_t cmp_src = 0, idx_src = 0;
    for (uint32_t L = 0; L < c.num_hidden_layers; ++L) {
        CedPlan& p = ced_[L];
        p.ratio           = c.compress_ratio(L);
        p.is_kv_source    = c.is_kv_source(L);
        p.is_index_source = c.is_index_source(L);
        if (p.is_kv_source) cmp_src = L;
        if (p.is_index_source) idx_src = L;
        p.cmp_src = cmp_src;
        p.idx_src = idx_src;
    }
}

// How many compressed positions each layer may read this step, and the window
// half of its top-k list. Both are pure functions of the position and the
// layer's ratio; the compressed half is the indexer's to write.
Result<void> Engine::prepare_ced(uint32_t position) {
    const TextConfig& c = model_cfg_.text;
    for (uint32_t L = 0; L < c.num_hidden_layers; ++L) {
        const CedPlan& p = ced_[L];
        const uint32_t n_cmp = p.ratio ? (position + 1) / p.ratio : 0u;
        if (n_cmp > kMaxIndexPositions)
            return fail(Err::ResourceExhausted,
                        std::format("layer {} wants {} compressed positions; the "
                                    "indexer's score plane holds {}", L, n_cmp,
                                    kMaxIndexPositions));
        // design §2.1's two-level top-k (layer 20 keeps candidate_topk_blocks
        // blocks, layers 24..36 score inside them) is DecodeLayer's: it
        // dispatches indexer.slang stages 6-8 once n_cmp passes 16,384.
        // A layer whose plane is read by others -- an index source, or a
        // window-only layer that has no source -- owns its top-k list; the rest
        // are pointed at their source's and only need the counts.
        //
        // The list is the window plus `min(index_topk, n_cmp)` compressed picks
        // (model.py Indexer.forward: `topk = min(self.index_topk, end_pos //
        // ratio)`), NOT the window plus every compressed position. Up to 512
        // compressed positions the two are the same number, which is why this
        // was invisible below ~1K tokens of context: past it, sparse_attn walked
        // n_cmp - 512 entries the indexer never wrote (zeros: window slot 0,
        // attended hundreds of times) and, past n_kv = 1024, overran its
        // per-head score stride -- the collapse and the NaN logits
        // docs/p3_prefill.md 8.3 item 2 reports (docs/p3_chat.md 5).
        const uint32_t n_sel = std::min<uint32_t>(n_cmp, c.index_topk);
        if (p.ratio == 0 || p.is_index_source) {
            if (auto r = cur_->kvs_.set_decode_topk(L, position, n_cmp, n_sel); !r) return r;
        } else if (auto r = cur_->kvs_.set_counts(L, n_cmp, c.sliding_window + n_sel); !r) {
            return r;
        }
    }
    return {};
}

Result<KvLayerView> Engine::effective_kv(uint32_t l) const {
    auto v = cur_->kvs_.layer(l);
    if (!v) return v;
    if (!produce_ced_ || l >= ced_.size() || ced_[l].ratio == 0) return v;
    auto cmp = cur_->kvs_.layer(ced_[l].cmp_src);
    if (!cmp) return cmp;
    auto idx = cur_->kvs_.layer(ced_[l].idx_src);
    if (!idx) return idx;
    v->cmp_kv       = cmp->cmp_kv;
    v->cmp_kv_host  = cmp->cmp_kv_host;
    v->top_idx      = idx->top_idx;
    v->top_idx_host = idx->top_idx_host;
    return v;
}

Result<void> Engine::load_decode_state(const std::string& dir) {
    if (!gpu_ready_) return fail(Err::FailedPrecondition, "call init_gpu() first");
    auto st = DecodeState::load(dir);
    if (!st) return std::unexpected(st.error());
    state_ = std::make_unique<DecodeState>(std::move(*st));

    const TextConfig& c = model_cfg_.text;
    // Sized from what the export actually holds, plus the room the remaining
    // steps will need; the compressed half grows by one row a step at ratio 1.
    // The PREFILL record's buffers are max_seq_len // ratio rows (2,074 / 4,149
    // at 4K, 8,513 / 17,026 at 17K) -- more than the widest per-step run
    // `max_compressed()` sees -- and seed_compressed is handed all of them.
    // Planes on the kv sources only (Track R2, docs/p4_kv_ux.md).
    KvStoreConfig kc = KvStoreConfig::for_model(
        c, std::max<uint32_t>(256, std::max(state_->max_compressed(), state_->max_prefill_rows()) + 64));
    if (kc.max_context > kMaxIndexPositions)
        return fail(Err::ResourceExhausted,
                    std::format("the export needs {} compressed positions and the "
                                "indexer's score plane holds {}", kc.max_context,
                                kMaxIndexPositions));
    if (auto r = cur_->kvs_.create(alloc_a_, kc); !r) return r;
    if (auto r = state_->seed_prefill(cur_->kvs_); !r) return r;
    // Producing the compressed KV and the top-k list is the default, and the
    // only thing that can stop it is an export that predates the prefill-state
    // record -- which cannot supply the index-key cache the indexer scores
    // against, and from which it cannot be recovered (§7.4's key comes off the
    // PRE-RoPE latent).
    produce_ced_    = state_->has_prefill_ced();
    cur_->prefill_loaded_ = true;

    auto tables = EngramTables::load(dir);
    if (!tables) return std::unexpected(tables.error());
    tables->scales = engram_scales_;
    if (auto r = cur_->engram_.create(device_, alloc_a_, cur_->dec_, manifest_, shards_, io_, pinned_, c,
                                *std::move(tables)); !r)
        return r;

    cur_->history_ = state_->prompt_ids();
    cur_->token_   = state_->prefill_len();
    clock_         = state_->prefill_len();
    log_info("engine: decode state loaded -- {} prompt tokens, {} reference steps, "
             "KV {} ({})",
             cur_->history_.size(), state_->steps(), human_bytes(cur_->kvs_.bytes()),
             produce_ced_ ? "prefill state seeded, every per-step tensor ours"
                          : "window REAL, compressed+topk LOADED per step");
    return {};
}

Result<void> Engine::reseed_decode_state() {
    if (!state_) return fail(Err::FailedPrecondition, "no decode state is loaded");
    cur_->kvs_.clear();
    if (auto r = state_->seed_prefill(cur_->kvs_); !r) return r;
    cur_->prefill_loaded_ = true;
    cur_->history_     = state_->prompt_ids();
    return {};
}

void Engine::shutdown() {
    // Everything that holds memory from an allocator has to let go before the
    // allocator does, and the allocators before the device.
    if (route_dump_) { std::fclose(route_dump_); route_dump_ = nullptr; }
    if(spec_diagnostics_){std::fclose(spec_diagnostics_);spec_diagnostics_=nullptr;}
    spec_diagnostics_started_=false;
    batch_route_requests_.clear();
    // ADDITIVE (Track W): the trace's name table and record count are written
    // on close, so a process that never closes leaves an unreadable file.
    if (cur_) cur_->layer_.set_tracer(nullptr);
    tracer_.close();
    // A P3 backfill writes into slab memory from the I/O threads: stop issuing
    // and let what is in flight land before the slabs go back.
    planner_.stop_backfill();
    if (io_.running()) io_.drain();
    state_.reset();
    dspark_.reset();
    for (auto& up : streams_) {
        Stream& s = *up;
        s.engram_.destroy();
        s.moe_.destroy();
        s.layer_.destroy();
        s.kvs_.destroy();
        s.tsq_.destroy();
        s.tok_pool_.destroy();
        s.tok_cmd_ = gpu::CommandBuffer{};
        for (gpu::CommandBuffer& c2 : s.se_cmd_) c2 = gpu::CommandBuffer{};
        s.tok_open_ = false;
        s.fence_.destroy();
        if (s.ffn_in_buf_.valid() && s.ffn_in_alloc_) s.ffn_in_alloc_->free(s.ffn_in_buf_);
        s.ffn_in_alloc_ = nullptr;
        if (s.logits_.valid()) alloc_a_.free(s.logits_);
        if (s.sample_.valid()) alloc_a_.free(s.sample_);
        if (s.topk_out_.valid()) alloc_a_.free(s.topk_out_);
        if (s.topk_hist_.valid()) alloc_a_.free(s.topk_hist_);
        // M1: the M > 1 forward's own runner, scratch and tail buffers.
        if (s.blogits_.valid()) alloc_a_.free(s.blogits_);
        if (s.bsample_.valid()) alloc_a_.free(s.bsample_);
        if (s.btopk_out_.valid()) alloc_a_.free(s.btopk_out_);
        if (s.btopk_hist_.valid()) alloc_a_.free(s.btopk_hist_);
        if (s.bdraft_.valid()) alloc_a_.free(s.bdraft_);
        if (s.brank_.valid()) alloc_a_.free(s.brank_);
        s.route_scratch_.destroy();
        s.bscratch_.destroy();
        s.mgt_.destroy();
        s.scratch_.destroy();
        s.dec_.destroy();
        s.attn_.destroy();
        s.timeline_.destroy();
        s.timings_.clear();
        s.history_.clear();
    }
    batch_ready_ = false;
    batch_cap_ = 0;
    store_.reset();      // the slabs came from alloc_a_/alloc_b_; give them back first
    pinned_.reset();
    io_.stop();
    shards_.close();
    alloc_a_.shutdown();
    alloc_b_.shutdown();
    device_.destroy();
    kv_.reset();
    weights_.clear();
    streams_.clear();
    streams_.push_back(std::make_unique<Stream>());
    cur_ = streams_[0].get();
    profiler_.close();
    ready_ = false;
    gpu_ready_ = false;
}

// --- Track MS: streams (docs/p4_multistream.md) ------------------------------

// The per-sequence resources of one stream. Everything here comes out of path
// A and is small; the expert cache, the pinned set and the planner are the
// process's and are never duplicated.
Result<void> Engine::create_stream(Stream& s) {
    const TextConfig& c = model_cfg_.text;
    const std::string dir = gpu::default_shader_dir();
    if (auto r = s.timeline_.create(device_, 0); !r) return r;
    if (auto r = s.attn_.create(device_, alloc_a_, dir); !r) return r;
    if (auto r = s.dec_.create(device_, alloc_a_, dir); !r) return r;
    if (auto r = s.scratch_.create(alloc_a_, 32ull << 20); !r) return r;
    auto lg = alloc_a_.allocate(uint64_t(c.vocab_size) * sizeof(float), true, true);
    if (!lg) return std::unexpected(lg.error());
    s.logits_ = *lg;
    auto sm = alloc_a_.allocate_host_coherent(sizeof(gpu::SampleOut));
    if (!sm) return std::unexpected(sm.error());
    s.sample_ = *sm;
    std::memset(s.sample_.host_ptr, 0, sizeof(gpu::SampleOut));
    {
        const uint64_t out_bytes =
            uint64_t(kTopKHeaderWords + kTopKThreads + 2ull * kTopKThreads * kTopKCapPerThread) * 4;
        auto to = alloc_a_.allocate_host_coherent(align_up(out_bytes, 4096));
        if (!to) return std::unexpected(to.error());
        s.topk_out_ = *to;
        std::memset(s.topk_out_.host_ptr, 0, static_cast<size_t>(s.topk_out_.bytes));
        auto th = alloc_a_.allocate_host_coherent(uint64_t(kTopKThreads) * kTopKBins * 4);
        if (!th) return std::unexpected(th.error());
        s.topk_hist_ = *th;
    }
    if (auto r = s.layer_.create(device_, s.attn_, s.scratch_, c); !r) return r;
    if (auto r = s.moe_.create(device_, alloc_a_, dir, store_, planner_, pinned_, c); !r) return r;
    // The MoE output stays on the GPU: the next layer's hc_post reads the
    // bridge's `y` by address instead of the host copying it into scratch.
    s.layer_.set_moe_output(s.moe_.y_address(), s.moe_.y_host());
    // The FFN input the host reads every layer, in cached host pages rather than
    // path A's write-combining mapping (DecodeLayer::set_ffn_input). Path A if
    // there is no path B: slower, still correct.
    {
        const uint64_t bytes = align_up(uint64_t(c.hidden_size) * sizeof(float), 1ull << 16);
        auto fb = alloc_b_.path() == MemoryPath::ExternalMemoryHost
                      ? alloc_b_.allocate_imported(bytes, /*device_address=*/true)
                      : alloc_a_.allocate(bytes, true, true);
        if (!fb) fb = alloc_a_.allocate(bytes, true, true);
        if (!fb) return std::unexpected(fb.error());
        s.ffn_in_buf_ = *fb;
        s.ffn_in_alloc_ = (fb->host_alloc != nullptr) ? &alloc_b_ : &alloc_a_;
        s.layer_.set_ffn_input(s.ffn_in_buf_.dev_addr, static_cast<float*>(s.ffn_in_buf_.host_ptr));
    }
    if (auto r = s.fence_.create(device_, 0); !r) return r;
    s.fence_value_ = 0;
    if (auto r = s.tok_pool_.create(device_); !r) return r;
    {
        auto cb = s.tok_pool_.acquire();
        if (!cb) return std::unexpected(cb.error());
        s.tok_cmd_ = *cb;
        for (gpu::CommandBuffer& c2 : s.se_cmd_) {
            auto cb2 = s.tok_pool_.acquire();
            if (!cb2) return std::unexpected(cb2.error());
            c2 = *cb2;
        }
    }
    // 40 layers x (attention + MoE) x 2 stamps, two engram layers, the tail.
    const uint32_t tsq_slots =
        tracer_.enabled() ? trace::Tracer::suggested_pool(c.num_hidden_layers) : 256u;
    if (auto r = s.tsq_.create(device_, tsq_slots); !r)
        log_warn("engine: no GPU timestamps ({}); the breakdown will be host-only",
                 r.error().str());
    s.timings_.assign(c.num_hidden_layers, LayerTiming{});
    s.route_ids_.assign(size_t(c.num_hidden_layers) * c.num_experts_per_tok, 0);
    return {};
}

Result<void> Engine::set_streams(uint32_t n) {
    if (n == 0) return fail(Err::InvalidArgument, "a run needs at least one stream");
    if (const char* e=std::getenv("DEEPMOE_BATCH_GPU_ROUTE"); n>1 && e && *e=='1')
        return fail(Err::FailedPrecondition,"batch GPU routing requires streams=1");
    if (n < streams_.size())
        return fail(Err::InvalidArgument,
                    std::format("streams only grow ({} are up, {} asked for)", streams_.size(), n));
    const size_t was = streams_.size();
    while (streams_.size() < n) {
        auto up = std::make_unique<Stream>();
        up->id = static_cast<uint32_t>(streams_.size());
        streams_.push_back(std::move(up));
    }
    for (auto& up : streams_)
        if (!up->made_) {
            if (auto r = create_stream(*up); !r) return r;
            up->made_ = true;
        }
    cur_ = streams_[0].get();
    if (streams_.size() != was)
        log_info("engine: {} decode stream(s)", streams_.size());
    return {};
}

Result<void> Engine::select_stream(uint32_t i) {
    if (i >= streams_.size())
        return fail(Err::OutOfRange, std::format("stream {} of {}", i, streams_.size()));
    cur_ = streams_[i].get();
    return {};
}

// TODO(design §11, §9.7): encoder over the full prompt, decoder bounded replay
// over the last 128 tokens, expert-major streaming above the length threshold.
Result<void> Engine::prefill(std::span<const uint32_t>) {
    return unimplemented("runtime::Engine::prefill (design §11, P5)");
}

Result<DecodeStepResult> Engine::slow_prefill(std::span<const uint32_t> prompt) {
    if (!gpu_ready_) return fail(Err::FailedPrecondition, "call init_gpu() first");
    if (prompt.empty()) return fail(Err::InvalidArgument, "the prompt is empty");
    if (!produce_ced_)
        return fail(Err::FailedPrecondition,
                    "slow_prefill produces the compressed KV and the top-k list, so "
                    "design 7.4's kernels have to be the ones running; "
                    "set_produce_ced(true)");
    const TextConfig& c = model_cfg_.text;
    if (prompt.size() > c.sliding_window)
        return fail(Err::Unimplemented,
                    std::format("{} prompt tokens against a {}-slot window: past the "
                                "window the ring wraps and a decode-shaped query no "
                                "longer sees the causal prefix, which is what design "
                                "11's chunked prefill is for",
                                prompt.size(), c.sliding_window));

    // Position 0 starts from nothing: an empty ring, an empty compressed plane,
    // an empty key cache and a compressor state of -inf.
    cur_->kvs_.clear();
    cur_->history_.assign(prompt.begin(), prompt.end());
    // `cur_->token_` is deliberately NOT reset. It is the cache's LRU clock, not a
    // position: winding it back makes everything already resident look newer
    // than what the next layer fetches, and the policy then evicts the slot it
    // just filled -- which surfaces as "expert (4, 1) is not resident at the
    // MoE dispatch" one dispatch later.

    const TimePoint t0 = Clock::now();
    DecodeStepResult last{};
    for (uint32_t p = 0; p < prompt.size(); ++p) {
        auto r = decode_step(prompt[p], p, -1);
        if (!r) return r;
        last = *r;
    }
    cur_->prefill_loaded_ = false;
    log_info("engine: slow prefill of {} tokens in {:.2f} s ({:.1f} ms/token), "
             "next token {}", prompt.size(), ms_since(t0) / 1000.0,
             ms_since(t0) / double(prompt.size()), last.token);
    return last;
}

// --- one decode step --------------------------------------------------------

// --- R1 round 2: per-turn reheat (docs/p4_hitrate.md §7) ---------------------

std::string Engine::HeatOrder::to_string() const {
    return std::format("reheat turn {}: decay {:.3f}, {} resident experts ranked in {:.1f} ms "
                       "({} above the floor), {} coldest freed ({} were free), {} keys to the backfill",
                       turn, decay, slots, ms, warm, evicted, free_slots, passed);
}

Result<Engine::HeatOrder> Engine::reheat(float decay) {
    if (!gpu_ready_) return fail(Err::FailedPrecondition, "call init_gpu() first");
    if (decay < 0.0f || decay > 1.0f)
        return fail(Err::InvalidArgument, std::format("reheat decay {} outside [0, 1]", decay));

    const TimePoint t0 = Clock::now();
    HeatOrder out;
    out.decay = decay;
    out.turn = ++reheat_turn_;
    out.free_slots = store_.free_slots();
    // The heat the store kept is the same EWMA the planner has been updating for
    // the chosen experts and the near misses, so it does not need a second
    // accumulator. Decaying it makes the last turn's routing the newest
    // information in the order.
    store_.decay_heat(decay);
    const std::vector<ExpertKey> heat = store_.heat_order();
    out.slots = static_cast<uint32_t>(heat.size());
    // How far down the order is still "this conversation". Everything below is
    // an expert the router has not wanted for several turns, and it is what the
    // pass is allowed to reclaim -- but only that: at 5,500 slots a full resident
    // set is ~1 GB of reads a slot, and a pass that evicted an eighth of it every
    // turn would spend 12 GB of NVMe on a topic that did not change. `decay_heat`
    // renormalises, so `head` is ~1.0 and these thresholds mean the same thing
    // from one turn to the next: 0.05 is a single weak note_heat, 10% of the
    // hottest expert in the cache.
    const float head = out.slots ? store_.slot_for(heat.front())->heat : 0.0f;
    const float floor_heat = std::max(0.05f, 0.1f * head);
    uint32_t warm = 0;                       // the prefix of the order at/above the floor
    while (warm < out.slots && store_.slot_for(heat[warm])->heat >= floor_heat) ++warm;
    // Fill the free slots first; when there are none, reclaim the tail. The tail
    // is where the previous topic lives, and evicting it is what makes room --
    // with no decay and no new routing this is exactly what the LRU would have
    // picked anyway, which is why it is a move of the reheat and not a second
    // eviction policy.
    uint32_t want = out.free_slots ? out.free_slots : std::max<uint32_t>(4, out.slots / 32);
    want = std::min(want, out.slots - warm);

    // What the backfill can actually do something with: keys that are NOT in the
    // cache. `Planner::backfill_pump` skips every resident key it is handed
    // (`if (store_->slot_of(key)) continue;`), so an order built resident-first
    // and then truncated to the budget -- which is what the first version did --
    // is a list of keys the pump walks straight past. That is why the 2,200-slot
    // A/B in docs/p4_hitrate.md §7 measured turn 3 decode hit 0.7210 against
    // 0.7210: on a saturated cache the pass evicted `slots/32` experts and then
    // fetched nothing, so it was a pure cost (the +32 MB/token of the §8 sweep).
    //
    // `StaticHeat` is the startup order (store/static_heat.inc, or a heat file
    // from the last run); its head is the best available estimate of what a
    // never-seen-token expert would score, and it is the only signal this
    // process has about an expert that is not resident -- `heat` lives on the
    // slot, so an evicted expert's heat is gone with it.
    std::vector<ExpertKey> order;
    order.reserve(want + out.free_slots);
    const std::vector<ExpertKey>& heat_table =
        heat_order_.empty() ? (heat_order_ = store::static_heat_order()) : heat_order_;
    for (const ExpertKey& k : heat_table) {
        if (order.size() >= size_t(want) + out.free_slots) break;
        if (!store_.resident(k)) order.push_back(k);
    }
    // Never evict without a candidate to put in the hole. With no non-resident
    // candidate left the pass is a no-op rather than a round of eviction the
    // next turn's misses have to pay back.
    const uint32_t have = static_cast<uint32_t>(order.size());
    want = out.free_slots >= have ? 0 : std::min(want, have - out.free_slots);
    for (uint32_t i = 0; i < want; ++i) {
        const ExpertKey& k = heat[out.slots - 1 - i];
        if (store_.evict_key(k)) ++out.evicted;
    }
    const uint32_t budget = out.free_slots + out.evicted;
    if (order.size() > budget) order.resize(budget);
    out.passed = static_cast<uint32_t>(order.size());
    out.warm = warm;
    out.ms = ms_since(t0);
    if (!order.empty())
        if (auto r = planner_.start_backfill(std::move(order), /*inflight=*/2, /*keep=*/true); !r)
            return std::unexpected(r.error());
    log_info("engine: {}", out.to_string());
    return out;
}

Result<void> Engine::embed_token(uint32_t token) {
    const TextConfig& c = model_cfg_.text;
    if (token >= c.vocab_size)
        return fail(Err::OutOfRange, std::format("token {} >= vocab {}", token, c.vocab_size));
    // `h = embed(ids).unsqueeze(2).repeat(1, 1, hc_mult, 1)`: one row, four
    // identical copies. The row is bf16 in a path-A mapping, so it is memcpy'd
    // out in one go (a write-combining READ, design §3.3 -- 10 KiB of it, about
    // 15 us) and widened on the host rather than read element by element.
    std::vector<uint16_t> row(c.hidden_size);
    std::memcpy(row.data(),
                static_cast<const std::byte*>(embed_->data_host) +
                    uint64_t(token) * c.hidden_size * 2,
                size_t(c.hidden_size) * 2);
    // Widen once into host memory, then memcpy each copy. Writing the four hc
    // copies interleaved (`x[j * dim + d]` with j innermost) would touch four
    // addresses 20 KiB apart per element, which flushes a write-combining
    // buffer per store instead of filling it -- the write-side twin of the
    // read problem runtime/moe_bridge.cpp documents.
    std::vector<float> wide(c.hidden_size);
    for (uint32_t d = 0; d < c.hidden_size; ++d) wide[d] = bf16_to_f32(row[d]);
    auto* x = static_cast<float*>(cur_->layer_.scratch().x.host);
    for (uint32_t j = 0; j < c.hc_mult; ++j)
        std::memcpy(x + size_t(j) * c.hidden_size, wide.data(),
                    size_t(c.hidden_size) * sizeof(float));
    // `make_identity_pre_mix`: the first sublayer collapses the four copies
    // onto copy 0. post/comb stay zero -- the first sublayer applies no hc_post.
    auto* mix = static_cast<float*>(cur_->layer_.scratch().mix_a.host);
    std::memset(mix, 0, 128);
    mix[0] = 1.0f;
    return {};
}

// --- the token loop's command buffer ----------------------------------------
//
// design §7.1 wants one pre-recorded buffer per token. What this implements is
// the closest thing the gate allows without the pointer table moving onto the
// GPU: the host must read a layer's ids between its gate and its MoE, so the
// buffer is CUT THERE and nowhere else. Each submit carries
//
//     [ MoE of layer L-1 ] [ attention of layer L, dispatches 1-9 + §7.4 ]
//
// with the MoE gated on the residency timeline (a semaphore wait inside the
// submit, design §7.1, not a host wait and a resubmit), and completion signals
// a second timeline the host fences on. A token is 40 of those plus one for the
// tail, plus one extra on each engram layer (below) -- 43 against the 128 of
// P2 step 2, whose shape was one submit per dispatch group.

Result<void> Engine::cmd_open() {
    if (cur_->tok_open_) return {};
    if (auto r = cur_->tok_cmd_.begin(); !r) return r;
    cur_->tok_open_ = true;
    if (cur_->tok_first_) {
        if (cur_->tsq_.count()) (void)cur_->tok_cmd_.reset_queries(cur_->tsq_, 0, cur_->tsq_.count());
        cur_->tsq_used_  = 0;
        cur_->tok_first_ = false;
    }
    return {};
}

uint32_t Engine::cmd_stamp() {
    if (!cur_->tok_open_ || cur_->tsq_.count() == 0 || cur_->tsq_used_ >= cur_->tsq_.count()) return ~0u;
    const uint32_t i = cur_->tsq_used_++;
    (void)cur_->tok_cmd_.write_timestamp(cur_->tsq_, i, /*bottom=*/true);
    return i;
}

Result<void> Engine::cmd_submit(TimelineValue wait_value) {
    if (!cur_->tok_open_) return {};
    if (auto r = cur_->tok_cmd_.end(); !r) return r;
    cur_->tok_open_ = false;
    gpu::Submission s;
    s.cmd = &cur_->tok_cmd_;
    TimelineValue w[1] = {wait_value};
    if (wait_value) {
        s.timeline    = &cur_->timeline_;
        s.wait_values = std::span<const TimelineValue>(w, 1);
    }
    s.signal_timeline    = &cur_->fence_;
    s.signal_value       = ++cur_->fence_value_;
    s.signal_on_complete = true;
    const TimePoint t0 = Clock::now();
    if (auto r = gpu::submit(device_, s); !r) {
        // H1a: the first submit of the process is where an over-sized expert
        // cache shows up. Every slab allocation succeeded -- the driver only
        // discovers it cannot make them all resident when work is queued -- so
        // `vkQueueSubmit2 failed (-2)` here means VK_ERROR_OUT_OF_DEVICE_MEMORY
        // on a cache that is too big, not a bug in the command buffer. 5,100
        // slots (34 path-A + 17 path-B slabs) died exactly here on 2026-09-19;
        // 5,000 (34 A + 16 B) ran clean. Say so, with the number to pass.
        if (!any_submit_ok_) {
            const uint64_t slots = store_.slot_count();
            return fail(Err::ResourceExhausted,
                        std::format(
                            "the FIRST GPU submit of this process failed ({}). The command buffer "
                            "is not the suspect: this is where a too-large expert cache is "
                            "discovered, because slab allocation succeeds and residency is only "
                            "checked when work is queued. This run has {} cache slots ({:.1f} GiB). "
                            "Retry with --cache-slots {} (100 fewer); if that also fails, keep "
                            "going down in steps of 100. See docs/p4_hitrate.md §4 (F4) and "
                            "docs/STATUS.md §1 'cache capacity'.",
                            r.error().str(), slots,
                            store_.capacity_bytes() / double(1ull << 30),
                            slots > 100 ? slots - 100 : 1));
        }
        return r;
    }
    any_submit_ok_ = true;
    if (cur_->open_guard_) { cur_->inflight_guard_ = cur_->open_guard_; cur_->open_guard_ = 0; }
    cur_->sub_ms_ += ms_since(t0);
    ++cur_->submits_;
    tracer_.note_submit();
    return {};
}

// Track R2 (docs/p4_kv_ux.md §2): the deadline on this wait used to be a flat
// 120 s, which is a POLICY ("the GPU is ours alone"), not a correctness check.
// A submission that is merely queued behind another process's work is
// indistinguishable from a wedged device at the semaphore, and 120 s of queueing
// is ordinary when several builds share this APU -- which is how a 64-step
// window replay died as `timeline wait for 5579 timed out` after running for
// 102 s. The budget is now DEEPMOE_GPU_WAIT_S (default 900 s), the wait is
// taken in slices so a slow machine says so instead of looking hung, and the
// failure names what it waited for and for how long.
double Engine::gpu_wait_budget_s() {
    static const double v = [] {
        const char* e = std::getenv("DEEPMOE_GPU_WAIT_S");
        const double x = e ? std::atof(e) : 0.0;
        return x > 0.0 ? x : 900.0;
    }();
    return v;
}

// Track G (docs/plan_p5.md (g)): the fence wake-up.
//
// `vkWaitSemaphores` parks this thread on a kernel object. When the GPU
// signals, the thread has to be made runnable and scheduled again, and the
// whole of that latency is paid INSIDE the gap the per-dispatch trace sees in
// front of every MoE dispatch -- the GPU is idle from the moment the gate
// dispatch retires until the host has woken, read the top-k, and submitted the
// MoE. Polling `vkGetSemaphoreCounterValue` instead trades a busy core for
// that latency.
//
// The trade is not free: the host arrives at this fence about 2 ms before the
// GPU finishes the layer's attention chain, so a spin long enough to catch the
// signal burns a core for ~85% of the step. DEEPMOE_FENCE_SPIN_US is therefore
// a budget in microseconds, default 0 = off (park immediately, the old
// behaviour); the spin always falls back to the blocking wait when the budget
// runs out, so no run can hang on it that would not have hung before.
double Engine::fence_spin_us() {
    static const double v = [] {
        const char* e = std::getenv("DEEPMOE_FENCE_SPIN_US");
        const double x = e ? std::atof(e) : 0.0;
        return x > 0.0 ? x : 0.0;
    }();
    return v;
}

// Track MS: the eviction guard with more than one stream in flight.
//
// A slot read by a submitted buffer is stamped with that buffer's guard number
// and may not be recycled until the store's completed timeline reaches it
// (design §5.3). With ONE stream the rule is "publish the guard of the buffer
// I just waited for", and that is what this does when `streams_.size() == 1`
// -- bit for bit the single-stream behaviour, including its conservatism about
// buffers that carry no guard of their own.
//
// With two, stream A's buffer can complete while stream B still has an OLDER
// guard outstanding, and publishing A's (larger) number would free slots B is
// still reading. So A's number is clamped to one below the oldest guard any
// OTHER stream has not yet proven complete -- which is any guard it has
// assigned for a layer still being recorded (`layer_guard_pending_`), any
// buffer it has recorded but not submitted (`open_guard_`), and the buffer it
// has submitted but not waited for (`inflight_guard_`).
void Engine::advance_store_guard(const Stream& me, TimelineValue mine) {
    if (!mine) return;
    TimelineValue safe = mine;
    const auto clamp = [&safe](TimelineValue v) {
        if (v && v - 1 < safe) safe = v - 1;
    };
    for (const auto& up : streams_) {
        if (up.get() == &me) continue;
        clamp(up->open_guard_);
        clamp(up->inflight_guard_);
    }
    store_.set_completed_timeline(safe);
}

Result<void> Engine::cmd_wait(uint64_t target) {
    const TimelineValue want = target ? TimelineValue(target) : cur_->fence_value_;
    const TimePoint t0 = Clock::now();
    if (const double spin_us = fence_spin_us(); spin_us > 0.0) {
        const double budget_ms = spin_us / 1000.0;
        for (;;) {
            auto v = cur_->fence_.value();
            if (!v) break;                       // fall through to the blocking wait
            if (*v >= want) {
                cur_->wait_ms_ += ms_since(t0);
                cur_->spin_hits_ += 1;
                if (cur_->inflight_guard_) {
                    const TimelineValue mine = cur_->inflight_guard_;
                    cur_->inflight_guard_ = 0;
                    advance_store_guard(*cur_, mine);
                }
                return {};
            }
            if (ms_since(t0) >= budget_ms) { cur_->spin_misses_ += 1; break; }
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
            __builtin_ia32_pause();
#endif
        }
    }
    const double budget_s = gpu_wait_budget_s();
    Result<void> r{};
    bool warned = false;
    for (;;) {
        r = cur_->fence_.wait(want, std::chrono::seconds(15));
        if (r || r.error().code != Err::Cancelled) break;   // done, or a real error
        const double waited = std::chrono::duration<double>(Clock::now() - t0).count();
        if (!warned && waited >= 15.0) {
            warned = true;
            log_warn("engine: still waiting for GPU fence {} after {:.0f} s (token {}, {} submits "
                     "this step) -- the queue is shared; giving it {:.0f} s "
                     "(DEEPMOE_GPU_WAIT_S)", want, waited, cur_->token_, cur_->submits_, budget_s);
        }
        if (waited >= budget_s) {
            r = fail(Err::Cancelled,
                     std::format("the GPU did not signal fence {} within {:.0f} s (token {}, "
                                 "{} submits this step). A queued submission and a wedged device "
                                 "look the same here: raise DEEPMOE_GPU_WAIT_S if the machine is "
                                 "shared", want, waited, cur_->token_, cur_->submits_));
            break;
        }
    }
    cur_->wait_ms_ += ms_since(t0);
    if (r && cur_->inflight_guard_) {
        const TimelineValue mine = cur_->inflight_guard_;
        cur_->inflight_guard_ = 0;
        advance_store_guard(*cur_, mine);
    }
    return r;
}

// Track BF: `read_timestamps` is the M = 1 path's, and it also fills a
// DecodeStepResult. The batch has no such result; all it owes the tracer is the
// raw tick prefix, after the LAST fence of the batch.
void Engine::flush_trace_batch() {
    if (!tracer_.enabled()) return;
    if (cur_->tsq_.count() == 0 || cur_->tsq_used_ == 0) return;
    auto raw = cur_->tsq_.read_range(0, cur_->tsq_used_);
    if (!raw) return;
    tracer_.token_end(raw->data(), cur_->tsq_used_, 0);
}

void Engine::read_timestamps(DecodeStepResult& res) {
    res.breakdown.gpu_timed = false;
    if (cur_->tsq_.count() == 0 || cur_->tsq_used_ == 0) return;
    auto raw = cur_->tsq_.read_range(0, cur_->tsq_used_);
    if (!raw) return;
    tracer_.token_end(raw->data(), cur_->tsq_used_, 0);
    const uint32_t bits = device_.caps().timestamp_valid_bits;
    const uint64_t mask = bits >= 64 ? ~0ull : ((1ull << bits) - 1);
    const double   ns   = device_.caps().timestamp_period_ns;
    auto span_ms = [&](const Stamp& s) {
        if (s.begin == ~0u || s.end == ~0u) return 0.0;
        const uint64_t a = (*raw)[s.begin] & mask, b = (*raw)[s.end] & mask;
        const uint64_t t = b >= a ? b - a : (mask - a) + b + 1;
        return double(t) * ns * 1e-6;
    };
    for (uint32_t L = 0; L < cur_->timings_.size(); ++L) {
        cur_->timings_[L].attn_ms    = span_ms(cur_->ts_attn_[L]);
        cur_->timings_[L].moe_gpu_ms = span_ms(cur_->ts_moe_[L]) + span_ms(cur_->ts_moe_early_[L]);
        cur_->timings_[L].moe_ms     = cur_->timings_[L].moe_gpu_ms + cur_->timings_[L].moe_host_ms;
        cur_->timings_[L].engram_ms  = cur_->engram_host_ms_[L] + span_ms(cur_->ts_engram_[L]);
    }
    res.breakdown.tail_ms   = span_ms(cur_->ts_tail_);
    res.breakdown.gpu_timed = true;
}

// The resident-only decision for ONE position, lifted out of `run_layer` so the
// M > 1 forward can make it per row of the batch (docs/p4_resident_routing.md
// §10: a verify batch routes row 0 exactly and the draft rows resident-only).
// `mode` is already resolved -- `Verify` is a phase rule the caller applies, not
// something this can see.
void Engine::set_resident_only(ResidentOnly m) {
    resident_only_ = m;
    const char* dynamic = std::getenv("DEEPMOE_MASK_DYNAMIC_LRU");
    store_.set_fixed_cache(m == ResidentOnly::Mask && !(dynamic && *dynamic == '1'));
    bool enabled = m == ResidentOnly::Mask;
    if (const char* e = std::getenv("DEEPMOE_IO_ENGRAM_DEADLINE"); e && *e == '0')
        enabled = false;
    io_.set_engram_wait_priority(enabled);
    log_info("engram wait priority: {}", enabled ? "on" : "off");
}

bool Engine::route_resident_only(uint32_t L, ResidentOnly ro, const uint32_t* ids_raw,
                                 const float* wts_raw, uint32_t topk, uint32_t* eff_ids,
                                 float* eff_w, uint16_t* kept_ids, float* kept_w,
                                 uint32_t& n_kept, std::vector<ExpertKey>& dropped) {
    const TextConfig& c = model_cfg_.text;
    uint8_t res[16];
    uint32_t n_res = 0;
    for (uint32_t i = 0; i < topk; ++i) {
        res[i] = store_.resident(ExpertKey{static_cast<uint16_t>(L),
                                           static_cast<uint16_t>(ids_raw[i])}) ? 1 : 0;
        n_res += res[i];
    }
    // (A) Stamp the LRU for every REQUESTED expert, resident or not. The
    // resident ones are stamped by the planner's `lookup` below; a
    // non-resident one has no slot to stamp, so its request stamp is parked
    // and handed to the P3 fetch, which is what lands in `last_use_token`
    // when the expert arrives. docs/p4_resident_routing.md section 8: step
    // 2 stamped the arrival at SUBMIT time, so an expert that was 5.6 s
    // late looked like the newest thing in the cache and evicted something
    // that was actually in use.
    {
        const TokenIndex ds = planner_.demand_stamp();
        for (uint32_t i = 0; i < topk; ++i) {
            if (res[i]) continue;
            const ExpertKey k{static_cast<uint16_t>(L), static_cast<uint16_t>(ids_raw[i])};
            rr_demand_[rr_pack(k)] = ds;   // a re-request overwrites with the newer stamp
        }
    }
    // `stall1`: one P0 fetch a layer, for the missing expert that carries
    // the most gate weight. At 18.8 MB / 4.5 GB/s that is about 4 ms
    // against a 2 ms layer -- the cheap middle ground between waiting for
    // up to six and waiting for none.
    if (ro == ResidentOnly::Stall1 && n_res < topk) {
        uint32_t best = topk;
        float    bw   = -1.0f;
        for (uint32_t i = 0; i < topk; ++i)
            if (!res[i] && wts_raw[i] > bw) { bw = wts_raw[i]; best = i; }
        if (best < topk) {
            const ExpertKey k{static_cast<uint16_t>(L),
                              static_cast<uint16_t>(ids_raw[best])};
            const TimePoint s0 = Clock::now();
            auto pr  = std::make_shared<std::promise<bool>>();
            auto fut = pr->get_future();
            // Shared, not captured by reference: `fetch` calls the callback
            // from the IoEngine thread on its own error paths too.
            auto f = planner_.fetch(k, IoPriority::BlockingMiss, clock_, L,
                                    [pr](bool ok) { pr->set_value(ok); },
                                    rr_demand_[rr_pack(k)]);
            if (f) {
                if (fut.wait_for(std::chrono::seconds(30)) == std::future_status::ready &&
                    fut.get() && store_.resident(k)) {
                    res[best] = 1;
                    ++n_res;
                    ++rr_.stall1_p0;
                    rr_demand_.erase(rr_pack(k));
                }
            }
            rr_.stall1_ms += ms_since(s0);
        }
    }
    // Nothing routed is resident: the layer runs on its shared expert alone,
    // and the seven slots still need one valid pointer-table row, so borrow
    // any expert this layer does have in the cache. With 5,100 slots over 40
    // layers this is ~1e-4 of layer-steps (docs/p4_resident_routing.md §2).
    // A layer with NOTHING resident cannot be expressed that way, so that
    // one case falls through to the ordinary demand path.
    uint32_t fill_id = ids_raw[0];
    bool     usable  = true;
    if (n_res == 0) {
        usable = false;
        for (uint32_t k = 0; k < c.n_routed_experts && !usable; ++k) {
            const uint32_t e = (ids_raw[0] + k) % c.n_routed_experts;
            if (store_.resident(ExpertKey{static_cast<uint16_t>(L),
                                          static_cast<uint16_t>(e)})) {
                fill_id = e;
                usable  = true;
            }
        }
    }
    if (!usable) return false;
    const ResidentRoute rr = resident_route(ids_raw, wts_raw, topk,
                                            std::span<const uint8_t>(res, topk),
                                            fill_id, eff_ids, eff_w);
    ++rr_.layers;
    rr_.requested     += topk;
    rr_.served        += rr.kept;
    rr_.skipped       += topk - rr.kept;
    rr_.mass_lost_sum += rr.mass_lost;
    rr_.shared_only   += rr.shared_only ? 1 : 0;
    n_kept = 0;
    for (uint32_t i = 0; i < topk; ++i) {
        if (!res[i]) continue;
        kept_ids[n_kept] = static_cast<uint16_t>(ids_raw[i]);
        kept_w[n_kept]   = eff_w[i];
        ++n_kept;
    }
    // The misses go to the background fetcher -- but only once this
    // layer's slots are guarded, or a P3 eviction could take a slot the
    // dispatch is about to read. Queued here, issued at the bottom of
    // the layer (`flush_resident_backfill`).
    for (uint32_t i = 0; i < topk; ++i)
        if (!res[i])
            dropped.push_back(ExpertKey{static_cast<uint16_t>(L),
                                        static_cast<uint16_t>(ids_raw[i])});
    return true;
}

// --- Track MS: `run_layer` in three phases (docs/p4_multistream.md) --------
//
// Phase boundaries are where a decode layer actually blocks:
//   (1) begin  ... ends having SUBMITTED the attention chain; the GPU is busy
//   (2) gate   ... waits that submit's fence, routes, and ISSUES the P0 misses
//   (3) moe    ... waits for residency (the NVMe stall) and records the MoE
// Between (2) and (3) this stream owns nothing but outstanding drive reads --
// which is exactly the ~2.5 ms/layer the single-stream engine spends idle
// (docs/p4_p0_queue.md) and exactly where another stream's (1) and (2) fit.
Result<void> Engine::layer_begin(Stream& s, uint32_t L, uint32_t position, bool& apply_post,
                                 LayerTiming& t) {
    cur_ = &s;
    Stream::LayerCtx& lc = s.lc_;
    lc = Stream::LayerCtx{};
    lc.t = &t;
    const TextConfig& c = model_cfg_.text;
    DecodeScratch& b = cur_->layer_.scratch();

    if (gate_probe_) cur_->gp_top_ = Clock::now();

    auto view = cur_->kvs_.layer(L);
    if (!view) return std::unexpected(view.error());

    LayerStep& st = lc.st;
    st.layer          = L;
    st.position       = position;
    st.compress_ratio = c.compress_ratio(L);
    st.apply_hc_post  = apply_post;
    st.kv             = *view;

    if (produce_ced_ && !ced_.empty() && st.compress_ratio) {
        const CedPlan& p = ced_[L];
        st.n_cmp          = view->n_cmp;
        st.run_compressor = p.is_kv_source;
        st.run_indexer    = p.is_index_source;
        st.cmp_complete   = ((position + 1) % st.compress_ratio) == 0;
        st.idx_key_write  = view->idx_key;
        // The caches this layer READS may belong to other layers: the compressed
        // KV and the index keys to the last kv_source at or below it, the
        // top-k list to the last index_source. The keys are the source's own
        // on every step, as in prefill -- NOT the reference decode's, which
        // scores a ratio-2 layer whose group is incomplete against whatever
        // was published last, i.e. layer 20's ratio-1 keys from the step
        // before (docs/p3_longctx.md §5.3; STATUS §7 0o).
        auto cmp = cur_->kvs_.layer(p.cmp_src);
        if (!cmp) return std::unexpected(cmp.error());
        auto idx = cur_->kvs_.layer(p.idx_src);
        if (!idx) return std::unexpected(idx.error());
        st.kv.cmp_kv  = cmp->cmp_kv;
        st.kv.top_idx = idx->top_idx;
        st.kv.idx_key = cmp->idx_key;
    }

    // The residency gate the open buffer's first dispatch -- the previous
    // layer's MoE -- is waiting on. Zero on layer 0, whose buffer has none.
    const TimelineValue prev_gate = L ? gpu::timeline_value(cur_->token_, L - 1) : 0;

    // The engram writes into the residual stream BEFORE the block (design
    // §2.1). design §7.7 normally defers the previous sublayer's hc_post into
    // this layer's first mega_mhc, so on an engram layer that hc_post has to be
    // materialised first -- otherwise the engram would be added to a stream
    // that is one sublayer behind.
    //
    // All of it joins the open buffer: the previous layer's MoE, its closing
    // hc_post, the engram's two dispatches and this layer's attention. The
    // rows were fetched at the start of the token (design §9.5), while layer
    // 0 ran; only a caller that skipped that pays for the fetch here.
    const bool engram = cur_->engram_.has_layer(L);
    lc.engram = engram;
    if (engram) {
        if (!cur_->engram_.fetched(L, position)) {
            const TimePoint f0 = Clock::now();
            if (auto r = cur_->engram_.fetch(L, cur_->history_, position); !r) return r;
            cur_->engram_host_ms_[L] += ms_since(f0);
        }
        if (auto r = cmd_open(); !r) return r;
        if (apply_post) {
            // `bind` for L-1 left MhcClose pointing at L-1's hc_ffn weights,
            // which is what closing L-1's block needs -- and `bind_close`
            // below stops this layer's `bind` from overwriting it before the
            // buffer runs.
            LayerStep prev = st;
            prev.layer = L - 1;
            if (auto r = cur_->layer_.record_close(cur_->tok_cmd_, prev); !r) return r;
        }
        cur_->ts_engram_[L].begin = cmd_stamp();
        const DeviceAddress in = apply_post ? b.xout.addr : b.x.addr;
        // One record, not two: the gemv and the gate are dispatched inside
        // EngramRunner::record, which Track W does not own.
        const uint32_t tr_eg = trace::open_dispatch(&tracer_, uint16_t(L), trace::Cls::Engram, 0,
                                                    "engram_gemv+gate");
        const TimePoint f0 = Clock::now();   // `record` waits for the rows to land
        if (auto r = cur_->engram_.record(cur_->tok_cmd_, L, in, b.x.addr); !r) return r;
        cur_->engram_host_ms_[L] += ms_since(f0);
        trace::close_dispatch(&tracer_, tr_eg);
        cur_->ts_engram_[L].end = cmd_stamp();
        apply_post = false;
        st.apply_hc_post = false;
        st.bind_close = false;
    }

    {
        const TimePoint b0 = Clock::now();
        if (auto r = cur_->layer_.bind(weights_[L], st); !r) return r;
        cur_->bind_ms_ += ms_since(b0);
    }
    if (auto r = cmd_open(); !r) return r;
    {
        const TimePoint r0 = Clock::now();
        cur_->ts_attn_[L].begin = cmd_stamp();
        if (auto r = cur_->layer_.record_attention(cur_->tok_cmd_, st); !r) return r;
        cur_->ts_attn_[L].end = cmd_stamp();
        cur_->rec_ms_ += ms_since(r0);
    }
    const TimePoint gp_sub0 = Clock::now();
    if (auto r = cmd_submit(prev_gate); !r) return r;
    const double gp_sub_us = gate_probe_ ? ms_since(gp_sub0) * 1000.0 : 0.0;
    // Track SE: the shared expert needs x and nothing else, and x exists the
    // moment the gate does. Submit its act_quant + dispatch A right behind the
    // gate, WITHOUT making it what layer_gate waits for, so the GPU computes it
    // while the host is waking up, reading the ids and submitting the routed
    // half -- the ~90-140 us a layer the per-dispatch trace shows idle in front
    // of every MoE dispatch on RADV.
    lc.gate_fence = cur_->fence_value_;
    if (shared_early_on()) {
        const uint64_t xa = cur_->layer_.ffn_in_addr();
        if (xa) {
            // Record into this layer's own buffer: swap it in for the length of
            // one cmd_open/cmd_submit so the stamps and the submit bookkeeping
            // are the ordinary ones.
            std::swap(cur_->tok_cmd_, cur_->se_cmd_[L & 1u]);
            struct SwapBack {
                Stream& st; uint32_t i;
                ~SwapBack() { std::swap(st.tok_cmd_, st.se_cmd_[i]); }
            } swap_back{*cur_, L & 1u};
            if (auto r = cmd_open(); !r) return r;
            const TimePoint r0 = Clock::now();
            const uint32_t tr_se = trace::open_dispatch(&tracer_, uint16_t(L), trace::Cls::Moe, 0,
                                                        "moe_shared_early");
            if (auto r = cur_->moe_.record_shared_early(cur_->tok_cmd_, L, xa); !r) return r;
            trace::close_dispatch(&tracer_, tr_se);
            cur_->rec_ms_ += ms_since(r0);
            if (auto r = cmd_submit(0); !r) return r;
            lc.shared_early = true;
        }
    }
    if (gate_probe_ && cur_->gp_open_) {
        GateSeg* g = static_cast<GateSeg*>(cur_->gp_open_);
        g->next_us += ms_between(cur_->gp_top_, Clock::now()) * 1000.0;
        g->sub_us  += gp_sub_us;
        cur_->gp_open_ = nullptr;
    }
    // design §9.5: the engram's 96 row reads depend only on the token ids, so
    // they go out the moment the first buffer of the token is on the GPU and
    // overlap it, instead of blocking the engram layers when they are reached.
    if (L == 0) {
        for (uint32_t E = 1; E < c.num_hidden_layers; ++E) {
            if (!cur_->engram_.has_layer(E)) continue;
            const TimePoint f0 = Clock::now();
            if (auto r = cur_->engram_.fetch(E, cur_->history_, position); !r) return r;
            cur_->engram_host_ms_[E] += ms_since(f0);
        }
    }
    return {};
}

Result<void> Engine::layer_gate(Stream& s, uint32_t L, uint32_t position) {
    cur_ = &s;
    Stream::LayerCtx& lc = s.lc_;
    LayerStep& st = lc.st;
    LayerTiming& t = *lc.t;
    const TextConfig& c = model_cfg_.text;
    DecodeScratch& b = cur_->layer_.scratch();
    (void)position;
    const TimePoint gp_w0 = Clock::now();
    lc.gp_w0 = gp_w0;
    if (auto r = cmd_wait(lc.gate_fence); !r) return r;
    const TimePoint gp_w1 = Clock::now();
    lc.gp_w1 = gp_w1;
    // The indexer wrote every compressed entry of the list sparse_attn just read.
    if (auto r = cur_->layer_.verify_after_attention(st); !r) return r;
    if (dspark_ && L>=37) capture_draft_hidden(L,position,static_cast<const float*>(b.xout.host));

    // design §7.1 / §7.8: the gate's ids are already in host-coherent memory.
    // Classify them, fetch the misses at P0, wait, host-signal the timeline the
    // MoE dispatch is gated on, then stage and record it.
    const uint32_t topk = c.num_experts_per_tok;
    const auto* ids_raw = static_cast<const uint32_t*>(b.gate_ids.host);
    const auto* wts_raw = static_cast<const float*>(b.gate_weights.host);
    const uint32_t*& ids = lc.ids;
    const float*&   wts = lc.wts;
    ids = ids_raw;
    wts = wts_raw;

    // --- Track Y: resident-only routing (docs/p4_resident_routing.md) ------
    // Drop the gate's non-resident experts, renormalise the rest over what is
    // left, and hand the dropped ones to the P3 fetcher. `ids`/`wts` below --
    // the MoE call, the guard, the route dump -- then describe what will
    // actually be computed, and `kept_*` is the subset the planner sees, so the
    // LRU is touched only by experts this step really used.
    uint32_t* eff_ids = lc.ids_eff;
    float*    eff_w = lc.wts_eff;
    uint16_t kept_ids[16];
    float    kept_w[16];
    uint32_t n_kept = topk;
    for (uint32_t i = 0; i < topk; ++i) {
        kept_ids[i] = static_cast<uint16_t>(ids_raw[i]);
        kept_w[i]   = wts_raw[i];
    }
    // `verify` picks one of the other two modes per STEP: the DSpark block's
    // first position routes exactly (mode off -- its misses are fetched at P0
    // and warm the cache for the rest of the block), the four draft positions
    // route resident-only. The phase is `cur_->token_ % 5`; which residue is the
    // exact one is arbitrary, since the block boundary is arbitrary.
    ResidentOnly ro = resident_only_;
    if (ro == ResidentOnly::Verify)
        ro = (cur_->token_ % kVerifyBlock) == 0 ? verify_first_ : verify_draft_;
    if (ro != ResidentOnly::Off && ro != ResidentOnly::Mask) {
        cur_->rr_pending_.clear();
        if (route_resident_only(L, ro, ids_raw, wts_raw, topk, eff_ids, eff_w, kept_ids, kept_w,
                                n_kept, cur_->rr_pending_)) {
            ids = eff_ids;
            wts = eff_w;
        }
    }
    MoeCall& call = lc.call;
    call.layer   = L;
    call.ids     = ids;
    call.weights = wts;
    call.topk    = topk;
    call.x       = cur_->layer_.ffn_norm_out();                  // ffn_norm output
    call.y       = nullptr;                                // stays on the GPU
    call.hidden  = c.hidden_size;
    // Track R1 (docs/p4_hitrate.md §4): when this layer waits on NVMe, dispatch
    // A over what is already resident -- the hits and the shared expert -- is
    // submitted before the wait instead of after it. `late` is what it did not
    // cover.
    cur_->layer_guard_pending_ = false;
    bool& split = lc.split;
    uint32_t* late = lc.late;
    uint32_t& n_late = lc.n_late;
    bool& staged = lc.staged;
    {
        const TimePoint g0 = Clock::now();
        lc.g0 = g0;
        uint16_t chosen[16];
        for (uint32_t i = 0; i < topk; ++i) chosen[i] = static_cast<uint16_t>(ids[i]);
        if (route_dump_)
            std::memcpy(cur_->route_ids_.data() + size_t(L) * topk, chosen, topk * sizeof(uint16_t));
        // design §9.3: the gate kernel already wrote the top-16 ids AND their raw
        // scores into the same two host-coherent buffers (gpu/shaders/gate.slang
        // stage 1, GatePush::record = 16), and the Planner's heat EWMA is defined
        // over them -- "so a bursty expert survives one bad round". They were
        // never passed, so every slot's heat stayed at exactly 0 in the decode
        // path: the reheat pass (docs/p4_hitrate.md §7) had nothing to rank, and
        // any future score-aware policy would have been ranking zeros. Entries
        // [topk, 16) carry the RAW score (only the first six are normalised by
        // route_scale, gate.slang), which is what the EWMA wants.
        uint16_t near_ids[16];
        float    near_scores[16];
        // The heat EWMA is defined over the GATE's own top-16 and raw scores, not
        // over what residency let through, so it reads the untouched buffers.
        for (uint32_t i = 0; i < 16; ++i) {
            near_ids[i]    = static_cast<uint16_t>(ids_raw[i]);
            near_scores[i] = wts_raw[i];
        }
        store::RouteDecision route;
        route.layer       = L;
        // Resident-only mode hands the planner exactly the experts the dispatch
        // will use, so the LRU is never touched by one that was skipped.
        route.chosen      = std::span<const uint16_t>(kept_ids, n_kept);
        route.weights     = std::span<const float>(kept_w, n_kept);
        route.near_ids    = std::span<const uint16_t>(near_ids, 16);
        route.near_scores = std::span<const float>(near_scores, 16);
        if (ro == ResidentOnly::Mask) {
            // Async fills can leave this hit as the only evictable slot. Hold
            // it while later misses in the SAME plan try to admit experts.
            guard_layer(L, {}, ids_raw);
            route.guard_hits = cur_->layer_guard_;
            cur_->open_guard_ = cur_->layer_guard_;
        }
        const TimePoint gp_p0 = Clock::now();
        lc.gp_p0 = gp_p0;
        auto plan = planner_.plan_layer(route, clock_);
        if (!plan) return std::unexpected(plan.error());
        lc.plan = std::move(*plan);
        if (ro == ResidentOnly::Mask) {
            // Use the planner's classification, not a second residency check:
            // a fill that completes during planning remains a miss this step.
            for (uint32_t i = 0; i < topk; ++i) {
                eff_ids[i] = ids_raw[i];
                const ExpertKey key{static_cast<uint16_t>(L), static_cast<uint16_t>(ids_raw[i])};
                eff_w[i] = std::find(lc.plan.hits.begin(), lc.plan.hits.end(), key) !=
                           lc.plan.hits.end() ? wts_raw[i] : 0.0f;
            }
            ids = eff_ids;
            wts = eff_w;
            call.ids = ids;
            call.weights = wts;
            ++rr_.layers;
            rr_.requested += topk;
            rr_.served += lc.plan.hits.size();
            rr_.skipped += lc.plan.misses.size();
            rr_.shared_only += lc.plan.hits.empty();
            double total = 0.0, kept = 0.0;
            for (uint32_t i = 0; i < topk; ++i) { total += wts_raw[i]; kept += eff_w[i]; }
            if (total > 0.0) rr_.mass_lost_sum += 1.0 - kept / total;
            // Touch an already-filling miss now; wait_layer normally does this
            // after joining it, but mask deliberately never joins the wait.
            for (const auto& [key, stamp] : lc.plan.joined) (void)store_.touch(key, stamp, true);
        }
        t.hits       = static_cast<uint32_t>(lc.plan.hits.size());
        t.misses     = static_cast<uint32_t>(lc.plan.misses.size());
        t.miss_bytes = lc.plan.miss_bytes;
        t.gate_ms    = ms_since(g0);
        if (ro != ResidentOnly::Mask && overlap_ && (!lc.plan.issued.empty() || !lc.plan.joined.empty())) {
            if (auto r = cur_->moe_.stage_input(call, lc.shared_early); !r) return r;
            staged = true;
            uint32_t early[16];
            uint32_t n_early = 0;
            for (uint32_t s = 0; s < topk; ++s) {
                const ExpertKey key{static_cast<uint16_t>(L), static_cast<uint16_t>(ids[s])};
                if (store_.resident(key)) early[n_early++] = s;
                else late[n_late++] = s;
            }
            if (n_late) {
                // The eviction guard (design §5.3, docs/p2_decode.md §13): the
                // early slots are read by a buffer that is in flight while the
                // host waits, so nothing may recycle them until the buffer that
                // carries dispatch B -- the last reader -- has completed.
                guard_layer(L, std::span<const uint32_t>(early, n_early), ids);
                if (auto r = cur_->moe_.stage_rows(call, std::span<const uint32_t>(early, n_early)); !r)
                    return r;
                // The shared expert, unless Track SE already sent it.
                if (!lc.shared_early) early[n_early++] = topk;
            }
            if (n_late && n_early == 0) {
                split = true;             // everything is late; the shared half is out already
            } else if (n_late) {
                if (auto r = cmd_open(); !r) return r;
                const TimePoint r0 = Clock::now();
                cur_->ts_moe_early_[L].begin = cmd_stamp();
                const uint32_t tr_me = trace::open_dispatch(&tracer_, uint16_t(L), trace::Cls::Moe,
                                                            0, "moe_gateup_early");
                if (auto r = cur_->moe_.record_gateup(cur_->tok_cmd_, std::span<const uint32_t>(early, n_early)); !r)
                    return r;
                trace::close_dispatch(&tracer_, tr_me);
                cur_->ts_moe_early_[L].end = cmd_stamp();
                cur_->rec_ms_ += ms_since(r0);
                if (auto r = cmd_submit(0); !r) return r;
                split = true;
            }
        }
    }
    return {};
}

Result<void> Engine::layer_moe(Stream& s, uint32_t L, bool& apply_post) {
    cur_ = &s;
    Stream::LayerCtx& lc = s.lc_;
    LayerTiming& t = *lc.t;
    const TextConfig& c = model_cfg_.text;
    MoeCall& call = lc.call;
    const uint32_t topk = c.num_experts_per_tok;
    const uint32_t* ids = lc.ids;
    const bool split = lc.split;
    const bool staged = lc.staged;
    uint32_t* late = lc.late;
    const uint32_t n_late = lc.n_late;
    const TimePoint gp_w0 = lc.gp_w0, gp_w1 = lc.gp_w1, gp_p0 = lc.gp_p0;
    {
        double gp_pwait_us = 0.0;
        {
            const TimePoint w0 = Clock::now();
            if (resident_only_ != ResidentOnly::Mask)
                if (auto r = planner_.wait_layer(lc.plan); !r) return std::unexpected(r.error());
            t.gate_ms += ms_since(w0);
            gp_pwait_us = ms_since(w0) * 1000.0;
            if (gate_probe_) {
                // (iii) is everything from the moment the ids are in hand to the
                // moment residency is proven: the planner's own lookup, the miss
                // issue, and -- on a miss layer -- Track R1's early dispatch.
                GateSeg& g = t.misses ? gp_miss_ : gp_hit_;
                g.fence_us += ms_between(gp_w0, gp_w1) * 1000.0;
                g.ids_us   += ms_between(gp_w1, gp_p0) * 1000.0;
                g.plan_us  += ms_between(gp_p0, w0) * 1000.0;
                g.pwait_us += gp_pwait_us;
                ++g.n;
                cur_->gp_open_ = &g;      // segments (iv) and (v) land in the same row
            }
        }
        // Every routed expert must be resident now. When one is not, say how
        // it got that way -- a hit that a later miss in the same layer evicted
        // and a fill whose read failed look identical at the MoE dispatch.
        for (uint32_t i = 0; i < topk; ++i) {
            if (resident_only_ == ResidentOnly::Mask && call.weights[i] == 0.0f) continue;
            const ExpertKey key{static_cast<uint16_t>(L), static_cast<uint16_t>(ids[i])};
            if (store_.resident(key)) continue;
            const bool was_hit = std::find(lc.plan.hits.begin(), lc.plan.hits.end(), key) !=
                                 lc.plan.hits.end();
            auto slot = store_.slot_for(key);
            return fail(Err::Internal,
                        std::format("layer {} expert {} is not resident after the gate: it was "
                                    "a {} this layer, its slot is {}; store: {}", L, ids[i],
                                    was_hit ? "HIT" : "miss",
                                    slot ? std::string(slot_state_name(slot->state))
                                         : std::string("gone"),
                                    store_.stats().to_string()));
        }
    }
    {
        const TimelineValue v = gpu::timeline_value(cur_->token_, L);
        auto cur = cur_->timeline_.value();
        if (cur && *cur < v)
            if (auto r = cur_->timeline_.signal(v); !r) return r;
    }
    profiler_.add_phase(Phase::NvmeStall, Nanos(int64_t(t.gate_ms * 1e6)));

    const TimePoint gp_s0 = Clock::now();
    if (split) {
        // The early dispatch reads the alternate slot list while it runs; it is
        // done by now in all but a pathological case, and this wait proves it
        // before the list is rewritten for the late slots.
        if (auto r = cmd_wait(); !r) return r;
        if (lc.shared_early && std::getenv("DEEPMOE_SE_CHECK")) {
            std::string first;
            const uint32_t bad = cur_->moe_.debug_check_x(call, &first);
            if (bad) log_warn("SE check (split): layer {} token {}: {} x elements differ; {}", L,
                              cur_->token_, bad, first);
        }
        guard_layer(L, std::span<const uint32_t>(late, n_late), ids);
        if (auto r = cur_->moe_.stage_rows(call, std::span<const uint32_t>(late, n_late)); !r) return r;
    } else {
        uint32_t all[16];
        uint32_t n_all = 0;
        for (uint32_t s = 0; s < topk; ++s)
            if (call.weights[s] != 0.0f) all[n_all++] = s;
        guard_layer(L, std::span<const uint32_t>(all, n_all), ids);
        if (staged) {
            if (auto r = cur_->moe_.stage_rows(call, std::span<const uint32_t>(all, n_all)); !r) return r;
        } else if (lc.shared_early) {
            static const bool se_check = std::getenv("DEEPMOE_SE_CHECK") != nullptr;
            if (se_check) {
                if (auto r = cmd_wait(); !r) return r;
                std::string first;
                const uint32_t bad = cur_->moe_.debug_check_x(call, &first);
                if (bad) log_warn("SE check: layer {} token {}: {} x elements differ; {}", L,
                                  cur_->token_, bad, first);
            }
            if (auto r = cur_->moe_.stage_input(call, true); !r) return r;
            if (auto r = cur_->moe_.stage_rows(call, std::span<const uint32_t>(all, n_all)); !r) return r;
        } else if (auto r = cur_->moe_.stage(call); !r) {
            return r;
        }
    }
    // stage_input resets the bridge's timing and stage_rows adds to it, so this
    // is the whole layer's host half whichever way it ran.
    if (gate_probe_ && cur_->gp_open_) static_cast<GateSeg*>(cur_->gp_open_)->stage_us += ms_since(gp_s0) * 1000.0;
    t.moe_host_ms = cur_->moe_.timing().host_ms;
    cur_->mx_ms_ += cur_->moe_.timing().x_read_ms;
    cur_->mq_ms_ += cur_->moe_.timing().quant_ms;
    cur_->mt_ms_ += cur_->moe_.timing().table_ms;

    if (auto r = cmd_open(); !r) return r;
    {
        const TimePoint r0 = Clock::now();
        cur_->ts_moe_[L].begin = cmd_stamp();
        if (split) {
            const uint32_t tr_a = trace::open_dispatch(&tracer_, uint16_t(L), trace::Cls::Moe, 1,
                                                       "moe_gateup");
            if (auto r = cur_->moe_.record_gateup(cur_->tok_cmd_, std::span<const uint32_t>(late, n_late)); !r)
                return r;
            trace::close_dispatch(&tracer_, tr_a);
            const uint32_t tr_b = trace::open_dispatch(&tracer_, uint16_t(L), trace::Cls::Moe, 2,
                                                       "moe_down");
            if (auto r = cur_->moe_.record_down(cur_->tok_cmd_); !r) return r;
            trace::close_dispatch(&tracer_, tr_b);
        } else if (lc.shared_early) {
            // Track SE: the shared expert's A is done; A over the routed six, then B
            // over all seven -- the Track R1 split, bit-identical to one shot.
            uint32_t routed[16];
            uint32_t n_rt = 0;
            for (uint32_t s2 = 0; s2 < topk; ++s2)
                if (call.weights[s2] != 0.0f) routed[n_rt++] = s2;
            const uint32_t tr_a = trace::open_dispatch(&tracer_, uint16_t(L), trace::Cls::Moe, 1,
                                                       "moe_gateup");
            if (n_rt)
                if (auto r = cur_->moe_.record_gateup(cur_->tok_cmd_, std::span<const uint32_t>(routed, n_rt)); !r)
                    return r;
            trace::close_dispatch(&tracer_, tr_a);
            const uint32_t tr_b = trace::open_dispatch(&tracer_, uint16_t(L), trace::Cls::Moe, 2,
                                                       "moe_down");
            if (auto r = cur_->moe_.record_down(cur_->tok_cmd_); !r) return r;
            trace::close_dispatch(&tracer_, tr_b);
        } else {
            // gate/up, h-quant and down, all recorded inside MoeBridge::record.
            const uint32_t tr_m = trace::open_dispatch(&tracer_, uint16_t(L), trace::Cls::Moe, 3,
                                                       "moe_gateup+hquant+down");
            if (auto r = cur_->moe_.record(cur_->tok_cmd_); !r) return r;
            trace::close_dispatch(&tracer_, tr_m);
        }
        cur_->ts_moe_[L].end = cmd_stamp();
        cur_->rec_ms_ += ms_since(r0);
        if (gate_probe_ && cur_->gp_open_) static_cast<GateSeg*>(cur_->gp_open_)->rec_us += ms_since(r0) * 1000.0;
    }
    // This buffer is the last reader of the layer's slots.
    cur_->open_guard_ = cur_->layer_guard_;
    // Track MS: with another stream waiting for the GPU, the MoE is worth a
    // submit of its own (~0.15 ms) instead of riding along with the NEXT
    // layer's attention -- which would not be recorded until this stream came
    // back round, i.e. after the other stream's whole layer.
    if (streams_.size() > 1 && ms_sched_ != MsSched::PingPong && ms_eager_moe()) {
        if (auto r = cmd_submit(gpu::timeline_value(cur_->token_, L)); !r) return r;
    }
    flush_resident_backfill(L);
    profiler_.note_hot_bytes(layer_hot_bytes_[L]);

    // A probe reads this layer's MoE output on the host, so it cannot wait
    // for the next layer's submit to carry it. It costs a submit a layer and
    // is never on in a real run.
    if (layer_probe) {
        if (auto r = cmd_flush(gpu::timeline_value(cur_->token_, L)); !r) return r;
        layer_probe(L, cur_->layer_);
    }
    apply_post = true;
    return {};
}

// The single-stream composition of the three: exactly the statements that
// used to be one function, in the same order.
Result<void> Engine::run_layer(uint32_t L, uint32_t position, bool& apply_post,
                               LayerTiming& t) {
    Stream& s = *cur_;
    if (auto r = layer_begin(s, L, position, apply_post, t); !r) return r;
    if (auto r = layer_gate(s, L, position); !r) return r;
    return layer_moe(s, L, apply_post);
}

// Track LX: on RADV the eager MoE submit -- a submit that waits on the
// host-signalled residency timeline, followed by a separate submit for the next
// layer -- trips the amdgpu job timeout (2 s on this kernel) in
// suite.multistream's Pipeline/Interleave cases (`ring comp_1.2.0 timeout` ->
// VK_ERROR_DEVICE_LOST), and passes bit-identically with it off; RADV_DEBUG=hang
// (syncshaders) also hides it. Root cause not found (docs/STATUS.md §7 0h), so
// on RADV the default is off; DEEPMOE_MS_EAGER_MOE=1 turns it back on.

// Track SE (docs/STATUS.md §7 0h): the shared expert's act_quant + dispatch A
// submitted right behind the gate, so it runs in the host round trip. RADV
// default ON (hot step ~77.2 -> ~74.3 ms over three alternating pairs, every
// warm pass's margin and l3_ppl's NLL bit-identical); elsewhere off until
// measured. DEEPMOE_SHARED_EARLY=0/1 overrides.
bool Engine::shared_early_on() const {
    static const int env = [] {
        const char* e = std::getenv("DEEPMOE_SHARED_EARLY");
        if (!e || !*e) return -1;
        return *e == '0' ? 0 : 1;
    }();
    const bool on = env >= 0 ? env != 0 : g_shared_early_default.load() != 0;
    // One stream by default: with two, the other stream's submits interleave
    // with this one's, and until 2026-10-01 nothing here had been measured. The
    // per-stream state it needs is per-stream already (`Stream::se_cmd_[2]`,
    // `Stream::moe_` and its two pointer-table pages), so the risk is not a
    // shared buffer but submission order -- shared-early issues an extra submit
    // with no wait behind the gate's, which is the same shape as the eager MoE
    // submit that raced under `--streams 2` on RADV (0h). Opt in with
    // `DEEPMOE_SHARED_EARLY_MS=1` to measure it; see STATUS §3 92.
    static const bool ms = [] {
        const char* e = std::getenv("DEEPMOE_SHARED_EARLY_MS");
        return e && *e && *e != '0';
    }();
    return on && (ms || streams_.size() == 1);
}

bool Engine::ms_eager_moe() {
    static const int env = [] {
        const char* e = std::getenv("DEEPMOE_MS_EAGER_MOE");
        if (!e || !*e) return -1;
        return (*e == '0') ? 0 : 1;
    }();
    if (env >= 0) return env != 0;
    return g_ms_eager_default.load(std::memory_order_relaxed) != 0;
}



// --- M1: the M > 1 forward (docs/p4_dspark_runtime.md §6.4) -----------------
//
// This is the second implementation of a decode step in this file, and that is
// the point: the gate (`spec_forward.batch_matches_m1`) runs the same tokens
// through both and asks for the same logits. What it shares with `run_layer` is
// the routing decision (`route_resident_only`), the planner contract and the
// command-buffer discipline; what it does not share is every kernel, because
// the M > 1 kernels are a separate family (gpu/shaders/mgt1_*.slang).
//
// The union has no MOE_OVERLAP split: starting on the resident half would need
// a second union table. Engram now uses independent per-row planes/bindings
// and one submission; DEEPMOE_BATCH_ENGRAM_EARLY=0 retains the old row fences.
Result<void> Engine::init_batch(uint32_t m_cap) {
    if (!gpu_ready_) return fail(Err::FailedPrecondition, "call init_gpu() first");
    // Track MS: the M > 1 runner, its 128 MB scratch and its tail buffers are
    // the SELECTED stream's, and `batch_ready_` is the engine's. DSpark is one
    // sequence speculating on itself, so the batch lives on stream 0 and says
    // so rather than quietly building a second copy on whatever is selected.
    if (cur_ != streams_.front().get())
        return fail(Err::FailedPrecondition,
                    std::format("the M > 1 forward runs on stream 0; stream {} is selected",
                                cur_->id));
    if (m_cap < 1 || m_cap > gpu::kMgtMaxM)
        return fail(Err::InvalidArgument,
                    std::format("forward_batch takes 1..{} tokens, not {}", gpu::kMgtMaxM, m_cap));
    if (batch_ready_) {
        if (m_cap <= batch_cap_) return {};
        return fail(Err::FailedPrecondition,
                    std::format("the batch scratch was built for M <= {}", batch_cap_));
    }
    const TextConfig& c = model_cfg_.text;
    const std::string dir = gpu::default_shader_dir();
    gpu::MgtSpec mgt_spec;
    if(const char* e=std::getenv("DEEPMOE_MGT_PAIR_DOT"))mgt_spec.pair_dot=*e=='1';
    const char* fold_env=std::getenv("DEEPMOE_MGT_FOLD_SCALE");
    mgt_spec.fold_scale=fold_env && *fold_env!='0';
    const char* cm_env=std::getenv("DEEPMOE_MGT_ATTN_CM");
    mgt_spec.attn_cm=cm_env && *cm_env!='0';
    if(mgt_spec.attn_cm && !device_.caps().cooperative_matrix)
        return fail(Err::Unavailable,"batch ATTN_CM requires cooperative matrix support");
    if(mgt_spec.attn_cm)
        log_warn("engine: experimental batch ATTN_CM; long-sequence regression gate is not passed");
    if (auto r = cur_->mgt_.create(device_, alloc_a_, dir,mgt_spec); !r) return r;
    // The batch activations. 6 columns of everything design §7.14 touches plus
    // the per-query score planes; 128 MB is the round number above what the
    // largest context this store can hold needs (the plane that grows with
    // context is idx_score, M x max_context floats).
    if (auto r = cur_->bscratch_.create(alloc_a_, 128ull << 20); !r) return r;
    const uint32_t maxc = std::max<uint32_t>(max_context(), 1);
    if (auto r = cur_->layer_.create_batch(cur_->mgt_, cur_->bscratch_, m_cap, maxc); !r) return r;

    auto lg = alloc_a_.allocate(uint64_t(m_cap) * c.vocab_size * sizeof(float), true, true);
    if (!lg) return std::unexpected(lg.error());
    cur_->blogits_ = *lg;
    auto sm = alloc_a_.allocate_host_coherent(
        align_up(uint64_t(m_cap) * sizeof(gpu::SampleOut), 4096));
    if (!sm) return std::unexpected(sm.error());
    cur_->bsample_ = *sm;
    std::memset(cur_->bsample_.host_ptr, 0, static_cast<size_t>(cur_->bsample_.bytes));
    auto tail_buffer = [&](gpu::GpuBuffer& dst, uint64_t bytes) -> Result<void> {
        auto b = alloc_a_.allocate_host_coherent(align_up(bytes, 4096));
        if (!b) return std::unexpected(b.error());
        dst = *b;
        return {};
    };
    if (auto r = tail_buffer(cur_->btopk_out_, uint64_t(m_cap) * gpu::kMgtTopKRecordWords * 4); !r) return r;
    if (auto r = tail_buffer(cur_->btopk_hist_, uint64_t(m_cap) * kTopKThreads * kTopKBins * 4); !r) return r;
    if (auto r = tail_buffer(cur_->bdraft_, uint64_t(m_cap) * 4); !r) return r;
    if (auto r = tail_buffer(cur_->brank_, uint64_t(m_cap) * sizeof(gpu::MgtRankOut)); !r) return r;

    // `ced_src` of tests/test_gpu_layer.cpp: list 0 serves every window-only
    // layer, list 1 + rank the rank-th index source and everything that reads it.
    batch_list_.assign(c.num_hidden_layers, 0);
    uint32_t rank = 0;
    for (uint32_t L = 0; L < c.num_hidden_layers; ++L) {
        if (c.is_index_source(L)) ++rank;
        batch_list_[L] = c.compress_ratio(L) ? rank : 0u;
    }
    batch_cap_   = m_cap;
    batch_ready_ = true;
    log_info("engine: M>1 forward ready (M <= {}, {} lists, {} compressed positions)", m_cap,
             cur_->layer_.batch().n_lists, cur_->layer_.batch().score_stride);
    return {};
}

Result<void> Engine::run_layer_batch(uint32_t L, uint32_t p0, uint32_t M, bool& apply_post) {
    const TextConfig& c = model_cfg_.text;
    BatchScratch& bb = cur_->layer_.batch();
    const uint64_t hcstride = uint64_t(c.hc_mult) * c.hidden_size * sizeof(float);
    if(batch_gpu_route_){bb.rope=cur_->saved_rope_[L];bb.rope_lat=cur_->saved_rope_lat_[L];}

    auto view = cur_->kvs_.layer(L);
    if (!view) return std::unexpected(view.error());

    BatchStep st;
    st.layer          = L;
    st.p0             = p0;
    st.m              = M;
    st.compress_ratio = c.compress_ratio(L);
    st.apply_hc_post  = apply_post;
    st.kv             = *view;
    st.list           = batch_list_[L];
    if (!ced_.empty() && st.compress_ratio) {
        const CedPlan& p = ced_[L];
        st.run_compressor = p.is_kv_source;
        st.run_indexer    = p.is_index_source;
        auto cmp = cur_->kvs_.layer(p.cmp_src);
        if (!cmp) return std::unexpected(cmp.error());
        st.kv.cmp_kv = cmp->cmp_kv;
        st.idx_key   = cmp->idx_key;      // the source's own keys, as at M = 1
    }

    // The engram writes into the residual stream BEFORE the block (design §2.1),
    // so the previous layer's deferred hc_post has to be materialised first --
    // the same order `run_layer` uses, one dispatch for all M rows.
    if (cur_->engram_.has_layer(L)) {
        const char* early_env=std::getenv("DEEPMOE_BATCH_ENGRAM_EARLY");
        const bool early=batch_gpu_route_||!early_env || *early_env!='0';
        if (auto r = cmd_open(); !r) return r;
        if (apply_post) {
            BatchStep prev = st;
            prev.layer = L - 1;
            if (auto r = cur_->layer_.record_close_batch(cur_->tok_cmd_, prev); !r) return r;
        }
        for (uint32_t m = 0; m < M; ++m) {
            // Track BF: the row fetch is host I/O; the rest of this loop body is
            // a submit + fence per row. The trace sees both as one gap.
            const TimePoint f0 = Clock::now();
            if(!early) {
                if (auto r = cur_->engram_.fetch(L, cur_->history_, p0 + m); !r) return r;
            } else if(!cur_->engram_.fetched(L,p0+m,m))
                return fail(Err::FailedPrecondition,"batch engram row was not prefetched");
            cur_->mq_ms_ += ms_since(f0);
            if (auto r = cmd_open(); !r) return r;
            const DeviceAddress in  = (apply_post ? bb.xout.addr : bb.x.addr) + m * hcstride;
            const DeviceAddress out = bb.x.addr + m * hcstride;
            const uint32_t tr_eg = trace::open_dispatch(&tracer_, uint16_t(L), trace::Cls::Engram,
                                                        uint16_t(m), "engram_row");
            const TimePoint land0=Clock::now();
            if (auto r = cur_->engram_.record(cur_->tok_cmd_, L, in, out,early?m:0); !r) return r;
            cur_->mq_ms_+=ms_since(land0); // record joins the asynchronous row reads
            trace::close_dispatch(&tracer_, tr_eg);
            // Distinct data and pointer-table bindings keep every row stable
            // through one submission. The arithmetic remains the decode kernel.
            if(!early)if (auto r = cmd_flush(0); !r) return r;
        }
        apply_post       = false;
        st.apply_hc_post = false;
        st.bind_close    = false;
    }

    {
        const TimePoint b0 = Clock::now();
        if (auto r = cur_->layer_.bind_batch(weights_[L], st); !r) return r;
        // `bind_batch` points hc_post's `A` at the batch scratch's own moe_y;
        // the MoE output actually lives in the union runner's y, where the M = 1
        // path's `set_moe_output` would have put it.
        const uint64_t y = cur_->moe_.union_y_address();
        uint64_t* s = cur_->mgt_.slots(gpu::MgtStage::MhcPost);
        s[gpu::slot::kA] = y;
        std::memcpy(cur_->mgt_.slots(gpu::MgtStage::MhcMix), s, gpu::kAttnStageStride);
        std::memcpy(cur_->mgt_.slots(gpu::MgtStage::MhcFinal), s, gpu::kAttnStageStride);
        if (st.bind_close) cur_->mgt_.slots(gpu::MgtStage::MhcClose)[gpu::slot::kA] = y;
        cur_->bind_ms_ += ms_since(b0);
    }
    if (auto r = cmd_open(); !r) return r;
    {
        const TimePoint r0 = Clock::now();
        if (auto r = cur_->layer_.record_attention_batch(cur_->tok_cmd_, st); !r) return r;
        cur_->rec_ms_ += ms_since(r0);
    }
    if(batch_gpu_route_){
        auto& runner=cur_->moe_.gpu_union();
        route_steps_.push_back(st);
        if(dspark_&&L>=37){if(auto r=runner.record_gpu_copy(cur_->tok_cmd_,bb.xout.addr,cur_->saved_hidden_[L-37].addr,M*c.hidden_size,true,c.hidden_size);!r)return r;}
        if(spec_inflight_&&st.run_compressor&&st.compress_ratio>1){
            if(auto r=runner.record_gpu_copy(cur_->tok_cmd_,bb.cmp_y.addr,cur_->saved_carry_k_[L].addr,M*c.head_dim);!r)return r;
            if(auto r=runner.record_gpu_copy(cur_->tok_cmd_,bb.cmp_g.addr,cur_->saved_carry_g_[L].addr,M*c.head_dim);!r)return r;
        }
        const auto trace_id=trace::open_dispatch(&tracer_,uint16_t(L),trace::Cls::Moe,4,"moe_gpu_union");
        if(auto r=runner.record_gpu_route(cur_->tok_cmd_,L,M,c.num_experts_per_tok,bb.gate_ids.addr,bb.gate_weights.addr,bb.u.addr,cur_->saved_routes_[L].addr);!r)return r;
        trace::close_dispatch(&tracer_,trace_id);apply_post=true;return {};
    }
    if (auto r = cmd_flush(0); !r) return r;
    if (auto r = cur_->layer_.verify_after_attention_batch(st); !r) return r;
    if (dspark_ && L>=37) for(uint32_t j=0;j<M;++j)
        capture_draft_hidden(L,p0+j,static_cast<const float*>(bb.xout.host)+size_t(j)*c.hc_mult*c.hidden_size);
    if (spec_inflight_ && st.run_compressor && st.compress_ratio>1) {
        BatchCarry carry;carry.layer=L;carry.kv.resize(size_t(M)*c.head_dim);carry.score.resize(carry.kv.size());
        wc_readback(carry.kv.data(),bb.cmp_y.host,carry.kv.size()*4);
        wc_readback(carry.score.data(),bb.cmp_g.host,carry.score.size()*4);
        batch_carry_.push_back(std::move(carry));
    }
    if (batch_probe) batch_probe(L, cur_->layer_);

    // --- routing, one decision per POSITION ---------------------------------
    const uint32_t topk = c.num_experts_per_tok;
    const auto* ids_all = static_cast<const uint32_t*>(bb.gate_ids.host);
    const auto* wts_all = static_cast<const float*>(bb.gate_weights.host);
    std::array<uint32_t, gpu::kMgtMaxM * 16> bids{};
    std::array<float,    gpu::kMgtMaxM * 16> bwts{};
    std::vector<uint16_t> near_ids;
    std::vector<float>    near_scores;
    near_ids.reserve(size_t(M) * 16);
    near_scores.reserve(size_t(M) * 16);
    cur_->rr_pending_.clear();
    for (uint32_t m = 0; m < M; ++m) {
        const uint32_t* ids_raw = ids_all + size_t(m) * 16;
        const float*    wts_raw = wts_all + size_t(m) * 16;
        for (uint32_t i = 0; i < 16; ++i) {
            near_ids.push_back(static_cast<uint16_t>(ids_raw[i]));
            near_scores.push_back(wts_raw[i]);
        }
        uint32_t eff_ids[16];
        float    eff_w[16];
        uint16_t kept_ids[16];
        float    kept_w[16];
        uint32_t n_kept = topk;
        const uint32_t* ids = ids_raw;
        const float*    wts = wts_raw;
        // The DSpark block's phase inside a verify batch is the ROW, not the LRU
        // clock: row 0 is the last accepted token and routes exactly, rows 1..
        // are the drafts and route resident-only (docs/p4_resident_routing.md §10).
        ResidentOnly ro = resident_only_;
        if (ro == ResidentOnly::Verify) ro = (m == 0) ? verify_first_ : verify_draft_;
        if (ro != ResidentOnly::Off && ro != ResidentOnly::Mask &&
            route_resident_only(L, ro, ids_raw, wts_raw, topk, eff_ids, eff_w, kept_ids, kept_w,
                                n_kept, cur_->rr_pending_)) {
            ids = eff_ids;
            wts = eff_w;
        }
        for (uint32_t i = 0; i < topk; ++i) {
            bids[size_t(m) * topk + i] = ids[i];
            bwts[size_t(m) * topk + i] = wts[i];
        }
    }

    if(spec_diagnostics_ && spec_inflight_)
        for(size_t i=0;i<size_t(M)*topk;++i)
            batch_route_requests_[size_t(L)*M*topk+i]=uint16_t(bids[i]);
    GpuMoeBridge::BatchCall bc;
    bc.layer   = L;
    bc.m       = M;
    bc.ids     = bids.data();
    bc.weights = bwts.data();
    bc.topk    = topk;
    bc.x       = static_cast<const float*>(bb.u.host);
    bc.y       = nullptr;
    bc.hidden  = c.hidden_size;

    // The union is what the batch READS, so it is what the planner is handed:
    // one expert is fetched once for the whole batch, which is the whole point
    // of the union dispatch (docs/p4_dspark_runtime.md §6.1).
    std::vector<uint32_t> un = cur_->moe_.union_experts(bc);
    std::vector<uint16_t> chosen(un.size());
    std::vector<float>    chosen_w(un.size(), 0.0f);
    for (size_t u = 0; u < un.size(); ++u) {
        chosen[u] = static_cast<uint16_t>(un[u]);
        for (uint32_t m = 0; m < M; ++m)
            for (uint32_t i = 0; i < topk; ++i)
                if (bids[size_t(m) * topk + i] == un[u])
                    chosen_w[u] = std::max(chosen_w[u], bwts[size_t(m) * topk + i]);
    }
    cur_->layer_guard_pending_ = false;
    {
        const TimePoint g0 = Clock::now();
        store::RouteDecision route;
        route.layer       = L;
        route.chosen      = std::span<const uint16_t>(chosen);
        route.weights     = std::span<const float>(chosen_w);
        route.near_ids    = std::span<const uint16_t>(near_ids);
        route.near_scores = std::span<const float>(near_scores);
        if (resident_only_ == ResidentOnly::Mask) {
            guard_layer(L, {}, un.data());
            route.guard_hits = cur_->layer_guard_;
            cur_->open_guard_ = cur_->layer_guard_;
        }
        auto plan = planner_.plan_layer(route, clock_);
        if (!plan) return std::unexpected(plan.error());
        cur_->timings_[L].hits       = static_cast<uint32_t>(plan->hits.size());
        cur_->timings_[L].misses     = static_cast<uint32_t>(plan->misses.size());
        cur_->timings_[L].miss_bytes = plan->miss_bytes;
        batch_miss_bytes_ += plan->miss_bytes;
        if (resident_only_ == ResidentOnly::Mask) {
            for (uint32_t m = 0; m < M; ++m) {
                double total = 0.0, kept = 0.0;
                uint32_t served = 0;
                for (uint32_t i = 0; i < topk; ++i) {
                    const size_t at = size_t(m) * topk + i;
                    total += bwts[at];
                    const ExpertKey key{static_cast<uint16_t>(L), static_cast<uint16_t>(bids[at])};
                    if (std::find(plan->hits.begin(), plan->hits.end(), key) == plan->hits.end())
                        bwts[at] = 0.0f;
                    else ++served;
                    kept += bwts[at];
                }
                ++rr_.layers; rr_.requested += topk; rr_.served += served;
                rr_.skipped += topk - served; rr_.shared_only += served == 0;
                if (total > 0) rr_.mass_lost_sum += 1.0 - kept / total;
            }
            for (const auto& [key, stamp] : plan->joined) (void)store_.touch(key, stamp, true);
            un = cur_->moe_.union_experts(bc);
        } else if (auto r = planner_.wait_layer(*plan); !r) return std::unexpected(r.error());
        cur_->timings_[L].gate_ms = ms_since(g0);
        for (uint32_t e : un)
            if (!store_.resident(ExpertKey{static_cast<uint16_t>(L), static_cast<uint16_t>(e)}))
                return fail(Err::Internal,
                            std::format("layer {} expert {} is not resident after the verify "
                                        "batch's gate; store: {}", L, e,
                                        store_.stats().to_string()));
    }
    {
        std::vector<uint32_t> slots(un.size());
        for (uint32_t i = 0; i < slots.size(); ++i) slots[i] = i;
        guard_layer(L, std::span<const uint32_t>(slots), un.data());
    }
    {
        const TimePoint h0 = Clock::now();
        if (auto r = cur_->moe_.stage_batch_union(bc); !r) return r;
        cur_->timings_[L].moe_host_ms = ms_since(h0);
        // Track BF: the two halves of the union's host cost -- `stage` is the
        // read of x out of GPU-visible memory plus act_quant, `table` is the
        // union build, the [M][slots] weight matrix and the pointer rows.
        cur_->mx_ms_ += cur_->moe_.union_info().stage_ms;
        cur_->mt_ms_ += cur_->moe_.union_info().table_ms;
    }
    batch_union_ += cur_->moe_.union_info().routed;
    if (auto r = cmd_open(); !r) return r;
    {
        const TimePoint r0 = Clock::now();
        const uint32_t tr_u = trace::open_dispatch(&tracer_, uint16_t(L), trace::Cls::Moe, 4,
                                                   "moe_union");
        if (auto r = cur_->moe_.record_batch_union(cur_->tok_cmd_); !r) return r;
        trace::close_dispatch(&tracer_, tr_u);
        cur_->rec_ms_ += ms_since(r0);
    }
    // This buffer is the last reader of the layer's slots; the next layer's
    // attention joins it, exactly as at M = 1.
    cur_->open_guard_ = cur_->layer_guard_;
    flush_resident_backfill(L);
    profiler_.note_hot_bytes(layer_hot_bytes_[L]);
    apply_post = true;
    return {};
}

Result<void> Engine::finish_gpu_routes(uint32_t p0,uint32_t M) {
    const auto& c=model_cfg_.text;const auto topk=c.num_experts_per_tok;
    batch_route_host_ms_.fill(0);
    for(const auto& st:route_steps_){const uint32_t L=st.layer;
        auto stamp=Clock::now();
        if(auto r=cur_->layer_.verify_after_attention_batch(st);!r)return r;
        if(spec_diagnostics_)batch_route_host_ms_[0]+=ms_since(stamp);
        stamp=Clock::now();
        if(dspark_&&L>=37)for(uint32_t m=0;m<M;++m){
            if(cur_->draft_hidden_.empty())cur_->draft_hidden_.resize(128*3*c.hidden_size);
            const uint32_t pos=p0+m,slot=pos%128;
            if(cur_->draft_position_[slot]!=pos){cur_->draft_position_[slot]=pos;cur_->draft_mask_[slot]=0;}
            wc_readback(cur_->draft_hidden_.data()+size_t(slot)*3*c.hidden_size+(L-37)*c.hidden_size,
                static_cast<const float*>(cur_->saved_hidden_[L-37].host)+size_t(m)*c.hidden_size,c.hidden_size*4);
            cur_->draft_mask_[slot]|=uint8_t(1u<<(L-37));
        }
        if(spec_inflight_&&st.run_compressor&&st.compress_ratio>1){
            BatchCarry carry;carry.layer=L;carry.kv.resize(size_t(M)*c.head_dim);carry.score.resize(carry.kv.size());
            wc_readback(carry.kv.data(),cur_->saved_carry_k_[L].host,carry.kv.size()*4);
            wc_readback(carry.score.data(),cur_->saved_carry_g_[L].host,carry.score.size()*4);batch_carry_.push_back(std::move(carry));
        }
        if(spec_diagnostics_)batch_route_host_ms_[1]+=ms_since(stamp);
        stamp=Clock::now();
        std::array<uint32_t,6*16*2> saved;wc_readback(saved.data(),cur_->saved_routes_[L].host,sizeof saved);
        if(spec_diagnostics_)batch_route_host_ms_[2]+=ms_since(stamp);
        stamp=Clock::now();
        std::vector<uint16_t> chosen,near_ids;std::vector<float> chosen_w,near_scores;
        std::set<uint32_t> kept_union;uint32_t snapshot_hits=0;
        for(uint32_t m=0;m<M;++m){uint32_t served=0;double total=0,kept=0;
            for(uint32_t i=0;i<16;++i){const auto e=saved[m*16+i];const float w=std::bit_cast<float>(saved[6*16+m*16+i]);
                if(e>=c.n_routed_experts)return fail(Err::Internal,"GPU route emitted invalid expert");
                near_ids.push_back(uint16_t(e));near_scores.push_back(w);if(i>=topk)continue;
                if(spec_diagnostics_&&spec_inflight_)batch_route_requests_[(size_t(L)*M+m)*topk+i]=uint16_t(e);
                total+=w;bool hit=route_snapshot_[(size_t(L)*c.n_routed_experts+e)*6]!=0;
                if(hit){++served;kept+=w;if(w!=0)kept_union.insert(e);}
                auto at=std::find(chosen.begin(),chosen.end(),uint16_t(e));
                if(at==chosen.end()){chosen.push_back(uint16_t(e));chosen_w.push_back(w);if(hit)++snapshot_hits;}
                else chosen_w[size_t(at-chosen.begin())]=std::max(chosen_w[size_t(at-chosen.begin())],w);
            }
            ++rr_.layers;rr_.requested+=topk;rr_.served+=served;rr_.skipped+=topk-served;rr_.shared_only+=served==0;
            if(total>0)rr_.mass_lost_sum+=1-kept/total;
        }
        store::RouteDecision route;route.layer=L;route.chosen=chosen;route.weights=chosen_w;route.near_ids=near_ids;route.near_scores=near_scores;
        if(spec_diagnostics_)batch_route_host_ms_[3]+=ms_since(stamp);
        stamp=Clock::now();
        auto plan=planner_.plan_layer(route,clock_);if(!plan)return std::unexpected(plan.error());
        for(const auto& [key,stamp]:plan->joined)(void)store_.touch(key,stamp,true);
        if(spec_diagnostics_)batch_route_host_ms_[4]+=ms_since(stamp);
        cur_->timings_[L].hits=snapshot_hits;cur_->timings_[L].misses=uint32_t(chosen.size())-snapshot_hits;
        cur_->timings_[L].miss_bytes=plan->miss_bytes;batch_miss_bytes_+=plan->miss_bytes;
        batch_union_+=kept_union.size();profiler_.note_hot_bytes(layer_hot_bytes_[L]);
    }
    return {};
}

Result<void> Engine::forward_batch(uint32_t p0, std::span<const uint32_t> tokens,
                                   std::span<BatchRow> rows, std::span<float> logits) {
    const auto host_begin=Clock::now();
    const auto loads_before=store_.stats();const auto planner_before=planner_.stats();
    batch_host_ms_.clear();batch_route_host_ms_.fill(0);
    if (!gpu_ready_) return fail(Err::FailedPrecondition, "call init_gpu() first");
    if (weights_.empty()) return fail(Err::FailedPrecondition, "no layer weights resolved");
    const TextConfig& c = model_cfg_.text;
    const uint32_t M = static_cast<uint32_t>(tokens.size());
    if (M == 0 || M > gpu::kMgtMaxM)
        return fail(Err::InvalidArgument,
                    std::format("forward_batch takes 1..{} tokens, not {}", gpu::kMgtMaxM, M));
    if (rows.size() < M)
        return fail(Err::InvalidArgument, std::format("{} rows for a batch of {}", rows.size(), M));
    if (!logits.empty() && logits.size() < size_t(M) * c.vocab_size)
        return fail(Err::InvalidArgument,
                    std::format("the logits span holds {} floats; a batch of {} needs {}",
                                logits.size(), M, size_t(M) * c.vocab_size));
    if (!produce_ced_)
        return fail(Err::FailedPrecondition,
                    "forward_batch produces design §7.4's state itself; the LOADED per-step "
                    "seeding has no batch form (set_produce_ced(true))");
    const char* route_env=std::getenv("DEEPMOE_BATCH_GPU_ROUTE");
    const bool gpu_route=route_env&&std::string_view(route_env)=="1";
    if(gpu_route&&streams_.size()>1)return fail(Err::FailedPrecondition,"batch GPU routing requires streams=1");
    if (auto r = init_batch(std::max(M, batch_cap_)); !r) return r;
    if (uint64_t(p0) + M > max_context())
        return fail(Err::ResourceExhausted,
                    std::format("a batch at {}..{} past the {}-position context", p0, p0 + M - 1,
                                max_context()));

    if(gpu_route&&resident_only_!=ResidentOnly::Mask)return fail(Err::FailedPrecondition,"batch GPU routing requires resident-only mask");
    if(gpu_route&&batch_probe)return fail(Err::FailedPrecondition,"batch GPU routing cannot provide live per-layer CPU probes");
    if(!route_config_logged_){log_info("verify batch: GPU snapshot routing {}",gpu_route?"on":"off");route_config_logged_=true;}
    batch_gpu_route_=gpu_route;route_steps_.clear();
    TimelineValue snapshot_guard=0;
    const auto original_rope=cur_->layer_.batch().rope,original_rope_lat=cur_->layer_.batch().rope_lat;
    struct FinishRoute{Engine& e;TimelineValue& guard;gpu::GpuScratch::View rope,rope_lat;~FinishRoute(){
        if(!e.batch_gpu_route_)return;
        if(e.cur_->tok_open_){(void)e.cur_->tok_cmd_.end();e.cur_->tok_open_=false;}
        e.cur_->mgt_.end_immutable();e.cur_->dec_.end_immutable();
        e.cur_->layer_.batch().rope=rope;e.cur_->layer_.batch().rope_lat=rope_lat;
        if(guard){auto fence=e.cur_->fence_.value();if(e.cur_->submits_==0||(fence&&*fence>=e.cur_->fence_value_))e.store_.set_completed_timeline(guard);}
        e.batch_gpu_route_=false;
    }} finish_route{*this,snapshot_guard,original_rope,original_rope_lat};
    if(gpu_route){
        auto& runner=cur_->moe_.gpu_union();
        if(auto r=runner.init_gpu_route(c.num_hidden_layers,gpu::default_shader_dir());!r)return r;
        if(!cur_->route_scratch_.capacity()){
            if(auto r=cur_->route_scratch_.create(alloc_a_,8ull<<20);!r)return r;
            auto take=[&](gpu::GpuScratch::View& v,uint64_t n)->Result<void>{auto r=cur_->route_scratch_.alloc(n);if(!r)return std::unexpected(r.error());v=*r;return {};};
            for(uint32_t l=0;l<c.num_hidden_layers;++l){
                if(auto r=take(cur_->saved_routes_[l],6*16*2*4);!r)return r;
                if(auto r=take(cur_->saved_rope_[l],6*c.qk_rope_head_dim*4);!r)return r;
                if(auto r=take(cur_->saved_rope_lat_[l],6*c.qk_rope_head_dim*4);!r)return r;
                if(auto r=take(cur_->saved_carry_k_[l],6*c.head_dim*4);!r)return r;
                if(auto r=take(cur_->saved_carry_g_[l],6*c.head_dim*4);!r)return r;
            }
            for(auto& v:cur_->saved_hidden_)if(auto r=take(v,6*c.hidden_size*4);!r)return r;
        }
        if(auto r=cur_->mgt_.begin_immutable();!r)return r;
        if(auto r=cur_->dec_.begin_immutable();!r)return r;
        snapshot_guard=++guard_clock_;
        auto snapshot=store_.guarded_snapshot(snapshot_guard);if(!snapshot)return std::unexpected(snapshot.error());
        route_snapshot_=std::move(*snapshot);
        auto shared=cur_->moe_.snapshot_with_shared(route_snapshot_,c.num_hidden_layers);
        if(auto r=runner.upload_snapshot(shared);!r)return r;
    }

    ++batch_forward_calls_;last_batch_layers_=0;
    const TimePoint t_start = Clock::now();
    if(spec_diagnostics_)batch_host_ms_.emplace_back("setup_snapshot",ms_since(host_begin));
    tracer_.token_begin(p0);
    if (cur_->history_.size() < size_t(p0) + M) cur_->history_.resize(size_t(p0) + M, 0);
    for (uint32_t m = 0; m < M; ++m) cur_->history_[p0 + m] = tokens[m];

    cur_->timings_.assign(c.num_hidden_layers, LayerTiming{});
    cur_->submits_ = 0;
    cur_->rec_ms_ = cur_->sub_ms_ = cur_->wait_ms_ = cur_->bind_ms_ = 0.0;
    cur_->mx_ms_ = cur_->mq_ms_ = cur_->mt_ms_ = 0.0;
    cur_->gp_open_ = nullptr;
    batch_union_ = 0;
    batch_miss_bytes_ = 0;
    if(spec_diagnostics_ && spec_inflight_)batch_route_requests_.assign(size_t(c.num_hidden_layers)*M*c.num_experts_per_tok,0);
    cur_->tok_first_ = true;
    if (cur_->tok_open_) { (void)cur_->tok_cmd_.end(); cur_->tok_open_ = false; }
    cur_->layer_.invalidate_candidates();

    // Hashes depend only on the single verification path, already in history.
    // Issue P2 before layer-0 P0 reads rather than joining that queue at L1/L14.
    // Normal main-model LRU/P0 work and its residency snapshot are unchanged.
    if(const char* e=std::getenv("DEEPMOE_BATCH_ENGRAM_EARLY");batch_gpu_route_||!e || *e!='0') {
        const TimePoint f0=Clock::now();
        for(const auto& table:cur_->engram_.tables().layers)
            for(uint32_t m=0;m<M;++m)
                if(auto r=cur_->engram_.fetch(table.layer,cur_->history_,p0+m,m);!r)return r;
        cur_->mq_ms_+=ms_since(f0);
    }

    // Once for the batch, not once a layer: the counts every layer may read are
    // the LAST position's, and the window half of every list is the same shape
    // for all M queries (docs/p4_dspark_runtime.md §6.4 item 3).
    if (auto r = prepare_ced(p0 + M - 1); !r) return r;
    BatchScratch& bb = cur_->layer_.batch();
    write_batch_window_lists(bb, c.sliding_window, p0, M);

    // The M embeddings, in the layout `Engine::embed_token` writes for M = 1.
    {
        std::vector<uint16_t> row(c.hidden_size);
        std::vector<float>    wide(c.hidden_size);
        auto* x = static_cast<float*>(bb.x.host);
        for (uint32_t m = 0; m < M; ++m) {
            const uint32_t tok = tokens[m];
            if (tok >= c.vocab_size)
                return fail(Err::OutOfRange, std::format("token {} >= vocab {}", tok, c.vocab_size));
            std::memcpy(row.data(),
                        static_cast<const std::byte*>(embed_->data_host) +
                            uint64_t(tok) * c.hidden_size * 2,
                        size_t(c.hidden_size) * 2);
            for (uint32_t d = 0; d < c.hidden_size; ++d) wide[d] = bf16_to_f32(row[d]);
            float* dst = x + size_t(m) * c.hc_mult * c.hidden_size;
            for (uint32_t j = 0; j < c.hc_mult; ++j)
                std::memcpy(dst + size_t(j) * c.hidden_size, wide.data(),
                            size_t(c.hidden_size) * sizeof(float));
            auto* mix = static_cast<float*>(bb.mix_a.host) + size_t(m) * 32;
            std::memset(mix, 0, 128);
            mix[0] = 1.0f;
        }
    }

    const auto record_begin=Clock::now();
    if(spec_diagnostics_)batch_host_ms_.emplace_back("input_ced_engram_issue",ms_since(t_start));
    bool apply_post = false;
    for (uint32_t L = 0; L < c.num_hidden_layers; ++L) {
        ++last_batch_layers_;
        if (auto r = run_layer_batch(L, p0, M, apply_post); !r) {
            if (cur_->tok_open_) { (void)cur_->tok_cmd_.end(); cur_->tok_open_ = false; }
            return r;
        }

    }
    // The tail runs the last layer's hc_post, the collapse, the model norm, the
    // head and the per-row argmax. `record_tail_batch` does the close itself, so
    // there is no `record_close_batch` for layer 39.
    BatchStep last;
    last.layer = c.num_hidden_layers - 1;
    last.p0    = p0;
    last.m     = M;
    DecodeLayer::BatchTail bt;
    bt.norm_w = norm_w_;
    bt.head_w = head_w_;
    bt.logits = cur_->blogits_.dev_addr;
    bt.sample = cur_->bsample_.dev_addr;
    const char* readout_env = std::getenv("DEEPMOE_SPEC_GPU_READOUT");
    const bool spec_readout = spec_inflight_ && (!readout_env || *readout_env != '0');
    if (spec_readout) {
        bt.accept_topk = cfg_.speculation.accept_topk;
        if (M > 1) {
            std::memcpy(cur_->bdraft_.host_ptr, tokens.data() + 1, size_t(M - 1) * 4);
            bt.draft_ids = cur_->bdraft_.dev_addr;
            bt.draft_rank = cur_->brank_.dev_addr;
        }
        if (!sampling_.greedy()) {
            bt.topk_out = cur_->btopk_out_.dev_addr;
            bt.topk_hist = cur_->btopk_hist_.dev_addr;
            bt.inv_t = 1.0f / sampling_.temperature;
        }
    }
    if (auto r = cmd_open(); !r) return r;
    if (auto r = cur_->layer_.record_tail_batch(cur_->tok_cmd_, last, bt); !r) return r;
    const auto flush_begin=Clock::now();
    if(spec_diagnostics_)batch_host_ms_.emplace_back("record_layers_tail",ms_since(record_begin));
    if (auto r = cmd_flush(0); !r) return r;
    const auto routes_begin=Clock::now();
    if(spec_diagnostics_){batch_host_ms_.emplace_back("submit_fence",ms_since(flush_begin));batch_host_ms_.emplace_back("queue_submit",cur_->sub_ms_);batch_host_ms_.emplace_back("fence_wait",cur_->wait_ms_);batch_host_ms_.emplace_back("engram_issue_land",cur_->mq_ms_);}
    if(batch_gpu_route_){
        store_.set_completed_timeline(snapshot_guard);snapshot_guard=0;
        if(cur_->submits_!=1)return fail(Err::Internal,"GPU verify must have exactly one queue submission");
        if(auto r=finish_gpu_routes(p0,M);!r)return r;
    }
    if(spec_diagnostics_)batch_host_ms_.emplace_back("finish_routes",ms_since(routes_begin));
    const auto readout_begin=Clock::now();
    flush_trace_batch();
    profiler_.note_hot_bytes(uint64_t(c.vocab_size) * c.hidden_size * 2);

    for (uint32_t m = 0; m < M; ++m) {
        gpu::SampleOut out{};
        std::memcpy(&out, static_cast<const std::byte*>(cur_->bsample_.host_ptr) + size_t(m) * sizeof out,
                    sizeof out);
        if (out.rows != c.vocab_size)
            return fail(Err::Internal,
                        std::format("verify row {} scanned {} rows, not the {}-wide vocabulary", m,
                                    out.rows, c.vocab_size));
        rows[m].argmax = out.token;
        rows[m].top1   = out.top1;
        rows[m].top2   = out.top2;
        if (spec_readout && m + 1 < M)
            std::memcpy(&rows[m].draft_rank,
                        static_cast<const std::byte*>(cur_->brank_.host_ptr) + size_t(m) * sizeof(gpu::MgtRankOut),
                        sizeof(gpu::MgtRankOut));
    }
    if (!logits.empty())
        wc_readback(logits.data(), cur_->blogits_.host_ptr, size_t(M) * c.vocab_size * sizeof(float));
    ++cur_->token_;
    ++clock_;
    if(spec_diagnostics_){
        batch_host_ms_.emplace_back("trace_readout",ms_since(readout_begin));
        static constexpr const char* labels[]={"route_index_validation","route_hidden_carry","route_readback","route_classify","route_planner"};
        for(size_t i=0;i<5;++i)batch_host_ms_.emplace_back(labels[i],batch_route_host_ms_[i]);
        const auto s=store_.stats();const auto ps=planner_.stats();
        batch_load_failures_={ps.p0_reserve_failed-planner_before.p0_reserve_failed,ps.p0_submit_failed-planner_before.p0_submit_failed,ps.p0_io_failed-planner_before.p0_io_failed,s.fills_failed-loads_before.fills_failed};
    }
    (void)t_start;
    return {};
}

Result<void> Engine::snapshot_batch_ring(uint32_t p0, uint32_t m) {
    const TextConfig& c = model_cfg_.text;
    if (m == 0 || m > gpu::kMgtMaxM) return fail(Err::InvalidArgument, "snapshot of 0 positions");
    std::vector<uint32_t> slots(m), layers(c.num_hidden_layers);
    for (uint32_t i = 0; i < m; ++i) slots[i] = (p0 + i) % c.sliding_window;
    for (uint32_t l = 0; l < c.num_hidden_layers; ++l) layers[l] = l;
    auto s = cur_->kvs_.snapshot_ring(slots, layers);
    if (!s) return std::unexpected(s.error());
    batch_snap_      = std::move(*s);
    batch_snap_p0_   = p0;
    batch_snap_m_    = m;
    batch_snap_hist_ = cur_->history_.size();
    return {};
}

Result<void> Engine::restore_batch_ring(uint32_t p0, uint32_t accepted, uint32_t m) {
    if (batch_snap_m_ == 0 || batch_snap_p0_ != p0 || batch_snap_m_ != m)
        return fail(Err::FailedPrecondition,
                    std::format("no ring snapshot for p0 {} m {} (have p0 {} m {})", p0, m,
                                batch_snap_p0_, batch_snap_m_));
    if (accepted + 1 >= m) return {};     // nothing was rejected
    // Positions p0 + accepted + 1 .. p0 + m - 1: the accepted drafts stay, and
    // so does the correction's own slot -- the next cycle's row 0 rewrites it
    // with the corrected token (docs/p3_dspark.md §3.5).
    KvStore::RingSnapshot sub;
    sub.latent_dim = batch_snap_.latent_dim;
    sub.layer      = batch_snap_.layer;
    const uint32_t row  = batch_snap_.latent_dim;
    const uint32_t srow = row / 32;
    std::vector<uint32_t> take;
    for (uint32_t i = accepted + 1; i < m; ++i) take.push_back(i);
    for (uint32_t i : take) sub.slot.push_back(batch_snap_.slot[i]);
    sub.val.resize(size_t(sub.layer.size()) * take.size() * row);
    sub.scale.resize(size_t(sub.layer.size()) * take.size() * srow);
    for (size_t li = 0; li < sub.layer.size(); ++li)
        for (size_t si = 0; si < take.size(); ++si) {
            const size_t src = li * batch_snap_.slot.size() + take[si];
            const size_t dst = li * take.size() + si;
            std::memcpy(sub.val.data() + dst * row, batch_snap_.val.data() + src * row, row);
            std::memcpy(sub.scale.data() + dst * srow, batch_snap_.scale.data() + src * srow, srow);
        }
    if (auto r = cur_->kvs_.restore_ring(sub); !r) return r;
    if (cur_->history_.size() > size_t(p0) + accepted + 2) cur_->history_.resize(size_t(p0) + accepted + 2);
    cur_->layer_.invalidate_candidates();
    return {};
}

Result<DecodeStepResult> Engine::collapse_and_sample(uint32_t position) {
    const TextConfig& c = model_cfg_.text;
    DecodeScratch& b = cur_->layer_.scratch();
    const uint32_t n_wg0 = (c.hidden_size + 255) / 256;

    // `h = layer.hc_pre(h, pre_mix); logits = head(norm(h))`. mega_mhc stage 0
    // with the post bit does the last layer's hc_post AND the collapse, stage 2
    // does the RMSNorm; there is no next sublayer to make mixes for
    // (gpu/shaders/head.slang).
    // The norm weight is the model's own `norm.weight`, not the layer's, so
    // MhcClose's slice is repointed and copied into MhcFinal's -- one slice per
    // stage is why the copy exists at all.
    uint64_t* close = cur_->attn_.slots(gpu::AttnStage::MhcClose);
    close[gpu::slot::kNormW] = norm_w_;
    std::memcpy(cur_->attn_.slots(gpu::AttnStage::MhcFinal), close, gpu::kAttnStageStride);

    uint64_t* hd = cur_->attn_.slots(gpu::AttnStage::Head);
    hd[gpu::slot::kHeadW]      = head_w_;
    hd[gpu::slot::kHeadX]      = b.u.addr;
    hd[gpu::slot::kHeadLogits] = cur_->logits_.dev_addr;

    uint64_t* sm = cur_->dec_.slots(gpu::DecodeStage::Argmax);
    sm[gpu::dslot::kHeadLogits] = cur_->logits_.dev_addr;
    sm[gpu::dslot::kHeadSample] = cur_->sample_.dev_addr;

    gpu::MhcPush mp{c.hidden_size, c.hc_mult, (2 + c.hc_mult) * c.hc_mult, n_wg0,
                    c.hc_sinkhorn_iters,
                    gpu::kMhcFlagPost,
                    static_cast<float>(c.rms_norm_eps), static_cast<float>(c.hc_eps)};
    gpu::HeadPush hp{c.vocab_size, c.hidden_size, 0};

    // The last layer's MoE is already in the open buffer; the collapse, the
    // head and the greedy argmax (design §7.11: four words come back, never the
    // 129,280-wide logit vector) go in behind it.
    const TimePoint t0 = Clock::now();
    if (auto r = cmd_open(); !r) return std::unexpected(r.error());
    cur_->ts_tail_.begin = cmd_stamp();
    auto rec = [&](gpu::AttnStage s, const void* push, uint32_t bytes,
                   uint32_t groups) -> Result<void> {
        const uint32_t tr = trace::open_dispatch(&tracer_, trace::kNoLayer, trace::Cls::Tail,
                                                 uint16_t(s), gpu::attn_stage_name(s));
        if (auto r = cur_->attn_.record(cur_->tok_cmd_, s, push, bytes, groups); !r) return r;
        trace::close_dispatch(&tracer_, tr);
        return cur_->tok_cmd_.barrier();
    };
    auto rec_dec = [&](gpu::DecodeStage s, const void* push, uint32_t bytes,
                       uint32_t groups) -> Result<void> {
        const uint32_t tr = trace::open_dispatch(&tracer_, trace::kNoLayer, trace::Cls::Tail,
                                                 uint16_t(0x100u + uint32_t(s)),
                                                 gpu::decode_stage_name(s));
        if (auto r = cur_->dec_.record(cur_->tok_cmd_, s, push, bytes, groups); !r) return r;
        trace::close_dispatch(&tracer_, tr);
        return {};
    };
    if (auto r = rec(gpu::AttnStage::MhcClose, &mp, sizeof mp, n_wg0); !r)
        return std::unexpected(r.error());
    if (auto r = rec(gpu::AttnStage::MhcFinal, &mp, sizeof mp, n_wg0); !r)
        return std::unexpected(r.error());
    if (auto r = rec(gpu::AttnStage::Head, &hp, sizeof hp,
                     cur_->attn_.gemv_groups(gpu::AttnStage::Head, c.vocab_size)); !r)
        return std::unexpected(r.error());
    if (auto r = rec_dec(gpu::DecodeStage::Argmax, &hp, sizeof hp, 1); !r)
        return std::unexpected(r.error());
    // Track P: a sampled step also reduces the logits to their top set and the
    // tail mass (gpu/shaders/sample_topk.slang), in the same buffer.
    const bool sample = cur_->sample_step_ && !sampling_.greedy();
    if (sample) {
        uint64_t* tk = cur_->dec_.slots(gpu::DecodeStage::SampleTopK);
        tk[gpu::dslot::kTopKLogits] = cur_->logits_.dev_addr;
        tk[gpu::dslot::kTopKOut]    = cur_->topk_out_.dev_addr;
        tk[gpu::dslot::kTopKHist]   = cur_->topk_hist_.dev_addr;
        gpu::TopKPush kp{c.vocab_size, kTopKDefaultK, 1.0f / sampling_.temperature,
                         kTopKBinsPerLogit};
        if (auto r = cur_->tok_cmd_.barrier(); !r) return std::unexpected(r.error());
        if (auto r = rec_dec(gpu::DecodeStage::SampleTopK, &kp, sizeof kp, 1); !r)
            return std::unexpected(r.error());
    }
    cur_->ts_tail_.end = cmd_stamp();
    if (auto r = cmd_flush(gpu::timeline_value(cur_->token_, c.num_hidden_layers - 1)); !r)
        return std::unexpected(r.error());
    (void)t0;
    profiler_.note_hot_bytes(uint64_t(c.vocab_size) * c.hidden_size * 2);

    gpu::SampleOut out{};
    std::memcpy(&out, cur_->sample_.host_ptr, sizeof out);
    if (out.rows != c.vocab_size)
        return fail(Err::Internal,
                    std::format("the sampler scanned {} rows, not the {}-wide vocabulary",
                                out.rows, c.vocab_size));

    DecodeStepResult res;
    res.token    = out.token;
    res.greedy_token = out.token;
    res.position = position;
    res.top1     = out.top1;
    res.top2     = out.top2;
    if (sample)
        if (auto r = sample_into(res, position); !r) return std::unexpected(r.error());
    return res;
}

Result<void> Engine::sample_into(DecodeStepResult& res, uint32_t position,
                               const uint32_t* readout, const float* logits) {
    const TimePoint t0 = Clock::now();
    const TextConfig& c = model_cfg_.text;
    const float T = sampling_.temperature, P = sampling_.top_p;
    const auto* w = readout ? readout : static_cast<const uint32_t*>(cur_->topk_out_.host_ptr);
    const auto* full = logits ? logits : static_cast<const float*>(cur_->logits_.host_ptr);
    if (w[0] != c.vocab_size || w[4] != kTopKCapPerThread || w[6] != kTopKBins)
        return fail(Err::Internal,
                    std::format("sample_topk header: rows {} cap {} bins {}", w[0], w[4], w[6]));
    TopKLogits tk;
    tk.rows      = w[0];
    tk.max_logit = std::bit_cast<float>(w[1]);
    tk.tail      = double(std::bit_cast<float>(w[2]));
    tk.bin       = w[3];
    tk.overflow  = w[5] != 0;
    if (!tk.overflow) {
        for (uint32_t t = 0; t < kTopKThreads; ++t) {
            const uint32_t n = std::min(w[kTopKHeaderWords + t], kTopKCapPerThread);
            const uint32_t* seg = w + kTopKHeaderWords + kTopKThreads + 2 * t * kTopKCapPerThread;
            for (uint32_t j = 0; j < n; ++j)
                tk.cand.push_back({seg[2 * j], std::bit_cast<float>(seg[2 * j + 1])});
        }
    }
    res.candidates = static_cast<uint32_t>(tk.cand.size());
    Nucleus nuc = nucleus_from_topk(tk, T, P);
    std::vector<float> host;   // only when the logits have to cross
    auto copy_logits = [&] {
        if (host.empty()) {
            host.resize(c.vocab_size);
            wc_readback(host.data(), full, host.size() * sizeof(float));
        }
    };
    if (check_topk_) {
        copy_logits();
        const TopKLogits emu = emulate_topk(host, kTopKDefaultK, T);
        auto key = [](const TopKLogits& k) {
            std::vector<uint64_t> v;
            for (const auto& cd : k.cand) v.push_back((uint64_t(cd.id) << 32) | std::bit_cast<uint32_t>(cd.logit));
            std::sort(v.begin(), v.end());
            return v;
        };
        res.topk_checked  = true;
        res.topk_mismatch = emu.max_logit != tk.max_logit || emu.bin != tk.bin ||
                            emu.overflow != tk.overflow ||
                            (!tk.overflow && key(emu) != key(tk)) ||
                            std::fabs(emu.tail - tk.tail) > 1e-4 * std::max(1.0, emu.tail);
        if (res.topk_mismatch) {
            ++topk_mismatches_;
            log_warn("sample_topk: kernel and emulation differ at position {}: M {} vs {}, bin {} vs "
                     "{}, {} vs {} candidates, tail {} vs {}", position, tk.max_logit, emu.max_logit,
                     tk.bin, emu.bin, tk.cand.size(), emu.cand.size(), tk.tail, emu.tail);
        }
    }
    if (!nuc.exact) {
        copy_logits();
        nuc = nucleus_from_full(host, T, P);
        res.topk_fallback = true;
    }
    if (nuc.ids.empty()) return fail(Err::Internal, "the nucleus is empty");
    const uint32_t tok = sample_nucleus(nuc, uniform01(sampling_.seed, uint64_t(position) + 1));
    res.token        = tok;
    res.sampled      = true;
    res.nucleus_size = static_cast<uint32_t>(nuc.ids.size());
    res.retained     = nuc.retained;
    res.kept         = nuc.kept;
    for (size_t i = 0; i < nuc.ids.size(); ++i)
        if (nuc.ids[i] == tok) { res.p_token = nuc.p[i]; break; }
    res.sample_ms = ms_since(t0);
    return {};
}

// --- Track P: conversations ---------------------------------------------------

uint32_t Engine::max_context() const {
    return std::min<uint32_t>(cur_->kvs_.config().max_context, kMaxIndexPositions);
}

// The per-stream half of `begin_session`: the KV store this sequence attends
// and its own engram row planes. The expert cache, the planner's clock and the
// P3 backfill are the process's and are set up once, by `begin_session`.
Result<void> Engine::begin_session_on(uint32_t stream, const SessionConfig& sc) {
    if (!gpu_ready_) return fail(Err::FailedPrecondition, "call init_gpu() first");
    if (stream >= streams_.size())
        return fail(Err::OutOfRange, std::format("stream {} of {}", stream, streams_.size()));
    Stream& s = *streams_[stream];
    const TextConfig& c = model_cfg_.text;
    // Track R2: planes on the kv sources only, allocated for 4,096 positions
    // and grown by the store as the context does.
    const uint32_t limit = std::min<uint32_t>(std::max<uint32_t>(sc.max_context, 256), kMaxIndexPositions);
    KvStoreConfig kc = KvStoreConfig::for_model(c, limit, std::min<uint32_t>(limit, 4096));
    if (auto r = s.kvs_.create(alloc_a_, kc); !r) return r;
    auto tables = sc.engram_tables_dir.empty() ? derive_engram_tables(cfg_.model_dir, c)
                                               : EngramTables::load(sc.engram_tables_dir);
    if (!tables)
        return fail(tables.error().code,
                    std::format("engram tables from '{}': {}",
                                sc.engram_tables_dir.empty() ? cfg_.model_dir + "/tokenizer.json"
                                                             : sc.engram_tables_dir,
                                tables.error().message));
    tables->scales = engram_scales_;
    if (auto r = s.engram_.create(device_, alloc_a_, s.dec_, manifest_, shards_, io_, pinned_, c,
                                  *std::move(tables)); !r)
        return r;
    s.kvs_.clear();
    s.history_.clear();
    s.draft_hidden_.clear();s.draft_position_.fill(-1);s.draft_mask_.fill(0);
    if(dspark_)dspark_->reset();
    s.prefill_loaded_ = false;
    log_info("engine: stream {} -- KV store {} in {} slab(s) (largest {}) for {} positions at "
             "capacity {}", stream, human_bytes(s.kvs_.bytes()), s.kvs_.slabs(),
             human_bytes(s.kvs_.largest_slab()), s.kvs_.config().max_context, s.kvs_.capacity());
    return {};
}

Result<void> Engine::begin_session(const SessionConfig& sc) {
    if (!gpu_ready_) return fail(Err::FailedPrecondition, "call init_gpu() first");
    session_cfg_ = sc;
    for (uint32_t i = 0; i < streams_.size(); ++i)
        if (auto r = begin_session_on(i, sc); !r) return r;
    state_.reset();
    produce_ced_ = true;
    bool backfill = sc.backfill;
    if (const char* e = std::getenv("DEEPMOE_BACKFILL"); e && *e) backfill = *e != '0';
    // One heat order for the whole process: the startup P3 backfill and every
    // later reheat pass rank non-resident experts by the same table, so
    // `DEEPMOE_HEAT_FILE` (tools/hitrate_bench.py --write-heat / --heat-recent)
    // steers both instead of only the first fill.
    if (heat_order_.empty()) {
        if (const char* hf = std::getenv("DEEPMOE_HEAT_FILE"); hf && *hf)
            heat_order_ = store::static_heat_order(hf);
        if (heat_order_.empty()) heat_order_ = store::static_heat_order();
    }
    if (store_.fixed_cache()) {
        auto warm=warm_cache_from_heat();if(!warm)return std::unexpected(warm.error());
        if(!store_.cache_frozen())return fail(Err::FailedPrecondition,"mask initial cache did not fully load; refusing partial fixed cache");
    } else if (backfill && store_.free_slots() > 0) {
        std::vector<ExpertKey> order = heat_order_;
        if (auto r = planner_.start_backfill(std::move(order)); !r)
            log_warn("engine: backfill: {}", r.error().str());
        else
            log_info("engine: P3 backfill started into {} free slots", store_.free_slots());
    }
    log_info("engine: session -- engram tables from {}",
             sc.engram_tables_dir.empty() ? std::string("derived from tokenizer.json")
                                          : sc.engram_tables_dir);
    if(cfg_.speculation.enabled){
        if(auto r=init_batch(6);!r)return r;
        for(uint32_t m=1;m<=cfg_.speculation.max_draft+1;++m)
            if(auto r=cur_->mgt_.ensure(m);!r)return r;
        if(const char* e=std::getenv("DEEPMOE_BATCH_GPU_ROUTE");e && *e=='1')
            if(auto r=cur_->moe_.gpu_union().init_gpu_route(model_cfg_.text.num_hidden_layers,gpu::default_shader_dir());!r)return r;
    }
    return {};
}

Result<void> Engine::set_context_tokens(std::span<const uint32_t> tokens) {
    if (!gpu_ready_) return fail(Err::FailedPrecondition, "call init_gpu() first");
    if (tokens.size() > max_context())
        return fail(Err::ResourceExhausted,
                    std::format("{} tokens against a {}-position context", tokens.size(), max_context()));
    cur_->history_.assign(tokens.begin(), tokens.end());
    return {};
}

void Engine::reset_context() {
    cur_->kvs_.clear();
    cur_->history_.clear();
    cur_->draft_hidden_.clear();
    cur_->draft_position_.fill(-1);cur_->draft_mask_.fill(0);
    if(dspark_)dspark_->reset();
    cur_->prefill_loaded_ = false;
    // `cur_->token_` stays: it is the expert cache's LRU clock (see slow_prefill).
}

Result<DecodeStepResult> Engine::feed(
    std::span<const uint32_t> tokens,
    const std::function<void(uint32_t, const DecodeStepResult&)>& on_step) {
    if (!gpu_ready_) return fail(Err::FailedPrecondition, "call init_gpu() first");
    if (!produce_ced_)
        return fail(Err::FailedPrecondition, "feed needs design 7.4's kernels on (begin_session)");
    if (tokens.empty()) return fail(Err::InvalidArgument, "nothing to feed");
    const uint32_t base = context_length();
    if (uint64_t(base) + tokens.size() > max_context())
        return fail(Err::ResourceExhausted,
                    std::format("{} + {} tokens exceed the session's {}-position context", base,
                                tokens.size(), max_context()));
    DecodeStepResult last{};
    for (uint32_t i = 0; i < tokens.size(); ++i) {
        cur_->sample_step_ = (i + 1 == tokens.size());
        auto r = decode_step(tokens[i], base + i, -1);
        cur_->sample_step_ = false;
        if (!r) return r;
        if (on_step) on_step(i, *r);
        last = *r;
    }
    return last;
}

Result<void> Engine::seed_from_prefill(const gpu::PrefillHandoff& h) {
    if (!gpu_ready_) return fail(Err::FailedPrecondition, "call init_gpu() first");
    const TextConfig& c = model_cfg_.text;
    if (h.layers.size() != c.num_hidden_layers)
        return fail(Err::InvalidArgument,
                    std::format("handoff has {} layers, the model {}", h.layers.size(),
                                c.num_hidden_layers));
    if (h.prompt.size() > max_context())
        return fail(Err::ResourceExhausted,
                    std::format("a {}-token prompt against a {}-position session", h.prompt.size(),
                                max_context()));
    cur_->kvs_.clear();
    for (uint32_t L = 0; L < c.num_hidden_layers; ++L) {
        const gpu::PrefillHandoff::Layer& l = h.layers[L];
        const uint32_t rows = static_cast<uint32_t>(l.win_kv.size() / c.head_dim);
        if (rows)
            if (auto r = cur_->kvs_.seed_window(L, l.win_kv.data(), rows); !r) return r;
        if (!l.cmp_cache.empty() && l.n_cmp) {
            if (auto r = cur_->kvs_.seed_compressed(L, l.cmp_cache.data(), l.n_cmp); !r) return r;
            if (auto r = cur_->kvs_.seed_index_k(L, l.index_k.data(), l.n_cmp); !r) return r;
        }
        if (!l.cmp_state_kv.empty() && l.ratio)
            if (auto r = cur_->kvs_.seed_cmp_state(L, l.cmp_state_kv.data(), l.cmp_state_score.data(),
                                             l.ratio); !r)
                return r;
    }
    cur_->history_.assign(h.prompt.begin(), h.prompt.end());
    produce_ced_ = true;
    cur_->prefill_loaded_ = false;
    return {};
}

Result<DecodeStepResult> Engine::gpu_prefill(std::span<const uint32_t> prompt, uint32_t replay) {
    if (!gpu_ready_) return fail(Err::FailedPrecondition, "call init_gpu() first");
    if (prompt.empty()) return fail(Err::InvalidArgument, "the prompt is empty");
    if (!cur_->engram_.tables().valid())
        return fail(Err::FailedPrecondition, "gpu_prefill needs the engram tables (begin_session)");
    const TextConfig& c = model_cfg_.text;
    const TimePoint t0 = Clock::now();
    gpu::PrefillRunner runner;
    if (auto r = runner.create(device_, alloc_a_, gpu::default_shader_dir()); !r) return std::unexpected(r.error());
    const double runner_ms = ms_since(t0);
    gpu::PrefillConfig pc;
    pc.max_tokens = static_cast<uint32_t>(prompt.size());
    pc.transit_segments = cfg_.prefill_transit_segments;
    if (const char* e = std::getenv("DEEPMOE_PF_LDS"); e && *e == '0') pc.lds_gemm = false;
    if (const char* e = std::getenv("DEEPMOE_PF_READ_AHEAD"); e && *e == '0') pc.read_ahead_min_rows = 0;
    pc.replay     = replay;
    pc.probe_layers = bool(dspark_);
    gpu::Prefill pf;
    {
        // The workspace has to fit what the GPU heaps have left, or the
        // submission loses the device (STATUS §6 17): refuse with the numbers.
        const uint64_t need = pf.workspace_bytes(c, pc);
        uint64_t room = 0;
        for (uint32_t i = 0; i < device_.caps().heaps.size(); ++i) room += heap_headroom(device_, i, nullptr, nullptr);
        if (room && need > room)
            return fail(Err::ResourceExhausted,
                        std::format("GPU prefill of {} tokens needs {} of workspace, {} left on the GPU "
                                    "heaps beside the {}-slot cache: a shorter prompt, or --cache-slots lower",
                                    prompt.size(), human_bytes(need), human_bytes(room), store_.slot_count()));
    }
    if (auto r = pf.create(device_, alloc_a_, runner, manifest_, shards_, io_, pinned_, c,
                           &cur_->engram_.tables(), pc); !r)
        return std::unexpected(r.error());
    const double create_ms = ms_since(t0);
    if(dspark_) pf.probe=[&](const gpu::PrefillProbe& b) {
        if(b.layer<37 || !b.block_in)return;
        const uint32_t first=b.rows>128?b.rows-128:0;
        for(uint32_t j=first;j<b.rows;++j) capture_draft_hidden(b.layer,b.row0+j,b.block_in+size_t(j)*c.hc_mult*c.hidden_size);
    };

    // Track R1 (docs/p4_hitrate.md §3): the experts the prefill streams land in
    // the decode cache under design §9.7.3's rule -- what a global LRU over the
    // prompt's routing table in token order would keep -- and a cached expert
    // is computed from where it is instead of being read again. Stamps are the
    // token-major access order `base + (pos * layers + layer) * topk + rank`.
    const uint64_t n_layers = c.num_hidden_layers, k6 = c.num_experts_per_tok;
    const TokenIndex stamp_base = planner_.reserve_stamps(uint64_t(prompt.size()) * n_layers * k6);
    struct Held { store::StreamAdmit a; TokenIndex stamp = 0; TimelineValue guard = 0; bool open = false; };
    std::vector<Held> held;
    std::set<TimelineValue> open_guards;   // of reservations not yet computed
    gpu::PfExpertSink sink;
    sink.reserve = [&](uint32_t layer, uint32_t expert, uint32_t pos, uint32_t rank) {
        gpu::PfExpertSink::Dest d;
        if (!handoff_) return d;   // DEEPMOE_PREFILL_HANDOFF=0: the transit, as before
        const TokenIndex stamp = stamp_base + (uint64_t(pos) * n_layers + layer) * k6 + rank;
        const TimelineValue guard = ++guard_clock_;
        auto a = planner_.admit_streamed({static_cast<uint16_t>(layer), static_cast<uint16_t>(expert)},
                                         stamp, guard);
        if (!a) {
            log_warn("engine: prefill handoff of ({}, {}): {}", layer, expert, a.error().str());
            return d;
        }
        if (a->kind == store::StreamKind::Drop) return d;
        d.kind = a->kind == store::StreamKind::Fill ? gpu::PfExpertSink::Kind::Fill
                                                    : gpu::PfExpertSink::Kind::Resident;
        d.host = a->addr.host_ptr;
        d.dev  = a->addr.dev_addr;
        d.cookie = held.size();
        held.push_back({*a, stamp, guard, true});
        open_guards.insert(guard);
        return d;
    };
    sink.release = [&](uint32_t, uint32_t, const gpu::PfExpertSink::Dest& d, bool ok) {
        Held& hd = held[d.cookie];
        if (!hd.open) return;
        hd.open = false;
        open_guards.erase(hd.guard);
        if (auto r = planner_.finish_streamed(hd.a, ok, hd.stamp); !r)
            log_warn("engine: prefill handoff settle: {}", r.error().str());
        // Batch i+1's reservations are made before batch i computes, so only
        // the guards below the oldest reservation still waiting may retire.
        store_.set_completed_timeline(open_guards.empty() ? guard_clock_ : *open_guards.begin() - 1);
    };
    sink.cached = [&](uint32_t layer, uint32_t expert) {
        return handoff_ && store_.resident({static_cast<uint16_t>(layer), static_cast<uint16_t>(expert)});
    };
    pf.expert_sink = &sink;
    const store::PlannerStats ps0 = planner_.stats();
    auto h = pf.run(prompt);
    pf.expert_sink = nullptr;
    // A failed prefill leaves reservations open: release them (a Fill whose
    // bytes may be partial is dropped) so no slot stays Filling.
    for (Held& hd : held)
        if (hd.open) {
            (void)planner_.finish_streamed(hd.a, false, hd.stamp);
            hd.open = false;
        }
    store_.set_completed_timeline(guard_clock_);
    if (!h) return std::unexpected(h.error());
    const gpu::PrefillTimes tm = pf.times();
    const TimePoint t_down = Clock::now();
    pf.destroy();
    runner.destroy();
    const double teardown_ms = ms_since(t_down);
    {
        const store::PlannerStats ps1 = planner_.stats();
        log_info("engine: prefill handoff -- {} experts already cached, {} kept, {} dropped; "
                 "cache {} resident of {}", ps1.streamed_resident - ps0.streamed_resident,
                 ps1.streamed_filled - ps0.streamed_filled,
                 ps1.streamed_dropped - ps0.streamed_dropped, store_.stats().resident,
                 store_.slot_count());
    }
    const TimePoint t_seed = Clock::now();
    if (auto r = seed_from_prefill(*h); !r) return std::unexpected(r.error());
    const double seed_ms = ms_since(t_seed);

    DecodeStepResult res;
    res.position     = static_cast<uint32_t>(prompt.size()) - 1;
    res.token        = h->first_token;
    res.greedy_token = h->first_token;
    res.top1 = h->top1;
    res.top2 = h->top2;
    res.wall_ms = ms_since(t0);
    if (!sampling_.greedy() && h->logits.size() == c.vocab_size) {
        const TimePoint s0 = Clock::now();
        const Nucleus nuc = nucleus_from_full(h->logits, sampling_.temperature, sampling_.top_p);
        if (!nuc.ids.empty()) {
            res.token = sample_nucleus(nuc, uniform01(sampling_.seed, uint64_t(res.position) + 1));
            res.sampled = true;
            res.topk_fallback = true;
            res.nucleus_size = static_cast<uint32_t>(nuc.ids.size());
            res.retained = 1.0;
            res.kept = nuc.kept;
            for (size_t i = 0; i < nuc.ids.size(); ++i)
                if (nuc.ids[i] == res.token) { res.p_token = nuc.p[i]; break; }
        }
        res.sample_ms = ms_since(s0);
    }
    // `setup` is the part of the wall clock outside Prefill::run: the runner and
    // buffers, the KV seeding, the teardown -- so the terms add up to the total.
    log_info("engine: GPU prefill of {} tokens in {:.1f} s: embed {:.0f}  engram io {:.0f} / gpu {:.0f}  "
             "mhc {:.0f}  attention {:.0f}  gate {:.0f}  shared {:.0f}  expert io {:.0f} / gpu {:.0f}  "
             "head {:.0f}  other {:.0f}  setup {:.0f} ms (runner {:.0f}, create {:.0f}, teardown {:.0f}, seed {:.0f}); {} experts / {}, {} dispatches, {} submits; "
             "first token {}",
             prompt.size(), res.wall_ms / 1e3, tm.embed, tm.engram_io, tm.engram, tm.mhc, tm.attention,
             tm.gate, tm.shared_expert, tm.expert_io, tm.expert_gpu, tm.head, tm.host,
             res.wall_ms - tm.total, runner_ms, create_ms - runner_ms, teardown_ms, seed_ms, tm.experts_read, human_bytes(tm.expert_bytes), tm.dispatches,
             tm.submits, res.token);
    // DEEPMOE_PF_OPS_JSON=FILE appends the per-op profile prefill_bench --ops-json writes
    if (const char* f = std::getenv("DEEPMOE_PF_OPS_JSON"))
        if (FILE* fp = std::fopen(f, "ab")) {
            std::fputs(tm.json(static_cast<uint32_t>(prompt.size()), "serve", "").c_str(), fp);
            std::fclose(fp);
        }
    return res;
}

// --- Track MS: a decode step, in the two halves a scheduler needs ----------
//
// `step_prologue` is everything before the layer loop and `step_epilogue`
// everything after it, so the single-stream loop and the interleaved one run
// the SAME code around the same forty layers.
Result<void> Engine::step_prologue(Stream& s, uint32_t in_token, uint32_t position,
                                   int32_t state_step, double& prep_ms) {
    cur_ = &s;
    const TextConfig& c = model_cfg_.text;
    if (s.history_.size() <= position) s.history_.resize(position + 1, 0);
    s.history_[position] = in_token;

    const TimePoint t_prep = Clock::now();
    {
        ScopedPhaseIf p(&profiler_, Phase::CpuSync);
        if (produce_ced_) {
            if (auto r = prepare_ced(position); !r) return r;
        } else if (state_step >= 0) {
            if (!state_) return fail(Err::FailedPrecondition, "no decode state is loaded");
            if (auto r = state_->seed_step(s.kvs_, static_cast<uint32_t>(state_step)); !r) return r;
        }
    }
    if (auto r = embed_token(in_token); !r) return r;
    prep_ms = ms_since(t_prep);

    // The token's first buffer resets the timestamp pool; everything after
    // appends to it and it is read once, after the last fence.
    s.timings_.assign(c.num_hidden_layers, LayerTiming{});
    s.ts_attn_.assign(c.num_hidden_layers, Stamp{});
    s.ts_moe_.assign(c.num_hidden_layers, Stamp{});
    s.ts_moe_early_.assign(c.num_hidden_layers, Stamp{});
    s.ts_engram_.assign(c.num_hidden_layers, Stamp{});
    s.engram_host_ms_.assign(c.num_hidden_layers, 0.0);
    s.ts_tail_   = Stamp{};
    s.submits_   = 0;
    s.rec_ms_ = s.sub_ms_ = s.wait_ms_ = s.bind_ms_ = 0.0;
    s.gp_open_ = nullptr;
    s.mx_ms_ = s.mq_ms_ = s.mt_ms_ = 0.0;
    s.tok_first_ = true;
    if (s.tok_open_) {                      // a previous step failed mid-buffer
        (void)s.tok_cmd_.end();
        s.tok_open_ = false;
    }
    // The residency timeline is per token and monotone; a token that starts
    // below where the last one left it would have every wait satisfied before
    // its experts were resident.
    {
        auto cur = s.timeline_.value();
        const TimelineValue base = gpu::timeline_value(s.token_, 0) - 1;
        if (cur && *cur == ~0ull)
            return fail(Err::Internal,
                        "the residency timeline reads UINT64_MAX, which is what a LOST "
                        "device reports -- most likely the expert cache's last path-B "
                        "import exhausted the host heap (see the slab pool's warning)");
        if (cur && *cur > base)
            return fail(Err::Internal,
                        std::format("the residency timeline is at {} but token {} starts "
                                    "at {}", *cur, s.token_, base));
    }
    return {};
}

Result<DecodeStepResult> Engine::step_epilogue(Stream& s, uint32_t position, double prep_ms,
                                               TimePoint t_start) {
    cur_ = &s;
    auto res = collapse_and_sample(position);
    if (!res) return res;
    res->wall_ms = ms_since(t_start);
    read_timestamps(*res);
    StepBreakdown& bd = res->breakdown;
    for (const LayerTiming& t : s.timings_) {
        bd.attn_ms     += t.attn_ms;
        bd.moe_gpu_ms  += t.moe_gpu_ms;
        bd.moe_host_ms += t.moe_host_ms;
        bd.gate_ms     += t.gate_ms;
        bd.engram_ms   += t.engram_ms;
        // Track MS: the step's own cache accounting. The store's counters are
        // the whole process's, so with two streams running a delta of them is
        // both streams' traffic and cannot give a per-stream hit rate.
        bd.requests   += t.hits + t.misses;
        bd.hits       += t.hits;
        bd.miss_bytes += t.miss_bytes;
    }
    bd.submits   = s.submits_;
    bd.record_ms = s.rec_ms_;
    bd.submit_ms = s.sub_ms_;
    bd.wait_ms   = s.wait_ms_;
    bd.bind_ms   = s.bind_ms_ + prep_ms;
    bd.moe_x_ms     = s.mx_ms_;
    bd.moe_quant_ms = s.mq_ms_;
    bd.moe_table_ms = s.mt_ms_;
    bd.other_ms = res->wall_ms - bd.attn_ms - bd.moe_gpu_ms - bd.moe_host_ms - bd.gate_ms -
                  bd.engram_ms - bd.tail_ms;
    // design 13.1's phases, from the same numbers: the GPU halves are only
    // known once the token's timestamps are read, which is now.
    profiler_.add_phase(Phase::HotGemv, Nanos(int64_t((bd.attn_ms + bd.tail_ms) * 1e6)));
    profiler_.add_phase(Phase::ExpertHit,
                        Nanos(int64_t((bd.moe_gpu_ms + bd.moe_host_ms) * 1e6)));
    res->record  = profiler_.token_end();
    if (route_dump_) write_route_record(position);
    ++s.token_;
    ++clock_;
    return res;
}

Result<DecodeStepResult> Engine::decode_step(uint32_t in_token, uint32_t position,
                                             int32_t state_step) {
    if (!gpu_ready_) return fail(Err::FailedPrecondition, "call init_gpu() first");
    if (weights_.empty()) return fail(Err::FailedPrecondition, "no layer weights resolved");
    const TextConfig& c = model_cfg_.text;
    Stream& s = *cur_;

    profiler_.token_begin(position);
    tracer_.token_begin(position);
    const TimePoint t_start = Clock::now();
    double prep_ms = 0.0;
    if (auto r = step_prologue(s, in_token, position, state_step, prep_ms); !r)
        return std::unexpected(r.error());

    bool apply_post = false;   // the very first sublayer has nothing to fold in
    for (uint32_t L = 0; L < c.num_hidden_layers; ++L)
        if (auto r = run_layer(L, position, apply_post, s.timings_[L]); !r) {
            if (s.tok_open_) { (void)s.tok_cmd_.end(); s.tok_open_ = false; }
            return std::unexpected(r.error());
        }
    return step_epilogue(s, position, prep_ms, t_start);
}

// --- Track MS: the interleaved step (docs/p4_multistream.md) ---------------
//
// Layer L of every stream, then layer L + 1. The three phases are ordered so
// that every stream's attention chain is on the GPU before any stream's gate
// blocks, and so that the P0 misses of ALL of them are outstanding at the
// drive together -- which is the second, independent gain: the engine's own
// measurement (docs/p4_p0_queue.md §2) is 3.80 chunks in flight against a
// queue that holds 8.
Result<void> Engine::decode_step_multi(std::span<const MultiStep> steps,
                                       std::span<DecodeStepResult> out) {
    if (!gpu_ready_) return fail(Err::FailedPrecondition, "call init_gpu() first");
    if (weights_.empty()) return fail(Err::FailedPrecondition, "no layer weights resolved");
    if (steps.empty()) return fail(Err::InvalidArgument, "no steps");
    if (out.size() != steps.size())
        return fail(Err::InvalidArgument, "one result per step, please");
    const TextConfig& c = model_cfg_.text;
    const uint32_t n = static_cast<uint32_t>(steps.size());
    for (const MultiStep& m : steps)
        if (m.stream >= streams_.size())
            return fail(Err::OutOfRange, std::format("stream {} of {}", m.stream, streams_.size()));
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = i + 1; j < n; ++j)
            if (steps[i].stream == steps[j].stream)
                return fail(Err::InvalidArgument, "a stream may take one step at a time");

    // The token-level control (design D2): no compute/stall overlap at all,
    // only whatever the drive gets out of two sequences asking at once.
    if (n == 1 || ms_sched_ == MsSched::PingPong) {
        for (uint32_t i = 0; i < n; ++i) {
            if (auto r = select_stream(steps[i].stream); !r) return r;
            auto res = decode_step(steps[i].in_token, steps[i].position, -1);
            if (!res) return std::unexpected(res.error());
            out[i] = *res;
        }
        return {};
    }

    std::vector<Stream*>  st(n);
    std::vector<bool>     apply_post(n, false);
    std::vector<double>   prep_ms(n, 0.0);
    std::vector<TimePoint> t_start(n);
    for (uint32_t i = 0; i < n; ++i) {
        st[i] = streams_[steps[i].stream].get();
        t_start[i] = Clock::now();
        profiler_.token_begin(steps[i].position);
        if (auto r = step_prologue(*st[i], steps[i].in_token, steps[i].position, -1, prep_ms[i]); !r)
            return r;
    }
    auto unwind = [&] {
        for (uint32_t i = 0; i < n; ++i)
            if (st[i]->tok_open_) { (void)st[i]->tok_cmd_.end(); st[i]->tok_open_ = false; }
    };
    auto begin_at = [&](uint32_t i, uint32_t L) -> Result<void> {
        bool ap = apply_post[i];
        auto r = layer_begin(*st[i], L, steps[i].position, ap, st[i]->timings_[L]);
        apply_post[i] = ap;
        return r;
    };
    auto moe_at = [&](uint32_t i, uint32_t L) -> Result<void> {
        bool ap = apply_post[i];
        auto r = layer_moe(*st[i], L, ap);
        apply_post[i] = ap;
        return r;
    };
    if (ms_sched_ == MsSched::Interleave) {
        // The phase-grouped schedule: every stream's attention, then every
        // gate, then every MoE. Both streams' misses are outstanding before
        // either waits -- but by the time the first MoE waits, every submit of
        // the round has already retired, so the GPU idles through the stall.
        // Measured at +9% (docs/p4_multistream.md §5); kept as the arm that
        // shows why the pipeline below is the one that matters.
        for (uint32_t L = 0; L < c.num_hidden_layers; ++L) {
            for (uint32_t i = 0; i < n; ++i)
                if (auto r = begin_at(i, L); !r) { unwind(); return r; }
            for (uint32_t i = 0; i < n; ++i)
                if (auto r = layer_gate(*st[i], L, steps[i].position); !r) { unwind(); return r; }
            for (uint32_t i = 0; i < n; ++i)
                if (auto r = moe_at(i, L); !r) { unwind(); return r; }
        }
        // fall through to the epilogues
    } else {
        // The pipeline. The rule is: NEVER enter a stall with an empty queue.
        // Before stream i blocks on its P0 misses, the next unit of GPU work is
        // already submitted -- the next stream's attention for this layer, or,
        // for the last stream, the first stream's attention for the next layer
        // (which its own MoE, recorded and submitted a moment ago, gates).
        if (auto r = begin_at(0, 0); !r) { unwind(); return r; }
        for (uint32_t L = 0; L < c.num_hidden_layers; ++L) {
            for (uint32_t i = 0; i < n; ++i) {
                if (auto r = layer_gate(*st[i], L, steps[i].position); !r) { unwind(); return r; }
                if (i + 1 < n) {
                    if (auto r = begin_at(i + 1, L); !r) { unwind(); return r; }
                } else if (L + 1 < c.num_hidden_layers) {
                    // Stream 0's MoE for this layer was submitted at i == 0, so
                    // its residency value is signalled and this submit does not
                    // block the queue on a semaphore.
                    if (auto r = begin_at(0, L + 1); !r) { unwind(); return r; }
                }
                if (auto r = moe_at(i, L); !r) { unwind(); return r; }
            }
        }
    }
    for (uint32_t i = 0; i < n; ++i) {
        auto res = step_epilogue(*st[i], steps[i].position, prep_ms[i], t_start[i]);
        if (!res) { unwind(); return std::unexpected(res.error()); }
        out[i] = *res;
    }
    return {};
}

Result<void> Engine::feed_multi(std::span<const MultiStep> steps,
                                std::span<DecodeStepResult> out) {
    if (!produce_ced_)
        return fail(Err::FailedPrecondition, "feed needs design 7.4's kernels on (begin_session)");
    std::vector<MultiStep> at(steps.begin(), steps.end());
    for (MultiStep& m : at) {
        if (m.stream >= streams_.size())
            return fail(Err::OutOfRange, std::format("stream {} of {}", m.stream, streams_.size()));
        Stream& s = *streams_[m.stream];
        m.position = static_cast<uint32_t>(s.history_.size());
        if (uint64_t(m.position) + 1 > s.kvs_.config().max_context)
            return fail(Err::ResourceExhausted,
                        std::format("stream {} is at its {}-position context", m.stream,
                                    s.kvs_.config().max_context));
    }
    return decode_step_multi(at, out);
}

// One record: u32 step (the LRU clock `cur_->token_`), u32 position, u16[layers x
// topk] gate ids in gate order, u8[layers] hits. 528 B at 40 x 6.
void Engine::guard_layer(uint32_t layer, std::span<const uint32_t> slots, const uint32_t* ids) {
    if (!cur_->layer_guard_pending_) {
        cur_->layer_guard_ = ++guard_clock_;
        cur_->layer_guard_pending_ = true;
    }
    for (uint32_t s : slots) {
        const ExpertKey key{static_cast<uint16_t>(layer), static_cast<uint16_t>(ids[s])};
        if (auto slot = store_.slot_of(key)) (void)store_.set_guard(*slot, cur_->layer_guard_);
    }
}

void Engine::write_route_record(uint32_t position) {
    const uint32_t step = static_cast<uint32_t>(cur_->token_);
    std::vector<uint8_t> hits(cur_->timings_.size());
    for (size_t L = 0; L < cur_->timings_.size(); ++L) hits[L] = static_cast<uint8_t>(cur_->timings_[L].hits);
    std::fwrite(&step, sizeof step, 1, route_dump_);
    std::fwrite(&position, sizeof position, 1, route_dump_);
    std::fwrite(cur_->route_ids_.data(), sizeof(uint16_t), cur_->route_ids_.size(), route_dump_);
    std::fwrite(hits.data(), 1, hits.size(), route_dump_);
    std::fflush(route_dump_);
}

Result<SampleResult> Engine::decode_step() {
    if (!state_) return fail(Err::FailedPrecondition, "no decode state is loaded");
    if (cur_->history_.empty()) return fail(Err::FailedPrecondition, "nothing to decode from");
    // A step's `position` is the position of its INPUT token, which is the last
    // one in the history; the token it produces is appended, so repeated calls
    // walk forward. `state_step` is -1 because there is no way to know which of
    // the export's per-step records this position corresponds to -- a caller
    // that needs the LOADED compressed KV uses the three-argument form.
    const uint32_t position = static_cast<uint32_t>(cur_->history_.size()) - 1;
    auto r = decode_step(cur_->history_.back(), position, -1);
    if (!r) return std::unexpected(r.error());
    cur_->history_.push_back(r->token);
    SampleResult s;
    s.token   = r->token;
    s.logprob = r->top1;
    s.margin  = r->margin();
    return s;
}

Result<GenerateResult> Engine::generate(std::span<const uint32_t> prompt,
                                        const GenerateOptions& opts) {
    if (opts.speculative) {
        if(!dspark_)return fail(Err::FailedPrecondition,"speculative generation needs speculation.enabled at engine startup");
        if(!opts.forced.empty())return fail(Err::InvalidArgument,"teacher forcing does not use speculative acceptance");
        std::vector<uint32_t> ids(prompt.begin(),prompt.end());
        if(ids.empty() && state_)ids=state_->prompt_ids();
        if(ids.empty())return fail(Err::InvalidArgument,"speculative generation requires a prompt");
        SessionConfig sc;sc.max_context=std::max<uint32_t>(4096,ids.size()+opts.max_tokens+6);
        if(auto r=begin_session(sc);!r)return std::unexpected(r.error());
        reset_context();
        auto first=gpu_prefill(ids,128);if(!first)return std::unexpected(first.error());
        GenerateResult result;
        if(opts.max_tokens)result.tokens.push_back(first->token);
        while(result.tokens.size()<opts.max_tokens) {
            auto next=speculative_step(result.tokens.back(),opts.max_tokens-result.tokens.size());
            if(!next)return std::unexpected(next.error());
            for(const auto& row:next->rows)result.tokens.push_back(row.token);
        }
        result.summary=profiler_.summary();return result;
    }
    if (!state_)
        return fail(Err::FailedPrecondition,
                    "generate needs the prefill state; call load_decode_state() "
                    "(prefill itself is design 11 / P5)");
    if (!prompt.empty() && prompt.size() != state_->prompt_ids().size())
        return fail(Err::InvalidArgument,
                    std::format("prompt has {} tokens but the loaded state was built "
                                "from {}", prompt.size(), state_->prompt_ids().size()));

    GenerateResult out;
    const uint32_t base = state_->decode_pos();
    // greedy_tokens[0] is the argmax of the PREFILL logits, i.e. the first
    // token and the input to step 0. Prefill is not ours yet, so it is taken
    // from the export.
    uint32_t next = state_->greedy_tokens().front();
    const uint32_t n = std::min<uint32_t>(opts.max_tokens, state_->steps());
    for (uint32_t s = 0; s < n; ++s) {
        const uint32_t in = (s < opts.forced.size()) ? opts.forced[s] : next;
        auto r = decode_step(in, base + s, static_cast<int32_t>(s));
        if (!r) return std::unexpected(r.error());
        out.tokens.push_back(r->token);
        next = r->token;
    }
    out.summary = profiler_.summary();
    return out;
}

Result<double> Engine::measure_submit_overhead(uint32_t iterations) {
    if (!gpu_ready_) return fail(Err::FailedPrecondition, "call init_gpu() first");
    if (iterations == 0) return fail(Err::InvalidArgument, "iterations must be >= 1");
    uint64_t* sm = cur_->dec_.slots(gpu::DecodeStage::Argmax);
    sm[gpu::dslot::kHeadLogits] = cur_->logits_.dev_addr;
    sm[gpu::dslot::kHeadSample] = cur_->sample_.dev_addr;
    gpu::HeadPush hp{model_cfg_.text.vocab_size, model_cfg_.text.hidden_size, 0};
    // One warm-up, so the first submit's pipeline bind does not land in the mean.
    if (auto r = cur_->dec_.dispatch_now(gpu::DecodeStage::Argmax, &hp, sizeof hp, 1); !r)
        return std::unexpected(r.error());
    const TimePoint t0 = Clock::now();
    for (uint32_t i = 0; i < iterations; ++i)
        if (auto r = cur_->dec_.dispatch_now(gpu::DecodeStage::Argmax, &hp, sizeof hp, 1); !r)
            return std::unexpected(r.error());
    return ms_since(t0) / iterations;
}

// Track Y: the dropped experts, at the lowest priority class that still fills
// the cache. `Planner::fetch` reclaims an LRU slot when the cache is full --
// which is the point: without an eviction the resident set would freeze at
// whatever the prefill left and resident-only routing would never refresh.
// Nothing here is waited on; a refusal (nothing evictable, or the class already
// saturated) is counted and dropped.
Result<uint32_t> Engine::warm_cache_from_heat(std::chrono::seconds timeout) {
    if (heat_order_.empty()) {
        if (const char* hf = std::getenv("DEEPMOE_HEAT_FILE"); hf && *hf)
            heat_order_ = store::static_heat_order(hf);
        if (heat_order_.empty()) heat_order_ = store::static_heat_order();
    }
    if (auto s=store_.stats(); s.free==0 && s.filling==0) return s.resident;
    std::vector<ExpertKey> order = heat_order_;
    // QD 8, the depth docs/p4_resident_routing.md's time model assumes, rather
    // than the conversational default of 2: nothing else is running.
    if (auto r = planner_.start_backfill(std::move(order), 8); !r) return std::unexpected(r.error());
    const TimePoint t0 = Clock::now();
    for (;;) {
        const auto s=store_.stats();if(s.resident>=store_.slot_count())break;
        if (Clock::now() - t0 > timeout) break;
        if(s.free>0 && !planner_.backfill_active())break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    planner_.stop_backfill();
    // All initial fills must settle before a fixed-cache session starts.
    return store_.stats().resident;
}

// (B) The bounded background miss queue (docs/p4_resident_routing.md section 8).
//
// Step 2 handed every miss straight to `Planner::fetch`, and the IoEngine's P3
// queue has no bound: 19,356 experts went in, the drive ran flat out at 4.1
// GB/s, and the mean P3 latency came out at 5,594 ms against a 100 ms step --
// so what the drive was reading had been wanted 56 steps earlier, and the cache
// it filled was a cache for a conversation that had already moved on.
//
// The fix is not more bandwidth (there is none) but a shorter queue: keep only
// the misses of the most recent `rr_queue_steps_` steps, drop the rest
// unissued, and drain what is left NEWEST-first with at most
// `rr_outstanding_cap_` experts out at the drive at a time. The drive moves the
// same number of bytes either way; this decides which bytes.
void Engine::flush_resident_backfill(uint32_t layer) {
    if (store_.fixed_cache()) {cur_->rr_pending_.clear();return;}
    if (resident_only_ == ResidentOnly::Off && cur_->rr_pending_.empty() && rr_queue_.empty()) return;
    for (const ExpertKey& key : cur_->rr_pending_) rr_queue_.push_back(RrMiss{key, clock_});
    cur_->rr_pending_.clear();

    const uint64_t cutoff = resident_queue_cutoff(clock_, rr_queue_steps_);
    while (!rr_queue_.empty() && rr_queue_.front().step < cutoff) {
        rr_demand_.erase(rr_pack(rr_queue_.front().key));
        rr_queue_.pop_front();
        ++rr_.bg_stale;
    }
    rr_.bg_depth_sum += rr_queue_.size();
    ++rr_.bg_depth_n;
    if (rr_queue_.size() > rr_.bg_depth_peak)
        rr_.bg_depth_peak = static_cast<uint32_t>(rr_queue_.size());

    while (!rr_queue_.empty() &&
           rr_outstanding_.load(std::memory_order_relaxed) < rr_outstanding_cap_ &&
           io_.inflight_chunks(IoPriority::Backfill) < rr_inflight_cap_) {
        const RrMiss m = rr_queue_.back();          // newest first
        rr_queue_.pop_back();
        // It landed on an earlier request, or is already being filled.
        if (store_.resident(m.key) || store_.slot_of(m.key)) {
            rr_demand_.erase(rr_pack(m.key));
            continue;
        }
        TokenIndex stamp = 0;
        if (auto it = rr_demand_.find(rr_pack(m.key)); it != rr_demand_.end()) stamp = it->second;
        rr_outstanding_.fetch_add(1, std::memory_order_relaxed);
        const TimePoint q0 = Clock::now();
        auto f = planner_.fetch(m.key, IoPriority::Backfill, clock_, layer,
                                [this, q0](bool) {
                                    // IoEngine dispatcher thread: counters only.
                                    rr_outstanding_.fetch_sub(1, std::memory_order_relaxed);
                                    rr_fetch_ns_.fetch_add(
                                        static_cast<uint64_t>((Clock::now() - q0).count()),
                                        std::memory_order_relaxed);
                                    rr_fetch_done_.fetch_add(1, std::memory_order_relaxed);
                                },
                                stamp);
        if (f) {
            ++rr_.bg_enqueued;
            rr_demand_.erase(rr_pack(m.key));
        } else {
            rr_outstanding_.fetch_sub(1, std::memory_order_relaxed);
            ++rr_.bg_refused;
        }
    }
}

std::string Engine::resident_route_report() const {
    if (resident_only_ == ResidentOnly::Off && rr_.layers == 0)
        return std::string("route     resident-only=off\n");
    const uint64_t done = rr_fetch_done_.load(std::memory_order_relaxed);
    const double   lat  = done ? double(rr_fetch_ns_.load(std::memory_order_relaxed)) /
                                 double(done) / 1e6
                               : 0.0;
    return std::format(
        "route     resident-only={} over {} layer-steps\n"
        "  experts requested {}  served {} ({:.4f})  skipped {}\n"
        "  gate mass lost    {:.4f} mean over those layer-steps\n"
        "  shared-expert-only layer-steps {}\n"
        "  background enqueued {}  refused {}  dropped stale {}\n"
        "  background queue depth mean {:.1f} peak {} (window {} steps, cap {} experts)\n"
        "  background fetch latency mean {:.1f} ms over {} completions\n"
        "  stall1 P0 fetches {}  {:.1f} ms total\n",
        resident_only_name(resident_only_),
        rr_.layers, rr_.requested, rr_.served, rr_.served_frac(), rr_.skipped,
        rr_.mass_lost(), rr_.shared_only, rr_.bg_enqueued, rr_.bg_refused, rr_.bg_stale,
        rr_.bg_depth_mean(), rr_.bg_depth_peak, rr_queue_steps_, rr_outstanding_cap_,
        lat, done, rr_.stall1_p0, rr_.stall1_ms);
}

std::span<const float> Engine::last_logits() const {
    if (!cur_->logits_.valid()) return {};
    return {static_cast<const float*>(cur_->logits_.host_ptr), model_cfg_.text.vocab_size};
}

// Track G: the gate host round trip, one line per residency class. The GPU-side
// number this is read against is the per-dispatch trace's "gap in front of the
// MoE dispatch" (docs/plan_p5.md (g)); everything here is the HOST clock, so
// the difference between the two is exactly the two latencies the host cannot
// see -- the fence wake-up and vkQueueSubmit -> GPU start.
std::string Engine::gate_probe_report() const {
    auto row = [](const char* tag, const GateSeg& g) {
        if (!g.n) return std::format("  {:<5} (no layer-steps)\n", tag);
        const double n = double(g.n);
        const double host = (g.fence_us + g.ids_us + g.plan_us + g.pwait_us +
                             g.stage_us + g.rec_us + g.next_us) / n;
        return std::format(
            "  {:<5} n={:<6} fence(i) {:8.1f}  ids(ii) {:5.1f}  plan(iii) {:6.1f}  "
            "pwait(iii) {:8.1f}  stage(iv) {:5.1f}  record(iv) {:5.1f}  "
            "next(v) {:6.1f} (submit {:5.1f})  | after the fence {:7.1f} us\n",
            tag, g.n, g.fence_us / n, g.ids_us / n, g.plan_us / n, g.pwait_us / n,
            g.stage_us / n, g.rec_us / n, g.next_us / n, g.sub_us / n,
            host - g.fence_us / n);
    };
    std::string out = "gate round trip, host clock, us per layer-step "
                      "(DEEPMOE_GATE_PROBE):\n";
    out += row("hit", gp_hit_);
    out += row("miss", gp_miss_);
    if (cur_->spin_hits_ + cur_->spin_misses_)
        out += std::format("  DEEPMOE_FENCE_SPIN_US={:.0f}: the spin caught the signal "
                           "{} of {} times\n", fence_spin_us(), cur_->spin_hits_,
                           cur_->spin_hits_ + cur_->spin_misses_);
    out += "  (i) blocked in cmd_wait; (ii) verify + top-k, already host-coherent; "
           "(iii) planner lookup and miss issue, then the wait on it; "
           "(iv) MoE staging and recording; (v) the next layer's prologue up to the "
           "vkQueueSubmit that carries this layer's MoE. 'after the fence' is (ii)"
           "..(v): the host half of the GPU-side gap the trace calls the gate gap.\n";
    return out;
}

std::string Engine::engram_status() const {
    if (!cur_) return {};
    const auto& eg = cur_->engram_;
    return std::format("{} lands, wait {:.3f} ms, stage {:.3f} ms",
                       eg.land_calls(), eg.wait_ns() / 1e6, eg.stage_ns() / 1e6);
}

std::string Engine::status() const {
    std::string s;
    s += std::format("model     {}\n", ready_ ? model_cfg_.summary() : std::string("(not loaded)"));
    s += std::format("weights   {} shards, {} (direct read, no repack)\n",
                     shards_.size(), human_bytes(shards_.total_bytes()));
    s += std::format("io        {}\n", io_.running() ? io_.backend_caps().name : "stopped");
    s += "          " + io_.stats().to_string();
    s += "engram    " + engram_status() + "\n";
    s += std::format("pinned    {} tensors, {} in {} regions\n", pinned_.tensor_count(),
                     human_bytes(pinned_.bytes_loaded()), pinned_.region_count());
    s += std::format("store     {}\n", store_.stats().to_string());
    s += std::format("planner   {}\n", planner_.stats().to_string());
    s += std::format("gpu       {}\n", device_.valid() ? device_.caps().device_name
                                                       : std::string("(not created)"));
    // What a transcript needs to be able to say for itself: which of the
    // inputs to a decode step this run computed and which it was handed.
    if (!gpu_ready_) {
        s += "LOADED    (no GPU: nothing has run)\n";
    } else if (!produce_ced_) {
        s += std::format("LOADED    the window KV after {} prompt tokens, and the "
                         "compressed KV + indexer top-k of every step, from {} "
                         "(design 7.4's kernels exist; this run is not using them)\n",
                         state_ ? state_->prefill_len() : 0,
                         state_ ? state_->dir() : std::string("(nowhere)"));
    } else if (cur_->prefill_loaded_) {
        s += std::format("LOADED    the state the prompt left behind, from {}: the "
                         "window KV, the compressed KV cache, the indexer key cache "
                         "and the compressor's carried group, after {} prompt tokens. "
                         "Every per-step tensor is ours (design 7.4)\n",
                         state_ ? state_->dir() : std::string("(nowhere)"),
                         state_ ? state_->prefill_len() : 0);
    } else {
        s += "LOADED    none -- the prompt state came from slow_prefill and every "
             "decode-step tensor from design 7.4's kernels\n";
    }
    return s;
}

}  // namespace deepmoe::runtime


namespace deepmoe::runtime {
void Engine::capture_draft_hidden(uint32_t L,uint32_t pos,const float* hc) {
    if(L<37 || L>39)return;
    constexpr uint32_t D=5120;
    if(cur_->draft_hidden_.empty())cur_->draft_hidden_.resize(128*3*D);
    const uint32_t slot=pos%128;
    if(cur_->draft_position_[slot]!=pos){cur_->draft_position_[slot]=pos;cur_->draft_mask_[slot]=0;}
    std::vector<float> host(4*D);wc_readback(host.data(),hc,host.size()*4);
    auto* out=cur_->draft_hidden_.data()+size_t(slot)*3*D+(L-37)*D;
    // mean(hc), rounded as the checkpoint's bf16 tensor.
    for(uint32_t d=0;d<D;++d)out[d]=bf16_to_f32(cpu::float_to_bf16((host[d]+host[D+d]+host[2*D+d]+host[3*D+d])*0.25f));
    cur_->draft_mask_[slot]|=uint8_t(1u<<(L-37));
}
Result<void> Engine::seed_draft(uint32_t pos) {
    if(dspark_->next_position()==pos+1)return {};
    const uint32_t first=pos>=127?pos-127:0;
    std::vector<float> hidden;hidden.reserve(size_t(pos-first+1)*15360);
    for(uint32_t p=first;p<=pos;++p) {
        if(cur_->draft_position_[p%128]!=p || cur_->draft_mask_[p%128]!=7)
            return fail(Err::FailedPrecondition,"DSpark needs main hidden means for the committed 128-token window");
        auto* row=cur_->draft_hidden_.data()+size_t(p%128)*15360;
        hidden.insert(hidden.end(),row,row+15360);
    }
    dspark_->reset();return dspark_->append(first,hidden);
}
Result<Engine::SpecStep> Engine::speculative_step(uint32_t root,uint32_t max_new,
                                                std::span<const uint32_t> stops) {
    if(!dspark_ || !produce_ced_ || current_stream()!=0 || !context_length() || !max_new)
        return fail(Err::FailedPrecondition,"DSpark step requires an initialized stream-0 session and nonempty context");
    const uint32_t p0=context_length(),V=model_cfg_.text.vocab_size;
    if(p0+1>=max_context())return fail(Err::FailedPrecondition,"DSpark context is full");
    uint32_t k=std::min({cfg_.speculation.max_draft,max_new-1,max_context()-p0-1});
    SpecStep out;out.cycle.k=k;
    DsparkRuntime::Output proposal;
    std::vector<uint32_t> initial;
    if(spec_diagnostics_ && !spec_diagnostics_started_) {
        auto slots=store_.evictable();
        std::sort(slots.begin(),slots.end(),[](const auto& a,const auto& b){
            if(a.last_use_token!=b.last_use_token)return a.last_use_token<b.last_use_token;
            return a.slot<b.slot;
        });
        for(const auto& slot:slots)initial.push_back(uint32_t(slot.key.layer)*384+slot.key.expert);
    }
    const TimePoint t0=Clock::now();
    if(auto r=seed_draft(p0-1);!r)return std::unexpected(r.error());
    const double draft_seed_ms=spec_diagnostics_ ? ms_since(t0) : 0.0;
    std::vector<uint32_t> input{root};
    if(k) {
        const char* tail_env=std::getenv("DEEPMOE_DSPARK_TRIM_TAIL");
        const bool trim_tail=!tail_env || *tail_env!='0';
        auto draft=dspark_->draft(p0-1,root,trim_tail?k:5,false);if(!draft)return std::unexpected(draft.error());
        proposal=std::move(*draft);
        if(cfg_.speculation.min_confidence) {
            auto selected=draft_prefix_from_confidence(std::span(proposal.confidence).first(k),*cfg_.speculation.min_confidence);
            if(!selected)return std::unexpected(selected.error());
            k=*selected;out.cycle.k=k;
        }
        input.insert(input.end(),proposal.tokens.begin(),proposal.tokens.begin()+k);
    }
    out.cycle.draft_ms=ms_since(t0);
    if(auto r=init_batch(cfg_.speculation.max_draft+1);!r)return std::unexpected(r.error());
    if(auto r=snapshot_batch_ring(p0,k+1);!r)return std::unexpected(r.error());
    // Verification may wrap onto committed hidden means in the 128-position
    // draft window. Preserve the overwritten slots as well as the target KV:
    // a rejected suffix must not prevent a later continuation from reseeding.
    std::vector<float> hidden_before(size_t(k+1)*15360);
    std::array<int64_t,6> hidden_positions{};
    std::array<uint8_t,6> hidden_masks{};
    for(uint32_t j=0;j<=k;++j) {
        const uint32_t slot=(p0+j)%128;
        std::memcpy(hidden_before.data()+size_t(j)*15360,
                    cur_->draft_hidden_.data()+size_t(slot)*15360,15360*sizeof(float));
        hidden_positions[j]=cur_->draft_position_[slot];hidden_masks[j]=cur_->draft_mask_[slot];
    }
    auto carry=cur_->kvs_.backup_rows(p0,p0);if(!carry)return std::unexpected(carry.error());
    batch_carry_before_=std::move(*carry);batch_carry_.clear();spec_inflight_=true;
    const char* readout_env = std::getenv("DEEPMOE_SPEC_GPU_READOUT");
    const bool gpu_readout = !readout_env || *readout_env != '0';
    out.cycle.gpu_readout = gpu_readout;
    std::vector<BatchRow> rows(k+1);
    std::vector<float> matrix(gpu_readout ? 0 : size_t(k+1)*V);
    const TimePoint tv=Clock::now();
    auto verify=forward_batch(p0,input,rows,matrix);spec_inflight_=false;
    if(!verify) { reset_context();return std::unexpected(verify.error()); }
    out.cycle.verify_ms=ms_since(tv);
    out.cycle.union_experts=batch_union_;out.cycle.miss_bytes=batch_miss_bytes_;
    for(const auto& t:cur_->timings_) {
        out.breakdown.attn_ms+=t.attn_ms;out.breakdown.moe_gpu_ms+=t.moe_gpu_ms;
        out.breakdown.moe_host_ms+=t.moe_host_ms;out.breakdown.gate_ms+=t.gate_ms;
        out.breakdown.engram_ms+=t.engram_ms;
        out.breakdown.requests+=t.hits+t.misses;out.breakdown.hits+=t.hits;
        out.breakdown.miss_bytes+=t.miss_bytes;
    }
    out.breakdown.engram_ms+=cur_->mq_ms_; // measured synchronous batch row fetches
    out.breakdown.record_ms=cur_->rec_ms_;out.breakdown.submit_ms=cur_->sub_ms_;
    out.breakdown.bind_ms=cur_->bind_ms_;out.breakdown.wait_ms=cur_->wait_ms_;
    out.breakdown.submits=cur_->submits_;
    out.cycle.stall_ms=out.breakdown.gate_ms;
    const TimePoint tc=Clock::now();
    Result<uint32_t> a = uint32_t(0);
    if (gpu_readout) {
        uint32_t accepted = 0;
        for (; accepted < k; ++accepted) {
            const auto& rank = rows[accepted].draft_rank;
            if (rank.rows != V || rank.token != input[accepted+1]) {
                a = fail(Err::Internal,"GPU draft rank header does not match verification row"); break;
            }
            if (rank.error) {
                a = fail(Err::InvalidArgument,"invalid draft token or nonfinite verification logit"); break;
            }
            if (rank.better >= cfg_.speculation.accept_topk) break;
        }
        if (a) a = accepted;
    } else a=accept_topk_prefix({input.data()+1,k},matrix,V,cfg_.speculation.accept_topk);
    if(!a){reset_context();return std::unexpected(a.error());}
    for(uint32_t j=0;j<=*a;++j) {
        DecodeStepResult r;r.position=p0+j;r.greedy_token=rows[j].argmax;
        r.top1=rows[j].top1;r.top2=rows[j].top2;r.token=j<*a?input[j+1]:rows[j].argmax;
        if(j==*a && !sampling_.greedy()) {
            if (gpu_readout) {
                std::vector<uint32_t> readout(gpu::kMgtTopKRecordWords);
                wc_readback(readout.data(), static_cast<const uint32_t*>(cur_->btopk_out_.host_ptr) +
                            size_t(j)*gpu::kMgtTopKRecordWords, readout.size()*4);
                const auto* full=static_cast<const float*>(cur_->blogits_.host_ptr)+size_t(j)*V;
                if (auto s=sample_into(r,p0+j,readout.data(),full);!s) {
                    reset_context();return std::unexpected(s.error());
                }
            } else {
                const TimePoint ts=Clock::now();
                auto nuc=nucleus_from_full({matrix.data()+size_t(j)*V,V},sampling_.temperature,sampling_.top_p);
                r.token=sample_nucleus(nuc,uniform01(sampling_.seed,uint64_t(p0)+j+1));
                r.sampled=true;r.topk_fallback=true;r.nucleus_size=nuc.ids.size();
                const auto it=std::find(nuc.ids.begin(),nuc.ids.end(),r.token);
                if(it!=nuc.ids.end())r.p_token=nuc.p[size_t(it-nuc.ids.begin())];
                r.sample_ms=ms_since(ts);
            }
        }
        out.rows.push_back(r);
        if(std::find(stops.begin(),stops.end(),r.token)!=stops.end())break;
    }
    const uint32_t n=uint32_t(out.rows.size());
    // Keep the public selected-token scoring API on the last emitted row,
    // including a stop in an accepted prefix, rather than stale M=1 logits.
    if (gpu_readout) {
        // The selected-token scoring API still needs a full row. Copy only
        // the last emitted row, rather than every speculative suffix row.
        std::vector<float> last(V);
        wc_readback(last.data(),static_cast<const float*>(cur_->blogits_.host_ptr)+size_t(n-1)*V,size_t(V)*4);
        std::memcpy(cur_->logits_.host_ptr,last.data(),size_t(V)*4);
    } else std::memcpy(cur_->logits_.host_ptr,matrix.data()+size_t(n-1)*V,size_t(V)*sizeof(float));
    out.cycle.accepted=std::min(*a,n);out.cycle.emitted=n;out.cycle.cpu_ms=ms_since(tc);
    const TimePoint tr=Clock::now();
    auto abort=[&](const Status& error)->Result<SpecStep>{reset_context();return std::unexpected(error);};
    if(auto r=restore_batch_ring(p0,n-1,k+1);!r)return abort(r.error());
    for(uint32_t j=n;j<=k;++j) {
        const uint32_t slot=(p0+j)%128;
        std::memcpy(cur_->draft_hidden_.data()+size_t(slot)*15360,
                    hidden_before.data()+size_t(j)*15360,15360*sizeof(float));
        cur_->draft_position_[slot]=hidden_positions[j];cur_->draft_mask_[slot]=hidden_masks[j];
    }
    // Restore carry, then overwrite exactly the slots the retained prefix wrote.
    if(auto r=cur_->kvs_.restore_carry(batch_carry_before_);!r)return abort(r.error());
    for(const auto& c:batch_carry_) {
        auto v=cur_->kvs_.layer(c.layer);if(!v)return abort(v.error());
        const uint32_t ratio=model_cfg_.text.compress_ratio(c.layer),D=model_cfg_.text.head_dim;
        for(uint32_t j=0;j<n;++j) {
            std::memcpy(v->cmp_state_kv_host+((p0+j)%ratio)*D,c.kv.data()+size_t(j)*D,D*4);
            std::memcpy(v->cmp_state_score_host+((p0+j)%ratio)*D,c.score.data()+size_t(j)*D,D*4);
        }
    }
    cur_->history_.resize(p0+n);cur_->kvs_.resolve_ring(p0+n);
    if(auto r=prepare_ced(p0+n-1);!r)return abort(r.error());
    std::vector<float> committed;committed.reserve(size_t(n)*15360);
    for(uint32_t p=p0;p<p0+n;++p) {
        auto* h=cur_->draft_hidden_.data()+size_t(p%128)*15360;committed.insert(committed.end(),h,h+15360);
    }
    if(auto r=dspark_->append(p0,committed);!r)return abort(r.error());
    out.cycle.rollback_ms=ms_since(tr);
    const double wall=ms_since(t0);
    // The batch path has no per-kernel GPU timestamps. Keep its unmeasured
    // time in 'other' so the reported components still account for the cycle.
    out.breakdown.other_ms=std::max(0.0,wall-out.breakdown.attn_ms-out.breakdown.moe_gpu_ms-
        out.breakdown.moe_host_ms-out.breakdown.gate_ms-out.breakdown.engram_ms);
    for(auto& r:out.rows)r.wall_ms=wall/n;
    if(spec_diagnostics_) {
        const TimePoint td=Clock::now();
        auto array=[](const auto& values){std::string s="[";for(const auto& x:values){if(s.size()>1)s+=",";s+=std::format("{}",x);}return s+"]";};
        auto write=[&](const std::string& text)->Result<void>{
            if(std::fwrite(text.data(),1,text.size(),spec_diagnostics_)!=text.size())
                return fail(Err::Io,"cannot write speculative diagnostics");
            return {};
        };
        if(!spec_diagnostics_started_) {
            const auto header=std::format("{{\"schema\":1,\"main_layers\":40,\"main_experts\":384,\"main_topk\":6,\"draft_experts\":128,\"initial_main_lru\":{},\"slots\":{},\"draft_pins\":384}}\n",array(initial),store_.slot_count());
            if(auto r=write(header);!r)return abort(r.error());
            spec_diagnostics_started_=true;
        }
        for(uint32_t j=0;j<k;++j)if(!std::isfinite(proposal.confidence[j]))
            return abort(Status{Err::InvalidArgument,"nonfinite draft confidence in diagnostics"});
        std::array<uint32_t,5> ranks; ranks.fill(UINT32_MAX);
        for(uint32_t j=0;j<k;++j) {
            if(gpu_readout)ranks[j]=rows[j].draft_rank.better;
            else {ranks[j]=0;const float chosen=matrix[size_t(j)*V+input[j+1]];
                for(uint32_t id=0;id<V;++id){const float x=matrix[size_t(j)*V+id];ranks[j]+=x>chosen || (x==chosen && id<input[j+1]);}}
        }
        std::string draft_stages="{";
        for(const auto& [name,elapsed]:proposal.timing_ms) {
            if(draft_stages.size()>1)draft_stages+=",";
            draft_stages+=std::format("\"{}\":{}",name,elapsed);
        }
        draft_stages+="}";
        std::string draft_gpu_stages="{";
        for(const auto& [name,elapsed]:proposal.gpu_timing_ms) {
            if(draft_gpu_stages.size()>1)draft_gpu_stages+=",";
            draft_gpu_stages+=std::format("\"{}\":{}",name,elapsed);
        }
        draft_gpu_stages+="}";
        auto line=std::format("{{\"position\":{},\"k\":{},\"accepted\":{},\"emitted\":{},\"tokens\":{},\"confidence\":{},\"target_rank\":{},\"draft_routes_valid\":{},\"draft_routes\":{},\"target_routes\":{},\"draft_ms\":{},\"draft_seed_ms\":{},\"draft_dispatches\":{},\"draft_submits\":{},\"draft_stages_ms\":{},\"draft_gpu_stages_ms\":{},\"verify_ms\":{},\"cpu_ms\":{},\"commit_ms\":{}}}\n",
            p0,k,out.cycle.accepted,n,array(input),array(std::span(proposal.confidence).first(k)),array(std::span(ranks).first(k)),
            proposal.expert_ids_valid,array(proposal.expert_ids),array(batch_route_requests_),out.cycle.draft_ms,
            draft_seed_ms,proposal.gpu_dispatches,proposal.gpu_submissions,draft_stages,draft_gpu_stages,
            out.cycle.verify_ms,out.cycle.cpu_ms,out.cycle.rollback_ms);
        line.resize(line.size()-2);line+=",\"verify_host_ms\":{";
        for(size_t i=0;i<batch_host_ms_.size();++i){if(i)line+=",";line+=std::format("\"{}\":{}",batch_host_ms_[i].first,batch_host_ms_[i].second);}
        line+=std::format("}},\"load_failures\":{{\"p0_reserve\":{},\"p0_submit\":{},\"p0_io\":{},\"fill\":{}}},\"cache_fixed\":{},\"cache_frozen\":{}}}\n",batch_load_failures_[0],batch_load_failures_[1],batch_load_failures_[2],batch_load_failures_[3],store_.fixed_cache(),store_.cache_frozen());
        if(auto r=write(line);!r)return abort(r.error());
        if(std::fflush(spec_diagnostics_))return abort(Status{Err::Io,"cannot flush speculative diagnostics"});
        // Include opt-in diagnostic logging in the public wall time. These
        // instrumented cells must be identified separately in speed reports.
        const double diagnostic_ms=ms_since(td);
        out.breakdown.other_ms+=diagnostic_ms;
        for(auto& r:out.rows)r.wall_ms=(wall+diagnostic_ms)/n;
    }
    return out;
}
}
