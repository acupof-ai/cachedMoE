// Vulkan plumbing that does not need the 510 GB checkpoint: the two memory
// paths of design §3.3 and the timeline-gated submission of design §7.1.
//
// tests/test_gpu_moe.cpp covers the kernels against the oracle and needs the
// weights; this file covers the mechanisms underneath them and needs only a
// GPU. Both skip themselves cleanly on a machine without one.
#include <cstdio>
#include <cstring>
#include <format>
#include <string>
#include <thread>
#include <vector>

#include "core/align.h"
#include "core/config.h"
#include "gpu/vulkan/cmdbuf.h"
#include "gpu/vulkan/descriptor.h"
#include "gpu/vulkan/device.h"
#include "gpu/vulkan/memory.h"
#include "gpu/vulkan/moe_kernels.h"
#include "gpu/vulkan/pipeline.h"
#include "gpu/vulkan/rawread.h"
#include "gpu/vulkan/timeline.h"
#include "store/expert_store.h"
#include "tests/test_framework.h"

using namespace deepmoe;

namespace {

// A skip is a pass; ctest keys its SKIP_REGULAR_EXPRESSION off "SKIP gpu:".
bool skip_without_gpu(gpu::Device& dev, const char* what) {
    if (auto r = dev.create(); !r) {
        std::printf("       SKIP gpu: %s needs a Vulkan device (%s)\n", what, r.error().str().c_str());
        return true;
    }
    if (auto r = dev.caps().check_required(); !r) {
        std::printf("       SKIP gpu: %s: %s\n", what, r.error().str().c_str());
        return true;
    }
    return false;
}

}  // namespace

// design §3.3: both paths must produce the (host_ptr, device_address) pair the
// ExpertStore expects, and both must survive a CPU write followed by a GPU read.
DEEPMOE_TEST(gpu, both_memory_paths_give_a_host_pointer_and_a_device_address) {
    gpu::Device dev;
    if (skip_without_gpu(dev, "memory paths")) return;

    const uint64_t bytes = 16u << 20;
    for (MemoryPath path : {MemoryPath::DeviceLocalHostVisible, MemoryPath::ExternalMemoryHost}) {
        gpu::MemoryAllocator alloc;
        auto init = alloc.init(dev, path);
        if (!init) {
            std::printf("       path %d unavailable: %s\n", static_cast<int>(path),
                        init.error().str().c_str());
            continue;
        }
        auto buf = alloc.allocate_slab(bytes);
        REQUIRE_OK(buf);
        CHECK(buf->host_ptr != nullptr);
        CHECK(buf->dev_addr != kNoDeviceAddress);
        CHECK_EQ(buf->bytes, bytes);
        // The IoEngine writes into this with FILE_FLAG_NO_BUFFERING, which
        // requires a sector-aligned destination (design §9.6).
        CHECK(is_aligned(buf->host_ptr));
        std::memset(buf->host_ptr, 0xA5, static_cast<size_t>(bytes));
        std::printf("       path %s: %.0f MB at host %p, device %#llx%s\n",
                    path == MemoryPath::DeviceLocalHostVisible ? "A" : "B",
                    bytes / 1e6, buf->host_ptr,
                    static_cast<unsigned long long>(buf->dev_addr),
                    buf->imported ? " (imported)" : "");
        alloc.free(*buf);
    }
}

// design §5.3: a slab may not exceed maxMemoryAllocationSize, and an import
// must be rejected rather than silently misaligned.
DEEPMOE_TEST(gpu, the_allocator_refuses_what_the_driver_cannot_do) {
    gpu::Device dev;
    if (skip_without_gpu(dev, "allocator limits")) return;

    gpu::MemoryAllocator alloc;
    REQUIRE_OK(alloc.init(dev, MemoryPath::DeviceLocalHostVisible));

    const uint64_t cap = dev.caps().max_memory_allocation_size;
    REQUIRE(cap > 0);
    auto too_big = alloc.allocate_slab(cap + kPageSize);
    CHECK(!too_big.has_value());
    if (!too_big) CHECK_EQ(static_cast<int>(too_big.error().code), static_cast<int>(Err::InvalidArgument));
    // design §5.3's 100 slots of 18,808,832 B must fit under the cap.
    CHECK(uint64_t(100) * layout::kExpertSlotBytes <= cap);

    if (dev.caps().external_memory_host) {
        gpu::MemoryAllocator b;
        REQUIRE_OK(b.init(dev, MemoryPath::ExternalMemoryHost));
        auto host = gpu::alloc_host_pages(1u << 20, /*try_large_pages=*/false);
        REQUIRE_OK(host);
        // One byte past the page boundary is not importable.
        auto bad = b.import_host_memory(static_cast<char*>(host->ptr) + 1, host->bytes - kPageSize);
        CHECK(!bad.has_value());
        auto good = b.import_host_memory(host->ptr, host->bytes);
        REQUIRE_OK(good);
        CHECK_EQ(good->host_ptr, host->ptr);
        CHECK(good->dev_addr != kNoDeviceAddress);
        b.free(*good);
        gpu::free_host_pages(*host);
    }
}

// design §5.3 / architecture.md §1.3: the interface inversion. The ExpertStore
// takes a SlabBacking from gpu/ and never learns which path it got.
DEEPMOE_TEST(gpu, the_expert_store_runs_on_a_vulkan_slab_backing) {
    gpu::Device dev;
    if (skip_without_gpu(dev, "vulkan slab backing")) return;

    gpu::MemoryAllocator alloc;
    REQUIRE_OK(alloc.init(dev, MemoryPath::DeviceLocalHostVisible));
    auto backing = alloc.make_slab_backing();
    REQUIRE_OK(backing);

    CacheConfig cache;
    cache.slots_per_slab = 4;
    cache.budget_bytes   = 4 * layout::kExpertSlotBytes;
    store::ExpertStore st;
    REQUIRE_OK(st.init(std::move(*backing), cache));
    REQUIRE_EQ(st.slot_count(), 4u);

    // A synthetic fill: the pointer table must come back holding device
    // addresses, not host pointers, once the backing is the Vulkan one.
    const ExpertKey key{3, 17};
    auto res = st.begin_fill(key);
    REQUIRE_OK(res);
    REQUIRE_OK(st.finish_fill(res->slot, /*ok=*/true, /*token=*/1));
    REQUIRE(st.resident(key));
    auto addr = st.lookup(key, 1);
    REQUIRE(addr.has_value());
    CHECK(addr->host_ptr != nullptr);
    CHECK(addr->dev_addr != kNoDeviceAddress);
    auto entry = st.table_entry(key, ExpertPart::W1Weight);
    REQUIRE_OK(entry);
    CHECK_EQ(*entry, addr->dev_addr);
    // ...and the six parts are the six offsets the manifest would have given.
    auto w2 = st.table_entry(key, ExpertPart::W2Weight);
    REQUIRE_OK(w2);
    CHECK(*w2 > *entry);
}

// design §7.1 / §7.8: the GPU must not run a MoE dispatch until the CPU has
// host-signalled "the experts for this layer are resident". This is that
// mechanism with nothing else attached.
DEEPMOE_TEST(gpu, a_timeline_wait_gates_a_submission_until_the_host_signals) {
    gpu::Device dev;
    if (skip_without_gpu(dev, "timeline gating")) return;

    gpu::MemoryAllocator alloc;
    REQUIRE_OK(alloc.init(dev, MemoryPath::DeviceLocalHostVisible));

    // Any real dispatch will do; rawread is the smallest one that exists.
    gpu::Pipeline pipe;
    gpu::PipelineLayoutSpec lspec;
    lspec.storage_buffers = 2;
    lspec.push_constant_size = 16;
    auto create = pipe.create(dev, gpu::default_shader_dir() + "/rawread.spv", lspec, {});
    if (!create) {
        std::printf("       SKIP gpu: timeline gating needs build/shaders/rawread.spv (%s)\n",
                    create.error().str().c_str());
        return;
    }
    auto src = alloc.allocate_slab(1u << 20);
    REQUIRE_OK(src);
    auto sink = alloc.allocate(256 * 16, /*host_visible=*/false, /*device_address=*/false);
    REQUIRE_OK(sink);

    gpu::DescriptorPool pool;
    REQUIRE_OK(pool.create(dev, 1, 2));
    std::vector<gpu::BufferBinding> binds(2);
    binds[0].binding = 0; binds[0].buffer = src->buffer;  binds[0].range = 1u << 20;
    binds[1].binding = 1; binds[1].buffer = sink->buffer;
    auto set = pool.allocate(pipe, binds);
    REQUIRE_OK(set);

    gpu::CommandPool cmds;
    REQUIRE_OK(cmds.create(dev));
    auto cb = cmds.acquire();
    REQUIRE_OK(cb);
    struct Push { uint32_t per_group, r0, r1, r2; } push{1024, 0, 0, 0};
    REQUIRE_OK(cb->begin());
    REQUIRE_OK(cb->bind(pipe, *set));
    REQUIRE_OK(cb->push(pipe, &push, sizeof(push)));
    REQUIRE_OK(cb->dispatch(1));
    REQUIRE_OK(cb->end());

    gpu::Timeline tl;
    REQUIRE_OK(tl.create(dev, /*initial=*/0));

    // Layer L of token T waits on timeline_value(T, L) and the token's end is
    // timeline_token_end(T); here: wait for 5, signal 6.
    const TimelineValue gate = 5, done = 6;
    const TimelineValue waits[] = {gate};
    gpu::Submission s;
    s.cmd = &*cb;
    s.timeline = &tl;
    s.wait_values = waits;
    s.signal_value = done;
    REQUIRE_OK(gpu::submit(dev, s));

    // The work is submitted but must not have run: nothing signalled `gate`.
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    auto v = tl.value();
    REQUIRE_OK(v);
    CHECK_EQ(*v, 0ull);
    CHECK(tl.wait(done, std::chrono::milliseconds(20)).has_value() == false);

    // The planner's host signal releases it.
    REQUIRE_OK(tl.signal(gate));
    REQUIRE_OK(tl.wait(done, std::chrono::seconds(5)));
    auto after = tl.value();
    REQUIRE_OK(after);
    CHECK_EQ(*after, done);
    std::printf("       timeline gated a submission at %llu and released it to %llu\n",
                static_cast<unsigned long long>(gate), static_cast<unsigned long long>(*after));

    // Signalling backwards is an error, not a silent no-op (design §5.3 uses
    // this counter as the eviction guard).
    CHECK(!tl.signal(gate).has_value());

    cmds.destroy();
    pool.destroy();
    pipe.destroy();
    alloc.free(*sink);
    alloc.free(*src);
}

// design §7.1 rule 2: the raw-read shader is the ceiling every GEMV is scored
// against, so it has to work on a machine that has never seen the checkpoint.
DEEPMOE_TEST(gpu, the_raw_read_ceiling_shader_runs) {
    gpu::Device dev;
    if (skip_without_gpu(dev, "raw read")) return;

    gpu::MemoryAllocator alloc;
    REQUIRE_OK(alloc.init(dev, MemoryPath::DeviceLocalHostVisible));
    gpu::RawReadKernel raw;
    auto create = raw.create(dev, alloc, gpu::default_shader_dir(), /*groups=*/64);
    if (!create) {
        std::printf("       SKIP gpu: raw read needs build/shaders/rawread.spv (%s)\n",
                    create.error().str().c_str());
        return;
    }
    auto buf = alloc.allocate_slab(64u << 20);
    REQUIRE_OK(buf);
    std::memset(buf->host_ptr, 0x5A, static_cast<size_t>(buf->bytes));
    const uint64_t want = raw.round_bytes(buf->bytes);
    REQUIRE(want > 0);
    (void)raw.run(*buf, want, 1);
    auto r = raw.run(*buf, want, 4);
    REQUIRE_OK(r);
    CHECK(r->gbps > 1.0);
    std::printf("       raw read %.0f MB x %u: %.1f GB/s (%s)\n", want / 1e6, r->passes,
                r->gbps, r->gpu_timed ? "gpu timestamps" : "wall clock");
    alloc.free(*buf);
}
