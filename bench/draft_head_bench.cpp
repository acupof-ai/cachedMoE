// Measure the existing native head with captured activations. One warmup and
// one timestamped batch; no per-dispatch profiling or repeated whole runs.
#include <chrono>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu/vulkan/draft_head.h"
#include "core/wc_read.h"

#include "gpu/vulkan/cmdbuf.h"
#include "gpu/vulkan/decode_kernels.h"
#include "gpu/vulkan/moe_kernels.h"
#include "model/layout.h"
#include "model/v41_config.h"
#include "storage/backend.h"
#include "store/pinned.h"

using namespace cachedmoe;
namespace fs = std::filesystem;
namespace {
template<class T> T require(Result<T> value) {
    if (!value) throw std::runtime_error(value.error().str());
    return std::move(*value);
}
void require(Result<void> value) {
    if (!value) throw std::runtime_error(value.error().str());
}
}

int main(int argc, char **argv) try {
    std::string model, mirror, hidden_path, out, format = "native_bf16";
    uint32_t rows = 2, iterations = 8;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc) throw std::runtime_error("each option requires a value");
        const std::string key = argv[i], value = argv[i + 1];
        if (key == "--model") model = value;
        else if (key == "--mirror") mirror = value;
        else if (key == "--hidden") hidden_path = value;
        else if (key == "--out") out = value;
        else if (key == "--format") format = value;
        else if (key == "--rows") rows = std::stoul(value);
        else if (key == "--iters") iterations = std::stoul(value);
        else throw std::runtime_error("unknown option: " + key);
    }
    if (model.empty() || mirror.empty() || hidden_path.empty() || out.empty() || !iterations || !rows || rows > 5)
        throw std::runtime_error("require --model --mirror --hidden --out; rows 1..5, positive iterations");
    if (format != "native_bf16" && format != "row_fp8")
        throw std::runtime_error("format must be native_bf16 or row_fp8");
    if (fs::exists(out)) throw std::runtime_error("output must be new");
    for (const auto &root : {model, mirror}) {
        const auto relative = fs::weakly_canonical(out).lexically_relative(fs::canonical(root));
        if (relative.empty() || *relative.begin() != "..")
            throw std::runtime_error("output must be outside checkpoint roots");
    }
    const auto config = require(V41Config::load(model + "/config.json"));
    const uint32_t dim = config.text.hidden_size, vocab = config.text.vocab_size;
    std::vector<float> hidden(size_t(rows) * dim);
    std::ifstream source(hidden_path, std::ios::binary);
    if (!source.read(reinterpret_cast<char *>(hidden.data()), hidden.size() * sizeof(float)))
        throw std::runtime_error("captured hidden block is incomplete");

    gpu::Device device;
    require(device.create({}));
    require(device.caps().check_required());
    gpu::MemoryAllocator allocator;
    require(allocator.init(device, MemoryPath::DeviceLocalHostVisible));
    const auto manifest = require(Manifest::load(model + "/" + layout::kManifestFile));
    store::ShardSet shards;
    require(shards.open_all(model, manifest, true));
    IoConfig io_config;
    storage::IoEngine io;
    require(io.start(require(storage::make_default_backend(io_config)), io_config));
    store::PinnedStore pinned;
    store::PinnedConfig pinned_config;
    pinned_config.region_bytes = 2ull << 30;
    require(pinned.init(require(allocator.make_slab_backing()), pinned_config));
    require(pinned.load(manifest, shards, io, {"head.weight"}));
    io.stop(); // No storage work is included in the GPU measurement.
    const auto *head = pinned.find("head.weight");
    if (!head || head->dtype != QuantType::Bf16 ||
        head->shape != std::vector<uint64_t>{vocab, dim} || head->data_bytes != uint64_t(vocab) * dim * 2)
        throw std::runtime_error("unexpected native BF16 head geometry");
    gpu::DraftHeadFp8 copy;
    uint64_t copy_bytes = 0;
    if (format == "row_fp8") {
        require(copy.create(allocator, *head, vocab, dim));
        copy_bytes = gpu::DraftHeadFp8::copy_bytes(vocab, dim);
        // Keep encoded hashes auditable without adding derived checkpoint files.
        for (const auto &[suffix, buffer] :
             {std::pair{".encoded.u8", &copy.weights()}, std::pair{".scales.f32", &copy.scales()}}) {
            if (fs::exists(out + suffix)) throw std::runtime_error("derived output must be new");
            std::vector<std::byte> cached(buffer->bytes);
            wc_readback(cached.data(), buffer->host_ptr, buffer->bytes);
            std::ofstream file(out + suffix, std::ios::binary);
            file.exceptions(std::ios::failbit | std::ios::badbit);
            file.write(reinterpret_cast<const char *>(cached.data()), cached.size());
        }
    }
    auto input = require(allocator.allocate(hidden.size() * sizeof(float), true, true));
    auto logits = require(allocator.allocate(uint64_t(rows) * vocab * sizeof(float), true, true));
    std::memcpy(input.host_ptr, hidden.data(), hidden.size() * sizeof(float));
    gpu::MgtRunner runner;
    gpu::MgtSpec spec;
    spec.draft_head_fp8 = format == "row_fp8";
    require(runner.create(device, allocator, gpu::default_shader_dir(), spec));
    require(runner.ensure(rows));
    const auto stage = spec.draft_head_fp8 ? gpu::MgtStage::DraftHeadFp8 : gpu::MgtStage::Head;
    auto *slots = runner.slots(stage);
    slots[gpu::mslot::kHW] = spec.draft_head_fp8 ? copy.weights().dev_addr : head->data;
    if (spec.draft_head_fp8) slots[gpu::mslot::kHScale] = copy.scales().dev_addr;
    slots[gpu::mslot::kHX] = input.dev_addr;
    slots[gpu::mslot::kHLogits] = logits.dev_addr;
    gpu::MgtHeadPush push;
    push.rows = vocab;
    push.k = dim;
    push.slices = runner.spec().head_slices;
    push.x_stride = dim;
    gpu::CommandPool pool;
    require(pool.create(device));
    gpu::QueryPool queries;
    require(queries.create(device, 2));
    auto record = [&](gpu::CommandBuffer &cmd) {
        for (uint32_t slice = 0; slice < push.slices; ++slice) {
            push.slice = slice;
            require(runner.record(cmd, rows, stage, &push, sizeof(push), runner.row_groups(vocab)));
            require(cmd.barrier());
        }
    };
    auto warmup = require(pool.acquire());
    require(warmup.begin());
    record(warmup);
    require(warmup.end());
    require(gpu::submit_and_wait(device, warmup));
    auto cmd = require(pool.acquire());
    require(cmd.begin());
    require(cmd.reset_queries(queries, 0, 2));
    require(cmd.write_timestamp(queries, 0, false));
    for (uint32_t iteration = 0; iteration < iterations; ++iteration) record(cmd);
    require(cmd.write_timestamp(queries, 1, true));
    require(cmd.end());
    const auto start = std::chrono::steady_clock::now();
    require(gpu::submit_and_wait(device, cmd));
    const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    const double ms = require(queries.elapsed_seconds(0, 1)) * 1000 / iterations;
    std::vector<float> cached_logits(size_t(rows) * vocab);
    wc_readback(cached_logits.data(), logits.host_ptr, logits.bytes);
    if (fs::exists(out + ".logits.f32")) throw std::runtime_error("logits output must be new");
    std::ofstream logits_file(out + ".logits.f32", std::ios::binary);
    logits_file.exceptions(std::ios::failbit | std::ios::badbit);
    logits_file.write(reinterpret_cast<const char *>(cached_logits.data()), cached_logits.size() * sizeof(float));
    std::ofstream result(out);
    result.exceptions(std::ios::failbit | std::ios::badbit);
    const auto text = std::format("{{\"format\":\"{}\",\"rows\":{},\"iterations\":{},"
                                  "\"slices\":{},\"weight_bytes\":{},\"additional_copy_bytes\":{},\"gpu_ms\":{},\"effective_gbps\":{},"
                                  "\"batch_wall_ms\":{},\"per_dispatch_queries\":false}}\n",
                                  format, rows, iterations, push.slices, spec.draft_head_fp8 ? copy_bytes : head->data_bytes, copy_bytes, ms,
                                  (spec.draft_head_fp8 ? copy_bytes : head->data_bytes) / (ms * 1e6), wall);
    result << text;
    std::cout << text;
    return 0;
} catch (const std::exception &error) {
    std::cerr << "draft_head_bench: " << error.what() << '\n';
    return 1;
}
