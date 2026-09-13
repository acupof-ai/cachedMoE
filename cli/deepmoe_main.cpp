// deepmoe CLI.
//
//   deepmoe info                     Vulkan heaps/limits, CPU features, layout budget
//   deepmoe bench nvme [options]     the P-1 NVMe micro-benchmark (design §9.2 Q6/Q7)
//   deepmoe run --model DIR ...      the token loop (stub until P2/P3)
//
// Ownership/threading: one process, one Engine, main thread only. Nothing here
// is a library; everything reusable lives in the modules.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <string>
#include <string_view>
#include <vector>

#include "core/config.h"
#include "core/log.h"
#include "core/status.h"
#include "cpu/gemv_avx512.h"
#include "gpu/vulkan/device.h"
#include "model/layout.h"
#include "runtime/engine.h"
#include "storage/backend.h"

using namespace deepmoe;

namespace {

int usage(int code = 2) {
    std::puts(
        "deepmoe -- MoE inference runtime for DeepSeek-V4.1-Flash on Strix Halo\n"
        "\n"
        "usage:\n"
        "  deepmoe info\n"
        "      Vulkan devices, heaps and the capabilities of design section 1.1,\n"
        "      CPU features, and the per-token byte budget of section 2.3.\n"
        "\n"
        "  deepmoe bench nvme [--file PATH] [--size-gb N] [--chunk-kb N,N,...]\n"
        "                     [--qd N,N,...] [--pattern seq|rand|both] [--keep]\n"
        "      Sequential and random read throughput against chunk size x queue\n"
        "      depth, through storage/IoEngine (design section 9.2, Q6/Q7).\n"
        "      Delegates to the nvme_bench executable when it is on PATH; the\n"
        "      same sweep is available directly as `nvme_bench`.\n"
        "\n"
        "  deepmoe run --model DIR [--prompt TEXT] [--max-tokens N]\n"
        "              [--profile FILE.jsonl] [--cache-gb N] [--no-gpu]\n"
        "      Load the repacked model and decode. Not implemented yet: the\n"
        "      kernels of design section 7 land in P2/P3.\n"
        "\n"
        "  common: -v / -vv raise the log level\n");
    return code;
}

std::string_view arg_value(int argc, char** argv, int& i, std::string_view name) {
    if (i + 1 >= argc) {
        std::fprintf(stderr, "%.*s needs a value\n", static_cast<int>(name.size()), name.data());
        std::exit(2);
    }
    return argv[++i];
}

int cmd_info() {
    std::puts("== deepmoe environment ==\n");

    // GPU (design §1.1)
    auto devices = gpu::Device::enumerate();
    if (!devices) {
        std::puts(std::format("vulkan: unavailable ({})", devices.error().str()).c_str());
    } else {
        for (const gpu::DeviceCaps& c : *devices) {
            std::fputs(c.to_string().c_str(), stdout);
            auto req = c.check_required();
            std::puts(req ? "  required capabilities: all present"
                          : std::format("  MISSING:\n{}", req.error().message).c_str());
            std::puts("");
        }
    }

    // CPU (design §8)
    std::puts(std::format("cpu: avx512-vnni {}", cpu::has_avx512_vnni() ? "yes" : "NO").c_str());

    // I/O backends (design §9.6)
    IoConfig io;
    auto ds = storage::make_directstorage_backend(io);
    std::puts(std::format("io: directstorage {}", ds ? "available"
                                                     : ds.error().str()).c_str());
    auto def = storage::make_default_backend(io);
    std::puts(std::format("io: default backend {}",
                          def ? (*def)->caps().name : def.error().str()).c_str());

    // The numbers every decision in the design rests on.
    namespace L = layout;
    std::puts("");
    std::puts("== model budget (design section 2.3) ==");
    std::puts(std::format("  routed expert payload {} B = {} x 4 KiB", L::kExpertBytes, L::kExpertSectors).c_str());
    std::puts(std::format("  slab slot             {} B = {} x 4 KiB (2 aligned runs, design section 5.1)",
                          L::kExpertSlotBytes, L::kExpertSlotSectors).c_str());
    std::puts(std::format("  experts per layer     {} ({:.2f} GB contiguous)",
                          L::kRoutedExperts, L::kExpertsPerLayerBytes / 1e9).c_str());
    std::puts(std::format("  routed experts total  {} ({:.1f} GB)",
                          L::kRoutedExpertCount, L::kRoutedExpertTotalBytes / 1e9).c_str());
    std::puts(std::format("  engram row            {} B = {} value + {} scale, in two planes",
                          L::kEngramRowBytes, L::kEngramValueRowBytes, L::kEngramScaleRowBytes).c_str());
    std::puts(std::format("  resident per token    {:.2f} GB", L::kHotBytesPerToken / 1e9).c_str());
    std::puts(std::format("  routed per token      {:.2f} GB ({} layers x {} experts)",
                          double(L::kNumLayers) * L::kExpertsPerTok * L::kExpertBytes / 1e9,
                          L::kNumLayers, L::kExpertsPerTok).c_str());
    return 0;
}

int cmd_bench(int argc, char** argv) {
    if (argc < 3 || std::strcmp(argv[2], "nvme") != 0) {
        std::fputs("only `deepmoe bench nvme` exists; run `bw_matrix` for the memory matrix\n", stderr);
        return 2;
    }
    // The sweep lives in bench/nvme_bench.cpp so it can be run standalone on a
    // machine with no model. Forward the arguments.
    std::string cmd = "nvme_bench";
    for (int i = 3; i < argc; ++i) { cmd += ' '; cmd += argv[i]; }
    std::puts(std::format("running: {}", cmd).c_str());
    const int rc = std::system(cmd.c_str());
    if (rc != 0)
        std::fputs("could not run nvme_bench; build it and put it on PATH "
                   "(it is built next to this binary)\n", stderr);
    return rc;
}

int cmd_run(int argc, char** argv) {
    RuntimeConfig cfg;
    bool want_gpu = true;
    std::string prompt;
    uint32_t max_tokens = 64;

    for (int i = 2; i < argc; ++i) {
        const std::string_view a = argv[i];
        if (a == "--model")            cfg.model_dir = arg_value(argc, argv, i, a);
        else if (a == "--prompt")      prompt = arg_value(argc, argv, i, a);
        else if (a == "--profile")     cfg.profile_jsonl = arg_value(argc, argv, i, a);
        else if (a == "--kvcache")     cfg.kvcache_dir = arg_value(argc, argv, i, a);
        else if (a == "--max-tokens")  max_tokens = static_cast<uint32_t>(std::atoi(arg_value(argc, argv, i, a).data()));
        else if (a == "--cache-gb")    cfg.cache.budget_bytes = uint64_t(std::atoll(arg_value(argc, argv, i, a).data())) << 30;
        else if (a == "--chunk-kb")    cfg.io.chunk_bytes = static_cast<uint32_t>(std::atoi(arg_value(argc, argv, i, a).data())) << 10;
        else if (a == "--qd")          cfg.io.max_inflight_ops = static_cast<uint32_t>(std::atoi(arg_value(argc, argv, i, a).data()));
        else if (a == "--no-gpu")      want_gpu = false;
        else if (a == "--buffered")    cfg.io.unbuffered = false;
        else { std::fprintf(stderr, "unknown option %.*s\n", static_cast<int>(a.size()), a.data()); return usage(); }
    }
    if (cfg.model_dir.empty()) {
        std::fputs("--model DIR is required (the output of tools/repack.py)\n", stderr);
        return 2;
    }

    runtime::Engine engine;
    if (auto r = engine.init(cfg); !r) {
        std::fprintf(stderr, "init failed: %s\n", r.error().str().c_str());
        return 1;
    }
    std::fputs(engine.status().c_str(), stdout);

    if (want_gpu) {
        if (auto r = engine.init_gpu(); !r) {
            std::fprintf(stderr, "gpu init: %s\n", r.error().str().c_str());
            if (r.error().code != Err::Unimplemented) return 1;
        }
    }

    runtime::GenerateOptions opts;
    opts.max_tokens  = max_tokens;
    opts.greedy      = cfg.temperature == 0.0f;
    opts.speculative = cfg.speculation.enabled;

    // TODO(design §15 P0): the tokenizer. Until it exists `--prompt` cannot be
    // turned into token ids, so an empty prompt is passed through and generate()
    // reports what is missing.
    (void)prompt;
    std::vector<uint32_t> tokens;
    auto out = engine.generate(tokens, opts);
    if (!out) {
        std::fprintf(stderr, "generate: %s\n", out.error().str().c_str());
        return 1;
    }
    std::fputs(out->summary.to_string().c_str(), stdout);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    // Log flags may appear anywhere; strip them before dispatching.
    std::vector<char*> args;
    args.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; ++i) {
        const std::string_view a = argv[i];
        if (a == "-v")       { set_log_level(LogLevel::Debug); continue; }
        if (a == "-vv")      { set_log_level(LogLevel::Trace); continue; }
        if (a == "-q")       { set_log_level(LogLevel::Warn);  continue; }
        args.push_back(argv[i]);
    }
    argc = static_cast<int>(args.size());
    argv = args.data();

    if (argc < 2) return usage();
    const std::string_view cmd = argv[1];
    if (cmd == "info")  return cmd_info();
    if (cmd == "bench") return cmd_bench(argc, argv);
    if (cmd == "run")   return cmd_run(argc, argv);
    if (cmd == "-h" || cmd == "--help" || cmd == "help") return usage(0);
    std::fprintf(stderr, "unknown command %.*s\n", static_cast<int>(cmd.size()), cmd.data());
    return usage();
}
