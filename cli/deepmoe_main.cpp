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
        "  deepmoe run --model DIR [--prompt-ids FILE] [--steps N]\n"
        "              [--state DIR] [--teacher-force] [--per-layer]\n"
        "              [--cache-gb N] [--profile FILE.jsonl] [--chunk-kb N] [--qd N]\n"
        "      Decode N tokens and print the section 13.1 per-token breakdown.\n"
        "      --prompt-ids is a file of token ids (there is no tokenizer in the\n"
        "      runtime yet, section 15 P0); --state is the oracle's L3 export,\n"
        "      which supplies the state prefill will supply once it exists --\n"
        "      the window KV after the prompt and, per step, the compressed KV\n"
        "      and indexer top-k of section 7.4. Default --state tests/data/l3.\n"
        "      --cache-gb 0 (the default) sizes the routed-expert cache from the\n"
        "      machine. See docs/p2_decode.md.\n"
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

// A whitespace- or comma-separated list of token ids. There is no tokenizer in
// the runtime yet (design §15 P0), so a prompt reaches `deepmoe run` as ids --
// which is also what makes a run reproducible against tools/oracle.py.
Result<std::vector<uint32_t>> read_prompt_ids(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return fail(Err::NotFound, std::format("cannot open '{}'", path));
    std::vector<uint32_t> ids;
    std::string tok;
    for (int c = std::fgetc(f);; c = std::fgetc(f)) {
        const bool sep = c == EOF || c == ',' || c == '[' || c == ']' ||
                         c == ' ' || c == '\n' || c == '\r' || c == '\t';
        if (!sep) { tok.push_back(static_cast<char>(c)); continue; }
        if (!tok.empty()) { ids.push_back(static_cast<uint32_t>(std::strtoul(tok.c_str(), nullptr, 10))); tok.clear(); }
        if (c == EOF) break;
    }
    std::fclose(f);
    if (ids.empty()) return fail(Err::InvalidArgument, std::format("'{}' has no token ids", path));
    return ids;
}

void print_token_line(uint32_t step, uint32_t in, const runtime::DecodeStepResult& r,
                      const runtime::Engine& engine) {
    double engram = 0, attn = 0, gate = 0, moe = 0, moe_gpu = 0, moe_host = 0;
    uint32_t hits = 0, misses = 0;
    uint64_t bytes = 0;
    for (const runtime::LayerTiming& t : engine.layer_timings()) {
        engram += t.engram_ms; attn += t.attn_ms; gate += t.gate_ms; moe += t.moe_ms;
        moe_gpu += t.moe_gpu_ms; moe_host += t.moe_host_ms;
        hits += t.hits; misses += t.misses; bytes += t.miss_bytes;
    }
    // design §13.1: every token's time, split into the buckets the design's
    // upper-bound model is written in.
    std::printf("%3u  %6u -> %6u  %8.1f ms | attn %6.1f  moe %6.1f (gpu %5.1f host %4.1f)  "
                "stall %7.1f  engram %4.1f  other %4.1f | hit %3u/%3u  nvme %6.1f MB  "
                "margin %.4f\n",
                step, in, r.token, r.wall_ms, attn, moe, moe_gpu, moe_host, gate, engram,
                r.wall_ms - attn - moe - gate - engram,
                hits, hits + misses, bytes / 1e6, r.margin());
}

// design 13.1 asks for the breakdown PER LAYER, which is where an anomaly is
// actually visible: one layer stalling for 200 ms, or what the two engram
// layers cost. The aggregate line hides all of it.
void print_layer_table(const runtime::Engine& engine) {
    std::puts("      layer  engram    attn    stall     moe    gpu   host  hit  nvme MB");
    uint32_t L = 0;
    for (const runtime::LayerTiming& t : engine.layer_timings()) {
        std::printf("      %5u  %6.2f  %6.2f  %7.1f  %6.2f %6.2f %6.2f  %u/6  %7.1f\n",
                    L, t.engram_ms, t.attn_ms, t.gate_ms, t.moe_ms, t.moe_gpu_ms,
                    t.moe_host_ms, t.hits, t.miss_bytes / 1e6);
        ++L;
    }
}

int cmd_run(int argc, char** argv) {
    RuntimeConfig cfg;
    cfg.cache.budget_bytes = 0;     // 0 = size it from the machine (see init_gpu)
    std::string prompt_ids_file;
    std::string state_dir = "tests/data/l3";
    uint32_t steps = 8;
    bool teacher_force = false;
    bool per_layer = false;

    for (int i = 2; i < argc; ++i) {
        const std::string_view a = argv[i];
        if (a == "--model")            cfg.model_dir = arg_value(argc, argv, i, a);
        else if (a == "--prompt-ids")  prompt_ids_file = arg_value(argc, argv, i, a);
        else if (a == "--state")       state_dir = arg_value(argc, argv, i, a);
        else if (a == "--profile")     cfg.profile_jsonl = arg_value(argc, argv, i, a);
        else if (a == "--kvcache")     cfg.kvcache_dir = arg_value(argc, argv, i, a);
        else if (a == "--steps")       steps = static_cast<uint32_t>(std::atoi(arg_value(argc, argv, i, a).data()));
        else if (a == "--teacher-force") teacher_force = true;
        else if (a == "--per-layer")   per_layer = true;
        else if (a == "--cache-gb")    cfg.cache.budget_bytes = uint64_t(std::atoll(arg_value(argc, argv, i, a).data())) << 30;
        else if (a == "--chunk-kb")    cfg.io.chunk_bytes = static_cast<uint32_t>(std::atoi(arg_value(argc, argv, i, a).data())) << 10;
        else if (a == "--qd")          cfg.io.max_inflight_ops = static_cast<uint32_t>(std::atoi(arg_value(argc, argv, i, a).data()));
        else if (a == "--buffered")    cfg.io.unbuffered = false;
        else { std::fprintf(stderr, "unknown option %.*s\n", static_cast<int>(a.size()), a.data()); return usage(); }
    }
    if (cfg.model_dir.empty()) {
        std::fputs("--model DIR is required (the 48 safetensors shards plus "
                   "deepmoe_manifest.json)\n", stderr);
        return 2;
    }

    runtime::Engine engine;
    if (auto r = engine.init(cfg); !r) {
        std::fprintf(stderr, "init failed: %s\n", r.error().str().c_str());
        return 1;
    }
    if (auto r = engine.init_gpu(); !r) {
        std::fprintf(stderr, "gpu init: %s\n", r.error().str().c_str());
        return 1;
    }
    // Prefill is design §11 / P5. Until it exists the state a decode step
    // starts from is the oracle's, and the run says so on the status line.
    if (auto r = engine.load_decode_state(state_dir); !r) {
        std::fprintf(stderr, "decode state '%s': %s\n(run "
                     "`tools/oracle.py --level l3 --out %s` first)\n",
                     state_dir.c_str(), r.error().str().c_str(), state_dir.c_str());
        return 1;
    }
    std::fputs(engine.status().c_str(), stdout);

    const runtime::DecodeState* st = engine.decode_state();
    std::vector<uint32_t> prompt;
    if (!prompt_ids_file.empty()) {
        auto ids = read_prompt_ids(prompt_ids_file);
        if (!ids) { std::fprintf(stderr, "%s\n", ids.error().str().c_str()); return 1; }
        prompt = *ids;
        if (prompt != st->prompt_ids()) {
            std::fprintf(stderr,
                         "--prompt-ids has %zu tokens and the loaded state was built from "
                         "%zu; they must be the same prompt, because the window KV in "
                         "'%s' is what that prompt left behind.\n",
                         prompt.size(), st->prompt_ids().size(), state_dir.c_str());
            return 1;
        }
    }

    if (auto ov = engine.measure_submit_overhead(64); ov)
        std::printf("submit    %.3f ms per submit+fence round trip; a decode step makes "
                    "about 128 of them (design 13.1's dispatch bucket)\n", *ov);

    const uint32_t n = std::min<uint32_t>(steps, st->steps());
    if (n < steps)
        std::printf("note: the loaded state covers %u steps, so %u were run\n",
                    st->steps(), n);
    std::printf("\nstep  in     -> out     wall      | the design 13.1 breakdown\n");

    uint32_t next = st->greedy_tokens().front();
    std::vector<uint32_t> produced;
    for (uint32_t s = 0; s < n; ++s) {
        const uint32_t in = teacher_force ? st->greedy_tokens()[s] : next;
        auto r = engine.decode_step(in, st->decode_pos() + s, static_cast<int32_t>(s));
        if (!r) { std::fprintf(stderr, "step %u: %s\n", s, r.error().str().c_str()); return 1; }
        print_token_line(s, in, *r, engine);
        if (per_layer) print_layer_table(engine);
        produced.push_back(r->token);
        next = r->token;
    }

    std::printf("\ntokens   ");
    for (uint32_t t : produced) std::printf("%u ", t);
    std::printf("\nreference");
    for (uint32_t s = 0; s < n; ++s) std::printf(" %u", st->greedy_tokens()[s + 1]);
    // Teacher-forced, every step starts from the reference's own input, so
    // every step is an independent comparison and all of them count. Free
    // running, a step after the first mismatch is decoding a sequence the
    // reference never produced -- and against LOADED compressed KV that belongs
    // to the reference's trajectory -- so only the leading run means anything.
    uint32_t match = 0;
    if (teacher_force) {
        for (uint32_t s = 0; s < n; ++s)
            match += (produced[s] == st->greedy_tokens()[s + 1]) ? 1 : 0;
    } else {
        while (match < n && produced[match] == st->greedy_tokens()[match + 1]) ++match;
    }
    std::printf("\n%u/%u tokens match the fp32 reference%s\n", match, n,
                teacher_force ? " (teacher-forced)" : " before divergence");
    std::fputs(engine.profiler().summary().to_string().c_str(), stdout);
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
