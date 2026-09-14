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
#include <cmath>

#include "cpu/dequant.h"
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
    uint32_t hits = 0, misses = 0;
    uint64_t bytes = 0;
    for (const runtime::LayerTiming& t : engine.layer_timings()) {
        hits += t.hits; misses += t.misses; bytes += t.miss_bytes;
    }
    const runtime::StepBreakdown& b = r.breakdown;
    // design 13.1: every token's time, split into the buckets the design's
    // upper-bound model is written in. The GPU halves are GPU timestamps.
    std::printf("%3u  %6u -> %6u  %8.1f ms | attn %5.1f  moe gpu %5.1f host %4.1f  "
                "stall %7.1f  engram %4.1f  tail %4.1f  other %5.1f  submits %3u | "
                "hit %3u/%3u  nvme %6.1f MB  margin %.4f\n",
                step, in, r.token, r.wall_ms, b.attn_ms, b.moe_gpu_ms, b.moe_host_ms,
                b.gate_ms, b.engram_ms, b.tail_ms, b.other_ms, b.submits,
                hits, hits + misses, bytes / 1e6, r.margin());
    std::printf("                                  other = record %4.1f + submit %4.1f + "
                "bind %4.1f + fence wait %5.1f (GPU time inside the fences: %5.1f); "
                "moe host = x %4.1f + act_quant %4.1f + table %4.1f\n",
                b.record_ms, b.submit_ms, b.bind_ms, b.wait_ms,
                b.attn_ms + b.moe_gpu_ms + b.tail_ms,
                b.moe_x_ms, b.moe_quant_ms, b.moe_table_ms);
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

// What a slow prefill produced, against the state the oracle's prompt left.
// Per POSITION as well as in aggregate: the aggregate cannot distinguish an
// error that compounds with the token from one that is uniform, and only the
// second would be a bug.
void print_prefill_state(const runtime::Engine& engine, const runtime::DecodeState& st) {
    const TextConfig& c = engine.model().text;
    const uint32_t n = static_cast<uint32_t>(st.prompt_ids().size());
    auto cos_of = [](const std::vector<float>& a, const float* b) {
        double num = 0, sa = 0, sb = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            num += double(a[i]) * double(b[i]);
            sa  += double(a[i]) * double(a[i]);
            sb  += double(b[i]) * double(b[i]);
        }
        return (sa > 0 && sb > 0) ? num / std::sqrt(sa * sb) : 1.0;
    };
    std::puts("  prefill state vs the oracle's, by layer (window KV cos at five prompt "
              "positions, then compressed KV / index keys)");
    for (uint32_t L = 0; L < c.num_hidden_layers; ++L) {
        auto v = engine.kv().layer(L);
        if (!v) continue;
        const runtime::StateTensor* g = st.tensor(0, std::format("L{:02d}.win_kv", L));
        if (!g) continue;
        const uint32_t blocks = c.head_dim / 32;
        std::printf("   L%-2u win", L);
        double worst = 1.0;
        for (uint32_t r : {0u, 1u, n / 4, n / 2, n - 1}) {
            std::vector<float> ours(c.head_dim);
            for (uint32_t d = 0; d < c.head_dim; ++d)
                ours[d] = cpu::fp8_e4m3_to_float(v->win_val_host[size_t(r) * c.head_dim + d]) *
                          cpu::e8m0_to_float(v->win_scale_host[size_t(r) * blocks + d / 32]);
            const double cs = cos_of(ours, g->f.data() + size_t(r) * c.head_dim);
            worst = std::min(worst, cs);
            std::printf(" p%u %.6f", r, cs);
        }
        const uint32_t ratio = c.compress_ratio(L);
        if (ratio && c.is_kv_source(L)) {
            const uint32_t rows = n / ratio;
            if (const runtime::StateTensor* gc = st.tensor(0, std::format("L{:02d}.cmp_cache", L))) {
                std::vector<float> ours(size_t(rows) * c.head_dim);
                for (size_t i = 0; i < ours.size(); ++i) ours[i] = cpu::bf16_to_float(v->cmp_kv_host[i]);
                std::printf(" | cmp %.6f", cos_of(ours, gc->f.data()));
            }
            if (const runtime::StateTensor* gk = st.tensor(0, std::format("L{:02d}.index_k", L))) {
                std::vector<float> ours(size_t(rows) * c.index_head_dim);
                for (size_t i = 0; i < ours.size(); ++i) ours[i] = cpu::bf16_to_float(v->idx_key_host[i]);
                std::printf(" | key %.6f", cos_of(ours, gk->f.data()));
            }
        }
        std::printf("\n");
    }
}

// FNV-1a over a host COPY of a GPU-visible vector: hashing straight off the
// mapping would be 5,120 write-combining reads (docs/p2_decode.md §3.3).
uint64_t fnv_copy(const float* p, size_t n) {
    std::vector<float> v(p, p + n);
    uint64_t h = 1469598103934665603ull;
    const auto* b = reinterpret_cast<const uint8_t*>(v.data());
    for (size_t i = 0; i < n * sizeof(float); ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

// docs/p2_decode.md §4.4: is a decode step bit-reproducible? The same warm step
// is run `runs` times, and at every layer four things are hashed -- the FFN
// sublayer input (after attention and hc_post), the gate scores, the MoE
// output and the stream after the attention half -- plus the logits. The first
// (layer, tensor) whose hash is not the same in every run is where the
// non-determinism enters; everything downstream of it differs by consequence.
int run_determinism(runtime::Engine& engine, const runtime::DecodeState& st, uint32_t runs) {
    const TextConfig& c = engine.model().text;
    const uint32_t dim = c.hidden_size;
    struct LayerHash { uint64_t u = 0, scores = 0, moe = 0, stream = 0; };
    std::vector<std::vector<LayerHash>> h(runs, std::vector<LayerHash>(c.num_hidden_layers));
    std::vector<uint64_t> logits(runs);
    // One unprobed step first, so every expert the step needs is resident and
    // no run differs from another by what the cache happened to hold.
    if (auto r = engine.decode_step(st.greedy_tokens().front(), st.decode_pos(), 0); !r) {
        std::fprintf(stderr, "warm-up: %s\n", r.error().str().c_str());
        return 1;
    }
    for (uint32_t k = 0; k < runs; ++k) {
        engine.layer_probe = [&](uint32_t L, const runtime::DecodeLayer& dl) {
            h[k][L].u      = fnv_copy(dl.ffn_norm_out(), dim);
            h[k][L].scores = fnv_copy(dl.gate_scores(), c.n_routed_experts);
            h[k][L].moe    = fnv_copy(dl.moe_out(), dim);
            h[k][L].stream = fnv_copy(dl.block_out(), size_t(dim) * c.hc_mult);
        };
        auto r = engine.decode_step(st.greedy_tokens().front(), st.decode_pos(), 0);
        engine.layer_probe = nullptr;
        if (!r) { std::fprintf(stderr, "run %u: %s\n", k, r.error().str().c_str()); return 1; }
        logits[k] = fnv_copy(engine.last_logits().data(), engine.last_logits().size());
        std::printf("run %u: token %u margin %.9f logits %016llx\n", k, r->token,
                    r->margin(), static_cast<unsigned long long>(logits[k]));
    }
    int first = -1;
    const char* what = "";
    for (uint32_t L = 0; L < c.num_hidden_layers && first < 0; ++L) {
        struct F { const char* n; uint64_t LayerHash::*m; } fields[] = {
            {"stream after attention hc_post", &LayerHash::stream},
            {"ffn_norm output (MoE input)", &LayerHash::u},
            {"gate scores", &LayerHash::scores},
            {"MoE output", &LayerHash::moe},
        };
        for (const F& f : fields) {
            bool same = true;
            for (uint32_t k = 1; k < runs; ++k) same &= (h[k][L].*f.m == h[0][L].*f.m);
            if (!same) { first = int(L); what = f.n; break; }
        }
    }
    bool logits_same = true;
    for (uint32_t k = 1; k < runs; ++k) logits_same &= logits[k] == logits[0];
    if (first < 0)
        std::printf("deterministic: %u runs, every layer's four tensors and the logits "
                    "bit-identical (%s)\n", runs, logits_same ? "logits too" : "BUT the logits differ");
    else
        std::printf("NOT deterministic: first difference at layer %d, %s\n", first, what);
    return 0;
}

int cmd_run(int argc, char** argv) {
    RuntimeConfig cfg;
    cfg.cache.budget_bytes = 0;     // 0 = size it from the machine (see init_gpu)
    std::string prompt_ids_file;
    std::string state_dir = "tests/data/l3";
    uint32_t steps = 8;
    bool teacher_force = false;
    bool per_layer = false;
    bool slow_prefill = false;
    bool loaded_ced = false;
    uint32_t warm = 0;
    uint32_t determinism = 0;

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
        else if (a == "--slow-prefill") slow_prefill = true;
        else if (a == "--loaded-ced")  loaded_ced = true;
        else if (a == "--warm")        warm = static_cast<uint32_t>(std::atoi(arg_value(argc, argv, i, a).data()));
        else if (a == "--determinism") determinism = static_cast<uint32_t>(std::atoi(arg_value(argc, argv, i, a).data()));
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
    if (loaded_ced) engine.set_produce_ced(false);
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

    if (slow_prefill) {
        auto pre = engine.slow_prefill(st->prompt_ids());
        if (!pre) { std::fprintf(stderr, "slow prefill: %s\n", pre.error().str().c_str()); return 1; }
        std::printf("slow prefill of %zu tokens -> token %u (reference %u) %s\n",
                    st->prompt_ids().size(), pre->token, st->greedy_tokens().front(),
                    pre->token == st->greedy_tokens().front() ? "MATCH" : "DIFFER");
        print_prefill_state(engine, *st);
    }
    std::fputs(engine.status().c_str(), stdout);

    if (auto ov = engine.measure_submit_overhead(64); ov)
        std::printf("submit    %.3f ms per submit+fence round trip; a decode step makes "
                    "about 128 of them (design 13.1's dispatch bucket)\n", *ov);

    // Past the export's last step there is no reference to compare against,
    // but with design 7.4's kernels producing the compressed KV there is also
    // nothing left to load, so a free-running decode can go as far as the KV
    // store was sized for. Teacher forcing needs the reference's tokens.
    const bool unbounded = engine.produce_ced() && !teacher_force;
    const uint32_t n = unbounded ? steps : std::min<uint32_t>(steps, st->steps());
    if (n < steps)
        std::printf("note: the loaded state covers %u steps, so %u were run\n",
                    st->steps(), n);
    std::printf("\nstep  in     -> out     wall      | the design 13.1 breakdown\n");

    if (determinism) return run_determinism(engine, *st, determinism);

    // --warm K: step 0 run K extra times first, at the same position with the
    // same input. The first fetches every expert that step routes to; every
    // repeat after it is a step whose experts are all resident, which is the
    // compute floor docs/p2_decode.md reports. Rewriting position 64's ring
    // slot and compressor slot with the same values leaves the state as it was.
    for (uint32_t w = 0; w < warm; ++w) {
        auto r = engine.decode_step(st->greedy_tokens().front(), st->decode_pos(), 0);
        if (!r) { std::fprintf(stderr, "warm-up %u: %s\n", w, r.error().str().c_str()); return 1; }
        std::printf("w");
        print_token_line(w, st->greedy_tokens().front(), *r, engine);
    }

    uint32_t next = st->greedy_tokens().front();
    std::vector<uint32_t> produced;
    for (uint32_t s = 0; s < n; ++s) {
        const uint32_t in = teacher_force ? st->greedy_tokens()[s] : next;
        auto r = engine.decode_step(in, st->decode_pos() + s,
                                    s < st->steps() ? static_cast<int32_t>(s) : -1);
        if (!r) { std::fprintf(stderr, "step %u: %s\n", s, r.error().str().c_str()); return 1; }
        print_token_line(s, in, *r, engine);
        if (per_layer) print_layer_table(engine);
        produced.push_back(r->token);
        next = r->token;
    }

    const uint32_t nref = std::min<uint32_t>(n, st->steps());
    std::printf("\ntokens   ");
    for (uint32_t t : produced) std::printf("%u ", t);
    std::printf("\nreference");
    for (uint32_t s = 0; s < nref; ++s) std::printf(" %u", st->greedy_tokens()[s + 1]);
    // Teacher-forced, every step starts from the reference's own input, so
    // every step is an independent comparison and all of them count. Free
    // running, a step after the first mismatch is decoding a sequence the
    // reference never produced -- and against LOADED compressed KV that belongs
    // to the reference's trajectory -- so only the leading run means anything.
    uint32_t match = 0;
    if (teacher_force) {
        for (uint32_t s = 0; s < nref; ++s)
            match += (produced[s] == st->greedy_tokens()[s + 1]) ? 1 : 0;
    } else {
        while (match < nref && produced[match] == st->greedy_tokens()[match + 1]) ++match;
    }
    std::printf("\n%u/%u tokens match the fp32 reference%s\n", match, nref,
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
