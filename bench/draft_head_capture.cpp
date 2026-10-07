// Owner TODO 4.9: capture the actual draft head input on one 64-token path.
// Observers copy existing results; they never launch another target forward.
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <format>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "model/layout.h"
#include "runtime/engine.h"

using namespace cachedmoe;
namespace fs = std::filesystem;

namespace {
template<class T> T require(Result<T> value) {
    if (!value)
        throw std::runtime_error(value.error().str());
    return std::move(*value);
}
void require(Result<void> value) {
    if (!value)
        throw std::runtime_error(value.error().str());
}
template<class T> std::string array(const T &values) {
    std::string text = "[";
    for (auto value : values) {
        if (text.size() > 1)
            text += ',';
        text += std::format("{}", value);
    }
    return text + ']';
}
} // namespace

int main(int argc, char **argv) try {
    RuntimeConfig cfg;
    fs::path prompt_path, out;
    uint32_t slots = 0;
    std::string head_mode;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc)
            throw std::runtime_error("each option requires a value");
        const std::string key = argv[i], value = argv[i + 1];
        if (key == "--model") cfg.model_dir = value;
        else if (key == "--mirror") cfg.model_mirrors.push_back(value);
        else if (key == "--prompt-ids") prompt_path = value;
        else if (key == "--out") out = value;
        else if (key == "--slots") slots = std::stoul(value);
        else if (key == "--head") head_mode = value;
        else throw std::runtime_error("unknown option: " + key);
    }
    if (!head_mode.empty() && head_mode != "native" && head_mode != "fp8")
        throw std::runtime_error("head must be native or fp8");
    if (cfg.model_dir.empty() || cfg.model_mirrors.size() != 1 || slots == 0 ||
        prompt_path.empty() || out.empty())
        throw std::runtime_error("require --model --mirror --slots --prompt-ids --out");
    if (fs::exists(out))
        throw std::runtime_error("output directory already exists");
    const auto destination = fs::weakly_canonical(out);
    for (const auto &root : {cfg.model_dir, cfg.model_mirrors.front()}) {
        const auto checkpoint = fs::weakly_canonical(root);
        auto left = checkpoint.begin(), right = destination.begin();
        while (left != checkpoint.end() && right != destination.end() && *left == *right) {
            ++left;
            ++right;
        }
        if (left == checkpoint.end())
            throw std::runtime_error("capture destination must be outside checkpoint roots");
    }
    std::ifstream source(prompt_path);
    source.exceptions(std::ios::badbit);
    std::vector<uint32_t> prompt;
    uint32_t token;
    while (source >> token)
        prompt.push_back(token);
    if (prompt.empty() || !source.eof())
        throw std::runtime_error("prompt must contain whitespace-separated token IDs");

    cfg.cache.budget_bytes = uint64_t(slots) * layout::kExpertSlotBytes;
    cfg.speculation.enabled = true;
    cfg.speculation.max_draft = 2; // This experiment's fixed k, not a runtime default.
    cfg.max_context = configuration::facts::MAX_CONTEXT;
    runtime::Engine engine;
    require(engine.init(cfg));
    const auto &actual = engine.config();
    if (!actual.gpu.draft_onecb || actual.gpu.batch_gpu_route || !actual.decode.dynamic_mask_lru)
        throw std::runtime_error("capture requires ONECB, CPU routing and dynamic mask policy");
    if (engine.io().live_source_count() != cfg.model_mirrors.size() + 1)
        throw std::runtime_error("capture requires both checkpoint read sources");
    // Resolve policy once through Engine::init, then reject wrong configurations
    // before creating the GPU device or reading the pinned weight payloads.
    require(engine.init_gpu());
    if (!head_mode.empty()) require(engine.dspark_runtime()->set_head_fp8(head_mode == "fp8"));
    engine.set_resident_only(runtime::Engine::ResidentOnly::Mask);
    runtime::SessionConfig session;
    session.max_context = cfg.max_context;
    require(engine.begin_session(session));
    runtime::SamplingParams sampling;
    sampling.temperature = 0; // Greedy target IDs make the offline estimate explicit.
    engine.set_sampling(sampling);
    auto first = require(engine.gpu_prefill(prompt));

    fs::create_directories(out);
    std::ofstream hidden(out / "hidden.f32", std::ios::binary);
    std::ofstream cycles(out / "cycles.jsonl");
    hidden.exceptions(std::ios::badbit | std::ios::failbit);
    cycles.exceptions(std::ios::badbit | std::ios::failbit);
    uint32_t hidden_count = 0;
    std::vector<uint32_t> proposals, target_top4, verified_greedy;
    engine.spec_verify_probe = [&](std::span<const uint32_t> input,
                                   std::span<const runtime::Engine::BatchRow> verified_rows,
                                   std::span<const float> logits) {
        proposals.assign(input.begin() + 1, input.end());
        target_top4.clear();
        verified_greedy.clear();
        for (const auto &row : verified_rows)
            verified_greedy.push_back(row.argmax);
        const uint32_t vocab = engine.model().text.vocab_size;
        std::vector<uint32_t> ids(vocab);
        for (size_t row = 0; row < input.size(); ++row) {
            std::iota(ids.begin(), ids.end(), 0);
            auto values = logits.subspan(row * vocab, vocab);
            std::partial_sort(ids.begin(), ids.begin() + configuration::facts::ACCEPT_TOP_K,
                              ids.end(), [&](uint32_t a, uint32_t b) {
                return values[a] > values[b] || (values[a] == values[b] && a < b);
            });
            target_top4.insert(target_top4.end(), ids.begin(),
                               ids.begin() + configuration::facts::ACCEPT_TOP_K);
        }
    };
    engine.dspark_runtime()->probe = [&](std::string_view name, std::span<const float> values) {
        if (name != "head_norm_out")
            return;
        const auto expected = size_t(configuration::facts::DRAFT_BLOCK_SIZE) * engine.model().text.hidden_size;
        if (values.size() != expected)
            throw std::runtime_error("unexpected draft hidden geometry");
        hidden.write(reinterpret_cast<const char *>(values.data()), values.size_bytes());
        ++hidden_count;
    };
    cycles << std::format("{{\"schema\":1,\"dim\":{},\"block_rows\":{},\"vocab\":{},"
                          "\"slot_bytes\":{},\"first_output\":{},\"outputs\":64,\"sampling\":\"greedy\"}}\n",
                          engine.model().text.hidden_size, configuration::facts::DRAFT_BLOCK_SIZE,
                          engine.model().text.vocab_size, layout::kExpertSlotBytes, first.token);
    uint32_t root = first.token, generated = 1;
    while (generated < 64) {
        const auto position = engine.context_length();
        const auto before = hidden_count;
        const auto calls = engine.batch_forward_calls();
        auto step = require(engine.speculative_step(root, 64 - generated));
        if (engine.io().live_source_count() != cfg.model_mirrors.size() + 1)
            throw std::runtime_error("checkpoint read source dropped during capture");
        if (engine.batch_forward_calls() != calls + 1)
            throw std::runtime_error("capture must use exactly one target forward per cycle");
        std::vector<uint32_t> emitted, targets;
        for (const auto &row : step.rows) {
            emitted.push_back(row.token);
            targets.push_back(row.greedy_token);
        }
        cycles << std::format("{{\"position\":{},\"root\":{},\"k\":{},\"accepted\":{},"
                              "\"hidden_index\":{},\"emitted\":{},\"target_greedy\":{},"
                              "\"proposal\":{},\"target_top4\":{},\"emitted_target_greedy\":{},"
                              "\"draft_ms\":{},\"verify_ms\":{}}}\n",
                              position, root, step.cycle.k, step.cycle.accepted,
                              hidden_count == before ? -1 : int(before), array(emitted),
                              array(verified_greedy), array(proposals), array(target_top4), array(targets),
                              step.cycle.draft_ms, step.cycle.verify_ms);
        if (emitted.empty())
            throw std::runtime_error("empty speculative output");
        generated += emitted.size();
        root = emitted.back();
    }
    std::cout << engine.status() << '\n' << engine.resident_route_report() << '\n';
    std::cout << "captured " << hidden_count << " draft inputs, " << generated << " output IDs\n";
    return 0;
} catch (const std::exception &error) {
    std::cerr << "draft_head_capture: " << error.what() << '\n';
    return 1;
}
