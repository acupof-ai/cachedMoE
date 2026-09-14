// `deepmoe serve`: one long-running Engine behind line-delimited JSON on
// stdin/stdout (Track P, docs/p3_chat.md §1).
//
// Requests, one JSON object per line:
//   {"op":"generate", "prompt_ids":[...] | "text":"...", "max_tokens":N,
//    "temperature":T, "top_p":P, "seed":S, "stop_ids":[...], "reuse":true}
//   {"op":"reset"}                       drop the KV state (the caches stay warm)
//   {"op":"tokenize", "text":"..."}      -> {"event":"tokens","ids":[...]}
//   {"op":"detokenize", "ids":[...]}     -> {"event":"text","text":"..."}
//   {"op":"status"}                      -> {"event":"status",...}
//   {"op":"quit"}
// Events, one JSON object per line on stdout:
//   {"event":"ready",...}  once, after the pinned set and the cache are up
//   {"event":"prefill","done":i,"total":n}
//   {"event":"token","id":..,"text":"..","t_ms":..,"step_ms":..,"p":..,"margin":..,"hit":..}
//   {"event":"done", <GenerateStats::json_fields>}
//   {"event":"error","message":".."}
// `temperature` <= 0 is greedy; the defaults are the model README's 1.0 / 0.95.
// Everything the engine logs goes to stderr: fd 1 is re-pointed at fd 2 and the
// protocol writes to a duplicate of the original stdout.
//
// Ownership/threading: main thread only.
#include <cstdio>
#include <cstdlib>
#include <format>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#else
#include <unistd.h>
#endif

#include "core/config.h"
#include "core/json.h"
#include "core/json_write.h"
#include "core/log.h"
#include "runtime/engine.h"
#include "runtime/session.h"
#include "text/tokenizer.h"

using namespace deepmoe;

namespace {

std::FILE* g_proto = nullptr;

void emit(const std::string& line) {
    std::fwrite(line.data(), 1, line.size(), g_proto);
    std::fputc('\n', g_proto);
    std::fflush(g_proto);
}

void emit_error(std::string_view msg) {
    emit("{\"event\":\"error\",\"message\":" + json_quote(msg) + "}");
}

bool read_line(std::string& out) {
    out.clear();
    for (;;) {
        const int c = std::fgetc(stdin);
        if (c == EOF) return !out.empty();
        if (c == '\n') return true;
        if (c != '\r') out.push_back(static_cast<char>(c));
    }
}

std::vector<uint32_t> uint_array(const JsonValue& v) {
    std::vector<uint32_t> out;
    if (auto a = v.as_array(); a)
        for (const JsonValue& x : **a)
            if (auto u = x.as_uint(); u) out.push_back(static_cast<uint32_t>(*u));
    return out;
}

std::string value_of(int argc, char** argv, int& i) {
    if (i + 1 >= argc) {
        std::fprintf(stderr, "%s needs a value\n", argv[i]);
        std::exit(2);
    }
    return argv[++i];
}

}  // namespace

int cmd_serve(int argc, char** argv) {
    RuntimeConfig cfg;
    cfg.cache.budget_bytes = 0;
    if (const char* e = std::getenv("DEEPMOE_MODEL_DIR")) cfg.model_dir = e;
    runtime::SessionConfig sc;
    runtime::SessionOptions so;
    bool check_topk = false;
    for (int i = 2; i < argc; ++i) {
        const std::string_view a = argv[i];
        if (a == "--model")                cfg.model_dir = value_of(argc, argv, i);
        else if (a == "--cache-gb")        cfg.cache.budget_bytes = uint64_t(std::atoll(value_of(argc, argv, i).c_str())) << 30;
        else if (a == "--max-context")     sc.max_context = uint32_t(std::atoi(value_of(argc, argv, i).c_str()));
        else if (a == "--engram-tables")   sc.engram_tables_dir = value_of(argc, argv, i);
        else if (a == "--gpu-prefill-min") so.gpu_prefill_min = uint32_t(std::atoi(value_of(argc, argv, i).c_str()));
        else if (a == "--replay")          so.replay = uint32_t(std::atoi(value_of(argc, argv, i).c_str()));
        else if (a == "--profile")         cfg.profile_jsonl = value_of(argc, argv, i);
        else if (a == "--check-topk")      check_topk = true;
        else {
            std::fprintf(stderr, "unknown option %.*s\n", int(a.size()), a.data());
            return 2;
        }
    }
    if (cfg.model_dir.empty()) {
        std::fputs("serve needs --model DIR (or DEEPMOE_MODEL_DIR)\n", stderr);
        return 2;
    }

    // The protocol owns the real stdout; everything else printed to fd 1 goes
    // to stderr.
    std::fflush(stdout);
#if defined(_WIN32)
    _setmode(_fileno(stdin), _O_BINARY);
    const int proto_fd = _dup(_fileno(stdout));
    g_proto = _fdopen(proto_fd, "wb");
    _dup2(_fileno(stderr), _fileno(stdout));
#else
    const int proto_fd = dup(fileno(stdout));
    g_proto = fdopen(proto_fd, "wb");
    dup2(fileno(stderr), fileno(stdout));
#endif
    if (!g_proto) { std::fputs("cannot duplicate stdout\n", stderr); return 1; }

    const TimePoint t0 = Clock::now();
    auto tok = text::Tokenizer::load(cfg.model_dir + "/tokenizer.json");
    if (!tok) { emit_error("tokenizer: " + tok.error().str()); return 1; }
    runtime::Engine engine;
    if (auto r = engine.init(cfg); !r) { emit_error("init: " + r.error().str()); return 1; }
    if (auto r = engine.init_gpu(); !r) { emit_error("gpu init: " + r.error().str()); return 1; }
    if (auto r = engine.begin_session(sc); !r) { emit_error("session: " + r.error().str()); return 1; }
    engine.set_check_topk(check_topk);
    runtime::Session session(engine, *tok, so);
    const double load_s = std::chrono::duration<double>(Clock::now() - t0).count();
    emit(std::format("{{\"event\":\"ready\",\"load_s\":{},\"max_context\":{},\"vocab\":{},"
                     "\"cache_gb\":{},\"cache_slots\":{},\"gpu_prefill_min\":{},\"check_topk\":{}}}",
                     json_number(load_s), engine.max_context(), tok->vocab_size(),
                     json_number(engine.store().capacity_bytes() / double(1ull << 30)),
                     engine.store().slot_count(), so.gpu_prefill_min, check_topk ? "true" : "false"));

    std::string line;
    while (read_line(line)) {
        if (line.find_first_not_of(" \t") == std::string::npos) continue;
        auto doc = json_parse(line);
        if (!doc || !doc->is_object()) { emit_error("bad request: not a JSON object"); continue; }
        const std::string op = doc->string_or("op", "");
        if (op == "quit") break;
        if (op == "reset") {
            session.reset();
            emit("{\"event\":\"reset\",\"context\":0}");
            continue;
        }
        if (op == "tokenize") {
            emit("{\"event\":\"tokens\",\"ids\":" + json_uint_array(tok->encode(doc->string_or("text", ""))) + "}");
            continue;
        }
        if (op == "detokenize") {
            const JsonValue* ids = doc->find("ids");
            const std::vector<uint32_t> v = ids ? uint_array(*ids) : std::vector<uint32_t>{};
            emit("{\"event\":\"text\",\"text\":" + json_quote(tok->decode(v)) + "}");
            continue;
        }
        if (op == "status") {
            emit(std::format("{{\"event\":\"status\",\"context\":{},\"max_context\":{},\"store\":{},"
                             "\"planner\":{}}}", engine.context_length(), engine.max_context(),
                             json_quote(engine.store().stats().to_string()),
                             json_quote(engine.planner().stats().to_string())));
            continue;
        }
        if (op != "generate") { emit_error("unknown op '" + op + "'"); continue; }

        runtime::GenerateRequest req;
        if (const JsonValue* ids = doc->find("prompt_ids"); ids && ids->is_array())
            req.prompt_ids = uint_array(*ids);
        else
            req.prompt_ids = tok->encode(doc->string_or("text", ""));
        req.max_tokens = static_cast<uint32_t>(doc->int_or("max_tokens", 256));
        req.sampling.temperature = static_cast<float>(doc->double_or("temperature", 1.0));
        req.sampling.top_p = static_cast<float>(doc->double_or("top_p", 0.95));
        req.sampling.seed = static_cast<uint64_t>(doc->int_or("seed", 0));
        if (const JsonValue* s = doc->find("stop_ids"); s && s->is_array()) req.stop_ids = uint_array(*s);
        req.reuse = doc->bool_or("reuse", true);

        auto st = session.generate(
            req,
            [&](const runtime::TokenEvent& ev) {
                emit(std::format("{{\"event\":\"token\",\"id\":{},\"text\":{},\"t_ms\":{},\"step_ms\":{},"
                                 "\"p\":{},\"margin\":{},\"nucleus\":{},\"hit\":{}}}",
                                 ev.id, json_quote(ev.text), json_number(ev.t_ms),
                                 json_number(ev.step_ms), json_number(ev.p), json_number(ev.margin),
                                 ev.nucleus, json_number(ev.hit_rate)));
            },
            [&](uint32_t done, uint32_t total) {
                emit(std::format("{{\"event\":\"prefill\",\"done\":{},\"total\":{}}}", done, total));
            });
        if (!st) {
            emit_error(st.error().str());
            // A failure mid-step leaves the KV store in an unknown state.
            session.reset();
            continue;
        }
        emit("{\"event\":\"done\"," + st->json_fields() + "}");
    }
    engine.shutdown();
    return 0;
}
