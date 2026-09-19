// `deepmoe serve`: one long-running Engine behind line-delimited JSON on
// stdin/stdout (Track P, docs/p3_chat.md §1; Track R2, docs/p4_kv_ux.md §4).
//
// Requests, one JSON object per line:
//   {"op":"generate", "prompt_ids":[...] | "text":"...", "max_tokens":N,
//    "temperature":T, "top_p":P, "seed":S, "stop_ids":[...], "reuse":true,
//    "session":"name"}
//   {"op":"cancel"}                      stop every generate read so far, between
//                                        tokens; the KV state stays consistent
//   {"op":"reset", "session":"name"}     drop that session's KV state (the caches stay warm)
//   {"op":"sessions"}                    -> {"event":"sessions","live":..,"sessions":[...]}
//   {"op":"drop", "session":"name"}      forget a session
//   {"op":"tokenize", "text":"..."}      -> {"event":"tokens","ids":[...]}
//   {"op":"detokenize", "ids":[...]}     -> {"event":"text","text":"..."}
//   {"op":"status"}                      -> {"event":"status",...}
//   {"op":"quit"}
// Events, one JSON object per line on stdout:
//   {"event":"ready",...}  once, after the pinned set and the cache are up
//   {"event":"session","name":..,"tokens":..,"replay_steps":..,"replay_ms":..}  on a switch
//   {"event":"prefill","done":i,"total":n}
//   {"event":"token","id":..,"text":"..","t_ms":..,"step_ms":..,"p":..,"margin":..,"hit":..}
//   {"event":"done","session":.., <GenerateStats::json_fields>}   finish: stop|length|context|cancel
//   {"event":"cancel","upto":n}          acknowledges a cancel (from the reader thread)
//   {"event":"error","message":".."}
// `temperature` <= 0 is greedy; the defaults are the model README's 1.0 / 0.95.
// Everything the engine logs goes to stderr: fd 1 is re-pointed at fd 2 and the
// protocol writes to a duplicate of the original stdout.
//
// Sessions: one is live in the KV store, the rest are parked -- only their
// non-SWA state, packed, plus token ids; the window ring is rebuilt by replaying
// <= 128 tokens when one comes back (runtime/session.h). The expert cache is shared.
//
// Ownership/threading: a reader thread owns stdin -- it answers `cancel` at once
// and queues everything else -- and the main thread runs the engine. `emit` is
// serialised by a mutex.
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <format>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
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
#include "model/layout.h"
#include "runtime/engine.h"
#include "runtime/session.h"
#include "text/tokenizer.h"

using namespace deepmoe;

namespace {

std::FILE* g_proto = nullptr;
std::mutex g_emit_mu;

void emit(const std::string& line) {
    std::lock_guard<std::mutex> lk(g_emit_mu);
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

// stdin, read on its own thread so a cancel reaches a running generate.
struct Inbox {
    struct Item { std::string line; uint64_t seq = 0; bool eof = false; };
    std::mutex mu;
    std::condition_variable cv;
    std::deque<Item> q;
    std::atomic<uint64_t> generates_read{0};   // generate requests enqueued so far
    std::atomic<uint64_t> cancel_upto{0};      // generates with seq <= this are cancelled

    void run() {
        std::string line;
        while (read_line(line)) {
            if (line.find_first_not_of(" \t") == std::string::npos) continue;
            Item it;
            it.line = line;
            auto doc = json_parse(line);
            const std::string op = (doc && doc->is_object()) ? doc->string_or("op", "") : "";
            if (op == "cancel") {
                const uint64_t upto = generates_read.load();
                cancel_upto.store(upto);
                emit(std::format("{{\"event\":\"cancel\",\"upto\":{}}}", upto));
                continue;
            }
            if (op == "generate") it.seq = generates_read.fetch_add(1) + 1;
            push(std::move(it));
            if (op == "quit") break;
        }
        Item end;
        end.eof = true;
        push(std::move(end));
    }
    void push(Item it) {
        {
            std::lock_guard<std::mutex> lk(mu);
            q.push_back(std::move(it));
        }
        cv.notify_one();
    }
    Item pop() {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !q.empty(); });
        Item it = std::move(q.front());
        q.pop_front();
        return it;
    }
};

}  // namespace

int cmd_serve(int argc, char** argv) {
    RuntimeConfig cfg;
    cfg.cache.budget_bytes = 0;
    if (const char* e = std::getenv("DEEPMOE_MODEL_DIR")) cfg.model_dir = e;
    runtime::SessionConfig sc;
    runtime::SessionOptions so;
    runtime::SessionPoolOptions po;
    bool check_topk = false;
    bool engine_reheat = false;
    std::string resident_only;   // Track Y: docs/p4_resident_routing.md
    uint32_t    streams = 1;     // Track MS: docs/p4_multistream.md
    bool        warm_cache = false;
    std::string ms_sched;
    bool kv_disk_off = false;
    bool kv_dir_given = false;
    for (int i = 2; i < argc; ++i) {
        const std::string_view a = argv[i];
        if (a == "--model")                cfg.model_dir = value_of(argc, argv, i);
        else if (a == "--mirror")          cfg.model_mirrors.emplace_back(value_of(argc, argv, i));
        else if (a == "--cache-gb")        cfg.cache.budget_bytes = uint64_t(std::atoll(value_of(argc, argv, i).c_str())) << 30;
        else if (a == "--cache-slots")     cfg.cache.budget_bytes = uint64_t(std::atoll(value_of(argc, argv, i).c_str())) * layout::kExpertSlotBytes;
        else if (a == "--max-context")     sc.max_context = uint32_t(std::atoi(value_of(argc, argv, i).c_str()));
        else if (a == "--engram-tables")   sc.engram_tables_dir = value_of(argc, argv, i);
        else if (a == "--gpu-prefill-min") so.gpu_prefill_min = uint32_t(std::atoi(value_of(argc, argv, i).c_str()));
        else if (a == "--replay")          so.replay = uint32_t(std::atoi(value_of(argc, argv, i).c_str()));
        else if (a == "--no-rollback")     so.rollback = false;
        else if (a == "--reheat")          { so.reheat = true; engine_reheat = true; }
        else if (a == "--reheat-decay")    { so.reheat_decay = static_cast<float>(std::atof(value_of(argc, argv, i).c_str())); engine_reheat = true; }
        else if (a == "--max-parked")      po.max_parked = uint32_t(std::atoi(value_of(argc, argv, i).c_str()));
        else if (a == "--park-budget-mb")  po.max_parked_bytes = uint64_t(std::atoll(value_of(argc, argv, i).c_str())) << 20;
        else if (a == "--kv-dir")          { po.disk.dir = value_of(argc, argv, i); kv_dir_given = true; }
        else if (a == "--kv-max-gb")       po.disk.max_bytes = uint64_t(std::atoll(value_of(argc, argv, i).c_str())) << 30;
        else if (a == "--no-kv-disk")      kv_disk_off = true;
        else if (a == "--profile")         cfg.profile_jsonl = value_of(argc, argv, i);
        else if (a == "--trace")           cfg.trace_file = value_of(argc, argv, i);
        else if (a == "--check-topk")      check_topk = true;
        else if (a == "--resident-only")   resident_only = value_of(argc, argv, i);
        // Track MS (docs/p4_multistream.md): decode streams inside this one
        // engine process, and how a multi-stream round is scheduled.
        else if (a == "--streams")         streams = uint32_t(std::atoi(value_of(argc, argv, i).c_str()));
        else if (a == "--ms-sched")        ms_sched = value_of(argc, argv, i);
        // The `run` path's --warm-cache: block until the P3 backfill has filled
        // every free slot from the static heat order, so a measurement starts
        // from a full cache instead of racing the fill.
        else if (a == "--warm-cache")      warm_cache = true;
        else {
            std::fprintf(stderr, "unknown option %.*s\n", int(a.size()), a.data());
            return 2;
        }
    }
    if (cfg.model_dir.empty()) {
        std::fputs("serve needs --model DIR (or DEEPMOE_MODEL_DIR)\n", stderr);
        return 2;
    }
    // Track R2's disk prefix cache, ON by default (docs/p4_kv_ux.md §8): a fresh
    // process that comes back to the same conversation rebuilt its whole prompt
    // every time -- 4,133 tokens at ~24 ms each, 101.6 s -- and the parked
    // context is already being written at shutdown, so the only thing missing
    // was the default. `DEEPMOE_KV_DIR` overrides the directory (empty string
    // disables it), `--kv-dir` sets it explicitly and `--no-kv-disk` turns it
    // off, which is what a benchmark that wants a cold prefill should use.
    if (!kv_dir_given && !kv_disk_off) {
        if (const char* e = std::getenv("DEEPMOE_KV_DIR"); e && *e) po.disk.dir = e;
        else if (e && !*e) po.disk.dir.clear();
        else {
            // Per model directory, so two checkpoints do not fight over one file
            // and a stale one is a miss rather than a corrupt hit (the header
            // carries the model tag as well).
            std::string base;
#if defined(_WIN32)
            if (const char* la = std::getenv("LOCALAPPDATA"); la && *la) base = la;
            else if (const char* tp = std::getenv("TEMP"); tp && *tp) base = tp;
#else
            if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg) base = xdg;
            else if (const char* home = std::getenv("HOME"); home && *home) base = std::string(home) + "/.cache";
#endif
            if (!base.empty()) {
                std::string tag = cfg.model_dir;
                for (char& ch : tag) if (ch == '\\' || ch == '/' || ch == ':') ch = '_';
                po.disk.dir = base + "/deepmoe/kv/" + tag;
            }
        }
    }
    // The disk cache stores the model identity in every file; a different
    // checkpoint is a miss, not a corrupt hit.
    if (!po.disk.dir.empty()) po.disk.model_tag = cfg.model_dir;

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
    engine.set_reheat(engine_reheat);
    // Track Y. The flag wins over DEEPMOE_ROUTE_RESIDENT_ONLY, which the load read.
    if (!resident_only.empty()) {
        if (resident_only == "all")      engine.set_resident_only(runtime::Engine::ResidentOnly::All);
        else if (resident_only == "stall1") engine.set_resident_only(runtime::Engine::ResidentOnly::Stall1);
        else if (resident_only == "verify") engine.set_resident_only(runtime::Engine::ResidentOnly::Verify);
        else if (resident_only == "off") engine.set_resident_only(runtime::Engine::ResidentOnly::Off);
        else { std::fprintf(stderr, "--resident-only takes off|all|stall1|verify, got '%s'\n", resident_only.c_str()); return 1; }
    }
    // Track MS: the extra streams, and the session each of them starts from.
    // They are created BEFORE begin_session so every one of them gets its own
    // KV store and engram planes from it.
    if (streams > 1) {
        if (auto r = engine.set_streams(streams); !r) {
            emit_error("streams: " + r.error().str());
            return 1;
        }
        if (auto r = engine.begin_session(sc); !r) {
            emit_error("session: " + r.error().str());
            return 1;
        }
        if (ms_sched == "pingpong") engine.set_ms_sched(runtime::Engine::MsSched::PingPong);
        else if (ms_sched == "interleave") engine.set_ms_sched(runtime::Engine::MsSched::Interleave);
        else if (ms_sched == "pipeline") engine.set_ms_sched(runtime::Engine::MsSched::Pipeline);
        else if (!ms_sched.empty()) {
            std::fprintf(stderr, "--ms-sched takes pipeline, interleave or pingpong, got '%s'\n",
                         ms_sched.c_str());
            return 1;
        }
    }
    if (warm_cache) {
        const TimePoint w0 = Clock::now();
        auto w = engine.warm_cache_from_heat();
        if (!w) log_warn("serve: warm-cache: {}", w.error().str());
        else log_info("serve: warm-cache filled {} slots in {:.1f} s", *w,
                      std::chrono::duration<double>(Clock::now() - w0).count());
    }
    runtime::SessionPool pool(engine, *tok, so, po);
    if (!po.disk.dir.empty()) {
        auto loaded = pool.restore_active_from_disk();
        // H1b: a refused restore is a degradation, not a failure -- the pool has
        // already reset the context, so the first turn simply prefills.
        if (loaded && loaded->cold_fallback)
            std::fprintf(stderr,
                         "serve: kv-disk restore declined (%s); starting cold, the first "
                         "turn prefills the whole prompt\n",
                         loaded->cold_reason.c_str());
        else if (loaded) std::fprintf(stderr, "serve: restored %u tokens from kv disk\n",
                                      engine.context_length());
        else if (loaded.error().code != Err::NotFound)
            std::fprintf(stderr, "serve: kv-disk restore failed: %s\n",
                         loaded.error().str().c_str());
    }
    const double load_s = std::chrono::duration<double>(Clock::now() - t0).count();
    emit(std::format("{{\"event\":\"ready\",\"load_s\":{},\"max_context\":{},\"vocab\":{},"
                     "\"cache_gb\":{},\"cache_slots\":{},\"gpu_prefill_min\":{},\"check_topk\":{},"
                     "\"engram_tables\":{},\"kv_mb\":{},\"rollback\":{},\"max_parked\":{},"
                     "\"reheat\":{},\"reheat_decay\":{},\"kv_disk\":{},\"kv_disk_dir\":{}}}",
                     json_number(load_s), engine.max_context(), tok->vocab_size(),
                     json_number(engine.store().capacity_bytes() / double(1ull << 30)),
                     engine.store().slot_count(), so.gpu_prefill_min, check_topk ? "true" : "false",
                     json_quote(sc.engram_tables_dir.empty() ? std::string("derived") : sc.engram_tables_dir),
                     json_number(engine.kv().bytes() / 1e6), so.rollback ? "true" : "false",
                     po.max_parked, so.reheat ? "true" : "false", json_number(so.reheat_decay),
                     po.disk.dir.empty() ? "false" : "true", json_quote(po.disk.dir)));

    Inbox inbox;
    std::thread reader([&] { inbox.run(); });

    auto switch_to = [&](const std::string& name) -> bool {
        if (name == pool.active()) return true;
        auto r = pool.activate(name);
        if (!r) {
            emit_error("session '" + name + "': " + r.error().str());
            return false;
        }
        if (r->cold_fallback)
            std::fprintf(stderr, "serve: session '%s' restore declined (%s); starting cold\n",
                         name.c_str(), r->cold_reason.c_str());
        emit(std::format("{{\"event\":\"session\",\"name\":{},\"tokens\":{},\"replay_steps\":{},"
                         "\"replay_ms\":{},\"evicted\":{},\"cold_fallback\":{}}}",
                         json_quote(name), engine.context_length(), r->plan.steps(),
                         json_number(r->ms), pool.evicted(),
                         r->cold_fallback ? "true" : "false"));
        return true;
    };

    for (;;) {
        Inbox::Item item = inbox.pop();
        if (item.eof) break;
        const std::string& line = item.line;
        auto doc = json_parse(line);
        if (!doc || !doc->is_object()) { emit_error("bad request: not a JSON object"); continue; }
        const std::string op = doc->string_or("op", "");
        const std::string session = doc->string_or("session", pool.active());
        if (op == "quit") break;
        if (op == "reset") {
            if (!switch_to(session)) continue;
            pool.reset_live();
            emit(std::format("{{\"event\":\"reset\",\"session\":{},\"context\":0}}", json_quote(session)));
            continue;
        }
        if (op == "sessions") {
            std::string arr;
            for (const auto& s : pool.list()) {
                if (!arr.empty()) arr += ",";
                arr += std::format("{{\"name\":{},\"tokens\":{},\"parked_mb\":{},\"live\":{}}}",
                                   json_quote(s.name), s.tokens, json_number(s.bytes / 1e6),
                                   s.live ? "true" : "false");
            }
            emit(std::format("{{\"event\":\"sessions\",\"live\":{},\"evicted\":{},\"sessions\":[{}]}}",
                             json_quote(pool.active()), pool.evicted(), arr));
            continue;
        }
        if (op == "drop") {
            const bool had = pool.drop(session);
            emit(std::format("{{\"event\":\"drop\",\"session\":{},\"existed\":{}}}", json_quote(session),
                             had ? "true" : "false"));
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
            emit(std::format("{{\"event\":\"status\",\"session\":{},\"context\":{},\"max_context\":{},"
                             "\"kv_mb\":{},\"kv_capacity\":{},\"kv_slabs\":{},\"kv_largest_slab_mb\":{},"
                             "\"store\":{},\"planner\":{},\"io\":{},\"route\":{}}}",
                             json_quote(pool.active()), engine.context_length(), engine.max_context(),
                             json_number(engine.kv().bytes() / 1e6), engine.kv().capacity(),
                             engine.kv().slabs(), json_number(engine.kv().largest_slab() / 1e6),
                             json_quote(engine.store().stats().to_string()),
                             json_quote(engine.planner().stats().to_string()),
                             json_quote(engine.io().stats().to_string()),
                             json_quote(engine.resident_route_report())));
            continue;
        }
        // Track R1 round 2 (docs/p4_hitrate.md §7): the same pass the turn
        // boundary runs, on demand. `decay` defaults to the session's.
        if (op == "reheat") {
            float decay = so.reheat_decay;
            if (const JsonValue* d = doc->find("decay"); d && d->as_double())
                decay = static_cast<float>(*d->as_double());
            auto h = engine.reheat(decay);
            if (!h) { emit_error("reheat: " + h.error().str()); continue; }
            emit(std::format("{{\"event\":\"reheat\",\"turn\":{},\"slots\":{},\"warm\":{},"
                             "\"evicted\":{},\"keys\":{},\"free_slots\":{},\"decay\":{},\"ms\":{}}}",
                             h->turn, h->slots, h->warm, h->evicted, h->passed, h->free_slots,
                             json_number(h->decay), json_number(h->ms)));
            continue;
        }
        // Track MS (docs/p4_multistream.md): N turns at once, one per engine
        // stream, layer-interleaved. Sessions are the STREAMS here, not the
        // parking pool: each stream keeps its own KV across turns, which is
        // what two concurrent conversations are.
        if (op == "generate_multi") {
            const JsonValue* rs = doc->find("requests");
            if (!rs || !rs->is_array()) { emit_error("generate_multi needs \"requests\":[...]"); continue; }
            std::vector<runtime::MultiTurn> turns;
            bool bad = false;
            uint32_t idx = 0;
            for (const JsonValue& rv : **rs->as_array()) {
                if (!rv.is_object()) { bad = true; break; }
                runtime::MultiTurn t;
                t.stream = static_cast<uint32_t>(rv.int_or("stream", idx));
                if (const JsonValue* ids = rv.find("prompt_ids"); ids && ids->is_array())
                    t.req.prompt_ids = uint_array(*ids);
                else
                    t.req.prompt_ids = tok->encode(rv.string_or("text", ""));
                t.req.max_tokens = static_cast<uint32_t>(rv.int_or("max_tokens", 256));
                t.req.sampling.temperature = static_cast<float>(rv.double_or("temperature", 1.0));
                t.req.sampling.top_p = static_cast<float>(rv.double_or("top_p", 0.95));
                t.req.sampling.seed = static_cast<uint64_t>(rv.int_or("seed", 0));
                if (const JsonValue* sp = rv.find("stop_ids"); sp && sp->is_array())
                    t.req.stop_ids = uint_array(*sp);
                t.req.reuse = rv.bool_or("reuse", true);
                turns.push_back(std::move(t));
                ++idx;
            }
            if (bad || turns.empty()) { emit_error("generate_multi: bad requests"); continue; }
            auto r = runtime::generate_multi(
                engine, *tok, so, turns,
                [&](uint32_t i, const runtime::TokenEvent& ev) {
                    const uint32_t sid = turns[i].stream;
                    emit(std::format("{{\"event\":\"token\",\"stream\":{},\"id\":{},\"text\":{},"
                                     "\"t_ms\":{},\"step_ms\":{},\"p\":{},\"margin\":{},"
                                     "\"nucleus\":{},\"hit\":{}}}",
                                     sid, ev.id, json_quote(ev.text), json_number(ev.t_ms),
                                     json_number(ev.step_ms), json_number(ev.p),
                                     json_number(ev.margin), ev.nucleus, json_number(ev.hit_rate)));
                },
                [&](uint32_t i, uint32_t done, uint32_t total) {
                    emit(std::format("{{\"event\":\"prefill\",\"stream\":{},\"done\":{},\"total\":{}}}",
                                     turns[i].stream, done, total));
                });
            if (!r) { emit_error(r.error().str()); continue; }
            for (size_t i = 0; i < r->turns.size(); ++i)
                emit(std::format("{{\"event\":\"done\",\"stream\":{},", turns[i].stream) +
                     r->turns[i].json_fields() + "}");
            emit(std::format("{{\"event\":\"done_multi\",\"streams\":{},\"decode_ms\":{},"
                             "\"decode_steps\":{},\"rounds\":{},\"full_rounds\":{},"
                             "\"aggregate_tok_s\":{}}}",
                             r->turns.size(), json_number(r->decode_ms), r->decode_steps,
                             r->rounds, r->full_rounds, json_number(r->aggregate_tok_s())));
            continue;
        }
        if (op != "generate") { emit_error("unknown op '" + op + "'"); continue; }

        const uint64_t seq = item.seq;
        auto cancelled = [&inbox, seq] { return inbox.cancel_upto.load() >= seq; };
        if (cancelled()) {
            runtime::GenerateStats gs;
            gs.finish = "cancel";
            gs.context_after = engine.context_length();
            emit("{\"event\":\"done\",\"session\":" + json_quote(session) + "," + gs.json_fields() + "}");
            continue;
        }
        if (!switch_to(session)) continue;

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
        req.cancel = cancelled;

        auto st = pool.live().generate(
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
            pool.reset_live();
            continue;
        }
        emit("{\"event\":\"done\",\"session\":" + json_quote(session) + "," + st->json_fields() + "}");
    }
    if (!po.disk.dir.empty()) {
        if (auto pr = pool.park_active(); !pr)
            std::fprintf(stderr, "serve: kv-disk save failed: %s\n", pr.error().str().c_str());
    }
    // The reader may still be blocked on stdin; it owns nothing the engine needs.
    reader.detach();
    engine.shutdown();
    return 0;
}
