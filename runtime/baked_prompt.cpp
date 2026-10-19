#include "runtime/baked_prompt.h"
#include "runtime/session.h"
#include "core/log.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <format>
#include <type_traits>

namespace cachedmoe::runtime {
namespace {
double ms_since(TimePoint start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }
constexpr uint64_t limit = 64ull << 20;
uint64_t hash(std::span<const char> bytes) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char b : bytes) { h ^= b; h *= 1099511628211ull; }
    return h;
}
struct Codec {
    std::vector<char> data;
    size_t at = 0;
    bool valid = true, reading = false;
    template<class T> void scalar(T& v) {
        static_assert(std::is_trivially_copyable_v<T>);
        if (reading) {
            if (sizeof(T) > data.size() - at) { valid = false; return; }
            std::memcpy(&v, data.data() + at, sizeof(T)); at += sizeof(T);
        } else {
            const auto* p = reinterpret_cast<const char*>(&v);
            data.insert(data.end(), p, p + sizeof(T));
        }
    }
    template<class T> void vec(std::vector<T>& v) {
        uint64_t n = v.size(); scalar(n);
        if (!valid || n > limit / sizeof(T)) { valid = false; return; }
        if (reading) {
            if (n * sizeof(T) > data.size() - at) { valid = false; return; }
            v.resize(n);
            if (n) std::memcpy(v.data(), data.data() + at, n * sizeof(T));
            at += n * sizeof(T);
        } else {
            const auto* p = reinterpret_cast<const char*>(v.data());
            if (n) data.insert(data.end(), p, p + n * sizeof(T));
        }
    }
    void state(BakedPrompt& p) {
        vec(p.tokens);
        scalar(p.first.token); scalar(p.first.position); scalar(p.first.greedy_token);
        scalar(p.first.top1); scalar(p.first.top2); scalar(p.first.candidates);
        scalar(p.first.nucleus_size); scalar(p.first.p_token); scalar(p.first.retained); scalar(p.first.kept);
        scalar(p.rows.first_pos); scalar(p.rows.end_pos);
        uint64_t n = p.rows.planes.size(); scalar(n);
        if (!valid || n > 64) { valid = false; return; }
        if (reading) p.rows.planes.resize(n);
        for (auto& pl : p.rows.planes) {
            scalar(pl.layer); scalar(pl.ratio); scalar(pl.first_row); scalar(pl.rows);
            vec(pl.cmp); vec(pl.key); vec(pl.carry_kv); vec(pl.carry_score);
        }
        scalar(p.ring.latent_dim); vec(p.ring.layer); vec(p.ring.slot);
        vec(p.ring.position); vec(p.ring.val); vec(p.ring.scale);
    }
};
}
Result<void> write_baked_prompt(const std::filesystem::path& path, uint64_t identity, const BakedPrompt& state) {
    Codec c;
    // Codec never changes fields in write mode.
    c.state(const_cast<BakedPrompt&>(state));
    if (!c.valid || c.data.size() > limit) return fail(Err::InvalidArgument, "baked prompt too large");
    const uint64_t header[] = {0x31444e554f424d43ull, identity, c.data.size(), hash(c.data)};
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) return fail(Err::Io, "baked prompt mkdir");
    auto tmp = path; tmp += ".tmp";
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(header), sizeof header);
    f.write(c.data.data(), c.data.size()); f.flush();
    if (!f.good()) return fail(Err::Io, "baked prompt write");
    f.close();
    std::filesystem::rename(tmp, path, ec);
    if (ec) return fail(Err::Io, "baked prompt rename");
    return {};
}
Result<BakedPrompt> read_baked_prompt(const std::filesystem::path& path, uint64_t identity) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return fail(Err::NotFound, "baked prompt miss");
    uint64_t header[4]{};
    if (!f.read(reinterpret_cast<char*>(header), sizeof header) || header[0] != 0x31444e554f424d43ull ||
        header[1] != identity || header[2] > limit)
        return fail(Err::Corrupt, "baked prompt identity/header");
    Codec c; c.reading = true; c.data.resize(header[2]);
    if (!f.read(c.data.data(), c.data.size()) || f.peek() != EOF || hash(c.data) != header[3])
        return fail(Err::Corrupt, "baked prompt checksum/length");
    BakedPrompt p; c.state(p);
    if (!c.valid || c.at != c.data.size() || p.tokens.empty() || p.tokens.size() > 8192)
        return fail(Err::Corrupt, "baked prompt payload");
    return p;
}

void Engine::configure_baked_prompts(const std::string& dir, uint64_t budget) {
    baked_dir_ = dir.empty() ? "" : (std::filesystem::path(dir) / "boundaries").string();
    baked_budget_ = budget;
    std::string identity = std::string(__DATE__) + __TIME__ + cfg_.model_dir + startup_decode_mode();
    for (size_t i = 0; i < configuration::kKeyNames.size(); ++i) {
        // Receipt sinks cannot change arithmetic or invalidate a bake.
        const auto key = static_cast<configuration::Key>(i);
        if (key == configuration::Key::PF_OPS_JSON || key == configuration::Key::SPEC_DIAGNOSTICS ||
            key == configuration::Key::DSPARK_PROFILE || key == configuration::Key::DSPARK_MEGA_DIAG) continue;
        const auto& raw = cfg_.environment->raw[static_cast<configuration::Key>(i)];
        identity += std::string(configuration::kKeyNames[i]) + "=" + (raw.text ? *raw.text : "<absent>") + "\n";
    }
    // Invalidate on checkpoint/shader replacement, without reading or changing weights.
    for (const auto& dirpath : {cfg_.model_dir, cfg_.environment->shader_dir}) {
        std::error_code ec;
        std::vector<std::string> entries;
        for (std::filesystem::directory_iterator it(dirpath, ec), end; !ec && it != end; it.increment(ec)) {
            if (!it->is_regular_file(ec)) continue;
            entries.push_back(it->path().filename().string() + ":" + std::to_string(it->file_size(ec)) + ":" +
                              std::to_string(it->last_write_time(ec).time_since_epoch().count()));
        }
        std::sort(entries.begin(), entries.end());
        for (const auto& s : entries) identity += s + "\n";
    }
#if defined(__linux__)
    // Any relink invalidates boundaries, including edits in other translation units.
    std::error_code exe_ec;
    const auto exe = std::filesystem::read_symlink("/proc/self/exe", exe_ec);
    if (exe_ec) { baked_dir_.clear(); return; }
    identity += std::to_string(std::filesystem::file_size(exe, exe_ec)) + ":" +
                std::to_string(std::filesystem::last_write_time(exe, exe_ec).time_since_epoch().count());
    if (exe_ec) { baked_dir_.clear(); return; }
#endif
    baked_identity_ = hash(identity);
}
std::string Engine::baked_path(std::span<const uint32_t> tokens) const {
    const auto bytes = std::span(reinterpret_cast<const char*>(tokens.data()), tokens.size_bytes());
    return (std::filesystem::path(baked_dir_) / std::format("{:016x}-{:016x}.bkv", baked_identity_, hash(bytes))).string();
}
bool Engine::has_baked_prompt(std::span<const uint32_t> tokens, const SamplingParams& sampling) const {
    if (baked_dir_.empty() || !sampling.greedy() || tokens.size() < 512 || tokens.size() > 8192 ||
        resident_only_ != ResidentOnly::Off || cfg_.speculation.enabled || streams_.size() != 1) return false;
    std::error_code ec;
    return std::filesystem::is_regular_file(baked_path(tokens), ec);
}
Result<void> Engine::bake_prompt(const DecodeStepResult& first) {
    if (baked_dir_.empty() || !sampling_.greedy() || context_length() < 512 || context_length() > 8192 ||
        resident_only_ != ResidentOnly::Off || cfg_.speculation.enabled || streams_.size() != 1) return {};
    const auto path = baked_path(cur_->history_);
    std::error_code ec;
    if (std::filesystem::exists(path, ec)) return {};
    const auto start = Clock::now();
    BakedPrompt p; p.tokens = cur_->history_; p.first = first;
    auto rows = cur_->kvs_.backup_rows(0, context_length());
    if (!rows) return std::unexpected(rows.error());
    p.rows = std::move(*rows);
    std::vector<uint32_t> slots(cur_->kvs_.config().window), layers(cur_->kvs_.config().layers);
    for (uint32_t i = 0; i < slots.size(); ++i) slots[i] = i;
    for (uint32_t i = 0; i < layers.size(); ++i) layers[i] = i;
    auto ring = cur_->kvs_.snapshot_ring(slots, layers);
    if (!ring) return std::unexpected(ring.error());
    p.ring = std::move(*ring);
    if (auto r = write_baked_prompt(path, baked_identity_, p); !r) return r;
    log_info("baked-prefix: saved {} tokens, {} bytes in {:.3f} ms ({})", p.tokens.size(),
             std::filesystem::file_size(path, ec), ms_since(start), path);
    enforce_kv_disk_budget({std::filesystem::path(baked_dir_).parent_path().string(), "", baked_budget_});
    return {};
}
Result<std::optional<DecodeStepResult>> Engine::restore_baked_prompt(std::span<const uint32_t> tokens, const SamplingParams& sampling) {
    if (baked_dir_.empty() || !sampling.greedy() || tokens.size() < 512 || tokens.size() > 8192 ||
        resident_only_ != ResidentOnly::Off || cfg_.speculation.enabled || streams_.size() != 1)
        return std::optional<DecodeStepResult>{};
    const auto start = Clock::now();
    auto p = read_baked_prompt(baked_path(tokens), baked_identity_);
    if (!p) {
        if (p.error().code != Err::NotFound) {
            log_warn("baked-prefix: {}", p.error().str());
            std::error_code ec;
            std::filesystem::remove(baked_path(tokens), ec);
        }
        return std::optional<DecodeStepResult>{};
    }
    if (p->tokens.size() != tokens.size() || !std::equal(tokens.begin(), tokens.end(), p->tokens.begin()))
        return std::optional<DecodeStepResult>{};
    // Validate every shape before touching live state. No persisted pointers or GPU handles.
    const auto& c = cur_->kvs_.config();
    if (p->rows.first_pos != 0 || p->rows.end_pos != tokens.size() ||
        p->ring.latent_dim != c.latent_dim || p->ring.layer.size() != c.layers ||
        p->ring.slot.size() != c.window || p->ring.position.size() != c.window ||
        p->ring.val.size() != size_t(c.layers) * c.window * c.latent_dim ||
        p->ring.scale.size() != size_t(c.layers) * c.window * c.window_scales() ||
        p->first.position + 1 != tokens.size() || p->first.token >= model_cfg_.text.vocab_size)
        return fail(Err::Corrupt, "baked prompt shapes");
    for (uint32_t i = 0; i < c.layers; ++i) if (p->ring.layer[i] != i) return fail(Err::Corrupt, "baked layers");
    for (uint32_t i = 0; i < c.window; ++i) if (p->ring.slot[i] != i) return fail(Err::Corrupt, "baked slots");
    reset_context();
    if (auto r = cur_->kvs_.restore_exact_rows(p->rows, tokens.size()); !r) return std::unexpected(r.error());
    if (auto r = cur_->kvs_.restore_ring(p->ring); !r) return std::unexpected(r.error());
    if (auto r = set_context_tokens(tokens); !r) return std::unexpected(r.error());
    p->first.wall_ms = ms_since(start);
    log_info("baked-prefix: restored exact {} tokens in {:.3f} ms", tokens.size(), p->first.wall_ms);
    return std::optional<DecodeStepResult>{p->first};
}
}
