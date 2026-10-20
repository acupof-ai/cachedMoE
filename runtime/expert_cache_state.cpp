#include "runtime/expert_cache_state.h"
#include "runtime/engine.h"
#include "core/log.h"
#include "model/layout.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <thread>

namespace cachedmoe::runtime {
namespace {
constexpr uint64_t kMagic = 0x31444952454d4343ull;
constexpr uint64_t kVersion = 1;
constexpr uint32_t kMaxKeys = layout::kNumLayers * layout::kRoutedExperts;
uint64_t hash_bytes(std::span<const char> bytes) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char b : bytes) { h ^= b; h *= 1099511628211ull; }
    return h;
}
uint64_t checksum(std::span<const uint32_t> keys) {
    return hash_bytes({reinterpret_cast<const char*>(keys.data()), keys.size_bytes()});
}
bool valid_keys(std::span<const ExpertKey> keys, uint32_t capacity) {
    if (keys.empty() || keys.size() > capacity || keys.size() > kMaxKeys) return false;
    std::array<bool, kMaxKeys> seen{};
    for (const auto key : keys) {
        if (key.layer >= layout::kNumLayers || key.expert >= layout::kRoutedExperts) return false;
        const auto index = key.layer * layout::kRoutedExperts + key.expert;
        if (seen[index]) return false;
        seen[index] = true;
    }
    return true;
}
}

Result<uint64_t> expert_cache_identity(const std::filesystem::path& model) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const auto root = fs::canonical(model, ec);
    if (ec) return fail(Err::Io, "expert cache checkpoint path");
    std::vector<std::string> entries;
    for (fs::directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
        const bool regular = it->is_regular_file(ec);
        if (ec) break;
        if (!regular) continue;
        const auto bytes = it->file_size(ec);
        if (ec) break;
        const auto modified = it->last_write_time(ec);
        if (ec) break;
        entries.push_back(it->path().filename().string() + ":" + std::to_string(bytes) + ":" +
                          std::to_string(modified.time_since_epoch().count()));
    }
    if (ec || entries.empty()) return fail(Err::Io, "expert cache checkpoint metadata");
    std::sort(entries.begin(), entries.end());
    std::string identity = root.string() + "\n";
    for (const auto& entry : entries) identity += entry + "\n";
    // No executable/shader identity is needed: these are IDs, not numerical state.
    return hash_bytes(identity);
}

Result<void> write_expert_cache_state(const std::filesystem::path& path, uint64_t identity,
                                     uint32_t capacity, std::span<const ExpertKey> keys) {
    if (!valid_keys(keys, capacity)) return fail(Err::InvalidArgument, "expert cache keys/capacity");
    std::vector<uint32_t> packed;
    packed.reserve(keys.size());
    for (const auto key : keys) packed.push_back(key.packed());
    const uint64_t header[] = {kMagic, kVersion, identity, capacity, packed.size(), checksum(packed)};
    std::error_code ec;
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) return fail(Err::Io, "expert cache mkdir");
    auto tmp = path; tmp += ".tmp";
    struct TempFile {
        std::filesystem::path path;
        bool owned = false;
        ~TempFile() {
            if (owned) { std::error_code ignored; std::filesystem::remove(path, ignored); }
        }
    } cleanup{tmp};
    // Exclusive creation refuses both stale files and symlinks without opening
    // their targets. Only a temporary file created by this call may be removed.
    std::ofstream f(tmp, std::ios::binary | std::ios::noreplace);
    if (!f.is_open()) return fail(Err::Io, "expert cache temporary file already exists or cannot be created");
    cleanup.owned = true;
    f.write(reinterpret_cast<const char*>(header), sizeof header);
    f.write(reinterpret_cast<const char*>(packed.data()), packed.size() * sizeof(uint32_t));
    f.flush();
    if (!f.good()) return fail(Err::Io, "expert cache write");
    f.close();
    if (f.fail()) return fail(Err::Io, "expert cache close");
    std::filesystem::rename(tmp, path, ec);
    if (ec) return fail(Err::Io, "expert cache rename");
    cleanup.owned = false;
    return {};
}

Result<std::vector<ExpertKey>> read_expert_cache_state(const std::filesystem::path& path,
                                                     uint64_t identity, uint32_t capacity) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return fail(Err::NotFound, "expert cache state unavailable");
    uint64_t header[6]{};
    if (!f.read(reinterpret_cast<char*>(header), sizeof header) ||
        header[0] != kMagic || header[1] != kVersion || header[2] != identity ||
        header[3] != capacity || header[4] == 0 || header[4] > capacity || header[4] > kMaxKeys)
        return fail(Err::Corrupt, "expert cache header/identity/capacity");
    std::vector<uint32_t> packed(header[4]);
    if (!f.read(reinterpret_cast<char*>(packed.data()), packed.size() * sizeof(uint32_t)) ||
        f.peek() != EOF || checksum(packed) != header[5])
        return fail(Err::Corrupt, "expert cache checksum/length");
    std::vector<ExpertKey> keys;
    keys.reserve(packed.size());
    for (const auto key : packed) keys.push_back(ExpertKey::unpack(key));
    if (!valid_keys(keys, capacity)) return fail(Err::Corrupt, "expert cache duplicate/range");
    return keys;
}

Result<uint32_t> Engine::restore_expert_cache(const std::string& path) {
    if (!gpu_ready_ || resident_only_ != ResidentOnly::Off || cfg_.speculation.enabled ||
        streams_.size() != 1 || store_.fixed_cache())
        return fail(Err::FailedPrecondition, "expert cache restore requires initialized off-plain engine");
    if (store_.stats().resident || store_.stats().filling)
        return fail(Err::FailedPrecondition, "expert cache restore requires empty startup cache");
    auto identity = expert_cache_identity(cfg_.model_dir);
    if (!identity) return std::unexpected(identity.error());
    auto keys = read_expert_cache_state(path, *identity, store_.slot_count());
    if (!keys) return std::unexpected(keys.error());
    uint64_t expected_bytes = 0;
    for (const auto key : *keys) {
        auto entry = manifest_.require_expert(key);
        if (!entry) return std::unexpected(entry.error());
        for (const auto& run : (*entry)->runs) expected_bytes += run.aligned_bytes;
    }
    const auto start = Clock::now();
    const uint64_t before_bytes = io_.stats().bytes_completed;
    // These are known former residents, not predicted routes or pinned experts.
    // P3 rank stamps stay below every new demand stamp; normal LRU resumes at READY.
    if (auto r = planner_.start_backfill(*keys, 8); !r) return std::unexpected(r.error());
    while (planner_.backfill_active() && Clock::now() - start < std::chrono::seconds(120))
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    planner_.stop_backfill();
    io_.drain();
    uint32_t restored = 0;
    for (const auto key : *keys) restored += store_.resident(key);
    const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
    log_info("expert-cache: restored {}/{} resident IDs in {:.3f} s before READY; "
             "original checkpoint bytes completed {}, expected {}",
             restored, keys->size(), seconds, io_.stats().bytes_completed - before_bytes, expected_bytes);
    if (restored != keys->size()) return fail(Err::Io, "expert cache restore incomplete; demand loading remains active");
    expert_cache_restored_ = true;
    return restored;
}

Result<void> Engine::save_expert_cache(const std::string& path) {
    if (!gpu_ready_ || !produce_ced_ || resident_only_ != ResidentOnly::Off || cfg_.speculation.enabled ||
        streams_.size() != 1 || store_.fixed_cache())
        return fail(Err::FailedPrecondition, "expert cache save requires initialized off-plain session");
    // This optional state must never write beneath a checkpoint or its mirror.
    std::error_code ec;
    const auto destination = std::filesystem::weakly_canonical(path, ec);
    if (ec) return fail(Err::Io, "expert cache destination path");
    auto roots = cfg_.model_mirrors;
    roots.push_back(cfg_.model_dir);
    for (const auto& root : roots) {
        const auto checkpoint = std::filesystem::canonical(root, ec);
        if (ec) return fail(Err::Io, "expert cache checkpoint path");
        const auto relative = destination.lexically_relative(checkpoint);
        if (!relative.empty() && *relative.begin() != "..")
            return fail(Err::InvalidArgument, "expert cache state cannot be written under a checkpoint");
    }
    auto identity = expert_cache_identity(cfg_.model_dir);
    if (!identity) return std::unexpected(identity.error());
    planner_.stop_backfill();
    io_.drain();
    std::vector<store::ExpertSlot> slots;
    for (uint32_t i = 0; i < store_.slot_count(); ++i) {
        auto slot = store_.slot_info(i);
        if (slot && slot->state == SlotState::Resident && slot->tier != Tier::Pinned)
            slots.push_back(*slot);
    }
    if (slots.empty()) {
        log_info("expert-cache: no resident IDs; preserving previous snapshot");
        return {};
    }
    std::sort(slots.begin(), slots.end(), [](const auto& a, const auto& b) {
        if (a.last_use_token != b.last_use_token) return a.last_use_token > b.last_use_token;
        return a.slot > b.slot;
    });
    std::vector<ExpertKey> keys;
    for (const auto& slot : slots) keys.push_back(slot.key);
    auto r = write_expert_cache_state(path, *identity, store_.slot_count(), keys);
    if (r) log_info("expert-cache: saved {} resident IDs (metadata only) to {}", keys.size(), path);
    return r;
}

} // namespace cachedmoe::runtime
