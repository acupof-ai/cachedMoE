#pragma once

#include "core/namespace.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_set>

namespace cachedmoe::environment {

enum class Source { Absent, Native, Canonical, Legacy };

struct Value {
    const char* value = nullptr;
    Source source = Source::Absent;
    bool present() const { return value != nullptr; }
};

struct Names {
    std::string canonical;
    std::string legacy;
    bool project = false;
};

inline Names names(std::string_view key) {
    constexpr std::string_view canonical = "CACHEDMOE_";
    constexpr std::string_view legacy = "DEEPMOE_";
    if (key.starts_with(canonical))
        return {std::string(key), std::string(legacy) + std::string(key.substr(canonical.size())), true};
    if (key.starts_with(legacy))
        return {std::string(canonical) + std::string(key.substr(legacy.size())), std::string(key), true};
    return {std::string(key), {}, false};
}

// Raw reads are only for alias resolution and exact environment snapshots.
// Returned pointers have getenv's lifetime; nothing is cached across mutations.
inline const char* raw_get(const char* key) { return std::getenv(key); }

namespace detail {

struct WarningState {
    std::mutex mutex;
    std::unordered_set<std::string> emitted;
};

// An external-linkage inline function has one shared local static across TUs.
inline WarningState& warnings() {
    static WarningState state;
    return state;
}

inline void warn_once(const Names& key, bool conflict) {
    auto& state = warnings();
    std::lock_guard lock(state.mutex);
    const std::string reason = conflict ? "conflict: canonical wins" : "legacy fallback";
    if (!state.emitted.insert(key.canonical + '\n' + reason).second) return;
    // Names and reason only: environment values may contain private data.
    std::fprintf(stderr, "cachedMoE env: %s / %s: %s\n",
                 key.canonical.c_str(), key.legacy.c_str(), reason.c_str());
}

} // namespace detail

inline Value lookup(const char* name) {
    const Names key = names(name);
    if (!key.project) {
        const char* value = raw_get(name);
        return {value, value ? Source::Native : Source::Absent};
    }
    const char* canonical = raw_get(key.canonical.c_str());
    const char* legacy = raw_get(key.legacy.c_str());
    if (canonical) {
        if (legacy && std::strcmp(canonical, legacy) != 0) detail::warn_once(key, true);
        return {canonical, Source::Canonical};
    }
    if (legacy) {
        detail::warn_once(key, false);
        return {legacy, Source::Legacy};
    }
    return {};
}

// Preserve consumer parsing (exact-one, first-one, not-zero, empty/presence).
inline const char* get(const char* name) { return lookup(name).value; }

} // namespace cachedmoe::environment
