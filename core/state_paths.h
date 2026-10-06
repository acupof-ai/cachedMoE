#pragma once

#include <array>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "core/status.h"

namespace cachedmoe::state_paths {

inline constexpr std::string_view kAppDirectory = "cachedmoe";
inline constexpr std::string_view kLegacyDirectory = "deepmoe";

enum class Platform { Posix, Windows };

// Native OS variables are independent of the project's runtime aliases. Copy
// them once so the selector never holds pointers across an environment change.
struct Environment {
    std::string xdg_cache_home;
    std::string home;
    std::string local_appdata;
    std::string temp;
    std::string tmpdir;
    std::string tmp;
    std::string tempdir;
};

inline Environment read_environment() {
    const auto value = [](const char* name) {
        const char* raw = std::getenv(name);
        return raw ? std::string(raw) : std::string();
    };
    return {value("XDG_CACHE_HOME"), value("HOME"), value("LOCALAPPDATA"),
            value("TEMP"), value("TMPDIR"), value("TMP"), value("TEMPDIR")};
}

inline constexpr Platform native_platform() {
#if defined(_WIN32)
    return Platform::Windows;
#else
    return Platform::Posix;
#endif
}

// The explicit temporary-variable order is also used by tools/state_paths.py.
// Do not use the STL/Python environment searches: their orders differ.
inline Result<std::filesystem::path> cache_home(
    const Environment& env, Platform platform,
    const std::filesystem::path& system_temp_override = {}) {
    namespace fs = std::filesystem;
    if (platform == Platform::Windows) {
        if (!env.local_appdata.empty()) return fs::path(env.local_appdata);
        if (!env.temp.empty()) return fs::path(env.temp);
    } else {
        if (!env.xdg_cache_home.empty()) return fs::path(env.xdg_cache_home);
        if (!env.home.empty()) return fs::path(env.home) / ".cache";
    }
    for (const std::string* value : {&env.tmpdir, &env.temp, &env.tmp, &env.tempdir})
        if (!value->empty()) return fs::path(*value);
    if (!system_temp_override.empty()) return system_temp_override;
    if (platform == Platform::Posix) return fs::path("/tmp");
#if defined(_WIN32)
    std::array<wchar_t, 32768> buffer{};
    const DWORD length = GetTempPathW(static_cast<DWORD>(buffer.size()), buffer.data());
    if (!length || length >= buffer.size())
        return fail(Err::Io, "cannot resolve the Windows system temporary directory",
                    GetLastError());
    return fs::path(buffer.data());
#else
    return fail(Err::Unavailable, "Windows system temporary directory is unavailable here");
#endif
}

struct StateRoot {
    std::filesystem::path path;
    std::string source;
    bool both_exist = false;
};

namespace detail {

inline Result<bool> directory_exists(const std::filesystem::path& path) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::file_status entry = fs::symlink_status(path, ec);
    if (entry.type() == fs::file_type::not_found &&
        (!ec || ec == std::errc::no_such_file_or_directory))
        return false;
    if (ec)
        return fail(Err::Io, "cannot inspect state path '" + path.string() + "': " +
                                ec.message(), static_cast<uint32_t>(ec.value()));

    // lstat distinguishes a missing root from a dangling root symlink. Only
    // valid directory symlinks may act as roots; broken links must not hide data.
    fs::file_status target = entry;
    if (fs::is_symlink(entry)) {
        target = fs::status(path, ec);
        if (ec)
            return fail(Err::Io, "cannot inspect state path target '" + path.string() +
                                    "': " + ec.message(), static_cast<uint32_t>(ec.value()));
    }
    if (!fs::is_directory(target))
        return fail(Err::AlreadyExists,
                    "state path exists but is not a directory: " + path.string());
    return true;
}

} // namespace detail

// Pure selection: inspect both roots, including the unused one, and never
// create, move, remove or overwrite a directory or its contents.
inline Result<StateRoot> choose_paths(const std::filesystem::path& canonical,
                                      const std::filesystem::path& legacy) {
    auto current_exists = detail::directory_exists(canonical);
    if (!current_exists) return std::unexpected(current_exists.error());
    auto legacy_exists = detail::directory_exists(legacy);
    if (!legacy_exists) return std::unexpected(legacy_exists.error());
    if (*current_exists)
        return StateRoot{canonical, "canonical-existing", *legacy_exists};
    if (*legacy_exists)
        return StateRoot{legacy, "legacy-existing", false};
    return StateRoot{canonical, "canonical-new", false};
}

inline Result<StateRoot> choose_directory(const std::filesystem::path& base) {
    return choose_paths(base / kAppDirectory, base / kLegacyDirectory);
}

inline Result<StateRoot> application_root() {
    auto base = cache_home(read_environment(), native_platform());
    if (!base) return std::unexpected(base.error());
    return choose_directory(*base);
}

// Keep the previous per-model identity/path tag. Explicit user directories
// bypass this helper and remain literal paths in the CLI.
inline std::filesystem::path model_kv_directory(const StateRoot& root,
                                               std::string model_dir) {
    for (char& ch : model_dir)
        if (ch == '\\' || ch == '/' || ch == ':') ch = '_';
    return root.path / "kv" / model_dir;
}

} // namespace cachedmoe::state_paths
