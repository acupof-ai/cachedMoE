// The serve adapter's disk-policy precedence, independently testable without
// starting an engine. Directory selection does not migrate or create files.
#pragma once

#include "core/config.h"
#include "core/state_paths.h"
#include "runtime/session.h"

#include <functional>

namespace cachedmoe::cli {

struct KvCommandLineOptions {
    bool disabled = false;
    bool explicit_directory = false;
    void set_directory(runtime::KvDiskOptions &disk, std::string value) {
        disk.dir = std::move(value);
        explicit_directory = true;
    }
    void disable() { disabled = true; }
};

using StateRootResolver = std::function<Result<state_paths::StateRoot>()>;

inline Result<std::optional<state_paths::StateRoot>> resolve_kv_options(
    runtime::KvDiskOptions &disk, bool disabled, bool explicit_directory,
    const configuration::RuntimeEnvironment &environment, const std::string &model,
    StateRootResolver resolve_root = [] { return state_paths::application_root(); }) {
    // Disabled wins even when --kv-dir appeared first or last in argv.
    if (disabled) {
        disk.dir.clear();
        disk.model_tag.clear();
        return std::optional<state_paths::StateRoot>{};
    }
    std::optional<state_paths::StateRoot> root;
    if (!explicit_directory) {
        if (environment.kv_dir)
            disk.dir = *environment.kv_dir;
        else {
            auto selected = resolve_root();
            if (!selected)
                return std::unexpected(selected.error());
            root = *std::move(selected);
            disk.dir = state_paths::model_kv_directory(*root, model).string();
        }
    }
    disk.model_tag = disk.dir.empty() ? std::string{} : model;
    return root;
}

} // namespace cachedmoe::cli
