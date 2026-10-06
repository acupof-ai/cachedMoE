#pragma once

#include "core/namespace.h"

#include <cstdlib>
#include <string>

#include "core/env.h"

namespace cachedmoe::test {

// Tests must isolate both aliases: an inherited canonical value must not
// override a deliberate legacy fixture, and both original values are restored.
class ScopedEnvironment {
public:
    const char* key;
    explicit ScopedEnvironment(const char* name, const char* value = nullptr)
        : key(name), names_(environment::names(name)),
          canonical_(snapshot(names_.canonical)), legacy_(snapshot(names_.legacy)) {
        set(nullptr);
        if (value) write(key, value);
    }
    ScopedEnvironment(const ScopedEnvironment&) = delete;
    ScopedEnvironment& operator=(const ScopedEnvironment&) = delete;
    ~ScopedEnvironment() {
        restore(names_.canonical, canonical_);
        if (names_.project) restore(names_.legacy, legacy_);
    }
    void set(const char* value) {
        write(names_.canonical.c_str(), nullptr);
        if (names_.project) write(names_.legacy.c_str(), nullptr);
        if (value) write(key, value);
    }
    static void write(const char* name, const char* value) {
#ifdef _WIN32
        _putenv_s(name, value ? value : "");
#else
        if (value) setenv(name, value, 1); else unsetenv(name);
#endif
    }

private:
    struct Snapshot { bool present = false; std::string value; };
    static Snapshot snapshot(const std::string& name) {
        if (name.empty()) return {};
        const char* value = environment::raw_get(name.c_str());
        return {value != nullptr, value ? value : ""};
    }
    static void restore(const std::string& name, const Snapshot& value) {
        if (!name.empty()) write(name.c_str(), value.present ? value.value.c_str() : nullptr);
    }
    environment::Names names_;
    Snapshot canonical_, legacy_;
};

} // namespace cachedmoe::test
