// Minimal leveled logger. Human-readable diagnostics only -- the measurement
// record goes through core/profiler.h as JSONL (design §4.1, §13.1).
//
// Ownership/threading: one process-wide sink guarded by a mutex; log() is safe
// from the IOCP completion threads, the planner thread and the submit thread.
// The level is a relaxed atomic, so filtering costs one load on the hot path.
#pragma once

#include <atomic>
#include <cstdio>
#include <format>
#include <mutex>
#include <string_view>
#include <utility>

namespace deepmoe {

enum class LogLevel : int { Trace = 0, Debug = 1, Info = 2, Warn = 3, Error = 4, Off = 5 };

namespace detail {
inline std::atomic<int> g_log_level{static_cast<int>(LogLevel::Info)};
inline std::mutex       g_log_mutex;
// Where Info/Debug/Trace go. Null = stdout, which is the default and what
// the one-shot commands want. `deepmoe serve` OWNS stdout -- the NDJSON
// protocol is duplicated off fd 1 and fd 1 is then pointed at stderr -- and a
// FILE* stdout pointed at a pipe is BLOCK buffered, so before Track PF every
// log_info a serve session wrote sat in an unflushed 4 KiB buffer and
// `--log web_serve.log` held only the few lines written with fprintf(stderr).
// serve calls set_log_stream(stderr).
inline std::atomic<std::FILE*> g_log_stream{nullptr};

inline constexpr std::string_view level_tag(LogLevel l) {
    switch (l) {
        case LogLevel::Trace: return "TRC";
        case LogLevel::Debug: return "DBG";
        case LogLevel::Info:  return "INF";
        case LogLevel::Warn:  return "WRN";
        case LogLevel::Error: return "ERR";
        case LogLevel::Off:   return "OFF";
    }
    return "???";
}

inline void emit(LogLevel l, std::string_view msg) {
    std::lock_guard lk(g_log_mutex);
    std::FILE* sink = g_log_stream.load(std::memory_order_relaxed);
    std::FILE* out = sink ? sink : ((l >= LogLevel::Warn) ? stderr : stdout);
    std::fprintf(out, "[%s] %.*s\n", level_tag(l).data(), static_cast<int>(msg.size()), msg.data());
    // Always: a diagnostic still sitting in a buffer when the process is
    // killed (which is how the web UI stops its engine) never existed.
    std::fflush(out);
}
}  // namespace detail

// Sends every level to `f` instead of the stdout/stderr split. Null restores
// the default. Relaxed atomic: safe to set before the first log line.
inline void set_log_stream(std::FILE* f) {
    detail::g_log_stream.store(f, std::memory_order_relaxed);
}

inline void set_log_level(LogLevel l) {
    detail::g_log_level.store(static_cast<int>(l), std::memory_order_relaxed);
}
inline LogLevel log_level() {
    return static_cast<LogLevel>(detail::g_log_level.load(std::memory_order_relaxed));
}
inline bool log_enabled(LogLevel l) { return static_cast<int>(l) >= detail::g_log_level.load(std::memory_order_relaxed); }

template <class... Args>
void log(LogLevel l, std::format_string<Args...> fmt, Args&&... args) {
    if (!log_enabled(l)) return;
    detail::emit(l, std::format(fmt, std::forward<Args>(args)...));
}

template <class... Args> void log_trace(std::format_string<Args...> f, Args&&... a) { log(LogLevel::Trace, f, std::forward<Args>(a)...); }
template <class... Args> void log_debug(std::format_string<Args...> f, Args&&... a) { log(LogLevel::Debug, f, std::forward<Args>(a)...); }
template <class... Args> void log_info (std::format_string<Args...> f, Args&&... a) { log(LogLevel::Info,  f, std::forward<Args>(a)...); }
template <class... Args> void log_warn (std::format_string<Args...> f, Args&&... a) { log(LogLevel::Warn,  f, std::forward<Args>(a)...); }
template <class... Args> void log_error(std::format_string<Args...> f, Args&&... a) { log(LogLevel::Error, f, std::forward<Args>(a)...); }

}  // namespace deepmoe
