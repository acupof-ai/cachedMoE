// Error type and Result alias used across every module boundary (design §14).
//
// Ownership/threading: Status is a value type, trivially copyable except for the
// message string; safe to move across threads. No exceptions cross a module
// boundary -- every fallible API returns Result<T>.
#pragma once

#include <expected>
#include <string>
#include <string_view>
#include <utility>

namespace deepmoe {

enum class Err {
    Ok = 0,
    Unimplemented,     // a designed-but-unwritten stub; carries a `TODO(design §x.y)`
    InvalidArgument,
    OutOfRange,
    NotFound,
    AlreadyExists,
    ResourceExhausted, // slab pool full, queue full, out of memory
    FailedPrecondition,// wrong state (e.g. evicting a Filling slot)
    Io,                // OS read/write failure; `os_code` holds GetLastError()/errno
    Unavailable,       // backend/device/extension not present on this machine
    Corrupt,           // manifest/config/file content did not parse or verify
    Cancelled,
    Internal,
};

constexpr std::string_view err_name(Err e) noexcept {
    switch (e) {
        case Err::Ok:                 return "ok";
        case Err::Unimplemented:      return "unimplemented";
        case Err::InvalidArgument:    return "invalid-argument";
        case Err::OutOfRange:         return "out-of-range";
        case Err::NotFound:           return "not-found";
        case Err::AlreadyExists:      return "already-exists";
        case Err::ResourceExhausted:  return "resource-exhausted";
        case Err::FailedPrecondition: return "failed-precondition";
        case Err::Io:                 return "io";
        case Err::Unavailable:        return "unavailable";
        case Err::Corrupt:            return "corrupt";
        case Err::Cancelled:          return "cancelled";
        case Err::Internal:           return "internal";
    }
    return "?";
}

struct Status {
    Err         code = Err::Internal;
    std::string message;
    uint32_t    os_code = 0;          // GetLastError() / errno, 0 when not an OS error

    Status() = default;
    Status(Err c, std::string m = {}, uint32_t os = 0)
        : code(c), message(std::move(m)), os_code(os) {}

    std::string str() const {
        std::string s(err_name(code));
        if (!message.empty()) { s += ": "; s += message; }
        if (os_code)          { s += " (os "; s += std::to_string(os_code); s += ")"; }
        return s;
    }
};

template <class T> using Result = std::expected<T, Status>;

inline std::unexpected<Status> fail(Err c, std::string m = {}, uint32_t os = 0) {
    return std::unexpected(Status{c, std::move(m), os});
}

// Convenience for the many stubs this skeleton carries.
inline std::unexpected<Status> unimplemented(std::string what) {
    return std::unexpected(Status{Err::Unimplemented, std::move(what)});
}

#define DEEPMOE_TRY(decl, expr)                                                \
    auto&& _dm_tmp_##__LINE__ = (expr);                                        \
    if (!_dm_tmp_##__LINE__) return std::unexpected(_dm_tmp_##__LINE__.error());\
    decl = *std::move(_dm_tmp_##__LINE__)

}  // namespace deepmoe
