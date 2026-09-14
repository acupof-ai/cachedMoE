// The writing half of core/json.h: just enough to emit one-line JSON events
// (`deepmoe serve`, `deepmoe tokenize`). Strings are expected to be UTF-8 and
// pass through unescaped apart from what JSON requires.
//
// Ownership/threading: pure functions.
#pragma once

#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <string_view>

namespace deepmoe {

// `s` as a quoted JSON string.
inline std::string json_quote(std::string_view s) {
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    out.push_back('"');
    return out;
}

inline std::string json_uint_array(std::span<const uint32_t> v) {
    std::string out = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) out.push_back(',');
        out += std::to_string(v[i]);
    }
    out.push_back(']');
    return out;
}

// A finite double as JSON; NaN and infinities (which JSON has no spelling for)
// as null.
inline std::string json_number(double d) {
    if (!(d == d) || d > 1.7e308 || d < -1.7e308) return "null";
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.9g", d);
    return buf;
}

}  // namespace deepmoe
