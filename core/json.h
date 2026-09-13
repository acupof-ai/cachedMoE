// A small read-only JSON DOM. Written in-tree rather than vendored, because the
// only JSON deepMoE reads is config.json and manifest.json (design §2.1, §5.1)
// and the rule is "no third-party deps".
//
// Not a general-purpose parser: no comments, no trailing commas, no duplicate
// key merging, and numbers are stored as double plus the raw text (manifest
// offsets exceed 2^53 only in theory, but int64 accessors parse the raw text so
// large byte offsets stay exact).
//
// Ownership/threading: JsonValue owns its children. Parsing is pure; a parsed
// document is immutable afterwards and therefore safe to share read-only
// between threads.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "core/status.h"

namespace deepmoe {

class JsonValue;
using JsonObject = std::map<std::string, JsonValue, std::less<>>;
using JsonArray  = std::vector<JsonValue>;

enum class JsonType : uint8_t { Null, Bool, Number, String, Array, Object };

class JsonValue {
public:
    JsonValue() = default;                                // null
    static JsonValue make_bool(bool b);
    static JsonValue make_number(double d, std::string raw);
    static JsonValue make_string(std::string s);
    static JsonValue make_array(JsonArray a);
    static JsonValue make_object(JsonObject o);

    JsonType type() const noexcept { return type_; }
    bool is_null()   const noexcept { return type_ == JsonType::Null; }
    bool is_bool()   const noexcept { return type_ == JsonType::Bool; }
    bool is_number() const noexcept { return type_ == JsonType::Number; }
    bool is_string() const noexcept { return type_ == JsonType::String; }
    bool is_array()  const noexcept { return type_ == JsonType::Array; }
    bool is_object() const noexcept { return type_ == JsonType::Object; }

    // Typed accessors. Each returns Corrupt if the node is of another type.
    Result<bool>        as_bool()   const;
    Result<double>      as_double() const;
    Result<int64_t>     as_int()    const;   // exact for |v| < 2^63, parsed from raw text
    Result<uint64_t>    as_uint()   const;
    Result<std::string_view> as_string() const;
    Result<const JsonArray*>  as_array()  const;
    Result<const JsonObject*> as_object() const;

    // Object lookup. `find` returns nullptr when absent, `at` returns NotFound.
    const JsonValue* find(std::string_view key) const;
    Result<const JsonValue*> at(std::string_view key) const;

    // Convenience readers used by v41_config / manifest. `path` is a single key.
    Result<int64_t>     int_at(std::string_view key) const;
    Result<uint64_t>    uint_at(std::string_view key) const;
    Result<double>      double_at(std::string_view key) const;
    Result<bool>        bool_at(std::string_view key) const;
    Result<std::string> string_at(std::string_view key) const;
    Result<std::vector<int64_t>> int_array_at(std::string_view key) const;

    // Same, but substitutes `dflt` when the key is absent or null.
    int64_t     int_or(std::string_view key, int64_t dflt) const;
    double      double_or(std::string_view key, double dflt) const;
    bool        bool_or(std::string_view key, bool dflt) const;
    std::string string_or(std::string_view key, std::string_view dflt) const;

    size_t size() const noexcept;   // array/object element count, else 0

private:
    JsonType    type_ = JsonType::Null;
    bool        bool_ = false;
    double      num_  = 0;
    std::string str_;               // string payload, or the raw number text
    JsonArray   arr_;
    JsonObject  obj_;
};

// Parses a complete document. `text` need not be NUL-terminated.
Result<JsonValue> json_parse(std::string_view text);

// Reads the file then parses it. Errors carry the path.
Result<JsonValue> json_parse_file(const std::string& path);

}  // namespace deepmoe
