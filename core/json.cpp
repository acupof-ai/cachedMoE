#include "core/json.h"

#include <charconv>
#include <format>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace deepmoe {
namespace {

constexpr int kMaxDepth = 64;

struct Parser {
    std::string_view s;
    size_t           i = 0;
    int              depth = 0;

    bool eof() const { return i >= s.size(); }
    char peek() const { return s[i]; }

    void skip_ws() {
        while (i < s.size()) {
            char c = s[i];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++i; else break;
        }
    }

    std::unexpected<Status> err(std::string what) const {
        return fail(Err::Corrupt, std::format("{} at byte {}", what, i));
    }

    Result<JsonValue> value() {
        if (depth >= kMaxDepth) return err("nesting too deep");
        skip_ws();
        if (eof()) return err("unexpected end of input");
        switch (peek()) {
            case '{': return object();
            case '[': return array();
            case '"': {
                auto r = string_literal();
                if (!r) return std::unexpected(r.error());
                return JsonValue::make_string(std::move(*r));
            }
            case 't':
                if (s.substr(i).starts_with("true"))  { i += 4; return JsonValue::make_bool(true); }
                return err("bad literal");
            case 'f':
                if (s.substr(i).starts_with("false")) { i += 5; return JsonValue::make_bool(false); }
                return err("bad literal");
            case 'n':
                if (s.substr(i).starts_with("null"))  { i += 4; return JsonValue{}; }
                return err("bad literal");
            default:  return number();
        }
    }

    Result<JsonValue> object() {
        ++i;  // '{'
        ++depth;
        JsonObject o;
        skip_ws();
        if (!eof() && peek() == '}') { ++i; --depth; return JsonValue::make_object(std::move(o)); }
        for (;;) {
            skip_ws();
            if (eof() || peek() != '"') return err("expected object key");
            auto k = string_literal();
            if (!k) return std::unexpected(k.error());
            skip_ws();
            if (eof() || peek() != ':') return err("expected ':'");
            ++i;
            auto v = value();
            if (!v) return std::unexpected(v.error());
            o.insert_or_assign(std::move(*k), std::move(*v));
            skip_ws();
            if (eof()) return err("unterminated object");
            if (peek() == ',') { ++i; continue; }
            if (peek() == '}') { ++i; --depth; return JsonValue::make_object(std::move(o)); }
            return err("expected ',' or '}'");
        }
    }

    Result<JsonValue> array() {
        ++i;  // '['
        ++depth;
        JsonArray a;
        skip_ws();
        if (!eof() && peek() == ']') { ++i; --depth; return JsonValue::make_array(std::move(a)); }
        for (;;) {
            auto v = value();
            if (!v) return std::unexpected(v.error());
            a.push_back(std::move(*v));
            skip_ws();
            if (eof()) return err("unterminated array");
            if (peek() == ',') { ++i; continue; }
            if (peek() == ']') { ++i; --depth; return JsonValue::make_array(std::move(a)); }
            return err("expected ',' or ']'");
        }
    }

    static void utf8_append(std::string& out, uint32_t cp) {
        if (cp < 0x80) out.push_back(static_cast<char>(cp));
        else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }

    Result<uint32_t> hex4() {
        if (i + 4 > s.size()) return err("truncated \\u escape");
        uint32_t v = 0;
        for (int k = 0; k < 4; ++k) {
            char c = s[i + k];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= static_cast<uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') v |= static_cast<uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= static_cast<uint32_t>(c - 'A' + 10);
            else return err("bad hex digit in \\u escape");
        }
        i += 4;
        return v;
    }

    Result<std::string> string_literal() {
        ++i;  // opening quote
        std::string out;
        while (true) {
            if (eof()) return err("unterminated string");
            char c = s[i++];
            if (c == '"') return out;
            if (c != '\\') {
                if (static_cast<unsigned char>(c) < 0x20) return err("raw control char in string");
                out.push_back(c);
                continue;
            }
            if (eof()) return err("unterminated escape");
            char e = s[i++];
            switch (e) {
                case '"':  out.push_back('"');  break;
                case '\\': out.push_back('\\'); break;
                case '/':  out.push_back('/');  break;
                case 'b':  out.push_back('\b'); break;
                case 'f':  out.push_back('\f'); break;
                case 'n':  out.push_back('\n'); break;
                case 'r':  out.push_back('\r'); break;
                case 't':  out.push_back('\t'); break;
                case 'u': {
                    auto hi = hex4();
                    if (!hi) return std::unexpected(hi.error());
                    uint32_t cp = *hi;
                    if (cp >= 0xD800 && cp <= 0xDBFF && i + 1 < s.size() && s[i] == '\\' && s[i + 1] == 'u') {
                        i += 2;
                        auto lo = hex4();
                        if (!lo) return std::unexpected(lo.error());
                        if (*lo >= 0xDC00 && *lo <= 0xDFFF)
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (*lo - 0xDC00);
                        else { utf8_append(out, cp); cp = *lo; }
                    }
                    utf8_append(out, cp);
                    break;
                }
                default: return err("unknown escape");
            }
        }
    }

    Result<JsonValue> number() {
        size_t start = i;
        if (!eof() && (peek() == '-' || peek() == '+')) ++i;
        bool any = false;
        while (!eof() && peek() >= '0' && peek() <= '9') { ++i; any = true; }
        if (!eof() && peek() == '.') {
            ++i;
            while (!eof() && peek() >= '0' && peek() <= '9') { ++i; any = true; }
        }
        if (!eof() && (peek() == 'e' || peek() == 'E')) {
            ++i;
            if (!eof() && (peek() == '-' || peek() == '+')) ++i;
            while (!eof() && peek() >= '0' && peek() <= '9') ++i;
        }
        if (!any) return err("expected a value");
        std::string raw(s.substr(start, i - start));
        // std::from_chars for double is not available in every libc++ build; strtod
        // on a NUL-terminated copy is exact enough and the raw text is kept anyway.
        double d = std::strtod(raw.c_str(), nullptr);
        return JsonValue::make_number(d, std::move(raw));
    }
};

}  // namespace

JsonValue JsonValue::make_bool(bool b)   { JsonValue v; v.type_ = JsonType::Bool;  v.bool_ = b; return v; }
JsonValue JsonValue::make_string(std::string s) { JsonValue v; v.type_ = JsonType::String; v.str_ = std::move(s); return v; }
JsonValue JsonValue::make_array(JsonArray a)    { JsonValue v; v.type_ = JsonType::Array;  v.arr_ = std::move(a); return v; }
JsonValue JsonValue::make_object(JsonObject o)  { JsonValue v; v.type_ = JsonType::Object; v.obj_ = std::move(o); return v; }
JsonValue JsonValue::make_number(double d, std::string raw) {
    JsonValue v; v.type_ = JsonType::Number; v.num_ = d; v.str_ = std::move(raw); return v;
}

Result<bool> JsonValue::as_bool() const {
    if (type_ != JsonType::Bool) return fail(Err::Corrupt, "expected bool");
    return bool_;
}
Result<double> JsonValue::as_double() const {
    if (type_ != JsonType::Number) return fail(Err::Corrupt, "expected number");
    return num_;
}
Result<int64_t> JsonValue::as_int() const {
    if (type_ != JsonType::Number) return fail(Err::Corrupt, "expected number");
    int64_t out = 0;
    const char* b = str_.data();
    const char* e = b + str_.size();
    if (b != e && *b == '+') ++b;
    auto [p, ec] = std::from_chars(b, e, out);
    if (ec == std::errc{} && p == e) return out;
    // Fall back through the double when the literal carries an exponent or a
    // fraction (e.g. 1e-20); such fields are never used as integers, but a
    // caller that asks gets the rounded value rather than an error.
    if (num_ < -9.22e18 || num_ > 9.22e18) return fail(Err::Corrupt, "integer out of range");
    return static_cast<int64_t>(num_);
}
Result<uint64_t> JsonValue::as_uint() const {
    if (type_ != JsonType::Number) return fail(Err::Corrupt, "expected number");
    uint64_t out = 0;
    const char* b = str_.data();
    const char* e = b + str_.size();
    if (b != e && *b == '+') ++b;
    auto [p, ec] = std::from_chars(b, e, out);
    if (ec == std::errc{} && p == e) return out;
    if (num_ < 0 || num_ > 1.8e19) return fail(Err::Corrupt, "unsigned out of range");
    return static_cast<uint64_t>(num_);
}
Result<std::string_view> JsonValue::as_string() const {
    if (type_ != JsonType::String) return fail(Err::Corrupt, "expected string");
    return std::string_view(str_);
}
Result<const JsonArray*> JsonValue::as_array() const {
    if (type_ != JsonType::Array) return fail(Err::Corrupt, "expected array");
    return &arr_;
}
Result<const JsonObject*> JsonValue::as_object() const {
    if (type_ != JsonType::Object) return fail(Err::Corrupt, "expected object");
    return &obj_;
}

const JsonValue* JsonValue::find(std::string_view key) const {
    if (type_ != JsonType::Object) return nullptr;
    auto it = obj_.find(key);
    return it == obj_.end() ? nullptr : &it->second;
}
Result<const JsonValue*> JsonValue::at(std::string_view key) const {
    if (const JsonValue* v = find(key)) return v;
    return fail(Err::NotFound, std::format("missing key '{}'", key));
}

Result<int64_t> JsonValue::int_at(std::string_view key) const {
    auto v = at(key); if (!v) return std::unexpected(v.error());
    auto r = (*v)->as_int();
    if (!r) return fail(Err::Corrupt, std::format("key '{}': {}", key, r.error().message));
    return *r;
}
Result<uint64_t> JsonValue::uint_at(std::string_view key) const {
    auto v = at(key); if (!v) return std::unexpected(v.error());
    auto r = (*v)->as_uint();
    if (!r) return fail(Err::Corrupt, std::format("key '{}': {}", key, r.error().message));
    return *r;
}
Result<double> JsonValue::double_at(std::string_view key) const {
    auto v = at(key); if (!v) return std::unexpected(v.error());
    auto r = (*v)->as_double();
    if (!r) return fail(Err::Corrupt, std::format("key '{}': {}", key, r.error().message));
    return *r;
}
Result<bool> JsonValue::bool_at(std::string_view key) const {
    auto v = at(key); if (!v) return std::unexpected(v.error());
    auto r = (*v)->as_bool();
    if (!r) return fail(Err::Corrupt, std::format("key '{}': {}", key, r.error().message));
    return *r;
}
Result<std::string> JsonValue::string_at(std::string_view key) const {
    auto v = at(key); if (!v) return std::unexpected(v.error());
    auto r = (*v)->as_string();
    if (!r) return fail(Err::Corrupt, std::format("key '{}': {}", key, r.error().message));
    return std::string(*r);
}
Result<std::vector<int64_t>> JsonValue::int_array_at(std::string_view key) const {
    auto v = at(key); if (!v) return std::unexpected(v.error());
    auto a = (*v)->as_array();
    if (!a) return fail(Err::Corrupt, std::format("key '{}': expected array", key));
    std::vector<int64_t> out;
    out.reserve((*a)->size());
    for (const JsonValue& e : **a) {
        auto n = e.as_int();
        if (!n) return fail(Err::Corrupt, std::format("key '{}': non-integer element", key));
        out.push_back(*n);
    }
    return out;
}

int64_t JsonValue::int_or(std::string_view key, int64_t dflt) const {
    const JsonValue* v = find(key);
    if (!v || v->is_null()) return dflt;
    auto r = v->as_int();
    return r ? *r : dflt;
}
double JsonValue::double_or(std::string_view key, double dflt) const {
    const JsonValue* v = find(key);
    if (!v || v->is_null()) return dflt;
    auto r = v->as_double();
    return r ? *r : dflt;
}
bool JsonValue::bool_or(std::string_view key, bool dflt) const {
    const JsonValue* v = find(key);
    if (!v || v->is_null()) return dflt;
    auto r = v->as_bool();
    return r ? *r : dflt;
}
std::string JsonValue::string_or(std::string_view key, std::string_view dflt) const {
    const JsonValue* v = find(key);
    if (!v || v->is_null()) return std::string(dflt);
    auto r = v->as_string();
    return r ? std::string(*r) : std::string(dflt);
}

size_t JsonValue::size() const noexcept {
    if (type_ == JsonType::Array)  return arr_.size();
    if (type_ == JsonType::Object) return obj_.size();
    return 0;
}

Result<JsonValue> json_parse(std::string_view text) {
    // Tolerate a UTF-8 BOM; ModelScope serves config.json without one, but
    // manifest.json may be written by a Python script on Windows.
    if (text.starts_with("\xEF\xBB\xBF")) text.remove_prefix(3);
    Parser p{text};
    auto v = p.value();
    if (!v) return v;
    p.skip_ws();
    if (!p.eof()) return fail(Err::Corrupt, std::format("trailing content at byte {}", p.i));
    return v;
}

Result<JsonValue> json_parse_file(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return fail(Err::Io, std::format("cannot open '{}'", path));
    std::string buf;
    char chunk[65536];
    size_t n;
    while ((n = std::fread(chunk, 1, sizeof chunk, f)) > 0) buf.append(chunk, n);
    bool bad = std::ferror(f) != 0;
    std::fclose(f);
    if (bad) return fail(Err::Io, std::format("read error on '{}'", path));
    auto v = json_parse(buf);
    if (!v) return fail(v.error().code, std::format("{}: {}", path, v.error().message));
    return v;
}

}  // namespace deepmoe
