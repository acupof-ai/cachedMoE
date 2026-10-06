// Shared NDJSON option parsing. Defaults belong to the request value type;
// single- and multi-stream adapters only supply the transport fields.
#pragma once

#include "core/json.h"
#include "runtime/session.h"

namespace cachedmoe::cli {

inline std::vector<uint32_t> token_ids(const JsonValue &value) {
    std::vector<uint32_t> result;
    if (auto array = value.as_array(); array)
        for (const JsonValue &item : **array)
            if (auto id = item.as_uint(); id)
                result.push_back(static_cast<uint32_t>(*id));
    return result;
}

inline void apply_generate_options(const JsonValue &value, runtime::GenerateRequest &request) {
    request.max_tokens = static_cast<uint32_t>(value.int_or("max_tokens", request.max_tokens));
    request.sampling.temperature =
        static_cast<float>(value.double_or("temperature", request.sampling.temperature));
    request.sampling.top_p = static_cast<float>(value.double_or("top_p", request.sampling.top_p));
    request.sampling.seed = static_cast<uint64_t>(value.int_or("seed", request.sampling.seed));
    request.reuse = value.bool_or("reuse", request.reuse);
    if (const auto *ids = value.find("stop_ids"); ids && ids->is_array())
        request.stop_ids = token_ids(*ids);
}

} // namespace cachedmoe::cli
