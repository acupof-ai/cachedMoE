// Native constants expand the same literal authority read by Python.
#pragma once

#include "core/namespace.h"

#include <cstdint>
#include <string_view>

namespace cachedmoe::configuration::facts {

#define CACHEDMOE_RUNTIME_FACT_INTEGER(type, name, value) \
    inline constexpr std::type name{value##ULL};
#define CACHEDMOE_RUNTIME_FACT_REAL(name, value) \
    inline constexpr double name = value;
#define CACHEDMOE_RUNTIME_FACT_TEXT(name, value) \
    inline constexpr std::string_view name{value};
#include "core/runtime_facts.def"
#undef CACHEDMOE_RUNTIME_FACT_INTEGER
#undef CACHEDMOE_RUNTIME_FACT_REAL
#undef CACHEDMOE_RUNTIME_FACT_TEXT

}  // namespace cachedmoe::configuration::facts
