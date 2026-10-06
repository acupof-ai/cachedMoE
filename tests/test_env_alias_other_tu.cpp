#include "core/env.h"

const char* env_alias_other_tu(const char* key) {
    return deepmoe::environment::get(key);
}
