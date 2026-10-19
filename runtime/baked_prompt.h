// Local, build-bound exact prompt state. Separate from legacy replay-based .pkv.
#pragma once
#include "runtime/engine.h"
#include <filesystem>

namespace cachedmoe::runtime {
struct BakedPrompt {
    std::vector<uint32_t> tokens;
    KvRowBackup rows;
    KvStore::RingSnapshot ring;
    DecodeStepResult first;
};
Result<void> write_baked_prompt(const std::filesystem::path&, uint64_t identity, const BakedPrompt&);
Result<BakedPrompt> read_baked_prompt(const std::filesystem::path&, uint64_t identity);
}
