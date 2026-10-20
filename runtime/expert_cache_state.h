// Restart hints contain only resident expert IDs in newest-first LRU order.
// Every restored weight is read from the original checkpoint, never this file.
#pragma once

#include "core/types.h"
#include "core/status.h"
#include <filesystem>
#include <span>
#include <vector>

namespace cachedmoe::runtime {

Result<uint64_t> expert_cache_identity(const std::filesystem::path& model);
Result<void> write_expert_cache_state(const std::filesystem::path& path, uint64_t identity,
                                     uint32_t capacity, std::span<const ExpertKey> newest_first);
Result<std::vector<ExpertKey>> read_expert_cache_state(const std::filesystem::path& path,
                                                     uint64_t identity, uint32_t capacity);

} // namespace cachedmoe::runtime
