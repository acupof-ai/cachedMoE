// The engram hash constants, derived at startup from the checkpoint's own
// tokenizer.json and config.json (Track R2, docs/p4_kv_ux.md §5).
//
// `inference/engram.py` builds three things and nothing else:
//
//   token map    `build_compressed_token_map`: every id decoded on its own,
//                normalised (NFKC, NFD, StripAccents, Lowercase, whitespace
//                runs to one space, a lone space kept, strip), and numbered by
//                first appearance of the normalised key. A token whose decoded
//                text holds U+FFFD is keyed by its raw token string instead.
//   multipliers  `numpy.random.default_rng(10007 * layer).integers(0, bound, 4)`
//                * 2 + 1, bound = (INT64_MAX // compressed_vocab) // 2. That is
//                SeedSequence -> PCG64 -> Lemire's bounded draw, reimplemented
//                here bit for bit.
//   primes       the next `n_heads` unused primes above `engram_vocab_size - 1`
//                for every (layer, n-gram order), in order; offsets are their
//                running sum per layer.
//
// The four normalisers that act per code point come from a table generated off
// HF tokenizers itself (tools/gen_engram_norm.py -> runtime/engram_norm.inc);
// the generator checks the per-code-point claim over the whole vocabulary.
// Validated identical to the L3 export's tables (tests/test_engram_tables.cpp).
//
// Ownership/threading: pure functions; no state.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "core/status.h"
#include "model/v41_config.h"
#include "runtime/engram.h"
#include "text/tokenizer.h"

namespace deepmoe::runtime {

// NFKC + NFD + StripAccents + Lowercase + the three whitespace rules of
// `build_compressed_token_map`, over valid UTF-8.
std::string engram_normalize(std::string_view utf8);

// `numpy.random.default_rng(seed).integers(0, high, size=n, dtype=np.int64)`.
std::vector<int64_t> numpy_rng_integers(uint64_t seed, int64_t high, size_t n);

Result<EngramTables> derive_engram_tables(const text::Tokenizer& tok, const TextConfig& cfg);
// Loads `<model_dir>/tokenizer.json` and derives from it.
Result<EngramTables> derive_engram_tables(const std::string& model_dir, const TextConfig& cfg);

}  // namespace deepmoe::runtime
