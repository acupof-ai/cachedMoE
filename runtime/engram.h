// Engram sublayer (layers 1 and 14) -- design §2.4, §7.10.
//
// Each engram layer owns a 384M-row hash table of 256 fp8 values (98.3 GB of
// values + 3.07 GB of scales per layer, 203 GB total). Per token it reads 24
// rows: 3 n-gram orders x 8 heads. The row index is a multiply-XOR hash of the
// compressed token id with the previous three tokens, each of the 24 buckets
// taking a different prime modulus.
//
// The 24 x 256 values concatenate into a 6144-wide vector that goes through
// wkv [25600 x 6144] fp8 (157 MB) to produce 4 key copies and 1 value; the gate
// is sigmoid of the signed square root of a normalised dot product.
//
// Because the addresses depend only on token ids, prefetch can start the moment
// a token exists -- including an unverified DSpark draft token (design §9.5).
//
// Ownership/threading: Engram borrows store/engram_prefetch.h and the GPU
// pipelines. Row assembly happens on the engine thread; the fetch itself is
// P2-priority I/O on the IoEngine.
#pragma once

#include <cstdint>
#include <span>

#include "core/status.h"
#include "core/types.h"
#include "model/layout.h"

namespace deepmoe::runtime {

class Engram {
public:
    virtual ~Engram() = default;

    // The 24 row indices for one token at one engram layer.
    // TODO(design §2.4): the exact hash constants must be lifted from
    // inference/model.py and locked by an L0 oracle test before this is
    // written -- see store/engram_prefetch.h.
    virtual Result<void> hash_rows(uint32_t layer,
                                   std::span<const uint32_t> recent_tokens,
                                   std::span<uint64_t> out_rows) = 0;

    // Records the two dispatches of design §7.10: the 6144 -> 25600 fp8 GEMV,
    // then the gating and residual update.
    // TODO(design §7.10): implement in P2/P3.
    virtual Result<void> record(uint32_t layer, TokenIndex token) = 0;
};

// The compressed vocabulary the hash is computed over: NFKC, lowercased,
// whitespace-normalised, 99,092 entries out of the 129,280 token vocabulary
// (engram_compressed_vocab_size in config.json).
// TODO(design §2.4): built by tools/repack.py from the tokenizer.
class CompressedVocab {
public:
    virtual ~CompressedVocab() = default;
    virtual uint32_t compress(uint32_t token_id) const = 0;
    virtual uint32_t size() const = 0;
};

}  // namespace deepmoe::runtime
