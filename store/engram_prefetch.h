// EngramPrefetcher (design §4.1, §7.10, §9.6 P2).
//
// The engram tables are 203 GB across two layers and permanently live on NVMe.
// They are also the one part of the model whose addresses are *known in
// advance*: a row index depends only on token ids, so the moment a token is
// decided -- including a DSpark draft token that has not been verified yet --
// its 24 rows (3 n-gram orders x 8 heads) can be queued at P2.
//
// Row layout (design §5.1): 264 B interleaved, 256 B of FP8 values followed by
// 8 B of E8M0 scale, so a single 4 KiB read picks up whole rows with scales.
// Reads are issued 4 KiB-aligned around the row, which is why Q7 asks for the
// latency distribution of 4 KiB random reads at QD 48.
//
// Ownership/threading: the prefetcher borrows the IoEngine and a ring buffer of
// GPU-visible staging rows. hash_token/enqueue are called from the engine
// thread; completions land on the IoEngine dispatcher thread.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "core/bytes.h"
#include "core/status.h"
#include "core/types.h"
#include "model/layout.h"
#include "storage/io_engine.h"

namespace deepmoe::store {

// The 24 rows one token needs from one engram layer.
struct EngramRows {
    uint32_t layer = 0;
    std::array<uint64_t, layout::kEngramRowsPerToken> row_index{};
    std::array<uint64_t, layout::kEngramRowsPerToken> file_offset{};
};

// Where a fetched row landed. `data` is 264 B: 256 B fp8 + 8 B E8M0.
struct EngramRowView {
    uint32_t   slot = 0;
    ByteSpan   data;
};

class EngramPrefetcher {
public:
    virtual ~EngramPrefetcher() = default;

    // Maps token ids to the 24 row indices for `layer`. The reference does
    // this with a compressed-vocabulary id (NFKC / lowercase / whitespace
    // normalised, 99,092 entries) hashed together with the previous three
    // tokens by multiply-XOR, each of the 24 buckets taking a different prime
    // modulus (design §2.4).
    //
    // TODO(design §2.4, §7.10): the exact multiplier/prime set has to be read
    // out of inference/model.py and checked against the L0 oracle before this
    // can be written. Everything downstream of it is address arithmetic.
    virtual Result<EngramRows> hash_token(uint32_t layer,
                                          std::span<const uint32_t> recent_token_ids) = 0;

    // Queues the 24 rows at P2 into the staging ring.
    virtual Result<void> enqueue(const EngramRows& rows, TokenIndex token) = 0;

    // Blocks until the rows for `token` have landed, then returns views into
    // the ring. Used by the engram kernel's input assembly (design §7.10).
    virtual Result<std::vector<EngramRowView>> acquire(uint32_t layer, TokenIndex token) = 0;

    // Releases the ring slots once the GPU has consumed them.
    virtual void release(TokenIndex token) = 0;
};

struct EngramPrefetchConfig {
    uint32_t ring_rows   = 512;   // staging rows; >= 24 x lookahead tokens
    uint32_t queue_depth = 48;    // design Q7 measures at this depth
};

// TODO(design §7.10): implement on top of storage::IoEngine and a GPU-visible
// ring buffer. Blocked on the hash definition above.
Result<std::unique_ptr<EngramPrefetcher>>
make_engram_prefetcher(storage::IoEngine& io,
                       std::span<const storage::File* const> engram_files,
                       const EngramPrefetchConfig& cfg);

}  // namespace deepmoe::store
