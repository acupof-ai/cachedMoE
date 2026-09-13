#include "store/engram_prefetch.h"

namespace deepmoe::store {

// TODO(design §7.10): the implementation needs (a) the exact compressed-vocab
// mapping and multiply-XOR hash from inference/model.py, verified against the
// L0 oracle, and (b) a GPU-visible ring for the 264 B rows. Until then the
// runtime simply has no engram path, which design §15 places in P2/P3.
Result<std::unique_ptr<EngramPrefetcher>>
make_engram_prefetcher(storage::IoEngine&,
                       std::span<const storage::File* const>,
                       const EngramPrefetchConfig&) {
    return unimplemented("store::make_engram_prefetcher (design §7.10)");
}

}  // namespace deepmoe::store
