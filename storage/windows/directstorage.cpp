// Optional DirectStorage backend (design §9.6).
//
// DirectStorage would let the runtime hand the drive a batch of reads whose
// destinations are GPU resources, skipping the CPU entirely. It is behind
// DEEPMOE_ENABLE_DIRECTSTORAGE and compiles only when <dstorage.h> is on the
// include path -- the SDK is a NuGet package this repo deliberately does not
// vendor or download. Without it the factory reports Unavailable and
// make_default_backend falls back to IOCP, which is the P-1 baseline anyway.
//
// Ownership/threading: same contract as the IOCP backend -- submit and poll
// from the IoEngine dispatcher thread, internal completion handled by the
// DirectStorage runtime's own threads.
#if defined(_WIN32)

#include "storage/backend.h"

#if defined(DEEPMOE_ENABLE_DIRECTSTORAGE) && defined(DEEPMOE_HAVE_DSTORAGE_H)
#include <dstorage.h>
#endif

namespace deepmoe::storage {

#if defined(DEEPMOE_ENABLE_DIRECTSTORAGE) && defined(DEEPMOE_HAVE_DSTORAGE_H)

// TODO(design §9.6): implement.
//   1. DStorageGetFactory -> IDStorageFactory, SetStagingBufferSize.
//   2. One IDStorageQueue per priority class (DSTORAGE_PRIORITY_REALTIME for
//      P0 blocking misses, NORMAL for lookahead, LOW for backfill) so the
//      preemption of §9.6 is expressed to the runtime rather than emulated.
//   3. DSTORAGE_REQUEST_SOURCE_FILE -> DSTORAGE_REQUEST_DESTINATION_BUFFER
//      pointing at the mapped slab slot; EnqueueRequest + Submit per batch.
//   4. An ID3D12Fence per batch drives poll(); the Vulkan side of that
//      interop is the open question this backend has to answer before it can
//      beat plain IOCP on a single SN740.
Result<std::unique_ptr<Backend>> make_directstorage_backend(const IoConfig&) {
    return fail(Err::Unimplemented, "DirectStorage backend (design §9.6)");
}

#else

Result<std::unique_ptr<Backend>> make_directstorage_backend(const IoConfig&) {
    return fail(Err::Unavailable,
                "DirectStorage is not compiled in "
                "(needs -DDEEPMOE_ENABLE_DIRECTSTORAGE and dstorage.h on the include path)");
}

#endif

}  // namespace deepmoe::storage

#endif  // _WIN32
