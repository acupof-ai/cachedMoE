// Abstract async-read backend. IoEngine owns the policy (priority, chunking,
// queue depth); a Backend owns only "get these bytes into this buffer and tell
// me when". Three implementations exist:
//
//   storage/windows/iocp.cpp          FILE_FLAG_NO_BUFFERING|OVERLAPPED + IOCP
//   storage/windows/directstorage.cpp DirectStorage, optional (design §9.6)
//   storage/linux/io_uring.cpp        raw io_uring + O_DIRECT, for CI
//
// Ownership/threading: a Backend is owned by the IoEngine. `submit` is called
// only from the engine's dispatcher thread; `poll` only from the same thread.
// A backend may use internal completion threads (IOCP does, two of them by
// default) but must hand results back through `poll` so the engine keeps a
// single-threaded view of request state.
//
// File lifetime: a Win32 handle can be associated with exactly ONE completion
// port for its whole lifetime, and the association cannot be undone -- not even
// by closing the port. A File handed to one IoEngine therefore must not be
// handed to another; reopen it instead. bench/nvme_bench.cpp does exactly that
// between measurement points.
#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

#include "core/config.h"
#include "core/status.h"
#include "storage/file.h"

namespace deepmoe::storage {

// One 4 KiB-aligned transfer. The engine never issues a chunk larger than
// BackendCaps::max_chunk_bytes.
struct ChunkRequest {
    uint64_t chunk_id = 0;      // engine-assigned, unique while in flight
    const File* file  = nullptr;  // borrowed; outlives the transfer
    uint64_t file_off = 0;      // sector-aligned
    uint32_t bytes    = 0;      // sector multiple
    void*    dst      = nullptr;  // sector-aligned; slab slot or staging buffer
};

struct ChunkCompletion {
    uint64_t chunk_id    = 0;
    uint32_t bytes_moved = 0;
    Status   status{Err::Ok};
    bool ok() const { return status.code == Err::Ok; }
};

struct BackendCaps {
    std::string name;
    uint32_t alignment        = 4096;
    uint32_t max_queue_depth  = 64;
    uint32_t max_chunk_bytes  = 32u << 20;
    bool     unbuffered       = true;
    // True when the backend can land bytes straight in GPU-visible memory with
    // no bounce buffer -- the zero-copy requirement of design §9.6.
    bool     direct_to_device_memory = true;
};

class Backend {
public:
    virtual ~Backend() = default;

    virtual const BackendCaps& caps() const = 0;

    // Queues one chunk. ResourceExhausted when the backend's own queue is full;
    // the engine then polls and retries rather than blocking.
    virtual Result<void> submit(const ChunkRequest& req) = 0;

    // Drains finished chunks into `out`, waiting at most `timeout` for the
    // first one. Returns how many were written.
    virtual Result<size_t> poll(std::span<ChunkCompletion> out,
                                std::chrono::milliseconds timeout) = 0;

    virtual uint32_t inflight() const = 0;

    // Best-effort cancel of everything queued; completions still arrive, with
    // Err::Cancelled. Called on engine shutdown.
    virtual void cancel_all() {}
};

// --- factories ---------------------------------------------------------------
// Each returns Unavailable when the platform or the optional SDK is missing, so
// the caller can fall back without #ifdef.

Result<std::unique_ptr<Backend>> make_iocp_backend(const IoConfig& cfg);
Result<std::unique_ptr<Backend>> make_directstorage_backend(const IoConfig& cfg);
Result<std::unique_ptr<Backend>> make_io_uring_backend(const IoConfig& cfg);

// Picks the best available backend for the host: DirectStorage if enabled and
// present, else IOCP on Windows, else io_uring on Linux.
Result<std::unique_ptr<Backend>> make_default_backend(const IoConfig& cfg);

}  // namespace deepmoe::storage
