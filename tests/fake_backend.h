// An in-memory storage::Backend for testing the IoEngine's chunking, priority
// ordering and preemption without touching a drive.
//
// It records the exact submit order so a test can assert that a P0 arriving
// mid-stream really does jump the queue (design §9.6), and it can be told to
// hold completions until the test releases them, which is how the "P0 preempts
// P1" case is made deterministic.
//
// Ownership/threading: the IoEngine owns it and calls submit/poll from its
// dispatcher thread. `log()` and `release_all()` are called from the test
// thread, so both are mutex-guarded.
#pragma once

#include <chrono>
#include <cstring>
#include <deque>
#include <mutex>
#include <vector>

#include "storage/backend.h"

namespace deepmoe::test {

struct SubmitRecord {
    uint64_t chunk_id;
    uint64_t file_off;
    uint32_t bytes;
};

class FakeBackend final : public storage::Backend {
public:
    // `content` is the whole virtual file; a chunk read at [off, off+bytes)
    // copies from it, so a test can verify the bytes actually landed.
    explicit FakeBackend(std::vector<std::byte> content, uint32_t queue_depth = 8)
        : content_(std::move(content)) {
        caps_.name = "fake";
        caps_.alignment = 4096;
        caps_.max_queue_depth = queue_depth;
        caps_.max_chunk_bytes = 64u << 20;
        caps_.unbuffered = false;
    }

    const storage::BackendCaps& caps() const override { return caps_; }

    Result<void> submit(const storage::ChunkRequest& req) override {
        std::lock_guard lk(m_);
        if (pending_.size() + held_.size() >= caps_.max_queue_depth)
            return fail(Err::ResourceExhausted, "fake queue full");
        log_.push_back(SubmitRecord{req.chunk_id, req.file_off, req.bytes});

        storage::ChunkCompletion c;
        c.chunk_id = req.chunk_id;
        if (req.file_off + req.bytes > content_.size()) {
            c.status = Status{Err::OutOfRange, "read past end of fake file"};
        } else {
            std::memcpy(req.dst, content_.data() + req.file_off, req.bytes);
            c.bytes_moved = req.bytes;
        }
        if (hold_) held_.push_back(c); else pending_.push_back(c);
        return {};
    }

    Result<size_t> poll(std::span<storage::ChunkCompletion> out,
                                 std::chrono::milliseconds) override {
        std::lock_guard lk(m_);
        size_t n = 0;
        while (n < out.size() && !pending_.empty()) {
            out[n++] = pending_.front();
            pending_.pop_front();
        }
        return n;
    }

    uint32_t inflight() const override {
        std::lock_guard lk(m_);
        return static_cast<uint32_t>(pending_.size() + held_.size());
    }

    // --- test controls ---
    void hold_completions(bool on) { std::lock_guard lk(m_); hold_ = on; }
    void release_all() {
        std::lock_guard lk(m_);
        while (!held_.empty()) { pending_.push_back(held_.front()); held_.pop_front(); }
        hold_ = false;
    }
    std::vector<SubmitRecord> log() const { std::lock_guard lk(m_); return log_; }
    size_t submit_count() const { std::lock_guard lk(m_); return log_.size(); }

private:
    mutable std::mutex m_;
    storage::BackendCaps caps_;
    std::vector<std::byte> content_;
    std::deque<storage::ChunkCompletion> pending_, held_;
    std::vector<SubmitRecord> log_;
    bool hold_ = false;
};

}  // namespace deepmoe::test
