// The opened safetensors shards, indexed exactly as the manifest indexes them.
//
// Since design §5.1 v0.5 there is no repack: an IoRequest names one of the 48
// original shards by the integer index the manifest's `files` array assigns.
// ShardSet is the mapping from that index to an open storage::File. It lives in
// store/ rather than storage/ because storage deliberately knows nothing about
// models or manifests (docs/architecture.md §1.1); store already depends on
// both.
//
// A Win32 handle can be associated with exactly one IOCP port for its whole
// lifetime (storage/backend.h), so a ShardSet handed to one IoEngine must not
// be handed to another -- open a second one instead.
//
// Ownership/threading: owns its Files and closes them on destruction. `at()` is
// const and lock-free; open_all() is not thread-safe and is called once at
// startup.
#pragma once

#include <format>
#include <string>
#include <vector>

#include "core/status.h"
#include "model/manifest.h"
#include "storage/file.h"

namespace deepmoe::store {

class ShardSet {
public:
    ShardSet() = default;
    ShardSet(ShardSet&&) = default;
    ShardSet& operator=(ShardSet&&) = default;
    ShardSet(const ShardSet&) = delete;
    ShardSet& operator=(const ShardSet&) = delete;

    // Opens every file the manifest lists, in index order. `dir` is the model
    // directory the manifest's relative paths hang off. The size of each file is
    // checked against the manifest, so a truncated or swapped shard is caught
    // here rather than as garbage weights later.
    Result<void> open_all(const std::string& dir, const Manifest& manifest,
                          bool unbuffered = true) {
        files_.clear();
        files_.reserve(manifest.files().size());
        for (uint32_t i = 0; i < manifest.files().size(); ++i) {
            const FileEntry& e = manifest.files()[i];
            storage::FileFlags flags = storage::FileFlags::Overlapped | storage::FileFlags::Random;
            if (unbuffered) flags = flags | storage::FileFlags::Unbuffered;
            auto f = storage::File::open(join(dir, e.path), flags);
            if (!f) return fail(f.error().code,
                                std::format("shard {} ('{}'): {}", i, e.path, f.error().message),
                                f.error().os_code);
            if (e.bytes && f->size() != e.bytes)
                return fail(Err::Corrupt,
                            std::format("shard {} ('{}') is {} B, the manifest says {}",
                                        i, e.path, f->size(), e.bytes));
            if (f->sector_size() > kPageSize)
                return fail(Err::FailedPrecondition,
                            std::format("'{}' lives on a volume with {} B sectors; the manifest "
                                        "is built for 4 KiB", e.path, f->sector_size()));
            files_.push_back(*std::move(f));
        }
        return {};
    }

    void close() { files_.clear(); }

    size_t size() const { return files_.size(); }
    bool   empty() const { return files_.empty(); }

    const storage::File* at(uint32_t index) const {
        return index < files_.size() ? &files_[index] : nullptr;
    }
    Result<const storage::File*> require(uint32_t index) const {
        if (const storage::File* f = at(index)) return f;
        return fail(Err::OutOfRange,
                    std::format("shard index {} but only {} are open", index, files_.size()));
    }

    uint64_t total_bytes() const {
        uint64_t n = 0;
        for (const storage::File& f : files_) n += f.size();
        return n;
    }

    static std::string join(const std::string& dir, const std::string& name) {
        if (dir.empty()) return name;
        const char last = dir.back();
        if (last == '/' || last == '\\') return dir + name;
#if defined(_WIN32)
        return dir + "\\" + name;
#else
        return dir + "/" + name;
#endif
    }

private:
    std::vector<storage::File> files_;
};

}  // namespace deepmoe::store
