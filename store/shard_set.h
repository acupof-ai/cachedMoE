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

    // --- Track D2 (docs/p4_dual_source.md): mirrors -------------------------
    // A mirror is another directory holding byte-identical copies of some or
    // all of the same shards -- on another physical drive, which is the whole
    // point. A shard whose mirror copy is missing, short or long is simply not
    // mirrored: the run still works, it just reads that shard from the primary.
    // Nothing here ever writes to the mirror root.
    struct Mirror {
        std::string root;
        std::vector<storage::File> files;   // one entry per manifest file index
        std::vector<uint8_t>       present; // 0 when that index is not mirrored
        uint32_t                   n_present = 0;
        uint64_t                   bytes = 0;
    };

    Result<void> open_mirror(const std::string& dir, const Manifest& manifest,
                             bool unbuffered = true) {
        if (files_.empty())
            return fail(Err::FailedPrecondition, "open_mirror before open_all");
        if (mirrors_.size() + 1 >= 4)
            return fail(Err::OutOfRange, "at most three mirrors");
        Mirror m;
        m.root = dir;
        m.files.resize(manifest.files().size());
        m.present.assign(manifest.files().size(), 0);
        for (uint32_t i = 0; i < manifest.files().size(); ++i) {
            const FileEntry& e = manifest.files()[i];
            storage::FileFlags flags = storage::FileFlags::Overlapped | storage::FileFlags::Random;
            if (unbuffered) flags = flags | storage::FileFlags::Unbuffered;
            auto f = storage::File::open(join(dir, e.path), flags);
            if (!f) continue;                       // not mirrored: fine
            if (f->size() != files_[i].size()) continue;   // not the same bytes: skip it
            if (f->sector_size() > kPageSize) continue;
            m.bytes += f->size();
            m.files[i] = *std::move(f);
            m.present[i] = 1;
            ++m.n_present;
        }
        if (m.n_present == 0)
            return fail(Err::NotFound,
                        std::format("mirror '{}' holds none of the {} shards at the right size",
                                    dir, files_.size()));
        mirrors_.push_back(std::move(m));
        return {};
    }

    size_t mirror_count() const { return mirrors_.size(); }
    const Mirror& mirror(size_t i) const { return mirrors_[i]; }
    // Source 0 is the primary; 1..mirror_count() are the mirrors. Null when
    // that source does not hold the shard.
    const storage::File* at(uint32_t index, uint32_t source) const {
        if (source == 0) return at(index);
        const size_t m = source - 1;
        if (m >= mirrors_.size() || index >= mirrors_[m].files.size()) return nullptr;
        return mirrors_[m].present[index] ? &mirrors_[m].files[index] : nullptr;
    }

    void close() { files_.clear(); mirrors_.clear(); }

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
    std::vector<Mirror>        mirrors_;
};

}  // namespace deepmoe::store
