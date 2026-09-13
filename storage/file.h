// Opened-file abstraction for the NVMe tier. Platform-neutral interface; the
// implementation lives in storage/windows/file_win.cpp and
// storage/linux/file_posix.cpp.
//
// Unbuffered mode is the point of this class: FILE_FLAG_NO_BUFFERING on Windows
// and O_DIRECT on Linux both require the file offset, the transfer length and
// the destination pointer to be multiples of the volume's sector size
// (design §9.6). `sector_size()` reports what the volume actually demands;
// the repacked layout of design §5.1 guarantees 4 KiB everywhere, so the
// runtime asserts sector_size() <= 4096 at open time.
//
// Ownership/threading: File owns the OS handle and closes it in the destructor;
// move-only, never copied. A File is safe to share between threads for
// positional reads -- neither read_at nor the async backends touch a shared
// file pointer.
#pragma once

#include <cstdint>
#include <string>

#include "core/bytes.h"
#include "core/status.h"

namespace deepmoe::storage {

#if defined(_WIN32)
using NativeHandle = void*;                       // HANDLE
inline NativeHandle invalid_handle() { return reinterpret_cast<void*>(~static_cast<uintptr_t>(0)); }
#else
using NativeHandle = int;                         // fd
inline NativeHandle invalid_handle() { return -1; }
#endif

enum class FileFlags : uint32_t {
    None       = 0,
    Unbuffered = 1u << 0,   // FILE_FLAG_NO_BUFFERING / O_DIRECT
    Overlapped = 1u << 1,   // FILE_FLAG_OVERLAPPED; required by the IOCP backend
    Sequential = 1u << 2,   // hint: prefill's expert-major stream (design §9.7)
    Random     = 1u << 3,   // hint: decode's per-expert and engram-row reads
    Write      = 1u << 4,   // open for read+write
    Create     = 1u << 5,   // create if absent (implies Write)
};

constexpr FileFlags operator|(FileFlags a, FileFlags b) {
    return static_cast<FileFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
constexpr bool has_flag(FileFlags v, FileFlags f) {
    return (static_cast<uint32_t>(v) & static_cast<uint32_t>(f)) != 0;
}

class File {
public:
    File() = default;
    ~File() { close(); }

    File(File&& o) noexcept { swap(o); }
    File& operator=(File&& o) noexcept { if (this != &o) { close(); swap(o); } return *this; }
    File(const File&) = delete;
    File& operator=(const File&) = delete;

    static Result<File> open(const std::string& path, FileFlags flags);
    static Result<File> open_read(const std::string& path, bool unbuffered = true) {
        FileFlags f = FileFlags::Overlapped;
        if (unbuffered) f = f | FileFlags::Unbuffered;
        return open(path, f);
    }

    void close();
    bool is_open() const { return open_; }

    NativeHandle       native() const { return handle_; }
    const std::string& path()   const { return path_; }
    uint64_t           size()   const { return size_; }
    uint32_t    sector_size()   const { return sector_size_; }
    bool         unbuffered()   const { return unbuffered_; }

    // Synchronous positional read. With unbuffered() the offset, dst pointer
    // and dst size must all be sector-aligned; the call returns
    // InvalidArgument rather than letting the OS fail obscurely.
    Result<size_t> read_at(uint64_t offset, MutBytes dst) const;

    // Synchronous positional write; same alignment rules. Used by the
    // benchmark harness to lay down its test file, not by the runtime.
    Result<size_t> write_at(uint64_t offset, ByteSpan src) const;

    // Grows the file to `bytes` without writing content. Fast on NTFS when the
    // caller has SeManageVolumePrivilege, a normal zero-fill otherwise.
    Result<void> set_size(uint64_t bytes);

    // Re-reads the OS size after external growth.
    Result<void> refresh_size();

private:
    void swap(File& o) noexcept {
        std::swap(handle_, o.handle_); std::swap(path_, o.path_);
        std::swap(size_, o.size_); std::swap(sector_size_, o.sector_size_);
        std::swap(open_, o.open_); std::swap(unbuffered_, o.unbuffered_);
    }

    NativeHandle handle_ = invalid_handle();
    std::string  path_;
    uint64_t     size_        = 0;
    uint32_t     sector_size_ = 4096;
    bool         open_        = false;
    bool         unbuffered_  = false;
};

// Sector size of the volume holding `path`, without opening a file. Used by the
// startup check that the repacked 4 KiB layout is legal on this volume.
Result<uint32_t> volume_sector_size(const std::string& path);

// Deletes a file, ignoring "already gone". Portable (std::remove).
Result<void> remove_file(const std::string& path);

// Directory for scratch files; honours TEMP/TMPDIR. Used by nvme_bench.
std::string temp_dir();

}  // namespace deepmoe::storage
