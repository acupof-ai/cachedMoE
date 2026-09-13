// POSIX implementation of storage/file.h, used by the Linux CI build.
// O_DIRECT is the counterpart of FILE_FLAG_NO_BUFFERING: same 4 KiB alignment
// contract on offset, length and destination pointer (design §9.6).
#if !defined(_WIN32)

#include "storage/file.h"

#include <cerrno>
#include <cstring>
#include <format>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include "core/align.h"

namespace deepmoe::storage {
namespace {

std::unexpected<Status> errno_err(std::string what) {
    const int e = errno;
    return fail(Err::Io, std::format("{}: {}", std::move(what), std::strerror(e)),
                static_cast<uint32_t>(e));
}

}  // namespace

Result<uint32_t> volume_sector_size(const std::string& path) {
    struct statvfs st {};
    // Fall back to the containing directory when the file itself is absent.
    if (::statvfs(path.c_str(), &st) != 0) {
        const size_t slash = path.find_last_of("/\\");
        const std::string dir = slash == std::string::npos ? std::string(".") : path.substr(0, slash);
        if (::statvfs(dir.c_str(), &st) != 0) return errno_err("statvfs");
    }
    const auto bs = static_cast<uint32_t>(st.f_bsize ? st.f_bsize : 4096);
    // O_DIRECT alignment is the *logical block* size, which is at most f_bsize
    // and in practice 512 or 4096; 4 KiB always satisfies both.
    return bs > 4096u ? 4096u : bs;
}

Result<File> File::open(const std::string& path, FileFlags flags) {
    const bool write  = has_flag(flags, FileFlags::Write) || has_flag(flags, FileFlags::Create);
    const bool create = has_flag(flags, FileFlags::Create);

    int oflags = write ? O_RDWR : O_RDONLY;
    if (create) oflags |= O_CREAT;
#if defined(O_DIRECT)
    if (has_flag(flags, FileFlags::Unbuffered)) oflags |= O_DIRECT;
#endif
#if defined(O_CLOEXEC)
    oflags |= O_CLOEXEC;
#endif

    const int fd = ::open(path.c_str(), oflags, 0644);
    if (fd < 0) return errno_err(std::format("open('{}')", path));

    File f;
    f.handle_     = fd;
    f.path_       = path;
    f.open_       = true;
    f.unbuffered_ = has_flag(flags, FileFlags::Unbuffered);

    struct stat st {};
    if (::fstat(fd, &st) != 0) { auto e = errno_err("fstat"); ::close(fd); f.handle_ = invalid_handle(); f.open_ = false; return e; }
    f.size_ = static_cast<uint64_t>(st.st_size);

    auto ss = volume_sector_size(path);
    f.sector_size_ = ss ? *ss : 4096u;
    if (f.unbuffered_ && f.sector_size_ > kPageSize) {
        ::close(fd);
        f.handle_ = invalid_handle();
        f.open_ = false;
        return fail(Err::FailedPrecondition,
                    std::format("block size {} exceeds the 4 KiB layout of '{}'", f.sector_size_, path));
    }

#if defined(POSIX_FADV_SEQUENTIAL)
    if (!f.unbuffered_) {
        if (has_flag(flags, FileFlags::Sequential)) ::posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
        if (has_flag(flags, FileFlags::Random))     ::posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
    }
#endif
    return f;
}

void File::close() {
    if (open_ && handle_ >= 0) ::close(handle_);
    handle_ = invalid_handle();
    open_   = false;
}

Result<void> File::refresh_size() {
    if (!open_) return fail(Err::FailedPrecondition, "file is not open");
    struct stat st {};
    if (::fstat(handle_, &st) != 0) return errno_err("fstat");
    size_ = static_cast<uint64_t>(st.st_size);
    return {};
}

Result<size_t> File::read_at(uint64_t offset, MutBytes dst) const {
    if (!open_) return fail(Err::FailedPrecondition, "file is not open");
    if (unbuffered_ && (!is_aligned(offset, sector_size_) || !is_aligned(dst.data(), sector_size_) ||
                        !is_aligned(dst.size(), sector_size_)))
        return fail(Err::InvalidArgument, "unbuffered read is not sector-aligned");
    size_t done = 0;
    while (done < dst.size()) {
        const ssize_t n = ::pread(handle_, dst.data() + done, dst.size() - done,
                                  static_cast<off_t>(offset + done));
        if (n < 0) { if (errno == EINTR) continue; return errno_err("pread"); }
        if (n == 0) break;                       // EOF
        done += static_cast<size_t>(n);
        // O_DIRECT never returns a partial transfer that would misalign the
        // next one, so the loop stays aligned.
        if (unbuffered_) break;
    }
    return done;
}

Result<size_t> File::write_at(uint64_t offset, ByteSpan src) const {
    if (!open_) return fail(Err::FailedPrecondition, "file is not open");
    if (unbuffered_ && (!is_aligned(offset, sector_size_) || !is_aligned(src.data(), sector_size_) ||
                        !is_aligned(src.size(), sector_size_)))
        return fail(Err::InvalidArgument, "unbuffered write is not sector-aligned");
    size_t done = 0;
    while (done < src.size()) {
        const ssize_t n = ::pwrite(handle_, src.data() + done, src.size() - done,
                                   static_cast<off_t>(offset + done));
        if (n < 0) { if (errno == EINTR) continue; return errno_err("pwrite"); }
        if (n == 0) break;
        done += static_cast<size_t>(n);
        if (unbuffered_) break;
    }
    return done;
}

Result<void> File::set_size(uint64_t bytes) {
    if (!open_) return fail(Err::FailedPrecondition, "file is not open");
#if defined(__linux__)
    // Allocate real extents so a benchmark measures the drive, not a hole.
    if (::posix_fallocate(handle_, 0, static_cast<off_t>(bytes)) != 0 &&
        ::ftruncate(handle_, static_cast<off_t>(bytes)) != 0)
        return errno_err("fallocate/ftruncate");
#else
    if (::ftruncate(handle_, static_cast<off_t>(bytes)) != 0) return errno_err("ftruncate");
#endif
    size_ = bytes;
    return {};
}

}  // namespace deepmoe::storage

#endif  // !_WIN32
