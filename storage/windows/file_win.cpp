// Win32 implementation of storage/file.h.
//
// The runtime opens every weight blob with
// FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED (design §9.6): no page cache
// (the 510 GB working set would only evict itself) and asynchronous completion
// through the IOCP backend.
#if defined(_WIN32)

#include "storage/file.h"

#include <format>
#include <string>
#include <vector>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "core/align.h"

namespace deepmoe::storage {
namespace {

HANDLE H(NativeHandle h) { return static_cast<HANDLE>(h); }

std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    int n = ::MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(static_cast<size_t>(n), L'\0');
    ::MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
    return w;
}

std::unexpected<Status> win_err(std::string what) {
    const DWORD e = ::GetLastError();
    return fail(Err::Io, std::move(what), static_cast<uint32_t>(e));
}

// Volume root of a path ("C:\\Users\\x" -> "C:\\"); empty for a relative path,
// which GetDiskFreeSpaceW then resolves against the current directory.
std::wstring volume_root(const std::string& path) {
    std::wstring w = widen(path);
    if (w.size() >= 2 && w[1] == L':') return w.substr(0, 2) + L"\\";
    if (w.size() >= 2 && (w[0] == L'\\' || w[0] == L'/') && (w[1] == L'\\' || w[1] == L'/')) {
        // \\server\share\...
        size_t p = w.find_first_of(L"\\/", 2);
        if (p == std::wstring::npos) return w + L"\\";
        p = w.find_first_of(L"\\/", p + 1);
        return (p == std::wstring::npos ? w : w.substr(0, p)) + L"\\";
    }
    return {};
}

}  // namespace

Result<uint32_t> volume_sector_size(const std::string& path) {
    DWORD spc = 0, bps = 0, freeClusters = 0, totalClusters = 0;
    std::wstring root = volume_root(path);
    if (!::GetDiskFreeSpaceW(root.empty() ? nullptr : root.c_str(),
                             &spc, &bps, &freeClusters, &totalClusters))
        return win_err(std::format("GetDiskFreeSpaceW('{}')", path));
    return bps ? static_cast<uint32_t>(bps) : 512u;
}

Result<File> File::open(const std::string& path, FileFlags flags) {
    const bool write  = has_flag(flags, FileFlags::Write) || has_flag(flags, FileFlags::Create);
    const bool create = has_flag(flags, FileFlags::Create);

    DWORD access = GENERIC_READ | (write ? GENERIC_WRITE : 0u);
    DWORD attrs  = FILE_ATTRIBUTE_NORMAL;
    if (has_flag(flags, FileFlags::Unbuffered)) attrs |= FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH;
    if (has_flag(flags, FileFlags::Overlapped)) attrs |= FILE_FLAG_OVERLAPPED;
    if (has_flag(flags, FileFlags::Sequential)) attrs |= FILE_FLAG_SEQUENTIAL_SCAN;
    if (has_flag(flags, FileFlags::Random))     attrs |= FILE_FLAG_RANDOM_ACCESS;

    HANDLE h = ::CreateFileW(widen(path).c_str(), access,
                             FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                             create ? OPEN_ALWAYS : OPEN_EXISTING, attrs, nullptr);
    if (h == INVALID_HANDLE_VALUE) return win_err(std::format("CreateFileW('{}')", path));

    File f;
    f.handle_ = h;
    f.path_   = path;
    f.open_   = true;
    f.unbuffered_ = has_flag(flags, FileFlags::Unbuffered);

    LARGE_INTEGER li{};
    if (!::GetFileSizeEx(h, &li)) { auto e = win_err("GetFileSizeEx"); ::CloseHandle(h); f.handle_ = invalid_handle(); f.open_ = false; return e; }
    f.size_ = static_cast<uint64_t>(li.QuadPart);

    auto ss = volume_sector_size(path);
    f.sector_size_ = ss ? *ss : 4096u;
    // The repacked layout of design §5.1 is built on 4 KiB; a 4K-native volume
    // with a larger physical sector would need repack.py re-run.
    if (f.unbuffered_ && f.sector_size_ > kPageSize) {
        ::CloseHandle(h);
        f.handle_ = invalid_handle();
        f.open_ = false;
        return fail(Err::FailedPrecondition,
                    std::format("volume sector size {} exceeds the 4 KiB layout of '{}'",
                                f.sector_size_, path));
    }
    return f;
}

void File::close() {
    if (open_ && H(handle_) != INVALID_HANDLE_VALUE) ::CloseHandle(H(handle_));
    handle_ = invalid_handle();
    open_   = false;
}

Result<void> File::refresh_size() {
    if (!open_) return fail(Err::FailedPrecondition, "file is not open");
    LARGE_INTEGER li{};
    if (!::GetFileSizeEx(H(handle_), &li)) return win_err("GetFileSizeEx");
    size_ = static_cast<uint64_t>(li.QuadPart);
    return {};
}

Result<size_t> File::read_at(uint64_t offset, MutBytes dst) const {
    if (!open_) return fail(Err::FailedPrecondition, "file is not open");
    if (unbuffered_ && (!is_aligned(offset, sector_size_) || !is_aligned(dst.data(), sector_size_) ||
                        !is_aligned(dst.size(), sector_size_)))
        return fail(Err::InvalidArgument, "unbuffered read is not sector-aligned");

    // The handle is FILE_FLAG_OVERLAPPED, so even a "synchronous" read needs an
    // OVERLAPPED with an event to wait on.
    OVERLAPPED ov{};
    ov.Offset     = static_cast<DWORD>(offset & 0xFFFFFFFFull);
    ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
    HANDLE ev = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!ev) return win_err("CreateEventW");
    ov.hEvent = ev;

    DWORD moved = 0;
    BOOL ok = ::ReadFile(H(handle_), dst.data(), static_cast<DWORD>(dst.size()), &moved, &ov);
    if (!ok) {
        const DWORD e = ::GetLastError();
        if (e == ERROR_IO_PENDING) {
            ok = ::GetOverlappedResult(H(handle_), &ov, &moved, TRUE);
        } else if (e == ERROR_HANDLE_EOF) {
            ::CloseHandle(ev);
            return size_t{0};
        } else {
            ::SetLastError(e);
            auto err = win_err(std::format("ReadFile('{}', off {}, {} B)", path_, offset, dst.size()));
            ::CloseHandle(ev);
            return err;
        }
    }
    ::CloseHandle(ev);
    if (!ok) return win_err("GetOverlappedResult");
    return static_cast<size_t>(moved);
}

Result<size_t> File::write_at(uint64_t offset, ByteSpan src) const {
    if (!open_) return fail(Err::FailedPrecondition, "file is not open");
    if (unbuffered_ && (!is_aligned(offset, sector_size_) || !is_aligned(src.data(), sector_size_) ||
                        !is_aligned(src.size(), sector_size_)))
        return fail(Err::InvalidArgument, "unbuffered write is not sector-aligned");

    OVERLAPPED ov{};
    ov.Offset     = static_cast<DWORD>(offset & 0xFFFFFFFFull);
    ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
    HANDLE ev = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!ev) return win_err("CreateEventW");
    ov.hEvent = ev;

    DWORD moved = 0;
    BOOL ok = ::WriteFile(H(handle_), src.data(), static_cast<DWORD>(src.size()), &moved, &ov);
    if (!ok) {
        const DWORD e = ::GetLastError();
        if (e == ERROR_IO_PENDING) {
            ok = ::GetOverlappedResult(H(handle_), &ov, &moved, TRUE);
        } else {
            ::SetLastError(e);
            auto err = win_err(std::format("WriteFile('{}', off {}, {} B)", path_, offset, src.size()));
            ::CloseHandle(ev);
            return err;
        }
    }
    ::CloseHandle(ev);
    if (!ok) return win_err("GetOverlappedResult");
    return static_cast<size_t>(moved);
}

Result<void> File::set_size(uint64_t bytes) {
    if (!open_) return fail(Err::FailedPrecondition, "file is not open");
    FILE_END_OF_FILE_INFO eof{};
    eof.EndOfFile.QuadPart = static_cast<LONGLONG>(bytes);
    if (!::SetFileInformationByHandle(H(handle_), FileEndOfFileInfo, &eof, sizeof eof))
        return win_err(std::format("SetFileInformationByHandle('{}', {} B)", path_, bytes));
    // Mark it valid so the extent is not a sparse zero range; without this the
    // first read of a fresh benchmark file measures the sparse-file fast path
    // rather than the drive.
    FILE_ALLOCATION_INFO alloc{};
    alloc.AllocationSize.QuadPart = static_cast<LONGLONG>(bytes);
    ::SetFileInformationByHandle(H(handle_), FileAllocationInfo, &alloc, sizeof alloc);
    size_ = bytes;
    return {};
}

}  // namespace deepmoe::storage

#endif  // _WIN32
