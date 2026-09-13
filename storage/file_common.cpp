// Portable parts of storage/file.h. The open/read/write bodies are in
// storage/windows/file_win.cpp and storage/linux/file_posix.cpp.
#include "storage/file.h"

#include <cstdio>
#include <cstdlib>

namespace deepmoe::storage {

Result<void> remove_file(const std::string& path) {
    if (std::remove(path.c_str()) == 0) return {};
    // Gone already is success; anything else is a real failure.
    if (std::FILE* f = std::fopen(path.c_str(), "rb")) { std::fclose(f); return fail(Err::Io, "cannot remove " + path); }
    return {};
}

std::string temp_dir() {
    for (const char* v : {"TEMP", "TMP", "TMPDIR"}) {
        if (const char* p = std::getenv(v); p && *p) return p;
    }
#if defined(_WIN32)
    return "C:\\Windows\\Temp";
#else
    return "/tmp";
#endif
}

}  // namespace deepmoe::storage
