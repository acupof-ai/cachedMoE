// dx_probe -- reproduce the deepmoe engine's I/O SHAPE without the engine.
//
// N handles (FILE_FLAG_NO_BUFFERING|FILE_FLAG_OVERLAPPED) on one IOCP,
// QD-limited 4 MiB random reads, optionally interleaved with 12 KiB reads
// (the `norm.weight` pinned-load shape).  Logs GB/s every 5 s.  Stops on the
// FIRST error and prints the exact Win32 code / length / offset.  A watchdog
// declares a stall if no completion arrives for `--stall-s` seconds.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <random>
#include <chrono>

static const uint32_t kPage = 4096;
static uint64_t align_down(uint64_t v) { return v & ~uint64_t(kPage - 1); }

struct Op {
    OVERLAPPED ov{};
    uint32_t   bytes = 0;
    uint64_t   off   = 0;
    int        h     = 0;
    void*      dst   = nullptr;
};

static std::atomic<uint64_t> g_bytes{0};
static std::atomic<uint64_t> g_reads{0};
static std::atomic<uint64_t> g_small{0};
static std::atomic<uint32_t> g_inflight{0};
static std::atomic<int>      g_stop{0};          // 1 = time up, 2 = error, 3 = stall
static std::atomic<uint64_t> g_last_done_ms{0};
static DWORD  g_err = 0;  static uint64_t g_err_off = 0;  static uint32_t g_err_len = 0;
static std::string g_err_file;

static uint64_t now_ms() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Ctx {
    HANDLE port = nullptr;
    std::vector<HANDLE>      files;
    std::vector<uint64_t>    sizes;
    std::vector<std::string> names;
};

static Ctx g;

static void completion_loop() {
    for (;;) {
        DWORD moved = 0; ULONG_PTR key = 0; OVERLAPPED* ov = nullptr;
        BOOL ok = ::GetQueuedCompletionStatus(g.port, &moved, &key, &ov, 1000);
        if (!ov) { if (g_stop.load()) return; continue; }      // timeout or quit packet
        Op* op = reinterpret_cast<Op*>(ov);
        if (!ok) {
            DWORD e = ::GetLastError();
            int expected = 0;
            if (g_stop.compare_exchange_strong(expected, 2)) {
                g_err = e; g_err_off = op->off; g_err_len = op->bytes;
                g_err_file = g.names[op->h];
            }
        } else {
            g_bytes.fetch_add(moved);
            g_reads.fetch_add(1);
            if (op->bytes <= 64u * 1024u) g_small.fetch_add(1);
        }
        g_last_done_ms.store(now_ms());
        g_inflight.fetch_sub(1);
        delete op;
    }
}

int main(int argc, char** argv) {
    std::string dir, label = "run";
    uint32_t handles = 48, qd = 24, secs = 180, big_kb = 4096, small_kb = 12, every = 0, stall_s = 30;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if      (a == "--dir")      dir = nx();
        else if (a == "--handles")  handles = (uint32_t)atoi(nx().c_str());
        else if (a == "--qd")       qd = (uint32_t)atoi(nx().c_str());
        else if (a == "--secs")     secs = (uint32_t)atoi(nx().c_str());
        else if (a == "--big-kb")   big_kb = (uint32_t)atoi(nx().c_str());
        else if (a == "--small-kb") small_kb = (uint32_t)atoi(nx().c_str());
        else if (a == "--small-every") every = (uint32_t)atoi(nx().c_str());  // 0 = no small reads
        else if (a == "--stall-s")  stall_s = (uint32_t)atoi(nx().c_str());
        else if (a == "--label")    label = nx();
        else { fprintf(stderr, "unknown %s\n", a.c_str()); return 2; }
    }
    if (dir.empty()) { fprintf(stderr, "need --dir\n"); return 2; }

    // ---- open up to `handles` shards -------------------------------------
    WIN32_FIND_DATAA fd;
    HANDLE fh = ::FindFirstFileA((dir + "\\*.safetensors").c_str(), &fd);
    std::vector<std::string> paths;
    if (fh != INVALID_HANDLE_VALUE) {
        do { paths.push_back(dir + "\\" + fd.cFileName); } while (::FindNextFileA(fh, &fd));
        ::FindClose(fh);
    }
    if (paths.empty()) { fprintf(stderr, "no .safetensors under %s\n", dir.c_str()); return 2; }

    g.port = ::CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, 2);
    uint64_t t_open = now_ms();
    for (size_t i = 0; i < paths.size() && g.files.size() < handles; ++i) {
        HANDLE h = ::CreateFileA(paths[i].c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                 OPEN_EXISTING,
                                 FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS,
                                 nullptr);
        if (h == INVALID_HANDLE_VALUE) {
            printf("OPEN FAILED %s win32 %lu\n", paths[i].c_str(), ::GetLastError());
            return 3;
        }
        LARGE_INTEGER sz{}; ::GetFileSizeEx(h, &sz);
        if ((uint64_t)sz.QuadPart < (uint64_t)big_kb * 1024 * 2) { ::CloseHandle(h); continue; }
        ::CreateIoCompletionPort(h, g.port, (ULONG_PTR)g.files.size(), 0);
        g.files.push_back(h); g.sizes.push_back((uint64_t)sz.QuadPart); g.names.push_back(paths[i]);
    }
    printf("[%s] opened %zu handles in %llu ms (dir=%s)\n", label.c_str(), g.files.size(),
           (unsigned long long)(now_ms() - t_open), dir.c_str());
    printf("[%s] qd=%u big=%u KiB small=%u KiB every=%u secs=%u\n",
           label.c_str(), qd, big_kb, small_kb, every, secs);
    fflush(stdout);

    // ---- buffers ----------------------------------------------------------
    const uint32_t big_bytes   = big_kb * 1024;
    const uint32_t small_bytes = (uint32_t)((small_kb * 1024 + kPage - 1) / kPage * kPage);
    std::vector<void*> bufs(qd);
    for (uint32_t i = 0; i < qd; ++i)
        bufs[i] = ::VirtualAlloc(nullptr, big_bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);

    std::vector<std::thread> workers;
    for (int i = 0; i < 2; ++i) workers.emplace_back(completion_loop);

    std::mt19937_64 rng(0xD1A6ull);
    const uint64_t t0 = now_ms();
    g_last_done_ms.store(t0);
    uint64_t next_log = t0 + 5000, last_bytes = 0, last_ms = t0;
    uint64_t issued = 0;

    while (!g_stop.load()) {
        uint64_t t = now_ms();
        if (t - t0 >= (uint64_t)secs * 1000) { g_stop.store(1); break; }
        if (t >= next_log) {
            uint64_t b = g_bytes.load();
            double gbps = (b - last_bytes) / 1e9 / ((t - last_ms) / 1000.0);
            printf("[%s] t=%3llus  %.3f GB/s  reads=%llu small=%llu inflight=%u\n",
                   label.c_str(), (unsigned long long)((t - t0) / 1000), gbps,
                   (unsigned long long)g_reads.load(), (unsigned long long)g_small.load(),
                   g_inflight.load());
            fflush(stdout);
            last_bytes = b; last_ms = t; next_log = t + 5000;
        }
        const uint64_t ld = g_last_done_ms.load();
        if (t > ld && t - ld > (uint64_t)stall_s * 1000) { g_stop.store(3); break; }
        if (g_inflight.load() >= qd) { std::this_thread::yield(); continue; }

        const bool small = every && (issued % every == 0);
        const uint32_t len = small ? small_bytes : big_bytes;
        const int hi = (int)(rng() % g.files.size());
        const uint64_t span = g.sizes[hi] - len;
        const uint64_t off = align_down(rng() % span);

        Op* op = new Op();
        op->ov.Offset     = (DWORD)(off & 0xFFFFFFFFull);
        op->ov.OffsetHigh = (DWORD)(off >> 32);
        op->bytes = len; op->off = off; op->h = hi; op->dst = bufs[issued % qd];
        g_inflight.fetch_add(1);
        ++issued;
        DWORD moved = 0;
        if (!::ReadFile(g.files[hi], op->dst, len, &moved, &op->ov)) {
            DWORD e = ::GetLastError();
            if (e != ERROR_IO_PENDING) {
                g_inflight.fetch_sub(1);
                int expected = 0;
                if (g_stop.compare_exchange_strong(expected, 2)) {
                    g_err = e; g_err_off = off; g_err_len = len; g_err_file = g.names[hi];
                }
                delete op;
                break;
            }
        }
    }

    const int why = g_stop.load();
    const double elapsed = (now_ms() - t0) / 1000.0;
    printf("[%s] STOP reason=%s elapsed=%.1fs total=%.2f GiB avg=%.3f GB/s reads=%llu small=%llu\n",
           label.c_str(), why == 1 ? "time" : why == 2 ? "ERROR" : "STALL",
           elapsed, g_bytes.load() / 1073741824.0,
           g_bytes.load() / 1e9 / (elapsed > 0 ? elapsed : 1),
           (unsigned long long)g_reads.load(), (unsigned long long)g_small.load());
    if (why == 2)
        printf("[%s] FIRST ERROR win32 %lu, %u B at off %llu in %s\n",
               label.c_str(), g_err, g_err_len, (unsigned long long)g_err_off, g_err_file.c_str());
    if (why == 3)
        printf("[%s] NO COMPLETION for %u s -- device wedged; do NOT retry, replug needed\n",
               label.c_str(), stall_s);
    fflush(stdout);
    // Deliberately do not join the completion threads or close handles on a
    // STALL: those calls block forever in uninterruptible I/O.
    ::fflush(nullptr);
    ::TerminateProcess(::GetCurrentProcess(), why == 1 ? 0 : (why == 2 ? 4 : 5));
    return 0;
}
