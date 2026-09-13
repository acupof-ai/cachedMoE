// NVMe micro-benchmark -- the P-1 deliverable that answers Q6 and Q7 of
// design §9.1:
//
//   Q6  sequential and random read throughput vs request size (64 KiB..18.8 MB)
//       x queue depth (1..64), unbuffered, into 4 KiB-aligned memory
//   Q7  latency distribution of 4 KiB random reads at QD 48 (the engram row
//       access pattern of design §7.10)
//
// It runs through storage::IoEngine rather than raw ReadFile, so what it
// measures is the thing the runtime will actually use: chunking, priority
// queueing, completion handling and all.
//
//   nvme_bench [--file PATH] [--size-gb N] [--chunk-kb LIST] [--qd LIST]
//              [--pattern seq|rand|both] [--reads N] [--keep] [--csv FILE]
//
// Defaults create a 2 GB file in %TEMP% and delete it afterwards.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "core/align.h"
#include "core/config.h"
#include "core/log.h"
#include "core/status.h"
#include "model/layout.h"
#include "storage/backend.h"
#include "storage/file.h"
#include "storage/io_engine.h"

using namespace deepmoe;
using namespace deepmoe::storage;

namespace {

struct Options {
    std::string file;
    uint64_t size_bytes = 2ull << 30;
    std::vector<uint32_t> chunk_kb{64, 256, 1024, 2048, 4096, 8192, 18360};  // 18360 KiB ~ one expert
    std::vector<uint32_t> queue_depths{1, 2, 4, 8, 16, 32, 64};
    bool seq = true, rand = true;
    uint32_t reads_per_point = 64;      // requests issued per (chunk, QD) point
    bool keep = false;
    std::string csv;
};

std::vector<uint32_t> parse_list(std::string_view s) {
    std::vector<uint32_t> out;
    size_t i = 0;
    while (i <= s.size()) {
        const size_t j = s.find(',', i);
        const std::string_view tok = s.substr(i, j == std::string_view::npos ? std::string_view::npos : j - i);
        if (!tok.empty()) out.push_back(static_cast<uint32_t>(std::atoi(std::string(tok).c_str())));
        if (j == std::string_view::npos) break;
        i = j + 1;
    }
    return out;
}

int usage() {
    std::puts(
        "nvme_bench -- design section 9.2, answers Q6 and Q7\n"
        "  --file PATH      test file (default: a temp file, deleted afterwards)\n"
        "  --size-gb N      size of the test file when it has to be created (default 2)\n"
        "  --chunk-kb LIST  comma-separated request sizes in KiB\n"
        "  --qd LIST        comma-separated target queue depths\n"
        "  --pattern P      seq | rand | both (default both)\n"
        "  --reads N        requests per measurement point (default 64)\n"
        "  --keep           do not delete a test file this run created\n"
        "  --csv FILE       also write the results as CSV\n");
    return 2;
}

// Lays down a real, non-sparse file so the measurement is of the drive.
Result<void> create_test_file(const std::string& path, uint64_t bytes) {
    std::puts(std::format("creating {} ({:.2f} GB) ...", path, bytes / 1e9).c_str());
    auto f = File::open(path, FileFlags::Create | FileFlags::Write | FileFlags::Overlapped
                            | FileFlags::Unbuffered | FileFlags::Sequential);
    if (!f) return std::unexpected(f.error());
    if (auto r = f->set_size(bytes); !r) return std::unexpected(r.error());

    constexpr size_t kBlock = 8u << 20;
    AlignedBuffer buf(kBlock);
    if (!buf) return fail(Err::ResourceExhausted, "cannot allocate the write buffer");
    // Incompressible-ish content, so no controller-side dedup flatters the read.
    std::mt19937_64 rng(0x5EED5EEDull);
    auto* w = reinterpret_cast<uint64_t*>(buf.data());
    for (size_t i = 0; i < kBlock / 8; ++i) w[i] = rng();

    const auto t0 = Clock::now();
    for (uint64_t off = 0; off < bytes; off += kBlock) {
        const size_t n = static_cast<size_t>(std::min<uint64_t>(kBlock, bytes - off));
        auto wrote = f->write_at(off, ByteSpan(buf.data(), n));
        if (!wrote) return std::unexpected(wrote.error());
    }
    const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
    std::puts(std::format("  wrote {:.2f} GB in {:.2f} s ({:.2f} GB/s)",
                          bytes / 1e9, secs, bytes / 1e9 / secs).c_str());
    return {};
}

struct Point {
    const char* pattern;
    uint32_t chunk_kb;
    uint32_t qd;
    uint64_t bytes;
    double   seconds;
    double   gbps;
    double   mean_latency_ms;
    double   max_latency_ms;
    uint32_t iops;
};

// Opens its own File and its own IoEngine. A Win32 handle is permanently bound
// to the first completion port it is associated with (storage/backend.h), so
// every measurement point needs a fresh handle.
Result<Point> measure(const std::string& path, const char* pattern,
                      uint32_t chunk_kb, uint32_t qd, uint32_t reads, bool sequential) {
    auto opened = File::open_read(path, true);
    if (!opened) return std::unexpected(opened.error());
    const File& f = *opened;

    const uint64_t req_bytes = align_up(uint64_t(chunk_kb) << 10, kPageSize);
    if (req_bytes > f.size()) return fail(Err::InvalidArgument, "request larger than the file");

    // Chunk size is the *request* size here: the engine must not split it
    // further, or the measurement stops being about Q6.
    IoConfig cfg;
    cfg.chunk_bytes        = static_cast<uint32_t>(req_bytes);
    cfg.max_inflight_ops   = qd;
    cfg.max_inflight_bytes = static_cast<uint32_t>(
        std::min<uint64_t>(uint64_t(qd) * req_bytes, 512ull << 20));
    cfg.completion_threads = 2;
    cfg.unbuffered         = true;

    auto backend = make_default_backend(cfg);
    if (!backend) return std::unexpected(backend.error());
    IoEngine engine;
    if (auto r = engine.start(std::move(*backend), cfg); !r) return std::unexpected(r.error());

    const uint64_t span = f.size() - req_bytes;
    const uint64_t stride = align_down(std::max<uint64_t>(span / std::max(reads, 1u), kPageSize));

    // One destination buffer per outstanding request, so nothing aliases.
    const uint32_t bufs = std::max(qd, 1u);
    std::vector<AlignedBuffer> dst(bufs);
    for (auto& b : dst)
        if (!b.reset(static_cast<size_t>(req_bytes), kPageSize))
            return fail(Err::ResourceExhausted, "cannot allocate the read buffers");

    std::mt19937_64 rng(0xC0FFEEull ^ (uint64_t(chunk_kb) << 32) ^ qd);
    std::uniform_int_distribution<uint64_t> pick(0, span / kPageSize);

    engine.reset_stats();
    std::atomic<uint32_t> done{0};
    std::atomic<uint32_t> failures{0};
    Status first_error{Err::Ok};

    const auto t0 = Clock::now();
    for (uint32_t i = 0; i < reads; ++i) {
        IoRequest r;
        r.priority = IoPriority::BlockingMiss;
        r.file     = &f;
        r.file_off = sequential ? align_down((uint64_t(i) * stride) % (span + 1))
                                : align_down(pick(rng) * kPageSize);
        r.bytes    = req_bytes;
        r.dst      = dst[i % bufs].data();

        // Throttle to the target queue depth: this is what makes QD the
        // independent variable rather than "everything at once".
        //
        // Throttling on inflight_chunks() would leak -- a request is not in
        // flight until the dispatcher picks it up, so the loop would race ahead
        // and queue far more than `qd`. queued_requests() counts everything
        // submitted and not yet completed, which is the depth we mean.
        while (engine.queued_requests() >= qd && failures.load() == 0)
            std::this_thread::yield();

        auto id = engine.submit(r, [&](const IoResult& res) {
            if (!res.ok() && failures.fetch_add(1, std::memory_order_relaxed) == 0)
                first_error = res.status;
            done.fetch_add(1, std::memory_order_relaxed);
        });
        if (!id) { engine.stop(); return std::unexpected(id.error()); }
    }
    engine.drain();
    const double secs = std::chrono::duration<double>(Clock::now() - t0).count();

    if (failures.load()) {
        engine.stop();
        return fail(Err::Io, std::format("{} request(s) failed: {}",
                                         failures.load(), first_error.str()));
    }

    const IoStats st = engine.stats();
    Point p;
    p.pattern         = pattern;
    p.chunk_kb        = chunk_kb;
    p.qd              = qd;
    p.bytes           = st.bytes_completed;
    p.seconds         = secs;
    p.gbps            = secs > 0 ? p.bytes / 1e9 / secs : 0.0;
    p.mean_latency_ms = st.mean_latency_ms();
    p.max_latency_ms  = st.latency_ns_max / 1e6;
    p.iops            = secs > 0 ? static_cast<uint32_t>(reads / secs) : 0;
    engine.stop();
    return p;
}

}  // namespace

int main(int argc, char** argv) {
    set_log_level(LogLevel::Warn);
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        auto next = [&]() -> std::string_view {
            if (i + 1 >= argc) { std::fputs("missing value\n", stderr); std::exit(2); }
            return argv[++i];
        };
        if (a == "--file")            o.file = next();
        else if (a == "--size-gb")    o.size_bytes = uint64_t(std::atoll(std::string(next()).c_str())) << 30;
        else if (a == "--chunk-kb")   o.chunk_kb = parse_list(next());
        else if (a == "--qd")         o.queue_depths = parse_list(next());
        else if (a == "--reads")      o.reads_per_point = static_cast<uint32_t>(std::atoi(std::string(next()).c_str()));
        else if (a == "--csv")        o.csv = next();
        else if (a == "--keep")       o.keep = true;
        else if (a == "--pattern") {
            const std::string_view p = next();
            o.seq = (p == "seq" || p == "both");
            o.rand = (p == "rand" || p == "both");
        }
        else if (a == "-h" || a == "--help") return usage();
        else { std::fprintf(stderr, "unknown option %.*s\n", static_cast<int>(a.size()), a.data()); return usage(); }
    }
    if (o.chunk_kb.empty() || o.queue_depths.empty()) return usage();

    bool created = false;
    if (o.file.empty()) {
        o.file = temp_dir() + "/deepmoe_nvme_bench.bin";
        created = true;
    }

    auto probe = File::open_read(o.file, true);
    if (!probe || probe->size() < o.size_bytes) {
        if (probe) probe->close();
        if (auto r = create_test_file(o.file, o.size_bytes); !r) {
            std::fprintf(stderr, "cannot create the test file: %s\n", r.error().str().c_str());
            return 1;
        }
        created = true;
        probe = File::open_read(o.file, true);
    }
    if (!probe) {
        std::fprintf(stderr, "cannot open %s: %s\n", o.file.c_str(), probe.error().str().c_str());
        return 1;
    }
    std::puts(std::format("\nfile      {} ({:.2f} GB)", probe->path(), probe->size() / 1e9).c_str());
    std::puts(std::format("unbuffered {}   sector {} B",
                          probe->unbuffered() ? "yes" : "no", probe->sector_size()).c_str());
    std::puts(std::format("one routed expert = {} B = {} KiB (design section 2.3)\n",
                          layout::kExpertBytes, layout::kExpertBytes >> 10).c_str());
    // Each measurement point reopens the file: a Win32 handle is permanently
    // bound to the first completion port it is associated with (storage/backend.h).
    probe->close();

    std::puts("pattern  chunk_kb     qd      GB/s    IOPS   mean_ms    max_ms");
    std::puts("---------------------------------------------------------------");

    std::vector<Point> results;
    const char* patterns[2] = {"seq", "rand"};
    for (int pi = 0; pi < 2; ++pi) {
        if (pi == 0 && !o.seq) continue;
        if (pi == 1 && !o.rand) continue;
        for (uint32_t chunk : o.chunk_kb) {
            for (uint32_t qd : o.queue_depths) {
                auto p = measure(o.file, patterns[pi], chunk, qd, o.reads_per_point, pi == 0);
                if (!p) {
                    std::puts(std::format("{:<8} {:>8} {:>6}   {}", patterns[pi], chunk, qd,
                                          p.error().str()).c_str());
                    continue;
                }
                results.push_back(*p);
                std::puts(std::format("{:<8} {:>8} {:>6} {:>9.3f} {:>7} {:>9.3f} {:>9.3f}",
                                      p->pattern, p->chunk_kb, p->qd, p->gbps, p->iops,
                                      p->mean_latency_ms, p->max_latency_ms).c_str());
            }
        }
    }

    // Q7: 4 KiB random reads at QD 48 -- the engram row pattern of design §7.10.
    std::puts("\nQ7: engram row pattern (4 KiB random, QD 48)");
    {
        auto p = measure(o.file, "rand4k", 4, 48, std::max(o.reads_per_point, 512u), false);
        if (p) {
            results.push_back(*p);
            std::puts(std::format("  {:.3f} GB/s, {} IOPS, mean {:.3f} ms, max {:.3f} ms",
                                  p->gbps, p->iops, p->mean_latency_ms, p->max_latency_ms).c_str());
        } else {
            std::puts(std::format("  failed: {}", p.error().str()).c_str());
        }
    }

    // The headline the design cares about: how long one 18.8 MB expert takes.
    auto best = std::max_element(results.begin(), results.end(),
                                 [](const Point& a, const Point& b) { return a.gbps < b.gbps; });
    if (best != results.end()) {
        const double expert_ms = layout::kExpertBytes / 1e9 / best->gbps * 1000.0;
        std::puts(std::format(
            "\nbest {:.3f} GB/s at {} KiB x QD {} ({})\n"
            "  -> one 18.8 MB expert in {:.2f} ms; a 6-expert layer miss at that rate is {:.1f} ms\n"
            "  -> 4.51 GB of routed weights per token would be {:.1f} s at 0% hit rate",
            best->gbps, best->chunk_kb, best->qd, best->pattern,
            expert_ms, expert_ms * 6,
            4.51 / best->gbps).c_str());
    }

    if (!o.csv.empty()) {
        std::FILE* c = std::fopen(o.csv.c_str(), "wb");
        if (c) {
            std::fputs("pattern,chunk_kb,qd,bytes,seconds,gbps,iops,mean_latency_ms,max_latency_ms\n", c);
            for (const Point& p : results)
                std::fputs(std::format("{},{},{},{},{:.6f},{:.4f},{},{:.4f},{:.4f}\n",
                                       p.pattern, p.chunk_kb, p.qd, p.bytes, p.seconds,
                                       p.gbps, p.iops, p.mean_latency_ms, p.max_latency_ms).c_str(), c);
            std::fclose(c);
            std::puts(std::format("\nwrote {}", o.csv).c_str());
        }
    }

    // (already closed above; each measurement point opened its own handle)
    if (created && !o.keep) {
        if (auto r = remove_file(o.file); r) std::puts(std::format("removed {}", o.file).c_str());
        else std::fprintf(stderr, "could not remove %s\n", o.file.c_str());
    }
    return 0;
}
