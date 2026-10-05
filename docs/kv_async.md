# Batched decode KV persistence

The running web profile had `--no-kv-disk`. In-memory KV was still reused, but
there was no disk checkpoint. With disk enabled, `serve` previously saved the
live context only when switching sessions or shutting down. A long-lived tab
could finish several decode turns without saving a new `.pkv` file.

`serve` now checkpoints after each completed or cancelled generation. It packs
one immutable CPU snapshot after the last GPU fence and enqueues a background
disk write. The live context stays installed. It adds no per-token disk write,
GPU dispatch, or disk join to the decode loop. Snapshot packing still costs CPU
time once per turn and is logged separately; this is not a zero-cost copy.

A single disk worker serialises the existing temporary-file/atomic-rename format
and disk quota eviction. Pending updates to one session coalesce to the newest
snapshot. The pending queue has a 256 MiB budget; one oversized snapshot is
allowed alone. Under sustained disk backlog it drops the oldest pending snapshot
rather than blocking decode. The in-flight write plus one oversized queued
snapshot can exceed 256 MiB. Persistence is best effort, not a crash-safe journal.

The worker only receives owned CPU data. It never reads mutable GPU KV. Session
reset removes pending work, waits for that session's in-flight write, and then
deletes the file, so the writer cannot recreate reset state. Loading from disk
and clean shutdown drain the queue. Session switching may wait when loading a
snapshot that has not landed yet; ordinary generation in the same session does
not wait for disk completion.

The `.pkv` layout and pack/unpack arithmetic are unchanged. The sliding-window
ring is still restored by bounded replay, not saved. `--no-kv-disk` continues to
disable persistence for clean-start benchmarks. Web use should omit that flag.

## Validation

CPU checks: 25/25 CTest suites and 32/32 Python gates passed. `suite.kvdisk` has
5 cases, including controlled background-write blocking, same-session coalescing,
owned snapshot data, cancellation/reset without resurrection, write failure,
and destructor draining.

The serial hardware smoke used Linux/RADV, two checkpoint read sources, 5,000
slots, exact routing and a 4K context capacity. Strata served OpenAI chat,
Anthropic chat, and streaming thinking/answer through the native backend. Each
turn queued and wrote KV before shutdown. Snapshots at 26/26/75 tokens packed
in **1.51 / 2.14 / 6.02 ms**, with 48,196 / 48,196 / 91,468-byte `.pkv` files.
A new engine loaded and restored the 75-token snapshot, replaying 75 positions
in 18.5 s, then answered correctly. The next unrelated prompt reset that state;
this checks persistence and restoration, not prefix-reuse speed or a benchmark.
There were zero thermal pauses. [Machine receipt](kv_async_receipt.json);
raw logs and the exact tested binary: `bench/results/kv_async/`.

A startup attempt exposed that Qwen's sampling-config validator rejected native
top-k=0. The Strata adapter now validates native sampling before engine startup
and passes its supported defaults to the common service. That fix has a CPU
main-entry regression test and preceded the successful native smoke.
