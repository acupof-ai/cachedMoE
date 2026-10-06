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
allowed alone. Queue pressure applies backpressure until the worker releases
space. It never discards another session's pending snapshot. The in-flight write
and the producer's packed snapshot are outside the pending budget, so total
snapshot memory can exceed 256 MiB. This is not a crash-safe journal.

The worker only receives owned CPU data. It never reads mutable GPU KV. Session
reset removes pending work, waits for that session's in-flight write, and then
deletes the file, so the writer cannot recreate reset state. Loading from disk
and clean shutdown drain the queue. Session switching may wait when loading a
snapshot that has not landed yet. Ordinary turn completion queues work without
waiting for disk completion, except when the pending budget is full. Before
evicting a parked session from memory, the pool waits for a successful atomic
save; a failure returns an error and retains the memory copy. `park_active` also
returns a save error. A failed budget enforcement can leave the newly requested
session active and retain extra parked sessions; it does not delete their data.
Same-session coalescing carries save acknowledgements to the newer snapshot.

Exit also propagates a failed required checkpoint through a nonzero `serve`
status, after draining the writer and releasing GPU resources. The web bridge
requires a successful engine exit; a timeout, forced kill or failed save is
reported as a failed or unconfirmed KV drain instead of silently returning
success. Ordinary round-end writes remain asynchronous.

The `.pkv` layout and pack/unpack arithmetic are unchanged. The sliding-window
ring is still restored by bounded replay, not saved. `--no-kv-disk` continues to
disable persistence for clean-start benchmarks. Web use should omit that flag.

## Validation

CPU checks after the queue fix: 25/25 CTest suites and 33/33 Python gates passed.
`suite.kvdisk` has 8 cases, including controlled background-write blocking, same-session coalescing,
owned snapshot data, cancellation/reset without resurrection, write failure,
and destructor draining. New tests require all sessions to land under a four-byte
queue budget, propagate a required save failure, and load a stale snapshot from
disk before checking its token prefix against a changed prompt. Prefix comparison
uses the same helper as generation: a mismatched suffix is discarded and fed
again, rather than treated as current conversation state. These are CPU tests;
the previous hardware smoke below predates the queue fix.

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
