// probe_resident -- the feasibility probe for a persistent-dispatch decode
// (Track W, docs/plan_p5.md §3(a)).
//
// The proposal is fleet-mi300x's shape: one dispatch for a layer (or a token),
// workgroups that never exit, and a device-side task queue they hand work to
// each other through. That is worth 41 submits x 0.156 ms plus every barrier
// tail -- but it only works if three things hold on a Radeon 8060S under
// Vulkan, and not one of them is documented:
//
//   1. RESIDENCY. How many workgroups of a given size actually run at once?
//      A megakernel whose workgroup N waits for workgroup M deadlocks if M was
//      never scheduled. Vulkan has no cooperative launch and no forward-
//      progress guarantee between workgroups: this is the number that decides
//      whether the whole idea is legal, not merely fast.
//
//   2. CROSS-WORKGROUP LATENCY. What does one producer->consumer handshake
//      through device memory cost? fleet measures 1.36 us idle / 5.88 us under
//      load on an MI300X. If it is tens of microseconds here, 40 layers x
//      several events a layer eats the submit saving.
//
//   3. FORWARD PROGRESS. Does a spinning workgroup starve the one it waits
//      for? Mode 2 answers it with a bounded spin so a "no" is a reported
//      timeout rather than a hung machine.
//
// EVERY WAIT IN THIS SHADER IS BOUNDED. `spin_limit` caps every loop; a loop
// that reaches the cap sets a bit in the result and returns. There is no path
// through this shader that does not terminate. That is deliberate: the machine
// this runs on is the owner's desktop and a TDR is a reboot.
//
// Mode 0 -- residency census
//   Every workgroup increments a live counter, publishes the maximum it has
//   ever seen, spins `spin_limit` iterations, then decrements. Dispatch
//   `groups` workgroups; `Out[kPeak]` is the largest number that were ever
//   live at once. Sweep `groups` upward: the peak stops tracking it at the
//   residency limit.
//
// Mode 1 -- ping-pong latency
//   Workgroup 0 writes round r to Flag[0] and waits for Flag[1] to reach r;
//   workgroup 1 does the mirror. `rounds` round trips, timed by the host
//   through GPU timestamps around the dispatch. Requires at least two
//   co-resident workgroups, so run mode 0 first.
//
// Mode 2 -- forward progress under a spinner
//   Workgroup 0 spins on a flag workgroup `groups - 1` sets after doing a
//   little work. If the last workgroup is never scheduled while the first
//   spins, workgroup 0's spin hits its cap and reports it. Repeat with
//   `groups` above and below the mode-0 residency limit: a timeout below the
//   limit is a scheduler property, a timeout above it is expected and is the
//   reason a persistent kernel may not exceed the limit.

[[vk::binding(0, 0)]] RWStructuredBuffer<uint> Ctl;   // [0] live, [1] peak, [2..] flags
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> Out;   // per-workgroup results

struct ProbeResidentPush {
    uint mode;         // 0 census, 1 ping-pong, 2 forward progress
    uint groups;       // how many workgroups the host dispatched
    uint spin_limit;   // hard cap on EVERY loop; 0 is treated as 1
    uint rounds;       // mode 1 only
};
[[vk::push_constant]] ProbeResidentPush pc;

static const uint kLive    = 0u;
static const uint kPeak    = 1u;
static const uint kFlagA   = 2u;
static const uint kFlagB   = 3u;
static const uint kDone    = 4u;

// Per-workgroup result words.
static const uint kOutSeen    = 0u;   // live count this workgroup observed
static const uint kOutTimeout = 1u;   // 1 = a loop hit spin_limit
static const uint kOutRounds  = 2u;   // rounds actually completed (mode 1)

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex) {
    const uint g     = gid.x;
    const uint limit = max(pc.spin_limit, 1u);
    const uint obase = g * 4u;

    if (lane == 0) {
        Out[obase + kOutSeen]    = 0u;
        Out[obase + kOutTimeout] = 0u;
        Out[obase + kOutRounds]  = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    if (pc.mode == 0u) {
        // --- residency census ------------------------------------------------
        uint live = 0u;
        if (lane == 0) {
            InterlockedAdd(Ctl[kLive], 1u, live);
            live += 1u;                       // this workgroup included
            Out[obase + kOutSeen] = live;
            InterlockedMax(Ctl[kPeak], live);
        }
        // Hold the slot so late workgroups are counted against early ones. A
        // bounded, side-effect-free spin: the compiler cannot hoist it out
        // because the accumulator reaches a store.
        uint burn = 0u;
        for (uint i = 0; i < limit; ++i) burn += Ctl[kLive] + i;
        if (lane == 0) {
            uint prev;
            InterlockedAdd(Ctl[kLive], 0xFFFFFFFFu, prev);   // -1
            Out[obase + kOutRounds] = burn | 1u;             // keep the loop alive
        }
        return;
    }

    if (pc.mode == 1u) {
        // --- ping-pong latency, workgroups 0 and 1 only ----------------------
        if (g > 1u) return;
        if (lane != 0) return;
        const uint mine  = (g == 0u) ? kFlagA : kFlagB;
        const uint other = (g == 0u) ? kFlagB : kFlagA;
        uint done = 0u;
        for (uint r = 1u; r <= max(pc.rounds, 1u); ++r) {
            Ctl[mine] = r;
            uint spins = 0u;
            // Bounded: the partner may not be resident at all.
            while (Ctl[other] < r) {
                if (++spins >= limit) {
                    Out[obase + kOutTimeout] = 1u;
                    Out[obase + kOutRounds]  = done;
                    return;
                }
            }
            done = r;
        }
        Out[obase + kOutRounds] = done;
        return;
    }

    // --- mode 2: forward progress ------------------------------------------
    if (lane != 0) return;
    const uint last = (pc.groups > 0u) ? pc.groups - 1u : 0u;
    if (g == last && last != 0u) {
        Ctl[kDone] = 1u;
        Out[obase + kOutRounds] = 1u;
        return;
    }
    if (g == 0u) {
        uint spins = 0u;
        while (Ctl[kDone] == 0u) {
            if (++spins >= limit) {
                // The last workgroup never ran while this one spun: a
                // persistent kernel at this workgroup count would hang.
                Out[obase + kOutTimeout] = 1u;
                Out[obase + kOutSeen]    = spins;
                return;
            }
        }
        Out[obase + kOutSeen]   = spins;
        Out[obase + kOutRounds] = 1u;
        return;
    }
    // Everyone in between exits immediately, freeing their slot.
    Out[obase + kOutRounds] = 1u;
}
