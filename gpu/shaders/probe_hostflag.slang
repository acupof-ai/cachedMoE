// probe_hostflag -- can a DEVICE workgroup wait on a HOST-written flag?
// (Track HG, docs/p4_hostflag_gate.md; docs/plan_p5.md §3(a) reopened.)
//
// STATUS §3 item 48 closed the workgroup<->workgroup handshake: two groups,
// both far below the ~406-group residency limit, timed out on the 3rd of
// 1,000 round trips. Co-residency does not imply forward progress, so a
// device-side task queue is illegal here.
//
// This shader asks a DIFFERENT question. The gate round trip of §3 item 53
// (0.40 ms/layer, 16 ms/token, of which 265 us is driver: vkQueueSubmit2
// 99.5 us + submit->GPU-start ~150 us + fence wake ~15 us) is paid because
// the command buffer is CUT at the gate. If ONE workgroup could spin on a
// flag the HOST writes, the cut disappears: one submit holds the whole token
// and the driver's 265 us is paid once instead of forty times.
//
// The forward-progress hazard is not the same one item 48 hit. There the
// spinner waited for a peer workgroup that the scheduler might never run --
// a dependency INSIDE the device's own scheduling domain. Here the spinner
// waits for the HOST, which is always running. What is still unknown, and is
// what mode 3 measures, is whether a workgroup that spins while other
// workgroups of the same command buffer are resident still gets its memory
// loads serviced, and whether the driver's TDR timer tolerates it at all.
//
// EVERY WAIT IN THIS SHADER IS BOUNDED. `spin_limit` caps the loop; hitting
// the cap sets a timeout word and returns. There is no non-terminating path.
// The host calibrates `spin_limit` from a measured per-iteration cost so the
// cap lands on a wall-clock budget (--max-spin-us), and it also bounds the
// SUM over all rounds below the Windows TDR window. This runs on the owner's
// desktop; a TDR is a reboot.
//
// Modes
//   0 REQUEST  one workgroup publishes `round` into Ctl[kReq]. The host is
//              polling that word in host-coherent memory -- this is exactly
//              the gate kernel publishing its top-16 today.
//   1 SPIN     one workgroup spins until Ctl[kFlag] >= round, or until the
//              cap. This is the mechanism under test.
//   2 CONSUME  one workgroup reads Ctl[kVal] -- the value the HOST wrote
//              between the request and the flag -- and accumulates it into
//              Out so a wrong or stale read is visible in the result. This
//              stands for the MoE dispatch reading the slot table.
//   3 BURN     `groups` workgroups of ordinary arithmetic, recorded BEFORE
//              the spin in the loaded runs. Background residency, so the spin
//              is not the only thing on the machine.
//
// Calibration re-uses mode 1 with a round the host never serves: the loop the
// host prices is bit-for-bit the loop under test, backoff included.

// `globallycoherent` is load-bearing, not decoration. A plain
// RWStructuredBuffer load may be served from the shader array's own caches,
// and the host's store never invalidates those: the spin would then read a
// stale flag forever and always hit its cap. globallycoherent forces the
// device-scope coherent path (GLC/DLC on this hardware); the flag itself is
// additionally read with an atomic, which is the strongest ordinary load a
// shader has. Whether even that reaches a HOST store on this driver is the
// first thing the probe has to answer.
[[vk::binding(0, 0)]] globallycoherent RWStructuredBuffer<uint> Ctl;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> Out;   // results, host coherent
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> Scratch;  // burn target, device local

struct ProbeHostFlagPush {
    uint mode;         // 0 request, 1 spin, 2 consume, 3 burn
    uint round;        // 1-based round index inside this command buffer
    uint spin_limit;   // hard cap on the spin loop; 0 is treated as 1
    uint burn_iters;   // mode 3: arithmetic iterations. mode 1: backoff depth.
    uint load_atomic;  // mode 1: 1 = read the flag with an atomic, 0 = plain
};
[[vk::push_constant]] ProbeHostFlagPush pc;

// Control-block words. Mirrored in bench/probes/hostflag_probe.cpp.
static const uint kReq     = 0u;   // device -> host: "round r is asking"
static const uint kFlag    = 1u;   // host -> device: "round r is served"
static const uint kVal     = 2u;   // host -> device: the payload for round r

// Per-round result words (4 per round).
static const uint kOutTimeout = 0u;
static const uint kOutSpins   = 1u;
static const uint kOutVal     = 2u;   // what CONSUME read
static const uint kOutBad     = 3u;   // 1 = CONSUME saw the wrong payload

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex) {
    const uint limit = max(pc.spin_limit, 1u);
    const uint obase = pc.round * 4u;

    if (pc.mode == 3u) {
        // --- background load ---------------------------------------------
        // Ordinary arithmetic on a device-local buffer. One store at the end
        // so nothing is dead code; the value is never read back.
        uint acc = gid.x * 2654435761u + lane;
        for (uint i = 0; i < max(pc.burn_iters, 1u); ++i)
            acc = acc * 1664525u + 1013904223u + i;
        if (lane == 0) Scratch[gid.x & 1023u] = acc;
        return;
    }

    if (gid.x != 0u || lane != 0u) return;

    if (pc.mode == 0u) {
        // --- request ------------------------------------------------------
        // The host is spinning on this word. Nothing else in the command
        // buffer may run before it lands, which the host's barrier gives us.
        Ctl[kReq] = pc.round;
        DeviceMemoryBarrier();
        return;
    }

    if (pc.mode == 1u) {
        // --- spin on the host flag ----------------------------------------
        uint spins = 0u;
        // The loop body re-reads through the buffer every iteration and the
        // compiler cannot hoist it: Ctl is an RWStructuredBuffer, so the load
        // is a non-cacheable global access, and `spins` reaches a store.
        // `burn_iters` is the backoff depth: 0 is a tight busy loop, N inserts
        // N arithmetic steps between two polls. This is the portable stand-in
        // for s_sleep, which neither Slang nor SPIR-V exposes -- the effect we
        // are after (fewer memory transactions a second, the wave off the
        // vector-memory port between polls) is the same, the power saving is
        // not. The host calibrates spin_limit AGAINST THIS LOOP, so the cap is
        // a wall-clock budget whatever the backoff is.
        // Two ways to read the flag, and the difference is not cosmetic.
        // An atomic is a read-modify-WRITE even when it adds zero: it dirties
        // the line in the device's own cache, and from then on every poll is
        // answered by that dirty line rather than by memory -- a host store
        // can never be seen, and on write-back it could even clobber one. The
        // plain globallycoherent load has no such hazard. The probe reports
        // both so the answer is not an artefact of this choice.
        uint pad = pc.round;
        uint seen = 0u;
        if (pc.load_atomic != 0u) InterlockedAdd(Ctl[kFlag], 0u, seen);
        else                      seen = Ctl[kFlag];
        while (seen < pc.round) {
            if (++spins >= limit) {
                Out[obase + kOutTimeout] = 1u;
                Out[obase + kOutSpins]   = spins;
                return;
            }
            for (uint b = 0; b < pc.burn_iters; ++b)
                pad = pad * 1664525u + 1013904223u;
            // Force ordering, then re-read atomically so the load cannot be
            // served from anything the host's store did not reach.
            DeviceMemoryBarrier();
            if (pc.load_atomic != 0u) InterlockedAdd(Ctl[kFlag], 0u, seen);
            else                      seen = Ctl[kFlag];
        }
        Out[obase + kOutSpins] = spins;
        Out[obase + kOutVal]   = pad | 1u;
        return;
    }

    if (pc.mode == 2u) {
        // --- consume ------------------------------------------------------
        // The contract the real gate needs: after the flag, the payload the
        // host wrote is visible. kOutBad is the correctness bit.
        const uint v = Ctl[kVal];
        Out[obase + kOutVal] = v;
        Out[obase + kOutBad] = (v == pc.round * 1000003u) ? 0u : 1u;
        return;
    }

    // Calibration re-uses mode 1 with a round the host never serves, so the
    // loop that is timed is bit-for-bit the loop that is measured. There is
    // no fourth mode.
}
