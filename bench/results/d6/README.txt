Track D6 -- docs/p4_dual_source.md s10 (2026-09-20)

abab_keepalive/   y_turns, ABAB 2 pairs, BOTH arms with --mirror.
                  off = DEEPMOE_MIRROR_KEEPALIVE_MS=0, on = 15.
                  5.2566 -> 5.2643 = +0.15%.  3,052 pokes, 0 refused, 0 errors.
abab_static/      same shape; on = DEEPMOE_MIRROR_STATIC_SPLIT=0.44 (keep-alive off).
                  5.2503 -> 5.1094 = -2.68%.  Split really did reach 44.0%.
gates/            suite.decode / decode_longctx / integration with the mirror on,
                  l3_ppl off (NLL 0.630051, run twice), full CPU ctest 46/46.
                  l3/off.txt is the FIRST l3_ppl run: E: dropped mid-backfill with
                  win32 433/55 and the health gate took it out; the run finished on
                  D: alone and the NLL was still bit-identical.  l3_rerun/ is the
                  immediate repeat: zero errors, E: back at 39.4%.

Verdict: both arms NO-GO, the default path is byte-for-byte D5's.
What the track actually produced is measurement: per-source P0 counters and
per-source idle gaps, both of which say the "the mirror falls asleep" story is
not what the 149 ms was.
