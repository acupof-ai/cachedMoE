#!/bin/bash
# Round 2: shallower P0 queues at 1 MiB (round 1, p0_sweep.sh: q4 4.68 < q8 4.72 < q12 4.90 ms/miss, 2 MiB x 8 5.00).
cd ~/projects/cachedMoE
for q in 2 3 4 6; do
  O=bench/results/linux/perf/p0sweep/r2_c1q$q
  mkdir -p $O
  .venv/bin/python tools/hitrate_bench.py --script bench/results/hitrate/long_turns.json --max-turns 3 \
      --out $O --env DEEPMOE_IO_P0_CHUNK_MB=1 --env DEEPMOE_IO_P0_QD=$q > $O.log 2>&1
  .venv/bin/python tools/perf_report.py --chat $O --record "p0sweep-r2-c1q$q-3turns" | sed -n 2,5p
  grep -o '"disks_during_run".*' $O/provenance.json | head -c 0; .venv/bin/python -c "
import json;d=json.load(open('$O/provenance.json')).get('disks_during_run',{});print('   disk writes during run:', {k:v['write_MB'] for k,v in d.items() if isinstance(v,dict)})"
done
