# cachedMoE rename compatibility

Current commands use `cachedmoe`, `CACHEDMOE_*` and `../cachedmoe-<track>`.
Historical reports, owner requirements and frozen commands keep their original
names. Renaming does not alter checkpoint files, shader entry points, numeric
constants or persisted format magic.

All six rename commits are complete through `f3149a6`. Configuration cleanup
through `43a2ab9` passes the build, three prefix combinations of 31 CPU suites
and 52 tool gates with model metadata enabled. The 52 SPIR-V files remain
byte-identical. Actual CLI guards and legacy source/KV readers also pass.
The local main fast-forward at `bec29fd` and its existing directory build also
pass, with three prefix combinations of 31 CPU suites and 52 tool gates.
The actual main executable passes the five CLI policy cases, three artifact
symlinks, old-source compilation and two legacy KV reads. The actual main executable also passes nine numerical jobs with ten registered
cases and no skips, preserving off/mask NLL .622784/.835581, short decode 6/8
and own-prefill 7/8, and 4K/16K 8/8. The RO mirror has 48/48 healthy shards.
Accepted changes are pushed at `cd5f1ea`; the owned clean worktree and branch
are removed. The guarded web service is restored and its live API, four old
histories and fresh speculative generation are verified.
See [configuration scope](runtime_configuration.md).

| Interface | Compatibility rule |
|---|---|
| Runtime environment | Read `CACHEDMOE_X` first; only absence falls back to `DEEPMOE_X`. Empty values are present. Existing consumer parsing stays unchanged. |
| CMake cache | New normal/cache values win, including empty. Legacy-only cache entries resolve before defaults and remain adjustable with old `-D` arguments. Once a canonical cached default exists, a later legacy argument cannot override it. |
| CMake module | `cmake/deepmoe_options.cmake` includes the canonical options module. Old internal list names and compile macros are not separate settings. |
| Linux artifacts | `build/deepmoe` resolves to `cachedmoe`; old test/library aliases refer to canonical artifacts. Process checks accept both exact names and retain ownership checks. |
| C++ source | `namespace deepmoe = cachedmoe` supports old source references. Rebuild objects; this is not compatibility with old binary ABI. |
| Saved state | Prefer an existing canonical root; reuse the legacy root when canonical is absent. Explicit `--kv-dir` is literal. Both-root selection is logged; no automatic migration or merge occurs. |

Warnings identify deprecated/conflicting keys without printing their values.
CLI overrides and benchmark arms remain explicit overrides; compatibility aliases
do not create another effective setting. Build options, runtime settings and
shader compile definitions remain separate domains.

The actual CPU KV reader linked against the renamed library loaded two existing
legacy snapshots with model-tag validation: `default` has 396 positions and four
planes; `web` has 4,710 positions and four planes. File SHA-256, size and mtime
were unchanged. The local raw evidence is
`bench/results/mask_quality/rename_prepared/final_config_validation/binary_shader_legacy_final_receipt.json`.
A separate isolated copy of the old web snapshot also restores all 4,710
positions on the actual GPU engine, replays the bounded 128-position window,
reuses 4,708 tokens after the normal two-token rollback, prefills only those
two tokens, and generates two new tokens. Normal quit drains disk writes.
The original snapshot and eight transcripts retain their SHA-256, size and
mtime. Raw evidence: `legacy_live_smoke/legacy_live_result.json` in the same
validation directory. The first smoke assertion omitted normal rollback; its
failed harness receipt and the corrected acceptance run are both preserved.
Current web/API acceptance also passes; the smoke generates 24 tokens in nine
cycles, verifies 18 draft tokens and accepts 14. Its 38-position disk snapshot
is readable. The existing old web KV and current web transcript contain
different token counts, so the old-KV restore proof uses an isolated copy and
does not claim current-chat prefix reuse. See [the web receipt](web_restore_receipt.json).
This is compatibility evidence,
not a new throughput or quality measurement.

Run the committed-tree audit after the final documentation commit:

```bash
python3 tools/rename_audit.py --ref HEAD \
  --out bench/results/mask_quality/rename_prepared/remaining_final.json
```

The output refuses to overwrite a receipt. Each old-name hit includes its exact
path, line, byte column, token, full line, source hash, role and reason. Unreviewed
live code/defaults block the audit; a preserved token does not exempt other
tokens on the same line. Binary files and legacy filenames are counted separately.
Current build/launch instructions live in [build.md](build.md) and
[the web guide](../tools/web/README.md); measured historical reports retain
their original commands and ledger bytes.

The complete nonignored working-tree scan also covers owner raw data: all
8,210 text hits have retention roles. Its conservative filename check flags
11 saved experiment binaries/source/chat paths. A separate exact-path/hash
review retains all 11 as historical evidence; no owner files are renamed.
Raw review: `final_config_validation/owner_untracked_filename_review.json`
under the same results root. The committed-tree audit is the release gate.
