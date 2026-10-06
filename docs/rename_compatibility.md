# cachedMoE rename compatibility

Current commands use `cachedmoe`, `CACHEDMOE_*` and `../cachedmoe-<track>`.
Historical reports, owner requirements and frozen commands keep their original
names. Renaming does not alter checkpoint files, shader entry points, numeric
constants or persisted format magic.

The first five rename commits are complete through `de74684`. Stage5 validation
passed 29 CPU suites, 46 tool gates, five real CLI root-selection cases and the
two real legacy KV reads below; all 52 SPIR-V files remained byte-identical.
Stage6 source validation has passed the build, 29 CPU suites and the same
52 shader hashes. Its documentation commit, final settings audit, new-executable
GPU numerical checks, live web restoration and delivery are still pending.

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
`bench/results/mask_quality/rename_prepared/legacy_kv_reader_receipt.json`.
This proves CPU format/read compatibility. GPU restoration, current web/API
configuration and session continuation still need final serial validation.

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
