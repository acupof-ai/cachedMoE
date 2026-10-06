# Runtime configuration

Configuration has one authority per scope. Equal values for different purposes
remain separate settings; startup policy is separate from live telemetry and
per-request controls.

| Scope | Authority | Consumers |
|---|---|---|
| Shared native/Python facts | `core/runtime_facts.def` | C++ constexpr header and strict Python reader |
| Native option defaults | `core/config.h` | Typed options before environment and CLI overrides |
| Native environment parsing | `core/runtime_environment.h` | CLI, engine, device, lazy GPU resources, cache and IO |
| Python launch and UI policy | `tools/runtime_defaults.py` | CLI tools, benchmark launchers and web configuration API |
| Environment alias precedence | `core/env.h`, `tools/runtime_env.py` | Native and Python entry points |
| Build and shader options | `cmake/cachedmoe_options.cmake` | CMake configuration and generated shader build commands |
| Model metadata defaults | `model/v41_config.h` | Optional JSON fields reuse their value-type defaults |

The canonical environment key wins whenever it is present, including an empty
value. A legacy key is used only when the canonical key is absent. Conflict
warnings identify keys without printing their values. Each historical parser
keeps its own rules for flags, numeric prefixes and empty values; these rules
are defined once in the native registry rather than repeated by consumers.

Native CLI validation and engine setup share an owned environment snapshot.
Changing process environment after setup cannot change the engine's lazy
pipelines or IO options. Standalone factories capture their setup epoch.
Public helpers that historically cache process settings keep a process epoch.
IO widening and capability-clamped derivation use the same parsed overrides.

Python launchers resolve the final environment before building argv, loading
the encoder or recording provenance. A repeated native `--model` argument uses
the final occurrence. Production launch clears inherited experiment controls;
benchmark profiles preserve their explicitly documented inheritance policy.
The web UI reads request defaults, bounds and effort presets from `/api/config`.
Power state, temperature and measured timing remain live observations.

The native request defaults to 256 output tokens; the web request defaults to
1,024. A library context defaults to 65,536 positions, a session to 4,096, and
the web launch uses the 1,048,576-position capacity. These are different scopes.
Capacity does not establish quality at a full 1M-token prompt.

Window replay uses one shared 128-token default. The library/session prefill
threshold and the measured RADV threshold have separate named defaults;
production Python launchers use the same RADV value as the native CLI.

Required checkpoint keys remain required. Optional scalar metadata has 39
fallback sites that read the corresponding field's initialized value. Legacy
decode exports keep their distinct rules: absent `config` leaves zero layers
and fails; a present object missing `n_layers` uses the compiled model layout.
Manifest format sentinels, derived offsets and model shapes keep their own
format/layout authorities.

The audit also checked older public fields with no runtime effect. They remain
for source compatibility and are explicitly reserved: `IoConfig.preempt_on_blocking`,
`RuntimeConfig.memory_path`, `RuntimeConfig.prefill`, `RuntimeConfig.io_core_mask`,
and `PrefetchConfig.min_width`, `max_width`, `precision_floor`. Prefetch
`lookahead_depth` and `lookahead_width` are logged, with no predictive scheduler.
Cache-policy alternatives retain their measured LRU fallback. These declarations
do not enable another implementation.

Regression checks cover alias conflicts and empty values, snapshot ownership,
late environment changes, reinitialization, parser families, two-stage IO,
native generation options and launch/encoder/provenance consistency. The shared
facts check compiles a real C++ header reader and compares its values and types
against Python:

```bash
python3 tools/tests/test_runtime_facts.py --check-cpp --cxx clang++
```

Numerical and live-web acceptance results belong in [STATUS.md](STATUS.md),
with [the curated configuration receipt](runtime_configuration_receipt.json)
and raw receipts under `bench/results/mask_quality/rename_prepared/`.
