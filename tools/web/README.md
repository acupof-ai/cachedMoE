# Local web chat

```bash
export CACHEDMOE_MODEL_DIR="$HOME/models/DeepSeek-V4.1-Flash"
python3 tools/web/server.py --max-context 1048576
```

Open <http://127.0.0.1:8080/> after the engine reports ready. The default UI language
is English. Generated text follows the prompt's language. The server starts one
`cachedmoe serve` process and queues requests; do not start another GPU job beside it.
The current local instance is described in [RUNNING.txt](RUNNING.txt).

## Controls and display

- Set temperature, top-p, output token limit, and an optional deterministic seed.
- Enable **Thinking** and select low (50), high (75), max (100), or a custom integer
  from 1 to 100. This uses the checkpoint's native reasoning-effort prompt field.
- Thought process and answer have separate panels. Thinking opens while streaming
  and collapses when the answer begins. Click it to reopen. Cancelled or truncated
  thinking remains in that panel, including after history is restored.
- **Token probabilities** shows each token's probability, cache hit rate, step time,
  and ID on hover. **Ctrl+Enter** sends, **Stop** cancels, **New chat** resets context
  while keeping the expert cache warm.
- The status strips show context capacity, decode speed, cache hits, disk wait,
  first-token latency, prefill, actual speculation settings, and hardware telemetry.

No CDN is required. Transcripts persist locally under
`$XDG_CACHE_HOME/cachedmoe/web_chat` (normally `~/.cache/cachedmoe/web_chat`) on Linux,
or `%LOCALAPPDATA%/cachedmoe/web_chat` on Windows. If the canonical root is absent
and an old `deepmoe` root exists, the server reuses it for both transcript and KV;
it does not migrate or merge files. When both roots exist, the canonical root wins
and the selected path is logged. An explicit `--kv-dir` keeps its exact path.
See [rename compatibility](../../docs/rename_compatibility.md). The session-name field selects a
conversation. Reloading restores it. Different tabs should use different names.
With disk KV enabled, completed decode turns now enqueue one batched background
checkpoint. The live KV stays in memory. [Persistence details](../../docs/kv_async.md).

## Configuration

| Option | Purpose |
|---|---|
| `--max-context N` | Context capacity; default and hard limit: 1,048,576 tokens |
| `--cache-gb N` / `--cache-slots N` | Explicit expert-cache budget; omitted means automatic sizing |
| `--mirror DIR` | Additional identical checkpoint read source; repeatable |
| `--no-mirror-auto` | Disable discovery of matching checkpoint mirrors |
| `--resident-only mask` | Experimental: skip unavailable routed experts; default is exact routing |
| `--mask-cache dynamic\|fixed` | Mask cache policy; dynamic LRU/loading is the default; fixed can cause repetition |
| `--dspark` | Enable experimental single-path speculation |
| `--spec-k N` / `--spec-top-k N` | Draft length (native block: five) and approximate target acceptance K |
| `--kv-dir DIR` / `--kv-max-gb N` | Disk KV cache location and budget |
| `--no-kv-disk` | Disable saved KV for a clean-start run |
| `--max-parked N` | Number of sessions parked in memory before disk parking |
| `--think` | Enable thinking for new sessions |
| `--exe PATH` | Engine executable; normally `build/cachedmoe` on Linux |
| `--host 127.0.0.1` / `--port 8080` | Listening address and port |
| `--log PATH` | Engine log file |

Keep the service on loopback. This local web bridge does not provide authentication.
Automatic cache sizing applies a budget cap and probes a GPU submission, reducing
its budget on allocation refusal. Explicit budgets are checked but never silently
reduced. See [STATUS](../../docs/STATUS.md) for the current machine's cache limits.

## Context and long prompts

KV grows from a 4K allocation as needed. Live bf16 KV costs about 3,200 bytes per
token, excluding windows, scratch, and parked conversations. Context capacity
and the memory required for GPU prefill are separate limits. A full 1M prompt
has not been validated end to end; larger startup capacity does not remove
prefill's working-memory requirements.

The composer previews total prompt tokens, reusable tokens, and work still needed.
Prefill estimates use live measurements. Prompts requiring at least 20,000 new
prefill tokens still ask for confirmation. Changing thinking effort can alter
the prompt prefix and require new prefill.

GPU prefill uses the exact streaming path even in mask mode. Decode throughput
is not prefill throughput. Read the [mask and speculation quality limits](../../README.md#quality-and-experimental-modes)
before enabling them.

## Validation

```bash
python3 tools/web/test_server.py
node --test tools/web/test_ui.cjs
```

These checks start no engine and use no GPU. Engine changes need a rebuild and
server restart; page-only changes appear on the next page load.
