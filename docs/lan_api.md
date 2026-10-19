# LAN API deployment (2026-10-19, acceptance passed)

The web UI and API adapters share one serial queue and one engine. Model ID:
`deepseek-v4.1-flash`. UI: `http://192.168.31.111:8080/`; OpenAI base:
`http://192.168.31.111:8080/v1`; Anthropic base: `http://192.168.31.111:8080`.
Independent SSH curls from flora, mac and cachyos return HTTP 200 for both UI
and models. The operator opened UFW TCP/8080 from `192.168.31.0/24`; firewall
acceptance is complete. Raw output is in `~/cmoeapi/peer-{flora,mac,cachyos}.txt`.

## Operations

The tracked `tools/web/cachedmoe-lan.service` is installed in
`~/.config/systemd/user/` and enabled, with `Linger=yes`. Start:

```sh
systemctl --user start cachedmoe-lan.service
```

Stop:

```sh
systemctl --user stop cachedmoe-lan.service
```

The service owns the sole GPU engine; stop it before any GPU test or bench.
Graceful SIGINT shutdown has a 150-second KV drain deadline. Output stays under
ignored `build/web_mask/`. Physical logout/reboot has not been exercised.

`tools/web/run-web.sh` now uses **5000 cache slots (previously 5500)** so real
client prompts fit GPU prefill. Other flags remain: dynamic mask, k2/top4,
CPU routing, native BF16 draft head, two read sources, max context 1048576,
GPU prefill minimum 16, disk KV 8 GB. Cold engine start to HTTP readiness was
**21.586733 seconds**, with OS page cache retained; load banner 16.3 seconds.
No binary, kernels, checkpoint, quantization or tokenizer data changed.
This is client capacity acceptance, not a new quality or performance baseline.

## Client proof and per-invocation settings

Codex 0.160.0 completed a real tool loop and exited 0. Its shell wrote exactly
`cachedMoE LAN API OK` plus newline to `~/cmoeapi/client-task/proof.txt`.
Full transcript: `~/cmoeapi/codex-transcript-final.jsonl`; exact invocation:
`~/cmoeapi/run-codex.sh`. The transcript includes recovery from a generated
workdir typo, successful shell write and final answer. Essential overrides:

```sh
CODEX_HOME=/home/chenkailun/cmoeapi/codex-home NO_PROXY='*' no_proxy='*' \
codex exec --ignore-user-config --ignore-rules --ephemeral \
  -c 'model_provider="cachedmoe"' -c 'model="deepseek-v4.1-flash"' \
  -c 'model_providers.cachedmoe={name="cachedMoE LAN",base_url="http://192.168.31.111:8080/v1",wire_api="responses",requires_openai_auth=false}' \
  -c 'features.apply_patch_freeform=false' -c 'features.multi_agent=false' \
  -c 'features.goals=false' -c 'web_search="disabled"' 'Your task'
```

Claude Code 2.1.283 returned `Hello` and exited 0. Full JSON transcript:
`~/cmoeapi/claude-transcript-isolated.json`. The original command was attempted
first, but user settings' `env` overrode the endpoint. Clearing proxy variables
alone did not fix that. The successful invocation skips settings sources per
invocation; global defaults are untouched:

```sh
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  ANTHROPIC_BASE_URL=http://192.168.31.111:8080 \
  ANTHROPIC_AUTH_TOKEN=local ANTHROPIC_MODEL=deepseek-v4.1-flash \
  ANTHROPIC_SMALL_FAST_MODEL=deepseek-v4.1-flash \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=deepseek-v4.1-flash \
  ANTHROPIC_DEFAULT_SONNET_MODEL=deepseek-v4.1-flash \
  ANTHROPIC_DEFAULT_OPUS_MODEL=deepseek-v4.1-flash \
  CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 NO_PROXY='*' no_proxy='*' \
  claude -p 'Reply with exactly the word Hello.' --model deepseek-v4.1-flash \
  --tools '' --no-session-persistence --output-format json --setting-sources ''
```

Reproducible wrapper: `~/cmoeapi/run-claude-direct.sh`. Claude's acceptance
uses no tools; full Claude tool protocol is untested. No template boundary
was demonstrated. The checkpoint has a native DSML tool template.

Final Codex turns used GPU prefill on 6208/6338/6530/6668 tokens, taking
25.906/28.542/27.937/30.121 seconds. Engine decode rates were
10.580569/9.474999/8.823008/9.637710 tok/s. Claude's 1403-token prompt took
21.457 seconds of GPU prefill; decode rate 2.734691 tok/s covers just one
decode step (2 generated tokens including EOS). These are short acceptance
turns, not sustained benchmarks. Exact counters and request/response paths
are in `lan_api_receipt.json`.

## API scope and evidence

`GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/messages` and
`POST /v1/responses` are implemented. Local non-stream and SSE acceptance
bodies remain in `~/cmoeapi/` and indexed in the receipt. Text streams
incrementally for Chat Completions and Messages; tool blocks wait for native
DSML parsing. Responses events are emitted after parsing the completion.
Function definitions, tool results and Responses namespaces map into the
checkpoint template; the adapter does not execute tools. Send full history.
`previous_response_id` and custom/freeform tools are rejected.

`--api-audit-dir` saves request bodies, parsed responses and engine counters
without authorization headers under `~/cmoeapi/api-audit/`, outside git.
The receipt hashes final evidence; transcripts/logs/KV stay outside git.
This trusted LAN listener has no auth or TLS. 58/58 tools gates passed.
Existing dynamic-mask quality limits remain; no numerical quality gate was
rerun because numerical code and executable are unchanged. Final publication
proof is `~/cmoeapi/published-followup.json`, written after the push.
