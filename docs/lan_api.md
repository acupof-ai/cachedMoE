# LAN API deployment (2026-10-19, blocked at host firewall)

The web UI and text/tool API adapters share one serial queue and one engine.
The model ID is `deepseek-v4.1-flash`. Existing UI routes remain available.

This host is configured with the tracked `tools/web/cachedmoe-lan.service`,
installed in `~/.config/systemd/user/`, enabled at user-manager startup.
`loginctl show-user chenkailun -p Linger` returned `Linger=yes`; the manager
survives logout. The service has a graceful SIGINT shutdown and a 150-second
KV drain deadline. Runtime output remains under ignored `build/web_mask/`.

Start:

```sh
systemctl --user start cachedmoe-lan.service
```

Stop:

```sh
systemctl --user stop cachedmoe-lan.service
```

Target URLs: UI `http://192.168.31.111:8080/`, OpenAI
`http://192.168.31.111:8080/v1`, Anthropic `http://192.168.31.111:8080`.

**These URLs currently work on the host only.** Real curls from flora, mac and
cachyos fail. UFW is enabled and the kernel logs show `[UFW BLOCK]` with
`DST=192.168.31.111 DPT=8080`. No root/sudo is available; `ufw status` says
`ERROR: You need to be root to run this script`. An administrator must allow
TCP 8080 from `192.168.31.0/24` before LAN acceptance can pass. No alternate
port, tunnel or proxy was substituted for the requested endpoint.

## API scope and evidence

`GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/messages` and
`POST /v1/responses` are implemented. Text streaming is incremental for Chat
Completions and Messages. Tool blocks wait for the native checkpoint DSML
parser; Responses events are emitted after parsing a completed generation.
Function definitions, tool results and Responses tool namespaces are mapped
into the checkpoint's existing template. The adapter does not execute tools.
Full input history must be sent each turn; `previous_response_id` and custom
freeform tools are rejected. This is a text/function subset, not full protocol
parity. Requests share the engine queue with web chats and use unique sessions.

Optional `--api-audit-dir` records request bodies, parsed responses and actual
engine counters without HTTP authorization headers. The machine launcher
currently enables it at `/home/chenkailun/cmoeapi/api-audit`; it contains client
prompt text. It is outside git. No global client configuration was edited.
This trusted-LAN listener has no authentication or TLS.

Real local HTTP checks and raw SSE bodies passed; see `lan_api_receipt.json`.
Real Codex runs are preserved but **no successful file-write/tool loop is
claimed**. Its initial namespace rejection was fixed. A subsequent 6208-token
prompt required 7.13 GiB GPU prefill workspace while only 7.03 GiB was free
beside the unchanged 5500-slot cache; the engine fell back to decode prefill.
The run was cancelled when the firewall blocker was confirmed. Claude Code
execution was not attempted after that blocker, per the task's stop rule.
The checkpoint DOES have a tool template; its absence is not the boundary.

## Per-invocation client settings (not acceptance claims)

Codex CLI 0.160.0 uses the Responses API. The saved reproducible invocation is
`/home/chenkailun/cmoeapi/run-codex.sh`; its essential overrides are:

```sh
codex exec --ignore-user-config --ignore-rules --ephemeral \
  -c 'model_provider="cachedmoe"' -c 'model="deepseek-v4.1-flash"' \
  -c 'model_providers.cachedmoe={name="cachedMoE LAN",base_url="http://192.168.31.111:8080/v1",wire_api="responses",requires_openai_auth=false}' \
  -c 'features.multi_agent=false' -c 'features.goals=false' \
  -c 'web_search="disabled"' 'Your task'
```

Claude Code 2.1.283 configuration to try after removing the blocker:

```sh
ANTHROPIC_BASE_URL=http://192.168.31.111:8080 \
ANTHROPIC_AUTH_TOKEN=local \
ANTHROPIC_MODEL=deepseek-v4.1-flash \
ANTHROPIC_DEFAULT_HAIKU_MODEL=deepseek-v4.1-flash \
ANTHROPIC_DEFAULT_SONNET_MODEL=deepseek-v4.1-flash \
ANTHROPIC_DEFAULT_OPUS_MODEL=deepseek-v4.1-flash \
NO_PROXY='*' no_proxy='*' \
claude -p 'Reply with exactly the word Hello.' --model deepseek-v4.1-flash \
  --tools '' --no-session-persistence
```

The complete pending Claude environment is in `~/cmoeapi/run-claude.sh`.
Configuration examples above are explicitly unproven end-to-end.
