#!/bin/bash
# Re-export the 4K / 17K long-context oracles with each kv source scoring its
# own index keys (tools/oracle_longctx.py's shim; the reference-rule exports
# are kept in traces/longctx/refkeys/). CPU only; no engine may run meanwhile.
cd ~/projects/cachedMoE
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
for n in ctx4k ctx16k; do
  .venv/bin/python tools/oracle_longctx.py run --name $n > traces/longctx/${n}_run.log 2>&1
  echo "== $n rc=$?"
done
