#!/bin/bash
# Re-export the 4K / 17K long-context oracles with each kv source scoring its
# own index keys (tools/oracle_longctx.py's shim; the reference-rule exports
# are kept in traces/longctx/refkeys/). CPU only; no engine may run meanwhile.
# The decode checkpoint is removed first so a rerun always decodes again under
# the current rule, from the prefill's own checkpoint (ckpt_decode_start.pt)
# when there is one -- a leftover ckpt_decode.pt would skip the decode.
cd ~/projects/cachedMoE
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
for n in ctx4k ctx16k; do
  rm -f traces/longctx/$n/ckpt_decode.pt
  .venv/bin/python tools/oracle_longctx.py run --name $n > traces/longctx/${n}_run.log 2>&1
  echo "== $n rc=$?"
done
