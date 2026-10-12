#!/bin/bash -e
# usage: restart-vllm.sh <DYNAMIC> <LAYERS> <MAP> <SCHEME>
# Restart our port-8010 vLLM on GPU7 inside container $C (dev helper).
REPO=$(cd "$(dirname "$0")/../.." && pwd)
C=${C:-iaxl.vllm.qiuyu}
P=$(ps -eo pid,args | grep -E "[v]llm serve .*--port 8010" | awk '{print $1}')
if [ -n "$P" ]; then
    docker exec $C kill -TERM "$P"
    for i in $(seq 1 30); do
        [ "$(nvidia-smi -i 7 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 100 ] && break
        sleep 2
    done
fi
echo "GPU7 used MiB: $(nvidia-smi -i 7 --query-gpu=memory.used --format=csv,noheader,nounits)"
# Wait for the old server's tee (cwd = our repo) to exit. If it is still writing
# when the new tee truncates log.kvshrink-vllm, it keeps writing at its old offset,
# leaving a NUL hole and inflating the file size used as the test's log start.
for i in $(seq 1 30); do
    # Checked inside the container: the host user cannot read /proc/<pid>/cwd of
    # the container's root processes.
    OLD_TEE=$(docker exec $C bash -c "for T in \$(pgrep -f '^tee log.kvshrink-vllm\$'); do [ \"\$(readlink /proc/\$T/cwd)\" = '$REPO' ] && echo \$T; done" | tr '\n' ' ')
    [ -z "${OLD_TEE// /}" ] && break
    sleep 2
done
[ -n "${OLD_TEE// /}" ] && { echo "old tee still running: $OLD_TEE" >&2; exit 1; }
docker exec -w "$REPO" -d $C bash -c "export CUDA_VISIBLE_DEVICES=7 VLLM_CPU_OMP_THREADS_BIND=64-127 PORT=8010 IAXL_API_CONTROLLER_PORT=18710 IAXL_API_WORKER_BASE_PORT=18810 KVSHRINK_VLLM_KV_ASYNC_LOAD_ENABLED=1 KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC=$1 KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS=$2 KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP=$3 KVSHRINK_ASYNC_LOAD_SCHEME=$4 VLLM_BATCH_INVARIANT=${VLLM_BATCH_INVARIANT:-0}; exec ./examples/kvshrink-vllm-serve.sh > /dev/null 2>&1"
for i in $(seq 1 90); do curl -sf http://localhost:8010/health >/dev/null && break; sleep 5; done
curl -s -o /dev/null -w 'health=%{http_code}\n' http://localhost:8010/health
grep -aE "async load scheme" "$REPO/log.kvshrink-vllm" | sed -E 's/^.*\] //'
