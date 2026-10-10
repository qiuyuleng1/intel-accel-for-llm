#!/bin/bash -e
#
# Queue-order verification for async KV load (doc/design/async-load-priority.zh-CN.md §7).
#
# Run inside the container while kvshrink-vllm-serve.sh is up, started with the
# scenario's KVSHRINK_* settings (§7.3):
#
#   ./tests/async-load-queue-test.sh s1 _data/queue-test/naive-s1
#   ./tests/async-load-queue-test.sh s2 _data/queue-test/naive-s2
#
# Writes trace.json, vllm.log (this run only), bench.log and report.json to OUT_DIR.

SCENARIO=${1:?usage: $0 s1|s2 OUT_DIR}
OUT_DIR=${2:?usage: $0 s1|s2 OUT_DIR}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

VLLM_LOG=${VLLM_LOG:-log.kvshrink-vllm}
CTRL=http://localhost:${IAXL_API_CONTROLLER_PORT:-18700}
SCHEME=${KVSHRINK_ASYNC_LOAD_SCHEME:?KVSHRINK_ASYNC_LOAD_SCHEME must be set}

export INPUT_LEN=${INPUT_LEN:-4000} OUTPUT_LEN=${OUTPUT_LEN:-128} HIT_RATE=${HIT_RATE:-95}
# One warmup request stores the shared prefix (cache miss) before the measured run;
# without it the measured requests pile up behind the miss's full prefill in one step.
export CONCURRENCY=${CONCURRENCY:-8} NUM_WARMUPS=${NUM_WARMUPS:-1}
case "$SCENARIO" in
    s1) export NUM_PROMPTS=${NUM_PROMPTS:-16} REQUEST_RATE=${REQUEST_RATE:-10} ;;
    s2) export NUM_PROMPTS=${NUM_PROMPTS:-32} REQUEST_RATE=${REQUEST_RATE:-inf} ;;
    *) echo "unknown scenario $SCENARIO" >&2; exit 1 ;;
esac

mkdir -p "$OUT_DIR"
log_start=$(stat -c %s "$VLLM_LOG")

curl -sf "$CTRL/v1/cache/queue_trace?queue=OMP-Main&enable=1" >/dev/null
"$SCRIPT_DIR/vllm-benchmark.sh" 2>&1 | tee "$OUT_DIR/bench.log"
sleep 2 # let trailing zip/unzip tasks finish
curl -sf "$CTRL/v1/cache/queue_trace?queue=OMP-Main&enable=0" >"$OUT_DIR/trace.json"
tail -c "+$((log_start + 1))" "$VLLM_LOG" >"$OUT_DIR/vllm.log"
curl -sf -X POST "$CTRL/v1/cache/evict" -d '{"count": 999999}' >/dev/null

python3 "$SCRIPT_DIR/async_load_queue_check.py" \
    --trace "$OUT_DIR/trace.json" --log "$OUT_DIR/vllm.log" \
    --scheme "$SCHEME" --scenario "$SCENARIO" --report "$OUT_DIR/report.json"
