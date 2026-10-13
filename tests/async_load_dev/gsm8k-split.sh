#!/bin/bash
# GSM8K accuracy, split_async_load_submit_first_n_batch_reqs, run twice on one server (dev helper).
# vLLM: container $C (v0.29), Qwen3-32B, TP=2 on GPU4,5, async-load settings from
# setvars.sh (DYNAMIC=1, map 0-3:0,4-6:4,7-:8). DDR is NOT evicted between runs,
# so run2 mostly hits the KV saved by run1 and exercises the split load path.
REPO=$(cd "$(dirname "$0")/../.." && pwd)
C=${C:-iaxl.vllm.qiuyu-v29}
GSM8K_DIR=/home/pese/qiuyu/Al-dsv4/KVCacheClip/internal/daocloud/test/accuracy-gsm8k
OUT=${OUT:-$REPO/_data/gsm8k-split}
MODEL=/home/pese/model-space/Qwen3-32B
cd "$REPO"

# Stop our previous port-8010 server, then wait for its tee to exit (see
# restart-vllm.sh for why).
P=$(ps -eo pid,args | grep -E "[v]llm serve .*--port 8010" | awk '{print $1}')
if [ -n "$P" ]; then
    docker exec $C kill -TERM "$P"
fi
for i in $(seq 1 60); do
    OLD=$(ps -eo pid,args | grep -E "[v]llm serve .*--port 8010" | awk '{print $1}')
    OLD_TEE=$(docker exec $C bash -c "for T in \$(pgrep -f '^tee log.kvshrink-vllm\$'); do [ \"\$(readlink /proc/\$T/cwd)\" = '$REPO' ] && echo \$T; done" | tr '\n' ' ')
    [ -z "$OLD" ] && [ -z "${OLD_TEE// /}" ] && break
    sleep 2
done

docker exec $C bash -c "mkdir -p $OUT && chown $(id -u):$(id -g) $OUT"
docker exec -w "$REPO" -d $C bash -c "export CUDA_VISIBLE_DEVICES=4,5 TP_SIZE=2 VLLM_CPU_OMP_THREADS_BIND='64-95|96-127' PORT=8010 IAXL_API_CONTROLLER_PORT=18710 IAXL_API_WORKER_BASE_PORT=18810 KVSHRINK_ASYNC_LOAD_SCHEME=split_async_load_submit_first_n_batch_reqs; exec ./examples/kvshrink-vllm-serve.sh > /dev/null 2>&1"
for i in $(seq 1 120); do curl -sf http://localhost:8010/health > /dev/null && break; sleep 5; done
curl -s -o /dev/null -w 'health=%{http_code}\n' http://localhost:8010/health
grep -aE "async load scheme|KVSHRINK_VLLM_KV_ASYNC_LOAD|TP_SIZE|VLLM_CPU_OMP" log.kvshrink-vllm | sed -E 's/^.*\] //' | head -12
echo "connector_md5=$(md5sum kvshrink/kvshrink_connector.py | cut -c1-8)"

export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1
for RUN in run1 run2; do
    log_start=$(stat -c %s log.kvshrink-vllm)
    echo "===== gsm8k $RUN ====="
    API_URL=http://127.0.0.1:8010/v1 MODEL=$MODEL \
        OUTPUT_DIR=$OUT/$RUN DATASET_DIR=$GSM8K_DIR/evalscope/datasets \
        bash "$GSM8K_DIR/run-gsm8k.sh" "$RUN" 200 50 > "$OUT/$RUN.log" 2>&1
    echo "exit=$?"
    tail -c "+$((log_start + 1))" log.kvshrink-vllm > "$OUT/$RUN.vllm.log"
    REPORT=$(ls -1t "$OUT/$RUN"/*/reports/*/gsm8k.json 2>/dev/null | head -1)
    echo "report=$REPORT"
    [ -n "$REPORT" ] && python3 -c "import json; print(json.dumps(json.load(open('$REPORT')), ensure_ascii=False)[:600])"
    echo "lookups=$(grep -ac get_num_new_matched_tokens $OUT/$RUN.vllm.log)" \
        "ext_hit=$(grep -a get_num_new_matched_tokens $OUT/$RUN.vllm.log | grep -acv 'externally-cached tokens: 0,')" \
        "async=$(grep -a get_num_new_matched_tokens $OUT/$RUN.vllm.log | grep -ac 'async=True')" \
        "first_n_submits=$(grep -ac 'submit kind=async reqs' $OUT/$RUN.vllm.log)" \
        "remaining_submits=$(grep -ac 'submit kind=async_remaining' $OUT/$RUN.vllm.log)" \
        "errors=$(grep -acE 'Traceback|RuntimeError|ValueError' $OUT/$RUN.vllm.log)"
done
# Free GPU4,5.
P=$(ps -eo pid,args | grep -E "[v]llm serve .*--port 8010" | awk '{print $1}')
[ -n "$P" ] && docker exec $C kill -TERM "$P"
echo ALL_DONE
