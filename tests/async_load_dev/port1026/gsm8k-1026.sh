#!/bin/bash
# usage: gsm8k-1026.sh   (dev helper, run on the host in tmux)
# GSM8K accuracy of the 1026 PR branch, same procedure as tests/async_load_dev/gsm8k-split.sh
# (evalscope, 200 questions, concurrency 50, temperature 0; two runs on one server, DDR
# not evicted between them, so run2 also hits the questions saved by run1). Qwen3-32B
# TP=2, map from 1026/setvars.sh (0-3:0,4-6:8,7-:16). Then the same two runs with async
# load disabled (KVSHRINK_VLLM_KV_ASYNC_LOAD_ENABLED=0) as the sync baseline of this
# build. Results: results/gsm8k-<time>/{async,sync}/.
# MODEL=<path> picks another model (default Qwen3-32B); MODES lists the modes to run
# (default "async sync"; "plain" = vLLM without the KV connector, e.g. as the baseline of a
# hybrid model, whose loads are async even with async load disabled). FEW_SHOT=<n> runs
# evalscope directly with the same arguments as run-gsm8k.sh plus that few-shot count (default
# 4 gives ~700-token prompts, under one 784-token block of a GDN hybrid model: no hits);
# with FEW_SHOT, MAX_TOKENS sets the output limit (default 2048, as run-gsm8k.sh).
HERE=$(cd "$(dirname "$0")" && pwd)
GSM8K_DIR=/home/pese/qiuyu/Al-dsv4/KVCacheClip/internal/daocloud/test/accuracy-gsm8k
export MODEL=${MODEL:-/media/model-space/Qwen3-32B}
MODES=${MODES:-"async sync"}
OUT=${OUT:-$HERE/results/gsm8k-$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1

echo "branch=$(git -C /home/pese/qiuyu/intel-accel-for-llm-1026 branch --show-current)" \
     "connector_md5=$(md5sum /home/pese/qiuyu/intel-accel-for-llm-1026/kvshrink/kvshrink_connector.py | cut -c1-8)" \
     "model=$MODEL modes=$MODES few_shot=${FEW_SHOT:-4}"
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1
for MODE in $MODES; do
    mkdir -p "$OUT/$MODE"
    ENABLED=1
    [ "$MODE" = sync ] && ENABLED=0
    KVC=1
    [ "$MODE" = plain ] && KVC=0
    echo "===== $MODE (KVSHRINK_VLLM_KV_ASYNC_LOAD_ENABLED=$ENABLED KV_CONNECTOR=$KVC) ====="
    KV_CONNECTOR=$KVC "$HERE/serve-1026.sh" "$OUT/$MODE/vllm.log" KVSHRINK_VLLM_KV_ASYNC_LOAD_ENABLED=$ENABLED || exit 1
    for RUN in run1 run2; do
        log_start=$(stat -c %s "$OUT/$MODE/vllm.log")
        echo "--- gsm8k $MODE $RUN ---"
        if [ -n "$FEW_SHOT" ]; then
            mkdir -p "$OUT/$MODE/$RUN"
            evalscope eval --model "$MODEL" --api-url http://127.0.0.1:8010/v1 --api-key EMPTY \
                --eval-type openai_api --datasets gsm8k \
                --dataset-dir "$GSM8K_DIR/evalscope/datasets" --limit 200 --eval-batch-size 50 \
                --work-dir "$OUT/$MODE/$RUN" \
                --generation-config "{\"max_tokens\": ${MAX_TOKENS:-2048}, \"temperature\": 0.0}" \
                --dataset-args "{\"gsm8k\": {\"few_shot_num\": $FEW_SHOT}}" \
                > "$OUT/$MODE/$RUN.log" 2>&1
        else
            API_URL=http://127.0.0.1:8010/v1 MODEL=$MODEL \
                OUTPUT_DIR=$OUT/$MODE/$RUN DATASET_DIR=$GSM8K_DIR/evalscope/datasets \
                bash "$GSM8K_DIR/run-gsm8k.sh" "$RUN" 200 50 > "$OUT/$MODE/$RUN.log" 2>&1
        fi
        echo "exit=$?"
        REPORT=$(ls -1t "$OUT/$MODE/$RUN"/*/reports/*/gsm8k.json 2>/dev/null | head -1)
        echo "report=$REPORT"
        [ -n "$REPORT" ] && python3 -c "import json; r = json.load(open('$REPORT')); print('score', r.get('score'), 'num', r.get('metrics', [{}])[0].get('num'))"
        python3 "$HERE/summarize_log.py" "$OUT/$MODE/vllm.log" "$log_start"
    done
done

P=$(ps -eo pid,args | grep -E "[v]llm serve .*--port 8010" | awk '{print $1}')
[ -n "$P" ] && docker exec iaxl.vllm.qiuyu-1026 kill -TERM $P
echo ALL_DONE
