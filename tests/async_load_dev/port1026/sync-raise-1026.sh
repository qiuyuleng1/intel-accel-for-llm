#!/bin/bash
# usage: sync-raise-1026.sh   (dev helper, run on the host in tmux)
# Targeted test of the 1026 PR branch: remaining layers raised in the same step as a sync
# batch (sync_raise_check.py). Qwen3-32B TP=2, VLLM_BATCH_INVARIANT=1,
# KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP=0-1:16,2-:0 (one request in flight: async with
# N=16; two or more: sync). Results: results/sync-raise-<time>/.
HERE=$(cd "$(dirname "$0")" && pwd)
MODEL=/media/model-space/Qwen3-32B
DELAYS=${DELAYS:-$(seq -s, 0 2 120)}
REPS=${REPS:-2}
OUT=${OUT:-$HERE/results/sync-raise-$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1

echo "branch=$(git -C /home/pese/qiuyu/intel-accel-for-llm-1026 branch --show-current)" \
     "commit=$(git -C /home/pese/qiuyu/intel-accel-for-llm-1026 rev-parse --short HEAD)" \
     "connector_md5=$(md5sum /home/pese/qiuyu/intel-accel-for-llm-1026/kvshrink/kvshrink_connector.py | cut -c1-8)" \
     "delays_ms=$DELAYS reps=$REPS"
"$HERE/serve-1026.sh" "$OUT/vllm.log" VLLM_BATCH_INVARIANT=1 \
    KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP=0-1:16,2-:0

export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1
python3 "$HERE/sync_raise_check.py" --model "$MODEL" --vllm-log "$OUT/vllm.log" \
    --out "$OUT/check" --delays-ms "$DELAYS" --reps "$REPS"
echo "check_exit=$?"
python3 "$HERE/summarize_log.py" "$OUT/vllm.log"

P=$(ps -eo pid,args | grep -E "[v]llm serve .*--port 8010" | awk '{print $1}')
[ -n "$P" ] && docker exec iaxl.vllm.qiuyu-1026 kill -TERM $P
echo ALL_DONE
