#!/bin/bash
# usage: rebase-regress-1028.sh   (dev helper, run on the host in tmux)
# One-off regression of the 1026 branch after the 10-28 rebase onto upstream dde3b4f,
# within a 10-minute budget: correctness-1026.sh with 3 rounds (own server,
# VLLM_BATCH_INVARIANT=1, default map), then GSM8K async run1 only (own server, same
# settings as gsm8k-1026.sh async). Results: results/correctness-<time>/ and
# results/gsm8k-rebase-<time>/.
HERE=$(cd "$(dirname "$0")" && pwd)
GSM8K_DIR=/home/pese/qiuyu/Al-dsv4/KVCacheClip/internal/daocloud/test/accuracy-gsm8k
MODEL=/media/model-space/Qwen3-32B
cd "$HERE" || exit 1
echo "$(date +%T) correctness start"
ROUNDS=3 ./correctness-1026.sh > /tmp/rb1028-correctness.out 2>&1
echo "$(date +%T) correctness done"

OUT=$HERE/results/gsm8k-rebase-$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT/async"
echo "commit=$(git -C /home/pese/qiuyu/intel-accel-for-llm-1026 rev-parse --short HEAD)" > "$OUT/run.log"
./serve-1026.sh "$OUT/async/vllm.log" KVSHRINK_VLLM_KV_ASYNC_LOAD_ENABLED=1 >> "$OUT/run.log" 2>&1
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1
echo "$(date +%T) gsm8k run1 start" | tee -a "$OUT/run.log"
API_URL=http://127.0.0.1:8010/v1 MODEL=$MODEL OUTPUT_DIR=$OUT/async/run1 \
    DATASET_DIR=$GSM8K_DIR/evalscope/datasets \
    bash "$GSM8K_DIR/run-gsm8k.sh" run1 200 50 > "$OUT/async/run1.log" 2>&1
echo "exit=$? $(date +%T) gsm8k run1 done" | tee -a "$OUT/run.log"
python3 "$HERE/summarize_log.py" "$OUT/async/vllm.log" >> "$OUT/run.log" 2>&1
P=$(ps -eo pid,args | grep -E "[v]llm serve .*--port 8010" | awk '{print $1}')
[ -n "$P" ] && docker exec iaxl.vllm.qiuyu-1026 kill -TERM $P
echo "$(date +%T) ALL_DONE $OUT"
