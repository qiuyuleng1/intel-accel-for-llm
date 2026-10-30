#!/bin/bash
# usage: correctness-1026.sh   (dev helper, run on the host in tmux)
# Output correctness of the 1026 PR branch under concurrency: Qwen3-32B TP=2,
# VLLM_BATCH_INVARIANT=1, map from 1026/setvars.sh (0-3:0,4-6:8,7-:16). Each round sends
# one ~5k-token prompt (cache miss, reference), then the same prompt $CONC times at once
# (DDR hits; with in-flight 1-3 sync, 4-6 async N=8, >=7 async N=16) and checks every
# hit output byte-for-byte against the reference, no U+FFFD / control characters, and
# externally-cached tokens > 0 (tests/async_load_correctness_check.py). $ROUNDS rounds,
# each with a fresh prompt. Results: results/correctness-<time>/.
# MODEL=<path> picks another model (default Qwen3-32B); RECORDS sets the prompt length
# (default 260 records, ~5k Qwen3 tokens); BATCH_INVARIANT=0 turns VLLM_BATCH_INVARIANT off
# (vLLM rejects it for GDN models; hits then need not be byte-identical); MAP overrides
# KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP.
HERE=$(cd "$(dirname "$0")" && pwd)
DEV=/home/pese/qiuyu/intel-accel-for-llm-1008
export MODEL=${MODEL:-/media/model-space/Qwen3-32B}
RECORDS=${RECORDS:-260}
SERVE_VARS=(VLLM_BATCH_INVARIANT=${BATCH_INVARIANT:-1})
if [ -n "$MAP" ]; then
    SERVE_VARS+=(KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP=$MAP)
fi
CONC=${CONC:-16}
ROUNDS=${ROUNDS:-3}
OUT=${OUT:-$HERE/results/correctness-$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1

echo "branch=$(git -C /home/pese/qiuyu/intel-accel-for-llm-1026 branch --show-current)" \
     "connector_md5=$(md5sum /home/pese/qiuyu/intel-accel-for-llm-1026/kvshrink/kvshrink_connector.py | cut -c1-8)" \
     "conc=$CONC rounds=$ROUNDS model=$MODEL records=$RECORDS serve_vars=${SERVE_VARS[*]}"
"$HERE/serve-1026.sh" "$OUT/vllm.log" "${SERVE_VARS[@]}" || exit 1

export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1
for ROUND in $(seq 1 "$ROUNDS"); do
    echo "===== round $ROUND ====="
    log_start=$(stat -c %s "$OUT/vllm.log")
    python3 "$DEV/tests/async_load_correctness_check.py" --api http://localhost:8010/v1/completions \
        --model "$MODEL" --vllm-log "$OUT/vllm.log" --concurrency "$CONC" --out "$OUT/round$ROUND" \
        --records "$RECORDS" \
        > "$OUT/round$ROUND.log" 2>&1
    echo "check_exit=$?"
    grep -E '"verdict"|"prompt_tokens"' "$OUT/round$ROUND.log"
    python3 "$HERE/summarize_log.py" "$OUT/vllm.log" "$log_start"
done
echo "===== whole server log ====="
python3 "$HERE/summarize_log.py" "$OUT/vllm.log"

P=$(ps -eo pid,args | grep -E "[v]llm serve .*--port 8010" | awk '{print $1}')
[ -n "$P" ] && docker exec iaxl.vllm.qiuyu-1026 kill -TERM $P
echo ALL_DONE
