#!/bin/bash
# usage: hybrid-correctness-1028.sh   (dev helper, run on the host in tmux)
# Correctness of the 1026 branch on the GDN hybrid Qwen3.5-27B (48 GDN + 16 attention
# layers, block_size 784): VLLM_BATCH_INVARIANT off (vLLM rejects it for GDN), so hits are
# judged by hybrid_round_summary.py (external tokens, answer to the prompt's question,
# U+FFFD / control characters) instead of byte identity. Two servers, 3 rounds each,
# 16 concurrent hits per round, ~20k-token prompts (1000 records, 25 full blocks):
# the default map (hybrid N = 48 / 56 / 64), then map 0-:4 (N = 52, 12 attention tail
# layers raised for every hit). Results: results/hybrid-<map>-<time>/.
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE" || exit 1
STAMP=$(date +%Y%m%d_%H%M%S)
for MAP_NAME in default tail12; do
    OUT=$HERE/results/hybrid-$MAP_NAME-$STAMP
    MAP=""
    [ "$MAP_NAME" = tail12 ] && MAP=0-:4
    echo "$(date +%T) start $MAP_NAME -> $OUT"
    OUT=$OUT MAP=$MAP MODEL=/media/model-space/Qwen3.5-27B BATCH_INVARIANT=0 ROUNDS=3 \
        CONC=16 RECORDS=1000 ./correctness-1026.sh > "/tmp/hybrid-$MAP_NAME.out" 2>&1
    if ! grep -q ALL_DONE "$OUT/run.log"; then
        echo "$(date +%T) FAILED $MAP_NAME (see $OUT/run.log)"
        exit 1
    fi
    python3 "$HERE/hybrid_round_summary.py" "$OUT" > "$OUT/summary.txt" 2>&1
    echo "$(date +%T) done $MAP_NAME"
done
echo "$(date +%T) ALL_DONE $STAMP"
