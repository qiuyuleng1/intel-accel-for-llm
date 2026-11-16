#!/bin/bash
# Kimi-K3 MLA hook check: Qwen3.8-27B gsm8k (30 questions), then Kimi-K3 smoke and gsm8k (26 shots, 30 questions).
# On the b300 host, in tmux: bash tests/bench_dev/mla_hook_check.sh
set -uo pipefail

REPO_DIR=/home/intel/qiuyu/intel-accel-for-llm-kimi-k3
CONTAINER=iaxl.vllm-kimi-k3-qiuyu
SERVE_STATE=tests/bench_dev/results/kimi-k3-serve.state

log() {
    echo "[$(date +%H:%M:%S)] $*"
}

log "step 1: Qwen3.8-27B gsm8k, 30 questions x 2 passes"
docker exec -w "$REPO_DIR" -e LIMIT=30 "$CONTAINER" \
    bash -lc 'bash tests/bench_dev/qwen38_gsm8k_two_passes.sh'
log "step 2: Kimi-K3 smoke"
docker exec -w "$REPO_DIR" "$CONTAINER" bash -lc 'bash tests/bench_dev/kimi_k3_smoke.sh'
log "step 3: Kimi-K3 gsm8k, 26 shots, 30 questions x 2 passes"
docker exec -w "$REPO_DIR" -e SHOTS=26 -e LIMIT=30 "$CONTAINER" \
    bash -lc 'bash tests/bench_dev/kimi_k3_gsm8k.sh'

log "step 4: stop the Kimi-K3 vLLM started by the smoke test"
docker exec -w "$REPO_DIR" "$CONTAINER" bash -lc \
    "if [[ -f $SERVE_STATE ]]; then read -r pgid log < $SERVE_STATE; kill -TERM -- -\$pgid; rm -f $SERVE_STATE; fi"
waited=0
while (( waited < 300 )); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{sum += $1} END {print sum}')
    if (( used == 0 )); then
        break
    fi
    sleep 5
    waited=$((waited + 5))
done
log "GPU memory used after ${waited}s: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
log "ALL_DONE"
