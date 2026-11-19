#!/bin/bash
# Qwen2.5-72B-Instruct at TP=8: kvshrink, vllm-prefix and vllm-wo-prefix
# (lmcache is not tested on b300 for now) at vLLM block sizes 16 and 256.
# Input 32000 tokens, output 128, concurrency 1 8 16 32 with
# 20 160 256 256 requests, hit rates 80 100 (vllm-wo-prefix: 80 only).
#
# Runs on the HOST. Every config is one kvshrink-benchmark.sh call:
#   kvshrink, vllm-prefix, vllm-wo-prefix: docker exec into the KVShrink
#       container. If it does not exist, ./start.sh starts it in a detached tmux
#       session and this script waits until `pip install -e .` finished.
#   lmcache: a new container of the LMCache image (docker run --rm).
# kvshrink stores its first layer uncompressed (IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS=1).
# A failed config is reported and the next one still runs.
#
# Results: bench-results/Qwen2.5-72B-Instruct_tp8_<time>/bs16 and /bs256, each
# with its own summary.csv, run.log and per-config serve logs.
# Print the plan without starting any serve or container:
#   DRY_RUN=1 bash test-qwen25-72B.sh
# Only some configs (same order as listed): CONFIGS=kvshrink bash test-qwen25-72B.sh
set -uo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$REPO_ROOT"

MODEL=/home/intel/model-space/Qwen2.5-72B-Instruct
BLOCK_SIZES="16 256"
# vllm-wo-prefix last: it is the longest and the only one that reuses nothing.
# Run a subset with e.g. CONFIGS=kvshrink.
CONFIGS=${CONFIGS:-"kvshrink vllm-prefix vllm-wo-prefix"}
DRY_RUN=${DRY_RUN:-0}
KVSHRINK_CONTAINER=$(sed -n 's/^CONTAINER_NAME=//p' setvars.sh)
# tmux session names cannot contain '.'.
KVSHRINK_TMUX_SESSION=${KVSHRINK_CONTAINER//./-}
LMCACHE_IMAGE=qiuyu/vllm-openai-lmcache:v0.33.0-lmcache0.5.6
LMCACHE_CONTAINER=iaxl.lmcache-bench
PERFSPECT_DIR=/home/intel/qiuyu/perfspect
MODEL_TAG=$(basename "$MODEL")
if [ "$DRY_RUN" = "1" ]; then
    RESULTS_ROOT=tests/bench_dev/results/dryrun-${MODEL_TAG}_tp8_$(date +%Y%m%d_%H%M%S)
else
    RESULTS_ROOT=bench-results/${MODEL_TAG}_tp8_$(date +%Y%m%d_%H%M%S)
fi
# Shared by every kvshrink-benchmark.sh call. MAX_MODEL_LEN is the model's
# max_position_embeddings (the script's default 201128 is for Qwen3.8).
BENCH_ENV="MODEL=$MODEL TP_SIZE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MAX_MODEL_LEN=32768"
BENCH_ENV+=" LEN_CONCURRENCY=32000:1,8,16,32 HIT_RATES='80 100'"
BENCH_ENV+=" NUM_PROMPTS_BY_CONCURRENCY='1:20 8:160 16:256 32:256'"
BENCH_ENV+=" IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS=1 DRY_RUN=$DRY_RUN"

mkdir -p "$RESULTS_ROOT"
exec > >(tee -a "$RESULTS_ROOT/test-qwen25-72B.log") 2>&1
cp "${BASH_SOURCE[0]}" "$RESULTS_ROOT/"

# Every given file must hold the expected value; prints the value counts.
expect_all_files() {
    local label=$1 expected=$2
    shift 2
    echo "  $label: $(cat "$@" | sort | uniq -c | tr -s ' ' | tr '\n' ';') expected all $expected"
    if cat "$@" | grep -qvx "$expected"; then
        echo "  ERROR: $label is not $expected everywhere" >&2
        return 1
    fi
}

# The host settings of PERFORMANCE_MODE_SETUP.md. A reboot resets them, and a
# run with any of them off is not comparable with the others (ACS on alone
# doubles every prefill at TP=8).
check_host_settings() {
    local failed=0
    local pstate_status cstate_names elc
    local state_dir state_name expected_disable

    pstate_status=$(cat /sys/devices/system/cpu/intel_pstate/status)
    echo "  intel_pstate status: $pstate_status expected active"
    if [ "$pstate_status" != "active" ]; then
        echo "  ERROR: intel_pstate is not active" >&2
        failed=1
    fi
    expect_all_files "scaling driver" intel_pstate /sys/devices/system/cpu/cpufreq/policy*/scaling_driver || failed=1
    expect_all_files "scaling governor" performance /sys/devices/system/cpu/cpufreq/policy*/scaling_governor || failed=1
    expect_all_files "EPB" 0 /sys/devices/system/cpu/cpu[0-9]*/power/energy_perf_bias || failed=1
    expect_all_files "no_turbo" 0 /sys/devices/system/cpu/intel_pstate/no_turbo || failed=1

    # Only POLL and C1 may be entered. b300 still lists C1E, C6 and C6P: they
    # are switched off through cpuidle sysfs, so each one must be disabled on
    # every CPU instead of being absent.
    cstate_names=$(cat /sys/devices/system/cpu/cpu0/cpuidle/state*/name | xargs)
    echo "  C-states: $cstate_names expected POLL C1 enabled, every other one disabled"
    for state_dir in /sys/devices/system/cpu/cpu0/cpuidle/state*; do
        state_name=$(cat "$state_dir/name")
        expected_disable=1
        if [ "$state_name" = "POLL" ] || [ "$state_name" = "C1" ]; then
            expected_disable=0
        fi
        expect_all_files "$state_name disable flag" "$expected_disable" \
            /sys/devices/system/cpu/cpu[0-9]*/cpuidle/"$(basename "$state_dir")"/disable || failed=1
    done

    elc=$(cd "$PERFSPECT_DIR" && sudo -n ./perfspect --noupdate config 2>&1 \
        | grep -oE "Efficiency Latency Control: +.*\)" | tr -s ' ')
    echo "  $elc expected Latency Optimized Mode (LOM)"
    if [[ "$elc" != *"Latency Optimized Mode (LOM)"* ]]; then
        echo "  ERROR: ELC is not LOM" >&2
        failed=1
    fi

    bash tools/check-acs.sh || failed=1
    return "$failed"
}

# QAT's usdm driver needs physically contiguous 4 MiB buffers once KVShrink
# compression starts. With the page cache filling host memory none was left and
# the serve died (page allocation failure order:10, Nov 5), so drop the page
# cache and compact both NUMA nodes before every serve start, as
# intel-accel-for-llm-1026 tests/python_overhead_dev/run_pyspy_bench.sh does.
reclaim_host_memory() {
    # Disabled (Nov 11): h8263 and b300 both have enough memory, and on b300
    # (2 TB) the compaction took ~3 minutes before every serve.
    return 0
    # sync
    # if ! sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches; for node in /sys/devices/system/node/node*/compact; do echo 1 > "$node"; done'; then
    #     echo "WARNING: could not drop caches / compact memory" >&2
    #     return 1
    # fi
    # # Last buddyinfo column: free blocks of order 10 = 4 MiB.
    # echo "  free 4 MiB blocks per NUMA node: $(awk '$4 == "Normal" {print $NF}' /proc/buddyinfo | xargs)"
}

# start.sh runs the container in the foreground (docker run -it --rm ... exec
# bash): it needs a terminal and the container is removed when that shell
# exits, so it runs in a detached tmux session that outlives this script.
ensure_kvshrink_container() {
    local started=0
    local main_pid
    if ! docker container inspect "$KVSHRINK_CONTAINER" > /dev/null 2>&1; then
        echo "$KVSHRINK_CONTAINER does not exist; running ./start.sh in tmux session $KVSHRINK_TMUX_SESSION"
        if ! tmux new-session -d -s "$KVSHRINK_TMUX_SESSION" -c "$REPO_ROOT" "MODEL=$MODEL ./start.sh"; then
            echo "ERROR: cannot create tmux session $KVSHRINK_TMUX_SESSION" >&2
            return 1
        fi
        # Keep the pane after start.sh exits so its error stays readable.
        tmux set-option -t "$KVSHRINK_TMUX_SESSION" remain-on-exit on > /dev/null
        started=1
    fi
    echo "waiting for $KVSHRINK_CONTAINER to finish pip install -e . ($(date +%T))"
    while true; do
        main_pid=$(docker inspect -f '{{.State.Pid}}' "$KVSHRINK_CONTAINER" 2> /dev/null || echo 0)
        # start.sh is done once the container's main process has exec'd a plain bash.
        if [ "$main_pid" != "0" ] && [ "$(tr '\0' ' ' < "/proc/$main_pid/cmdline" 2> /dev/null)" = "bash " ]; then
            break
        fi
        if [ "$started" = "1" ] && [ "$(tmux display-message -p -t "$KVSHRINK_TMUX_SESSION" '#{pane_dead}')" = "1" ]; then
            echo "ERROR: ./start.sh exited before $KVSHRINK_CONTAINER was ready; see: tmux attach -t $KVSHRINK_TMUX_SESSION" >&2
            return 1
        fi
        sleep 20
    done
    echo "$KVSHRINK_CONTAINER ready ($(date +%T))"
    if ! docker exec "$KVSHRINK_CONTAINER" test -d "$MODEL"; then
        echo "ERROR: $KVSHRINK_CONTAINER does not see $MODEL (setvars.sh mounts /home/intel/model-space); exit that container's shell and rerun" >&2
        return 1
    fi
}

# One kvshrink-benchmark.sh call for one config at one block size.
run_config() {
    local config=$1 block_size=$2 results_dir=$3
    local bench_cmd="CONFIGS=$config RESULTS_DIR=$results_dir BLOCK_SIZE=$block_size $BENCH_ENV bash kvshrink-benchmark.sh"
    if [ "$config" = "lmcache" ]; then
        docker run --rm --name "$LMCACHE_CONTAINER" --gpus all --network host --ipc host \
            -v "$REPO_ROOT:$REPO_ROOT" -v /media/model-space:/media/model-space -w "$REPO_ROOT" \
            --entrypoint bash "$LMCACHE_IMAGE" \
            -c "LMCACHE_CHUNK_SIZE=$block_size $bench_cmd"
    else
        docker exec -w "$REPO_ROOT" "$KVSHRINK_CONTAINER" bash -lc "$bench_cmd"
    fi
}

echo "==========================================================="
echo "Qwen2.5-72B test  model=$MODEL  block_sizes=[$BLOCK_SIZES]  configs=[$CONFIGS]"
echo "  $BENCH_ENV"
echo "  KVShrink container=$KVSHRINK_CONTAINER  LMCache image=$LMCACHE_IMAGE"
echo "  results -> $RESULTS_ROOT"
echo "==========================================================="

# Fail before starting anything when the host is not in performance mode.
echo "Host settings (PERFORMANCE_MODE_SETUP.md):"
if ! check_host_settings && [ "$DRY_RUN" != "1" ]; then
    echo "ERROR: host settings differ from PERFORMANCE_MODE_SETUP.md; restore them first" >&2
    exit 1
fi

# The KVShrink container is a precondition: fail now, while someone is watching.
if [ "$DRY_RUN" != "1" ] && ! ensure_kvshrink_container; then
    exit 1
fi

failed=""
for block_size in $BLOCK_SIZES; do
    results_dir=$RESULTS_ROOT/bs$block_size
    for config in $CONFIGS; do
        echo ""
        echo "### bs$block_size $config ($(date +'%F %T')) -> $results_dir"
        if [ "$DRY_RUN" != "1" ]; then
            reclaim_host_memory
        fi
        run_config "$config" "$block_size" "$results_dir"
        status=$?
        if [ "$status" != "0" ]; then
            echo "ERROR: bs$block_size $config exited with status $status" >&2
            failed+=" bs$block_size/$config"
        fi
    done
done

echo ""
echo "Done ($(date +'%F %T')): $RESULTS_ROOT/bs*/summary.csv"
if [ -n "$failed" ]; then
    echo "FAILED:$failed" >&2
    exit 1
fi
