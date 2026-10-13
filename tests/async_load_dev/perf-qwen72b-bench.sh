#!/bin/bash
# usage: perf-qwen72b-bench.sh   (dev helper; run on the host, e.g. in tmux)
#
# TTFT benchmark of Qwen2.5-72B-Instruct (TP=4, GPU4-7, QAT + IAA + DSA) for the
# three KVSHRINK_ASYNC_LOAD_SCHEME values of the PR branch: naive,
# batch_reqs_async_load_submit and split_async_load_submit. Each scheme starts its
# own server with perf-qwen72b-serve.sh, runs a one-time prewarm, then the
# hit-rate x concurrency matrix with `vllm bench serve` inside the container; the
# DDR cache is evicted after the prewarm and after every cell. Results go to
# _data/perf-qwen72b/<timestamp>/ (summary.csv, per-cell logs/JSON, server logs).
#
# Precondition: DSA work queues on NUMA1 are enabled (dsa-numa1.sh enable).
set -o pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
HERE="$REPO/tests/async_load_dev"
C=${C:-iaxl.vllm.qiuyu-v29}
MODEL=${MODEL:-/home/pese/model-space/Qwen2.5-72B-Instruct}
PORT=8010
CONTROLLER=http://localhost:18710

SCHEMES=${SCHEMES:-"naive batch_reqs_async_load_submit split_async_load_submit"}
INPUT_LEN=${INPUT_LEN:-8000}
OUTPUT_LEN=${OUTPUT_LEN:-128}
HIT_RATES=${HIT_RATES:-"80 100"}
CONCURRENCIES=${CONCURRENCIES:-"1 4 8 12"}
# Measured prompts per concurrency ("concurrency:num_prompts"); at most 96 to bound
# host memory.
PROMPTS_MAP=${PROMPTS_MAP:-"1:16 4:48 8:96 12:96"}
# Per-cell warmup requests; they also store the shared prefix in DDR.
NUM_WARMUPS=${NUM_WARMUPS:-5}
# One-time prewarm after each server start (torch.compile, QAT/IAA/DSA, DDR pool).
PREWARM_CONC=${PREWARM_CONC:-12}
PREWARM_PROMPTS=${PREWARM_PROMPTS:-24}
BASE_SEED=${BASE_SEED:-10000}
# DRY_RUN=1: print every step; start nothing, send no request, write no file.
DRY_RUN=${DRY_RUN:-0}

OUT=${OUT:-$REPO/_data/perf-qwen72b/$(date +'%Y%m%d_%H%M%S')}
if [ "$DRY_RUN" != "1" ]; then
    docker exec $C bash -c "mkdir -p $OUT && chown $(id -u):$(id -g) $OUT"
    exec > >(tee -a "$OUT/run.log") 2>&1
fi
SUMMARY_CSV="$OUT/summary.csv"

stop_server() {
    if [ "$DRY_RUN" = "1" ]; then
        echo "    [dry-run] stop port-$PORT vLLM"
        return
    fi
    local pid
    pid=$(ps -eo pid,args | grep -E "[v]llm serve .*--port $PORT" | awk '{print $1}')
    if [ -n "$pid" ]; then
        docker exec $C kill -TERM "$pid"
    fi
}
trap stop_server EXIT

evict_ddr_cache() {
    if [ "$DRY_RUN" = "1" ]; then
        echo "    [dry-run] POST $CONTROLLER/v1/cache/evict"
        return
    fi
    sleep 2 # let in-flight KV stores settle
    curl -sf -X POST "$CONTROLLER/v1/cache/evict" -d '{"count": 500000000000}' > /dev/null \
        || echo "    WARNING: evict failed"
}

reclaim_host_memory() {
    # QAT's usdm driver needs 2 MB physically contiguous pages, allocated after the
    # pinned scratch pools; on this shared host the page cache fragments
    # memory and QAT init failed with "page allocation failure: order:9".
    # Drop clean page cache and compact NUMA node1 only, where QAT 4-7 and GPU4-7
    # live: whole-host compaction took minutes (root in the container).
    if [ "$DRY_RUN" = "1" ]; then
        echo "    [dry-run] drop_caches=3 + node1 compact"
        return
    fi
    docker exec $C bash -c "sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /sys/devices/system/node/node1/compact"
    echo "    reclaimed host memory: $(free -g | awk '/^Mem:/ {print "free=" $4 "G available=" $7 "G"}')"
    grep Normal /proc/buddyinfo | sed 's/^/    /'
}

check_server_alive() {
    # check_server_alive <stage> <serve_log>: abort the whole run if vLLM died.
    if [ "$DRY_RUN" = "1" ]; then
        echo "    [dry-run] check $PORT/health after $1"
        return
    fi
    if ! curl -sf "http://localhost:$PORT/health" > /dev/null; then
        echo "ERROR: server died during $1, see $2"
        exit 1
    fi
}

num_prompts_for() {
    local concurrency=$1
    local pair
    for pair in $PROMPTS_MAP; do
        if [ "${pair%%:*}" = "$concurrency" ]; then
            echo "${pair##*:}"
            return
        fi
    done
    echo "ERROR: no PROMPTS_MAP entry for concurrency $concurrency" >&2
    exit 1
}

bench() {
    # bench <hit_rate> <concurrency> <num_prompts> <num_warmups> <seed> <log> [extra args]
    local hit=$1 concurrency=$2 num_prompts=$3 num_warmups=$4 seed=$5 log=$6
    shift 6
    local prefix_len=$((INPUT_LEN * hit / 100))
    local random_len=$((INPUT_LEN - prefix_len))
    local cmd="vllm bench serve \
        --backend vllm --model $MODEL --tokenizer $MODEL --trust-remote-code \
        --dataset-name random --host localhost --port $PORT \
        --random-prefix-len $prefix_len --random-input-len $random_len \
        --random-output-len $OUTPUT_LEN --ignore-eos \
        --percentile-metrics ttft,tpot --metric-percentiles 50,95 \
        --request-rate inf --max-concurrency $concurrency \
        --num-prompts $num_prompts --num-warmups $num_warmups --seed $seed $*"
    if [ "$DRY_RUN" = "1" ]; then
        echo "    [dry-run] in $C: $(echo $cmd)"
        return
    fi
    docker exec -w "$REPO" $C bash -c "export no_proxy=localhost,127.0.0.1; $cmd" > "$log" 2>&1
}

append_csv() {
    # append_csv <scheme> <hit_rate> <concurrency> <num_prompts> <result_json>
    python3 - "$SUMMARY_CSV" "$@" <<'PY'
import csv
import json
import os
import sys

summary, scheme, hit, concurrency, num_prompts, result_json = sys.argv[1:7]
metrics = [
    "completed", "duration", "request_throughput",
    "mean_ttft_ms", "median_ttft_ms", "p95_ttft_ms",
    "mean_tpot_ms", "median_tpot_ms", "p95_tpot_ms",
]
row = {
    "scheme": scheme,
    "hit_rate": hit,
    "concurrency": concurrency,
    "num_prompts": num_prompts,
}
try:
    with open(result_json) as f:
        result = json.load(f)
except OSError as error:
    sys.stderr.write(f"    WARNING: cannot read {result_json}: {error}\n")
    result = {}
for metric in metrics:
    value = result.get(metric, "")
    if isinstance(value, float):
        value = round(value, 2)
    row[metric] = value

write_header = not os.path.exists(summary)
with open(summary, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    if write_header:
        writer.writeheader()
    writer.writerow(row)
print(f"    -> {scheme} hit={hit} conc={concurrency}: "
      f"mean_ttft={row['mean_ttft_ms']} median_ttft={row['median_ttft_ms']} "
      f"p95_ttft={row['p95_ttft_ms']} ms")
PY
}

run_scheme() {
    local scheme=$1
    local serve_log="$OUT/$scheme.serve.log"
    echo "===== scheme=$scheme connector_md5=$(md5sum "$REPO/kvshrink/kvshrink_connector.py" | cut -c1-8) ====="

    reclaim_host_memory
    DRY_RUN=$DRY_RUN "$HERE/perf-qwen72b-serve.sh" "$scheme" "$serve_log"
    echo "    serve log -> $serve_log"
    if [ "$DRY_RUN" != "1" ] && ! curl -sf "http://localhost:$PORT/health" > /dev/null; then
        echo "ERROR: server for $scheme did not start, see $serve_log"
        exit 1
    fi

    echo "    prewarm: conc=$PREWARM_CONC prompts=$PREWARM_PROMPTS hit=100"
    bench 100 "$PREWARM_CONC" "$PREWARM_PROMPTS" 0 "$BASE_SEED" "$OUT/$scheme.prewarm.log"
    check_server_alive "$scheme prewarm" "$serve_log"
    evict_ddr_cache

    local hit concurrency num_prompts tag
    for hit in $HIT_RATES; do
        for concurrency in $CONCURRENCIES; do
            num_prompts=$(num_prompts_for "$concurrency")
            tag="${scheme}_hr${hit}_c${concurrency}"
            echo "    --- $tag prompts=$num_prompts ---"
            bench "$hit" "$concurrency" "$num_prompts" "$NUM_WARMUPS" \
                $((BASE_SEED + concurrency + hit)) "$OUT/$tag.log" \
                --save-result --result-dir "$OUT" --result-filename "$tag.json"
            if [ "$DRY_RUN" != "1" ]; then
                grep -a "Successful requests" "$OUT/$tag.log"
                check_server_alive "$tag" "$serve_log"
                append_csv "$scheme" "$hit" "$concurrency" "$num_prompts" "$OUT/$tag.json"
            fi
            evict_ddr_cache
        done
    done

    stop_server
}

if [ "$(git -C "$REPO" branch --show-current)" != "perf/async-load-optimization" ]; then
    echo "ERROR: check out perf/async-load-optimization first"
    exit 1
fi
for wq in wq8.0 wq10.0 wq12.0 wq14.0; do
    if [ "$(cat /sys/bus/dsa/devices/$wq/state)" != "enabled" ]; then
        echo "ERROR: DSA $wq not enabled; run: docker exec -w $REPO $C bash tests/async_load_dev/dsa-numa1.sh enable"
        exit 1
    fi
done

echo "out=$OUT schemes=$SCHEMES input=$INPUT_LEN output=$OUTPUT_LEN hit=$HIT_RATES" \
     "conc=$CONCURRENCIES prompts=$PROMPTS_MAP warmups=$NUM_WARMUPS"
for scheme in $SCHEMES; do
    run_scheme "$scheme"
done
echo ALL_DONE
