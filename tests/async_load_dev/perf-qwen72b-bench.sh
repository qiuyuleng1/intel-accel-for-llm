#!/bin/bash
# usage: perf-qwen72b-bench.sh   (dev helper; run on the host, e.g. in tmux)
#
# TTFT benchmark of Qwen2.5-72B-Instruct (TP=8, GPU0-7, QAT + IAA + DSA) for each
# entry of $SCHEMES (run from the dev branch): "sync" starts the server with
# KVSHRINK_VLLM_KV_ASYNC_LOAD_ENABLED=0 (every DDR hit loads synchronously); any
# other entry is a KVSHRINK_ASYNC_LOAD_SCHEME value with async load enabled and
# runs once per async-load layer map of $DYNAMIC_MAPS (label
# <scheme>_L<layers>, e.g. ..._L0-4-8 for "0-3:0,4-6:4,7-:8") and once per fixed
# layer count of $FIXED_LAYERS (KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC=0; label
# <scheme>_N<layers>, ..._Nall for -1). Each scheme / layer setting starts its
# own server with perf-qwen72b-serve.sh; then, for every input length of
# $INPUT_LENS, it runs a prewarm and $REPEATS times the hit-rate x concurrency
# matrix with `vllm bench serve` inside the container; the DDR cache is evicted
# after every prewarm and every cell. Results go to _data/perf-qwen72b/<timestamp>/
# (summary.csv, per-cell logs/JSON, server logs, failed_cells.txt).
#
# A server that dies (vLLM exits after a step hangs for 5 min) fails only the
# cell it was running: the cell is listed in failed_cells.txt instead of
# summary.csv, a new server (<label>.r<N>.serve.log) is started and prewarmed,
# and the run goes on with the next cell. After MAX_RESTARTS restarts of one
# label, or when a server does not start, the rest of that label is skipped and
# the next one runs.
#
# Precondition: DSA configured with tools/setup_dsa_cnt.sh (one user WQ per DSA).
set -o pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
HERE="$REPO/tests/async_load_dev"
C=${C:-iaxl.vllm.qiuyu-v29}
MODEL=${MODEL:-/home/pese/model-space/Qwen2.5-72B-Instruct}
PORT=8010
CONTROLLER=http://localhost:18710

# Async schemes still compared plus the sync baseline, in run order (others:
# naive, split_async_load_submit_first_n_batch_reqs,
# split_async_load_submit_first_n_per_req).
# 10-21 round: 32k fixed-layer sweep of priority_3level_first_n_per_req against
# sync. 20261020_065412 ran prio / promote_raise / batch with maps 8/16, 16/32,
# 40/79 at 8k / 16k / 32k, hit 80-100, c4-c32.
SCHEMES=${SCHEMES:-"priority_3level_first_n_per_req sync"}
# KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP values run by every async scheme
# ("START-END:LAYERS" rules over the number of in-flight requests); sync ignores
# them and runs once. 20261020_065412: with the map the 32k c4 / c8 cells mixed
# sync and two layer counts, and the async share of one cell differed by map
# (32k hit 98 c4: 25 / 47 / 68 % for 8/16, 16/32, 40/79), so the layer count
# could not be compared on its own. Map 16/32, the recommended one, stays as the
# reference of the fixed-layer sweep below.
DYNAMIC_MAPS=${DYNAMIC_MAPS-"0-3:0,4-6:16,7-:32"}
# KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS values run by every async scheme with
# KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC=0: every DDR hit loads async and
# prefill starts after that many layers (-1: after all 80 layers; 1..79 valid).
FIXED_LAYERS=${FIXED_LAYERS-"4 8 16 32 48 64 -1"}
# Layer setting of the current server, read by perf-qwen72b-serve.sh: a map or a
# fixed layer count, at most one of them (both empty: sync).
export DYNAMIC_MAP=""
export ASYNC_LOAD_LAYERS=""
# Input lengths run in order by every server; each + 128 output tokens must fit
# max-model-len 32765 of kvshrink-vllm-serve.sh.
INPUT_LENS=${INPUT_LENS:-"32000"}
OUTPUT_LEN=${OUTPUT_LEN:-128}
# Hit 80 is left out: at hit 80 every setting had the same end-to-end latency in
# 20261020_065412 and only moved time between TTFT and TPOT.
HIT_RATES=${HIT_RATES:-"95 98 100"}
CONCURRENCIES=${CONCURRENCIES:-"4 8"}
# Runs of the hit-rate x concurrency matrix per input length. The same cell run
# twice differed by up to 25 % (0.73-1.25 between 10-19 and 10-20), so one run
# cannot rank settings that are closer than that.
REPEATS=${REPEATS:-2}
# Highest concurrency per input length ("input_len:max_concurrency"); larger
# entries of $CONCURRENCIES are skipped for that length. A load takes
# 80 layers x (tokens / 16) blocks x 8 KiB of scratch per rank, so 8k c32, 16k
# c16 and 32k c8 all need ~9.8 GiB before stores; 16k c32 and 32k c16 would need
# ~19.5 GiB and do not fit the 16 GiB scratch pool (see the pool sizes below).
MAX_CONC_MAP=${MAX_CONC_MAP:-"16000:16 32000:8"}
# Measured prompts per concurrency ("concurrency:num_prompts"), 8 waves each
# except c1 and c16. The largest cells, measured as DDR bytes per rank at the end
# of the cell: 8k / hit 80 / c32 8.1 GiB and 16k / hit 80 / c16 9.9 GiB (10-16),
# 32k / hit 80 / c8 with 96 prompts 19.9 GiB (10-15); the DDR pool is evicted
# after every cell.
PROMPTS_MAP=${PROMPTS_MAP:-"1:16 4:32 8:64 16:96 32:160"}
# Per-cell warmup requests; they also store the shared prefix in DDR.
NUM_WARMUPS=${NUM_WARMUPS:-5}
# Prewarm before each input length's matrix (torch.compile, QAT/IAA/DSA, DDR pool).
PREWARM_CONC=${PREWARM_CONC:-32}
PREWARM_PROMPTS=${PREWARM_PROMPTS:-64}
# Per-rank pools, read by perf-qwen72b-serve.sh. Scratch 16 / DDR 25 GiB is the
# only setting with no hang so far (24 servers, 10-12..10-16); with scratch 24 or
# 32 GiB, 2 of 3 servers hung together with the whole host (journald and
# networkd stuck ~6 min, recovered when vLLM exited; cause unknown).
# ScratchPool.allocate() raises when the pool is exhausted, so it must hold every
# load and store in flight: a load takes 80 layers x (tokens / 16) blocks x 8 KiB
# per rank, i.e. 8k c32, 16k c16 and 32k c8 all need ~9.8 GiB before stores. The DDR
# pool size is only a budget (Mem capacity_bytes): a put over it still succeeds
# with "[Mem] WARNING: Over budget!" and nothing is evicted before
# evict_ddr_cache, so hit rates are unaffected.
export DDR_POOL_GB_PER_RANK=${DDR_POOL_GB_PER_RANK:-25}
export SCRATCH_POOL_GB_PER_RANK=${SCRATCH_POOL_GB_PER_RANK:-16}
BASE_SEED=${BASE_SEED:-10000}
# TRACE=1: per measured cell, record the OMP-Main queue trace of all ranks
# (<tag>.trace.json.gz), save per-request start times / TTFTs in <tag>.json
# (--save-detailed) and start the server with the async_load step / submit /
# promote / activate log lines (KVSHRINK_ASYNC_LOAD_EVENT_LOG=1). Profiling
# only: both add some per-task work, so compare traced runs with traced runs.
TRACE=${TRACE:-0}
export ASYNC_LOAD_EVENT_LOG=$TRACE
DETAIL_ARG=""
if [ "$TRACE" = "1" ]; then
    DETAIL_ARG="--save-detailed"
fi
# Server restarts allowed per scheme after a server died (see the header).
MAX_RESTARTS=${MAX_RESTARTS:-3}
# Labels to skip, to resume an interrupted run: rerun with the same OUT (run.log,
# summary.csv and failed_cells.txt are appended) and the finished labels, e.g.
# SKIP_LABELS="priority_3level_first_n_per_req_L0-4-8".
SKIP_LABELS=${SKIP_LABELS:-}
# DRY_RUN=1: print every step; start nothing, send no request, write no file.
DRY_RUN=${DRY_RUN:-0}

OUT=${OUT:-$REPO/_data/perf-qwen72b/$(date +'%Y%m%d_%H%M%S')}
if [ "$DRY_RUN" != "1" ]; then
    docker exec $C bash -c "mkdir -p $OUT && chown $(id -u):$(id -g) $OUT"
    exec > >(tee -a "$OUT/run.log") 2>&1
fi
SUMMARY_CSV="$OUT/summary.csv"
FAILED_CELLS="$OUT/failed_cells.txt"
# Current server of run_scheme: name prefix of its logs and its serve log.
SERVER_TAG=""
SERVE_LOG=""
RESTARTS=0

stop_server() {
    # Stop every port-$PORT vLLM server and `vllm bench serve` client: a client
    # left by an interrupted run keeps its connections open, and the server's
    # graceful shutdown then waits for them forever.
    if [ "$DRY_RUN" = "1" ]; then
        echo "    [dry-run] stop port-$PORT vLLM"
        return
    fi
    local pids
    pids=$(ps -eo pid,args | grep -E "[v]llm (serve|bench serve) .*--port $PORT" | awk '{print $1}')
    if [ -n "$pids" ]; then
        docker exec $C kill -TERM $pids
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
    # Drop clean page cache and compact both NUMA nodes (TP=8 uses QAT/GPU on
    # both); whole-host compaction took minutes (root in the container).
    if [ "$DRY_RUN" = "1" ]; then
        echo "    [dry-run] drop_caches=3 + node0/node1 compact"
        return
    fi
    docker exec $C bash -c "sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /sys/devices/system/node/node0/compact; echo 1 > /sys/devices/system/node/node1/compact"
    echo "    reclaimed host memory: $(free -g | awk '/^Mem:/ {print "free=" $4 "G available=" $7 "G"}')"
    grep Normal /proc/buddyinfo | sed 's/^/    /'
}

server_alive() {
    # server_alive: true while our port-$PORT vLLM answers /health.
    if [ "$DRY_RUN" = "1" ]; then
        return 0
    fi
    curl -sf "http://localhost:$PORT/health" > /dev/null
}

record_failure() {
    # record_failure <what>: report a failed server start / prewarm / cell and
    # append it with the current serve log to $FAILED_CELLS.
    echo "ERROR: $1 failed, see $SERVE_LOG"
    if [ "$DRY_RUN" != "1" ]; then
        echo "$1 $SERVE_LOG" >> "$FAILED_CELLS"
    fi
}

wait_server_gone() {
    # wait_server_gone: after stop_server, wait until every port-$PORT vLLM
    # server / bench client has exited and GPU0-7 memory is released. Those
    # processes and the GPU processes that belong to container $C and are still
    # alive after 3 min are killed; if anything is left after 4 min the whole run
    # stops, since no new server could start.
    if [ "$DRY_RUN" = "1" ]; then
        echo "    [dry-run] wait for port-$PORT vLLM to exit and GPU0-7 to free"
        return
    fi
    local i pids used leftover container_id
    container_id=$(docker inspect -f '{{.Id}}' $C)
    for i in $(seq 1 48); do
        pids=$(ps -eo pid,args | grep -E "[v]llm (serve|bench serve) .*--port $PORT" | awk '{print $1}')
        used=$(nvidia-smi -i 0,1,2,3,4,5,6,7 --query-gpu=memory.used --format=csv,noheader,nounits \
            | sort -n | tail -1)
        if [ -z "$pids" ] && [ "$used" -lt 1024 ]; then
            return
        fi
        if [ "$i" = "36" ]; then
            for leftover in $pids $(nvidia-smi -i 0,1,2,3,4,5,6,7 --query-compute-apps=pid \
                    --format=csv,noheader); do
                if grep -q "$container_id" "/proc/$leftover/cgroup" 2> /dev/null; then
                    echo "    killing leftover process $leftover of $C"
                    docker exec $C kill -9 "$leftover"
                fi
            done
        fi
        sleep 5
    done
    echo "ERROR: port-$PORT vLLM processes ($(echo $pids)) or GPU memory (max $used MiB) did not go away, stopping the run"
    exit 1
}

queue_trace() {
    # queue_trace <enable 0|1> <out_json>: drain the OMP-Main trace of all ranks
    # into <out_json> (gzipped), then set tracing on/off. No-op unless TRACE=1.
    if [ "$TRACE" != "1" ]; then
        return
    fi
    if [ "$DRY_RUN" = "1" ]; then
        echo "    [dry-run] GET $CONTROLLER/v1/cache/queue_trace?queue=OMP-Main&enable=$1 > $2"
        return
    fi
    curl -sf --max-time 300 "$CONTROLLER/v1/cache/queue_trace?queue=OMP-Main&enable=$1" > "$2" \
        || echo "    WARNING: queue_trace enable=$1 failed"
    if [ "$2" != "/dev/null" ]; then
        gzip -f "$2"
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

max_conc_for() {
    # max_conc_for <input_len>: the MAX_CONC_MAP cap of <input_len>; empty if none.
    local pair
    for pair in $MAX_CONC_MAP; do
        if [ "${pair%%:*}" = "$1" ]; then
            echo "${pair##*:}"
            return
        fi
    done
}

map_label() {
    # map_label <dynamic_map>: the layer counts of a map, "0-3:0,4-6:4,7-:8" -> L0-4-8.
    local entry
    local label=L
    local IFS=,
    for entry in $1; do
        label="$label${entry##*:}-"
    done
    echo "${label%-}"
}

fixed_label() {
    # fixed_label <layers>: label of a fixed layer count, 16 -> N16, -1 -> Nall.
    if [ "$1" = "-1" ]; then
        echo Nall
    else
        echo "N$1"
    fi
}

bench() {
    # bench <input_len> <hit_rate> <concurrency> <num_prompts> <num_warmups> <seed> <log> [extra args]
    local input_len=$1 hit=$2 concurrency=$3 num_prompts=$4 num_warmups=$5 seed=$6 log=$7
    shift 7
    local prefix_len=$((input_len * hit / 100))
    local random_len=$((input_len - prefix_len))
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
    # append_csv <scheme> <dynamic_map> <fixed_layers> <input_len> <hit_rate> <concurrency> <num_prompts> <repeat> <result_json>
    if [ "$DRY_RUN" = "1" ]; then
        echo "    [dry-run] append $9 to $SUMMARY_CSV"
        return
    fi
    python3 - "$SUMMARY_CSV" "$@" <<'PY'
import csv
import json
import os
import sys

(
    summary, scheme, dynamic_map, fixed_layers, input_len, hit, concurrency,
    num_prompts, repeat, result_json,
) = sys.argv[1:11]
metrics = [
    "completed", "duration", "request_throughput",
    "mean_ttft_ms", "median_ttft_ms", "p95_ttft_ms",
    "mean_tpot_ms", "median_tpot_ms", "p95_tpot_ms",
]
row = {
    "scheme": scheme,
    "dynamic_map": dynamic_map if dynamic_map else "none",
    "fixed_layers": fixed_layers if fixed_layers else "none",
    "input_len": input_len,
    "hit_rate": hit,
    "concurrency": concurrency,
    "num_prompts": num_prompts,
    "repeat": repeat,
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
print(f"    -> {scheme} map={dynamic_map or 'none'} fixed={fixed_layers or 'none'} "
      f"input={input_len} hit={hit} conc={concurrency} repeat={repeat}: "
      f"mean_ttft={row['mean_ttft_ms']} median_ttft={row['median_ttft_ms']} "
      f"p95_ttft={row['p95_ttft_ms']} ms")
PY
}

start_server() {
    # start_server <scheme>: start a server for <scheme> logging to $SERVE_LOG;
    # false if it does not become healthy.
    reclaim_host_memory
    DRY_RUN=$DRY_RUN "$HERE/perf-qwen72b-serve.sh" "$1" "$SERVE_LOG"
    echo "    serve log -> $SERVE_LOG"
    server_alive
}

prewarm() {
    # prewarm <input_len>: warm the current server up at <input_len>, then evict
    # the DDR cache; false if the server died. Concurrency is PREWARM_CONC capped
    # by MAX_CONC_MAP, so the prewarm fits the scratch pool like the matrix does.
    local conc=$PREWARM_CONC
    local max_conc
    max_conc=$(max_conc_for "$1")
    if [ -n "$max_conc" ] && [ "$conc" -gt "$max_conc" ]; then
        conc=$max_conc
    fi
    echo "    prewarm: input=$1 conc=$conc prompts=$PREWARM_PROMPTS hit=100"
    bench "$1" 100 "$conc" "$PREWARM_PROMPTS" 0 "$BASE_SEED" \
        "$OUT/$SERVER_TAG.in$1.prewarm.log"
    if ! server_alive; then
        return 1
    fi
    evict_ddr_cache
}

restart_server() {
    # restart_server <scheme> <label> <input_len>: replace the dead server of
    # <label> with a new <scheme> server prewarmed at <input_len>. False, with no
    # server left running, if <label> already used MAX_RESTARTS or the new server
    # fails to start/prewarm.
    stop_server
    wait_server_gone
    if [ "$RESTARTS" -ge "$MAX_RESTARTS" ]; then
        echo "    $2: all $MAX_RESTARTS restarts used, skipping the rest of $2"
        return 1
    fi
    RESTARTS=$((RESTARTS + 1))
    SERVER_TAG="$2.r$RESTARTS"
    SERVE_LOG="$OUT/$SERVER_TAG.serve.log"
    echo "    restart $RESTARTS/$MAX_RESTARTS of $2"
    if ! start_server "$1"; then
        record_failure "$SERVER_TAG start"
    elif ! prewarm "$3"; then
        record_failure "$SERVER_TAG in$3 prewarm"
    else
        return 0
    fi
    stop_server
    wait_server_gone
    return 1
}

run_scheme() {
    # run_scheme <scheme> <dynamic_map> <fixed_layers>: run every input length of
    # <scheme> on its own server with layer map <dynamic_map> or fixed layer count
    # <fixed_layers> (at most one non-empty; both empty for sync).
    local scheme=$1
    local label=$scheme
    DYNAMIC_MAP=$2
    ASYNC_LOAD_LAYERS=$3
    if [ -n "$DYNAMIC_MAP" ]; then
        label="${scheme}_$(map_label "$DYNAMIC_MAP")"
    elif [ -n "$ASYNC_LOAD_LAYERS" ]; then
        label="${scheme}_$(fixed_label "$ASYNC_LOAD_LAYERS")"
    fi
    if [[ " $SKIP_LABELS " == *" $label "* ]]; then
        echo "===== label=$label skipped (SKIP_LABELS) ====="
        return
    fi
    RESTARTS=0
    SERVER_TAG=$label
    SERVE_LOG="$OUT/$SERVER_TAG.serve.log"
    echo "===== label=$label map=${DYNAMIC_MAP:-none} fixed_layers=${ASYNC_LOAD_LAYERS:-none} connector_md5=$(md5sum "$REPO/kvshrink/kvshrink_connector.py" | cut -c1-8) ====="

    if ! start_server "$scheme"; then
        record_failure "$SERVER_TAG start"
        stop_server
        wait_server_gone
        return
    fi

    local input_len max_conc rep hit concurrency num_prompts tag
    for input_len in $INPUT_LENS; do
        max_conc=$(max_conc_for "$input_len")
        if ! prewarm "$input_len"; then
            record_failure "$SERVER_TAG in$input_len prewarm"
            if ! restart_server "$scheme" "$label" "$input_len"; then
                return
            fi
        fi

        for rep in $(seq 1 "$REPEATS"); do
            for hit in $HIT_RATES; do
                for concurrency in $CONCURRENCIES; do
                    if [ -n "$max_conc" ] && [ "$concurrency" -gt "$max_conc" ]; then
                        continue
                    fi
                    num_prompts=$(num_prompts_for "$concurrency")
                    tag="${label}_in${input_len}_hr${hit}_c${concurrency}"
                    if [ "$REPEATS" -gt 1 ]; then
                        tag="${tag}_r${rep}"
                    fi
                    echo "    --- $tag prompts=$num_prompts ---"
                    queue_trace 1 /dev/null
                    bench "$input_len" "$hit" "$concurrency" "$num_prompts" "$NUM_WARMUPS" \
                        $((BASE_SEED + concurrency + hit)) "$OUT/$tag.log" \
                        --save-result --result-dir "$OUT" --result-filename "$tag.json" $DETAIL_ARG
                    queue_trace 0 "$OUT/$tag.trace.json"
                    if [ "$DRY_RUN" != "1" ]; then
                        grep -a "Successful requests" "$OUT/$tag.log"
                    fi
                    if ! server_alive; then
                        record_failure "$tag"
                        if ! restart_server "$scheme" "$label" "$input_len"; then
                            return
                        fi
                        continue
                    fi
                    append_csv "$scheme" "$DYNAMIC_MAP" "$ASYNC_LOAD_LAYERS" "$input_len" \
                        "$hit" "$concurrency" "$num_prompts" "$rep" "$OUT/$tag.json"
                    evict_ddr_cache
                done
            done
        done
    done

    stop_server
    wait_server_gone
}

echo "branch=$(git -C "$REPO" branch --show-current) commit=$(git -C "$REPO" rev-parse --short HEAD)"
# Same selection setvars.sh will make; fails if any rank has no enabled DSA WQ on
# its GPU's NUMA node.
if ! docker exec -w "$REPO" $C bash -c 'source tools/auto_config.sh && cpu_auto_detect 8 && qat_auto_detect 8 && dsa_auto_detect 8'; then
    echo "ERROR: device auto-detection failed; configure DSA with ./tools/setup_dsa_cnt.sh"
    exit 1
fi

echo "out=$OUT schemes=$SCHEMES maps=$DYNAMIC_MAPS fixed_layers=$FIXED_LAYERS input=$INPUT_LENS" \
     "output=$OUTPUT_LEN hit=$HIT_RATES conc=$CONCURRENCIES max_conc=$MAX_CONC_MAP" \
     "prompts=$PROMPTS_MAP repeats=$REPEATS warmups=$NUM_WARMUPS trace=$TRACE" \
     "ddr_gb=$DDR_POOL_GB_PER_RANK scratch_gb=$SCRATCH_POOL_GB_PER_RANK"
for scheme in $SCHEMES; do
    if [ "$scheme" = "sync" ]; then
        run_scheme sync "" ""
        continue
    fi
    for dynamic_map in $DYNAMIC_MAPS; do
        run_scheme "$scheme" "$dynamic_map" ""
    done
    for layers in $FIXED_LAYERS; do
        run_scheme "$scheme" "" "$layers"
    done
done
if [ -s "$FAILED_CELLS" ]; then
    echo "FAILED (also in $FAILED_CELLS):"
    sed 's/^/    /' "$FAILED_CELLS"
fi
echo ALL_DONE
