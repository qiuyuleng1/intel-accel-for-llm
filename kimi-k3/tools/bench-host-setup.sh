#!/bin/bash
# Host-side helpers shared by the HOST benchmark scripts (test-qwen38-27B.sh,
# tests/bench_dev/scratch-check-qwen38.sh). Source it; it defines functions only.
#   check_host_settings      fail unless the host matches PERFORMANCE_MODE_SETUP.md
#   reclaim_host_memory      drop the page cache and compact memory before a serve
#   ensure_kvshrink_container start the KVShrink container if needed and wait for it
# ensure_kvshrink_container uses the caller's REPO_ROOT, MODEL,
# KVSHRINK_CONTAINER, KVSHRINK_TMUX_SESSION and KVSHRINK_START_ENV (variable
# assignments put in front of ./start.sh, e.g. a TP_SIZE and GPU/QAT/DSA
# binding that setvars.sh cannot detect; unset = none).

PERFSPECT_DIR=${PERFSPECT_DIR:-/home/intel/qiuyu/perfspect}

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
# exits, so it runs in a detached tmux session that outlives the caller.
ensure_kvshrink_container() {
    local started=0
    local main_pid
    if ! docker container inspect "$KVSHRINK_CONTAINER" > /dev/null 2>&1; then
        echo "$KVSHRINK_CONTAINER does not exist; running ./start.sh in tmux session $KVSHRINK_TMUX_SESSION"
        if ! tmux new-session -d -s "$KVSHRINK_TMUX_SESSION" -c "$REPO_ROOT" "${KVSHRINK_START_ENV:-} MODEL=$MODEL ./start.sh"; then
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
