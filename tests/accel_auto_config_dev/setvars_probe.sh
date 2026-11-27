#!/bin/bash
# Sources setvars.sh like a script in the container would, then prints what it exported (PROBE lines).
REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
started_ns=$(date +%s%N)
source "$REPO_DIR/setvars.sh"
status=$?
echo "PROBE setvars_status=$status"
echo "PROBE setvars_ms=$((($(date +%s%N) - started_ns) / 1000000))"
for name in IAXL_DSA_H2D_CROSSOVER_KB IAXL_DSA_D2H_CROSSOVER_KB IAXL_ZIP_SEGMENT_SIZE_KB VLLM_CPU_OMP_THREADS_BIND KVSHRINK_DSA_DEVICES; do
    echo "PROBE $name=$(printenv "$name" || echo '<unset>')"
done
docker_arg=no
for arg in "${DOCKER_RUN_ARGS[@]}"; do
    [[ "$arg" == IAXL_IN_CONTAINER=1 ]] && docker_arg=yes
done
echo "PROBE docker_run_args_has_in_container=$docker_arg"
