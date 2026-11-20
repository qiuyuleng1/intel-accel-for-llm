#!/bin/bash
# Runs "$@" in the container after setvars.sh, like benchmark/kvstore/kvstore_benchmark.sh does.
REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "$REPO_DIR/setvars.sh" || exit 1
export LD_PRELOAD="$IAXL_LD_PRELOAD${LD_PRELOAD:+:$LD_PRELOAD}"
exec "$@"
