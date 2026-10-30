#!/bin/bash
# usage: pipeline-1026.sh   (dev helper, run on the host in tmux)
# Waits for the 1026 dev image, starts the container (tmux session iaxl1026-ctr), waits
# for `pip install -e .` in it, then runs: rpc_check.py (remote payload plumbing, no
# daemon), correctness-1026.sh, gsm8k-1026.sh. The standalone TaskQueue check
# (task_queue_check.cpp) is built and run on the host separately.
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=/home/pese/qiuyu/intel-accel-for-llm-1026
C=iaxl.vllm.qiuyu-1026
IMAGE=$(cd "$REPO" && source setvars.sh > /dev/null && echo "$IAXL_DEV_DOCKER_IMAGE")

echo "$(date +%T) waiting for image $IMAGE"
until docker image inspect "$IMAGE" > /dev/null 2>&1; do
    if ! tmux has-session -t iaxl1026-build 2> /dev/null; then
        echo "image build session is gone and $IMAGE does not exist"
        exit 1
    fi
    sleep 30
done

echo "$(date +%T) starting container $C"
tmux new -d -s iaxl1026-ctr "$HERE/start-1026-container.sh; exec bash"
until docker exec $C true 2> /dev/null; do
    sleep 10
done
echo "$(date +%T) waiting for pip install -e . in $C"
# By process name: with --pid host, `pgrep -f` also matches the host's `docker run`,
# whose command line contains the pip command.
until docker exec $C bash -c "! pgrep -x pip > /dev/null && cd $REPO && python3 -c 'import iaxl.torch_ext; from iaxl.torch_ext import task_queue_new_bundle_id'" > /dev/null 2>&1; do
    if ! docker exec $C true 2> /dev/null; then
        echo "container $C exited (see tmux session iaxl1026-ctr)"
        exit 1
    fi
    sleep 20
done
echo "$(date +%T) extension built"

docker exec -w "$REPO" $C python3 "$HERE/rpc_check.py" > "$HERE/results-rpc_check.log" 2>&1
echo "rpc_check exit=$? ($(tail -1 "$HERE/results-rpc_check.log"))"

"$HERE/correctness-1026.sh"
"$HERE/gsm8k-1026.sh"
echo "$(date +%T) PIPELINE_DONE"
