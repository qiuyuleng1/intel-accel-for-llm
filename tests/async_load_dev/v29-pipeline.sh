#!/bin/bash
# Wait for container $C (built by build-v29-container.sh) and its pip install,
# then run queue + correctness tests for all three schemes (dev helper).
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export C=${C:-iaxl.vllm.qiuyu-v29}
for i in $(seq 1 720); do
    docker inspect "$C" > /dev/null 2>&1 && break
    sleep 10
done
docker inspect "$C" > /dev/null 2>&1 || { echo "CONTAINER_MISSING"; exit 1; }
for i in $(seq 1 180); do
    docker exec $C grep -q PIP_EXIT /tmp/pip_install.log 2>/dev/null && break
    sleep 10
done
docker exec $C tail -3 /tmp/pip_install.log
docker exec $C grep -q "PIP_EXIT=0" /tmp/pip_install.log || { echo "PIP_FAILED"; exit 1; }
docker exec $C python3 -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
for scheme in split_async_load_submit naive batch_reqs_async_load_submit; do
    PREFIX=v29 "$REPO/tests/async_load_dev/run-queue-and-correctness.sh" $scheme s1 s2
done
echo ALL_DONE
