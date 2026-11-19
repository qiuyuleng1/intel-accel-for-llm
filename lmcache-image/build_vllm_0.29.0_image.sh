#!/usr/bin/env bash
# Builds $IMAGE (vLLM 0.29.0 + LMCache 0.5.7 compiled against its torch) from
# Dockerfile.vllm-0.29.0, then checks on one GPU that lmcache.cuda_ops imports.
# Usage: bash build_vllm_0.29.0_image.sh      CHECK_GPU=0 by default.
# For b300 (B300 GPUs, arch 10.3; it reaches the internet without a proxy),
# build and check on b300's docker daemon over ssh:
#   env -u http_proxy -u https_proxy DOCKER_HOST=ssh://b300 \
#       CUDA_ARCH=10.3 bash build_vllm_0.29.0_image.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LMCACHE_REPO=$(cd "$SCRIPT_DIR/../.." && pwd)
BASE_IMAGE=${BASE_IMAGE:-vllm/vllm-openai:v0.29.0}
IMAGE=${IMAGE:-qiuyu/vllm-openai-lmcache:v0.29.0-lmcache0.5.7}
LMCACHE_TAG=v0.5.7
CHECK_GPU=${CHECK_GPU:-0}
CUDA_ARCH=${CUDA_ARCH:-12.0}
CONTEXT_DIR=$SCRIPT_DIR/build-vllm-0.29.0

# The build context holds only the tagged LMCache source and the vLLM plugin,
# not this experiment dir.
mkdir -p "$CONTEXT_DIR"
git -C "$LMCACHE_REPO" archive --format=tar -o "$CONTEXT_DIR/lmcache-src.tar" "$LMCACHE_TAG"
rm -rf "$CONTEXT_DIR/lmcache_hybrid_patch"
cp -r "$SCRIPT_DIR/lmcache_hybrid_patch" "$CONTEXT_DIR/"

# The host reaches PyPI only through its proxy; pass it to the build steps.
docker build -f "$SCRIPT_DIR/Dockerfile.vllm-0.29.0" \
    --build-arg http_proxy --build-arg https_proxy --build-arg no_proxy \
    --build-arg BASE_IMAGE="$BASE_IMAGE" --build-arg CUDA_ARCH="$CUDA_ARCH" \
    -t "$IMAGE" "$CONTEXT_DIR"

docker run --rm --gpus "\"device=$CHECK_GPU\"" --entrypoint python3 "$IMAGE" -c '
from importlib.metadata import entry_points, version
import lmcache.cuda_ops
import lmcache.integration.vllm.lmcache_mp_connector
print("vllm", version("vllm"), "lmcache", version("lmcache"), "torch", version("torch"))
print("import OK: lmcache.cuda_ops, lmcache_mp_connector")
print("vllm.general_plugins:", sorted(e.name for e in entry_points(group="vllm.general_plugins")))
'
echo "[build] done: $IMAGE"
