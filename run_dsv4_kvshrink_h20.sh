#!/bin/bash -e
# ---------------------------------------------------------------------------
# DSv4 (DeepSeek-V4-Flash) KVShrink on the H20 server (8xH20, sm_90).
#
# H20 natively supports FlashMLA / DeepGemm / TileLang, so NO vLLM Triton
# fallback patches are needed (unlike L20 / RTX 6000D). The DSv4 per-layer
# KV-transfer hook is installed automatically at runtime by the connector
# (kvshrink/dsv4_patch.py) -- no source patch step required.
#
# This bring-up DISABLES the Intel accelerator compression/transfer paths
# (QAT / IAA / DSA) and stores the KV cache RAW (no DEFLATE): a plain
# GPU<->CPU-pinned<->DDR tiering path with no hardware dependency.
#
# Workflow:
#   1) On the H20 host:            bash run_dsv4_kvshrink_h20.sh
#        -> builds the iaxl dev image and drops you into the container.
#   2) Inside the container:       bash examples/kvshrink-vllm-serve-dsv4.sh
#   3) In a second host shell:     docker exec -it -w "$PWD" iaxl.vllm bash
#        -> then: bash tests/vllm-test.sh
# ---------------------------------------------------------------------------

# ---- Model / parallelism (H20 server5: /ssd/hf_models/DeepSeek-V4-Flash) ----
export MODEL="${MODEL:-/ssd/hf_models/DeepSeek-V4-Flash}"
export TP_SIZE="${TP_SIZE:-4}"
export PORT="${PORT:-8000}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"

# ---- Disable Intel accelerator compression + DSA transfer -------------------
# Store the KV cache uncompressed (raw). CPU worker threads stay enabled to
# drive the GPU<->CPU<->DDR staging path (they only copy, no compression).
export IAXL_KV_COMPRESSION=0   # no DEFLATE (raw D2H/H2D staging)
export IAXL_QAT_ZIP_ENABLE=0   # no Intel QAT compression workers
export IAXL_IAA_ZIP_ENABLE=0   # no Intel IAA (QPL) compression workers
export IAXL_DSA_GD_ENABLE=0    # no Intel DSA + GDRCopy transfers
export IAXL_CPU_ZIP_ENABLE="${IAXL_CPU_ZIP_ENABLE:-1}"  # keep CPU copy workers

# H20 is sm_90: the base image must contain DeepSeek-V4 support.
export IAXL_BASE_DOCKER_IMAGE="${IAXL_BASE_DOCKER_IMAGE:-vllm/vllm-openai:v0.23.0}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
exec bash start.sh
