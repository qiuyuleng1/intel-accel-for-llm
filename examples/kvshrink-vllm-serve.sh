#!/bin/bash -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../setvars.sh"
: "${MODEL:?Set MODEL in setvars.sh or the environment}"

export LD_PRELOAD="$IAXL_LD_PRELOAD${LD_PRELOAD:+:$LD_PRELOAD}"

parallel_args=(-tp "$TP_SIZE")
if [[ "${DP_SIZE:-1}" -gt 1 ]]; then
    parallel_args+=(--data-parallel-size "$DP_SIZE")
fi

# vLLM CPU inference sizes its KV cache from VLLM_CPU_KVCACHE_SPACE (GiB) instead.
device_args=(--gpu-memory-utilization 0.8)
[[ "$DEVICE" == cpu ]] && device_args=()
# Unset BLOCK_SIZE keeps vLLM's default: 128 on CPU (16 for MLA models), 16 or the attention backend's preference on GPU.
[[ -n "$BLOCK_SIZE" ]] && device_args+=(--block-size "$BLOCK_SIZE")
# Unset CUDAGRAPH_MODE keeps vLLM's choice: FULL_AND_PIECEWISE, or PIECEWISE when the KV connector requires it.
[[ -n "$CUDAGRAPH_MODE" ]] && device_args+=(--compilation-config "{\"cudagraph_mode\": \"$CUDAGRAPH_MODE\"}")
# TORCH_PROFILER_DIR enables vLLM's torch profiler: POST /start_profile records
# PROFILER_MAX_ITERATIONS worker steps, then it stops and writes its trace and
# profiler_out_<rank>.txt (CUDA time per kernel) into that directory.
[[ -n "$TORCH_PROFILER_DIR" ]] && device_args+=(--profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"$TORCH_PROFILER_DIR\", \"torch_profiler_with_stack\": false, \"max_iterations\": ${PROFILER_MAX_ITERATIONS:-20}}")

# Benchmark baselines: BASELINE=no_cache is plain vLLM without KV reuse,
# BASELINE=prefix_cache is plain vLLM with its GPU prefix cache.
# GPU_PREFIX_CACHING=1 keeps vLLM's GPU prefix cache on next to KVShrink.
kvshrink_prefix_caching_arg=--no-enable-prefix-caching
[[ "$GPU_PREFIX_CACHING" == 1 ]] && kvshrink_prefix_caching_arg=--enable-prefix-caching
cache_args=(--kv-transfer-config '{"kv_connector":"KVShrinkConnector","kv_connector_module_path":"kvshrink.kvshrink_connector","kv_role":"kv_both"}' "$kvshrink_prefix_caching_arg")
[[ "$BASELINE" == no_cache ]] && cache_args=(--no-enable-prefix-caching)
[[ "$BASELINE" == prefix_cache ]] && cache_args=(--enable-prefix-caching)

vllm serve "$MODEL" \
    "${cache_args[@]}" \
    --trust-remote-code \
    "${parallel_args[@]}" \
    "${device_args[@]}" \
    --max-model-len "${MAX_MODEL_LEN:-32765}" \
    --port "${PORT:-8000}" \
    2>&1 | tee log.kvshrink-vllm
