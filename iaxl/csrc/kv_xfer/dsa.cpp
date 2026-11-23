// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "kv_pool.h"
#include "kv_xfer.h"
#include "env.h"
#include "iaxl_common.h"

extern "C" {
#include "dsa_gd.h"
}

extern "C" int64_t dsa_memcpy_batch_async(void *const dest[], const void *const src[],
                                          const size_t n[], size_t count);
extern "C" int dsa_memcpy_batch_wait(uint64_t ticket);

namespace kv_xfer {

using kv_pool::KVPool;
using kv_pool::PoolDevice;

// Serialises batch submission and mapping release; dsa_memcpy_batch_wait() needs no lock.
static std::mutex dsa_context_mutex;

// One GDRCopy mapping per KV pool base: re-pinning on every copy overflowed libgdrapi's handles.
struct DsaRegion {
    void *bar_base = nullptr;
    int refs = 0;
};
static std::mutex dsa_region_mutex;
static std::unordered_map<uintptr_t, DsaRegion> dsa_regions; // by KV pool base

// Contiguous block size from which each direction uses CUDA batch; INT64_MAX = always DSA.
static std::atomic<int64_t> dsa_h2d_crossover{INT64_MAX};
static std::atomic<int64_t> dsa_d2h_crossover{INT64_MAX};

// 128 MiB per copy, 2 warmup + 10 timed copies, keep the fastest: most stable in the 6000d test.
static constexpr int64_t kCalibrateBytes = 128LL << 20;
static constexpr int64_t kCalibrateFirstBlockSize = 256LL << 10;
static constexpr int64_t kCalibrateBlockSizeStep = 128LL << 10;
static constexpr int64_t kCalibrateLastBlockSize = 4LL << 20;
static constexpr int kCalibrateWarmupCopies = 2;
static constexpr int kCalibrateTimedCopies = 10;

// kv_pool hooks: called with the registry lock held, so never call back into kv_pool here.
static void dsa_on_register(const KVPool &p) {
    if (!envs.IAXL_DSA_GD_ENABLE || p.device != PoolDevice::GPU)
        return;
    std::lock_guard<std::mutex> lock(dsa_region_mutex);
    auto [it, inserted] = dsa_regions.try_emplace(p.base);
    it->second.refs++;
    if (!inserted)
        return;
    void *bar_base = nullptr;
    const bool mapped = dsa_gd_default_register(p.base, p.total_bytes) == 0 &&
                        dsa_gd_default_gpu_bar_addr(p.base, p.total_bytes, &bar_base) == 0;
    if (!mapped)
        fprintf(stderr,
                "dsa: GDRCopy cannot map KV pool %s (%zu bytes); the GPU BAR1 may be smaller "
                "than the KV cache. Set IAXL_DSA_GD_ENABLE=0 to copy with CUDA instead.\n",
                p.name.c_str(), p.total_bytes);
    IAXL_CHECK(mapped, "dsa: KV pool registration failed");
    it->second.bar_base = bar_base;
}

static void dsa_on_unregister(const KVPool &p) {
    if (!envs.IAXL_DSA_GD_ENABLE || p.device != PoolDevice::GPU)
        return;
    std::lock_guard<std::mutex> lock(dsa_region_mutex);
    auto it = dsa_regions.find(p.base);
    IAXL_CHECK(it != dsa_regions.end(), "dsa: GPU KV pool was not registered");
    if (--it->second.refs > 0)
        return;
    // Wait for in-flight DSA batches before the mapping goes away.
    std::lock_guard<std::mutex> submit_lock(dsa_context_mutex);
    dsa_memcpy_batch_wait(UINT64_MAX);
    IAXL_CHECK(dsa_gd_default_unregister(p.base, p.total_bytes) == 0,
               "dsa: KV pool mapping not found");
    dsa_regions.erase(it);
}

void dsa_install_pool_hooks() { kv_pool::add_pool_hooks({dsa_on_register, dsa_on_unregister}); }

void *dsa_gpu_bar_base(char *gpu_base) {
    std::lock_guard<std::mutex> lock(dsa_region_mutex);
    auto it = dsa_regions.find(reinterpret_cast<uintptr_t>(gpu_base));
    if (it == dsa_regions.end())
        return nullptr;
    return it->second.bar_base;
}

bool dsa_copy_chunks_batch(char *gpu_bar_base, int64_t chunk_stride, int64_t outer_dims,
                           int64_t inner_size, int64_t outer_block_size, bool is_h2d,
                           const std::vector<int64_t> &chunk_indices,
                           const std::vector<char *> &cpu_ptrs, uint64_t &ticket) {

    if (gpu_bar_base == nullptr) // IAXL_DSA_GD_ENABLE=0: the KV pool has no BAR mapping
        return false;
    const int64_t crossover = is_h2d ? dsa_h2d_crossover.load() : dsa_d2h_crossover.load();
    if (inner_size >= crossover) // CUDA batch was measured faster at this block size
        return false;

    std::lock_guard<std::mutex> lock(dsa_context_mutex);

    if ((inner_size % 8) != 0)
        return false;

    const size_t num_chunks = chunk_indices.size();
    if (num_chunks != cpu_ptrs.size())
        return false;
    const size_t count = num_chunks * static_cast<size_t>(outer_dims);
    if (count == 0)
        return true;

    std::vector<void *> dest(count), src(count);
    std::vector<size_t> nbytes(count);

    size_t k = 0;
    for (size_t i = 0; i < num_chunks; ++i) {
        const int64_t gpu_chunk_off = chunk_indices[i] * chunk_stride;
        char *cpu_chunk = cpu_ptrs[i];
        for (int64_t o = 0; o < outer_dims; ++o) {
            char *gpu_bar = gpu_bar_base + gpu_chunk_off + o * outer_block_size;
            char *cpu_ptr = cpu_chunk + o * inner_size;
            if (is_h2d) {
                dest[k] = gpu_bar;
                src[k] = cpu_ptr;
            } else {
                dest[k] = cpu_ptr;
                src[k] = gpu_bar;
            }
            nbytes[k] = static_cast<size_t>(inner_size);
            ++k;
        }
    }

    const int64_t t = dsa_memcpy_batch_async(
        dest.data(), const_cast<const void *const *>(src.data()), nbytes.data(), count);
    if (t < 0)
        return false;
    // Tickets are monotonic, so remembering the latest one is enough to cover every batch
    // this context has issued.
    ticket = static_cast<uint64_t>(t);
    return true;
}

void dsa_wait_ticket(uint64_t ticket) {
    if (!envs.IAXL_DSA_GD_ENABLE || ticket == 0)
        return;
    IAXL_CHECK(dsa_memcpy_batch_wait(ticket) == 0, "dsa_wait_ticket: DSA batch failed");
}

bool dsa_auto_fallback_enabled() {
    return envs.IAXL_DSA_AUTO_FALLBACK_ENABLE && envs.IAXL_DSA_GD_ENABLE;
}

void dsa_set_crossover(int64_t h2d_bytes, int64_t d2h_bytes) {
    dsa_h2d_crossover.store(h2d_bytes);
    dsa_d2h_crossover.store(d2h_bytes);
}

// CUDA reaches the GPU buffer at gpu_buffer, DSA reaches the same memory at gpu_bar_buffer.
struct CalibrationResources {
    int device_index;
    char *gpu_buffer;
    char *gpu_bar_buffer;
    char *host_buffer;
    cudaStream_t stream;
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
};

static double calibrate_cuda_batch_ms(const CalibrationResources &resources, int64_t block_size,
                                      bool h2d, const std::vector<int64_t> &chunk_indices,
                                      const std::vector<char *> &cpu_ptrs) {
    double best_ms = std::numeric_limits<double>::max();
    for (int copy = 0; copy < kCalibrateWarmupCopies + kCalibrateTimedCopies; ++copy) {
        IAXL_CHECK(cudaEventRecord(resources.start_event, resources.stream) == cudaSuccess,
                   "dsa calibrate: cudaEventRecord failed");
        cuda_copy_chunks_batch(resources.gpu_buffer, block_size, 1, block_size, block_size, h2d,
                               chunk_indices, cpu_ptrs, resources.stream);
        IAXL_CHECK(cudaEventRecord(resources.stop_event, resources.stream) == cudaSuccess,
                   "dsa calibrate: cudaEventRecord failed");
        IAXL_CHECK(cudaEventSynchronize(resources.stop_event) == cudaSuccess,
                   "dsa calibrate: cudaEventSynchronize failed");
        float elapsed_ms = 0.0f;
        IAXL_CHECK(cudaEventElapsedTime(&elapsed_ms, resources.start_event,
                                        resources.stop_event) == cudaSuccess,
                   "dsa calibrate: cudaEventElapsedTime failed");
        if (copy >= kCalibrateWarmupCopies && elapsed_ms < best_ms)
            best_ms = elapsed_ms;
    }
    return best_ms;
}

static double calibrate_dsa_ms(const CalibrationResources &resources, int64_t block_size,
                               bool h2d, const std::vector<int64_t> &chunk_indices,
                               const std::vector<char *> &cpu_ptrs) {
    double best_ms = std::numeric_limits<double>::max();
    for (int copy = 0; copy < kCalibrateWarmupCopies + kCalibrateTimedCopies; ++copy) {
        const auto start = std::chrono::steady_clock::now();
        uint64_t ticket = 0;
        IAXL_CHECK(dsa_copy_chunks_batch(resources.gpu_bar_buffer, block_size, 1, block_size,
                                         block_size, h2d, chunk_indices, cpu_ptrs, ticket),
                   "dsa calibrate: DSA batch submit failed");
        dsa_wait_ticket(ticket);
        const std::chrono::duration<double, std::milli> elapsed =
            std::chrono::steady_clock::now() - start;
        if (copy >= kCalibrateWarmupCopies && elapsed.count() < best_ms)
            best_ms = elapsed.count();
    }
    return best_ms;
}

// Smallest scanned size where CUDA batch beats DSA; INT64_MAX when DSA wins from 256 KB to 4 MB.
static int64_t calibrate_direction(const CalibrationResources &resources, bool h2d) {
    const char *direction = h2d ? "H2D" : "D2H";
    int measured_sizes = 0;
    for (int64_t block_size = kCalibrateFirstBlockSize; block_size <= kCalibrateLastBlockSize;
         block_size += kCalibrateBlockSizeStep) {
        const int64_t block_count = kCalibrateBytes / block_size;
        std::vector<int64_t> chunk_indices(block_count);
        std::vector<char *> cpu_ptrs(block_count);
        for (int64_t block_index = 0; block_index < block_count; ++block_index) {
            chunk_indices[block_index] = block_index;
            cpu_ptrs[block_index] = resources.host_buffer + block_index * block_size;
        }
        // Scattered GPU blocks like real KV; fixed seed keeps the order the same every run.
        std::mt19937 generator(0);
        std::shuffle(chunk_indices.begin(), chunk_indices.end(), generator);
        const double cuda_batch_ms =
            calibrate_cuda_batch_ms(resources, block_size, h2d, chunk_indices, cpu_ptrs);
        const double dsa_ms = calibrate_dsa_ms(resources, block_size, h2d, chunk_indices, cpu_ptrs);
        measured_sizes++;
        if (cuda_batch_ms < dsa_ms) {
            if (envs.IAXL_DEBUG_LOG) {
                const double copied_bytes = static_cast<double>(block_count * block_size);
                fprintf(stderr,
                        "[kv_xfer] dsa calibrate GPU%d %s: crossover %lld KB (DSA %.1f GB/s, "
                        "CUDA batch %.1f GB/s, %d block sizes measured)\n",
                        resources.device_index, direction, static_cast<long long>(block_size >> 10),
                        copied_bytes / dsa_ms / 1e6, copied_bytes / cuda_batch_ms / 1e6,
                        measured_sizes);
            }
            return block_size;
        }
    }
    if (envs.IAXL_DEBUG_LOG)
        fprintf(stderr,
                "[kv_xfer] dsa calibrate GPU%d %s: no crossover up to %lld KB (%d block sizes "
                "measured), all copies use DSA\n",
                resources.device_index, direction,
                static_cast<long long>(kCalibrateLastBlockSize >> 10), measured_sizes);
    return INT64_MAX;
}

std::pair<int64_t, int64_t> dsa_calibrate_crossover(int device_index) {
    CalibrationResources resources{};
    resources.device_index = device_index;
    IAXL_CHECK(cudaSetDevice(device_index) == cudaSuccess, "dsa calibrate: cudaSetDevice failed");
    IAXL_CHECK(cudaMalloc(reinterpret_cast<void **>(&resources.gpu_buffer), kCalibrateBytes) ==
                   cudaSuccess,
               "dsa calibrate: cudaMalloc of 128 MiB failed");
    IAXL_CHECK(cudaHostAlloc(reinterpret_cast<void **>(&resources.host_buffer), kCalibrateBytes,
                             cudaHostAllocDefault) == cudaSuccess,
               "dsa calibrate: cudaHostAlloc of 128 MiB failed");
    const uint64_t gpu_address = reinterpret_cast<uint64_t>(resources.gpu_buffer);
    void *bar_buffer = nullptr;
    const bool mapped =
        dsa_gd_default_register(gpu_address, kCalibrateBytes) == 0 &&
        dsa_gd_default_gpu_bar_addr(gpu_address, kCalibrateBytes, &bar_buffer) == 0;
    IAXL_CHECK(mapped, "dsa calibrate: GDRCopy cannot map 128 MiB, set "
                       "IAXL_DSA_AUTO_FALLBACK_ENABLE=0 to skip calibration");
    resources.gpu_bar_buffer = static_cast<char *>(bar_buffer);
    IAXL_CHECK(cudaStreamCreateWithFlags(&resources.stream, cudaStreamNonBlocking) == cudaSuccess,
               "dsa calibrate: cudaStreamCreate failed");
    IAXL_CHECK(cudaEventCreate(&resources.start_event) == cudaSuccess,
               "dsa calibrate: cudaEventCreate failed");
    IAXL_CHECK(cudaEventCreate(&resources.stop_event) == cudaSuccess,
               "dsa calibrate: cudaEventCreate failed");

    const int64_t h2d_crossover = calibrate_direction(resources, true);
    const int64_t d2h_crossover = calibrate_direction(resources, false);

    cudaEventDestroy(resources.stop_event);
    cudaEventDestroy(resources.start_event);
    cudaStreamDestroy(resources.stream);
    IAXL_CHECK(dsa_gd_default_unregister(gpu_address, kCalibrateBytes) == 0,
               "dsa calibrate: mapping not found");
    cudaFreeHost(resources.host_buffer);
    cudaFree(resources.gpu_buffer);
    return {h2d_crossover, d2h_crossover};
}

} // namespace kv_xfer
