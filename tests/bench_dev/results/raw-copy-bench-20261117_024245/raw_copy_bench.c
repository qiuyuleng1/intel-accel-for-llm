// Raw (uncompressed) KV block copy throughput on the host, done the way kvshrink
// does it without compression (iaxl/csrc/kv_zip/kv_zip.cpp, IAXL_KV_COMPRESSION=0):
//   put: per block malloc(header + block) and memcpy from the ScratchPool block
//   get: memcpy from the stored buffer back into a ScratchPool block
// Blocks are 802,816 bytes (784 tokens of one layer on one TP rank of
// Qwen3.8-27B, from the serve logs) and are copied in OpenMP parallel-for
// batches of 37 blocks (one layer of one 32k hit-95 request). Run once per
// OMP_NUM_THREADS; prints GB/s per phase. Put round 1 writes into memory the
// allocator has never handed out (page faults, like a DDR pool that is still
// filling up); later rounds reuse memory freed by the previous round.
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BLOCK_BYTES 802816
#define HEADER_BYTES 64
#define BATCH_BLOCKS 37
#define TOTAL_BLOCKS 4070
#define SCRATCH_BLOCKS 1024
#define PUT_ROUNDS 3
#define GET_ROUNDS 2

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void put_all(char **stored, const char *scratch) {
    for (int first = 0; first < TOTAL_BLOCKS; first += BATCH_BLOCKS) {
        int count = BATCH_BLOCKS;
        if (TOTAL_BLOCKS - first < BATCH_BLOCKS)
            count = TOTAL_BLOCKS - first;
#pragma omp parallel for
        for (int k = 0; k < count; k++) {
            int block = first + k;
            char *buf = malloc(HEADER_BYTES + BLOCK_BYTES);
            if (buf == NULL)
                abort();
            memset(buf, 0, HEADER_BYTES);
            memcpy(buf + HEADER_BYTES, scratch + (size_t)(block % SCRATCH_BLOCKS) * BLOCK_BYTES,
                   BLOCK_BYTES);
            stored[block] = buf;
        }
    }
}

static void get_all(char *const *stored, char *scratch) {
    for (int first = 0; first < TOTAL_BLOCKS; first += BATCH_BLOCKS) {
        int count = BATCH_BLOCKS;
        if (TOTAL_BLOCKS - first < BATCH_BLOCKS)
            count = TOTAL_BLOCKS - first;
#pragma omp parallel for
        for (int k = 0; k < count; k++) {
            int block = first + k;
            memcpy(scratch + (size_t)(block % SCRATCH_BLOCKS) * BLOCK_BYTES,
                   stored[block] + HEADER_BYTES, BLOCK_BYTES);
        }
    }
}

static void free_all(char **stored) {
    for (int block = 0; block < TOTAL_BLOCKS; block++)
        free(stored[block]);
}

int main(void) {
    const size_t scratch_bytes = (size_t)SCRATCH_BLOCKS * BLOCK_BYTES;
    char *scratch = aligned_alloc(4096, scratch_bytes);
    char **stored = malloc(TOTAL_BLOCKS * sizeof(char *));
    if (scratch == NULL || stored == NULL)
        abort();
    memset(scratch, 1, scratch_bytes);
    const int threads = omp_get_max_threads();
    const double gigabytes = (double)TOTAL_BLOCKS * BLOCK_BYTES / 1e9;

    for (int round = 1; round <= PUT_ROUNDS; round++) {
        double start = now_s();
        put_all(stored, scratch);
        double seconds = now_s() - start;
        const char *memory = "reused memory";
        if (round == 1)
            memory = "fresh memory";
        printf("threads %2d  put round %d (%s): %.2f GB in %.3f s = %.1f GB/s\n", threads, round,
               memory, gigabytes, seconds, gigabytes / seconds);
        if (round < PUT_ROUNDS)
            free_all(stored);
    }
    for (int round = 1; round <= GET_ROUNDS; round++) {
        double start = now_s();
        get_all(stored, scratch);
        double seconds = now_s() - start;
        printf("threads %2d  get round %d: %.2f GB in %.3f s = %.1f GB/s\n", threads, round,
               gigabytes, seconds, gigabytes / seconds);
    }
    free_all(stored);
    free(stored);
    free(scratch);
    return 0;
}
