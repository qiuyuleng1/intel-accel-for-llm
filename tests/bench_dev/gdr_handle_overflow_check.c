// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Reproduces the kvshrink crash seen on B300 ("dsa batch failed ... status=0x84" after tens of
// thousands of "gdr_map failed rc=16/22").
//
// Suspected cause: dsa_gd_default_register() pins [base, base + span) again every time the
// copied span grows and keeps every old pin. gdrdrv gives each pin a handle that only grows
// (by the pinned size in 4 KiB pages), but libgdrapi 2.5 stores the handle in 32 bits. After
// 2^32 pages (16 TiB) of pins in one process the library sends truncated handles; gdrdrv
// resolves a handle it does not know to its oldest pin, so gdr_map() fails and the
// gdr_unpin_buffer() that dsa_gd does on that failure removes the oldest pin, which is still
// mapped and in use.
//
// The check pins a small "victim" buffer first (so it is the oldest pin), then grows the
// pinned span of a large buffer through the same dsa_gd_default_register() call kvshrink
// uses, and after every call reads the victim through its BAR mapping. If the cause is right,
// registering starts failing once the cumulative pinned size passes 16 TiB, and at that same
// call the victim read gets SIGBUS and a DSA write to the victim fails.

#include <setjmp.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cuda_runtime.h>

#include "dsa_gd.h"

int dsa_memcpy(void *dest, const void *src, size_t n);

#define VICTIM_BYTES (2UL << 20)
#define HOST_PAGE_BYTES 4096UL
#define HANDLE_LIMIT_PAGES (1ULL << 32)
#define VICTIM_PATTERN 0x1122334455667788ULL
#define DSA_PATTERN 0x5a5a5a5a5a5a5a5aULL

enum victim_state { VICTIM_OK = 0, VICTIM_WRONG_DATA = 1, VICTIM_SIGBUS = 2 };

static sigjmp_buf g_sigbus_jump;

static void on_sigbus(int sig) {
    (void)sig;
    siglongjmp(g_sigbus_jump, 1);
}

static double now_sec(void) {
    struct timespec t;

    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static const char *victim_state_name(enum victim_state state) {
    if (state == VICTIM_OK)
        return "ok";
    if (state == VICTIM_WRONG_DATA)
        return "WRONG_DATA";
    return "SIGBUS";
}

// A CPU read through the BAR mapping gets SIGBUS once gdrdrv has torn the mapping down.
static enum victim_state read_victim(volatile const uint64_t *victim_bar) {
    if (sigsetjmp(g_sigbus_jump, 1))
        return VICTIM_SIGBUS;
    if (victim_bar[0] != VICTIM_PATTERN)
        return VICTIM_WRONG_DATA;
    return VICTIM_OK;
}

// Writes one host page into the victim through DSA (the kvshrink load direction) and checks
// the result with cudaMemcpy, which does not go through the BAR mapping.
static void dsa_write_victim(const char *when, char *victim_bar, char *victim_gpu,
                             uint64_t *host_page) {
    uint64_t readback = 0;
    int rc;

    rc = dsa_memcpy(victim_bar + HOST_PAGE_BYTES, host_page, HOST_PAGE_BYTES);
    cudaMemcpy(&readback, victim_gpu + HOST_PAGE_BYTES, sizeof(readback),
               cudaMemcpyDeviceToHost);
    printf("dsa write to victim %s: dsa_memcpy rc=%d, value read back by cudaMemcpy %s\n", when,
           rc, readback == DSA_PATTERN ? "matches" : "DOES NOT match");
}

int main(int argc, char **argv) {
    int device = 1;
    size_t churn_gib = 128;
    size_t step_mib = 128;
    size_t churn_bytes, step_bytes, span, word;
    char *victim_gpu = NULL;
    char *churn_gpu = NULL;
    void *victim_bar = NULL;
    uint64_t *host_page;
    uint64_t pattern = VICTIM_PATTERN;
    uint64_t pinned_pages;
    unsigned long call_index = 0;
    struct sigaction action;
    double start;

    // Unbuffered so our lines stay in order with the stderr lines dsa_gd prints.
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc > 1)
        device = atoi(argv[1]);
    if (argc > 2)
        churn_gib = strtoull(argv[2], NULL, 0);
    if (argc > 3)
        step_mib = strtoull(argv[3], NULL, 0);
    churn_bytes = churn_gib << 30;
    step_bytes = step_mib << 20;
    printf("device=%d churn buffer=%zu GiB span step=%zu MiB handle limit=2^32 pages (16 TiB)\n",
           device, churn_gib, step_mib);

    if (cudaSetDevice(device) != cudaSuccess ||
        cudaMalloc((void **)&victim_gpu, VICTIM_BYTES) != cudaSuccess ||
        cudaMalloc((void **)&churn_gpu, churn_bytes) != cudaSuccess) {
        printf("SETUP FAILED: cuda allocation\n");
        return 2;
    }
    cudaMemset(victim_gpu, 0, VICTIM_BYTES);
    cudaMemcpy(victim_gpu, &pattern, sizeof(pattern), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    host_page = aligned_alloc(HOST_PAGE_BYTES, HOST_PAGE_BYTES);
    for (word = 0; word < HOST_PAGE_BYTES / sizeof(uint64_t); word++)
        host_page[word] = DSA_PATTERN;

    memset(&action, 0, sizeof(action));
    action.sa_handler = on_sigbus;
    sigaction(SIGBUS, &action, NULL);

    if (dsa_gd_default_register((uint64_t)(uintptr_t)victim_gpu, VICTIM_BYTES) != 0 ||
        dsa_gd_default_gpu_bar_addr((uint64_t)(uintptr_t)victim_gpu, VICTIM_BYTES,
                                    &victim_bar) != 0) {
        printf("SETUP FAILED: victim register\n");
        return 2;
    }
    printf("victim read before churn: %s\n",
           victim_state_name(read_victim((volatile const uint64_t *)victim_bar)));
    dsa_write_victim("before churn", (char *)victim_bar, victim_gpu, host_page);
    pinned_pages = VICTIM_BYTES / HOST_PAGE_BYTES;

    printf("%8s %10s %14s %12s %6s %8s %8s\n", "call", "span_GiB", "cum_pinned_TiB",
           "cum/2^32pg", "rc", "victim", "time_s");
    start = now_sec();
    span = 0;
    while (span + step_bytes <= churn_bytes) {
        enum victim_state state;
        int rc;

        span += step_bytes;
        call_index++;
        rc = dsa_gd_default_register((uint64_t)(uintptr_t)churn_gpu, span);
        // Every call pins a new, larger span (the pin itself succeeds even when the map
        // after it fails), so the handle space used grows by the full span each time.
        pinned_pages += span / HOST_PAGE_BYTES;
        state = read_victim((volatile const uint64_t *)victim_bar);

        if (rc != 0 || state != VICTIM_OK || call_index % 64 == 0) {
            printf("%8lu %10.2f %14.3f %12.4f %6d %8s %8.1f\n", call_index,
                   (double)span / (1UL << 30),
                   (double)pinned_pages * HOST_PAGE_BYTES / (1ULL << 40),
                   (double)pinned_pages / HANDLE_LIMIT_PAGES, rc, victim_state_name(state),
                   now_sec() - start);
            fflush(stdout);
        }
        if (rc != 0 || state != VICTIM_OK) {
            dsa_write_victim("after first failure", (char *)victim_bar, victim_gpu, host_page);
            printf("RESULT: first failure at call %lu, cumulative pinned %.3f TiB\n", call_index,
                   (double)pinned_pages * HOST_PAGE_BYTES / (1ULL << 40));
            return 0;
        }
    }
    printf("RESULT: no failure up to cumulative pinned %.3f TiB\n",
           (double)pinned_pages * HOST_PAGE_BYTES / (1ULL << 40));
    return 1;
}
