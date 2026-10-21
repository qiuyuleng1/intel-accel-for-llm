// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Micro-benchmark of TaskQueue::set_group_priority (profiling helper, not a test).
//
// For each (P1, P2 groups x tasks per group, P3) shape: block the worker with a
// gate task, fill the waiting levels, time one set_group_priority(group, P0) for a
// group in the middle of P2, release the gate and let the worker drain. Repeated
// REPS times; prints median / p99 / max of the call and ns per waiting task.
//
// Build and run (inside the dev container, pinned to one core):
//   g++ -O2 -std=c++17 -Iinclude -o ../../_data/task_queue_bench \
//       test/task_queue_bench.cpp -lpthread
//   taskset -c 2 ../../_data/task_queue_bench

#include "task_queue.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <future>
#include <memory>
#include <vector>

namespace {

constexpr int REPS = 200;

struct Shape {
    size_t p1_tasks;
    size_t p2_groups;
    size_t p2_tasks_per_group;
    size_t p3_tasks;
};

// Captures a shared_ptr like the real unzip lambda, so std::function holds a
// non-trivial callable.
void submit_noop(TaskQueue &queue, int priority, uint64_t group) {
    auto token = std::make_shared<int>(0);
    queue.submit([token]() {}, priority, std::string(), group);
}

double run_shape(const Shape &shape, std::vector<double> &call_us) {
    TaskQueue queue("bench");
    queue.init();
    size_t moved_total = 0;
    for (int rep = 0; rep < REPS; ++rep) {
        std::promise<void> started;
        std::promise<void> release;
        std::shared_future<void> gate(release.get_future());
        auto gate_done = queue.submit(
            [&started, gate]() {
                started.set_value();
                gate.wait();
            },
            TaskQueue::PRIORITY_P0);
        started.get_future().wait();

        // Group ids: P1 heads use groups 1..p1_tasks/8, P2 tails use 100000+g, P3 is 0.
        for (size_t i = 0; i < shape.p1_tasks; ++i)
            submit_noop(queue, TaskQueue::PRIORITY_P1, 1 + i / 8);
        for (size_t g = 0; g < shape.p2_groups; ++g)
            for (size_t t = 0; t < shape.p2_tasks_per_group; ++t)
                submit_noop(queue, TaskQueue::PRIORITY_P2, 100000 + g);
        for (size_t i = 0; i < shape.p3_tasks; ++i)
            submit_noop(queue, TaskQueue::PRIORITY_P3, 0);

        const uint64_t target = 100000 + shape.p2_groups / 2;
        auto t0 = std::chrono::steady_clock::now();
        size_t moved = queue.set_group_priority(target, TaskQueue::PRIORITY_P0);
        auto t1 = std::chrono::steady_clock::now();
        call_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        moved_total += moved;

        release.set_value();
        gate_done.get();
        // Drain: a P3 task runs last, so its completion means the queue is empty.
        queue.submit([]() {}, TaskQueue::PRIORITY_P3).get();
    }
    return static_cast<double>(moved_total) / REPS;
}

} // namespace

int main() {
    const std::vector<Shape> shapes = {
        {0, 1, 72, 0},        {64, 8, 72, 0},       {256, 32, 72, 0},
        {256, 64, 72, 0},     {256, 32, 72, 1280},  {256, 32, 72, 6400},
        {256, 128, 72, 6400},
    };
    std::printf("%8s %10s %8s %10s %8s %10s %10s %10s %12s\n", "P1", "P2", "P3", "waiting",
                "moved", "median_us", "p99_us", "max_us", "ns_per_task");
    for (const Shape &shape : shapes) {
        std::vector<double> call_us;
        double moved = run_shape(shape, call_us);
        std::sort(call_us.begin(), call_us.end());
        const size_t p2 = shape.p2_groups * shape.p2_tasks_per_group;
        const size_t waiting = shape.p1_tasks + p2 + shape.p3_tasks;
        const double median = call_us[call_us.size() / 2];
        const double p99 = call_us[call_us.size() * 99 / 100];
        std::printf("%8zu %10zu %8zu %10zu %8.0f %10.1f %10.1f %10.1f %12.1f\n", shape.p1_tasks,
                    p2, shape.p3_tasks, waiting, moved, median, p99, call_us.back(),
                    median * 1000.0 / static_cast<double>(waiting));
    }
    return 0;
}
