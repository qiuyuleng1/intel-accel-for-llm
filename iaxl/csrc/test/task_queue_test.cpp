// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "task_queue.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, msg)                                                                           \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "[FAIL] %s (%s:%d)\n", (msg), __FILE__, __LINE__);                \
            ++g_failures;                                                                          \
        } else {                                                                                   \
            std::printf("[ok]   %s\n", (msg));                                                     \
        }                                                                                          \
    } while (0)

static int64_t steady_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

static const TaskTraceEvent *find(const std::vector<TaskTraceEvent> &events,
                                  const std::string &label) {
    for (const auto &e : events)
        if (e.label == label)
            return &e;
    return nullptr;
}

// future.get() returns when the task body finishes, slightly before the worker stores
// its trace. The worker is serial, so once an untraced no-op submitted afterwards has
// run, every earlier trace is stored.
static std::vector<TaskTraceEvent> flush_and_drain(TaskQueue &queue) {
    const bool enabled = queue.trace_enabled();
    queue.set_trace_enabled(false);
    queue.submit([]() {}).get();
    queue.set_trace_enabled(enabled);
    return queue.drain_trace();
}

static void test_disabled_by_default() {
    TaskQueue queue("TQ-TRACE-OFF");
    queue.init();
    CHECK(!queue.trace_enabled(), "trace is disabled by default");
    queue.submit([]() {}, TaskQueue::PRIORITY_HIGH, "x").get();
    CHECK(flush_and_drain(queue).empty(), "no events recorded while trace is disabled");
}

static void test_timestamps_and_order() {
    TaskQueue queue("TQ-TRACE");
    queue.init();
    queue.set_trace_enabled(true);

    // Hold the worker inside "gate" so that "low" and "high" must wait in the queue.
    std::promise<void> gate_started;
    std::promise<void> release_gate;
    std::shared_future<void> gate(release_gate.get_future());
    auto gate_done = queue.submit(
        [&]() {
            gate_started.set_value();
            gate.wait();
        },
        TaskQueue::PRIORITY_HIGH, "gate");
    gate_started.get_future().wait();

    auto low_done = queue.submit([]() {}, TaskQueue::PRIORITY_LOW, "low");
    auto high_done = queue.submit([]() {}, TaskQueue::PRIORITY_HIGH, "high");

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const int64_t release_ns = steady_ns();
    release_gate.set_value();
    gate_done.get();
    high_done.get();
    low_done.get();

    auto events = flush_and_drain(queue);
    CHECK(events.size() == 3, "one event per executed task");
    const auto *g = find(events, "gate");
    const auto *lo = find(events, "low");
    const auto *hi = find(events, "high");
    CHECK(g && lo && hi, "labels are preserved");
    if (!(g && lo && hi))
        return;

    bool monotonic = true;
    for (const auto &e : events)
        monotonic = monotonic && e.enqueue_ns <= e.start_ns && e.start_ns <= e.end_ns;
    CHECK(monotonic, "enqueue_ns <= start_ns <= end_ns for every event");

    CHECK(g->priority == TaskQueue::PRIORITY_HIGH && lo->priority == TaskQueue::PRIORITY_LOW &&
              hi->priority == TaskQueue::PRIORITY_HIGH,
          "priority is recorded");
    CHECK(g->seq < lo->seq && lo->seq < hi->seq, "seq follows submit order");
    CHECK(g->end_ns <= hi->start_ns && hi->end_ns <= lo->start_ns,
          "timestamps show gate -> high -> low execution order");
    CHECK(hi->start_ns >= release_ns && lo->start_ns >= release_ns,
          "queued tasks start only after the blocking task is released");
    CHECK(hi->start_ns - hi->enqueue_ns >= 20 * 1000 * 1000,
          "queue wait (start_ns - enqueue_ns) covers the time spent behind the blocking task");
    CHECK(lo->high_depth == 0 && lo->low_depth == 0, "low saw an empty queue at submit");
    CHECK(hi->high_depth == 0 && hi->low_depth == 1, "high saw one LOW task waiting at submit");

    CHECK(queue.drain_trace().empty(), "drain_trace clears the buffer");
}

static void test_throwing_task_is_recorded() {
    TaskQueue queue("TQ-TRACE-THROW");
    queue.init();
    queue.set_trace_enabled(true);
    auto failed =
        queue.submit([]() { throw std::runtime_error("expected"); }, TaskQueue::PRIORITY_HIGH,
                     "throw");
    bool propagated = false;
    try {
        failed.get();
    } catch (const std::runtime_error &) {
        propagated = true;
    }
    CHECK(propagated, "exception still propagates through the future");
    auto events = flush_and_drain(queue);
    CHECK(events.size() == 1 && events[0].label == "throw", "throwing task is still recorded");
}

static void test_disable_stops_recording() {
    TaskQueue queue("TQ-TRACE-TOGGLE");
    queue.init();
    queue.set_trace_enabled(true);
    queue.submit([]() {}, TaskQueue::PRIORITY_HIGH, "on").get();
    queue.set_trace_enabled(false);
    queue.submit([]() {}, TaskQueue::PRIORITY_HIGH, "off").get();
    auto events = flush_and_drain(queue);
    CHECK(events.size() == 1 && events[0].label == "on",
          "tasks submitted after disabling are not recorded");
}

static void test_capacity_drop() {
    TaskQueue queue("TQ-TRACE-CAP");
    queue.init();
    queue.set_trace_enabled(true);
    const size_t extra = 5;
    std::future<void> last;
    for (size_t i = 0; i < TaskQueue::TRACE_CAPACITY + extra; i++)
        last = queue.submit([]() {});
    last.get();
    auto events = flush_and_drain(queue);
    CHECK(queue.trace_dropped() == extra, "events beyond TRACE_CAPACITY are counted as dropped");
    CHECK(events.size() == TaskQueue::TRACE_CAPACITY, "buffer keeps exactly TRACE_CAPACITY events");
    queue.reset_trace_dropped();
    CHECK(queue.trace_dropped() == 0, "reset_trace_dropped clears the counter");
}

static void test_untraced_submit_compatible() {
    TaskQueue queue("TQ-COMPAT");
    queue.init();
    std::vector<int> order;
    std::promise<void> release;
    std::shared_future<void> gate(release.get_future());
    auto first = queue.submit([&]() { gate.wait(); });
    auto low = queue.submit([&]() { order.push_back(1); }, TaskQueue::PRIORITY_LOW);
    auto high = queue.submit([&]() { order.push_back(0); });
    release.set_value();
    first.get();
    high.get();
    low.get();
    CHECK(order == std::vector<int>({0, 1}), "label-less submit keeps HIGH before LOW");

    bool rejected = false;
    try {
        queue.submit([]() {}, 7);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    CHECK(rejected, "invalid priority is rejected");
}

int main() {
    test_disabled_by_default();
    test_timestamps_and_order();
    test_throwing_task_is_recorded();
    test_disable_stops_recording();
    test_capacity_drop();
    test_untraced_submit_compatible();

    if (g_failures) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
