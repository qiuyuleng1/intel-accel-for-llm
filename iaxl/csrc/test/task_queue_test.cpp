// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "task_queue.h"

#include <algorithm>
#include <array>
#include <atomic>
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
    CHECK(lo->depth == (std::array<uint32_t, 4>{0, 0, 0, 0}), "low saw an empty queue at submit");
    CHECK(hi->depth == (std::array<uint32_t, 4>{0, 0, 0, 1}),
          "high saw one LOW (P3) task waiting at submit");
    CHECK(hi->submit_priority == hi->priority && hi->group == 0 && hi->p0_enter_ns == 0,
          "plain submit records submit_priority == priority, group 0, p0_enter_ns 0");

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

// Blocks the worker inside one task so that later submits stay queued until release().
class WorkerGate {
  public:
    explicit WorkerGate(TaskQueue &queue) : gate_(release_.get_future()) {
        done_ = queue.submit(
            [this]() {
                started_.set_value();
                gate_.wait();
            },
            TaskQueue::PRIORITY_P0, "gate", kGateGroup);
        started_.get_future().wait();
    }

    void release() {
        release_.set_value();
        done_.get();
    }

    static constexpr uint64_t kGateGroup = 99;

  private:
    std::promise<void> started_;
    std::promise<void> release_;
    std::shared_future<void> gate_;
    std::future<void> done_;
};

static std::vector<std::string> run_order(const std::vector<TaskTraceEvent> &events) {
    std::vector<TaskTraceEvent> sorted = events;
    std::sort(sorted.begin(), sorted.end(),
              [](const TaskTraceEvent &a, const TaskTraceEvent &b) {
                  return a.start_ns < b.start_ns;
              });
    std::vector<std::string> labels;
    for (const auto &e : sorted) {
        if (e.label != "gate")
            labels.push_back(e.label);
    }
    return labels;
}

static void test_four_levels_order() {
    TaskQueue queue("TQ-LEVELS");
    queue.init();
    queue.set_trace_enabled(true);
    WorkerGate gate(queue);
    std::vector<std::future<void>> done;
    done.push_back(queue.submit([]() {}, TaskQueue::PRIORITY_P3, "p3"));
    done.push_back(queue.submit([]() {}, TaskQueue::PRIORITY_P2, "p2"));
    done.push_back(queue.submit([]() {}, TaskQueue::PRIORITY_P1, "p1"));
    done.push_back(queue.submit([]() {}, TaskQueue::PRIORITY_P0, "p0"));
    gate.release();
    for (auto &f : done)
        f.get();
    auto events = flush_and_drain(queue);
    CHECK(run_order(events) == std::vector<std::string>({"p0", "p1", "p2", "p3"}),
          "worker takes P0, then P1, then P2, then P3");
    const auto *p0 = find(events, "p0");
    CHECK(p0 && p0->depth == (std::array<uint32_t, 4>{0, 1, 1, 1}),
          "depth records waiting tasks per level at submit");
}

static void test_group_raise_to_p0() {
    TaskQueue queue("TQ-RAISE");
    queue.init();
    queue.set_trace_enabled(true);
    WorkerGate gate(queue);
    std::vector<std::future<void>> done;
    // A's tail (group 5, P2) is queued before sync batch S (P0) and B's head (P1).
    done.push_back(queue.submit([]() {}, TaskQueue::PRIORITY_P2, "a_tail_0", 5));
    done.push_back(queue.submit([]() {}, TaskQueue::PRIORITY_P2, "a_tail_1", 5));
    done.push_back(queue.submit([]() {}, TaskQueue::PRIORITY_P2, "c_tail", 7));
    done.push_back(queue.submit([]() {}, TaskQueue::PRIORITY_P0, "s"));
    done.push_back(queue.submit([]() {}, TaskQueue::PRIORITY_P1, "b_head", 6));
    const size_t moved = queue.set_group_priority(5, TaskQueue::PRIORITY_P0);
    done.push_back(queue.submit([]() {}, TaskQueue::PRIORITY_P0, "s_late"));
    gate.release();
    for (auto &f : done)
        f.get();
    CHECK(moved == 2, "set_group_priority returns the number of moved tasks");
    auto events = flush_and_drain(queue);
    CHECK(run_order(events) ==
              std::vector<std::string>({"s", "a_tail_0", "a_tail_1", "s_late", "b_head", "c_tail"}),
          "raised tasks join P0 in the order they entered P0, ahead of P1 and P2");
    const auto *a0 = find(events, "a_tail_0");
    const auto *s = find(events, "s");
    const auto *c = find(events, "c_tail");
    CHECK(a0 && a0->priority == TaskQueue::PRIORITY_P0 &&
              a0->submit_priority == TaskQueue::PRIORITY_P2 && a0->group == 5,
          "raised task records execution priority, submit priority and group");
    CHECK(a0 && s && a0->p0_enter_ns > s->enqueue_ns, "raised task records p0_enter_ns");
    CHECK(s && c && s->p0_enter_ns == 0 && c->p0_enter_ns == 0,
          "tasks never moved keep p0_enter_ns 0");
}

static void test_raise_noop_cases() {
    TaskQueue queue("TQ-RAISE-NOOP");
    queue.init();
    WorkerGate gate(queue);
    auto p1 = queue.submit([]() {}, TaskQueue::PRIORITY_P1, "", 0);
    CHECK(queue.set_group_priority(0, TaskQueue::PRIORITY_P0) == 0, "group 0 is never moved");
    CHECK(queue.set_group_priority(12345, TaskQueue::PRIORITY_P0) == 0,
          "unknown group moves nothing");
    CHECK(queue.set_group_priority(WorkerGate::kGateGroup, TaskQueue::PRIORITY_P0) == 0,
          "running task is not affected");
    bool rejected = false;
    try {
        queue.set_group_priority(5, 4);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    CHECK(rejected, "set_group_priority rejects an invalid priority");
    gate.release();
    p1.get();
}

// Cache-miss retry is submitted at PRIORITY_LOW with group 0: it must stay behind the
// zip task it waits for even after the original request's group is raised.
static void test_retry_stays_behind_zip() {
    TaskQueue queue("TQ-RETRY");
    queue.init();
    queue.set_trace_enabled(true);
    WorkerGate gate(queue);
    std::vector<std::future<void>> done;
    done.push_back(queue.submit([]() {}, TaskQueue::PRIORITY_LOW, "zip"));
    done.push_back(queue.submit([]() {}, TaskQueue::PRIORITY_LOW, "unzip_retry"));
    done.push_back(queue.submit([]() {}, TaskQueue::PRIORITY_P2, "tail", 5));
    queue.set_group_priority(5, TaskQueue::PRIORITY_P0);
    gate.release();
    for (auto &f : done)
        f.get();
    CHECK(run_order(flush_and_drain(queue)) ==
              std::vector<std::string>({"tail", "zip", "unzip_retry"}),
          "retry stays behind zip after a group raise");
}

static void test_concurrent_raise_and_submit() {
    TaskQueue queue("TQ-RAISE-MT");
    queue.init();
    constexpr int kThreads = 4;
    constexpr int kPerThread = 500;
    std::atomic<int> ran{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            std::vector<std::future<void>> done;
            for (int i = 0; i < kPerThread; ++i) {
                const uint64_t group = static_cast<uint64_t>(t * kPerThread + i) / 8 + 1;
                done.push_back(queue.submit([&]() { ran.fetch_add(1); },
                                            TaskQueue::PRIORITY_P2, "", group));
                if (i % 8 == 7)
                    queue.set_group_priority(group, TaskQueue::PRIORITY_P0);
            }
            for (auto &f : done)
                f.get();
        });
    }
    for (auto &th : threads)
        th.join();
    CHECK(ran.load() == kThreads * kPerThread,
          "concurrent submit and set_group_priority run every task exactly once");
}

// One forward activates A (group 5) and B (group 6) while this step's sync batch S
// (group 9) is still in P0: all their waiting tasks end up in P0 ordered by layer.
static void test_groups_raise_by_order() {
    TaskQueue queue("TQ-RAISE-ORD");
    queue.init();
    queue.set_trace_enabled(true);
    WorkerGate gate(queue);
    std::vector<std::future<void>> done;
    done.push_back(queue.submit([]() {}, TaskQueue::PRIORITY_P2, "a8", 5, 8));
    done.push_back(queue.submit([]() {}, TaskQueue::PRIORITY_P2, "a9", 5, 9));
    done.push_back(queue.submit([]() {}, TaskQueue::PRIORITY_P2, "b8", 6, 8));
    done.push_back(queue.submit([]() {}, TaskQueue::PRIORITY_P2, "b9", 6, 9));
    done.push_back(queue.submit([]() {}, TaskQueue::PRIORITY_P2, "c8", 7, 8));
    done.push_back(queue.submit([]() {}, TaskQueue::PRIORITY_P0, "other"));
    done.push_back(queue.submit([]() {}, TaskQueue::PRIORITY_P0, "s0", 9, 0));
    done.push_back(queue.submit([]() {}, TaskQueue::PRIORITY_P0, "s8", 9, 8));
    done.push_back(queue.submit([]() {}, TaskQueue::PRIORITY_P0, "s9", 9, 9));
    done.push_back(queue.submit([]() {}, TaskQueue::PRIORITY_P1, "d0", 8, 0));
    const size_t moved = queue.set_groups_priority({9, 5, 6}, TaskQueue::PRIORITY_P0);
    done.push_back(queue.submit([]() {}, TaskQueue::PRIORITY_P0, "late"));
    gate.release();
    for (auto &f : done)
        f.get();
    CHECK(moved == 7, "set_groups_priority returns the number of appended tasks");
    auto events = flush_and_drain(queue);
    CHECK(run_order(events) == std::vector<std::string>({"other", "s0", "a8", "b8", "s8", "a9",
                                                         "b9", "s9", "late", "d0", "c8"}),
          "groups are appended to P0 by (order_key, seq), behind other P0 tasks");
    const auto *a8 = find(events, "a8");
    const auto *s0 = find(events, "s0");
    const auto *other = find(events, "other");
    const auto *c8 = find(events, "c8");
    CHECK(a8 && s0 && a8->p0_enter_ns > 0 && s0->p0_enter_ns == a8->p0_enter_ns,
          "moved and re-appended tasks share one p0_enter_ns");
    CHECK(s0 && s0->priority == TaskQueue::PRIORITY_P0 &&
              s0->submit_priority == TaskQueue::PRIORITY_P0 && s0->order_key == 0,
          "re-appended P0 task keeps its submit priority and records order_key");
    CHECK(other && c8 && other->p0_enter_ns == 0 && c8->p0_enter_ns == 0 &&
              c8->priority == TaskQueue::PRIORITY_P2 && other->order_key == -1,
          "tasks of other groups are not touched; order_key defaults to -1");
}

static void test_groups_raise_noop_cases() {
    TaskQueue queue("TQ-RAISE-ORD-NOOP");
    queue.init();
    queue.set_trace_enabled(true);
    WorkerGate gate(queue);
    std::vector<std::future<void>> done;
    done.push_back(queue.submit([]() {}, TaskQueue::PRIORITY_P3, "zip"));
    done.push_back(queue.submit([]() {}, TaskQueue::PRIORITY_P2, "tail", 5, 3));
    CHECK(queue.set_groups_priority({}, TaskQueue::PRIORITY_P0) == 0, "no groups moves nothing");
    CHECK(queue.set_groups_priority({0, 12345}, TaskQueue::PRIORITY_P0) == 0,
          "group 0 and unknown groups move nothing");
    CHECK(queue.set_groups_priority({WorkerGate::kGateGroup}, TaskQueue::PRIORITY_P0) == 0,
          "running task is not affected");
    bool rejected = false;
    try {
        queue.set_groups_priority({5}, -1);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    CHECK(rejected, "set_groups_priority rejects an invalid priority");
    CHECK(queue.set_groups_priority({5}, TaskQueue::PRIORITY_P1) == 1, "raise to P1 moves the task");
    gate.release();
    for (auto &f : done)
        f.get();
    auto events = flush_and_drain(queue);
    const auto *tail = find(events, "tail");
    CHECK(tail && tail->priority == TaskQueue::PRIORITY_P1 && tail->p0_enter_ns == 0,
          "raise to P1 does not record p0_enter_ns");
    CHECK(run_order(events) == std::vector<std::string>({"tail", "zip"}),
          "group-0 P3 task stays behind");
}

static void test_concurrent_groups_raise_and_submit() {
    TaskQueue queue("TQ-RAISE-ORD-MT");
    queue.init();
    constexpr int kThreads = 4;
    constexpr int kPerThread = 500;
    std::atomic<int> ran{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            std::vector<std::future<void>> done;
            for (int i = 0; i < kPerThread; ++i) {
                const uint64_t group = static_cast<uint64_t>(t * kPerThread + i) / 8 + 1;
                const int priority = i % 2 ? TaskQueue::PRIORITY_P2 : TaskQueue::PRIORITY_P0;
                done.push_back(queue.submit([&]() { ran.fetch_add(1); }, priority, "", group,
                                            kPerThread - i));
                if (i % 8 == 7)
                    queue.set_groups_priority({group, group - 1}, TaskQueue::PRIORITY_P0);
            }
            for (auto &f : done)
                f.get();
        });
    }
    for (auto &th : threads)
        th.join();
    CHECK(ran.load() == kThreads * kPerThread,
          "concurrent submit and set_groups_priority run every task exactly once");
}

int main() {
    test_disabled_by_default();
    test_timestamps_and_order();
    test_throwing_task_is_recorded();
    test_disable_stops_recording();
    test_capacity_drop();
    test_untraced_submit_compatible();
    test_four_levels_order();
    test_group_raise_to_p0();
    test_raise_noop_cases();
    test_retry_stays_behind_zip();
    test_concurrent_raise_and_submit();
    test_groups_raise_by_order();
    test_groups_raise_noop_cases();
    test_concurrent_groups_raise_and_submit();

    if (g_failures) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
