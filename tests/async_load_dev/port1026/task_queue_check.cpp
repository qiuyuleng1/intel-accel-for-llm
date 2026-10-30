// Standalone check of the 4-level TaskQueue and set_bundles_priority() in the 1026 PR
// branch (dev helper, not part of the PR).
// Build: g++ -std=c++17 -O1 -pthread -I <repo>/iaxl/csrc/include task_queue_check.cpp
#include "task_queue.h"

#include <cstdio>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool ok, const char *what) {
    std::printf("%s: %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok)
        ++failures;
}

struct Recorder {
    std::mutex mutex;
    std::vector<std::string> order;
    void add(const std::string &name) {
        std::lock_guard<std::mutex> lock(mutex);
        order.push_back(name);
    }
};

static std::string joined(const std::vector<std::string> &names) {
    std::string out;
    for (const std::string &name : names) {
        if (!out.empty())
            out += ",";
        out += name;
    }
    return out;
}

// Blocks the worker so that every later submit waits in its level.
struct Gate {
    std::promise<void> open;
    std::shared_future<void> opened = open.get_future().share();
};

int main() {
    {
        TaskQueue queue("check-levels");
        queue.init();
        Gate gate;
        Recorder rec;
        std::shared_future<void> opened = gate.opened;
        auto gate_done = queue.submit([opened]() { opened.wait(); }, TaskQueue::PRIORITY_P0);
        std::vector<std::future<void>> done;
        done.push_back(queue.submit([&rec]() { rec.add("p3a"); }, TaskQueue::PRIORITY_P3));
        done.push_back(queue.submit([&rec]() { rec.add("p2a"); }, TaskQueue::PRIORITY_P2));
        done.push_back(queue.submit([&rec]() { rec.add("p1a"); }, TaskQueue::PRIORITY_P1));
        done.push_back(queue.submit([&rec]() { rec.add("p0a"); }, TaskQueue::PRIORITY_P0));
        done.push_back(queue.submit([&rec]() { rec.add("p2b"); }, TaskQueue::PRIORITY_P2));
        done.push_back(queue.submit([&rec]() { rec.add("p0b"); }, TaskQueue::PRIORITY_P0));
        gate.open.set_value();
        gate_done.get();
        for (auto &future : done)
            future.get();
        std::string got = joined(rec.order);
        std::printf("  order: %s\n", got.c_str());
        check(got == "p0a,p0b,p1a,p2a,p2b,p3a", "most urgent level first, FIFO within a level");
        check(TaskQueue::PRIORITY_HIGH == TaskQueue::PRIORITY_P0 &&
                  TaskQueue::PRIORITY_LOW == TaskQueue::PRIORITY_P3,
              "HIGH / LOW aliases are P0 / P3");
    }

    for (int sort_after_raise = 0; sort_after_raise <= 1; ++sort_after_raise) {
        // Sync batch S (no bundle, sort keys = layers) at P0 behind a P0 task X without sort key;
        // tails of request A (bundle 1) and B (bundle 2) at P2, C has no bundle. Raise A, then
        // optionally sort P0.
        TaskQueue queue("check-bundles");
        queue.init();
        Gate gate;
        Recorder rec;
        std::shared_future<void> opened = gate.opened;
        auto gate_done = queue.submit([opened]() { opened.wait(); }, TaskQueue::PRIORITY_P0);
        std::vector<std::future<void>> done;
        done.push_back(queue.submit([&rec]() { rec.add("A2"); }, TaskQueue::PRIORITY_P2, 1, 2));
        done.push_back(queue.submit([&rec]() { rec.add("A0"); }, TaskQueue::PRIORITY_P2, 1, 0));
        done.push_back(queue.submit([&rec]() { rec.add("B0"); }, TaskQueue::PRIORITY_P2, 2, 0));
        done.push_back(queue.submit([&rec]() { rec.add("A1"); }, TaskQueue::PRIORITY_P2, 1, 1));
        done.push_back(queue.submit([&rec]() { rec.add("C0"); }, TaskQueue::PRIORITY_P2, 0, 0));
        done.push_back(queue.submit([&rec]() { rec.add("B1"); }, TaskQueue::PRIORITY_P2, 2, 1));
        done.push_back(queue.submit([&rec]() { rec.add("X"); }, TaskQueue::PRIORITY_P0));
        done.push_back(queue.submit([&rec]() { rec.add("S0"); }, TaskQueue::PRIORITY_P0, 0, 0));
        done.push_back(queue.submit([&rec]() { rec.add("S1"); }, TaskQueue::PRIORITY_P0, 0, 1));
        done.push_back(queue.submit([&rec]() { rec.add("S2"); }, TaskQueue::PRIORITY_P0, 0, 2));
        done.push_back(queue.submit([&rec]() { rec.add("P1"); }, TaskQueue::PRIORITY_P1, 0, 0));
        size_t zero_moved = queue.set_bundles_priority({0}, TaskQueue::PRIORITY_P0);
        size_t moved = queue.set_bundles_priority({1}, TaskQueue::PRIORITY_P0);
        if (sort_after_raise)
            queue.sort_priority_level(TaskQueue::PRIORITY_P0);
        gate.open.set_value();
        gate_done.get();
        for (auto &future : done)
            future.get();
        std::string got = joined(rec.order);
        std::printf("  order (sort=%d): %s (moved %zu)\n", sort_after_raise, got.c_str(), moved);
        check(zero_moved == 0, "bundle id 0 is never moved");
        check(moved == 3, "returns the number of tasks moved from less urgent levels");
        if (sort_after_raise) {
            check(got == "X,S0,A0,S1,A1,S2,A2,P1,B0,C0,B1",
                  "sort_priority_level: stable by sort key within P0 only; others keep place");
        } else {
            check(got == "X,S0,S1,S2,A2,A0,A1,P1,B0,C0,B1",
                  "set_bundles_priority alone appends in queue order, no sorting");
        }
    }

    {
        // A bundle task already more urgent than the target level stays where it is.
        TaskQueue queue("check-no-demote");
        queue.init();
        Gate gate;
        Recorder rec;
        std::shared_future<void> opened = gate.opened;
        auto gate_done = queue.submit([opened]() { opened.wait(); }, TaskQueue::PRIORITY_P0);
        std::vector<std::future<void>> done;
        done.push_back(queue.submit([&rec]() { rec.add("g_p0"); }, TaskQueue::PRIORITY_P0, 7, 5));
        done.push_back(queue.submit([&rec]() { rec.add("p1"); }, TaskQueue::PRIORITY_P1));
        done.push_back(queue.submit([&rec]() { rec.add("g_p2"); }, TaskQueue::PRIORITY_P2, 7, 0));
        size_t moved = queue.set_bundles_priority({7}, TaskQueue::PRIORITY_P1);
        gate.open.set_value();
        gate_done.get();
        for (auto &future : done)
            future.get();
        std::string got = joined(rec.order);
        std::printf("  order: %s (moved %zu)\n", got.c_str(), moved);
        check(moved == 1 && got == "g_p0,p1,g_p2", "never moves a task to a less urgent level");
    }

    {
        TaskQueue queue("check-invalid");
        queue.init();
        bool submit_threw = false;
        try {
            queue.submit([]() {}, TaskQueue::NUM_PRIORITIES);
        } catch (const std::invalid_argument &) {
            submit_threw = true;
        }
        bool raise_threw = false;
        try {
            queue.set_bundles_priority({1}, -1);
        } catch (const std::invalid_argument &) {
            raise_threw = true;
        }
        bool sort_threw = false;
        try {
            queue.sort_priority_level(TaskQueue::NUM_PRIORITIES);
        } catch (const std::invalid_argument &) {
            sort_threw = true;
        }
        check(submit_threw && raise_threw && sort_threw,
              "out-of-range priority throws std::invalid_argument");
    }

    std::printf("%s (%d failures)\n", failures == 0 ? "ALL PASS" : "FAILED", failures);
    return failures == 0 ? 0 : 1;
}
