// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <pthread.h>

// One executed task, recorded only while tracing is enabled.
// Timestamps are std::chrono::steady_clock nanoseconds (CLOCK_MONOTONIC on Linux,
// the same clock as Python's time.monotonic_ns()).
struct TaskTraceEvent {
    std::string label;           // caller-supplied tag, empty if none
    int priority;                // level the task was taken from (TaskQueue::PRIORITY_P0..P3)
    int submit_priority;         // level passed to submit()
    uint64_t group;              // group passed to submit(); 0 = no group
    uint64_t seq;                // submit order within this queue
    int64_t enqueue_ns;          // submit() pushed the task
    int64_t p0_enter_ns;         // set_group_priority() moved the task into P0; 0 if never
    int64_t start_ns;            // worker began running the task
    int64_t end_ns;              // task returned (or threw)
    int64_t release_ns;          // the task's callable and what it captured were destroyed
    std::array<uint32_t, 4> depth; // tasks already waiting in P0..P3 when this task was submitted
};

class TaskQueue {
  public:
    // Levels, most urgent first. The worker always takes the front task of the most
    // urgent non-empty level; each level is FIFO in the order tasks entered it.
    static constexpr int PRIORITY_P0 = 0;
    static constexpr int PRIORITY_P1 = 1;
    static constexpr int PRIORITY_P2 = 2;
    static constexpr int PRIORITY_P3 = 3;
    static constexpr int NUM_PRIORITIES = 4;
    static constexpr int PRIORITY_HIGH = PRIORITY_P0;
    static constexpr int PRIORITY_LOW = PRIORITY_P3;
    // Events beyond this are dropped (counted by trace_dropped()) to bound memory.
    static constexpr size_t TRACE_CAPACITY = 1u << 18;

    explicit TaskQueue(const char *name) : name_(name) {}

    const char *name() const { return name_; }

    ~TaskQueue() { shutdown(); }

    void init() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ == State::RUNNING)
            return;
        if (state_ != State::CREATED) {
            throw std::runtime_error("TaskQueue cannot be restarted after shutdown");
        }

        worker_ = std::thread([this]() {
            pthread_setname_np(pthread_self(), name_);
            worker_loop();
        });
        state_ = State::RUNNING;
    }

    // Queue func at the tail of level `priority` (PRIORITY_P0..PRIORITY_P3).
    // `group` tags the task for set_group_priority(); 0 means no group.
    // label is only stored when tracing is enabled; callers may pass an empty string
    // (or build it only if trace_enabled()) to avoid the formatting cost.
    // Throws std::invalid_argument if priority is out of range and std::runtime_error
    // if the queue is not running.
    template <typename F>
    std::future<void> submit(F &&func, int priority = PRIORITY_HIGH, std::string label = {},
                             uint64_t group = 0) {
        check_priority(priority);
        auto task = std::make_shared<std::packaged_task<void()>>(std::forward<F>(func));
        auto future = task->get_future();

        Item item;
        item.fn = [task]() { (*task)(); };
        item.priority = priority;
        item.submit_priority = priority;
        item.group = group;
        item.traced = trace_enabled_.load(std::memory_order_relaxed);
        if (item.traced) {
            item.label = std::move(label);
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (state_ != State::RUNNING) {
                throw std::runtime_error("TaskQueue submit rejected: queue is not running");
            }
            // Taken under the lock, like p0_enter_ns, so timestamps follow the
            // order in which tasks entered a level.
            if (item.traced)
                item.enqueue_ns = now_ns();
            item.seq = next_seq_++;
            for (int level = 0; level < NUM_PRIORITIES; ++level) {
                item.depth[level] = static_cast<uint32_t>(tasks_[level].size());
            }
            tasks_[priority].push_back(std::move(item));
        }
        cv_.notify_one();

        return future;
    }

    // Move every waiting task of `group` that sits in a less urgent level than
    // `priority` to the tail of level `priority`, keeping their relative order.
    // Tasks already running or finished are not affected; group 0 is never moved.
    // Tasks moved into P0 record p0_enter_ns. Returns the number of tasks moved.
    // Throws std::invalid_argument if priority is out of range.
    size_t set_group_priority(uint64_t group, int priority) {
        check_priority(priority);
        if (group == 0)
            return 0;
        size_t moved = 0;
        std::lock_guard<std::mutex> lock(mutex_);
        const int64_t moved_ns = now_ns();
        for (int level = priority + 1; level < NUM_PRIORITIES; ++level) {
            std::deque<Item> kept;
            for (Item &item : tasks_[level]) {
                if (item.group != group) {
                    kept.push_back(std::move(item));
                    continue;
                }
                item.priority = priority;
                if (priority == PRIORITY_P0)
                    item.p0_enter_ns = moved_ns;
                tasks_[priority].push_back(std::move(item));
                ++moved;
            }
            tasks_[level].swap(kept);
        }
        return moved;
    }

    // Tracing affects tasks submitted after the call; already-queued tasks keep
    // the setting they were submitted with.
    void set_trace_enabled(bool enabled) {
        trace_enabled_.store(enabled, std::memory_order_relaxed);
    }

    bool trace_enabled() const { return trace_enabled_.load(std::memory_order_relaxed); }

    // Return all recorded events (in completion order) and clear the buffer.
    std::vector<TaskTraceEvent> drain_trace() {
        std::lock_guard<std::mutex> lock(trace_mutex_);
        std::vector<TaskTraceEvent> out;
        out.swap(trace_events_);
        return out;
    }

    // Number of events dropped because the buffer was full since the last reset.
    uint64_t trace_dropped() const { return trace_dropped_.load(std::memory_order_relaxed); }

    void reset_trace_dropped() { trace_dropped_.store(0, std::memory_order_relaxed); }

    void shutdown() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (state_ == State::CREATED) {
                state_ = State::STOPPED;
                return;
            }
            if (state_ == State::STOPPED)
                return;
            if (state_ == State::STOPPING) {
                state_cv_.wait(lock, [this]() { return state_ == State::STOPPED; });
                return;
            }
            state_ = State::STOPPING;
        }
        cv_.notify_one();

        if (worker_.joinable()) {
            worker_.join();
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            state_ = State::STOPPED;
        }
        state_cv_.notify_all();
    }

  private:
    enum class State { CREATED, RUNNING, STOPPING, STOPPED };

    struct Item {
        std::function<void()> fn;
        std::string label;
        int priority = PRIORITY_HIGH;
        int submit_priority = PRIORITY_HIGH;
        uint64_t group = 0;
        bool traced = false;
        uint64_t seq = 0;
        int64_t enqueue_ns = 0;
        int64_t p0_enter_ns = 0;
        std::array<uint32_t, NUM_PRIORITIES> depth{};
    };

    static void check_priority(int priority) {
        if (priority < 0 || priority >= NUM_PRIORITIES) {
            throw std::invalid_argument("TaskQueue priority must be in [PRIORITY_P0, PRIORITY_P3]");
        }
    }

    bool all_empty() const {
        for (const auto &level : tasks_) {
            if (!level.empty())
                return false;
        }
        return true;
    }

    static int64_t now_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }

    void record_trace(Item &item, int64_t start_ns, int64_t end_ns, int64_t release_ns) {
        std::lock_guard<std::mutex> lock(trace_mutex_);
        if (trace_events_.size() >= TRACE_CAPACITY) {
            trace_dropped_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        trace_events_.push_back(TaskTraceEvent{std::move(item.label), item.priority,
                                               item.submit_priority, item.group, item.seq,
                                               item.enqueue_ns, item.p0_enter_ns, start_ns, end_ns,
                                               release_ns,
                                               item.depth});
    }

    void worker_loop() {
        while (true) {
            Item item;

            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this]() { return state_ != State::RUNNING || !all_empty(); });

                if (state_ == State::STOPPING && all_empty()) {
                    return;
                }

                for (auto &level : tasks_) {
                    if (!level.empty()) {
                        item = std::move(level.front());
                        level.pop_front();
                        break;
                    }
                }
            }

            if (!item.traced) {
                item.fn();
                continue;
            }
            const int64_t start_ns = now_ns();
            item.fn();
            const int64_t end_ns = now_ns();
            // Destroy the callable here (otherwise done when `item` goes out of scope)
            // so the trace shows how long releasing what the task captured takes.
            item.fn = nullptr;
            record_trace(item, start_ns, end_ns, now_ns());
        }
    }

    const char *name_;
    std::thread worker_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable state_cv_;
    std::array<std::deque<Item>, NUM_PRIORITIES> tasks_;
    State state_ = State::CREATED;
    uint64_t next_seq_ = 0;

    std::atomic<bool> trace_enabled_{false};
    std::atomic<uint64_t> trace_dropped_{0};
    std::mutex trace_mutex_;
    std::vector<TaskTraceEvent> trace_events_;
};
