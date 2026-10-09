// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

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
    std::string label;      // caller-supplied tag, empty if none
    int priority;           // TaskQueue::PRIORITY_*
    uint64_t seq;           // submit order within this queue
    int64_t enqueue_ns;     // submit() pushed the task
    int64_t start_ns;       // worker began running the task
    int64_t end_ns;         // task returned (or threw)
    uint32_t high_depth;    // HIGH tasks already waiting when this task was submitted
    uint32_t low_depth;     // LOW tasks already waiting when this task was submitted
};

class TaskQueue {
  public:
    static constexpr int PRIORITY_HIGH = 0;
    static constexpr int PRIORITY_LOW = 1;
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

    // label is only stored when tracing is enabled; callers may pass an empty string
    // (or build it only if trace_enabled()) to avoid the formatting cost.
    template <typename F>
    std::future<void> submit(F &&func, int priority = PRIORITY_HIGH, std::string label = {}) {
        if (priority != PRIORITY_HIGH && priority != PRIORITY_LOW) {
            throw std::invalid_argument("TaskQueue priority must be PRIORITY_HIGH or PRIORITY_LOW");
        }
        auto task = std::make_shared<std::packaged_task<void()>>(std::forward<F>(func));
        auto future = task->get_future();

        Item item;
        item.fn = [task]() { (*task)(); };
        item.priority = priority;
        item.traced = trace_enabled_.load(std::memory_order_relaxed);
        if (item.traced) {
            item.label = std::move(label);
            item.enqueue_ns = now_ns();
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (state_ != State::RUNNING) {
                throw std::runtime_error("TaskQueue submit rejected: queue is not running");
            }
            item.seq = next_seq_++;
            item.high_depth = static_cast<uint32_t>(high_priority_tasks_.size());
            item.low_depth = static_cast<uint32_t>(low_priority_tasks_.size());
            if (priority == PRIORITY_HIGH) {
                high_priority_tasks_.push_back(std::move(item));
            } else {
                low_priority_tasks_.push_back(std::move(item));
            }
        }
        cv_.notify_one();

        return future;
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
        bool traced = false;
        uint64_t seq = 0;
        int64_t enqueue_ns = 0;
        uint32_t high_depth = 0;
        uint32_t low_depth = 0;
    };

    static int64_t now_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }

    void record_trace(Item &item, int64_t start_ns, int64_t end_ns) {
        std::lock_guard<std::mutex> lock(trace_mutex_);
        if (trace_events_.size() >= TRACE_CAPACITY) {
            trace_dropped_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        trace_events_.push_back(TaskTraceEvent{std::move(item.label), item.priority, item.seq,
                                               item.enqueue_ns, start_ns, end_ns, item.high_depth,
                                               item.low_depth});
    }

    void worker_loop() {
        while (true) {
            Item item;

            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this]() {
                    return state_ != State::RUNNING || !high_priority_tasks_.empty() ||
                           !low_priority_tasks_.empty();
                });

                if (state_ == State::STOPPING && high_priority_tasks_.empty() &&
                    low_priority_tasks_.empty()) {
                    return;
                }

                if (!high_priority_tasks_.empty()) {
                    item = std::move(high_priority_tasks_.front());
                    high_priority_tasks_.pop_front();
                } else {
                    item = std::move(low_priority_tasks_.front());
                    low_priority_tasks_.pop_front();
                }
            }

            if (!item.traced) {
                item.fn();
                continue;
            }
            const int64_t start_ns = now_ns();
            item.fn();
            record_trace(item, start_ns, now_ns());
        }
    }

    const char *name_;
    std::thread worker_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable state_cv_;
    std::deque<Item> high_priority_tasks_;
    std::deque<Item> low_priority_tasks_;
    State state_ = State::CREATED;
    uint64_t next_seq_ = 0;

    std::atomic<bool> trace_enabled_{false};
    std::atomic<uint64_t> trace_dropped_{0};
    std::mutex trace_mutex_;
    std::vector<TaskTraceEvent> trace_events_;
};
