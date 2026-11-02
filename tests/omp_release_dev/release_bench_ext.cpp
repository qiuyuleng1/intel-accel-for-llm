// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Hands KV blocks to IAXL's TaskQueue the way Context::unzip_from_mem and Context::zip_to_mem do,
// so that releasing what a task holds can be timed without a GPU or vLLM. Every task copies one
// source block into each of its blocks, standing in for decompression, and records when its body
// started and ended; the gap to the next task's start is the release plus taking the next task.

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <cstring>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "kv_pool.h"
#include "task_queue.h"

namespace py = pybind11;

namespace {

int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// What UnzipFromMemWork holds today (Python tensors), or under scheme B' (tensors made in C++).
struct TensorBlocksWork {
    std::vector<std::string> chunk_labels;
    std::vector<int64_t> chunk_indices;
    std::vector<torch::Tensor> blocks;
    std::shared_ptr<std::promise<void>> promise;
};

// What the decompression task would hold under scheme B: one address per block.
struct AddressBlocksWork {
    std::vector<std::string> chunk_labels;
    std::vector<int64_t> chunk_indices;
    std::vector<char *> blocks;
    std::shared_ptr<std::promise<void>> promise;
};

struct TaskTimes {
    int64_t start_ns = 0;
    int64_t end_ns = 0;
};

} // namespace

class ReleaseBench {
  public:
    ReleaseBench(const torch::Tensor &pool, int64_t max_tasks) {
        TORCH_CHECK(pool.device().is_cpu() && pool.is_contiguous() && pool.dim() >= 1 &&
                        pool.size(0) > 0,
                    "pool must be a non-empty contiguous CPU tensor");
        TORCH_CHECK(max_tasks > 0, "max_tasks must be positive");
        pool_ = pool;
        num_blocks_ = pool.size(0);
        block_bytes_ = static_cast<int64_t>(pool.nbytes()) / num_blocks_;
        pool_base_ = static_cast<char *>(pool.data_ptr());
        source_block_.assign(block_bytes_, 0x5A);
        // Views made here have no Python object, so copying and dropping them never takes the GIL.
        cpp_views_ = pool.unbind(0);
        task_times_.resize(max_tasks);
        kept_futures_.resize(max_tasks);
        queue_.init();
    }

    int64_t block_bytes() const { return block_bytes_; }

    // Blocks the queue thread until resume_queue(), so that all tasks of a round wait behind each
    // other and every gap between two tasks is one release plus taking the next task.
    void hold_queue() {
        resume_ = std::promise<void>();
        std::shared_future<void> resumed = resume_.get_future().share();
        queue_.submit([resumed]() { resumed.wait(); });
    }

    void resume_queue() { resume_.set_value(); }

    // Context::unzip_from_mem today: one Python tensor per block, dropped when the task is
    // destroyed on the queue thread.
    void submit_py_tensors(int64_t task_id, const std::string &label, const std::string &tensor_key,
                           const std::vector<std::string> &chunk_labels,
                           const std::vector<int64_t> &chunk_indices,
                           const std::vector<torch::Tensor> &blocks) {
        check_task_id(task_id);
        auto work = std::make_shared<TensorBlocksWork>();
        work->chunk_labels = kv_pool::make_chunk_labels(label, tensor_key, chunk_labels);
        work->chunk_indices = chunk_indices;
        work->blocks = blocks;
        submit_tensor_blocks(task_id, std::move(work));
    }

    // Scheme B': only slot numbers come from Python; the task holds views made in C++.
    void submit_cpp_views(int64_t task_id, const std::string &label, const std::string &tensor_key,
                          const std::vector<std::string> &chunk_labels,
                          const std::vector<int64_t> &chunk_indices,
                          const std::vector<int64_t> &slots) {
        check_task_id(task_id);
        auto work = std::make_shared<TensorBlocksWork>();
        work->chunk_labels = kv_pool::make_chunk_labels(label, tensor_key, chunk_labels);
        work->chunk_indices = chunk_indices;
        work->blocks.reserve(slots.size());
        for (int64_t slot : slots) {
            check_slot(slot);
            work->blocks.push_back(cpp_views_[slot]);
        }
        submit_tensor_blocks(task_id, std::move(work));
    }

    // Scheme B: only slot numbers come from Python; the task holds addresses in the pool.
    void submit_slots(int64_t task_id, const std::string &label, const std::string &tensor_key,
                      const std::vector<std::string> &chunk_labels,
                      const std::vector<int64_t> &chunk_indices, const std::vector<int64_t> &slots) {
        check_task_id(task_id);
        auto work = std::make_shared<AddressBlocksWork>();
        work->chunk_labels = kv_pool::make_chunk_labels(label, tensor_key, chunk_labels);
        work->chunk_indices = chunk_indices;
        work->blocks.reserve(slots.size());
        for (int64_t slot : slots) {
            check_slot(slot);
            work->blocks.push_back(pool_base_ + slot * block_bytes_);
        }
        work->promise = make_promise();
        auto *self = this;
        queue_.submit([work, self, task_id]() {
            const int64_t start_ns = now_ns();
            for (char *block : work->blocks)
                self->fill_block(block);
            work->promise->set_value();
            self->task_times_[task_id] = {start_ns, now_ns()};
        });
    }

    // Context::zip_to_mem today: the lambda holds the tensors by value and its future is kept
    // until wait_kept_future(), so whichever of the queue thread and the waiting thread lets go of
    // the task last drops the tensors.
    void submit_py_tensors_kept_future(int64_t task_id, const std::string &label,
                                       const std::string &tensor_key,
                                       const std::vector<std::string> &chunk_labels,
                                       const std::vector<torch::Tensor> &blocks) {
        check_task_id(task_id);
        auto full_chunk_labels = kv_pool::make_chunk_labels(label, tensor_key, chunk_labels);
        auto blocks_copy = blocks;
        auto *self = this;
        kept_futures_[task_id] = queue_.submit([=]() {
            const int64_t start_ns = now_ns();
            // zip_to_mem stores one cache entry per label, so the lambda keeps the labels too.
            IAXL_CHECK(full_chunk_labels.size() == blocks_copy.size(), "one label per block");
            for (const torch::Tensor &block : blocks_copy)
                self->fill_block(static_cast<char *>(block.data_ptr()));
            self->task_times_[task_id] = {start_ns, now_ns()};
        });
    }

    // Times future.get() the way Context::zip_wait() runs it (GIL released): when the queue thread
    // has already destroyed its side of the task, get() drops the lambda on this thread.
    int64_t wait_kept_future(int64_t task_id) {
        check_task_id(task_id);
        TORCH_CHECK(kept_futures_[task_id].valid(), "no kept future for task ", task_id);
        const int64_t start_ns = now_ns();
        kept_futures_[task_id].get();
        return now_ns() - start_ns;
    }

    // An empty last task: once it has run, every earlier task has been run and destroyed.
    void submit_fence() { fence_ = queue_.submit([]() {}); }

    bool fence_done() const {
        return fence_.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    }

    void wait_fence() {
        TORCH_CHECK(fence_.valid(), "submit_fence() must be called first");
        fence_.get();
        unzip_futures_.clear();
    }

    std::vector<int64_t> start_times(int64_t first_task_id, int64_t count) const {
        check_task_range(first_task_id, count);
        std::vector<int64_t> times;
        for (int64_t task_id = first_task_id; task_id < first_task_id + count; task_id++)
            times.push_back(task_times_[task_id].start_ns);
        return times;
    }

    std::vector<int64_t> end_times(int64_t first_task_id, int64_t count) const {
        check_task_range(first_task_id, count);
        std::vector<int64_t> times;
        for (int64_t task_id = first_task_id; task_id < first_task_id + count; task_id++)
            times.push_back(task_times_[task_id].end_ns);
        return times;
    }

    void shutdown() { queue_.shutdown(); }

  private:
    void submit_tensor_blocks(int64_t task_id, std::shared_ptr<TensorBlocksWork> work) {
        work->promise = make_promise();
        auto *self = this;
        queue_.submit([work, self, task_id]() {
            const int64_t start_ns = now_ns();
            for (const torch::Tensor &block : work->blocks) {
                // kv_zip_decompress_batch also reads each tensor's byte size before writing to it.
                IAXL_CHECK(block.nbytes() == static_cast<size_t>(self->block_bytes_),
                           "block byte size mismatch");
                self->fill_block(static_cast<char *>(block.data_ptr()));
            }
            work->promise->set_value();
            self->task_times_[task_id] = {start_ns, now_ns()};
        });
    }

    std::shared_ptr<std::promise<void>> make_promise() {
        auto promise = std::make_shared<std::promise<void>>();
        // Context keeps this future until unzip_wait(), so the queue thread never frees its state.
        unzip_futures_.push_back(promise->get_future());
        return promise;
    }

    void fill_block(char *block) const { std::memcpy(block, source_block_.data(), block_bytes_); }

    void check_task_id(int64_t task_id) const {
        TORCH_CHECK(0 <= task_id && task_id < static_cast<int64_t>(task_times_.size()),
                    "task_id out of range: ", task_id);
    }

    void check_task_range(int64_t first_task_id, int64_t count) const {
        TORCH_CHECK(count > 0, "count must be positive");
        check_task_id(first_task_id);
        check_task_id(first_task_id + count - 1);
    }

    void check_slot(int64_t slot) const {
        TORCH_CHECK(0 <= slot && slot < num_blocks_, "slot out of range: ", slot);
    }

    torch::Tensor pool_;
    int64_t num_blocks_ = 0;
    int64_t block_bytes_ = 0;
    char *pool_base_ = nullptr;
    std::vector<char> source_block_;
    std::vector<torch::Tensor> cpp_views_;
    std::vector<TaskTimes> task_times_;
    std::vector<std::future<void>> kept_futures_;
    std::vector<std::future<void>> unzip_futures_;
    std::promise<void> resume_;
    std::future<void> fence_;
    // Declared last so that it is destroyed first: its thread must stop before the data it uses.
    TaskQueue queue_{"BENCH-Main"};
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    auto release_gil = py::call_guard<py::gil_scoped_release>();
    py::class_<ReleaseBench>(m, "ReleaseBench")
        .def(py::init<const torch::Tensor &, int64_t>(), py::arg("pool"), py::arg("max_tasks"))
        .def_property_readonly("block_bytes", &ReleaseBench::block_bytes)
        .def("hold_queue", &ReleaseBench::hold_queue, release_gil)
        .def("resume_queue", &ReleaseBench::resume_queue, release_gil)
        .def("submit_py_tensors", &ReleaseBench::submit_py_tensors, py::arg("task_id"),
             py::arg("label"), py::arg("tensor_key"), py::arg("chunk_labels"),
             py::arg("chunk_indices"), py::arg("blocks"), release_gil)
        .def("submit_cpp_views", &ReleaseBench::submit_cpp_views, py::arg("task_id"),
             py::arg("label"), py::arg("tensor_key"), py::arg("chunk_labels"),
             py::arg("chunk_indices"), py::arg("slots"), release_gil)
        .def("submit_slots", &ReleaseBench::submit_slots, py::arg("task_id"), py::arg("label"),
             py::arg("tensor_key"), py::arg("chunk_labels"), py::arg("chunk_indices"),
             py::arg("slots"), release_gil)
        .def("submit_py_tensors_kept_future", &ReleaseBench::submit_py_tensors_kept_future,
             py::arg("task_id"), py::arg("label"), py::arg("tensor_key"),
             py::arg("chunk_labels"), py::arg("blocks"), release_gil)
        .def("wait_kept_future", &ReleaseBench::wait_kept_future, py::arg("task_id"), release_gil)
        .def("submit_fence", &ReleaseBench::submit_fence, release_gil)
        // Keeps the GIL: a polling loop must not hand it to the queue thread through this call.
        .def("fence_done", &ReleaseBench::fence_done)
        .def("wait_fence", &ReleaseBench::wait_fence, release_gil)
        .def("start_times", &ReleaseBench::start_times, py::arg("first_task_id"), py::arg("count"))
        .def("end_times", &ReleaseBench::end_times, py::arg("first_task_id"), py::arg("count"))
        .def("shutdown", &ReleaseBench::shutdown, release_gil);
}
