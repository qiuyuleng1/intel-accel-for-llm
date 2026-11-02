## 1. 队列线程上每个任务结束到下一个任务开始的间隔(release + 取下一个任务)

| 主线程状态 | 传法 | 间隔个数 | 间隔中位(µs) | 间隔平均(µs) | 间隔p95(µs) | 间隔最大(µs) | 任务执行中位(µs) |
|---|---|---|---|---|---|---|---|
| wait_in_cpp | py_tensors | 790 | 279.0 | 283.5 | 300.9 | 322.9 | 627.5 |
| wait_in_cpp | py_tensors_kept_future | 790 | 1.5 | 1.7 | 2.6 | 6.4 | 641.2 |
| wait_in_cpp | slots | 790 | 10.2 | 10.3 | 11.8 | 15.1 | 625.6 |
| wait_in_cpp | cpp_views | 790 | 18.1 | 18.2 | 21.0 | 23.9 | 642.5 |
| wait_in_cpp | slots_no_labels | 790 | 1.2 | 1.4 | 2.4 | 6.7 | 639.1 |
| torch_ops_loop | py_tensors | 790 | 3035.8 | 3009.0 | 3341.4 | 3744.1 | 597.3 |
| torch_ops_loop | slots | 790 | 10.9 | 11.0 | 12.5 | 15.8 | 639.9 |
| torch_ops_loop | cpp_views | 790 | 18.1 | 18.3 | 21.3 | 23.5 | 637.9 |
| torch_ops_loop | slots_no_labels | 790 | 1.2 | 1.4 | 2.4 | 7.6 | 632.7 |
| pure_python_loop | py_tensors | 21 | 1815582.9 | 1844257.5 | 2174555.7 | 2189704.0 | 521.7 |
| pure_python_loop | slots | 21 | 11.4 | 11.3 | 12.9 | 13.3 | 311.3 |
| pure_python_loop | cpp_views | 21 | 17.9 | 17.4 | 18.5 | 19.1 | 312.7 |
| pure_python_loop | slots_no_labels | 21 | 1.2 | 1.2 | 1.8 | 2.3 | 306.2 |

列说明:主线程状态 / 传法 见文末;间隔个数 = 统计到的间隔数(每轮最后一个任务后面没有任务,不计);间隔 = 下一个任务开始时刻 − 本任务结束时刻,即释放本任务持有的数据 + 队列取下一个任务;任务执行 = 任务函数本身(把 1000 个 block 各写一遍)的耗时。

## 2. 间隔拆分(按中位数相减)

| 主线程状态 | 现状间隔(µs) | tensor 部分(µs) | tensor 每 block(ns) | label 部分(µs) | 下限(µs) | B 去掉的比例 | B' 去掉的比例 | B 去掉的比例(按均值) |
|---|---|---|---|---|---|---|---|---|
| wait_in_cpp | 279.0 | 268.8 | 269 | 9.0 | 1.2 | 96.3% | 93.5% | 96.4% |
| torch_ops_loop | 3035.8 | 3024.9 | 3025 | 9.7 | 1.2 | 99.6% | 99.4% | 99.6% |
| pure_python_loop | 1815582.9 | 1815571.5 | 1815571 | 10.2 | 1.2 | 100.0% | 100.0% | 100.0% |

列说明:现状间隔 = py_tensors 的间隔中位;tensor 部分 = py_tensors − slots(两者只差传 tensor 还是传编号);tensor 每 block = tensor 部分 ÷ 每任务 block 数;label 部分 = slots − slots_no_labels(只差有没有 label 字符串);下限 = slots_no_labels,即没有 tensor 也没有 label 时剩下的间隔;B / B' 去掉的比例 = (现状 − slots 或 cpp_views) ÷ 现状;最后一列用两者的间隔均值代替中位数计算。

## 3. 主线程上每次调用的耗时(主线程状态 wait_in_cpp 的各轮)

| 传法 | allocate(µs) | submit(µs) | release(µs) | wait(µs) |
|---|---|---|---|---|
| py_tensors | 19.9 / 20.6 | 162.3 / 163.9 | 44.0 / 48.4 | - |
| py_tensors_kept_future | 15.9 / 16.2 | 160.6 / 161.9 | 43.2 / 52.0 | 75.3 / 77.4 |
| slots | 3.9 / 4.0 | 127.5 / 128.1 | 2.7 / 2.7 | - |
| cpp_views | 3.2 / 3.2 | 138.2 / 139.1 | 2.3 / 2.3 | - |
| slots_no_labels | 2.9 / 2.9 | 27.7 / 28.5 | 2.0 / 2.1 | - |

列说明:每格是一个任务(1000 个 block)一次调用耗时的 中位 / 均值;allocate / release = 从 pool 取、还 block(现状用 ScratchPool,返回 Python tensor;方案 B 用整数编号);submit = 调一次 C++ 提交函数,含 pybind 把列表逐个转成 C++ 值;wait = zip_to_mem 形态下主线程在 C++ 里 future.get() 的耗时(此时队列线程已做完,lambda 在这里销毁)。

## 传法 / 主线程状态

- py_tensors: 现状 unzip_from_mem:每个 block 一个 ScratchPool 的 Python tensor + label 字符串

- py_tensors_kept_future: 现状 zip_to_mem:同上,但 lambda 捕获 tensor 且 future 留到 wait

- slots: 方案 B:只传 slot 编号,C++ 存地址;label 字符串照旧

- cpp_views: 方案 B':只传 slot 编号,C++ 用自建的 view(无 Python 对象);label 照旧

- slots_no_labels: 下限:只传 slot 编号,且不传 label 字符串

- wait_in_cpp: 主线程停在 C++ 里等(已放 GIL)

- torch_ops_loop: 主线程循环调用小 torch 算子(每个算子前后放、拿一次 GIL)

- pure_python_loop: 主线程循环跑纯 Python 代码(不主动放 GIL)
