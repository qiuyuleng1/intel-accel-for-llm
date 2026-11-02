## 1. 队列线程上每个任务结束到下一个任务开始的间隔(release + 取下一个任务)

| 主线程状态 | 传法 | 间隔个数 | 间隔中位(µs) | 间隔平均(µs) | 间隔p95(µs) | 间隔最大(µs) | 任务执行中位(µs) |
|---|---|---|---|---|---|---|---|
| wait_in_cpp | py_tensors | 790 | 147.0 | 147.3 | 153.9 | 160.5 | 203.6 |
| wait_in_cpp | py_tensors_kept_future | 790 | 1.2 | 1.3 | 2.0 | 6.9 | 205.2 |
| wait_in_cpp | slots | 790 | 5.6 | 5.8 | 7.1 | 40.2 | 195.7 |
| wait_in_cpp | cpp_views | 790 | 9.2 | 9.3 | 10.8 | 14.7 | 206.2 |
| wait_in_cpp | slots_no_labels | 790 | 0.9 | 1.1 | 2.3 | 4.0 | 205.3 |
| torch_ops_loop | py_tensors | 790 | 1673.7 | 1675.9 | 1944.0 | 2237.4 | 188.6 |
| torch_ops_loop | slots | 790 | 5.7 | 5.8 | 7.1 | 10.7 | 199.0 |
| torch_ops_loop | cpp_views | 790 | 9.3 | 9.4 | 10.8 | 13.8 | 201.4 |
| torch_ops_loop | slots_no_labels | 790 | 0.8 | 1.1 | 2.2 | 5.3 | 194.8 |
| pure_python_loop | py_tensors | 21 | 511035.4 | 512712.7 | 561573.5 | 571695.3 | 168.1 |
| pure_python_loop | slots | 21 | 6.0 | 5.9 | 7.1 | 7.2 | 153.4 |
| pure_python_loop | cpp_views | 21 | 9.4 | 9.4 | 10.2 | 10.8 | 154.7 |
| pure_python_loop | slots_no_labels | 21 | 1.0 | 1.1 | 2.1 | 2.2 | 152.6 |

列说明:主线程状态 / 传法 见文末;间隔个数 = 统计到的间隔数(每轮最后一个任务后面没有任务,不计);间隔 = 下一个任务开始时刻 − 本任务结束时刻,即释放本任务持有的数据 + 队列取下一个任务;任务执行 = 任务函数本身(把 500 个 block 各写一遍)的耗时。

## 2. 间隔拆分(按中位数相减)

| 主线程状态 | 现状间隔(µs) | tensor 部分(µs) | tensor 每 block(ns) | label 部分(µs) | 下限(µs) | B 去掉的比例 | B' 去掉的比例 | B 去掉的比例(按均值) |
|---|---|---|---|---|---|---|---|---|
| wait_in_cpp | 147.0 | 141.4 | 283 | 4.7 | 0.9 | 96.2% | 93.7% | 96.1% |
| torch_ops_loop | 1673.7 | 1668.0 | 3336 | 4.9 | 0.8 | 99.7% | 99.4% | 99.7% |
| pure_python_loop | 511035.4 | 511029.4 | 1022059 | 5.0 | 1.0 | 100.0% | 100.0% | 100.0% |

列说明:现状间隔 = py_tensors 的间隔中位;tensor 部分 = py_tensors − slots(两者只差传 tensor 还是传编号);tensor 每 block = tensor 部分 ÷ 每任务 block 数;label 部分 = slots − slots_no_labels(只差有没有 label 字符串);下限 = slots_no_labels,即没有 tensor 也没有 label 时剩下的间隔;B / B' 去掉的比例 = (现状 − slots 或 cpp_views) ÷ 现状;最后一列用两者的间隔均值代替中位数计算。

## 3. 主线程上每次调用的耗时(主线程状态 wait_in_cpp 的各轮)

| 传法 | allocate(µs) | submit(µs) | release(µs) | wait(µs) |
|---|---|---|---|---|
| py_tensors | 10.1 / 10.4 | 82.8 / 83.3 | 20.8 / 21.3 | - |
| py_tensors_kept_future | 8.1 / 8.2 | 80.8 / 81.5 | 21.1 / 29.7 | 38.8 / 38.4 |
| slots | 1.9 / 1.9 | 64.0 / 64.3 | 1.1 / 1.1 | - |
| cpp_views | 1.7 / 1.7 | 69.8 / 70.3 | 1.0 / 1.0 | - |
| slots_no_labels | 1.4 / 1.5 | 14.2 / 14.9 | 0.8 / 0.8 | - |

列说明:每格是一个任务(500 个 block)一次调用耗时的 中位 / 均值;allocate / release = 从 pool 取、还 block(现状用 ScratchPool,返回 Python tensor;方案 B 用整数编号);submit = 调一次 C++ 提交函数,含 pybind 把列表逐个转成 C++ 值;wait = zip_to_mem 形态下主线程在 C++ 里 future.get() 的耗时(此时队列线程已做完,lambda 在这里销毁)。

## 传法 / 主线程状态

- py_tensors: 现状 unzip_from_mem:每个 block 一个 ScratchPool 的 Python tensor + label 字符串

- py_tensors_kept_future: 现状 zip_to_mem:同上,但 lambda 捕获 tensor 且 future 留到 wait

- slots: 方案 B:只传 slot 编号,C++ 存地址;label 字符串照旧

- cpp_views: 方案 B':只传 slot 编号,C++ 用自建的 view(无 Python 对象);label 照旧

- slots_no_labels: 下限:只传 slot 编号,且不传 label 字符串

- wait_in_cpp: 主线程停在 C++ 里等(已放 GIL)

- torch_ops_loop: 主线程循环调用小 torch 算子(每个算子前后放、拿一次 GIL)

- pure_python_loop: 主线程循环跑纯 Python 代码(不主动放 GIL)
