## 1. 队列线程上每个任务结束到下一个任务开始的间隔(release + 取下一个任务)

| 主线程状态 | 传法 | 间隔个数 | 间隔中位(µs) | 间隔平均(µs) | 间隔p95(µs) | 间隔最大(µs) | 任务执行中位(µs) |
|---|---|---|---|---|---|---|---|
| wait_in_cpp | py_tensors | 790 | 558.2 | 562.4 | 585.3 | 606.8 | 1341.2 |
| wait_in_cpp | py_tensors_kept_future | 790 | 1.5 | 1.7 | 2.2 | 5.8 | 1357.3 |
| wait_in_cpp | slots | 790 | 18.7 | 18.8 | 21.3 | 47.6 | 1342.4 |
| wait_in_cpp | cpp_views | 790 | 35.6 | 35.8 | 41.0 | 66.3 | 1349.6 |
| wait_in_cpp | slots_no_labels | 790 | 1.5 | 1.7 | 2.8 | 6.3 | 1350.0 |
| torch_ops_loop | py_tensors | 790 | 5435.6 | 5592.0 | 7111.7 | 8603.4 | 1327.9 |
| torch_ops_loop | slots | 790 | 20.1 | 20.2 | 22.3 | 27.9 | 1344.8 |
| torch_ops_loop | cpp_views | 790 | 35.9 | 36.2 | 41.7 | 46.5 | 1343.0 |
| torch_ops_loop | slots_no_labels | 790 | 1.4 | 1.7 | 2.8 | 5.7 | 1342.2 |
| pure_python_loop | py_tensors | 21 | 3353240.1 | 3431777.2 | 4010493.6 | 4116664.6 | 1349.6 |
| pure_python_loop | slots | 21 | 22.4 | 21.6 | 24.4 | 24.6 | 1001.2 |
| pure_python_loop | cpp_views | 21 | 35.1 | 34.7 | 38.4 | 40.0 | 974.3 |
| pure_python_loop | slots_no_labels | 21 | 1.3 | 1.4 | 2.0 | 2.4 | 645.8 |

列说明:主线程状态 / 传法 见文末;间隔个数 = 统计到的间隔数(每轮最后一个任务后面没有任务,不计);间隔 = 下一个任务开始时刻 − 本任务结束时刻,即释放本任务持有的数据 + 队列取下一个任务;任务执行 = 任务函数本身(把 2000 个 block 各写一遍)的耗时。

## 2. 间隔拆分(按中位数相减)

| 主线程状态 | 现状间隔(µs) | tensor 部分(µs) | tensor 每 block(ns) | label 部分(µs) | 下限(µs) | B 去掉的比例 | B' 去掉的比例 | B 去掉的比例(按均值) |
|---|---|---|---|---|---|---|---|---|
| wait_in_cpp | 558.2 | 539.5 | 270 | 17.2 | 1.5 | 96.7% | 93.6% | 96.7% |
| torch_ops_loop | 5435.6 | 5415.5 | 2708 | 18.7 | 1.4 | 99.6% | 99.3% | 99.6% |
| pure_python_loop | 3353240.1 | 3353217.7 | 1676609 | 21.1 | 1.3 | 100.0% | 100.0% | 100.0% |

列说明:现状间隔 = py_tensors 的间隔中位;tensor 部分 = py_tensors − slots(两者只差传 tensor 还是传编号);tensor 每 block = tensor 部分 ÷ 每任务 block 数;label 部分 = slots − slots_no_labels(只差有没有 label 字符串);下限 = slots_no_labels,即没有 tensor 也没有 label 时剩下的间隔;B / B' 去掉的比例 = (现状 − slots 或 cpp_views) ÷ 现状;最后一列用两者的间隔均值代替中位数计算。

## 3. 主线程上每次调用的耗时(主线程状态 wait_in_cpp 的各轮)

| 传法 | allocate(µs) | submit(µs) | release(µs) | wait(µs) |
|---|---|---|---|---|
| py_tensors | 41.2 / 41.3 | 327.1 / 328.7 | 88.1 / 93.2 | - |
| py_tensors_kept_future | 32.5 / 32.9 | 331.4 / 331.8 | 87.2 / 96.4 | 149.8 / 152.7 |
| slots | 7.6 / 7.7 | 252.8 / 254.1 | 5.3 / 5.3 | - |
| cpp_views | 5.9 / 6.0 | 274.8 / 275.6 | 4.6 / 4.6 | - |
| slots_no_labels | 5.5 / 5.7 | 54.3 / 55.4 | 4.2 / 4.3 | - |

列说明:每格是一个任务(2000 个 block)一次调用耗时的 中位 / 均值;allocate / release = 从 pool 取、还 block(现状用 ScratchPool,返回 Python tensor;方案 B 用整数编号);submit = 调一次 C++ 提交函数,含 pybind 把列表逐个转成 C++ 值;wait = zip_to_mem 形态下主线程在 C++ 里 future.get() 的耗时(此时队列线程已做完,lambda 在这里销毁)。

## 传法 / 主线程状态

- py_tensors: 现状 unzip_from_mem:每个 block 一个 ScratchPool 的 Python tensor + label 字符串

- py_tensors_kept_future: 现状 zip_to_mem:同上,但 lambda 捕获 tensor 且 future 留到 wait

- slots: 方案 B:只传 slot 编号,C++ 存地址;label 字符串照旧

- cpp_views: 方案 B':只传 slot 编号,C++ 用自建的 view(无 Python 对象);label 照旧

- slots_no_labels: 下限:只传 slot 编号,且不传 label 字符串

- wait_in_cpp: 主线程停在 C++ 里等(已放 GIL)

- torch_ops_loop: 主线程循环调用小 torch 算子(每个算子前后放、拿一次 GIL)

- pure_python_loop: 主线程循环跑纯 Python 代码(不主动放 GIL)
