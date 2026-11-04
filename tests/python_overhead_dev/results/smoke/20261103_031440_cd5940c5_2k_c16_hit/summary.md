# py-spy 结果:2k_c16,hit

## 这次运行是什么

- vLLM 0.29.0 + kvshrink,模型 `/media/model-space/Qwen3-32B`,TP=1,GPU 5,vLLM 进程绑定 CPU 64-127。调度、模型计算和 kvshrink 的全部函数都在 EngineCore 进程的主线程上。
- 一次 bench = `vllm bench serve` 发 64 个请求,输入 2048 token、输出 128 token,并发 16。
- 场景 hit:所有请求的 prompt 相同,测量前已存进 DDR,kvshrink 只加载、不保存。
- py-spy 从测量 bench 开始前采到结束后,采样率 100 次/秒,只采 EngineCore 进程(pid 2101436)。一个样本约代表 10 ms;“毫秒/请求”= 样本数 ÷ 采样率 × 1000 ÷ 请求数。相对误差约 1/√样本数。
- 代码 git 28f2d3e,代码 md5 cd5940c5,py-spy py-spy 0.4.3。

## 核对

| 核对项 | 结果 |
|---|---|
| baseline 这次 bench 的请求数(connector 日志) | 64 |
| baseline 里有外部命中(externally-cached tokens > 0)的请求数 | 64 |
| baseline 里 externally-cached tokens 的不同取值(前 5 个) | 2032 |
| pyspy 这次 bench 的请求数(connector 日志) | 64 |
| pyspy 里有外部命中(externally-cached tokens > 0)的请求数 | 64 |
| pyspy 里 externally-cached tokens 的不同取值(前 5 个) | 2032 |
| vLLM 日志里 `ScratchPool exhausted` 的次数 | 0 |
| py-spy 跟不上采样率的警告次数 | 0 |
| py-spy 自报的样本数 / 错误数 | 5296 / 77 |

列说明:“核对项”是检查什么;“结果”是从 connector 日志、vLLM 日志或 py-spy 输出数出来的值。miss 场景外部命中的请求数应为 0;hit 场景应等于请求数。

### 有没有 py-spy 的 bench 对比(干扰检查)

| 指标 | 没有 py-spy | 有 py-spy | 有 / 没有 |
|---|---|---|---|
| duration | 33.23 | 60.58 | 1.823 |
| completed | 64.00 | 64.00 | 1.000 |
| request_throughput | 1.93 | 1.06 | 0.548 |
| mean_ttft_ms | 406.59 | 1891.51 | 4.652 |
| median_ttft_ms | 404.86 | 1177.29 | 2.908 |
| mean_tpot_ms | 61.66 | 103.29 | 1.675 |
| median_tpot_ms | 61.71 | 102.45 | 1.660 |
| mean_e2el_ms | 8237.34 | 15009.85 | 1.822 |
| median_e2el_ms | 8240.14 | 14331.54 | 1.739 |

列说明:“指标”是 `vllm bench serve` 结果里的字段(duration 单位秒,throughput 单位请求/秒,其余单位 ms,mean 是均值、median 是中位数,统计对象是这次 bench 的全部请求);后两列是两次 bench 的值;“有 / 没有”是比值,越接近 1 说明 py-spy 干扰越小。

## 表 1:按入口函数汇总(毫秒/请求)

| 入口函数 | Python 开销 | 其中 Python 本身 | 其中其他 C 代码 | 对照:ScratchPool | 对照:IAXL C++(含参数转换) | 样本数 |
|---|---|---|---|---|---|---|
| `KVShrinkConnector.wait_for_layer_load` | 27.50 | 0.31 | 27.19 | 4.53 | 16.41 | 310 |
| `KVShrinkConnector.get_num_new_matched_tokens` | 3.75 | 3.59 | 0.16 | 0.00 | 1.88 | 36 |
| `KVShrinkConnector.save_kv_layer` | 3.44 | 3.44 | 0.00 | 0.00 | 0.00 | 22 |
| `KVShrinkConnector.start_load_kv` | 3.28 | 1.56 | 1.72 | 3.59 | 13.28 | 129 |
| `KVShrinkConnector.get_finished` | 0.62 | 0.47 | 0.16 | 3.44 | 0.94 | 32 |
| `KVShrinkConnector.build_connector_meta` | 0.47 | 0.31 | 0.16 | 0.00 | 0.00 | 3 |
| `KVShrinkConnector.update_state_after_alloc` | 0.16 | 0.16 | 0.00 | 0.00 | 0.00 | 1 |
| `KVShrinkConnector.request_finished_all_groups` | 0.16 | 0.16 | 0.00 | 0.00 | 0.00 | 1 |
| 主线程上 kvshrink 合计 | 39.38 | 10.00 | 29.38 | 11.56 | 32.50 | 534 |
| 其他线程上的 kvshrink | 2.34 | 0.94 | 1.41 | 0.00 | 0.00 | 15 |

主线程 CPU 忙碌总时间(含 vLLM 自己的代码):739.22 毫秒/请求(4731 个样本);kvshrink 的 Python 开销占其中 5.3%。

列说明:“入口函数”是调用链里最外面的 kvshrink 函数,即 vLLM 调用的那个;“Python 开销”= 后两列之和,是本次要量的量;“其中 Python 本身”是 Python 语句自己的时间,“其中其他 C 代码”是它直接调用的 xxhash、numpy、torch、libc 等库的时间;两个“对照”列不算 Python 开销:ScratchPool 取块还块归同事,IAXL C++ 是 IAXL 扩展库里的时间(库没有符号,pybind11 的参数转换也在这一列);“样本数”是这一行全部类别的样本数。“其他线程上的 kvshrink”是管理接口、内存池淘汰等后台线程。

## 表 2:Python 开销最多的前 20 个代码位置

| 代码位置 | 所在函数 | 入口函数 | 毫秒/请求 | 其中其他 C 代码(库: 样本数) | 占 Python 开销的比例 | 样本数 |
|---|---|---|---|---|---|---|
| `iaxl/kvflow/flow.py:458` | `KVFlow.get_wait` | `KVShrinkConnector.wait_for_layer_load` | 13.91 | 13.91(? 89) | 35.3% | 89 |
| `iaxl/kvflow/flow.py:457` | `KVFlow.get_wait` | `KVShrinkConnector.wait_for_layer_load` | 12.66 | 12.66(? 81) | 32.1% | 81 |
| `iaxl/kvflow/flow.py:387` | `KVFlow.get` | `KVShrinkConnector.start_load_kv` | 1.41 | 1.41(libc.so.6 9) | 3.6% | 9 |
| `iaxl/utils/hash.py:13` | `compute_hash` | `KVShrinkConnector.get_num_new_matched_tokens` | 1.41 | - | 3.6% | 9 |
| `kvshrink/kvshrink_connector.py:718` | `KVShrinkConnector.save_kv_layer` | `KVShrinkConnector.save_kv_layer` | 1.25 | - | 3.2% | 8 |
| `kvshrink/kvshrink_connector.py:709` | `KVShrinkConnector.save_kv_layer` | `KVShrinkConnector.save_kv_layer` | 0.78 | - | 2.0% | 5 |
| `iaxl/kvflow/flow.py:460` | `KVFlow.get_wait` | `KVShrinkConnector.wait_for_layer_load` | 0.62 | 0.62(libtorch_python.so 4) | 1.6% | 4 |
| `kvshrink/kvshrink_connector.py:712` | `KVShrinkConnector.save_kv_layer` | `KVShrinkConnector.save_kv_layer` | 0.47 | - | 1.2% | 3 |
| `iaxl/utils/hash.py:14` | `compute_hash` | `KVShrinkConnector.get_num_new_matched_tokens` | 0.47 | - | 1.2% | 3 |
| `kvshrink/kvshrink_connector.py:283` | `KVShrinkConnector.get_num_new_matched_tokens` | `KVShrinkConnector.get_num_new_matched_tokens` | 0.47 | - | 1.2% | 3 |
| `kvshrink/kvshrink_connector.py:713` | `KVShrinkConnector.save_kv_layer` | `KVShrinkConnector.save_kv_layer` | 0.31 | - | 0.8% | 2 |
| `kvshrink/kvshrink_connector.py:717` | `KVShrinkConnector.save_kv_layer` | `KVShrinkConnector.save_kv_layer` | 0.31 | - | 0.8% | 2 |
| `iaxl/kvflow/flow.py:439` | `KVFlow.get_wait` | `KVShrinkConnector.get_finished` | 0.31 | - | 0.8% | 2 |
| `kvshrink/kvshrink_connector.py:669` | `KVShrinkConnector.wait_for_layer_load` | `KVShrinkConnector.wait_for_layer_load` | 0.31 | - | 0.8% | 2 |
| `kvshrink/kvshrink_connector.py:716` | `KVShrinkConnector.save_kv_layer` | `KVShrinkConnector.save_kv_layer` | 0.31 | - | 0.8% | 2 |
| `kvshrink/kvshrink_connector.py:336` | `KVShrinkConnector.get_num_new_matched_tokens` | `KVShrinkConnector.get_num_new_matched_tokens` | 0.31 | - | 0.8% | 2 |
| `iaxl/kvflow/flow.py:351` | `KVFlow.get` | `KVShrinkConnector.start_load_kv` | 0.31 | - | 0.8% | 2 |
| `kvshrink/kvshrink_connector.py:299` | `KVShrinkConnector.get_num_new_matched_tokens.<locals>.<genexpr>` | `KVShrinkConnector.get_num_new_matched_tokens` | 0.16 | - | 0.4% | 1 |
| `iaxl/utils/hash.py:35` | `generate_block_hashs` | `KVShrinkConnector.get_num_new_matched_tokens` | 0.16 | - | 0.4% | 1 |
| `kvshrink/kvshrink_connector.py:285` | `KVShrinkConnector.get_num_new_matched_tokens` | `KVShrinkConnector.get_num_new_matched_tokens` | 0.16 | - | 0.4% | 1 |

列说明:“代码位置”是最里面的 kvshrink 帧的文件和行号,即那一刻正在执行的 kvshrink 代码;“所在函数”是这一行所在的函数;“入口函数”同表 1;“毫秒/请求”是这一行的 Python 开销;“其中其他 C 代码”是其中花在其他库里的部分,括号里是库名和样本数;“占 Python 开销的比例”是这一行 ÷ 主线程上 kvshrink 的 Python 开销合计;“样本数”是这一行的样本数。

## 附 1:各类别里样本最多的 C/C++ 函数(核对归类用)

| 类别 | C/C++ 函数(库) | 样本数 |
|---|---|---|
| 其他 C 代码 | `0x3b75fa00 (?)` | 170 |
| 其他 C 代码 | `pybind11_object_dealloc (libtorch_python.so)` | 5 |
| 其他 C 代码 | `_IO_file_seekoff (libc.so.6)` | 5 |
| 其他 C 代码 | `0x7e5dd918f533 (libc.so.6)` | 3 |
| 其他 C 代码 | `pthread_condattr_setpshared (libc.so.6)` | 1 |
| 其他 C 代码 | `THPVariable_get_shape (libtorch_python.so)` | 1 |
| 其他 C 代码 | `THPVariable_dtype (libtorch_python.so)` | 1 |
| 其他 C 代码 | `PYXXH64_vectorcall (_xxhash.c)` | 1 |
| 其他 C 代码 | `__create_fn__.<locals>.__init__ (<string>)` | 1 |
| IAXL C++ | `0x7e5a89cdbf68 (torch_ext.cpython-312-x86_64-linux-gnu.so)` | 202 |
| IAXL C++ | `0x7e5a89cdc149 (torch_ext.cpython-312-x86_64-linux-gnu.so)` | 3 |
| IAXL C++ | `0x7e5a89cdba05 (torch_ext.cpython-312-x86_64-linux-gnu.so)` | 2 |
| IAXL C++ | `0x7e5a89cdc48d (torch_ext.cpython-312-x86_64-linux-gnu.so)` | 1 |
| ScratchPool | `tc_realloc (libtcmalloc_minimal.so.4.5.16)` | 1 |

列说明:“类别”同表 1;“C/C++ 函数(库)”是样本里最里面的 kvshrink 帧下面第一个 C/C++ 函数及其所在库(没有符号的显示为地址);“样本数”是落在它上面的主线程样本数。

## 附 2:各线程的样本数

| 线程 | 样本数 |
|---|---|
| MainThread | 4731 |
| thread (2106889) | 493 |
| Thread-3 (process_output_sockets) | 37 |
| Thread-2 (process_input_sockets) | 14 |
| iaxl-monitor | 6 |
| mgmt-ctrl_0 | 5 |
| Thread-27 (process_request_thread) | 3 |
| Thread-30 (process_request_thread) | 3 |
| Thread-29 (process_request_thread) | 2 |
| mgmt-worker-0 | 1 |
| tqdm_monitor | 1 |

列说明:“线程”是 py-spy 记录的线程名;“样本数”是这个线程在忙(没睡眠、没阻塞)时被采到的次数。
