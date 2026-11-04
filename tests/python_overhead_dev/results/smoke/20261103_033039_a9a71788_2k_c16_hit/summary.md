# py-spy 结果:2k_c16,hit

## 这次运行是什么

- vLLM 0.29.0 + kvshrink,模型 `/media/model-space/Qwen3-32B`,TP=1,GPU 5,vLLM 进程绑定 CPU 64-127。调度、模型计算和 kvshrink 的全部函数都在 EngineCore 进程的主线程上。
- 一次 bench = `vllm bench serve` 发 64 个请求,输入 2048 token、输出 128 token,并发 16。
- 场景 hit:所有请求的 prompt 相同,测量前已存进 DDR,kvshrink 只加载、不保存。
- py-spy 从测量 bench 开始前采到结束后,采样率 100 次/秒,只采 EngineCore 进程(pid 2119996),参数 `--rate 100 --nonblocking --threads --format raw --full-filenames`:不暂停被测进程,也不记录 C/C++ 函数,在 C/C++ 里花的时间算在调用它的那一行 Python 上。一个样本约代表 10 ms;“毫秒/请求”= 样本数 ÷ 采样率 × 1000 ÷ 请求数。相对误差约 1/√样本数。
- 代码 git 28f2d3e,代码 md5 a9a71788,py-spy py-spy 0.4.3。

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
| py-spy 自报的样本数 / 错误数 | 412 / 133 |
| 主线程样本里调用链不完整的样本数(占比) | 55(14.0%) |

列说明:“核对项”是检查什么;“结果”是从 connector 日志、vLLM 日志或 py-spy 输出数出来的值。miss 场景外部命中的请求数应为 0;hit 场景应等于请求数。“调用链不完整”指最外层不是进程入口 `<module>` 的样本:py-spy 不暂停进程时,偶尔读到一半就变了的调用链。

### 有没有 py-spy 的 bench 对比(干扰检查)

| 指标 | 没有 py-spy | 有 py-spy | 有 / 没有 |
|---|---|---|---|
| duration | 33.44 | 33.16 | 0.992 |
| completed | 64.00 | 64.00 | 1.000 |
| request_throughput | 1.91 | 1.93 | 1.008 |
| mean_ttft_ms | 440.39 | 404.88 | 0.919 |
| median_ttft_ms | 387.75 | 403.24 | 1.040 |
| mean_tpot_ms | 61.70 | 61.56 | 0.998 |
| median_tpot_ms | 61.76 | 61.57 | 0.997 |
| mean_e2el_ms | 8275.91 | 8223.10 | 0.994 |
| median_e2el_ms | 8196.20 | 8210.47 | 1.002 |

列说明:“指标”是 `vllm bench serve` 结果里的字段(duration 单位秒,throughput 单位请求/秒,其余单位 ms,mean 是均值、median 是中位数,统计对象是这次 bench 的全部请求);后两列是两次 bench 的值;“有 / 没有”是比值,越接近 1 说明 py-spy 干扰越小。

## 表 1:按入口函数汇总(毫秒/请求)

| 入口函数 | Python 开销 | 对照:ScratchPool | 对照:IAXL C++(含参数转换) | 样本数 |
|---|---|---|---|---|
| `KVShrinkConnector.start_load_kv` | 0.47 | 0.00 | 0.47 | 6 |
| `KVShrinkConnector.wait_for_layer_load` | 0.31 | 0.47 | 2.19 | 19 |
| `KVShrinkConnector.save_kv_layer` | 0.31 | 0.00 | 0.00 | 2 |
| `KVShrinkConnector.get_num_new_matched_tokens` | 0.16 | 0.00 | 0.16 | 2 |
| `KVStoreLocal.get_wait` | 0.00 | 0.16 | 0.00 | 1 |
| `KVShrinkConnector.get_finished` | 0.00 | 0.16 | 0.00 | 1 |
| 主线程上 kvshrink 合计 | 1.25 | 0.78 | 2.81 | 31 |
| 其他线程上的 kvshrink | 0.16 | 0.00 | 0.00 | 1 |

主线程 CPU 忙碌总时间(含 vLLM 自己的代码):61.56 毫秒/请求(394 个样本);kvshrink 的 Python 开销占其中 2.0%。

列说明:“入口函数”是调用链里最外面的 kvshrink 函数,即 vLLM 调用的那个;“Python 开销”是本次要量的量:kvshrink 的 Python 语句,加上它们直接调用的 Python 代码和xxhash、numpy、torch 等库的 C 代码;两个“对照”列不算 Python 开销:ScratchPool 取块还块归同事,IAXL C++ 是调用 IAXL 扩展的那些行上的时间(C++ 执行、pybind11 转换参数、空转等待,哪些行见附 1);以上单位都是毫秒/请求;“样本数”是这一行三类的样本总数。“其他线程上的 kvshrink”是管理接口、内存池淘汰等后台线程。

## 表 2:Python 开销最多的前 20 个代码位置

| 代码位置 | 源代码 | 所在函数 | 入口函数 | 毫秒/请求 | 占 Python 开销的比例 | 样本数 |
|---|---|---|---|---|---|---|
| `kvshrink/kvshrink_connector.py:718` | `for req_id, request in metadata.reqs_to_save.requests.items():` | `KVShrinkConnector.save_kv_layer` | `KVShrinkConnector.save_kv_layer` | 0.31 | 25.0% | 2 |
| `iaxl/kvflow/flow.py:399` | `results[tensor_key] = task = Task(` | `KVFlow.get` | `KVShrinkConnector.start_load_kv` | 0.16 | 12.5% | 1 |
| `iaxl/utils/hash.py:12` | `def compute_hash(*args, **kwargs) -> int:` | `compute_hash` | `KVShrinkConnector.get_num_new_matched_tokens` | 0.16 | 12.5% | 1 |
| `iaxl/kvflow/flow.py:407` | `self._pending_gets[id(task)] = task` | `KVFlow.get` | `KVShrinkConnector.start_load_kv` | 0.16 | 12.5% | 1 |
| `iaxl/kvflow/flow.py:463` | `self.chunk_pool.release(result.cpu_tensors)` | `KVFlow.get_wait` | `KVShrinkConnector.wait_for_layer_load` | 0.16 | 12.5% | 1 |
| `iaxl/kvflow/flow.py:165` | `while (tensors := self.chunk_pool.allocate(count, shape, dtype)) is None:` | `KVFlow._allocate` | `KVShrinkConnector.start_load_kv` | 0.16 | 12.5% | 1 |
| `kvshrink/kvshrink_connector.py:669` | `if not self._current_get_tasks and not self._active_promoted_tasks:` | `KVShrinkConnector.wait_for_layer_load` | `KVShrinkConnector.wait_for_layer_load` | 0.16 | 12.5% | 1 |

列说明:“代码位置”是最里面的 kvshrink 帧的文件和行号,即那一刻正在执行的 kvshrink 代码;“源代码”是这一行的内容;“所在函数”是这一行所在的函数;“入口函数”同表 1;“毫秒/请求”是这一行的 Python 开销;“占 Python 开销的比例”是这一行 ÷ 主线程上 kvshrink 的Python 开销合计;“样本数”是这一行的样本数。

## 附 1:算作 IAXL C++ 的代码行(前 15 个,核对归类用)

| 代码位置 | 源代码 | 所在函数 | 入口函数 | 毫秒/请求 | 占 IAXL C++ 的比例 | 样本数 |
|---|---|---|---|---|---|---|
| `iaxl/kvflow/flow.py:458` | `result.ctx.xfer_wait()` | `KVFlow.get_wait` | `KVShrinkConnector.wait_for_layer_load` | 1.25 | 44.4% | 8 |
| `iaxl/kvflow/flow.py:457` | `result.ctx.unzip_wait()` | `KVFlow.get_wait` | `KVShrinkConnector.wait_for_layer_load` | 0.94 | 33.3% | 6 |
| `iaxl/kvflow/flow.py:387` | `ctx.unzip_from_mem(` | `KVFlow.get` | `KVShrinkConnector.start_load_kv` | 0.47 | 16.7% | 3 |
| `iaxl/kvflow/flow.py:484` | `return self.record.has(label, chunk_labels)` | `KVFlow.has` | `KVShrinkConnector.get_num_new_matched_tokens` | 0.16 | 5.6% | 1 |

列说明:同表 2,只是这里列的是被算作 IAXL C++ 的行(调用 IAXL 扩展的行);“占 IAXL C++ 的比例”是这一行 ÷ 主线程上 IAXL C++ 的合计。

## 附 2:各线程的样本数

| 线程 | 样本数 |
|---|---|
| MainThread | 394 |
| thread (2123586) | 13 |
| Thread-3 (process_output_sockets) | 2 |
| mgmt-ctrl_0 | 1 |
| Thread-2 (process_input_sockets) | 1 |
| thread (0) | 1 |

列说明:“线程”是 py-spy 记录的线程名;“样本数”是这个线程在忙(没睡眠、没阻塞)时被采到的次数。
