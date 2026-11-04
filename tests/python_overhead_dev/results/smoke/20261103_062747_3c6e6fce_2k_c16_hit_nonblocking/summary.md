# py-spy 结果:2k_c16,hit

## 这次运行是什么

- vLLM 0.29.0 + kvshrink,模型 `/media/model-space/Qwen3-32B`,TP=1,GPU 5,vLLM 进程绑定 CPU 64-127。调度、模型计算和 kvshrink 的全部函数都在 EngineCore 进程的主线程上。
- 一次 bench = `vllm bench serve` 发 64 个请求,输入 2048 token、输出 128 token,并发 16。
- 场景 hit:所有请求的 prompt 相同,测量前已存进 DDR,kvshrink 只加载、不保存。
- py-spy 从测量 bench 开始前采到结束后,采样率 100 次/秒,只采 EngineCore 进程(pid 56530),参数 `--rate 100 --threads --format raw --full-filenames --nonblocking`:不暂停被测进程,不记录 C/C++ 函数,在 C/C++ 里花的时间算在调用它的那一行 Python 上。一个样本约代表 10 ms;“毫秒/请求”= 样本数 ÷ 采样率 × 1000 ÷ 请求数。相对误差约 1/√样本数。
- 代码 git 28f2d3e,代码 md5 3c6e6fce,py-spy py-spy 0.4.3。

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
| py-spy 自报的样本数 / 错误数 | 1330 / 359 |
| 主线程样本里调用链不完整的样本数(占比) | 141(11.3%) |

列说明:“核对项”是检查什么;“结果”是从 connector 日志、vLLM 日志或 py-spy 输出数出来的值。miss 场景外部命中的请求数应为 0;hit 场景应等于请求数。“调用链不完整”指最外层不是进程入口 `<module>` 的样本:py-spy 不暂停进程时,偶尔读到一半就变了的调用链。

### 有没有 py-spy 的 bench 对比(干扰检查)

| 指标 | 没有 py-spy | 有 py-spy | 有 / 没有 |
|---|---|---|---|
| duration | 33.56 | 33.88 | 1.009 |
| completed | 64.00 | 64.00 | 1.000 |
| request_throughput | 1.91 | 1.89 | 0.991 |
| mean_ttft_ms | 466.75 | 490.88 | 1.052 |
| median_ttft_ms | 463.02 | 380.17 | 0.821 |
| mean_tpot_ms | 61.61 | 62.01 | 1.007 |
| median_tpot_ms | 61.57 | 61.92 | 1.006 |
| mean_e2el_ms | 8290.62 | 8365.80 | 1.009 |
| median_e2el_ms | 8292.20 | 8280.37 | 0.999 |

列说明:“指标”是 `vllm bench serve` 结果里的字段(duration 单位秒,throughput 单位请求/秒,其余单位 ms,mean 是均值、median 是中位数,统计对象是这次 bench 的全部请求);后两列是两次 bench 的值;“有 / 没有”是比值,越接近 1 说明 py-spy 干扰越小。

### py-spy 数的主线程忙碌时间对不对(和内核记的 CPU 时间比)

| 量 | 没有 py-spy 的 bench | 有 py-spy 的 bench |
|---|---|---|
| EngineCore 主线程 CPU 时间(内核) | 258.28 | 242.50 |
| EngineCore 全进程 CPU 时间(内核) | 550.78 | 567.19 |
| py-spy 数的主线程忙碌时间 | - | 194.69 |
| 校准系数 = 内核 ÷ py-spy(主线程) | - | 1.246 |

列说明:单位都是毫秒/请求(校准系数除外)。“内核”两行是 /proc 里 EngineCore 的 utime + stime(内核记账的、真正占用 CPU 的时间),在每次 bench 前后各读一次,差值 ÷ 请求数;主线程的线程号就是进程号,全进程包括IAXL、torch 等全部线程。“py-spy 数的主线程忙碌时间”= 主线程被判为忙的样本数 ÷ 采样率 ÷ 请求数。py-spy 采样窗口比 bench 略长(前后各多几秒,那时 EngineCore 没有请求),所以它和同一次 bench 的“主线程 CPU 时间(内核)”应基本相等;差很多说明 py-spy 漏采或多采。“校准系数”= 有 py-spy 的 bench 的“主线程 CPU 时间(内核)”÷“py-spy 数的主线程忙碌时间”,表 1 用它把 py-spy 直接数的主线程时间换算成内核口径。

## 表 1:按入口函数汇总(毫秒/请求)

| 入口函数 | Python 开销(py-spy 直接数) | Python 开销(按内核校准) | 对照:ScratchPool | 对照:IAXL C++(含参数转换) | 样本数 |
|---|---|---|---|---|---|
| `KVShrinkConnector.save_kv_layer` | 4.69 | 5.84 | 0.00 | 0.00 | 30 |
| `KVShrinkConnector.start_load_kv` | 0.94 | 1.17 | 1.09 | 3.59 | 36 |
| `KVShrinkConnector.wait_for_layer_load` | 0.47 | 0.58 | 0.94 | 4.84 | 40 |
| `KVShrinkConnector.get_num_new_matched_tokens` | 0.47 | 0.58 | 0.00 | 0.31 | 5 |
| `KVShrinkConnector.build_connector_meta` | 0.31 | 0.39 | 0.00 | 0.00 | 2 |
| `KVShrinkConnector.get_finished` | 0.16 | 0.19 | 0.62 | 0.31 | 7 |
| `KVStoreLocal.get_wait` | 0.00 | 0.00 | 0.16 | 0.00 | 1 |
| `ScratchPool.release.<locals>.<genexpr>` | 0.00 | 0.00 | 0.16 | 0.00 | 1 |
| 主线程上 kvshrink 合计 | 7.03 | 8.76 | 2.97 | 9.06 | 122 |
| 其他线程上的 kvshrink | 0.16 | - | 0.00 | 0.00 | 1 |

主线程 CPU 忙碌总时间(含 vLLM 自己的代码):py-spy 直接数 194.69 毫秒/请求(1246 个样本),内核记 242.50 毫秒/请求;kvshrink 的 Python 开销占 py-spy 主线程忙碌样本的 3.6%。

列说明:“入口函数”是调用链里最外面的 kvshrink 函数,即 vLLM 调用的那个;“Python 开销”是本次要量的量:kvshrink 的 Python 语句,加上它们直接调用的 Python 代码和xxhash、numpy、torch 等库的 C 代码;“py-spy 直接数”= 样本数 ÷ 采样率 ÷ 请求数;“按内核校准”= 直接数的值 × 校准系数(见上一节),假设 py-spy 漏掉的样本均匀分布在各段代码上,其他线程不校准(校准系数只对主线程测过);两个“对照”列不算 Python 开销,是 py-spy 直接数的值:ScratchPool 取块还块归同事,IAXL C++ 是调用 IAXL 扩展的那些行上的时间(C++ 执行、pybind11 转换参数、空转等待,哪些行见附 1);以上单位都是毫秒/请求;“样本数”是这一行三类的样本总数。“其他线程上的 kvshrink”是管理接口、内存池淘汰等后台线程。

## 表 2:Python 开销最多的前 20 个代码位置

| 代码位置 | 源代码 | 所在函数 | 入口函数 | 毫秒/请求 | 占 Python 开销的比例 | 样本数 |
|---|---|---|---|---|---|---|
| `kvshrink/kvshrink_connector.py:718` | `for req_id, request in metadata.reqs_to_save.requests.items():` | `KVShrinkConnector.save_kv_layer` | `KVShrinkConnector.save_kv_layer` | 3.75 | 53.3% | 24 |
| `kvshrink/kvshrink_connector.py:712` | `metadata = self._get_connector_metadata()` | `KVShrinkConnector.save_kv_layer` | `KVShrinkConnector.save_kv_layer` | 0.47 | 6.7% | 3 |
| `kvshrink/kvshrink_connector.py:688` | `success = self._store().get_wait(` | `KVShrinkConnector.wait_for_layer_load` | `KVShrinkConnector.wait_for_layer_load` | 0.16 | 2.2% | 1 |
| `kvshrink/kvshrink_connector.py:285` | `request.all_token_ids[:-1], self.block_size` | `KVShrinkConnector.get_num_new_matched_tokens` | `KVShrinkConnector.get_num_new_matched_tokens` | 0.16 | 2.2% | 1 |
| `iaxl/kvflow/flow.py:463` | `self.chunk_pool.release(result.cpu_tensors)` | `KVFlow.get_wait` | `KVShrinkConnector.wait_for_layer_load` | 0.16 | 2.2% | 1 |
| `iaxl/kvflow/flow.py:399` | `results[tensor_key] = task = Task(` | `KVFlow.get` | `KVShrinkConnector.start_load_kv` | 0.16 | 2.2% | 1 |
| `iaxl/kvflow/flow.py:463` | `self.chunk_pool.release(result.cpu_tensors)` | `KVFlow.get_wait` | `KVShrinkConnector.get_finished` | 0.16 | 2.2% | 1 |
| `iaxl/kvflow/flow.py:165` | `while (tensors := self.chunk_pool.allocate(count, shape, dtype)) is None:` | `KVFlow._allocate` | `KVShrinkConnector.start_load_kv` | 0.16 | 2.2% | 1 |
| `iaxl/utils/hash.py:13` | `parts = [repr(arg) for arg in args]` | `compute_hash` | `KVShrinkConnector.get_num_new_matched_tokens` | 0.16 | 2.2% | 1 |
| `iaxl/kvflow/flow.py:372` | `cpu_tensors = self._allocate(num_chunks, chunk_shape, tensor.dtype)` | `KVFlow.get` | `KVShrinkConnector.start_load_kv` | 0.16 | 2.2% | 1 |
| `iaxl/utils/hash.py:16` | `return xxhash.xxh64(text.encode("utf-8")).intdigest() & 0x7FFFFFFFFFFFFFFF` | `compute_hash` | `KVShrinkConnector.get_num_new_matched_tokens` | 0.16 | 2.2% | 1 |
| `kvshrink/kvshrink_connector.py:468` | `return metadata` | `KVShrinkConnector.build_connector_meta` | `KVShrinkConnector.build_connector_meta` | 0.16 | 2.2% | 1 |
| `iaxl/kvflow/flow.py:407` | `self._pending_gets[id(task)] = task` | `KVFlow.get` | `KVShrinkConnector.start_load_kv` | 0.16 | 2.2% | 1 |
| `kvshrink/kvshrink_connector.py:713` | `if not isinstance(metadata, KVShrinkConnectorMetadata):` | `KVShrinkConnector.save_kv_layer` | `KVShrinkConnector.save_kv_layer` | 0.16 | 2.2% | 1 |
| `iaxl/kvflow/flow.py:460` | `result.ctx = None` | `KVFlow.get_wait` | `KVShrinkConnector.wait_for_layer_load` | 0.16 | 2.2% | 1 |
| `kvshrink/kvshrink_connector.py:383` | `start = num_computed_tokens // self.block_size` | `KVShrinkConnector._add_request_to_save` | `KVShrinkConnector.build_connector_meta` | 0.16 | 2.2% | 1 |
| `iaxl/kvflow/flow.py:368` | `for tensor_key, tensor in tensors.items():` | `KVFlow.get` | `KVShrinkConnector.start_load_kv` | 0.16 | 2.2% | 1 |
| `iaxl/kvflow/flow.py:349` | `assert tensor.is_contiguous(), "all GPU tensors must be contiguous"` | `KVFlow.get` | `KVShrinkConnector.start_load_kv` | 0.16 | 2.2% | 1 |
| `kvshrink/kvshrink_connector.py:717` | `layer_keys = [self._layer_key[layer_name]]` | `KVShrinkConnector.save_kv_layer` | `KVShrinkConnector.save_kv_layer` | 0.16 | 2.2% | 1 |
| `kvshrink/kvshrink_connector.py:701` | `def save_kv_layer(` | `KVShrinkConnector.save_kv_layer` | `KVShrinkConnector.save_kv_layer` | 0.16 | 2.2% | 1 |

列说明:“代码位置”是最里面的 kvshrink 帧的文件和行号,即那一刻正在执行的 kvshrink 代码;“源代码”是这一行的内容;“所在函数”是这一行所在的函数;“入口函数”同表 1;“毫秒/请求”是这一行的 Python 开销;“占 Python 开销的比例”是这一行 ÷ 主线程上 kvshrink 的Python 开销合计;“样本数”是这一行的样本数。

## 附 1:算作 IAXL C++ 的代码行(前 15 个,核对归类用)

| 代码位置 | 源代码 | 所在函数 | 入口函数 | 毫秒/请求 | 占 IAXL C++ 的比例 | 样本数 |
|---|---|---|---|---|---|---|
| `iaxl/kvflow/flow.py:458` | `result.ctx.xfer_wait()` | `KVFlow.get_wait` | `KVShrinkConnector.wait_for_layer_load` | 4.53 | 50.0% | 29 |
| `iaxl/kvflow/flow.py:387` | `ctx.unzip_from_mem(` | `KVFlow.get` | `KVShrinkConnector.start_load_kv` | 3.44 | 37.9% | 22 |
| `iaxl/kvflow/flow.py:457` | `result.ctx.unzip_wait()` | `KVFlow.get_wait` | `KVShrinkConnector.wait_for_layer_load` | 0.31 | 3.4% | 2 |
| `iaxl/kvflow/flow.py:484` | `return self.record.has(label, chunk_labels)` | `KVFlow.has` | `KVShrinkConnector.get_num_new_matched_tokens` | 0.31 | 3.4% | 2 |
| `iaxl/kvflow/flow.py:439` | `if not result.ctx.unzip_is_complete():` | `KVFlow.get_wait` | `KVShrinkConnector.get_finished` | 0.16 | 1.7% | 1 |
| `iaxl/kvflow/flow.py:445` | `if not result.ctx.xfer_is_complete():` | `KVFlow.get_wait` | `KVShrinkConnector.get_finished` | 0.16 | 1.7% | 1 |
| `iaxl/kvflow/flow.py:185` | `return Context.create(tensor, chunk_dim, direction, description, work_stream=work_stream)` | `KVFlow._create_ctx` | `KVShrinkConnector.start_load_kv` | 0.16 | 1.7% | 1 |

列说明:同表 2,只是这里列的是被算作 IAXL C++ 的行(调用 IAXL 扩展的行);“占 IAXL C++ 的比例”是这一行 ÷ 主线程上 IAXL C++ 的合计。

## 附 2:各线程的样本数

| 线程 | 样本数 |
|---|---|
| MainThread | 1246 |
| thread (61197) | 63 |
| thread (0) | 8 |
| Thread-2 (process_input_sockets) | 6 |
| Thread-3 (process_output_sockets) | 6 |
| thread (66504) | 1 |

列说明:“线程”是 py-spy 记录的线程名;“样本数”是这个线程在忙(没睡眠、没阻塞)时被采到的次数。
