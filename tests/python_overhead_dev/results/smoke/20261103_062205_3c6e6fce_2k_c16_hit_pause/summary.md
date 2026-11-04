# py-spy 结果:2k_c16,hit

## 这次运行是什么

- vLLM 0.29.0 + kvshrink,模型 `/media/model-space/Qwen3-32B`,TP=1,GPU 5,vLLM 进程绑定 CPU 64-127。调度、模型计算和 kvshrink 的全部函数都在 EngineCore 进程的主线程上。
- 一次 bench = `vllm bench serve` 发 64 个请求,输入 2048 token、输出 128 token,并发 16。
- 场景 hit:所有请求的 prompt 相同,测量前已存进 DDR,kvshrink 只加载、不保存。
- py-spy 从测量 bench 开始前采到结束后,采样率 100 次/秒,只采 EngineCore 进程(pid 51739),参数 `--rate 100 --threads --format raw --full-filenames`:每次采样把被测进程暂停一下再读调用栈,不记录 C/C++ 函数,在 C/C++ 里花的时间算在调用它的那一行 Python 上。一个样本约代表 10 ms;“毫秒/请求”= 样本数 ÷ 采样率 × 1000 ÷ 请求数。相对误差约 1/√样本数。
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
| py-spy 自报的样本数 / 错误数 | 3528 / 1 |
| 主线程样本里调用链不完整的样本数(占比) | 1(0.0%) |

列说明:“核对项”是检查什么;“结果”是从 connector 日志、vLLM 日志或 py-spy 输出数出来的值。miss 场景外部命中的请求数应为 0;hit 场景应等于请求数。“调用链不完整”指最外层不是进程入口 `<module>` 的样本:py-spy 不暂停进程时,偶尔读到一半就变了的调用链。

### 有没有 py-spy 的 bench 对比(干扰检查)

| 指标 | 没有 py-spy | 有 py-spy | 有 / 没有 |
|---|---|---|---|
| duration | 33.49 | 35.89 | 1.072 |
| completed | 64.00 | 64.00 | 1.000 |
| request_throughput | 1.91 | 1.78 | 0.933 |
| mean_ttft_ms | 480.34 | 729.85 | 1.519 |
| median_ttft_ms | 420.68 | 744.51 | 1.770 |
| mean_tpot_ms | 61.54 | 64.56 | 1.049 |
| median_tpot_ms | 61.52 | 64.43 | 1.047 |
| mean_e2el_ms | 8295.30 | 8928.75 | 1.076 |
| median_e2el_ms | 8252.87 | 8913.41 | 1.080 |

列说明:“指标”是 `vllm bench serve` 结果里的字段(duration 单位秒,throughput 单位请求/秒,其余单位 ms,mean 是均值、median 是中位数,统计对象是这次 bench 的全部请求);后两列是两次 bench 的值;“有 / 没有”是比值,越接近 1 说明 py-spy 干扰越小。

### py-spy 数的主线程忙碌时间对不对(和内核记的 CPU 时间比)

| 量 | 没有 py-spy 的 bench | 有 py-spy 的 bench |
|---|---|---|
| EngineCore 主线程 CPU 时间(内核) | 274.38 | 319.84 |
| EngineCore 全进程 CPU 时间(内核) | 554.38 | 756.56 |
| py-spy 数的主线程忙碌时间 | - | 509.84 |

列说明:单位都是毫秒/请求。“内核”两行是 /proc 里 EngineCore 的 utime + stime(内核记账的、真正占用 CPU 的时间),在每次 bench 前后各读一次,差值 ÷ 请求数;主线程的线程号就是进程号,全进程包括IAXL、torch 等全部线程。“py-spy 数的主线程忙碌时间”= 主线程被判为忙的样本数 ÷ 采样率 ÷ 请求数。py-spy 采样窗口比 bench 略长(前后各多几秒,那时 EngineCore 没有请求),所以它和同一次 bench 的“主线程 CPU 时间(内核)”应基本相等;差很多说明 py-spy 漏采或多采。

## 表 1:按入口函数汇总(毫秒/请求)

| 入口函数 | Python 开销 | 对照:ScratchPool | 对照:IAXL C++(含参数转换) | 样本数 |
|---|---|---|---|---|
| `KVShrinkConnector.save_kv_layer` | 3.91 | 0.00 | 0.00 | 25 |
| `KVShrinkConnector.get_num_new_matched_tokens` | 3.28 | 0.00 | 0.94 | 27 |
| `KVShrinkConnector.start_load_kv` | 1.88 | 1.25 | 7.19 | 66 |
| `KVShrinkConnector.build_connector_meta` | 0.94 | 0.00 | 0.00 | 6 |
| `KVShrinkConnector.wait_for_layer_load` | 0.78 | 1.72 | 10.31 | 82 |
| `KVShrinkConnector.get_finished` | 0.47 | 1.41 | 0.94 | 18 |
| `KVShrinkConnector.update_state_after_alloc` | 0.16 | 0.00 | 0.00 | 1 |
| 主线程上 kvshrink 合计 | 11.41 | 4.38 | 19.38 | 225 |
| 其他线程上的 kvshrink | 0.94 | 0.00 | 0.00 | 6 |

主线程 CPU 忙碌总时间(含 vLLM 自己的代码):509.84 毫秒/请求(3263 个样本);kvshrink 的 Python 开销占其中 2.2%。

列说明:“入口函数”是调用链里最外面的 kvshrink 函数,即 vLLM 调用的那个;“Python 开销”是本次要量的量:kvshrink 的 Python 语句,加上它们直接调用的 Python 代码和xxhash、numpy、torch 等库的 C 代码;两个“对照”列不算 Python 开销:ScratchPool 取块还块归同事,IAXL C++ 是调用 IAXL 扩展的那些行上的时间(C++ 执行、pybind11 转换参数、空转等待,哪些行见附 1);以上单位都是毫秒/请求;“样本数”是这一行三类的样本总数。“其他线程上的 kvshrink”是管理接口、内存池淘汰等后台线程。

## 表 2:Python 开销最多的前 20 个代码位置

| 代码位置 | 源代码 | 所在函数 | 入口函数 | 毫秒/请求 | 占 Python 开销的比例 | 样本数 |
|---|---|---|---|---|---|---|
| `kvshrink/kvshrink_connector.py:718` | `for req_id, request in metadata.reqs_to_save.requests.items():` | `KVShrinkConnector.save_kv_layer` | `KVShrinkConnector.save_kv_layer` | 1.56 | 13.7% | 10 |
| `iaxl/utils/hash.py:13` | `parts = [repr(arg) for arg in args]` | `compute_hash` | `KVShrinkConnector.get_num_new_matched_tokens` | 1.25 | 11.0% | 8 |
| `kvshrink/kvshrink_connector.py:712` | `metadata = self._get_connector_metadata()` | `KVShrinkConnector.save_kv_layer` | `KVShrinkConnector.save_kv_layer` | 1.09 | 9.6% | 7 |
| `iaxl/kvflow/flow.py:407` | `self._pending_gets[id(task)] = task` | `KVFlow.get` | `KVShrinkConnector.start_load_kv` | 0.47 | 4.1% | 3 |
| `iaxl/kvflow/flow.py:460` | `result.ctx = None` | `KVFlow.get_wait` | `KVShrinkConnector.wait_for_layer_load` | 0.47 | 4.1% | 3 |
| `kvshrink/kvshrink_connector.py:713` | `if not isinstance(metadata, KVShrinkConnectorMetadata):` | `KVShrinkConnector.save_kv_layer` | `KVShrinkConnector.save_kv_layer` | 0.47 | 4.1% | 3 |
| `iaxl/kvflow/flow.py:399` | `results[tensor_key] = task = Task(` | `KVFlow.get` | `KVShrinkConnector.start_load_kv` | 0.47 | 4.1% | 3 |
| `kvshrink/kvshrink_connector.py:336` | `logger.info(` | `KVShrinkConnector.get_num_new_matched_tokens` | `KVShrinkConnector.get_num_new_matched_tokens` | 0.47 | 4.1% | 3 |
| `kvshrink/kvshrink_connector.py:716` | `group_idx = self.layer_group[layer_name]` | `KVShrinkConnector.save_kv_layer` | `KVShrinkConnector.save_kv_layer` | 0.31 | 2.7% | 2 |
| `kvshrink/kvshrink_connector.py:717` | `layer_keys = [self._layer_key[layer_name]]` | `KVShrinkConnector.save_kv_layer` | `KVShrinkConnector.save_kv_layer` | 0.31 | 2.7% | 2 |
| `iaxl/kvflow/flow.py:343` | `assert isinstance(tensor, RemoteTensor) or tensor.device.type == self.device_type, (` | `KVFlow.get` | `KVShrinkConnector.start_load_kv` | 0.31 | 2.7% | 2 |
| `iaxl/utils/hash.py:14` | `parts += [f"{k}={repr(v)}" for k, v in sorted(kwargs.items())]` | `compute_hash` | `KVShrinkConnector.get_num_new_matched_tokens` | 0.31 | 2.7% | 2 |
| `iaxl/utils/hash.py:37` | `cur_hash = hash_block_tokens(` | `generate_block_hashs` | `KVShrinkConnector.get_num_new_matched_tokens` | 0.16 | 1.4% | 1 |
| `iaxl/utils/hash.py:24` | `return compute_hash(is_first_block, prev_block_hash, cur_block_token_ids)` | `hash_block_tokens` | `KVShrinkConnector.get_num_new_matched_tokens` | 0.16 | 1.4% | 1 |
| `kvshrink/kvshrink_connector.py:773` | `self._deferred_finished_req_ids.update(finished_req_ids)` | `KVShrinkConnector.get_finished` | `KVShrinkConnector.get_finished` | 0.16 | 1.4% | 1 |
| `iaxl/kvflow/flow.py:352` | `if priorities is not None and tensor_key not in priorities:` | `KVFlow.get` | `KVShrinkConnector.start_load_kv` | 0.16 | 1.4% | 1 |
| `kvshrink/kvshrink_connector.py:444` | `state = self._req_states[req_id]` | `KVShrinkConnector.build_connector_meta` | `KVShrinkConnector.build_connector_meta` | 0.16 | 1.4% | 1 |
| `iaxl/kvflow/flow.py:460` | `result.ctx = None` | `KVFlow.get_wait` | `KVShrinkConnector.get_finished` | 0.16 | 1.4% | 1 |
| `kvshrink/kvshrink_connector.py:297` | `(` | `KVShrinkConnector.get_num_new_matched_tokens` | `KVShrinkConnector.get_num_new_matched_tokens` | 0.16 | 1.4% | 1 |
| `kvshrink/kvshrink_connector.py:448` | `ids.extend(new_ids)` | `KVShrinkConnector.build_connector_meta` | `KVShrinkConnector.build_connector_meta` | 0.16 | 1.4% | 1 |

列说明:“代码位置”是最里面的 kvshrink 帧的文件和行号,即那一刻正在执行的 kvshrink 代码;“源代码”是这一行的内容;“所在函数”是这一行所在的函数;“入口函数”同表 1;“毫秒/请求”是这一行的 Python 开销;“占 Python 开销的比例”是这一行 ÷ 主线程上 kvshrink 的Python 开销合计;“样本数”是这一行的样本数。

## 附 1:算作 IAXL C++ 的代码行(前 15 个,核对归类用)

| 代码位置 | 源代码 | 所在函数 | 入口函数 | 毫秒/请求 | 占 IAXL C++ 的比例 | 样本数 |
|---|---|---|---|---|---|---|
| `iaxl/kvflow/flow.py:458` | `result.ctx.xfer_wait()` | `KVFlow.get_wait` | `KVShrinkConnector.wait_for_layer_load` | 10.16 | 52.4% | 65 |
| `iaxl/kvflow/flow.py:387` | `ctx.unzip_from_mem(` | `KVFlow.get` | `KVShrinkConnector.start_load_kv` | 6.88 | 35.5% | 44 |
| `iaxl/kvflow/flow.py:484` | `return self.record.has(label, chunk_labels)` | `KVFlow.has` | `KVShrinkConnector.get_num_new_matched_tokens` | 0.78 | 4.0% | 5 |
| `iaxl/kvflow/flow.py:458` | `result.ctx.xfer_wait()` | `KVFlow.get_wait` | `KVShrinkConnector.get_finished` | 0.62 | 3.2% | 4 |
| `iaxl/kvflow/flow.py:185` | `return Context.create(tensor, chunk_dim, direction, description, work_stream=work_stream)` | `KVFlow._create_ctx` | `KVShrinkConnector.start_load_kv` | 0.31 | 1.6% | 2 |
| `iaxl/kvstore/kvstore.py:351` | `self.tensorzip.record.touch(label, block_hashs)` | `KVStoreLocal.has` | `KVShrinkConnector.get_num_new_matched_tokens` | 0.16 | 0.8% | 1 |
| `iaxl/kvflow/flow.py:439` | `if not result.ctx.unzip_is_complete():` | `KVFlow.get_wait` | `KVShrinkConnector.get_finished` | 0.16 | 0.8% | 1 |
| `iaxl/kvflow/flow.py:457` | `result.ctx.unzip_wait()` | `KVFlow.get_wait` | `KVShrinkConnector.wait_for_layer_load` | 0.16 | 0.8% | 1 |
| `iaxl/kvflow/flow.py:445` | `if not result.ctx.xfer_is_complete():` | `KVFlow.get_wait` | `KVShrinkConnector.get_finished` | 0.16 | 0.8% | 1 |

列说明:同表 2,只是这里列的是被算作 IAXL C++ 的行(调用 IAXL 扩展的行);“占 IAXL C++ 的比例”是这一行 ÷ 主线程上 IAXL C++ 的合计。

## 附 2:各线程的样本数

| 线程 | 样本数 |
|---|---|
| MainThread | 3263 |
| thread (53277) | 147 |
| thread (0) | 76 |
| Thread-3 (process_output_sockets) | 20 |
| Thread-2 (process_input_sockets) | 12 |
| mgmt-worker-0 | 3 |
| iaxl-monitor | 2 |
| Thread-22 (process_request_thread) | 1 |
| Thread-24 (process_request_thread) | 1 |
| Thread-25 (process_request_thread) | 1 |
| Thread-23 (process_request_thread) | 1 |
| mgmt-ctrl_0 | 1 |

列说明:“线程”是 py-spy 记录的线程名;“样本数”是这个线程在忙(没睡眠、没阻塞)时被采到的次数。
