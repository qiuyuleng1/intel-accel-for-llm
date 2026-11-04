# py-spy 结果:2k_c16,miss

## 这次运行是什么

- vLLM 0.29.0 + kvshrink,模型 `/media/model-space/Qwen3-32B`,TP=1,GPU 5,vLLM 进程绑定 CPU 64-127。调度、模型计算和 kvshrink 的全部函数都在 EngineCore 进程的主线程上。
- 一次 bench = `vllm bench serve` 发 800 个请求,输入 2048 token、输出 128 token,并发 16。
- 场景 miss:每个请求的 prompt 都是新的,kvshrink 只保存、不加载。
- py-spy 从测量 bench 开始前采到结束后,采样率 100 次/秒,只采 EngineCore 进程(pid 149866),参数 `--rate 100 --threads --format raw --full-filenames --nonblocking`:不暂停被测进程,不记录 C/C++ 函数,在 C/C++ 里花的时间算在调用它的那一行 Python 上。一个样本约代表 10 ms;“毫秒/请求”= 样本数 ÷ 采样率 × 1000 ÷ 请求数。相对误差约 1/√样本数。
- 代码 git 28f2d3e,代码 md5 27e553aa,py-spy py-spy 0.4.3。

## 核对

| 核对项 | 结果 |
|---|---|
| baseline 这次 bench 的请求数(connector 日志) | 800 |
| baseline 里有外部命中(externally-cached tokens > 0)的请求数 | 0 |
| baseline 里 externally-cached tokens 的不同取值(前 5 个) | 0 |
| pyspy 这次 bench 的请求数(connector 日志) | 800 |
| pyspy 里有外部命中(externally-cached tokens > 0)的请求数 | 2 |
| pyspy 里 externally-cached tokens 的不同取值(前 5 个) | 0, 2032 |
| vLLM 日志里 `ScratchPool exhausted` 的次数 | 0 |
| py-spy 跟不上采样率的警告次数 | 0 |
| py-spy 自报的样本数 / 错误数 | 90159 / 3951 |
| 主线程样本里调用链不完整的样本数(占比) | 1535(1.7%) |

列说明:“核对项”是检查什么;“结果”是从 connector 日志、vLLM 日志或 py-spy 输出数出来的值。miss 场景外部命中的请求数应为 0;hit 场景应等于请求数。“调用链不完整”指最外层不是进程入口 `<module>` 的样本:py-spy 不暂停进程时,偶尔读到一半就变了的调用链。

### 有没有 py-spy 的 bench 对比(干扰检查)

| 指标 | 没有 py-spy | 有 py-spy | 有 / 没有 |
|---|---|---|---|
| duration | 1158.82 | 1155.92 | 0.997 |
| completed | 800.00 | 800.00 | 1.000 |
| request_throughput | 0.69 | 0.69 | 1.003 |
| mean_ttft_ms | 7692.23 | 7215.41 | 0.938 |
| median_ttft_ms | 8679.48 | 7740.27 | 0.892 |
| mean_tpot_ms | 121.90 | 125.19 | 1.027 |
| median_tpot_ms | 114.20 | 121.38 | 1.063 |
| mean_e2el_ms | 23173.08 | 23114.56 | 0.997 |
| median_e2el_ms | 23186.59 | 23162.58 | 0.999 |

列说明:“指标”是 `vllm bench serve` 结果里的字段(duration 单位秒,throughput 单位请求/秒,其余单位 ms,mean 是均值、median 是中位数,统计对象是这次 bench 的全部请求);后两列是两次 bench 的值;“有 / 没有”是比值,越接近 1 说明 py-spy 干扰越小。

### py-spy 数的主线程忙碌时间对不对(和内核记的 CPU 时间比)

| 量 | 没有 py-spy 的 bench | 有 py-spy 的 bench |
|---|---|---|
| EngineCore 主线程 CPU 时间(内核) | 1187.14 | 1163.88 |
| EngineCore 全进程 CPU 时间(内核) | 5082.93 | 5026.23 |
| py-spy 数的主线程忙碌时间 | - | 1122.26 |
| 校准系数 = 内核 ÷ py-spy(主线程) | - | 1.037 |

列说明:单位都是毫秒/请求(校准系数除外)。“内核”两行是 /proc 里 EngineCore 的 utime + stime(内核记账的、真正占用 CPU 的时间),在每次 bench 前后各读一次,差值 ÷ 请求数;主线程的线程号就是进程号,全进程包括IAXL、torch 等全部线程。“py-spy 数的主线程忙碌时间”= 主线程被判为忙的样本数 ÷ 采样率 ÷ 请求数。py-spy 采样窗口比 bench 略长(前后各多几秒,那时 EngineCore 没有请求),所以它和同一次 bench 的“主线程 CPU 时间(内核)”应基本相等;差很多说明 py-spy 漏采或多采。“校准系数”= 有 py-spy 的 bench 的“主线程 CPU 时间(内核)”÷“py-spy 数的主线程忙碌时间”,表 1 用它把 py-spy 直接数的主线程时间换算成内核口径。

## 表 1:按入口函数汇总(毫秒/请求)

| 入口函数 | Python 开销(py-spy 直接数) | Python 开销(按内核校准) | 对照:ScratchPool | 对照:IAXL C++(含参数转换) | 样本数 |
|---|---|---|---|---|---|
| `KVShrinkConnector.save_kv_layer` | 5.34 | 5.54 | 0.34 | 951.86 | 76603 |
| `KVShrinkConnector.get_num_new_matched_tokens` | 0.40 | 0.41 | 0.00 | 0.53 | 74 |
| `KVShrinkConnector.get_finished` | 0.33 | 0.34 | 0.57 | 0.93 | 146 |
| `KVShrinkConnector.wait_for_layer_load` | 0.23 | 0.23 | 0.00 | 0.00 | 18 |
| `KVShrinkConnector.build_connector_meta` | 0.10 | 0.10 | 0.00 | 0.00 | 8 |
| `KVStoreLocal.put` | 0.06 | 0.06 | 0.00 | 0.38 | 35 |
| `ScratchPool.release.<locals>.<genexpr>` | 0.00 | 0.00 | 0.25 | 0.00 | 20 |
| 主线程上 kvshrink 合计 | 6.45 | 6.69 | 1.16 | 953.69 | 76904 |
| 其他线程上的 kvshrink | 0.46 | - | 0.00 | 0.99 | 116 |

主线程 CPU 忙碌总时间(含 vLLM 自己的代码):py-spy 直接数 1122.26 毫秒/请求(89781 个样本),内核记 1163.88 毫秒/请求;kvshrink 的 Python 开销占 py-spy 主线程忙碌样本的 0.6%。

列说明:“入口函数”是调用链里最外面的 kvshrink 函数,即 vLLM 调用的那个;“Python 开销”是本次要量的量:kvshrink 的 Python 语句,加上它们直接调用的 Python 代码和xxhash、numpy、torch 等库的 C 代码;“py-spy 直接数”= 样本数 ÷ 采样率 ÷ 请求数;“按内核校准”= 直接数的值 × 校准系数(见上一节),假设 py-spy 漏掉的样本均匀分布在各段代码上,其他线程不校准(校准系数只对主线程测过);两个“对照”列不算 Python 开销,是 py-spy 直接数的值:ScratchPool 取块还块归同事,IAXL C++ 是调用 IAXL 扩展的那些行上的时间(C++ 执行、pybind11 转换参数、空转等待,哪些行见附 1);以上单位都是毫秒/请求;“样本数”是这一行三类的样本总数。“其他线程上的 kvshrink”是管理接口、内存池淘汰等后台线程。

## 表 2:Python 开销最多的前 20 个代码位置

| 代码位置 | 源代码 | 所在函数 | 入口函数 | 毫秒/请求 | 占 Python 开销的比例 | 样本数 |
|---|---|---|---|---|---|---|
| `kvshrink/kvshrink_connector.py:718` | `for req_id, request in metadata.reqs_to_save.requests.items():` | `KVShrinkConnector.save_kv_layer` | `KVShrinkConnector.save_kv_layer` | 2.58 | 39.9% | 206 |
| `kvshrink/kvshrink_connector.py:72` | `return [p[0] for p in pairs], [p[1] for p in pairs]` | `ReqMeta.for_group` | `KVShrinkConnector.save_kv_layer` | 0.50 | 7.8% | 40 |
| `kvshrink/kvshrink_connector.py:719` | `block_ids, block_hashes = request.for_group(group_idx)` | `KVShrinkConnector.save_kv_layer` | `KVShrinkConnector.save_kv_layer` | 0.34 | 5.2% | 27 |
| `kvshrink/kvshrink_connector.py:722` | `tasks = self._store().put(` | `KVShrinkConnector.save_kv_layer` | `KVShrinkConnector.save_kv_layer` | 0.20 | 3.1% | 16 |
| `kvshrink/kvshrink_connector.py:65` | `pairs = [` | `ReqMeta.for_group` | `KVShrinkConnector.save_kv_layer` | 0.17 | 2.7% | 14 |
| `iaxl/kvflow/flow.py:308` | `self.chunk_pool.release(result.cpu_tensors)` | `KVFlow.put_wait` | `KVShrinkConnector.get_finished` | 0.17 | 2.7% | 14 |
| `iaxl/kvflow/flow.py:267` | `results[tensor_key] = task = Task(` | `KVFlow.put` | `KVShrinkConnector.save_kv_layer` | 0.16 | 2.5% | 13 |
| `kvshrink/kvshrink_connector.py:709` | `if self._connector_metadata is None or layer_name not in self.layer_group:` | `KVShrinkConnector.save_kv_layer` | `KVShrinkConnector.save_kv_layer` | 0.14 | 2.1% | 11 |
| `kvshrink/kvshrink_connector.py:66` | `(block_id, block_hash)` | `ReqMeta.for_group` | `KVShrinkConnector.save_kv_layer` | 0.14 | 2.1% | 11 |
| `iaxl/kvflow/flow.py:245` | `cpu_tensors = self._allocate(num_chunks, chunk_shape, tensor.dtype)` | `KVFlow.put` | `KVShrinkConnector.save_kv_layer` | 0.12 | 1.9% | 10 |
| `iaxl/utils/hash.py:13` | `parts = [repr(arg) for arg in args]` | `compute_hash` | `KVShrinkConnector.get_num_new_matched_tokens` | 0.12 | 1.9% | 10 |
| `iaxl/kvstore/kvstore.py:240` | `result = self.tensorzip.put(` | `KVStoreLocal.put` | `KVShrinkConnector.save_kv_layer` | 0.12 | 1.9% | 10 |
| `kvshrink/kvshrink_connector.py:670` | `return` | `KVShrinkConnector.wait_for_layer_load` | `KVShrinkConnector.wait_for_layer_load` | 0.11 | 1.7% | 9 |
| `kvshrink/kvshrink_connector.py:669` | `if not self._current_get_tasks and not self._active_promoted_tasks:` | `KVShrinkConnector.wait_for_layer_load` | `KVShrinkConnector.wait_for_layer_load` | 0.11 | 1.7% | 9 |
| `iaxl/kvflow/flow.py:233` | `chunk_shape = list(first_t.shape)` | `KVFlow.put` | `KVShrinkConnector.save_kv_layer` | 0.09 | 1.4% | 7 |
| `kvshrink/kvshrink_connector.py:717` | `layer_keys = [self._layer_key[layer_name]]` | `KVShrinkConnector.save_kv_layer` | `KVShrinkConnector.save_kv_layer` | 0.09 | 1.4% | 7 |
| `iaxl/kvflow/flow.py:275` | `self._pending_puts[id(task)] = task` | `KVFlow.put` | `KVShrinkConnector.save_kv_layer` | 0.09 | 1.4% | 7 |
| `kvshrink/kvshrink_connector.py:712` | `metadata = self._get_connector_metadata()` | `KVShrinkConnector.save_kv_layer` | `KVShrinkConnector.save_kv_layer` | 0.06 | 1.0% | 5 |
| `iaxl/kvflow/flow.py:223` | `assert tensor.device == first_t.device, (` | `KVFlow.put` | `KVShrinkConnector.save_kv_layer` | 0.05 | 0.8% | 4 |
| `kvshrink/kvshrink_connector.py:716` | `group_idx = self.layer_group[layer_name]` | `KVShrinkConnector.save_kv_layer` | `KVShrinkConnector.save_kv_layer` | 0.05 | 0.8% | 4 |

列说明:“代码位置”是最里面的 kvshrink 帧的文件和行号,即那一刻正在执行的 kvshrink 代码;“源代码”是这一行的内容;“所在函数”是这一行所在的函数;“入口函数”同表 1;“毫秒/请求”是这一行的 Python 开销;“占 Python 开销的比例”是这一行 ÷ 主线程上 kvshrink 的Python 开销合计;“样本数”是这一行的样本数。

## 附 1:算作 IAXL C++ 的代码行(前 15 个,核对归类用)

| 代码位置 | 源代码 | 所在函数 | 入口函数 | 毫秒/请求 | 占 IAXL C++ 的比例 | 样本数 |
|---|---|---|---|---|---|---|
| `iaxl/kvflow/flow.py:256` | `ctx.xfer_wait_cur_stream(sync_cur_stream=True)` | `KVFlow.put` | `KVShrinkConnector.save_kv_layer` | 947.86 | 99.4% | 75829 |
| `iaxl/kvflow/flow.py:263` | `ctx.zip_to_mem(` | `KVFlow.put` | `KVShrinkConnector.save_kv_layer` | 1.57 | 0.2% | 126 |
| `iaxl/kvflow/flow.py:259` | `ctx.xfer_chunks_batch(chunk_indices, cpu_tensors)` | `KVFlow.put` | `KVShrinkConnector.save_kv_layer` | 1.31 | 0.1% | 105 |
| `iaxl/kvflow/flow.py:304` | `result.ctx.zip_wait()` | `KVFlow.put_wait` | `KVShrinkConnector.get_finished` | 0.91 | 0.1% | 73 |
| `iaxl/kvflow/flow.py:472` | `self.record.sync()` | `KVFlow.record_flush` | `KVShrinkConnector.save_kv_layer` | 0.60 | 0.1% | 48 |
| `iaxl/kvflow/flow.py:484` | `return self.record.has(label, chunk_labels)` | `KVFlow.has` | `KVShrinkConnector.get_num_new_matched_tokens` | 0.51 | 0.1% | 41 |
| `iaxl/kvflow/flow.py:185` | `return Context.create(tensor, chunk_dim, direction, description, work_stream=work_stream)` | `KVFlow._create_ctx` | `KVShrinkConnector.save_kv_layer` | 0.45 | 0.0% | 36 |
| `iaxl/kvflow/flow.py:263` | `ctx.zip_to_mem(` | `KVFlow.put` | `KVStoreLocal.put` | 0.38 | 0.0% | 30 |
| `iaxl/kvflow/flow.py:260` | `ctx.xfer_finish()` | `KVFlow.put` | `KVShrinkConnector.save_kv_layer` | 0.06 | 0.0% | 5 |
| `iaxl/kvstore/kvstore.py:351` | `self.tensorzip.record.touch(label, block_hashs)` | `KVStoreLocal.has` | `KVShrinkConnector.get_num_new_matched_tokens` | 0.01 | 0.0% | 1 |
| `iaxl/kvflow/flow.py:295` | `if result.ctx is not None and not result.ctx.zip_is_complete():` | `KVFlow.put_wait` | `KVShrinkConnector.get_finished` | 0.01 | 0.0% | 1 |

列说明:同表 2,只是这里列的是被算作 IAXL C++ 的行(调用 IAXL 扩展的行);“占 IAXL C++ 的比例”是这一行 ÷ 主线程上 IAXL C++ 的合计。

## 附 2:各线程的样本数

| 线程 | 样本数 |
|---|---|
| MainThread | 89781 |
| Thread-3 (process_output_sockets) | 141 |
| Thread-2 (process_input_sockets) | 69 |
| iaxl-monitor | 59 |
| Thread-225 (process_request_thread) | 36 |
| mgmt-ctrl_0 | 21 |
| mgmt-worker-0 | 18 |
| mgmt-controller | 12 |
| Thread-1 (_report_usage_worker) | 10 |
| Thread-215 (process_request_thread) | 4 |
| tqdm_monitor | 3 |
| thread (173646) | 1 |
| thread (172224) | 1 |
| thread (173570) | 1 |
| thread (153534) | 1 |
| Thread-212 (process_request_thread) | 1 |

列说明:“线程”是 py-spy 记录的线程名;“样本数”是这个线程在忙(没睡眠、没阻塞)时被采到的次数。
