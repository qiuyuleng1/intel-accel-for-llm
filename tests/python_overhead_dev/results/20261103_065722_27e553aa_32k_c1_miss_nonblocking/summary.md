# py-spy 结果:32k_c1,miss

## 这次运行是什么

- vLLM 0.29.0 + kvshrink,模型 `/media/model-space/Qwen3-32B`,TP=1,GPU 5,vLLM 进程绑定 CPU 64-127。调度、模型计算和 kvshrink 的全部函数都在 EngineCore 进程的主线程上。
- 一次 bench = `vllm bench serve` 发 50 个请求,输入 32768 token、输出 128 token,并发 1。
- 场景 miss:每个请求的 prompt 都是新的,kvshrink 只保存、不加载。
- py-spy 从测量 bench 开始前采到结束后,采样率 100 次/秒,只采 EngineCore 进程(pid 83572),参数 `--rate 100 --threads --format raw --full-filenames --nonblocking`:不暂停被测进程,不记录 C/C++ 函数,在 C/C++ 里花的时间算在调用它的那一行 Python 上。一个样本约代表 10 ms;“毫秒/请求”= 样本数 ÷ 采样率 × 1000 ÷ 请求数。相对误差约 1/√样本数。
- 代码 git 28f2d3e,代码 md5 27e553aa,py-spy py-spy 0.4.3。

## 核对

| 核对项 | 结果 |
|---|---|
| baseline 这次 bench 的请求数(connector 日志) | 50 |
| baseline 里有外部命中(externally-cached tokens > 0)的请求数 | 0 |
| baseline 里 externally-cached tokens 的不同取值(前 5 个) | 0 |
| pyspy 这次 bench 的请求数(connector 日志) | 50 |
| pyspy 里有外部命中(externally-cached tokens > 0)的请求数 | 0 |
| pyspy 里 externally-cached tokens 的不同取值(前 5 个) | 0 |
| vLLM 日志里 `ScratchPool exhausted` 的次数 | 0 |
| py-spy 跟不上采样率的警告次数 | 0 |
| py-spy 自报的样本数 / 错误数 | 126524 / 4644 |
| 主线程样本里调用链不完整的样本数(占比) | 1863(1.5%) |

列说明:“核对项”是检查什么;“结果”是从 connector 日志、vLLM 日志或 py-spy 输出数出来的值。miss 场景外部命中的请求数应为 0;hit 场景应等于请求数。“调用链不完整”指最外层不是进程入口 `<module>` 的样本:py-spy 不暂停进程时,偶尔读到一半就变了的调用链。

### 有没有 py-spy 的 bench 对比(干扰检查)

| 指标 | 没有 py-spy | 有 py-spy | 有 / 没有 |
|---|---|---|---|
| duration | 1520.59 | 1520.93 | 1.000 |
| completed | 50.00 | 50.00 | 1.000 |
| request_throughput | 0.03 | 0.03 | 1.000 |
| mean_ttft_ms | 22773.18 | 22774.40 | 1.000 |
| median_ttft_ms | 22772.61 | 22774.54 | 1.000 |
| mean_tpot_ms | 60.14 | 60.19 | 1.001 |
| median_tpot_ms | 60.14 | 60.18 | 1.001 |
| mean_e2el_ms | 30411.11 | 30418.01 | 1.000 |
| median_e2el_ms | 30410.08 | 30418.43 | 1.000 |

列说明:“指标”是 `vllm bench serve` 结果里的字段(duration 单位秒,throughput 单位请求/秒,其余单位 ms,mean 是均值、median 是中位数,统计对象是这次 bench 的全部请求);后两列是两次 bench 的值;“有 / 没有”是比值,越接近 1 说明 py-spy 干扰越小。

### py-spy 数的主线程忙碌时间对不对(和内核记的 CPU 时间比)

| 量 | 没有 py-spy 的 bench | 有 py-spy 的 bench |
|---|---|---|
| EngineCore 主线程 CPU 时间(内核) | 26215.60 | 26197.20 |
| EngineCore 全进程 CPU 时间(内核) | 96043.20 | 96072.20 |
| py-spy 数的主线程忙碌时间 | - | 25238.80 |
| 校准系数 = 内核 ÷ py-spy(主线程) | - | 1.038 |

列说明:单位都是毫秒/请求(校准系数除外)。“内核”两行是 /proc 里 EngineCore 的 utime + stime(内核记账的、真正占用 CPU 的时间),在每次 bench 前后各读一次,差值 ÷ 请求数;主线程的线程号就是进程号,全进程包括IAXL、torch 等全部线程。“py-spy 数的主线程忙碌时间”= 主线程被判为忙的样本数 ÷ 采样率 ÷ 请求数。py-spy 采样窗口比 bench 略长(前后各多几秒,那时 EngineCore 没有请求),所以它和同一次 bench 的“主线程 CPU 时间(内核)”应基本相等;差很多说明 py-spy 漏采或多采。“校准系数”= 有 py-spy 的 bench 的“主线程 CPU 时间(内核)”÷“py-spy 数的主线程忙碌时间”,表 1 用它把 py-spy 直接数的主线程时间换算成内核口径。

## 表 1:按入口函数汇总(毫秒/请求)

| 入口函数 | Python 开销(py-spy 直接数) | Python 开销(按内核校准) | 对照:ScratchPool | 对照:IAXL C++(含参数转换) | 样本数 |
|---|---|---|---|---|---|
| `KVShrinkConnector.save_kv_layer` | 66.00 | 68.51 | 5.80 | 22252.00 | 111619 |
| `KVShrinkConnector.get_num_new_matched_tokens` | 9.20 | 9.55 | 0.00 | 6.80 | 80 |
| `KVShrinkConnector.wait_for_layer_load` | 4.00 | 4.15 | 0.00 | 0.00 | 20 |
| `KVShrinkConnector.get_finished` | 3.80 | 3.94 | 24.80 | 35.80 | 322 |
| `KVStoreLocal.put` | 2.20 | 2.28 | 0.00 | 1.80 | 20 |
| `KVShrinkConnector.build_connector_meta` | 0.20 | 0.21 | 0.00 | 0.00 | 1 |
| `ScratchPool.release.<locals>.<genexpr>` | 0.00 | 0.00 | 4.60 | 0.00 | 23 |
| 主线程上 kvshrink 合计 | 85.40 | 88.64 | 35.20 | 22296.40 | 112085 |
| 其他线程上的 kvshrink | 9.00 | - | 0.00 | 15.60 | 123 |

主线程 CPU 忙碌总时间(含 vLLM 自己的代码):py-spy 直接数 25238.80 毫秒/请求(126194 个样本),内核记 26197.20 毫秒/请求;kvshrink 的 Python 开销占 py-spy 主线程忙碌样本的 0.3%。

列说明:“入口函数”是调用链里最外面的 kvshrink 函数,即 vLLM 调用的那个;“Python 开销”是本次要量的量:kvshrink 的 Python 语句,加上它们直接调用的 Python 代码和xxhash、numpy、torch 等库的 C 代码;“py-spy 直接数”= 样本数 ÷ 采样率 ÷ 请求数;“按内核校准”= 直接数的值 × 校准系数(见上一节),假设 py-spy 漏掉的样本均匀分布在各段代码上,其他线程不校准(校准系数只对主线程测过);两个“对照”列不算 Python 开销,是 py-spy 直接数的值:ScratchPool 取块还块归同事,IAXL C++ 是调用 IAXL 扩展的那些行上的时间(C++ 执行、pybind11 转换参数、空转等待,哪些行见附 1);以上单位都是毫秒/请求;“样本数”是这一行三类的样本总数。“其他线程上的 kvshrink”是管理接口、内存池淘汰等后台线程。

## 表 2:Python 开销最多的前 20 个代码位置

| 代码位置 | 源代码 | 所在函数 | 入口函数 | 毫秒/请求 | 占 Python 开销的比例 | 样本数 |
|---|---|---|---|---|---|---|
| `kvshrink/kvshrink_connector.py:718` | `for req_id, request in metadata.reqs_to_save.requests.items():` | `KVShrinkConnector.save_kv_layer` | `KVShrinkConnector.save_kv_layer` | 37.80 | 44.3% | 189 |
| `kvshrink/kvshrink_connector.py:719` | `block_ids, block_hashes = request.for_group(group_idx)` | `KVShrinkConnector.save_kv_layer` | `KVShrinkConnector.save_kv_layer` | 4.80 | 5.6% | 24 |
| `kvshrink/kvshrink_connector.py:72` | `return [p[0] for p in pairs], [p[1] for p in pairs]` | `ReqMeta.for_group` | `KVShrinkConnector.save_kv_layer` | 3.60 | 4.2% | 18 |
| `kvshrink/kvshrink_connector.py:66` | `(block_id, block_hash)` | `ReqMeta.for_group` | `KVShrinkConnector.save_kv_layer` | 3.20 | 3.7% | 16 |
| `kvshrink/kvshrink_connector.py:709` | `if self._connector_metadata is None or layer_name not in self.layer_group:` | `KVShrinkConnector.save_kv_layer` | `KVShrinkConnector.save_kv_layer` | 2.80 | 3.3% | 14 |
| `iaxl/utils/hash.py:16` | `return xxhash.xxh64(text.encode("utf-8")).intdigest() & 0x7FFFFFFFFFFFFFFF` | `compute_hash` | `KVShrinkConnector.get_num_new_matched_tokens` | 2.80 | 3.3% | 14 |
| `kvshrink/kvshrink_connector.py:65` | `pairs = [` | `ReqMeta.for_group` | `KVShrinkConnector.save_kv_layer` | 2.60 | 3.0% | 13 |
| `kvshrink/kvshrink_connector.py:669` | `if not self._current_get_tasks and not self._active_promoted_tasks:` | `KVShrinkConnector.wait_for_layer_load` | `KVShrinkConnector.wait_for_layer_load` | 2.20 | 2.6% | 11 |
| `iaxl/utils/hash.py:14` | `parts += [f"{k}={repr(v)}" for k, v in sorted(kwargs.items())]` | `compute_hash` | `KVShrinkConnector.get_num_new_matched_tokens` | 1.80 | 2.1% | 9 |
| `kvshrink/kvshrink_connector.py:670` | `return` | `KVShrinkConnector.wait_for_layer_load` | `KVShrinkConnector.wait_for_layer_load` | 1.60 | 1.9% | 8 |
| `kvshrink/kvshrink_connector.py:717` | `layer_keys = [self._layer_key[layer_name]]` | `KVShrinkConnector.save_kv_layer` | `KVShrinkConnector.save_kv_layer` | 1.60 | 1.9% | 8 |
| `kvshrink/kvshrink_connector.py:713` | `if not isinstance(metadata, KVShrinkConnectorMetadata):` | `KVShrinkConnector.save_kv_layer` | `KVShrinkConnector.save_kv_layer` | 1.40 | 1.6% | 7 |
| `kvshrink/kvshrink_connector.py:712` | `metadata = self._get_connector_metadata()` | `KVShrinkConnector.save_kv_layer` | `KVShrinkConnector.save_kv_layer` | 1.40 | 1.6% | 7 |
| `iaxl/utils/hash.py:13` | `parts = [repr(arg) for arg in args]` | `compute_hash` | `KVShrinkConnector.get_num_new_matched_tokens` | 1.40 | 1.6% | 7 |
| `iaxl/kvflow/flow.py:277` | `return results` | `KVFlow.put` | `KVStoreLocal.put` | 1.20 | 1.4% | 6 |
| `kvshrink/kvshrink_connector.py:67` | `for block_id, block_hash in zip(` | `ReqMeta.for_group` | `KVShrinkConnector.save_kv_layer` | 1.20 | 1.4% | 6 |
| `iaxl/kvflow/flow.py:305` | `result.ctx = None` | `KVFlow.put_wait` | `KVShrinkConnector.get_finished` | 1.00 | 1.2% | 5 |
| `iaxl/utils/hash.py:37` | `cur_hash = hash_block_tokens(` | `generate_block_hashs` | `KVShrinkConnector.get_num_new_matched_tokens` | 1.00 | 1.2% | 5 |
| `iaxl/kvflow/flow.py:245` | `cpu_tensors = self._allocate(num_chunks, chunk_shape, tensor.dtype)` | `KVFlow.put` | `KVShrinkConnector.save_kv_layer` | 1.00 | 1.2% | 5 |
| `iaxl/utils/hash.py:24` | `return compute_hash(is_first_block, prev_block_hash, cur_block_token_ids)` | `hash_block_tokens` | `KVShrinkConnector.get_num_new_matched_tokens` | 1.00 | 1.2% | 5 |

列说明:“代码位置”是最里面的 kvshrink 帧的文件和行号,即那一刻正在执行的 kvshrink 代码;“源代码”是这一行的内容;“所在函数”是这一行所在的函数;“入口函数”同表 1;“毫秒/请求”是这一行的 Python 开销;“占 Python 开销的比例”是这一行 ÷ 主线程上 kvshrink 的Python 开销合计;“样本数”是这一行的样本数。

## 附 1:算作 IAXL C++ 的代码行(前 15 个,核对归类用)

| 代码位置 | 源代码 | 所在函数 | 入口函数 | 毫秒/请求 | 占 IAXL C++ 的比例 | 样本数 |
|---|---|---|---|---|---|---|
| `iaxl/kvflow/flow.py:256` | `ctx.xfer_wait_cur_stream(sync_cur_stream=True)` | `KVFlow.put` | `KVShrinkConnector.save_kv_layer` | 22207.60 | 99.6% | 111038 |
| `iaxl/kvflow/flow.py:304` | `result.ctx.zip_wait()` | `KVFlow.put_wait` | `KVShrinkConnector.get_finished` | 35.40 | 0.2% | 177 |
| `iaxl/kvflow/flow.py:263` | `ctx.zip_to_mem(` | `KVFlow.put` | `KVShrinkConnector.save_kv_layer` | 17.00 | 0.1% | 85 |
| `iaxl/kvflow/flow.py:259` | `ctx.xfer_chunks_batch(chunk_indices, cpu_tensors)` | `KVFlow.put` | `KVShrinkConnector.save_kv_layer` | 15.20 | 0.1% | 76 |
| `iaxl/kvflow/flow.py:472` | `self.record.sync()` | `KVFlow.record_flush` | `KVShrinkConnector.save_kv_layer` | 8.80 | 0.0% | 44 |
| `iaxl/kvflow/flow.py:484` | `return self.record.has(label, chunk_labels)` | `KVFlow.has` | `KVShrinkConnector.get_num_new_matched_tokens` | 6.60 | 0.0% | 33 |
| `iaxl/kvflow/flow.py:185` | `return Context.create(tensor, chunk_dim, direction, description, work_stream=work_stream)` | `KVFlow._create_ctx` | `KVShrinkConnector.save_kv_layer` | 2.20 | 0.0% | 11 |
| `iaxl/kvflow/flow.py:263` | `ctx.zip_to_mem(` | `KVFlow.put` | `KVStoreLocal.put` | 1.80 | 0.0% | 9 |
| `iaxl/kvflow/flow.py:260` | `ctx.xfer_finish()` | `KVFlow.put` | `KVShrinkConnector.save_kv_layer` | 1.00 | 0.0% | 5 |
| `iaxl/kvflow/flow.py:295` | `if result.ctx is not None and not result.ctx.zip_is_complete():` | `KVFlow.put_wait` | `KVShrinkConnector.get_finished` | 0.40 | 0.0% | 2 |
| `iaxl/kvstore/kvstore.py:351` | `self.tensorzip.record.touch(label, block_hashs)` | `KVStoreLocal.has` | `KVShrinkConnector.get_num_new_matched_tokens` | 0.20 | 0.0% | 1 |
| `iaxl/kvflow/flow.py:478` | `self.record.submit(label, chunk_labels)` | `KVFlow.put_finish` | `KVShrinkConnector.save_kv_layer` | 0.20 | 0.0% | 1 |

列说明:同表 2,只是这里列的是被算作 IAXL C++ 的行(调用 IAXL 扩展的行);“占 IAXL C++ 的比例”是这一行 ÷ 主线程上 IAXL C++ 的合计。

## 附 2:各线程的样本数

| 线程 | 样本数 |
|---|---|
| MainThread | 126194 |
| Thread-3 (process_output_sockets) | 95 |
| Thread-2 (process_input_sockets) | 64 |
| iaxl-monitor | 64 |
| mgmt-ctrl_0 | 25 |
| Thread-227 (process_request_thread) | 23 |
| Thread-340 (process_request_thread) | 17 |
| Thread-1 (_report_usage_worker) | 14 |
| mgmt-controller | 10 |
| mgmt-worker-0 | 8 |
| tqdm_monitor | 5 |
| Thread-217 (process_request_thread) | 3 |
| thread (0) | 2 |

列说明:“线程”是 py-spy 记录的线程名;“样本数”是这个线程在忙(没睡眠、没阻塞)时被采到的次数。
