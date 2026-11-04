# py-spy 结果:32k_c1,hit

## 这次运行是什么

- vLLM 0.29.0 + kvshrink,模型 `/media/model-space/Qwen3-32B`,TP=1,GPU 5,vLLM 进程绑定 CPU 64-127。调度、模型计算和 kvshrink 的全部函数都在 EngineCore 进程的主线程上。
- 一次 bench = `vllm bench serve` 发 50 个请求,输入 32768 token、输出 128 token,并发 1。
- 场景 hit:所有请求的 prompt 相同,测量前已存进 DDR,kvshrink 只加载、不保存。
- py-spy 从测量 bench 开始前采到结束后,采样率 100 次/秒,只采 EngineCore 进程(pid 141285),参数 `--rate 100 --threads --format raw --full-filenames --nonblocking`:不暂停被测进程,不记录 C/C++ 函数,在 C/C++ 里花的时间算在调用它的那一行 Python 上。一个样本约代表 10 ms;“毫秒/请求”= 样本数 ÷ 采样率 × 1000 ÷ 请求数。相对误差约 1/√样本数。
- 代码 git 28f2d3e,代码 md5 27e553aa,py-spy py-spy 0.4.3。

## 核对

| 核对项 | 结果 |
|---|---|
| baseline 这次 bench 的请求数(connector 日志) | 50 |
| baseline 里有外部命中(externally-cached tokens > 0)的请求数 | 50 |
| baseline 里 externally-cached tokens 的不同取值(前 5 个) | 32752 |
| pyspy 这次 bench 的请求数(connector 日志) | 50 |
| pyspy 里有外部命中(externally-cached tokens > 0)的请求数 | 50 |
| pyspy 里 externally-cached tokens 的不同取值(前 5 个) | 32752 |
| vLLM 日志里 `ScratchPool exhausted` 的次数 | 0 |
| py-spy 跟不上采样率的警告次数 | 0 |
| py-spy 自报的样本数 / 错误数 | 15693 / 4583 |
| 主线程样本里调用链不完整的样本数(占比) | 2072(13.6%) |

列说明:“核对项”是检查什么;“结果”是从 connector 日志、vLLM 日志或 py-spy 输出数出来的值。miss 场景外部命中的请求数应为 0;hit 场景应等于请求数。“调用链不完整”指最外层不是进程入口 `<module>` 的样本:py-spy 不暂停进程时,偶尔读到一半就变了的调用链。

### 有没有 py-spy 的 bench 对比(干扰检查)

| 指标 | 没有 py-spy | 有 py-spy | 有 / 没有 |
|---|---|---|---|
| duration | 414.22 | 414.18 | 1.000 |
| completed | 50.00 | 50.00 | 1.000 |
| request_throughput | 0.12 | 0.12 | 1.000 |
| mean_ttft_ms | 671.48 | 669.16 | 0.997 |
| median_ttft_ms | 670.69 | 673.03 | 1.003 |
| mean_tpot_ms | 59.94 | 59.95 | 1.000 |
| median_tpot_ms | 59.92 | 59.93 | 1.000 |
| mean_e2el_ms | 8283.88 | 8283.13 | 1.000 |
| median_e2el_ms | 8281.78 | 8284.88 | 1.000 |

列说明:“指标”是 `vllm bench serve` 结果里的字段(duration 单位秒,throughput 单位请求/秒,其余单位 ms,mean 是均值、median 是中位数,统计对象是这次 bench 的全部请求);后两列是两次 bench 的值;“有 / 没有”是比值,越接近 1 说明 py-spy 干扰越小。

### py-spy 数的主线程忙碌时间对不对(和内核记的 CPU 时间比)

| 量 | 没有 py-spy 的 bench | 有 py-spy 的 bench |
|---|---|---|
| EngineCore 主线程 CPU 时间(内核) | 3923.80 | 3912.60 |
| EngineCore 全进程 CPU 时间(内核) | 6644.20 | 6646.20 |
| py-spy 数的主线程忙碌时间 | - | 3049.40 |
| 校准系数 = 内核 ÷ py-spy(主线程) | - | 1.283 |

列说明:单位都是毫秒/请求(校准系数除外)。“内核”两行是 /proc 里 EngineCore 的 utime + stime(内核记账的、真正占用 CPU 的时间),在每次 bench 前后各读一次,差值 ÷ 请求数;主线程的线程号就是进程号,全进程包括IAXL、torch 等全部线程。“py-spy 数的主线程忙碌时间”= 主线程被判为忙的样本数 ÷ 采样率 ÷ 请求数。py-spy 采样窗口比 bench 略长(前后各多几秒,那时 EngineCore 没有请求),所以它和同一次 bench 的“主线程 CPU 时间(内核)”应基本相等;差很多说明 py-spy 漏采或多采。“校准系数”= 有 py-spy 的 bench 的“主线程 CPU 时间(内核)”÷“py-spy 数的主线程忙碌时间”,表 1 用它把 py-spy 直接数的主线程时间换算成内核口径。

## 表 1:按入口函数汇总(毫秒/请求)

| 入口函数 | Python 开销(py-spy 直接数) | Python 开销(按内核校准) | 对照:ScratchPool | 对照:IAXL C++(含参数转换) | 样本数 |
|---|---|---|---|---|---|
| `KVShrinkConnector.save_kv_layer` | 48.80 | 62.61 | 0.00 | 0.00 | 244 |
| `KVShrinkConnector.get_num_new_matched_tokens` | 16.60 | 21.30 | 0.00 | 8.60 | 126 |
| `KVShrinkConnector.wait_for_layer_load` | 2.00 | 2.57 | 5.20 | 108.60 | 579 |
| `KVShrinkConnector.start_load_kv` | 1.60 | 2.05 | 5.80 | 58.60 | 330 |
| `KVStoreLocal.get_wait` | 0.00 | 0.00 | 0.40 | 0.00 | 2 |
| `ScratchPool.release.<locals>.<genexpr>` | 0.00 | 0.00 | 0.40 | 0.00 | 2 |
| 主线程上 kvshrink 合计 | 69.00 | 88.53 | 11.80 | 175.80 | 1283 |
| 其他线程上的 kvshrink | 1.80 | - | 0.00 | 0.00 | 9 |

主线程 CPU 忙碌总时间(含 vLLM 自己的代码):py-spy 直接数 3049.40 毫秒/请求(15247 个样本),内核记 3912.60 毫秒/请求;kvshrink 的 Python 开销占 py-spy 主线程忙碌样本的 2.3%。

列说明:“入口函数”是调用链里最外面的 kvshrink 函数,即 vLLM 调用的那个;“Python 开销”是本次要量的量:kvshrink 的 Python 语句,加上它们直接调用的 Python 代码和xxhash、numpy、torch 等库的 C 代码;“py-spy 直接数”= 样本数 ÷ 采样率 ÷ 请求数;“按内核校准”= 直接数的值 × 校准系数(见上一节),假设 py-spy 漏掉的样本均匀分布在各段代码上,其他线程不校准(校准系数只对主线程测过);两个“对照”列不算 Python 开销,是 py-spy 直接数的值:ScratchPool 取块还块归同事,IAXL C++ 是调用 IAXL 扩展的那些行上的时间(C++ 执行、pybind11 转换参数、空转等待,哪些行见附 1);以上单位都是毫秒/请求;“样本数”是这一行三类的样本总数。“其他线程上的 kvshrink”是管理接口、内存池淘汰等后台线程。

## 表 2:Python 开销最多的前 20 个代码位置

| 代码位置 | 源代码 | 所在函数 | 入口函数 | 毫秒/请求 | 占 Python 开销的比例 | 样本数 |
|---|---|---|---|---|---|---|
| `kvshrink/kvshrink_connector.py:718` | `for req_id, request in metadata.reqs_to_save.requests.items():` | `KVShrinkConnector.save_kv_layer` | `KVShrinkConnector.save_kv_layer` | 45.80 | 66.4% | 229 |
| `iaxl/utils/hash.py:13` | `parts = [repr(arg) for arg in args]` | `compute_hash` | `KVShrinkConnector.get_num_new_matched_tokens` | 4.40 | 6.4% | 22 |
| `iaxl/utils/hash.py:16` | `return xxhash.xxh64(text.encode("utf-8")).intdigest() & 0x7FFFFFFFFFFFFFFF` | `compute_hash` | `KVShrinkConnector.get_num_new_matched_tokens` | 3.20 | 4.6% | 16 |
| `iaxl/utils/hash.py:14` | `parts += [f"{k}={repr(v)}" for k, v in sorted(kwargs.items())]` | `compute_hash` | `KVShrinkConnector.get_num_new_matched_tokens` | 2.60 | 3.8% | 13 |
| `iaxl/utils/hash.py:24` | `return compute_hash(is_first_block, prev_block_hash, cur_block_token_ids)` | `hash_block_tokens` | `KVShrinkConnector.get_num_new_matched_tokens` | 1.60 | 2.3% | 8 |
| `iaxl/utils/hash.py:15` | `text = "\|".join(parts)` | `compute_hash` | `KVShrinkConnector.get_num_new_matched_tokens` | 1.00 | 1.4% | 5 |
| `kvshrink/kvshrink_connector.py:670` | `return` | `KVShrinkConnector.wait_for_layer_load` | `KVShrinkConnector.wait_for_layer_load` | 1.00 | 1.4% | 5 |
| `kvshrink/kvshrink_connector.py:669` | `if not self._current_get_tasks and not self._active_promoted_tasks:` | `KVShrinkConnector.wait_for_layer_load` | `KVShrinkConnector.wait_for_layer_load` | 0.80 | 1.2% | 4 |
| `kvshrink/kvshrink_connector.py:709` | `if self._connector_metadata is None or layer_name not in self.layer_group:` | `KVShrinkConnector.save_kv_layer` | `KVShrinkConnector.save_kv_layer` | 0.80 | 1.2% | 4 |
| `kvshrink/kvshrink_connector.py:701` | `def save_kv_layer(` | `KVShrinkConnector.save_kv_layer` | `KVShrinkConnector.save_kv_layer` | 0.80 | 1.2% | 4 |
| `iaxl/kvstore/kvstore.py:355` | `mask = np.array(results, dtype=np.bool_)` | `KVStoreLocal.has` | `KVShrinkConnector.get_num_new_matched_tokens` | 0.80 | 1.2% | 4 |
| `kvshrink/kvshrink_connector.py:283` | `str(block_hash)` | `KVShrinkConnector.get_num_new_matched_tokens` | `KVShrinkConnector.get_num_new_matched_tokens` | 0.80 | 1.2% | 4 |
| `iaxl/utils/hash.py:37` | `cur_hash = hash_block_tokens(` | `generate_block_hashs` | `KVShrinkConnector.get_num_new_matched_tokens` | 0.80 | 1.2% | 4 |
| `kvshrink/kvshrink_connector.py:717` | `layer_keys = [self._layer_key[layer_name]]` | `KVShrinkConnector.save_kv_layer` | `KVShrinkConnector.save_kv_layer` | 0.60 | 0.9% | 3 |
| `iaxl/kvflow/flow.py:372` | `cpu_tensors = self._allocate(num_chunks, chunk_shape, tensor.dtype)` | `KVFlow.get` | `KVShrinkConnector.start_load_kv` | 0.60 | 0.9% | 3 |
| `kvshrink/kvshrink_connector.py:712` | `metadata = self._get_connector_metadata()` | `KVShrinkConnector.save_kv_layer` | `KVShrinkConnector.save_kv_layer` | 0.40 | 0.6% | 2 |
| `kvshrink/kvshrink_connector.py:716` | `group_idx = self.layer_group[layer_name]` | `KVShrinkConnector.save_kv_layer` | `KVShrinkConnector.save_kv_layer` | 0.40 | 0.6% | 2 |
| `kvshrink/kvshrink_connector.py:282` | `block_hashes = [` | `KVShrinkConnector.get_num_new_matched_tokens` | `KVShrinkConnector.get_num_new_matched_tokens` | 0.20 | 0.3% | 1 |
| `iaxl/kvflow/flow.py:399` | `results[tensor_key] = task = Task(` | `KVFlow.get` | `KVShrinkConnector.start_load_kv` | 0.20 | 0.3% | 1 |
| `kvshrink/kvshrink_connector.py:336` | `logger.info(` | `KVShrinkConnector.get_num_new_matched_tokens` | `KVShrinkConnector.get_num_new_matched_tokens` | 0.20 | 0.3% | 1 |

列说明:“代码位置”是最里面的 kvshrink 帧的文件和行号,即那一刻正在执行的 kvshrink 代码;“源代码”是这一行的内容;“所在函数”是这一行所在的函数;“入口函数”同表 1;“毫秒/请求”是这一行的 Python 开销;“占 Python 开销的比例”是这一行 ÷ 主线程上 kvshrink 的Python 开销合计;“样本数”是这一行的样本数。

## 附 1:算作 IAXL C++ 的代码行(前 15 个,核对归类用)

| 代码位置 | 源代码 | 所在函数 | 入口函数 | 毫秒/请求 | 占 IAXL C++ 的比例 | 样本数 |
|---|---|---|---|---|---|---|
| `iaxl/kvflow/flow.py:458` | `result.ctx.xfer_wait()` | `KVFlow.get_wait` | `KVShrinkConnector.wait_for_layer_load` | 106.60 | 60.6% | 533 |
| `iaxl/kvflow/flow.py:387` | `ctx.unzip_from_mem(` | `KVFlow.get` | `KVShrinkConnector.start_load_kv` | 58.60 | 33.3% | 293 |
| `iaxl/kvflow/flow.py:484` | `return self.record.has(label, chunk_labels)` | `KVFlow.has` | `KVShrinkConnector.get_num_new_matched_tokens` | 7.80 | 4.4% | 39 |
| `iaxl/kvflow/flow.py:457` | `result.ctx.unzip_wait()` | `KVFlow.get_wait` | `KVShrinkConnector.wait_for_layer_load` | 2.00 | 1.1% | 10 |
| `iaxl/kvstore/kvstore.py:351` | `self.tensorzip.record.touch(label, block_hashs)` | `KVStoreLocal.has` | `KVShrinkConnector.get_num_new_matched_tokens` | 0.80 | 0.5% | 4 |

列说明:同表 2,只是这里列的是被算作 IAXL C++ 的行(调用 IAXL 扩展的行);“占 IAXL C++ 的比例”是这一行 ÷ 主线程上 IAXL C++ 的合计。

## 附 2:各线程的样本数

| 线程 | 样本数 |
|---|---|
| MainThread | 15247 |
| thread (143655) | 174 |
| Thread-3 (process_output_sockets) | 100 |
| Thread-2 (process_input_sockets) | 71 |
| thread (0) | 69 |
| Thread-1 (_report_usage_worker) | 10 |
| mgmt-ctrl_0 | 5 |
| mgmt-worker-0 | 5 |
| mgmt-controller | 4 |
| iaxl-monitor | 2 |
| Thread-113 (process_request_thread) | 1 |
| thread (148341) | 1 |
| thread (149212) | 1 |
| tqdm_monitor | 1 |
| thread (148889) | 1 |
| thread (148547) | 1 |

列说明:“线程”是 py-spy 记录的线程名;“样本数”是这个线程在忙(没睡眠、没阻塞)时被采到的次数。
