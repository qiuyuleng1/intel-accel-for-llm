# OMP-Main 任务 release 开销:分析与修复方案

> 状态:分析稿(2026-10-16),未实现,计划单独开 PR。数据来自开发分支 `perf/async-load-optimization-dev`。

## 术语

| 术语 | 含义 |
|---|---|
| OMP-Main | `omp_queue()` 返回的单线程 `TaskQueue`;所有层的解压(unzip)、压缩(zip)任务都在这一个线程上串行执行 |
| run | 任务函数本身的执行时间,trace 字段 `end_ns − start_ns` |
| release | 任务函数返回后,销毁该任务持有的数据(lambda 及其捕获)所花的时间,`release_ns − end_ns` |
| pick | release 结束到开始执行下一个任务,`下一个任务的 start_ns − release_ns` |
| block | KV cache 的一个 block(16 token);ScratchPool 的一个槽位对应一个 block |
| 引用数 | `at::Tensor` 背后 `TensorImpl` 的 C++ 强引用计数(`intrusive_ptr` 的 refcount) |

`release_ns` 由 `TaskQueue::worker_loop()` 记录:开 trace 时,先 `item.fn = nullptr`(显式销毁 callable),再取时间戳。`tests/async_load_dev/trace_timeline.py` 的 `worker_overhead()` 按"下一个任务已在排队"的情况统计 run / release / pick。

## 1. 现象(实测)

### 1.1 任务之间的间隔几乎全部是 release

数据:`_data/perf-qwen72b/gap8k_20261015_093354`,Qwen2.5-72B,TP=8,8k 输入,命中率 100,`batch_reqs_async_load_submit`,并发 32,rank 0;只统计下一个任务在当前任务结束前已入队的情况,共 4044 次。

| 阶段 | 平均(µs) | 中位(µs) | p95(µs) |
|---|---|---|---|
| run | 3161.0 | 2141.5 | 6767.7 |
| release | 766.2 | 505.3 | 2034.9 |
| pick | 0.6 | 0.5 | 0.9 |

### 1.2 release 与任务里的请求数成正比

数据:`_data/perf-qwen72b/trace8k_20261015_084115`,rank 0,并发 8 / 16 / 32 合并。这一轮的 trace 还没有 `release_ns`,表中"间隔"是下一个任务的 `start_ns − 本任务 end_ns`,即 release + pick;pick 只有 0.5 µs(1.1),所以间隔 ≈ release。

| 方案 | 任务里的请求数 | 次数 | 间隔中位(µs) | 每个请求的间隔(µs) | 任务执行中位(µs) |
|---|---|---|---|---|---|
| batch | 1 | 6501 | 177.3 | 177.3 | 1051.6 |
| batch | 2 | 4441 | 405.5 | 202.7 | 1939.4 |
| batch | 4 | 477 | 835.5 | 208.9 | 3545.6 |
| batch | 8 | 159 | 1912.6 | 239.1 | 6880.6 |
| batch | 10 | 80 | 2334.1 | 233.4 | 8271.2 |
| sync | 1 | 1738 | 343.7 | 343.7 | 1148.9 |
| sync | 7 | 1027 | 1845.6 | 263.7 | 6626.6 |
| sync | 16 | 237 | 4258.0 | 266.1 | 14861.0 |
| sync | 24 | 79 | 6684.3 | 278.5 | 21981.7 |

列说明:"任务里的请求数"= 上一个任务(一层)覆盖的请求个数;"每个请求的间隔"= 间隔中位 ÷ 请求数;"任务执行中位"= 上一个任务的 run 中位。8k 输入、block_size 16 时每个请求每层约 500 个 block,折合每个 block 约 0.35–0.7 µs。

## 2. 原因

### 2.1 代码事实(读代码确认)

1. `iaxl/kvflow/scratch_pool.py`:`ScratchPool.__init__()` 用 `self.pool.unbind(0)` 一次性生成全部 block 的 view,存在 `self._blocks`,进程存活期间一直持有;每个 view 还设了 Python 属性 `block_idx`。`allocate()` 返回的就是这些已有的 Python tensor 对象。
2. `iaxl/kvflow/flow.py`:`KVFlow.get()` 每个 tensor_key(每层)调用一次 `ctx.unzip_from_mem(..., cpu_tensors, ...)`,`cpu_tensors` 是 `allocate()` 返回的约 500 个 block tensor 的列表;`KVFlow.put()` 对 `zip_to_mem` 同理。
3. `iaxl/csrc/torch_ext/torch_ext.cpp`:`unzip_from_mem` / `zip_to_mem` 的绑定带 `py::call_guard<py::gil_scoped_release>()`;pybind 在释放 GIL 之前,把 Python 列表转成 `std::vector<torch::Tensor>`。
4. `iaxl/csrc/torch_ext/zip.cpp`:`UnzipFromMemWork` 持有一份 `std::vector<torch::Tensor> cpu_tensors` 和每个 block 一个的 `chunk_labels` 字符串;work 只被提交给队列的 lambda 持有,任务结束后在 OMP-Main 线程上销毁(`TaskQueue::worker_loop()` 中 `item.fn = nullptr` 处,或 `item` 出作用域时)。`zip_to_mem` 的 lambda 按值捕获 `cpu_tensors_copy`,同理。
5. PyTorch 2.13(容器 `iaxl.vllm.qiuyu-v29`,`torch 2.13.0+cu130`),`c10/util/intrusive_ptr.h` 的 Note [PyObject preservation for Tensor and Storages]:带 Python 对象的 `TensorImpl`,引用数从 1 变 2 时给 Python 对象 incref,从 2 变 1 时 decref;`reset_not_null_()` 中 `has_pyobject && new_refcount == 1` 时调用 `decref_pyobject()`。
6. `c10/core/impl/PyInterpreter.h`:`decref` 的注释为"Run Py_DECREF on a PyObject. We DO NOT assume the GIL is held on call."。PyTorch 源码 `torch/csrc/PyInterpreter.cpp` 中 `ConcretePyInterpreterVTable::decref()` 的实现是 `pybind11::gil_scoped_acquire gil; Py_DECREF(pyobj);`(读的是 PyTorch GitHub main 分支;容器里只有头文件,没有 .cpp)。

### 2.2 一个 block 的引用数变化

| 时刻 | 引用数 | 线程 | 对 Python 对象的操作 |
|---|---|---|---|
| ScratchPool 初始化后(Python view 持有) | 1 | — | — |
| pybind 把列表转成 `std::vector<torch::Tensor>` | 1 → 2 | Python 主线程(持有 GIL) | `Py_INCREF` |
| `work->cpu_tensors = cpu_tensors` | 2 → 3 | 主线程(已释放 GIL) | 无 |
| pybind 参数析构 | 3 → 2 | 主线程 | 无 |
| 任务结束,work 被销毁 | 2 → 1 | **OMP-Main** | **`gil_scoped_acquire` + `Py_DECREF`** |

即每个 block、每层、每个请求,OMP-Main 上都要拿一次 GIL。8k 输入的一个请求在一个 rank 上约为 80 层 × 500 block = 4 万次。(若任务在 pybind 参数析构之前就已执行完,2 → 1 会落在主线程上;只有队列空闲时才可能,少见。)

### 2.3 推断(未验证)

- release 的主要开销就是这 500 次 GIL 往返;每个 block 0.35–0.7 µs 与此吻合。
- OMP-Main 是纯 C++ 线程,没有 Python 线程状态;按 pybind11 `gil_scoped_acquire` 的实现,在这种线程上每次可能都要新建、销毁一个 `PyThreadState`,让单次往返更贵。
- 长尾(p95 约 2 ms)来自和 Python 主线程抢 GIL:主线程在执行 Python 字节码时,OMP-Main 要等它让出,最坏等一个切换间隔(默认 5 ms)。
- 主线程也被拖慢:每次被抢 GIL 都要让出、再拿回;提交时 pybind 逐个转换 500 个 tensor 也有开销。两者都未测。
- 旧版 PyTorch 在 Python 对象一直存活时,C++ 释放引用只是原子减一,不碰 Python 对象;2.13 的机制下才变成拿 GIL。具体哪个版本引入,未核对。

### 2.4 未确认

- `chunk_labels`(每个 block 一个字符串,在主线程分配、在 OMP-Main 释放)占 release 的多少。
- 以上推断尚无 profiling 证据,见 6 的第 0 步。

## 3. 影响

### 3.1 实测:OMP-Main 时间里 release 的占比

数据:`_data/perf-qwen72b/raise8k_20261016_022243`,8k 输入,命中率 100,8 个 rank 加总。

| 方案 | 并发 | run(ms) | release(ms) | release 占比 | 多用的队列时间 | 队列忙碌率 |
|---|---|---|---|---|---|---|
| promote_raise | 8 | 47611 | 12563 | 20.9% | 26.3% | 14.7% |
| promote_raise | 16 | 71223 | 17661 | 19.9% | 24.7% | 20.5% |
| promote_raise | 32 | 116100 | 26406 | 18.5% | 22.7% | 29.0% |
| prio | 8 | 46014 | 11535 | 20.0% | 25.0% | 14.4% |
| prio | 16 | 68537 | 15592 | 18.5% | 22.7% | 19.1% |
| prio | 32 | 113112 | 24020 | 17.5% | 21.2% | 25.3% |
| batch | 8 | 44657 | 11559 | 20.6% | 25.6% | 13.8% |
| batch | 16 | 67141 | 15954 | 19.2% | 23.6% | 19.2% |
| batch | 32 | 103025 | 23231 | 18.4% | 22.4% | 24.6% |

列说明:promote_raise / prio / batch 分别是 `priority_3level_first_n_per_req_promote_raise` / `priority_3level_first_n_per_req` / `batch_reqs_async_load_submit`;"release 占比"= release ÷ (run + release);"多用的队列时间"= 有别的任务在排队时发生的 release ÷ run,即同样的加载工作因 release 多花的队列时间(实测这部分几乎等于全部 release:每个请求一次提交 80 层,一个任务结束时后面基本总有任务排队);"队列忙碌率"= (run + release) ÷ 整段测试时长。

### 3.2 估算:trace 回放去掉 release

方法:对每个 rank 的 trace 按单线程队列重新排一遍。任务按 `enqueue_ns` 到达、进入 `submit_priority` 级,在 `p0_enter_ns` 时刻移到 P0 末尾,级内按进入该级的时间先进先出;任务执行时长取实测 `end_ns − start_ns`;release 取实测值(用于校验)或 0。回放脚本是临时写的,未入库。

校验:保留 release 回放,任务开始时间与实测平均相差 0.03–1.2 ms;回放出的 head_wait / load_total 与实测一致(例:promote_raise 并发 32 回放 284/22、467/230 ms,实测 284/22、468/230 ms)。

| 方案 | 并发 | load_total 平均:有 → 无 release(ms) | 缩短 | head_wait 平均:有 → 无 release(ms) | 缩短的 ms 占平均 TTFT |
|---|---|---|---|---|---|
| promote_raise | 8 | 195 → 133 | 32% | 82 → 30 | 15% |
| promote_raise | 16 | 301 → 213 | 29% | 147 → 63 | 15% |
| promote_raise | 32 | 467 → 372 | 20% | 284 → 62 | 11% |
| prio | 8 | 287 → 194 | 32% | 42 → 23 | 18% |
| prio | 16 | 502 → 403 | 20% | 69 → 23 | 9% |
| prio | 32 | 989 → 702 | 29% | 101 → 14 | 15% |
| batch | 8 | 259 → 205 | 21% | 50 → 38 | 10% |
| batch | 16 | 404 → 303 | 25% | 171 → 111 | 15% |
| batch | 32 | 840 → 674 | 20% | 378 → 282 | 14% |

列说明:每个请求取 8 个 rank 中最慢的;load_total = 请求的加载任务第一次入队 → 最后一层加载完;head_wait = 第一次入队 → 前 N 层加载完;"缩短的 ms 占平均 TTFT"= load_total 平均缩短的毫秒数 ÷ 该格实测平均 TTFT(promote_raise 416 / 603 / 870 ms,prio 510 / 1054 / 1866 ms,batch 515 / 673 / 1174 ms)。

加载时间缩短的比例(20–32%)大于 release 占比(约 20%):请求集中到达时,队列处理得越快,排队等待缩短得越多。

### 3.3 对 TTFT(推断)

按上表最后一列,平均 TTFT 约慢了 9–18%。这个换算很粗:

- 回放固定了请求到达时间和提权时刻;实际加载变快后 promote 会提前、step 会变短,下游会继续受益,回放没算。
- 没算 release 抢 GIL 对 Python 主线程的影响。
- TTFT 还包含按 step 推进的调度时间,加载省下的毫秒数不一定一比一反映到 TTFT。

准确值只能在修复后用同样配置重跑得到。sync 方案的加载走同一个队列,同样受影响,数据待补(`_data/perf-qwen72b/hitsweep8k_20261016_063957` 中的 sync trace 带 `release_ns`)。

## 4. 方案

### 4.1 方案 A:把销毁挪出 OMP-Main

任务执行完后,不在 OMP-Main 上析构 `cpu_tensors` / `chunk_labels`,而是交给一个后台线程销毁。

- A+:后台线程先拿一次 GIL(`gil_scoped_acquire` 可重入),再批量销毁一整批,把每批的 GIL 获取降到 1 次。

### 4.2 方案 B:不再逐个 block 传 `torch::Tensor`,只传 block 编号

- pool 在创建时向 C++ 登记一次:基地址、每个 block 的字节数、block 总数。已有先例:RDMA 路径在 `KVFlow._ensure_pool()` 中用 `rdma_register_local(pool.data_ptr(), pool.nbytes, block_bytes)` 登记 pool,并以 `block_idx` 作描述符下标。
- get / put 只传一串整数编号,C++ 用"基地址 + 编号 × block 字节数"寻址;热路径上不再生成、转换、释放 per-block 的 tensor。
- B':C++ 启动时自建 block 表(不挂 Python 对象的 view,或直接存指针),对外只收编号;解压、压缩、传输函数的内部接口基本不变,改动面比 B 小。

### 4.3 对比

| 维度 | A | A+ | B / B' |
|---|---|---|---|
| 去掉 OMP-Main 上的 release | 是 | 是 | 是 |
| 每请求每层拿 GIL 次数 | 约 500(换线程) | 约 1 | 0 |
| Python 主线程开销(被抢 GIL、提交时 pybind 转换) | 不变 | 被抢 GIL 减少,转换不变 | 都去掉 |
| 改动范围 | 小:`zip.cpp`、`TaskQueue` | 小 | 中到大:`Context` 的 get / put 接口、解压 / 压缩 / 传输函数、`flow.py`、`ScratchPool`、`tensor_ops.py`、RDMA / remote pool 路径、测试 |
| 新增风险 | 多一个线程;退出时须在释放 GIL 的状态下 join,否则死锁;后台积压无上限 | 同 A | 失去 tensor 自带的形状 / dtype 检查,需显式检查编号越界和字节数;需保证 pool 生命周期 |
| 治症状还是根因 | 症状 | 大部分 | 根因 |

列说明:"每请求每层拿 GIL 次数"按 8k 输入、每层约 500 个 block 估;"Python 主线程开销"指 vLLM worker 进程主线程上的额外开销。

## 5. 推荐:方案 B(按 B' 实施)

1. 问题出在"每个 block 都以带 Python 对象的 tensor 进出 C++"。A 只是换线程付这笔账,500 次 GIL 往返、抢主线程 GIL、提交时的转换都还在。
2. B 同时去掉 OMP-Main 和主线程两侧的开销。目标平台 B200 / B300 计算更快,主线程更不能被 GIL 打断。
3. 已有先例:RDMA 路径已按 block 编号寻址,B 是把它推广到所有路径。

A+ 作为备选:若 B 的改动范围对当期不合适,可先用 A+ 止血。

## 6. 实现计划(单独 PR)

**第 0 步:验证根因(不改代码)。** 在 hit 100、并发 32 运行时,对一个 rank 的 OMP-Main 线程 `perf record -g`(或 `py-spy dump --native`)。预期 release 阶段的栈集中在 `decref_pyobject` → 拿 GIL / 线程状态创建销毁;同时看字符串释放的占比。决策:GIL 路径占 release 大头则做 B';若主要是 malloc / free,重新评估。

**第 1 步:C++。** 增加 pool 登记接口与 block 表;新增按编号的 `unzip_from_mem` / `zip_to_mem` 入口(旧入口暂留);检查编号越界、block 字节数和 dtype 一致。

**第 2 步:Python。** `ScratchPool.allocate()` / `release()` 直接处理编号,热路径不再生成 tensor 列表;`Task` 存编号;`KVFlow.get()` / `put()` 改用新入口。

**第 3 步:其他调用方。** 排查 `tensor_ops.py`、remote pool、RDMA 路径和测试里对 `cpu_tensors` 的使用,逐个迁移。

**第 4 步:验证。**

| 项 | 方法 | 通过标准 |
|---|---|---|
| 数据正确 | 单测:pool 槽位反复复用下 put / get 逐字节一致 | 全部一致 |
| release 消除 | 同配置 trace,`trace_timeline.py` 的 run / release / pick | release 中位从约 500 µs 降到微秒级;pick 不变 |
| 无 GIL | 第 0 步同样的 `perf` | OMP-Main 栈中不再出现 `decref_pyobject` / 拿 GIL |
| 主线程开销 | 测 `start_load_kv()` 中提交耗时 | 报告前后对比 |
| 端到端性能 | `perf-qwen72b-bench.sh`,8k,命中率 90 / 95 / 98 / 100,并发 8 / 16 / 32,带 trace | 报告 TTFT、TPOT、load_total、head_wait 前后对比 |
| 输出正确 | `VLLM_BATCH_INVARIANT=1`,temperature 0、固定 seed,命中前后输出逐字节比较,扫描 U+FFFD / 非法控制字符;connector log 中 `externally-cached tokens` > 0 | 一致、无乱码 |

**第 5 步:拆 PR。** C++ 接口 + 测试;Python 切换;删除旧路径。

## 7. 证据目录(本机)

| 目录 | 内容 |
|---|---|
| `_data/perf-qwen72b/gap8k_20261015_093354` | 1.1 的 run / release / pick |
| `_data/perf-qwen72b/trace8k_20261015_084115` | 1.2 的间隔随请求数变化(无 `release_ns`) |
| `_data/perf-qwen72b/raise8k_20261016_022243` | 3.1、3.2 的占比与回放 |
| `_data/perf-qwen72b/hitsweep8k_20261016_063957` | 命中率 90–100、含 sync,带 `release_ns`(待分析) |
