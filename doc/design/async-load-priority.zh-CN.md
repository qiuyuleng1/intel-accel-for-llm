# KVShrink 异步加载(async load)队列调度优化设计

> 状态:讨论稿(2026-10-08)。开发分支 `perf/async-load-optimization-dev`,PR 分支 `perf/async-load-optimization`。

## 术语

| 术语 | 含义 |
|---|---|
| L | 模型总层数,下文例子取 48 |
| N | 提前 promote 所需的层数(`KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS`),下文例子取 4 |
| promote | worker 在 `get_finished()` 中检测到某 async 请求前 N 层已加载完成,向 vLLM 上报 `finished_recving`,该请求随后可被调度进 prefill |
| 头 / 尾 | 一个 async 请求的前 N 层 / 后 L−N 层 |
| sync 批 | 同一 step 内所有 sync 请求合并后的一次 `get()` |
| `S_k` / `A_k` | sync 批的第 k 层任务 / 请求 A 的第 k 层任务 |

## 1. 背景

当前 async load 流程:scheduler 判定请求走 async 后,worker 在 `start_load_kv()` 中**一次性把该请求全部 L 层的解压任务提交进队列**;`get_finished()` 轮询到前 N 层完成即 promote;剩余层在 prefill forward 的 `wait_for_layer_load()` 中按需等待。

观察到的现象:并发多个 async 请求时,后到请求的加载迟迟不开始,pipeline 表现混乱。

## 2. 代码事实

以下是读代码确认到的事实,是后续分析的依据。

### 2.1 所有请求、所有层、压缩与解压,共用一个单线程串行队列

| 位置 | 事实 |
|---|---|
| `iaxl/csrc/torch_ext/context.h` `omp_queue()` | 全局唯一 `static TaskQueue("OMP-Main")` |
| `iaxl/csrc/include/task_queue.h` | **1 个 worker 线程**;两个 deque(HIGH / LOW);先取 HIGH 再取 LOW,级内严格 FIFO;优先级在提交时固定,之后不可改 |
| `iaxl/csrc/torch_ext/zip.cpp` `Context::unzip_from_mem` | 解压任务提交为 `PRIORITY_HIGH` |
| `iaxl/csrc/torch_ext/zip.cpp` `Context::zip_to_mem` | 压缩任务提交为 `PRIORITY_LOW` |
| `iaxl/csrc/kv_zip/kv_zip.cpp` `zip_pipeline` | 每个任务内部开 OMP parallel region,占满全部 QAT/IAA/CPU 压缩 worker |
| `iaxl/kvflow/flow.py` `KVFlow.get()` | 对传入的每个 layer 按顺序各提交一个解压任务 |

任务粒度 = **一个 (请求, 层) 的整批 block**;同一时刻只有一个任务在执行,不可抢占。

GPU 拷贝另有 H2D / D2H 两个队列,但解压任务是"解压完直接在任务内部发起 H2D 拷贝",不影响上述结论。

### 2.2 提交顺序:sync 批内按层合并;sync 批与 async 请求之间、async 请求之间按请求排

`kvshrink_connector.py` `start_load_kv()`:

1. 所有 sync 请求的 block 先合并,调**一次** `get()` → 队列里是 `[S0 S1 ... S47]`,每个 `S_k` 含本 step 所有 sync 请求的第 k 层。**sync 请求之间已是按层排的。**
2. 每个 async 请求各调一次 `get()` → `[A0 A1 ... A47]`,然后 `[B0 B1 ... B47]`。

一个 step 提交完后队列内容:

```
[S0 S1 ... S47][A0 A1 ... A47][B0 B1 ... B47]
```

跨 step 来看,之前 step 提交的任务排在前面。

### 2.3 `start_load_kv()` 每个 step 都会被调用

vLLM `vllm/v1/worker/kv_connector_model_runner_mixin.py` `_get_kv_connector_output()` 在每次 `execute_model` 开头无条件调用 `start_load_kv()`;本 step 没有可运行 token 时走 `kv_connector_no_forward()`,同样调用。

因此 connector 有稳定的"每 step 一次"钩子。但 promote 之后 `metadata.reqs_to_load` 已不含该请求,如需分批提交,worker 侧必须自行保存 block_ids / block_hashes(当前只保存了 `Task`)。

## 3. 三个具体问题

### (a) 先到 async 请求 A 的尾,堵住后到 async 请求 B 的头

```
step t:   A 提交 → [A0 A1 ... A47]
step t+1: B 提交 → [A0 A1 ... A47 B0 B1 ... B47]
```

B 要 promote 需要 B0~B3 完成,但 B0 排在 A47 之后 → B 要等 A 全部 48 层解压完才开始动。这是最初观察到的现象。影响:B 的 TTFT 变长。

### (b) 先到 async 请求 A 的尾,堵住后到 sync 请求 C

```
step t:   A(async) 提交 → [A0 ... A47]
step t+1: C(sync)  提交 → [A0 ... A47 C0 ... C47]
```

C 是 sync,本 step forward 算 layer 0 之前就在 `wait_for_layer_load(L0)` 等 C0;C0 排在 A47 之后 → **forward 卡住等 A 全部解压完,GPU 空转**。影响:整个 batch(含正在 decode 的其他请求)全部停住,比 (a) 严重。

同一形态还出现在"已 promote 请求的尾 + 本 step 的 sync 批"之间:

```
[A4 A5 ... A47]      ← A 上一 step promote,尾早已入队
[S0 S1 ... S47]      ← 本 step sync 批,排在后面
```

forward 等 S0,S0 前面是 A 的 44 层。

### (c) 同一 batch 内多个已 promote 请求,层序错开

A、B 都 promote,进入同一个 prefill batch,队列剩余:

```
[A4 A5 ... A47 B4 B5 ... B47]
```

forward 算到 layer 4 时 `wait_for_layer_load(L4)` 遍历所有已 promote 请求,要 A4 和 B4 **都**完成才放行;B4 排在 A47 之后 → forward 在 layer 4 等 A 剩余 44 层。

根因:forward 是**按层**消费的(算 layer k 需要 batch 内所有请求的第 k 层,不需要任何请求的第 k+1 层),但队列是按请求排的。

## 4. 候选方案

### 方案一:分批提交(Python 侧)

首次 `start_load_kv()` 只提交前 N 层;promote 后,下一 step 的 `start_load_kv()` 再提交其余 L−N 层。

效果:队列里只剩两类解压任务——"未 promote 请求的头(各 N 层)"和"已 promote 请求的尾(各 L−N 层,forward 正在等)"。(a)(b) 中"被堵 L 层"缩为"被堵 N 层";(c) 不变(尾仍以 per-request 顺序提交)。

代价:尾失去 promote 前的提前解压时间。forward 算 layer i 时队列同时解压 layer i+1、i+2…,是流水线;**不卡的条件是单层解压时间 ≤ 单层 prefill 计算时间**。该条件是否满足需测量(第 6 节)。

### 方案二:队列多级可变优先级(C++ 侧)

一次性提交 L 层不变;`TaskQueue` 改为多级且优先级可变:

| 级别 | 内容 | 级内排序 |
|---|---|---|
| P0 | forward 正在等的:sync 批全部层 + 已 promote 请求的尾 | 按 layer 号跨请求交错,同层按提交序 |
| P1 | 未 promote 请求的头 | 按提交序 FIFO |
| P2 | 未 promote 请求的尾(预取) | 按提交序 FIFO |

promote 时对该请求的尾调用 `set_priority(P0)`。

P1 用 FIFO 而非按层交错是有意的:头的目标是让**某一个**请求尽快凑齐 N 层去被调度;按层交错(A0 B0 A1 B1…)会让两个请求都拖到最后才凑齐。

P2 存在的唯一理由:让 B 的头(P1)能压过 A 尚未 promote 的尾。若 P1、P2 合并,A 的 48 层(先提交)与 B 的头(后提交)同级 FIFO,(a) 复现。

### 方案三:方案一 + 方案二叠加(推荐)

用方案一的提交方式,尾在 promote 前不进队列 → 队列里不存在"未 promote 的尾"这一类任务 → **P2 不再需要,也不需要可变优先级**。promote 后尾直接以 P0 提交。队列只需:

| 级别 | 内容 | 级内排序 |
|---|---|---|
| P0 | sync 批 + 已 promote 请求的尾 | 按 layer 号 |
| P1 | 未 promote 请求的头 | FIFO |

优先级和 layer 号在提交时指定,之后不变。

## 5. 方案对比

"(a)(b)(c)" 列写的是 B 的头 / C 的 layer 0 / forward 的 layer k **前面最多有多少无关任务**。

| 维度 | 原方案 | 方案一 | 方案二 | 方案三 |
|---|---|---|---|---|
| **(a)** B 头前面的无关任务 | 前面每个 async 请求各 L 层 | 前面未 promote 请求各 **N** 层 + 已 promote 请求各 L−N 层(后者 forward 正在等,排前面合理) | 只等前面请求的头(P1)和 P0;所有未 promote 尾(P2)排 B 后面 | 同方案二 |
| **(b)** C 的 layer 0 前面的无关任务 | 前面每个 async 请求各 L 层 | 前面未 promote 请求各 **N** 层 + 已 promote 请求各 L−N 层 | **0**:C 是 P0 且按层排,C0 到最前 | 同方案二 |
| **(c)** forward 在 layer k 等 B_k | B_k 前面有 A_{k+1}..A_{L−1} | **不变** | **0**:P0 按层排 → `[A4 B4 A5 B5 …]` | 同方案二 |
| 尾的提前解压时间 | 有 | **无**,靠流水线 | 有(P2 利用队列空闲预取) | 无,同方案一 |
| 尾不卡 forward 的条件 | 宽松 | 单层解压 ≤ 单层 prefill 计算 | 宽松 | 单层解压 ≤ 单层 prefill 计算 |
| 队列优先级 | 2 级固定 | 2 级固定 | 3 级**可变** | 2 级固定 |
| C++ 改动 | — | 无 | 多级 + 可变优先级 + `set_priority` 绑定 + 按层排序 | 2 级 + 按层排序键;提交接口加 priority / layer 号参数 |
| Python 改动 | — | worker 保存 async 请求元数据;promote 后补提交尾 | `get()` 加 per-layer priority;promote 时提权 | 方案一改动 + `get()` 加 priority / layer 号参数 |

结论:方案一单独能把 (a)(b) 从 L 层压到 N 层,(c) 不动;方案二三个全解但需可变优先级;方案三三个全解且队列只需 2 级固定优先级,改动最小。方案三唯一代价是尾没有提前量,是否可接受取决于第 7 节的测量。三个方案都实现并测性能,见第 8 节。

## 6. 会不会让 async 请求排不上号?

担心:P0 永远压在 P1 上,async 请求的头一直做不了。

P0 的任务来源:

1. **sync 请求**:按 dynamic map,并发到阈值以上新请求才走 async,高并发时 sync 来源很少。
2. **已 promote 请求的尾**:一个 async 请求只有在它的头(P1)做完后才会 promote,然后才产生尾的 P0 任务。即 **P0 的第二类来源是 P1 完成的结果**。若 P1 被饿着,就没有新 promote,P0 很快空,P1 就能跑。自限闭环,不会永久饿死。

会出现的是**短时延迟**:A 刚 promote,其 L−N 层尾进 P0,B 的头要等这些做完。这是有意取舍:A 的尾卡的是正在跑的 forward(整个 batch GPU 停转),B 的头卡的只是 B 一个请求的 TTFT。

## 7. 动手前先测两个数

1. 用 profiler(已有 `unzip_from_mem_work` scope)测**单个 (请求, 层) 任务的解压耗时**,对比**单层 prefill 计算耗时**和**一个 decode step 耗时**。
   - 决定方案三是否可行(单层解压 ≤ 单层计算 → 可行;否则需保留 P2 预取,即退回方案二)。
   - 若 48 层约 100~200 ms,(a)(b) 的排队延迟为百毫秒级,与 TTFT 同量级,值得改。
2. 记录 **HIGH deque 积压长度随时间的曲线**,确认高并发时确实出现多请求排队。

## 8. 实现复杂度评估

决定:**三个方案都实现,都测性能**。下面按"公共部分 + 各方案增量"拆分。

### 8.1 公共基础设施(三个方案都依赖,先做)

| 项 | 内容 | 为什么必须 |
|---|---|---|
| 队列可观测性 | `TaskQueue` 记录每个任务的入队 / 开始 / 结束时间,按 (priority, tensor_key, description) 打 profiler scope 或暴露到 `/v1/cache/metrics` | 没有它,(a)(b)(c) 的"排队等待时间"无法测量,方案对比没有证据 |
| 运行时开关 | 环境变量 `KVSHRINK_ASYNC_LOAD_SCHEME=0/1/2/3`(0 = 原方案) | 一次编译切换对比,避免每测一个方案重编一次扩展 |
| 第 7 节的测量 | 单层解压 vs 单层 prefill 计算耗时 | 决定方案一/三的流水线是否会卡 |

规模:C++ ~60 行(队列计时)+ Python ~30 行。

### 8.2 方案一:分批提交(纯 Python)

改动点全部在 `kvshrink/kvshrink_connector.py`:

| # | 位置 | 改什么 |
|---|---|---|
| 1 | `start_load_kv()` async 提交处 | async 请求只调 `get(layer_names=self._layer_names[:N])`;把 `block_ids / hashes / N` 存进新 dict `_async_req_meta[req_id]` |
| 2 | `get_finished()` promote 分支 | promote 后把 req_id 放进 `_tail_to_submit` |
| 3 | `start_load_kv()` 开头 | 遍历 `_tail_to_submit`,调 `get(layer_names=self._layer_names[N:])`,把返回的 Task dict **合并**进该请求已有的 Task dict |
| 4 | `wait_for_layer_load()` | 逻辑不改,但 `KVFlow.get_wait()` 有 `assert key in get_results`——第 3 步的合并不做,等 layer N 时直接 assert 挂 |
| 5 | `get_finished()` deferred-finish 分支 | 请求在 promote 后、尾提交前被 abort:要清 `_async_req_meta` / `_tail_to_submit`,否则泄漏,且下一 step 会给已死请求提交尾 |

`KVStore.get(layer_names=...)` 已支持子集,不需要动。

- 规模:~80~120 行 Python,新增 2 组 worker 状态。
- 风险 1:worker 状态从 3 组 dict(`_pending_load_tasks` / `_early_promoted_tasks` / `_active_promoted_tasks`)变成 5 组。**建议动手前先重构**:合并成一个 per-request 的 `AsyncLoadState`(字段:tasks, block_ids, block_hashes, head_layers, phase),这是后两个方案叠加的地基。
- 风险 2(现有问题,顺带发现):`start_load_kv()` 只要本 step 有 forward(`attn_metadata is not None`)就把所有 `_early_promoted_tasks` 搬进 `_active_promoted_tasks`,**不管该请求是否真的被调度进了这个 batch**。若 scheduler 因 token 预算未排进来,这个 forward 会白等它的尾。方案一不改变这点,但测性能时它会混进来。
- 复杂度:**低~中**。难点在状态清理,不在算法。

### 8.3 方案二:多级可变优先级(C++ 为主)

| # | 文件 | 改什么 |
|---|---|---|
| 1 | `iaxl/csrc/include/task_queue.h` | 两个 deque 换成一个条目列表,每条 `{shared_ptr<atomic<int>> prio, int64 sort_key, uint64 seq, fn}`;worker 在锁内线性扫最小 `(prio, sort_key, seq)`。队列长度 ≤ 并发数 × 层数(几千),线性扫可接受。**保留旧 `submit(fn, priority)` 重载**——`kv_pool/record.cpp` 的 `Record` 和 H2D / D2H 队列也用这个类 |
| 2 | 同上 | 新增 `submit(fn, prio_cell, sort_key)` 重载;调用方持有 `prio_cell` 即可随时改 |
| 3 | `torch_ext/context.h` / `torch_ext/zip.cpp` | `unzip_from_mem` 加 `priority, sort_key` 参数;`Context` 持有 `prio_cell`;新增 `set_priority(int)`;**cache-miss 重试路径**(`UnzipFromMemWork::execute` 中重新 submit 处)要复用同一个 cell,不能再硬编码 LOW |
| 4 | `torch_ext/torch_ext.cpp` | pybind:`unzip_from_mem` 加两个带默认值的参数;绑定 `set_priority` |
| 5 | `iaxl/kvflow/flow.py` `get()` | 加 `priorities` / `sort_keys`(按 tensor_key 的 dict),透传 |
| 6 | `iaxl/kvstore/kvstore.py` `get()` | 接收 per-layer priority;`sort_key = self.layer_names.index(name)` |
| 7 | connector `start_load_kv()` | sync → P0;async 前 N 层 → P1,其余 → P2 |
| 8 | connector `get_finished()` promote 分支 | 对该请求所有 `Task.ctx.set_priority(P0)`(ctx 可能已是 None,要判空) |

- 规模:C++ ~200~250 行(含重载兼容和重试路径),Python ~60 行,需重编扩展。
- 风险 1:线程安全。`set_priority` 从 Python 线程改 atomic,worker 在锁内读;任务已在执行时改优先级无效但无害,契约要写清楚。
- 风险 2:`Context` 的 move 构造 / 赋值要把 `prio_cell` 一起搬,漏了就是悬空。
- 风险 3:`TaskQueue` 目前没有单测(`tests/` 下只有两个 shell 脚本),要新加 gtest 覆盖"提权后顺序正确"。
- 复杂度:**中~高**。难在可变优先级 cell 的生命周期(跟着 Context 走、重试复用、move 语义),不在排序本身。

### 8.4 方案三:方案一的提交方式 + 2 级固定优先级 + P0 按层排

增量 = 方案一全部 + 方案二第 1、3、4、5、6 项的简化版:

| 项 | 相对方案二的简化 |
|---|---|
| `TaskQueue` | 不需要可变 cell。3 个容器:P0 用 `std::multimap<(sort_key, seq), fn>`,P1 和 LOW(zip)各一个 deque。~50 行 |
| `unzip_from_mem` | 加 `priority, sort_key` 两个参数即可,**不需要 `set_priority`**,`Context` 不加新成员,move 语义不动 |
| 重试路径 | 重试时沿用原 priority / sort_key(局部捕获即可) |
| connector | 方案一改动之上:首次提交头传 P1,补提交尾传 P0;sync 传 P0。promote 分支不调任何提权接口 |

- 规模:C++ ~80~100 行,Python = 方案一 + ~20 行。
- 风险:继承方案一的状态管理风险;C++ 侧风险明显低于方案二。
- 复杂度:**中**。

### 8.5 汇总

| | 方案一 | 方案二 | 方案三 |
|---|---|---|---|
| C++ 行数(估) | 0 | 200~250 | 80~100 |
| Python 行数(估) | 80~120 | ~60 | 100~140 |
| 需重编扩展 | 否 | 是 | 是 |
| 新增 worker 状态 | 2 组(建议重构合并) | 0 | 同方案一 |
| 新增 C++ 接口 | 无 | `submit` 重载、`set_priority`、pybind 两处 | `submit` 重载、pybind 一处 |
| 最难的点 | 状态清理(abort / deferred finish) | 可变优先级 cell 的生命周期 | 同方案一 |
| 复杂度 | 低~中 | 中~高 | 中 |

### 8.6 实现顺序

**公共基础设施 → connector 状态重构 → 方案一 → 方案三 → 方案二**,原因:

1. 方案三 = 方案一 + 一小块 C++,自然递进。
2. 方案二的 priority / sort_key 透传(第 3~6 项)与方案三完全相同;做完方案三后,方案二只剩"可变 cell + `set_priority` + promote 时提权"这一块增量。
3. 全部挂在 `KVSHRINK_ASYNC_LOAD_SCHEME` 开关下;方案二和三共存于同一份 `TaskQueue`(可变 cell 对方案三而言就是"从不改"),一次编译跑四组对比。

## 9. 待办

- [ ] 公共:`TaskQueue` 任务计时 + 暴露指标
- [ ] 公共:`KVSHRINK_ASYNC_LOAD_SCHEME` 开关
- [ ] 公共:第 7 节两个测量,结果附到本文档
- [ ] connector worker 侧状态重构:3 组 dict → per-request `AsyncLoadState`(先出字段设计)
- [ ] 方案一实现
- [ ] 方案三:`TaskQueue` 2 级固定优先级 + P0 按 layer 号排序;`unzip_from_mem` / `KVFlow.get()` / `KVStore.get()` 透传 priority / layer 号;connector 接入
- [ ] 方案二:`TaskQueue` 可变 cell + `Context::set_priority` + pybind;connector promote 时提权
- [ ] gtest:`TaskQueue` 排序 / 提权正确性
- [ ] 性能测试:(a)(b)(c) 三个场景 × 四个方案(含原方案),排队延迟 + TTFT + GPU 空转时间对比
- [ ] 现有问题:`_early_promoted_tasks` 搬入 `_active_promoted_tasks` 不判断请求是否在本 batch(单独评估是否修)

## 10. 进度记录

- 2026-10-08:完成问题分析、三个方案设计与复杂度评估;创建分支 `perf/async-load-optimization-dev`。下一步:connector `AsyncLoadState` 字段设计 → 公共基础设施。
