# KVShrink 异步加载(async load)队列调度优化设计

> 状态:讨论稿(2026-10-09)。开发分支 `perf/async-load-optimization-dev`,PR 分支 `perf/async-load-optimization`。

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
| `iaxl/csrc/include/task_queue.h` | **1 个 worker 线程**;两个 deque(HIGH / LOW);先取 HIGH 再取 LOW,级内严格 FIFO;优先级在提交时固定,之后不可改(10-14 起改为 P0..P3 四级、可按组提权,见 §8.4、§11.10) |
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

## 3. 两个具体问题

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

### 已知问题 K1:同一 step 提交的多个 async 请求按请求排,不按层排

不属于 (a)(b) 这类难点:改法简单,与现有 sync 路径一致。改法命名为 `batch_reqs_async_load_submit`,作为所有方案的公共前提(§4.1)。

**现象**:`start_load_kv()` 里 async 请求逐请求提交——

```python
for req_id, request in async_reqs:
    self._pending_load_tasks[req_id] = self._store().get(
        block_indices=request.block_ids,      # 只含这一个请求的 block
        block_hashs=request.block_hashes,
        description=req_id,                   # 未传 layer_names → 全部 L 层
    )
```

`KVFlow.get()` 对每层各提交一个任务,所以同一 step 的 A、B 在队列里是 `[A0 ... A47][B0 ... B47]`。后果:

1. B 的头排在 A 的尾后面。实测见 §11.3 `naive-s1`:step 2 提交的 7 个请求,首个任务依次等 6、128、243、359、472、587、701 ms,前面没有更早 step 的任务。
2. A、B 都 promote 并进入同一 batch 时,forward 算 layer k 要等 A_k 和 B_k,B_k 排在 A_{k+1}..A_{L−1} 后面。

根因:forward 按层消费(算 layer k 需要 batch 内所有请求的第 k 层),而这里按请求提交。

**改法**(`batch_reqs_async_load_submit`):与 sync 相同,把本 step 所有 async 请求打成一个 batch,调一次 `get()`,队列变为 `[L0(A+B) L1(A+B) ... L47(A+B)]`。A、B 前 N 层同时就绪、同时 promote,尾也按层排。

**改法的约束**(读代码得出,未实测):

| 项 | 说明 |
|---|---|
| 头变慢 | 每层任务包含 A+B 的 block,单个请求前 N 层就绪时间 ≈ N × (A+B 的单层解压时间)。单层时间与 block 数近似成正比是推断,未测 |
| 共用 Task | 各请求的 `_pending_load_tasks[req_id]` 指向同一组 Task;`KVFlow.get_wait()` 对已等过的任务(`ctx is None`)直接跳过,重复等待安全;各请求仍可按自己的 N 判断 promote |
| abort / finish | 某个请求结束时等待的是整组任务(含其他请求的 block) |
| 覆盖范围 | 只覆盖同一 step 提交的请求;不同 step 的请求仍按批次先后排,即 (a) |
| 与 `split_async_load_submit` 的关系 | promote 后补提交尾时,同一 step 内补提交的尾也按同样方式打成 batch |

### 已知问题 K2:promote 后的 forward 等尾,拖长 step,推迟后到请求的提交

推断,未验证,不单独测试。

**现象**(早期两请求测试,B 比 A 晚 20 / 50 ms 发送;时间相对 A 第 0 层入队):A 前 4 层完成于 9.3 / 8.6 ms,A 最后一层完成于 181.3 / 165.7 ms,B 第 0 层入队于 188.6 / 178.6 ms。B 在队列里没有等待,但提交本身被推迟了约 170 ms。

**推断的原因**:`start_load_kv` 每个 step 只在开头调用一次。A 在约 9 ms promote 后,下一个 step 做 A 的 prefill forward,它在 `wait_for_layer_load` 里逐层等 A 自己的尾,这个 step 一直持续到 A 的尾全部完成;B 只能在下一个 step 才被调度和提交。

**影响**:这段延迟发生在任务进队列之前,queue trace 看不到;本文的方案只改队列内顺序,对它无效。

### 已知问题 K3:已 promote 的请求在未被调度的 forward 里也被等待

**现象**(读代码):`start_load_kv()` 只要本 step 有 forward(`attn_metadata is not None`),就把所有 `_early_promoted_tasks` 搬进 `_active_promoted_tasks`,不检查这些请求是否被调度进了本次 forward。之后 `wait_for_layer_load()` 每算一层都会等这些请求的这一层。

**推断很常见**(未实测):vLLM 开了异步调度(日志 `Asynchronous scheduling is enabled`),scheduler 在 step k 执行时已经排好 step k+1;step k 里 promote 的请求往往到 step k+2 才被调度,但 connector 在 step k+1 就把它搬进 active。

**影响**:不含该请求的 forward 白等它的尾。`split_async_load_submit` 下尾恰好在这个 step 开头才提交,forward 要从头等整段尾的解压,影响比 `naive` 更大。

**改法**(对所有方案生效,在 `split_async_load_submit` 之前单独提交):

1. scheduler 侧 `build_connector_meta()` 把 `scheduler_output.num_scheduled_tokens` 的 key(本 step 被调度的请求)放进 connector metadata。
2. worker 侧 `start_load_kv()` 只把本 step 被调度的已 promote 请求搬进 active,其余留在 early,等它真正被调度的 step 再搬。

安全性:请求只在被调度的那次 forward 里被等;在 split 下,它的尾在 promote 后的下一次 `start_load_kv()` 就已提交,不会晚于它被调度的那个 step,所以不会出现 `key not in get_results`。


## 4. 方案

### 4.0 命名

| 名字 | 做法 | 前提 |
|---|---|---|
| `naive` | 现状:每个 async 请求单独调一次 `get()`,一次提交全部 L 层 | 对照组,不加前提 |
| `batch_reqs_async_load_submit` | 同一 step 的 async 请求打成一个 batch,调一次 `get()`;每层一个任务,覆盖 batch 内所有请求的 block(与 sync 路径相同) | 本身就是公共前提 |
| `split_async_load_submit_first_n_batch_reqs` | async 请求先只提交头(前 N 层);promote 后的下一次 `start_load_kv` 再提交尾(后 L−N 层)。同一 step 的头组 batch,按层分段每段一次 `get()` | `batch_reqs_async_load_submit` |
| `split_async_load_submit_first_n_per_req` | 同上,但头逐个请求提交:每个请求一次 `get()`,按到达顺序;尾仍组 batch 按层提交 | `batch_reqs_async_load_submit` |
| `priority_3level_first_n_per_req` | 一次提交全部 L 层;队列加 3 级优先级(P0/P1/P2),头逐个请求提交;请求进入 forward(K3 搬入 active)时把它的尾从 P2 调到 P0。已实现(10-14,§11.10) | `batch_reqs_async_load_submit` |
| `priority_3level_first_n_per_req_promote_raise` | 同上,但 promote 时就把尾调到 P0。已实现(§11.12) | `batch_reqs_async_load_submit` |
| `priority_3level_first_n_per_req_remaining_by_layer` | 同 `_first_n_per_req`,但同一 step 激活的请求与本 step 的 sync 批一次提权,在 P0 里按层排。10-22 实现(§4.3 末尾) | `batch_reqs_async_load_submit` |
| `priority_3level_first_n_batch_reqs_remaining_by_layer` | 同上,但同一 step 的头组 batch 按层分段提交。10-22 实现 | `batch_reqs_async_load_submit` |
| `split_priority_2level` | 按 `split_async_load_submit` 分两次提交;队列加 2 级固定优先级(P0/P1):头用 P1,尾和 sync 用 P0,P0 内先进先出 | `batch_reqs_async_load_submit` |

列说明:"名字"用于文档、代码和 `KVSHRINK_ASYNC_LOAD_SCHEME` 的取值;"做法"写的是与 `naive` 的区别;"前提"指该方案是否建立在 `batch_reqs_async_load_submit` 之上。

2026-10-13 起 `split_async_load_submit` 改名为 `split_async_load_submit_first_n_batch_reqs`(行为不变,旧名启动时报错;PR 分支暂保留旧名),`priority_3level` 同样拆成两个变种。下文未带后缀的 `split_async_load_submit` / `priority_3level` 指该方案的两个变种共有的部分。10-13 预留的 `priority_3level_first_n_batch_reqs`(尾也跨请求共用任务)10-22 决定不做,枚举值待删,由 `_remaining_by_layer` 两个变种取代(§4.3 末尾)。

`split_priority_2level` 不是 `split_async_load_submit` 与 `priority_3level` 的叠加:它沿用 split 的提交方式,但优先级是另一套 2 级固定方案(对比见 4.4)。

### 4.1 公共前提:`batch_reqs_async_load_submit`

即已知问题 K1 的改法(§3)。它只决定同一 step 内多个 async 请求怎么排,与"头尾怎么提交、队列有没有优先级"无关,所以作为后三个方案的公共前提先做;`naive` 保持原样作为对照组。它单独也有可测的效果:§11.3 `naive-s1` step 2 中后几个请求数百毫秒的排队应当消失。

### 4.2 `split_async_load_submit`

新请求只提交头(前 N 层)。头可能要多个 step 才加载完:`get_finished()` 每个 step 用 `wait=False` 检查,前 N 层全部完成时 promote(上报 `finished_recving`),同时把请求加入"待补尾"集合。每次 `start_load_kv()` 都检查该集合,把其中所有请求(不论来自哪个 step)打成 batch 提交尾(第 N..L−1 层)。

```
start_load_kv
  1. sync 批                     (不变)
  2. 待补尾集合 → 打成 batch 提交尾
  3. 本 step 新到的 async 请求 → 打成 batch 提交头
forward
get_finished
  前 N 层完成的请求 → promote,并加入待补尾集合
```

步骤 2 在步骤 3 之前:刚 promote 的请求可能就在本 step 的 forward 里,forward 等的是它的尾;新请求的头不会被本次 forward 用到。

尾一定在请求被 forward 用到之前提交:每个 step 的顺序是 `start_load_kv` → forward → `get_finished`,promote 发生在 `get_finished`,请求最早进入下一个 step 的 forward,而那个 step 的 `start_load_kv` 先执行;no-forward step 也会调用 `start_load_kv`。

各请求的 N 可能不同(动态表),头和尾都按层分段:同一段内"需要这一层的请求集合"相同,每段调用一次 `get(layer_names=该段的层)`。头的第 j 层含 N > j 的请求,尾的第 j 层含 N ≤ j 的请求。N = −1(等全部层才完成)的请求把全部层当作头,没有尾。

效果:队列里只剩两类解压任务——"未 promote 请求的头(各 N 层)"和"已 promote 请求的尾(各 L−N 层,forward 正在等)"。(a)(b) 中"被堵 L 层"缩为"被堵 N 层"。

代价:尾失去 promote 前的提前解压时间。forward 算 layer i 时队列同时解压 layer i+1、i+2…,是流水线;**不卡的条件是单层解压时间 ≤ 单层 prefill 计算时间**。本轮不测这一条件(目前是软件解压,换硬件加速和 GPU 后结论会变)。

**头的两种提交方式**(两个变种只在这一点不同,尾都组 batch 按层提交):

| 变种 | 同一 step 来 A、B(N=4)时的队列 | 头完成时刻 |
|---|---|---|
| `_first_n_batch_reqs` | `[L0(A+B) L1(A+B) L2(A+B) L3(A+B)]` | A、B 同时,= 4 × (A+B 单层) |
| `_first_n_per_req` | `[A0 A1 A2 A3][B0 B1 B2 B3]` | A 先完成(4 × A 单层),B 与上一行相同 |

`_first_n_per_req` 让先到的请求先凑齐 N 层,与 §4.3 "P1 按提交序 FIFO"的理由相同;代价是每个任务的固定开销多付几次。v29 trace 实测(CPU 软件压缩)单层任务耗时与所含请求数近似成正比:覆盖 1 / 2 / 4 个请求时平均 1.73 / 3.29 / 6.33 ms,合并只省约 8%。头完成后要等 `get_finished()` 才 promote(v29 split 实测头完成 → promote 平均 66~77 ms),头完成时刻相差几毫秒时往往落在同一个 step、同时 promote,收益待实测。

### 4.3 `priority_3level`

一次性提交 L 层不变;`TaskQueue` 改为 3 级且优先级可变:

| 级别 | 内容 | 级内排序 |
|---|---|---|
| P0 | forward 正在等的:sync 批全部层 + 已进入 forward 的请求的尾 | 按进入 P0 的顺序 FIFO(10-14 定,见本节末尾) |
| P1 | 未 promote 请求的头 | 按提交序 FIFO |
| P2 | 尚未进入 forward 的请求的尾(预取) | 按提交序 FIFO |
| P3 | 压缩任务(KV 存入 DDR,现有 `PRIORITY_LOW`) | 按提交序 FIFO |

请求被 K3 搬入 active(本 step 的 forward 开始等它)时,对该请求的组调用 `set_group_priority(组号, P0)`(10-14 定,见本节末尾)。

P3 不变,只是写明:压缩排在所有解压之后。风险(推断):P2 预取持续占队列时压缩被压后,存储用的 pinned 中转池(ScratchPool)要等压缩完才释放,满了会让引擎报错退出;请求结束也要等它的存储完成。`batch_reqs_async_load_submit` 下尾全是 HIGH,同样压在压缩前面,不是本方案新引入的。

P1 用 FIFO 而非按层交错是有意的:头的目标是让**某一个**请求尽快凑齐 N 层去被调度;按层交错(A0 B0 A1 B1…)会让两个请求都拖到最后才凑齐。

P2 存在的唯一理由:让 B 的头(P1)能压过 A 尚未 promote 的尾。若 P1、P2 合并,A 的 48 层(先提交)与 B 的头(后提交)同级 FIFO,(a) 复现。

与公共前提的关系:同一 batch 的请求共用尾任务。batch 内各请求 N 相同时它们同时 promote;N 不同时,第一个 promote 的请求会把整批的尾提到 P0,其余请求的尾随之提前,无害。

**2026-10-13 讨论结论(先只做 `priority_3level_first_n_per_req`)**:

- **优先级存放在哪**:存在 `TaskQueue` 的排队条目 `Item` 上,不存进 `Context`。`Context`(每层一次 `get()` 生成的解压 / 拷贝对象,由 Python `Task.ctx` 持有)不加字段,只把参数原样往下传:connector → `KVStore.get` → `KVFlow.get` → `Context::unzip_from_mem` → `omp_queue().submit`。理由:排队顺序本来由队列管,优先级放在队列里,"按优先级取任务"和"改优先级"在同一个类、同一把锁内完成;不需要 §8.4 原方案的共享优先级 cell,也就没有 cell 的生命周期和 `Context` 手写 move 漏字段的问题。
- **任务粒度**:`Item` 是队列里的一个任务 = 一次 `get()` 中一层的解压任务,覆盖这次 `get()` 涉及的所有请求的 block。`_first_n_per_req` 下一次 `get()` 只含一个请求,所以一个 `Item` = (一个请求, 一层);`_first_n_batch_reqs` 下一个 `Item` = (同 step 一批请求, 一层)。
- **怎么改优先级(按组提权)**:每个任务带一个组号。`_first_n_per_req` 下一个请求一次 `get()`,组号就是请求的编号(Python 生成,记 `req_id → 组号`,请求结束时随现有清理一起删)。请求进入 forward 时(10-14 改,见下方补充)connector 调一次 `set_group_priority(组号, P0)`,队列持锁把该组仍在 P2 排队的任务移到 P0;此时头已执行完,不在队列里。已在执行或已完成的任务不受影响。队列不维护组表,无需清理。与原 cell 方案相比:原方案 promote 时 Python 要对该请求每个尾层的 `Task.ctx` 各调一次 `set_priority`(L−N 次 Python→C++ 调用);按组提权只调一次,由队列在锁内扫 P2 把同组的 L−N 个 `Item` 移走,逐个移动的工作仍在,但都在 C++ 内一次完成。不属于任何组的任务(sync 批、压缩、H2D/D2H 拷贝)组号为 0,提权不会碰到。
- **`Item` 字段**:现有 `fn / label / priority / traced / seq / enqueue_ns / high_depth / low_depth`;`priority` 改为 0..3(P0..P3);新增 `group`(组号);新增 `submit_priority`(提交时的优先级,trace 用,`priority` 记执行时的);`high_depth / low_depth` 换成 4 个数(入队时 P0..P3 各排着几个)。
- **P0 内顺序:先进先出,不按层号排序(10-14 定,适用于所有带 P0 的方案;10-22 对 `_remaining_by_layer` 新方案部分作废,见本节末尾)**:10-08 原设计是"P0 按 layer 号跨请求交错",用意是:A 的尾(A5..A63)已在 P0,本 step 来了 sync 请求 C,先进先出时 C0 要排在 A 的尾后面;按层号排则 C0..C4 先做、之后 A/C 按层交替,让 A、C 同一次 forward 能更早开始。不做的理由:C 的第 0 层排在更早进入 P0 的 A 的尾后面是预期行为,先到的先加载完;我们不追求把先后到达的请求凑成一个大 batch 同时就绪。不做省掉的东西:`Item` 不加层号字段;P0 用普通 deque,不需要有序容器;`get()` / `unzip_from_mem` 不透传层号;判定脚本的排序键不含层号;不需要按层排序的单测。
- **`_first_n_batch_reqs` 变种(以后做;10-22 作废,见本节末尾)**:同一 step 内各请求 N 不同时(按到达时的在途数查映射表,可能一个 N=4、一个 N=8),按层合并会出现"同一层对 A 是尾、对 B 是头"的混合任务。做法:复用 split 的 `_first_n_layer_segments()` / `_remaining_layer_segments()` 分段,每个任务要么是头要么是尾,不出现混合任务。

**2026-10-14 补充(实现时定下的细节)**:

- **提权时机:请求进入 forward 时,不在 promote 时**。§11.5 实测几乎所有 async 请求 promote 后都要再等一个 step 才被调度(held)。若 promote 时就提到 P0,不在本次 forward 里的请求会占住 P0,排在本 step 的 sync 批前面。改为在 `start_load_kv()` 的 K3 循环里(请求被搬进 `_active_promoted_tasks`)提权:代码量与 promote 时提权相同,只是调用位置不同;该循环在 forward 之前执行,来得及。提权后删掉 `req_id → 组号`。
- **P0 顺序 = 进入 P0 的顺序**:被提权的任务按原顺序追加到 P0 末尾,排在已在 P0 的任务(如更早提交的 sync 批)之后。实现就是 deque 尾部追加。
- **trace 字段**:`priority`(执行时所在级别)、`submit_priority`(提交时级别)、`group`、`p0_enter_ns`(被 `set_group_priority` 移入 P0 的时间,未移过为 0)、`depth[4]`(提交时 P0..P3 各排着几个)。`enqueue_ns` 与 `p0_enter_ns` 都在队列锁内取,二者先后与进入级别的先后一致。判定脚本用它们还原每个任务在任一时刻所在的级别(§7.6)。
- **数值**:`PRIORITY_HIGH = P0 = 0`,`PRIORITY_LOW = P3 = 3`(原来 LOW = 1);旧的两级调用方行为不变。
- **已知问题(记录,不处理)**:
  - 远程存储(`IAXL_RDMA_ENABLE`,`KVStoreRemote`)没有 OMP-Main 优先级队列:选 `priority_3level_*` 且开远程存储时启动直接报错,以后再处理。
  - 请求在进入 forward 前结束(abort)时,它的尾一直留在 P2,排在所有 P1 后面;结束清理要等这些尾做完,期间该请求的 GPU block 不释放。`naive` / `batch` 下尾是 HIGH,没有这个问题。

**2026-10-22 讨论:本次 forward 的 P0 任务(激活请求的尾 + sync 批)按层排(10-22 已实现,实现细节见本节末尾)**

只新增两个方案,都在请求进入 forward 时提权;不做 promote 时提权的按层版本(它会让 P0 跨 step 残留,见下方 sync 批一条的实测)。

- **问题**(代码事实 + 实测):`set_group_priority` 一次提一个组,被提的任务按原顺序追加到 P0 末尾。同一次 forward 激活 A、B 时 P0 是 `[A 的尾][B 的尾]`,forward 算第 k 层要等 B_k,而 B_k 排在 A 的全部尾后面,即已知问题 K1 第 2 点在 P0 里重现。尾在请求激活后等同 sync(forward 逐层等它),所以必须按层排。实测:§11.13 prio 的 10 请求激活步在第一个尾层停了 572 ms,期间 OMP-Main 86% 忙于 P0 解压。
- **只在提交时按层交错不够**:prio 的尾在请求到达时就以 P2 提交,同一次 forward 激活的请求多半在不同 step 到达,提交时凑不到一起;forward 等的是 P0 里的顺序,它在提权时决定,逐组提权会重新变成按请求排。
- **做法(做法 A)**:
  1. 队列任务带层号(排序键,无层号为 −1),由 `get()` 透传,层号取该层在全部层中的位置。
  2. 新增一次提权多个组:持锁把这些组仍在 P0 / P1 / P2 排队(未开始执行)的任务取出,按(层号,提交顺序 `seq`)排序,追加到 P0 末尾,记 `p0_enter_ns`;正在执行或已完成的任务不动。取 P0 里的任务是为了把本 step 的 sync 批一起重排(见下方 sync 批一条)。同一层内各组谁先谁后不影响 forward(第 k 层要等齐),实现时用 `seq` 而不是组在参数里的先后,这样判定脚本只凭 trace 就能还原顺序。
  3. connector 在 K3 激活循环里收集本 step 激活的全部组,加上本 step sync 批的组,循环后调用一次,替代逐请求提权;本 step 没有激活请求时不调用(sync 批本身已按层)。

  P2 预取保留(§11.10:960 个尾任务中 566 个在提权前已预取完)。P0 在不同批之间仍先进先出,只在同一次提权内部按层排。与 10-14 的关系:"先进 P0 的任务不被后到的插队"不变(10-14 设想的场景——上一个 forward 的尾还在 P0 时来了新的 sync 请求——在 prio 下不会出现,见下方 sync 批一条);"`Item` 不加层号、`get()` 不透传层号、不需要按层排序的单测"这几条对新方案作废。当时只考虑了不同 forward 之间的先后,漏了同一次 forward 激活多个请求时 P0 内的顺序。
- **不采用**:
  - 尾也跨请求共用任务(原 `_first_n_batch_reqs` 设想):一个尾任务属于多个请求,组号映射要改成一对多,且 A 激活会把同段 B 的尾也提进 P0(这正是 10-14 把提权从 promote 改到进入 forward 时要避免的)。
  - 尾推迟到激活时才按层提交(split 式):丢掉 P2 预取。
- **头**:两种都实现、实测后取舍。逐请求:先到的请求最早凑齐 N 层、最早 promote;同 step 合并:按 `_first_n_layer_segments()` 分段(各请求 N 不同时不出现混合任务),同 step 的请求同时 promote,任务数少(任务 release 占 OMP-Main 忙碌时间 18–21%,见 [task-release-overhead.zh-CN.md](task-release-overhead.zh-CN.md))。提交方式:头逐请求的变种沿用现在 prio 的提交(每个请求一次 `get()`,前 N 层 P1、其余 P2,一个组号),只改提权;头合并的变种把头、尾分开提交——头按段合并 `get()`(P1),尾每个请求一次 `get()`(P2,带组号),请求的 Task dict = 头任务(可能共用)+ 自己的尾任务,合并方式同 split。
- **尾的任务粒度**:一个任务 = (一个请求, 一层),请求间不合并成一个任务。请求到达时一次 `get()` 以 P2 提交它的第 N..L−1 层,一层一个任务,带该请求的组号(与现在的 prio 相同)。请求间的按层顺序在激活时由多组提权排出:同一 step 激活 A、B、C 时 P0 为 `A_N B_N C_N A_{N+1} B_{N+1} C_{N+1} …`。尾不在到达时就跨请求按层提交,是因为同一次 forward 激活的请求多半在不同 step 到达。
- **名字**:`priority_3level_first_n_per_req_remaining_by_layer`(头逐请求)、`priority_3level_first_n_batch_reqs_remaining_by_layer`(头合并)。名字只提到尾,但两个方案都把同 forward 的 sync 批一起按层排(10-22 定,名字不改)。预留未实现的 `priority_3level_first_n_batch_reqs` 从枚举中删除。
- **sync 批一起按层排(10-22 定,两个新方案内置,不单列方案、不加开关)**:`start_load_kv()` 先以 P0 提交本 step 的 sync 批(一层一个任务,覆盖所有 sync 请求),再提交 async,最后在 K3 循环里提权,所以 P0 是 `[sync 批未执行的部分][提上来的尾]`,forward 第一个尾层要等 sync 批剩下的全部层,即 sync 与尾之间的 K1 第 2 点:

  | 排法 | 提权后 P0 的顺序 | forward 第 N 层(第一个尾层)要等 |
  |---|---|---|
  | 现在的 prio | `S_0…S_79 │ A_N…A_79 │ B_N…B_79` | S 剩下的全部层 + A 的全部尾 + B_N |
  | 只排尾 | `S_0…S_79 │ A_N B_N A_{N+1} B_{N+1} …` | S 剩下的全部层 + A_N + B_N |
  | 尾与 sync 一起按层(采用) | `S_0…S_{N−1} │ S_N A_N B_N │ S_{N+1} A_{N+1} B_{N+1} …` | S_N + A_N + B_N |

  列说明:`S` 为本 step 的 sync 批,`A`、`B` 为本次 forward 激活的 async 请求,`X_k` 为其第 k 层的解压任务,N 为头层数,共 80 层;`│` 只为分组显示。"要等"指 forward 算第 N 层前 OMP-Main 必须先做完的任务。

  为什么按层对本次 forward 最优(推理):前提 (a) P0 由单个 OMP-Main 串行执行,(b) 提权时 P0 里的任务都属于本次 forward(见下方代码与实测)。forward 开始第 k 层的最早时刻不早于"第 0..k 层全部任务做完",按层排时这批任务对每个 k 都是队列前缀,每层都达到这个下界;sync 请求与 A、B 同在一个 forward,第 k 层本来就要等齐三方,没有损失。

  实现:sync 批提交时也带组号(仍为 P0),多组提权时把它仍在 P0 排队的任务一起取出重排。不采用的替代:把 sync 提交挪到提权之后(FIFO 下 sync 排到尾后面,同样不对);先以低级别提交再一起提权(两步之间 OMP-Main 可能先取 P1)。不加"关掉 sync 合并"的实验开关,性能只在四个 priority 方案之间比较。

  出现频率:`use_async = num_new_tokens > 0 and selected_layers != 0`,固定 N 时没有 sync 请求,只有动态 map 含 0 档时才有(如 map B 在途 ≤ 3)。实测 `hit98diag_20261020_030016` prio rank 0(8k c16,map `0-3:0,4-6:4,7-:8`):有 sync 批的 step 7 个,有激活的 step 71 个,两者同 step 0 个。10-21 的 32k 扫描未开事件日志,无法统计;新方案的性能测试开事件日志统计。

  step 开始时 P0 是否为空(前提 b 的依据):
  - 代码:prio 下进入 P0 的只有 (1) 本 step 的 sync 批(`KVFlow.get()` 不传 `priorities` 时各层都是 0;prio 的 async `get()` 只用 P1 / P2;压缩与解压重试在 P3),(2) K3 激活循环里的 `set_group_priority(…, P0)`,对象是本次 forward 调度的请求。forward 的 `wait_for_layer_load()` 对 sync 批和每个激活请求逐层 `get_wait(wait=True)`,解压任务完成才返回;所以只要该 forward 对每层都调用了 `wait_for_layer_load()`,本 step 进入 P0 的任务在 forward 结束时都已执行完。promote_raise 在 `get_finished()` promote 时就提权,已 promote 但未被调度的请求的尾会跨 step 留在 P0。
  - 实测(`hit98diag_20261020_030016`,8k c16,map `0-3:0,4-6:4,7-:8`,rank 0 / 3 / 7;脚本 `tests/async_load_dev/analysis/p0_at_step_start.py`,统计每个 `step_start` 时刻已进 P0 但未开始执行的解压任务):prio 在 hit 98 / 100 的 802 / 806 个 step 开始时 P0 都是空的;promote_raise 在 hit 98 有 19–20 个、hit 100 有 76–83 个 step 开始时 P0 非空,最多 571 个任务。
  - 所以对两个新方案,把 sync 批一起按层排就是"本次 forward 要等的全部 P0 任务按层排"。
- **实现规模(估计)**:`task_queue.h` 约 60–80 行(现有 `set_group_priority` 的扩展);`zip.cpp` / `torch_ext.cpp` / `flow.py` / `kvstore.py` 透传层号与多组提权接口;connector 约 20–40 行(头合并的变种另加分段,复用 split 代码);单测约 6 项 + TSan;§7 判定脚本认识新的 P0 顺序。
- **预期(推断)**:实现正确的把握约 85%;实测有提升的把握约 60%——hit 95 / 98 下同一 step 激活多个请求时,第一个尾层的停顿可从数百 ms 降到约(请求数 × 1 ms),省下的是 GPU 空闲时间;hit 100 时加载是瓶颈,两种顺序结束时间接近。同一 step 激活多个请求在 32k c4 / c8 下有多常见,尚未统计。
- **实现(10-22)**:
  - `TaskQueue`([task_queue.h](../../iaxl/csrc/include/task_queue.h)):`submit()` 多一个参数 `order_key`(默认 −1),存在排队条目上并写进 trace;新增 `set_groups_priority(groups, priority)`,做法即上面第 2 步,返回追加的任务数。旧的 `set_group_priority` 不变。trace 的 `p0_enter_ns` 含义改为"最后一次被追加到 P0 的时间":被重排的 sync 任务提交级别就是 P0,但也会记 `p0_enter_ns`。
  - 透传:`Context::unzip_from_mem(..., order_key)` → `KVFlow.get(order_keys=)` → `KVStore.get()`。`KVStore.get()` 对所有方案都把该层在 `layer_names` 中的位置作为 `order_key` 传下去(只影响 trace 与 `set_groups_priority`);Python 新增 `task_queue_set_groups_priority` / `KVFlow.set_groups_priority` / `KVStore.set_groups_priority`。
  - connector([kvshrink_connector.py](../../kvshrink/kvshrink_connector.py)):两个新方案下 sync 批每个 step 分配一个组号;K3 循环收集激活请求的组号,循环后连同 sync 组调用一次 `set_groups_priority(…, P0)`。头逐请求的变种复用 `_submit_priority_loads()`;头合并的变种为新函数 `_submit_priority_batch_head_loads()`:头按 `_first_n_layer_segments()` 分段提交(P1,组号 0,日志 `kind=async`),尾每个请求一次 `get()`(P2,带组号,日志 `kind=async_remaining`),都在请求到达的 step 提交。`_submit_layer_segments()` 加了 `priority` 参数(默认 P0,split 行为不变)。枚举删除 `PRIORITY_3LEVEL_FIRST_N_BATCH_REQS`,该名字启动时报错。
  - 判定脚本([async_load_queue_check.py](../../tests/async_load_queue_check.py)):两个新方案下额外检查每个任务 `order_key` = 层号;sync 任务组号非 0 且只属于本 step 的 sync 批,只能在本 step 的激活里被重排;同一次激活提上来的任务只有一个 `p0_enter_ns`(一次调用);头合并变种的头为 P1 / 组号 0 / 不被提权,尾覆盖第 N..L−1 层且与头同一 step 提交。排序键加入 `order_key`:`(级别, 进入该级别的时间, order_key, seq)`。报告新增 `p0_raise_calls` / `p0_raise_calls_multi_group` / `p0_raise_calls_with_sync`(按 `p0_enter_ns` 归并的提权调用次数、其中含 ≥ 2 个组的次数、含 sync 组的次数)。用改后的脚本重跑 `v29p3-priority-s1/s2`、`v29fnpr-split-s1/s2`,结论与原报告相同(PASS / PASS / PASS / UNCOVERED)。
  - 单测:`task_queue_test` 48 项全部通过(原 35 项 + 新 13 项:按层排序且同组已在 P0 的任务一起重排、其他组不动、空组 / 组 0 / 未知组 / 正在执行的任务不移动、非法级别报错、提到 P1 不记 `p0_enter_ns`、并发提交与多组提权);TSan 0 告警。

### 4.4 `split_priority_2level`

按 `split_async_load_submit` 提交:尾在 promote 前不进队列,队列里不存在"未 promote 的尾"这类任务,因此**不需要 P2,也不需要改优先级**。promote 后尾直接以 P0 提交。队列只需:

| 级别 | 内容 | 级内排序 |
|---|---|---|
| P0 | sync 批 + 已 promote 请求的尾 | FIFO |
| P1 | 未 promote 请求的头 | FIFO |

优先级在提交时指定,之后不变。

与 `priority_3level` 的区别:

| | `priority_3level` | `split_priority_2level` |
|---|---|---|
| 优先级级数 | 3(P0/P1/P2) | 2(P0/P1) |
| 优先级能否改 | 能:请求进入 forward 时 `set_group_priority` | 不能:提交时定死 |
| 尾在 promote 前预取(P2) | 有 | 无 |
| P0 内顺序 | FIFO | FIFO |

## 5. 方案对比

"(a)(b)" 两行写的是 B 的头 / C 的 layer 0 **前面最多有多少无关任务**。`naive` 列是现状;其余三列都建立在 `batch_reqs_async_load_submit` 之上。

| 维度 | `naive` | `split_async_load_submit` | `priority_3level` | `split_priority_2level` |
|---|---|---|---|---|
| **(a)** B 头前面的无关任务 | 前面每个 async 请求各 L 层 | 前面未 promote 请求各 **N** 层 + 已 promote 请求各 L−N 层(后者 forward 正在等,排前面合理) | 只等前面请求的头(P1)和 P0;所有未 promote 尾(P2)排 B 后面 | 同 `priority_3level` |
| **(b)** C 的 layer 0 前面的无关任务 | 前面每个 async 请求各 L 层 | 前面未 promote 请求各 **N** 层 + 已 promote 请求各 L−N 层 | 只有 P0 中更早提交的任务(已 promote 请求的尾、更早的 sync 批);P1 / P2 都排在 C 后面 | 同 `priority_3level` |
| 尾的提前解压时间 | 有 | **无**,靠流水线 | 有(P2 利用队列空闲预取) | 无,同 `split_async_load_submit` |
| 尾不卡 forward 的条件 | 宽松 | 单层解压 ≤ 单层 prefill 计算 | 宽松 | 单层解压 ≤ 单层 prefill 计算 |
| 队列优先级 | 2 级固定(HIGH/LOW) | 2 级固定(HIGH/LOW) | 3 级**可变** | 2 级固定 |
| C++ 改动 | — | 无 | 4 级 + 按组提权 `set_group_priority` | 2 级;提交接口加 priority 参数 |
| Python 改动 | — | worker 保存 async 请求元数据;promote 后按 batch 补提交尾 | `get()` 加 per-layer priority;请求进入 forward 时提权 | `split_async_load_submit` 的改动 + `get()` 加 priority 参数 |

结论:`split_async_load_submit` 单独能把 (a)(b) 从 L 层压到 N 层;`priority_3level` 解决 (a)(b) 但需可变优先级;`split_priority_2level` 解决 (a)(b) 且队列只需 2 级固定优先级,代价是尾没有提前量(本轮不评估,见 4.2)。P0 内不按层号排序,先进先出(10-14 定,理由见 4.3 末尾)。另外,已知问题 K2(§3)的延迟发生在任务进队列之前,本文方案对它无效。

## 6. 会不会让 async 请求排不上号?

担心:P0 永远压在 P1 上,async 请求的头一直做不了。

P0 的任务来源:

1. **sync 请求**:按 dynamic map,并发到阈值以上新请求才走 async,高并发时 sync 来源很少。
2. **已 promote 请求的尾**:一个 async 请求只有在它的头(P1)做完后才会 promote,然后才产生尾的 P0 任务。即 **P0 的第二类来源是 P1 完成的结果**。若 P1 被饿着,就没有新 promote,P0 很快空,P1 就能跑。自限闭环,不会永久饿死。

会出现的是**短时延迟**:A 刚 promote,其 L−N 层尾进 P0,B 的头要等这些做完。这是有意取舍:A 的尾卡的是正在跑的 forward(整个 batch GPU 停转),B 的头卡的只是 B 一个请求的 TTFT。

## 7. 队列顺序验证测试

### 7.1 目的

只验证一件事:**OMP-Main 单线程队列的提交结构和执行顺序,是否与各方案的预期一致**。不看 TTFT,也不测解压耗时与 prefill / decode 耗时的对比(目前是软件解压,以后会换硬件加速和更高端的 GPU,这些数字没有参考价值)。

### 7.2 负载

用 `tests/vllm-benchmark.sh`(`vllm bench serve`,random 数据集)产生请求,由 `tests/async-load-queue-test.sh s1|s2 <输出目录>` 串起来(开 trace → 跑 benchmark → 取 trace 和本次 log → evict DDR → 判定)。该数据集一次运行内所有请求共用同一个前缀(长度 = 输入长度 × 命中率)。

| 参数 | S1 | S2 | 说明 |
|---|---|---|---|
| 输入长度 / 输出长度 | 4k / 128 | 4k / 128 | 8 × 4.1k 远小于 GPU7 的 KV 容量 66k token,不会触发抢占(connector 不支持抢占恢复) |
| 命中率 | 95% | 95% | 每个请求 64 层 × 约 240 个 block,单个请求全部层解压约 115 ms |
| 并发 / 请求数 | 8 / 16 | 8 / 32 | |
| 请求速率 | 10 req/s | inf | S1:平均间隔 100 ms,小于单个请求的加载时间,且请求分散在不同 step;S2:用 benchmark 默认值,不为凑覆盖去调低速率(低速率的负载没有实际意义) |
| warmup | 1 | 1 | 由 warmup 请求承担未命中、把前缀存进 DDR |

实测修正(§11.3):速率为 inf 时前 8 个请求落在同一个 step,补进的请求在前面结束后(约 8 s)才到,队列早已空,覆盖不到 (a);warmup 为 0 时第 0 个测量请求是未命中,要做完整 prefill,其余请求在这个长 step 里到达并堆进下一个 step,同样覆盖不到 (a)。

### 7.3 场景与环境变量

启动 vLLM 时用 `export` 覆盖,不改 `setvars.sh` 默认值。

| 变量 | S1:全部 async | S2:sync 与 async 混合 |
|---|---|---|
| `KVSHRINK_VLLM_KV_ASYNC_LOAD_ENABLED` | 1 | 1 |
| `KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC` | 0 | 1 |
| `KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS` | 4 | -1(不使用) |
| `KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP` | 不使用 | `0-3:0,4-:4` |
| `KVSHRINK_ASYNC_LOAD_SCHEME` | 每次运行一个方案 | 同 S1 |

- S1:每个命中请求都走 async,N=4。覆盖"同一 step 多个 async 请求"(K1)和"后到 async 请求碰上前面的尾"((a))。
- S2:scheduler 在途请求数(`len(self._req_states)`,含当前请求)为 1~3 时走 sync,≥4 时走 async 且 N=4,与实际配置一致(低并发 sync,高并发 async)。
- S2 覆盖 (b) 的难点(推断):async 请求 A 提交时除 A 外至少有 3 个在途请求;之后到达的 C 要走 sync,除 C 外最多 2 个在途(含 A)。即在 A 的尾还在队列里的约 115 ms 内,A 之外的 3 个请求至少要有 2 个结束。实测 0.5 req/s、32 个请求一次都没出现(§11.3)。S2 改用速率 inf;覆盖不到 (b) 就记为未覆盖,不再为凑覆盖调负载。

### 7.4 各方案的预期队列

记号:`Rk_j` = 请求 k 第 j 层的解压任务;`Lj(R1..R8)` = 一个覆盖 R1~R8 的第 j 层任务;`S` = sync 批;L = 64,N = 4。

**S1:R1~R8 同一 step 到达**

| 方案 | 队列 |
|---|---|
| `naive` | `[R1_0..R1_63][R2_0..R2_63]…[R8_0..R8_63]`,512 个任务;R8_0 排第 449 位 |
| `batch_reqs_async_load_submit` | `[L0(R1..R8)]…[L63(R1..R8)]`,64 个任务;8 个请求在 L3 完成后同时 promote |
| `split_async_load_submit` | 本 step `[L0..L3(R1..R8)]`,4 个任务;promote 后下一个 step `[L4..L63(R1..R8)]`,60 个任务 |
| `priority_3level` | 同 batch,64 个任务;L0~L3 为 P1,L4~L63 为 P2,promote 后提到 P0(10-14 实现的 `_first_n_per_req`:每个请求一次 `get()` 共 64 个任务,P2 在请求进入 forward 时提到 P0) |
| `split_priority_2level` | 同 split;头为 P1,尾为 P0 |

实际上 8 个请求不一定落在同一个 step;判定时以 7.5 的 step log 为准,对每个 step 套用上表。

**S1:补进的请求 R9 到达时,R9_0 前面允许有什么**

| 方案 | 允许排在 R9_0 前面的任务 |
|---|---|
| `naive` / `batch_reqs_async_load_submit` | 队列里所有尚未执行的任务(FIFO) |
| `split_async_load_submit` | 更早提交的头 + 已 promote 请求的尾;**不能有未 promote 请求的尾** |
| `priority_3level` | P0 + 更早提交的 P1;**不能有 P2** |
| `split_priority_2level` | P0 + 更早提交的 P1 |

**S2:sync 批 S 提交时,队列里仍有 async 任务((b))**

| 方案 | 允许排在 S0 前面的任务 |
|---|---|
| `naive` / `batch_reqs_async_load_submit` | 队列里所有尚未执行的任务,包括未 promote 的尾 |
| `split_async_load_submit` | 更早提交的头 + 已 promote 请求的尾 |
| `priority_3level` / `split_priority_2level` | 正在执行的任务 + P0 中更早提交的任务(已 promote 请求的尾、更早的 sync 批);**不能有 P1 / P2** |

### 7.5 要看的 log

| log | 内容 | 状态 |
|---|---|---|
| `GET /v1/cache/queue_trace` | 每个解压任务的 `label=unzip\|<req_id 列表>\|<层名>`、`priority`、`seq`、`enqueue_ns / start_ns / end_ns`;按 `start_ns` 排序即实际执行顺序。10-14 起另有 `submit_priority` / `group` / `p0_enter_ns` / `depth`(§4.3 末尾) | 已有 |
| connector `get_num_new_matched_tokens` | 每个请求实际走 sync 还是 async、N 是多少 | 已有 |
| connector `start_load_kv` | 每次 `get()` 一行:`async_load step=<k> t=<ns> submit kind=sync\|async reqs=<id,...> layers=<a>-<b>` | 已加 |
| connector `get_finished` | `async_load step=<k> t=<ns> promote reqs=<id,...>` | 已加 |
| connector `start_load_kv`(K3) | `async_load step=<k> t=<ns> activate reqs=<本次搬入 active 的请求> held=<留在 early 的请求>`;`priority_3level` 在打这行之前提权 | 已加 |

`t` 是 `time.monotonic_ns()`,与 trace 同一时钟。sync 批的 description 也改为逗号分隔的 req_id,trace 里每个任务都能对应到具体的 submit 行。

`priority_3level` 的优先级会中途改变:trace 的 `priority` 是执行时所在级别,`submit_priority` 是提交时级别,`p0_enter_ns` 是被提权移入 P0 的时间。

### 7.6 判定

判定脚本读取 7.5 的 log,对每次运行输出三项:

1. **提交结构是否与 7.4 一致**:每个 step 的任务数、层范围、req_id 组成。例:`batch_reqs_async_load_submit` 同一 step 恰好 64 个任务且 label 含该 step 全部 async 请求;split 类方案中,任何请求在 promote 记录之前不得有 L4 及以上的任务。
2. **违序数**:用 trace 重放队列,对每个任务检查它开始执行的那一刻,是否有"已入队、未开始、排序键更小"的任务。排序键:FIFO 类方案 `(HIGH/LOW, seq)`;优先级类方案取任务**在该时刻**的 `(所在级别, 进入该级别的时间, seq)`:被提权的任务在 `p0_enter_ns` 之前处于提交级别(进入时间 = `enqueue_ns`),之后处于 P0(进入时间 = `p0_enter_ns`)。必须为 0。
3. **场景覆盖数**:S1 中"R9 类请求到达时前面有尾"的次数;S2 中"sync 批提交时队列里有 async 任务"的次数。

每次运行的结论只有三种:

| 结论 | 条件 |
|---|---|
| PASS | 结构错误 0、违序 0、覆盖数 > 0 |
| FAIL | 结构错误 > 0 或违序 > 0 |
| 未覆盖 | 结构错误 0、违序 0,但覆盖数 = 0:队列行为没有错,只是负载没造出要测的场景 |

每个方案 S1、S2 都要跑,无论之前的方案是否覆盖到。不同方案的提交方式不同,同一负载下造出的场景也可能不同。

先在 `naive` 上跑 S1、S2,确认预期和判定脚本本身是对的,再用于其他方案。

## 8. 实现复杂度评估

决定:`naive` 作为对照组保持不动;`batch_reqs_async_load_submit` 作为公共前提先做;`split_async_load_submit`、`priority_3level`、`split_priority_2level` 都实现,都测性能。下面按"公共部分 + 各方案增量"拆分。

### 8.1 公共基础设施(已完成,见 §11.1)

| 项 | 内容 | 为什么必须 |
|---|---|---|
| 队列可观测性 | `TaskQueue` 记录每个任务的入队 / 开始 / 结束时间,按 (priority, tensor_key, description) 打标签,经 `GET /v1/cache/queue_trace` 读取 | 没有它,(a)(b) 的"排队等待时间"无法测量,方案对比没有证据 |
| 运行时开关 | 环境变量 `KVSHRINK_ASYNC_LOAD_SCHEME=<4.0 中的名字>`(默认 `naive`;未实现的名字启动时报错) | 一次编译切换对比,避免每测一个方案重编一次扩展 |
| 队列顺序验证(§7) | connector 的 step / promote log;`vllm-benchmark.sh` 参数可覆盖;判定脚本 | 各方案验收的证据 |

### 8.2 `batch_reqs_async_load_submit`(公共前提,纯 Python)

改动全在 `kvshrink/kvshrink_connector.py`:

| # | 位置 | 改什么 |
|---|---|---|
| 1 | `start_load_kv()` async 提交处 | 把 `async_reqs` 的 block_ids / block_hashes 合并,调一次 `get()`;description 用逗号分隔的 req_id 列表,trace 里可以看出 batch 由哪些请求组成 |
| 2 | 同上 | batch 内每个请求的 `_pending_load_tasks[req_id]` 指向这同一组 Task;`_pending_load_layers[req_id]` 仍按请求记录 N |
| 3 | `get_finished()` promote 分支 | 不改:各请求按自己的 N 判断 promote;对同一组 Task 重复等待安全(`get_wait` 遇到 `ctx is None` 直接跳过) |
| 4 | `get_finished()` deferred-finish 分支 | 不改:请求结束时等整组 Task 完成 |

- 规模:~20 行 Python。复杂度:**低**。
- 风险:同 batch 请求越多,单个请求的头越晚就绪;abort 一个请求时要等同 batch 其他请求的 block 也加载完(见 §3 K1 约束表)。

### 8.3 `split_async_load_submit`(纯 Python)

改动全在 `kvshrink/kvshrink_connector.py`。术语:"Task dict" = `self._store().get(...)` 的返回值 `Dict[层名, Task]`,`Task.ctx` 是该层解压任务的 C++ `Context`;connector 的 `_pending_load_tasks` / `_early_promoted_tasks` / `_active_promoted_tasks` 存的都是它。

| # | 位置 | 改什么 |
|---|---|---|
| 1 | `start_load_kv()` 提交头 | 按 4.2 分段提交头。**每个请求存一份自己的 Task dict 副本**:`_pending_load_tasks[r] = dict(head_tasks)`(只复制"层名 → Task"映射,Task 对象仍共用,不重复提交);同时把请求的 block_ids / block_hashes / N 存进"加载元数据" |
| 2 | `get_finished()` promote 分支 | 前 N 层完成时 promote(现有逻辑),同时把 req_id 加入"待补尾"集合 |
| 3 | `start_load_kv()` 提交尾(在提交头之前) | 取出待补尾集合的全部请求,按 4.2 分段提交尾;把返回的 Task **原地 `update`** 进每个请求自己的 dict(此时该 dict 已被 `_early_promoted_tasks` 引用,原地更新后无需再搬) |
| 4 | `wait_for_layer_load()` | 不改。`KVFlow.get_wait()` 有 `assert key in get_results`,第 3 步保证 forward 等第 N 层之前尾已并入 dict |
| 5 | `get_finished()` 处理已结束请求 | 现有逻辑不变(等该请求 dict 内任务全部完成再清理);另外把它从"加载元数据"和"待补尾"集合删掉 |

**为什么每个请求要自己的 dict**:同 batch 的请求可能 N 不同、promote 时刻不同。例:A 的 N=4,B 的 N=8,共用一个 dict = {L0..L7}(L4~L7 只含 B)。A 先 promote 并补尾(L4~L63 只含 A)后 `update`,B 原来的 L4~L7 被覆盖成只含 A 的任务,B 的 forward 等第 4~7 层时等的是 A 的任务,B 的数据可能还没加载完。各自一份 dict 就不会互相覆盖。

**为什么共用 Task 对象没问题**:`get_wait` 等完一个 Task 会把 `ctx` 置为 `None` 并释放 `cpu_tensors`;另一个请求再等同一个 Task 时看到 `ctx is None` 直接跳过。该层任务覆盖 batch 内所有请求的 block,完成一次即对所有请求都完成。

**新增状态**(采用方式 A:原有三个 dict 不动):"加载元数据"(每请求 block_ids / block_hashes / N;promote 后 `reqs_to_load` 里已没有该请求,补尾只能用它)和"待补尾"集合。`split_priority_2level` 复用,`priority_3level` 不需要。

**abort / 提前结束**(`get_finished` 中 promote 循环在处理已结束请求的循环之前,同一次调用内):

| abort 时刻 | 结果 |
|---|---|
| 头还在加载 | 头完成前每次都跳过;头完成的那次,先被 promote(进 early、进待补尾、进 `finished_recving`),同一次调用里又被清理(dict 只含头,已完成),从待补尾集合删掉,不会补尾 |
| 已 promote、尾未提交 | dict 只含头且已完成,当次清理,不会补尾 |
| 尾已提交 | dict 含全部层,等全部完成后才清理,保证解压不会写入已释放的 GPU block |

第一种情况里,已 abort 的请求仍会出现在 `finished_recving`;这是 `naive` 就有的现有行为,不在本文范围内。

- 规模:~100~140 行 Python。
- 前提:已知问题 K3 先修复(见 §3),否则尾提交的那个 step,不含该请求的 forward 会从头等整段尾。
- 复杂度:**低~中**。难点在状态清理,不在算法。

### 8.4 `priority_3level`(C++ 为主)

2026-10-13 起按 §4.3 末尾的讨论结论做(按组提权,`Context` 不改,先做 `_first_n_per_req`,P0 先按先进先出)。改动如下;其后的表格是原"可变优先级 cell"方案,保留作对比,不再采用。

| # | 文件 | 改什么(10-13 方案) |
|---|---|---|
| 1 | `iaxl/csrc/include/task_queue.h` | 4 个等待区 P0..P3;`Item` 加 `group`、`submit_priority`,depth 换成 4 个数;新增 `set_group_priority(group, priority)`,持锁把该组 P2 中的任务移到 P0。保留旧 `submit(fn, priority)`:`PRIORITY_HIGH` → P0,`PRIORITY_LOW` → P3,Record / H2D / D2H 队列与压缩任务行为不变 |
| 2 | `torch_ext/zip.cpp`、`torch_ext/torch_ext.cpp` | `unzip_from_mem` 加 `priority`、`group` 两个参数(pybind 给默认值,现有调用方不变);缓存未命中重试**不改**,仍以 `PRIORITY_LOW`(= P3,组号 0)入队,见下方"重试优先级";`Context` 不加字段 |
| 3 | `iaxl/kvflow/flow.py`、`iaxl/kvstore/kvstore.py` | `get()` 接收每层优先级和组号并透传;新增 `set_group_priority` 封装 |
| 4 | connector | `start_load_kv()`:sync → P0;每个 async 请求一次 `get()`,前 N 层 P1、其余 P2,组号按请求记录;请求被 K3 搬入 active 时提权到 P0(10-14 改,原为 promote 时);请求结束时删 `req_id → 组号`;开远程存储时启动报错 |
| 5 | `csrc/test/task_queue_test.cpp` | 单测:4 级取任务顺序、提权后顺序改变、旧 HIGH/LOW 调用行为不变、并发提权与入队(TSan) |
| 6 | `tests/async_load_queue_check.py` | 优先级会变:违序检查按任务**在该时刻**所在级别判断,需要 trace 的 `submit_priority` 与 `p0_enter_ns`;另查每个任务的提交级别、组号,以及提权时间介于 promote 与 activate 之间 |

10-14 已按上表实现,验证结果见 §11.10。

**重试优先级(10-14 定)**:解压任务在 `mem->get()` 取不到数据时重试,原因是这层 KV 的压缩任务还在队列里、数据还没写进 DDR。重试以 `PRIORITY_LOW`(P3)入队,与压缩同级先进先出,排在更早入队的压缩任务后面,压缩做完再取就有数据;第二次仍取不到会 `IAXL_CHECK` 退出。所以重试**不能**沿用原任务的 P0 / P1 / P2,否则会跑在压缩前面。重试组号为 0,提权只移动 P2,碰不到它。

已知行为(记录,不处理):重试排在**所有**已入队的压缩任务后面,不只是它要等的那一个。若重试的是 forward 正在等的任务(P0 的尾或 sync 批),而压缩积压较多,forward 要等整段压缩做完。这是现有行为,4 级方案下不变;实际发生频率未测(trace 中 `unzip_retry` 标签可统计),本轮不看。

原方案(不再采用):

| # | 文件 | 改什么 |
|---|---|---|
| 1 | `iaxl/csrc/include/task_queue.h` | 两个 deque 换成一个条目列表,每条 `{shared_ptr<atomic<int>> prio, int64 sort_key, uint64 seq, fn}`;worker 在锁内线性扫最小 `(prio, sort_key, seq)`。队列长度 ≤ 并发数 × 层数(几千),线性扫可接受。**保留旧 `submit(fn, priority)` 重载**——`kv_pool/record.cpp` 的 `Record` 和 H2D / D2H 队列也用这个类 |
| 2 | 同上 | 新增 `submit(fn, prio_cell, sort_key)` 重载;调用方持有 `prio_cell` 即可随时改 |
| 3 | `torch_ext/context.h` / `torch_ext/zip.cpp` | `unzip_from_mem` 加 `priority, sort_key` 参数;`Context` 持有 `prio_cell`;新增 `set_priority(int)`;**cache-miss 重试路径**(`UnzipFromMemWork::execute` 中重新 submit 处)要复用同一个 cell,不能再硬编码 LOW(10-14 注:此条是错的,重试必须排在压缩后面,见上方"重试优先级") |
| 4 | `torch_ext/torch_ext.cpp` | pybind:`unzip_from_mem` 加两个带默认值的参数;绑定 `set_priority` |
| 5 | `iaxl/kvflow/flow.py` `get()` | 加 `priorities` / `sort_keys`(按 tensor_key 的 dict),透传 |
| 6 | `iaxl/kvstore/kvstore.py` `get()` | 接收 per-layer priority;`sort_key = self.layer_names.index(name)` |
| 7 | connector `start_load_kv()` | sync → P0;async 前 N 层 → P1,其余 → P2 |
| 8 | connector `get_finished()` promote 分支 | 对该请求所有 `Task.ctx.set_priority(P0)`(ctx 可能已是 None,要判空) |

- 规模:C++ ~200~250 行(含重载兼容和重试路径),Python ~60 行,需重编扩展。
- 风险 1:线程安全。`set_priority` 从 Python 线程改 atomic,worker 在锁内读;任务已在执行时改优先级无效但无害,契约要写清楚。
- 风险 2:`Context` 的 move 构造 / 赋值要把 `prio_cell` 一起搬,漏了就是悬空。
- 风险 3:`TaskQueue` 的优先级排序和提权要新加单测(在 `iaxl/csrc/test/task_queue_test.cpp` 里扩展),覆盖"提权后顺序正确"。
- 风险 4:公共前提下尾任务被同 batch 请求共用,提权作用于整批(见 4.3)。
- 复杂度:**中~高**。难在可变优先级 cell 的生命周期(跟着 Context 走、重试复用、move 语义),不在排序本身。

### 8.5 `split_priority_2level`

代码上 = `split_async_load_submit` 的全部改动 + `priority_3level` 第 1、3、4、5、6 项的简化版(复用其 priority 透传路径,不用其提权):

| 项 | 相对 `priority_3level` 的简化 |
|---|---|
| `TaskQueue` | 不需要提权。3 个 deque:P0、P1、LOW(zip),均 FIFO。~50 行 |
| `unzip_from_mem` | 加 `priority` 参数即可,**不需要提权接口**,`Context` 不加新成员,move 语义不动 |
| 重试路径 | 不改,仍以 `PRIORITY_LOW` 入队,排在压缩后面(理由同 8.4"重试优先级") |
| connector | `split_async_load_submit` 改动之上:首次提交头传 P1,补提交尾传 P0;sync 传 P0。promote 分支不调任何提权接口 |

- 规模:C++ ~80~100 行,Python = `split_async_load_submit` + ~20 行。
- 风险:继承 `split_async_load_submit` 的状态管理风险;C++ 侧风险明显低于 `priority_3level`。
- 复杂度:**中**。

### 8.6 汇总

| | `batch_reqs_async_load_submit` | `split_async_load_submit` | `priority_3level` | `split_priority_2level` |
|---|---|---|---|---|
| C++ 行数(估) | 0 | 0 | 200~250 | 80~100 |
| Python 行数(估) | ~20 | 100~140 | ~60 | 120~160 |
| 需重编扩展 | 否 | 否 | 是 | 是 |
| 新增 worker 状态 | 无(同 batch 共用 Task dict) | 加载元数据 + 待补尾集合;每请求一份 Task dict 副本 | 无 | 同 `split_async_load_submit` |
| 新增 C++ 接口 | 无 | 无 | `submit` 重载、`set_priority`、pybind 两处 | `submit` 重载、pybind 一处 |
| 最难的点 | 共用 Task 时的结束 / abort 语义 | 状态清理(abort / deferred finish) | 可变优先级 cell 的生命周期 | 同 `split_async_load_submit` |
| 复杂度 | 低 | 低~中 | 中~高 | 中 |

列说明:行数是相对 `naive` 的增量估算;后三列都不含 `batch_reqs_async_load_submit` 自身的改动。

### 8.7 实现顺序

**公共基础设施(已完成)→ `batch_reqs_async_load_submit`(已完成)→ K3 修复 → `split_async_load_submit` → `split_priority_2level` → `priority_3level`**,原因:

1. `batch_reqs_async_load_submit` 是后三个方案的前提,改动小,收益已有实测依据(§11.3 `naive-s1` step 2)。
2. K3 不修,split 下不含该请求的 forward 会从头等整段尾;它对所有方案生效,单独提交。
3. `split_priority_2level` = split 的提交方式 + 一小块 C++,放在 `split_async_load_submit` 之后。
4. `priority_3level` 的 priority 透传(8.4 第 3~6 项)与 `split_priority_2level` 相同;做完后只剩"可变 cell + `set_priority` + promote 时提权"。
5. 全部挂在 `KVSHRINK_ASYNC_LOAD_SCHEME` 开关下;两种优先级共存于同一份 `TaskQueue`(可变 cell 对 `split_priority_2level` 而言就是"从不改"),一次编译跑五组对比(`naive` + 公共前提 + 三个方案)。

## 9. 验收标准与最小测试

本节区分两类内容:

- **验收标准**:某个方案算“做完”的定义。
- **最小测试**:不追求完整 benchmark,只用于快速证明“这个方案的核心机制确实生效”。

### 9.1 公共验收前提(所有方案都适用)

无论哪个方案,都先满足以下前提,否则不能进入性能对比:

| 项 | 验收标准 | 简单测试 |
|---|---|---|
| 运行时切换 | `KVSHRINK_ASYNC_LOAD_SCHEME=<名字>` 能切到 4.0 的五种实现,不需改代码重编 | 同一二进制下分别启动 5 次,日志打印当前 scheme 名字 |
| 可观测性 | 能记录每个 unzip 任务的 `enqueue_ts`、`start_ts`、`end_ts`;至少能按请求 ID、layer、priority 追踪 | 构造 1 个 async 请求,检查日志或 metrics 中能看到完整三元时间戳 |
| 功能正确性 | async load 不改变模型输出;cache hit 前后生成文本一致 | 固定 seed、`temperature=0`,同一请求跑“无命中”和“有命中”两次,逐字节比对输出 |
| 无明显泄漏 | 请求 finish / abort 后,worker 侧与该请求相关的状态能清空 | 构造 1 个 promote 前 abort、1 个 promote 后 abort,检查 connector 内状态计数恢复为 0 |

### 9.2 `batch_reqs_async_load_submit` 验收标准(公共前提)

| 维度 | 验收标准 | 简单测试 |
|---|---|---|
| 提交行为 | 同一 step 的多个 async 请求,每层只提交一个任务,覆盖所有请求的 block | queue_trace 中该 step 的 unzip 任务数 = 层数,label 的 description 含全部 req_id |
| promote 正确 | 各请求按自己的 N promote | 构造同一 step 的 A、B,检查两者都被 promote 且输出与未命中时逐字节一致 |
| 结束 / abort | 同 batch 中一个请求结束或 abort,其余请求不受影响,状态清空 | A、B 同 batch,abort A,检查 B 输出正确、connector 状态计数归 0 |
| 效果 | 同一 step 内多个 async 请求不再依次排队 | §7 S1 中 `steps_with_multiple_async_reqs` 的那些 step,各请求首个任务的等待不再随提交顺序递增;对比 §11.3 `naive-s1` step 2 的 6→701 ms |

### 9.3 `split_async_load_submit` 验收标准

核心不是"性能一定最好",而是"**尾不再与头同时入队**"。

| 维度 | 验收标准 | 简单测试 |
|---|---|---|
| 提交行为 | 新请求只提交前 N 层;promote 前尾(L−N 层)不在队列中 | §7 判定脚本:每个请求头的层都 < N;任何时刻 trace 里没有未 promote 请求的尾 |
| promote 后补提交 | 请求在某次 `get_finished()` promote 后,紧接着的下一次 `start_load_kv()` 提交尾 | 判定脚本:尾的 submit log 在该请求 promote log 之后的第一个 step;每个请求所有提交的层合起来恰好是 0..L−1 |
| 提交顺序 | 同一 step 内尾先于新请求的头提交 | 判定脚本:同一 step 的 submit log 中尾在头之前 |
| 任务合并正确 | 每个请求自己的 Task dict 在其 forward 等第 N 层前已含全部层,不出现 `key not in get_results` | S1/S2 运行无报错;输出正确性检查 |
| (a) 缓解生效 | 后到 async 请求 B 的头前面,不再有前面请求的整段 L 层尾 | 人工构造 A、B 两个 async 请求,打印任务入队顺序,确认 B0..B{N-1} 前面没有 A_N..A_{L-1} |
| (b) 缓解生效 | 后到 sync 请求 C 的 layer 0 前面,最多只会被未 promote 请求的头和已 promote 请求的尾挡住,不会再被“未 promote 尾”挡住 | 先提交 async A,再提交 sync C,检查 C0 前面不存在 A_N..A_{L-1} 这种未 promote 尾 |
| abort 清理 | promote 后但尾尚未提交时 abort,不会给已结束的请求补提交尾 | **不测**:abort 需落在"promote 之后、补尾之前"的一个 step 内,难以稳定构造;靠代码审查(见 8.3 abort 表) |

同一 step 内补提交的尾按 batch 提交,检查方法同 9.2 的"提交行为"。

### 9.4 `priority_3level` 验收标准

核心是"**任务可提权**"。P0 内先进先出,不验按层排序(4.3 末尾)。

| 维度 | 验收标准 | 简单测试 |
|---|---|---|
| 多级队列生效 | P0 / P1 / P2 任务进入队列后,worker 取任务顺序符合 `(priority, seq)` | 在 `task_queue_test.cpp` 里加用例:手工 submit 若干假任务,断言执行顺序与预期一致 |
| 可变优先级生效 | 任务入队后但尚未执行前,`set_group_priority(组号, P0)` 能改变其出队顺序 | 同上加用例:先入队 A(P2)、B(P1),再把 A 的组提到 P0,断言 A 先执行 |
| retry 排在压缩后面 | cache miss 重试的 unzip 任务以 P3 入队,不受提权影响 | 单测:队列中有 P3 压缩任务时按重试路径入队一个 P3 任务(组号 0),再对原任务的组提权,断言它仍在压缩之后执行 |
| (a) 缓解生效 | B 的头(P1)能越过 A 的未 promote 尾(P2) | 构造 A、B 两个 async 请求,查看队列执行顺序,确认 B0..B{N-1} 先于 A_N..A_{L-1} |
| (b) 缓解生效 | sync 批(C 的 S0)作为 P0,能越过所有 P1/P2 | 先让队列中存在 async 任务,再提交 1 个 sync 批,检查 S0 先于所有 P1/P2 执行 |
| 线程安全/生命周期 | 并发提权与入队、任务 finish、abort 不导致 crash | 开启 ASAN/Debug 或至少跑多次并发压测,确认无 crash / double free / use-after-free |

验收的关键证据,不是 TTFT 下降本身,而是**队列顺序符合预期**。TTFT 只是后续 benchmark 指标。

### 9.5 `split_priority_2level` 验收标准

验收 = `split_async_load_submit` 的提交行为成立 + 2 级固定优先级成立。

| 维度 | 验收标准 | 简单测试 |
|---|---|---|
| 头尾分批提交 | 同 `split_async_load_submit`:首次只提交头,promote 后补提交尾 | 复用 9.3 的测试 |
| 固定优先级生效 | sync 与已 promote 尾以 P0 提交;未 promote 头以 P1 提交 | queue_trace 中检查每个 unzip 任务的 priority |
| (a) 缓解生效 | 因为未 promote 尾根本不入队,B 头不会再被它挡住 | 构造 A、B 两个 async 请求,确认 B0..B{N-1} 前面不存在 A_N..A_{L-1} |
| (b) 缓解生效 | sync 批 C 的 S0 作为 P0,能越过其他请求的头(P1) | 先排一些 P1,再提交 C,检查 S0 先执行 |
| 流水线可运行 | 从 layer 0 算到 layer N 期间,队列能推进尾层加载,不会在 layer N 立即因为尾未提交而断流 | 跑 1 个 async 请求到 layer N,日志显示 layer N~N+1 的尾任务已在 layer 0~N-1 期间开始执行 |

`split_priority_2level` **不要求**"尾一定完全不卡 forward";它要求的是:尾已经在与前面层计算重叠地推进。是否会完全不卡,取决于“单层解压时间 ≤ 单层 prefill 计算时间”,这是性能结论,不是功能验收前提。

### 9.6 推荐的最小测试集合

为了尽快判断"做的东西有没有把核心机制做对",建议最少保留下面 5 组测试。

| 测试 | 目的 | 适用方案 |
|---|---|---|
| T1: 单请求 async,检查第一次只提交头/第二次补提交尾 | 验证分批提交状态机 | `split_async_load_submit`、`split_priority_2level` |
| T2: A(async) → B(async),打印任务入队/执行顺序 | 验证 (a) 是否缓解 | 所有,对比 `naive` |
| T3: A(async) → C(sync),检查 C 的 S0 何时执行 | 验证 (b) 是否缓解 | 所有,对比 `naive` |
| T4: 同一 step 到达的 A、B(async),检查队列是否为 `L0(A+B) L1(A+B) ...` | 验证公共前提 | `batch_reqs_async_load_submit`(非 `naive` 的方案都依赖它) |
| T5: promote 后 abort / finish,检查状态是否清空 | 验证状态清理 | `split_async_load_submit`、`split_priority_2level`,`priority_3level` 也建议跑 |

若时间只够做最少的 smoke test,顺序为:T4 → T1 → T3 → T5。T4 改动最小、收益已有实测依据;T3 对应 (b),最严重。

T2、T4 由 §7 的 S1 覆盖(S1 开头的 8 个请求同时到达对应 T4,后面补进的请求对应 T2),T3 由 S2 覆盖;T1、T5 是状态机测试,单独做。

## 10. 待办

- [x] 公共:`TaskQueue` 任务计时 + 暴露接口
- [x] 公共:`KVSHRINK_ASYNC_LOAD_SCHEME` 开关
- [x] 公共:开关取值改为 4.0 中的名字(目前只实现 `naive`)
- [x] §7 准备:connector 加 `start_load_kv` submit log 和 `get_finished` promote log
- [x] §7 准备:`tests/vllm-benchmark.sh` 参数改为可用环境变量覆盖(输入长度、命中率、并发、请求数、warmup、请求速率、seed),默认值不变
- [x] §7 准备:判定脚本 `tests/async_load_queue_check.py` + 执行脚本 `tests/async-load-queue-test.sh`
- [x] §7:`naive` 上 S1 PASS;S2 未覆盖(§11.3)。S2 速率改为 inf(不为凑覆盖调负载),后续每个方案照跑 S2。
- [x] `batch_reqs_async_load_submit`:实现(8.2);§7 S1、S2 均无 FAIL(均为未覆盖);输出正确性通过;9.2 中"结束 / abort"未测(§11.4)
- [x] 已知问题 K3 修复(在 split 之前,单独提交):scheduler 把本 step 被调度的请求放进 connector metadata,worker 只把被调度的已 promote 请求搬进 active;`naive`、`batch_reqs_async_load_submit` 各跑 §7 S1、S2 无 FAIL,输出正确性均通过(§11.5)
- [x] `split_async_load_submit`:按 4.2 / 8.3 实现;判定脚本加 split 的结构检查(9.3);§7 S1 PASS、S2 未覆盖,输出正确性通过;abort 清理只做代码审查(§11.6)
- [ ] `split_priority_2level`:`TaskQueue` 2 级固定优先级(P0 内 FIFO);`unzip_from_mem` / `KVFlow.get()` / `KVStore.get()` 透传 priority;connector 接入;9.5 验收 + §7 S1、S2 都跑(结论不得为 FAIL)
- [x] `priority_3level_first_n_per_req`:按 §4.3 末尾 / §8.4 实现(按组提权,请求进入 forward 时提权,`Context` 不改);单测 + TSan 通过;§7 S1、S2 均 PASS,输出正确性通过(§11.10)
- [x] 删除预留未实现的 `priority_3level_first_n_batch_reqs` 枚举值(10-22 决定,尾不跨请求共用任务,§4.3 末尾)
- [x] `TaskQueue`:任务带层号 + 一次提权多个组并按层排序;`get()` 透传层号;单测 + TSan(§4.3 末尾)
- [x] 新方案 `priority_3level_first_n_per_req_remaining_by_layer`、`priority_3level_first_n_batch_reqs_remaining_by_layer`(含 sync 批一起按层排):实现、§7 S1 / S2、输出正确性(§11.15)
- [x] 上面两个新方案:GSM8K(TP=2)0.955–0.960,TP=2 输出正确性 S1 / S2 通过(§11.15)
- [x] 上面两个新方案:32k 性能(四个 priority 方案之间比较,开事件日志)——实际跑了 sync / prio / 两个新方案 / batch × map 4/8、8/16、16/32 × 8k / 16k / 32k × hit 80 / 98 / 100(§11.16);固定层数与 map 的对比见 §11.17
- [ ] batch 在 hit 100、小 map 下比所有 priority 方案快(§11.16),原因未查(FIFO 顺序还是任务数);影响方案取舍
- [ ] 试 map `0-3:0,4-6:16,7-:48`(32k c8 下固定 N = 48 优于 map 16/32,§11.17)
- [ ] "sync 批与激活的尾一起按层排":只在 TP=1 的 `rbl-perreq-s2` 队列测试中出现 2 次;`_batch_reqs_` 下与 TP=2 下(GSM8K、正确性检查)都没有被覆盖,输出正确性也未在该路径上验证(§11.15)
- [ ] 统计 32k c4 / c8 下同一 step 激活多个请求、以及 sync 与激活同 step 的频率(决定按层提权的收益上限)
- [ ] 已知问题:`priority_3level_*` 不支持远程存储(`IAXL_RDMA_ENABLE`),目前启动报错,以后处理
- [ ] 分析:`v29p3-priority-s1` 中两次新请求的头等了 58 / 61 ms(§11.10),原因未查
- [x] 性能测试:72B 脚本(`perf-qwen72b-bench.sh`)适配 dev 分支与新方案、TP=8 用 GPU0-7;支持动态 map、固定层数(`DYNAMIC=0`)和每格重复次数(§11.12)
- [ ] `priority_3level_first_n_per_req_promote_raise`:只跑过 GSM8K(§11.11),§7 S1、S2 与输出正确性检查未跑
- [ ] 32k 并发上限:scratch 在 `KVFlow.get()` 提交时为每层申请、`get_wait(wait=True)` 时才归还,占用 ≈ 在途请求数 × 每请求完整 KV;32k 每请求每 rank 1.22 GiB,16 GiB 只够 c8。改为任务开始执行时申请、H2D 完成后释放(§11.12)
- [ ] 动态 map 按在途请求数(`len(self._req_states)`)选层数,32k c4 / c8 正好落在分界附近,同一格里 sync 与两种层数混在一起;考虑按新 token 数 / 命中率选(§11.12)。§11.14:32k 下只有 hit 100 的 async 有真实吞吐收益,hit 95 / 98 端到端持平或略差,这一条更值得做
- [ ] async 让 TPOT 变差(32k hit 95 / 98 下 N ≥ 48 比 sync 高 9–15%)的原因未查(§11.14)
- [ ] hit 98 下 store 步每层 ~14 ms 的去向未定:用 connector 的 `wait_ms` / `save_ms`(event log 的 `step_end` 行,已加未跑)拆分(§11.13)
- [ ] `record_test.cpp` 并发段报 `Invalid KV cache label: 0:0`(改动前的 `task_queue.h` 同样出现,与本方案无关),该测试也未接入 Makefile
- [x] P0 内顺序:10-14 定为先进先出,不按层号排序,适用于所有带 P0 的方案(理由见 4.3 末尾)
- [x] split 拆成两个变种:`split_async_load_submit` 改名 `split_async_load_submit_first_n_batch_reqs`;新增 `split_async_load_submit_first_n_per_req`(头逐个请求提交);判定脚本对 `_first_n_per_req` 改为"每次头提交只含 1 个请求"。10-14 在 GPU2 跑:S1 PASS、S2 未覆盖,输出正确性通过(§11.10)
- [ ] PR 分支同步改名(目前 PR 分支仍是 `split_async_load_submit`,`tests/async_load_dev/perf-qwen72b-bench.sh` 也按 PR 分支用旧名)
- [ ] 日志(整理 PR 前):`start_load_kv()` 的 `async_load step=… activate reqs=… held=…` 降为 DEBUG。它只用于验证 K3 / split 时人工统计,高并发下几乎每个 step 一行
- [ ] 日志(整理 PR 前):删除 `_log_load_submit()` 及其调用(submit log)。它只供 §7 判定脚本把 trace 任务对应到提交,生产上不需要
- [ ] 整理 PR 前:删除本分支在 sync / batch / split 路径 `get()` 上新加的 `description=` 参数(`naive` 路径的 `description=req_id` 是 main 原有的,保留)。它只用于 queue trace 标签把任务对应到请求
- [ ] 方案取舍(与同事讨论后定):正式 PR 暂时保留 `naive`、`batch_reqs_async_load_submit`、`split_async_load_submit` 三个可切换方案,默认 `split_async_load_submit`。若 split 的性能确认更好,后续删掉 `naive` / `batch_reqs_async_load_submit` 及 `KVSHRINK_ASYNC_LOAD_SCHEME` 开关
- [ ] 代码清理(以后做):`_early_promoted_tasks` 与 `_active_promoted_tasks` 合并为一个"已 promote" dict。promote 只发生在 forward 之后的 `get_finished`,forward 进行中不会新增;二者的区别只是"是否已进入某次 forward"。需在 K3 修复之后重新评估(K3 让"搬入 active"变为按请求是否被调度)
- [ ] OMP-Main 任务 release 开销(单独 PR):release 约占 OMP-Main 忙碌时间的 18–21%,原因、影响估算与修复方案见 [task-release-overhead.zh-CN.md](task-release-overhead.zh-CN.md)

## 11. 进度记录

- 2026-10-08:完成问题分析、方案设计与复杂度评估;创建分支 `perf/async-load-optimization-dev`。
- 2026-10-09:完成公共基础设施,并在 `naive` 上跑了 T2;方案改用 4.0 的名字,`batch_reqs_async_load_submit` 定为公共前提。完成 §7 测试工具(submit / promote log、benchmark 参数、判定脚本),`naive` 的 S1 PASS;S2 在实际映射表下未覆盖 (b)(§11.3)。step 层面的延迟记为已知问题 K2,不单独测试。完成 `batch_reqs_async_load_submit`,S1、S2 无 FAIL,输出正确(§11.4)。
- 2026-10-10:完成 split 详细设计;修复 K3,`naive`、`batch_reqs_async_load_submit` 的 S1、S2 无 FAIL,输出正确(§11.5)。输出正确性检查改为要求 vLLM 以 `VLLM_BATCH_INVARIANT=1` 启动;S2 速率改为 inf。完成 `split_async_load_submit`,S1 PASS、S2 未覆盖,输出正确(§11.6)。
- 2026-10-12:connector 代码整理(命名、显式循环、K3 搬入 active 移到所有提交之后),修正测试工具切 log 的问题,三个方案在 v0.23 上重测无 FAIL(§11.7)。rebase 到 `origin/main`(vLLM v0.29.0),在新容器里三个方案 S1、S2 无 FAIL、输出正确(§11.8)。split 下 GSM8K 跑 2 次:0.955 / 0.950(§11.9)。
- 2026-10-13:72B 性能脚本;split 拆成 `_first_n_batch_reqs` / `_first_n_per_req` 两个变种;`priority_3level` 改为按组提权的设计。
- 2026-10-14:定下 P0 按进入顺序 FIFO、请求进入 forward 时提权、重试保持 LOW、`p0_enter_ns`;实现 `priority_3level_first_n_per_req`,单测 / TSan 通过,GPU2 上 S1、S2 PASS、输出正确;补跑 `split_..._first_n_per_req`(§11.10)。
- 2026-10-16:trace 增加 `release_ns`,确认任务间隔几乎全部是 release;分析与修复方案另立文档 [task-release-overhead.zh-CN.md](task-release-overhead.zh-CN.md)。
- 2026-10-20:72B(TP=8)hit 98 诊断:变慢全部在 store 步,原因未定(§11.13);72B 全矩阵 360 格(prio / raise / batch × 3 个 map + sync,8k / 16k / 32k),无失败(§11.12)。
- 2026-10-21:prio、raise、batch 的 GSM8K 各 2 次,0.945–0.955,与 split 同水平(§11.11);72B 脚本加固定层数与重复次数,下一轮做 32k 固定层数扫描(§11.12)。
- 2026-10-21 / 22:32k 固定层数扫描 108 格无失败:N ≤ 8 比 sync 慢,N ≥ 32 的 TTFT 提升在 hit 100 是真实吞吐提升,在 hit 95 / 98 只是 TTFT 与 TPOT 的再分配(§11.14)。
- 2026-10-22:prio 逐组提权使尾在 P0 里按请求排(K1 第 2 点重现);定下做法 A(任务带层号 + 多组按层提权)、两个新方案名,删除预留的 `priority_3level_first_n_batch_reqs`;按代码与 trace 确认 prio 在 step 开始时 P0 为空(§4.3 末尾)。随后定:只新增这两个方案(都在进入 forward 时提权);同 forward 的 sync 批一起按层排,作为内置行为,不加开关;名字不改。未改代码。
- 2026-10-22:实现两个新方案(`TaskQueue` 任务带层号 + `set_groups_priority`,透传到 `KVStore.get()`,connector 与判定脚本),单测 48 项 + TSan 通过;GPU7 上 §7 S1、S2 与输出正确性无 FAIL,所有提权都按层号执行;`restart-vllm.sh` 补开事件日志(§11.15)。随后 TP=2:GSM8K 两次 0.955 / 0.955(per_req)、0.955 / 0.960(batch_reqs);输出正确性 S1 / S2 均通过;`restart-vllm.sh` 支持 `TP`,`run-queue-and-correctness.sh` 支持 `QUEUE=0`(§11.15)。
- 2026-10-22 / 23:72B 性能 351 格无失败(sync / prio / 两个新方案 / batch × 3 个 map):新方案修好 prio 在 hit 98 的前向等待(33–38 s → 14–15 s),整体比 prio 快约 2%;batch 不比任何 priority 方案差,小 map 下更好;map 16/32 下四个 async 方案相同(§11.16)。32k 固定层数 N = 48 与 map 16/32 打平(§11.17)。

### 11.1 公共基础设施

- `TaskQueue`(`iaxl/csrc/include/task_queue.h`):开启后为每个任务记录 `label / priority / seq / enqueue_ns / start_ns / end_ns / high_depth / low_depth`,时钟为 steady_clock(= Python `time.monotonic_ns()`);关闭时只多一次原子读。上限 262144 条,超出计入 `dropped`。
- 标签(`torch_ext/zip.cpp`):`<kind>|<description>|<tensor_key>`,kind ∈ {`unzip`, `unzip_retry`, `zip`};description 是本次 `get()` 涉及的 req_id(逗号分隔;sync 批同样如此),put 为空。
- 读取:`GET /v1/cache/queue_trace?queue=OMP-Main&enable=1|0`(worker 端;controller 转发)。每次调用返回并清空已记录事件。
- 单测:`iaxl/csrc/test/task_queue_test.cpp`(`make test-task-queue`),21 项通过;TSan 0 告警(需 `setarch $(uname -m) -R` 运行)。10-14 起的 4 级队列与新 trace 字段见 §11.10。

### 11.2 测试环境(本机配置,不进 PR)

| 项 | 值 | 说明 |
|---|---|---|
| 容器 | `iaxl.vllm.qiuyu`,镜像 `vllm-iaxl-dev-qiuyu` | 避开同机其他人的 `iaxl.vllm`;所有 GPU 可见,去掉 `--rm` |
| 模型 / 并行 | Qwen3-32B 本地路径,TP=1 | |
| GPU / CPU | `CUDA_VISIBLE_DEVICES=7`,`VLLM_CPU_OMP_THREADS_BIND=64-127` | GPU7 在 NUMA1;`auto_config.sh` 用 `nvidia-smi` 枚举 GPU,不认 `CUDA_VISIBLE_DEVICES`,需手动指定 CPU |
| 压缩后端 | QAT=0,IAA=0,DSA=0,CPU zip 60 线程 | |
| vLLM | `gpu-memory-utilization 0.95`,`max-model-len 32765`,prefix caching 关 | 0.8 时单卡 KV 只剩 3.6 GiB,起不来 |
| 端口 | API 8010,controller 18710,worker 18810 | 容器是 host 网络,避开默认端口 |
| async 配置 | `KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC=0`,`..._LAYERS=4` | 固定 N=4,保证单请求也走 async |
| v0.29 容器(§11.8 起) | `iaxl.vllm.qiuyu-v29`,镜像 `vllm-iaxl-dev-qiuyu-v29`(base `vllm/vllm-openai:v0.29.0`),vllm 0.29.0 / torch 2.13.0+cu130 | v0.23 的容器和镜像保留不动;由 `tests/async_load_dev/build-v29-container.sh` 构建 |
| 测试脚本 | `tests/async_load_dev/`:`restart-vllm.sh`、`run-queue-and-correctness.sh`、`v29-pipeline.sh`、`gsm8k-split.sh`(`SCHEME=` 选方案,结果 `_data/gsm8k/<方案>/`)、`perf-qwen72b-bench.sh` / `perf-qwen72b-serve.sh`(72B TP=8) | dev 辅助脚本;日志在 `_data/queue-test/logs/` |
| 分析脚本 | `tests/async_load_dev/analysis/`:§11.11–11.15 与 §4.3 末尾用到的临时分析脚本,未按仓库规范整理 | 在仓库根目录运行 |
| 测试结果副本 | `tests/async_load_dev/results/`:本文引用的 `_data/<路径>` 都以相同相对路径、相同时间戳文件名复制在此;不含 `<label>.serve.log`(换成截到 "Application startup complete" 的 `<label>.serve.startup.log`,即服务配置)和 `*.trace.json.gz` | `_data/` 本机被 `.git/info/exclude` 排除 |

### 11.3 §7 队列顺序验证:`naive`

证据目录(本机):`_data/queue-test/<运行名>/`,含 `bench.log`、`trace.json`、`vllm.log`(本次运行段)、`report.json`。

| 运行名 | 场景 | 关键参数 | 命中 async / sync | 结构错误 | 违序 | (a) 次数 | (b) 次数 | 结论 |
|---|---|---|---|---|---|---|---|---|
| `naive-s1` | S1 | inf,warmup 0 | 15 / 0 | 0 | 0 | 0 | — | 未覆盖 |
| `naive-s1-rate10` | S1 | 10 req/s,warmup 0 | 15 / 0 | 0 | 0 | 0 | — | 未覆盖 |
| `naive-s1-rate10-w1` | S1 | 10 req/s,warmup 1 | 16 / 0 | 0 | 0 | 7 | — | **PASS** |
| `naive-s2` | S2 | map `0-3:0,4-:4`,0.5 req/s,32 个请求(实测峰值在途 7) | 22 / 10 | 0 | 0 | — | 0 | 未覆盖 |
| `naive-s2-map-inv` | S2 | map `0-3:4,4-:0`,10 req/s | 5 / 11 | 0 | 0 | — | 1 | 不采纳:映射表与实际配置相反 |

列说明:"命中 async / sync"为命中 DDR 的请求中走 async、sync 的个数;"结构错误""违序"见 7.6;"(a) 次数"= async 提交时队列里有更早 step 的 async 任务未执行的次数;"(b) 次数"= sync 批提交时队列里有更早 step 的 async 任务未执行的次数。

PASS 的具体表现,与 7.4 中 `naive` 的预期一致:

- (a):step 677 提交的请求,首个任务前面有前一个请求的 59 个未 promote 尾,排队 172 ms;step 681 的请求前面有 240 个未 promote 尾 + 16 个头 + 11 个已 promote 尾,排队 518 ms。
- K1(取自 `naive-s1`,该轮对 (a) 无效,但 K1 的观测有效):step 2 一次提交 7 个请求,首个任务依次等 6、128、243、359、472、587、701 ms,前面没有更早 step 的任务,等待全部来自同一 step 内排在前面的请求。

### 11.4 `batch_reqs_async_load_submit`

代码:`start_load_kv()` 中,scheme 为 `batch_reqs_async_load_submit` 时把本 step 所有 async 请求的 block 合并,调一次 `get()`(description 为全部 req_id);每个请求的 `_pending_load_tasks[req_id]` 指向同一组 Task,promote / 结束逻辑不变。`naive` 路径保持原样。

**§7 队列顺序验证**(判定脚本已支持 PASS / FAIL / UNCOVERED 三种结论):

| 运行名 | 场景 | 命中 async / sync | 提交次数 | 结构错误 | 违序 | 含多个 async 请求的 step | (a) 次数 | (b) 次数 | 结论 |
|---|---|---|---|---|---|---|---|---|---|
| `batch-s1` | S1(10 req/s,warmup 1) | 16 / 0 | 10 | 0 | 0 | 4 | 0 | — | 未覆盖 |
| `batch-s2` | S2(map `0-3:0,4-:4`,0.5 req/s,32 个请求,峰值在途 8) | 20 / 12 | 32 | 0 | 0 | 0 | — | 0 | 未覆盖 |

列说明:"提交次数"= submit log 行数(每次 `get()` 一行);"含多个 async 请求的 step"= 同一 step 内 async 请求 ≥ 2 的 step 数;其余同 11.3。

- K1 已消除:`batch-s1` 中 16 个请求只产生 10 次提交,4 个 step 各自把 2~3 个请求合成一次提交;每次提交的首个任务等待 0~3.5 ms,不再随提交顺序递增(对比 `naive-s1` step 2 的 6→701 ms)。
- (a) 未覆盖:每次提交时队列里都没有更早 step 的任务。每次运行的请求到达时间是随机的(seed 取当前时间),本轮没造出;未分析是否与合并提交有关。
- (b) 未覆盖,与 `naive-s2` 相同。

**输出正确性**(`_data/queue-test/batch-correctness/`,含请求体、全部响应和 connector log):同一 prompt(4963 token)先发 1 次(未命中,`externally-cached tokens: 0`)作参考,再并发 4 次(均命中,`externally-cached tokens: 4960`,其中 3 个在 step 476 合并为一次提交并同时 promote);`temperature=0`、`seed=0`,4 个输出与参考逐字节一致,无 U+FFFD 或非法控制字符。

**9.2 验收项状态**:提交行为 ✔(结构检查:每个 step 至多一次 async 提交);promote 正确 ✔(合并的 3 个请求同时 promote,输出正确);效果 ✔(见上);结束 / abort **未测**。

### 11.5 K3 修复

代码(`kvshrink/kvshrink_connector.py`):

- `KVShrinkConnectorMetadata` 增加 `scheduled_req_ids`,`build_connector_meta()` 填入 `scheduler_output.num_scheduled_tokens` 的 key(本 step 被调度的请求)。
- `start_load_kv()` 在有 forward 时,只把 `_early_promoted_tasks` 中属于 `scheduled_req_ids` 的请求搬进 `_active_promoted_tasks`,其余留在 early 等下一个 step。
- 新增 log:`async_load step=<n> t=<ns> activate reqs=<本次搬入的请求> held=<留在 early 的请求>`。

**输出正确性检查的方法修正**(`tests/async_load_correctness_check.py`):

- 第一次检查(`naive` S1 配置,普通模式)FAIL:4 个命中请求中 2 个从第 1 个 token 起与参考不同。
- 原因(实测):参考请求单独跑(batch=1),命中请求与其他请求混在一个 batch 里,kernel 的归约顺序不同,贪心解码分叉,与加载无关。证据:以 `VLLM_BATCH_INVARIANT=1` 重启后同样的检查连跑 2 次(`bi-naive-1`、`bi-naive-2`),4 个命中请求均与参考逐字节一致,每次各有 3 个请求经过 held → activate。
- 因此正确性检查必须在 `VLLM_BATCH_INVARIANT=1` 下运行;队列顺序测试仍用普通模式(batch invariant kernel 更慢,会改变时序)。§11.4 的 10-09 正确性结果是普通模式下得到的,当时 4 个输出恰好一致。
- 脚本修正:取本次 log 改为按字节偏移(log 中有 `\r`,按行数会错位,导致参考请求的 log 行没取到)。

**§7 队列顺序验证**(普通模式,S2 速率 inf):

| 运行名 | 方案 | 场景 | 命中 async / sync | 提交次数 | 结构错误 | 违序 | 含多个 async 请求的 step | (a) 次数 | (b) 次数 | 经过 held 的请求 | 结论 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `k3-naive-s1` | `naive` | S1 | 16 / 0 | 16 | 0 | 0 | 4 | 7 | — | 15 | **PASS** |
| `k3-naive-s2` | `naive` | S2 | 29 / 3 | 31 | 0 | 0 | 10 | 12 | 0 | 29 | 未覆盖 |
| `k3-batch-s1` | `batch_reqs_async_load_submit` | S1 | 16 / 0 | 12 | 0 | 0 | 4 | 3 | — | 15 | **PASS** |
| `k3-batch-s2` | `batch_reqs_async_load_submit` | S2 | 29 / 3 | 12 | 0 | 0 | 7 | 3 | 0 | 29 | 未覆盖 |

列说明:"经过 held 的请求"= 在某个 step 的 activate log 中出现在 `held=` 里的不同请求个数,即 promote 后至少等了一个 step 才被 forward 等待的请求;其余列同 11.3、11.4。

- 几乎所有 async 请求都经过 held(推断:开了 async scheduling,scheduler 构建下一个 step 时还没收到该请求的 promote,所以该请求不在下一个 step 里,要再等一个 step)。修复前这些请求的尾会被不相关的 forward 全部等完。
- S2 的 (b) 仍未覆盖,按约定记为未覆盖,不为凑覆盖调负载。

**输出正确性**(`VLLM_BATCH_INVARIANT=1`,每次 1 个参考请求 + 4 个并发命中请求,prompt 4963 token,命中 4960 token):

| 运行名 | 方案 | 服务配置 | 参考请求 externally-cached | 4 个命中请求 externally-cached | 与参考逐字节一致 | 乱码 | 经过 held 的请求 | 结论 |
|---|---|---|---|---|---|---|---|---|
| `bi-naive-1` / `bi-naive-2` | `naive` | S1 | 0 | 4960 ×4 | 4 / 4 | 无 | 3 / 3 | PASS |
| `k3-naive-s2-correctness` | `naive` | S2 | 0 | 4960 ×4 | 4 / 4 | 无 | 1 | PASS |
| `k3-batch-s1-correctness` | `batch_reqs_async_load_submit` | S1 | 0 | 4960 ×4 | 4 / 4 | 无 | 2 | PASS |
| `k3-batch-s2-correctness` | `batch_reqs_async_load_submit` | S2 | 0 | 4960 ×4 | 4 / 4 | 无 | 1 | PASS |

列说明:"服务配置"= 启动 vLLM 时的 §7.3 场景环境变量;"externally-cached"= connector log `get_num_new_matched_tokens` 中从 DDR 加载的 token 数,参考请求为 0 表示未命中;"乱码"= U+FFFD 或非法控制字符;"经过 held 的请求"同上。

### 11.6 `split_async_load_submit`

代码(`kvshrink/kvshrink_connector.py`,`KVSHRINK_ASYNC_LOAD_SCHEME=split_async_load_submit`):

- `start_load_kv()`:sync 批 → `_submit_split_loads()`:先把 `_pending_remaining_load_req_ids` 中全部请求的剩余层打成 batch 提交,再把本 step 新 async 请求的前 N 层打成 batch 提交。
- 分段:`_first_n_layer_segments()` / `_remaining_layer_segments()` 按各请求的 N(N=−1 时取 L)切层,同一段内需要这些层的请求集合相同,每段调用一次 `get(layer_names=该段)`(`_submit_layer_segments()`,返回每个请求新提交的 Task)。submit log 的 kind:前 N 层为 `async`,剩余层为 `async_remaining`(本节下面 `k3-split-*` 的运行日志中为旧名 `async_tail`)。
- 每个请求一份自己的 Task dict,**只含它参与的层**(与 8.3 第 1 项写的 `dict(head_tasks)` 不同:整份复制会把同 batch 其他请求才有的层也放进来,不必要)。补剩余层时把新 Task 原地 `update` 进 `_early_promoted_tasks[req_id]`(K3 的搬入 active 在所有提交之后,此时请求一定还在 early)。
- 新状态:`_remaining_load_meta`(`SplitLoadMeta`:block_ids / block_hashes / num_first_n_layers,仅有剩余层的请求)和 `_pending_remaining_load_req_ids`。promote 时若请求在 `_remaining_load_meta` 中则加入待提交列表;请求结束清理时两者都删掉,因此 promote 后、补剩余层前结束的请求不会补。
- `async_load_layers` 只接受 −1 或 [1, L),其他值 `raise ValueError`(配置解析已保证,不做截断)。
- 以上为 10-12 整理后的代码(§11.7);10-10 初版用的是 `_split_load_meta` / `_tail_pending` 等旧名。

判定脚本(`tests/async_load_queue_check.py`)新增的 split 检查(9.3):每个请求的头恰为 0..h−1、尾恰为 h..L−1,各在一个 step 内提交;尾在 promote 的下一个 step 提交,且 trace 中尾的任务入队时间不早于 promote;同一 step 内尾先于头提交;同一 step 内同类分段的层不重叠。头与尾的 description 相同,改为按 description + 层号把 trace 任务对应到 submit log。用新脚本重跑 11.5 的四组数据,结果不变。

**§7 队列顺序验证**(普通模式,S2 速率 inf):

| 运行名 | 场景 | 命中 async / sync | 提交次数 | 结构错误 | 违序 | 含多个 async 请求的 step | (a) 次数 | (b) 次数 | 结论 |
|---|---|---|---|---|---|---|---|---|---|
| `k3-split-s1` | S1 | 16 / 0 | 19 | 0 | 0 | 4 | 1 | — | **PASS** |
| `k3-split-s2` | S2 | 29 / 3 | 22 | 0 | 0 | 7 | 2 | 0 | 未覆盖 |

列说明同 11.4;"提交次数"含头、尾、sync 各段的 `get()` 次数。

**新到达的提交前面排着什么**(只统计含第 0 层的提交,即一批新请求或一个 sync 批的到达;实测):

| 运行名 | 提交数 | 前面:已 promote 的尾 | 前面:未 promote 的尾 | 前面:头 | 首个任务等待 最大 / 平均(ms) |
|---|---|---|---|---|---|
| `k3-naive-s1` | 16 | 172 | 420 | 28 | 340.7 / 119.6 |
| `k3-batch-s1` | 12 | 78 | 0 | 0 | 67.6 / 16.7 |
| `k3-split-s1` | 10 | 44 | 0 | 0 | 215.0 / 61.2 |
| `k3-naive-s2` | 31 | 255 | 300 | 20 | 683.7 / 142.7 |
| `k3-batch-s2` | 12 | 37 | 0 | 0 | 233.0 / 26.4 |
| `k3-split-s2` | 12 | 62 | 0 | 0 | 224.3 / 57.8 |

列说明:"前面:xxx"= 该提交的首个任务入队时,来自更早 step、尚未开始执行的任务个数(按类别累加,类别见 7.6);"首个任务等待"= 首个任务从入队到开始执行的时间。

- split 下新到达的提交前面只有已 promote 的尾,没有未 promote 的尾,符合 4.2。
- batch 的两轮也没有"未 promote 的尾",原因未分析(可能是本轮到达时刻恰好没造出)。
- 三个方案各自一轮、请求到达时刻随机(seed 取当前时间),各轮队列负载不同,**等待时间不能用于方案间比较**。

**输出正确性**(`VLLM_BATCH_INVARIANT=1`):

| 运行名 | 服务配置 | 4 个请求的路径 | 与参考逐字节一致 | 乱码 | 结论 |
|---|---|---|---|---|---|
| `k3-split-s1-correctness` | S1 | 4 个 async:先 1 个、后 3 个合并提交头(层 0-3),各自 promote 后下一 step 补尾(层 4-63) | 4 / 4 | 无 | PASS |
| `k3-split-s2-correctness` | S2 | 3 个 sync + 1 个 async(头 0-3,promote 后下一 step 补尾 4-63) | 4 / 4 | 无 | PASS |

列说明:"4 个请求的路径"取自 connector log 的 submit / promote 行;其余同 11.5。

### 11.7 代码整理与 v0.23 上的回归

**connector 整理**(提交 `d477b45`,行为不变):

- split 相关的状态和变量按代码已有术语改名(first-N / remaining,不再用头 / 尾);`SplitLoadMeta` 改为 dataclass;分段用 `LayerSegment`。
- 推导式改为显式循环;`_submit_layer_segments()` 返回每个请求的 Task,由调用方赋值。
- K3 的"搬入 active"移到 `start_load_kv()` 末尾(所有提交之后),补剩余层时只需查 `_early_promoted_tasks`。naive / batch 分支由提前 `return` 改为 `if / elif`,逻辑不变。

**测试工具问题:切 log 切错**(实测 + 推断):

- 现象(实测):一轮 split S1 判 FAIL,benchmark 16 个请求全部成功,但切出的 vLLM log 只有 7 条 `get_num_new_matched_tokens`,且从半行开始;trace 中前 8 个请求的任务找不到 submit 行,报 260 个结构错误。10-10 的 `k3-naive-s1` log 片段中也有 9753 个 NUL 字节。
- 原因(推断):重启 vLLM 时旧的 `tee log.kvshrink-vllm` 还未退出,新 tee 截断文件从头写,旧 tee 仍在原偏移处写,文件被撑大(留下 NUL 空洞);测试脚本用文件大小作起点,切丢了开头。
- 修正:`restart-vllm.sh` 在启动新 vLLM 前,在容器内等 cwd 为本仓库的旧 tee 退出(host 用户读不到容器 root 进程的 `/proc/<pid>/cwd`)。之后每轮结果都检查 log 片段:从完整的一行开始、无 NUL、`get_num_new_matched_tokens` 条数 = 1(warmup) + 请求数。

**v0.23 上的回归**(connector md5 `76ea7e4b` = `d477b45`):

| 运行名 | 方案 | 场景 | 命中 async / sync | 提交次数 | 结构错误 | 违序 | 含多个 async 请求的 step | (a) 次数 | (b) 次数 | 队列结论 | 正确性 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `r4-split-s1` | split | S1 | 16 / 0 | 16 | 0 | 0 | 4 | 1 | — | **PASS** | PASS |
| `r3-split-s2` | split | S2 | 29 / 3 | 22 | 0 | 0 | 7 | 0 | 0 | 未覆盖 | PASS |
| `r3-naive-s1` | naive | S1 | 16 / 0 | 16 | 0 | 0 | 5 | 7 | — | **PASS** | PASS |
| `r3-naive-s2` | naive | S2 | 29 / 3 | 31 | 0 | 0 | 10 | 12 | 0 | 未覆盖 | PASS |
| `r3-batch-s1` | batch | S1 | 16 / 0 | 10 | 0 | 0 | 4 | 4 | — | **PASS** | PASS |
| `r2-batch-s2` | batch | S2 | 29 / 3 | 12 | 0 | 0 | 7 | 4 | 0 | 未覆盖 | PASS |

列说明同 11.4、11.5;"正确性"为同一配置下 `VLLM_BATCH_INVARIANT=1` 的输出正确性检查(4 个命中请求与参考逐字节一致、无乱码)。`r3-split-s1` 因上述切 log 问题作废,由 `r4-split-s1` 重跑。

### 11.8 rebase 到 vLLM v0.29.0

`origin/main` 的 `1ddbe90 kvshrink: upgrade base vLLM to v0.29.0 (#15)` 把基础镜像升到 v0.29.0,并修改 connector:

- `_view_as_blocks()`:v0.29 把所有层放在一块连续分配里,给出非连续的 view;connector 按 `KVCacheConfig` 的 page 大小把每层看作 `(num_blocks, page)`,store 用 `block_dim=0`。
- `build_connector_meta()` 中强制 `scheduler_output.has_sync_kv_loads = True`:否则 v0.29 把 async 的 `start_load_kv` 推迟到 forward 之后;我们的 load 只是 host 侧提交,必须在 forward 前执行。

rebase 结果:只有 `setvars.sh` 一处冲突(基础镜像用上游 v0.29.0,开发镜像名保留本机的);connector 由 git 自动合并,"rebase 前备份 → rebase 后"的 connector 增删行与上游改动完全一致,`build_connector_meta()` 中两边改动并存(先 `has_sync_kv_loads = True`,再构造带 `scheduled_req_ids` 的 metadata)。rebase 前的状态备份在本地分支 `backup/async-load-dev-pre-v0.29`。

**v0.29 上的验证**(容器 `iaxl.vllm.qiuyu-v29`,vllm 0.29.0 / torch 2.13.0+cu130,connector md5 `4633759c` = rebase 后 HEAD):

| 运行名 | 方案 | 场景 | 命中 async / sync | 提交次数 | 结构错误 | 违序 | 含多个 async 请求的 step | (a) 次数 | (b) 次数 | 经过 held 的请求 | 队列结论 | 正确性 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `v29-split-s1` | split | S1 | 16 / 0 | 25 | 0 | 0 | 2 | 5 | — | 15 | **PASS** | PASS |
| `v29-split-s2` | split | S2 | 29 / 3 | 26 | 0 | 0 | 9 | 5 | 0 | 28 | 未覆盖 | PASS |
| `v29-naive-s1` | naive | S1 | 16 / 0 | 16 | 0 | 0 | 4 | 6 | — | 15 | **PASS** | PASS |
| `v29-naive-s2` | naive | S2 | 29 / 3 | 31 | 0 | 0 | 8 | 13 | 0 | 29 | 未覆盖 | PASS |
| `v29-batch-s1` | batch | S1 | 16 / 0 | 16 | 0 | 0 | 0 | 5 | — | 15 | **PASS** | PASS |
| `v29-batch-s2` | batch | S2 | 29 / 3 | 14 | 0 | 0 | 7 | 6 | 0 | 28 | 未覆盖 | PASS |

列说明同 11.5、11.7。六份 log 片段均从完整的一行开始、无 NUL,`get_num_new_matched_tokens` 条数为 17 / 33(1 个 warmup + 16 / 32 个请求)。

- 正确性(从保存的 `responses.json` 复核):6 组各 4 个命中请求与参考逐字节一致,参考与命中输出中乱码字符均为 0;直接给出答案的 3 组(split S1、naive S1、naive S2)答案与各自 prompt 中 Record 0042 一致。
- v0.29 下 K3 的 held → activate 路径照常触发,split 的结构检查全部通过;上游的 `_view_as_blocks` + `block_dim=0` 与 split 按层 `get(layer_names=...)` 配合正常。
- `v29-batch-s1` 的"含多个 async 请求的 step"为 0:本轮请求恰好各自在不同 step 到达(到达时刻随机),16 个请求 16 次提交,结构检查无误。

### 11.9 GSM8K 精度:`split_async_load_submit`

配置:容器 `iaxl.vllm.qiuyu-v29`(vLLM 0.29.0),Qwen3-32B(bf16),TP=2 在 GPU4、5,`VLLM_CPU_OMP_THREADS_BIND=64-95|96-127`;`KVSHRINK_ASYNC_LOAD_SCHEME=split_async_load_submit`,async 配置用 `setvars.sh` 默认(`DYNAMIC=1`,map `0-3:0,4-6:4,7-:8`)。评测用 `Al-dsv4/KVCacheClip/internal/daocloud/test/accuracy-gsm8k/run-gsm8k.sh`(evalscope,200 题,并发 50,temperature 0,max_tokens 2048),同一个服务连跑 2 次,中间不清 DDR。脚本 `tests/async_load_dev/gsm8k-split.sh`,结果 `_data/gsm8k-split/`。

| 运行 | 答对 / 总数 | mean_acc | 命中外部缓存的请求 | 外部加载 token 平均 / 最少 / 最多 | 选到的 N(请求数) | 前 N 层 / 剩余层提交次数 | 报错行 |
|---|---|---|---|---|---|---|---|
| run1 | 191 / 200 | **0.955** | 199 / 200 | 573 / 0 / 576 | N=0:3,N=4:3,N=8:194 | 180 / 180 | 0 |
| run2 | 190 / 200 | **0.950** | 200 / 200 | 653 / 608 / 720 | N=0:3,N=4:3,N=8:194 | 164 / 162 | 0 |

列说明:"命中外部缓存的请求"= connector log 中 `externally-cached tokens` > 0 的请求数;"外部加载 token"= 每个请求从 DDR 加载的 token 数;"选到的 N"= 动态映射表按并发度选的 N(0 表示走 sync),N=4 与 N=8 同时出现,分段提交被覆盖;"前 N 层 / 剩余层提交次数"= 两类分段各自的 `get()` 次数;"报错行"= log 中 `Traceback` / `RuntimeError` / `ValueError` 行数。

- run1 也几乎全部命中:gsm8k 的 prompt 共用同一段 few-shot 示例(约 576 token),第一个请求存进 DDR 后其余请求都命中;run2 还命中了题目本身。
- 逐题对比(实测):最终答案相同 191 / 200;对错不同 3 题(index 98、114 run1 对 run2 错;197 run1 错 run2 对);输出全文相同 29 / 200;400 条输出乱码字符为 0。
- 两次相差 1 题(0.005)。没有开 `VLLM_BATCH_INVARIANT`,并发 50 时各请求所在 batch 的组成每次不同,贪心解码会分叉(与 11.5 的现象一致);推断该差异为运行间波动,未见 split 加载导致的问题。客户报告为 0.955(qwen3-32b-fp8,模型不同,仅供参考)。

### 11.10 `priority_3level_first_n_per_req`

**代码**(设计见 §4.3 末尾、§8.4):

| 文件 | 改动 |
|---|---|
| `iaxl/csrc/include/task_queue.h` | 4 个 deque(P0..P3);`submit(fn, priority, label, group = 0)`,priority 越界抛 `std::invalid_argument`;`set_group_priority(group, priority)` 持锁把该组在更低级别排队的任务按原顺序移到目标级别末尾,移入 P0 的记 `p0_enter_ns`,返回移动个数,组号 0 不动;`enqueue_ns` 改在锁内取;trace 新增 `submit_priority` / `group` / `p0_enter_ns` / `depth[4]`(替换 `high_depth` / `low_depth`) |
| `iaxl/csrc/torch_ext/context.h`、`zip.cpp`、`torch_ext.cpp` | `unzip_from_mem(..., priority = 0, group = 0)`;重试不变(LOW,组号 0);trace 输出新字段;新增 `task_queue_set_group_priority(group, priority)` |
| `iaxl/kvflow/flow.py`、`iaxl/kvstore/kvstore.py` | `get(..., priorities, group)` 透传(`priorities` 缺层时 `ValueError`);新增 `set_group_priority` |
| `kvshrink/async_load_config.py` | 方案名 `priority_3level_first_n_per_req`(已实现)、`priority_3level_first_n_batch_reqs`(未实现,启动报错) |
| `kvshrink/kvshrink_connector.py` | `_submit_priority_loads()`:每个 async 请求一次 `get()`,前 N 层 P1、其余 P2,一请求一个组号;K3 搬入 active 时 `set_group_priority(组号, P0)` 并删映射;结束清理时删映射;开远程存储时启动报错 |
| `tests/async_load_queue_check.py` | 新方案的结构检查(每次 async 提交 1 个请求、提交级别、组号唯一、提权时间介于 promote 与 activate 之间)与按时刻级别的违序检查(§7.6) |
| `tests/async_load_dev/restart-vllm.sh` | GPU 号 / 绑核改为环境变量 `GPU` / `CPU_BIND`(默认 7 / 64-127) |

**单测**(`iaxl.vllm.qiuyu-v29` 容器内):`task_queue_test` 35 项全部通过(原 22 项 + 新 13 项:4 级顺序、提权后按进入 P0 顺序、不移动的情况、重试排在压缩后、并发提权与入队);TSan 0 告警。`record_test` 前 9 项(含 3 项 `TaskQueue` HIGH/LOW 检查)通过,之后并发段报 `Invalid KV cache label: 0:0`,用改动前的 `task_queue.h` 编译结果完全相同,为已有问题。

**§7 队列顺序 + 输出正确性**(GPU2,`VLLM_CPU_OMP_THREADS_BIND=0-63`,Qwen3-32B,TP=1,输入 4k;connector md5 `586c6133`):

| 运行名 | 方案 | 场景 | 命中 async / sync | 提交次数 | 结构错误 | 违序 | (a) 次数 | (b) 次数 | 队列结论 | 正确性 |
|---|---|---|---|---|---|---|---|---|---|---|
| `v29p3-priority-s1` | `priority_3level_first_n_per_req` | S1 | 16 / 0 | 16 | 0 | 0 | 12 | — | **PASS** | PASS |
| `v29p3-priority-s2` | `priority_3level_first_n_per_req` | S2 | 27 / 5 | 31 | 0 | 0 | 8 | 2 | **PASS** | PASS |
| `v29fnpr-split-s1` | `split_async_load_submit_first_n_per_req` | S1 | 16 / 0 | 28 | 0 | 0 | 4 | — | **PASS** | PASS |
| `v29fnpr-split-s2` | `split_async_load_submit_first_n_per_req` | S2 | 29 / 3 | 44 | 0 | 0 | 7 | 0 | 未覆盖 | PASS |
| `v29p3reg-naive-s1` | `naive`(`TaskQueue` 改动后的回归) | S1 | 16 / 0 | 16 | 0 | 0 | 7 | — | **PASS** | PASS |

列说明同 11.7;"正确性"为 `VLLM_BATCH_INVARIANT=1` 下 1 个参考请求(`externally-cached tokens` = 0)+ 4 个命中请求(各 4960 token),4 个输出与参考逐字节一致、乱码字符 0 个。

**提权是否发生**(`v29p3-priority-s1` trace,实测):

| 提交级别 | 执行级别 | 被提权(`p0_enter_ns` > 0) | 任务数 |
|---|---|---|---|
| P1(前 N 层) | P1 | 否 | 64 |
| P2(其余层) | P0 | 是 | 394 |
| P2 | P2 | 否 | 566 |

列说明:"提交级别"= `submit_priority`;"执行级别"= `priority`;"任务数"为 unzip 任务条数(另有 1024 个 zip 任务,P3)。566 个 P2 任务未被提权就执行完,推断为请求 activate 前趁队列空闲预取完成,未单独核对。

- 超车:64 个 P1 任务中 48 个比更早入队的 P2 任务先执行。
- 负对照:把 trace 中的提权记录抹掉(`p0_enter_ns` 置 0、`priority` 改回提交级别)再判定,结果 FAIL、违序 53,判定脚本能识别提权。
- (b) 第一次被覆盖:`v29p3-priority-s2` 中 sync 批两次到达时前面分别排着 62、61 个更早 step 的尾任务,其第 0 层任务等待 0.7 ms、0.9 ms。
- 未解释:S1 中两次新请求的首个任务等了 58.2 ms、61.0 ms(其余同类提交 0.7~4.1 ms),原因未查(§10)。
- 判定报告的 `async_tail_promoted` 按"请求已 promote"分类;本方案下已 promote 但未 activate 的请求,尾仍在 P2,该分类不等于"在 P0"。

本节只验证队列顺序与输出正确性,未测 TTFT。

### 11.11 GSM8K 精度:`priority_3level_first_n_per_req`、`_promote_raise`、`batch_reqs_async_load_submit`

配置与 §11.9 相同(Qwen3-32B bf16,TP=2 在 GPU4、5,map `0-3:0,4-6:4,7-:8`,200 题,并发 50,temperature 0,同一服务连跑 2 次、中间不清 DDR),另开 `KVSHRINK_ASYNC_LOAD_EVENT_LOG=1` 记录每次提交 / promote。10-21 依次跑三个方案。脚本 `SCHEME=<方案> tests/async_load_dev/gsm8k-split.sh`(脚本现在显式传 `MODEL`:容器环境变量里的 `MODEL` 是 72B,第一次启动误加载了 72B,已停掉重跑),结果 `_data/gsm8k/<方案>/`,汇总行在 `_data/gsm8k/<方案>.run.log`。

| 方案 | 运行 | 答对 / 总数 | mean_acc | 命中外部缓存的请求 | 外部加载 token 平均 / 最少 / 最多 | 选到的 N(请求数) | sync / async 提交行 | 报错行 | 乱码 |
|---|---|---|---|---|---|---|---|---|---|
| prio | run1 | 191 / 200 | **0.955** | 198 / 200 | 570 / 0 / 576 | N=0:3,N=4:3,N=8:194 | 2 / 394 | 0 | 0 |
| prio | run2 | 189 / 200 | **0.945** | 200 / 200 | 653 / 608 / 720 | 同上 | 6 / 394 | 0 | 0 |
| raise | run1 | 191 / 200 | **0.955** | 198 / 200 | 570 / 0 / 576 | 同上 | 2 / 394 | 0 | 0 |
| raise | run2 | 189 / 200 | **0.945** | 200 / 200 | 653 / 608 / 720 | 同上 | 4 / 394 | 0 | 0 |
| batch | run1 | 189 / 200 | **0.945** | 199 / 200 | 573 / 0 / 576 | 同上 | 2 / 178 | 0 | 0 |
| batch | run2 | 191 / 200 | **0.955** | 200 / 200 | 653 / 608 / 720 | 同上 | 4 / 178 | 0 | 0 |

列说明:prio / raise / batch 分别是 `priority_3level_first_n_per_req` / `priority_3level_first_n_per_req_promote_raise` / `batch_reqs_async_load_submit`;"sync / async 提交行"= vLLM log 中 `submit kind=sync` / `submit kind=async` 的行数,两个 TP rank 合计(prio / raise 每个 async 请求一次 `get()`,197 个请求 × 2 rank = 394;batch 同一 step 的请求合成一次);"乱码"= 400 条输出中 U+FFFD 与非法控制字符个数;其余同 §11.9。

**同一方案 run1 与 run2 逐题对比**(实测):

| 方案 | 最终答案相同 | 对错不同的题 | 输出相同 |
|---|---|---|---|
| prio | 187 / 200 | 8 题:run1 对 5 题(12、85、93、147、187),run2 对 3 题(40、114、139) | 47 / 200 |
| raise | 188 / 200 | 8 题:run1 对 5 题(7、62、87、161、181),run2 对 3 题(39、114、122) | 33 / 200 |
| batch | 189 / 200 | 6 题:run1 对 2 题(147、187),run2 对 4 题(7、93、114、138) | 42 / 200 |
| split(§11.9) | 191 / 200 | 3 题 | 29 / 200 |

列说明:题号为 evalscope 的 `index`;"输出相同"= 评审记录 `sample_score.score.prediction` 全文相同(与 §11.9 的 29 / 200 同一口径)。

- 三个方案 6 次在 0.945–0.955,与 split 的 0.955 / 0.950 同一水平;同一方案两次相差 0–2 题,方向两边都有(batch 是 run2 更好)。
- 对错翻转的题在方案之间重复:114 在四个方案里都翻转,7、93、147、187 在两个方案里翻转。推断为对数值扰动敏感的题,差异来自运行间波动(未开 `VLLM_BATCH_INVARIANT`,与 §11.9 同),未见加载方案导致的问题。
- 局限:GSM8K 每个请求只加载约 576–720 token(共享的 few-shot 前缀),只覆盖短加载;map 用的是 setvars 默认值,不是 §11.12 推荐的 map。

### 11.12 72B 性能:三个方案 × 三个 map(10-20)

配置:容器 `iaxl.vllm.qiuyu-v29`,Qwen2.5-72B-Instruct,TP=8 在 GPU0-7,QAT + IAA 压缩、DSA + GDRCopy 传输,每 rank DDR 25 GiB / scratch 16 GiB,`max_num_batched_tokens=8192`;输出 128 token,每格 5 个 warmup,每格结束清 DDR;输入 8k / 16k / 32k,hit 80 / 95 / 98 / 100,并发 c4 / c8 / c16 / c32(16k 最多 c16,32k 最多 c8),prompt 数 c4:32、c8:64、c16:96、c32:160。方案 prio / raise / batch,各跑 3 个 map:A = `0-3:0,4-6:8,7-:16`,B = `0-3:0,4-6:16,7-:32`,C = `0-3:0,4-6:40,7-:79`;加 sync(`KVSHRINK_VLLM_KV_ASYNC_LOAD_ENABLED=0`)。10 个 server × 36 格 = 360 格,06:54–21:33,无失败格、无重启。结果 `_data/perf-qwen72b/20261020_065412/`。

**各配置整体**(实测):

| 配置 | TTFT 变化 | 优于 sync 的格数 | 最差格 | p95 TTFT 变化 | TPOT 变化 |
|---|---|---|---|---|---|
| prio A | −18.2% | 25 / 36 | +20%(16k h98 c16) | −3.2% | +6.8% |
| prio B | −22.2% | 29 / 36 | +19%(16k h80 c16) | −5.6% | +7.6% |
| prio C | −20.4% | 27 / 36 | +50%(16k h80 c16) | −2.2% | +5.9% |
| raise A | −21.6% | 23 / 36 | +17%(8k h98 c16) | −2.7% | +6.3% |
| raise B | −21.9% | 29 / 36 | +30%(16k h80 c16) | −3.9% | +4.6% |
| raise C | −19.8% | 27 / 36 | +49%(16k h80 c16) | −0.8% | +6.0% |
| batch A | −22.0% | 30 / 36 | +14%(8k h80 c32) | −5.0% | +7.1% |
| batch B | −21.6% | 26 / 36 | +33%(16k h80 c16) | −2.2% | +5.8% |
| batch C | −21.5% | 28 / 36 | +32%(16k h80 c16) | +2.7% | +5.7% |

列说明:"TTFT / p95 TTFT / TPOT 变化"= 每格"该配置 ÷ sync"取几何平均后减 1,负数为更好;"最差格"= 平均 TTFT 相对 sync 变慢最多的格(h 命中率,c 并发)。

**按命中率**(平均 TTFT 相对 sync,各长度、并发的几何平均,依次为 map A / B / C):

| 方案 | hit 80 | hit 95 | hit 98 | hit 100 |
|---|---|---|---|---|
| prio | −9 / −6 / −4% | −14 / −12 / −11% | −2 / −14 / −16% | −42 / −49 / −43% |
| raise | −7 / −5 / −3% | −11 / −11 / −12% | −7 / −15 / −16% | −51 / −48 / −43% |
| batch | −8 / −5 / −2% | −12 / −12 / −16% | −16 / −18 / −21% | −46 / −45 / −41% |

- 噪声(实测):与 10-19 那一轮的 20 个相同格子比,各配置两轮之比的中位数 0.99–1.00,单格 0.73–1.25。单格 25% 以内的差别不能用来给配置排序;9 个配置整体相差约 4 个百分点,在噪声内。
- hit 98:map `0-3:0,4-6:4,7-:8`(10-16、10-19)和 A(10-19 的第二个 map,本轮仍在跑)下,prio / raise 在 hit 98 c16 / c32 比 sync 慢,本轮 prio A 仍是 8k c16 +10%、c32 +16%,16k c16 +20%;换成 B / C 后这几格 prio / raise 都比 sync 快(−2% ~ −33%)。
- 输入长度:8k / 16k / 32k 整体收益为 −17 ~ −22% / −21 ~ −27% / −16 ~ −23%,看不出随长度变化的趋势。

**低命中、高并发时异步"变慢"只是 TTFT 与 TPOT 的再分配**(实测,16k h80 c16):

| 配置 | 平均 TTFT ms | 平均 TPOT ms | TTFT + 127 × TPOT ms | c × 时长 ÷ 请求数 ms | 平均排队 | 平均运行 |
|---|---|---|---|---|---|---|
| sync | 13517 | 226 | 42261 | 42287 | 0.5 | 13.2 |
| prio A | 13858 | 225 | 42407 | 42448 | 2.0 | 12.7 |
| prio B | 16086 | 211 | 42900 | 42942 | 2.7 | 12.0 |
| prio C | 20216 | 175 | 42455 | 42478 | 4.2 | 11.2 |

列说明:"TTFT + 127 × TPOT"= 每个请求的平均端到端时延(输出固定 128 token);"c × 时长 ÷ 请求数"= 按 Little 定律从吞吐反推的端到端时延;"平均排队 / 平均运行"= 该格时间窗内 vLLM 每 10 s 统计行 `Waiting` / `Running` 的平均值(`Waiting` 含等待 KV 加载的请求)。

- 本格 10 个配置的端到端时延都在 42.1–42.9 s,时长 253–258 s,prefill 吞吐 1216–1240 token/s;32k h80 c8(42.2–42.8 s)和 8k h80 c32(44.3–44.8 s)同样。异步没有改变吞吐,map 越大请求在加载完前 N 层之前排队越久(TTFT 升),在跑的请求变少、每步更快(TPOT 降)。
- 对照 16k h100 c16:异步把端到端时延从 11.5 s 降到 10.0–10.8 s、时长从 69 s 降到 61–65 s,是真实的吞吐提升。
- 推断:hit 80 时 sync 的加载在请求自己那一步里逐层与 prefill 重叠,几乎不额外耗时,异步没有可隐藏的加载,只推迟准入。
- 未解释:8k c32 / 16k c16 / 32k c8 的 hit 80 prefill 吞吐都约 1.2k token/s,不随上下文长度变化,瓶颈可能在随新 token 数增长的部分(例如 store),未验证。

**动态 map 在 c4 / c8 下把"层数"和"走不走 async"混在一起**(实测,prio):

| 格子 | map | 走 async 的请求 | 选到的 N(请求数) | 平均 TTFT ms | sync ms |
|---|---|---|---|---|---|
| 32k h98 c4 | A | 25% | 8:7 | 1764 | 1849 |
| 32k h98 c4 | B | 47% | 16:15 | 1281 | 1849 |
| 32k h98 c4 | C | 68% | 40:21 | 1328 | 1849 |
| 32k h98 c8 | B | 75% | 16:31,32:16 | 3619 | 3966 |

列说明:"走 async 的请求"= serve.log 中 `async=True` 的请求 ÷ 命中外部缓存的请求;"选到的 N"= async 请求选到的层数及请求数。

- map 按在途请求数(`len(self._req_states)`)选规则,32k 只有 c4 / c8,正好在 0-3 / 4-6 / ≥7 的分界附近;在途数又随配置变化,同一格里 A 只有 25% 的请求走 async,C 有 68%。所以 32k 上"B 比 A 好"的证据(最明显的是 h98 c4,0.73 倍)不干净。
- 当前推荐:prio + map B(整体 −22.2%,29 / 36 格优于 sync,hit 98 不再输给 sync,最差 +19%)。prio 与 raise 在本轮分不出高下(B 下整体 −22.2% / −21.9%);选 prio 是因为它有 §7 S1 / S2 与输出正确性记录、逻辑更简单、最差格更小。

**32k 并发上不去**(代码事实):`KVFlow.get()` 提交时为每层 `chunk_pool.allocate()`,`get_wait(wait=True)` 时才 `release()`(`iaxl/kvflow/flow.py`),H2D 早已完成也不还。32k 一个请求每 rank 80 层 × 2000 块 × 8 KiB ≈ 1.22 GiB,c8 ≈ 9.8 GiB(10-20 实测峰值 8.86 GiB / rank,0.5 s 采样),c16 ≈ 19.5 GiB,超过 16 GiB 时 `ScratchPool.allocate()` 抛异常。scratch 开到 24 / 32 GiB 的 3 个 server 中有 2 个与整机一起卡住(推断与 (scratch + DDR) × 8 ≈ 416 GB 的内存压力有关,未验证)。见 §10。

**下一轮**(10-21 已跑,结果见 §11.14):只跑 prio 与 sync,32k,hit 95 / 98 / 100,c4 / c8,每格重复 2 次(同一 seed);层数设置为 map B 加 `KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC=0` 下固定 N = 4 / 8 / 16 / 32 / 48 / 64 / −1(标签 `_N4` … `_Nall`),共 9 个 server × 12 格 = 108 格。目的:请求全部走 async 时单独比较 N,并看 map 的"0-3 走 sync"在 32k 下是否有利。

### 11.13 72B hit 98 诊断(10-20)

配置:8k,hit 98 / 100,c16,map `0-3:0,4-6:4,7-:8`,`TRACE=1`(OMP-Main trace + event log),prio / raise / batch / sync 各 2 格。结果 `_data/perf-qwen72b/hit98diag_20261020_030016/`。

| 方案 | hit 98 平均 TTFT ms | hit 100 平均 TTFT ms |
|---|---|---|
| prio | 2033 | 1035 |
| raise | 1918 | 609 |
| batch | 1784 | 684 |
| sync | 1554 | 1157 |

实测:

- 变慢全部在 store 步(请求 2% 的新 token,即 160 token / 10 块,算完要存回 DDR 的那一步)。sync 下加载同样请求数的一步,hit 98 / hit 100 耗时:1 个请求 198 / 137 ms,4 个 557 / 380 ms,10 个 1234 / 883 ms,约多 35 + 32 × 请求数 ms。
- 加载没有变慢:sync 下 OMP 解压每请求每层 ~1.0–1.1 ms,hit 98 与 hit 100 相同;hit 98 的 10 请求步里,OMP 解压完第 L 层的时刻比前向到达第 L 层早 88 ms(第 0 层)到 447 ms(第 79 层),前向不是在等解压。
- put 本身的开销可以忽略:同一层相邻两个请求的 put(以 zip 任务入队时刻计)只隔 ~0.03 ms。put 首个 tensor 调用 `xfer_wait_cur_stream(sync_cur_stream=True)`,即 `cudaStreamSynchronize(cur_stream)`,阻塞时长等于 GPU 上剩余的工作量,是否构成每层的主要耗时未测。
- prio 的 10 请求激活步:已预加载的第 0–7 层每层 ~14 ms(普通 decode 每层 ~0.65 ms);第 7 → 8 层(第一个尾层)停了 572 ms,期间 OMP-Main 86% 忙于这些请求尾层的 P0 解压,即尾层在激活前没预取完;之后每层 ~14 ms。sync 处理同样 10 个请求一步 1.23 s,prio 1.6–1.7 s。

未定:store 步每层 ~14 ms 花在哪(前向计算、stream 同步、CPU 争用)。connector 已加 `wait_for_layer_load` / `save_kv_layer` 的累计耗时(仅 event log 开启时,`step_end wait_ms=… save_ms=…`),尚未跑(§10)。

### 11.14 32k 固定层数扫描(10-21)

配置:与 §11.12 相同的 72B TP=8 环境;只跑 `priority_3level_first_n_per_req` 与 sync;32k,hit 95 / 98 / 100,c4 / c8(prompt 32 / 64),每格重复 2 次(同一 seed,同一组 prompt);层数设置:map B `0-3:0,4-6:16,7-:32`,以及 `KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC=0` 下固定 N = 4 / 8 / 16 / 32 / 48 / 64 / −1(−1 = 全部 80 层加载完才开始)。`perf-qwen72b-bench.sh` 默认值(提交 `26388cc` 中的版本;`run.log` 里记的 commit 是启动时的 HEAD `757215e`,脚本当时未提交但内容相同)。10-21 05:45 – 10-22 凌晨,9 个 server × 12 格 = 108 格,无失败格、无重启。结果 `_data/perf-qwen72b/20261021_054543/`(副本见 §11.2),分析脚本 `tests/async_load_dev/analysis/fixed_layers.py`。

**各层数设置的平均 TTFT 相对 sync**(实测):

| 设置 | 6 格合计 | hit 95 | hit 98 | hit 100 | c4 | c8 |
|---|---|---|---|---|---|---|
| map B | −25% | −8% | −9% | −49% | −24% | −25% |
| N = 4 | +19% | +12% | +42% | +5% | +11% | +26% |
| N = 8 | +13% | +9% | +31% | +2% | +11% | +16% |
| N = 16 | −16% | −7% | +4% | −38% | −15% | −17% |
| N = 32 | −23% | −1% | −5% | −52% | −17% | −29% |
| N = 48 | **−27%** | −6% | −9% | **−54%** | −17% | **−36%** |
| N = 64 | −26% | −11% | −9% | −50% | −15% | −35% |
| N = −1 | −22% | −9% | −5% | −45% | −9% | −33% |

列说明:每格先取 2 次重复的平均 TTFT 均值,除以 sync 同格的均值,再对所选的格取几何平均减 1;负数为比 sync 快。

**端到端:只有 hit 100 是真实收益**(实测,相对 sync 的几何平均,c4 与 c8 合计):

| 设置 | hit 95:TTFT / TPOT / 端到端 / 时长 | hit 98:TTFT / TPOT / 端到端 / 时长 | hit 100:TTFT / TPOT / 端到端 / 时长 |
|---|---|---|---|
| map B | −8 / +7 / +0 / +0% | −9 / +10 / +4 / +4% | −49 / +6 / −9 / −8% |
| N = 16 | −7 / +7 / +1 / +1% | +4 / +7 / +6 / +7% | −38 / +10 / −4 / −3% |
| N = 32 | −1 / +3 / +1 / +1% | −5 / +8 / +4 / +4% | −52 / +5 / −11 / −10% |
| N = 48 | −6 / +9 / +2 / +2% | −9 / +13 / +6 / +6% | −54 / +1 / −15 / −13% |
| N = 64 | −11 / +14 / +2 / +3% | −9 / +15 / +7 / +7% | −50 / −0 / −14 / −13% |
| N = −1 | −9 / +14 / +4 / +4% | −5 / +14 / +8 / +8% | −45 / −1 / −13 / −12% |

列说明:"端到端"= 平均 TTFT + 127 × 平均 TPOT(输出固定 128 token);"时长"= bench 的 `duration`,与吞吐成反比;N = 4 / 8 在三个命中率下端到端都比 sync 差(+1% ~ +15%),未列。

实测结论:

- N ≤ 8 在所有格都不如 sync,连 hit 100 也没有收益(+2% / +5%)。N ≥ 32 的 TTFT 接近,N = 48 整体最好(−27%),尤其 c8(−36%);c4 下 N = 32 / 48 只有 −17%。
- hit 100:N ≥ 32 的端到端 −11% ~ −15%、时长 −10% ~ −13%,是真实的吞吐提升。
- hit 95 / 98:TTFT 的提升(−1% ~ −11%)伴随 TPOT 变差(+3% ~ +15%),端到端与时长持平或变差(hit 98 下 N ≥ 32 为 +4% ~ +8%),和 §11.12 中 hit 80 的情形相同,只是时间在 TTFT 与 TPOT 之间重新分配。
- map B 整体 −25%,与最好的固定 N 接近:c4 下最好(−24%,固定 N 最多 −17%),c8 下不如 N = 48 / 64(−25% 对 −36% / −35%)。c4 下 map B 只有 25% ~ 84% 的请求走 async(hit 95 25%,hit 98 25% ~ 46%,hit 100 79% ~ 84%),其余走 sync;hit 95 / 98 c4 下它比全部 async 的 N = 32 / 48 快(0.87 / 0.83 对 1.00 ~ 1.02)。
- 重复性:同一个 server 上同一格 2 次的差别中位数 2%,54 对里只有 1 对 ≥ 20%:map B hit 98 c4,r1 1318 ms、r2 1706 ms,两次走 async 的请求分别为 46% 与 25%——动态 map 在 c4 下走不走 async 本身就是波动源。与 10-20 那一轮同格比,map B 为 0.93 ~ 1.18 倍,sync 为 0.83 ~ 1.00 倍。
- 固定 N 下所有命中请求都走 async 且选到设定的 N(connector log 核对)。

推断:

- N 太小时,请求很快被 promote,前向随后在尾层上等加载,等于把 sync 的等待搬到了前向里,又多了 promote / 提权的步边界开销。
- 32k 下只有新 token 极少(hit 100)时 async 才提高吞吐;hit 95 / 98 时新 token 已有 640 ~ 1600 个,sync 的加载能与自己的 prefill 重叠(同 §11.12 对 hit 80 的推断)。这支持"按新 token 数 / 命中率决定走不走 async"(§10),未验证。
- 目前对 32k 的建议:若只看 TTFT,c8 用 N = 48,c4 用 map B;一个可以试的 map 是 `0-3:0,4-6:16,7-:48`(未测)。若看吞吐 / 端到端,hit 95 / 98 用 sync 不吃亏。

### 11.15 `priority_3level_first_n_*_remaining_by_layer`(10-22)

代码见 §4.3 末尾"实现(10-22)"。两个方案的区别只在提交头:`_per_req_` 每个请求一次 `get()`(头 P1、尾 P2,同一组号);`_batch_reqs_` 把本 step 所有 async 请求的头合成按层分段的 `get()`(P1,组号 0,不提权),每个请求的尾再各自一次 `get()`(P2,各自组号)。激活时两者相同:本 step 激活的所有请求的组号 + 本 step 的 sync 组号,一次 `set_groups_priority(…, P0)`。

**单测**(`iaxl.vllm.qiuyu-v29` 容器内):`task_queue_test` 48 项全部通过;TSan 0 告警。

**测试脚本修正**:10-20 起 submit / promote / activate 行降为 DEBUG,只在 `KVSHRINK_ASYNC_LOAD_EVENT_LOG=1` 时输出,而 `restart-vllm.sh` 没开,§7 判定脚本找不到任何 submit 行,所有任务报"无 submit 行"而 FAIL。已在 `restart-vllm.sh` 中打开。10-20 之后没有跑过 §7,之前的结果不受影响。

**§7 队列顺序 + 输出正确性**(GPU7,`VLLM_CPU_OMP_THREADS_BIND=64-127`,Qwen3-32B,TP=1,输入 4k;connector md5 `4dafb666`):

| 运行名 | 方案 | 场景 | 命中 async / sync | 提交次数 | 结构错误 | 违序 | (a) 次数 | (b) 次数 | 队列结论 | 正确性 |
|---|---|---|---|---|---|---|---|---|---|---|
| `rbl-perreq-r2-s1` | `_per_req_remaining_by_layer` | S1 | 16 / 0 | 16 | 0 | 0 | 2 | — | **PASS** | PASS |
| `rbl-perreq-s2` | `_per_req_remaining_by_layer` | S2 | 27 / 5 | 31 | 0 | 0 | 8 | 2 | **PASS** | PASS |
| `rbl-batchreqs-s1` | `_batch_reqs_remaining_by_layer` | S1 | 16 / 0 | 24 | 0 | 0 | 2 | — | **PASS** | PASS |
| `rbl-batchreqs-s2` | `_batch_reqs_remaining_by_layer` | S2 | 29 / 3 | 41 | 0 | 0 | 2 | 0 | 未覆盖 | PASS |
| `rbl-batchreqs-r2-s2` | `_batch_reqs_remaining_by_layer` | S2 | 29 / 3 | 41 | 0 | 0 | 2 | 0 | 未覆盖 | PASS |

列说明同 11.7、11.10;方案名省略前缀 `priority_3level_first_n`。"提交次数"= submit log 行数(sync 批、头的每个分段、每个请求的尾各算一次)。正确性为同一配置下 `VLLM_BATCH_INVARIANT=1` 的检查:参考请求 `externally-cached tokens` = 0,4 个命中请求各 4960,与参考逐字节一致,乱码 0。

- 新增的结构检查(§4.3 末尾)全部通过:每个任务 `order_key` = 层号;sync 组号非 0、每 step 唯一、只在本 step 的激活里被重排;`_batch_reqs_` 的头为 P1、组号 0、从未提权,尾 N..L−1 与头同 step 提交。
- `rbl-perreq-s1` 作废(实测):该轮 vLLM log 的 EngineCore pid(1287175)小于本轮 tmux 进程的 pid,是上一轮被中断时刚拉起的旧服务(未开事件日志);新服务没占到端口,测试打到了旧服务上。由 `rbl-perreq-r2-s1` 重跑。同轮的 `rbl-perreq-s1-correctness` 用的是新服务(事件日志 288 行),PASS。其余各轮 log 都有事件日志行,EngineCore pid 各不相同。
- `rbl-batchreqs-s2` 未覆盖 (b):3 个 sync 请求到达时队列里没有更早 step 的 async 任务,与以往 split / batch 的 S2 相同。为了覆盖"sync 批一起按层排"重跑一次(`rbl-batchreqs-r2-s2`,新服务、新 seed,trace 不同),各项计数与第一次几乎相同(被提权任务 987 对 993,其余相同)。推断:S2 速率为 inf,32 个请求同时到达,调度结构基本固定,重跑不增加覆盖;按约定不为凑覆盖调负载。

**每次提权的实际执行顺序**(实测,`tests/async_load_dev/analysis/raise_by_layer.py`):

| 运行名 | 提权次数 | 含多个组 | 含 sync 批 | 被提权任务数 | 单次最多组数 | 执行顺序按层号非降 | 多组提权的平均组间切换次数 |
|---|---|---|---|---|---|---|---|
| `rbl-perreq-r2-s1` | 6 | 2 | 0 | 371 | 3 | 6 / 6 | 150.5 |
| `rbl-perreq-s2` | 7 | 6 | 2 | 1131 | 5 | 7 / 7 | 176.7 |
| `rbl-batchreqs-s1` | 5 | 4 | 0 | 490 | 3 | 5 / 5 | 80.5 |
| `rbl-batchreqs-s2` | 8 | 4 | 0 | 993 | 5 | 8 / 8 | 229.2 |
| `rbl-batchreqs-r2-s2` | 8 | 4 | 0 | 987 | 5 | 8 / 8 | 229.2 |

列说明:一次提权 = trace 中 `p0_enter_ns` 相同的全部 unzip 任务(`set_groups_priority` 给一次移动的任务记同一个时间);"含多个组"= 一次移动 ≥ 2 个组的提权次数;"含 sync 批"= 其中含 `submit_priority` = 0 的任务(本 step 的 sync 批被一起重排)的提权次数;"执行顺序按层号非降"= 按 `start_ns` 排序后层号不下降的提权次数 / 提权次数;"组间切换次数"= 按执行顺序相邻两任务组号不同的次数,按请求排时为"组数 − 1",按层交错时接近"组数 × 层数"。

- 所有提权都按层号执行,多组提权时各请求的尾按层交错执行(切换 80–229 次,按请求排只会有 1–4 次)。§4.3 末尾说的 K1 第 2 点(提权后 P0 里按请求排)在这两个方案下消除。
- "sync 批与激活的尾一起按层排"只在 `rbl-perreq-s2` 出现(2 次);`_batch_reqs_` 三轮都没有出现(sync 请求到达的 step 恰好没有激活 async 请求)。两个方案的这段代码相同(`start_load_kv()` 的 K3 循环之后)。

**GSM8K(TP=2)**:配置与 §11.11 相同(Qwen3-32B bf16,TP=2 在 GPU4、5,map `0-3:0,4-6:4,7-:8`,200 题,并发 50,temperature 0,同一服务连跑 2 次、中间不清 DDR,开事件日志),connector md5 `4dafb666`。结果 `_data/gsm8k/<方案>/`。

| 方案 | 运行 | 答对 / 总数 | mean_acc | 命中外部缓存的请求 | 外部加载 token 平均 / 最少 / 最多 | 选到的 N(请求数) | sync / async / async_remaining 提交行 | 多请求激活行 | 报错行 | 乱码 |
|---|---|---|---|---|---|---|---|---|---|---|
| per_req | run1 | 191 / 200 | **0.955** | 198 / 200 | 570 / 0 / 576 | N=0:3,N=4:3,N=8:194 | 2 / 394 / 0 | 82 | 0 | 0 |
| per_req | run2 | 191 / 200 | **0.955** | 200 / 200 | 653 / 608 / 720 | 同上 | 4 / 394 / 0 | 108 | 0 | 0 |
| batch_reqs | run1 | 191 / 200 | **0.955** | 199 / 200 | 573 / 0 / 576 | 同上 | 2 / 182 / 394 | 86 | 0 | 0 |
| batch_reqs | run2 | 192 / 200 | **0.960** | 200 / 200 | 653 / 608 / 720 | 同上 | 4 / 170 / 394 | 100 | 0 | 0 |

列说明:per_req / batch_reqs 分别是 `priority_3level_first_n_per_req_remaining_by_layer` / `priority_3level_first_n_batch_reqs_remaining_by_layer`;"sync / async / async_remaining 提交行"= vLLM log 中三种 `submit kind=` 的行数,两个 TP rank 合计(per_req 每个 async 请求一次 `get()`,头尾同一次,197 × 2 = 394;batch_reqs 的头按 step 合并,尾每请求一次);"多请求激活行"= `activate reqs=` 中含 ≥ 2 个请求的行数,两个 rank 合计,每行对应一次多组按层提权;其余同 §11.9、§11.11。

- 准确率 0.955–0.960,与 split(0.955 / 0.950)、prio / raise / batch(0.945–0.955)同水平。同一方案两次逐题比:per_req 最终答案相同 191 / 200、对错不同 2 题;batch_reqs 相同 192 / 200、对错不同 3 题(未开 batch invariant,推断为运行间波动,同 §11.9)。
- TP=2 下多组提权大量发生(每次 41–54 次 / rank);sync 批与激活同 step 的次数为 0(sync 只在开头 1–2 个 step 出现),GSM8K 没有覆盖"sync 批一起按层排"。

**输出正确性(TP=2)**:GPU4、5,`VLLM_BATCH_INVARIANT=1`,检查方法与上面相同(`QUEUE=0 TP=2 GPU=4,5 CPU_BIND='64-95|96-127' run-queue-and-correctness.sh`)。每轮 log 中有 `Worker_TP0` / `Worker_TP1` 两个进程,两个 rank 的 activate 行数相同。

| 运行名 | 方案 | 服务配置 | 4 个请求的路径(rank 0) | 与参考逐字节一致 | 乱码 | 结论 |
|---|---|---|---|---|---|---|
| `tp2-perreq-s1-correctness` | per_req | S1 | 4 个 async;step 73 一次激活 2 个请求(多组按层提权),另 1 个 held 一个 step | 4 / 4 | 无 | PASS |
| `tp2-perreq-s2-correctness` | per_req | S2 | 3 个 sync + 1 个 async(sync 与激活不在同一 step) | 4 / 4 | 无 | PASS |
| `tp2-batchreqs-s1-correctness` | batch_reqs | S1 | 4 个 async;step 70 两个请求的头合成一次提交,step 74 一次激活 2 个请求 | 4 / 4 | 无 | PASS |
| `tp2-batchreqs-s2-correctness` | batch_reqs | S2 | 3 个 sync + 1 个 async(sync 与激活不在同一 step) | 4 / 4 | 无 | PASS |

列说明同 §11.6 的输出正确性表;参考请求 `externally-cached tokens` = 0,4 个命中请求各 4960。

- TP=2 下未覆盖:"sync 批与激活的尾一起按层排"(GSM8K 与正确性检查都没有遇到),见 §10。
- TTFT / 吞吐见 §11.16。

### 11.16 72B 性能:两个 remaining_by_layer 新方案(10-22)

配置:与 §11.12 相同的 72B TP=8 环境;方案 sync / prio / rbl_pr / rbl_br / batch,分别是 `KVSHRINK_VLLM_KV_ASYNC_LOAD_ENABLED=0` / `priority_3level_first_n_per_req` / `priority_3level_first_n_per_req_remaining_by_layer` / `priority_3level_first_n_batch_reqs_remaining_by_layer` / `batch_reqs_async_load_submit`;map 4/8、8/16、16/32(`0-3:0,4-6:<a>,7-:<b>`);输入 8k / 16k / 32k,hit 80 / 98 / 100,c4–c32(16k 最多 c16,32k 最多 c8),每格 1 次;13 个 server × 27 格 = 351 格,10-22 10:10 – 10-23 01:24,无失败格、无重启。运行顺序:sync,然后按 map,同一 map 的 4 个 async 方案连着跑。所有 server(含 sync)开 `KVSHRINK_ASYNC_LOAD_EVENT_LOG=1`。结果 `_data/perf-qwen72b/20261022_101029/`;分析脚本 `tests/async_load_dev/analysis/round1022.py`(性能)、`events1022.py`(事件日志),输出另存为结果目录下的 `analysis_*.txt`。

**各配置相对 sync**(实测,27 格几何平均):

| 配置 | TTFT | p95 TTFT | TPOT | 端到端 | 时长 | TTFT 优于 sync 的格 | 最差格 |
|---|---|---|---|---|---|---|---|
| prio 4/8 | −10% | +2% | +7% | +2% | +2% | 18 / 27 | +15%(8k h98 c16) |
| rbl_pr 4/8 | −13% | −5% | +8% | +2% | +2% | 19 / 27 | +14%(8k h80 c32) |
| rbl_br 4/8 | −11% | −3% | +7% | +1% | +2% | 21 / 27 | +17%(8k h80 c32) |
| batch 4/8 | −21% | −4% | +8% | +1% | +1% | 23 / 27 | +13%(8k h80 c32) |
| prio 8/16 | −19% | −3% | +8% | +1% | +2% | 18 / 27 | +18%(8k h98 c16) |
| rbl_pr 8/16 | −22% | −6% | +8% | +0% | +1% | 22 / 27 | +14%(8k h80 c32) |
| rbl_br 8/16 | −21% | −5% | +9% | +1% | +1% | 20 / 27 | +18%(8k h80 c32) |
| batch 8/16 | −24% | −4% | +7% | −1% | −0% | 22 / 27 | +11%(8k h80 c32) |
| prio 16/32 | −24% | −5% | +7% | −1% | −0% | 24 / 27 | +30%(16k h80 c16) |
| rbl_pr 16/32 | −24% | −5% | +7% | −0% | +0% | 23 / 27 | +31%(16k h80 c16) |
| rbl_br 16/32 | −25% | −3% | +8% | −0% | +0% | 22 / 27 | +31%(16k h80 c16) |
| batch 16/32 | −25% | −2% | +7% | −1% | −0% | 23 / 27 | +31%(16k h80 c16) |

列说明:每格"该配置 ÷ sync",对 27 格取几何平均后减 1,负数为更好;"端到端"= 平均 TTFT + 127 × 平均 TPOT;"时长"= bench 的 `duration`;"最差格"= 平均 TTFT 相对 sync 变慢最多的格(h 命中率,c 并发)。

**按命中率**(TTFT / 端到端,各 9 格几何平均):

| 配置 | hit 80 | hit 98 | hit 100 |
|---|---|---|---|
| prio 4/8 | −7 / +1% | −3 / +5% | −19 / −0% |
| rbl_pr 4/8 | −8 / +1% | −10 / +3% | −21 / +2% |
| rbl_br 4/8 | −8 / +1% | −9 / +3% | −17 / +1% |
| batch 4/8 | −8 / +1% | −12 / +3% | −38 / −2% |
| prio 8/16 | −9 / +0% | −1 / +6% | −41 / −2% |
| rbl_pr 8/16 | −8 / +1% | −12 / +3% | −40 / −2% |
| rbl_br 8/16 | −7 / +1% | −16 / +3% | −36 / −1% |
| batch 8/16 | −9 / +1% | −13 / +2% | −43 / −5% |
| prio 16/32 | −7 / +1% | −14 / +3% | −46 / −5% |
| rbl_pr 16/32 | −5 / +1% | −14 / +3% | −46 / −5% |
| rbl_br 16/32 | −7 / +1% | −21 / +3% | −43 / −4% |
| batch 16/32 | −6 / +1% | −21 / +3% | −44 / −6% |

**方案两两比较**(同 map 同格的 TTFT 之比,27 格几何平均;"前向等待"= rank 0 各 step `step_end` 的 `wait_ms` 之和,27 格合计之比):

| 比较 | map | TTFT | 前者更快的格 | 单格范围 | 端到端 | 前向等待 |
|---|---|---|---|---|---|---|
| rbl_pr / prio | 4/8 | −4% | 18 / 27 | 0.78–1.09 | +0% | 0.83 |
| rbl_pr / prio | 8/16 | −3% | 20 / 27 | 0.77–1.16 | −1% | 0.86 |
| rbl_pr / prio | 16/32 | +0% | 10 / 27 | 0.87–1.19 | +0% | 0.92 |
| rbl_br / prio | 4/8 | −1% | 18 / 27 | 0.82–1.22 | −0% | 0.84 |
| rbl_br / prio | 8/16 | −2% | 12 / 27 | 0.53–1.15 | −0% | 0.87 |
| rbl_br / prio | 16/32 | −1% | 9 / 27 | 0.65–1.30 | +1% | 0.88 |
| rbl_br / rbl_pr | 全部 | +1% | 33 / 81 | 0.61–1.34 | — | — |
| batch / prio | 4/8 | −12% | 19 / 27 | 0.55–1.06 | −1% | 0.60 |
| batch / prio | 8/16 | −6% | 19 / 27 | 0.71–1.14 | −2% | 0.66 |
| batch / prio | 16/32 | −1% | 8 / 27 | 0.64–1.18 | −0% | 0.84 |
| rbl_br / batch | 4/8 | +12% | 8 / 27 | 0.87–1.86 | +1% | 1.40 |
| rbl_br / batch | 8/16 | +4% | 10 / 27 | 0.56–1.37 | +2% | 1.32 |
| rbl_br / batch | 16/32 | +0% | 10 / 27 | 0.55–1.53 | +1% | 1.04 |

**前向等待(秒,rank 0,各命中率 9 格合计,含每格 5 个 warmup;实测)**:

| map | 命中率 | sync | prio | rbl_pr | rbl_br | batch |
|---|---|---|---|---|---|---|
| 4/8 | 98 | 12.7 | 37.6 | 14.9 | 14.9 | 15.2 |
| 4/8 | 100 | 98.5 | 106.3 | 102.7 | 103.9 | 67.5 |
| 8/16 | 98 | 12.7 | 33.3 | 14.2 | 13.8 | 15.9 |
| 8/16 | 100 | 98.5 | 93.3 | 93.4 | 95.0 | 64.6 |
| 16/32 | 98 | 12.7 | 16.7 | 13.0 | 11.9 | 16.3 |
| 16/32 | 100 | 98.5 | 60.1 | 57.3 | 55.5 | 47.8 |

列说明:hit 80 下各配置 8.6–10.9 s,差别小,未列;sync 只有一组数据,各 map 行重复。每个配置每个命中率约 10.3k 个 step,其中一次激活 ≥ 2 个请求的 step 63–145 个,sync 批与激活同 step 的只有 0–12 个。

**噪声与事件日志开销**(实测,与 10-20 那一轮相同配置相同格比较平均 TTFT):sync、prio 8/16、prio 16/32、batch 8/16、batch 16/32 共 135 对,中位数 1.00,118 对在 ±10% 内、131 对在 ±20% 内;"配置 ÷ sync"两轮之比的几何平均 −1% ~ +2%。其中 sync 两轮(10-20 未开事件日志、本轮开)27 对中位数 1.00、25 对在 ±10% 内,看不出事件日志的开销。

实测结论:

- 新方案修好了 prio 在 hit 98 的问题:map 4/8、8/16 下 prio 的 hit 98 TTFT 只比 sync 快 1–3%,rbl_pr / rbl_br 快 9–16%;对应的前向等待从 33–38 s 降到 14–15 s(与 batch 同水平)。差别集中在 c8 及以上的格(如 32k c8 map 4/8:prio 12.7 s,rbl 2.9 s;TTFT 4147 → 3378 ms),包括 sync 批与激活同 step 次数为 0 的格(8k c32:4.8 → 1.9 s)。
- 全部 81 格合计,rbl_pr / prio 为 −2%、rbl_br / prio 为 −2%,在两轮之间的漂移(±1–2%)附近;map 16/32 下四个方案整体相同(−24% ~ −25%)。
- rbl_pr 与 rbl_br 分不出高下(+1%,33 / 81)。
- batch 不比任何 priority 方案差:map 4/8、8/16 下明显更好(比 prio 快 12% / 6%,rbl_br 比 batch 慢 12% / 4%),主要在 hit 100(4/8:batch −38%,三个 priority 方案 −17% ~ −21%),hit 100 的前向等待也比 prio 少 31–36%;map 16/32 下与其他方案相同。
- 端到端与时长:所有配置相对 sync 在 −1% ~ +2% 之间,只有 hit 100 为 −2% ~ −6%;与 §11.12、§11.14 一致,TTFT 的提升在 hit 80 / 98 主要是 TTFT 与 TPOT 的再分配。
- 最好的配置:map 16/32 下任意方案,或 batch 8/16(TTFT −24% ~ −25%)。

推断:

- prio 在 hit 98 慢的原因是激活后各请求的尾在 P0 里按请求排(§4.3 末尾 K1 第 2 点):前向要第 k 层时,别的请求的全部层还排在前面。按层提权后前向按层拿到数据。sync 批合并的情况太少(每配置每命中率 ≤ 12 个 step),对结果没有可见影响。
- batch 在 hit 100、小 map 下更快,可能有两个原因(未验证):一是 FIFO 下先到请求的尾先执行,前向等待的尾更早完成,而 priority 方案让后到请求的头(P1)插到尾(P2)前面;二是 batch 同 step 的请求每层共用一个任务,任务数少,任务 release 开销小([task-release-overhead.zh-CN.md](task-release-overhead.zh-CN.md))。
- 每格只跑 1 次,单格两轮之间可差到 0.80–1.81 倍;上面"方案两两比较"中几个百分点的差别只能看整体与机制(前向等待),不能看单格。

**最优"方案 + map"**(实测,10-20 与 10-22 两轮所有组合放在同一组 27 格上,即 8k / 16k / 32k × hit 80 / 98 / 100 × 各并发,每轮除以本轮 sync;按平均 TTFT 排序的前 7 名):

| 组合 | 平均 TTFT | 10-20 TTFT / 端到端 / p95 | 10-22 TTFT / 端到端 / p95 |
|---|---|---|---|
| raise 16/32 | −25.2% | −25.2 / −2.2 / −4.4% | — |
| rbl_br 16/32 | −25.0% | — | −25.0 / −0.0 / −2.7% |
| batch 16/32 | −24.9% | −24.6 / −1.4 / −1.8% | −25.2 / −0.9 / −2.2% |
| raise 8/16 | −24.9% | −24.9 / −0.9 / −2.7% | — |
| prio 16/32 | −24.9% | −25.4 / −0.2 / −6.2% | −24.4 / −0.6 / −5.1% |
| batch 8/16 | −24.4% | −25.1 / −0.9 / −5.4% | −23.6 / −0.6 / −3.8% |
| rbl_pr 16/32 | −24.0% | — | −24.0 / −0.2 / −5.2% |

列说明:"平均 TTFT"= 有数据的各轮几何平均相对 sync 再取平均;raise = `priority_3level_first_n_per_req_promote_raise`(只在 10-20 跑过);map 40/79 的三个方案为 −22.4% ~ −23.3%,map 4/8 为 −10% ~ −21%。

- map:每个方案的最优 map 都是 16/32(batch 的 8/16 与之持平);40/79 与 4/8 都更差。
- 方案:16/32 下五个方案相差 1.2 个百分点(−24.0% ~ −25.2%),而同一组合两轮之间就差 0.6–1.5 个百分点(prio 16/32:−25.4 / −24.4;batch 16/32:−24.6 / −25.2;batch 8/16:−25.1 / −23.6),数据分不出唯一最优的方案。端到端都在 0 ~ −2%。
- 若只选一个(推断):按同一轮(10-22)比较,prio / rbl_pr / rbl_br 在各自最优 map(都是 16/32)下 TTFT −24.4 / −24.0 / −25.0%、p95 −5.1 / −5.2 / −2.7%,前向等待 1 : 0.92 : 0.88。prio 的"两轮都复现"只是因为它跑过两轮,不是更好的证据;新方案在最优 map 下与 prio 打平,在较小 map 下明显更好(见上面 hit 98),优先选新方案。

**rbl_pr 与 rbl_br,各自最优 map(都是 16/32,在 4/8、8/16、16/32 中选;40/79 未测)**(实测,10-22 同一轮逐格配对,"rbl_br ÷ rbl_pr"的几何平均):

| 范围 | TTFT | rbl_br 更快的格 | 单格范围 | p95 TTFT | TPOT | 端到端 |
|---|---|---|---|---|---|---|
| 全部 27 格 | −1% | 12 / 27 | 0.62–1.34 | +3% | +1% | +0% |
| hit 80 | −2% | 6 / 9 | 0.75–1.08 | −2% | +2% | +0% |
| hit 98 | −8% | 5 / 9 | 0.62–1.08 | −3% | +2% | +0% |
| hit 100 | +6% | 1 / 9 | 0.97–1.34 | +13% | +0% | +1% |
| 8k | +2% | 6 / 12 | 0.88–1.34 | +5% | −0% | +0% |
| 16k | +2% | 2 / 9 | 0.84–1.08 | +5% | −1% | +0% |
| 32k | −12% | 4 / 6 | 0.62–1.03 | −5% | +7% | +0% |

事件日志(rank 0,hit 80 / 98 / 100 各 9 格合计):前向等待 rbl_pr 9.0 / 13.0 / 57.3 s,rbl_br 8.6 / 11.9 / 55.5 s(合计 79.3 对 76.0 s,−4%);一次激活 ≥ 2 个请求的 step 135 / 124 / 81 对 145 / 112 / 103;sync 批与激活同 step 2 / 10 / 0 对 5 / 1 / 0。

- 整体只差 1%(TTFT),小于同一组合两轮之间的 0.6–1.5 个百分点;端到端相同。
- 分场景有方向:rbl_br 在 hit 98(−8%)和 32k(−12%)更快,在 hit 100 更慢(+6%,9 格中 8 格更慢,p95 +13%)。各子集只有 6–9 格、各跑 1 次,只作参考。
- rbl_br 前向等待少 4%,与 hit 98 / 32k 的 TTFT 方向一致;hit 100 等待也少 3%,但 TTFT 反而更慢,原因未查。

**async 请求的时间线:prio / rbl_pr / rbl_br**(实测,只用 10-22 这一轮;rank 0 事件日志;脚本 `tests/async_load_dev/analysis/profile1022.py`)

一个 async 请求在 worker 上经过三段:

```
t_submit     start_load_kv 提交 get():前 N 层(P1)+ 尾(P2)进 OMP-Main 队列
   |  (1) 加载前 N 层:排队 + 解压 + 拷贝,直到前 N 层全部到 GPU
t_promote    get_finished 报告 promote,scheduler 下一次可以调度它
   |  (2) 等进 forward:scheduler 把它放进某个 step(含 held 的 step)
t_activate   该 step 的 start_load_kv 把它的尾提到 P0("激活")
   |  (3) forward 逐层等尾:该 step 的 forward 每层调 wait_for_layer_load(),
   |      等这一层所有请求(本 step 激活的各请求的尾 + 本 step 的 sync 批)到 GPU
step_end     wait_ms = 80 层等待时间之和
```

| map | 方案 | hit 98 (1) | hit 98 (2) | hit 98 (3) | hit 98 合计 | hit 100 (1) | hit 100 (2) | hit 100 (3) | hit 100 合计 |
|---|---|---|---|---|---|---|---|---|---|
| 4/8 | prio | 629 | 266 | 307 | 1202 | 328 | 259 | 540 | 1127 |
| 4/8 | rbl_pr | 622 | 204 | 76 | 902 | 295 | 265 | 533 | 1093 |
| 4/8 | rbl_br | 634 | 233 | 75 | 942 | 336 | 255 | 640 | 1231 |
| 8/16 | prio | 746 | 189 | 310 | 1245 | 390 | 125 | 197 | 713 |
| 8/16 | rbl_pr | 621 | 279 | 59 | 959 | 380 | 127 | 193 | 700 |
| 8/16 | rbl_br | 588 | 103 | 69 | 760 | 383 | 99 | 358 | 840 |
| 16/32 | prio | 741 | 192 | 67 | 999 | 392 | 96 | 96 | 584 |
| 16/32 | rbl_pr | 758 | 124 | 41 | 923 | 393 | 89 | 96 | 578 |
| 16/32 | rbl_br | 612 | 165 | 38 | 816 | 404 | 115 | 149 | 668 |

列说明:单位 ms;(1) 加载前 N 层 = t_promote − t_submit;(2) 等进 forward = t_activate − t_promote;(3) forward 逐层等尾 = 该请求被激活的那个 step 的 `wait_ms`,即 forward 在 80 层 `wait_for_layer_load()` 里的等待之和(只有这一步要等尾:之后各层都已加载,再等是空操作);各列按请求平均,每格约 560–620 个请求(该命中率 9 格合计,含每格 5 个 warmup)。事件日志只记每个 step 的 80 层合计,没有逐层的值。hit 80 下 (2) 约 2 s(prefill 占满 token 预算),三者差别被它淹没,未列。"合计"不是 TTFT:TTFT 还包括提交前的排队与 prefill 计算。对照:三个方案的 `save_ms` 合计都是 1750–1760 s。

实测结论:

- 新方案 vs prio:差别在 (3)。hit 98 下 prio 为 307 / 310 / 67 ms,rbl_pr 为 76 / 59 / 41 ms,rbl_br 为 75 / 69 / 38 ms;hit 100 下 rbl_pr 与 prio 相同(533 / 193 / 96 对 540 / 197 / 96),rbl_br 更大。
- rbl_pr vs rbl_br:没有一方在所有场景占优。
  - hit 98 rbl_br 更好:(1) 更快(16/32:612 对 758 ms;8/16:588 对 621 ms),合计 816 对 923 ms、760 对 959 ms;与 TTFT 的 hit 98 −8% 一致。
  - hit 100 rbl_pr 更好:rbl_br 的 (3) 大(16/32:149 对 96 ms;8/16:358 对 193 ms;4/8:640 对 533 ms),合计 668 对 578 ms;与 TTFT 的 hit 100 +6% 一致。
- 推断:rbl_br 同一 step 的请求共用前 N 层的提交,任务更少、(1) 更快;但这些请求也因此同时 promote、同时激活,一次 forward 要等的尾更多(16/32 hit 100 一次激活 ≥ 2 个请求的 step:103 对 81),(3) 更大。

### 11.17 固定层数(`..._LAYERS_DYNAMIC=0`)与动态 map 对比(32k)

数据:固定层数只有 10-21 那一轮(§11.14),只跑了 `priority_3level_first_n_per_req`、32k、c4 / c8,每格 2 次取平均。动态 map 用同一方案在三轮里的结果:10-20(§11.12,map 8/16、16/32、40/79)、10-21(map 16/32,与固定层数同一轮)、10-22(§11.16,map 4/8、8/16、16/32)。各轮的 sync 不同,每格先除以本轮的 sync 再比较。三轮共有的格:32k × hit 98 / 100 × c4 / c8,共 4 格。脚本 `round1022.py` 的 B 部分。

| 轮 | 设置 | TTFT | p95 TTFT | TPOT | 端到端 | 时长 | TTFT h98 | TTFT h100 | 端到端 h98 | 端到端 h100 | TTFT c4 | TTFT c8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10-21 | N = 4 | +22% | +22% | +0% | +8% | +8% | +42% | +5% | +15% | +2% | +13% | +32% |
| 10-21 | N = 8 | +16% | +16% | +1% | +6% | +6% | +31% | +2% | +12% | +1% | +14% | +17% |
| 10-21 | N = 16 | −20% | −5% | +9% | +1% | +2% | +4% | −38% | +6% | −4% | −16% | −23% |
| 10-21 | N = 32 | −32% | −8% | +7% | −4% | −3% | −5% | −52% | +4% | −11% | −25% | −39% |
| 10-21 | **N = 48** | **−36%** | −7% | +7% | **−5%** | −4% | −9% | **−54%** | +6% | −15% | −25% | **−45%** |
| 10-21 | N = 64 | −32% | −4% | +7% | −4% | −3% | −9% | −50% | +7% | −14% | −23% | −41% |
| 10-21 | N = −1 | −27% | +2% | +6% | −3% | −2% | −5% | −45% | +8% | −13% | −16% | −38% |
| 10-21 | map 16/32 | −32% | −11% | +8% | −3% | −2% | −9% | −49% | +4% | −9% | −29% | −35% |
| 10-20 | map 8/16 | −25% | −4% | +14% | +2% | +3% | +0% | −44% | +8% | −3% | −21% | −29% |
| 10-20 | map 16/32 | −38% | −15% | +15% | −2% | −1% | −21% | −52% | +5% | −8% | **−35%** | −41% |
| 10-20 | map 40/79 | −37% | −11% | +13% | −3% | −2% | −22% | −49% | +6% | −11% | **−35%** | −38% |
| 10-22 | map 4/8 | −10% | +2% | +11% | +4% | +5% | −0% | −19% | +8% | +1% | −13% | −6% |
| 10-22 | map 8/16 | −30% | −5% | +17% | +3% | +3% | −5% | −48% | +8% | −2% | −23% | −36% |
| 10-22 | map 16/32 | −35% | −8% | +14% | −2% | −1% | −13% | −51% | +5% | −8% | −25% | −43% |

列说明:每格"该设置 ÷ 本轮 sync",对所选格取几何平均后减 1,负数为更好;前 5 列为 4 格合计,其后按命中率(各 2 格)、并发(各 2 格)拆开;"端到端""时长"同 §11.16。加上 hit 95(只有 10-20、10-21 有,6 格)时:N = 48 −27%,N = 64 −26%,map 16/32 −25%(10-21)/ −30%(10-20),结论不变。

实测结论:

- 最好的固定层数(N = 48)与最好的 map(16/32、40/79)TTFT 相差 4 个百分点以内:同一轮(10-21)N = 48 −36%、map 16/32 −32%;map 16/32 在三轮里分别为 −38% / −32% / −35%,同一设置跨轮的变化就有 6 个百分点。端到端 N = 48 −5%、map 16/32 −2% ~ −3%,同样在噪声附近。
- 按并发拆开有方向性:c4 下 map 更好(同一轮 −29% 对 N = 48 的 −25%;10-20 的 map 16/32、40/79 为 −35%),c8 下 N = 48 更好(−45% 对 −35% ~ −43%)。c4 下 map 让在途 ≤ 3 的请求走 sync,固定 N 则全部走 async。
- 固定层数选错的代价大:N = 4 / 8 比 sync 慢 16–22%,N = 16 只有 −20%;map 中最小的 4/8 也只有 −10%。
- 固定层数没有 8k / 16k 的数据,8k / 16k 上无法比较。

推断:32k 下固定 N = 48 与 map 16/32 打平,固定层数没有明显优势,且需要按输入长度选 N;建议保留动态 map(16/32)。若要拿到 c8 下 N = 48 的那部分收益,可以试 `0-3:0,4-6:16,7-:48`(§11.14 已提,未测)。
