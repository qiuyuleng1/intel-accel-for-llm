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
| `split_async_load_submit` | async 请求先只提交头(前 N 层);promote 后的下一次 `start_load_kv` 再提交尾(后 L−N 层) | `batch_reqs_async_load_submit` |
| `priority_3level` | 一次提交全部 L 层;队列加 3 级优先级(P0/P1/P2),promote 时把尾从 P2 调到 P0 | `batch_reqs_async_load_submit` |
| `split_priority_2level` | 按 `split_async_load_submit` 分两次提交;队列加 2 级固定优先级(P0/P1):头用 P1,尾和 sync 用 P0,P0 内按层号排 | `batch_reqs_async_load_submit` |

列说明:"名字"用于文档、代码和 `KVSHRINK_ASYNC_LOAD_SCHEME` 的取值;"做法"写的是与 `naive` 的区别;"前提"指该方案是否建立在 `batch_reqs_async_load_submit` 之上。

`split_priority_2level` 不是 `split_async_load_submit` 与 `priority_3level` 的叠加:它沿用 split 的提交方式,但优先级是另一套 2 级固定方案,与 `priority_3level` 只共用"P0 内按层号排序"这一点(对比见 4.4)。

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

### 4.3 `priority_3level`

一次性提交 L 层不变;`TaskQueue` 改为 3 级且优先级可变:

| 级别 | 内容 | 级内排序 |
|---|---|---|
| P0 | forward 正在等的:sync 批全部层 + 已 promote 请求的尾 | 按 layer 号跨请求交错,同层按提交序 |
| P1 | 未 promote 请求的头 | 按提交序 FIFO |
| P2 | 未 promote 请求的尾(预取) | 按提交序 FIFO |

promote 时对该请求的尾调用 `set_priority(P0)`。

P1 用 FIFO 而非按层交错是有意的:头的目标是让**某一个**请求尽快凑齐 N 层去被调度;按层交错(A0 B0 A1 B1…)会让两个请求都拖到最后才凑齐。

P2 存在的唯一理由:让 B 的头(P1)能压过 A 尚未 promote 的尾。若 P1、P2 合并,A 的 48 层(先提交)与 B 的头(后提交)同级 FIFO,(a) 复现。

与公共前提的关系:同一 batch 的请求共用尾任务。batch 内各请求 N 相同时它们同时 promote;N 不同时,第一个 promote 的请求会把整批的尾提到 P0,其余请求的尾随之提前,无害。

### 4.4 `split_priority_2level`

按 `split_async_load_submit` 提交:尾在 promote 前不进队列,队列里不存在"未 promote 的尾"这类任务,因此**不需要 P2,也不需要改优先级**。promote 后尾直接以 P0 提交。队列只需:

| 级别 | 内容 | 级内排序 |
|---|---|---|
| P0 | sync 批 + 已 promote 请求的尾 | 按 layer 号 |
| P1 | 未 promote 请求的头 | FIFO |

优先级和 layer 号在提交时指定,之后不变。

与 `priority_3level` 的区别:

| | `priority_3level` | `split_priority_2level` |
|---|---|---|
| 优先级级数 | 3(P0/P1/P2) | 2(P0/P1) |
| 优先级能否改 | 能:promote 时 `set_priority` | 不能:提交时定死 |
| 尾在 promote 前预取(P2) | 有 | 无 |
| P0 内按层号排序 | 是 | 是 |

## 5. 方案对比

"(a)(b)" 两行写的是 B 的头 / C 的 layer 0 **前面最多有多少无关任务**。`naive` 列是现状;其余三列都建立在 `batch_reqs_async_load_submit` 之上。

| 维度 | `naive` | `split_async_load_submit` | `priority_3level` | `split_priority_2level` |
|---|---|---|---|---|
| **(a)** B 头前面的无关任务 | 前面每个 async 请求各 L 层 | 前面未 promote 请求各 **N** 层 + 已 promote 请求各 L−N 层(后者 forward 正在等,排前面合理) | 只等前面请求的头(P1)和 P0;所有未 promote 尾(P2)排 B 后面 | 同 `priority_3level` |
| **(b)** C 的 layer 0 前面的无关任务 | 前面每个 async 请求各 L 层 | 前面未 promote 请求各 **N** 层 + 已 promote 请求各 L−N 层 | **0**:C 是 P0 且按层排,C0 到最前 | 同 `priority_3level` |
| 尾的提前解压时间 | 有 | **无**,靠流水线 | 有(P2 利用队列空闲预取) | 无,同 `split_async_load_submit` |
| 尾不卡 forward 的条件 | 宽松 | 单层解压 ≤ 单层 prefill 计算 | 宽松 | 单层解压 ≤ 单层 prefill 计算 |
| 队列优先级 | 2 级固定(HIGH/LOW) | 2 级固定(HIGH/LOW) | 3 级**可变** | 2 级固定 |
| C++ 改动 | — | 无 | 多级 + 可变优先级 + `set_priority` 绑定 + 按层排序 | 2 级 + 按层排序键;提交接口加 priority / layer 号参数 |
| Python 改动 | — | worker 保存 async 请求元数据;promote 后按 batch 补提交尾 | `get()` 加 per-layer priority;promote 时提权 | `split_async_load_submit` 的改动 + `get()` 加 priority / layer 号参数 |

结论:`split_async_load_submit` 单独能把 (a)(b) 从 L 层压到 N 层;`priority_3level` 解决 (a)(b) 但需可变优先级;`split_priority_2level` 解决 (a)(b) 且队列只需 2 级固定优先级,代价是尾没有提前量(本轮不评估,见 4.2)。P0 内按层号排序在公共前提之下作用变小(同一 step 内已按层排),只剩不同 step 提交的 P0 任务之间需要它,是否保留待实测。另外,已知问题 K2(§3)的延迟发生在任务进队列之前,本文方案对它无效。

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
| `priority_3level` | 同 batch,64 个任务;L0~L3 为 P1,L4~L63 为 P2,promote 后提到 P0 |
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
| `priority_3level` / `split_priority_2level` | S0 是 P0 且层号为 0,只等正在执行的那一个任务 |

### 7.5 要看的 log

| log | 内容 | 状态 |
|---|---|---|
| `GET /v1/cache/queue_trace` | 每个解压任务的 `label=unzip\|<req_id 列表>\|<层名>`、`priority`、`seq`、`enqueue_ns / start_ns / end_ns`;按 `start_ns` 排序即实际执行顺序 | 已有 |
| connector `get_num_new_matched_tokens` | 每个请求实际走 sync 还是 async、N 是多少 | 已有 |
| connector `start_load_kv` | 每次 `get()` 一行:`async_load step=<k> t=<ns> submit kind=sync\|async reqs=<id,...> layers=<a>-<b>` | 已加 |
| connector `get_finished` | `async_load step=<k> t=<ns> promote reqs=<id,...>` | 已加 |

`t` 是 `time.monotonic_ns()`,与 trace 同一时钟。sync 批的 description 也改为逗号分隔的 req_id,trace 里每个任务都能对应到具体的 submit 行。

`priority_3level` 的优先级会中途改变,做该方案时 trace 要记录执行时的优先级,而不是提交时的。

### 7.6 判定

判定脚本读取 7.5 的 log,对每次运行输出三项:

1. **提交结构是否与 7.4 一致**:每个 step 的任务数、层范围、req_id 组成。例:`batch_reqs_async_load_submit` 同一 step 恰好 64 个任务且 label 含该 step 全部 async 请求;split 类方案中,任何请求在 promote 记录之前不得有 L4 及以上的任务。
2. **违序数**:用 trace 重放队列,对每个任务检查它开始执行的那一刻,是否有"已入队、未开始、排序键更小"的任务。排序键:FIFO 类方案 `(HIGH/LOW, seq)`;优先级类方案 `(priority, 层号, seq)`。必须为 0。
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
- 风险 3:`TaskQueue` 的优先级排序和提权要新加单测(在 `iaxl/csrc/test/task_queue_test.cpp` 里扩展),覆盖"提权后顺序正确"。
- 风险 4:公共前提下尾任务被同 batch 请求共用,提权作用于整批(见 4.3)。
- 复杂度:**中~高**。难在可变优先级 cell 的生命周期(跟着 Context 走、重试复用、move 语义),不在排序本身。

### 8.5 `split_priority_2level`

代码上 = `split_async_load_submit` 的全部改动 + `priority_3level` 第 1、3、4、5、6 项的简化版(复用其 priority / layer 号透传路径,不用其可变优先级):

| 项 | 相对 `priority_3level` 的简化 |
|---|---|
| `TaskQueue` | 不需要可变 cell。3 个容器:P0 用 `std::multimap<(sort_key, seq), fn>`,P1 和 LOW(zip)各一个 deque。~50 行 |
| `unzip_from_mem` | 加 `priority, sort_key` 两个参数即可,**不需要 `set_priority`**,`Context` 不加新成员,move 语义不动 |
| 重试路径 | 重试时沿用原 priority / sort_key(局部捕获即可) |
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
4. `priority_3level` 的 priority / sort_key 透传(8.4 第 3~6 项)与 `split_priority_2level` 相同;做完后只剩"可变 cell + `set_priority` + promote 时提权"。
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

核心是"**任务可提权**"和"**P0 按 layer 排**"。

| 维度 | 验收标准 | 简单测试 |
|---|---|---|
| 多级队列生效 | P0 / P1 / P2 任务进入队列后,worker 取任务顺序符合 `(priority, sort_key, seq)` | 在 `task_queue_test.cpp` 里加用例:手工 submit 若干假任务,断言执行顺序与预期一致 |
| 可变优先级生效 | 任务入队后但尚未执行前,`set_priority(P0)` 能改变其出队顺序 | 同上加用例:先入队 A(P2)、B(P1),再把 A 提到 P0,断言 A 先执行 |
| retry 保持优先级 | cache miss 重试后的 unzip 任务仍保留原 priority / sort_key | 人工触发一次 retry,检查重试前后日志中的 priority / sort_key 相同 |
| (a) 缓解生效 | B 的头(P1)能越过 A 的未 promote 尾(P2) | 构造 A、B 两个 async 请求,查看队列执行顺序,确认 B0..B{N-1} 先于 A_N..A_{L-1} |
| (b) 缓解生效 | sync 批(C 的 S0)作为 P0,能越过所有 P1/P2 | 先让队列中存在 async 任务,再提交 1 个 sync 批,检查 S0 最先执行 |
| P0 按 layer 排 | 不同次提交进入 P0 的任务,按 layer 号交错执行而不是按提交批次排 | 同上加用例:分两批提交 P0 任务,断言执行顺序按 layer 号 |
| 线程安全/生命周期 | 不因 `Context` move、任务 finish、abort 导致悬空 cell 或 crash | 开启 ASAN/Debug 或至少跑多次并发压测,确认无 crash / double free / use-after-free |

验收的关键证据,不是 TTFT 下降本身,而是**队列顺序符合预期**。TTFT 只是后续 benchmark 指标。

### 9.5 `split_priority_2level` 验收标准

验收 = `split_async_load_submit` 的提交行为成立 + 2 级固定优先级成立。

| 维度 | 验收标准 | 简单测试 |
|---|---|---|
| 头尾分批提交 | 同 `split_async_load_submit`:首次只提交头,promote 后补提交尾 | 复用 9.3 的测试 |
| 固定优先级生效 | sync 与已 promote 尾以 P0 提交;未 promote 头以 P1 提交 | queue_trace 中检查每个 unzip 任务的 priority |
| P0 按 layer 排 | 不同次提交进入 P0 的任务按 layer 号交错 | 复用 9.4 的 P0 排序单测 |
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
- [ ] `split_async_load_submit`:按 4.2 / 8.3 实现(待补尾集合、分段提交、每请求 Task dict 副本、加载元数据、清理);判定脚本加 split 的结构检查(9.3);§7 S1、S2 都跑(结论不得为 FAIL)+ 输出正确性检查;abort 清理只做代码审查
- [ ] `split_priority_2level`:`TaskQueue` 2 级固定优先级 + P0 按 layer 号排序;`unzip_from_mem` / `KVFlow.get()` / `KVStore.get()` 透传 priority / layer 号;connector 接入;9.5 验收 + §7 S1、S2 都跑(结论不得为 FAIL)
- [ ] `priority_3level`:`TaskQueue` 可变 cell + `Context::set_priority` + pybind;trace 记录执行时优先级;connector promote 时提权;9.4 验收 + §7 S1、S2 都跑(结论不得为 FAIL)
- [ ] 代码清理(以后做):`_early_promoted_tasks` 与 `_active_promoted_tasks` 合并为一个"已 promote" dict。promote 只发生在 forward 之后的 `get_finished`,forward 进行中不会新增;二者的区别只是"是否已进入某次 forward"。需在 K3 修复之后重新评估(K3 让"搬入 active"变为按请求是否被调度)

## 11. 进度记录

- 2026-10-08:完成问题分析、方案设计与复杂度评估;创建分支 `perf/async-load-optimization-dev`。
- 2026-10-09:完成公共基础设施,并在 `naive` 上跑了 T2;方案改用 4.0 的名字,`batch_reqs_async_load_submit` 定为公共前提。完成 §7 测试工具(submit / promote log、benchmark 参数、判定脚本),`naive` 的 S1 PASS;S2 在实际映射表下未覆盖 (b)(§11.3)。step 层面的延迟记为已知问题 K2,不单独测试。完成 `batch_reqs_async_load_submit`,S1、S2 无 FAIL,输出正确(§11.4)。
- 2026-10-10:完成 split 详细设计;修复 K3,`naive`、`batch_reqs_async_load_submit` 的 S1、S2 无 FAIL,输出正确(§11.5)。输出正确性检查改为要求 vLLM 以 `VLLM_BATCH_INVARIANT=1` 启动;S2 速率改为 inf。

### 11.1 公共基础设施

- `TaskQueue`(`iaxl/csrc/include/task_queue.h`):开启后为每个任务记录 `label / priority / seq / enqueue_ns / start_ns / end_ns / high_depth / low_depth`,时钟为 steady_clock(= Python `time.monotonic_ns()`);关闭时只多一次原子读。上限 262144 条,超出计入 `dropped`。
- 标签(`torch_ext/zip.cpp`):`<kind>|<description>|<tensor_key>`,kind ∈ {`unzip`, `unzip_retry`, `zip`};description 是本次 `get()` 涉及的 req_id(逗号分隔;sync 批同样如此),put 为空。
- 读取:`GET /v1/cache/queue_trace?queue=OMP-Main&enable=1|0`(worker 端;controller 转发)。每次调用返回并清空已记录事件。
- 单测:`iaxl/csrc/test/task_queue_test.cpp`(`make test-task-queue`),21 项通过;TSan 0 告警(需 `setarch $(uname -m) -R` 运行)。

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
