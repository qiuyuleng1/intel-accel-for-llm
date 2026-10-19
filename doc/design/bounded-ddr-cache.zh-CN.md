# 有界 DDR 缓存（bounded DDR cache）设计讨论记录

> 分支：`feat/bounded-ddr-cache`（从 `main` 拉出）
> 状态：方案讨论中，**尚未实现任何代码**；LRU-A / LRU-B 二选一待定。
> 依据：本仓库源码 + 本地 vLLM 0.29.0 源码（`../vllm-0.29.0`，以下简写为 `vllm-0.29.0/`）。

结论标注约定：
- 【代码证据】引用代码位置；
- 【文档】引用仓库文档；
- 【技术硬约束】无法绕开的技术限制；
- 【设计选择】代码"选择"这样做，而非"只能"这样做；
- 【推测】未经代码验证的判断，附置信度。

---

## 1. 问题

用 [tests/vllm-benchmark.sh](../../tests/vllm-benchmark.sh) 做高并发测试（如并发 32）时，需要较大的 `num_prompts`（如 320 或更多）才能得到稳定结果，但 host 内存经常不够，导致无法完成 benchmark。目标机器形态示例：8 卡、每卡约 198GB 显存，但 host 内存只有约 1TB。

脚本头部注释给出的手动 `/v1/cache/evict` 只能在两轮测试之间执行，**单轮 case 跑的过程中无法回收**。

---

## 2. 两个环境变量

### 2.1 `IAXL_DDR_POOL_SIZE_GB`：DDR 缓存池预算（"仓库"）

- 作用：卸载到 host DRAM 的 KV cache 池容量预算。【代码证据】[kvstore.py](../../iaxl/kvstore/kvstore.py#L95-L99) → `KVFlow(cache_size_gb=...)` → [flow.py](../../iaxl/kvflow/flow.py#L103-L107) `Mem(capacity_bytes=...)`。
- 默认值：未设置时为 **当前 available RAM 的 1/10**。【代码证据】[kvstore.py](../../iaxl/kvstore/kvstore.py#L24-L26)
- **它是软预算，不是硬上限**：超出时只每 1 万次打印一条 `Over budget` 警告，然后继续 malloc，不会自动淘汰。【代码证据】[mem.cpp](../../iaxl/csrc/kv_pool/mem.cpp#L621-L628)
- 这是 OOM 的直接原因：单轮测试中没有任何自动淘汰。淘汰只能由外部调用 `/v1/cache/evict` 触发（[kvstore.py](../../iaxl/kvstore/kvstore.py#L141-L144) → [flow.py](../../iaxl/kvflow/flow.py#L546)）。

### 2.2 `IAXL_SCRATCH_POOL_SIZE_GB`：pinned 中转池（"装卸月台"）

- 作用：GPU ↔ DDR 之间的**固定大小 pinned 中转缓冲**，不存放 KV。【代码证据】[scratch_pool.py](../../iaxl/kvflow/scratch_pool.py#L4-L10)、[scratch_pool.py](../../iaxl/kvflow/scratch_pool.py#L29)
- 循环复用：put/get 时先 `allocate` 借出 block，传输完成后 `release` 归还（[flow.py](../../iaxl/kvflow/flow.py#L209)、[flow.py](../../iaxl/kvflow/flow.py#L271)）。池子不会增长，用尽时直接抛出 `ScratchPool exhausted`（[scratch_pool.py](../../iaxl/kvflow/scratch_pool.py#L48)）。因此它决定的是**同时在途的传输块数上限**。
- 开启 RDMA 时，整块 scratch 还会被注册为 RDMA 区域（[flow.py](../../iaxl/kvflow/flow.py#L146-L147)）。
- pinned 内存是锁页的、不可换出，所以这个值越大，实际占用的 RAM 越多。

### 2.3 两者都是 per-rank 的

- 【代码证据】每个 rank 进程有自己独立的 KVStore → KVFlow → `ScratchPool` 和 `Mem`（[flow.py](../../iaxl/kvflow/flow.py#L144)、[flow.py](../../iaxl/kvflow/flow.py#L103)）。remote_pool daemon 也是每个 TP rank 一个进程（[daemon.py](../../iaxl/remote_pool/daemon.py#L5-L7)）。
- 因此**整机占用 = 本机 rank 数 × 设定值**。例如 TP=8、scratch 用默认 8GB，就是 64GB pinned 内存。
- `IAXL_DDR_POOL_SIZE_GB` 的默认值由**每个 rank 各自**按"available RAM 的 1/10"计算，8 个 rank 加起来约等于 80% 的 RAM，而且只是软预算。建议显式设置一个保守值。
- 这两个变量只在 daemon/worker 侧生效，client 侧设置无效。【文档】[remote-pool-kvstore.zh-CN.md](../usage/remote-pool-kvstore.zh-CN.md#L148-L149)

---

## 3. 数据路径与拷贝次数

### 3.1 次数

| 模式 | save | load |
|---|---|---|
| **不压缩** | **2 次**：GPU→scratch（DMA），scratch→DDR（CPU memcpy） | **2 次**：DDR→scratch（CPU memcpy），scratch→GPU（DMA） |
| QAT / IAA 压缩 | **4 次**：GPU→scratch（DMA），scratch→加速器输入缓冲（memcpy），硬件压缩写入输出缓冲，输出缓冲→DDR（memcpy） | **4 次**：上述过程反向 |
| CPU zlib 压缩 | **3 次**：zlib 直接读取 scratch，省掉一次 memcpy | **3 次** |

"多拷一次"指的就是不压缩时每个方向多出的那 1 次 CPU memcpy。另外，压缩路径还会在 scratch 上原地做 shuffle / 截断，这些是对数据的遍历处理，不计为拷贝。

### 3.2 代码证据

- save：DMA（[flow.py](../../iaxl/kvflow/flow.py#L224)）→ `zip_to_mem`（[flow.py](../../iaxl/kvflow/flow.py#L228)）→ 等待 DMA 完成（[zip.cpp](../../iaxl/csrc/torch_ext/zip.cpp#L132)）→ 压缩或拷贝 → `Mem.put`（[zip.cpp](../../iaxl/csrc/torch_ext/zip.cpp#L148)，只保存指针，不再拷贝）。
  - 不压缩时：先 `malloc`，再 `memcpy`（[kv_zip.cpp](../../iaxl/csrc/kv_zip/kv_zip.cpp#L151-L167)）。
  - 压缩时：拷入加速器 slot（QAT [qat_zip.c](../../iaxl/csrc/qat_zip/qat_zip.c#L327)，IAA [iaa_zip.c](../../iaxl/csrc/iaa_zip/iaa_zip.c#L343)），压缩完成后再 `malloc` + `memcpy` 写入 DDR（[kv_zip.cpp](../../iaxl/csrc/kv_zip/kv_zip.cpp#L182-L192)）。CPU zlib 直接读 src，不经过 slot 拷贝（[cpu_zip.c](../../iaxl/csrc/cpu_zip/cpu_zip.c#L80-L84)）。
- load：先拿 deletion guard，再调用 `Mem.get` 拿到指针（[zip.cpp](../../iaxl/csrc/torch_ext/zip.cpp#L51-L56)）→ 解压或 memcpy 到 scratch（[zip.cpp](../../iaxl/csrc/torch_ext/zip.cpp#L91)；raw 拷贝见 [kv_zip.cpp](../../iaxl/csrc/kv_zip/kv_zip.cpp#L221-L225)，解压后写回见 [kv_zip.cpp](../../iaxl/csrc/kv_zip/kv_zip.cpp#L250-L258)）→ DMA 到 GPU（[zip.cpp](../../iaxl/csrc/torch_ext/zip.cpp#L99)）。

### 3.3 为什么需要中转

- 【技术硬约束】GPU 异步 DMA 要求源/目的是 pinned 内存；压缩后的大小要等压缩完成才知道，所以只能先写入临时缓冲，再按实际大小分配。
- 【设计选择，置信度 80%】不压缩时，scratch→DDR 这一次 memcpy 是可以省掉的，例如把 DDR 池做成一大块 pinned 区域、按块切分，直接 DMA 进去。代价是要 pin 数百 GB 内存，开 RDMA 时还要注册这部分内存。

### 3.4 压缩不作为解决手段

用户实测 deflate 对 KV cache 压缩效果较差（约 20%），**不把压缩当作解决内存问题的手段**。

---

## 4. Benchmark workload 的事实

### 4.1 所有请求共享同一段 80% 前缀

- 【代码证据】`vllm bench serve` 的 random 数据集只生成**一次**前缀，然后拼到每个请求前面（`vllm-0.29.0/vllm/benchmarks/datasets/datasets.py:636-637` "Generate prefix once"）。各请求的后缀不同（`datasets.py:755`，`offset + index + arange`）。
- 结果：`num_prompts` 个请求共享同一段 `random_prefix_len`，只有 20% 的后缀各不相同。

### 4.2 warmup 重复发送第 0 号请求

- 【代码证据】warmup 把 `input_requests[0]` 重复发送 `num_warmups` 次（`vllm-0.29.0/vllm/benchmarks/serve.py:832-833`、`:898`）。
- 因此 warmup 结束后，共享前缀（以及第 0 号请求自己的后缀）已经在 DDR 中。

### 4.3 测试关闭了 GPU prefix caching

- 【代码证据】[kvshrink-vllm-serve.sh](../../examples/kvshrink-vllm-serve.sh#L23) 使用 `--no-enable-prefix-caching`。

### 4.4 save 份数：共享前缀只存 1 份

- key 是链式块 hash，前缀 token 相同则 hash 相同（[hash.py](../../iaxl/utils/hash.py#L18-L45)）。
- save 时只保存 `not exists` 的块（[kvshrink_connector.py](../../kvshrink/kvshrink_connector.py#L312-L320)），`exists` 来自 `store.has()`（[kvshrink_connector.py](../../kvshrink/kvshrink_connector.py#L229)）。
- 即使多个请求同时写同一个 key，`Mem` 也是按 group 覆盖（[mem.cpp](../../iaxl/csrc/kv_pool/mem.cpp#L547-L551)），DDR 中仍然只有 1 份。
- 正式测量阶段，每个请求只保存自己的那 20% 后缀。

### 4.5 load 份数：每个请求各 load 一次，共 N 次（置信度 90%）

- 本地 GPU 前缀命中为 0，所以每个请求都会从 connector 拿到完整的前缀命中（[kvshrink_connector.py](../../kvshrink/kvshrink_connector.py#L245-L246)）。
- 每个请求的每一层，都要走一遍"`Mem.get` → DDR→scratch（memcpy 或解压）→ H2D 到该请求自己的 GPU 块"（[flow.py](../../iaxl/kvflow/flow.py#L320-L335)）。
- 并发 32 时就是 32 次，整轮共 `num_prompts` 次；每个 TP rank 各自处理自己的分片。
- 同步加载会把多个请求合并成一批（[kvshrink_connector.py](../../kvshrink/kvshrink_connector.py#L483)），但**不按 hash 去重**。
- 【技术硬约束】H2D 必须做 N 次：关闭 prefix caching 后，每个请求的 GPU 块都是私有的。
- 【设计选择】DDR→scratch 的 memcpy / 解压重复 N 次，可以在同一批内去重（只解一次，再 H2D 到多个目标块）。**已记录，但用户明确要求本次不做。**

### 4.6 前缀与后缀交界处的 token

- bench 客户端把"前缀 + 后缀"整体 decode 后再 re-encode，服务端还会再 tokenize 一次。交界处的 token 合并方式可能因请求而异，所以前缀的最后一个块可能每个请求都不一样。实际共享的块数为 `prefix/bs` 或少 1 个。
- 这个现象**现在就存在**，与本文讨论的任何方案无关。

---

## 5. 现有代码中与"淘汰"相关的约束和风险

以下都是设计任何淘汰方案时必须面对的问题：

1. **Record 在 put 发起时就写入，此时数据还没进 DDR。**【代码证据】[kvstore.py](../../iaxl/kvstore/kvstore.py#L189-L191)。如果在 Mem 层直接丢弃数据，Record 仍然显示"存在"。
2. **读不到数据时只重试一次，然后 abort。**【代码证据】[zip.cpp](../../iaxl/csrc/torch_ext/zip.cpp#L59-L82)，`IAXL_CHECK` 会调用 `abort()`（[iaxl_common.h](../../iaxl/csrc/include/iaxl_common.h#L13-L20)）。也就是说，"Record 显示存在但 Mem 里没有"会直接让服务崩溃。
3. **SCHEDULER 侧的 `has()` 只查 tp0 的 Record。**【代码证据】[kvshrink_connector.py](../../kvshrink/kvshrink_connector.py#L170-L175)（`rank=self.dp_rank * self.tp_size`）。如果每个 rank 各自淘汰，结果可能不一致：rank k 已经删掉的块，tp0 判断为命中，rank k 读不到就 abort。
4. **`Mem::evict_unlocked`（被 `evict_to_size` 调用）不删除 Record**（[mem.cpp](../../iaxl/csrc/kv_pool/mem.cpp#L754)）。只有 `evict_groups_unlocked` 会删（[mem.cpp](../../iaxl/csrc/kv_pool/mem.cpp#L739)）。
5. **deletion guard 是全局的，读取方先拿 guard 再去拿 `mutex_`。**【代码证据】[zip.cpp](../../iaxl/csrc/torch_ext/zip.cpp#L51-L56)。所以如果在持有 `mutex_` 时等待 guard，会死锁。
6. **潜在竞态（置信度 80%）**：`evict_groups` 先 `wait_deletion_guard()`、再加锁（[mem.cpp](../../iaxl/csrc/kv_pool/mem.cpp#L220-L223)），这两步不是原子的。如果有读取方恰好在两步之间拿到 guard 并取走指针，就可能出现释放后使用（use-after-free）。两轮之间手动 evict 时几乎不会触发，流量进行中 evict 时则是真实风险。
7. **`Mem.remove` 也会等待全局 guard**（[mem.cpp](../../iaxl/csrc/kv_pool/mem.cpp#L445-L446)）。如果在 forward 线程里调用，会被正在进行的 unzip 卡住。
8. **vLLM 0.29 支持把 load 失败降级为重算。**【代码证据】`kv_load_failure_policy: "recompute" | "fail"`，默认 `"fail"`（`vllm-0.29.0/vllm/config/kv_transfer.py:69`）；接口 `get_block_ids_with_load_errors`（`vllm-0.29.0/vllm/distributed/kv_transfer/kv_connector/v1/base.py:371`）；多个 worker 上报的失败块会取并集（`vllm-0.29.0/vllm/distributed/kv_transfer/kv_connector/utils.py:158`）。

---

## 6. 术语："SCHEDULER 侧"指什么

本文所说的"SCHEDULER 侧"**不是** vLLM 的 `scheduler.py`，我们不修改 vLLM。

vLLM 会把同一个 `KVShrinkConnector` 类实例化两次：一次在调度进程中（`role=SCHEDULER`），一次在每个 GPU worker 进程中（`role=WORKER`）。【代码证据】[kvshrink_connector.py](../../kvshrink/kvshrink_connector.py#L165)。

文件内分成两段：
- `Scheduler Side Methods`（[L212](../../kvshrink/kvshrink_connector.py#L212)）：`get_num_new_matched_tokens`、`update_state_after_alloc`、`_add_request_to_save`、`request_finished`、`build_connector_meta`。
- `Worker Side Methods`（[L368](../../kvshrink/kvshrink_connector.py#L368)）。

另外，调度进程中的 has-only KVStore 就是 controller 角色（[kvstore.py](../../iaxl/kvstore/kvstore.py#L123)）。

---

## 7. 方案

### 7.1 a1：只保存前 N 个 prompt token（静态截断）

**做法**：在 `_add_request_to_save`（[kvshrink_connector.py](../../kvshrink/kvshrink_connector.py#L303-L326)）中，只保留块下标小于 `ceil(N / block_size)` 的块。

**环境变量命名**：
- 项目约定：connector 开关使用 `KVSHRINK_VLLM_KV_<功能>_<参数>` 格式（[setvars.sh](../../setvars.sh#L28-L31)）。
- 默认值**只写在 setvars.sh**；代码通过 `required()` 读取，不写默认值（[async_load_config.py](../../kvshrink/async_load_config.py#L189-L205)）。
- `-1` 表示"全部"（同 `KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS=-1`）。
- 新增变量需同步加入 setvars.sh 的汇总打印（[setvars.sh](../../setvars.sh#L77-L80)）和 [kvshrink.md](kvshrink.md)。
- 建议名称：**`KVSHRINK_VLLM_KV_SAVE_PREFIX_TOKENS`**。默认 `-1` 表示全部保存（即现状）；设为 N 表示只保存覆盖前 N 个 token 的块。这是 vLLM 服务端的变量，不是 bench 客户端的参数。

**截断位置：只能向上取整（原则：宁可多存，也要保证 benchmark 结果正确）**
- 截断只决定"存哪些块"，是否命中由链式 hash 决定。
- 多存几个后缀块，只会让每个请求多占一两个块的内存，对命中率没有影响。
- 少存则可能漏掉真正共享的块，导致命中率低于 80%。
- 取 `N = random_prefix_len + block_size`。这样命中率与不截断时完全一致（交界处问题见 4.6）。

**缺点**：
- 只适用于这类 benchmark。
- 从第一个请求起就不再保存后缀，save 路径的开销（D2H、memcpy、malloc）比真实情况少，测得的 TTFT/TPOT 可能偏乐观。【推测，置信度 70%】原因是 zip 和 unzip 共用同一个 `omp_queue`。

置信度：90%。

### 7.2 a2：同一个块被 ≥2 个请求见过才保存（已否决）

**做法**：SCHEDULER 侧记录每个块 hash 出现在多少个不同请求里；只有达到 2 次的块才保存。需要按 req_id 去重计数，因为请求排队等待时 `get_num_new_matched_tokens` 会被反复调用（`vllm-0.29.0/vllm/v1/core/sched/scheduler.py:857`；connector 中已有 "Discarded stale state" 的处理，[kvshrink_connector.py](../../kvshrink/kvshrink_connector.py#L220-L221)）。

**改动位置**：只改 [kvshrink_connector.py](../../kvshrink/kvshrink_connector.py)，全部在 SCHEDULER 侧方法里，不改 vLLM。上面引用的 `scheduler.py:857` 只是用来说明为什么会被反复调用。
- `__init__`：新增计数表（hash → 不同请求数，设上限）和"已计数 req_id"集合。
- `get_num_new_matched_tokens`（[L215](../../kvshrink/kvshrink_connector.py#L215)）：算完 `block_hashes` 后，如果该 req_id 还没计过数，就给它的每个 hash 计数 +1。
- `_add_request_to_save`（[L303](../../kvshrink/kvshrink_connector.py#L303)）：在 `missing` 的过滤条件（[L319](../../kvshrink/kvshrink_connector.py#L319)）里加上"计数 ≥ 2"。
- `request_finished`（[L328](../../kvshrink/kvshrink_connector.py#L328)）：把该 req_id 从"已计数"集合中移除。

**结论**：用户认为实现不优雅、需要维护的集合较大，**已否决**。

### 7.3 d：硬上限，满了就停止 offload

**做法**：在 SCHEDULER 侧的 `_add_request_to_save` 中累计"已保存块数 × 每块字节数"（每块字节数取自 `kv_cache_config` 中各层 `page_size_bytes` 之和）。达到预算后，不再加入新的块。

**为什么必须放在 SCHEDULER 侧而不是 Mem 层**：
- 第 5 节第 1、2 条：Mem 层丢弃数据会导致"Record 显示存在、实际读不到"，进而 abort。
- 第 5 节第 3 条：在 SCHEDULER 侧决定，各 TP rank 收到同一份 metadata，天然一致。

**复杂度**：远低于 LRU。不需要 pin，不需要给 worker 下发删除指令，不改 C++。唯一需要补的是：手动 evict 之后要重置计数。【推测，置信度 70%】可以在同进程的 controller 处挂钩实现。

**缺点**：
- 脚本中 `seed=$(date +%s)`，每轮前缀都不同。如果两轮之间不 evict，上一轮的冷后缀会占满上限，新一轮的前缀存不进去，命中率归 0。
- 达到上限后不再有 save 开销，保真度不如 LRU（但好于 a1，a1 从一开始就没有）。

置信度：单轮 85%。

### 7.4 拆分子轮再合并（不改代码）——助手单方面提出，未经讨论、未确认

> 来源：助手在说明"重放方案不成立"时，顺带提出的替代方案。用户没有讨论过，也没有认可。保留在这里仅作记录，是否删除由用户决定。

**做法**：
- 把一轮测试拆成 K 个子轮，子轮之间调用一次 evict。
- 每个子轮加 `--save-result --save-detailed`，事后把各子轮逐请求的 `ttfts`/`itls` 合并，再计算 P50/P95。
- 每个子轮的请求数应 ≥ 4 × 并发数，以减小开头和结尾的边界效应。

置信度：75%（边界效应与一轮长跑不完全相同）。

### 7.5 重放有限 prompt 集（单独使用不成立）

关闭 prefix caching 后，同一个 prompt 第二次发送时，kvshrink 会命中前缀和后缀。原因是命中长度由第一个不存在的块决定（[kvshrink_connector.py](../../kvshrink/kvshrink_connector.py#L236-L245)），而后缀在第一次出现时已被保存。结果命中率接近 100%，而不是 80%，**会破坏 benchmark 的语义**。

### 7.6 保真度对比

| 方案 | 是否保留后缀的 save 开销 |
|---|---|
| LRU（A 或 B） | 保留（另外增加淘汰本身的开销） |
| d 硬上限 | 达到上限前保留，之后消失 |
| a1 静态截断 | 从一开始就消失 |

保真度排序：LRU > d > a1。

### 7.7 LRU-A：在 iaxl Mem 层自动淘汰（worker / daemon 进程内）

**做法**：
1. 在 `Mem::put_unlocked` 插入数据之后，如果 `current_bytes_ > capacity_bytes_`，就沿 Mem 已有的 LRU 链表从最旧开始淘汰，直到降到低水位（例如容量的 90%，用一个新的 `IAXL_` 变量控制）。LRU 链表在 put 和 get 时都会更新（[mem.cpp](../../iaxl/csrc/kv_pool/mem.cpp#L569-L571)、[mem.cpp](../../iaxl/csrc/kv_pool/mem.cpp#L637-L640)）。
2. 淘汰时同时删除 Record，写法参照 `evict_groups_unlocked`（[mem.cpp](../../iaxl/csrc/kv_pool/mem.cpp#L727-L740)），让 `has()` 返回 false。已持久化的 group 可以保留 Record，因为它们能从磁盘回读。
3. 给每个 group 加引用计数：get 时 +1，解压完成后 -1；淘汰时跳过被引用的 group。这样可以替代全局 deletion guard（第 5 节第 5、6 条）。
4. 兜底：把读不到时的 abort（[zip.cpp](../../iaxl/csrc/torch_ext/zip.cpp#L82)）改为上报失败块。connector 的 worker 侧实现 `get_block_ids_with_load_errors`，vLLM 设置 `kv_load_failure_policy=recompute` 去重算。这一步是**必需的**，原因有二：`has()` 和 worker 的 `get` 之间存在时间窗；各 rank 各自淘汰时结果可能不一致（第 5 节第 3 条）。

**优势**：
- **结构性优势（LRU-B 做不到）**
  - 支持远端共享 DDR 池。【文档】一个池节点可以服务多个推理节点（[remote-pool.md](remote-pool.md#L9)）。LRU-B 由每个推理节点的 connector 各自决策，无法共同管住同一个共享池的预算。置信度 90%。
  - 内存统计是真实值：`current_bytes_` 就是实际 malloc 的大小，即压缩后大小（[mem.cpp](../../iaxl/csrc/kv_pool/mem.cpp#L567)）。从磁盘回读进 DDR 的数据（[mem.cpp](../../iaxl/csrc/kv_pool/mem.cpp#L305-L318)）也会被计入。
- **工程性优势**
  - `IAXL_DDR_POOL_SIZE_GB` 会变成真正的上限，不需要引入第二套预算配置。
  - 淘汰顺序反映真实访问（所有 put/get 都会更新 LRU 链表）。
  - 不依赖 vLLM connector 的 metadata 和生命周期钩子。
  - 实现它的前提条件会顺带修掉两处已有问题：读不到时 abort（第 5 节第 2 条），以及 `evict_groups` 的潜在竞态（第 5 节第 6 条）。

**劣势**：
- 各 rank 的淘汰结果不一致，只能靠 recompute 兜底。一旦触发重算，会污染 TTFT。【推测，置信度 85%】在本 workload 中概率很低，因为共享前缀一直排在 LRU 最不容易被淘汰的那一端。
- 需要修改 C++ 热路径：持锁期间做 free，且引用计数的正确性要保证。

置信度：60%。

### 7.8 LRU-B：由 connector 的 SCHEDULER 侧驱动淘汰

**做法**：
1. SCHEDULER 侧维护一个按块 hash 排序的 LRU：save 时加入，load 时移到最新位置。
2. 正在 load/save 的请求所涉及的块要 pin 住，直到请求完成。请求完成可通过 `update_connector_output` 的 `finished_sending` / `finished_recving` 得知（`vllm-0.29.0/vllm/distributed/kv_transfer/kv_connector/v1/base.py:533`）。
3. 按"每块字节数 × 块数"估算用量，超出预算时，从 LRU 最旧端选出未被 pin 的块，写入 metadata 的新字段 `blocks_to_evict` 下发。
4. 每个 worker 在**后台线程**中调用已有的 Python 绑定 `Mem.remove` 和 `Record.remove`（[torch_ext.cpp](../../iaxl/csrc/torch_ext/torch_ext.cpp#L229)、[torch_ext.cpp](../../iaxl/csrc/torch_ext/torch_ext.cpp#L428)）。必须放在后台线程，原因见第 5 节第 7 条。
5. 可选兜底：与 LRU-A 第 4 步相同（`kv_load_failure_policy=recompute`）。

**优势**：
- 各 TP rank 天然一致，因为在同一个 step 收到同一份淘汰列表。
- 没有 `has()` 到 `get` 之间的时间窗问题（有 pin 保护）。
- 基本只改 Python。
- 不会因为重算而污染 benchmark 结果。

**劣势**：
- 用量是按未压缩大小估算的，开启压缩时会偏保守，导致提前淘汰。
- 看不到从磁盘回读进 DDR 的数据。
- 只覆盖 kvshrink。remote_pool 还需要新增 remove RPC，`KVStoreRemote` 目前只有 put/get/has/persist/evict(count)（[kvstore_remote.py](../../iaxl/remote_pool/kvstore_remote.py#L114-L178)），并且无法处理多个客户端共享同一个池的情况。
- pin 的生命周期管理比较复杂。

置信度：75%。

### 7.9 方案汇总

| 方案 | 改动位置 | 置信度 | 状态 |
|---|---|---|---|
| a1 静态截断 | connector（Python）+ setvars.sh + 文档 | 90% | 备选 |
| a2 ≥2 次才保存 | connector | 85% | **否决** |
| d 硬上限 | connector | 85%（单轮） | 备选，复杂度最低 |
| 子轮拆分合并 | 只改脚本 | 75% | 助手提出，未讨论 |
| 重放 | — | — | **不成立** |
| 压缩 | — | — | **不作为手段** |
| LRU-A | iaxl C++ + connector 少量 | 60% | 待选 |
| LRU-B | connector（Python） | 75% | 待选 |

建议：
- 如果只服务本机 benchmark，选 LRU-B。
- 如果要做成 iaxl 的通用能力，或者 benchmark 需要覆盖 remote_pool，选 LRU-A。

---

## 8. 已确定的事项

- 工作分支：`feat/bounded-ddr-cache`。
- 环境变量命名遵循第 7.1 节所述的项目约定，不另起格式。
- 截断类参数一律向上取整，宁可多存。
- "同一批内 DDR→scratch 去重"只做记录，本次不实现。
- 不使用压缩来缓解内存问题。

## 9. 待定事项

- 选择 LRU-A 还是 LRU-B。
- 是否同时提供 a1 或 d 作为轻量选项。

## 10. 讨论过程中的更正记录

| 原说法 | 更正 |
|---|---|
| "LRU 代码 90% 已就绪，只差在 put 中调用 `evict_to_size`" | 过于乐观。`evict_unlocked` 不删除 Record，此外还有 guard 死锁/竞态和各 rank 不一致的问题（第 5 节）。 |
| "实际 load 次数远小于 32，因为 GPU prefix cache 会接管" | 错误。测试使用了 `--no-enable-prefix-caching`，每个请求都会 load 一次（4.5）。 |
| "推荐重放有限 prompt 集" | 单独使用不成立，命中率会变成约 100%（7.5）。 |
| "a1 的截断点 K 应比 prefix_len 少一个 block" | 方向错误，应当向上取整、宁可多存（7.1）。 |
| 曾随口提出 `KVSHRINK_SAVE_MAX_TOKENS` | 不符合项目命名约定，应为 `KVSHRINK_VLLM_KV_SAVE_PREFIX_TOKENS`（7.1）。 |
| "a2 达到上限后 save 开销才消失" | a1 和 a2 从一开始就没有后缀的 save 开销，只有 d 是达到上限后才消失（7.6）。 |
| 引用 controller 角色时写成 `kvstore.py` 的 L117 | 正确位置是 L123。 |
