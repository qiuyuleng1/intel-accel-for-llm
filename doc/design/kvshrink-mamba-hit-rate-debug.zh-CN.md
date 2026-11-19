# KVShrink 在混合模型（Qwen3.8-27B）上命中率偏低的 debug 记录

状态：debug 中。本文档随 debug 进度更新，最新进度见文末「进度记录」。

## 1. 现象

b300 正式测试中，同一个 hitrate 下 kvshrink 每个请求的实际命中，经常低于 vllm-prefix
（vLLM 自己的 GPU prefix caching）。最明显的是输入 32000、并发 32 和 64：几乎所有请求只
命中 15680 token（49.0%），hitrate 设成 80 还是 95 都一样；vllm-prefix 同样的运行命中
78–93%。只有 kvshrink 的命中和 vllm-prefix 对齐之后，两者的性能对比才有意义，所以先查命中。

## 2. 测试设置

- 结果目录：`bench-results/Qwen3.8-27B_tp4_20261111_095943`（b300 原始目录，本地有 rsync 副本）。
- 代码：main `bff83ae` 加 b300 本地设置（`setvars.sh`、`examples/kvshrink-vllm-serve.sh`、
  `tools/setup_qat.sh`、IAA 按 rank 绑定），未提交。
- 程序：`test-qwen38-27B.sh`。每个配置启动一个 vLLM 0.29 服务（Qwen3.8-27B，TP=4，
  GPU 0/2/4/6，每个 rank 绑定本 NUMA 节点的 CPU、QAT、DSA、IAA），用 `vllm bench serve`
  发随机文本请求，输出 128 token。
- 配置：
  - kvshrink：开 KVShrink DDR 缓存，关 vLLM GPU prefix caching；`kvshrink/vllm_patch.py`
    把 `mamba_cache_mode` 从 `none` 强制改成 `align`。
  - vllm-prefix：只开 vLLM GPU prefix caching（vLLM 自动设 `mamba_cache_mode='align'`）。
  - vllm-wo-prefix：两种缓存都不开。
  - lmcache：所有存储都报 `RuntimeError: Number of heads (784) exceeds max threads per block
    in y-dim (32)`，命中全为 0，跑完 15 次运行后停掉，本文不讨论。
- 一次运行：输入长度、hitrate、并发各取一个值，跑一次 `vllm bench serve`。kvshrink 和
  vllm-prefix 每次发 并发 × 10 个请求，vllm-wo-prefix 发 并发 × 2 个；每次运行前另有 5 个
  热身请求（不计入结果）；每个输入长度开始前先发一批完全相同的请求预热缓存
  （vllm-wo-prefix 不预热）。
- hitrate：每个请求开头有多少比例的 token 和其他请求相同（设计值，不是实测）。
- vLLM 自动选择的参数（三个配置的 serve.log 相同）：
  - block size 784（`Setting attention block size to 784 tokens`）；
  - chunk 大小 `max_num_batched_tokens=16384`（`Chunked prefill is enabled with
    max_num_batched_tokens=16384`），align 模式下每段 chunk 结尾向下对齐到 block 边界，
    即一段 chunk 最多 20 个 block = 15680 token。

## 3. TTFT（ms，均值 / 中位）

样本数 = 这次运行的请求数。来源：`summary.csv` 的 `mean_ttft_ms`、`median_ttft_ms`。
vllm-wo-prefix 没有缓存，hitrate 对它没有意义，只跑了 80。

| 输入长度 | hitrate | block size | 并发 | kvshrink | vllm-prefix | vllm-wo-prefix |
|---|---|---|---|---|---|---|
| 32000 | 80 | 784 | 32 | 2855 / 3052 | 2247 / 2380 | 5684 / 3223 |
| 32000 | 80 | 784 | 64 | 3587 / 3142 | 3346 / 3655 | 9749 / 3270 |
| 32000 | 80 | 784 | 128 | 3798 / 1971 | 3485 / 2953 | 32422 / 32380 |
| 32000 | 80 | 784 | 256 | 5681 / 2562 | 3957 / 2417 | 80787 / 96624 |
| 32000 | 80 | 784 | 300 | 8420 / 5331 | 3408 / 1658 | 97353 / 118670 |
| 32000 | 95 | 784 | 32 | 2845 / 3067 | 1193 / 1291 | — |
| 32000 | 95 | 784 | 64 | 3738 / 3351 | 1330 / 1161 | — |
| 32000 | 95 | 784 | 128 | 2634 / 773 | 1221 / 583 | — |
| 32000 | 95 | 784 | 256 | 15022 / 14008 | 10829 / 10734 | — |
| 32000 | 95 | 784 | 300 | 18987 / 18930 | 14495 / 14611 | — |
| 128000 | 80 | 784 | 64 | 10970 / 10123 | 11273 / 11418 | 100742 / 120097 |
| 128000 | 80 | 784 | 80 | 8211 / 5411 | 5007 / 2603 | 130592 / 159971 |
| 128000 | 95 | 784 | 64 | 18726 / 19304 | 21458 / 22468 | — |
| 128000 | 95 | 784 | 80 | 24689 / 25547 | 27158 / 28389 | — |
| 256000 | 80 | 784 | 32 | 23811 / 27988 | 26607 / 29877 | 128489 / 152088 |
| 256000 | 80 | 784 | 40 | 15381 / 8109 | 28661 / 32628 | 166059 / 202258 |
| 256000 | 95 | 784 | 32 | 16586 / 16806 | 20139 / 21221 | — |
| 256000 | 95 | 784 | 40 | 22670 / 23219 | 25757 / 27153 | — |

## 4. 实际命中

### 4.1 怎么测

- kvshrink：每个请求一个值，取 connector 日志 `get_num_new_matched_tokens` 的
  `externally-cached tokens`（从 DDR 载入、不用重算的 token 数）。只统计正式测量的请求
  （`req-cmpl-bench-<8 位十六进制>-<序号>`），5 个热身请求的 id 不同，不计入。同一请求打了
  多行日志时各行的值都相同（0 个请求不同）。脚本 `tests/bench_dev/hit_per_run.py`，输出
  `bench-results/Qwen3.8-27B_tp4_20261111_095943/analysis/hit_per_run.txt`。
- vllm-prefix：vLLM 不按请求打印 GPU prefix cache 命中，只能用计数器
  `vllm:prefix_cache_hits_total` / `vllm:prefix_cache_queries_total` 这次运行前后的差值之比。
  这是整次运行的总比值，含 5 个热身请求，不能拆到每个请求。

### 4.2 设计上限怎么算

设计上限 = ⌊输入 × hitrate ÷ 784⌋ × 784，即共享前缀向下取整到整 block。例：
32000 × 80% = 25600，25600 ÷ 784 = 32.65 → 32 个 block = 25088 token = 78.4%。
它假设每个 block 边界都能有 mamba 状态，没有考虑「只有 chunk 结尾才保存状态」（见第 5 节）。

### 4.3 结果

kvshrink 列：每个不同的命中值 × 有多少个请求是这个值（不加总）。命中 98.0% / 99.8% 的请求
和缓存里某个请求完全相同（例如第 1 个请求和热身请求相同）。

| 输入长度 | hitrate | 并发 | 请求数 | 设计上限 | kvshrink：命中 token（占输入 %）× 请求数 | vllm-prefix：命中 / 查询 |
|---|---|---|---|---|---|---|
| 32000 | 80 | 32 | 320 | 25088 (78.4%) | 15680 (49.0%) ×319；31360 (98.0%) ×1 | 78.2% |
| 32000 | 80 | 64 | 640 | 25088 (78.4%) | 15680 (49.0%) ×638；31360 (98.0%) ×2 | 78.3% |
| 32000 | 80 | 128 | 1280 | 25088 (78.4%) | 25088 (78.4%) ×1155；15680 (49.0%) ×122；31360 (98.0%) ×3 | 78.4% |
| 32000 | 80 | 256 | 2560 | 25088 (78.4%) | 25088 (78.4%) ×2469；15680 (49.0%) ×81；31360 (98.0%) ×10 | 78.4% |
| 32000 | 80 | 300 | 3000 | 25088 (78.4%) | 25088 (78.4%) ×2903；15680 (49.0%) ×81；31360 (98.0%) ×16 | 78.4% |
| 32000 | 95 | 32 | 320 | 29792 (93.1%) | 15680 (49.0%) ×319；31360 (98.0%) ×1 | 92.6% |
| 32000 | 95 | 64 | 640 | 29792 (93.1%) | 15680 (49.0%) ×639；31360 (98.0%) ×1 | 92.8% |
| 32000 | 95 | 128 | 1280 | 29792 (93.1%) | 29792 (93.1%) ×1151；15680 (49.0%) ×127；31360 (98.0%) ×2 | 93.0% |
| 32000 | 95 | 256 | 2560 | 29792 (93.1%) | 29792 (93.1%) ×2406；15680 (49.0%) ×138；31360 (98.0%) ×16 | 93.1% |
| 32000 | 95 | 300 | 3000 | 29792 (93.1%) | 29792 (93.1%) ×2887；15680 (49.0%) ×86；31360 (98.0%) ×27 | 93.1% |
| 128000 | 80 | 64 | 640 | 101920 (79.6%) | 100352 (78.4%) ×627；94080 (73.5%) ×11；127792 (99.8%) ×2 | 79.6% |
| 128000 | 80 | 80 | 800 | 101920 (79.6%) | 100352 (78.4%) ×787；94080 (73.5%) ×11；127792 (99.8%) ×2 | 79.6% |
| 128000 | 95 | 64 | 640 | 121520 (94.9%) | 120736 (94.3%) ×639；127792 (99.8%) ×1 | 94.7% |
| 128000 | 95 | 80 | 800 | 121520 (94.9%) | 120736 (94.3%) ×796；127792 (99.8%) ×4 | 94.8% |
| 256000 | 80 | 32 | 320 | 204624 (79.9%) | 203840 (79.6%) ×319；255584 (99.8%) ×1 | 79.7% |
| 256000 | 80 | 40 | 400 | 204624 (79.9%) | 203840 (79.6%) ×397；255584 (99.8%) ×3 | 79.9% |
| 256000 | 95 | 32 | 320 | 243040 (94.9%) | 241472 (94.3%) ×319；255584 (99.8%) ×1 | 94.4% |
| 256000 | 95 | 40 | 400 | 243040 (94.9%) | 241472 (94.3%) ×399；255584 (99.8%) ×1 | 94.5% |

要点（实测）：

- vllm-prefix 每次运行都基本等于设计上限。
- kvshrink 输入 32000、并发 32 / 64：几乎全部只命中 15680 = 20 个 block，正好是一段 chunk
  （16384 向下对齐到 784）。hitrate 80 和 95 的命中相同，TTFT 也几乎相同（2855 对 2845 ms）。
- kvshrink 其他情况多数到达或接近上限，但也有落在 20 个 block 倍数上的
  （128000 / 80 有 11 个请求命中 94080 = 120 个 block；256000 / 80 命中 203840 = 260 个 block）。
- 32000 并发 256 / 300 下 hitrate 95 的 TTFT 比 80 高，不是因为实际命中更少（95 的命中更高）。

## 5. 分析（读代码，尚未用实验验证）

### 5.1 mamba 状态只在 chunk 结尾保存

vLLM 0.29 align 模式下，线性注意力层（mamba）的状态只在每段 chunk 结束时写出，chunk 结尾
对齐到 block 边界（`vllm/v1/core/sched/scheduler.py` `_mamba_block_aligned_split`）。命中要求
注意力 KV 和 mamba 状态在同一位置都存在，所以只能命中到「之前某个请求某段 chunk 恰好结束」
的位置。一个 32000 token 的请求单独算时，chunk 在 15680、31360 结束，只共享前 25600 token
的请求最多命中 15680。

### 5.2 vllm-prefix 为什么能到上限：共享前缀分叉点

- `vllm/v1/core/kv_cache_manager.py` `get_computed_blocks`（vLLM 自己的 GPU KV block 管理，
  运行在调度进程，负责分配 block 和查 prefix cache）：全注意力层能匹配的 block 多于 mamba
  层有状态的 block 时，返回 `shared_prefix_boundary` = 全注意力层的匹配长度（注释：
  Marconi-style APC），即新请求和缓存里的请求开始不同的位置。
- `_mamba_block_aligned_split` 把 chunk 截停在这个分叉点（block 向下取整），于是分叉点的
  mamba 状态被保存，之后的请求可以命中到这里。

例（输入 32000 = 40 个满 block + 640；共享前缀 25600 = 32 个满 block + 512）：

| 请求 | 发生什么 | 命中 |
|---|---|---|
| A（第一个用到这个前缀的） | 全部要算；chunk 在第 20、40 个 block 结束；注意力 KV 存 40 个 block，mamba 状态只在 20、40 | 0 |
| B（和 A 共享 32 个 block） | 注意力能匹配 32，mamba 状态只有 20；命中取两者都有的 20。分叉点 = 32，B 的 chunk 停在 32，存下 32 处的 mamba 状态 | 20 个 block（49%） |
| C 及之后 | 32 处注意力 KV 和 mamba 状态都有 | 32 个 block（78.4%） |

B 存下 32 处状态之前到达的请求也只能命中 20（推断：这是 vllm-prefix 并发 32 时 78.2% 略低于
78.4% 的原因）。

### 5.3 kvshrink 为什么没有这个效果

kvshrink 配置关闭了 vLLM prefix caching，`get_computed_blocks` 直接返回 0，分叉点永远是 0，
chunk 不会在分叉点停，mamba 状态只在 chunk 结尾保存。所以 kvshrink 保存状态的位置和
vllm-prefix 不一致。高并发下多数请求能到 32 个 block（推断）：很多请求分摊 16384 的预算，
chunk 结尾落在各种位置，只要有一个落在第 32 个 block，之后的请求都能命中那里。这也让命中
依赖并发和时序，结果不稳定。

### 5.4 修复方向

不可行：打开 vLLM prefix caching 但让显存缓存查找一直返回 0。分叉点就是查显存缓存得到的，
查找返回 0 则分叉点也是 0；若只丢掉命中结果、保留真实查找，分叉点又取决于显存里还剩什么，
高并发下更不稳定。

倾向：GPU prefix caching 继续关闭，由 kvshrink connector 用自己 DDR 里的索引算两个数：
(a) 注意力 KV 在 DDR 中最长能匹配到哪个 block；(b) 注意力 KV 和 mamba 状态都存在的最长位置。
返回 (b) 作为命中，并把 `request.shared_prefix_boundary` 设成 (a)，让 vLLM 现成的
`_mamba_block_aligned_split` 把 chunk 停在 (a)。改的是 `kvshrink/kvshrink_connector.py`。
调度器先查本地缓存并给 `shared_prefix_boundary` 赋值，再调用 connector 的
`get_num_new_matched_tokens(request, ...)`，connector 有机会设置这个值。

待确认：

1. kvshrink 是否存了每个 block 的注意力 KV。
2. prefix caching 关闭时，`shared_prefix_boundary` 在调度器后续步骤中是否被使用或覆盖
   （如 `single_type_kv_cache_manager.py` 中 `reachable_boundaries`）。
3. kvshrink 是否在任意 chunk 结尾都保存 mamba 状态。

### 5.5 debug 步骤 1 结论（读代码确认，11-12）

代码版本：b300 容器 `iaxl.vllm-fddeef0-qiuyu`、6000d 容器 `iaxl.vllm-fddeef0-benchmark` 里的
vLLM 都是 0.29.0，`scheduler.py`、`kv_cache_manager.py`、`single_type_kv_cache_manager.py`、
`kv_cache_coordinator.py`、`request.py` 的 md5 和本地 `vllm-0.29.0` 源码相同。`kvshrink/`
目录在 6000d 的 1030 仓库（`fddeef0`）和 B300 测试目录（`bff83ae`）之间没有差别。

kvshrink 怎么判断命中（`kvshrink/kvshrink_connector.py` `get_num_new_matched_tokens`，L293 起）：

1. 从第 1 个 block 起，数注意力 KV 在 DDR 里连续存在多少个 block（L314），即 5.4 的 (a)；
2. 有 mamba 层时，从 (a) 往回退，直到某个 block 的 mamba 状态存在（L322-L331），即 (b)；
3. 返回 (b) × block size 作为命中。

kvshrink 存什么（`_add_request_to_save`，L393 起）：注意力 KV 每个满 block 都存（DDR 里已有的
跳过）；mamba 状态只在 chunk 结尾正好落在 block 边界时存，存的是结尾那个 block。

vLLM 调度器这边（`vllm/v1/core/sched/scheduler.py`）：

- `KVShrinkConnector` 没有覆盖 `supports_divergent_local_hybrid_hits`（默认 False），所以
  `_get_local_prefix_cache_hit`（L468-L478）走 `get_computed_blocks`；prefix caching 关闭时它
  直接返回 0（`kv_cache_manager.py` L247-L248），`request.shared_prefix_boundary = 0`（L839-L844）。
- 接着调用 `connector.get_num_new_matched_tokens(request, ...)`（L850-L857），connector 拿到的
  就是这个 `request`。
- 之后只有 `hit_diverged` 为真时才会重新赋值（L880-L885），这里永远为假。
- `shared_prefix_boundary` 只在两处被读：`_mamba_block_aligned_split` 决定 chunk 停在哪
  （L456-L461）；`cache_blocks` 决定 GPU 缓存保留哪些 block，但它只在 prefix caching 打开时
  调用（`kv_cache_manager.py` L550、L757）。
- 异步载入时，请求在载入完成后直接按已算好的 token 数继续调度（L934-L938），不再重新查找，
  `shared_prefix_boundary` 仍是之前设的值。

所以 5.4 的三个前提都成立：

1. 注意力 KV 每个 block 都存了；
2. prefix caching 关闭时，在 `get_num_new_matched_tokens` 里设 `request.shared_prefix_boundary`
   不会被覆盖，只影响 chunk 在哪里停；
3. 任何落在 block 边界的 chunk 结尾都会存 mamba 状态。

而且 kvshrink 已经算出了 (a)（第 1 步），只是退回 (b) 之后就丢掉了。修改方案（未改）：退回
之后若 (a) > (b)，设 `request.shared_prefix_boundary = (a) × block size`。

### 5.6 debug 步骤 2 结果：6000d 最小复现（实测，11-12）

设置：6000d，vLLM 0.29，Qwen3.8-27B，TP=4，GPU0-3，`--max-num-batched-tokens 16384`、
`--enable-prompt-tokens-details`，MAX_MODEL_LEN 256128；两种配置各启动一次 vLLM，各按顺序
（一次一个）发 5 个 32000 token 的请求（prompt 为 token id）：请求 1 是新的，请求 2-5 和
请求 1 共享前 25600 token、后 6400 token 各不相同；输出 16 token，temperature 0。
命中 = 响应 `usage.prompt_tokens_details.cached_tokens`；kvshrink 的值和 connector 日志
`externally-cached tokens` 逐个相同。两种配置启动日志都是 block size 784、
`max_num_batched_tokens=16384`。结果：`intel-accel-for-llm-1030/tests/kvshrink_mamba_hit_dev/
results/repro-20261112_025832_238a3ea9/`（`run.log`、各配置 `serve.log`、`requests.jsonl`）。

| 请求 | 和之前请求共享的前缀 | 设计上限 | kvshrink 命中 | vllm-prefix 命中 |
|---|---|---|---|---|
| 1 | 0 | 0 | 0 | 0 |
| 2 | 25600 | 25088 | 15680 (49.0%) | **0** |
| 3 | 25600 | 25088 | 15680 (49.0%) | 25088 (78.4%) |
| 4 | 25600 | 25088 | 15680 (49.0%) | 25088 (78.4%) |
| 5 | 25600 | 25088 | 15680 (49.0%) | 25088 (78.4%) |

和预期的差别：vllm-prefix 请求 2 命中 0，不是预期的 15680。解释（读代码，未单独验证）：
vLLM GPU 缓存不保留每段 chunk 结尾的 mamba 状态，只保留 prompt 末尾和分叉点处的
（`single_type_kv_cache_manager.py` `cache_blocks` 的 `reachable_boundaries`）。请求 1 的状态
只留了第 40 个 block 的，请求 2 能用的 mamba 状态为 0，所以命中 0；但它的注意力 KV 匹配到
32 个 block，分叉点 = 32，请求 2 的 chunk 停在 25088，存下第 32 个 block 的状态，请求 3-5
命中 25088。kvshrink 保留了第 20 个 block 的状态，所以请求 2-5 都是 15680，但没有分叉点，
第 32 个 block 的状态永远不会被存。5.2 表中请求 B 的命中应为 0（vLLM），这里更正。

推论：按 5.5 的方案修改后，kvshrink 请求 2 仍命中 15680（比 vllm-prefix 的 0 多），请求 3-5
应命中 25088，与 vllm-prefix 相同。

### 5.7 输入长度

问题根源不是输入长度。没有分叉点时，最坏少命中 19 个 block（14896 token）：32k 约占输入
47%，128k 约 12%，256k 约 6%，所以 32k 最明显，长输入看起来接近正常但同样靠运气。保留 32k
用来验证修复；不靠改输入长度或改 chunk 大小绕开（调大 chunk 时只共享前缀的请求可能命中 0，
调小则改变所有配置的 prefill 效率）。

### 5.8 debug 步骤 4 结果：修改后在 6000d 验证（实测，11-12）

和 5.6 相同的设置和请求，只跑 kvshrink，kvshrink 代码为本目录改后的 `kvshrink/`（md5 标记
`b49277a4`；改前那次是 `238a3ea9`）。结果：`intel-accel-for-llm-1030/tests/kvshrink_mamba_hit_dev/
results/repro-20261112_031738_b49277a4/`。分叉点取自 connector 日志新增的
`shared_prefix_boundary=`（只有改后的代码会打印，证明 vLLM 加载的是改后的代码）。

| 请求 | 设计上限 | kvshrink 改前 | kvshrink 改后 | 改后的分叉点 | vllm-prefix（5.6） |
|---|---|---|---|---|---|
| 1 | 0 | 0 | 0 | 0 | 0 |
| 2 | 25088 | 15680 | 15680 | 25088 | 0 |
| 3 | 25088 | 15680 | **25088** | 0 | 25088 |
| 4 | 25088 | 15680 | **25088** | 0 | 25088 |
| 5 | 25088 | 15680 | **25088** | 0 | 25088 |

和预期完全一致：请求 2 命中 15680，分叉点 25088（注意力 KV 匹配 32 个 block，mamba 状态只到
第 20 个）；请求 3-5 命中 25088，等于设计上限和 vllm-prefix；请求 3-5 的注意力 KV 和 mamba 状态都到
第 32 个 block，不再需要分叉点（0）。serve.log 中 `Traceback`、` ERROR `、`IAXL check failed`、
`cache key missing after retry`、`cache miss on unzip` 均为 0。

未覆盖：只验证了命中数，没有验证输出是否正确（请求是随机 token，输出 16 token 未保存）；
只有并发 1、hitrate 80、输入 32000。

### 5.9 debug 步骤 5 结果：b300 确认（实测，11-12）

b300，改后的 kvshrink（`kvshrink_connector.py` md5 `2b396593`），只跑 kvshrink，参数取自 11-11
正式测试的一部分、随机种子相同（同样的请求）：输入 32000 并发 32 / 64、128000 并发 64、256000
并发 32，hitrate 80 / 95。结果 `bench-results/Qwen3.8-27B_tp4_20261112_032448`（本地有副本），
`analysis/hit_per_run.txt`。serve.log：`Traceback`、` ERROR `、`IAXL check failed`、
`cache key missing after retry`、`dsa batch failed` 均为 0；`cache miss on unzip` 1801 行。

每次运行每个请求的命中（不同的值 × 请求数），和改前（11-11 正式测试同一次运行）对比：

| 输入 | hitrate | 并发 | 设计上限 | 改前 kvshrink | 改后 kvshrink |
|---|---|---|---|---|---|
| 32000 | 80 | 32 | 25088 | 15680 ×319；31360 ×1 | **25088 ×318**；15680 ×1；31360 ×1 |
| 32000 | 95 | 32 | 29792 | 15680 ×319；31360 ×1 | **29792 ×318**；15680 ×1；31360 ×1 |
| 32000 | 80 | 64 | 25088 | 15680 ×638；31360 ×2 | **25088 ×637**；15680 ×1；31360 ×2 |
| 32000 | 95 | 64 | 29792 | 15680 ×639；31360 ×1 | **29792 ×638**；15680 ×1；31360 ×1 |
| 128000 | 80 | 64 | 101920 | 100352 ×627；94080 ×11；127792 ×2 | **101920 ×638**；127792 ×2 |
| 128000 | 95 | 64 | 121520 | 120736 ×639；127792 ×1 | **121520 ×638**；120736 ×1；127792 ×1 |
| 256000 | 80 | 32 | 204624 | 203840 ×319；255584 ×1 | **204624 ×318**；203840 ×1；255584 ×1 |
| 256000 | 95 | 32 | 243040 | 241472 ×319；255584 ×1 | **243040 ×318**；241472 ×1；255584 ×1 |

每次运行只有 1 个请求低于上限，正是记下分叉点的那个请求（connector 日志里非 0 的
`shared_prefix_boundary` 每次运行 1 行，值等于该运行的设计上限）。命中 98.0% / 99.8% 的是和热身
请求内容相同的请求。

### 5.10 输出正确性：6000d gsm8k（实测，11-12）

6000d，改后的 kvshrink 和 vllm-prefix，vLLM 设置同 5.6；evalscope gsm8k 测试集前 200 题，每个
设置跑两遍，50 并发，max_tokens 2048，temperature 0，不开 thinking。结果
`intel-accel-for-llm-1030/tests/kvshrink_mamba_hit_dev/results/gsm8k-20261112_034746_10291e21_limit200/`
（`report.txt`）。

| 配置 | few-shot | 第几遍 | prompt token | 准确率 | 乱码题数 | 命中 |
|---|---|---|---|---|---|---|
| kvshrink | 26 | 1 | 5413-5523 | 97.5% | 0 | 4704 ×199；0 ×1 |
| kvshrink | 26 | 2 | 5413-5523 | 98.0% | 0 | 4704 ×185；5488 ×15 |
| kvshrink | 140 | 1 | 27795-27905 | 99.0% | 0 | 27440 ×150；20384 ×49；4704 ×1 |
| kvshrink | 140 | 2 | 27795-27905 | 99.0% | 0 | 27440 ×200 |
| vllm-prefix | 26 | 1 | 5413-5523 | 98.0% | 0 | 计数器 85.9% |
| vllm-prefix | 26 | 2 | 5413-5523 | 98.0% | 0 | 计数器 87.4% |
| vllm-prefix | 140 | 1 | 27795-27905 | 98.5% | 0 | 计数器 98.2% |
| vllm-prefix | 140 | 2 | 27795-27905 | 98.5% | 0 | 计数器 98.6% |

kvshrink 的命中取自 connector 日志（每个请求一个值）；vllm-prefix 为 vLLM 计数器这一遍的
命中 / 查询 token。两种配置准确率相近，没有乱码。kvshrink serve.log 的 1 个 `Traceback` 和
8 行 ` ERROR ` 是脚本停 vLLM 时（03:57:00 发 SIGTERM，之前所有请求都已返回 200）的
`EngineDeadError`，不是运行中的错误。

没有覆盖到改动：800 行 connector 日志的 `shared_prefix_boundary` 全是 0。gsm8k 每题独有的部分
（题目）短于一个 block（784），prompt 最后一个满 block 完全落在所有题共享的示例里；vLLM 总会
把 chunk 停在最后一个满 block，那里的状态本来就会存，不需要分叉点。要验证分叉点处状态的正确性，
需要每题独有的部分至少跨过一个满 block。

### 5.11 输出正确性：经分叉点状态命中的请求（实测，11-12）

6000d，容器 `iaxl.vllm-fddeef0-benchmark`，GPU0-3，只跑改后的 kvshrink，vLLM 设置同 5.6。64 道
gsm8k 测试题，每题 prompt = 指令 + 100 个共享示例 + 按题号抽的 10 个独有示例 + 题目（20949-21943
token）；题 1、2 单独发，其余 16 并发；temperature 0，不开 thinking，max_tokens 1024。05:16 开始，
05:19:37 结束。结果 `intel-accel-for-llm-1030/tests/kvshrink_mamba_hit_dev/results/
junction-20261112_051601_1c83c3a5/`（`report.txt`、`requests.jsonl` 含每题完整回答）。
每题 connector 日志的命中和响应 `cached_tokens` 逐题相同（报告脚本逐题核对）。

| 命中来源 | 命中 token | 题数 | 答对 | 准确率 | 乱码 |
|---|---|---|---|---|---|
| 分叉点 | 18816 | 62 | 60 | 96.8% | 0 |
| chunk 结尾（题 2，记下分叉点 18816） | 15680 | 1 | 1 | 100% | 0 |
| 无命中（题 1） | 0 | 1 | 1 | 100% | 0 |

和预期一致：题 2 记下分叉点 18816，题 3-64 全部命中 18816。另有 1 个请求记下分叉点 19600，
没有题命中 19600。答错的 2 题都是完整、连贯的推理，答案不同：题 13（柠檬树回本年数，答 12，
标准 13；11-12 的 26-shot gsm8k 中这道题也答错）和题 46（"2/5 times more" 理解成 2/5 倍，答 44，
标准 104）。serve.log 的 1 个 `Traceback` / 8 行 ` ERROR ` 是 05:19:07 脚本停 vLLM 时的
`EngineDeadError`；`cache miss on unzip` 0 行。

## 6. `[KVClip] cache miss on unzip (will retry): mamba:<hash>:<N>`

来自 `iaxl/csrc/torch_ext/zip.cpp`：从 DDR 载入时按标签取压缩数据，取不到就打印这行并以低
优先级重排一次；第二次取到则照常解压拷到 GPU，再取不到则 `IAXL_CHECK` 失败
（`cache key missing after retry`）。第一次取不到的原因代码没写，推测是写入未完成就开始载入
（未验证）。只影响载入等待，不影响命中数；但只重试一次。

这次 kvshrink serve.log（实测）：共 3403 行（3399 行完整 `mamba:` 标签，4 行多线程打印交错，
也是 mamba），495 个不同标签，全部是 mamba 状态；重试后仍缺失 0 次。按日志位置分到各次运行：

| 输入长度 | 运行 | 次数 |
|---|---|---|
| 32000 | 预热 | 688 |
| 32000 | 并发 32，hitrate 80 / 95 | 280 / 314 |
| 32000 | 并发 64，hitrate 80 / 95 | 307 / 309 |
| 32000 | 并发 128，hitrate 80 / 95 | 315 / 315 |
| 32000 | 并发 256，hitrate 80 / 95 | 280 / 315 |
| 32000 | 并发 300，hitrate 80 | 280 |
| 32000 并发 300 hitrate 95，及全部 128000 / 256000 | | 0 |

`std::cout` 有输出缓冲，位置可能晚几十行，vLLM 停止时缓冲中的行会丢失。

## 7. Debug 计划

在 6000d（本机，8 块 RTX 6000D，模型 `/media/model-space/Qwen3.8-27B`）复现和修改，在 b300
确认。6000d 用 TP=4（预期 block size 784，以日志为准），并加：

- `--max-num-batched-tokens 16384`：6000d 默认 8192（11-10 运行日志），b300 是 16384，
  chunk 结尾位置要和 b300 一致；
- `--enable-prompt-tokens-details`：响应的 `usage.prompt_tokens_details.cached_tokens` 给出
  每个请求的命中，vllm-prefix 才有按请求的数据。

步骤：

1. 读代码：vLLM `get_computed_blocks_for_connector`、scheduler 中有 connector 时分叉点的设置；
   kvshrink connector 的保存位置和 `get_num_new_matched_tokens`。确认 5.4 的三个前提。
2. 6000d 最小复现：同一组请求分别发给 kvshrink 和 vllm-prefix。输入 32000，先发 1 个请求
   存缓存，再逐个发 4 个只共享前 25600 token 的请求，记录每个请求的命中。
3. 给出修复方案。
4. 6000d 验证：每个请求 kvshrink 命中等于 vllm-prefix。
5. b300 确认：输入 32000 并发 32 / 64、hitrate 80 / 95，128000 和 256000 各一个，按请求对比；
   对齐后再跑正式测试。

## 8. 进度记录

- 11-12：建立本文档；完成第 4 节统计和第 5 节初步代码阅读。
- 11-12：debug 步骤 1 完成（5.5），三个前提都成立。下一步：步骤 2，6000d 最小复现。
- 11-12 02:58：步骤 2 开始。6000d 现有容器 `iaxl.vllm-fddeef0-benchmark`（1030 仓库，
  `kvshrink/` 与本目录相同，未安装或修改任何东西），脚本
  `intel-accel-for-llm-1030/tests/kvshrink_mamba_hit_dev/run_repro.sh`、`repro_client.py`
  （未跟踪），tmux `kvshrink-hit-repro`，结果
  `tests/kvshrink_mamba_hit_dev/results/repro-20261112_025832_238a3ea9`。预期（假设成立时）：
  kvshrink 请求 2-5 都命中 15680；vllm-prefix 请求 2 命中 15680，请求 3-5 命中 25088。
- 11-12 03:06：步骤 2 完成（结果见 5.6），EXIT=0，vLLM 已停，GPU0-3 已释放。
- 11-12 03:17：用户批准修改。已改本目录 `kvshrink/kvshrink_connector.py`
  `get_num_new_matched_tokens`：退回到有 mamba 状态的 block 后，若注意力匹配的 block 更多，
  设 `request.shared_prefix_boundary` = 注意力匹配 block 数 × block size；日志行末加
  `shared_prefix_boundary=`。改后的 `kvshrink/` 复制到 1030 仓库
  `tests/kvshrink_mamba_hit_dev/kvshrink_src/kvshrink/`（6000d 容器只挂载 1030 仓库），用
  `KVSHRINK_SRC` 经 PYTHONPATH 只替换 kvshrink 包。步骤 4 验证开始：tmux `kvshrink-hit-fix`，
  只跑 kvshrink，结果 `tests/kvshrink_mamba_hit_dev/results/repro-20261112_031738_b49277a4`。
- 11-12 03:20：步骤 4 完成（结果见 5.8），符合预期，EXIT=0，vLLM 已停，GPU0-3 已释放。
  下一步：输出正确性验证和 b300 确认（步骤 5）。
- 11-12 03:24 UTC：步骤 5 开始。改后的 `kvshrink_connector.py` 已同步到 b300（md5 `2b396593`，
  与本地相同）。b300 tmux `kvshrink-hit-fix`：`CONFIGS=kvshrink LEN_CONCURRENCY="32
- 11-12 05:19：分叉点正确性测试跑完（EXIT=0，结果见 5.11）：62 题经分叉点状态命中 18816，答对
  60（96.8%），无乱码。
- 11-12：修复提交到 `/home/pese/qiuyu/B300/intel-accel-for-llm-dsa-gdr-register-once` 分支
  `fix/kvshrink-mamba-shared-prefix-boundary`（`e2512d8`，基于 origin/main `bff83ae`，注释改为一行），
  该版本 connector（md5 `08597b6d`）已同步到本目录和 b300。为了让 lmcache 能存 KV，新增
  `ATTENTION_BACKEND`（`test-qwen38-27B.sh`、`kvshrink-vllm-serve.sh`、`lmcache-vllm-serve.sh`，不设
  时同原来，b300 默认 FLASHINFER），设了就加 `--attention-backend`、结果目录加后缀
  （如 `_flash_attn`）。正式重跑由用户执行：kvshrink（FLASHINFER）+ 四个配置（FLASH_ATTN）。000:32,64
  128000:64 256000:32"`，hitrate 80 / 95，种子与 11-11 正式测试相同（同样的请求），结果
  `bench-results/Qwen3.8-27B_tp4_20261112_032448`。
- 11-12 03:38：输出正确性测试（6000d）。用户要求：不要求逐字相同，要求结果正确、无乱码；只和
  vllm-prefix 比较准确率，不用 vllm-wo-prefix。两个测试都用 evalscope gsm8k（每题两遍，50 并发，
  max_tokens 2048，temperature 0，不开 thinking），vLLM 设置同 5.6：26-shot（约 5.5k token，同
  b300 11-11）和 140-shot（约 25k token，长于一段 chunk，会走分叉点）。脚本
  `intel-accel-for-llm-1030/tests/kvshrink_mamba_hit_dev/gsm8k_correctness.sh`、
  `gsm8k_correctness_report.py`。先跑 20 题试跑：tmux `kvshrink-gsm8k-pilot`，结果
  `tests/kvshrink_mamba_hit_dev/results/gsm8k-20261112_033804_10291e21_limit20`。
- 11-12 03:47：按用户要求取消试跑（试跑未完成，目录保留），直接跑 200 题完整测试：6000d tmux
  `kvshrink-gsm8k`，结果 `tests/kvshrink_mamba_hit_dev/results/gsm8k-20261112_034746_10291e21_limit200`。
- 11-12 04:46 UTC：b300 确认跑完（EXIT=0，结果见 5.9）：每次运行只有记下分叉点的 1 个请求低于
  设计上限，其余全部到达上限。
- 11-12 04:05：6000d gsm8k 跑完（EXIT=0，结果见 5.10）：准确率与 vllm-prefix 相近、无乱码；但
  分叉点一次也没触发，改动的正确性还没验证。报告脚本按 evalscope 1.14.1 的评分字段修正后重新生成。
- 11-12 05:16：分叉点正确性测试（6000d，只跑 kvshrink，约 10 分钟）：64 道 gsm8k 测试题，每题
  prompt = 指令 + 100 个共享示例 + 按题号抽的 10 个独有示例 + 题目（约 21k token）。实测分词：
  第 1、2 题共享前 19426 token = 24 个满 block，分叉点应为 18816；第 1 题最后一个满 block 是 27。
  预期：题 1 命中 0，题 2 命中 15680 并记下分叉点 18816，题 3-64 命中 18816（来源为分叉点）。
  脚本 `run_junction_correctness.sh`、`junction_correctness.py`，tmux `kvshrink-junction`，结果
  `tests/kvshrink_mamba_hit_dev/results/junction-20261112_051601_1c83c3a5`。
