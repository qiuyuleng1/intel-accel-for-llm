# KVShrink TPOT 偏高调查（2026-11-02 ~ 11-09）

本文记录这次调查里提出过的每个猜测、为验证它做的实验、实测结果，以及由结果得出的结论。
每个实验都先说明它要回答什么问题、怎么做，再给结果；**实测**和**推断**分开写。
所有脚本、结果、日志都在本仓库里，路径见第 14 节。

## 0. 结论摘要

1. **KVShrink 在命中率 100 的 benchmark 里 TPOT 比 vllm-prefix 高，主因是：关掉 vLLM 的 GPU 前缀缓存后，
   每个请求在显存里有一份自己的 KV 拷贝；共用一份时 attention kernel 快得多。**
   - Qwen3-32B、TP8、48 个请求同时 decode、上下文约 3.28 万 token 时，attention kernel 每次调用
     各自一份 620 µs，共用一份 263 µs；每步多出约 23 ms（72.0 ms 对 49.2 ms）。（实测，第 9 节）
   - 同样的现象在 LMCache（关前缀缓存）和不接任何 connector 的 vLLM（关前缀缓存）上完全一样：
     attention 每次调用 620 / 621 µs。所以这**不是 KVShrink 特有的问题**。（实测，第 11 节）
2. 以下猜测被实测排除或证明影响很小：
   - async load 的三种设置（默认 / 同步 / 等全部层）：TPOT 都比 vllm-prefix 高 26–51%，彼此差不多。（第 4 节）
   - KVShrink 被强制的 PIECEWISE CUDA graph：只让 vllm-prefix 的 TPOT 升高 0.6–8.8%。（第 5 节）
   - Python GIL 争用、worker 主线程抢不到 CPU：主线程等 CPU 的时间 ≤ 3 ms / 整次运行。（第 8 节）
   - attention backend 不同、cascade attention：所有配置都是 FlashAttention 2、同一个 kernel，没有 cascade。（第 10 节）
3. 为什么共用一份更快：**推断**是共用的那 16.8 MB（每层每卡）放得进 112 MB 的 L2 缓存，私有拷贝 805 MB 放不下；
   这张卡（RTX 6000D）禁止读硬件性能计数器，Nsight Compute 量不了，所以没有直接证实。（第 10 节）
4. 这个 benchmark 让所有请求共用同一个前缀，是共享最极端的情况，会放大这个差距；
   真实 serving 里并发请求的前缀多数不同，差距应当小很多（推断，未测）。（第 12 节）

## 1. 环境与术语

**机器**：h8263-6000d，8 张 NVIDIA RTX 6000D（每张约 83 GB 显存、L2 缓存 112 MB、156 个 SM），GPU 之间走 PCIe。
11-06 重启后主机内存为 1.5 TB。主机按 PERFORMANCE_MODE_SETUP.md（未包含在本分支）设置性能模式
（该文件 11-05 03:03 建立；之前的运行没有记录主机设置）。

**软件**：

| 用途 | 容器 / 镜像 | vLLM | 说明 |
|---|---|---|---|
| KVShrink、vllm-prefix、vllm-wo-prefix | 容器 `iaxl.vllm-78aaef4-benchmark`（镜像 `vllm-iaxl-78aaef4-benchmark:1.5.1`） | 0.29.0 | 代码 = 提交 78aaef4 + 本分支的工作区改动 |
| LMCache、vllm-prefix-v033 | 镜像 `qiuyu/vllm-openai-lmcache:v0.33.0-lmcache0.5.6`（每次 `docker run --rm`） | 0.33.0 | LMCache 0.5.6，MP 模式 |

**配置名**：

| 名字 | 含义 |
|---|---|
| kvshrink / default | KVShrink connector，GPU 前缀缓存关（`--no-enable-prefix-caching`），QAT + IAA 压缩，DSA 传输 |
| kvshrink-gpu-prefix | 同上，但打开 GPU 前缀缓存（`GPU_PREFIX_CACHING=1`） |
| vllm-prefix | 不接 connector 的 vLLM，打开 GPU 前缀缓存（上限参考） |
| vllm-prefix-piecewise | vllm-prefix，但 CUDA graph 强制 PIECEWISE（和 KVShrink 一样） |
| vllm-wo-prefix | 不接 connector 的 vLLM，关前缀缓存，什么都不复用 |
| lmcache | LMCache MP connector（vLLM 0.33），关前缀缓存 |
| vllm-prefix-v033 | vLLM 0.33 的 vllm-prefix（在 LMCache 镜像里），作为 lmcache 的同版本参考 |

**术语**：

- **一次 bench 运行**：一次 `vllm bench serve`，输入长度、命中率、并发数各取一个值。
- **命中率 X**：每个请求的前 X% token 是所有请求相同的前缀（`--random-prefix-len`），其余随机。
  命中率 100 时所有请求的 prompt 完全相同。
- **TPOT**：每个请求的（总耗时 − 首 token 时间）/（输出 token 数 − 1），再对一次运行的 n 个请求取平均值 / 中位数。单位 ms。
- **一步**：vLLM worker 执行一次 `execute_model`，batch 里每个请求生成 1 个 token（或做一段 prefill）。
- **私有拷贝 / 共用一份**：同一个前缀的 KV，在显存里是每个请求各占一份物理块，还是所有请求的 block table 指向同一批物理块。
  只有打开 GPU 前缀缓存时才会共用。
- **CUDA graph 模式**：vLLM 默认 FULL_AND_PIECEWISE（decode 整步录成一张图）；KVShrinkConnector 的
  `requires_piecewise_for_cudagraph()` 返回 True，vLLM 因此强制 PIECEWISE（attention 部分不进图）。

## 2. 早期 benchmark（11-02 ~ 11-05，调查之前）

这些运行不是为 TPOT 调查做的，记录在这里是因为它们已经出现了同样的差距。11-02 和 11-04 的运行**没有**检查主机性能模式。

### 2.1 Qwen3.8-27B TP4（11-02）

三次尝试都在第一个运行（KVShrink，输入 100000，并发 31，命中率 80）中途失败，没有数据：

| 结果目录 | 结果 |
|---|---|
| `bench-results/Qwen3.8-27B_tp4_20261102_073127` | worker 进程意外退出（`VllmWorker-2 died unexpectedly`），EngineDeadError |
| `bench-results/Qwen3.8-27B_tp4_20261102_074937` | 同上（`VllmWorker-1`） |
| `bench-results/Qwen3.8-27B_tp4_20261102_080417` | 运行中途停止，run.log 没有错误记录 |

### 2.2 Qwen3.8-27B TP8（11-02 09:56）

**实验**：Qwen3.8-27B，TP8，block 400；K1 = kvshrink，B1 = vllm-prefix，B0 = vllm-wo-prefix；
输入 100000 / 200000，输出 128，并发 8 / 16 / 32，每次运行请求数 = 并发 × 10。
结果：`bench-results/Qwen3.8-27B_tp8_20261102_095646/summary.csv`。

命中率 100 的 TPOT（平均 / 中位，ms）：

| 输入 | 并发 | n | kvshrink | vllm-prefix | kvshrink / vllm-prefix |
|---|---|---|---|---|---|
| 100000 | 8 | 80 | 63.1 / 61.5 | 27.8 / 28.1 | 2.27 |
| 100000 | 16 | 160 | 124.1 / 123.3 | 51.2 / 51.2 | 2.42 |
| 100000 | 32 | 320 | 290.9 / 279.4 | 94.6 / 96.6 | 3.07 |
| 200000 | 8 | 80 | 123.6 / 123.5 | 39.0 / 39.3 | 3.17 |
| 200000 | 16 | 160 | 248.8 / 248.1 | 75.3 / 76.5 | 3.31 |

最后一列是平均 TPOT 之比。这次差距比 11-05 的结果大得多（见第 3 节）；两次的输入长度和主机设置都不同，原因没有单独查。

### 2.3 Qwen2.5-72B-Instruct TP8（11-04）

**实验**：Qwen2.5-72B-Instruct（dense，80 层），TP8，vLLM block 16 和 256 各跑一遍（LMCache chunk = block），
输入 32000，输出 128，并发 1 / 2 / 4 / 8 / 16，请求数 20 / 80 / 96 / 96 / 96，命中率 80 / 100。
结果：`bench-results/Qwen2.5-72B-Instruct_tp8_20261104_100929/bs16`、`bs256`。

命中率 100 的 TPOT（平均 / 中位，ms，n = 96），以及用 `running-reqs-per-run.py` 从 serve.log 统计的同时在算的请求数（平均 / 最大，
vLLM 每 10 s 打印一次，样本少）：

| block | 并发 | kvshrink | vllm-prefix | lmcache | lmcache 在算的请求数 |
|---|---|---|---|---|---|
| 16 | 8 | 48.1 / 47.8 | 40.8 / 41.0 | 28.2 / 28.2 | 2.4 / 3 |
| 16 | 16 | 80.3 / 80.4 | 57.2 / 57.5 | 28.2 / 28.3 | 2.3 / 3 |
| 256 | 8 | 48.5 / 47.5 | 41.5 / 41.7 | 68.1 / 68.2 | 7.4 / 8 |
| 256 | 16 | 79.0 / 78.1 | 58.3 / 53.6 | 278.8 / 279.9 | 15.3 / 16 |

- kvshrink 并发 16 比 vllm-prefix 高 36–40%。
- LMCache 在 chunk 16 时同时只有 2–3 个请求在算（其余在等加载），TPOT 低是因为 batch 小，不能比较（第 11.1 节同样现象）。
- LMCache chunk 256 并发 16 的 TPOT 279 ms，原因没有查。

### 2.4 Qwen2.5-72B KVShrink 重跑（11-05）

- 11-05 03:07（`..._20261105_030708`）：第一个运行时 serve 退出，没有数据。
- 11-05 03:23（`..._20261105_032337`）：主机已设成性能模式、ACS 关闭（日志里有检查记录），只跑 kvshrink：

| block | 并发 8 TPOT 平均 / 中位 | 并发 16 TPOT 平均 / 中位 |
|---|---|---|
| 16 | 38.2 / 38.4 | 55.8 / 56.1 |
| 256 | 38.2 / 38.0 | 57.0 / 56.8 |

和 11-04 的 kvshrink（并发 16：80.3）相比低很多。两次的代码相同（提交 78aaef4，`code_md5.txt` 只有
kvshrink-benchmark.sh 的一行打印不同），vLLM 版本、CUDA graph 模式、前缀缓存设置也相同；
11-04 那次的主机设置没有记录。11-05 没有重跑 vllm-prefix，所以没有同条件的对比。

### 2.5 Qwen3.8-27B TP8（11-05 09:19）

`bench-results/Qwen3.8-27B_tp8_20261105_091908`：vLLM 中途退出，之后所有运行的请求都因连接被拒失败（completed = 0）。
此后 kvshrink-benchmark.sh 在每次运行后用 `/health` 检查 serve 是否还活着。

## 3. 起点：Qwen3.8-27B TP8 四配置 benchmark（11-05 09:43）

**问题**：四种配置在同样负载下的 TTFT / TPOT。
**实验**：`test-qwen38-27B.sh`（未包含在本分支）；Qwen3.8-27B（混合模型：64 层里 16 层 full attention、48 层 linear attention），TP8；
kvshrink / vllm-prefix 在 vLLM 0.29（block 400），lmcache 在 vLLM 0.33（block 448）；
输入 32768（并发 16 / 32 / 64 / 96）和 131072（并发 16 / 24），输出 128，命中率 80 / 100，请求数 = 并发 × 10。
结果：`bench-results/Qwen3.8-27B_tp8_20261105_094321/summary.csv`，耗时约 6 小时 48 分。

命中率 100 的 TPOT（平均 / 中位，ms）；括号里是平均值相对 vllm-prefix 的变化：

| 输入 | 并发 | n | vllm-prefix | lmcache | kvshrink |
|---|---|---|---|---|---|
| 32768 | 16 | 160 | 27.2 / 26.9 | 26.6 / 26.8（−2%） | 31.9 / 32.0（+17%） |
| 32768 | 32 | 320 | 39.7 / 40.0 | 38.8 / 39.3（−2%） | 49.4 / 50.0（+24%） |
| 32768 | 64 | 640 | 75.7 / 77.5 | 69.3 / 71.2（−8%） | 90.8 / 93.1（+20%） |
| 32768 | 96 | 960 | 90.4 / 92.5 | 94.8 / 98.1（+5%） | 121.0 / 124.5（+34%） |
| 131072 | 16 | 160 | 47.6 / 48.2 | 59.2 / 58.1（+24%） | 63.4 / 65.5（+33%） |
| 131072 | 24 | 240 | 64.5 / 66.5 | 75.2 / 77.2（+17%） | 85.6 / 88.7（+33%） |

- 命中率 80 时每个请求还要 prefill 约 20% 的 token，TPOT 被 prefill 主导（例如 32768 并发 96：vllm-prefix 696.8，kvshrink 731.9），
  后续调查只看命中率 100。
- lmcache 在这次运行里同时在算的请求数接近并发数（并发 64 时平均 58.3、最大 61），所以它的 TPOT 有效。

**最初的猜测**：KVShrink TPOT 偏高与 async load（异步加载、提前放行）有关。

## 4. 猜测：async load 导致 TPOT 偏高

### 4.1 Qwen3.8 不适合做这个测试（代码事实）

KVShrink 在混合模型上把 mamba 层排在最前面，提前放行的层数要加上全部 mamba 层
（`kvshrink_connector.py`：`selected_layers += self.num_mamba_layers`）。默认映射在并发 ≥ 7 时选 16 层，
16 + 48 = 64 层，即全部层，所以在 Qwen3.8 上"默认"和"等全部层再放行"是一回事。因此换成 dense 的 Qwen3-32B（64 层）。

### 4.2 Qwen3-32B 三种 async 设置（11-06 07:51）

**实验**：`test-qwen3-32B-async-load.sh`；Qwen3-32B，TP8，block 16，vLLM 0.29；输入 32768，输出 128，命中率 100，
并发 16（80 个请求）和 48（240 个请求），每次运行前 5 个 warm-up，每个输入长度先 prewarm；ScratchPool 48 GB，DDR 池 40 GB。

| 设置 | 含义 |
|---|---|
| default | setvars.sh 默认：并发 ≥ 7 时前 16 层加载完就放行，其余 48 层在 prefill 那一步里加载 |
| async-off | 同步加载，64 层都在 prefill 那一步里加载 |
| async-all-layers | 64 层全部加载完才放行 |

结果：`bench-results/Qwen3-32B_tp8_asyncload_20261106_075118`。TPOT 平均 / 中位（ms），括号里是平均值相对 vllm-prefix：

| 设置 | 并发 16（n = 80） | 并发 48（n = 240） |
|---|---|---|
| vllm-prefix | 29.6 / 29.7 | 70.0 / 72.8 |
| default | 37.3 / 37.8（+26%） | 101.7 / 105.9（+45%） |
| async-off | 38.9 / 39.4（+31%） | 95.4 / 98.2（+36%） |
| async-all-layers | 37.3 / 37.6（+26%） | 105.5 / 111.4（+51%） |

**结论（实测）**：三种 async 设置都比 vllm-prefix 高 26–51%，彼此差别小。async load 的设置不是主因。

## 5. 猜测：KVShrink 被强制的 PIECEWISE CUDA graph

**实验**：vllm-prefix 强制 `CUDAGRAPH_MODE=PIECEWISE`，其余同第 4.2 节。结果：`bench-results/Qwen3-32B_tp8_asyncload_20261106_083903`。

| 配置 | 并发 16 | 并发 48 |
|---|---|---|
| vllm-prefix（FULL_AND_PIECEWISE） | 29.6 / 29.7 | 70.0 / 72.8 |
| vllm-prefix-piecewise | 29.8 / 30.0（+0.6%） | 76.2 / 80.0（+8.8%） |

**结论（实测）**：PIECEWISE 只解释一小部分；kvshrink 仍比 vllm-prefix-piecewise 高 25–39%。

## 6. 猜测：KVShrink 自身的开销 还是 KV 的布局

**实验**：kvshrink-gpu-prefix（打开 GPU 前缀缓存）。prewarm 之后所有命中都来自显存里的前缀缓存，
KVShrink 的各个钩子照常运行、CUDA graph 也是 PIECEWISE，但不再加载任何东西，所有请求共用一份前缀。
结果：`bench-results/Qwen3-32B_tp8_asyncload_20261106_085337`。

| 配置 | 并发 16 | 并发 48 |
|---|---|---|
| vllm-prefix-piecewise | 29.8 / 30.0 | 76.2 / 80.0 |
| kvshrink-gpu-prefix | 30.3 / 30.4 | 74.8 / 78.4 |

**结论（实测）**：打开前缀缓存后，KVShrink 和 vllm-prefix-piecewise 一样快。剩下的差距只可能来自两件事：
(a) 每个请求一份私有拷贝（KV 布局）；(b) 加载 KV 时和计算抢资源。下面用代码打点分别验证。

## 7. 代码打点：每步记录（step trace，11-09 03:04）

**问题**：差距出在哪些步？加载占了多少时间？每步的 KV 块布局是什么样？

**做法**：新增 [kvshrink/step_trace.py](../../kvshrink/step_trace.py)，`KVSHRINK_STEP_TRACE=1` 时包住 vLLM 的
`execute_model`（V2 和 V1 model runner），每个 TP rank 每步打一行 `step-trace`。主要字段：

| 字段 | 含义 |
|---|---|
| reqs / decode_reqs / prefill_tokens | 这一步的请求数、其中做 decode 的请求数、其余请求的 token 数 |
| period_ms | 这一步开始到下一步开始的时间（"每步时长"） |
| main_cpu_ms / main_runq_ms | worker 主线程在这一步用的 CPU 时间 / 可运行但等 CPU 的时间（`/proc/thread-self/schedstat`） |
| other_threads_cpu_ms | 同进程其他线程用的 CPU 时间 |
| start_load_kv_ms 等 | KVShrink connector 各个钩子在这一步里花的时间 |
| loads_started / loads_unfinished / promoted | 这一步提交的加载、尚未完成的加载、提前放行的加载 |
| block_refs / block_unique / block_runs | rank 0 每 20 步：batch 中所有请求 block table 的块号总数、去重后的物理块数、块号连续段数 |

块布局是根据 scheduler 发给 worker 的块号在 rank 0 上镜像出来的（V2 runner 的 block table 在 GPU 上，读回会同步）。
`gpu_ms`（CUDA event 计时）因为 CPU 跑在 GPU 前面，在下一步读取时还没完成，全部是 −1，没有可用数据。

第一次尝试（`bench-results/Qwen3-32B_tp8_asyncload_steptrace_20261109_025506`）只包了 V1 runner，而 worker 用的是 V2，
没有任何 step-trace 行，作废。

**实验**：default、kvshrink-gpu-prefix、vllm-prefix-piecewise 三个设置，负载同第 4.2 节，打开 step trace。
结果：`bench-results/Qwen3-32B_tp8_asyncload_steptrace_20261109_030413`；分析：
[results/step-trace-analysis-20261109_030413.txt](results/step-trace-analysis-20261109_030413.txt)（`step-trace-analysis.py`）。

这次运行的 TPOT（平均 / 中位）：default 37.7 / 38.2、99.0 / 103.3；kvshrink-gpu-prefix 29.6 / 29.7、74.6 / 78.5；
vllm-prefix-piecewise 30.5 / 30.9、76.1 / 80.9（并发 16 / 48）。

**7.1 满批、纯 decode、没有任何加载活动的步**（rank 0；"满批"= 请求数 ≥ 并发 − 2）：

| 设置 | 并发 48：每步时长 平均 / 中位（n） | 并发 16：每步时长 平均 / 中位（n） | 主线程 CPU（并发 48） | 主线程等 CPU |
|---|---|---|---|---|
| default | 77.9 / 72.0（398） | 33.6 / 32.8（501） | 11.1 ms | 0 |
| kvshrink-gpu-prefix | 52.9 / 49.1（451） | 27.0 / 25.8（538） | 11.0 ms | 0 |
| vllm-prefix-piecewise | 53.6 / 49.2（430） | 26.4 / 26.0（539） | 10.5 ms | 0 |

**实测**：没有任何加载活动、主线程 CPU 一样、不等 CPU 的步，default 每步仍多出约 23 ms（中位 72.0 对 49.1）。
这说明差距来自 GPU 上的计算，与加载无关。

**7.2 KV 块布局**（rank 0，并发 48，中位，n = 35–36 次采样）：

| 设置 | 引用的块总数 | 不同的物理块数 | 块号连续段数 |
|---|---|---|---|
| kvshrink 默认 | 98,390 | 98,390 | 19,739 |
| kvshrink 打开前缀缓存 | 98,420 | 2,297 | 950 |
| vllm-prefix PIECEWISE | 98,416 | 2,303 | 863 |

列说明：每块 16 token；"不同的物理块数" = 去重后的块数；"块号连续段数" = 每个请求的块号序列按"相邻差 1"切成的段数之和。

**7.3 加载相关的开销**（default，并发 48，rank 0，734 步共 70.86 s）：

| 项目 | 时间 | 占步时长 |
|---|---|---|
| start_load_kv（主线程提交加载） | 2.88 s | 4.1% |
| wait_for_layer_load 中真正阻塞的时间 | 1.38 s | 1.9% |
| get_finished | 0.11 s | 0.2% |
| save_kv_layer | 0.07 s | 0.1% |

- 每提交一个请求的加载，主线程在 start_load_kv 里花约 12–13 ms（一步里提交 1 / 2 / 3 个：12.7 / 24.8 / 36.1 ms）。
- 满批且有加载活动的步，其他线程（压缩 / 传输）平均每步用 1.43 s CPU（中位 1.09 s，n = 17）；没有加载活动时中位 1.5 ms。
- 按步的类型分（default，并发 48，全部步）：没有加载活动 475 步（占步时长 48.9%，平均 73.0 ms）、
  有 prefill 146 步（30.7%，平均 149.1 ms）、有加载活动 113 步（20.3%，平均 127.6 ms）。
- 三个设置里，紧跟在一个含 16-token prefill 的步之后的那一步都约 210–236 ms。由于每步时长看的是"这一步开始到下一步开始"，
  它可能反映的是上一步混合 batch 的 GPU 时间；没有 gpu_ms，无法归属（未解决）。

## 8. 猜测：GIL 争用 / 主线程抢不到 CPU（11-09 03:22）

**做法**：主机上用 [proc-thread-sampler.py](proc-thread-sampler.py) 每秒采样所有 `VLLM::Worker_TP*` 线程的 CPU 时间和等 CPU 时间
（`/proc/<pid>/task/<tid>/stat`、`schedstat`），采样器自己绑在 CPU 255；KVShrink 打开 `KVSHRINK_THREAD_MAP=1`，
把 Python 线程名和 Linux 线程号打进日志，用来区分 Python 线程和 native 线程。
运行：default、kvshrink-gpu-prefix，负载同第 4.2 节。
结果：`bench-results/Qwen3-32B_tp8_asyncload_procthreads_20261109_032233`，采样：
[results/proc-threads-20261109_032233.csv](results/proc-threads-20261109_032233.csv)，分析：
[results/proc-thread-analysis-20261109_032233.txt](results/proc-thread-analysis-20261109_032233.txt)。

这次运行的 TPOT（没有 step trace）：default 37.5 / 38.1、103.8 / 108.8；kvshrink-gpu-prefix 29.8 / 30.0、71.9 / 74.7。

default 每次运行的线程 CPU 时间（秒，rank 0；8 个 rank 相近）：

| 运行 | 运行时长 | 主线程 CPU | 主线程等 CPU | 其他 Python 线程 CPU（最大的 WorkerAsyncOutputCopy） | native 线程 CPU 合计 |
|---|---|---|---|---|---|
| 并发 16 | 27.4 s | 9.29 | 0.001 | 0.15 | 138.8（每个 openmp_worker 11.7） |
| 并发 48 | 74.3 s | 12.84 | 0.001 | 0.17 | 473.7（每个 openmp_worker 40.6） |

8 个 rank 中主线程等 CPU 最多 0.003 s，其他 Python 线程最多 0.27 s。kvshrink-gpu-prefix 的 native 线程 CPU 只有 0.4–1.0 s。

**结论（实测）**：主线程几乎不等 CPU，其他 Python 线程几乎不用 CPU，GIL 争用和主线程抢不到 CPU 都可以排除。
加载用的 CPU 全在 native 线程（OpenMP 压缩 / 传输线程）。

## 9. kernel 级 profile：多出的 23 ms 在哪个 kernel（11-09 05:10）

**问题**：同样是 48 个请求满批 decode、没有到达和加载，各自一份每步 72 ms、共用一份 49 ms，多出的时间花在哪个 GPU kernel？

**做法**：[profile-decode-qwen3-32b.sh](profile-decode-qwen3-32b.sh)（在容器里跑）：
1. 启动 vLLM（Qwen3-32B，TP8，block 16，设置同 benchmark），打开 vLLM 自带的 torch profiler（`--profiler-config`）和 step trace；
2. 发 1 个请求存下 32768 token 的 prompt；
3. 同时发 48 个相同的请求（命中率 100），等 48 个都在 decode 后，`POST /start_profile` 录 20 步；
4. [profile-kernel-summary.py](profile-kernel-summary.py) 从 8 个 rank 的 trace 里按 kernel 名字分类，20 步合计 ÷ 20 = 每步。

一次性发 48 个请求是为了让录到的 20 步都是满批纯 decode（benchmark 里这种步只是夹在请求到达之间的一段段）。
有效性：profile 窗口前后的每步时长 default 71.8–72.0 ms、gpu-prefix 49.1–49.3 ms，与 benchmark 的稳定步（72.0 / 49.1）一致；
窗口内没有加载活动，每步时长不变，说明 profiler 没有拖慢 GPU。
结果：[results/profile-decode-qwen3-32b-20261109_051007](results/profile-decode-qwen3-32b-20261109_051007)，
汇总：[results/profile-kernel-summary-20261109_051007.txt](results/profile-kernel-summary-20261109_051007.txt)。

`/start_profile` 时日志里出现的 `External init callback must run in same thread as registerClient`
（API server 进程的 PyTorch profiler 前端记录没能启动）和 `ADF_UIO_PROXY err` / `[iaa_zip] ...`
（worker 在 prewarm 时初始化 QAT/IAA 时写入 stdout 缓冲区的旧输出，被 profiler 的输出一起冲出来）不影响结果；
`[qat_zip] init done`（stderr，立即输出）在 prewarm 时就已出现。

结果（rank 0，每步 ms；8 个 rank 之间相差 ≤ 0.3 ms）：

| kernel 类别 | 各自一份 | 共用一份 | 差 |
|---|---|---|---|
| attention（`flash_fwd_splitkv_kernel`） | 39.69 | 16.82 | **+22.87** |
| allreduce（NCCL） | 21.87 | 21.88 | 0.0 |
| GEMM | 7.56 | 7.77 | −0.2 |
| attention 合并（`flash_fwd_splitkv_combine_kernel`） | 0.40 | 0.27 | +0.1 |
| allgather + 其他 | 1.48 | 1.50 | 0.0 |
| **所有 kernel 合计** | **71.00** | **48.25** | **+22.75** |
| 实测每步时长 | ≈72.0 | ≈49.2 | +22.8 |

attention 单次调用（n = 1280 = 20 步 × 64 层）：各自一份 平均 620.2 / 中位 619.4 µs（8 个 rank 平均 620.0–622.3），
共用一份 262.9 / 263.1 µs（261.0–267.6）。

**结论（实测）**：多出的约 23 ms 全部在 attention kernel 上，单次调用慢 2.36 倍；通信、矩阵乘等两边完全一样。

## 10. 为什么共用一份的 attention 更快

### 10.1 L2 缓存（推断）

- 每张卡 1 个 KV head（8 个 KV head / TP8），每 token 每层 K + V = 2 × 128 × 2 字节 = 512 B；
  3.28 万 token → 每个请求每层 16.8 MB。
- 共用一份：48 个请求读同一块 16.8 MB，放得进 112 MB 的 L2，第一次从显存读进 L2 后基本都在 L2 命中。
- 各自一份：48 份不同的数据，每层 805 MB，是 L2 的 7 倍，每次都要从显存读。
- 等效读速：共用 805 MB / 263 µs ≈ 3.06 TB/s；各自一份 ≈ 1.30 TB/s。
- FlashAttention 2 本身不知道"共同前缀"：每个线程块按 block table 读自己那个请求的 KV，硬件 L2 自动让相同地址的读取命中；
  但从 L2 读进 SM、以及计算，仍然是 48 份，所以共用时是 263 µs 而不是 620 / 48。

### 10.2 Nsight Compute 量不了（实测）

在 KVShrink 容器里装了 Nsight Compute 2026.1.1（与驱动支持的 CUDA 13.2 配套），在 GPU 0 上测一个读 256 MiB 的 kernel：

```
==ERROR== Profiling is not supported on the specific SKU of device 0.
```

容器以 root、privileged 运行，满足 `RmProfilingAdminOnly=1`；是 RTX 6000D 这个型号禁止读硬件性能计数器。
所以"从显存读了多少字节"无法直接测量，10.1 节保持为推断。

### 10.3 attention backend 与 cascade attention（实测）

- 所有运行（kvshrink 默认、kvshrink 打开前缀缓存、vllm-prefix、vllm-prefix-piecewise、lmcache、vllm-prefix-v033、vllm 关前缀缓存）
  都是 `FLASH_ATTN` backend、FlashAttention 2、V2 model runner。
- vLLM 0.29 的 cascade attention 只在 V1 runner（`v1/worker/gpu_model_runner.py`）里；V2 runner 的目录 `v1/worker/gpu/` 里没有任何引用。
- trace 里每层每步正好 1 次 `flash_fwd_splitkv_kernel` + 1 次合并 kernel（1280 次 = 20 × 64）；cascade 会多一次 attention 和一个结果合并 kernel，没有出现。
- 两种布局下 kernel 模板参数完全相同，grid `[1, 6, 48]`、block 128、shared memory 81920 字节：
  48 = 48 个请求（同一张卡的 8 个 q head 共用 1 个 KV head，放在同一个线程块里），6 = 每个请求的 token 切成 6 段并行。
- cascade attention（vLLM V1 `flash_attn.py`）的做法：把所有请求的 query 拼成一个序列，对共同前缀算一次 attention（真正"读一份算 48 个"），
  再对各自后缀算一次，最后按 softmax 归一化量合并；条件是共同前缀 ≥ 256 token、请求数 ≥ 8，且前缀在显存里必须是同一批物理块。

## 11. 与 LMCache 和不接 connector 的 vLLM 对比

### 11.1 LMCache 的 KV 布局（代码 + 日志）

- GPU 上：LMCache 把 KV 写进 vLLM 给每个请求分配的显存块。我们的 LMCache 配置也是 `--no-enable-prefix-caching`，
  11-05 那次所有 lmcache 运行的 GPU 前缀命中都是 0，所以同样是**每个请求一份私有拷贝**。
- CPU 上（LMCache server 的 L1）：一个对象 = 一个 chunk 的所有层，形状 [K/V, 层, token, 头数×头维度]，pinned 内存，不压缩；
  key 含 chunk 内容哈希和 TP rank，同一前缀只存一份。
- 加载：异步（`get_num_new_matched_tokens` 返回 `need_to_load > 0`，lmcache_mp_connector.py 第 1351 行），
  由 LMCache server 进程通过 CUDA IPC 写入 vLLM 的显存块；`wait_for_layer_load` 是空函数。

### 11.2 Qwen3-32B LMCache benchmark（11-09 05:59，LMCache 结果无效）

**实验**：`test-qwen3-32B-async-load.sh` 的 lmcache 与 vllm-prefix-v033 设置，负载同第 4.2 节，LMCache chunk 16（= vLLM block）、L1 40 GB。
结果：`bench-results/Qwen3-32B_tp8_asyncload_lmcache_20261109_055927`。

| 配置 | vLLM | 并发 16 TPOT | 并发 48 TPOT | 并发 16 TTFT | 并发 48 TTFT | 并发 48 在算的请求数 平均 / 最大 |
|---|---|---|---|---|---|---|
| vllm-prefix（11-06） | 0.29 | 29.6 / 29.7 | 70.0 / 72.8 | 495 / 421 | 960 / 692 | — |
| kvshrink default（11-06） | 0.29 | 37.3 / 37.8 | 101.7 / 105.9 | 654 / 537 | 1,480 / 1,068 | — |
| vllm-prefix-v033 | 0.33 | 27.6 / 27.7 | 67.2 / 69.4 | 504 / 407 | 983 / 627 | 42.8 / 48 |
| lmcache | 0.33 | 19.2 / 19.2 | 19.3 / 19.3 | 11,623 / 12,298 | 38,114 / 41,420 | **2.6 / 3** |

- vLLM 0.33 比 0.29 的 vllm-prefix 快 7%（并发 16）/ 4%（并发 48）：版本差异不是 kvshrink 26–45% 差距的原因。
- **LMCache 的 TPOT 不能比较**：同一时刻最多 3 个请求在算，vLLM 日志一直是 `KV fetch: 45 in progress`；
  223.7 s 里加载了 7.93M token（约 3.5 万 token/s，约 1.1 个请求/s）。11-04 Qwen2.5-72B chunk 16 也是这样（第 2.3 节），
  11-05 Qwen3.8 chunk 448 则正常。chunk 16 导致拷贝很碎（每层每块 8 KB）**是推断，未验证**。
- vllm-prefix-v033 的 serve.log 有 8 行 ERROR（`EngineDeadError`），出现在 bench 全部结束后脚本停止 vLLM 时，不影响结果。

### 11.3 kernel 级 profile：LMCache、vLLM 0.33、vLLM 关前缀缓存（11-09 06:30、07:19）

方法同第 9 节。LMCache 镜像里没有装 KVShrink，step trace 通过 PYTHONPATH 加载
（[step_trace_plugin/](step_trace_plugin) 的 dist-info 把 `kvshrink/step_trace.py` 注册成 vLLM 插件）。
LMCache 加载慢，输出改为 2048 token，保证第 48 个请求开始 decode 时前面的请求还在生成；profile 写完即停。
开始 profile 的条件后来改为"rank 0 的 step trace 显示 48 个请求都在 decode、没有任何 prefill"
（不复用 KV 时请求在分块 prefill 期间就已算作"在跑"）。

结果：[results/profile-decode-qwen3-32b-lmcache-20261109_063020](results/profile-decode-qwen3-32b-lmcache-20261109_063020)（lmcache、vllm-prefix-v033）、
[results/profile-decode-qwen3-32b-woprefix-20261109_071929](results/profile-decode-qwen3-32b-woprefix-20261109_071929)（vllm 关前缀缓存）；
汇总：`results/profile-kernel-summary-*.txt`。

**总表**（块统计取 rank 0 在 profile 窗口内那一步；kernel 时间取 rank 0 的 20 步，attention n = 1280）：

| 设定 | vLLM | 引用的块总数 | 不同的物理块数 | 块号连续段数 | attention 每次调用 平均 / 中位（µs） | attention 每步（ms） | allreduce 每步（ms） | 所有 kernel 每步（ms） | 实测每步时长（ms） |
|---|---|---|---|---|---|---|---|---|---|
| kvshrink 默认 | 0.29 | 98,438 | 98,438 | 2,269 | 620.2 / 619.4 | 39.69 | 21.87 | 71.00 | ≈72.0 |
| kvshrink 打开前缀缓存 | 0.29 | 98,448 | 2,239 | 238 | 262.9 / 263.1 | 16.82 | 21.88 | 48.25 | ≈49.2 |
| vllm 关前缀缓存 | 0.29 | 98,652 | 98,652 | 2,500 | 621.2 / 620.5 | 39.76 | 22.10 | 71.29 | ≈71.6 |
| lmcache | 0.33 | 98,446 | 98,446 | 2,284 | 619.8 / 619.0 | 39.67 | 12.28 | 61.48 | ≈61.5 |
| vllm-prefix-v033 | 0.33 | 98,448 | 2,239 | 238 | 261.2 / 261.8 | 16.72 | 12.22 | 38.53 | ≈38.8 |

列说明同第 7.2 节；"实测每步时长"是 step trace 的 period_ms。8 个 rank 的 attention 每次调用平均值：
lmcache 619.8–621.9 µs，vllm-prefix-v033 260.2–267.0 µs，vllm 关前缀缓存 621.2–623.7 µs。

**结论（实测）**：

1. 只要显存里每个请求各自一份，三种方案（KVShrink、LMCache、不接 connector 的 vLLM）attention 都是约 620 µs；
   共用一份约 262 µs，两个 vLLM 版本一样。
2. 私有拷贝在两个版本上都让每步多约 23 ms：0.29 上 72.0 对 49.2，0.33 上 61.5 对 38.8。
3. 两个版本的差别在 allreduce：0.33 每次调用约 95 µs、每步 12.2 ms，0.29 约 170 µs、每步 21.9 ms（原因未查）。
   这也解释了第 3 节 Qwen3.8 上 lmcache（0.33，私有拷贝）为什么看起来和 vllm-prefix（0.29，共用）差不多（推断；Qwen3.8 只有 16 层 attention，私有拷贝的代价本来就小）。
4. 在稳定 decode 阶段 KVShrink 本身几乎不增加开销：和 vllm 关前缀缓存相比，所有 kernel 每步 71.00 对 71.29 ms，实测每步 72.0 对 71.6 ms
   （还混着 PIECEWISE 对 FULL_AND_PIECEWISE 的差别）。
5. 块号是否连续不影响 attention：单波测试里 kvshrink 默认只有 2,269 段（benchmark 里 19,739 段），attention 一样是 620 µs。起作用的是"不同的物理块数"。

其他观察：LMCache profile 时命中 token 1,212,398，少于 48 × 32,752 = 1,572,096，约 11 个请求没从 LMCache 命中、自己 prefill（同样是私有拷贝，不影响上表；原因未查）。

## 12. 讨论

- **不是 KVShrink 独有的问题**，准确地说是"tiering 方案关掉 vLLM GPU 前缀缓存"时的问题。LMCache 常见部署是开着前缀缓存的。
- **打开前缀缓存也不完全消除私有拷贝**（vLLM 0.29 代码事实，影响未测）：异步加载时（`load_kv_async=True`）
  加载进来的块要等加载完成后才登记进前缀缓存（`scheduler.py` 里 `delay_cache_blocks=load_kv_async`，
  完成后在 `_update_waiting_for_remote_kv` 里 `cache_blocks`）；第一个请求还在加载时到达的同前缀请求仍会各自加载一份。
  同步加载在分配块时就登记。KVShrink 默认关前缀缓存的原因不清楚；"打开前缀缓存 + 真正发生加载"没有测过。
- **benchmark 放大了差距**：命中率 100 时所有请求的 32k 前缀完全相同，vllm-prefix 得到了真实 serving 里很少有的 L2 共享。
  真实 serving 里不同用户的前缀通常不同，vllm-prefix 也要读多份数据，差距应当缩小（推断，未测）。
  多个并发请求共享同一个长前缀的场景（热点长文档 RAG、带长 system prompt 的 agent、固定 few-shot 的批量评测）仍会出现，但共享部分通常只有几千 token。
- **显存容量也是代价**：48 个请求私有时用 98,446 个块，共用时 2,239 个。任何并发请求之间有共享前缀时，私有拷贝都会浪费显存、降低能同时运行的请求数（对最大并发的影响未测）。
- 如果要让 benchmark 更像真实 serving，可以改成多个不同前缀（例如 8 篇文档、每篇被 6 个请求共享），控制共享程度（建议，未做）。

## 13. 未解决 / 未验证的事项

1. step trace 的 GPU 计时（gpu_ms）全部 −1；请求到达步、prefill 后那一步（约 210–236 ms）的开销没有归属。
2. L2 解释是推断：RTX 6000D 禁止读硬件计数器，Nsight Compute 不可用。
3. vLLM 0.33 的 allreduce 为什么比 0.29 快（95 对 170 µs 每次调用）没有查。
4. LMCache chunk 16 加载慢（约 1.1 个请求/s）的原因没有验证；Qwen2.5-72B LMCache chunk 256 并发 16 TPOT 279 ms 没有解释。
5. LMCache profile 时约 11 个请求没有命中，原因没查。
6. KVShrink 打开前缀缓存且真正发生加载时的行为（正确性、私有拷贝持续多久）没有测。
7. Qwen2.5-72B 11-04 → 11-05 KVShrink TPOT 从 80.3 降到 55.8（并发 16，同代码）：11-04 主机设置没有记录，11-05 没有重跑 vllm-prefix。
8. Qwen3.8 11-02（100k / 200k，差 2.3–3.3 倍）与 11-05（32k / 128k，+17–34%）差距大小不同，没有单独查。
9. 多个不同前缀的 benchmark 没有跑。

## 14. 文件索引

**脚本**（仓库根目录）：

| 文件 | 作用 |
|---|---|
| `kvshrink-benchmark.sh` | 在容器里跑一个或多个配置的 benchmark，写 summary.csv / run.log / serve.log |
| `test-qwen38-27B.sh`（未包含在本分支） | 第 3 节 Qwen3.8 四配置 benchmark（主机上运行） |
| `test-qwen25-72B.sh`（未包含在本分支） | 第 2.3 / 2.4 节 Qwen2.5-72B benchmark |
| `test-qwen3-32B-async-load.sh` | 第 4–8、11.2 节 Qwen3-32B 各设置（default / async-off / async-all-layers / vllm-prefix / vllm-prefix-piecewise / kvshrink-gpu-prefix / lmcache / vllm-prefix-v033） |
| `examples/kvshrink-vllm-serve.sh` | KVShrink / vllm-prefix / vllm-wo-prefix 的 serve；可选 `CUDAGRAPH_MODE`、`GPU_PREFIX_CACHING`、`TORCH_PROFILER_DIR` |
| `examples/lmcache-vllm-serve.sh`（未包含在本分支） | LMCache MP serve；`BASELINE=prefix_cache` 跑不接 LMCache 的 vLLM 0.33；可选 `TORCH_PROFILER_DIR`。lmcache / vllm-prefix-v033 设置依赖它 |
| `tools/bench-host-setup.sh`、`tools/check-acs.sh` | 主机性能模式 / ACS 检查、清页缓存、确保 KVShrink 容器在运行 |
| `kvshrink/step_trace.py` | 第 7 节每步记录（`KVSHRINK_STEP_TRACE=1`、`KVSHRINK_THREAD_MAP=1`） |

**tests/bench_dev/**：

| 文件 | 作用 |
|---|---|
| `step-trace-analysis.py` | 第 7 节分析 step-trace 行 |
| `proc-thread-sampler.py`、`proc-thread-analysis.py` | 第 8 节线程 CPU 采样与分析 |
| `running-reqs-per-run.py` | 从 serve.log 统计每次运行在算 / 等待 / 等加载的请求数 |
| `profile-decode-qwen3-32b.sh` | 第 9、11.3 节单波 torch profile（default / gpu-prefix / vllm-wo-prefix / lmcache / vllm-prefix-v033） |
| `profile-kernel-summary.py` | 从 torch trace 按 kernel 类别汇总每步时间 |
| `step_trace_plugin/` | 在 LMCache 镜像里把 step_trace 注册成 vLLM 插件 |

**结果**：

| 目录 / 文件 | 对应章节 |
|---|---|
| `bench-results/Qwen3.8-27B_tp4_20261102_*` | 2.1 |
| `bench-results/Qwen3.8-27B_tp8_20261102_095646` | 2.2 |
| `bench-results/Qwen2.5-72B-Instruct_tp8_20261104_*`、`..._20261105_*` | 2.3、2.4 |
| `bench-results/Qwen3.8-27B_tp8_20261105_091908` | 2.5 |
| `bench-results/Qwen3.8-27B_tp8_20261105_094321` | 3 |
| `bench-results/Qwen3-32B_tp8_asyncload_20261106_075118`、`_083903`、`_085337` | 4、5、6 |
| `bench-results/Qwen3-32B_tp8_asyncload_steptrace_20261109_025506`（作废）、`_030413` | 7 |
| `bench-results/Qwen3-32B_tp8_asyncload_procthreads_20261109_032233` | 8 |
| `bench-results/Qwen3-32B_tp8_asyncload_lmcache_20261109_055927` | 11.2 |
| `tests/bench_dev/results/step-trace-analysis-20261109_030413.txt` | 7 |
| `tests/bench_dev/results/proc-threads-20261109_032233.csv`、`proc-thread-analysis-20261109_032233.txt` | 8 |
| `tests/bench_dev/results/profile-decode-qwen3-32b-20261109_051007`、`profile-kernel-summary-20261109_051007.txt` | 9 |
| `tests/bench_dev/results/profile-decode-qwen3-32b-lmcache-20261109_063020`、`profile-kernel-summary-{lmcache,vllm-prefix-v033}-20261109_063020.txt` | 11.3 |
| `tests/bench_dev/results/profile-decode-qwen3-32b-woprefix-20261109_071929`、`profile-kernel-summary-vllm-wo-prefix-20261109_071929.txt` | 11.3 |
