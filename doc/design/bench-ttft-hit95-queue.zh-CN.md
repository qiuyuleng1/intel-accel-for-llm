# 高并发时 hitrate 95 的 TTFT 反而比 hitrate 80 高：原因与验证

状态：已查明（11-16）。

结论：vLLM 0.29 的 Python API server（vLLM 里接收 HTTP 请求、把 prompt 文本分词后交给引擎的进程）
默认只用 1 个线程分词。hitrate 95 时引擎每个请求要算的 token 少、跑得快，请求就在进入引擎之前排队
等分词，这段排队时间全部算进 TTFT；hitrate 80 时引擎本身是瓶颈，请求在引擎里排队，TTFT 反而短。
不是 KV 加载慢。加 `--renderer-num-workers 8`（8 个分词线程）或换 Rust 前端都能消掉大部分排队
（见“11-16 验证二”）。

## 现象

b300 正式测试（Qwen3.8-27B，TP=4，GPU 0/2/4/6，输出 128 token），输入 32000、并发 256 和 300 时，
vllm-prefix 和 kvshrink 都是 hitrate 95 的 TTFT 比 hitrate 80 高很多；并发 ≤ 128 时没有这个现象。

| 配置 | 并发 | TTFT 均值 hitrate 80 (ms) | TTFT 均值 hitrate 95 (ms) |
|---|---|---|---|
| vllm-prefix | 256 | 3957 | 10829 |
| vllm-prefix | 300 | 3408 | 14495 |
| kvshrink | 256 | 4670 | 9835 |
| kvshrink | 300 | 7150 | 15578 |

数据：vllm-prefix 来自 `bench-results/Qwen3.8-27B_tp4_20261111_095943`，kvshrink（修复后，FlashInfer）
来自 `bench-results/Qwen3.8-27B_tp4_20261112_091700`。

## 11-13 的分析（实测数据 + 推断）

`vllm bench serve` 是闭环发请求：并发 256 表示客户端始终保持 256 个请求在途，完成一个才发下一个；
TTFT 从客户端发出请求开始计时，请求在哪里排队都算进 TTFT。

实测：serve.log 里 vLLM 每 10 秒打印的 `Engine 000: ... Running: N reqs, Waiting: M reqs`，取每次运行
期间的平均值（去掉开头 3 个点）。"每秒完成请求数" = 引擎每秒生成 token 数 ÷ 128；"每个请求总时间"
= TTFT + 127 × TPOT。

| 配置 | 并发 | hitrate | 引擎里的请求数（运行 + 排队） | 每秒完成请求数 | TTFT 均值 (ms) | TPOT 均值 (ms) | 每个请求总时间 (s) |
|---|---|---|---|---|---|---|---|
| vllm-prefix | 128 | 80 | 108 + 9 | 8.2 | 3485 | 98 | 15.9 |
| vllm-prefix | 128 | 95 | 124 + 0 | 11.8 | 1221 | 77 | 11.0 |
| vllm-prefix | 256 | 80 | 239 + 11 | 8.3 | 3957 | 213 | 31.0 |
| vllm-prefix | 256 | 95 | 129 + 0 | 12.0 | 10829 | 81 | 21.2 |
| vllm-prefix | 300 | 80 | 291 + 4 | 8.3 | 3408 | 257 | 36.0 |
| vllm-prefix | 300 | 95 | 126 + 0 | 12.1 | 14495 | 79 | 24.6 |
| kvshrink | 128 | 80 | 109 + 12 | 6.8 | 3955 | 119 | 19.1 |
| kvshrink | 128 | 95 | 115 + 2 | 10.0 | 2011 | 87 | 13.1 |
| kvshrink | 256 | 80 | 237 + 10 | 7.1 | 4670 | 259 | 37.5 |
| kvshrink | 256 | 95 | 161 + 2 | 10.0 | 9835 | 124 | 25.5 |
| kvshrink | 300 | 80 | 270 + 26 | 6.6 | 7150 | 302 | 45.5 |
| kvshrink | 300 | 95 | 152 + 2 | 9.6 | 15578 | 121 | 30.9 |

要点：

- hitrate 80：引擎里的请求数（250-296）≈ 客户端并发，瓶颈是引擎 prefill（每个请求要算 6912 token，
  每秒约 8 个）；新请求很快被 prefill，TTFT 短，但 prefill 和 decode 挤在同一步，TPOT 高。
- hitrate 95、并发 256 / 300：客户端在途 256 / 300 个，引擎里只有约 126-163 个，且几乎不排队
  （Waiting 0-2），即有 100 多个请求还没进引擎。此时引擎每秒完成 10-12 个请求，比 hitrate 80 多约
  40%，每个请求的总时间也更短。
- Little 定律核对（vllm-prefix，并发 256，hitrate 95）：256 ÷ 12.0 ≈ 21.3 s 在途，≈ 实测总时间
  21.2 s；在引擎里 ≈ 129 ÷ 12.0 ≈ 10.7 s（≈ decode 127 × 81 ms ≈ 10.3 s）；其余约 10.6 s 在进引擎
  之前，≈ TTFT 均值 10.8 s。

结论（推断）：hitrate 95 时 TTFT 变高不是处理变慢，而是排队位置从"引擎内"变成了"引擎之前"，后者整段
算进 TTFT。系统上限约每秒 12 个请求，且瓶颈不在引擎（不排队，GPU KV cache 只用约 5%），而在请求
进引擎之前：最可能是 vLLM API server 进程（单个 Python 进程解析约 32000 token 的文本 prompt、分词、
流式返回），也可能是 `vllm bench` 客户端。现有日志区分不了。

## 11-16：进入引擎的速度有上限，约等于单线程分词速度（日志实测 + 源码）

`tests/bench_dev/engine_stats_per_run.py` 对每次运行取 serve.log 引擎日志的平均（规则同上），输出在各
结果目录的 `analysis/engine_stats_<config>.txt`。vllm-prefix（只用 vLLM 自己的 GPU 前缀缓存，不从 DDR
加载 KV），hitrate 95：

| 输入 | 并发 | 每秒完成请求数 | 每秒进入引擎的 prompt token | 引擎里（运行 + 等待） | 引擎外 | TTFT 均值 (ms) | TPOT 均值 (ms) |
|---|---|---|---|---|---|---|---|
| 32000 | 128 | 11.58 | 370,656 | 124 + 0 | 4 | 1221 | 76.9 |
| 32000 | 256 | 11.88 | 380,288 | 129 + 0 | 127 | 10829 | 81.4 |
| 32000 | 300 | 11.96 | 382,720 | 126 + 0 | 174 | 14495 | 79.2 |
| 128000 | 64 | 2.61 | 334,592 | 6 + 0 | 58 | 21458 | 15.2 |
| 128000 | 80 | 2.62 | 336,000 | 6 + 0 | 74 | 27158 | 14.9 |
| 256000 | 32 | 1.31 | 334,592 | 5 + 0 | 27 | 20139 | 26.5 |
| 256000 | 40 | 1.32 | 337,152 | 5 + 0 | 35 | 25757 | 26.4 |

- 每秒完成请求数：summary.csv 的 `request_throughput_req_s`。
- 每秒进入引擎的 prompt token = 每秒完成请求数 × 输入长度（闭环稳定时，进入速度等于完成速度）。
- 引擎里：引擎日志 Running + Waiting 的平均（去掉开头 3 条、最后 2 条）。
- 引擎外 = 并发 − 引擎里：客户端已经发出、还没进引擎的请求数。

要点：

- 三种输入长度下，每秒进入引擎的 prompt token 都停在约 33.5 万–38 万；并发再加，只是引擎外的请求变多，
  引擎里几乎不等待，TPOT 只有 15–81 ms。瓶颈在请求进引擎之前。
- 不是 KV 加载：vllm-prefix 根本不从 DDR 加载 KV，现象一样。
- 源码：Python API server 在一个线程池里分词，线程数是 `--renderer-num-workers`，默认 1
  （vLLM 0.29.0 `vllm/renderers/base.py` 第 91–92 行）。
- 单线程分词速度（只用 CPU、1 个核，Qwen3.8-27B 的 tokenizer，文本和 bench random 数据集一样由随机
  token 解码得到，`tests/bench_dev/tokenizer_speed.py`）：

  | 机器 | 32k | 128k | 256k |
  |---|---|---|---|
  | b300 | 49.1 万 token/s | 33.6 万 | 29.6 万 |
  | 6000d | 49.3 万 | 35.7 万 | 32.6 万 |

  128k 的上限（33.5 万）和 b300 单线程分词速度（33.6 万）一致；32k 的上限（38 万）低于分词速度，差的
  部分推测是每个请求的其它开销；256k 的上限比分词速度高约 14%，没有解释。
- hitrate 80 时引擎是瓶颈：32k 并发 256 引擎里有 250 个请求，每秒进入 26.3 万 token。

其它配置在 hitrate 95 下的同一数据（每秒进入引擎的 prompt token，万 / 引擎外请求数）：

| 配置（注意力后端） | 32k 并发 256 | 32k 并发 300 | 128k 并发 64 | 128k 并发 80 | 256k 并发 32 | 256k 并发 40 |
|---|---|---|---|---|---|---|
| vllm-prefix（FlashInfer） | 38.0 / 127 | 38.3 / 174 | 33.5 / 58 | 33.6 / 74 | 33.5 / 27 | 33.7 / 35 |
| kvshrink（FlashInfer） | 31.6 / 90 | 30.6 / 144 | 26.6 / 55 | 29.4 / 69 | 29.8 / 24 | 29.2 / 32 |
| kvshrink（FLASH_ATTN） | 22.4 / 11 | 22.5 / 16 | 23.1 / 53 | 23.1 / 69 | 22.3 / 21 | 22.2 / 29 |
| lmcache（FLASH_ATTN） | 24.3 / 7 | 24.5 / 8 | 25.2 / 3 | 25.1 / 3 | 23.1 / 2 | 23.2 / 4 |

数据：vllm-prefix `Qwen3.8-27B_tp4_20261111_095943`，kvshrink FlashInfer `..._20261112_091700`，
kvshrink FLASH_ATTN `..._20261112_130640_flash_attn`，lmcache `..._20261113_014438_flash_attn`。

- kvshrink 的上限比 vllm-prefix 低（FlashInfer 下 26.6 万–31.6 万 vs 33.5 万–38.3 万），引擎外照样堆积
  24–144 个请求；kvshrink 运行里引擎内等待（含等 KV 加载）只有 0–4 个，所以也不是 KV 加载慢。
- 同为 FLASH_ATTN，128k/256k 下 lmcache 每秒进入 23 万–25 万 token、引擎外只有 2–4 个，kvshrink 进入
  22 万–23 万、引擎外 21–69 个。kvshrink 为什么进得更慢没查（推测：kvshrink 的 CPU 线程和 API server
  的分词线程抢 CPU，未验证）。

## 11-16 验证一：HF 分词能不能多线程并行（6000d，只用 CPU）

`--renderer-num-workers` 只有在多个线程能同时分词时才有用；如果分词时一直占着 Python 全局锁（GIL），
多线程也只能轮流跑。`tests/bench_dev/tokenizer_thread_scaling.py`：同一段 12.8 万 token 的随机文本，
N 个线程同时分词，每个线程 4 次，8 个 CPU 核，每行跑 1 次。tokenizers 0.23.2（b300 容器同版本）。

| 线程数 | 总分词 token 数 | 墙钟时间 (s) | 总吞吐 (token/s) | 相对 1 线程 |
|---|---|---|---|---|
| 1 | 547,616 | 1.61 | 340,614 | 1.0× |
| 2 | 1,095,232 | 1.65 | 664,106 | 1.9× |
| 4 | 2,190,464 | 1.79 | 1,220,460 | 3.6× |
| 8 | 4,380,928 | 2.02 | 2,169,006 | 6.4× |

每次分词得到 136,904 个 token（随机 token 解码后重新分词会多约 7%）。结论：分词时会释放 GIL，多线程能
并行。

## 11-16 验证二：b300 上换前端（GPU 实测）

- 问题：把分词线程加到 8 个，或换 Rust 前端（`VLLM_USE_RUST_FRONTEND=1`，Rust 程序 `vllm-rs` 代替
  Python API server，引擎和 GPU worker 不变），排队能不能消掉。
- 运行：vllm-prefix，Qwen3.8-27B，TP=4（GPU 0/2/4/6），FlashInfer，输入 32000、hitrate 95、并发 256、
  1024 条请求（并发 × 4），输出 128。种子和 5 个热身请求同 11-11，不做预热（预热用的种子不同，前缀也不同，
  起不到预热作用；正式运行的测试请求和热身请求已经把前缀放进 GPU 缓存）。
- 4 组只换前端，每组单独启动一次 vLLM，各跑一次：
  - python：默认，1 个分词线程；
  - python-workers8：`--renderer-num-workers 8`；
  - rust：Rust 前端，请求线程数用默认值 min(CPU 数, 32) = 32；
  - rust-threads8：Rust 前端，`VLLM_RS_REQUEST_WORKER_THREADS=8`。
- 4 组都加了 `--chat-template tests/bench_dev/qwen38-27B-chat-template-no-generation.jinja`：模型自带的
  对话模板里有 `{% generation %}`，vllm-rs 不认识，启动即退出；这份是删掉那两行的副本。bench 发的是
  `/v1/completions`，不用对话模板，不影响结果。
- 运行期间 `tests/bench_dev/cpu_thread_sampler.py` 每秒记录各进程每个线程的 CPU%。

| 组 | 总耗时 (s) | 每秒完成请求数 | TTFT 均值 / 中位数 / p95 (ms) | TPOT 均值 (ms) | 引擎每请求收到的 token | 引擎每请求实际计算的 token | 前端进程 CPU 均值 / 最大 (%) | 最忙的分词线程 CPU (%) |
|---|---|---|---|---|---|---|---|---|
| python | 69.8 | 14.68 | 6069 / 5011 / 14286 | 86.1 | 31,969 | 2,253 | 127 / 141 | 93 |
| python-workers8 | 50.6 | 20.23 | 3925 / 3315 / 10294 | 68.1 | 31,969 | 2,253 | 260 / 803 | 8 个线程，各约 28 |
| rust | 40.7 | 25.18 | 2237 / 1345 / 7747 | 61.8 | 30,679 | 1,745 | 19 / 109 | 都 < 1 |
| rust-threads8 | 40.8 | 25.11 | 2239 / 1345 / 7791 | 62.0 | 30,679 | 1,745 | 18 / 93 | 都 < 1 |

- 总耗时：第一条请求发出到最后一条完成。TTFT、TPOT 的均值 / 中位数 / p95 都对 1024 条请求计算。
- 引擎每请求收到的 token：vLLM 前缀缓存查询计数 ÷ 1030（1024 条，加 bench 的 1 个测试请求和 5 个
  热身请求）；实际计算的 token = 收到的 − 前缀缓存命中的，即 GPU 真正要算的部分。
- CPU%：以一个核为 100%，只算正式计时那段时间。前端进程在 python 组是 API server，在 rust 组是 vllm-rs。
- 引擎外排队（引擎日志平均）：python 88 个、引擎内等待 0（7 条日志）；python-workers8 11 个、引擎内
  等待 52（5 条）。rust 两组每组只有 4–5 条日志，平均值不可靠，不列。

结论（实测）：

1. Python 默认前端被分词线程卡住：那个线程占 93% 的一个核，88 个请求堵在引擎外，引擎里不等待。
2. 只加 `--renderer-num-workers 8`：分词分到 8 个线程，引擎外只剩 11 个，引擎里开始有 52 个等待，瓶颈
   回到引擎。每秒完成请求数 +38%，TTFT 均值 6.07 s → 3.92 s。这两组发给引擎的 token 完全相同，对比
   干净。
3. Rust 前端整个进程平均只用约 0.2 个核；8 个和 32 个请求线程结果相差 < 0.5%，对 Rust 来说分词不是
   瓶颈。`--renderer-num-workers` 是 Python 前端的参数，Rust 前端不用。

### 问题：同一段文本，Rust 前端分出的 token 少 4%

- 实测：引擎每请求收到的 token，Rust 30,679、Python 31,969；GPU 每请求实际计算的 token 少 23%（1,745 vs
  2,253）。所以 Rust 比 python-workers8 快，一部分是 GPU 的活少了，这次的数据分不开两部分。
- 推测（未验证）：Rust 前端默认用另一个分词库 fastokens（不是 HF tokenizers），对这种随机 token 拼出来
  的文本切分结果不同。如果正常文本也不同，就是正确性问题：用 Rust 前端时模型看到的输入和 Python 不同。
  6000d 上 Qwen2.5-32B 的 Rust/Python 对比两边命中率完全一样（`doc/vllm-python-rust-frontend.md`），
  那个模型没有这个差异。
- 待做：容器里以不连引擎的方式启动 `vllm-rs render`，把 bench 的第 1 条 prompt 和一段正常中英文文本
  的 token id 与 HF tokenizer 逐个比对（只用 CPU）。

### Rust 前端在原设置下的两个点（并发 × 10）

`bench-results/Qwen3.8-27B_tp4_frontend_20261116_080613/rust`，和 11-11 的 Python 对照（同种子、同请求
数，但不是同一次运行，且 11-11 有预热、Rust 这次的 32k 预热并发是 256 不是 300）：

| 点 | 前端 | 总耗时 (s) | 每秒完成请求数 | TTFT 均值 (ms) | TPOT 均值 (ms) |
|---|---|---|---|---|---|
| 32k，并发 256，2560 条 | Python（11-11） | 215.4 | 11.9 | 10829 | 81.4 |
| 32k，并发 256，2560 条 | Rust | 98.1 | 26.1 | 1673 | 63.8 |
| 128k，并发 64，640 条 | Python（11-11） | 244.9 | 2.6 | 21458 | 15.2 |
| 128k，并发 64，640 条 | Rust | 121.1 | 5.3 | 2488 | 75.5 |

128k 的 TPOT 从 15 ms 升到 75 ms：前端不再卡后，进入引擎的请求变多，GPU 更忙。

## 未解释 / 局限

- kvshrink hitrate 95 的进入上限比 vllm-prefix 低，原因没查。
- 256k 的进入上限比单线程分词速度高约 14%，没有解释。
- 4 组前端实验里 EngineCore 进程的主线程都占约 96–97% 的一个核。只看 CPU% 分不清是在调度还是在空转
  等待，所以它会不会成为下一个上限还不知道。
- 前端实验每组只跑了一次，每次 40–70 s；引擎日志每 10 s 一条，引擎内外的排队数是粗略值。
- 结果报告里除 TTFT 外要同时列每秒完成请求数和 TPOT，只看 TTFT 会误导。

## 数据与脚本

| 内容 | 位置 |
|---|---|
| 前端实验（4 组，32k 并发 256） | `bench-results/Qwen3.8-27B_tp4_frontend_20261116_083551/`，汇总 `analysis/frontend_run_summary.txt` |
| Rust 原设置两个点 | `bench-results/Qwen3.8-27B_tp4_frontend_20261116_080613/rust/` |
| 两次中断的运行（无有效数据） | `..._frontend_20261116_075654`（ACS 检查未过）、`..._frontend_20261116_075948`（Rust 对话模板报错） |
| 前端实验脚本 | `test-qwen38-27B-frontend.sh`（`kvshrink-benchmark.sh` 新增 `PREWARM=0`） |
| 分析脚本 | `tests/bench_dev/engine_stats_per_run.py`、`frontend_run_summary.py`、`cpu_thread_sampler.py`、`tokenizer_speed.py`、`tokenizer_thread_scaling.py` |

注意：11-16 06:39 运行的 `tests/bench_dev/fix_nvls_fabric_state.sh` 对 8 张 GPU 做了重置，ACS 随之恢复开启，
测试脚本的主机设置检查因此失败；已按 PERFORMANCE_MODE_SETUP.md 用 `sudo bash tools/disable-acs.sh --apply`
关闭（记录在 b300 的 `tests/bench_dev/logs/disable-acs-*.log`）。以后再跑这个修复脚本，需要重新关 ACS。
