# Rust frontend vs Python frontend 性能对比（Qwen2.5-32B-Instruct，2×L20）

日期：2026-11-09，机器：spr-inspur-18

标注约定：

- 【实测】：本次实验测得的数据。
- 【代码证据】：引用的源码位置。行号对应工作区的 vLLM 0.33.0 源码，实验用的是 0.29.0 镜像。
- 【我的推测】：推断，附置信度。

## 1. 结论

1. 在 Qwen2.5-32B-Instruct、TP2、2×L20 上，Rust frontend 和 Python frontend 的吞吐差异 < 1%【实测】。
   - 不开前缀缓存：一致。
   - 开前缀缓存、前缀全部命中：也一致。
2. Rust 的 TTFT 略低，差 1–6%，即几到十几 ms。每组只跑一次，没测运行之间的波动，不能确认是真实差异【我的推测：可能在噪声范围内，60%】。
3. frontend 不是瓶颈【实测】：
   - 前缀全部命中、输出 16 token 时，Python 主进程只占单核的 4–6%，`vllm-rs` 进程占 1–2%。
   - 端到端时延主要由 GPU 决定：TPOT 约 46–76 ms。
4. RFC #40846 里“Rust 快很多倍”的数据来自 0.6B 模型、4×GB200、DP=4、并发 1024。在那种配置下 frontend CPU 才是瓶颈，结果不适用于 32B 模型、2×L20、低并发的部署【我的推测，80%】。
5. 对当前部署来说，换 Rust frontend 没有明显收益，也不需要再换小模型复测。只有将来要部署小模型、做高请求率服务时才值得测。

## 2. 环境

| 项 | 值 |
|---|---|
| GPU | 2× NVIDIA L20（46 GB），拓扑 SYS（跨 NUMA，无 NVLink） |
| 驱动 | 580.173.02 |
| CPU / 内存 | 144 核 / 2 TB |
| 镜像 | `vllm-iaxl-dev-qiuyu-vllm-block-hashes-dev:1.5.1`，即 vLLM 0.29.0 + vllm-rs 0.29.0。vLLM 文件和 wheel RECORD 的哈希一致 |
| 模型 | `/media/model-space/Qwen/Qwen2.5-32B-Instruct` |
| 服务端参数 | `--tensor-parallel-size 2 --max-model-len 8192`，其余默认，GPU 显存利用率 0.92 |
| KV cache | 73,232 token【实测，服务端日志】 |
| 插件 | 镜像自带 `kvshrink` 插件，服务端和客户端都设 `VLLM_PLUGINS=""` 关掉 |

开启 Rust frontend：设 `VLLM_USE_RUST_FRONTEND=1`。

- 日志里会出现 "Launching Rust frontend"。
- 进程列表里会多一个 `vllm-rs frontend --listen-fd ...` 进程【实测】。

## 3. 相关代码事实

- Rust frontend 会强制 `api_server_count=1`。Python 在 DP=1 时默认也是 1，API server 跑在主进程里。所以两边对比的都是单个 frontend 进程【代码证据】vllm/entrypoints/cli/serve.py L113-136。
- 两边的 tokenizer 不同【代码证据 + 实测】：
  - Rust 默认用 `fastokens`（rust/src/tokenizer/src/hf.rs L8），日志里有 "loading tokenizer with fastokens"。
  - Python 用 HF tokenizers。
- 默认 `max_num_seqs` 和 `max_num_batched_tokens`【代码证据】vllm/engine/arg_utils.py L2997-L3006：
  - 单卡显存 < 70 GiB 时，在线服务默认 `max_num_seqs=256`、`max_num_batched_tokens=2048`。
  - L20 属于这一档，所以并发超过 256 会在引擎调度器里排队，不能用来测 frontend。
  - 这是 0.33 源码，0.29 大概率一样【我的推测，85%】。
- 前缀缓存命中上限【代码证据】vllm/v1/core/kv_cache_manager.py L296-L302：
  - 命中长度最多到 `num_tokens - 1`，因为最后一个 token 必须重新计算才能得到 logits。
  - 命中长度再按 block（16 token）向下取整。
  - 所以 2048 token 的 prompt 最多命中 2032 token（99.22%），1024 token 最多命中 1008 token（98.44%）。命中率达不到 100%。
- 两边的 `/metrics` 都有 `vllm:prefix_cache_queries_total` 和 `vllm:prefix_cache_hits_total`。日志里也都会打 "Prefix cache hit rate"【实测 + 代码证据】rust/src/llm/src/log_stats.rs。

## 4. 实验一：不开前缀缓存（baseline）

配置：

- 服务端：`--no-enable-prefix-caching`。
- 客户端：`vllm bench serve --backend vllm`（`/v1/completions`），random 数据集。
- 输入 2048、输出 256，`--ignore-eos`，`--num-warmups 2`，`--seed 0`。
- 两组：并发 1 / 20 条请求，并发 8 / 80 条请求。
- 每组跑一次。

结果【实测】（时延单位 ms）：

| frontend | 并发 | TTFT mean | TTFT P50 | TTFT P99 | TPOT mean | out tok/s | req/s |
|---|---|---|---|---|---|---|---|
| python | 1 | 1310.01 | 1310.10 | 1311.88 | 45.86 | 19.69 | 0.08 |
| rust | 1 | 1298.20 | 1298.10 | 1299.79 | 45.85 | 19.71 | 0.08 |
| python | 8 | 4400.66 | 5209.21 | 10507.45 | 76.22 | 85.85 | 0.34 |
| rust | 8 | 4380.41 | 5185.81 | 10454.13 | 76.09 | 86.04 | 0.34 |

所有指标两边相差都在 1% 以内。

## 5. 实验二：开前缀缓存，前缀全部命中

配置：

- 服务端：开 prefix caching（默认）。
- prompt 构造：共享前缀 (输入长度 − 16) token，加 16 token 的唯一尾部，即 `--random-prefix-len`，`--random-input-len 16`。
- 输出 16 token，`--ignore-eos`。输出短是为了减少 GPU 计算，突出 frontend 开销。
- 两组：输入 2048 / 并发 8 / 80 条请求，输入 1024 / 并发 16 / 160 条请求。
- 预热：每组先用完全相同的参数跑一遍，不计入结果，保证前缀已经进了缓存。之后再跑一遍计入结果。
- 命中率：计入那一遍前后各抓一次 `/metrics`，用计数器差值计算。
- frontend CPU：计入那一遍期间，用 `top -b -d 1` 每秒采样一次 frontend 进程。
  - Python：服务端容器的主进程 `vllm serve`。
  - Rust：`vllm-rs frontend` 进程。
  - 数值是单核百分比。前约 7 个采样是客户端容器启动，CPU 为 0，不计入均值。

结果【实测】（时延单位 ms，计入的那一遍运行时长约 10–11 s）：

| frontend | 输入 | 并发 | 完成 | 命中率 | FE CPU (%) | req/s | out tok/s | TTFT mean | TTFT P50 | TTFT P99 | TPOT mean | E2E mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| python | 2048 | 8 | 80/80 | 99.22% | 4–5 | 8.06 | 128.97 | 207.52 | 213.44 | 233.38 | 52.19 | 990.39 |
| rust | 2048 | 8 | 80/80 | 99.22% | 1–2 | 8.07 | 129.04 | 202.23 | 208.58 | 219.42 | 52.51 | 989.88 |
| python | 1024 | 16 | 160/160 | 98.44% | 4–6 | 14.35 | 229.54 | 269.63 | 275.36 | 296.56 | 56.25 | 1113.40 |
| rust | 1024 | 16 | 160/160 | 98.44% | 1–2 | 14.35 | 229.63 | 265.24 | 271.54 | 285.42 | 56.53 | 1113.19 |

解读：

- 命中率正好等于上限，前缀全部命中。
- 吞吐两边相差 < 0.1%。
- Rust 的 TTFT 均值低约 2%，P99 低 4–6%。TPOT 反而高约 0.6%。
- frontend CPU 两边都远没跑满。Rust 省下的只是每秒几个百分点的单核 CPU，转化不成吞吐。

## 6. 和 RFC #40846 的对比

RFC #40846（njhill，2026-04-25）的配置：

- Qwen3-0.6B，DP=4，4×GB200，vLLM 0.19，并发 1024。
- 原文自己说："obviously not a realistic real-world configuration"。

原文表头在抓取时丢失了，下面两列是根据正文文字确认的：

| 场景 | frontend | 吞吐 | P50 TTFT (ms) |
|---|---|---|---|
| in32/out512，不开前缀缓存 | Rust | 559.79 | 50.51 |
| | Python, api-server-count=4 | 509.56 | 165.95 |
| | Python, api-server-count=16 | 521.80 | 58.97 |
| 约 1 万条 chat prompt，前缀缓存已预热，out16 | Rust | 837 | 596.92 |
| | Python, api-server-count=4 | 162.23 | 6076.09 |
| | Python, api-server-count=32 | 785.98 | 657.15 |

- “Rust 快很多倍”来自和 Python 4 个 API server 的对比，第二个场景吞吐差约 5.2 倍。Python 把 API server 数量加大后，吞吐差距缩小到约 7%（559.79/521.80）和 6.5%（837/785.98）。
- 它的前提是模型很小、GPU 很快、请求率极高，frontend CPU 成为瓶颈。本机的 32B 模型每秒只有 8–14 个请求，没有这种瓶颈。

## 7. 踩过的坑

1. **主机 curl 访问 `/health` 走了 Intel 代理**：主机 curl 7.81 不支持 `no_proxy` 里的 CIDR（`127.0.0.0/8`），请求被转到代理。修复：用 `curl --noproxy '*'`。
2. **bench 客户端写不了结果目录**：镜像里 `/home/ubuntu` 属于 uid 1000、权限 0750，而客户端以 uid 1001 运行，没法建目录。修复：把结果目录挂载到 `/out`。
3. **客户端里 `kvshrink` 插件导入报错**：客户端也设 `VLLM_PLUGINS=""`。
4. **客户端里 `No module named vllm._C` 和 `Triton 0 active driver(s)` 警告**：客户端容器没有 GPU，不影响 bench。
5. **并发不能超过 `max_num_seqs`**（L20 默认 256）：超过的部分在引擎排队，测不到 frontend。

## 8. 复现

脚本和数据都在本目录：

| 文件 | 内容 |
|---|---|
| `run_bench.sh` + `summarize.py` | 实验一，结果在 `results/`，日志 `run.log` |
| `run_prefix.sh` + `summarize_prefix.py` | 实验二，结果在 `results_prefix/`，日志 `run_prefix.log` |

两个脚本都是：

- 依次用 Python frontend 和 Rust frontend 启动服务端容器 `rfbench-server`，每次启动约 1.5 分钟。
- 用另一个 `docker run --rm` 容器跑 `vllm bench serve`。
- 跑完自动删除服务端容器。

换模型只需改脚本里的 `MODEL`。运行方式：

```bash
cd rust_frontend_bench
tmux new-session -d -s rfbench "bash -c './run_prefix.sh 2>&1 | tee run_prefix.log; exec bash'"
```
