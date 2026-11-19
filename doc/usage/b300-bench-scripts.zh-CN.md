# B300 基准测试脚本使用说明（2026-11）

分支 `benchmark/b300-2026-11` 只做备份，不合入 main。它保存了 2026-11 在 B300 上
（Folsom 实验室的 aquila，8 张 NVIDIA B300 SXM6 AC）做测试的三样东西：

- 测试脚本；
- 脚本依赖的代码改动；
- 这次测试的结果日志。

目的：换一台新机器时，照本文准备好环境，就能直接用这套脚本重做测试。

## 1. 分支里有什么

| 位置 | 内容 | 在本分支能直接跑吗 |
|---|---|---|
| 根目录 `test-*.sh`、`kvshrink-benchmark.sh`、`night-kimi-then-qwen.sh` | Qwen 的大测试脚本（第 4 节） | 能 |
| `examples/`、`tools/`、`setvars.sh`、`iaxl/utils/affinity.py`、`kvshrink/kvshrink_connector.py` | 大脚本依赖的 serve 脚本、主机工具和代码改动（第 3 节） | 能 |
| `tests/bench_dev/` | 小脚本：探测、正确性测试、拷回结果、分析（第 7 节） | 能 |
| `kimi-k3/` | Kimi-K3 的脚本、文档和结果，从 `intel-accel-for-llm-kimi-k3` 仓库拷来，目录结构与那个仓库相同 | 不能，见第 8 节 |
| `lmcache-image/` | LMCache 对照组所用镜像的构建脚本和 Dockerfile（第 2.6 节） | 要放回 LMCache 仓库才能跑 |
| `bench-results/`、`tests/bench_dev/results/` | 这次的结果（第 9 节） | — |

各列含义：
- 位置：文件或目录在本分支里的路径。
- 内容：它是什么。
- 在本分支能直接跑吗：不切别的分支、不挪文件，能否照本文的命令运行。

几点说明：
- **每个结果目录里都存有当次运行用的脚本副本**，例如 `test-qwen38-27B-rust.sh`、
  `kvshrink-benchmark.sh`、`git_diff.patch`。要复现某一次运行，以目录里的副本为准；
  根目录的脚本是最后一次运行时的版本。
- 本分支**不含** vLLM 的 `serve.log` 和 LMCache 的 `lmcache_server.log`：两者共约
  527 MB，单个最大 147 MB，超过 GitHub 单文件 100 MB 的上限。它们留在 6000d 本机
  `/home/pese/qiuyu/B300/intel-accel-for-llm-b300/bench-results/` 和
  `/home/pese/qiuyu/B300/intel-accel-for-llm-kimi-k3/` 下。
- 本分支也**不含** B300 的远程登录方式，因为它涉及内网地址和 VPN。

## 2. 新机器上的准备

### 2.1 主机性能模式和 PCIe ACS

照 [PERFORMANCE_MODE_SETUP.md](../../PERFORMANCE_MODE_SETUP.md) 设置。**每次重启后都要重做**，
因为 ACS 每次开机都会重新打开。

大脚本开始测试前会调用 `check_host_settings`（在 `tools/bench-host-setup.sh` 里），
不满足要求就直接退出。它检查两样：

- perfspect 报告的性能模式；
- `tools/check-acs.sh`：GPU 上方的 PCIe 交换口 ACS 必须关闭。

perfspect 不在本仓库里。它要放在 `PERFSPECT_DIR`（默认 `/home/intel/qiuyu/perfspect`），
并且要能免密 `sudo -n` 运行。

### 2.2 加速器

| 设备 | 命令 | 说明 |
|---|---|---|
| QAT | `tools/setup_qat.sh` | 本分支把 `[SSL] NumProcesses` 从 4 改成 8，因为 TP8 的 8 个 rank 共用 QAT |
| IAA | `tools/setup_iaa.sh` | — |
| DSA | `tools/setup_dsa_cnt.sh 1` 或 `2` | 参数是每个 DSA 设备上的 WQ 数。Qwen TP4：4 个 rank 对 4 个 DSA 设备，用 1。Kimi-K3 TP8：8 个 rank 对 4 个设备，用 2 |

各列含义：
- 设备：加速器种类。
- 命令：在宿主机上运行的设置命令。各脚本的具体参数见脚本开头注释。
- 说明：这次在 B300 上为什么这样设。

### 2.3 KVShrink 容器（setvars.sh + start.sh）

kvshrink、vllm-prefix、vllm-wo-prefix 这三个 config 在 KVShrink 容器里跑，
大脚本通过 `docker exec` 进容器执行。容器不存在时，大脚本里的
`ensure_kvshrink_container` 会在一个后台 tmux 会话里运行 `./start.sh` 启动容器，
并等到容器里的 `pip install -e .` 完成。

本分支的 `setvars.sh` 是 B300 上实际用的版本，与 main 的区别如下：

| 项 | main | 本分支（B300） | 新机器上要改吗 |
|---|---|---|---|
| `IAXL_DEV_DOCKER_IMAGE` | `vllm-iaxl-dev:$IAXL_VERSION` | `vllm-iaxl-dev-fddeef0-qiuyu:$IAXL_VERSION` | 按需改成自己的镜像名 |
| `CONTAINER_NAME` | `${CONTAINER_NAME:-iaxl.vllm}` | `iaxl.vllm-fddeef0-qiuyu`（写死） | 按需改；大脚本用 `sed -n 's/^CONTAINER_NAME=//p' setvars.sh` 读这一行，所以必须保持 `CONTAINER_NAME=<名字>` 这种写法 |
| `IAXL_IAA_ZIP_ENABLE` 默认值 | 0 | 1 | 否 |
| `IAXL_DSA_GD_ENABLE` 默认值 | 0 | 1 | 否 |
| `TP_SIZE` 默认值 | auto_config 推出的值 | 8 | 否（Qwen 脚本会显式传 `TP_SIZE=4`） |
| 模型目录挂载 | 只挂 `$MODEL` | 另外整个挂载 `MODEL_SPACE=/home/intel/model-space` | 改成新机器的模型目录 |

各列含义：
- 项：`setvars.sh` 里的变量或行为。
- main：上游 main 的写法。
- 本分支（B300）：这次测试用的写法。
- 新机器上要改吗：换机器时是否必须改。

还有两处代码改动是脚本必需的：

- `iaxl/utils/affinity.py`：把 `KVSHRINK_IAA_DEVICES` 映射成每个 rank 的
  `IAXL_IAA_DEVICES`。Qwen 脚本设置了 `KVSHRINK_IAA_DEVICES='0|1|2|3'`；没有这个改动，
  IAA 不会按 rank 绑定。
- `kvshrink/kvshrink_connector.py`：与上游 #67（f4df091，"stop prefill at the
  shared-prefix boundary on hybrid models"）的改动相同。本分支基于 bff83ae，那时
  还没有 #67。

### 2.4 模型

脚本里的模型路径是写死的：

```text
/home/intel/model-space/Qwen3.8-27B            test-qwen38-27B*.sh
/home/intel/model-space/Qwen3-32B              test-qwen3-32B.sh
/home/intel/model-space/Qwen2.5-72B-Instruct   test-qwen25-72B.sh
/home/intel/model-space/Kimi-K3                kimi-k3/test-kimi-k3.sh
```

### 2.5 GPU / NUMA / 加速器绑定

B300 的拓扑记录在 [b300-topo.md](../../b300-topo.md) 和
[B300-ACCELERATORS.zh-CN.md](../../B300-ACCELERATORS.zh-CN.md)：每个 NUMA 节点有
2 张 GPU、1 个 QAT、1 个 IAA、1 个 DSA。

Qwen TP4 脚本每个 NUMA 节点只用 1 张 GPU，绑定写在 `ACCEL_ENV` 里
（以 `test-qwen38-27B-rust.sh` 为例）：

```bash
ACCEL_ENV="TP_SIZE=4 CUDA_VISIBLE_DEVICES=0,2,4,6"
ACCEL_ENV+=" VLLM_CPU_OMP_THREADS_BIND='0-23|24-47|48-71|72-95'"
ACCEL_ENV+=" KVSHRINK_QAT_DEVICES='0|1|2|3' KVSHRINK_DSA_DEVICES='wq0.0|wq2.0|wq4.0|wq6.0'"
ACCEL_ENV+=" KVSHRINK_IAA_DEVICES='0|1|2|3'"
```

`|` 分隔的第 i 项分给第 i 个 rank。换机器时，要按新机器的拓扑改这几行，
让每个 rank 的 GPU、CPU 和加速器都在同一个 NUMA 节点上。

`tests/bench_dev/tensor_xfer_runs_b300.sh` 里还写死了容器名 `iaxl.vllm-fddeef0-qiuyu`，
换机器时一并修改。

### 2.6 LMCache 镜像

lmcache 这个 config 不在 KVShrink 容器里跑，而是用单独的 LMCache 镜像：

| 镜像 | 基础镜像 | 用到它的脚本 | 构建文件 |
|---|---|---|---|
| `qiuyu/vllm-openai-lmcache:v0.29.0-lmcache0.5.7` | `vllm/vllm-openai:v0.29.0` | `test-qwen38-27B.sh`、`test-qwen38-27B-rust.sh`、`test-qwen3-32B.sh`、`kimi-k3/test-kimi-k3.sh`、`kimi-k3/tests/bench_dev/kimi_k3_lmcache.sh` | `lmcache-image/build_vllm_0.29.0_image.sh`、`lmcache-image/Dockerfile.vllm-0.29.0` |
| `qiuyu/vllm-openai-lmcache:v0.33.0-lmcache0.5.6`（旧，11-04） | `vllm/vllm-openai:v0.33.0` | `test-qwen25-72B.sh` | `lmcache-image/build_vllm_0.33.0_image.sh`、`lmcache-image/Dockerfile.vllm-0.33.0` |

各列含义：
- 镜像：docker 镜像名。
- 基础镜像：Dockerfile 的 `FROM`。
- 用到它的脚本：哪些脚本里写了这个镜像名。
- 构建文件：本分支里对应的构建脚本和 Dockerfile。

两个镜像都把 LMCache 用基础镜像里的 torch 重新编译，并装上
`lmcache-image/lmcache_hybrid_patch`。它是一个 vLLM 插件，移植自
intel-accel-for-llm PR #41，作用是让 LMCacheMPConnector 在关闭 prefix caching 时
也能跑混合模型。

构建脚本用 `LMCACHE_REPO=$SCRIPT_DIR/../..` 找 LMCache 仓库，再从中
`git archive v0.5.7`（或 `v0.5.6`）取源码。所以 `lmcache-image/` 不能在本仓库里
直接运行，要先把整个目录的内容放回一个 LMCache 仓库里、仓库根目录往下两层的位置。
这次用的是 `/home/pese/qiuyu/LMCache_v0.5.6/qiuyu_exp/qwen38_hybrid/`，
那个仓库里要有对应的 tag。在 B300 上构建的命令（摘自构建脚本开头注释，
通过 ssh 用 b300 的 docker 构建）：

```bash
cd <LMCache 仓库>/qiuyu_exp/qwen38_hybrid
env -u http_proxy -u https_proxy DOCKER_HOST=ssh://b300 \
    CUDA_ARCH=10.3 bash build_vllm_0.29.0_image.sh
```

`CUDA_ARCH` 默认是 12.0（RTX 6000D），B300 是 10.3。

## 3. 脚本依赖关系

下面每一行表示"上一层的脚本会调用或读取它"。"外部"指本仓库以外的东西。

```text
kvshrink-benchmark.sh（在容器里跑；所有 Qwen 大脚本共用，一次调用测一个 config）
├── examples/kvshrink-vllm-serve.sh   kvshrink / vllm-prefix / vllm-wo-prefix
│   └── setvars.sh → iaxl/utils/affinity.py（按 rank 绑定 CPU、QAT、DSA、IAA）
├── examples/lmcache-vllm-serve.sh    lmcache / vllm-prefix-v033
└── 外部：vllm bench serve；http://localhost:18800/v1/cache/simple_status（读 KVShrink 统计）

test-qwen38-27B-rust.sh / test-qwen38-27B.sh / test-qwen3-32B.sh / test-qwen38-27B-frontend.sh（在宿主机跑）
├── setvars.sh（只读出 CONTAINER_NAME）
├── tools/bench-host-setup.sh（source 进来）
│   ├── tools/check-acs.sh
│   ├── ./start.sh（容器不存在时，在后台 tmux 会话里启动）
│   └── 外部：perfspect（PERFSPECT_DIR）
├── kvshrink-benchmark.sh（kvshrink 等：docker exec 进 KVShrink 容器；
│                          lmcache：docker run --rm 一个 LMCache 镜像的新容器）
├── tests/bench_dev/qwen38-27B-chat-template-no-generation.jinja（只有 -rust 和 -frontend 用）
└── tests/bench_dev/cpu_thread_sampler.py（只有 -frontend 用）

test-qwen25-72B.sh（宿主机）
├── setvars.sh、tools/check-acs.sh、./start.sh、外部 perfspect
│   （没有 source bench-host-setup.sh，而是在脚本里自带了一份较早的同名函数）
└── kvshrink-benchmark.sh

night-kimi-then-qwen.sh（宿主机，依次跑 Kimi-K3 和 Qwen）
├── ../intel-accel-for-llm-kimi-k3/tools/setup_dsa_cnt.sh 2   （要求旁边有 kimi 仓库）
├── ../intel-accel-for-llm-kimi-k3/test-kimi-k3.sh
├── 等 GPU 全部空闲后：setup_dsa_cnt.sh 1
└── test-qwen38-27B-rust.sh

tests/bench_dev/tensor_xfer_runs_b300.sh（宿主机）
└── benchmark/tensor_xfer/tensor_xfer_benchmark.sh（main 已有），在 KVShrink 容器里跑 1、4、8 个 rank
```

## 4. 大脚本

所有大脚本都在**宿主机的 tmux 会话里**运行，因为一次要跑几小时。都支持
`DRY_RUN=1`：只打印计划，不启动 vLLM，也不启动容器，输出写到
`tests/bench_dev/results/dryrun-<名字>/`。建议先 dry run 看一遍计划，再正式跑：

```bash
DRY_RUN=1 bash test-qwen38-27B-rust.sh
bash test-qwen38-27B-rust.sh
```

| 脚本 | 测什么 | 常用环境变量（默认值） | 结果目录 |
|---|---|---|---|
| `test-qwen38-27B-rust.sh` | Qwen3.8-27B，TP4（GPU0/2/4/6），Rust 前端。每个 setting 启动一次 vLLM：kvshrink-zip-dsa、lmcache、vllm-prefix、kvshrink-nozip-dsa、kvshrink-nozip-nodsa、vllm-wo-prefix、kvshrink-zip-dsa-flash-attn；不在默认列表里的还有 `-omp8`、`-omp12` 变体 | `SETTINGS`（上面 7 个）、`HIT_RATES`（`80 95`）、`LEN_CONCURRENCY`（`32000:32,64,128,256 128000:80 256000:40`）、`SCRATCH_POOL_SIZE_GB`（8）、`RUN_NAME` | `bench-results/Qwen3.8-27B_tp4_rust_<时间>/<setting>/` |
| `test-qwen38-27B.sh` | 同上，但用 Python 前端，config 为 kvshrink、vllm-prefix、vllm-wo-prefix、lmcache | `CONFIGS`、`HIT_RATES`（`80 95`）、`LEN_CONCURRENCY`（`32000:32,64,128,256,300 128000:64,80 256000:32,40`）、`ATTENTION_BACKEND` | `bench-results/Qwen3.8-27B_tp4_<时间>/` |
| `test-qwen38-27B-frontend.sh` | 命中率 95 时 API 前端是否成为瓶颈：只跑 vllm-prefix，输入 32000、并发 256，比较 4 种前端 | `VARIANTS`（`rust python rust-threads8 python-workers8`） | `bench-results/Qwen3.8-27B_tp4_frontend_<时间>/<variant>/` |
| `test-qwen3-32B.sh` | Qwen3-32B，TP8，block size 16 和 256 | `CONFIGS`、`HIT_RATES`（`80 100`） | `bench-results/Qwen3-32B_tp8_<时间>/bs16`、`/bs256` |
| `test-qwen25-72B.sh` | Qwen2.5-72B-Instruct，TP8，block size 16 和 256 | `CONFIGS`（`kvshrink vllm-prefix vllm-wo-prefix`） | `bench-results/Qwen2.5-72B-Instruct_tp8_<时间>/` |
| `night-kimi-then-qwen.sh` | 11-17 夜间：DSA 设成 2 个 WQ → Kimi-K3 TP8 → GPU 空闲后 DSA 改回 1 个 WQ → Qwen3.8-27B Rust 前端的 3 个 `-omp8` setting | `DRY_RUN` | 日志 `tests/bench_dev/logs/night-<时间>.log`；结果在两个仓库各自的 `bench-results/` 下 |
| `tests/bench_dev/tensor_xfer_runs_b300.sh` | 主机与 GPU 之间的拷贝带宽：4 种方法、两个方向，分别在 1、4、8 个 rank 下测 | `RUN_NAME`、`RANKS_LIST`（`1 4 8`） | `bench-results/tensor_xfer_<时间>/ranks<N>/` |

各列含义：
- 脚本：在仓库根目录运行的命令名。
- 测什么：模型、并行方式、比较哪些 config。
- 常用环境变量：可以在命令前覆盖的变量，括号里是默认值；完整列表见各脚本开头注释。
- 结果目录：结果写到哪里。

术语：
- **config**：一种 KV 复用方式，即 kvshrink、lmcache、vllm-prefix（vLLM 自带 GPU
  prefix cache）、vllm-wo-prefix（不复用）。
- **setting**：config 再加上开关组合。例如 `kvshrink-nozip-dsa` 表示 kvshrink、
  不压缩、用 DSA 传输。
- **一次测量**：一次 `vllm bench serve`，输入长度、并发、命中率各取一个值；
  请求数为并发 × 10（vllm-wo-prefix 为 × 2）。
- 不要把并发设得超过 vLLM 在该输入长度下报告的 "Maximum concurrency"，先用
  `tests/bench_dev/probe-max-concurrency-qwen38-b300.sh` 测出来。

注意：B300 上这次**只正式跑了 Qwen3.8-27B 和 Kimi-K3**。`test-qwen25-72B.sh` 只做过
11-10 的 dry run，`test-qwen3-32B.sh` 在 B300 上没有跑过。

`kvshrink-benchmark.sh` 也可以不经过大脚本、直接在容器里跑，用来单独测一个 config
（摘自它的开头注释）：

```bash
CONFIGS="kvshrink vllm-prefix vllm-wo-prefix" RESULTS_DIR=bench-results/<name> \
    TP_SIZE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    LEN_CONCURRENCY="100000:8,16,32 200000:8,16" bash kvshrink-benchmark.sh
```

## 5. 结果目录和完成标记

以 `test-qwen38-27B-rust.sh` 的结果为例：

```text
bench-results/Qwen3.8-27B_tp4_rust_<时间>/
├── test-qwen38-27B-rust.sh、*.jinja      当次运行用的脚本副本
├── test-qwen38-27B-rust.log              大脚本自己的日志
├── <setting>/
│   ├── summary.csv                       每次测量一行
│   ├── run.log                           kvshrink-benchmark.sh 的日志
│   ├── <config>/serve.log                vLLM 日志（本分支不含）
│   ├── <config>/*.json                   每次测量 vllm bench serve 的原始结果
│   └── SETTING_DONE                      这个 setting 跑完：退出码和时间
├── analysis/                             事后分析的输出
└── ALL_DONE                              所有 setting 跑完
```

Kimi-K3 的结果用 `CONFIG_DONE/<layout>-<config>` 代替 `SETTING_DONE`。

`summary.csv` 常用列：

| 列 | 含义 |
|---|---|
| `config`、`tp_size`、`block_size_tokens` | config、TP 大小、vLLM block size（单位：token） |
| `input_len_tokens`、`output_len_tokens` | 每个请求的输入、输出 token 数 |
| `max_concurrency`、`hit_rate_pct`、`num_prompts` | 并发、目标命中率（%）、请求数 |
| `request_throughput_req_s` | 每秒完成的请求数 |
| `mean_ttft_ms`、`median_ttft_ms`、`p95_ttft_ms` | 首 token 时间（ms）的平均值、中位数、P95 |
| `mean_tpot_ms`、`median_tpot_ms`、`p95_tpot_ms` | 第 2 个 token 起每个输出 token 的时间（ms） |
| `preemptions` | 这次测量中 vLLM 抢占请求的次数 |
| `external_prefix_cache_hit_tokens` | vLLM 统计到的、从外部（KVShrink 或 LMCache）命中的 token 数 |
| `kvshrink_hit_requests`、`kvshrink_mean_hit_tokens` | KVShrink 命中的请求数、平均每个命中请求的命中 token 数 |
| `compression_ratio`、`ddr_used_gb_after` | KVShrink 压缩比、测量后 DDR 池已用 GB |

各列含义：
- 列：`summary.csv` 的列名。
- 含义：这列记录的是什么、单位是什么。

## 6. 把结果拷回 6000d，再做分析

B300 连不到 6000d，所以测试脚本不能主动推送结果，只能在 6000d 上拉。
在 6000d 的 tmux 会话里运行（要求 `ssh b300` 能连上）：

```bash
bash tests/bench_dev/pull-b300-bench-results.sh <结果目录名>          # Qwen
bash kimi-k3/tests/bench_dev/pull-b300-kimi-results.sh <结果目录名>  # Kimi-K3（在 kimi 仓库里运行）
```

拉取脚本每 60 秒（`CHECK_INTERVAL_S`）用一次 ssh 查看有哪些完成标记
（`SETTING_DONE` / `CONFIG_DONE`）。Qwen 版每个跑完的 setting 只拷一次，拷完在本地写
`SYNCED`；Kimi 版则在每个 (layout, config) 跑完后把整个运行目录拷一次。
出现 `ALL_DONE` 且全部拷完后退出。远端路径是 `REMOTE_RESULTS`
（默认 `/home/intel/qiuyu/intel-accel-for-llm-b300/bench-results`），换机器时要改。

分析脚本只读日志和结果，不需要 GPU：

| 脚本 | 输入 | 输出 |
|---|---|---|
| `python3 tests/bench_dev/rust_run_tables.py bench-results/<目录> ...` | `test-qwen38-27B-rust.sh` 的结果 | 每个输入长度、命中率一张表，各 setting 并排 |
| `python3 tests/bench_dev/engine_stats_per_run.py bench-results/<目录> <config>` | 同上或 `test-qwen38-27B.sh` 的结果 | 每次测量期间 vLLM 的 Running/Waiting 请求数和 GPU KV 占用 |
| `python3 tests/bench_dev/hit_per_run.py bench-results/<目录>` | `test-qwen38-27B.sh` 的结果 | 每次测量中 kvshrink 每个请求实际命中的 token 数 |
| `python3 tests/bench_dev/lmcache_hit_per_run.py bench-results/<目录>` | 同上 | lmcache 每个请求的命中情况 |
| `python3 tests/bench_dev/frontend_run_summary.py <目录> [variant ...]` | `test-qwen38-27B-frontend.sh` 的结果 | 每种前端的汇总 |
| `kimi-k3/tests/bench_dev/analysis/*` | Kimi-K3 的结果 | 每次测量的命中值、TTFT 拆分 |

各列含义：
- 脚本：运行命令。
- 输入：哪种结果目录。
- 输出：打印什么。

只有 `rust_run_tables.py` 只读 `summary.csv`，在本分支的结果上也能跑。其余脚本都要读
`serve.log` 或 `lmcache_server.log`，本分支不含这两种日志，所以只能在 6000d 本机
保存了完整日志的目录上跑。

## 7. 小脚本（tests/bench_dev/）

| 脚本 | 用途 |
|---|---|
| `probe-max-concurrency.sh`、`probe-max-concurrency-qwen38-b300.sh` | 读 vLLM 报告的 GPU KV 容量和各输入长度下的最大并发，用来定 `LEN_CONCURRENCY` |
| `gsm8k_two_passes_b300.sh`、`_glm47_b300.sh`、`_minimax_b300.sh`、`gsm8k_compare.py` | 正确性测试：在同一个 kvshrink vLLM 上把 gsm8k 跑两遍（第 2 遍从 KVShrink 读第 1 遍存下的 KV），再比较两遍的答案 |
| `rust_kvshrink_smoke.sh`、`rust_kvshrink_smoke_client.py` | kvshrink 与 Rust 前端能否一起工作：同一个约 4000 token 的请求发两次，检查第二次命中且答案正确 |
| `kimi_k3_smoke.sh`、`kimi_k3_smoke_client.py` | Kimi-K3 冒烟测试的早期版本（bf16 模型），最终版本在 `kimi-k3/tests/bench_dev/` |
| `tensor_xfer_runs_b300.sh` | 见第 4 节 |
| `raw_copy_bench.c/.sh` | 不压缩时，主机上拷贝 KV block 的吞吐量（1、4、12、20 个 OpenMP 线程） |
| `gdr_handle_overflow_check.c/.sh` | 复现 B300 上 "dsa batch failed ... status=0x84" 崩溃 |
| `fix_nvls_fabric_state.sh` | nvidia-fabricmanager 重启但 GPU 没有 reset 后，重新同步 NVLink fabric 状态 |
| `nccl_allreduce_check.py` | 最小的 NCCL all-reduce 检查（torchrun） |
| `cpu_thread_sampler.py` | 每秒采样一次 vLLM 各线程的 CPU 占用（`-frontend` 脚本使用） |
| `tokenizer_speed.py`、`tokenizer_thread_scaling.py` | tokenizer 单线程速度，以及多线程能否提速 |
| `qwen38-27B-chat-template-no-generation.jinja` | 删掉 `{% generation %}` 的 chat template；没有它，vllm-rs 无法启动 |
| `recipes/moonshotai_Kimi-K3.json` | Kimi-K3 的 vLLM recipe |

各列含义：
- 脚本：`tests/bench_dev/` 下的文件。
- 用途：它解决什么问题；用法见各文件开头注释。

## 8. Kimi-K3（kimi-k3/）

`kimi-k3/` 里的文件来自 `intel-accel-for-llm-kimi-k3` 仓库的工作区，是 11-17 实际运行时
的版本，其中一部分在那个仓库里还没有 commit。目录结构与那个仓库相同，
例如 `kimi-k3/test-kimi-k3.sh` 对应那个仓库的 `test-kimi-k3.sh`。

在本分支里不能直接跑，有两个原因：

1. 需要 fork 上 `kimi-k3-support-dev` 分支的代码（dd14303：Kimi-K3 的 kvshrink 支持、
   MLA 修复、`kvshrink/vllm_patch.py` 等），这些代码还没有合入 main。
2. `kimi-k3/kvshrink-benchmark.sh`、`examples/*.sh`、`setvars.sh`、`tools/auto_config.sh`
   和本分支根目录的同名文件不同：Kimi 版多了 `DP_SIZE`、`KEEP_SERVE`、按 DP rank
   热身等功能。

运行方法：check out `kimi-k3-support-dev`，把 `kimi-k3/` 下的文件拷回仓库根目录的
对应位置，然后：

```bash
bash tools/setup_dsa_cnt.sh 2     # TP8：每个 DSA 设备 2 个 WQ
DRY_RUN=1 bash test-kimi-k3.sh
bash test-kimi-k3.sh              # 默认 LAYOUTS=tp8，CONFIGS="kvshrink kvshrink-nozip"
```

lmcache 这个 config 会 `docker exec` 进一个常驻的 LMCache 容器
`lmcache.vllm-kimi-k3-qiuyu`；这个容器的创建命令由 `test-kimi-k3.sh` 打印出来。
部署细节见 [kimi-k3/doc/design/kimi-k3-deployment.zh-CN.md](../../kimi-k3/doc/design/kimi-k3-deployment.zh-CN.md)。

## 9. 这次的结果

结论和数据表在 [kimi-k3/doc/design/b300-bench-results.zh-CN.md](../../kimi-k3/doc/design/b300-bench-results.zh-CN.md)，
它同时覆盖 Qwen3.8-27B 和 Kimi-K3。其他相关文档：

- [doc/design/bench-ttft-hit95-queue.zh-CN.md](../design/bench-ttft-hit95-queue.zh-CN.md)：命中率 95 时 TTFT 高的原因（请求排队）
- [doc/design/kvshrink-mamba-hit-rate-debug.zh-CN.md](../design/kvshrink-mamba-hit-rate-debug.zh-CN.md)：混合模型上 kvshrink 命中率的排查
- [doc/vllm-python-rust-frontend.md](../vllm-python-rust-frontend.md)：vLLM 的 Python 前端与 Rust 前端

结果目录索引：

| 目录 | 产生它的脚本 | 里面的 setting / config | 有 ALL_DONE |
|---|---|---|---|
| `bench-results/Qwen3.8-27B_tp4_20261111_034030` | `test-qwen38-27B.sh` | kvshrink、lmcache、vllm-prefix、vllm-wo-prefix | — |
| `bench-results/Qwen3.8-27B_tp4_20261111_080256`、`_094004` | `test-qwen38-27B.sh` | kvshrink | — |
| `bench-results/Qwen3.8-27B_tp4_20261111_091522` | `test-qwen38-27B.sh` | kvshrink、vllm-prefix、vllm-wo-prefix | — |
| `bench-results/Qwen3.8-27B_tp4_20261111_095943` | `test-qwen38-27B.sh` | kvshrink、lmcache、vllm-prefix、vllm-wo-prefix | — |
| `bench-results/Qwen3.8-27B_tp4_20261112_032448`、`_091700` | `test-qwen38-27B.sh` | kvshrink | — |
| `bench-results/Qwen3.8-27B_tp4_20261112_130640_flash_attn` | `test-qwen38-27B.sh` | kvshrink、lmcache（FLASH_ATTN） | — |
| `bench-results/Qwen3.8-27B_tp4_20261113_014438_flash_attn` | `test-qwen38-27B.sh` | lmcache（FLASH_ATTN） | — |
| `bench-results/Qwen3.8-27B_tp4_frontend_20261116_*`（4 个） | `test-qwen38-27B-frontend.sh` | rust、python、rust-threads8、python-workers8（最后一个 `_083551` 四种都有） | — |
| `bench-results/Qwen3.8-27B_tp4_rust_20261116_100254` | `test-qwen38-27B-rust.sh` | 默认 7 个 setting | 有 |
| `bench-results/Qwen3.8-27B_tp4_rust_20261117_024301` | `test-qwen38-27B-rust.sh` | kvshrink-nozip-dsa-omp12 | 有 |
| `bench-results/Qwen3.8-27B_tp4_rust_20261117_030926` | `test-qwen38-27B-rust.sh` | kvshrink-nozip-dsa-omp12、kvshrink-nozip-nodsa-omp12 | 有 |
| `bench-results/Qwen3.8-27B_tp4_rust_omp8_scratch64g_20261117_092637` | `night-kimi-then-qwen.sh` → `test-qwen38-27B-rust.sh` | 3 个 `-omp8` setting，ScratchPool 64 GB/rank | 有 |
| `bench-results/tensor_xfer_20261118_015550` | `tensor_xfer_runs_b300.sh` | ranks1、ranks4 | — |
| `bench-results/tensor_xfer_20261118_025858` | `tensor_xfer_runs_b300.sh` | ranks1、ranks4、ranks8 | 有 |
| `kimi-k3/bench-results/Kimi-K3_20261113_095102` | `test-kimi-k3.sh` | dp2tp4、tp8 两种 layout | — |
| `kimi-k3/bench-results/Kimi-K3_tp8_omp8_scratch64g_20261117_092637` | `night-kimi-then-qwen.sh` → `test-kimi-k3.sh` | tp8：kvshrink、kvshrink-nozip | 有 |
| `kimi-k3/bench-results/dryrun-*`、`tests/bench_dev/results/dryrun-*` | 各大脚本的 `DRY_RUN=1` | 只有计划 | — |
| `tests/bench_dev/results/gsm8k-*`、`kimi-k3/tests/bench_dev/results/kimi-k3-gsm8k-*` | gsm8k 正确性测试 | — | — |
| `kimi-k3/tests/bench_dev/results/kimi-k3-*`（smoke、capacity、dp-layouts、lmcache） | `kimi-k3/tests/bench_dev/` 下同名脚本 | — | — |
| `tests/bench_dev/results/probe-*`、`raw-copy-bench-*`、`gdr-handle-overflow-*` | 同名小脚本 | — | — |

各列含义：
- 目录：结果目录（多个目录共用一行时，用 `_时间` 后缀列出其余的）。
- 产生它的脚本：由目录里保存的脚本副本确定。
- 里面的 setting / config：目录下有哪些子目录。
- 有 ALL_DONE：目录里是否有 `ALL_DONE` 标记。只有 `test-qwen38-27B-rust.sh`、
  `tensor_xfer_runs_b300.sh` 和 11-17 的 `test-kimi-k3.sh` 写这个标记；其他脚本
  不写，所以它们的目录显示"—"，不代表没跑完。

哪次运行是哪个结论的依据，以结果文档为准。
