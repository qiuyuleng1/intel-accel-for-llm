# DSA 自动回退：校准稳定性实验方案

状态：方案已确认（2026-11-19），2026-11-20 已在 6000d 上运行，结果见第 10 节。2026-11-20 决定：校准采用设置 D（每次拷贝 128 MiB，热身 2 次 + 计时 10 次，取最快一次）。

## 1. 背景

- 功能：DSA 初始化时现场测一遍 DSA 和 `cudaMemcpy3DBatchAsync`（下文简称 CUDA batch）的拷贝带宽，连续块大小达到交界点的拷贝改走 CUDA batch。开关是隐藏环境变量 `IAXL_DSA_AUTO_FALLBACK_ENABLE`，默认开。
- 校准规则（计划）：从 256K 开始、步长 128K、最多到 4M，H2D 和 D2H 各自逐点测；第一个"CUDA batch 比 DSA 快"的片段大小就是该方向的交界点，测到就停；到 4M 还没有出现，就全部用 DSA。
- 还没定的两项：每次拷贝的总大小（暂定 64 MiB）、热身和计时次数（暂定热身 1 次 + 计时 3 次，取最快一次）。本实验就是用来定这两项。

## 2. 要回答的问题

每次拷贝的总大小取 64 MiB 或 128 MiB、计时取 3 次或 10 次时，在同一张卡上重复测：

1. 每个片段大小的 GB/s 在重复运行之间变化多大；
2. "第一个 CUDA batch 比 DSA 快的片段大小"（交界点）在重复运行之间会不会变。

## 3. 程序和硬件

- 机器：6000d（8× RTX 6000D），容器 `iaxl.vllm`（在 1119 仓库下运行 `./start.sh` 启动）。
- 程序：现有的 `benchmark/tensor_xfer/tensor_xfer_benchmark.sh`，单进程，不跑 vLLM。脚本按 `setvars.sh` 默认绑定 rank 0：GPU0、DSA 工作队列 `wq0.0`，并用 `numactl --cpunodebind=0 --membind=0` 运行。
- 只测两种方法：`batch`（CUDA batch，即 `cudaMemcpy3DBatchAsync`）和 `iaxl`（IAXL DSA）。
- 片段大小：256K 到 2M，步长 128K，共 15 个点；H2D 和 D2H 都测。只测到 2M，因为 6000d 旧图上 H2D 交界点在 512K 到 1M 之间，2M 已足够覆盖，又能控制总时长。

## 4. 一次运行

一次运行 = 启动一次 benchmark 进程，把 15 个片段大小 × 2 个方向 × 2 种方法各测一遍。每个片段大小、每种方法：

- 把"每次拷贝的总大小"切成大小相同的片段，主机内存是 pinned 内存，目的地顺序随机打乱；
- 先热身 W 次，再计时 N 次，取最快一次的时间换算成 GB/s（benchmark 原有逻辑）。

命令示例（设置 A，第 1 次运行）：

```bash
docker exec -w "$PWD" iaxl.vllm bash benchmark/tensor_xfer/tensor_xfer_benchmark.sh \
    --methods batch,iaxl --direction both \
    --frag-sizes 256K,384K,512K,640K,768K,896K,1M,1152K,1280K,1408K,1536K,1664K,1792K,1920K,2M \
    --total-gib 0.0625 --warmup 1 --iterations 3 \
    --output-dir "$PWD/tests/dsa_fallback_dev/results/stability_<时间>/A/run1" --plot '' \
    > tests/dsa_fallback_dev/results/stability_<时间>/A/run1.log 2>&1
```

## 5. 设置

| 设置 | 每次拷贝的总大小 | `--total-gib` | 热身次数 `--warmup` | 计时次数 `--iterations` |
|---|---|---|---|---|
| A | 64 MiB | 0.0625 | 1 | 3 |
| B | 64 MiB | 0.0625 | 2 | 10 |
| C | 128 MiB | 0.125 | 1 | 3 |
| D | 128 MiB | 0.125 | 2 | 10 |

- 每种设置重复 10 次运行，共 40 次运行。
- 运行顺序交错进行（A、B、C、D、A、B、C、D……），让机器状态随时间的变化平均地落到 4 种设置上。

## 6. 每个数字怎么算

- benchmark 每次运行对每个片段大小、每种方法输出一行：最快一次的毫秒数和对应的 GB/s。
- 每种设置有 10 次运行，因此每个片段大小、每种方法有 10 个 GB/s，样本数 10。
- 平均、中位数：这 10 个 GB/s 的平均值和中位数。
- 波动 %：(10 个 GB/s 里的最大值 − 最小值) ÷ 平均值。
- 交界点：每次运行单独算一次。从 256K 往上找，第一个 CUDA batch 的 GB/s 大于 DSA 的 GB/s 的片段大小；15 个点都没有就记为"无"。

## 7. 结果表格

表 1：每种设置、每个方向各一张，共 8 张。示例标题："H2D，设置 A（64 MiB，热身 1 + 计时 3），10 次运行"。

| 片段大小 | DSA 平均 | DSA 中位数 | DSA 波动 % | CUDA batch 平均 | CUDA batch 中位数 | CUDA batch 波动 % | CUDA batch 比 DSA 快的运行次数 |
|---|---|---|---|---|---|---|---|
| 256K | | | | | | | /10 |
| … | | | | | | | |

- 平均、中位数的单位是 GB/s，算法见第 6 节。
- "CUDA batch 比 DSA 快的运行次数"：10 次运行里，同一次运行中 CUDA batch 的 GB/s 大于 DSA 的 GB/s 的次数。

表 2：交界点稳不稳。

| 设置 | H2D 交界点：每个值出现的次数 | D2H 交界点：每个值出现的次数 |
|---|---|---|
| A（64 MiB，热身 1 + 计时 3） | | |
| B（64 MiB，热身 2 + 计时 10） | | |
| C（128 MiB，热身 1 + 计时 3） | | |
| D（128 MiB，热身 2 + 计时 10） | | |

- 每格写成"640K：2 次，768K：8 次"的形式，次数加起来等于 10。

## 8. 运行步骤

1. 试跑：设置 A 跑 2 次运行，把真实输出整理成第 7 节的表格，确认格式后再进行下一步。
2. 全量：4 种设置 × 10 次运行 = 40 次运行，在 tmux 会话 `dsa-calib-stability` 里跑。估算每次运行 10 到 20 秒，共 10 到 15 分钟。
3. 脚本（待写）：`tests/dsa_fallback_dev/stability_runs.sh` 负责运行，`tests/dsa_fallback_dev/stability_tables.py` 负责出表。结果目录 `tests/dsa_fallback_dev/results/stability_<时间>/`。
4. 跑之前先用 `nvidia-smi` 确认 GPU0 空闲。

## 9. 这个实验做不到的

- 计时方式不同：benchmark 对 CUDA batch 用 CUDA event 计时，只算 GPU 上的时间；校准计划用墙钟，从提交算到拷完，包含 CPU 提交开销。DSA 两边都是墙钟。
- 片段顺序不同：benchmark 的目的地顺序随机打乱，校准计划用倒序。
- 因此本实验只回答"重复测量的波动有多大"，不能精确代表校准最终会选哪个点。校准代码写好后，再用真正的校准函数重复跑一次确认。
- 6000d 的结果不代表 b300：两台机器的交界点不同。
- 6000d 的 D2H 交界点低于 256K（旧图目测在 64K 到 128K 之间），所以 D2H 在本实验范围内只能看波动，看不到交界点的变化。

## 10. 结果（2026-11-20，6000d）

运行信息：

- 时间：05:18 到 05:24 UTC，40 次运行全部完成，所有拷贝校验 PASS。每次运行 7 到 12 秒。
- 代码：git 7de59f2（`feat/dsa-auto-fallback-dev`，和 origin/main 相同，没有功能改动）。
- 镜像：`vllm-iaxl-dev-pr70:1.6.0`。1119 的 `vllm-iaxl-dev:1.6.0` 不存在，这个镜像的 `docker/Dockerfile.dev`、`requirements.txt` 和基础镜像都与 1119 相同；扩展在容器里用 1119 的代码重新编译。
- 和第 4 节命令的差别：`--output-dir` 直接用结果目录。`--plot ''` 时 benchmark 不写文件，只建目录，这样避免容器以 root 身份在每次运行下建空目录。没有单独试跑，直接跑了全量。
- 结果目录：`tests/dsa_fallback_dev/results/stability_20261120_051727/`，8 张完整的表 1 在其中的 `tables.md`。

表 2：交界点，每个值在 10 次运行里出现的次数。

| 设置 | H2D 交界点 | D2H 交界点 |
|---|---|---|
| A（64 MiB，热身 1 + 计时 3） | 512K：10 次 | 256K：10 次 |
| B（64 MiB，热身 2 + 计时 10） | 512K：10 次 | 256K：10 次 |
| C（128 MiB，热身 1 + 计时 3） | 384K：10 次 | 256K：10 次 |
| D（128 MiB，热身 2 + 计时 10） | 384K：10 次 | 256K：10 次 |

H2D 交界点附近和真实块大小附近的 GB/s（10 次运行的平均值）。Qwen 的块是 784K，Kimi 的块是 864K，分别落在 768K 到 896K 之间。

H2D，DSA 平均 GB/s：

| 片段大小 | 设置 A | 设置 B | 设置 C | 设置 D |
|---|---|---|---|---|
| 256K | 44.23 | 44.26 | 44.16 | 44.16 |
| 384K | 44.30 | 44.30 | 44.21 | 44.24 |
| 512K | 44.30 | 44.35 | 44.25 | 44.26 |
| 768K | 44.33 | 44.37 | 44.29 | 44.32 |
| 896K | 44.34 | 44.39 | 44.29 | 44.36 |

H2D，CUDA batch 平均 GB/s：

| 片段大小 | 设置 A | 设置 B | 设置 C | 设置 D |
|---|---|---|---|---|
| 256K | 40.62 | 40.68 | 40.74 | 40.81 |
| 384K | 43.82 | 43.87 | 44.36 | 44.41 |
| 512K | 45.96 | 46.00 | 46.50 | 46.53 |
| 768K | 47.99 | 48.08 | 48.80 | 48.84 |
| 896K | 48.69 | 48.76 | 49.38 | 49.43 |

D2H，DSA 平均 GB/s：

| 片段大小 | 设置 A | 设置 B | 设置 C | 设置 D |
|---|---|---|---|---|
| 256K | 43.09 | 43.11 | 35.70 | 35.75 |
| 1M | 43.22 | 43.24 | 35.82 | 35.87 |
| 2M | 43.23 | 43.27 | 35.86 | 35.89 |

结论（实测）：

- 重复运行很稳：同一设置 10 次运行之间，每个片段大小的波动 %，DSA 最大 0.5%，CUDA batch 最大 1.5%。
- 同一设置内交界点 10 次全部相同。热身 1 + 计时 3 和热身 2 + 计时 10 得到的交界点相同，波动也相近。
- 每次拷贝的总大小会让 H2D 交界点移动一档：64 MiB 是 512K，128 MiB 是 384K。384K 处两种方法几乎打平：64 MiB 时 CUDA batch 比 DSA 慢 1.1%（43.82 对 44.30），128 MiB 时快 0.3%（44.36 对 44.21）。
- 真实块大小（784K、864K）附近，40 次运行都是 CUDA batch 更快（768K 快 8.3% 到 10.2%，896K 快 9.8% 到 11.5%），4 种设置的选择相同。
- D2H 在所有片段大小上都是 CUDA batch 更快，交界点低于 256K，本实验看不到。
- 意外发现：DSA 的 D2H 在 128 MiB 时约 35.8 GB/s，64 MiB 时约 43.1 到 43.9 GB/s，低 17%，10 次运行都一样。原因没有查。6000d 上它不改变选择（D2H 本来就全是 CUDA batch 更快）。

决定（2026-11-20）：校准采用设置 D（每次拷贝 128 MiB，热身 2 次 + 计时 10 次，取最快一次）。4 种设置的单次运行时间相差不大（设置 A 平均 7.1 秒，设置 D 平均 11.4 秒）。

## 11. 功能测试结果（2026-11-20，6000d）

被测代码：`feat/dsa-auto-fallback-dev` 上未提交的改动（C++ 校准 + KVStore 启动时校准，整组用 TP 组第一个 rank 的结果），加上 kvstore benchmark 的修改：`Benchmark.setup()` 里创建 KVStore 的部分拆成 `create_store()`，多 rank 时所有 rank 同时调用。原因：创建 KVStore 时要等同组所有 rank 的 `/v1/health`，原来 rank 0 单独先创建，会等满 60 秒后报错。

结果目录：

- `tests/dsa_fallback_dev/results/tests_20261120_072455/`：第 1 步；第 4 步 Qwen3-32B。
- `tests/dsa_fallback_dev/results/tests_20261120_074556_rest/`：第 2、3 步（benchmark 修改后）；第 4 步 Qwen3.8-27B。
- 脚本：`tests/dsa_fallback_dev/run_tests.sh`，每项结果在各目录的 `summary.txt`，第 4 步明细在 `step4/<模型>/check.txt`。

每一步做什么、结果：

| 步骤 | 做什么 | 结果 |
|---|---|---|
| 1 | 3 种环境变量组合查开关；GPU0 上一个进程连续调 3 次 `dsa_calibrate_crossover` | 开关 3 种都对；3 次都是 H2D 512K、D2H 256K，每次 0.35 秒；前后空闲显存相同（84569 MiB） |
| 2 | KVStore + mock 数据，784K 块，32 层；用 `dsa_set_crossover` 强制 H2D/D2H 各走 DSA 或 CUDA batch，4 种组合；PUT 再 GET，比较每层 32 个块 | 4 种组合数据全部一致；每个方向的 backend 日志与强制的一致 |
| 3 | `kvstore_benchmark.sh --dsa`：1 rank 16K 块、1 rank 784K 块、2 rank 16K 块；另外只起 rank 1、`IAXL_API_TIMEOUT=5` | 3 次运行 Verification passed；2 rank 时两个 rank 用 rank 0 的同一个分界值；超时测试 7.6 秒后抛 `RuntimeError`，报错写明在等 rank 0 的 `/v1/health` |
| 4 | vLLM TP=8，不压缩，DSA；Qwen3-32B 和 Qwen3.8-27B 各起两次（`IAXL_DSA_AUTO_FALLBACK_ENABLE=1` / `=0`）；4 个约 15K token 的 prompt 各发 3 轮，temperature 0，生成 32 token | 见下表 |

第 4 步（每个模型每个设置 12 个请求）：

| 模型 | 连续块大小 | `=1` 时整组分界值（rank 0 测） | `=1` 两个方向走的路 | `=0` 两个方向走的路 | 第 2、3 轮命中 token | 回复 |
|---|---|---|---|---|---|---|
| Qwen3-32B | 8 KB | H2D 384K，D2H 256K | 都走 DSA | 都走 DSA | 15328 | 24 条都是通顺英文；prompt 0 答出正确值 80393，其余是推理开头（32 token 不够写到答案） |
| Qwen3.8-27B | 400 KB | H2D 256K，D2H 256K | 都走 CUDA batch | 都走 DSA | 15200 | 24 条都直接答出正确值（80393、88312、96231、4150） |

- 8 个 rank 都打出同一个分界值；`=0` 时没有 rank 校准；没有 Traceback。
- vLLM 启动多出的时间：等同组 rank 最多 0.11 秒，测量 0.24 到 0.51 秒，等 rank 0 结果最多 0.13 秒，合计不到 1 秒。
- 正确值按 prompt 的生成公式 `(prompt 编号 × 7919 + 17 × 104729) mod 100000` 计算。

发现：GPU0 测出的 H2D 分界值会变（实测）。

| 测量场景 | 同时在测的 rank 数 | GPU0 的 H2D 分界 |
|---|---|---|
| 第 1 步，一个进程连续 3 次 | 1 | 512K、512K、512K |
| 第 3 步，1 rank（4 次运行） | 1 | 384K、512K、384K、384K |
| 第 3 步，2 rank | 2 | 256K |
| 第 4 步，Qwen3-32B vLLM | 8 | 384K |
| 第 4 步，Qwen3.8-27B vLLM | 8 | 256K |

- D2H 每次都是 256K（测量范围的最小值）。
- 单独测时，384K 处两条路几乎打平（DSA 44.4 GB/s，CUDA batch 44.8 GB/s），所以 H2D 分界在 384K 和 512K 之间来回。
- 8 个 rank 同时测时，大多数卡的 H2D 只有 DSA 21 到 27 GB/s、CUDA batch 约 27.5 GB/s；每次有 1 到 2 张卡测到约 44 GB/s（Qwen3-32B 那次是 GPU0，Qwen3.8-27B 那次是 GPU5、GPU7），是哪几张每次不同。
- 两次 vLLM 启动里，8 个 rank 各自测出的 H2D 分界：Qwen3-32B 是 256K 六个、384K、512K；Qwen3.8-27B 是 256K 六个、512K、640K。
- 影响：Qwen3.8-27B TP=8 的块是 400 KB，落在 256K 到 512K 之间。这次 rank 0 测出 256K，H2D 走 CUDA batch；如果 rank 0 测出 512K，H2D 就走 DSA。也就是说，这个模型在 6000d 上 H2D 走哪条路，取决于每次启动时 rank 0 的测量。
- 决定（2026-11-20）：接受这个差异，不改代码。

### 11.1 8 个 rank 同时测时速度下降的原因：两张 GPU 共用一条 PCIe 上行链路

DSA 的配置符合"一张 GPU 一个 DSA 设备"（`tests/dsa_fallback_dev/results/topology_survey.txt`）：

- 8 个 DSA 设备 dsa0、dsa2 … dsa14，每个只有一个 dedicated WQ（`wqN.0`，size 256，4 个 engine），空闲时 clients=0。
- vLLM 里 rank i 用 `wq(2i).0`（`KVSHRINK_DSA_DEVICES=wq0.0|wq2.0|…|wq14.0`），每个 rank 的 `dsa_init` 只映射自己那一个 WQ。
- NUMA 对得上：GPU0–3 和 dsa0–6 在 NUMA 0，GPU4–7 和 dsa8–14 在 NUMA 1。

PCIe 拓扑：GPU 两两挂在一个 Broadcom PEX890xx Gen5 交换芯片下，交换芯片只有一条 Gen5 x16 链路连到 CPU（根端口 35:02.0、45:02.0、b5:02.0、c5:02.0）。GPU0+1、2+3、4+5、6+7 各共用一条。DSA 和 CUDA batch 的拷贝都经过这条链路。

验证实验（`tests/dsa_fallback_dev/pair_contention.sh`）：两张 GPU 同时各自调 20 次 `dsa_calibrate_crossover`，每张用自己的 DSA WQ。

| 组合 | 是否共用上行链路 | GPU 和 WQ | H2D 分界（20 次中出现次数） | H2D 的 DSA GB/s（前 5 次） |
|---|---|---|---|---|
| GPU0 单独 | — | GPU0 / wq0.0 | 512K：20 | 44.4 |
| GPU0 + GPU2 同时 | 否 | GPU0 / wq0.0 | 384K：20 | 44.4–44.5 |
| | | GPU2 / wq4.0 | 384K：17，512K：3 | 43.3–43.7 |
| GPU0 + GPU1 同时 | 是 | GPU0 / wq0.0 | 384K：3，512K：15，768K：2 | 23.6–23.7 |
| | | GPU1 / wq2.0 | 256K：3，384K：3，512K：14 | 24.3–25.0（一次 44.4） |

- 不共用链路的两张卡同时测，速度和单独测一样；共用链路的两张卡同时测，DSA 各自只剩约一半。
- 8 个 rank 同时测时，没用 DSA 的 CUDA batch 也从约 45 GB/s 降到约 27.5 GB/s，说明瓶颈是链路，不是 DSA。
- 推断（日志没有时间戳，未证实）：8 个 rank 同时测时，每对 GPU 里测到约 44 GB/s 的那张，测了更多块大小，测这些块时搭档已经停止 H2D、改测 D2H（链路的另一个方向），所以没有争用。两次 vLLM 里的高值都出现在搭档只测了 1 个块大小（256K）的那张卡上。
