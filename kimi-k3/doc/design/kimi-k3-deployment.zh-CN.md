# Kimi-K3 在 b300 上怎么部署（讨论记录）

状态：讨论中，下面的方案都还没试。标注说明：**实测** = 在 b300 上查到/跑出的；**代码** = 读源码得到的；
**推算** = 根据实测数字计算，没跑过；**资料** = 官方 recipe / vLLM 文档 / GitHub。

## 1. 问题

kvshrink 每个 vLLM rank（每张 GPU 对应的 worker 进程）要用一个 DSA / IAA / QAT 设备搬运、压缩 KV。
b300 每种加速器只有 4 个，Kimi-K3 现在用 TP8 跑，有 8 个 rank，加速器不够。

现在的分配规则（**代码**，`tools/auto_config.sh`）：每个 rank 独占设备，设备数少于 TP 数直接报错，
例如 `ERROR: TP_SIZE=8 requires at least 8 DSA devices`。这是我们软件的规定，不是硬件限制：DSA / IAA
的共享工作队列支持多个进程提交，QAT 一个设备有多个实例。

目前 TP8 + kvshrink（加速器和压缩全关）已验证能用：命中达到上限，gsm8k 30 题 × 4 遍正确率都是 96.7%，
无乱码（结果在 b300 `tests/bench_dev/results/kimi-k3-gsm8k-20261112_084942_f4dc227b_limit30/`）。

## 2. 机器拓扑（实测）

4 个 NUMA 节点，每个节点 2 张 GPU、1 个 DSA、1 个 IAA。QAT 个数还没确认（`lspci` 数到 8 行相关设备）。

| NUMA | GPU | DSA | IAA |
|---|---|---|---|
| 0 | GPU0、GPU1 | dsa0 | iax1 |
| 1 | GPU2、GPU3 | dsa2 | iax3 |
| 2 | GPU4、GPU5 | dsa4 | iax5 |
| 3 | GPU6、GPU7 | dsa6 | iax7 |

每张 B300 显存 275,040 MiB ≈ 288 GB（nvidia-smi）。

## 3. Kimi-K3 的结构

- 93 层：69 层 KDA（线性注意力，每个请求存一个固定大小的状态）+ 24 层 MLA（每个 token 存 KV）。
- MLA（**实测**，`config.json`）：96 个 query 头；每个 token 的 KV 压成一个 576 维向量
  （`kv_lora_rank` 512 + `qk_rope_head_dim` 64），96 个头读的都是这一个向量。
- 权重按类型拆分（**实测**，只读 safetensors 文件头里的张量大小，共 1560.9 GB）：

| 权重 | 大小 |
|---|---|
| 路由专家 | 1446.5 GB |
| attention 投影（KDA q/k/v/g/o 约 69 GB，MLA 约 3 GB） | 72.4 GB |
| 共享专家 | 24.3 GB |
| latent MoE 的 up / down 投影 | 9.5 GB |
| embedding + lm_head | 4.7 GB |
| gate、第一层 dense MLP、视觉编码器 | 约 3.3 GB |

## 4. MLA 在 TP 下：计算切开，KV 不切

**代码**（vLLM 0.29 `vllm/models/kimi_k3/nvidia/mla.py`）：

- L195-198 `num_local_heads = num_heads // tp_size`：96 个头分到 8 张卡，每张卡算 12 个头，权重和计算切开了。
- L424-427 KV cache 规格 `num_kv_heads=1, head_size=576`，与 TP 无关：每张卡存完整的 576 维向量。

所以 TP8 时 8 张卡的 MLA KV 内容相同，kvshrink 每个 rank 各存一份，DDR 和加速器都做了 8 份一样的活
（**推断**，未实测比较 8 个 rank 存的数据）。KDA 按卡切开，每张卡存的内容不同。

## 5. 部署方式对比

术语：
- **TP**（tensor parallel）：每层权重切成 TP 份，TP 张卡一起算同一批请求。
- **PP**（pipeline parallel）：前一半层放一组卡，后一半层放另一组卡。
- **DP**（data parallel）：每组卡各放一份 attention，处理不同的请求；DP 组之间 KV 不重复。
- **EP**（expert parallel）：专家分到所有卡上，每张卡放一部分专家。
- **DCP**（decode context parallel）：TP 组内把 MLA 的 KV 按 token 切开，每张卡只存 1/DCP，消除重复。
- **AFD**（Attention-FFN Disaggregation）：一组卡只算 attention、存 KV，另一组卡只算 FFN（专家），
  每层之间互相传中间结果。

MLA KV 总份数：N 个请求、每个 DP 组 TP 张卡时，总份数 = N × TP（开 DCP=TP 时为 N）。4 个加速器平分，
每个负担 N × TP ÷ 4。KDA 状态总量在各方式下相同。

每张卡权重 = 72.4 ÷ TP + 41.8（其他，可能 ÷ TP，也可能每卡一份）+ 专家 1446.5 ÷ 8 = 180.8（**推算**）。

| 方式 | attention 在几张卡 | 每卡权重（推算） | MLA 总份数 | 每个加速器负担 | 可行性 |
|---|---|---|---|---|---|
| TP8（现在） | 8 | 约 195 GB | 8N | 2N | 能跑（实测） |
| TP4 × PP2 | 8（每卡只有一半层） | 约 195 GB | 4N | N | K3 实现了 `SupportsPP`（代码）；PP + kvshrink 未验证 |
| TP8 + DCP8 | 8 | 约 195 GB | N | N/4 | K3 MLA 支持 DCP（代码 `MLADCPManager`）；kvshrink 要支持 DCP 的 KV 布局 |
| DP2 × TP4 + EP8 | 8（两组各 4） | 209–241 GB | 4N | N | 显存放得下；不在官方 recipe 里，未验证 |
| DP4 × TP2 + EP8 | 8（四组各 2） | 238–259 GB | 2N | N/2 | 显存利用率 0.9 时预算约 259 GB，几乎不剩 KV 空间 |
| DP8 + EP8（DEP8） | 8 | 约 295 GB（共享专家切开约 274 GB） | N | N/4 | 不行：放不下；recipe `strategy_min_gpus.multi_node_dep = 16` |
| AFD 4 + 4 | 4 | 专家卡 1446.5 ÷ 4 = 362 GB | — | — | 不行：专家放不进 4 张卡；vLLM 也没有 AFD（RFC #22799 关闭，PR #25162、#29772 未合入） |

除 AFD 外，每种方式 8 张卡上都有 attention 和 KV，都是 8 个 rank 要搬 KV，所以**每个 NUMA 节点上 2 个
rank 共用 1 个加速器是必须的**，`auto_config.sh` 的独占规则要改。各方式的差别在于每个加速器要搬多少 MLA。

## 6. 已排除的想法

- **只让 attention 用 4 张卡**：只有 AFD 能做到，见上表，不可行。
- **TP8 下只让 4 张卡存 MLA**：不成立。每张卡算自己的 12 个头都要完整的 MLA KV，读回时还得写进全部 8 张卡，
  读的工作量没减少。
- **DP8 + EP8**：显存放不下（见上表）。

## 7. 待确认 / 待尝试

1. 加速器共用：改 `auto_config.sh`，同一 NUMA 的 2 个 rank 共用 DSA / IAA / QAT；测 2 个 rank 同时搬运时
   每个 rank 的带宽，判断加速器是不是瓶颈。
2. 实测 MLA 重复：同一请求比较 8 个 rank 存进 DDR 的 MLA 块；看 MLA 和 KDA 各占多少 GB。
3. DP2 × TP4 + EP8：
   - vLLM 0.29 上 K3 能否这样启动，KV 容量多少；
   - kvshrink 在 DP 下两组能否共享 DDR 缓存（现在存储目录按 `Kimi-K3_rank0..7` 命名，编号规则未查），
     不能共享的话 vLLM 怎么把同前缀请求分到同一组；
   - `auto_config.sh` 的设备分配按 TP 数算，DP 下要按总 rank 数。
4. DCP：kvshrink 支持 DCP 布局要改多少。
5. QAT 设备个数。

## 8. 参考

- 官方 recipe：b300 仓库 `tests/bench_dev/recipes/moonshotai_Kimi-K3.json`（recipes.vllm.ai/moonshotai/Kimi-K3）。
  `strategy_min_gpus`：`single_node_tp` 8、`multi_node_tep` 8、`multi_node_dep` 16、`multi_node_tp_dp` 16、
  `multi_node_tp_pp` 16。
- vLLM 0.29 文档：`docs/serving/data_parallel_deployment.md`、`docs/serving/expert_parallel_deployment.md`。
- LMCache 0.5.3 的 K3 recipe：`docs/source/recipes/kimi_k3.rst`（TP8，`--max-num-batched-tokens` 在
  [768, 1536)，`--enable-prefix-caching`）。

## 9. 决定与实测记录

决定（11-13）：不比较性能，目标是加速器压力小，优先级 DP8 + EP > DP4TP2 + EP > DP2TP4 + EP；每种配置只
launch 一次（`tests/bench_dev/kimi_k3_dp_layouts.sh`），`--gpu-memory-utilization 0.9`、`--max-model-len -1`、
加速器和压缩全开、DDR 池 1200 GB、scratch pool 每 rank 8 GB；能 launch 的配置给出测试矩阵
（32k / 128k / 256k / 512k / 1M 各支持多少并发）。DP 各组缓存不共享，warmup 用 `X-data-parallel-rank`
保证每个 DP rank 都收到。

### 9.1 加速器共用（2 张 GPU 共用本 NUMA 的 1 个设备）

- 改动：`tools/auto_config.sh`（QAT 设备少于 worker 时按 NUMA 共用；DSA 按 WQ 分配；新增
  `iaa_auto_detect`），`setvars.sh`（`KVSHRINK_IAA_DEVICES`），`iaxl/utils/affinity.py`（绑定 IAA）。
- b300 DSA 改为每个设备 2 个 dedicated WQ（各 128 深，原来 1 个 256 深）：`tools/setup_dsa_cnt.sh 2`。
  **测别的模型前要改回**：`IMAGE=vllm-iaxl-dev-kimi-k3-qiuyu:1.6.0 bash tools/setup_dsa_cnt.sh 1`。
- QAT 配置（实测，`/etc/4xxx_dev*.conf`）：`ServicesEnabled = dc`、`NumberDcInstances = 8`、`NumProcesses = 8`。
- 启动前验证（实测，`tests/bench_dev/k3_accel_share_check.sh`，结果
  `tests/bench_dev/results/k3-accel-share-20261113_075344/run.log`）：
  - 8 个 worker 的分配：QAT `0|0|1|1|2|2|3|3`，DSA `wq0.0|wq0.1|wq2.0|wq2.1|wq4.0|wq4.1|wq6.0|wq6.1`，
    IAA NUMA `0|0|1|1|2|2|3|3`，都是本 NUMA 的设备。
  - `kvstore_benchmark.sh --ranks 8 --qat --iaa --dsa`（8 个进程同时存取 512 MiB mock 数据）：8 个 rank
    校验全部 pass，压缩比 1.39，每 rank 存 5.4-5.6 GiB/s、取 6.6-7.8 GiB/s。`ADF_UIO_PROXY err ... /dev/qat_dev_processes`
    每进程 1 行，之前能用的 Qwen3.8 TP4 运行里也有，QAT 照常初始化（每进程 4 个实例）。

### 9.2 DP8 + EP8（实测：launch 失败）

加载权重时 8 个 worker 都 `torch.OutOfMemoryError`：每卡已分配 265.09 GiB，再要 676 MiB 时只剩 153 MiB
（每卡总量 267.69 GiB）。和第 5 节推算一致：DP8 每卡权重放不下。结果
`tests/bench_dev/results/kimi-k3-20261113_075650_6c8ea66e/serve.log`。

### 9.3 DP4TP2 + EP（实测：launch 失败，0.9 下放不下）

权重加载成功，每卡 `Model loading took 239.73 GiB`，block size 2944。之后测显存时 OOM：
`profile_cudagraph_memory → _init_minimal_kv_cache_for_profiling` 申请 77.62 GiB 失败。
即使不看这一步，每卡权重 239.73 GiB 已接近显存利用率 0.9 的预算（0.9 × 267.69 = 240.9 GiB），不剩 KV 空间。
**结论：DP4TP2 + EP 在 0.9 下不支持**（11-13 与用户确认）。
结果 `tests/bench_dev/results/kimi-k3-20261113_075820_6c8ea66e/serve.log`。

### 9.4 DP2TP4 + EP（实测：launch 失败，原因是测显存那一步的 KV 太大）

权重加载成功，每卡 `Model loading took 221.78 GiB`，block size 1536。测显存时录 CUDA graph OOM（申请 598 MiB，
只剩 445 MiB）。原因（读代码 `vllm/v1/worker/gpu/cudagraph_utils.py` `_init_minimal_kv_cache_for_profiling`）：
测显存前 vLLM 先分配 `min(max_num_seqs, max_cudagraph_capture_size)` 个 block 的 KV（默认 1024 个），
DP 布局下 block 很大（KDA 状态按 TP 切得少，block size 跟着变大），1024 个 block 就占几十 GiB
（DP4TP2 是 77.62 GiB）。TP8 时 block 只有 768，所以没出问题。调小 `--max-num-seqs` 可减少这一步的 KV。
结果 `tests/bench_dev/results/kimi-k3-20261113_080201_6c8ea66e/serve.log`。

### 9.5 DP2TP4 + EP + `--max-num-seqs 128`（实测：launch 成功）

结果目录 `tests/bench_dev/results/`：launch 与冒烟 `kimi-k3-20261113_081431_6c8ea66e/`，gsm8k
`kimi-k3-gsm8k-20261113_082517_3a3a7495_limit30/`。加速器和压缩全开（QAT + IAA + DSA，每 rank 绑本 NUMA 设备）。

- 加载 520 s 就绪。每卡权重 221.78 GiB，block size 1536，每卡 KV 显存 6.21 GiB。
- 每个 DP 组 KV 容量 227,239 token（两个 DP 组各一份）；`--max-model-len -1` 自动定为 225,792（147 个 block）。
- 冒烟：每条请求发给 DP0、DP1 各一次，18 条全对、无乱码；每组第一次见到的请求命中 0（两组缓存不共享），
  重复的请求命中到上限（10k prompt 9216 = 6 块，20k prompt 19968 = 13 块）。
- gsm8k 30 题 × 4 遍：96.7% / 96.7% / 100% / 100%，乱码 0；错的只有第 12 题（TP8 时也错）。pass2 命中到上限
  （26 例题 4608 = 3 块，140 例题 23040 = 15 块）。日志 Traceback / ERROR / `cache key missing after retry` /
  `cache miss on unzip` 都是 0。DDR 监控 zip/raw = 0.70（压缩后是原来的 70%）。

测试矩阵（推算：每组并发 = ⌊227,239 ÷ (输入 + 128 输出)⌋，与 `kvshrink-benchmark.sh` 的算法相同，
没算 KDA 状态另占的 block，实际可能略少）：

| 输入长度 | 单请求放得下（≤ 225,792） | 每 DP 组同时放得下 | 两组合计 |
|---|---|---|---|
| 32k | 是 | 7 | 14 |
| 128k | 是 | 1 | 2 |
| 256k / 512k / 1M | 否 | — | — |


