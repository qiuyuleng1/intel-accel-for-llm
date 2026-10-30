# GLM-5.3-Flash 在 8×RTX 6000D（sm_120）上的测试记录

本文汇总 2026-09 到 2026-10 在本机 8×RTX 6000D 上用各版本 vLLM 跑 GLM-5.3-Flash 的测试结果和分析。每条结论标注来源：**实测**（本机运行得到）、**读代码**（读 vLLM / FlashInfer 源码或 git 历史得到）、**推断**（由前两者推出，未直接验证）。

## 1. 结论

| vLLM 版本 | 有没有 GLM-5.3-Flash 模型代码 | 本机能不能跑 | 依据 |
|---|---|---|---|
| 官方镜像 `vllm/vllm-openai:glm53-flash` | 有 | 不能，启动后第一次前向报 `pe_dim must be 64 for fp8_ds_mla` | 实测 |
| 社区镜像 `cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1` | 有 | 能，GSM8K 0.9773 | 实测 |
| v0.29.0 | 没有 | 不能，加 rope 补丁也不行 | 读代码 |
| v0.32.0（最新正式版） | 有 | 能启动，但输出全错（logits 全是 NaN），之后卡死 | 实测 |
| nightly `0.33.0rc1.dev157+gedad8b03c` | 有 | 能，不需要任何补丁，GSM8K 0.9742 | 实测 |

表中各列：
- **vLLM 版本**：所测的镜像或版本。
- **有没有 GLM-5.3-Flash 模型代码**：vLLM 里是否有 `Glm5NextForConditionalGeneration` 的实现。
- **本机能不能跑**：TP=8、fp8 KV 下能否启动并给出正确输出。
- **依据**：见本文开头的三种来源。

要点：
- 目前在本机能正确跑的有两个：nightly（不打补丁）和社区镜像。两者 GSM8K 准确率没有显著差异（第 6.5 节）。
- nightly 在并发 64 时平均每个请求比社区镜像慢 3.5 倍（实测，原因未查，第 8 节）。
- v0.32.0 输出全错的最可能原因是 FlashInfer 0.7.0.post1 的 SM120 稀疏 attention kernel 会读到 KV cache 里从未写过的位置，被其中的 NaN 污染；FlashInfer 0.7.1 起已修复（推断，第 7 节）。

## 2. 环境

- 8× NVIDIA RTX 6000D，每卡 85651 MiB，compute capability 12.0（sm_120，工作站 Blackwell，不是 B200 的 sm_100）。8 卡之间只有 PCIe，没有 NVLink。
- 模型：`/media/model-space/GLM-5.3-Flash`，原生 FP8（e4m3，按块量化），62 个 safetensors 分片，305.79 GiB。
- 所有测试都是 TP=8、`--kv-cache-dtype fp8`。sm_120 上 GLM-5.3 的稀疏 attention 只有 `FLASHINFER_MLA_SPARSE_SM120` 一个后端，它只接受 `fp8_ds_mla` 这种 KV 布局，不支持 bf16 KV（读代码）。

## 3. 模型结构中和本次测试有关的部分

- 45 层，其中 34 层是 KDA 线性注意力，11 层（第 3、7、…、43 层）是 DeepSeek-V3.2 式的稀疏 MLA；MoE 为 288 个路由专家 + 1 个共享专家，每个 token 选 8 个；带 1 层 MTP 和视觉塔。
- 稀疏 MLA 层不用 rope：`qk_rope_head_dim=0`、`qk_nope_head_dim=256`、`kv_lora_rank=512`。DeepSeek-V3.2 的 rope 维度是 64，sm_120 上原有的 kernel 只支持带 64 维 rope 的布局，这是 9 月份跑不起来的原因。
- 稀疏 MLA 的 indexer 每次选 `index_topk=2048` 个位置，GLM 特有的 kpool（`index_kpool=4`）把相邻 4 个 key 合成一个打分单位，再加上最多 3 个还没凑满 4 个的尾部 token。vLLM 为此分配的候选位置缓冲宽度是 2048+3 向上取整到 128 的倍数，即 **2176**，有效候选最多 2051 个，其余位置填 -1（读代码，`vllm/models/glm5next/common/model.py::get_sparse_topk_buffer_width`）。所以**每个 query 都有被填 -1 的候选位置**，短 prompt 几乎全是 -1。这一点和第 7 节的乱码原因直接相关。

## 4. 各版本 vLLM 的来历和支持情况

### 4.1 9 月份用过的两个镜像

| 镜像 | vLLM 版本号 | 对应的提交 | 从 main 分出的位置 | 镜像创建时间 | FlashInfer |
|---|---|---|---|---|---|
| `vllm/vllm-openai:glm53-flash` | `0.1.dev20051+g487ecf187` | `487ecf187d`（2026-08-25），在 GLM 开发分支 `glm5_next` 上，比 main 多 142 个提交，不在 main 上 | `b908a21f9a`（2026-08-12） | 2026-08-26 | 0.6.17 |
| `cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1` | 同上 | Dockerfile 为 `FROM vllm/vllm-openai:glm53-flash@sha256:2c6da6c6…`，加一层 Python 文本替换补丁 | 同上 | 2026-08-26 | 0.6.17 |

表中各列：
- **vLLM 版本号**：镜像内 `vllm.__version__`。
- **对应的提交**：由版本号里的短哈希在 GitHub 上查到的完整提交（读代码）。
- **从 main 分出的位置**：该提交与 main 的共同祖先（`git merge-base`）。
- **镜像创建时间**：`docker images` 显示的时间。

两个镜像的 main 基础（08-12）都早于 v0.29.0 从 main 分出的位置（`f5c3cc240b`，2026-08-31），但多了 GLM 开发分支上的模型代码，这部分 v0.29.0 没有。

### 4.2 相关提交进入正式版的时间（读代码，git 历史）

| PR | 内容 | 合入 main | 第一个包含它的正式版 |
|---|---|---|---|
| #53906 | 新增 GLM-5.3-Flash 模型 | 2026-09-03 | v0.30.0 |
| #55277 | sm_120 支持 NoPE：`concat_and_cache_mla` 允许 `pe_dim=0`，在 KV 行的 rope 段写 0 | 2026-09-23 | v0.31.0 |
| #57169 | GLM-5.3 KV 布局调整，其中包括 sm_120 上 indexer 的分页大小只用 64（DeepGEMM 在 sm_120 只支持 64） | 2026-10-09 | v0.32.0 |
| #58305 | sm_120 上 NoPE 的 KV 行从 656 字节缩成 528 字节（去掉全 0 的 rope 段） | 2026-10-28 | 还没进正式版 |

v0.29.0 里搜不到 `glm5next`、`Glm5Next`、`index_kpool`，git 也确认 #53906 不在 v0.29.0 的历史里。v0.29.0 启动时会在识别模型结构这一步失败，所以只给 v0.29.0 加 rope 补丁跑不起来，需要把整个模型（KDA、kpool indexer、MTP、多模态等）移植过去（读代码）。

### 4.3 社区镜像具体改了什么（读代码，它的 Dockerfile）

1. 把 q 和 k 的 rope 部分各补 64 维 0，按 576 维走 sm_120 已有的 GLM_NSA kernel。补的是 0，点积不变，这一步**无损**。
2. 把 top-k 候选缓冲宽度从 2176 改成 2048（当时的 kernel 只接受 2048）。
3. 为了塞进 2048，indexer 只展开前 511 个打分单位（2044 个 token），加上 3 个尾部 token，共 2047 个候选，**丢掉排名最低的那个打分单位（4 个 token）**。这一步和参考实现**不等价**，影响很小，但不是数学上完全一致。nightly 保留全部 2051 个候选。
4. sm_120 上 indexer 分页大小只用 64，稀疏 MLA 的 kernel 块大小固定为 64（与 #57169 的修复相同）。

另外社区镜像每个 token 的 KV 行是 656 字节，其中 128 字节是 0；nightly（#58305）是 528 字节。

## 5. 测试方法

脚本在本机 `/home/pese/qiuyu/intel-accel-for-llm-0901/glm53-sm120-test/`（未纳入本仓库）：

| 脚本 | 作用 |
|---|---|
| `run-glm53-image.sh` | 用指定镜像启动 vLLM（TP=8），等服务可用后依次跑 `glm53_probe.py`、`glm53_check.py`；vLLM 日志实时写入结果目录；容器提前退出时立刻报错；结束时删除容器 |
| `glm53_probe.py` | 数值探针，见下 |
| `glm53_check.py` | 功能检查，见下 |
| `run-gsm8k.sh` / `gsm8k_summary.py` | 两个镜像依次跑 GSM8K 全量并逐题对比 |
| `nan_argmax_check.py` | 验证 v0.32.0 的采样函数遇到全 NaN 的 logits 会选哪个 token |

数值探针（`glm53_probe.py`，用 `/v1/completions` 的 `prompt_logprobs`）：
- 一段 108 token 的英文段落，计算模型给每个 token 的平均负对数似然（单位 nats/token，越小越好）。正常模型约 1~3，模型坏掉时接近 8 以上。
- 同一段紧接着再出现一遍，算第二遍的平均负对数似然。能从上文照抄说明 attention 和 KV 读写正常，正常值约 0~0.3。
- "The capital of France is" 的下一个 token 及其概率。

功能检查（`glm53_check.py`，`/v1/chat/completions`，temperature=0、seed 固定、`reasoning_effort=low`，一次只发一个请求）：
- 5 道算术题，答案必须完全正确。
- 长文找口令：在一段约 1.7 万 / 4.4 万 token 的日志里分别把口令藏在 10% / 50% / 90% 位置，要求答出口令。长度远超 2048，indexer 必须真的选对位置。
- 同一个长文请求再发一遍，回答和思考过程逐字节比对。
- 生成 1500 token 的长文，检查连贯性并粗略计算解码速度。
- 所有输出扫描 U+FFFD 和非法控制字符。

## 6. 测试结果（均为实测）

### 6.1 9 月：官方 `glm53-flash` 镜像

权重加载、MoE（DeepGEMM FP8）、KDA 都正常，只有稀疏 MLA 没有可用后端：

| 尝试的后端 | 结果 |
|---|---|
| `FLASHINFER_MLA_SPARSE_SM120`（自动选择） | `concat_and_cache_mla` 报 `pe_dim must be 64`，模型 `pe_dim=0` |
| `FLASHINFER_MLA_SPARSE_SM90`（强制） | 不支持该 compute capability，且需要 FlashInfer ≥ 0.6.18（镜像是 0.6.17） |
| `FLASH_ATTN_MLA_SPARSE`（强制） | 不支持该 compute capability |
| `TRITON_MLA` | 不支持稀疏 |

### 6.2 9 月：社区镜像

- 启动：仍选 `FLASHINFER_MLA_SPARSE_SM120` + `fp8_ds_mla`，没有 `pe_dim` 报错；KV 每卡 32.04 GiB、共 3,883,681 tokens（max-num-seqs 16）；初始化 215 s。
- 5285 token 的输入回答正确；1500 token 的输出约 101 token/s。

### 6.3 10-28：v0.32.0（`sha256:befccf0e…`，FlashInfer 0.7.0.post1，torch 2.13.0）

- 不打补丁就能启动：选 `FLASHINFER_MLA_SPARSE_SM120` + `fp8_ds_mla`，KV 每卡 28.71 GiB、共 2,939,606 tokens，初始化 320.64 s。
- **输出全错**：5 道算术题（输入 37~47 token）全部生成满 2048 token，内容是 `locklocklock…`；长文找口令（16,863 / 43,680 token）同样全是 `lock`。短 prompt 也错，说明不是长上下文的问题。
- 之后 worker 无响应，5 分钟后 `TimeoutError`，引擎退出、请求返回 500。卡死前我额外并发发了一个探测请求，**不能排除它与卡死有关**。

### 6.4 10-29：nightly（`sha256:2e735ffe…`，vLLM `0.33.0rc1.dev157+gedad8b03c`，FlashInfer 0.7.2.post1，torch 2.14.0）

该 nightly 对应 main 在 2026-10-28 05:01 UTC 的提交，包含 #55277、#57169、#58305。启动参数与 v0.32.0 完全相同，不打任何补丁。

- 启动：KV 每卡 28.77 GiB、共 4,128,768 tokens，比 v0.32.0 多 40%。#58305 的 656→528 字节最多只能解释约 24%，其余来源未分析。启动到可服务 472 s（首次启动含编译）。
- 结果：

| 检查项 | 输入 token 数 | 结果 |
|---|---|---|
| 数值探针：段落第一遍平均负对数似然 | 108 | 1.526 |
| 数值探针：段落第二遍平均负对数似然 | 108 | 0.189 |
| 数值探针：France 下一个 token | 12 | `' Paris'`，概率 0.98 |
| 5 道算术题 | 37~47 | 全对 |
| 长文找口令（10% / 50% / 90%） | 16,863 | 3/3 |
| 长文找口令（10% / 50% / 90%） | 43,680 | 3/3 |
| 同一请求重发，逐字节比对 | 16,863 | 回答和思考过程都一致 |
| 生成 1500 token 长文 | 40 | 连贯，约 88 token/s（总 token 数除以总耗时，含 prefill） |

表中"输入 token 数"取自服务端返回的 `usage.prompt_tokens`。所有输出没有乱码字符和非法控制字符。

### 6.5 10-29：GSM8K，nightly 与社区镜像对比

两个镜像设置完全相同：TP=8、fp8 KV、`max-num-seqs 64`、evalscope 1.9.1、全量 1319 题、4-shot（evalscope 默认）、temperature 0、`max_tokens 16384`、`reasoning_effort` 用模型默认的 max（模型卡要求评测用 max）、并发 64。GLM-5.3-Flash 没有公布 GSM8K 分数，所以用 9 月已验证的社区镜像做参照。

| 镜像 | 答对/总数 | 准确率 | 被截断 | 平均输出 token | 乱码字符 | 平均单请求耗时 | 跑完全部题 |
|---|---|---|---|---|---|---|---|
| nightly | 1285/1319 | 0.9742 | 0 | 222 | 0 | 95.8 s | 33.5 min |
| 社区镜像 | 1289/1319 | 0.9773 | 1 | 236 | 0 | 27.6 s | 14.4 min |

表中各列：
- **答对/总数、准确率**：evalscope 判分，准确率 = 答对数 ÷ 1319。
- **被截断**：输出写满 16384 token 仍未结束的题数。社区镜像这 1 题是第 119 题，两个镜像都答错。
- **平均输出 token**：每题输出长度（含思考过程）的平均值。
- **乱码字符**：所有输出中 U+FFFD 和非法控制字符的总数。
- **平均单请求耗时**：evalscope 记录的每个请求从发出到返回的平均时间。
- **跑完全部题**：1319 题的总耗时。

逐题对比：
- 两边答案相同的有 1308 题。
- 对错不一致 8 题：nightly 对、社区镜像错 2 题（182、640）；社区镜像对、nightly 错 6 题（340、406、901、951、1046、1092）。
- 6 比 2 的配对符号检验，双侧 p≈0.29，差异不显著。**nightly 精度与参照组相当，没有发现精度问题。**

启动前的数值探针：nightly 1.526 / 0.189 / Paris 0.98，社区镜像 1.571 / 0.178 / Paris 0.97，两者一致。

## 7. v0.32.0 输出全错的原因分析

### 7.1 logits 全是 NaN（实测）

v0.32.0 使用 V2 model runner，temperature=0 时的采样函数是 `vllm/v1/worker/gpu/sample/gumbel.py::gumbel_sample`。在 v0.32.0 镜像里把全 NaN 的 logits（词表 154880）喂给这个函数，选出的是 **token 1023，正是 "lock"**；对照组（第 777 个位置最大）选出 777。v0.32.0 每个请求从第一个 token 起就是 "lock" 重复，而且与输入无关，与"logits 全是 NaN"吻合。

### 7.2 NaN 最可能的来源（读代码 + FlashInfer 发布说明，推断）

1. FlashInfer 0.7.0.post1 的 SM120 DSV3.2/GLM 解码 kernel 遇到 -1 候选时，会把 KV cache 的第 0 个槽位读进来，再把它的注意力分数设成 -1e30 来屏蔽（`decode_dsv3_2_kernel.cuh` 第 400-403 行注释："The IO warp gathered slot 0 into smem for the -1 case (idx clamped to 0); setting qk = -inf kills the contribution in softmax"）。
2. 权重因此是 0，但乘 V 时 0 × NaN 仍是 NaN。只要第 0 个槽位里有 NaN 或 Inf，输出就被污染。
3. vLLM 的第 0 个槽位属于 null block（`BlockPool` 取出的第一个块，从不写入），内容是显存里的残留数据。
4. 第 3 节说明 GLM-5.3 的每个 query 都有 -1 候选，所以每个请求都会读到这个槽位，短 prompt 也不例外。
5. nightly 带的 FlashInfer 0.7.2.post1 改为让 -1 候选读一行共享的全 0 行，注释写明原因："Masked candidates gather the shared zero row, never a mutable cache slot: a NaN there would leak through 0 * NaN in the value MMA"（`common/kv_cache_io.cuh`、`kernels/dsv32_fp8/decode.cuh`、`common/zero_row.cuh`）。FlashInfer v0.7.1 发布说明中对应 #5075 "runtime KV row stride + canonical 528B GLM53_NOPE rows + NaN-safe masked gathers"。

未证实的部分：没有读出第 0 个槽位的实际内容，也没有排除 MoE、KDA 等模块同时有问题。要坐实只需一次组件替换实验：在 v0.32.0 镜像里只把 FlashInfer 升级到 0.7.2.post1，看输出是否恢复正常。

### 7.3 #58305 不是修好乱码的原因（读代码）

- #58305 在 nightly 里：git 上 `dbcf568821` 是 `edad8b03c` 的祖先；nightly 镜像中 `flashinfer_mla_sparse_sm120.py` 的 `get_fp8_ds_mla_row_bytes()` 对 NoPE 返回 528。
- 它只把 KV 行从 656 字节缩到 528 字节，去掉的 128 字节是全 0 的 rope 段，kernel 本来就不读。
- v0.32.0 自带的 FlashInfer 0.7.0.post1 中 GLM53_NOPE 的每 token 字节数就是 656（`_bytes_per_token_for_model_type` 返回 `_BPT_DSV3_2 = 656`），v0.32.0 的 vLLM 写的也是 656，两边一致，没有布局不一致的问题。
- #58305 是 vLLM 对 FlashInfer #5075 的跟进（同一作者）。FlashInfer 发布说明写明旧的 656 字节布局仍然可用。

## 8. 未解决的问题

1. **nightly 大 batch 性能**（实测，原因未查）：并发 64 时 nightly 平均每个请求 95.8 s，社区镜像 27.6 s；单个请求时 nightly 解码约 88 token/s 是正常的，问题出在大 batch。
2. v0.32.0 的 NaN 来源只做了代码分析，未做第 7.2 节末尾的组件替换实验。
3. 未测：MTP 投机解码（`--speculative-config '{"method":"mtp","num_speculative_tokens":5}'`）、65536 以上的上下文长度、多模态输入、工具调用。

## 9. 复现

nightly 启动命令（与本文所有 nightly 测试一致；GSM8K 时 `--max-num-seqs 64`）：

```bash
docker run -d --name glm53-test --gpus all --ipc=host --shm-size 32g -p 8001:8001 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e HF_HUB_OFFLINE=1 \
  -v /media/model-space:/models:ro -v $HOME/.cache/vllm-glm53-nightly:/root/.cache \
  vllm/vllm-openai:nightly /models/GLM-5.3-Flash \
  --served-model-name GLM-5.3-Flash --host 0.0.0.0 --port 8001 \
  --tensor-parallel-size 8 --max-num-seqs 16 --max-model-len 65536 \
  --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8 --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45
```

社区镜像使用相同参数，只把镜像换成 `cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1`。从启动到可服务：nightly 首次 472 s（含编译）、保留缓存目录后第二次 411 s；社区镜像 452 s。

用脚本复现（在 `/home/pese/qiuyu/intel-accel-for-llm-0901` 下）：

```bash
# 启动 + 数值探针 + 功能检查
IMAGE=vllm/vllm-openai:nightly bash glm53-sm120-test/run-glm53-image.sh
# 两个镜像依次跑 GSM8K 全量并对比
bash glm53-sm120-test/run-gsm8k.sh
```

各次测试的原始结果（vLLM 日志、每个请求和响应原文、GSM8K 逐题结果）在 `/home/pese/qiuyu/intel-accel-for-llm-0901/glm53-sm120-test/results/`：

| 目录 | 内容 |
|---|---|
| `v0.32.0_tp8_20261028_092512/` | 6.3 节 |
| `nightly_tp8_baseline_20261029_014824/` | 6.4 节 |
| `gsm8k_20261029_020634/` | 6.5 节 |
