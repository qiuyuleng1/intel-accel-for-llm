# DSv4 支持工作记录（intel-accel-for-llm-0901 / iaxl，vLLM 0.23.0）

> 目标：把 `/home/pese/qiuyu/Al-dsv4/KVCacheClip`（kvclip，vLLM 0.20.2，**已跑通**）
> 里对 **DeepSeek-V4-Flash（DSv4）** 的支持，**原样搬迁 + 适配**到本仓库
> （kvshrink connector，vLLM 0.23.0）。**不重新设计**，沿用已验证的方案：
> per-layer hook（打 patch）+ SupportsHMA + group_block_ratio + 异步存取。
>
> 所有改动留在本仓库，rsync 到 **H20**（8×H20 sm_90）测试；不在本地 6000D 测。

---

## 0. 硬约束

- 参考实现：`Al-dsv4/KVCacheClip/kvclip/integration/vllm/v1/kvclip_connector.py`
  + `kvclip/integration/vllm/patch/v0.20.2/dsv4-layer-hook.patch`。忽略其中 kvfuse。
- H20：`ubuntu@118.195.144.97`，`/data/qiuyu/Al-dsv4/`，key `/home/pese/qiuyu/new_id_rsa_h20`，
  代理 SOCKS4A `proxy-shz.intel.com:1080`，模型 `/ssd/hf_models/DeepSeek-V4-Flash`。
  H20 原生支持 FlashMLA/DeepGemm/TileLang → **不需要** Triton fallback 补丁。
- 关闭 QAT/IAA 压缩、DSA 传输；本次 bring-up 用 `IAXL_KV_COMPRESSION=0`（raw）。
- 参考 `run_vllm0.20.2_dsv4_kvshrink_h20.sh`（H20 版）；`_6000d.sh` 里的 Triton 补丁 H20 不需要。

---

## 1. KVCacheClip 方案（被搬迁的原型）

| 组件 | 做法 |
|------|------|
| Connector | `KVClipConnector(KVConnectorBase_V1, SupportsHMA)` |
| DSv4 检测 | `register_kv_caches` 里 `len(shapes)>1 and use_mla` → `_dsv4_mode=True` |
| 存哪些 | 167 个 tensor 中存 105 个，剔除 `compressor.state_cache`（block 边界为 0，可重算） |
| per-layer hook | **patch** `deepseek_v4_attention`，在 `attention_impl()` 前后插
  `connector.wait_for_layer_load()` / `save_kv_layer()`（原生不走 `@maybe_transfer_kv_layer`） |
| HMA | `SupportsHMA` + `request_finished_all_groups` + `_group_block_ratio`
  （hash_bs/group_bs：MLA=1、SWA=4…），多组 block_ids 分片 |
| SWA chunk 标签 | `f"{h}:{r}"`（1 个 hash → ratio 个 GPU block）；MLA 用裸 `h` |
| 存/取 | load：每 HMA group 一次异步 `get()`；save：每 storable key 一次异步 `put()`；
  block 全部 43 层写完后 `finish()` 写 presence 记录（裸 `h`），scheduler `has()` 读之 |
| patch 应用 | `kvclip/integration/vllm/vllm-serve.sh`：`patch -p1 -d $VLLM_SITE_PACKAGES < dsv4-layer-hook.patch` |

---

## 2. vLLM 0.23.0 关键差异（适配点）

| 方面 | 0.20.2（kvclip） | 0.23.0（本仓库） |
|------|------------------|------------------|
| DSv4 模型位置 | `vllm/model_executor/layers/deepseek_v4_attention.py`（含 `deepseek_v4_attention` 自定义 op 函数） | `vllm/models/deepseek_v4/`（包）；**没有** `deepseek_v4_attention` op；attention 走 `DeepseekV4Attention.attention_impl`（`@eager_break_during_capture`） |
| 注意力层前缀 | `model.layers.X.attn` | 同（`self.attn = ...MLAAttention(prefix=f"{prefix}.attn")`） |
| storable key | `{prefix}` / `{prefix}.swa_cache` / `{prefix}.indexer.k_cache` | 同 |
| hook 施加方式 | `.patch` 文件改 op 函数 | **runtime monkey-patch** 包住 `attention_impl`（`kvshrink/dsv4_patch.py`），效果一致但不依赖源码行号，自动生效 |
| 存储后端 | kvclip `KVStore`/`TensorZip`（`hybrid=True`、`finish()`） | iaxl `KVStore`/`KVFlow`（本次给它加了 `hybrid` + `finish()`） |
| HMA API | `SupportsHMA` + `request_finished_all_groups` | 同（0.23.0 `kv_connector/v1/base.py` 已有） |
| fp8 | `--kv-cache-dtype fp8` | 同（`attention.py` 断言 `fp8*`） |

---

## 3. 本仓库已落地的改动

### 3.1 iaxl 存储层（为承接 DSv4 而适配，等价于 kvclip 对 TensorZip 的改动）
- `iaxl/kvflow/flow.py`：`put()`/`get()` 的 `chunk_shape` 改为**逐 tensor** 计算，去掉
  「同 shape/dtype」硬断言 → 一次 `get()` 可传入一个 HMA group 内的**异构** shape
  （MLA `[N,64,584]` + indexer `[N,64,132]`）。`ScratchPool` 已按 `(shape,dtype)` 分桶。
  （这与 kvclip「Bug 修复 2026-05-27」对 TensorZip 的修复同源。）
- `iaxl/kvstore/kvstore.py`：`KVStore.__init__` 增加 `hybrid: bool`；hybrid 下 `put()`
  **不**自动 `put_finish`；新增 `finish(block_hashs)` = `put_finish(裸 hash)+record_flush`。

### 3.2 Connector 移植（`kvshrink/kvshrink_connector.py`）
- `class KVShrinkConnector(KVConnectorBase_V1, SupportsHMA)`。
- `__init__`：HMA `block_size` 覆盖（用 max group block_size=256 做 hash）；
  `_group_block_ratio`；`_is_dsv4`（config 检测）；DSv4 状态字段。
- `ReqMeta`/`RequestMetadata`：增加 `all_group_block_ids`。
- `get_num_new_matched_tokens`：DSv4 强制 `use_async=False`（走 per-layer hook）。
- `update_state_after_alloc` / `_add_request_to_save` / `build_connector_meta`：按
  `_group_block_ratio` 分片/展开多组 block_ids（含 kvclip 的「跳过部分块」逻辑）。
- `request_finished_all_groups`：委托 `request_finished`。
- `register_kv_caches` → DSv4 分支 `_register_dsv4_kv_caches`：剔除 compressor state、
  建 `_layer_to_group` / `_layer_name_to_storable_keys` / `_dsv4_group_keys`、
  `KVStore(hybrid=True)`、并安装 attention hook。
- DSv4 worker helpers：`_dsv4_start_load_kv`（按 group 合并异步 get）、
  `_dsv4_group_slots`（hash→block_indices/chunk_labels，MLA 裸 hash / SWA `h:r`）、
  `_dsv4_wait_for_layer_load`（按层 keys `get_wait`）、`_dsv4_save_kv_layer`
  （按 key 异步 put + 记录 save 进度）、`_dsv4_wait_for_save`（drain put + 满 43 层 `finish()`）。
- `get_finished`：DSv4 分支——save 已在 wait_for_save 同步 drain，finished 直接回报。
- **非 DSv4 路径完全保留**（早返回分支，不影响原有普通模型逻辑）。

### 3.3 per-layer hook（0.23.0 版 dsv4-layer-hook）
- `kvshrink/dsv4_patch.py`：`install_dsv4_attention_hook()` 在 worker 上 monkey-patch
  `DeepseekV4Attention.attention_impl`，前后插 `wait_for_layer_load(self.prefix)` /
  `save_kv_layer(self.prefix, None, None)`（`has_kv_transfer_group`+`is_v1_kv_transfer_group`
  +`has_connector_metadata` 守卫）。在 `_register_dsv4_kv_caches` 里安装（幂等）。
  需 `--enforce-eager`（DSv4 部署本就用它）。

### 3.4 运行脚本
- `examples/kvshrink-vllm-serve-dsv4.sh`（容器内）：`--kv-cache-dtype fp8 --enforce-eager
  --no-enable-prefix-caching --no-disable-hybrid-kv-cache-manager`（HMA 开），不传 `--block-size`。
- `run_dsv4_kvshrink_h20.sh`（H20 host）：`MODEL`/`TP=4`，关闭
  `IAXL_KV_COMPRESSION/QAT/IAA/DSA`（保留 CPU raw 拷贝 worker），base 镜像 `vllm/vllm-openai:v0.23.0`，
  调 `start.sh` 起容器。

---

## 4. 当前进度

| 项 | 状态 |
|----|------|
| 重新研读 KVCacheClip 完整实现 + patch + 应用机制 | ✅ |
| iaxl KVFlow 异构 shape | ✅ |
| iaxl KVStore hybrid + finish | ✅ |
| Connector 移植（SupportsHMA + per-layer + HMA 分组） | ✅（py 编译通过） |
| 0.23.0 attention hook（dsv4_patch.py） | ✅ |
| serve/run 脚本 | ✅ |
| rsync 到 H20 + 测试 | ⬜（待与用户一起做；H20 无 0.23.0 镜像但可联网） |

---

## 5. 风险 / 待确认（H20 上核实）

1. **0.23.0 镜像**：H20 现有 `v0.20.2`，无带 DSv4 的 `vllm/vllm-openai:v0.23.0`；机器可联网，
   需构建 iaxl dev 镜像（base 需带 DSv4 的 0.23.0）。
2. **`self.prefix` 精确值**：hook 用 `self.prefix` 作 layer_name，连 connector 的
   `_layer_name_to_storable_keys`（由实际 kv_caches key 去后缀推得）。已确认 0.20.2 日志
   前缀为 `model.layers.X.attn`；0.23.0 `model.py` 亦为 `prefix=f"{prefix}.attn"`。仍需实测确认。
3. **HMA group 与 storable key 的映射**：`_layer_to_group` 来自
   `kv_cache_config.kv_cache_groups[*].layer_names`，需确认其 key 与 register 收到的
   kv_caches key 命名一致（否则 group_block_ratio 落空 → SWA 分片错误）。实测校验。
4. **monkey-patch 时机**：在 worker `register_kv_caches` 安装，早于真实请求 forward；
   dummy/profiling forward 无 connector metadata → hook no-op。需日志确认 hook 已安装。
5. **`finish()` presence 与 `get()` chunk 一致性**：presence 记录键=裸 `h`；chunk 键=`h`/`h:r`。
   实测确认第二次相同 prompt 命中（`externally-cached tokens>0`）且输出正确。

---

## 6. 测试 / 同步流程（备忘）

```bash
# rsync（容器 bind mount 用 --inplace）
rsync -avz --inplace \
  -e "ssh -i /home/pese/qiuyu/new_id_rsa_h20 -o ProxyCommand='socat - SOCKS4A:proxy-shz.intel.com:%h:%p,socksport=1080'" \
  /home/pese/qiuyu/intel-accel-for-llm-0901/ \
  ubuntu@118.195.144.97:/data/qiuyu/intel-accel-for-llm-0901/
```

H20 上：`bash run_dsv4_kvshrink_h20.sh` → 容器内 `bash examples/kvshrink-vllm-serve-dsv4.sh`
→ 另一终端 `docker exec ... ; bash tests/vllm-test.sh`。验证：启动不 crash；相同长 prompt
第二次 `externally-cached tokens>0`；输出与 baseline 一致；日志有 `Installed DSv4 attention
KV-transfer hook` 与 `DSv4: N storable KV caches`。

---

## 7. H20 部署与测试进度（2026-09-02，下班存档）

### 7.1 连接 H20 的正确姿势（**重要，之前踩坑**）
- **不要**用 `eval "ssh ... $SSHOPT ..."` 拼 ProxyCommand——嵌套引号会破坏 `%h:%p`，
  导致连接落到**本地 6000D**（`h8263-6000d`，RTX 6000D），不是 H20！
- 已建 SSH config：`/home/pese/qiuyu/ssh_h20.conf`（Host `h20` → 118.195.144.97 + socat 代理）。
  统一用 `ssh -F /home/pese/qiuyu/ssh_h20.conf h20 ...`，rsync 用 `-e "ssh -F /home/pese/qiuyu/ssh_h20.conf"`，目标 `h20:/data/qiuyu/intel-accel-for-llm-0901/`。
- 已验证：真 H20 = `aice-h20-97`，8×H20 96GB（当前全空闲），`/ssd/hf_models/DeepSeek-V4-Flash` 在位。

### 7.2 镜像构建（已完成）
- `docker/Dockerfile.dev` 已加 `ARG PIP_INDEX_URL`（默认 pypi）。H20 在国内，直连 PyPI 下大 wheel
  （cmake 30MB）会**卡死**。用 aliyun 镜像即可：
  ```bash
  ssh -F /home/pese/qiuyu/ssh_h20.conf h20 'cd /data/qiuyu/intel-accel-for-llm-0901 && \
    docker build --network host -f docker/Dockerfile.dev -t vllm-iaxl-dev . \
    --build-arg BASE=vllm/vllm-openai:v0.23.0 \
    --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/'
  ```
- ✅ 镜像 `vllm-iaxl-dev:latest`（21.4GB）已在 H20 build 成功。

### 7.3 起服务（detached 容器，已跑通启动）
- docker 无 `nvidia` named runtime，**用 `--gpus all`（不要 `--runtime nvidia`）**。
- 启动脚本模板已存于 H20 `/tmp/launch_dsv4.sh`（容器名 `iaxl.dsv4`，端口 8010，TP=4，
  `pip install -e .` 后 `exec bash examples/kvshrink-vllm-serve-dsv4.sh`，log 落
  `/data/qiuyu/intel-accel-for-llm-0901/log.kvshrink-vllm-dsv4`）。
- ✅ **服务成功启动**（`Application startup complete`），移植机制全部正确（MEASURED 日志）：
  - `[HMA] _group_block_ratio: {0:1, 1:4, 2:4, 3:64, 4:32}`
  - `Installed DSv4 attention KV-transfer hook`（4 个 worker 都装上）
  - `DSv4: 105 storable KV caches (dropped 62 compressor states), 43 op-layers, 6 shapes,
    num_blocks=45706, groups={0:62, 1:22, 2:21}`
  - 第一次请求 `externally-cached tokens: 0` 正确。

### 7.4 ⛔ 当前阻塞点（明天从这里继续）
**forward 阶段崩溃**，EngineCore 500。根因（MEASURED，日志堆栈）：
```
AssertionError: all GPU tensors must be contiguous
  at iaxl/kvflow/flow.py:175  (put())
  <- kvshrink_connector._dsv4_save_kv_layer -> KVStore.put
```
- 即 DSv4 在 **HMA 下部分 KV-cache tensor 是非连续（strided）视图**（HMA tensor-sharing），
  而 iaxl `KVFlow.put/get` 里的 `assert tensor.is_contiguous()` 过严，直接挡下。
- **对照 KVCacheClip**：其 C++ `cuda_xfer.h` 明确支持 strided GPU tensor
  （"GPU (strided) → CPU (contiguous)"），所以旧方案能跑。
- **iaxl 侧调研**：iaxl 的 C++（`iaxl/csrc/torch_ext/context.h`）用 `cudaMemcpy2DAsync`，
  `chunk_stride = tensor.stride(chunk_dim=0)*elsize`，**能处理"chunk 维 strided"**；但
  `inner_size` 由 sizes（非 strides）算，**假设内层维连续**。因此能否直接放开 assert，
  取决于 DSv4 那些非连续 tensor 的**确切 stride 形态**。
- **下一步（已做到一半）**：在 `_dsv4_save_kv_layer`/`KVFlow.put` 加了 stride/contiguity
  诊断日志，准备重启服务、抓真实 stride，判断：
  - 若只是「chunk 维(dim0) strided、内层连续」→ 直接把 `is_contiguous()` assert 改成
    「内层连续即可」，走 memcpy2D 路径（最省）。
  - 若内层也不连续 → 在 put/get 里对该 tensor 做 `.contiguous()`（多一次 copy）或按 KVCacheClip
    的 strided 拷贝实现补齐。
- ⚠️ 注意：该诊断改动可能还在本地未 rsync / 未重编。明天先 `git diff` 看 flow.py 当前状态。

### 7.5 明天 quickstart
```bash
# 1) 连 H20，看容器/日志
ssh -F /home/pese/qiuyu/ssh_h20.conf h20 'docker ps -a --filter name=iaxl.dsv4; \
  tail -50 /data/qiuyu/intel-accel-for-llm-0901/log.kvshrink-vllm-dsv4'
# 2) 改完 flow.py 后：本地 rsync（--inplace）→ 重启容器内 serve（editable install，Python 改动 rsync 即生效；
#    iaxl 是 pip install -e . 的 C++ 扩展，若改了 .py 直接生效，改了 C++ 需容器内重编）
rsync -az --inplace -e "ssh -F /home/pese/qiuyu/ssh_h20.conf" \
  /home/pese/qiuyu/intel-accel-for-llm-0901/ h20:/data/qiuyu/intel-accel-for-llm-0901/
# 3) 重启 serve：docker exec iaxl.dsv4 里 Ctrl+C 杀 serve 后重跑 examples/kvshrink-vllm-serve-dsv4.sh，
#    或 docker rm -f iaxl.dsv4 && bash /tmp/launch_dsv4.sh
# 4) 测试：curl 长 prompt(>256 tok) 两次，验证第二次 externally-cached tokens>0 且输出一致
```

### 7.6 本地遗留清理（可选）
- 本地 6000D（`h8263-6000d`）上误 build 了一个 `vllm-iaxl-dev` 镜像（之前连错机器所致），
  确认无用后可 `docker rmi vllm-iaxl-dev` 清掉。

---

## 8. 打通 + 精度验证（2026-09-03）——PORT 完成

> 7.4 标记的 forward 崩溃已解决。共修 3 个 bug，服务打通、缓存命中正确、GSM8K 已测。

### 8.1 三个 bug（全部 FIXED）

**Bug 1 — 非连续 tensor 断言过严（`iaxl/kvflow/flow.py`）**
- MEASURED（H20 日志）：**全部 105/105 storable cache 都非连续**。典型
  `indexer.k_cache [45706,64,132] stride=(8640,132,1)`——dim0 被 HMA padding（8640 > 自然 8448），
  但**内层连续**（per_block_contiguous=True）。根因：HMA 把每个 block slot pad 到公共 page_size，
  使 `stride(0)` 大于自然值。**不止 SWA，是全部 cache**。
- FIX：`put()`/`get()` 里把 `assert tensor.is_contiguous()` 放宽为
  `if not contiguous: assert chunk_dim==0 and tensor.select(0,0).is_contiguous()`。
  正确性依据：kv_xfer `cuda.cpp:340` 的 `cudaMemcpy2DAsync` 按 `chunk_stride=stride(0)` 逐 chunk
  拷贝 `inner_size` 字节，dim0 strided、内层连续正是它能处理的形态。

**Bug 2 — KV cache label 分隔符冲突（`kvshrink/kvshrink_connector.py`）**
- 真实请求崩 worker：`Invalid KV cache label: <hash>:0`。根因：iaxl `kv_pool.h` 用 `LABEL_SEP=':'`，
  `validate_label_component()` **拒绝**含 `:` `/` `\` 的 label。我的 SWA 子标签 `f"{h}:{r}"`
  （抄自 KVCacheClip，其 C++ 没把 `:` 设为保留符）撞车。
- FIX：`_dsv4_group_slots` 里 SWA 子标签 `f"{h}:{r}"` → **`f"{h}_{r}"`**。

**Bug 3 — warm 掉点：需保存 C4A 压缩器状态（`kvshrink/kvshrink_connector.py`）**
- 现象：warm（读 DDR）比 cold 掉 1.5 个点（见 8.2）。根因：我最初把**全部 62 个**
  `compressor.state_cache` 都当"可重算"丢弃，但 KVCacheClip 实际会存 **C4A** 那部分状态
  （其默认 `KVCLIP_SAVE_COMPRESSOR_STATE=c4a`，我据过时文档抄漏了）。
- FIX：移植 `_keep_storable`——保留 `compressor.state_cache` 中 `tensor.shape[1]==4` 的 C4A 状态
  （block_size=4），丢弃 `shape[1]==8` 的 C128A（block_size=8）。模块级
  `KVSHRINK_SAVE_COMPRESSOR_STATE` 默认 `"c4a"`。storable 从 105 → **147**。
- 诊断确认（MEASURED）：`c4a -> 147 storable (state caches kept: 42)`，
  C4A `(45706,4,512)×21 + (45706,4,2048)×21` 保留，C128A `(45706,8,1024)×20` 丢弃。

### 8.2 GSM8K 精度（MEASURED，200 题，fp8 KV，parallel=50，evalscope）

首轮 cold/warm 对比（c4a 修复前 vs 后，TP4）：

| 轮次 | TP | 缓存 | 存 C4A state | 外部命中 | Accuracy |
|------|----|------|--------------|----------|----------|
| run1_cold | 4 | cold | 否（修复前） | — | **99.0%** |
| run2_warm | 4 | warm | 否（修复前） | 200/200, 102144 tok | **97.5%** |
| run3_cold | 4 | cold | 是（c4a） | — | **99.0%** |
| run4_warm | 4 | warm | 是（c4a） | 200/200, 102144 tok | **98.5%** |

方差复测（c4a 修复后，同一份 DDR 缓存反复 warm / 切 TP8）：

| 轮次 | TP | 缓存 | 外部命中 | Accuracy |
|------|----|------|----------|----------|
| run5_warm | 4 | warm | 200/200 | 98.5% |
| run6_warm | 4 | warm | 200/200 | **99.0%** |
| run7_warm | 4 | warm | 200/200 | 98.0% |
| run8_cold | 8 | cold | — | 98.5% |
| run9_warm | 8 | warm | 200/200 | 98.5% |
| run10_warm | 8 | warm | 200/200 | 98.5% |

- **列义**：TP=张量并行度；缓存=cold 空 DDR 冷启动/边算边写、warm 完全命中上轮 DDR；存 C4A state=
  是否搬运 C4A 压缩器状态；外部命中=从 DDR 读回并命中的请求数（分母 200）；Accuracy=evalscope main
  子集准确率（分母 200，0.5%=1 题）。
- **MEASURED**：C4A 修复把 warm 掉点从 1.5pt（99→97.5）收窄到 0.5pt（99→98.5）。TP4 warm 跨 4 轮
  在 **98.0%–99.0%** 波动（run6 触到 99）。**TP8：cold=warm=98.5% 三轮全相等**，warm 两轮 100% 命中
  （400/400 请求，204544 tokens）。
- **INFERRED（已坐实）**：残余 0.5% 是**运行间噪声**：warm 分布（98–99）与 cold（99）重叠，且 TP8 上
  cold 与 warm **逐点相等**——KV 回读路径**不引入系统性掉点**。TP4(~99) vs TP8(~98.5) 的差异来自不同
  TP 的 reduction 数值顺序 + fp8，属噪声带内，非缓存 bug。无需上 `all` 模式。
- **切 TP 必须清 DDR 缓存**（`_data/kvcache`）：不同 TP 分片不同，旧缓存不可复用。TP8 启动时
  `tensor_parallel_size=8`、`147 storable`、`num_blocks=64341`（比 TP4 的 45706 大，8 卡显存更多）。

### 8.3 结论
DSv4 在 iaxl / vLLM 0.23.0 上**已打通并验证（TP4 + TP8）**：服务正常启动、per-layer hook 生效、
HMA 分组存取正确、缓存 100% 命中、cold 精度无损。warm 精度 = cold（噪声范围内，TP8 上逐点相等）。
QAT/IAA 压缩与 DSA 传输全程关闭（`IAXL_KV_COMPRESSION=0`，CPU raw 拷贝）。

---

## 9. PR 拆分与分支策略（2026-09-03）

### 9.1 进上游 PR 的文件（5 个，`dsv4-dev` 分支）
只留 DSv4 功能本身、与机器无关、已实测验证的改动：
- `iaxl/kvflow/flow.py`（非连续 tensor 传输）
- `iaxl/kvstore/kvstore.py`（hybrid + finish）
- `kvshrink/kvshrink_connector.py`（SupportsHMA + per-layer + HMA 分组 + C4A state）
- `kvshrink/dsv4_patch.py`（runtime attention hook）
- `examples/kvshrink-vllm-serve-dsv4.sh`（容器内 serve 脚本，实测在跑的就是它）
- commit：`feat(kvshrink): support DeepSeek-V4-Flash hybrid MLA KV offload via SupportsHMA + per-layer hook`

### 9.2 不进上游 PR（仅留备份分支 `dsv4-h20-dev-backup`）
- `docker/Dockerfile.dev`：只加了可配置 `ARG PIP_INDEX_URL`（默认官方源不变），为国内镜像 build。
  与 DSv4 功能无关，不进 DSv4 PR。
- `run_dsv4_kvshrink_h20.sh`：**机器专属** bring-up 包装（名字带 h20，套 start.sh），且**未实际用它跑通**
  （实际用 detached 的 `/tmp/launch_dsv4.sh`，容器内同样 exec `examples/kvshrink-vllm-serve-dsv4.sh`）。
- `DSV4_SUPPORT_WORKLOG.md`：内部工作日志，不进上游。
- `iaxl/torch_ext.cpython-312-x86_64-linux-gnu.so`：编译产物，**任何分支都不提**。

### 9.3 分支
- `dsv4-dev`：干净的 PR 分支（上述 5 个文件）。
- `dsv4-h20-dev-backup`：**备份分支**，含全部工作成果（5 个 PR 文件 + Dockerfile.dev + run_h20 + 本日志），
  **不入 .so，不 merge 主分支**。
