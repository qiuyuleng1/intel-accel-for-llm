# KVShrink 运行中 DSA 报 `dsa batch failed` 的调查记录

记录时间：2026-11-03。本文把**实测事实**、**代码事实**和**推断**分开写；凡是推断都标了"推断"，没有被实验证实。

## 0. 结论摘要

- 2026-11-02 的 TP=8 正式测试中，K1（KVShrink）运行约 5.7 小时，vLLM 日志里出现 195 次 `dsa batch failed`。
- 状态码有 3 种：`0x5`（191 次）、`0x3`（3 次）、`0x83`（1 次）。按内核头文件定义，`0x3`/`0x83` 是"缺页（page fault）且未开启 block-on-fault"，`0x83` 表示缺页发生在写访问；`0x5` 是"批里至少一条描述符失败"，具体原因代码没有打印。
- 每次失败后，这一次拷贝会整批改用 CUDA 拷贝重做，所以 vLLM 没有报错退出，请求全部完成。
- 不是一开始就出错：第一次在 DSA 开始使用约 27 分钟后；之后集中出现在两段压力最大的测试里，最后 1.5 小时几乎持续出现、越来越多。
- 与 ScratchPool 不够用（`ScratchPool exhausted` 等待）没有明显关系。
- 读代码发现一个可疑点（推断，未证实）：GPU 显存映射"注册时用整段范围、查找时只用开头一小段"，可能选中一个覆盖不到整段的映射，使 DSA 访问到映射之外的地址而缺页。见第 5 节 H1。

## 1. 背景：DSA 在 KVShrink 里做什么

- **加载 KV（DDR → GPU）**：DDR 池里是压缩后的 KV → QAT/IAA 解压到 ScratchPool（每张卡一块锁页主机内存）→ DSA 把 ScratchPool 的数据拷到 GPU 显存。
- **保存 KV（GPU → DDR）**：方向相反，DSA 先把 GPU 显存拷到 ScratchPool，再压缩进 DDR 池。
- **DSA 怎么访问 GPU 显存**：DSA 不能直接访问显存。代码先用 GDRCopy（`gdr_pin_buffer` + `gdr_map`）把一段显存通过 GPU 的 BAR 映射成进程里的普通虚拟地址，DSA 再通过 PASID（共享进程虚拟地址）按这个地址读写。
- **出错后的处理**：[iaxl/csrc/kv_xfer/cuda.cpp](../../iaxl/csrc/kv_xfer/cuda.cpp) 第 257 行先调用 DSA 路径，如果返回失败，就对这一整批改用 `cudaMemcpy3DBatchAsync` 重做。`using CUDA backend` 只打印一次（`std::call_once`），之后每次拷贝仍然先尝试 DSA。

## 2. 环境（2026-11-02 正式测试时）

| 项 | 值 |
|---|---|
| CPU | Intel Xeon 6767P，2 个 NUMA 节点，256 个逻辑 CPU |
| 主机内存 | 503.5 GiB |
| GPU | 8 × NVIDIA RTX 6000D（84 GB），驱动 595.91.07；GPU 0–3 在 NUMA 0，4–7 在 NUMA 1 |
| 内核 | 6.8.0-136-generic，启动参数 `intel_iommu=on,sm_on iommu=pt` |
| GDRCopy 驱动 | gdrdrv 2.6 |
| 自动 NUMA 迁移 | `kernel.numa_balancing=0`（关） |
| 透明大页 | `madvise` |
| DSA 设备 | 每张卡一个 DSA 设备、一个 WQ：rank0→wq0.0（dsa0）… rank7→wq14.0（dsa14） |
| DSA WQ 配置（wq0.0，其它相同） | `mode=dedicated size=256 priority=10 block_on_fault=0 type=user max_transfer_size=2147483648 max_batch_size=1024 ats_disable=0 prs_disable=0`；`dsa0 pasid_enabled=1` |
| 代码 | 提交 `b37a1b6`；未提交的改动只在测试脚本、`tools/setup_qat.sh`、`examples/kvshrink-vllm-serve.sh`、`setvars.sh`，不涉及 `iaxl/csrc` |
| vLLM | 0.29.0（容器 `iaxl.vllm-qwen38-benchmark`，镜像 `vllm-iaxl-dev-qwen38-benchmark:1.5.1`） |
| 模型 | Qwen3.8-27B（48 层 GDN 线性注意力 + 16 层全注意力的混合模型），TP=8，block size 400 token，`--max-model-len 201128` |
| K1 开关 | `IAXL_DSA_GD_ENABLE=1 IAXL_KV_COMPRESSION=1 IAXL_QAT_ZIP_ENABLE=1 IAXL_IAA_ZIP_ENABLE=1 IAXL_CPU_ZIP_ENABLE=0 IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS=0` |
| 池大小 | `IAXL_DDR_POOL_TOTAL_SIZE_GB=120`（每卡 15 GB），`IAXL_SCRATCH_POOL_SIZE_GB=16`（每卡） |

## 3. 现象（实测）

数据目录：`/home/pese/qiuyu/intel-accel-for-llm-1030/bench-results/Qwen3.8-27B_tp8_20261102_095646/`（在 1030 仓库，未搬到本分支）。K1 只启动了一次 vLLM，先跑 100k 再跑 200k，每个并发依次跑命中率 80、95、100。

### 3.1 报错原文和状态码

| 报错原文 | 整个 K1 日志里的次数 | 落在正式测试时间段内的次数 | 含义（`/usr/include/linux/idxd.h`） |
|---|---|---|---|
| `dsa batch failed, status=0x5` | 191 | 190 | `DSA_COMP_BATCH_FAIL`：批里至少一条描述符失败 |
| `dsa batch failed, status=0x3` | 3 | 3 | `DSA_COMP_PAGE_FAULT_NOBOF`：缺页，未开 block-on-fault |
| `dsa batch failed, status=0x83` | 1 | 1 | 同 0x3，`0x80` 位（`DSA_COMP_STATUS_WRITE`）表示是写访问时缺页 |

"整个 K1 日志"比"正式测试时间段内"多 1 次 `0x5`，那一次发生在 200k 的预热阶段。

### 3.2 每次正式测试的次数

| 输入长度 | 并发 | 命中率 % | DSA 报错次数 | 测试计时时长 (s) |
|---|---|---|---|---|
| 100k | 8 | 80 / 95 / 100 | 4 / 1 / 0 | 591 / 262 / 157 |
| 100k | 16 | 80 / 95 / 100 | 0 / 0 / 0 | 1172 / 508 / 293 |
| 100k | 32 | 80 / 95 / 100 | 0 / 0 / **34** | 2308 / 991 / 558 |
| 200k | 8 | 80 / 95 / 100 | 3 / 6 / 0 | 1278 / 556 / 310 |
| 200k | 16 | 80 / 95 / 100 | **28 / 68 / 50** | 2558 / 1096 / 597 |

- "DSA 报错次数"：summary.csv 的 `dsa_batch_failed_count` 列，统计的是这次测试在 serve.log 里那一段（含该次测试的 5 个 vLLM 预热请求）。
- "测试计时时长"：该次测试结果 json 里的 `duration`，不含发请求前生成提示词的时间。

### 3.3 时间线

DSA 报错行本身没有时间戳，这里用它前面最近一条带时间的日志行的时间代替，误差在 10 秒以内。

| 时间 (11-02) | 在跑什么 | DSA 报错 |
|---|---|---|
| 09:56:54 | K1 的 vLLM 开始启动 | — |
| 10:00:16 | 第一次读写 KV，8 张卡的 ScratchPool 分配，DSA 开始被使用（100k 预热） | — |
| 10:00–10:21 | 100k 预热 | 0 |
| 10:21–10:39 | 100k 并发 8，命中率 80/95 | 4 / 1，**第一次在 10:27:41** |
| 10:39–12:41 | 100k 并发 8 命中率 100；并发 16 全部；并发 32 命中率 80、95 | 0 |
| 12:41–12:50 | 100k 并发 32 命中率 100 | 34（每分钟 2–7 次） |
| 12:50–13:11 | 200k 预热 | 1 |
| 13:11–14:01 | 200k 并发 8 | 3 / 6 / 0 |
| 14:01–15:38 | 200k 并发 16 | 28 / 68 / 50 |

200k 并发 16 期间每 10 分钟的次数依次为：1、2、4、5、13、7、32、32、13、37。15:28–15:37 几乎每分钟都有（3–13 次/分钟）。195 次分布在 112 个不同的秒里，不是在同一时刻成批打印。4 次非 `0x5` 的时间：12:43:57（0x3）、15:06:58（0x3）、15:25:23（0x83）、15:30:51（0x3）。

### 3.4 其它运行中的情况

| 运行 | DSA 报错 |
|---|---|
| TP=4 的 3 次正式测试尝试（ScratchPool 每卡 8 GB，`ScratchPool exhausted` 很多） | 0 |
| TP=8 ScratchPool 实验 `/home/pese/qiuyu/intel-accel-for-llm-1030/tests/bench_dev/results/scratch-tp8-in100000-20261102_091324`（脚本有误，请求几乎没命中，基本没有 ScratchPool 等待） | 1（`0x5`，09:25 左右） |
| TP=8 ScratchPool 实验 `/home/pese/qiuyu/intel-accel-for-llm-1030/tests/bench_dev/results/scratch-tp8-in100000-20261102_093418`（16 GB，32 个 100k 请求同时加载，ScratchPool 等待 1696 次） | 0 |

### 3.5 内核日志

`sudo dmesg -T` 从 10-23 开始，覆盖了 11-02 的整个运行。11-02 09:50–22:59 之间没有任何包含 `idxd`、`dmar`、`iommu`、`pasid`、`prq`、`page request` 的内核消息。

### 3.6 与 ScratchPool 等待的关系

- ScratchPool 不够时，Python 在 [iaxl/kvflow/flow.py](../../iaxl/kvflow/flow.py) 第 163 行 `_allocate()` 里等待，等到有空块才把地址交给 DSA。DSA 拿到的地址总是已分配好的锁页内存。
- K1 日志里 `ScratchPool exhausted` 共 17213 行，DSA 报错 195 次；其中 71 次报错前 20 行日志内没有任何 ScratchPool 等待。
- 上表中 ScratchPool 等待最多的两次实验（TP=4、TP=8 16 GB 实验）DSA 报错为 0。

结论：没有证据表明 DSA 报错由 ScratchPool 大小引起。

## 4. 代码事实

### 4.1 调用链

1. [iaxl/csrc/kv_xfer/cuda.cpp](../../iaxl/csrc/kv_xfer/cuda.cpp) 第 257 行：`copy_chunks_batch` 先调用 `dsa_copy_chunks_batch(...)`，返回 false 时改用 CUDA 拷贝。
2. [iaxl/csrc/kv_xfer/dsa.cpp](../../iaxl/csrc/kv_xfer/dsa.cpp)：
   - 第 49–56 行：算出本次涉及的 GPU 范围 `span`（本批最大块号 × `chunk_stride` + …）。
   - 第 57 行：`dsa_gd_default_register(gpu_base, span)`，确保有一个映射覆盖 `[gpu_base, gpu_base + span)`。
   - 第 61 行：`dsa_gd_default_gpu_bar_addr(gpu_base, inner_size, &gpu_bar_base)`，**只用 `inner_size`（一个块的大小）去查映射**，得到 `gpu_bar_base`。
   - 第 66–86 行：每个块的 GPU 地址 = `gpu_bar_base + chunk_index * chunk_stride + o * outer_block_size`，最远到 `gpu_base + span`。
   - 第 88 行：调用 `dsa_memcpy_batch`。
3. [iaxl/csrc/dsa_gd/dsa_memcpy.c](../../iaxl/csrc/dsa_gd/dsa_memcpy.c)：
   - 第 383 行：子描述符标志是 `CRAV | RCR | CC`，**没有 `BOF`（block on fault）**；WQ 本身 `block_on_fault=0`。所以遇到缺页时 DSA 直接报失败，不会等待缺页处理。
   - 第 393–397 行：一批只有 1 条描述符时，直接提交这一条，它的完成记录就是 `bcomp`，所以打印出来的是这条描述符自己的状态（`0x3`/`0x83`）。
   - 一批多于 1 条时，提交 batch 描述符，`bcomp` 只给出 `0x5`；各子描述符的完成记录在 `comp[j]` 里，**代码没有打印**。
   - 第 479–482 行：打印 `dsa batch failed, status=0x%x`，并让整个 `dsa_memcpy_batch` 返回 -1。

### 4.2 GPU 显存映射的管理（[iaxl/csrc/dsa_gd/dsa_gd.c](../../iaxl/csrc/dsa_gd/dsa_gd.c)）

- 映射保存在单向链表 `ctx->maps` 里。新映射插在**链表头**（第 166–167 行），之后**从不删除**（只有进程退出或 `dsa_gd_reset` 时才清空）。
- `find_enclosing(ctx, dptr, n)`（第 107 行）：从链表头开始，返回**第一个**满足 `m->va <= dptr && dptr + n <= m->va + m->map_size` 的映射。
- `mapping_for(ctx, dptr, n)`（第 114 行）：先 `find_enclosing`，找不到就新建一个覆盖 `[dptr 对齐到 GPU 页, dptr + n)` 的映射。
- `dsa_gd_default_register(gpu_base, span)` → `mapping_for(gpu_base, span)`。
- `dsa_gd_default_gpu_bar_addr(gpu_base, inner_size)` → `resolve` → `mapping_for(gpu_base, inner_size)`：从链表头找第一个覆盖 `[gpu_base, gpu_base + inner_size)` 的映射，返回 `m->bar + (gpu_base - m->va)`。

## 5. 推断与假设（均未证实）

### H1（最可疑）：查映射时范围用错，选中了覆盖不到整段的映射

推理过程：

1. 第 57 行注册时保证"存在某个映射覆盖整段 `[gpu_base, gpu_base + span)`"。
2. 第 61 行查找时只要求"覆盖开头一个块 `[gpu_base, gpu_base + inner_size)`"，并且返回链表里**第一个**满足条件的映射。
3. 如果链表头附近有一个**更新、但更短**的映射 M_small，它覆盖了 `gpu_base` 开头的一个块，却没覆盖到 `gpu_base + span`，查找就会返回 M_small。整段注册时找到的那个大映射 M_big 在链表后面，用不上。
4. 于是本批里块号较大的那些地址 = `M_small.bar + 偏移`，超出了 M_small 的 BAR 映射范围。DSA 访问到一段没有映射的虚拟地址 → 缺页 → `0x3`/`0x83`（单条）或 `0x5`（批）。

M_small 怎么来：只要某次对**另一个起始地址**（比如另一层的 KV 张量）注册时，现有映射都不能完整覆盖，就会新建映射插到链表头。只有在不同张量的 GPU 地址范围有重叠时，才会出现"新映射覆盖了本张量的开头"的情况。vLLM 对混合模型（注意力层 + mamba/GDN 层）可能让不同层共用同一块底层显存，各层只是不同的视图（**需要确认**）；KVShrink 对注意力层和 mamba 层都会做 DSA 拷贝（`label="mamba"`），它们的 `gpu_base` 和 `span` 不同。

和现象对得上的地方：

- 不是一开始就出：映射是随着用到的块号和层逐渐建立起来的，运行越久，链表里各种长短的映射越多。
- 压力大、长输入时更多：用到的块号范围更大，`span` 更长，更容易超出短映射。
- 读、写两种缺页都有：GPU→主机（保存）时 GPU 一侧是读，主机→GPU（加载）时 GPU 一侧是写。

**附带风险（推断）**：如果超出 M_small 的地址恰好落在另一段有效映射（例如另一个 BAR 映射）里，DSA 不会缺页，而是**静默读写错误的显存**。本次测试没有做输出正确性检查，这个风险没有被排除。

### H2：缺页发生在 ScratchPool 锁页内存一侧

ScratchPool 是 `torch.empty(..., pin_memory=True)` 一次性分配的锁页内存，地址固定，理论上不应缺页。可能性较低，但没有数据排除。

### H3：其他

IOMMU/ATS 地址转换失效、GDRCopy 映射数量或 BAR 空间相关的问题等。没有任何证据支持或排除；第 3.5 节显示内核日志里没有相关消息。

## 6. 建议的诊断（都还没做）

1. **打印失败描述符的详细信息**：在 [dsa_memcpy.c](../../iaxl/csrc/dsa_gd/dsa_memcpy.c) 第 481 行附近，`0x5` 时遍历这一批的 `comp[j]`，打印 `status`、`fault_addr`（完成记录里硬件写入的出错地址）、`bytes_completed`、该条的 `src`/`dst`/`xfer_size`；同时打印拷贝方向（H2D/D2H）。据此判断出错地址落在 ScratchPool 范围还是 GPU BAR 映射范围。
2. **直接检验 H1**：在 [dsa.cpp](../../iaxl/csrc/kv_xfer/dsa.cpp) 第 61 行之后，检查查到的映射是否覆盖 `[gpu_base, gpu_base + span)`；不覆盖时打印 `gpu_base`、`span`、`m->va`、`m->map_size` 并计数。如果"不覆盖"的次数和 DSA 失败一一对应，H1 基本成立。可以同时试验把第 61 行查找用的长度从 `inner_size` 改成 `span`，看报错是否消失。
3. **确认显存是否重叠**：打印 KVShrink 收到的每一层（含 mamba 层）张量的 `data_ptr()` 和字节数，看不同层的地址范围是否重叠。
4. **记录映射数量**：定期打印 `ctx->maps` 链表长度和每个映射的 `[va, va + map_size)`。

## 7. 复现方法

### 7.1 完整复现（已知会出现，但要跑几个小时）

在容器内、1030 仓库根目录（`/home/pese/qiuyu/intel-accel-for-llm-1030`，`kvshrink-benchmark.sh` 只在那里），按 11-02 的配置只跑 K1。注意要把 `kvshrink-benchmark.sh` 里的 `IAXL_DDR_POOL_TOTAL_SIZE_GB` / `IAXL_SCRATCH_POOL_SIZE_GB` 改回 120 / 16；当前脚本已改成 80 / 24。

```bash
CONFIGS=K1 TP_SIZE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    LEN_CONCURRENCY="100000:8,16,32 200000:8,16" bash kvshrink-benchmark.sh
```

结果里 `summary.csv` 的 `dsa_batch_failed_count` 列、每次测试的 `<tag>_dsa_batch_failed.log` 记录报错次数和原文。

### 7.2 较快的复现尝试（未验证能否复现）

- 只跑压力最大的组合：`CONFIGS=K1 LEN_CONCURRENCY="200000:16"`（11-02 时这一组三次测试共 146 次报错，但它是在 vLLM 已运行约 4 小时后跑的；新启动的 vLLM 能否很快复现，未知）。
- 如果 H1 成立，可以写一个不依赖 vLLM 的小程序确定性复现：在一块 GPU 缓冲区上先 `dsa_gd_default_register(X, 大范围)`，再对一个起始地址 Z < X、结尾在 X 之后但在 X + 大范围之前的区间注册，然后 `dsa_gd_default_gpu_bar_addr(X, 一个块)` 并用 DSA 拷贝 `[X, X + 大范围)` 的末尾，观察是否缺页。

### 7.3 不依赖 vLLM 的复现尝试（11-03，已做）

要回答的问题：不启动 vLLM、只用 KVStore 的 put/get，能不能出现同样的 DSA 报错。

程序：仓库自带的 `benchmark/kvstore/kvstore_benchmark.sh`，代码未改。每张卡一个进程。一次"运行"指：每个进程在 GPU 上生成 mock KV 数据，然后依次做 预热写入 → 预热读回 → 正式写入 → 正式读回，最后抽查每层 3 个块的数据。写入时 DSA 把 GPU 上的块拷到 ScratchPool 再压缩进 DDR 池，读回时方向相反。每次写入或读回是 16 层 × 4611 块 × 400 KB = 28.14 GiB，所以一次运行里每个进程经 DSA 搬运 112.6 GiB，约 30 秒。

配置：GPU 0–3（4 个进程，DSA 队列 wq0.0/wq2.0/wq4.0/wq6.0）；QAT+IAA+DSA 开；KV 形状 `2 9221 400 1 256` bf16（每块 400 KB，与 11-02 相同），16 层；ScratchPool 每卡 16 GB；DDR 池每卡 15 GB。命令（在容器内、仓库根目录）：

```bash
export TP_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 IAXL_SCRATCH_POOL_SIZE_GB=16 IAXL_DDR_POOL_TOTAL_SIZE_GB=60
bash benchmark/kvstore/kvstore_benchmark.sh --ranks 4 --qat --iaa --dsa --data-source mock \
    --kv-cache-shape 2 9221 400 1 256 --num-layers 16
```

结果（实测，`tests/dsa_pagefault_dev/results/dsa-repro-kvstore-20261103_025817/`，`run2.log` 和 `loop/iter01–20.log`，汇总在 `loop/summary.txt`）：

- 共 21 次运行，只有 1 次（`loop/iter05.log`）出现 DSA 报错：6 次，全部是 `0x5`。其余 20 次为 0。
- iter05 里 4 个进程各至少失败一次（`using CUDA backend` 打印了 4 次，它每个进程只打印一次、且只在 DSA 失败后才会走到）。
- iter05 的报错都在 03:15:04–03:15:08，即该进程第一次用 DSA（03:15:03.8 分配 ScratchPool）之后约 4 秒内。DDR 池 "Over budget 20970 MB"（≈ 一次写入压缩后的大小）在 03:15:08 之后才出现，所以报错发生在预热写入阶段（GPU → ScratchPool 方向）。
- 21 次运行的数据抽查都通过（失败的那批已整批改用 CUDA 重做）。
- 内核日志：测试期间没有 idxd/iommu 消息。03:14:58（iter05 报错前约 5 秒）有一条 `bash (2100208): drop_caches: 3`，是机器上别的任务执行的。11-02 当天内核日志里没有 drop_caches，所以它解释不了 11-02 的报错；它和 iter05 是否有关，未知。
- 测试期间机器上还有别人的 vLLM / lmcache 任务在 GPU 4、6、7 上运行，主机内存占用在 35–116 GiB 之间变化。

结论：不依赖 vLLM 可以出现同样的报错，但 21 次运行只出现 1 次，还不能稳定复现。

### 7.4 稳定复现：关压缩、只开 DSA，DSA 工作时让内核整理内存（11-03，已做）

背景：内存整理（memory compaction）是 Linux 内核做的：把普通进程内存（malloc 一类）搬到别的物理位置，凑出连续的大块空闲内存。触发者：申请大块连续内存的程序或驱动（当场整理）、内核后台线程 `kcompactd`（本机 `compaction_proactiveness=20`，主动整理开着）、root 写 `/proc/sys/vm/compact_memory`（本实验用这个人为制造）。

程序：同 7.3 的 kvstore benchmark，但只加 `--dsa`、不加 `--qat --iaa`，所以压缩关闭，QAT/IAA 不启动。脚本 `tests/dsa_pagefault_dev/dsa-repro-kvstore-loop.sh`：每次运行 4 个进程（GPU 0–3），最多 300 秒；`COMPACT_MEMORY=1` 时从第一次 DSA 拷贝开始反复执行 `echo 1 > /proc/sys/vm/compact_memory`，直到这次运行结束；每次运行记录整台机器在这段时间里搬动的页数（`/proc/vmstat` 的 `pgmigrate_success` 差值）。机器在实验前重启过，DSA 队列用 `tools/setup_dsa_cnt.sh` 重新配置（每设备 1 个独占队列，大小 256，`block_on_fault=0`，与 11-02 相同）。

| 条件 | 运行次数 | 有 DSA 报错的运行 | 每次运行的报错数 | 每次运行期间搬动的页数 |
|---|---|---|---|---|
| 不干扰（`dsa-repro-kvstore-dsaonly-drop_none-compact0-20261103_055254`） | 10 | 0 | 0 | 0–2360 |
| 反复整理内存（`dsa-repro-kvstore-dsaonly-drop_none-compact1-20261103_060109`） | 10 | **10** | 1–12，全部 `0x5` | 137 万–358 万 |

（目录都在 `tests/dsa_pagefault_dev/results/` 下，每个目录的 `summary.txt` 每次运行一行。）

结论（实测）：DSA 工作时内核整理内存，DSA 报错稳定出现（10/10）；不整理时 10/10 不出。所有运行的数据抽查都通过。

推断（未证实）：内核搬页时会先把页表项改成"正在迁移"，这段时间里 DSA 按虚拟地址访问会缺页；队列 `block_on_fault=0`，DSA 不等待，直接报错。`0x5` 表示批里某条描述符失败并写下了自己的完成记录；如果是读描述符数组时缺页，应为 `0x6`（`DSA_COMP_BATCH_PAGE_FAULT`），从未出现。所以缺页的应是完成记录、ScratchPool 或 GPU 显存映射之一，具体是哪个要看出错地址（下一步：在 `dsa_memcpy.c` 出错时打印每条失败描述符的 `fault_addr`、源/目标地址，以及拷贝方向）。另外本机 `compact_unevictable_allowed=1`，即内核整理时连 `mlock` 锁住的页也可以搬，所以只 `mlock` 不能保证不被搬。

### 7.5 出错地址和 block_on_fault=1（11-03，已做）

出错地址（实测）：在 `dsa_memcpy.c` 加了出错时的打印（每条失败描述符的状态、`fault_addr`、`bytes_completed`、源/目标地址；`dsa.cpp` 打印方向）。同 7.4 的条件（关压缩、反复整理内存）跑 5 次（`dsa-repro-kvstore-dsaonly-drop_none-compact1-20261103_061514`），5/5 出错，共 31 条描述符失败：GPU→主机 18 条，状态 `0x83`（写缺页），出错地址 = 目标地址 + 已拷字节数；主机→GPU 13 条，状态 `0x3`（读缺页），出错地址 = 源地址 + 已拷字节数。31 条全部落在主机一侧的 ScratchPool 块里，GPU 一侧和完成记录都是 0 条。

ScratchPool 内存类型（实测）：容器里用与 ScratchPool 相同的 `torch.empty(..., pin_memory=True)` 申请 1 GiB，`/proc/self/smaps` 显示 `rw-s ... /dev/zero (deleted)`、`Anonymous: 0 kB`、`Locked: 0 kB`，即 torch 2.13 的锁页内存是共享映射（shmem），不是进程私有内存。

block_on_fault=1（实测）：`tools/setup_dsa.sh` 增加第 2 个参数，`./tools/setup_dsa_cnt.sh 1 1` 把队列设为 `block_on_fault=1`；`dsa_memcpy.c` 在队列开启时给每条拷贝描述符加 `IDXD_OP_FLAG_BOF`（加在 batch 描述符上会被硬件拒绝，状态 `0x11` = `DSA_COMP_INVALID_FLAGS`，所以只加在拷贝描述符上）。同样条件跑 3 次（`dsa-repro-kvstore-dsaonly-bof1b-compact1-20261103_063659`）：

| 条件 | 运行次数 | 有 DSA 报错的运行 | 写入 GiB/s 均值（范围） | 读回 GiB/s 均值（范围） |
|---|---|---|---|---|
| bof=0，不整理 | 10 | 0 | 10.0（9.2–11.2） | 68.3（62.8–69.9） |
| bof=0，反复整理 | 10 | 10 | 11.0（10.1–12.6） | 67.9（65.2–70.2） |
| bof=1，反复整理 | 3 | 0 | 11.2（10.2–12.7） | 68.1（66.6–69.0） |

（写入/读回：benchmark 正式那一次 PUT/GET 的 4 个进程合计速度，取最慢进程的时间；关压缩时写入还包括把原始数据拷进 DDR 池，所以比读回慢。）

结论（实测）：打开 block_on_fault 后，同样的整理内存条件下不再报错，速度没有可见变化。这说明这些缺页是内核能处理的临时缺页，不是错误地址。

### 7.6 验证"内核会去搬 torch 锁页内存"及修法 A（11-03，已做）

验证 1（不用 DSA，`tests/dsa_pagefault_dev/pinned-compaction-check.py`，容器内 root 运行）：每种方式申请 8 GiB（2,097,152 个 4 KB 页），整理一次内存，记录整台机器 `/proc/vmstat` 的 `pgmigrate_fail`（搬页失败次数）增量，并用 `/proc/self/pagemap` 抽查 8192 个页整理前后的物理页号。`pgmigrate_fail` 是整台机器的计数，包括别的进程的页，所以先测"不申请"时的值作为背景。

| 申请方式 | smaps 显示 | 一次整理的 `pgmigrate_fail` 增量（两轮） | 抽查页物理地址变化 |
|---|---|---|---|
| 不申请（背景） | — | 3,153 / 3,153 | — |
| `torch.empty(pin_memory=True)`（ScratchPool 现状） | `rw-s /dev/zero`，Anonymous 0 | 2,078,015 / 1,512,568 | 0 / 8192 |
| 同上，加 `PYTORCH_CUDA_ALLOC_CONF=pinned_use_cuda_host_register:True` | `rw-p`，Anonymous 8 GiB | 4,667 / 242 | 0 / 8192 |

验证 2（`dsa-repro-kvstore-dsaonly-hostreg-compact1-20261103_064418`）：队列改回 `block_on_fault=0`，条件同 7.4（关压缩、反复整理内存），只加 `PINNED_HOST_REGISTER=1`：3 次运行 0 次 DSA 报错，0 次退回 CUDA；写入 13.8 / 11.8 / 10.7 GiB/s，读回 69.8 / 68.6 / 69.5 GiB/s，与 7.5 表中 bof=0 不整理时（写入 10.0、读回 68.3 均值）相当。

结论：torch 锁页内存（共享映射）在内核整理内存时会被尝试搬移并失败，抽查到的页物理地址不变；改为私有内存 + `cudaHostRegister` 后，失败次数回到背景水平，DSA 报错消失。推断（未用更细的手段证实）：每次"尝试搬移后失败"期间，该页的页表项被临时替换，DSA 在 `block_on_fault=0` 时撞上即报缺页。若要逐页证实，可用内核 tracepoint `migrate:set_migration_pte`（记录被替换页表项的虚拟地址）按 ScratchPool 地址范围过滤。

## 8. 相关文件

| 内容 | 路径 |
|---|---|
| K1 完整 vLLM 日志（195 次报错都在这里，含 `\r`，用 `tr -d '\r'` 处理后再 grep） | `/home/pese/qiuyu/intel-accel-for-llm-1030/bench-results/Qwen3.8-27B_tp8_20261102_095646/K1/serve.log` |
| 每次测试摘出的报错原文 | `/home/pese/qiuyu/intel-accel-for-llm-1030/bench-results/Qwen3.8-27B_tp8_20261102_095646/K1/in<长度>/<tag>_dsa_batch_failed.log` |
| 每次测试的结果和统计 | `/home/pese/qiuyu/intel-accel-for-llm-1030/bench-results/Qwen3.8-27B_tp8_20261102_095646/summary.csv`、`K1/in<长度>/<tag>.json` |
| 测试脚本 | `/home/pese/qiuyu/intel-accel-for-llm-1030/kvshrink-benchmark.sh` |
| 不依赖 vLLM 的复现脚本和结果（本分支） | `tests/dsa_pagefault_dev/`、`tests/dsa_pagefault_dev/results/` |
| DSA 拷贝入口 | `iaxl/csrc/kv_xfer/dsa.cpp`、`iaxl/csrc/kv_xfer/cuda.cpp` |
| DSA 提交和等待 | `iaxl/csrc/dsa_gd/dsa_memcpy.c` |
| GPU 显存映射（GDRCopy） | `iaxl/csrc/dsa_gd/dsa_gd.c` |
| ScratchPool | `iaxl/kvflow/scratch_pool.py`、`iaxl/kvflow/flow.py` |
