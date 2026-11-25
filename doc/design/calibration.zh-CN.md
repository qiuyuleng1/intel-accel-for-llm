# 启动前校准：DSA 分界值与 QAT/IAA 分段大小

状态（2026-11-23）：

- 已定：需求（第 3 节）。
- 已确认：DSA 校准的设计（第 5 节）。代码已在 feat/dsa-auto-fallback-dev 上改好（11-23，没提交、没编译、没测试）。
- QAT/IAA 分段校准：设计见第 6 节，11-24 代码已写在 feat/dsa-auto-fallback-dev 上（没提交、没编译、没测试）。

## 1 背景

### 1.1 DSA 和 CUDA batch 拷贝的分界值

KVStore 在 GPU 上的 vLLM KV cache 和 CPU 内存池之间拷贝 KV 块：

- 读回 KV 到 GPU 是 H2D，存 KV 到 CPU 是 D2H；
- 开了 `IAXL_DSA_GD_ENABLE=1` 时，用 Intel DSA 拷贝（C++ `dsa_copy_chunks_batch`）；
- 没开时，用 CUDA batch 拷贝（C++ `copy_chunks_batch` 里的 `cudaMemcpy3DBatchAsync`）。

本文里"连续拷贝大小"指一个块在一层里连续存放的字节数，也就是 C++ 里的 `inner_size`。实测下来，连续拷贝小的时候 DSA 快，大的时候 CUDA batch 快。两者一样快的那个连续拷贝大小，叫"分界值"。分界值随机器、拷贝方向、同时在拷贝的 GPU 数而变。

6000d（8 张 RTX 6000D）上的实测：

- D2H 分界值一直是 256 KB；H2D 在 256、384、512 KB 之间变，有一次是 768 KB。
- GPU 0+1、2+3、4+5、6+7 每两张卡在同一个 PCIe 交换芯片下面，共用一条 Gen5 x16 上行链路。两张卡同方向同时拷贝时，每张卡大约只有一半带宽：DSA 约 24 GB/s，CUDA batch 约 27.5 GB/s；单卡单独拷贝时约 44–46 GB/s。所以只在一张卡上测出来的分界值，不代表所有 rank 一起拷贝时的情况。

几个模型的连续拷贝大小：

| 模型 | 并行方式 | vLLM 块大小（token） | 连续拷贝大小 |
|---|---|---|---|
| Qwen3.8-27B | TP=4 | 784 | 784 KB |
| Qwen3.8-27B | TP=8 | 400 | 400 KB |
| Qwen3-32B | TP=8 | 16 | 8 KB |

列说明：
- **模型**：vLLM 加载的模型。
- **并行方式**：tensor parallel 的 rank 数。
- **vLLM 块大小（token）**：vLLM 一个 KV cache 块里放多少个 token。
- **连续拷贝大小**：一个块在一层里连续存放的字节数，即 C++ 的 `inner_size`。这个值决定 DSA 和 CUDA batch 哪个快。

### 1.2 QAT/IAA 压缩的分段大小

`kv_zip` 压缩一个块时，可以把块切成多段，每段作为一个压缩请求发给 QAT/IAA。段的大小由 `IAXL_ZIP_SEGMENT_SIZE_KB` 决定（第 6.1 节）。段大小对吞吐影响很大，用户实测见第 6.2 节。

## 2 之前的实现（已放弃）

做法：

- vLLM 启动后，各 rank 的 KVStore 先等所有 TP rank 起来；
- rank 0 在 C++ 里测分界值（`dsa_calibrate_crossover`），通过 KVStore 的 HTTP 接口（`/v1/dsa/crossover`）发给其它 rank；
- 由开关 `IAXL_DSA_AUTO_FALLBACK_ENABLE` 控制开不开。

代码存档：

```text
开发分支：feat/dsa-auto-fallback-dev，提交 0233c5d、fd9e98e
PR 分支：feat/dsa-auto-fallback，提交 707f142、7636963、01e98e6
旧设计文档：doc/design/dsa-auto-fallback.zh-CN.md（不再更新）
```

放弃的原因：11-23 用户和同事讨论后认为太复杂，C++ 和 KVStore 里到处都是校准代码。

性能结论：Qwen3.8-27B，TP=4，不压缩，命中率 100%，并发 32，共 320 个请求。回退开和关各跑了 4 到 5 次，TTFT 没有稳定差异。

## 3 需求（用户 11-23 原话，按原顺序）

### 3.1 第一次（改方案）

> 和同事讨论过了，当前这个实现太复杂了，改成在 examples/kvshrink-vllm-serve.sh 的 vllm serve 前面跑 calibration，并且除了 calibrate dsa 的两个 crossnode，还要 calibrate qat 的 segment。
>
> 1. 跑一个简化的 dsa + cudamemcpy 的 benchmark（和当前的 calibration 类似），返回两个值，分别是 h2d 和 d2h，然后搞两个环境变量，把这俩值设置上，放到 dsa.h 还是哪里去用。不再有单独的 dsa_auto_fallback 环境变量，只要 dsa 开启，就要做这个 calibration。这个 calibration 脚本可以放 tools 下面
> 2. 因为不同 qat 的 segment 对于 qat 的 throughput 影响很大，所以我们加一个 qat（也许还有 iaa）的 segment calibration。benchmark/kvstore/kvstore_segment_sweep.sh 这个跑的时间太久了，可以参考 tests/iaxl_unittest.py 来设计 calibration 脚本。注意，这时候要返回一组值，而不是 dsa calibration 那样的单独一个值就可以。原因是这时候 vllm 还没有启动，我们不知道 vllm 的 kvcache 到底什么大小，比如才 64k 的话，这个 calibration 结果是 128k 最好，那结果毫无意义，所以我们希望返回 128k，32k，...，这时候我们就可以把 64k 拆成两个 32k 了。`export IAXL_ZIP_SEGMENT_SIZE_KB=${IAXL_ZIP_SEGMENT_SIZE_KB:-0}` 你可以通过这个环境变量去找在哪里拆，然后可能代码里需要加个小的算法，来针对那一串返回值设计拆分的策略。相关的 calibration 代码也放 tools 下面
> 3. 你考虑一下吧，先和我讨论设计。先不要改代码。此外，feat/dsa-auto-fallback-dev 之前所有改动都提个 commit，我们代码做存档。

### 3.2 第二次（纠正我的第一版设计，见第 4 节）

> 1. 没有用户手动设置分界值这个选项。分界值都是 calibration 来的。
> 2. 你把 dsa_calibrate_crossover 这个函数删了，我们所有的代码改动差不多都要删了！是单独的一个 benchmark/tensor_xfer/tensor_xfer_benchmark.sh 脚本，放在 tools 下面！当前实现太复杂了，所以我们不在 c++ 和 kvstore 里到处做 calibration 了！而是单独的脚本。你不要考虑现有的函数了，推倒重来。
> 3. 测试肯定不是单卡测试啊！tp+dp 所有的同时来测。
>
> 你先别管 qat calibration 了，我们先把 dsa calibration 改好再说！

### 3.3 整理后的 DSA 校准需求

1. 校准是一个独立脚本，原型是 `benchmark/tensor_xfer/tensor_xfer_benchmark.sh`（DSA 和 CUDA batch 拷贝的对比 benchmark），做一个简化版，放在 `tools/` 下面。
2. 在 `examples/kvshrink-vllm-serve.sh` 里，`vllm serve` 之前运行。
3. 只要 DSA 开着（`IAXL_DSA_GD_ENABLE=1`）就运行；没有单独的开关，`IAXL_DSA_AUTO_FALLBACK_ENABLE` 删掉。
4. 输出两个值：H2D 分界值、D2H 分界值。serve 脚本把这两个值设成两个环境变量，C++ 的 DSA 代码读这两个变量来决定回退到 CUDA batch。
5. 没有手动设分界值的选项，分界值只来自校准。
6. TP×DP 的所有 rank 同时测，不是单卡测。
7. C++ 和 KVStore 里的校准代码全部删掉（`dsa_calibrate_crossover`、rank 0 通过 HTTP 发布等），推倒重来。

### 3.4 整理后的 QAT/IAA 分段校准需求（暂缓）

1. 加 QAT（可能还有 IAA）的分段大小校准，在 `vllm serve` 之前运行，代码放在 `tools/` 下面。
2. `benchmark/kvstore/kvstore_segment_sweep.sh` 太慢，参考 `tests/iaxl_unittest.py` 来设计。
3. 返回一组段大小（比如 128K、32K、…），不是一个值。原因：校准时 vLLM 还没启动，不知道块多大。例如块只有 64K、而校准最好的是 128K 时，128K 没法用，要能退到"拆成两个 32K"。
4. 拆分发生在 `IAXL_ZIP_SEGMENT_SIZE_KB` 被用到的地方（第 6.1 节），代码里加一个小算法，根据那一组值决定怎么拆。

## 4 讨论记录

### 4.1 11-23 第 1 轮：我的第一版 DSA 设计被否

| 我的第一版提议 | 用户的决定 |
|---|---|
| 用户手动设了分界值的环境变量时，跳过校准 | 没有手动设置这个选项，分界值只来自校准 |
| tools 下的 Python 脚本调用现有的 C++ 函数 `dsa_calibrate_crossover` | 删掉这个函数，之前的代码改动基本都删；以 `tensor_xfer_benchmark.sh` 为原型写一个独立脚本，放在 tools 下 |
| 只在 GPU0 上单独测（或者所有 TP 卡同时测，二选一） | 必须 TP×DP 所有 rank 同时测 |
| 同时给出了 QAT/IAA 分段校准的方案和 6 个问题 | 先不管 QAT，先做 DSA |

列说明：
- **我的第一版提议**：11-23 第一次回复里提出的做法。
- **用户的决定**：用户对这条的回复，以后按这一列执行。

### 4.2 11-23 第 2 轮：第二版草案的问题和用户的决定

| 问题 | 用户的决定 |
|---|---|
| 测出分界值后，还测不测更大的大小 | 不测：某个方向测出分界值就停，接着测另一个方向（和之前的 C++ 校准一样） |
| 校准进程要不要把两个分界值设成 0，防止测 DSA 时被回退判断转去走 CUDA | 不需要：校准在 serve 脚本 export 这两个变量之前运行，又没有手动设置的选项，所以校准时这两个变量不存在，C++ 读到 0 |
| 脚本放在哪 | `tools/calibration/` |
| 环境变量名 | `IAXL_DSA_H2D_CROSSOVER_KB` / `IAXL_DSA_D2H_CROSSOVER_KB`，单位 KB，0 表示不回退 |
| 所有 rank 怎么合成一个值 | 每个大小取各 rank 的平均 GB/s |
| 计时方式 | 和之前的 C++ 校准代码一致，不改（见 5.2 节）。我提议的"两种方法都用 CPU 时钟、取中位数"作废 |
| 要不要检查拷贝的数据是否正确 | 不检查 |
| `kvstore_benchmark.sh`、`tensor_xfer_benchmark.sh` 要不要跑校准 | 和这次改动没关系，维持现状 |
| 用时 | 我第二版写的"几十秒以内"是错的：那是按 `tensor_xfer_benchmark.py` 每个大小重新分配内存、31 个大小全测、做正确性检查估出来的。按已定的做法预计 10 秒内（推算，见 5.2 节） |

列说明：
- **问题**：第二版草案里需要定的点。
- **用户的决定**：用户的回复，以后按这一列执行。

## 5 DSA 校准设计（已确认，代码未改）

### 5.1 流程

```text
examples/kvshrink-vllm-serve.sh
  source setvars.sh                      # 决定 IAXL_DSA_GD_ENABLE、TP_SIZE、DP_SIZE、每个 rank 的 CPU/DSA
  export LD_PRELOAD=...
  {                                      # 校准和 vLLM 的输出一起 tee 进 log.kvshrink-vllm
    if IAXL_DSA_GD_ENABLE 开着:
        python3 tools/calibration/dsa_calibration.py   # 起 TP_SIZE×DP_SIZE 个进程同时测，标准输出 "<H2D KB> <D2H KB>"
        export IAXL_DSA_H2D_CROSSOVER_KB=<H2D 分界值>
        export IAXL_DSA_D2H_CROSSOVER_KB=<D2H 分界值>
    vllm serve ...                       # 每个 rank 的 C++ 加载时读这两个变量
  } 2>&1 | tee log.kvshrink-vllm
```

校准出错（任何一个进程失败）时不启动 vLLM，日志里有 `[dsa_calibration] failed: ...`。因为有 `| tee`，serve 脚本的退出码是 tee 的（0），和 main 里 vLLM 自己失败时一样。

### 5.2 校准脚本 `tools/calibration/dsa_calibration.py`

从 `tensor_xfer_benchmark.py` 和 `tensor_xfer_benchmark_multi_ranks.py` 简化而来，只有这一个 Python 文件。serve 脚本已经 source 过 setvars.sh、设好了 LD_PRELOAD，所以不用再包一层 .sh。单独运行：先 `source setvars.sh`，再 `LD_PRELOAD=$IAXL_LD_PRELOAD python3 tools/calibration/dsa_calibration.py`。

- **rank 和绑定**：起 `TP_SIZE × DP_SIZE` 个进程，第 r 个进程用第 r 张 GPU。CPU 和 DSA 的绑定和 KVShrink connector 里一样：`bind_cpu_affinity(r, TP_SIZE × DP_SIZE, VLLM_CPU_OMP_THREADS_BIND)`、`bind_intel_accel(r)`，r 是全局 rank，等于 dp_rank × TP_SIZE + tp_rank。
- **测的两种方法**，其余方法（逐块 CUDA 拷贝、Triton kernel）都去掉：
  - CUDA batch：`cudaMemcpy3DBatchAsync`，和 C++ 回退后走的是同一个 API；
  - DSA：`SliceCopier`。这是 `iaxl/tensor_ops.py` 里的 Python 类，把一块 GPU 内存和一块 CPU 内存登记成 KV 池，每次拷贝调用 C++ `copy_chunks_batch` → `dsa_copy_chunks_batch`，和 KVStore 拷 KV 是同一个 C++ 函数。
- **同时开始**：每次拷贝前（热身和计时都算）所有 rank 先过 barrier，在同一时刻开始，沿用 `tensor_xfer_benchmark_multi_ranks.py` 的同步方式。
- **参数和计时**，和之前的 C++ 校准代码一致：
  - 每次拷 128 MiB；GPU 上的块打乱顺序（固定随机种子，每次一样），CPU 上按顺序；
  - CUDA batch 用 CUDA event 计时（在 GPU 上打两个时间戳）；DSA 用 CPU 时钟，从提交量到等待结束；
  - 热身 2 次，计时 3 次，取最快的一次（11-23 用户把计时次数从 10 改成 3；5.7 节的测试结果是 10 次时测的）；
  - 连续拷贝大小从 256 KB 开始，步长 128 KB，最大 4 MB；先测 H2D，再测 D2H。
- **分界值和提前停**：
  - 每测完一个大小，主进程收齐所有 rank 的 GB/s，CUDA batch 和 DSA 各取平均；
  - CUDA batch 的平均值超过 DSA 的平均值，这个大小就是该方向的分界值，所有 rank 一起停止这个方向，后面的大小不测；
  - 到 4 MB 都没超过，输出 0，表示不回退。
- **内存只分配一次**：每个进程一块 128 MiB 锁页 CPU 内存和一块 128 MiB GPU 内存，每个大小用其中一段。
- **不检查拷贝的数据是否正确**。
- **输出**（除了给 serve 脚本读的那一行，其余都进终端和 `log.kvshrink-vllm`）：
  - 总控程序的标准输出只有一行 `<H2D 分界值 KB> <D2H 分界值 KB>`，给 serve 脚本读去设环境变量；
  - 总控程序的标准错误只有一行总结，例如 `[dsa_calibration] crossover H2D 512 KB, D2H 256 KB (0 = DSA faster up to 4096 KB), 4 workers, 8.3 s`；出错时是 `[dsa_calibration] failed: rank3 exited with code 1`；
  - 每个测量程序（每个 rank 一个）调用 iaxl 的 C++ 库做 DSA 拷贝，库里现有的打印会打出约 5 行：`[iaxl] config: ...`、`dsa_init: mapped 1 WQ(s) ...`、`[kv_xfer] copy_chunks_batch: contiguous block size = ...`、`copy_chunks_batch H2D/D2H: using Intel DSA backend`。用户决定保留（TP=4 共约 20 行），测量程序出错时的 Python 报错也在这里。
- **用时**：预计 10 秒内（推算）。之前 C++ 校准在单卡上实测约 0.35 秒；其余时间是进程启动、加载 torch、初始化 CUDA，这部分没测过。写好后试跑一次，以实测为准。

### 5.3 C++ 改动（和 main 比）

- `iaxl/csrc/include/env.h`、`iaxl/csrc/common/envs.c`：加两个字段 `IAXL_DSA_H2D_CROSSOVER_KB`、`IAXL_DSA_D2H_CROSSOVER_KB`，用 `env_nonnegative_int` 读，默认 0（不回退，和 main 的行为一样）。
- `iaxl/csrc/kv_xfer/dsa.cpp` 的 `dsa_copy_chunks_batch`：连续拷贝大小 ≥ 该方向的分界值时返回 false。`copy_chunks_batch` 收到 false，就接着走 main 里已有的 CUDA batch 代码。
- `iaxl/csrc/kv_xfer/cuda.cpp`：main 里"using Intel DSA backend"和"using CUDA backend"两行日志各打一次，不分方向。回退后 H2D 和 D2H 可能走不同的方法，所以改成每个方向各打一次。
- 其它 C++ 文件、KVStore、benchmark 都和 main 一样。

### 5.4 相对 PR 分支 `feat/dsa-auto-fallback` 删掉的内容

- `iaxl/csrc/include/kv_xfer.h`、`iaxl/csrc/torch_ext/torch_ext.cpp`：校准、开关、设置分界值这三个函数的声明和导出。
- `iaxl/csrc/kv_xfer/dsa.cpp`：C++ 校准代码（约 160 行）。
- `iaxl/csrc/kv_xfer/cuda.cpp`：为 C++ 校准拆出来的 `cuda_copy_chunks_batch`，恢复成 main 的写法。
- `iaxl/kvstore/kvstore.py`、`iaxl/kvstore/api.py`：KVStore 里的校准流程、rank 0 HTTP 发布、`wait_for_worker`。
- `benchmark/kvstore/kvstore_benchmark.py`、`kvstore_benchmark_multi_ranks.py`：`create_store` 拆分。原来是为了让所有 rank 一起创建 KVStore、避免校准互相等待死锁。
- `IAXL_DSA_AUTO_FALLBACK_ENABLE`。

### 5.5 还没定的

- 分支已定：继续用开发分支 `feat/dsa-auto-fallback-dev`，在上面加新提交（先把旧实现的文件恢复成 main，再改）。
- serve 脚本退出码（见 5.1 节）：要不要在脚本开头加 `set -o pipefail`，让校准失败时退出码不是 0。`kvshrink-benchmark.sh` 用 `kill -0` 判断 serve 进程还在不在，不看退出码，所以不加也能发现失败。
- 测试什么时候跑（见 5.6 节）。11-23 tangyang 的 vLLM（TP=8）占着 8 张卡、每张约 65 GB，当前先不测。

### 5.6 测试计划（11-23 记录，还没执行）

编译：11-23 已在容器 iaxl.vllm 里 `pip install -e . --no-build-isolation`，成功（日志 `tests/dsa_calibration_dev/logs/build_20261123_082402.log`，最后一行 `EXIT=0`）。

脚本和日志都放在开发目录 `tests/dsa_calibration_dev/` 下。每一步在 tmux 里跑，先只跑 1a 一次，把真实输出给用户确认后再跑其余的。

**测试 1：只跑校准脚本，不起 vLLM**

| 编号 | 配置 | 验证什么 | 通过标准 |
|---|---|---|---|
| 1a | TP=4，DP=1（GPU 0–3），跑 1 次 | 脚本能跑通，输出格式对 | 退出码 0；标准输出只有一行两个整数；总控程序只打一行总结；每个测量程序打出 `copy_chunks_batch H2D/D2H: using Intel DSA backend`（校准时没有分界值变量，测的是真的 DSA）；给出实际用时 |
| 1b | TP=4 再跑 2 次 | 结果稳不稳 | 不判通过与否，列出 3 次的 H2D、D2H 分界值和用时 |
| 1c | TP=2，DP=2（4 个进程） | DP 的情况：进程数是 TP×DP，每个 rank 用自己的 GPU 和 DSA 工作队列 | 4 个 rank 都有数据，退出码 0；每个测量程序的 `dsa_init: mapped 1 WQ(s) ... (first wqX.0)` 对应自己 rank 的工作队列 |
| 1d | TP=8（8 张卡） | 8 个进程同时测 | 8 个 rank 都有数据，退出码 0 |
| 1e | 故意让 rank 3 起不来（`CUDA_VISIBLE_DEVICES=0,1,2,99`） | 出错时不会卡住 | 几秒内退出，退出码非 0，打出 `[dsa_calibration] failed: rank3 ...`，没有残留进程 |

列说明：
- **编号**：测试步骤的编号。
- **配置**：跑校准脚本时的并行设置和用哪几张 GPU。
- **验证什么**：这一步要确认的行为。
- **通过标准**：怎样算这一步通过。

**测试 2：serve 脚本 + vLLM 端到端**

配置：Qwen3.8-27B，TP=4（GPU 0–3），不压缩。连续拷贝大小是 784 KB。

- **2a**：`IAXL_DSA_GD_ENABLE=1`。
  - `log.kvshrink-vllm` 里，vLLM 启动前有测量程序打的那几行和总控程序的一行总结；
  - 每个 rank 的 C++ 日志打出 H2D、D2H 各用了哪个后端，要和规则对得上：某方向分界值不为 0 且 ≤ 784 KB 时走 CUDA，否则走 DSA；
  - 用 `tests/dsa_fallback_dev/e2e_requests.py` 发 3 轮同样的长 prompt，第 1 轮存 KV，第 2、3 轮读回；
  - 通过标准：每个回答都对（答案可以从 prompt 的公式算出来）；第 2、3 轮 KVShrink 有命中；日志里没有 Traceback。旧的 `e2e_check.py` 匹配的是旧日志格式，要另写检查脚本。
- **2b**：`IAXL_DSA_GD_ENABLE=0`。通过标准：不跑校准，vLLM 正常起来，回答正确。
- **2c**：故意让校准失败（`KVSHRINK_DSA_DEVICES` 写一个不存在的 WQ）。通过标准：几秒内停下，不启动 vLLM，日志里有 `[dsa_calibration] failed`。

测完停掉自己起的 vLLM。

**用时（推算）**：测试 1 一两分钟；测试 2 主要是两次加载模型，约 15–20 分钟。

**不包含**：
- 用 `tensor_xfer_benchmark.sh` 测回退。这个 benchmark 和分界值无关，维持现状，不给它设分界值。
- TTFT 性能对比，要不要做由用户定。

### 5.7 测试结果（11-23 08:49–08:56 UTC，6000d）

结果目录 `tests/dsa_calibration_dev/results/tests_20261123_084912`（全部）和 `tests_20261123_085551_2c`（修 bug 后重跑 2c）。

| 编号 | 配置 | 结果 | H2D 分界值 | D2H 分界值 | 用时 |
|---|---|---|---|---|---|
| 1a | TP=4 | PASS | 256 KB | 256 KB | 3.0 s |
| 1b | TP=4，再跑 2 次 | PASS | 256 KB，256 KB | 256 KB，256 KB | 3.1 s，3.0 s |
| 1c | TP=2，DP=2 | PASS | 256 KB | 256 KB | 3.0 s |
| 1d | TP=8 | PASS | 256 KB | 256 KB | 4.2 s |
| 1e | rank 3 用不存在的 GPU 99 | PASS：2.3 s 后退出码 1，没有残留进程 | — | — | — |
| 2a | vLLM，DSA 开 | PASS：4 个 rank 的 H2D、D2H 都走 CUDA（784 KB ≥ 256 KB），回答都对，第 3 轮全部命中 | 256 KB | 256 KB | vLLM 100 s 就绪 |
| 2b | vLLM，DSA 关 | PASS：不跑校准，都走 CUDA，回答都对，全部命中 | — | — | vLLM 100 s 就绪 |
| 2c | vLLM，校准失败 | 第一次 FAIL，修后重跑 PASS：6 s 后退出码 1，没启动 vLLM，`log.kvshrink-vllm` 里有 failed 那一行 | — | — | — |

列说明：
- **编号**：对应 5.6 节的测试步骤。
- **配置**：这一步的并行设置或场景。
- **结果**：按 5.6 节的通过标准判断，冒号后是关键现象。
- **H2D / D2H 分界值**：校准脚本输出的值，"—"表示这一步没有校准结果。
- **用时**：测试 1 是校准脚本从启动到退出的时间；测试 2 是 vLLM 从启动到能接受请求的时间。

2c 第一次失败的原因：serve 脚本靠第一行 `#!/bin/bash -e` 的 `-e` 在出错时停下，但用 `bash examples/kvshrink-vllm-serve.sh` 运行时，bash 会忽略第一行的 `-e`，结果校准失败后 vLLM 照样启动了。修改：校准那行改成 `crossovers=$(python3 ...) || exit 1`。

之后的改动（用户要求）：`benchmark/tensor_xfer/tensor_xfer_benchmark.sh` 和 `benchmark/kvstore/kvstore_benchmark.sh` 都显式 `export IAXL_DSA_H2D_CROSSOVER_KB=0 IAXL_DSA_D2H_CROSSOVER_KB=0`，保证这两个 benchmark 开 DSA 时一定走 DSA。

## 6 QAT/IAA 分段校准

### 6.1 现有的切分代码

`iaxl/csrc/kv_zip/kv_zip.cpp`：

- `segment_size_for(orig_size)`：`orig_size` 是一个块的字节数。`IAXL_ZIP_SEGMENT_SIZE_KB` 是 0，或者 ≥ 块大小时，整块作为一段；否则段大小就是这个值。
- `segment_count`：段数 = 块大小 / 段大小，向下取整，最少 1 段。除不尽时余数并到最后一段，例如 784 KB 按 256 KB 切成 256 + 256 + 272 KB。
- QAT、IAA、CPU 共用一个任务池，一个段由哪个后端处理是动态分配的，所以所有后端共用同一个段大小。
- `IAXL_ZIP_SEGMENT_SIZE_KB` 在 C++ 的 `envs` 里进程启动时读一次（`envs.c`），默认值来自 `setvars.sh`：`export IAXL_ZIP_SEGMENT_SIZE_KB=${IAXL_ZIP_SEGMENT_SIZE_KB:-0}`。

### 6.2 用户实测（`kvstore_segment_sweep.sh`）

设置：块 2048 KiB，4 层，模拟数据，`qat_instances=4`，`iaa_instances=4`，`omp_threads=4`。

| 段大小 KiB | QAT 压缩率 | QAT 压缩 GB/s | QAT 解压 GB/s | IAA 压缩率 | IAA 压缩 GB/s | IAA 解压 GB/s |
|---|---|---|---|---|---|---|
| 16 | 1.390 | 7.16 | 12.06 | 1.390 | 7.76 | 14.89 |
| 32 | 1.392 | 9.32 | 17.26 | 1.393 | 9.66 | 18.17 |
| 64 | 1.392 | 10.70 | 19.71 | 1.394 | 10.54 | 21.02 |
| 128 | 1.393 | 11.07 | 20.46 | 1.395 | 10.71 | 20.86 |
| 256 | 1.392 | 10.39 | 20.84 | 1.395 | 11.07 | 21.62 |
| 512 | 1.392 | 9.02 | 20.78 | 1.395 | 11.82 | 23.24 |
| 1024 | 1.392 | 9.43 | 20.30 | 1.395 | 11.74 | 22.84 |
| 0（整块 2048） | 1.388 | 9.13 | 18.46 | 1.393 | 13.61 | 21.84 |

列说明：
- **段大小 KiB**：`IAXL_ZIP_SEGMENT_SIZE_KB` 的值。0 表示不切，整块 2048 KiB 作为一段。
- **压缩率**：原始字节数 / 压缩后字节数，越大越好。
- **压缩 GB/s、解压 GB/s**：每秒处理的原始字节数（10^9 字节）。压缩发生在存 KV 时，解压发生在读回 KV 时，影响 TTFT。
- QAT 列只开 QAT，IAA 列只开 IAA。

从表里看：

- QAT 压缩在 128K 最快，256K–512K 的解压最快，但 128K–1024K 之间只差约 2%；
- IAA 压缩整块最快，解压在 512K 最快。

### 6.3 设计（11-23、11-24 讨论；11-24 代码已写，测试结果见 6.6）

改动：`env.h` / `envs.c`（`IAXL_ZIP_SEGMENT_SIZE_KB` 读成一串数，函数 `env_nonnegative_int_list`，写错时报错退出）、`kv_zip.cpp` 的 `segment_size_for`（取第一个不大于块大小的值，0 或没有就整块一段）、`setvars.sh`（删掉手动设置）、`examples/kvshrink-vllm-serve.sh`（压缩开着且 QAT 或 IAA 开着时先校准再 export）、新文件 `tools/calibration/zip_calibration.py`（复制来的 mock 函数放在同一个文件里）。

- **测法**：参考 `tests/iaxl_unittest.py`，用 `torch_ext.compress_tensors` / `decompress_tensors`。对每个候选大小 s，造一批正好 s 大的 CPU tensor，每个 tensor 正好一段，所以测的就是段大小为 s 时的吞吐。一个进程里就能测完所有大小，不用改 `IAXL_ZIP_SEGMENT_SIZE_KB`。
- **输出**：按吞吐从高到低排序的一组段大小，例如 `256,512,128,1024,64,2048,32,16`。
- **拆分算法**（`segment_size_for`）：按顺序找第一个 ≤ 块大小的值，作为段大小；没有就整块一段。例如排序结果是 `128,32,…`、块是 64K 时，选 32K，拆成两段，和需求 3.4 第 3 条的例子一致。

已定（11-23 用户）：

| 问题 | 用户的决定 |
|---|---|
| 块头里的段大小 | 现有代码：每个压缩块的 `BlockHeader` 记着 `segment_size`，解压按块头来，所以每次启动段大小不同也能解压之前存的数据 |
| 压缩和解压的段大小是否必须一致 | 必须一致（每段是一份独立的压缩数据，解压只能按压缩时的切法），所以一个选择同时决定压缩和解压速度 |
| 按什么排序 | 只看解压吞吐，解压才影响 TTFT |
| 测几个 rank | 和 DSA 一样，TP×DP 所有 rank 同时测 |
| 候选大小 | 16、32、64、128、256、512、1024、2048 KB |
| 环境变量 | `IAXL_ZIP_SEGMENT_SIZE_KB` 直接接受一串值（只写一个数时和现在一样） |
| setvars.sh 里的手动设置 | 取消，只用校准结果；`kvstore_segment_sweep.sh` 每次运行设一个值来扫描，要保持能用 |
| 测试数据 | 不准用随机数；把 `kvstore_benchmark.py` 的 `generate_mock_kv_cache`（约 40 行）复制到 `tools/calibration/` |
| 吞吐接近时 | 不加"相差几 % 算一样"的规则，测出哪个最好用哪个（一次 vLLM 启动只设一次） |
| 脚本位置 | `tools/calibration/zip_calibration.py`，和 `dsa_calibration.py` 分开 |
| QAT 和 IAA 是否各用各的段大小 | 不分开（11-24 用户和同事讨论后定）：QAT 开着时只校准 QAT，不管 IAA 开没开，QAT 的结果同时用于 QAT 和 IAA；QAT 没开、只开 IAA 时才校准 IAA。现有代码本来就是所有后端共用一个段大小，所以 `kv_zip.cpp` 的压缩分配不用改 |
| 只测 QAT（或只测 IAA）时的线程数 | 设成被测后端的实例数，和 vLLM 里这个后端实际分到的线程数一样 |
| CPU 压缩 | 和 QAT 或 IAA 一起开时忽略，只测 QAT 或 IAA；只开 CPU 压缩时不校准，整块一段（和现在一样） |
| 和 `benchmark/` 的关系 | 校准代码不准依赖 `benchmark/` 里的任何文件，防止有人改了 benchmark 影响校准。需要 benchmark 里的函数时，复制到 `tools/`，或者改成 benchmark 引用 `tools/` 里的函数；遇到这种函数要先告诉用户 |
| mock 数据怎么拿 | 把 `kvstore_benchmark.py` 的 `generate_mock_kv_cache`、它调用的 `quantize_kv` 和 4 个常量（`SEED`、`FP8_MAX`、`INT4_QMAX`、`INT4_GROUP_SIZE`）全部复制到 `tools/calibration/`，benchmark 不动 |
| 校准用的数据类型 | 先默认 bf16。vLLM 的 KV cache 类型由 `--kv-cache-dtype` 决定，serve 脚本和 setvars.sh 都没设，默认跟模型（bf16）；bf16 时 kv_zip 会 shuffle 并且只把高字节交给 QAT/IAA，fp8/int4 不会，所以有人通过 `VLLM_SERVE_EXTRA_ARGS` 改成 fp8 时校准结果不准 |
| CPU 推理（`DEVICE=cpu`） | CPU 推理不支持 DSA，支持 QAT/IAA 压缩，所以也要做压缩校准。复制来的 `generate_mock_kv_cache` 改成没有 GPU 时在 CPU 上生成数据（7 处 `"cuda"` 换成按 `torch.cuda.is_available()` 选的设备）；`examples/recipes/vllm-cpu/setvars.pre.sh` 不动（用户没说要删里面注释掉的 `IAXL_ZIP_SEGMENT_SIZE_KB=1024`，我删了，11-24 已恢复） |
| `doc/design/kv-zip.md` | "Segments" 一段按新做法更新 |

列说明：
- **问题**：讨论中要定的点。
- **用户的决定**：用户的回复，以后按这一列执行。

### 6.4 QAT 和 IAA 公平比较（11-24 实测，6000d）

起因：用户原来的 sweep 里两边都是 4 个实例，但 QAT 只有设备 0 在干活，IAA 的 4 个实例分在两个 NUMA 节点上，QPL 把任务分给了全部 8 个 IAA 设备，两边用的设备数不一样。

做法：`tests/zip_calibration_dev/fair_compare.sh`，单 rank，段大小 512 KB，块 2048 KiB，4 层，mock 数据。每次运行期间统计每个设备实际处理了多少：QAT 读跑前、跑后两次 `/sys/kernel/debug/qat_4xxx_*/fw_counters` 的固件请求数，取差；IAA 用 `sudo perf stat` 读 `iaxN/event_category=1,event=0x2/`（不跑东西时为 0）。结果目录 `tests/zip_calibration_dev/results/fair_compare_20261124_014910`。

| 运行 | 实际干活的设备（计数器实测） | 实例数 | OMP 线程数 | 压缩率 | 压缩 GB/s | 解压 GB/s |
|---|---|---|---|---|---|---|
| QAT 4 设备 | QAT 0、1、2、3（NUMA 0），各约 1000 个请求 | 16 | 16 | 1.392 | 30.25 | 67.16 |
| IAA 4 设备 | iax1、3、5、7（NUMA 0），各约 75.5 万计数 | 16 | 16 | 1.395 | 15.07 | 73.93 |
| IAA，sweep 的配置（`auto`） | 8 个 IAA 全用 | 4 | 4 | 1.395 | 11.26 | 21.51 |
| QAT，sweep 的配置 | 只有 QAT 0（4096 个请求），另外 7 个为 0 | 4 | 4 | 1.392 | 9.11 | 20.68 |

列说明：
- **运行**：前两行是 4 设备对 4 设备的公平比较，后两行重现用户原来 sweep 的配置。
- **实际干活的设备**：运行期间计数器不为 0 的设备。
- **实例数、OMP 线程数**：日志里 `[iaxl] config` 行的 `qat_instances` / `iaa_instances` 和 `omp_threads`。
- **压缩率、压缩 GB/s、解压 GB/s**：kvstore benchmark 的输出，每个数只测一次（512 MiB，用时 7–60 毫秒），单次波动可能不小。

推测（没验证）：从 sweep 的配置到 4 设备，设备数和实例数、线程数是同时变的；IAA 用 8 个设备时解压反而只有 21.5，所以吞吐更可能受实例数和线程数限制，而不是设备数。

### 6.5 测试计划（11-24 记录，结果见 6.6）

先在容器 iaxl.vllm 里重新编译（`env.h`、`envs.c`、`kv_zip.cpp` 改了）。脚本和日志放在 `tests/zip_calibration_dev/`，每一步在 tmux 里跑。

**测试 1：段大小怎么选、环境变量怎么读**（不起 vLLM，只用 `torch_ext.compress_tensors` / `decompress_tensors`）

| 编号 | `IAXL_ZIP_SEGMENT_SIZE_KB` | 每个 tensor 大小 | 期望的段大小 / 段数 |
|---|---|---|---|
| 1a | 不设 | 64 KB、784 KB、2048 KB | 整块 1 段 |
| 1b | `0` | 784 KB | 整块 1 段 |
| 1c | `128`（只写一个数） | 784 KB | 128 KB，6 段（最后一段 144 KB） |
| 1d | `256,32` | 64 KB | 32 KB，2 段 |
| 1e | `256,32` | 784 KB | 256 KB，3 段 |
| 1f | `2048,512` | 2048 KB | 2048 KB，1 段 |
| 1g | `abc`、`-1`、`1,,2`、17 个数 | 任意 | 报错退出，提示格式不对 |
| 1h | 用 `256` 压缩并存文件，换一个进程用 `32` 解压 | 784 KB | 解压结果和原数据完全一样（解压按块头，不看当前变量） |

列说明：
- **编号**：测试步骤。
- **`IAXL_ZIP_SEGMENT_SIZE_KB`**：这一步设的值。
- **每个 tensor 大小**：压缩的单个 tensor 字节数。
- **期望的段大小 / 段数**：`[kv_zip] block_size=... segment_size=... segments=...` 日志里应该看到的值。

每一步都检查解压后和原数据逐字节相同；每种设置单独起一个进程（环境变量在库加载时读一次）。另外跑一次现有的 `tests/iaxl_unittest.py`。

**测试 2：校准脚本单独跑**（不起 vLLM）

| 编号 | 配置 | 验证什么 | 通过标准 |
|---|---|---|---|
| 2a | TP=4，只开 QAT | 脚本能跑通，输出格式对 | 退出码 0；标准输出一行、8 个段大小；总结一行写 QAT；测量程序只初始化 QAT，线程数等于 QAT 实例数；给出用时 |
| 2b | TP=4，QAT 和 IAA 都开 | 两个都开时只测 QAT | 总结写 QAT；测量程序里没有 `[iaa_zip]` 初始化 |
| 2c | TP=4，只开 IAA | 只开 IAA 时测 IAA | 总结写 IAA；线程数等于 IAA 实例数 |
| 2d | TP=2，DP=2 | 进程数是 TP×DP | 4 个测量程序，退出码 0 |
| 2e | rank 3 用不存在的 GPU 99 | 出错不卡住 | 几秒内退出，退出码非 0，没有残留进程 |
| 2f | 看不到 GPU 的进程里直接调 `generate_mock_kv_cache` | CPU 推理时能生成数据 | 在 CPU 上生成成功；压缩率和 GPU 生成的数据接近 |
| 2g | `kvstore_segment_sweep.sh --backends qat --sizes 512` | 原来的扫描脚本还能用（只设一个数） | 出结果，512 那一行不是 failed |

列说明：
- **编号**：测试步骤。
- **配置**：这一步的并行设置和开的后端。
- **验证什么**：要确认的行为。
- **通过标准**：怎样算通过。

**测试 3：serve 脚本 + vLLM 端到端**（Qwen3.8-27B，TP=4，GPU 0–3，块 784 KB）

- 3a：压缩开、只开 QAT、DSA 开。`log.kvshrink-vllm` 里有 DSA 和压缩两行校准总结；vLLM 的 `[kv_zip] block_size=... segment_size=...` 和规则对得上（校准列表里第一个不大于 784 KB 的值）；3 轮请求回答都对，第 2、3 轮命中，没有 Traceback。
- 3b：QAT 和 IAA 都开。压缩校准总结写 QAT，其余同 3a。
- 3c：只开 CPU 压缩。没有压缩校准那一行，vLLM 日志里整块一段，回答正确。

用时（推算）：编译约 3 分钟；测试 1 约 2 分钟；测试 2 约 3 分钟；测试 3 主要是三次加载模型，约 15 分钟。

不包含：CPU 推理（`DEVICE=cpu`）端到端（这台机器上没有 CPU 版容器）；TTFT 性能对比。

### 6.6 测试结果（11-24 07:40–07:53 UTC，6000d）

结果目录 `tests/zip_calibration_dev/results/tests_20261124_073957`，`summary.txt` 每一步一行 PASS/FAIL。编译通过。2e 以外全部通过；2e 失败是测试脚本设计错了，不是校准代码的问题（见下面测试 2）。

**测试 1**：全部通过。

| 编号 | `IAXL_ZIP_SEGMENT_SIZE_KB` | tensor 大小 | 日志里的段大小 | 日志里的段数 |
|---|---|---|---|---|
| 1a | 不设 | 64 KB | 64 KB | 1 |
| 1a | 不设 | 784 KB | 784 KB | 1 |
| 1a | 不设 | 2048 KB | 2048 KB | 1 |
| 1b | `0` | 784 KB | 784 KB | 1 |
| 1c | `128` | 784 KB | 128 KB | 6 |
| 1d | `256,32` | 64 KB | 32 KB | 2 |
| 1e | `256,32` | 784 KB | 256 KB | 3 |
| 1f | `2048,512` | 2048 KB | 2048 KB | 1 |

列说明：
- **编号**、**`IAXL_ZIP_SEGMENT_SIZE_KB`**、**tensor 大小**：同 6.5。
- **日志里的段大小 / 段数**：`[kv_zip] block_size=... segment_size=... segments=...` 这一行的实测值，全部和 6.5 的期望一样。每一行解压后都和原数据逐字节相同。

- 1g：`abc`、`-1`、`1,,2`、17 个数，4 种都报错退出（退出码 134，即 abort），提示 `[iaxl] ERROR: IAXL_ZIP_SEGMENT_SIZE_KB=abc must be up to 16 comma-separated non-negative integers`。
- 1h：用 256 KB 压缩存文件，另一个进程设成 32 解压，结果和原数据相同。
- `tests/iaxl_unittest.py`：2 个测试通过。

**测试 2**：2e 以外通过。

| 段大小 KB | 2a 只开 QAT，TP=4 | 2b QAT 和 IAA 都开，TP=4 | 2c 只开 IAA，TP=4 | 2d 只开 QAT，TP=2 DP=2 |
|---|---|---|---|---|
| 16 | 11.0 | 11.0 | 13.2 | 10.8 |
| 32 | 15.7 | 15.7 | 16.4 | 15.4 |
| 64 | 18.4 | 18.4 | 18.5 | 18.1 |
| 128 | 20.2 | 20.2 | 20.5 | 20.1 |
| 256 | 20.8 | 20.7 | 21.1 | 20.7 |
| 512 | 21.0 | 21.0 | 20.8 | 21.0 |
| 1024 | 21.1 | 21.1 | 20.7 | 21.1 |
| 2048 | 20.7 | 20.7 | 20.5 | 20.7 |
| 输出的排序 | 1024,512,256,2048,128,64,32,16 | 1024,512,256,2048,128,64,32,16 | 256,512,1024,128,2048,64,32,16 | 1024,512,256,2048,128,64,32,16 |
| 被测后端 | QAT | QAT | IAA | QAT |
| 校准用时 s | 7.0 | 6.6 | 7.0 | 6.4 |

列说明：
- **段大小 KB**：候选段大小。
- **2a–2d 各列**：一次校准运行。数字是解压 GB/s：每个 rank 解压 3 次取最快，4 个 rank 取平均（见 6.3）。每列只跑了一次。
- **输出的排序**：校准脚本标准输出那一行，就是 serve 脚本 export 的 `IAXL_ZIP_SEGMENT_SIZE_KB`。
- **被测后端**：总结行写的后端。每个测量程序只初始化了这个后端（另一个后端的初始化行为 0 行），线程数等于它的实例数（4），CPU 压缩关。
- **校准用时**：总结行里的用时，从脚本开始到测完；加上 Python 启动，整个命令 7.7–8.3 s。

- 2e：**失败，是测试脚本设计错了**。测试让 rank 3 用不存在的 GPU 99，想让它出错。但压缩校准不需要 GPU：rank 3 看不到 GPU 时，按 6.3 的决定在 CPU 上生成 mock 数据，正常测完，退出码 0，输出和 2a 一样。这个测试是照着 DSA 校准写的（DSA 校准必须用 GPU），对压缩校准不成立。要换一种让 rank 3 出错的方法，重跑。
- 2f：看不到 GPU 时在 CPU 上生成 mock 数据成功；CPU 数据压缩率 1.3922，GPU 数据 1.3919。
- 2g：`kvstore_segment_sweep.sh --backends qat --sizes 512` 正常出结果：压缩率 1.392，压缩 9.05 GB/s，解压 20.68 GB/s。

**测试 3**：Qwen3.8-27B，TP=4，GPU 0–3，三次都通过，模型 105–110 s 就绪。

| 运行 | 开的压缩后端 | DSA 校准总结 | 压缩校准 | vLLM 的块大小 | vLLM 的段大小 | 段数 |
|---|---|---|---|---|---|---|
| 3a | QAT | H2D 256 KB，D2H 256 KB，4.2 s | QAT，1024,512,256,2048,128,64,32,16，6.2 s | 784 KB | 512 KB | 1 |
| 3b | QAT 和 IAA | H2D 256 KB，D2H 256 KB，3.6 s | QAT，1024,512,256,2048,128,64,32,16，6.2 s | 784 KB | 512 KB | 1 |
| 3c | 只开 CPU | H2D 256 KB，D2H 256 KB，3.6 s | 没有这一行（不校准） | 784 KB | 784 KB | 1 |

列说明：
- **运行**：一次 vLLM 启动和 3 轮请求。
- **开的压缩后端**：`IAXL_QAT_ZIP_ENABLE` / `IAXL_IAA_ZIP_ENABLE` / `IAXL_CPU_ZIP_ENABLE` 里开着的。
- **DSA 校准总结**：`log.kvshrink-vllm` 里 `[dsa_calibration]` 那一行。
- **压缩校准**：`log.kvshrink-vllm` 里 `[zip_calibration]` 那一行的被测后端、排序、用时。
- **vLLM 的块大小 / 段大小 / 段数**：vLLM 里 4 个 rank 打的 `[kv_zip]` 那一行，4 个 rank 一样。

每次运行的请求：4 个 prompt，各 15,333–15,337 个 token，按顺序发 3 轮，temperature 0。三次运行的每个请求结果都一样：第 1 轮命中 0；第 2、3 轮每个请求命中 14,896 个 token（vLLM 日志 `externally-cached tokens`，即从 KVShrink 读回并解压）；3 轮回答都一样且正确（80393、88312、96231、4150）；没有 Traceback。

看到的一点：3a、3b 选的是 512 KB（排序里第一个不大于 784 KB 的值），但段数按"块大小 / 段大小，向下取整"算（6.1，原有代码），784 / 512 = 1，所以实际是整块 784 KB 一段，没有切开，而校准没测过 784 KB 这个段大小。推测（没验证）：校准里 128–2048 KB 的解压吞吐在 20.1–21.1 GB/s 之间，最多差 5%，影响应该很小。
