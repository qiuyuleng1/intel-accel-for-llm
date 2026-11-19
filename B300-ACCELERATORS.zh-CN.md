# B300 上加速器怎么分给 vLLM 的各个 rank（讨论记录，未定稿）

记录时间：2026-11-11。机器：b300（`aquila`）。
"rank" 指 vLLM 的一个 worker 进程，一个 rank 管一张 GPU；TP=8 就是 8 个 rank。
标【实测】的是在机器上查到或跑过的，标【推断】的是读代码或按原理推出来的、没有跑过验证。

## 1. 硬件：每个 NUMA 节点上有什么【实测】

| NUMA 节点 | CPU 编号（逻辑 CPU） | 物理核数 | GPU | QAT（编号 / PCI） | DSA（设备 / 已启用的队列） |
|---|---|---|---|---|---|
| 0 | 0-23, 96-119 | 24 | GPU0 `10:00.0`, GPU1 `38:00.0` | 0 / `01:00.0` | dsa0 / wq0.0 |
| 1 | 24-47, 120-143 | 24 | GPU2 `5e`, GPU3 `71` | 1 / `06:00.0` | dsa2 / wq2.0 |
| 2 | 48-71, 144-167 | 24 | GPU4 `90`, GPU5 `b9` | 2 / `81:00.0` | dsa4 / wq4.0 |
| 3 | 72-95, 168-191 | 24 | GPU6 `dd`, GPU7 `ef` | 3 / `86:00.0` | dsa6 / wq6.0 |

- 列说明：CPU 编号前一段是物理核，后一段是对应的超线程；QAT 编号是 `IAXL_QAT_DEVICES` / `KVSHRINK_QAT_DEVICES` 里用的序号（按 PCI 地址排序）。
- NUMA 0、1 在第一颗 CPU，NUMA 2、3 在第二颗 CPU（每颗 48 个物理核）。
- 另有 IAA：iax1/3/5/7，每个 NUMA 节点一个。
- 对比本机（h8263）：8 个 QAT，128 个物理核，TP=8 时每个 rank 独占 1 个 QAT、16 个物理核。

**核心矛盾**：B300 有 8 张 GPU，但 QAT、DSA、IAA 各只有 4 个（每个 NUMA 节点一个），物理核 96 个，TP=8 时每个 rank 只能分到 12 个物理核。

## 2. 现在的主机设置【实测】

- QAT：外部驱动（QAT20.L.1.2.30-00178），配置 `/etc/4xxx_dev*.conf` 的 `[SSL]` 段：`NumberDcInstances 8`、`NumProcesses 8`、`LimitDevAccess 0`。重启后保留。
- DSA：每个设备 1 个队列，dedicated（独占）模式，队列深度 256。重启后丢失，要重新跑 `tools/setup_dsa.sh`。
- 设备文件权限：`/dev/qat_*`、`/dev/usdm_drv` 是 `root:qat 660`；`/dev/dsa/wq*` 是 `root:root 600`。`intel` 用户不在 `qat` 组，在宿主机上以 `intel` 身份直接跑用不了 QAT 和 DSA。我们在容器里以 root 运行，不受影响。

## 3. 代码对"多个 rank 共用一个加速器"的支持

| 项 | 代码在哪里 | 结论 |
|---|---|---|
| 每个 rank 用哪个 QAT | `iaxl/utils/affinity.py` 的 `bind_intel_accel(rank)`：把 `KVSHRINK_QAT_DEVICES` 按 `\|` 切开，取第 rank 段 | 允许两段写同一个编号，例如 `0\|0\|1\|1\|2\|2\|3\|3`【推断：读代码】 |
| QAT 进程怎么接入 | `iaxl/csrc/qat_zip/qat_zip.c`：`icp_sal_userStartMultiProcess("SSL", CPA_FALSE)`，再按 PCI 总线号挑出指定设备上的 instance | 多进程接入同一设备是 QAT 驱动支持的用法，进程数上限是 `NumProcesses`【推断】 |
| QAT 进程名额 | `LimitDevAccess 0` 时，一个进程会在**每个** QAT 设备上都占一个名额 | 8 个 rank 正好用完每个设备的 8 个名额；此时再起任何一个用 QAT 的进程（例如单独的测试程序）都会失败【推断】 |
| 自动分 QAT | `tools/auto_config.sh` 的 `qat_auto_detect`：QAT 数 < TP 时直接报错；否则按编号平分，不看 NUMA | TP=8 走不通自动分配，必须手写 `KVSHRINK_QAT_DEVICES`【实测：source setvars.sh 报 `TP_SIZE=8 requires at least 8 QAT devices, but only 4 were found`】 |
| 自动分 DSA | `dsa_auto_detect`：每个 rank 要一个**没被别的 rank 用过的 DSA 设备**，且和 GPU 在同一 NUMA 节点 | TP=8 时即使每个设备拆成 2 个队列也通不过，必须手写 `KVSHRINK_DSA_DEVICES`【推断：读代码】 |
| DSA 能否共用 | dedicated 队列同一时间只能被一个进程打开 | 两个 rank 不能共用 `wq0.0`；要用 `tools/setup_dsa.sh 2` 把每个设备拆成 2 个 dedicated 队列（深度各 128），每个 rank 一个【推断】 |
| CPU 预算 | setvars 检查：压缩线程数 + 保留数 ≤ 每个 rank 分到的 CPU 数 | TP=8：QAT 8 + IAA 4 = 12 个线程，加保留 4 = 16 > 12，报错【实测：`IAXL_OMP_THREAD_NUM (12) + reserved (4) exceeds the smallest rank CPU allocation (12)`】 |

## 4. 另一份代码 `/home/intel/intel-accel-for-llm` 的做法【实测】

不是我们的代码，不清楚是谁在用。代码版本 commit `28f2d3e`，比我们的 `fddeef0` 旧。

- 不用容器，宿主机 Python 环境 `~/.venv-iaxl`：vLLM 0.33.0，torch 2.14 + CUDA 13.2。
- Qwen2.5-72B，**TP=2**，`CUDA_VISIBLE_DEVICES=0,2`（GPU0 在 NUMA0，GPU2 在 NUMA1）。
- CPU：`VLLM_CPU_OMP_THREADS_BIND="0-23|24-47"`，每个 rank 24 个物理核。
- QAT：由 setvars 自动平分，结果是 `KVSHRINK_QAT_DEVICES=0,1|2,3`。rank1 的 GPU 在 NUMA1，分到的 QAT 却在 NUMA2、3，也就是另一颗 CPU 上，跨 CPU 访问。
- DSA：`wq0.0|wq2.0`，每个 rank 独占一个；QAT instance 4 个，压缩线程 4 个；IAA 关闭。
- 日志：`vllm-qat-dsa.log`、`vllm-cpu-only.log`（11-09 晚上），各 45 个请求，都有 `externally-cached tokens`（例如 6400），没有 ERROR。
- 他们是在我们 11-10 早上改 QAT 驱动、重配 DSA **之前**跑的。现在设备权限变了，他们再以 `intel` 身份跑 QAT/DSA 很可能失败【推断】。

## 5. 可选方案（待定）

| 方案 | GPU | 每个 rank 的 QAT | 每个 rank 的 DSA 队列 | 每个 rank 的 CPU | 需要改什么 | 风险 / 代价 |
|---|---|---|---|---|---|---|
| TP=8 共用 | 全部 8 张 | 同 NUMA 的 QAT，2 个 rank 共用一个 | 同 NUMA 的 DSA 拆成 2 个队列，每个 rank 一个 | 12 物理核；或加上超线程共 24 个逻辑 CPU | 手写 `KVSHRINK_QAT_DEVICES`、`KVSHRINK_DSA_DEVICES`；`setup_dsa.sh 2`；解决 CPU 预算（见下） | 每个 rank 的 QAT 吞吐约为独占时的一半【推断】；QAT 进程名额全部用完；结果和本机 TP=8（独占）不完全可比 |
| TP=4 独占 | GPU0/2/4/6（每个 NUMA 一张） | 1 个，独占 | 1 个，独占 | 24 物理核 | 只设 `TP_SIZE=4`、`CUDA_VISIBLE_DEVICES=0,2,4,6`，自动分配应能通过【推断】 | 只用一半 GPU |
| TP=2（照搬另一份代码） | GPU0、GPU2 | 2 个（自动平分，rank1 跨 CPU） | 1 个，独占 | 24 物理核 | 只设 TP 和 GPU | 跨 CPU；和本机 TP=8 差别最大 |

TP=8 时 CPU 预算的三个解决办法（三选一，未定）：
1. 每个 rank 加上超线程，用 24 个逻辑 CPU，例如 rank0 `0-11,96-107`。手写 `VLLM_CPU_OMP_THREADS_BIND`，或改 `auto_config.sh`。
2. 每个 QAT 设备只用 4 个 instance（`IAXL_QAT_ZIP_INSTANCES_PER_DEVICE=4`），线程数 4 + 4 + 保留 4 = 12。
3. 关掉 IAA（`IAXL_IAA_ZIP_ENABLE=0`），线程数 8 + 保留 4 = 12。

## 6. 还没验证的

- 两个进程同时用同一个 QAT 设备压缩（计划：容器里同时跑两个 `qat_zip_test`，都设 `IAXL_QAT_DEVICES=0`）。
- 镜像里的 vLLM 0.29.0 只编了到 sm_100 的 GPU 代码，B300 是 sm_103，能不能正常跑要启动 vLLM 才知道。
