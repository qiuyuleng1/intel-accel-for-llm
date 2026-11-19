# 主机性能模式配置

本文记录在 `h8263-6000d` 上运行 KVShrink 基准测试时使用的主机级性能配置。

> 这些配置作用于整台主机。请以 `root` 用户身份或使用 `sudo` 在主机上执行命令，
> 不要在容器内执行。部分运行时配置可能在重启后恢复默认值，因此每次开始基准测试前
> 都应恢复并验证这些配置。

## 目标配置

| 配置项 | 目标值 | 含义 | 重启后的行为 |
| --- | --- | --- | --- |
| 能效偏置（Energy Performance Bias，EPB） | `Performance (0)` | 在 0–15 的 EPB 范围内选择最强的性能偏好。 | 可能恢复为固件默认值；此前的默认值为 `6`。 |
| 能效偏好（Energy Performance Preference，EPP） | 空白／不可用 | 此平台未向 Linux 暴露 HWP/EPP，因此改用 EPB。 | 除非 BIOS 后续启用 HWP/EPP，否则无需恢复此项。 |
| 调频策略（Scaling Governor） | 所有 256 个逻辑 CPU 均为 `performance` | 请求使用允许的最高 P-state，但仍受 Turbo、功耗和温度限制。 | 可能恢复为 `schedutil`。 |
| 调频驱动（Scaling Driver） | `intel_pstate` | 表示 Intel P-state 以主动模式运行；被动模式显示为 `intel_cpufreq`。 | 可能恢复为被动模式，并显示为 `intel_cpufreq`。 |
| C-state | `POLL: Enabled, C1: Enabled` | 系统仅提供轮询和浅层 C1 空闲状态；更深的 C1E/C6 状态不可用或已禁用。 | 通常由 BIOS 控制，但重启后仍需验证。 |
| 能效延迟控制（Efficiency Latency Control，ELC） | `Latency Optimized Mode (LOM)` | 为 Uncore、计算芯粒和 I/O 芯粒选择延迟优先策略。 | 运行时 TPMI 配置可能恢复为 `Optimized Power Mode (OPM)`。 |
| PCIe ACS（Access Control Services） | 所有带 ACS 的 PCIe 设备 `ACSCtl` 全为 0（关闭） | 同一 PCIe 交换芯片下两块 GPU 之间的数据直接在交换芯片里转发，不再绕到 CPU。ACS 开启时实测：8 卡 128 MiB NCCL all-reduce 27.5 ms（关闭后 9.9 ms），Qwen2.5-72B TP8 32k 输入的首 token 延迟 21.6 s（关闭后 10.4 s）；解码速度不受影响。 | **每次重启都会恢复为开启**，必须重新关闭。 |

### 表格列说明

- **配置项**：所控制的 CPU 功耗、频率、空闲状态、Uncore 策略或 PCIe 转发方式。
- **目标值**：本次性能测试要求使用的配置值。
- **含义**：该配置产生的实际功能效果。
- **重启后的行为**：重启后可能恢复默认值、因而需要重新检查的内容。

## 首次设置或重启后恢复配置

如果当前配置如下：

```text
Energy Performance Bias       Balanced Performance (6)
Scaling Governor              schedutil
Scaling Driver                intel_cpufreq
```

说明 Intel P-state 当前处于被动模式（`passive`）。无需重启服务器，按下面的顺序在
**主机上**执行即可切换到目标配置：

1. 将 Intel P-state 切换为主动模式（`active`）；
2. 将所有 CPUFreq 策略的 governor 设置为 `performance`；
3. 将所有逻辑 CPU 的 EPB 设置为 `0`；
4. 如 ELC 尚未是 LOM，再将 ELC 设置为延迟优化模式；
5. 备份并关闭 PCIe ACS；
6. 执行本文后面的完整验证命令。

切换 Intel P-state 模式会重新创建 CPUFreq 策略对象，因此必须先切换为 `active`，
然后再设置 governor。不要在 Docker 容器内执行这些主机级命令。

### 1. 恢复 Intel P-state 主动模式、调频策略和 EPB

在主机上执行：

```bash
sudo bash -c '
set -e

echo active > /sys/devices/system/cpu/intel_pstate/status

for f in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do
    echo performance > "$f"
done

for f in /sys/devices/system/cpu/cpu[0-9]*/power/energy_perf_bias; do
    echo 0 > "$f"
done
'
```

在 Intel P-state 的主动模式与被动模式之间切换时，系统会重新创建 CPUFreq 策略对象。
因此，应先将模式设置为 `active`，再恢复调频策略和 EPB 值。

### 2. 恢复 ELC 延迟优化模式

使用 PerfSpect 自带的 ELC 配置命令，以确保所有计算芯粒和 I/O 芯粒采用一致配置：

```bash
cd /home/pese/qiuyu/perfspect
sudo ./perfspect --noupdate config --elc latency
```

如果 `sudo ./perfspect --noupdate config` 已经报告
`Efficiency Latency Control: Latency Optimized Mode (LOM)`，则此步骤可以跳过。
重复执行 `--elc latency` 也是幂等的，不会启动性能测试。

不要只在这台主机上使用 Intel PCM 的 `bhs-power-mode.sh --latency-optimized-mode`
脚本。该脚本会使计算芯粒上未使用的阈值字段保留为 OPM 值，导致 PerfSpect 将最终配置
汇总为 `mixed`。PerfSpect 的 `--elc latency` 命令会配置所有相关字段，并将模式一致地
报告为 LOM。

### 3. 备份并关闭 PCIe ACS

本机 8 块 GPU 两两挂在同一个 PCIe 交换芯片下（GPU 0/1、2/3、4/5、6/7）。ACS 开启时，
交换芯片端口的 `ReqRedir+ CmpltRedir+` 会把两块 GPU 之间的请求和完成包都转发到 CPU
再折返，NCCL 走这条路比绕道 CPU 内存还慢，TP8 下每次 prefill 慢约一倍。内核开启
IOMMU（`intel_iommu=on,sm_on`，DSA 需要）时每次启动都会重新打开 ACS，因此每次重启后
都必须重新关闭。8 月 27 日 02:13 关闭 ACS 前留下的快照见
`/home/pese/qiuyu/acs-backups/acs-before-20260827_021300.txt`，同一次开机里随后跑的 KVShrink DSA
测试正常完成。

先备份当前值，再关闭（在 `intel-accel-for-llm-1030` 仓库根目录执行）：

```bash
cd /home/pese/qiuyu/intel-accel-for-llm-1030
sudo bash /home/pese/qiuyu/Alcyoneus-internal/acs-snapshot-restore.sh snapshot \
    /home/pese/qiuyu/acs-backups/acs-before-$(date +%Y%m%d_%H%M%S).txt
sudo bash tools/disable-acs.sh --apply
```

`disable-acs.sh` 不带 `--apply` 时只打印现状，不写寄存器。需要回退时，用备份文件
逐个写回原值，或直接重启：

```bash
sudo bash /home/pese/qiuyu/Alcyoneus-internal/acs-snapshot-restore.sh restore \
    /home/pese/qiuyu/acs-backups/acs-before-<时间>.txt
```

## 设置后及每次基准测试前进行验证

以下命令均为状态读取命令，不会启动 benchmark。设置完成后应立即执行一次；服务器
重启后或开始新的基准测试前也应再次执行。

### CPU 频率策略和 EPB

```bash
echo "Intel P-state status:"
cat /sys/devices/system/cpu/intel_pstate/status

echo "Scaling drivers:"
cat /sys/devices/system/cpu/cpufreq/policy*/scaling_driver |
    sort | uniq -c

echo "Scaling governors:"
cat /sys/devices/system/cpu/cpufreq/policy*/scaling_governor |
    sort | uniq -c

echo "EPB values:"
cat /sys/devices/system/cpu/cpu[0-9]*/power/energy_perf_bias |
    sort -n | uniq -c

echo "Turbo disabled flag (0 means Turbo is enabled):"
cat /sys/devices/system/cpu/intel_pstate/no_turbo
```

预期输出：

```text
Intel P-state status:
active

Scaling drivers:
    256 intel_pstate

Scaling governors:
    256 performance

EPB values:
    256 0

Turbo disabled flag (0 means Turbo is enabled):
0
```

计数 `256` 表示有 256 个逻辑 CPU 使用对应值。EPB 没有单位；`0` 表示最高性能偏好。

### C-states

```bash
for d in /sys/devices/system/cpu/cpu0/cpuidle/state*; do
    printf '%s: disable=%s, exit_latency=%s us, target_residency=%s us\n' \
        "$(cat "$d/name")" \
        "$(cat "$d/disable")" \
        "$(cat "$d/latency")" \
        "$(cat "$d/residency")"
done
```

预期状态：

```text
POLL: disable=0, exit_latency=0 us, target_residency=0 us
C1: disable=0, exit_latency=1 us, target_residency=2 us
```

`disable=0` 表示该状态已启用。应保持 C1 启用：实测退出延迟仅为 1 微秒；禁用 C1 会
增加空闲功耗，并可能降低 Turbo 的可用余量。此目标配置有意不提供 C1E 或 C6。

### ELC 模式

```bash
cd /home/pese/qiuyu/perfspect
sudo ./perfspect --noupdate config
```

配置摘要中必须包含：

```text
Efficiency Latency Control:      Latency Optimized Mode (LOM)
```

如果报告为 `Optimized Power Mode (OPM)` 或 `mixed`，请重新执行上面的 ELC 恢复命令。

### PCIe ACS

```bash
cd /home/pese/qiuyu/intel-accel-for-llm-1030
bash tools/check-acs.sh
```

脚本逐块 GPU 找到它所在的交换芯片端口，打印该端口的 `ACSCtl`，任何一位为 `+` 就报错
退出（退出码 1）。预期输出（每块 GPU 一行，共 8 行，最后一行为结论）：

```text
GPU 0000:38:00.0  port 0000:37:00.0  ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
...
ACS is off on every GPU switch port.
```

`test-qwen25-72B.sh` 开头会自动检查本文的所有配置项（P-state、调频驱动、governor、EPB、
Turbo、C-state、ELC 和 ACS），任何一项不符合目标值就直接退出，不会启动任何测试。

## EPP 为空的原因

PerfSpect 从 HWP 请求寄存器中读取 EPP。在这台机器上，固件／CPUID 未暴露 HWP 和
HWP-EPP，因此：

- 无法读取 EPP MSR；
- CPUFreq sysfs 中不存在 `energy_performance_preference`；
- PerfSpect 将 EPP 留空；
- EPB 仍然可用，并被设置为 `0`。

因此，在当前 BIOS 配置下，EPP 字段为空属于预期现象，并不表示性能配置失败。

## PerfSpect 目标配置摘要

恢复配置后，PerfSpect 中的相关字段应为：

```text
Intel Turbo Boost              Enabled
Energy Performance Bias       Performance (0)
Energy Performance Preference
Scaling Governor               performance
Scaling Driver                 intel_pstate
C-states                       POLL: Enabled, C1: Enabled
Efficiency Latency Control     Latency Optimized Mode (LOM)
```

`performance` 不会强制每个核心始终保持最高 Turbo 频率。实际频率仍受活跃核心数量、
Turbo 倍频、封装功耗、温度以及工作负载指令组合的限制。
