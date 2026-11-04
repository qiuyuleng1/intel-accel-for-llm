# kvshrink Python 开销:line_profiler 逐行测量

> 状态:第一阶段 11-04 测完,范围是单卡单 rank(TP=1)、全部命中(只走加载)、两种负载、两个代码版本(#53 前后)。没有改产品代码。全部未命中(保存路径)、多 rank、打开压缩以后再看。
>
> 分支:`perf/python-overhead-dev`,11-04 快进到 78aaef4(upstream/main)。文中的代码链接按 78aaef4;报告里的行号是各自代码版本的源文件行号。
>
> 11-03 的 py-spy 方案已放弃(3.5 节),那一版文档移到 `tests/python_overhead_dev/pyspy-method-doc-20261103.zh-CN.md`。

本文的结论分三种性质:**实测**(跑出来的数)、**确认**(读代码确认)、**推断**(没有直接证据)。

## 一、要回答的问题与范围

### 1.1 问题

在一次真实的 vLLM 运行里,kvshrink 在 EngineCore 主线程上执行的 Python 代码,每一行执行了多少次、一共花了多少时间、每次多长?

覆盖下面 5 个文件里定义的全部函数:

| 文件 | 在 kvshrink 里做什么 |
|---|---|
| `kvshrink/kvshrink_connector.py` | vLLM 调用的 connector |
| `iaxl/kvstore/kvstore.py` | connector 调用的 KVStore |
| `iaxl/kvflow/flow.py` | KVStore 调用的 KVFlow:每层的保存、加载任务在这里交给 C++ |
| `iaxl/kvflow/scratch_pool.py` | 加载时从内存里的临时缓冲池(ScratchPool)取块、还块 |
| `iaxl/utils/hash.py` | 把 prompt 分块算 hash |

列说明:“文件”是代码路径;“在 kvshrink 里做什么”是它的角色。

一行代码里调用的 C/C++(IAXL 扩展、xxhash、numpy、torch)和 vLLM 函数,时间都算在这一行上,报告里不单列。哪些行主要是在调用或等待 C++,第五节逐个指出。

### 1.2 测了什么、没测什么

| 项 | 本次 | 说明 |
|---|---|---|
| 场景 | 只测全部命中:每个请求的 KV 都从 DDR 加载 | 11-04 用户定。全部未命中(只走保存)没有测 |
| 负载 | 32k 输入、并发 1、10 个请求;2k 输入、并发 16、160 个请求 | 11-04 用户定(4.1 节) |
| 代码版本 | 28f2d3e(#52)和 78aaef4(#53) | #53 把临时缓冲池的块从张量改成块号,改的正是加载路径;两个版本各测一次做对比 |
| 线程 | EngineCore 主线程 | 例外见 3.4 节“线程” |
| 不回答 | 这些时间有没有拉长首 token 时间、每个输出 token 的时间 | 报告只给主线程上花的时间;哪些在关键路径上,第六节按代码推断 |

列说明:“项”是范围的一个方面;“本次”是这次测的内容;“说明”是来源或理由。

11-02 曾定“ScratchPool 取块还块、Python 列表交给 C++ 时的参数转换由同事负责,不计入”。同事的 #53 已经改了这两处,这次报告照常覆盖它们,用来对比 #53 前后。

## 二、背景:kvshrink 的 Python 代码在哪里跑

### 2.1 进程和线程

`vllm serve` 启动后的进程,TP=1 和 TP>1 不一样:

| 进程 | `ps` 里看到的名字 | TP=1(第一阶段) | TP>1(以后) |
|---|---|---|---|
| API Server | `vllm serve ...` 命令行 | 收 HTTP 请求、分词,不跑 kvshrink | 同左 |
| EngineCore | `VLLM::EngineCore` | 调度和模型计算都在这里;kvshrink 的全部函数都在它的主线程上跑 | 只做调度:每一步决定算哪些请求、分配显存块;跑“调度侧”函数 |
| Worker,每张 GPU 一个 | `VLLM::Worker_TP0`、`VLLM::Worker_TP1` … | 没有这个进程 | 把模型计算交给 GPU;跑“worker 侧”函数 |

列说明:“`ps` 里看到的名字”是 vLLM 用 setproctitle 设的进程名(确认:vLLM 0.29 `vllm/utils/system_utils.py` 的 `set_process_title`、`vllm/v1/executor/multiproc_executor.py` 的 `setup_proc_title_and_log_prefix`);后两列是这个进程在两种配置下做什么。

TP=1 的情况是确认的:vLLM 0.29 `vllm/config/parallel.py` 第 987–988 行,只用一张卡时 executor 是 `uni`;`vllm/v1/executor/uniproc_executor.py` 里 worker 的方法直接在调用它的线程里执行,不另开进程。

几个概念:
- **主线程**:每个进程里 Python 解释器启动时自带的线程。TP=1 时,EngineCore 的主线程依次做调度、模型计算和 kvshrink 的全部函数。
- **kvshrink 的其他 Python 线程**:管理接口的 HTTP 服务线程,以及 EngineCore 里每 10 秒查一次内存池状态的 `iaxl-monitor` 线程。它们大部分时间在睡眠。
- **一步**:调度一次 + 模型前向一次。解码时每一步给每个在跑的请求各生成 1 个 token。
- 模型前向时,vLLM 在每个 attention 层的前后各调用一次 kvshrink(确认:vLLM 0.29 `vllm/model_executor/layers/attention/kv_transfer_utils.py` 第 50–57 行)。Qwen3-32B 有 64 层,所以每步各调用 64 次。

### 2.2 vLLM 调用的 kvshrink 函数

都在 [kvshrink_connector.py](../../kvshrink/kvshrink_connector.py) 里。TP=1 时这些函数全部在 EngineCore 的主线程上。

| 函数 | TP>1 时在哪个进程的主线程 | 什么时候调用 | Python 部分做什么 |
|---|---|---|---|
| [`get_num_new_matched_tokens`](../../kvshrink/kvshrink_connector.py#L274) | EngineCore | 新请求排队时,每次尝试调度它调 1 次;这一步没排上,下一步再调一次 | 整个 prompt 分块算 hash;查 DDR 后用 numpy 截断结果;写一行日志 |
| [`update_state_after_alloc`](../../kvshrink/kvshrink_connector.py#L345) | EngineCore | 请求分到显存块后调 1 次 | 切片,记下要加载的块 |
| [`build_connector_meta`](../../kvshrink/kvshrink_connector.py#L425) | EngineCore | 每步调度结束调 1 次 | 对这一步调度的每个请求,列出要保存的块 |
| [`start_load_kv`](../../kvshrink/kvshrink_connector.py#L573) | Worker | 每步前向开始前调 1 次 | 对每个要加载的请求,一次性提交所有层的加载:每层定优先级、建任务对象 |
| [`wait_for_layer_load`](../../kvshrink/kvshrink_connector.py#L668) | Worker | 每步每个 attention 层前调 1 次 | 有加载时等这一层的数据到 GPU;没有加载时直接返回 |
| [`save_kv_layer`](../../kvshrink/kvshrink_connector.py#L701) | Worker | 每步每个 attention 层后调 1 次 | 对每个要保存的请求:算块列表、建任务对象 |
| [`wait_for_save`](../../kvshrink/kvshrink_connector.py#L729) | Worker | 每步前向结束后调 1 次 | 只有 hybrid(mamba)模型才做事 |
| [`get_finished`](../../kvshrink/kvshrink_connector.py#L747) | Worker | 每步前向结束后调 1 次 | 查还没完成的加载、保存做完没有 |

列说明:“函数”是 vLLM 调用的 kvshrink 函数(链接到定义处);“TP>1 时在哪个进程的主线程”对应 2.1 节;“什么时候调用”是调用时机和次数;最后一列是这个函数里属于 Python 的工作。另外还有 `request_finished`、`request_finished_all_groups`,请求结束时各调 1 次。

## 三、测量方法:line_profiler

### 3.1 line_profiler 是什么、报告怎么读

line_profiler 是逐行计时的 Python 分析工具:先把要测的函数登记给它,之后这些函数每执行一行,它记一次时间。结束后每个函数输出一段:源代码原样带缩进,每行前面几列数字。版本 5.1.0,11-04 在容器里 `pip install line_profiler`。

报告 B(5.1 节)里的一段,单位 µs,是 10 个请求加起来的时间:

```
Total time: 0.00888795 s
File: /home/pese/qiuyu/intel-accel-for-llm-1026/iaxl/kvflow/scratch_pool.py
Function: ScratchPool.release at line 62

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    62                                               def release(self, indices: List[int]):
    63       640       7667.7     12.0     86.3          self._free.extend(indices)
    64       640       1220.3      1.9     13.7          self._release_count += len(indices)
```

| 列 | 含义 |
|---|---|
| `Total time`(每段开头) | 这个函数一共花的时间,单位秒 |
| `Line #` | 源文件行号 |
| `Hits` | 这一行执行的次数 |
| `Time` | 这一行一共花的时间,单位 µs |
| `Per Hit` | `Time ÷ Hits`,单位 µs |
| `% Time` | 占这个函数总时间的比例 |
| `Line Contents` | 源代码 |

列说明:“列”是报告里的列名;“含义”是这一列的数怎么来的。没执行过的行只有源代码,前面是空的。

读报告的两个要点:
- **跨多行的调用**:第一行的 `Hits` 是实际调用次数的 2 倍,`Per Hit` 只有每次调用的一半(实测:报告 B 里 flow.py 第 358 行 `ctx.unzip_from_mem(` 的 `Hits` 是 1280,它实际调用了 10 个请求 × 64 层 = 640 次)。本文说的“每次”“每层”都是 `Time ÷ 实际调用次数`。
- **一行里调用了别的代码**:时间全算在这一行。例如 flow.py 第 429 行 `result.ctx.xfer_wait()` 的时间,是主线程在 C++ 里等这一层的数据搬到 GPU。

### 3.2 怎么接到 vLLM 上(不改 kvshrink)

- [`tests/python_overhead_dev/lineprofiled_kvshrink_connector.py`](../../tests/python_overhead_dev/lineprofiled_kvshrink_connector.py) 定义 `LineProfiledKVShrinkConnector`,继承 `KVShrinkConnector`,覆盖 vLLM 调用的 10 个函数(2.2 节)。每个都调用原函数;开始测量以后,调用期间打开 line_profiler(`enable_by_count` / `disable_by_count`)。
- vLLM 这样加载它:`--kv-transfer-config '{"kv_connector":"LineProfiledKVShrinkConnector","kv_connector_module_path":"lineprofiled_kvshrink_connector","kv_role":"kv_both"}'`,并把 `tests/python_overhead_dev` 加进 `PYTHONPATH`。
- 登记的函数:1.1 节 5 个文件自己定义的全部函数和方法,不含它们 import 进来的(例如 vLLM 的函数)。
- 开始和停止:脚本往结果目录的 `control` 文件写一行 `<序号> start` 或 `<序号> stop`。主线程每次进入上面 10 个函数时读这个文件(最多每 0.2 秒读一次)。所以 start 在测量 bench 的第一个请求时生效;stop 写入后,脚本再发一个只生成 1 个 token 的小请求让它生效,同时写出报告。处理过的命令和时间记在 `control.log`。
- 只统计测量 bench:预热、存入两次 bench 不计时(4.3 节)。

### 3.3 测量对 vLLM 的影响(实测)

同一个 vLLM 里,存入 bench(不开 profiler)和测量 bench(开)发的是同一批请求(4.2 节),两者对比见 5.3 节。开 profiler 后:
- 每个输出 token 的时间多 9.6 ms(32k)、11–16 ms(2k);
- 首 token 时间(中位数)多 75–411 ms,旧代码多得更多(5.6 节)。

### 3.4 报告里的数要怎么看

- **放大的原因**(确认:line_profiler 5.1.0 源码 `_line_profiler.pyx` 第 426、442 行):每次打开,它对整个 Python 解释器打开“每执行一行回调一次”(`sys.monitoring.set_events`),关闭时再撤掉。所以 kvshrink 的函数运行期间,它调用的其他 Python 代码(vLLM 的函数、生成器表达式)每一行也会触发回调,多出的时间算在 kvshrink 发起调用的那一行上。每步开关一百多次,本身也有开销(推断)。
- **小数不可信**:单次只有零点几到一两 µs 的行,主要是计时本身的开销,例如 `_store()` 里的 `return self.kvstore` 每次 1.7 µs(报告 A)。只有单次几百 µs 以上的行,绝对值才有参考意义。
- **大部分开销在报告之外**:解码时每步多出约 10 ms,而报告里 kvshrink 每步只有 0.4–0.53 ms(5.4、5.5 节)(推断)。
- **两次运行的波动约 15%**:报告 A、B 里算 hash 的那几行(两个版本代码相同)每请求 15.0、17.3 ms。所以版本对比时,15% 以内的差别不算数。
- **线程**:line_profiler 不分线程。测量期间 kvshrink 其他线程调到这 5 个文件里的函数也记了进来:报告 A、C 里的 `KVStoreLocal.simple_status`、`_log_pool_status` 是 `iaxl-monitor` 线程调的,每份合计不到 3 ms;报告 B、D 里没有出现。

### 3.5 为什么不用 py-spy、cProfile

- **py-spy**(11-03,每 10 ms 读一次调用链的采样工具):
  - 默认的暂停方式多数了 59%,首 token 时间变成 1.5 倍;
  - 不暂停方式(`--nonblocking`)少数了 20%,而且 11-04 分析发现 44–70% 的 kvshrink Python 样本调用链是不可能的(例如显示 `save_kv_layer` 第 718 行在调用 `FlashAttentionImpl.forward`),逐行归属不可信;
  - 冒烟结果在 `/home/pese/qiuyu/intel-accel-for-llm-1026/tests/python_overhead_dev/results/smoke/20261103_062205_3c6e6fce_2k_c16_hit_pause/` 和 `/home/pese/qiuyu/intel-accel-for-llm-1026/tests/python_overhead_dev/results/smoke/20261103_062747_3c6e6fce_2k_c16_hit_nonblocking/`;4 次正式运行 `/home/pese/qiuyu/intel-accel-for-llm-1026/tests/python_overhead_dev/results/20261103_*_nonblocking/` 不采用。
- **cProfile**:只能统计到函数,给不出逐行时间,11-04 停掉。
- 不再使用的脚本(都在 `tests/python_overhead_dev/`):`run_pyspy_bench.sh`、`analyze_pyspy.py`、`run_cprofile_bench.sh`、`analyze_cprofile.py`、`profiled_kvshrink_connector.py`、`timed_kvshrink.py`。

## 四、实验设计

### 4.1 配置

| 参数 | 值 | 说明 |
|---|---|---|
| 模型 | `/media/model-space/Qwen3-32B` | 11-02 确认 |
| TP | 1(单卡单 rank) | 11-02 确认;多 rank 以后再看 |
| 两种负载 | 32k 输入、并发 1、10 个请求;2k 输入、并发 16、160 个请求 | 11-04 用户定。32k 只放得下并发 1(见下面“显存”);并发 16 时相当于 16 路各跑 10 个请求,和并发 1 的 10 个对应。line_profiler 每一次执行都计入,不需要像采样那样跑很多请求 |
| 输出长度 | 128 token | 11-03 确认 |
| `--gpu-memory-utilization` | 0.9 | `examples/kvshrink-vllm-serve.sh` 默认 0.8,TP=1 时放不下一个 32k 请求 |
| `--max-model-len` | 32896(32768 + 128) | 两种负载用同一个值 |
| vLLM 的 prefix caching | 关(`--no-enable-prefix-caching`) | 否则同一个 prompt 第二次发时命中的是显存里的缓存,不走 kvshrink 的加载 |
| KV 压缩 | 关(`IAXL_KV_COMPRESSION=0`);DSA 也关(`IAXL_DSA_GD_ENABLE=0`) | 11-03 用户决定。QAT 的用户态进程名额整台机器共用,别人的任务占满时,vLLM 在第一次压缩时退出(11-03 发生过) |
| `IAXL_SCRATCH_POOL_SIZE_GB` | 16 | 默认 8。一个 32k 请求的全部层就要约 8 GiB(2047 块 × 64 层 × 每块 64 KiB) |
| `IAXL_DDR_POOL_SIZE_GB` | 64 | 默认按主机内存的 80% 算,太大:这台机器上还有别人的任务 |
| 异步加载 | 默认(`setvars.sh` 的 `0-3:0,4-6:8,7-:16`) | 在跑的请求不到 4 个时同步加载;7 个及以上时异步,前 16 层加载完就开始计算(确认:[setvars.sh](../../setvars.sh#L33)、connector [L320–L331](../../kvshrink/kvshrink_connector.py#L320-L331);实测见 5.5 节) |
| 显卡和 CPU | GPU 5;vLLM 绑 CPU 64-127 | 11-03 用户同意用 GPU 5。GPU 4-7 在 NUMA 节点 1,它的物理核是 64-127 |
| 端口 | vLLM 8020;IAXL 管理接口 18720(controller)、18820(worker) | 避开这台机器上别人的服务(容器用主机网络) |

列说明:“参数”是 vLLM 或 IAXL 的设置;“值”是这次用的值;“说明”是来源或理由。

**显存(11-03 vLLM 启动日志实测)**

| 项 | 数值 | 日志里的字段 |
|---|---|---|
| 模型权重 | 61.03 GiB | `Model loading took` |
| 留给 KV 的显存 | 10.81 GiB | `Available KV cache memory` |
| KV 能放的 token 数 | 44,272 | `GPU KV cache size` |
| 按 32,896 token 一个请求算的最大并发 | 1.35 | `Maximum concurrency` |

列说明:“项”是显存里的一部分;“数值”是日志里的值;“日志里的字段”是从哪一行来的。每个 token 的 KV 是 256 KiB(`config.json` 的 64 层 × 8 个 KV head × 128 维 × K、V 两份 × 2 字节)。32k 一个请求 32,896 token,只放得下 1 个;2k 并发 16 是 16 × 2176 = 34,816 token,放得下。

### 4.2 怎么做到全部命中

- 存入 bench 和测量 bench 参数完全相同:`vllm bench serve` 的 random 数据集,`--random-prefix-len <输入长度> --random-input-len 0 --seed 4`,请求数也相同。这样一次 bench 里所有请求的 prompt 完全相同。请求数不同时前缀会不同(确认:vLLM 0.29 random 数据集先按请求数抽样,再抽共享前缀),所以存入 bench 必须和测量 bench 请求数一样。
- 存入 bench 的第一个请求没命中,把这个 prompt 的 KV 存进 DDR,后面的都命中。实测(connector 日志):32k 是 1 个没命中、9 个命中;2k 是 1 个没命中、159 个命中。
- **怎么证明测量 bench 全部命中**:每个请求在 connector 日志里的 `externally-cached tokens` 都等于“prompt 长度减 1,再向下取整到 16 的倍数”(算 hash 时去掉了最后一个 token):32k 是 32752,2k 是 2032(5.2 节)。这次 bench 的日志按 bench 前后记下的字节位置切出来(日志里有 `\r`,按行号会对不齐)。
- 压缩关闭后 DDR 里是原始 KV,只存一个 prompt(32k 约 8 GiB),不会触发自动淘汰。

### 4.3 一次运行的步骤

一次运行 = 一种负载,由 [`tests/python_overhead_dev/run_lineprofiler_bench.sh`](../../tests/python_overhead_dev/run_lineprofiler_bench.sh) 在宿主机的 tmux 里完成,约 7 分钟:

1. 检查:容器在运行,端口 8020 没有别的 vLLM。
2. 启动 vLLM(4.1 节的参数,加载 `LineProfiledKVShrinkConnector`),日志运行时直接写进结果目录;起不来就失败退出。
3. 预热:2 × 并发个请求,随机 prompt(seed 1),不计时。
4. 存入 bench(4.2 节),不计时。
5. 写 start,跑测量 bench,确认 `control.log` 里有 `started`。
6. 写 stop,发一个小请求,确认 `control.log` 里有 `stopped`。
7. 停掉 vLLM,确认 GPU 5 显存释放。
8. 写 `line_profile_report.txt`:开头几行写请求数、命中核对、列的含义、代码版本,下面是 line_profiler 的原始输出。

每次 bench 后检查失败请求数为 0 且 EngineCore 还在,否则立刻失败退出(`vllm bench serve` 全部请求失败也返回成功)。

```bash
# 在宿主机仓库根目录,tmux 里
tests/python_overhead_dev/run_lineprofiler_bench.sh                    # 32k 并发 1,10 个请求
WORKLOAD=2k_c16 tests/python_overhead_dev/run_lineprofiler_bench.sh    # 2k 并发 16,160 个请求
SMOKE=1 tests/python_overhead_dev/run_lineprofiler_bench.sh            # 2 × 并发个请求,结果在 results/line_profiler/smoke/
```

注意:在 tmux 里按 Ctrl-C 只会停掉脚本,vLLM 和 bench 还在跑(脚本退出前要打印日志,而 `tee` 已经被 Ctrl-C 关掉),要按进程号手动停(11-04 遇到过)。

### 4.4 结果目录

`tests/python_overhead_dev/results/line_profiler/<时间>_<代码 md5>_<负载>_hit/`。代码 md5 是 kvshrink、iaxl 的 Python 文件和两个脚本一起算的,不含 C++。

| 文件 | 内容 |
|---|---|
| `line_profile_report.txt` | 报告:开头的说明 + line_profiler 原始输出 |
| `line_profile.txt` | line_profiler 原始输出 |
| `line_profile.lprof` | 原始数据,`python -m line_profiler line_profile.lprof` 可以重新输出 |
| `control`、`control.log` | 开始、停止命令,以及生效的时间 |
| `vllm.log` | vLLM 日志,运行时直接写入 |
| `bench_<名字>.json`、`bench_<名字>.log` | 每次 bench 的结果和输出;名字是 warmup、store、measured |
| `bench_offsets.txt` | 每次 bench 前后 `vllm.log` 的字节位置 |

列说明:“文件”是结果目录里的文件名;“内容”是里面存的东西。

## 五、结果

### 5.1 四份报告

| 报告 | 负载 | 代码版本 | 报告文件的绝对路径 | 测量时间段(11-04) |
|---|---|---|---|---|
| A | 32k 并发 1 | 28f2d3e(#52) | `/home/pese/qiuyu/intel-accel-for-llm-1026/tests/python_overhead_dev/results/line_profiler/20261104_033321_29cb131e_32k_c1_hit/line_profile_report.txt` | 03:38:44–03:40:24 |
| B | 32k 并发 1 | 78aaef4(#53) | `/home/pese/qiuyu/intel-accel-for-llm-1026/tests/python_overhead_dev/results/line_profiler/20261104_054449_31a8c87e_32k_c1_hit/line_profile_report.txt` | 05:50:10–05:51:47 |
| C | 2k 并发 16 | 28f2d3e(#52) | `/home/pese/qiuyu/intel-accel-for-llm-1026/tests/python_overhead_dev/results/line_profiler/20261104_051900_56b63407_2k_c16_hit/line_profile_report.txt` | 05:23:50–05:25:39 |
| D | 2k 并发 16 | 78aaef4(#53) | `/home/pese/qiuyu/intel-accel-for-llm-1026/tests/python_overhead_dev/results/line_profiler/20261104_055154_31a8c87e_2k_c16_hit/line_profile_report.txt` | 05:56:37–05:58:17 |

列说明:“报告”是本文引用时用的名字;“负载”见 4.1 节;“代码版本”是 kvshrink 和 iaxl 的 git 版本(78aaef4 的 C++ 扩展 05:44 重新编译过);“测量时间段”是 `control.log` 里 started 到 stopped 的时间。A 和 C 的 md5 不同,只是因为测试脚本加了 `WORKLOAD`,kvshrink 和 iaxl 的代码相同。

不采用的两次运行:
- `/home/pese/qiuyu/intel-accel-for-llm-1026/tests/python_overhead_dev/results/line_profiler/smoke/20261104_031959_591046d7_32k_c1_hit/`:冒烟测试,2 个请求,单位还是 ms。
- `/home/pese/qiuyu/intel-accel-for-llm-1026/tests/python_overhead_dev/results/line_profiler/20261104_032637_29cb131e_32k_c1_hit/`:运行中被这台机器上别人的任务干扰,中途停掉,没有报告。

### 5.2 核对

| 报告 | 测量 bench 的请求数 / 失败数 | `externally-cached tokens`(次数 × 值) | ScratchPool 空转 |
|---|---|---|---|
| A | 10 / 0 | 10 × 32752 | 0 |
| B | 10 / 0 | 10 × 32752 | 0 |
| C | 160 / 0 | 160 × 2032 | 0 |
| D | 160 / 0 | 164 × 2032 | 0 |

列说明:“请求数 / 失败数”来自 `bench_measured.json`;“`externally-cached tokens`”是测量 bench 那段 connector 日志里这个字段出现的次数和取值,每个请求都等于 4.2 节的值就是全部命中;“ScratchPool 空转”是 `vllm.log` 里 `ScratchPool exhausted` 警告的次数,报告里 `_allocate` 的等待循环也一次没执行。

D 是 164 次:有 4 个请求第一次询问后没被调度上,vLLM 下一步又问了一次(报告 D 第 280 行 `Discarded stale state` 执行了 4 次),这 4 次的结果也是 2032。

### 5.3 首 token 时间和每个输出 token 的时间

| 报告 | 首 token:存入 bench(不开 profiler) | 首 token:测量 bench(开) | 输出 token 间隔:存入 bench | 输出 token 间隔:测量 bench |
|---|---|---|---|---|
| A | 801.7 / 2994.6 | 1090.6 / 1069.2 | 59.9 / 59.9 | 69.5 / 69.5 |
| B | 710.6 / 2914.5 | 785.3 / 792.8 | 59.9 / 60.0 | 69.6 / 69.7 |
| C | 378.0 / 501.1 | 789.1 / 851.6 | 61.8 / 61.8 | 78.2 / 78.0 |
| D | 357.9 / 476.7 | 496.5 / 526.1 | 61.7 / 61.7 | 72.6 / 73.7 |

列说明:每格是“中位数 / 平均值”,单位 ms,来自 `vllm bench serve` 的结果(`bench_store.json`、`bench_measured.json`)。“首 token”是从发出请求到收到第一个输出 token 的时间;“输出 token 间隔”是之后每个输出 token 的平均时间(`tpot`)。样本数:32k 每次 bench 10 个请求,2k 每次 160 个。存入 bench 的第一个请求没命中(4.2 节),平均值包含它,32k 时这一个请求就把平均值拉到 2.9 s 以上,所以看中位数;32k 的中位数只有 9 个命中请求做样本。

### 5.4 32k 并发 1(报告 B,括号里是报告 A)

一个请求从调度到解码的顺序,每请求 = 报告时间 ÷ 10:

- **调度阶段 `get_num_new_matched_tokens`**:22.4 ms(19.6)。两个版本这段代码相同,差别是波动(3.4 节)。
  - connector 第 282–285 行算 hash:17.3 ms(15.0),2047 块,每块约 8.5 µs。每块的做法是把 16 个 token 的列表 `repr` 成字符串、拼接、编码、算 xxhash([hash.py](../../iaxl/utils/hash.py#L12-L44)),全是 Python 逐块循环。
  - 第 288 行 `has`:3.4 ms(3.0),其中 C++ 的 `Record.has` 3.1 ms(2.75)。
  - 第 296 行 `next(...)` 逐个找第一个没命中的块:1.3 ms(1.3)。
  - 第 336 行 `logger.info`:0.29 ms(0.29)。
- **提交加载 `start_load_kv`**:15.6 ms(35.3)。并发 1 是同步加载,一次提交 64 层;每层在 KVFlow.get 里:
  - flow.py `ctx.unzip_from_mem`(C++):每层 173 µs(408);
  - `_allocate` 取临时块:每层 16 µs(81)。
- **前向计算中逐层等数据 `wait_for_layer_load`**:255 ms(488)。
  - flow.py `xfer_wait`(C++,等这一层的数据搬到 GPU):每层 3.8 ms,每请求 243 ms(5.7 ms,366 ms);
  - `release` 还临时块:每层 17 µs,每请求 1.1 ms(1.6 ms,103 ms);
  - `unzip_wait`:每请求 0.3 ms(7.2)。
- **解码每步(1280 步)**:约 0.4 ms(约 0.4 ms)。`save_kv_layer` 64 次共 282 µs;`wait_for_layer_load` 没有加载时直接返回,64 次共 57 µs;`build_connector_meta` 26 µs;其余约 30 µs。
- **合计**:kvshrink 主线程每请求 334 ms(585),其中 77% 是 C++ 调用和等待,主要是 `xfer_wait`。其余约 76 ms 以 Python 为主:36 ms 是 128 个解码步的 `save_kv_layer` 加起来,17 ms 是算 hash。

### 5.5 2k 并发 16(报告 D,括号里是报告 C)

- **加载方式**(实测,报告 D 第 589–596、642–645 行):160 个请求里 157 个异步加载、3 个同步加载(在跑的请求不到 4 个时)。异步的请求先加载 16 层或 8 层就开始计算:第 643 行执行了 2440 次,按 16 层和 8 层两种算,是 148 个 16 层、9 个 8 层。
- **每个请求一次性的**:
  - `get_num_new_matched_tokens` 每次 1.5 ms(2.1):算 hash 0.92 ms(127 块),`has` 0.30 ms,`logger.info` 0.16 ms。
  - 提交异步加载(connector 第 653–662 行):每请求 2.4 ms(5.6)。其中 `for_group` 0.16 ms;`store.get` 2.2 ms,64 层,每层 `unzip_from_mem` 21 µs(45)。
  - `get_finished` 里查前 N 层到齐没有:每请求 0.28 ms(1.2)。
  - 前向计算中等其余层(connector 第 688 行):每请求 2.0 ms(33.7);每层 32 µs(526),其中 `xfer_wait` 18.5 µs(257)、`unzip_wait` 1.7 µs(196)、`release` 2.8 µs(74)。也就是说,现在其余层的数据基本在前向计算用到之前就到了。
  - 同步加载 2 次(3 个请求):每次 37 ms(9.4)。只有 2 次,报告里看不出原因,不影响结论。
- **每步(1322 步有前向)**:约 0.53 ms。`save_kv_layer` 295 µs;`build_connector_meta` 99 µs;`wait_for_layer_load` 没有加载时直接返回 59 µs;`get_finished` 46 µs;其余约 30 µs。`save_kv_layer`、`build_connector_meta` 两个版本基本一样。
- **合计**:kvshrink 主线程每步 1.32 ms(5.7),每步约 62 ms,占约 2%。现在约 72% 是 Python,C++ 调用和等待约 28%。

### 5.6 #53 前后对比

- **加载路径上按块的 Python 开销基本没了**(实测):
  - `release`:每层 1.6 ms → 17 µs(32k),74 → 2.8 µs(2k)。旧代码给每块建一个张量对象,还块时用生成器表达式逐个读 `t.block_idx`;新代码直接还块号列表([scratch_pool.py](../../iaxl/kvflow/scratch_pool.py#L51-L64))。
  - `_allocate`:每层 81 → 16 µs(32k)。
  - `unzip_from_mem`(C++,含参数转换):每层 408 → 173 µs(32k),45 → 21 µs(2k)。C++ 不再收每块一个张量的列表,改收块号列表。
- **等数据的时间也少了很多**(实测):32k `xfer_wait` 每请求 366 → 243 ms;2k 等其余层每请求 33.7 → 2.0 ms。`xfer_wait`、`unzip_wait` 是 C++,#53 也改了这部分 C++(`context.h`、`xfer.cpp`、`zip.cpp`),这次分不开是 Python 少了还是 C++ 变快。
- **不开 profiler 的首 token 时间**(实测,中位数):32k 801.7 → 710.6 ms(−91 ms,−11%),2k 378.0 → 357.9 ms(−20 ms,−5%);每个输出 token 的时间不变。
- **旧代码被 profiler 放大得特别多**(实测):开 profiler 时首 token 差 305 ms(32k)、293 ms(2k),不开时只差 91、20 ms。所以报告 A、C 里加载路径的数(例如 `release` 每请求 103 ms)不能当真实开销。旧 `release` 的生成器表达式在 3.4 节的机制下,每次迭代都可能触发回调(推断)。

## 六、结论:哪些值得优化

读代码时列的候选点(11-02)和新加的几项,按 78aaef4、全部命中的实测:

| # | 名称 | 代码位置 | 实测(报告 B、D) | 判断 |
|---|---|---|---|---|
| 1 | prompt 分块算 hash | [hash.py](../../iaxl/utils/hash.py#L12-L44),调用处 [connector L282–L287](../../kvshrink/kvshrink_connector.py#L282-L287) | 每请求 32k 17.3 ms,2k 0.92 ms | 唯一可能值得做的,见 6.1 节 |
| 2 | 列出要保存的块 | [`_add_request_to_save`](../../kvshrink/kvshrink_connector.py#L375-L406) | 全部命中时没有要存的块,每次在第 386 行就返回:每次 2–5 µs,2k 每步 16 次共约 33 µs | 本场景不用做;未命中没测 |
| 3 | 每层重算块列表 | [`ReqMeta.for_group`](../../kvshrink/kvshrink_connector.py#L64-L72) | 只在提交异步加载时调 1 次:2k 每请求 0.16 ms。保存时每层调 1 次,要未命中场景才有,没测 | 本场景不用做 |
| 4 | KVStore、KVFlow 每次调用的其余 Python | [KVStore.get](../../iaxl/kvstore/kvstore.py#L273)、[KVFlow.get](../../iaxl/kvflow/flow.py#L317) | KVFlow.get 里除 C++ 调用以外的行:每请求 32k 约 1.7 ms,2k 约 0.95 ms | 不用做 |
| 5 | 每步查完成状态 | [`get_finished`](../../kvshrink/kvshrink_connector.py#L747-L806) | 每步 32k 9 µs,2k 46 µs | 不用做 |
| 6 | ScratchPool 用光时空转 | [KVFlow._allocate](../../iaxl/kvflow/flow.py#L193-L207) | 4 次运行都没出现(5.2 节) | — |
| 7 | 解码时 `save_kv_layer` 每步空跑 64 次 | [L701–L727](../../kvshrink/kvshrink_connector.py#L701-L727) | 每步 0.28–0.30 ms,解码时没有要存的数据 | 收益小,不建议单独做 |
| 8 | 没有加载时 `wait_for_layer_load` 每步 64 次 | [L668–L670](../../kvshrink/kvshrink_connector.py#L668-L670) | 每步 57–59 µs | 不用做 |
| 9 | 每个新请求打一行 info 日志 | [L336–L342](../../kvshrink/kvshrink_connector.py#L336-L342) | 每请求 0.16–0.29 ms | 不用做 |
| 10 | ScratchPool 取块还块、参数转换 | [scratch_pool.py](../../iaxl/kvflow/scratch_pool.py#L51-L64) | 见 5.6 节 | #53 已解决 |

列说明:“#”是编号;“名称”是这一项在做什么;“代码位置”按 78aaef4;“实测”是报告 B(32k)、D(2k)里的数,开着 profiler;“判断”是要不要做。第 7–9 项是看了报告以后加的。

**哪些在关键路径上(推断)**
- 算 hash 在开始加载之前、在调度时执行,这期间没有别的工作同时进行,所以直接加到首 token 时间上:32k 约 17 ms,占首 token 710.6 ms 的约 2.4%。这是开着 profiler 的数,真实值更小(6.1 节)。
- 解码每步 kvshrink 约 0.4–0.53 ms,而每步约 60–62 ms,不到 1%;这两种负载下 GPU 是瓶颈,CPU 跑在 GPU 前面,这部分不影响速度。
- 32k 最大的时间是 `xfer_wait`(每请求 243 ms,占 kvshrink 主线程时间的 73%),是 C++ 和硬件的事,不在本次范围。

### 6.1 hash 优化(方案,还没做)

**现在做什么**

每来一个新请求,vLLM 调度它时调用 `get_num_new_matched_tokens`。kvshrink 把整个 prompt 每 16 个 token 切成一块,每块把 16 个 token 的列表 `repr` 成字符串、和上一块的 hash 拼接、编码、算 xxhash([hash.py](../../iaxl/utils/hash.py#L12-L44)),再 `str()` 成 DDR 的键([connector L282–L287](../../kvshrink/kvshrink_connector.py#L282-L287)),拿这些键去 DDR 里查有没有存过。

- 每个请求只算一次,在开始加载 KV 之前,在 EngineCore 主线程上。这段时间主线程不做别的事,所以算 hash 的时间全部加到这个请求的首 token 时间上(推断)。
- 开始生成以后不再算,每个输出 token 的时间和它无关(确认)。

**现在耗时(实测,开着 line_profiler,报告 B、D)**

| 场景 | 每请求块数 | hash 耗时(每请求) | 占首 token 时间 |
|---|---|---|---|
| 32k 输入、并发 1、全部命中 | 2047 | 17.3 ms | 约 2.4% |
| 2k 输入、并发 16、全部命中 | 127 | 0.92 ms | 约 0.3% |

列说明:“hash 耗时”是 connector 第 282–285 行的总时间 ÷ 调用次数(32k 10 次,2k 164 次),是平均值;报告 A 里同一段代码是 15.0 ms,差别是两次运行的波动(3.4 节)。“占首 token 时间”= hash 耗时 ÷ 存入 bench(不开 profiler)首 token 时间的中位数:32k 710.6 ms(9 个命中请求),2k 357.9 ms(159 个)。

**不开 profiler 的真实耗时(推断,没测)**

line_profiler 每执行一行都要额外计时:报告里最简单的行(例如 hash.py 第 43 行 `prev_block_hash = cur_hash`)每次 0.2 µs,正常只要几十纳秒。算一块要经过约 18 次逐行计时(报告 B 里这些行的 `Hits` 加起来 ÷ 块数),每块多算约 3 µs。扣掉以后,32k 真实约 11 ms(每块约 5.5 µs),2k 约 0.5 ms,误差可能有好几毫秒。

**改法**

kvshrink 不再自己算,直接用 vLLM 已经算好的每块 hash(`request.block_hashes`,每块 32 字节),每块用一次 `.hex()` 转成字符串作 DDR 的键。
- 只要配置了 KV connector,vLLM 就会为每个请求算这些 hash,关掉 prefix caching 也算(确认:vLLM 0.29 `vllm/v1/engine/core.py` 第 225–233 行)。这部分开销改不改都在,kvshrink 再算一遍是重复工作。
- 默认算法 sha256、种子固定,不同进程算出来一样(确认:vLLM 0.29 `vllm/v1/core/kv_cache_utils.py` 第 118–131 行)。

**预期效果(推断)**

| 场景 | 现在:开 profiler(实测) / 不开(估算) | 改后(估算) | 每请求省 | 首 token 快多少 |
|---|---|---|---|---|
| 32k 并发 1 | 17.3 ms / 约 11 ms | 不到 1 ms | 约 10–16 ms | 约 1.4–2.3% |
| 2k 并发 16 | 0.92 ms / 约 0.5 ms | 不到 0.1 ms | 不到 1 ms | 不到 0.3%,基本看不出来 |

列说明:“现在”见上面两段;“改后”只剩每块一次 `.hex()`,每块零点几 µs;“每请求省”= 现在 − 改后,下限用不开 profiler 的估算,上限用开 profiler 的实测;“首 token 快多少”= 每请求省 ÷ 上一张表的首 token 时间中位数。

- 只有长输入才有明显收益;解码速度不变。
- 没测:DDR 的键从约 19 个字符变成 64 个字符,C++ 查 DDR 的 `Record.has`(32k 每请求 3.1 ms)可能略慢。

**代价**
- DDR 里的键换格式:改之前存进去的数据命中不了;共享同一份存储的所有实例要一起升级。
- 用户如果把 vLLM 的 hash 算法设成 xxhash 又没设 `PYTHONHASHSEED`,每个进程的 hash 是随机的,同一个 prompt 在不同进程算出不同的键,要禁止这种配置或报错(确认:同上 `kv_cache_utils.py`)。

**要先定的事**
1. 要支持哪些 vLLM 版本(11-04 已问用户,待回答)。
2. DDR 键换格式能不能接受。
3. 先跑一个几秒的小测试,把上面的“估算”换成实测:不启动 vLLM、不用 GPU,在容器里分别计时现在的 hash 和改后的 `.hex()` 转换,输入 32767 个 token,各跑多次,给平均值和中位数。

## 七、已定和待定事项

已定(11-02 至 11-04 和用户确认):

| 事项 | 结论 |
|---|---|
| 测量工具 | line_profiler 5.1.0,逐行报告;不用 py-spy、cProfile(3.5 节) |
| 场景 | 只测全部命中 |
| 负载和请求数 | 32k 并发 1:10 个;2k 并发 16:160 个 |
| 输出长度 | 128 |
| 报告单位 | µs |
| 代码版本 | 28f2d3e 和 78aaef4 各测一次 |
| 压缩和 DSA | 都关 |
| 显卡 | GPU 5 |

列说明:“事项”是要定的问题;“结论”是定下来的做法,理由见对应章节。

待定:
1. **算 hash 要不要改**(6.1 节):先定支持哪些 vLLM 版本、DDR 键换格式能不能接受,并跑小测试拿到实测数;要改的话先给出方案和改前改后的代码。
2. **全部未命中(保存路径)**:这次没有测,第六节第 2、3 项的保存部分要靠它。
3. **多 rank**:TP、并发等参数以后再定。
4. **压缩打开时**:压缩与否只改变 IAXL C++ 里做的事,kvshrink 的 Python 代码路径相同(推断,没有测过)。

## 八、环境

| 项 | 值 | 来源 |
|---|---|---|
| 容器 | `iaxl.vllm.qiuyu-python-overhead-dev`,镜像 `vllm-iaxl-dev-qiuyu-python-overhead-dev:1.5.1` | 11-03 启动 |
| 代码 | 分支 `perf/python-overhead-dev`;报告 A、C 用 28f2d3e,报告 B、D 用 78aaef4(11-04 快进,05:44 容器里 `pip install -e . --no-build-isolation` 重新编译) | 报告开头、`git log` |
| Python | 3.12.3 | 容器内实测 |
| torch | 2.13.0+cu130 | 容器内实测 |
| vLLM | 0.29.0 | 容器内实测 |
| line_profiler | 5.1.0 | 容器内 `pip install`,11-04 |
| 显卡 | 8 × RTX 6000D,每张 85651 MiB | 宿主机 `nvidia-smi` |
| 主机内存 | 503 GB | 宿主机 `free -g` |
| CPU | 256 个逻辑 CPU,2 个 NUMA 节点(节点 0:0-63、128-191;节点 1:64-127、192-255) | 宿主机 `lscpu` |

列说明:“项”是软件或硬件;“值”是版本或大小;“来源”是从哪里查到的。
