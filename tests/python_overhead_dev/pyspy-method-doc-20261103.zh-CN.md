# kvshrink Python 开销:用 py-spy 测量的方案

> 状态:第一阶段(单卡单 rank,TP=1)11-03 开始测量;多 rank 以后再看。没有改产品代码。第三节的候选点是读代码推断;测量方法和冒烟测试结果见第四节,正式结果跑完后补在后面。
>
> 分支:`perf/python-overhead-dev`,代码基线 28f2d3e(upstream/main)。文中行号都按这个版本核对过。

本文的结论分三种性质:**实测**(跑出来的数)、**确认**(读代码确认)、**推断**(没有直接证据)。

## 一、要回答的问题与范围

### 1.1 问题

在一次真实的 vLLM 运行里,kvshrink 自己的 Python 代码在 CPU 上忙了多少时间?花在哪些函数、哪些代码行?

“kvshrink 自己的 Python 代码”指下面这些文件:

| 文件 | 在 kvshrink 里做什么 |
|---|---|
| `kvshrink/kvshrink_connector.py` | vLLM 调用的 connector |
| `iaxl/kvstore/kvstore.py` | connector 调用的 KVStore |
| `iaxl/kvflow/flow.py` | KVStore 调用的 KVFlow:每层的保存、加载任务在这里交给 C++ |
| `iaxl/utils/hash.py` | 把 prompt 分块算 hash |

列说明:“文件”是代码路径;“在 kvshrink 里做什么”是它的角色。

这些 Python 代码直接调用的第三方库的 C 代码也算在内:xxhash、numpy(11-02 确认),以及 torch(11-03 确认)。torch 只出现在一处:`KVFlow.put` / `get` 开头读 GPU KV cache tensor 的属性(`.dim()`、`.device`、`.is_contiguous()`、`.shape`、`.dtype`),用来做 assert 检查和算每块的形状([flow.py L218–L245](../../iaxl/kvflow/flow.py#L218-L245)、[L341–L372](../../iaxl/kvflow/flow.py#L341-L372));connector 和 KVStore 在跑 bench 时不调 torch(确认)。

### 1.2 不在本次范围

| 不算的内容 | 代码里的例子 | 为什么不算 |
|---|---|---|
| IAXL 的 C++ 代码 | 压缩、解压、拷贝、任务队列 | 不是 Python |
| Python 列表交给 C++ 时逐个元素的转换 | `unzip_from_mem`、`zip_to_mem`、`xfer_chunks_batch` 收到的 block 号、label、tensor 列表 | 已知问题,同事负责 |
| ScratchPool 取块、还块 | [scratch_pool.py](../../iaxl/kvflow/scratch_pool.py#L45-L59) 的 `allocate`、`release`:每层建一个长度等于块数的 tensor 列表、逐个读 `block_idx` | 同事负责(11-02 确认) |
| CPU 等 GPU | `KVFlow.put` 每层调用一次 `xfer_wait_cur_stream(sync_cur_stream=True)`([flow.py](../../iaxl/kvflow/flow.py#L256)),主线程停下来等 GPU 算完这一层 | 和 GPU 同步有关,先不看 |
| 等待 | 等 GPU、等 C++ 任务完成、等锁 | 这时 CPU 没在干活 |
| vLLM 跑在 CPU 上时的 in-place 路径 | [flow.py](../../iaxl/kvflow/flow.py#L243) 的 `[tensor[i] for i in chunk_indices]` | 我们跑 GPU,不经过这段代码 |

列说明:“不算的内容”是本次不测的一类开销;“代码里的例子”是对应的代码;“为什么不算”是理由。

IAXL C++、参数转换、ScratchPool 这三类 py-spy 也会采到,结果里单独列出,只作对照(4.4 节;IAXL C++ 和参数转换合成一列)。“CPU 等 GPU”如果是空转等待,会落在“IAXL C++”那一列(4.3 节)。

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
- **主线程**:每个进程里 Python 解释器启动时自带的线程。TP=1 时,EngineCore 的主线程依次做调度、模型计算和 kvshrink 的全部函数;TP>1 时,调度侧函数在 EngineCore 主线程,worker 侧函数在各 Worker 的主线程。
- **kvshrink 的其他 Python 线程**:管理接口的 HTTP 服务线程(EngineCore 和 Worker 里都有),以及 EngineCore 里每 10 秒查一次内存池状态的 `iaxl-monitor` 线程。它们大部分时间在睡眠。
- **一步**:调度一次 + 模型前向一次。
- 模型前向时,vLLM 在每个 attention 层的前后各调用一次 kvshrink(确认:vLLM 0.29 `vllm/model_executor/layers/attention/kv_transfer_utils.py` 第 50–57 行)。

### 2.2 vLLM 调用的 kvshrink 函数

都在 [kvshrink_connector.py](../../kvshrink/kvshrink_connector.py) 里。TP=1 时这些函数全部在 EngineCore 的主线程上。

| 函数 | TP>1 时在哪个进程的主线程 | 什么时候调用 | Python 部分做什么(不含 1.2 节不算的内容) |
|---|---|---|---|
| [`get_num_new_matched_tokens`](../../kvshrink/kvshrink_connector.py#L274) | EngineCore | 新请求排队时,每次尝试调度它调 1 次;这一步没排上,下一步再调一次 | 整个 prompt 分块算 hash;查 DDR 后用 numpy 截断结果;写一行日志 |
| [`update_state_after_alloc`](../../kvshrink/kvshrink_connector.py#L345) | EngineCore | 请求分到显存块后调 1 次 | 切片,记下要加载的块 |
| [`build_connector_meta`](../../kvshrink/kvshrink_connector.py#L425) | EngineCore | 每步调度结束调 1 次 | 对这一步调度的每个请求,列出要保存的块 |
| [`start_load_kv`](../../kvshrink/kvshrink_connector.py#L573) | Worker | 每步前向开始前调 1 次 | 对每个要加载的请求,一次性提交所有层的加载:每层定优先级、建任务对象 |
| [`wait_for_layer_load`](../../kvshrink/kvshrink_connector.py#L668) | Worker | 每步每个 attention 层前调 1 次 | 有加载时等这一层加载完(等待不算);Python 部分只有简单的判断和循环 |
| [`save_kv_layer`](../../kvshrink/kvshrink_connector.py#L701) | Worker | 每步每个 attention 层后调 1 次 | 对每个要保存的请求:算块列表、建任务对象 |
| [`wait_for_save`](../../kvshrink/kvshrink_connector.py#L729) | Worker | 每步前向结束后调 1 次 | 只有 hybrid(mamba)模型才做事 |
| [`get_finished`](../../kvshrink/kvshrink_connector.py#L747) | Worker | 每步前向结束后调 1 次 | 查还没完成的加载、保存做完没有 |

列说明:“函数”是 vLLM 调用的 kvshrink 函数(链接到定义处);“TP>1 时在哪个进程的主线程”对应 2.1 节;“什么时候调用”是调用时机和次数;最后一列是这个函数里属于 Python 的工作。

## 三、候选点(读代码推断,未测)

按调度侧、worker 侧分组(TP=1 时两组都在 EngineCore 主线程上)。ScratchPool 取块、还块归同事,不在这张表里。

| # | 名称 | 代码位置 | 在哪跑、多久一次 | 随什么变大 | 为什么怀疑 |
|---|---|---|---|---|---|
| | **调度侧** | | | | |
| 1 | prompt 分块算 hash | [hash.py](../../iaxl/utils/hash.py#L12-L44),调用处 [connector L282–L287](../../kvshrink/kvshrink_connector.py#L282-L287) | `get_num_new_matched_tokens` 里,每个新请求每次尝试调度 1 次 | prompt 的 token 数 | 每块都要:切出一个小列表、`repr` 成字符串、拼接、`encode`、xxhash,结果再转成 `str`;调用前 `all_token_ids[:-1]` 还把整个 token 列表复制一遍 |
| 2 | 列出要保存的块 | [`_add_request_to_save`](../../kvshrink/kvshrink_connector.py#L375-L406),以及 [L432](../../kvshrink/kvshrink_connector.py#L432) 复制块号列表 | `build_connector_meta` 里,每步每个被调度的请求 1 次 | 这个请求这一步的块数 | 逐块判断要不要保存;新请求还要复制整份块号列表 |
| | **worker 侧** | | | | |
| 3 | 每层重算块列表 | [`ReqMeta.for_group`](../../kvshrink/kvshrink_connector.py#L64-L72),调用处 [L719](../../kvshrink/kvshrink_connector.py#L719)、[L653](../../kvshrink/kvshrink_connector.py#L653) | `save_kv_layer` 里每层 × 每个要保存的请求 1 次;`start_load_kv` 里每个要加载的请求 1 次 | 块数 | 同一个请求每层算出来的结果都一样,但每层都把块号和 hash 配对、过滤、再拆成两个新列表 |
| 4 | KVStore、KVFlow 每次调用的其余 Python | [KVStore.put](../../iaxl/kvstore/kvstore.py#L212)、[KVStore.get](../../iaxl/kvstore/kvstore.py#L274)、[KVFlow.put](../../iaxl/kvflow/flow.py#L199)、[KVFlow.get](../../iaxl/kvflow/flow.py#L319);`start_load_kv` 里给每层定优先级 [L638–L647](../../kvshrink/kvshrink_connector.py#L638-L647) | 保存:每层每个请求 1 次;加载:每个请求 1 次,内部循环所有层 | 层数 | 每次调用都要:对传入的每一层做一串 assert 检查、建任务对象、登记到等待列表;`KVStore.put` 每次用 `list.index` 在所有层名里线性查找([L237](../../iaxl/kvstore/kvstore.py#L237));`KVStore.get` 每次重建层号字典([L294–L296](../../iaxl/kvstore/kvstore.py#L294-L296))。单次不大,次数多 |
| 5 | 每步查完成状态 | [`get_finished`](../../kvshrink/kvshrink_connector.py#L747-L806)、[KVFlow.get_wait](../../iaxl/kvflow/flow.py#L420) | 每步 1 次 | 还没完成的请求数 × 层数 | 对每个还没完成的加载请求,逐层调用 C++ 查状态(每层 2 次) |
| 6 | ScratchPool 用光时空转 | [KVFlow._allocate](../../iaxl/kvflow/flow.py#L163-L177) | 偶发:ScratchPool 用光时 | 等待时长 | 循环里不睡眠,一直查有没有块还回来,会打 `ScratchPool exhausted: waited ...` 警告。空转会被算成 Python 在忙,污染结果,所以第一阶段把 ScratchPool 设大来避免(5.1 节),日志里出现就重跑 |

列说明:
- “#”:编号,只在本文内引用。
- “名称”:这一项在做什么。
- “代码位置”:函数或行号(链接)。
- “在哪跑、多久一次”:所在函数和执行频率。
- “随什么变大”:耗时和哪个数量成正比。块数 = 输入 token 数 ÷ block_size(默认 16),例如 32k 输入约 2000 块。
- “为什么怀疑”:读代码看到的可能慢的原因。

py-spy 的结果会直接显示哪些真的花了时间,也可能出现这张表里没有的地方。

## 四、测量方法:py-spy

### 4.1 py-spy 是什么

py-spy 是一个 Python 采样分析工具,在被测进程**之外**单独运行:
- 每隔固定时间(采样率 100 时每 10 ms)读一次被测进程的内存,取出每个 Python 线程当前的调用链(从最外层函数一直到那一刻正在执行的代码行)。取一次叫一个**样本**。
- 不改 vLLM、不改 kvshrink 代码,也不用重启 vLLM:按进程号附着到已经在跑的进程上即可。
- 本次**不加 `--native`**,调用链里只有 Python 函数;在 C/C++ 里花的时间(IAXL 扩展、torch、xxhash 等)算在调用它的那一行 Python 上。不加的原因:
  - IAXL 扩展库去掉了符号(`nm` 显示 `no symbols`,11-03 确认),加了也只能看到地址;
  - 加了以后每次采样要暂停 vLLM 更久,11-03 冒烟测试里 bench 总时长变成 1.82 倍(4.8 节)。
- 样本足够多时,一段代码被采到的次数和它占用的 CPU 时间成正比。

### 4.2 一个样本长什么样

用 `--format raw` 输出时,文件里一行是一条调用链和它被采到的次数。下面是示意:格式按 py-spy 源码 `src/flamegraph.rs`,内容是举例,不是实测,中间的 vLLM 函数用 `…` 省略。

```
thread (0x7F3A2C1B7740): MainThread;…;save_kv_layer (/…/kvshrink/kvshrink_connector.py:719);for_group (/…/kvshrink/kvshrink_connector.py:65) 37
```

- 用 `;` 隔开的是调用链,从左到右由外到内。
- 第一项是线程编号和线程名(`--threads` 加上的)。
- 其余每一项是“函数名 (文件:行号)”。最右边一项是那一刻正在执行的代码。
- 行尾的 37 是这条调用链一共被采到 37 次。

这一行的意思是:主线程在 `save_kv_layer` 调用的 `for_group` 里、执行第 65 行时,一共被采到 37 次。

调用 IAXL 扩展的样本(示意,行号是真实代码行,`…` 处省略):

```
thread (0x7F3A2C1B7740): MainThread;…;wait_for_layer_load (/…/kvshrink/kvshrink_connector.py:688);get_wait (/…/iaxl/kvstore/kvstore.py:…);get_wait (/…/iaxl/kvflow/flow.py:458) 50
```

最里面是 [flow.py 第 458 行](../../iaxl/kvflow/flow.py#L458) `result.ctx.xfer_wait()`:那一刻主线程在 IAXL 的 C++ 里等传输完成。调用链里看不到 C++ 函数,但这一行是调用 IAXL 扩展的行,按 4.4 节归“IAXL C++”,不算进 Python 开销。

### 4.3 哪些样本不记录

py-spy 默认不记录**空闲线程**:每次采样先读 `/proc` 里每个线程的状态,只有状态是 R(正在运行或等着上 CPU)的线程才记录,睡眠或阻塞(等锁、等 IO、等 C++ 任务完成)的跳过(确认:py-spy 源码 `src/python_spy.rs`,先读线程状态,再读调用链)。所以统计到的是 CPU 在忙的时间,等待时间不在里面,对应 1.2 节“等待不算”。

CUDA 的同步等待可能是空转(CPU 一直在查 GPU 好了没有),这时线程算“在忙”(推断)。这种等待落在调用 IAXL 扩展或 torch 的那一行上:IAXL 的归“IAXL C++”(4.4 节);torch 的(例如 vLLM 自己的 `Event.synchronize`)不在 kvshrink 里,不计入。

**py-spy 数的忙碌时间准不准(11-03 实测)**

每次 bench 前后各读一次 `/proc` 里 EngineCore 主线程的 utime + stime(内核记账的、真正占用 CPU 的时间),和 py-spy 数的主线程忙碌时间比。负载:2k 输入、并发 16、64 个请求、全部命中,压缩关闭(5.1 节),两种采样方式各跑一次。

| 量 | 暂停方式(默认) | 不暂停方式(`--nonblocking`) |
|---|---|---|
| 主线程 CPU 时间,内核记,没开 py-spy 的 bench | 274 | 258 |
| 主线程 CPU 时间,内核记,开 py-spy 的 bench | 320 | 242.5 |
| py-spy 数的主线程忙碌时间 | 510 | 195 |
| py-spy 数的 ÷ 内核记的(同一次 bench) | 1.59 | 0.80 |
| py-spy 读失败次数 / 主线程样本数 | 1 / 3263 | 359 / 1246 |
| 主线程调用链不完整的占比 | 0% | 11.3% |

列说明:前三行单位是毫秒/请求(这次 bench 的总量 ÷ 64 个请求);“py-spy 数的”= 主线程被判为忙的样本数 × 10 ms ÷ 64;“读失败”是 py-spy 某一轮读整个进程失败的次数,这一轮所有线程都丢了;“调用链不完整”是最外层不是进程入口 `<module>` 的样本。结果目录 `tests/python_overhead_dev/results/smoke/20261103_062205_3c6e6fce_2k_c16_hit_pause/` 和 `.../20261103_062747_3c6e6fce_2k_c16_hit_nonblocking/`。

- 暂停方式多数了 59%,原因没有查明;它还让主线程多用 17% 的 CPU(274 → 320),首 token 时间变成 1.5 倍(4.8 节)。
- 不暂停方式少数了 20%。不暂停时,只有线程正在执行 Python、调用链正在变,才会读失败;359 次失败如果都发生在主线程忙的时候,会丢 359 ÷ (1246 + 359) = 22%,和实测的 20% 对得上(推断:失败几乎都发生在主线程忙时)。
- **采用不暂停方式**,结果同时给出 py-spy 直接数的值和按内核校准的值(4.5 节)。11-03 和用户确认。

### 4.4 一个样本怎么归类

只统计主线程:TP=1 时是 EngineCore 的主线程;TP>1 时是 EngineCore 和 Worker_TP0 的主线程。其他线程里的 kvshrink 代码(管理接口、内存池自动淘汰)单独汇总一行。对每个样本:

1. 在调用链里找**最里面那个 kvshrink 帧**:文件在 `kvshrink/` 或 `iaxl/` 下的 Python 函数。找不到,说明这个样本是 vLLM 自己的代码,不计入。
2. 按下表从上到下判断,满足哪一行就归哪一类:

| 类别 | 判断规则 | 例子 | 算不算 Python 开销 |
|---|---|---|---|
| ScratchPool | 最里面的 kvshrink 帧在 `iaxl/kvflow/scratch_pool.py` 里 | `allocate`、`release` | 不算,单列(同事负责) |
| IAXL C++(含参数转换) | 这个帧正在执行的那一行调用 IAXL 扩展 | `ctx.unzip_from_mem(...)`、`result.ctx.xfer_wait()`、`self.record.has(...)`、`Context.create(...)` | 不算,单列 |
| Python 开销 | 以上都不是 | 列表推导、字典操作、`repr`、logging;以及这些行直接调用的 xxhash、numpy、torch 的 C 代码 | 算 |

列说明:“类别”是归类结果;“判断规则”是怎么从样本判断;“例子”是落进这一类的代码;“算不算 Python 开销”按 1.1、1.2 节。

**哪些行算“调用 IAXL 扩展”**:分析脚本用 Python 的 `ast` 模块读 kvshrink 的源码,找出两类调用所在的行:
- 对名字是 `ctx`、`Context`、`_iqt`、`record`、`mem`、`storage` 的对象调用方法(包括 `result.ctx.xfer_wait()` 这样的属性):这些名字在 kvshrink 里都指向 IAXL 扩展对象;
- 直接调用 `task_queue_set_bundles_priority`、`task_queue_sort_priority_level`、`Record`、`Mem`、`Storage`。

跨多行的调用,第一行和方法名所在的行都算(Python 3.12 在调用执行期间报的是方法名那一行)。结果里的“附 1”列出被算作 IAXL C++ 的行和源代码,用来人工核对。

这样归的局限:
- **Python 本身和其他库的 C 代码分不开**:不加 `--native` 时看不到一行 Python 里面调用了哪个 C 函数。两者都算 Python 开销(1.1 节),合成一列,不影响要量的量。
- **参数转换和 IAXL 自己的 C++ 分不开**:两者都在调用 IAXL 扩展的那一行上。都不算 Python 开销,合成一列。
- 调用 IAXL 扩展的那一行如果还在同一行里用 Python 算参数(例如同一行里的列表推导),这部分 Python 时间也会算进 IAXL C++,Python 开销偏小(推断)。
- 不暂停方式下,约 11% 的主线程样本调用链不完整(4.3 节):最外面几层被截掉,最里面的帧还在。所以这些样本的“代码位置”和类别可信,“入口函数”可能不对(表 1 里会出现 `KVStoreLocal.get_wait` 这样的“入口”)。

3. 记下这个样本的两个位置:
   - **入口函数**:调用链里最外面的 kvshrink 帧,也就是 vLLM 调用的那个函数(2.2 节的表)。
   - **代码位置**:最里面的 kvshrink 帧的文件和行号。

### 4.5 样本怎么换算成时间

- 采样率为 r(次/秒)时,一个样本代表约 1/r 秒。采样率 100:一个样本约 10 ms。
- 某一类在一个线程里共 n 个样本:估计耗时 = n / r 秒。
- **每请求耗时(py-spy 直接数)** = 估计耗时 ÷ 这次 bench 发的请求数。采样覆盖整次 bench,前后各多几秒(那时 EngineCore 没有请求,主线程空闲,不产生样本)。
- **按内核校准**:校准系数 = 同一次 bench 内核记的主线程 CPU 时间 ÷ py-spy 数的主线程忙碌时间(4.3 节);校准后的毫秒/请求 = 直接数的值 × 校准系数。等价于“kvshrink Python 开销占 py-spy 主线程忙碌样本的比例 × 内核记的主线程 CPU 时间”。假设:py-spy 漏掉的样本均匀分布在各段代码上。如果读失败更多落在频繁调用 Python 函数的代码上,kvshrink 的 Python 开销仍会被低估;这一点用 py-spy 本身测不出来(推断)。其他线程和两个对照列不校准。
- **误差**:n 个样本的相对误差约 1/√n。例如 n = 100 时约 ±10%,n = 400 时约 ±5%。结果表里每个数都附上样本数。
- py-spy 跟不上采样率时会打印 `behind in sampling, results may be inaccurate`。出现这条,上面的换算就不准,要降低采样率重跑。

### 4.6 采哪些进程

| 阶段 | 采哪个进程 | 为什么 |
|---|---|---|
| 第一阶段,TP=1 | EngineCore | kvshrink 的全部函数都在它的主线程上(2.1 节) |
| 以后,TP>1 | EngineCore 和 Worker_TP0 | 调度侧函数在 EngineCore,worker 侧函数在 Worker |

列说明:“阶段”是哪种配置;“采哪个进程”见 2.1 节;“为什么”是要在这个进程里看的函数。

- TP>1 时,其他 Worker(TP1 …)做的事和 TP0 一样:处理同样的请求和块,只是 KV 的切片不同。所以只采 TP0(推断;第一次可以同时采 TP1 核对一次)。两个 py-spy 同时运行,各采一个进程。
- **找进程号**:容器用 `--pid host` 启动,容器里能看到这台机器上所有进程,包括别人的 vLLM。所以不能只按进程名找,要按端口和父子关系找:命令行是 `python3 …/vllm serve … --port 8020` 的进程 → 它的子进程 `VLLM::EngineCore`;TP>1 时再往下 → EngineCore 的子进程 `VLLM::Worker_TP0`。只认第一个词是 python 的进程:启动脚本的 bash 外壳和 `setvars.sh` 分出的子 shell 命令行里也有 `vllm serve … --port 8020`(11-03 因此误判过一次)。TP=1 时的父子关系 11-03 已用 `ps` 核对。

### 4.7 命令和选项

采样(容器内,每个进程一条):

```bash
taskset -c 60-63 \
    py-spy record --pid <EngineCore 进程号> --rate 100 --threads --format raw \
        --full-filenames --nonblocking -o <结果目录>/pyspy_enginecore.txt
# bench 结束后给 py-spy 发 SIGINT(相当于 Ctrl-C),它停止采样并写文件
```

开始前先存档一次当前调用链(只暂停一次,不影响测量):

```bash
py-spy dump --pid <EngineCore 进程号> --native
```

| 选项 | 作用 | 为什么这样选 |
|---|---|---|
| `record --pid <PID>` | 附着到已经在跑的进程,持续采样 | 不用改 vLLM 的启动方式 |
| `--rate 100` | 每秒采样 100 次(默认值) | 干扰小;样本不够就加请求数 |
| `--threads` | 每个样本带上线程编号和线程名 | 只统计主线程 |
| `--format raw` | 输出纯文本,一行一条调用链加样本数 | 方便脚本统计;同一个文件也能拖进 https://www.speedscope.app 看火焰图 |
| `--full-filenames` | 文件名写全路径 | 准确判断代码属于 kvshrink 还是 vLLM |
| `--nonblocking` | 采样时不暂停被测进程 | 暂停方式多数 59%、拖慢首 token(4.3、4.8 节);不暂停少数的 20% 按内核 CPU 时间校准(4.5 节) |
| 不加 `--native` | 调用链里只有 Python 函数 | IAXL 扩展库没有符号;而且 `--native` 不能和 `--nonblocking` 同时用(4.1 节) |
| 不加 `--idle` | 不记录空闲线程(默认) | 只统计 CPU 在忙的时间(4.3 节) |
| 不加 `--gil` | 不管线程有没有拿着 GIL 都记录 | GIL 是 Python 的全局锁,同一时刻只有一个线程能执行 Python;C++ 函数执行时常常先把它放开。加了 `--gil` 就看不到放开 GIL 后的 C++ 时间,没法和 Python 开销对照 |
| 不加 `--duration` | 一直采样,收到 SIGINT 才停并写文件 | 让采样覆盖整次 bench |
| `taskset -c 60-63` | 把 py-spy 自己限制在 NUMA 节点 0 的 4 个 CPU 上 | vLLM 绑在节点 1 的 CPU 64-127,py-spy 不和它抢 CPU |

列说明:“选项”是命令行参数;“作用”是它让 py-spy 做什么;“为什么这样选”是本次用或不用它的理由。选项含义、默认值和“`--native` 不能和 `--nonblocking` 同时用”按 py-spy 源码 `src/config.rs` 确认。

### 4.8 对 vLLM 的干扰怎么检查

同一次运行里,同样的 bench 先不开 py-spy 跑一次(基线),再开 py-spy 跑一次,比较两次的总时长、首 token 时间、每个输出 token 的时间和主线程的内核 CPU 时间。11-03 三种采样方式的冒烟测试(2k 输入、并发 16、64 个请求、全部命中):

| 采样方式 | bench 总时长 | 首 token 时间均值 | 每个输出 token 时间均值 | 主线程内核 CPU 时间 |
|---|---|---|---|---|
| 暂停 + `--native`(压缩开) | 1.82 | 4.65 | 1.67 | 没有测 |
| 暂停(压缩关) | 1.07 | 1.52 | 1.05 | 1.17 |
| 不暂停 `--nonblocking`(压缩关) | 1.01 | 1.05 | 1.01 | 0.94 |

列说明:每格是“开 py-spy 的 bench ÷ 没开 py-spy 的 bench”,越接近 1 干扰越小;“首 token 时间”和“每个输出 token 时间”是 `vllm bench serve` 的 mean_ttft_ms、mean_tpot_ms;“主线程内核 CPU 时间”见 4.3 节。第一行在加内核 CPU 时间核对之前跑的,是改成不加 `--native` 的原因。

每次正式运行的 `summary.md` 都带这两项检查。

### 4.9 py-spy 能回答什么、不能回答什么

能回答:
- 一次 bench 里,kvshrink 的 Python 开销(4.4 节)在主线程上一共花了多少 CPU 时间,折合每请求多少毫秒;
- 这些时间分在哪些入口函数、哪些代码行;
- 作为对照,ScratchPool、IAXL C++(含参数转换)各花了多少。

不能回答:
- **每次调用**多少 µs:py-spy 只给总量,不知道函数被调了几次;
- 这些时间有没有让 GPU 空等:属于 1.2 节先不看的 GPU 同步部分;
- 比采样间隔(10 ms)短的单次调用,只能靠大量重复在统计上看到,看不到某一次具体多长;
- Python 开销里有多少是 Python 语句本身、多少是 xxhash/numpy/torch 的 C 代码(4.4 节);
- 校准假设是否成立:py-spy 漏掉的样本是不是均匀分布在各段代码上(4.5 节)。

## 五、实验设计

### 5.1 第一阶段的配置

**参数**

| 参数 | 值 | 说明 |
|---|---|---|
| 模型 | `/media/model-space/Qwen3-32B` | 11-02 确认 |
| TP | 1(单卡单 rank) | 11-02 确认;多 rank 以后再看 |
| 两种负载 | 32k 输入、并发 1、50 个请求;2k 输入、并发 16、800 个请求 | 11-03 确认。32k 只放得下并发 1(见下面“显存”);2k 并发 16 看多个请求同时在跑时的开销 |
| 输出长度 | 128 token | 11-03 确认。只影响 decode 步数 |
| `--gpu-memory-utilization` | 0.9 | `examples/kvshrink-vllm-serve.sh` 默认 0.8,TP=1 时放不下一个 32k 请求 |
| `--max-model-len` | 32896(32768 + 128) | 两种负载用同一个值 |
| vLLM 的 prefix caching | 关(`--no-enable-prefix-caching`) | 否则同一个 prompt 第二次发时命中的是显存里的缓存,不走 kvshrink 的加载 |
| KV 压缩 | 关(`IAXL_KV_COMPRESSION=0`);DSA 也关(`IAXL_DSA_GD_ENABLE=0`) | 11-03 用户决定。QAT 的用户态进程名额整台机器共用,别人的任务占满时,vLLM 在第一次压缩时退出(11-03 发生过)。压缩关闭后 KV 原样存,kvshrink 不启动 QAT(确认:[kv_zip.cpp](../../iaxl/csrc/kv_zip/kv_zip.cpp#L294) 不压缩时直接返回,QAT 在第一次真正压缩或解压时才初始化;vLLM 日志 `compression=OFF`) |
| `IAXL_SCRATCH_POOL_SIZE_GB` | 16 | 默认 8。TP=1 时一个 32k 请求的全部层就要约 8 GiB(2047 块 × 64 层 × 每块 64 KiB),默认值一点余量都没有,会触发第三节第 6 项的空转 |
| `IAXL_DDR_POOL_SIZE_GB` | 64 | 默认按主机内存的 80% 算,太大:这台机器上还有别人的任务 |
| `IAXL_DDR_POOL_EVICT_INTERVAL_S` | 10 | 默认 30。全部未命中时内存池几十秒就写满,每 10 秒检查一次使用率 |
| 异步加载 | 默认(`setvars.sh` 的 `0-3:0,4-6:8,7-:16`) | 并发 1 时落在 `0-3:0`,即同步加载;并发 16 时落在 `7-:16`,前 16 层加载完就开始 prefill(确认:[setvars.sh](../../setvars.sh#L33)、connector [L320–L331](../../kvshrink/kvshrink_connector.py#L320-L331)) |
| 显卡和 CPU | GPU 5;vLLM 绑 CPU 64-127;py-spy 绑 CPU 60-63 | 11-03 用户同意用 GPU 5。GPU 4-7 在 NUMA 节点 1,它的物理核是 64-127 |
| 端口 | vLLM 8020;IAXL 管理接口 18720(controller)、18820(worker) | 避开这台机器上别人的服务(容器用主机网络) |

列说明:“参数”是 vLLM 或 IAXL 的设置;“值”是第一阶段用的值;“说明”是来源或理由。

**显存(11-03 vLLM 启动日志实测)**

| 项 | 数值 | 日志里的字段 |
|---|---|---|
| 模型权重 | 61.03 GiB | `Model loading took` |
| 留给 KV 的显存 | 10.81 GiB | `Available KV cache memory` |
| KV 能放的 token 数 | 44,272 | `GPU KV cache size` |
| 按 32,896 token 一个请求算的最大并发 | 1.35 | `Maximum concurrency` |

列说明:“项”是显存里的一部分;“数值”是日志里的值;“日志里的字段”是从哪一行来的。每个 token 的 KV 是 256 KiB(`config.json` 的 64 层 × 8 个 KV head × 128 维 × K、V 两份 × 2 字节)。

- 32k:一个请求 32,896 token,只放得下 1 个,所以只跑并发 1。
- 2k 并发 16:16 × (2048 + 128) = 34,816 token,小于 44,272,放得下。

**样本数**

- 2k 并发 16:800 个请求。11-03 冒烟测试(64 个请求、全部命中)采到 45 个 kvshrink Python 开销的样本,800 个请求预计约 560 个,相对误差约 ±4%(推断)。
- 32k 并发 1:50 个请求(11-02 定)。每个请求的块数是 2k 的 16 倍,预计样本不会少(推断);跑完看实际样本数,不够再加请求。
- 相对误差 ≈ 1/√样本数(4.5 节)。

**两个场景**

保存和加载走的 Python 代码不同,分成两个场景。请求都由 `vllm bench serve` 的 random 数据集生成:

| 场景 | 走哪条路径 | 怎么构造请求 | 怎么证明 |
|---|---|---|---|
| 全部未命中 | 只有保存 | `--random-prefix-len 0 --random-input-len <输入长度>`;预热、基线、py-spy 三次 bench 用不同的 seed(1、2、3),每次都是新 prompt | 这次 bench 的每个请求在 connector 日志里都是 `externally-cached tokens: 0` |
| 全部命中 | 只有加载 | `--random-prefix-len <输入长度> --random-input-len 0`:一次 bench 里所有请求的 prompt 完全相同。先用同样的 seed(4)和请求数跑一次“存入”bench,把这个 prompt 存进 DDR,再跑基线和 py-spy 两次 | 每个请求的 `externally-cached tokens` 都等于“prompt 长度减 1,再向下取整到 block_size 的倍数”(算 hash 时去掉了最后一个 token;2k 时是 2032) |

列说明:“走哪条路径”是这个场景里 kvshrink 主要做的事;“怎么构造请求”是 bench 的参数;“怎么证明”是事后从日志核对场景对不对的依据,结果在每次运行的 `summary.md` “核对”表里。

- 存入 bench 必须和测量 bench 用同样的 seed 和请求数:random 数据集先按请求数抽样,再抽共享前缀,请求数不同,前缀就不同(确认:vLLM 0.29 random 数据集代码)。
- 全部未命中的 prompt 是从随机起点开始的连续 token,800 个 2k prompt 里可能有约 2 个完全相同(推断),核对表会显示出来。
- 压缩关闭后 DDR 里是原始 KV:全部命中只存一个 prompt(32k 约 8 GiB),不会触发自动淘汰;全部未命中会写满 64 GB 的内存池,自动淘汰在 kvshrink 的其他线程上运行,结果表里单独一行(11-03 32k 实测:使用率到 92% 时一次淘汰约 2000 组,用时 0.02 秒)。

### 5.2 一次运行的步骤

一次运行 = 一种负载 × 一个场景,由 `tests/python_overhead_dev/run_pyspy_bench.sh` 在宿主机的 tmux 里完成:

1. 检查:容器在运行,GPU 5 显存小于 1000 MiB,端口 8020 没有别的 vLLM。
2. 启动 vLLM(5.1 节的参数),serve 日志在运行时直接写进结果目录;起不来就失败退出。按 4.6 节找到 EngineCore,写 `meta.json`,做一次 `py-spy dump --native` 存档。
3. 预热:2 × 并发个请求,随机 prompt(seed 1),不计入结果。
4. 全部命中场景:存入 bench(5.1 节)。
5. 基线 bench:不开 py-spy。
6. 启动 py-spy(4.7 节),跑测量 bench,结束后给 py-spy 发 SIGINT。
7. 每次 bench 前后记下:vLLM 日志的字节位置(用来切出这次 bench 的 connector 日志;日志里有 `\r`,按行号会对不齐)、EngineCore 主线程和全进程的 utime + stime(4.3 节)。bench 后检查失败请求数为 0 且 EngineCore 还在,否则立刻失败退出(`vllm bench serve` 全部请求失败也返回成功)。
8. 停掉 vLLM,确认 GPU 5 显存释放。
9. 分析脚本 `analyze_pyspy.py` 写 `summary.md`(5.3 节)。

四次正式运行依次跑:32k 并发 1 未命中 → 32k 并发 1 命中 → 2k 并发 16 未命中 → 2k 并发 16 命中,某一次失败就停。

```bash
# 在宿主机仓库根目录,tmux 里
WORKLOAD=32k_c1 SCENARIO=miss tests/python_overhead_dev/run_pyspy_bench.sh
SMOKE=1 WORKLOAD=2k_c16 SCENARIO=hit tests/python_overhead_dev/run_pyspy_bench.sh   # 4 × 并发个请求,只验证流程
DRY_RUN=1 WORKLOAD=32k_c1 SCENARIO=miss tests/python_overhead_dev/run_pyspy_bench.sh # 只打印命令
```

### 5.3 产出

每次运行一份 `summary.md`,依次是:

1. **这次运行是什么**:负载、场景、py-spy 参数、代码 md5。
2. **核对**:基线和测量两次 bench 各自的请求数、外部命中的请求数和 `externally-cached tokens` 的取值(5.1 节“怎么证明”);`ScratchPool exhausted` 次数;py-spy 跟不上采样率的次数;py-spy 的样本数和读失败次数;主线程调用链不完整的占比。
3. **干扰检查**:基线和测量 bench 的总时长、吞吐、首 token 时间、每个输出 token 时间、端到端时间(均值和中位数)及比值(4.8 节)。
4. **py-spy 数的主线程忙碌时间对不对**:两次 bench 的主线程和全进程内核 CPU 时间、py-spy 数的主线程忙碌时间、校准系数(4.3、4.5 节)。
5. **表 1:按入口函数汇总**,列为:入口函数 | Python 开销(py-spy 直接数)| Python 开销(按内核校准)| 对照:ScratchPool | 对照:IAXL C++(含参数转换)| 样本数。最后两行是“主线程上 kvshrink 合计”和“其他线程上的 kvshrink”,表下写主线程忙碌总时间(py-spy 直接数和内核记的)以及 kvshrink Python 开销的占比。单位都是毫秒/请求。
6. **表 2:Python 开销最多的前 20 个代码位置**,列为:代码位置 | 源代码 | 所在函数 | 入口函数 | 毫秒/请求 | 占 Python 开销的比例 | 样本数。
7. **附 1**:被算作 IAXL C++ 的代码行(核对 4.4 节的归类)。**附 2**:各线程的样本数。

每张表下都有列说明。

### 5.4 文件位置

脚本在 `tests/python_overhead_dev/`:`run_pyspy_bench.sh`(运行)和 `analyze_pyspy.py`(分析)。结果目录是 `tests/python_overhead_dev/results/<时间>_<代码 md5>_<负载>_<场景>_<采样方式>/`,冒烟测试在 `results/smoke/` 下。代码 md5 是 kvshrink、iaxl 的 Python 文件和两个脚本一起算的。

| 文件 | 内容 |
|---|---|
| `vllm.log` | vLLM serve 日志,运行时直接写入 |
| `meta.json` | 参数、进程号、代码 md5、py-spy 版本和参数、时钟频率 |
| `pyspy_dump_enginecore.txt` | 第 2 步的 dump |
| `pyspy_enginecore.txt` | py-spy 原始样本(raw 格式) |
| `pyspy_enginecore.stdout` | py-spy 自己的输出:样本数、错误数、有没有跟不上 |
| `bench_<名字>.json`、`bench_<名字>.log` | 每次 bench 的结果和输出;名字是 warmup、store(只有命中场景)、baseline、pyspy |
| `bench_offsets.txt` | 每次 bench 前后 vllm.log 的字节位置 |
| `bench_cpu_ticks.txt` | 每次 bench 前后 EngineCore 主线程、全进程的 utime + stime(时钟周期数) |
| `summary.md` | 5.3 节的内容 |

列说明:“文件”是结果目录里的文件名;“内容”是里面存的东西。

## 六、运行前的准备(11-03 已完成)

1. **权限**:容器用 `--privileged --pid host` 启动([start.sh](../../start.sh#L26-L27)),py-spy 能读 EngineCore 的内存(dump 跑通)。
2. **安装**:容器里 `pip install py-spy`,版本 0.4.3。
3. **符号**:IAXL 扩展库 `iaxl/torch_ext.cpython-312-x86_64-linux-gnu.so` 没有符号(`nm` 显示 `no symbols`),所以不加 `--native`(4.1 节)。
4. **dump**:能看到 `MainThread` 和 kvshrink 的函数名。
5. **显存**:见 5.1 节。

## 七、已定和待定事项

已定(11-02、11-03 和用户确认):

| 事项 | 结论 |
|---|---|
| 请求数 | 32k 并发 1:50;2k 并发 16:800 |
| 并发 16 | 用 2k 输入在 TP=1 下跑 |
| 输出长度 | 128 |
| torch | 和 xxhash、numpy 一样算进 Python 开销 |
| 采样方式 | 不暂停(`--nonblocking`),不加 `--native`,结果同时给直接数和按内核校准的值 |
| 压缩和 DSA | 都关 |
| 显卡 | GPU 5 |

列说明:“事项”是要定的问题;“结论”是定下来的做法,理由见对应章节。

待定:
1. **第二阶段(多 rank)**:TP、并发等参数以后再定。
2. **压缩打开时**:这次没有测。压缩与否只改变 IAXL C++ 里做的事,kvshrink 的 Python 代码路径相同(推断,没有测过)。

## 八、环境

| 项 | 值 | 来源 |
|---|---|---|
| 容器 | `iaxl.vllm.qiuyu-python-overhead-dev`,镜像 `vllm-iaxl-dev-qiuyu-python-overhead-dev:1.5.1` | 11-03 启动 |
| 代码 | 分支 `perf/python-overhead-dev`,git 28f2d3e | `meta.json` |
| Python | 3.12.3 | 容器内实测 |
| torch | 2.13.0+cu130 | 容器内实测 |
| vLLM | 0.29.0 | `meta.json` |
| py-spy | 0.4.3 | `meta.json` |
| 显卡 | 8 × RTX 6000D,每张 85651 MiB | 宿主机 `nvidia-smi` |
| 主机内存 | 503 GB | 宿主机 `free -g` |
| CPU | 256 个逻辑 CPU,2 个 NUMA 节点(节点 0:0-63、128-191;节点 1:64-127、192-255) | 宿主机 `lscpu` |

列说明:“项”是软件或硬件;“值”是版本或大小;“来源”是从哪里查到的。

