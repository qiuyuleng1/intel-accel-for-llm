# OMP-Main 释放 tensor 的开销:问题、方案与无 GPU 实验

> 状态:**暂停(11-02 决定这个优化暂不做)**。本文是讨论稿 + 无 GPU 实验结果 + 方案 B 改动计划;IAXL 产品代码未修改。重新开始时,先补齐 12.9 节决定对 12.4–12.7 节代码的影响(保留 tensor 版本接口),并定下 12.7 节末尾的待定项。
>
> 代码基线:分支 `perf/omp-main-task-release-dev`,b37a1b6(11-02 已快进到 upstream/main)。
>
> `doc/design/task-release-overhead.zh-CN.md` 是之前开发时留下的分析稿,本文只引用其中的真实 vLLM 实测数据,不沿用其中的方案。

本文的结论分三种性质:**实测**(跑出来的数)、**确认**(读代码或读二进制确认)、**推断**(没有直接证据)。

## 一、背景:问题出在哪

### 1.1 vLLM 进程里的两个相关线程

线程是一个程序里能同时、独立往下执行的一条流水线。vLLM 每张 GPU 对应一个 worker 进程,每个 worker 进程里有:

| 线程 | 谁创建的 | 做什么 |
|---|---|---|
| 主线程 | Python 解释器启动时自带 | 执行 vLLM 的全部 Python 代码:准备每一步的输入、调用模型计算、调用 kvshrink 的加载和保存函数 |
| OMP-Main | IAXL 在 `TaskQueue::init()` 里用 C++ 创建 | 后台专门做解压(加载 KV)和压缩(保存 KV) |

### 1.2 GIL 与“线程档案”

- **GIL**:Python 的一把全局锁。同一时刻只有一个线程能读写 Python 对象,想读写就得先拿到这把锁。
- **线程档案(PyThreadState)**:Python 为每个要碰 Python 对象的线程保留的一份记录。Python 自己创建的线程(比如主线程)一直有档案;OMP-Main 是 C++ 创建的,没有档案,每次要碰 Python 对象时都要临时建一份,用完删掉。
- 确认:在容器里反汇编 torch 2.13 的 `libtorch_python.so`,torch 放掉一个 Python 对象引用的函数 `ConcretePyInterpreterVTable::decref` 依次做:建档案 → 拿 GIL → 计数减 1 → 删档案并放 GIL。

### 1.3 ScratchPool 是什么

`iaxl/kvflow/scratch_pool.py` 管理一块固定大小的 CPU 内存,作为 GPU 与内存缓存之间的中转区。

- **为什么需要**:内存缓存里存的是压缩后的数据,不能直接拷进 GPU;压缩和解压都在 CPU 上进行,需要一块 CPU 内存放未压缩的数据。GPU 异步拷贝还要求这块内存是锁页内存,而锁页内存分配很慢,所以启动时一次分好、反复使用。
- **大小与切分**:大小由 `IAXL_SCRATCH_POOL_SIZE_GB` 决定,默认 8 GB,旧文档里真实的 Qwen2.5-72B 实验用 16 GB。按“一层一个 KV block”的大小切成等大的块,Qwen2.5-72B TP=8 时一块 8 KB,16 GB 约 200 万块。
- **怎么用**(`iaxl/kvflow/flow.py`):
  - 加载:取若干块 → 解压到这些块 → 从这些块拷到 GPU → 完成后还回。
  - 保存:取若干块 → 从 GPU 拷到这些块 → 压缩进内存缓存 → 完成后还回。
- **现状的实现**:初始化时用 `pool.unbind(0)` 为每一块生成一个 Python tensor 对象,并挂上 `block_idx` 属性记住编号,一直保留。`allocate(n)` 返回 n 个这样的 Python tensor,`release()` 收回。只在主线程上调用,不加锁。
- 只在 GPU 场景使用;CPU 推理场景直接读写 KV cache,不经过它。

### 1.4 现状:加载一个请求的一层时发生什么(确认)

1. 主线程:从 ScratchPool 取约 500 个 block,拿到 500 个 Python tensor 对象。
2. 主线程:调用 `unzip_from_mem`,把这 500 个 Python tensor 传进 C++。C++ 为每个 tensor 留一个“引用”(表示“我在用它”)。按 torch 2.13 的规则,C++ 每留一个引用,对应 Python 对象的计数就加 1;此时主线程持有 GIL,代价小。
3. 任务进入 OMP-Main 的队列,任务里存着这 500 个引用。
4. OMP-Main:执行任务,把数据解压到这 500 个 block,并安排把它们拷到 GPU。
5. OMP-Main:任务结束,丢掉 500 个引用。每丢一个,就要把对应 Python 对象的计数减 1,所以每一个都要走一遍“建档案 → 拿 GIL → 减 1 → 删档案、放 GIL”。**这一步就是慢的地方。**
6. 主线程:等加载完成后,把 500 个 block 还给 ScratchPool。

旧文档在真实 vLLM 里实测:第 5 步(释放,release)约占 OMP-Main 总时间的 17.5–20.9%;一个请求一层的 release 中位数约 177 µs。

## 二、要优化的是哪个函数

两个函数都在 Python 和 C++ 之间传逐个 block 的 tensor,而且都从 ScratchPool 拿 tensor 列表:

| 函数 | 做什么 | tensor 在哪个线程释放 | 影响 |
|---|---|---|---|
| `unzip_from_mem` | 加载:从内存缓存取出 → 解压 → 拷到 GPU | OMP-Main | 拖慢加载。旧文档测到的约 20% 就是它,**是这次的主要目标** |
| `zip_to_mem` | 保存:从 GPU 拷出 → 压缩 → 存进内存缓存 | 主线程,在等待保存完成的 `zip_wait()` 里(读代码推断,6.4 节实测确认) | 拖慢主线程 |

方案 B 会把 ScratchPool 改成只给编号,所以两个函数都要跟着改,两个都受益。

## 三、方案 B(已选定)与 B'

### 3.1 方案 B:改成传编号后的流程

1. 主线程:从 pool 取约 500 个 block,拿到 500 个整数编号。
2. 主线程:调用新的 `unzip_from_mem`,传入 500 个整数。C++ 把它们转成 C++ 的整数以后,和 Python 那边就没有关系了。C++ 用“pool 起始地址 + 编号 × 每块字节数”算出 500 个内存地址。
3. 任务里存这 500 个地址(普通整数)。
4. OMP-Main:解压到这 500 个地址,并安排拷到 GPU(拷贝函数本来就是按地址拷的)。
5. OMP-Main:任务结束,释放一个整数数组。**不涉及任何 Python 对象,不需要 GIL。**
6. 主线程:把 500 个编号放回 pool 的空闲列表。

为什么能解决:第 5 步慢,是因为 C++ 手里拿着 Python 对象的引用,放手时必须改 Python 的计数。方案 B 下 C++ 从头到尾不拿任何“每个 block 一个”的 Python 对象,放手时也就没有要还给 Python 的东西。

pool 的内存不会被提前释放:pool 在 C++ 里有一个代表对象(第十二节的 `BlockPool`),它持有整块 pool。整个进程只建这一次;每个任务只额外持有这个代表对象的一份 C++ 共享指针,增减它的计数不碰 Python。

保存路径同理:`zip_to_mem` 和“从 GPU 拷到 pool”的函数都改为收编号,任务里存地址,`zip_wait()` 里也就不再逐个 block 拿 GIL。

### 3.2 方案 B'(不采用)

Python 同样只传编号,但 C++ 启动时为每一块预先建一个 C++ tensor 对象(与 Python 无关),任务里存这些 C++ tensor;任务结束时对 500 个 C++ tensor 各做一次计数减 1(不用 GIL)。

不采用的原因:
- 实测剩下的释放时间是 B 的 1.6–1.9 倍(6.2 节);
- 每块要常驻一个 C++ 对象(16 GB pool 约 200 万个)。

它唯一的好处是解压、压缩函数基本不用改。

### 3.3 解压、压缩代码要不要改

方案 B 要改,但只改“从哪里拿到地址和大小”,不改算法。

`iaxl/csrc/kv_zip/kv_zip.cpp` 的 `kv_zip_compress_batch` / `kv_zip_decompress_batch` 现在接收 tensor 列表,但只从 tensor 上读四样东西:

| 读的东西 | 用途 | 位置 |
|---|---|---|
| 数据地址(`data_ptr()`) | 从哪里读、往哪里写 | 压缩:第 305、318 行;解压:第 456 行 |
| 字节数(`numel × element_size`) | 块大小,解压时用来核对 | 第 121、280、426 行 |
| 每个元素的字节数(`element_size()`) | lossy 截断的掩码 | 第 319 行 |
| 是否 bf16(`dtype()`) | 是否做 shuffle | 第 320 行 |

改成直接传“地址列表 + 块字节数 + dtype”以后,QAT/IAA/CPU 的调用、segment 格式、shuffle、lossy 都不动。风险在于这几项参数传错会静默写坏数据,所以必须做改前/改后对照的逐字节测试(第十二节)。

**为什么要传 dtype**

方案 B 里 dtype 只在一个地方用到:压缩函数 `kv_zip_compress_batch`。
- 解压不需要:压缩时用了哪些设置(shuffle、lossy),都写在缓存里每个块的块头里,解压时直接读。
- `zip_to_mem` / `unzip_from_mem` 的调用方不传 dtype:它们只传 pool,压缩时由 C++ 从 `pool->dtype()` 取。dtype 只在建 `BlockPool` 时从 ScratchPool 那块内存上读一次。

压缩需要 dtype 的两个原因(现在都从 tensor 上读):

| 用途 | 现在从哪读 | 为什么需要 |
|---|---|---|
| lossy 截断的掩码 | 第 319 行 `t.element_size()` | 截断的是每个元素的低位,要知道一个元素占几个字节(bf16 是 2) |
| 要不要做 shuffle | 第 320 行 `t.dtype() == kBFloat16` | 只有 bf16 才做字节重排 |

方案 B 去掉了 tensor,这两样信息只能显式告诉压缩函数,所以压缩函数多了一个 dtype 参数。

Python 侧还有一些 `shape` / `dtype` 参数,但它们和方案 B 无关:
- `ScratchPool.allocate(count, shape, dtype)`、`available_count(shape, dtype)`、`total_count(shape, dtype)`、`KVFlow._allocate(count, shape, dtype)` 是以前设计留下的。
- 现有代码里这些参数本来就没被用到;加上 12.7 节 `_ensure_pool` 的检查后更是多余。
- 这次删不删待定(12.7 节)。

### 3.4 BlockPool 和 ScratchPool 的关系

BlockPool 主要是算地址(offset)用的,另外负责让内存保持有效。两者用同一块内存,分工不同:

| | ScratchPool(Python,已有) | BlockPool(C++,新加) |
|---|---|---|
| 谁建 | KVFlow 第一次 put/get 时建 | ScratchPool 初始化时顺手建一个:`self.block_pool = BlockPool(self.pool)` |
| 管内存吗 | 是,分配那一大块锁页内存(`self.pool`) | 不分配,只引用同一块内存 |
| 管“哪些块空闲”吗 | 是:空闲编号列表,`allocate()` / `release()` | 不管 |
| 做什么 | 在主线程上发编号、收编号 | ① 编号 → 地址(pool 起始地址 + 编号 × 每块字节数),并检查越界;② 保活:每个 C++ 任务持有它的一份共享指针,任务没结束内存就不会被释放;③ 告诉 C++ 每块字节数和 dtype |

列说明:两列分别对应两个对象;各行依次对比它们的创建、内存、空闲管理和职责。

加载一个请求的一层时,两者这样配合:
1. 主线程:`slots = scratch_pool.allocate(500)`,拿到 500 个编号。
2. 主线程:`ctx.unzip_from_mem(..., scratch_pool.block_pool, slots)`。
3. C++:`pool->addresses(slots)` 算出 500 个地址存进任务;任务同时拿一份 BlockPool 的共享指针。
4. OMP-Main:按地址解压;任务结束时只释放地址数组和共享指针,不碰 Python。
5. 主线程:加载完成后 `scratch_pool.release(slots)`。

为什么不把 ScratchPool 整个搬进 C++:空闲列表只在主线程上用,没有性能问题;搬进 C++ 改动更大,对这次的释放问题没有帮助。

实验里用 memcpy 代替解压,不影响对 release 的测量:release 只取决于任务结束时手里拿着什么,和任务里怎么解压无关(推断)。

## 四、实验是怎么做的

### 4.1 为什么不用 vLLM

只想测“释放”这一步。为此反复启动 vLLM 不合理,而且 GPU 当时不能用。

### 4.2 在哪里跑

- **容器**:`iaxl.vllm.qiuyu-omp-main-task-release-dev`,镜像 `vllm-iaxl-dev-qiuyu-omp-main-task-release-dev:1.5.1`。这个镜像就是 10-26 为 1026 代码编的那个,只是加了一个新名字;Dockerfile 和依赖此后都没变。
- **启动方式**:按 README 用 `./start.sh` 启动,启动时 `pip install -e .` 用当前分支代码编译 IAXL。`start.sh` 已在本地去掉 `--rm`:启动它的终端关闭后,容器会停止但不会被删除,可以用 `docker start <容器名>` 恢复。
- **用到的 IAXL 部分**:实验程序只用了三样,任务队列 `TaskQueue`(`task_queue.h`)、拼 label 的函数 `make_chunk_labels`(`kv_pool.h`)和 `ScratchPool`。

### 4.3 实验程序和真实函数的异同

实验程序是 `tests/omp_release_dev/release_bench.py` 加一个小的 C++ 部分 `release_bench_ext.cpp`(运行时现编)。拿现状这种传法和真实的 `unzip_from_mem` 逐步对比:

| 步骤 | 真实 `unzip_from_mem` | 实验 | 是否一样 |
|---|---|---|---|
| Python 传入的三个列表 | label 字符串、GPU block 编号、block tensor,各约 500 个 | 同样三个列表 | 一样 |
| 列表转成 C++ 数据 | pybind 逐个元素转换 | 同 | 一样 |
| 拼完整 label | `kv_pool::make_chunk_labels`,层名用层号(upstream #49 起) | 调用同一个函数,层名用层号 | 一样 |
| 任务里存的东西 | label、编号、tensor、完成通知 | 同样四样 | 一样 |
| 交给哪个队列 | IAXL 的 `TaskQueue`(OMP-Main) | `TaskQueue` 的另一个实例 | 同一份代码 |
| 任务里做什么 | 查缓存、解压、提交拷到 GPU | 把每个 block 的 8 KB 写一遍,代替解压 | **不一样** |
| 任务结束后 | OMP-Main 销毁任务存的东西 | 同 | 一样 |

结论:要释放的东西和真实代码完全相同,不同的只是任务里干的活。

### 4.4 一个任务、一轮,以及释放时间怎么算

- **一个任务**:模拟一个请求的一层。主线程从 pool 拿 N 个 block 交给 C++;队列线程把 N 个 block 各写一遍,然后释放。
- **一轮**:模拟一个请求,连续提交 80 个任务(80 层)。
- **释放时间**:队列线程无法直接给“释放”计时,所以记下每个任务的开始和结束时刻,用两者之差代表:
  - **间隔 = 下一个任务的开始时刻 − 本任务的结束时刻 ≈ 释放时间**。其中“从队列取下一个任务”只占约 1 µs(见“下限”传法)。
  - 这要求下一个任务提前排好队,所以每轮先把队列停住,把 80 个任务都提交好再放开。
  - 一轮 80 个任务产生 79 个间隔。

### 4.5 改变的三个条件

**条件 1:每个任务的 block 数**:500 / 1000 / 2000,对应 8k / 16k / 32k 输入(每 block 16 个 token)。

**条件 2:传法**

| 传法 | 传什么 | 模拟什么 |
|---|---|---|
| 现状(`py_tensors`) | Python tensor + label + 编号 | 现状 `unzip_from_mem` |
| 方案 B(`slots`) | 整数编号,C++ 算地址;label、编号照旧 | 方案 B |
| 方案 B'(`cpp_views`) | 整数编号,C++ 用预先建好的 C++ tensor;label、编号照旧 | 方案 B' |
| 下限(`slots_no_labels`) | 只传编号,不传 label | 用来说明方案 B 剩下的时间是 label 字符串 |
| 保存形态(`py_tensors_kept_future`) | 和现状一样,但按 `zip_to_mem` 的方式持有,由主线程等待时释放 | 现状 `zip_to_mem` |

**条件 3:主线程在做什么。** OMP-Main 释放时要拿 GIL,能不能马上拿到,取决于同一进程的主线程那一刻在干什么:

| 主线程状态 | 实验里主线程在做什么 | 对应 vLLM 主线程的什么时候 | OMP-Main 什么时候能拿到 GIL |
|---|---|---|---|
| 等 GPU(`wait_in_cpp`) | 停着等待,不占 GIL | 等 GPU 算完、等结果 | 随时能拿 |
| 调 torch 函数(`torch_ops_loop`) | 不停调用很小的 torch 函数 | 跑模型时,Python 一个接一个地调用 torch 函数,让 GPU 计算 | torch 每个函数进入 C++ 部分时会临时放开 GIL,OMP-Main 要等到这个空档 |
| 纯 Python(`pure_python_loop`) | 不停跑纯 Python 循环 | 跑调度、整理数据等纯 Python 逻辑 | Python 默认每 5 ms 才强制让出一次,最坏情况 |

真实 vLLM 主线程在这三种状态之间切换,比例未知,所以三种都测。

### 4.6 其他固定设置(都与真实运行一致)

| 设置 | 取值 | 为什么 |
|---|---|---|
| malloc(分配内存的库) | tcmalloc,`TCMALLOC_RELEASE_RATE=0` | 真实 vLLM 由 `examples/kvshrink-vllm-serve.sh` 启动,它预加载 tcmalloc,并读 `setvars.sh`(upstream #50 起默认 `TCMALLOC_RELEASE_RATE=0`) |
| label 里的层名 | 层号,如 `kv:<hash>:12` | upstream #49 起 connector 用层号 |
| pool | 16 GB | 旧文档里真实运行时的大小 |
| CPU | 整个实验进程限制在 CPU 16–31(16 个) | 真实运行时一个 rank 的主线程和 IAXL 线程共用一段 16 个 CPU,本机 TP=8 时 rank 1 = 16–31(第七节)。两个线程都没有绑到单个核 |

- **公平性**:几种传法按轮轮流跑(第 1 轮现状、第 1 轮 B、第 1 轮 B'……再第 2 轮),这样运行过程中机器状态的变化对各传法影响相同。
- **预热**:每种组合先跑 1 轮,不计入结果。
- **样本数**:每种组合正式跑 10 轮,共 790 个间隔,报告中位数和均值。纯 Python 状态下现状太慢(一个任务 0.5–3.4 秒),只跑 3 轮 × 8 个任务,共 21 个间隔。
- **运行次数**:每种 block 数跑一次,共三次运行。

## 五、结果怎么读

所有数字都来自第十一节列出的三个结果目录(11-02 运行,每种 block 数一个)。每个单元格写成“中位数 / 均值”,单位 µs;“省”是方案 B 相比现状省掉的比例。

## 六、结果(实测)

### 6.1 加载路径:OMP-Main 上的释放时间

| 主线程状态 | block 数 | 现状 | 方案 B | 省(按中位数) | 省(按均值) |
|---|---|---|---|---|---|
| 等 GPU | 500 | 147.0 / 147.3 | 5.6 / 5.8 | 96.2% | 96.1% |
| 等 GPU | 1000 | 279.0 / 283.5 | 10.2 / 10.3 | 96.3% | 96.4% |
| 等 GPU | 2000 | 558.2 / 562.4 | 18.7 / 18.8 | 96.7% | 96.7% |
| 调 torch 函数 | 500 | 1673.7 / 1675.9 | 5.7 / 5.8 | 99.7% | 99.7% |
| 调 torch 函数 | 1000 | 3035.8 / 3009.0 | 10.9 / 11.0 | 99.6% | 99.6% |
| 调 torch 函数 | 2000 | 5435.6 / 5592.0 | 20.1 / 20.2 | 99.6% | 99.6% |
| 纯 Python | 500 | 511035 / 512713 | 6.0 / 5.9 | 约 100% | 约 100% |
| 纯 Python | 1000 | 1815583 / 1844258 | 11.4 / 11.3 | 约 100% | 约 100% |
| 纯 Python | 2000 | 3353240 / 3431777 | 22.4 / 21.6 | 约 100% | 约 100% |

列说明:
- 一行 = 一种主线程状态 × 一种 block 数。
- 现状 / 方案 B = 一个任务(一个请求的一层)的释放时间(间隔),中位数 / 均值。前两种主线程状态各 790 个样本,纯 Python 21 个。
- 纯 Python 时现状的单位仍是 µs,即约 0.51 / 1.82 / 3.35 秒。

结论:
- 现状的释放时间和 block 数成正比:主线程等 GPU 时每个 block 约 0.28–0.29 µs,主线程忙时更慢。三种 block 数下各轮的中位数都很稳定。
- 方案 B 在所有情况下省 96% 以上。剩下的 5.6–18.7 µs(主线程等 GPU 时)主要是释放 label 字符串:不传 label 的“下限”传法只剩 0.9 / 1.1、1.2 / 1.4、1.5 / 1.7 µs(500 / 1000 / 2000)。

### 6.2 方案 B 对比方案 B'(主线程状态:等 GPU)

| block 数 | 方案 B | 方案 B' |
|---|---|---|
| 500 | 5.6 / 5.8 | 9.2 / 9.3 |
| 1000 | 10.2 / 10.3 | 18.1 / 18.2 |
| 2000 | 18.7 / 18.8 | 35.6 / 35.8 |

说明:数值是释放时间,中位数 / 均值,µs。B' 是 B 的 1.6–1.9 倍,因为任务里仍有约 500 个 C++ tensor 要逐个计数减 1。

### 6.3 加载路径:主线程上每个任务的开销(主线程状态:等 GPU)

| block 数 | 从 pool 取 block | 提交给 C++ | 还 block 给 pool | 三项合计 |
|---|---|---|---|---|
| 500 现状 | 10.1 / 10.4 | 82.8 / 83.3 | 20.8 / 21.3 | 113.7 / 115.0 |
| 500 方案 B | 1.9 / 1.9 | 64.0 / 64.3 | 1.1 / 1.1 | 67.0 / 67.3 |
| 1000 现状 | 19.9 / 20.6 | 162.3 / 163.9 | 44.0 / 48.4 | 226.2 / 232.9 |
| 1000 方案 B | 3.9 / 4.0 | 127.5 / 128.1 | 2.7 / 2.7 | 134.1 / 134.8 |
| 2000 现状 | 41.2 / 41.3 | 327.1 / 328.7 | 88.1 / 93.2 | 456.4 / 463.2 |
| 2000 方案 B | 7.6 / 7.7 | 252.8 / 254.1 | 5.3 / 5.3 | 265.7 / 267.1 |

列说明:
- 每格是一个任务(一层)一次调用的耗时,中位数 / 均值,µs,各 800 个样本。
- 三项合计 = 三列直接相加(分别对中位数和均值相加)。
- “提交给 C++”包括把三个列表逐个转成 C++ 数据。

结论:
- 方案 B 在主线程上每层省约 47 / 92 / 191 µs(500 / 1000 / 2000,按中位数,实测)。乘以 80 层,每个请求约省 3.7 / 7.4 / 15.3 ms(推断,按层数相乘)。
- 方案 B 下“提交”仍有 64–253 µs,其中 label 字符串约 50 / 100 / 199 µs(= 方案 B 的提交 − 不传 label 的提交 14.2 / 27.7 / 54.3 µs)。这属于第十节第 4 项,暂不做。

### 6.4 保存路径(`zip_to_mem`)

| block 数 | OMP-Main 上的间隔 | 主线程等待保存完成的那一次调用 |
|---|---|---|
| 500 | 1.2 / 1.3 | 38.8 / 38.4 |
| 1000 | 1.5 / 1.7 | 75.3 / 77.4 |
| 2000 | 1.5 / 1.7 | 149.8 / 152.7 |

说明:
- 数值是中位数 / 均值,µs。
- OMP-Main 上几乎没有释放,说明保存任务的 tensor 不在 OMP-Main 上释放,而是在主线程等待保存完成时释放(右列)。
- 方案 B 下右列应该会消失(推断,实验没有单独模拟保存路径的方案 B)。

### 6.5 和真实 vLLM 的对照

旧文档里真实 vLLM 一个请求一层(约 500 个 block)的释放中位数约 177 µs;本实验 500 个 block、主线程等 GPU 时为 147 µs,量级一致。

### 6.6 这个实验没覆盖什么

- 任务里没有真实的查缓存、解压、拷到 GPU,只测“释放”这一步。
- 三种主线程状态是人为构造的,真实 vLLM 在它们之间切换,比例未知。
- 方案 B 是在实验程序里模仿的。真实实现写完后,要用第十二节的验证步骤再测。端到端能快多少,要等 GPU 可用后跑 vLLM 才知道。

## 七、真实代码里线程怎么绑 CPU(确认)

| 谁 | 怎么绑 | 默认绑到哪 |
|---|---|---|
| vLLM worker 进程(包括主线程,以及之后创建的线程) | connector 初始化时 `iaxl/utils/affinity.py` 的 `bind_cpu_affinity()` 调 `os.sched_setaffinity`,用 `VLLM_CPU_OMP_THREADS_BIND` 里本 rank 的那一项 | `setvars.sh` 自动检测:每个 rank 用 GPU 所在 NUMA 节点上的一段 CPU。本机 TP=8 时每 rank 16 个:rank 0 = 0–15,rank 1 = 16–31,…;TP=2 时每 rank 32 个 |
| IAXL 的线程(OMP-Main、H2D、D2H 及压缩/解压工作线程) | connector 把 `KVSHRINK_CPU_OMP_THREADS_BIND` 里本 rank 的那一项写进 `IAXL_CPU_OMP_THREADS_BIND`,`TaskQueue` 起线程时 `bind_cpus()` 按它绑定 | 默认等于 `VLLM_CPU_OMP_THREADS_BIND`,即和主线程同一段 CPU |

列说明:“怎么绑”是代码位置和使用的环境变量;“默认绑到哪”是不设置这些变量时的值。

结论:真实运行时,主线程和 OMP-Main 共用本 rank 的同一段 CPU,都没有绑到单个核。实验按 TP=8 的 rank 1 设置(16–31)。

## 八、代码基线

11-02 已把本分支从 1c74d57 快进到 upstream/main 的 b37a1b6,新增 5 个提交:

| 提交 | 内容 | 对本工作的影响 |
|---|---|---|
| 29b2c88 (#45) | 内存缓存初始化时预热 | 无关 |
| ecc4bcd (#47) | scheduler 驱动的 LRU 淘汰 | 方案 B 要改的文件里只有 `kv_pool.h` 被改(+7 行,淘汰相关) |
| be405a7 (#48) | 用 C++20 编译 | `xfer.cpp` 的 lambda 捕获写法有改动,方案 B 要改这个文件 |
| fd58a31 (#49) | KVStore 的层名改用层号 | label 变短;实验已跟着改 |
| b37a1b6 (#50) | 默认 `TCMALLOC_RELEASE_RATE=0` | 实验已跟着设 |

更新方法(已执行):

```bash
git fetch upstream
git stash push setvars.sh           # 收起本地改的镜像名/容器名
git merge --ff-only upstream/main   # 本分支没有自己的提交,直接快进
git stash pop                       # 放回本地改动
```

本地改动 `setvars.sh`(镜像名、容器名)和 `start.sh`(去掉 `--rm`)已在本开发分支上单独提交(11-02);做 PR 时要去掉这个提交。

## 九、结论与下一步

**结论**

1. 加载路径的 release 几乎全部来自“任务持有每个 block 的 Python tensor”。
2. 方案 B 把它去掉约 96%(主线程等 GPU 时)到 99% 以上(主线程忙时),并且每个请求在主线程上再省约 3.7 / 7.4 / 15.3 ms(500 / 1000 / 2000 个 block)。
3. 采用方案 B,不采用 B'(3.2 节)。

**下一步**(11-02 决定暂不做,以下为重新开始时的顺序)

1. 按第十二节的计划实施(待确认)。
2. 按第十二节第 7 步验证:改前/改后逐字节对照、KVFlow 级别计时。
3. GPU 可用时跑端到端 vLLM。

## 十、其他类似的地方(暂不做)

先区分三类开销。“Python 和 C++ 之间传大数组都有开销”只对第 A 类普遍成立:

| 类 | 是什么 | 在哪个线程 | 哪些参数会有 |
|---|---|---|---|
| A | pybind 把 Python 列表逐个元素转成 C++ 数据 | 主线程,持有 GIL | 所有列表参数(int / str / tensor) |
| B | C++ 放掉最后一个额外引用时改 Python 计数,要拿 GIL | 谁最后放谁付;不是 Python 线程的还要建档案 | 只有 tensor(以及 Python 对象)元素 |
| C | 普通的堆分配、释放 | 分配在主线程,释放在 OMP-Main 或主线程 | `std::string` 等 |

热路径(每请求每层一次;8k 输入约 500 block、80 层):

| # | 位置 | 从 Python 传进 C++、长度和 block 数成正比的参数 | 开销 |
|---|---|---|---|
| 1 | `kvflow/flow.py` `get()` → `unzip_from_mem` | 约 500 个 label 字符串、约 500 个 GPU block 编号、约 500 个 tensor | A ×1500;B ×500(OMP-Main);C:新拼约 500 个字符串,在 OMP-Main 释放 |
| 2 | `kvflow/flow.py` `put()` → `zip_to_mem` | label、tensor | A ×1000;B ×500,在主线程 `zip_wait()`;C 同上 |
| 3 | `kvflow/flow.py` `put()` → `xfer_chunks_batch` | GPU block 编号、tensor | A ×1000;参数用完时主线程持有 GIL,约 500 次计数减 1,不用切线程 |
| 4 | 同一请求的 80 层 | label 和 GPU block 编号每层都一样,只有层名不同 | 每请求重复转换 80 次;实验中主线程上 label 部分每层约 50–199 µs(6.3 节) |
| 5 | `kvflow/scratch_pool.py` allocate / release | 不跨语言,但每层要建约 500 个元素的列表、读约 500 次 `t.block_idx` | 主线程 Python 开销;实验中每层 31–129 µs(6.3 节,两项相加) |
| 6 | `kvflow/flow.py` CPU in-place:`[tensor[i] for i in chunk_indices]` | 每层新建约 500 个 Python tensor | 比 GPU 路径更重:OMP-Main 释放时要在 GIL 下销毁整个 Python 对象(推断) |
| 7 | remote_pool daemon:`remote_pool/rpc.py` 的 `unpack_blocks` 还原成列表后再走 1–3 | 同 1–3 | 同 1–3;daemon 主线程是 Python 忙等循环,OMP-Main 更难拿到 GIL(推断,参考 6.1 节纯 Python 状态) |

列说明:“开销”用上一张表的 A/B/C 标注,“×N”是每请求每层发生的次数。

- 低频的地方(每请求一次):scheduler 里的 `Record.has`、`put_finish` 里的 `Record.submit`,只有 A 类开销。
- 确认没问题的地方:H2D/D2H 队列的 lambda 只捕获裸地址和 int64 编号(`csrc/torch_ext/xfer.cpp`);`Context.create` 每层只持有一个 GPU tensor。

方案 B 覆盖 1、2、3、5;4 和 7 单独讨论。6(CPU in-place)暂不改,等 GPU 路径改好、调通后再回来看(12.9 节)。

## 十一、文件与数据

| 路径 | 内容 |
|---|---|
| `tests/omp_release_dev/release_bench_ext.cpp`、`release_bench.py`、`run_release_bench.sh` | 实验程序;运行:`BLOCKS=500 tests/omp_release_dev/run_release_bench.sh` |
| `tests/omp_release_dev/results/20261102_030722_1c902c28_blocks500` | 500 个 block |
| `tests/omp_release_dev/results/20261102_030805_1c902c28_blocks1000` | 1000 个 block |
| `tests/omp_release_dev/results/20261102_030919_1c902c28_blocks2000` | 2000 个 block |
| `tests/omp_release_dev/results/superseded_20261030/` | 10-30 在旧代码基线上的结果,不再使用;只在本机保留,未提交(约 40 MB) |
| `tests/omp_release_dev/build/` | 实验程序的 C++ 部分在运行时编译出的产物,未提交 |

列说明:结果目录名 = 时间 _ 实验代码 md5 _ block 数。每个目录里:
- `summary.md`:汇总表;
- `tasks.json`:每个任务的开始、结束时刻和间隔;
- `main_thread_calls.json`:主线程每次调用的耗时;
- `meta.json`:参数、源文件 md5、torch / Python 版本、CPU、LD_PRELOAD、TCMALLOC_RELEASE_RATE;
- `run.log`:运行日志。

## 十二、方案 B 改动计划(待确认,尚未修改)

### 12.1 总览

| 步骤 | 文件 | 改什么 | 估计改动行数 |
|---|---|---|---|
| 1 | `iaxl/csrc/include/block_pool.h`(新增) | C++ 的 `BlockPool`:持有整块 pool,把编号换成地址并检查越界 | 约 45 |
| 2 | `iaxl/csrc/include/kv_zip.h`、`iaxl/csrc/kv_zip/kv_zip.cpp` | 压缩、解压改为接收地址 + 块字节数(+ dtype) | 约 25 |
| 3 | `iaxl/csrc/torch_ext/context.h`、`xfer.cpp` | `zip_to_mem` / `unzip_from_mem` 改为接收 pool + 编号;`xfer_chunks_batch` 增加接收地址的版本 | 约 35 |
| 4 | `iaxl/csrc/torch_ext/zip.cpp` | 解压任务和压缩 lambda 存地址和 pool,不再存 tensor | 约 30 |
| 5 | `iaxl/csrc/torch_ext/torch_ext.cpp` | 绑定 `BlockPool`;更新三个函数的绑定 | 约 25 |
| 6 | `iaxl/kvflow/scratch_pool.py`、`iaxl/kvflow/flow.py` | ScratchPool 发编号;KVFlow 在经过 ScratchPool 的路径上传编号(CPU in-place 暂不改,见 12.9 节);`_ensure_pool` 检查块形状和 dtype 与已有 pool 一致(12.7 节) | 约 60 |
| 7 | `tests/omp_release_dev/`(新增脚本) | 改前/改后逐字节对照;KVFlow 级别计时 | 约 200 |

列说明:“估计改动行数”是增加行加删除行的估计。产品代码合计约 220 行,验证脚本约 200 行。

**不改的地方**:
- `kvstore.py`、connector、remote_pool(都经过 `flow.py`,接口不变);
- RDMA 传输(仍收地址,地址 = pool 起始地址 + 编号 × 块字节数,和现在一样);
- `Context.xfer_chunks_batch(chunk_indices, cpu_tensors)` 这个旧的 Python 接口保留不删。

### 12.2 步骤 1:新增 `block_pool.h`

改后(新文件):

```cpp
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <vector>

// A contiguous CPU tensor cut into equal blocks along dim 0: the scratch pool, or a CPU KV cache
// layer read and written in place. Python passes block numbers (slots) and C++ turns them into
// addresses, so no per-block tensor reaches a task: dropping a Python-backed tensor on a queue
// thread takes the GIL, once per block.
class BlockPool {
  public:
    explicit BlockPool(const torch::Tensor &blocks) : blocks_(blocks) {
        TORCH_CHECK(blocks.device().is_cpu() && blocks.is_contiguous() && blocks.dim() >= 1 &&
                        blocks.size(0) > 0,
                    "BlockPool: blocks must be a non-empty contiguous CPU tensor");
        base_ = static_cast<char *>(blocks.data_ptr());
        num_blocks_ = blocks.size(0);
        block_bytes_ = blocks.stride(0) * blocks.element_size();
    }

    // The addresses of `slots`, in order.
    std::vector<char *> addresses(const std::vector<int64_t> &slots) const {
        std::vector<char *> result;
        result.reserve(slots.size());
        for (int64_t slot : slots) {
            TORCH_CHECK(0 <= slot && slot < num_blocks_, "BlockPool: slot ", slot,
                        " out of range [0, ", num_blocks_, ")");
            result.push_back(base_ + slot * block_bytes_);
        }
        return result;
    }

    size_t block_bytes() const { return static_cast<size_t>(block_bytes_); }
    c10::ScalarType dtype() const { return blocks_.scalar_type(); }

  private:
    // Held so the memory outlives every task that took addresses from this pool.
    torch::Tensor blocks_;
    char *base_ = nullptr;
    int64_t num_blocks_ = 0;
    int64_t block_bytes_ = 0;
};
```

说明:
- Python 侧持有一个 `BlockPool`;每个任务再持有它的一份 C++ 共享指针(`std::shared_ptr`)。共享指针的计数增减不碰 Python。
- 所以即使 Python 先放手,pool 内存也会活到最后一个任务结束。
- 编号越界在主线程提交时就报错,抛 Python 异常,不会走到 OMP-Main。

### 12.3 步骤 2:`kv_zip.h` / `kv_zip.cpp`

参数名用 `cpu_blocks`,因为函数内已有局部变量叫 `blocks`(`std::vector<ZipBlock>`)。

`kv_zip.h` 改前:

```cpp
void kv_zip_compress_batch(const std::vector<torch::Tensor> &tensors, std::vector<char *> &out_bufs,
                                                     std::vector<size_t> &out_sizes, std::vector<size_t> &orig_sizes,
                                                     bool compress = true);

void kv_zip_decompress_batch(const std::vector<const char *> &data_ptrs,
                                                         const std::vector<torch::Tensor> &tensors);
```

改后:

```cpp
// `cpu_blocks` are the addresses of equally sized blocks of `block_bytes` bytes of `dtype` elements.
void kv_zip_compress_batch(const std::vector<char *> &cpu_blocks, size_t block_bytes,
                           c10::ScalarType dtype, std::vector<char *> &out_bufs,
                           std::vector<size_t> &out_sizes, std::vector<size_t> &orig_sizes,
                           bool compress = true);

// Decompression reads the shuffle / lossy settings from each cached block's header, not from dtype.
void kv_zip_decompress_batch(const std::vector<const char *> &data_ptrs,
                             const std::vector<char *> &cpu_blocks, size_t block_bytes);
```

`kv_zip.cpp`,压缩入口,改前:

```cpp
void kv_zip_compress_batch(const std::vector<torch::Tensor> &tensors, std::vector<char *> &out_bufs,
                           std::vector<size_t> &out_sizes, std::vector<size_t> &orig_sizes,
                           bool compress) {
    const size_t n = tensors.size();
    for (size_t i = 0; i < n; i++) {
        const auto &t = tensors[i];
        IAXL_CHECK(t.is_contiguous() && t.device().type() == c10::DeviceType::CPU,
                   "kv_zip: tensor must be a contiguous CPU tensor");
        orig_sizes[i] = tensor_bytes(t);
        IAXL_CHECK(orig_sizes[i] <= static_cast<size_t>(INT_MAX),
                   "kv_zip: tensor byte size exceeds zip integer length range");
    }
```

改后(连续、CPU 两项检查由 `BlockPool` 构造时完成):

```cpp
void kv_zip_compress_batch(const std::vector<char *> &cpu_blocks, size_t block_bytes,
                           c10::ScalarType dtype, std::vector<char *> &out_bufs,
                           std::vector<size_t> &out_sizes, std::vector<size_t> &orig_sizes,
                           bool compress) {
    const size_t n = cpu_blocks.size();
    IAXL_CHECK(block_bytes <= static_cast<size_t>(INT_MAX),
               "kv_zip: block byte size exceeds zip integer length range");
    for (size_t i = 0; i < n; i++)
        orig_sizes[i] = block_bytes;
```

不压缩分支(第 305 行),改前 / 改后:

```cpp
            memcpy(buf + start, tensors[i].data_ptr(), h.orig_size);
```
```cpp
            memcpy(buf + start, cpu_blocks[i], h.orig_size);
```

压缩分支(第 316–320 行),改前:

```cpp
        const auto &t = tensors[i];
        ZipBlock &b = blocks[i];
        b.data = static_cast<char *>(t.data_ptr());
        b.mask = lossy_mask(static_cast<int>(t.element_size()));
        const bool shuffle = data_shuffle_enabled() && t.dtype() == torch::kBFloat16;
```

改后:

```cpp
        ZipBlock &b = blocks[i];
        b.data = cpu_blocks[i];
        b.mask = lossy_mask(static_cast<int>(c10::elementSize(dtype)));
        const bool shuffle = data_shuffle_enabled() && dtype == torch::kBFloat16;
```

解压(第 415–427 行、第 456 行),改前:

```cpp
void kv_zip_decompress_batch(const std::vector<const char *> &data_ptrs,
                             const std::vector<torch::Tensor> &tensors) {
    const size_t n = tensors.size();
    ...
        IAXL_CHECK(h.orig_size == tensor_bytes(tensors[i]),
                   "kv_zip: block size does not match tensor byte size");
    ...
        char *dst = static_cast<char *>(tensors[r.block].data_ptr()) +
                    size_t(r.index) * h.segment_size;
```

改后:

```cpp
void kv_zip_decompress_batch(const std::vector<const char *> &data_ptrs,
                             const std::vector<char *> &cpu_blocks, size_t block_bytes) {
    const size_t n = cpu_blocks.size();
    ...
        IAXL_CHECK(h.orig_size == block_bytes, "kv_zip: cached block size does not match block size");
    ...
        char *dst = cpu_blocks[r.block] + size_t(r.index) * h.segment_size;
```

第 121 行的 `tensor_bytes()` 不再使用,删除。其余代码(segment、QAT/IAA/CPU、shuffle、lossy)不动。

### 12.4 步骤 3:`context.h` / `xfer.cpp`

`context.h` 改前:

```cpp
    void xfer_chunks_batch(const std::vector<int64_t> &chunk_indices,
                           const std::vector<torch::Tensor> &cpu_tensors);
    ...
    void zip_to_mem(kv_pool::Mem &mem, const std::string &label, const std::string &tensor_key,
                    const std::vector<std::string> &chunk_labels,
                    const std::vector<torch::Tensor> &cpu_tensors, bool compress = true);

    // priority / bundle_id / sort_key go to TaskQueue::submit() of the decompression task.
    void unzip_from_mem(kv_pool::Mem &mem, const std::string &label, const std::string &tensor_key,
                        const std::vector<std::string> &chunk_labels,
                        const std::vector<int64_t> &chunk_indices,
                        const std::vector<torch::Tensor> &cpu_tensors,
                        int priority = TaskQueue::PRIORITY_HIGH, uint64_t bundle_id = 0,
                        int64_t sort_key = -1);
```

改后(另在文件头 `#include "block_pool.h"`):

```cpp
    void xfer_chunks_batch(const std::vector<int64_t> &chunk_indices,
                           const std::vector<torch::Tensor> &cpu_tensors);
    // Same transfer with the CPU blocks given by address; they must stay valid until xfer_wait().
    void xfer_chunks_batch(const std::vector<int64_t> &chunk_indices,
                           const std::vector<char *> &cpu_blocks);
    ...
    // Compresses `slots` of `pool`, one per chunk label.
    void zip_to_mem(kv_pool::Mem &mem, const std::string &label, const std::string &tensor_key,
                    const std::vector<std::string> &chunk_labels,
                    const std::shared_ptr<BlockPool> &pool, const std::vector<int64_t> &slots,
                    bool compress = true);

    // Decompresses into `slots` of `pool`, then copies them to `chunk_indices` of the GPU tensor;
    // priority / bundle_id / sort_key go to TaskQueue::submit() of the decompression task.
    void unzip_from_mem(kv_pool::Mem &mem, const std::string &label, const std::string &tensor_key,
                        const std::vector<std::string> &chunk_labels,
                        const std::vector<int64_t> &chunk_indices,
                        const std::shared_ptr<BlockPool> &pool, const std::vector<int64_t> &slots,
                        int priority = TaskQueue::PRIORITY_HIGH, uint64_t bundle_id = 0,
                        int64_t sort_key = -1);
```

`xfer.cpp` 改前(第 23–45 行):

```cpp
void Context::xfer_chunks_batch(const std::vector<int64_t> &chunk_indices,
                                const std::vector<torch::Tensor> &cpu_tensors) {
    PROFILE_SCOPE_FMT(...);

    auto chunk_indices_copy = chunk_indices;
    bool h2d = (direction_ == GpuTransferDirection::H2D);
    auto x = xctx_;
    auto ops = ops_;

    std::vector<char *> cpu_ptrs;
    cpu_ptrs.reserve(cpu_tensors.size());
    for (const auto &tensor : cpu_tensors) {
        cpu_ptrs.push_back((char *)tensor.data_ptr());
    }

    queue_->submit([=, this, chunk_indices = std::move(chunk_indices_copy)]() {
        PROFILE_SCOPE_FMT(...);
        ops->copy_chunks_batch(x, chunk_indices, cpu_ptrs, h2d);
    });
}
```

改后(tensor 版本只取地址,转给地址版本;传输逻辑只有一份):

```cpp
void Context::xfer_chunks_batch(const std::vector<int64_t> &chunk_indices,
                                const std::vector<torch::Tensor> &cpu_tensors) {
    std::vector<char *> cpu_blocks;
    cpu_blocks.reserve(cpu_tensors.size());
    for (const auto &tensor : cpu_tensors) {
        cpu_blocks.push_back((char *)tensor.data_ptr());
    }
    xfer_chunks_batch(chunk_indices, cpu_blocks);
}

void Context::xfer_chunks_batch(const std::vector<int64_t> &chunk_indices,
                                const std::vector<char *> &cpu_blocks) {
    PROFILE_SCOPE_FMT(...);  // 同改前

    auto chunk_indices_copy = chunk_indices;
    bool h2d = (direction_ == GpuTransferDirection::H2D);
    auto x = xctx_;
    auto ops = ops_;

    queue_->submit([=, this, chunk_indices = std::move(chunk_indices_copy)]() {
        PROFILE_SCOPE_FMT(...);  // 同改前
        ops->copy_chunks_batch(x, chunk_indices, cpu_blocks, h2d);
    });
}
```

### 12.5 步骤 4:`zip.cpp`

解压任务结构,改前:

```cpp
struct UnzipFromMemWork : std::enable_shared_from_this<UnzipFromMemWork> {
    Context *ctx;
    kv_pool::Mem *mem;
    std::vector<std::string> chunk_labels;
    std::vector<int64_t> chunk_indices;
    std::vector<torch::Tensor> cpu_tensors;
    std::shared_ptr<std::promise<void>> promise;
```

改后:

```cpp
struct UnzipFromMemWork : std::enable_shared_from_this<UnzipFromMemWork> {
    Context *ctx;
    kv_pool::Mem *mem;
    std::vector<std::string> chunk_labels;
    std::vector<int64_t> chunk_indices;
    // Addresses rather than tensors: this work is destroyed on OMP-Main, where dropping a
    // Python-backed tensor takes the GIL once per block. `pool` keeps the addresses valid.
    std::shared_ptr<BlockPool> pool;
    std::vector<char *> cpu_blocks;
    std::shared_ptr<std::promise<void>> promise;
```

`execute()` 里,改前:

```cpp
            kv_zip_decompress_batch(data_ptrs, cpu_tensors);
            METRICS_ADD_DECOMPRESS(metrics_decompress, metrics_sum_tensor_bytes(cpu_tensors));
        ...
            ctx->xfer_chunks_batch(chunk_indices, cpu_tensors);
```

改后:

```cpp
            kv_zip_decompress_batch(data_ptrs, cpu_blocks, pool->block_bytes());
            METRICS_ADD_DECOMPRESS(metrics_decompress, cpu_blocks.size() * pool->block_bytes());
        ...
            ctx->xfer_chunks_batch(chunk_indices, cpu_blocks);
```

`unzip_from_mem()`,改前:

```cpp
void Context::unzip_from_mem(..., const std::vector<int64_t> &chunk_indices,
                             const std::vector<torch::Tensor> &cpu_tensors, int priority,
                             uint64_t bundle_id, int64_t sort_key) {
    ...
    work->chunk_indices = chunk_indices;
    work->cpu_tensors = cpu_tensors;
```

改后:

```cpp
void Context::unzip_from_mem(..., const std::vector<int64_t> &chunk_indices,
                             const std::shared_ptr<BlockPool> &pool,
                             const std::vector<int64_t> &slots, int priority, uint64_t bundle_id,
                             int64_t sort_key) {
    TORCH_CHECK(slots.size() == chunk_labels.size() && chunk_indices.size() == chunk_labels.size(),
                "unzip_from_mem: one slot and one chunk index per chunk label");
    ...
    work->chunk_indices = chunk_indices;
    work->pool = pool;
    work->cpu_blocks = pool->addresses(slots);
```

`zip_to_mem()`,改前:

```cpp
void Context::zip_to_mem(..., const std::vector<std::string> &chunk_labels,
                         const std::vector<torch::Tensor> &cpu_tensors, bool compress) {
    ...
    const size_t n = cpu_tensors.size();
    ...
    auto cpu_tensors_copy = cpu_tensors;
    ...
                    kv_zip_compress_batch(cpu_tensors_copy, compressed_bufs, compressed_sizes,
                                          unzip_sizes, compress);
```

改后:

```cpp
void Context::zip_to_mem(..., const std::vector<std::string> &chunk_labels,
                         const std::shared_ptr<BlockPool> &pool,
                         const std::vector<int64_t> &slots, bool compress) {
    TORCH_CHECK(slots.size() == chunk_labels.size(),
                "zip_to_mem: one slot per chunk label");
    ...
    const size_t n = slots.size();
    ...
    // Addresses rather than tensors: the lambda is dropped where its future is released last,
    // usually zip_wait() on the Python thread, which would take the GIL per block for tensors.
    auto cpu_blocks = pool->addresses(slots);
    ...
                    kv_zip_compress_batch(cpu_blocks, pool->block_bytes(), pool->dtype(),
                                          compressed_bufs, compressed_sizes, unzip_sizes, compress);
```

lambda 原本用 `[=]` 捕获,改后会按值捕获 `pool`(共享指针)和 `cpu_blocks`(地址)。文件头的 `metrics_sum_tensor_bytes()` 不再使用,删除。

### 12.6 步骤 5:`torch_ext.cpp`

新增(放在 `Context` 绑定之前):

```cpp
    py::class_<BlockPool, std::shared_ptr<BlockPool>>(
        m, "BlockPool",
        "A contiguous CPU tensor cut into equal blocks along dim 0. zip_to_mem / unzip_from_mem /\n"
        "xfer_chunks_batch address its blocks by slot number.")
        .def(py::init<const torch::Tensor &>(), py::arg("blocks"));
```

`xfer_chunks_batch`:C++ 里有了两个同名函数,原绑定要用 `static_cast` 指明 tensor 版本(本文件里 `xfer_wait_stream` 已有同样写法)。另加一个接收 pool + 编号的绑定:

```cpp
        .def("xfer_chunks_batch",
             static_cast<void (Context::*)(const std::vector<int64_t> &,
                                           const std::vector<torch::Tensor> &)>(
                 &Context::xfer_chunks_batch),
             "Transfer batch of chunks (async)", py::arg("chunk_indices"),
             py::arg("cpu_tensors"), py::call_guard<py::gil_scoped_release>())
        .def(
            "xfer_chunks_batch",
            [](Context &self, const std::vector<int64_t> &chunk_indices,
               const std::shared_ptr<BlockPool> &pool, const std::vector<int64_t> &slots) {
                self.xfer_chunks_batch(chunk_indices, pool->addresses(slots));
            },
            "Transfer batch of chunks between the GPU tensor and `slots` of `pool` (async)",
            py::arg("chunk_indices"), py::arg("pool"), py::arg("slots"),
            py::call_guard<py::gil_scoped_release>())
```

`zip_to_mem` / `unzip_from_mem` 的参数名列表,改前 / 改后:

```cpp
             py::arg("cpu_tensors"), py::arg("compress") = true,                  // zip_to_mem 改前
             py::arg("pool"), py::arg("slots"), py::arg("compress") = true,      // zip_to_mem 改后

             py::arg("chunk_indices"), py::arg("cpu_tensors"),                    // unzip_from_mem 改前
             py::arg("chunk_indices"), py::arg("pool"), py::arg("slots"),        // unzip_from_mem 改后
```

### 12.7 步骤 6:`scratch_pool.py` / `flow.py`

`scratch_pool.py`,改前:

```python
from ..envs import envs
...
        self.pool = torch.empty((num_blocks, *self.block_shape), dtype=dtype, device="cpu", pin_memory=pin_memory)
        assert not pin_memory or self.pool.is_pinned(), "ScratchPool: pin_memory=True did not take effect"
        self._blocks: Tuple[torch.Tensor, ...] = self.pool.unbind(0)
        for i, t in enumerate(self._blocks):
            t.block_idx = i  # type: ignore[attr-defined]
        self._free: List[int] = list(range(num_blocks))  # LIFO: pop from the end
...
    @property
    def num_blocks(self) -> int:
        return len(self._blocks)

    def allocate(self, count: int, shape: Optional[Sequence[int]] = None,
                 dtype: Optional[torch.dtype] = None) -> Optional[List[torch.Tensor]]:
        ...
        idx = free[cut:]
        del free[cut:]
        self._allocate_count += count
        blocks = self._blocks
        return [blocks[i] for i in idx]

    def release(self, tensors: List[torch.Tensor]):
        self._free.extend(t.block_idx for t in tensors)  # type: ignore[attr-defined]
        self._release_count += len(tensors)
```

改后(`total_count()`、`status()` 里的 `len(self._blocks)` 一并改为 `self.num_blocks`,模块说明同步更新):

```python
from ..envs import envs
from ..torch_ext import BlockPool
...
        self.pool = torch.empty((num_blocks, *self.block_shape), dtype=dtype, device="cpu", pin_memory=pin_memory)
        assert not pin_memory or self.pool.is_pinned(), "ScratchPool: pin_memory=True did not take effect"
        # C++ turns slots into addresses in `pool`; no per-block Python tensor is created.
        self.block_pool = BlockPool(self.pool)
        self._free: List[int] = list(range(num_blocks))  # LIFO: pop from the end
...
    @property
    def num_blocks(self) -> int:
        return self.pool.shape[0]

    def allocate(self, count: int, shape: Optional[Sequence[int]] = None,
                 dtype: Optional[torch.dtype] = None) -> Optional[List[int]]:
        ...
        slots = free[cut:]
        del free[cut:]
        self._allocate_count += count
        return slots

    def release(self, slots: List[int]):
        self._free.extend(slots)
        self._release_count += len(slots)
```

`flow.py` 的 `Task`,改前 / 改后:

```python
    cpu_tensors: List[torch.Tensor] = None
```
```python
    # Scratch pool slots to release after the wait; None when the KV cache is read in place.
    slots: List[int] = None
```

`flow.py` 的 `__init__` 新增一行,以及一个 put / get 共用的小函数:

```python
        # CPU KV cache layers read in place, as block pools keyed by tensor key (created on first use).
        self._in_place_pools: Dict[str, BlockPool] = {}
```
```python
    def _in_place_pool(self, tensor_key: str, tensor: torch.Tensor) -> BlockPool:
        # A CPU KV cache layer is its own block pool: chunk i is slot i.
        pool = self._in_place_pools.get(tensor_key)
        if pool is None:
            pool = BlockPool(tensor)
            self._in_place_pools[tensor_key] = pool
        return pool
```

`put()`,改前:

```python
            if self._in_place:
                cpu_tensors = [tensor[i] for i in chunk_indices]
            else:
                cpu_tensors = self._allocate(num_chunks, chunk_shape, tensor.dtype)
            ...
            if not self._in_place:
                ctx.xfer_chunks_batch(chunk_indices, cpu_tensors)
            ...
            ctx.zip_to_mem(
                self.mem, label, tensor_key, chunk_labels, cpu_tensors, compress
            )

            results[tensor_key] = task = Task(
                ctx=ctx,
                cpu_tensors=None if self._in_place else cpu_tensors,
```

改后:

```python
            if self._in_place:
                pool = self._in_place_pool(tensor_key, tensor)
                slots = chunk_indices
            else:
                pool = self.chunk_pool.block_pool
                slots = self._allocate(num_chunks, chunk_shape, tensor.dtype)
            ...
            if not self._in_place:
                ctx.xfer_chunks_batch(chunk_indices, pool, slots)
            ...
            ctx.zip_to_mem(
                self.mem, label, tensor_key, chunk_labels, pool, slots, compress
            )

            results[tensor_key] = task = Task(
                ctx=ctx,
                slots=None if self._in_place else slots,
```

`get()` 同样修改:`unzip_from_mem(..., chunk_indices, pool, slots, priority=..., ...)`。

`put_wait()` / `get_wait()`,改前 / 改后:

```python
                self.chunk_pool.release(result.cpu_tensors)
                result.cpu_tensors = None
```
```python
                self.chunk_pool.release(result.slots)
                result.slots = None
```

`get_wait()` 里的判断 `if result.cpu_tensors is not None:` 改为 `if result.slots is not None:`。`_allocate()` 的变量名 `tensors` 改为 `slots`,返回类型改为 `List[int]`。

`flow.py` 的 `_ensure_pool()`(11-02 确认加入):现有代码只在第一次调用时按当时的块形状和 dtype 建一个 ScratchPool,之后的调用即使块形状不同也照样用它。改为不一致就报错。改前:

```python
    def _ensure_pool(self, block_shape: Tuple[int, ...], dtype: torch.dtype):
        if self.chunk_pool is None:
            self.chunk_pool = ScratchPool(
                block_shape, dtype, pin_memory=self.device_type not in (None, "cpu")
            )
            if envs.IAXL_RDMA_ENABLE:
                pool = self.chunk_pool.pool
                _iqt.rdma_register_local(pool.data_ptr(), pool.nbytes, self.chunk_pool.block_bytes)
```

改后:

```python
    def _ensure_pool(self, block_shape: Tuple[int, ...], dtype: torch.dtype):
        if self.chunk_pool is None:
            self.chunk_pool = ScratchPool(
                block_shape, dtype, pin_memory=self.device_type not in (None, "cpu")
            )
            if envs.IAXL_RDMA_ENABLE:
                pool = self.chunk_pool.pool
                _iqt.rdma_register_local(pool.data_ptr(), pool.nbytes, self.chunk_pool.block_bytes)
            return
        # One scratch pool serves every layer. Hybrid models fit because vLLM gives all layers the
        # same page size, but a layer with another block shape or dtype would be copied with the
        # wrong block size, so refuse it.
        if block_shape != self.chunk_pool.block_shape or dtype != self.chunk_pool.dtype:
            raise ValueError(
                f"KV block {block_shape} {dtype} differs from the scratch pool's "
                f"{self.chunk_pool.block_shape} {self.chunk_pool.dtype}: one scratch pool holds "
                "one block shape and dtype"
            )
```

`put()` / `get()` 传入的 `block_shape` 已经是 tuple,`ScratchPool.block_shape` 也是 tuple,可以直接比较。

待定:`ScratchPool.allocate` / `available_count` / `total_count` 和 `KVFlow._allocate` 里从未使用的 `shape`、`dtype` 参数(3.3 节)这次是否一并删除,约 10 行。

### 12.8 步骤 7:验证

| 验证 | 怎么做 | 通过标准 |
|---|---|---|
| 改前/改后逐字节对照 | 新脚本在 GPU 卡 4 上用 `KVFlow` 做 put → put_wait → get(写到另一组 GPU block)→ get_wait,保存 get 出来的数据。**改代码前**先跑一遍,存为对照;改后再跑,逐字节比较。配置逐一覆盖:bf16 / fp8、shuffle 开/关、lossy 0 / 2、压缩开/关、`IAXL_ZIP_SEGMENT_SIZE_KB` 小到一个 block 分多个 segment;每种配置连续多轮,使 slot 反复复用 | 改后与改前完全一致;不压缩、不 lossy 时,get 出来的数据还等于放进去的数据 |
| 编号越界 | 传一个超出范围的编号 | 抛 Python 异常,进程不崩 |
| 释放时间(真实代码) | 同一脚本对“一个请求 80 层、每层 500 / 1000 / 2000 个 block”的 get,计时“第一层提交 → 最后一层 `unzip_wait` 返回”,改前 / 改后各跑 10 次 | 报告中位数 / 均值;改后应少约 80 × 第 6.1 节的单层差值 |
| CPU in-place 路径 | 需要 `DEVICE=cpu` 编译的 IAXL。在单独的 git worktree 里编一份(不覆盖现在的 CUDA 版),跑同样的对照 | 同第一项 |
| 端到端 | GPU 可用时,vLLM + 正确性检查(`VLLM_BATCH_INVARIANT=1`、temperature 0、命中前后逐字节比较、扫乱码)+ 性能 | 输出一致;release 下降 |

列说明:“通过标准”是这一项算通过的条件。前三项不需要 vLLM,只用 GPU 卡 4、5 和 QAT。

### 12.9 CPU in-place 路径:暂不改(11-02 决定)

**是什么**
- IAXL 也支持 vLLM 跑在 CPU 上(`DEVICE=cpu` 编译,见 `doc/usage/vllm-cpu-inference.en.md`)。这时 KV cache 本来就在 CPU 内存里,压缩、解压直接读写 KV cache 的块,不经过 ScratchPool,这叫 in-place(就地)。
- 开关在 `flow.py` 的 `_in_place`:CPU 部署且没开 shuffle / lossy 时为真。
- 它把 block 交给 C++ 的方式是 `[tensor[i] for i in chunk_indices]`:每层新建约 500 个 Python tensor。所以同样有逐 block 拿 GIL 的问题,而且可能更重(推断,未测)。
- 我们的 GPU 版 IAXL 永远不会走这段代码,要测它得另编一份 `DEVICE=cpu` 的 IAXL。

**决定**:先把 GPU 路径改好、调通,再回来看它。这次实施对计划的影响:
- C++ 的 `zip_to_mem` / `unzip_from_mem` 保留现有的 tensor 版本,专给 CPU in-place 用。新增按“pool + 编号”的版本给 ScratchPool 路径用。两个版本在 C++ 内部都先换成地址,解压 / 压缩的执行代码只有一份。
- `flow.py` 的 in-place 分支保持不变。12.7 节里的 `_in_place_pools`、`_in_place_pool()` 不加。
- 12.8 节的“CPU in-place 路径”验证这次不做。

**验证用卡**:GPU 卡 4、5(11-02 确认)。
