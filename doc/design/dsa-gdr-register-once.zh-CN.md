# kvshrink 在 B300 上因 DSA 报错崩溃：原因、讨论与修复

记录 2026-11-11 在 b300（8× NVIDIA B300）上发现、定位并修复的 kvshrink 崩溃问题，以及过程中的全部讨论和决定。

代码改动在 https://github.com/intel/intel-accel-for-llm/pull/63。

- 修复代码：`/home/pese/qiuyu/B300/intel-accel-for-llm-dsa-gdr-register-once`，分支 `fix/dsa-gdr-register-once`，基于 main 的 `dfc8769`。
- 测试代码与本文档：`/home/pese/qiuyu/B300/intel-accel-for-llm-b300`，分支 `fix/dsa-gdr-register-once-dev`。b300 上对应目录是 `/home/intel/qiuyu/intel-accel-for-llm-b300`。

文中标注：**实测** = 跑出来的结果；**读源码确认** = 逐行读过代码；**推断** = 没有直接证据的判断。

---

## 1. 名词

| 名词 | 含义 |
|---|---|
| DSA | Intel CPU 上的拷贝硬件。kvshrink 用 DSA 在 GPU 显存和 CPU 内存之间搬 KV cache |
| BAR1 | GPU 在 PCI 上开给 CPU 的地址窗口（lspci 里显示为 `Region 2`）。CPU 和 DSA 通过这个窗口直接读写显存 |
| GDRCopy | NVIDIA 的开源工具，把一段显存映射成 CPU 地址，让 DSA 能直接读写显存。由两部分组成：`libgdrapi`（用户态库，链接进 vLLM 进程）和 `gdrdrv`（Linux 内核驱动）。libgdrapi 自己不干活，把请求通过系统调用转交给 gdrdrv |
| 登记显存 | 本文自定的说法：我们的代码对一段显存先调 `gdr_pin_buffer()`（请 gdrdrv 把这段显存固定住），再调 `gdr_map()`（把这段显存映射成 vLLM 进程里的 CPU 地址）。撤销登记调 `gdr_unpin_buffer()` |
| 登记号（handle） | gdrdrv 给每一次登记分配的号码。libgdrapi 之后用登记号告诉 gdrdrv “我说的是哪一次登记” |
| KV pool | 我们的 C++ 代码（`iaxl/csrc/kv_pool/pool.cpp`）为每一层 KV cache 建的记录：起始地址、block 数、每个 block 的大小、总字节数 `total_bytes`。connector 启动时由 `register_kv_pool_gpu()` 建立 |
| 区域 | vLLM 0.29 在每张 GPU 上把全部 KV cache 放进一整块显存，这块显存从头到尾切成若干个连续区域，见第 4 节 |

---

## 2. 现象（实测）

**触发的测试**：b300 上 Qwen3.8-27B、TP=4（GPU 0/2/4/6）的冒烟测试。

```
HIT_RATES=80 LEN_CONCURRENCY="32000:300 128000:80 256000:40" NUM_PROMPTS_BY_CONCURRENCY="300:300 80:80 40:40" bash test-qwen38-27B.sh
结果目录: bench-results/Qwen3.8-27B_tp4_20261111_034030
```

kvshrink 在输入 32000 的预热阶段（600 个请求、并发 300、命中率 100）崩溃。`kvshrink/serve.log` 里：

| 现象 | 实测 |
|---|---|
| `gdr_map failed` 次数 | 45,076（`rc=16` 30,968 次，`rc=22` 14,108 次） |
| 第一次 `gdr_map failed` | 03:44:49，vLLM 开始服务后约 1 分钟；前 30,870 次全是 `rc=16`，之后 `rc=16` 和 `rc=22` 交替 |
| 第一次 DSA 报错 | `dsa batch failed, ticket=8415 status=0x84`，之后 ticket 8416…8444 的每一批都报同样的错 |
| 崩溃 | 03:51:33，`IAXL check failed: dsa_memcpy_batch_wait(ticket) == 0 (dsa_wait_ticket: DSA batch failed) at iaxl/csrc/kv_xfer/dsa.cpp:107`，VllmWorker-1 退出 |
| `gdr_pin_buffer failed` 次数 | 0 |

说明：
- `gdr_map` 失败本身不致命：代码会改用 CUDA 拷贝。所以从 03:44:49 到崩溃，kvshrink 的搬运已经有一部分悄悄走了 CUDA。
- DSA 状态 0x84：低位 0x04 是“页错误，内核无法修复”，最高位表示错误发生在写入时（按 DSA 规范解读）。
- 日志里 `sub-desc 40..48: status=0xa0, 0xa4 …` 这些递增的值不是真的错误，是打印函数把整块 1024 个槽位都打出来了，其中包括没初始化的槽位。

本机对照（实测）：本机 1030、1026 两个目录的 24 个 kvshrink serve.log 里，`gdr_map failed` 是 0 次。

---

## 3. 原因

### 3.1 我们的代码反复登记同一块显存（读源码确认）

修复前，kvshrink 每次在 GPU 和 CPU 之间搬一层的若干个 block，都会调用一次 `dsa_copy_chunks_batch()`（`iaxl/csrc/kv_xfer/dsa.cpp`）。这个函数每次都做两件事：

1. 算出“从这层 KV 显存的起始地址，到这次要搬的 block 里编号最大的那个 block 的末尾”这段范围，调用 `dsa_gd_default_register()` 登记。
2. `dsa_gd.c` 的 `mapping_for()` 检查已有的登记能否覆盖这段范围。覆盖不了，就新登记一段更长的范围，旧的登记不撤销。

举例（数字是举例）：

| 第几次搬运 | 这次要搬的 block | 修复前的代码做什么 |
|---|---|---|
| 1 | 10–99 | 新登记“起始地址到 block 99 末尾” |
| 2 | 20–80 | 已被覆盖，不登记 |
| 3 | 100–150 | 新登记“起始地址到 block 150 末尾”，第 1 次的登记保留 |
| 4 | 300–400 | 新登记“起始地址到 block 400 末尾”，前两次的登记保留 |

vLLM 从小编号往大编号发 block（`vllm/v1/core/block_pool.py`：空闲队列按 0、1、2… 建立，从队头取，读源码确认）。所以在全部 block 第一次被用完之前，几乎每一步推理都会把每个区域重新登记成更长的范围。

### 3.2 登记号只增不减（读源码确认，b300 上的 `/usr/src/gdrdrv-2.6/gdrdrv.c`）

`gdr_generate_mr_handle()`：新登记号 = 这个进程之前所有登记的总大小（以 4 KB 为单位）；撤销的登记，登记号不回收。每个 vLLM 进程各自打开 gdrdrv，各自有一个计数器。

### 3.3 libgdrapi 只用 32 位保存登记号（读源码确认）

容器里的 libgdrapi 是 2.5（`/usr/lib/x86_64-linux-gnu/libgdrapi.so.2.5`），`gdrapi_internal.h` 里 `struct gdr_memh_t` 的 `handle` 是 `uint32_t`；gdrdrv 那边是 64 位（`gdr_hnd_t` = `__u64`）。一个进程累计登记超过 2^32 × 4 KB = 16 TiB 后，libgdrapi 手里的登记号被截断，是错的。

### 3.4 gdrdrv 找不到登记号时返回最早的登记（读源码确认）

`gdr_mr_from_handle_unlocked()` 遍历登记列表找登记号，找不到时没有返回空，而是返回列表最后一项，也就是这个进程最早的那次登记。

### 3.5 完整过程

1. libgdrapi 用截断后的错误登记号请 gdrdrv 映射，gdrdrv 拿到的是最早的那次登记；这次登记早已映射过，gdrdrv 返回错误 16（EBUSY），即 `gdr_map failed rc=16`。
2. 我们的代码看到映射失败，按原有逻辑调用 `gdr_unpin_buffer()` 撤销刚才这次登记；libgdrapi 交给 gdrdrv 的还是那个错误登记号，gdrdrv 撤销的是**最早那次、仍在使用的登记**，并删除那段显存的 CPU 地址映射。
3. 我们的代码不知道那段映射已被删除，继续把那段地址交给 DSA。DSA 往这段地址写时，地址已不对应任何显存，内核无法修复页错误，DSA 返回 0x84；之后凡是用到这段地址的批次都失败，vLLM 进程崩溃。

### 3.6 为什么本机没有遇到（推断）

累计登记大小 ≈“用完全部 block 所需的步数” × “KV 显存总大小的一半”，大致与 KV 显存大小的平方成正比。b300 每卡 KV 显存约 196.6 GiB，本机每卡显存只有 b300 的约三分之一，累计登记量小得多，到不了 16 TiB。

### 3.7 排除的其他可能

| 假设 | 结论 | 依据 |
|---|---|---|
| GPU 的 BAR 太小（同事问过“是不是没开 64G BAR”） | 不是 | 实测：nvidia-smi 显示 8 张卡的 BAR1 Total 都是 524288 MiB（512 GiB）；lspci 输出里显示出来的卡，`Region 2` 都是 512G，Physical Resizable BAR 当前 512GB（支持的最大值）；`gdr_pin_buffer failed` 0 次；复现测试中同时存在 16 TiB 的登记也没有一次登记失败 |
| PR #44 已经修过 | 不是同一个问题 | PR #44 只改 `iaxl/remote_pool/kvstore_remote.py`，处理的是 RDMA 网卡的内存登记（按 `data_ptr()` 去重），不涉及 GDRCopy |
| 新代码已经整块登记一次（同事的说法） | 公开代码里没有 | 2026-11-11 拉取的 origin main（`08eba1e`）、`v1.5.0`（`075d079`）、`v1.5.0.post`（`d238951`）的 dsa.cpp 都还是按最大 block 登记；6 个未合并 PR 也没有相关改动 |

查 BAR 的命令（b300 上）：

```bash
nvidia-smi -q -d MEMORY | grep -E "^GPU|BAR1 Memory Usage" -A1 | grep -E "^GPU|Total"
lspci -vv -d 10de: | grep -E "^[0-9a-f]{2}:|Region 2"
```

---

## 4. vLLM 0.29 的 KV 显存布局（Qwen3.8-27B、TP=4、b300）

- vLLM 0.29 在每张 GPU 上只分配**一整块显存**放全部 KV cache（`vllm/v1/worker/utils.py` 的 `allocate_kv_cache()`，`torch.zeros(buf_size, ...)`），b300 上约 196.6 GiB。
- 这一整块显存切成 **16 个连续区域**，每个区域 12.29 GiB；每个区域内部按 block 编号依次排列，一个 block 占 784 KB。

```
每张 GPU 上放 KV cache 的一整块显存（约 196.6 GiB）：
| 区域 0 (12.29 GiB) | 区域 1 (12.29 GiB) | ... | 区域 15 (12.29 GiB) |

区域 0 的内部（区域 1~15 结构相同）：
| block 0 (784 KB) | block 1 (784 KB) | ... | block 16434 (784 KB) |
```

- vLLM 的 KV cache group = 同一种注意力类型、共用一张 block 表的一组层。Qwen3.8-27B 共 64 层：16 层普通注意力为 1 个 group，48 层线性注意力分成 3 个 group，每个 group 16 层（读 `kv_cache_utils.py` 推断）。4 个 group 共用同一整块显存（注释 “Groups alias from byte 0”），每个 group 里排第 r 的那一层都用区域 r，所以区域 r 由 4 层共用。
- 实测：kvshrink 日志 `Registered 64 KV cache layers (48 mamba) with shape [16435, 401408]`（每层 16,435 个 block，每个 block 401,408 个 bf16 = 784 KB），`contiguous block size = 784.00 KB`。16 × 16,435 × 784 KB = 196.6 GiB，与 probe 实测的可用 KV 显存一致。
- block size 784 由 TP 决定（TP=4 为 784，TP=8 为 400），与 GPU 型号无关。

我们的 KV pool 记录的是“这一层所在区域的起始地址和区域大小 12.29 GiB”；共用同一区域的 4 层，起始地址和大小相同。

---

## 5. 复现测试（实测）

**目的**：不跑 vLLM，单独验证第 3 节的整条链。

**程序**：`tests/bench_dev/gdr_handle_overflow_check.c`（构建并运行：`tests/bench_dev/gdr_handle_overflow_check.sh`），直接编进我们的 `dsa_gd.c`、`dsa_memcpy.c`，调用 kvshrink 用的同一个函数 `dsa_gd_default_register()`。b300 的 GPU 1、DSA 队列 wq0.0，在 iaxl 容器里运行。

**步骤**：
1. 登记一块 2 MiB 的显存（“被害块”，最早的登记），写入已知值；用 DSA 往被害块写一次，确认 DSA 正常。
2. 在一块 128 GiB 的显存上按 kvshrink 的方式反复登记，每次比上次长 128 MiB；每次登记后用 CPU 通过映射地址读被害块。
3. 第一次出错时，再用 DSA 往被害块写一次，然后结束。

**跑之前的预测**：第 513 次登记时登记号超过 32 位。

**结果**（`tests/bench_dev/results/gdr-handle-overflow-20261111_054618/`，代码 md5 在 `code.md5`）：

| 第几次登记 | 本次登记长度 GiB | 累计登记 TiB | 累计 ÷ 2^32 页 | 登记返回 | 读被害块 |
|---|---|---|---|---|---|
| 448 | 56.00 | 12.28 | 0.767 | 0 | 正常 |
| 512 | 64.00 | 16.03 | 1.002 | 0 | 正常 |
| 513 | 64.12 | 16.09 | 1.006 | -1（`gdr_map failed rc=16`） | SIGBUS |

列说明：“累计 ÷ 2^32 页”超过 1 表示登记号越过 32 位上限；“登记返回” 0 为成功；SIGBUS 表示被害块的映射已被删除，CPU 读它时进程收到总线错误。表中只列了每 64 次的打印和出错的那一行。

之后用 DSA 写被害块：`dsa op failed, status=0x84`，与 kvshrink 崩溃时一致。

**结论**：出错正好在预测的第 513 次，`rc=16` → SIGBUS → `0x84` 的顺序与 kvshrink 日志一致。**没直接证明的部分**：kvshrink 那次运行没记录累计登记量，“确实累计到 16 TiB”是根据错误特征一致推断的。测试的一个缺陷：最后 “value read back by cudaMemcpy matches” 那句没有意义（同一位置的值是前面那次成功的 DSA 写入留下的）。

---

## 6. 修复方案的讨论与决定

### 6.1 核心思想（与同事讨论后确定）

参考 RDMA 的做法：**登记 KV pool 时把整个区域登记一次，搬运时只算地址偏移**。

### 6.2 RDMA 是怎么做的（读源码确认，`iaxl/csrc/kv_xfer/rdma.cpp`）

RDMA 代码跑在 remote_pool 的 daemon 进程里，用 NIXL 读写 vLLM 进程所在 GPU 的显存。

| 时刻 | RDMA 做什么 |
|---|---|
| 初始化（`rdma_init()`） | 调用 `set_pool_hooks({on_register, on_unregister})`，把 RDMA 的两个函数地址填进 pool.cpp 的 hook |
| 登记 KV pool（`on_register()`） | 按起始地址维护表 `regs`，起始地址相同的 KV pool 共用一项并记引用数；某个起始地址第一次出现时：远端 GPU 的 KV pool 建一张描述符列表（block b 的第 o 段放在第 `o × num_blocks + b` 项），本地内存的 KV pool 用 `registerMem` 整块登记一次 |
| 撤销 KV pool（`on_unregister()`） | 引用数减 1，减到 0 才释放 |
| 创建 Context（`rdma_context_create()`） | 只从表里取出已建好的描述符列表 |
| 每次搬运（`copy_chunks_batch()`） | 只算“用表里第几项”：`remote_idx = o × num_blocks + chunk_index`，`local_idx = lb × od + o`，然后发一个 NIXL 请求 |

DSA 搬运时算地址的代码（GPU 侧 = BAR 起始地址 + block 编号 × `chunk_stride` + 段号 × `outer_block_size`）修复前已经有了，DSA 和 RDMA 的区别只在“什么时候登记、登记多大”。

### 6.3 什么时候登记：登记 KV pool 时，不在创建 Context 时

- 登记 KV pool 只在 connector 启动时发生，每层一次（b300 实测 64 次）。
- 创建 Context：每次存或取 KV，对每一层都创建一个 Context（`iaxl/kvflow/flow.py` 的 `put()`、`get()`），次数极多，而且在请求处理过程中。RDMA 也没有在创建 Context 时登记。

### 6.4 hook 是什么

hook 就是 pool.cpp 里保存的函数指针。`PoolHooks`（`kv_pool.h`）有两个函数指针 `on_register`、`on_unregister`；pool.cpp 登记一个 KV pool 后、撤销一个 KV pool 前，如果函数指针不为空就调用。hook 让各传输方式（RDMA、DSA）在 KV pool 登记或撤销时得到通知，而通用的 pool.cpp 不用写任何传输方式相关的代码。

### 6.5 DSA 登记怎样接到“登记 KV pool”上：三种做法

| 做法 | 内容 | 问题 |
|---|---|---|
| (a) | DSA 也调用 `set_pool_hooks()` 填自己的函数 | 修复前登记表只有一组 hook，`set_pool_hooks()` 是直接赋值。同一个进程里同时开启 RDMA 和 DSA 时（即同时调用 `rdma_init()` 并用 hook 登记 DSA），后设置的登记函数会替换先设置的，pool.cpp 只会调用 RDMA 的登记函数和 DSA 的登记函数中的一个。现在 `rdma_init()` 只在 daemon 进程调用、vLLM 进程用 Python 版 RDMA，所以现在不冲突，但隐患太大，否决 |
| (b) | `register_kv_pool_gpu()` 和 `unregister_kv_pool()` 里用 `#if defined(CUDA_SUPPORT) && defined(DSA_SUPPORT)` 直接调用 DSA 的函数 | 不会被替换，但各设备通用的 pool.cpp 里多了 DSA 专用代码；RDMA 走 hook、DSA 走直接调用，同一件事两套机制 |
| (c) | 把 hook 改成可以挂多组：`set_pool_hooks()` 改为 `add_pool_hooks()`，登记表里的 hook 改成列表 | **采用**。`PoolHooks` 的注释本来就写明是 “Transport backend callbacks”，pool.cpp 的设计就是“各传输方式通过 hook 接入”；修复前的缺陷只在于只能存一组 |

(c) 下 pool.cpp 的循环不知道、也不需要知道有几组 hook，只按加入顺序逐组调用。每个登记函数自己看 `p.device` 决定要不要处理：

| `p.device` | DSA 的 `dsa_on_register()` | RDMA 的 `on_register()` |
|---|---|---|
| `GPU` | `IAXL_DSA_GD_ENABLE=1` 时用 GDRCopy 整块登记，否则跳过 | 用 NIXL `registerMem` 整块登记 |
| `CPU` | 跳过 | 用 NIXL `registerMem` 整块登记 |
| `Remote` | 跳过 | 建 NIXL 描述符列表 |

vLLM 进程（CUDA 构建）的列表里只有 DSA 一组（导入 torch_ext 时加入）；daemon 进程里先是 DSA 一组，后是 RDMA 一组（`rdma_init()` 时加入）。两个登记函数互不调用、互不依赖。

(c) 需要处理好的三点：DSA 的 hook 必须在登记任何 KV pool 之前挂上（在 torch_ext 模块加载时挂）；DSA 的 hook 只处理 GPU 上的 KV pool；pool.cpp 调用 hook 时持有登记表的锁，所以 DSA 的 hook 里不能再调用 pool.cpp 的函数（持锁时间较长只发生在启动和撤销时，可以接受）。

### 6.6 其他决定

| 问题 | 决定 | 理由 |
|---|---|---|
| `IAXL_DSA_GD_RESET_ON_DESTROY`（每个 Context 销毁时撤销全部登记，给 BAR1 放不下全部 KV 的显卡用） | **删除**这个开关 | 与“启动时登记一次”冲突：开着时第一个 Context 销毁就会撤销启动时的登记，之后 DSA 用的地址全部失效。b300（512 GiB）和本机（128 GiB）的 BAR1 都大于整卡显存，用不到 |
| BAR1 放不下全部 KV 时怎么办 | `IAXL_DSA_GD_ENABLE=1` 时启动登记失败就**报错退出**，提示可以设 `IAXL_DSA_GD_ENABLE=0` 改用 CUDA 拷贝 | 以前登记失败会悄悄改用 CUDA。注意：gdrdrv 没加载等其他原因导致登记失败，现在也会报错退出。b300 和本机都无法真实触发这条路径 |
| 撤销 KV pool 时要不要释放登记 | **释放**（与 RDMA 一致，按引用数，最后一个使用者撤销时释放） | 释放前先等全部在途 DSA 批次完成，并持有提交锁，防止 DSA 读写已释放的显存 |
| 累计登记量上限检查（快到 16 TiB 时报错） | **不做** | 整块登记一次后累计登记量约 196.6 GiB，正常运行到不了上限；这个检查只防将来有人改回反复登记 |
| `dsa_gd_reset()` / `dsa_gd_default_reset()` | 保留，没有删除 | 修复后没有调用方，但属于 dsa_gd.h 对外公开的函数 |

讨论过但没采用的修法：
- 只改 dsa.cpp 一处，把登记范围从“到这次最大 block”改成“整个区域”：改动最小，但仍在搬运时登记、每个区域第一次搬运多一次登记耗时；改为按 RDMA 的思路在登记 KV pool 时登记。
- 在 dsa_gd.c 里用 `cuMemGetAddressRange` 查整次显存分配再登记：可能多登记不相关的显存，PyTorch 开启 expandable_segments 时可能登记失败。
- 在 connector 启动时由 Python 按 `untyped_storage()` 整块登记：依赖“全部 KV 在同一块显存”，旧版本 vLLM 不一定如此。

---

## 7. 最终改动

修复分支 `fix/dsa-gdr-register-once`（目录 `/home/pese/qiuyu/B300/intel-accel-for-llm-dsa-gdr-register-once`）：

| commit | 内容 |
|---|---|
| `f59ee88` fix(kv_xfer): map each GPU KV pool through GDRCopy once at registration | 修复本身，12 个文件，新增 136 行、删除 51 行；已 push 到个人 fork |
| `d3f4041` update | 把新增的多行注释缩成一行，3 个文件，只改注释 |

| 文件 | 改动 |
|---|---|
| `iaxl/csrc/include/kv_pool.h`、`iaxl/csrc/kv_pool/pool.cpp` | `set_pool_hooks()` 改为 `add_pool_hooks()`，hook 改成列表，登记/撤销 KV pool 时逐组调用 |
| `iaxl/csrc/kv_xfer/rdma.cpp` | `rdma_init()` 改为调用 `add_pool_hooks()` |
| `iaxl/csrc/include/dsa_gd.h`、`iaxl/csrc/dsa_gd/dsa_gd.c` | 新增 `dsa_gd_unregister()`、`dsa_gd_default_unregister()`：释放一个登记 |
| `iaxl/csrc/kv_xfer/dsa.cpp` | 新增 `dsa_on_register()` / `dsa_on_unregister()`（按起始地址的表 `dsa_regions`，记 BAR 地址和引用数）、`dsa_install_pool_hooks()`、`dsa_gpu_bar_base()`；`dsa_copy_chunks_batch()` 不再登记，直接用传入的 BAR 地址；删除 `dsa_context_reset()` |
| `iaxl/csrc/include/kv_xfer.h` | 声明随 dsa.cpp 修改 |
| `iaxl/csrc/kv_xfer/cuda.cpp` | 创建 Context 时查好 BAR 地址存进 Context，搬运时直接用；删除 `IAXL_DSA_GD_RESET_ON_DESTROY` 的处理 |
| `iaxl/csrc/torch_ext/torch_ext.cpp` | Python 导入 torch_ext 模块时挂上 DSA 的 hook |
| `iaxl/csrc/include/env.h`、`iaxl/csrc/common/envs.c`、`setvars.sh` | 删除 `IAXL_DSA_GD_RESET_ON_DESTROY` |

修复后每个 vLLM 进程登记 16 次（每个区域一次），累计 196.6 GiB，是 32 位登记号上限（16 TiB）的 1.2%。Qwen3-32B 这类非混合模型每层一个区域，每个进程登记 64 次，累计也等于全部 KV 显存。

---

## 8. 验证

### 8.1 编译检查

修复后的 `iaxl/csrc` 在 b300 的 iaxl 容器里用正式编译的宏（`CUDA_SUPPORT=1 DSA_SUPPORT=1` 等）逐个做 `-fsyntax-only`：dsa.cpp、cuda.cpp、pool.cpp、rdma.cpp、torch_ext.cpp、dsa_gd.c、envs.c 全部通过。之后在 b300 测试目录完整重新编译，`pip install -e .` 成功（`tests/bench_dev/logs/build-dsa-fix-20261111_080403.log`，最后一行 `BUILD_EXIT=0`）。

### 8.2 冒烟测试（实测）

```
CONFIGS=kvshrink HIT_RATES=80 LEN_CONCURRENCY="32000:300" NUM_PROMPTS_BY_CONCURRENCY="300:300" bash test-qwen38-27B.sh
结果目录: bench-results/Qwen3.8-27B_tp4_20261111_080256（08:02:56 开始，08:09:28 结束）
```

请求分三段依次进行：预热 600 个请求（并发 300、命中率 100，上次崩溃的阶段）→ 热身 5 个 → 正式测量 300 个请求（输入 32000、输出 128、并发 300、命中率 80，只有一轮）。测试代码是修复代码加临时打印 `TEMP-DEBUG`（每登记一个区域打印一行）。

| 检查项 | 预期 | 实测 |
|---|---|---|
| `gdr_map failed` / `dsa batch failed` / `IAXL check failed` | 0 | 都是 0；summary.csv 的 `dsa_batch_failed_count` 也是 0 |
| `using Intel DSA backend` | 大于 0 | 4（4 个 vLLM 进程各一次） |
| 每个 vLLM 进程的登记次数 / 累计大小 | 16 次 / 约 196.6 GiB | 4 个进程各 16 次，每个 196.61 GiB，每次 13,194,280,960 字节 |
| 进程是否崩溃 | 不崩溃 | 不崩溃 |

每个请求从 kvshrink 缓存取回的 token 数（connector 日志 `externally-cached tokens`，905 行 = 600 + 5 + 300，按日志顺序分段）：

| 阶段 | 请求数 | 命中 token 数 → 请求数 |
|---|---|---|
| 预热（命中率 100） | 600 | 31360 → 593；15680 → 6；0 → 1 |
| 热身 | 5 | 31360 → 2；15680 → 2；0 → 1 |
| 正式测量（命中率 80） | 300 | 25088 → 218；15680 → 81；31360 → 1 |

31360 = 40 个 block（输入 32000 能对齐的最大值）；25088 = 32 个 block（命中率 80 时共享前缀 25600 能对齐的最大值）；15680 = 20 个 block。

正式测量那一轮（300 个请求，只有一波，不是稳定状态性能）：耗时 62.5 秒；总吞吐 154,291 token/秒；TTFT 平均 / 中位 / p95 = 35,734 / 38,540 / 57,621 毫秒；TPOT 平均 / 中位 / p95 = 187.3 / 178.0 / 342.9 毫秒；被挤出 0 次；每个 rank 的 DDR 缓存 0.41 → 34.88 GB（容量 250 GB）；压缩比 1.276。

### 8.3 gsm8k 两轮准确率（实测）

```
脚本: tests/bench_dev/gsm8k_two_passes_b300.sh（在 iaxl 容器里运行）
结果目录: tests/bench_dev/results/gsm8k-20261111_081551_a82a7c51（代码 md5 a82a7c51）
```

做法照搬 1026 的 `tests/vllm_block_hashes_dev/gsm8k_two_passes.sh`：1 个 kvshrink vLLM（设置同冒烟测试，vLLM 自带的 GPU 前缀缓存关闭），evalscope 1.14.1 跑 gsm8k 前 200 题，50 题并发，temperature 0，max_tokens 2048，关闭 thinking。默认 4 个示例题时题目只有 652–762 token，填不满一个 784 token 的 block，所以拼到 26 个示例题，每题约 5500 token（约 7 个 block）。第 1 轮计算并存入 DDR（26 个示例题是所有题共享的开头，第 1 轮靠后的题已能命中）；第 2 轮发同样的 200 题，从 DDR 取回 KV。

| 轮次 | evalscope 准确率 | connector 日志行数 |
|---|---|---|
| 第 1 轮 | 0.98 | 200 |
| 第 2 轮 | 0.975 | 200 |

serve.log 里 `gdr_map failed`、`dsa batch failed`、`IAXL check failed` 都是 0 次。逐题对比脚本 `gsm8k_compare.py`（从 1026 复制）读 evalscope 1.14.1 的结果格式时报错（`KeyError: 'acc'`，新版本的分数字段名是 `accuracy`），所以每题命中多少 token、哪些题对错翻转、输出是否逐字节相同，**还没有统计**。

---

## 9. 未完成与后续

1. 推送 `d3f4041` 到个人 fork（普通 push，不需要 `--force`）。
2. 删除测试目录 `iaxl/csrc/kv_xfer/dsa.cpp` 里的 `TEMP-DEBUG` 临时打印，同步 `d3f4041` 的注释改动，然后重新编译。测试目录的 `iaxl/csrc` 现在 = 修复（缩注释前的版本）+ 临时打印。
3. gsm8k 逐题对比没有统计（第 8.3 节）。
4. 冒烟测试正式测量里 81 个请求只命中 20 个 block（最多 32 个），原因没分析。推断可能与混合模型有关：命中某个位置需要那个位置存有线性注意力层的状态。没有核实。
5. 冒烟测试只跑了输入 32000，没跑 128000、256000。
6. 完整测试还卡在另一个独立问题上：LMCache 在 b300 上零命中（LMCache 把 KV 张量里长度为 784 的 block size 维当成了头数，`Number of heads (784) exceeds max threads per block in y-dim (32)`），与本修复无关。
7. Qwen3-32B 在 b300 上也会遇到本问题（推断：累计登记约 37 TiB，按每步 8192 token 估算，未核实 vLLM 实际参数）。修复后不受影响。

## 10. 当时 git 操作上的一个错误

修复 commit `f59ee88` 已经 push 后，我用 `git commit --amend` 把注释改动并进了 `f59ee88`（变成本地 `c17dcd1`），与已 push 的分支分叉。随后用 `git reset --soft f59ee88` 退回已 push 的 commit，把注释改动另外提交为 `d3f4041 update`。`c17dcd1` 没有 push 过。规则：不改写 git 历史，后续改动一律提交新 commit。
