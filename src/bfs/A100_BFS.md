# BFS CUDA 版本（A100 / SM80）说明

本文档位于 `gardenia/src/bfs/`，只聚焦本目录下的 CUDA 实现（`*.cu`），并以 **NVIDIA A100（SM80）** 为目标说明：

- 这些 CUDA 文件分别代表什么（从名字就能看懂的“暗号” + 通俗例子）。
- 目前哪些 CUDA 版本**无法直接编译/链接成可运行的 BFS 可执行文件**，原因是什么，能否修改。

> 背景：`src/bfs/main.cc` 固定调用 `BFSSolver(Graph&, ...)`（见 `bfs.h` 声明），因此能否“编译成可执行文件”不只取决于 CUDA 语法是否能过，还取决于 **Solver 接口是否匹配**。

---

## 0) A100 快速开始（只编译可运行的 CUDA 版本）

### 0.1 编译（A100 / sm_80）

`src/common.mk` 里已经把 GPU 架构设成 `sm_80`（A100）：`-gencode arch=compute_80,code=sm_80`。

在 BFS 目录编译：

```bash
cd /root/gpgpu-sim/accel-sim-framework/gpu-app-collection/gardenia/src/bfs
make -j bfs_linear_base bfs_linear_lb
```

可执行文件会被移动到 `gardenia/bin/`（由 `src/common.mk` 的 `BIN=../../bin` 控制）。

### 0.2 运行（示例图）

`main.cc` 的参数格式（`<filetype> <graph-prefix> [symmetrize] [reverse] [source]`）：

```bash
cd /root/gpgpu-sim/accel-sim-framework/gpu-app-collection/gardenia/bin
./bfs_linear_base mtx ../datasets/4 0 0 0
./bfs_linear_lb   mtx ../datasets/4 0 0 0
```

---

## 1) 文件名怎么读：把名字拆成“前缀 + 后缀”

这一目录的 CUDA 文件名基本可以按下面方式理解：

- `linear_*`：**数据驱动（data-driven）**，每一轮只处理“当前 frontier 队列”里的顶点（像“只处理待办清单”，而不是每次都翻整本通讯录）。
- `topo_*`：**拓扑驱动（topology-driven）**，每一轮通常会扫一遍全图顶点（用 `front[]/visited[]` 标记哪些是 frontier）。
- `hybrid_*`：**混合/方向优化（direction-optimizing）**，在 top-down（push）与 bottom-up（pull）之间切换（Beamer BFS 思路）。

常见后缀含义：

- `*_base`：最直观的映射方式（常见是 thread-per-vertex 或简单扫描）。
- `*_vector`：更“向量化”的映射（常见是 **warp-per-vertex**：一个 warp 协作处理一个顶点的邻居）。
- `*_lb`：**load-balance**（负载均衡），用 warp/CTA 协作或 CUB BlockScan 把“高出度顶点”的边工作拆给更多线程。
- `*_pb`：**push-based**（偏 push 的写法），先把“下一层会被访问的点”标记出来（如 `visited[dst]=true`），再由 `update` 把它们变成真正 frontier/深度。
- `*_bu` / `bottom_up`：**bottom-up（pull）**，从“未访问点”出发，检查是否有邻居在 frontier 里（后期 frontier 很大时常更划算）。
- `*_tex`：使用 **纹理缓存（texture）** 读取 CSR（偏旧式 CUDA API，属于 legacy 用法）。
- `fusion`：把多轮 BFS/多步融合到更少 kernel（甚至单 kernel）里，通常配合全局同步（global barrier）。
- `atomic_free`：尝试减少/避免原子操作的版本（但本仓库里该文件当前存在明显实现缺陷，见下文）。
- `merrill`：Merrill BFS（历史实现，依赖 back40/b40c 代码库）。

---

## 1.5) 通俗理解：八卦传播的故事

为了直观理解这些不同 BFS 实现的区别，我们可以想象一个**“八卦传播”**的场景：从 0号（八卦源头）开始传播消息。

**场景设定**：
- **0号**：源头。
- **1号（网红）**：认识 1000 个人。
- **2号（普通人）**：只认识 1 个人。

| 算法 | 对应文件 | 形象比喻 | 核心逻辑 | 优缺点 |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Base** | `linear_base.cu` | **邮递员派件**<br>(一人送一家) | **Top-Down (Push)**<br>GPU 派一个线程处理 1号，派另一个线程处理 2号。<br>处理 1号的线程要发 1000 封信（累死），处理 2号的线程发 1 封信（闲死）。 | **缺点**：负载严重不均，网红节点会卡死整个 GPU。<br>**优点**：逻辑简单，适合图很稀疏且均匀时。 |
| **Linear LB** | `linear_lb.cu` | **搬家公司**<br>(多人帮一家) | **Load Balanced**<br>发现 1号是网红，于是派**一组线程**（甚至整个 Warp）一起帮 1号 发信。<br>2号还是由单人处理。 | **优点**：解决了“网红卡死”问题。<br>**缺点**：调度逻辑复杂，开销稍大。 |
| **Topo Base** | `topo_base.cu` | **村口大喇叭**<br>(全员点名) | **Topology-Driven**<br>不维护“待办清单”。每一轮直接拿大喇叭问全图所有人：“你有八卦吗？有就传给邻居！” | **缺点**：如果全村 100万人 只有 2人 知道八卦，依然要问遍 100万人。<br>**优点**：不用维护队列，适合极度稠密的图。 |
| **Bottom Up** | `bottom_up.cu` | **吃瓜群众**<br>(主动打听) | **Bottom-Up (Pull)**<br>让**不知道八卦的人**去问邻居：“你知道吗？”<br>一旦问到一个邻居知道，自己就标记为“已吃瓜”，并**立刻停止**询问其他邻居。 | **优点**：八卦爆发期（Frontier 很大）效率极高，利用“提前退出”机制。<br>**缺点**：刚开始没人知道时，大家瞎打听效率低。 |
| **Hybrid** | `hybrid_base.cu` | **精明经理**<br>(灵活切换) | **Direction-Optimizing**<br>刚开始人少用 **邮递员模式**；<br>中间爆发期切成 **吃瓜群众模式**；<br>收尾时切回 **邮递员模式**。 | **优点**：集百家之长，工业界标准做法。<br>**缺点**：实现最复杂。 |

---

## 2) CUDA 文件清单：每个文件“代表什么”+ 适用场景例子

下面按 `linear/topo/hybrid/其它` 分类说明。

### 2.1 `linear_*`：数据驱动（frontier 队列）

**核心直觉**：frontier 很小时，`linear_*` 会更省，因为每轮只处理 “frontier 列表里的点”，不会扫全图。

- `linear_base.cu`
  - 含义：最基础的 **top-down BFS**（frontier 队列），近似 “一个线程处理一个 frontier 顶点并遍历它的邻居”。
  - 例子：从一个源点开始，前几轮 frontier 很小（1、几十、几百），这时 `linear_base` 像“只做待办清单”一样省工作量。
  - 当前状态：**可直接编译成可执行文件** `bfs_linear_base`（接口是 `BFSSolver(Graph&,...)`）。

- `linear_lb.cu`
  - 含义：`linear_base` 的 **负载均衡版**（`lb = load-balance`），更擅长处理“某些点出度极大”的情况（用 warp/CTA + CUB BlockScan 分摊边遍历）。
  - 例子：社交图里有“超级节点”（百万邻居）。`linear_base` 可能让一个线程独自扫完百万邻居；`linear_lb` 会把这部分边工作拆给更多线程协作。
  - 当前状态：**可直接编译成可执行文件** `bfs_linear_lb`（接口是 `BFSSolver(Graph&,...)`）。

- `linear_vector.cu`
  - 含义：数据驱动的 **warp-per-vertex** 版本（一个 warp 协作处理一个顶点的邻居）。
  - 例子：当 frontier 顶点的邻居规模在几十~几百时，一个 warp 协作通常比单线程更高效。
  - 当前状态：**不支持直接生成可执行文件（需修改）**，原因见第 3 节（Solver 接口不匹配：它实现的是旧签名 `BFSSolver(int m, int nnz, ...)`）。

- `linear_tex.cu`
  - 含义：用 **texture reference + tex1Dfetch** 读取 CSR（`_tex`）。
  - 例子：把 CSR 数组当作只读纹理读取，期望利用纹理缓存提升随机访问的带宽/延迟。
  - 当前状态：
    - **不支持直接生成可执行文件（需修改）**：同样是旧 `BFSSolver(int m, int nnz, ...)` 签名；
    - 代码使用 legacy texture API（`texture<>` + `cudaBindTexture`），在新 CUDA 上通常还能编译，但不建议作为长期方案。

- `linear_pb.cu`
  - 含义：尝试做 push-based / 分阶段（标记 visited → update 生成下一轮 frontier）的数据驱动版本（`pb`）。
  - 例子：先用 frontier 去“标记哪些点会被访问”（写 `visited[dst]=1`），再用一个 `update` kernel 把这些点转成下一层队列，从而减少一些原子冲突。
  - 当前状态：**不支持直接生成可执行文件（需修改）**（旧签名）；而且该文件内还有明显的调试痕迹（例如循环中途提前 `return`），即使改成可链接也不建议直接当作“正确实现”使用。

### 2.2 `topo_*`：拓扑驱动（扫描全图）

**核心直觉**：frontier 很大时（尤其是 BFS 后期），扫描全图顶点可能并不亏；实现也更直接。

- `topo_base.cu`
  - 含义：最直观的 topology-driven BFS：每轮 kernel 扫一遍顶点 `src`，若 `front[src]` 为真就扩展它的邻居。
  - 例子：当 frontier 接近全图的一大部分时（比如 BFS 中后期），扫描全顶点的代价接近“反正都要扫很多”。
  - 当前状态：**不支持直接生成可执行文件（需修改）**（旧签名）。

- `topo_vector.cu`
  - 含义：拓扑驱动的 warp-per-vertex（或 warp-per-src 扫描）思路，配合 `visited/expanded` 做状态控制。
  - 例子：当图的出度差异大，warp 协作能减少分歧（divergence）。
  - 当前状态：**不支持直接生成可执行文件（需修改）**（旧签名）。

- `topo_lb.cu`
  - 含义：拓扑驱动 + 负载均衡（`lb`），把高出度顶点的边遍历拆分到更多线程执行（使用 CUB BlockScan 等技巧）。
  - 例子：同样是“超级节点”场景，但这里 frontier 是用 `front[]` 标记的，不是队列。
  - 当前状态：**不支持直接生成可执行文件（需修改）**（旧签名）。

- `topo_pb.cu`
  - 含义：拓扑驱动的 push-based（`pb`）：先从 `front[]` 把邻居标记到 `visited[]`，再由 `update` 把“新发现”写入 depth 并生成下一轮 `front[]`。
  - 例子：把“访问标记”和“frontier 更新”拆成两步，逻辑更像 CPU BFS：先扩展，再收集下一层。
  - 当前状态：**不支持直接生成可执行文件（需修改）**（旧签名）。

### 2.3 `bottom_up.cu` 与 `hybrid_*`：pull / 方向优化

这些版本通常需要 **入边（reverse graph / in-neighbor）**，因为 bottom-up 的检查方式是：

> 对于每个“未访问”的点 `dst`，看看它是否存在某个入邻居 `src` 在当前 frontier 里。

- `bottom_up.cu`
  - 含义：纯 bottom-up（pull）BFS。
  - 例子：BFS 后期 frontier 很大时，top-down 会产生大量重复尝试（很多边都指向已访问点）；bottom-up 反而能更快收敛。
  - 当前状态：**不支持直接生成可执行文件（需修改）**（旧签名；并且需要 reverse graph 输入）。

- `hybrid_base.cu`
  - 含义：方向优化 BFS（Beamer DO-BFS）：根据 frontier/边数估计（`alpha/beta`）在 top-down 与 bottom-up 之间切换。
  - 例子：社交图这类直径小、后期 frontier 很大的图，hybrid 往往比纯 top-down 更快。
  - 当前状态：**不支持直接生成可执行文件（需修改）**（旧签名；需要 out_graph + in_graph + degree 等信息）。

- `hybrid_vector.cu` / `hybrid_lb.cu` / `hybrid_tile.cu`
  - 含义：`hybrid` 的不同并行映射/负载均衡版本（`vector/lb/tile`）。
  - 例子：同一套方向优化思想，不同 kernel 组织方式影响吞吐与分歧。
  - 当前状态：**不支持直接生成可执行文件（需修改）**（旧签名；同样需要 reverse graph/degree 等）。

### 2.4 其它：`atomic_free` / `fusion` / `merrill`

- `atomic_free.cu`
  - 含义：尝试“少用原子”的 BFS（靠 `visited/expanded` 或 `dist>depth` 的方式减少 atomicCAS）。
  - 当前状态：该文件当前实现里存在明显问题：`d_num_frontier` 未分配就传入 kernel，随后还 `cudaFree(d_num_frontier)`；这类问题会导致运行时错误。即使修接口能链接，也建议先修复逻辑后再用。

- `fusion.cu`
  - 含义：kernel fusion + 全局同步（global barrier），把多轮 BFS 融进更少 kernel（降低 kernel launch 频率）。
  - 例子：frontier 每一轮都很小但轮数多时，fusion 思路可能减少 launch 开销（但实现复杂、对并发 resident block 数有要求）。
  - 当前状态：**不支持直接生成可执行文件（需修改）**（旧签名）。

- `merrill.cu`
  - 含义：Merrill BFS（历史实现），依赖 back40computing / b40c 相关头文件与库。
  - 当前状态：本仓库默认缺少 `back40computing-read-only` 目录与 `b40c_*` 头文件，且 Solver 也是旧签名；因此 **无法直接编译/链接**。要用它需要补齐依赖并做接口适配。

---

## 3) 目前“可直接编译运行”的 CUDA BFS：只有哪两个？为什么？

**当前能直接编译成可执行文件并运行的 CUDA 版本：**

- `linear_base.cu` → `bfs_linear_base`
- `linear_lb.cu` → `bfs_linear_lb`

原因很简单：`src/bfs/main.cc` 调用的是：

```cpp
BFSSolver(g, source, &distances[0]);
```

而这两个文件实现了匹配的函数签名：`BFSSolver(Graph &g, int source, DistT *dist)`。

**其它 `.cu` 大多实现的是旧签名**（`BFSSolver(int m, int nnz, ...)` 或需要 in/out CSR 等更多参数），所以即使 `Makefile` 里写了诸如 `bfs_topo_base`、`bfs_hybrid_base` 等链接目标，直接 `make bfs_topo_base` 往往会在链接阶段失败（缺少 `BFSSolver(Graph&,...)` 符号）。

---

## 4) 哪些 CUDA 版本目前“不支持编译成可运行版本”：原因 + 是否能改

这里的“不支持”指：**不能直接通过 `make bfs_xxx` 产出可运行可执行文件**。

### 4.1 主因：`BFSSolver` 接口不匹配（可改，工作量中等）

受影响文件（典型）：`topo_*`、`bottom_up.cu`、`hybrid_*`、`linear_vector.cu`、`linear_tex.cu`、`fusion.cu`、`linear_pb.cu` 等。

为什么：这些文件实现的 `BFSSolver(...)` 参数列表与 `main.cc` 期待的不一致。

能否修改：可以，常见两条路线：

1) **加一层适配 wrapper（推荐）**：在对应 `.cu` 里增加 `BFSSolver(Graph&,...)`，从 `Graph` 中取出 CSR 指针（以及需要的话取 reverse CSR / degree），再调用旧实现或改写 kernel 入参。
2) **把旧实现整体改成 Graph 版**：把 kernel/host 端的 CSR 类型统一成 `Graph` 使用的 `rowptr=uint64_t*`、`colidx=VertexId*`，避免 int 溢出与重复拷贝（改动更大，但长期更干净）。

> 如果要跑 bottom-up/hybrid：还需要在运行时传 `reverse=1` 让 `Graph` 构建 reverse graph（main 参数第 4 个）。

### 4.2 依赖缺失：`merrill.cu`（可改，但需要引入第三方代码）

为什么：依赖 `b40c_test_util.h` 与 `b40c/...` 头文件，以及 `../../back40computing-read-only` 目录（本仓库默认没有）。

能否修改：

- 需要先把 back40computing/b40c 代码引入到仓库（或改成你自己的 BFS 实现），然后再做 `BFSSolver(Graph&,...)` 适配。

### 4.3 文件自身实现问题：`atomic_free.cu`、`linear_pb.cu`（先修 bug 再谈适配）

- `atomic_free.cu`：存在未分配指针就传 kernel / free 的问题（运行必炸）。
- `linear_pb.cu`：包含明显调试/未完成逻辑（例如迭代中途提前退出），不建议直接当作“正确可用版本”。

这些都能修，但建议先把它们当作“研究代码/草稿”看待。

### 4.4 legacy CUDA API：`linear_tex.cu`（可改，建议换 texture object）

为什么：使用 `texture<>` + `cudaBindTexture` + `tex1Dfetch`（legacy texture reference API）。

在很多新 CUDA 版本上仍可编译，但更推荐迁移到 **texture object**（更现代、更可控）。

---

## 5) 在 A100 上看 SASS ↔ 源码对应：怎么编译/反汇编

本目录 `Makefile` 已经提供了 `cubin` 与 `nvdisasm` 的辅助规则（会利用 `src/common.mk` 默认加入的 `-lineinfo` 来做行号映射）。

### 5.1 只看某一个文件（推荐）

以 `linear_base.cu` 为例：

```bash
cd /root/gpgpu-sim/accel-sim-framework/gpu-app-collection/gardenia/src/bfs

# 生成 cubin（只编译，不链接）
make linear_base.cubin

# 反汇编成 SASS（带行号信息）
make linear_base.sass

# 更详细的反汇编信息
make SASS_VERBOSE=1 linear_base.sass

# device debug（-G -g，方便对齐源码；但会改变优化与指令形态）
make DEBUG=1 linear_base.sass
```

> 注意：`make sass-all` 会尝试编译本目录下所有 `*.cu`，其中 `merrill.cu` 这类文件可能因为缺依赖而失败；因此建议按文件单独生成 `.sass`。

---

## 6) 我想把 `topo_* / hybrid_* / bottom_up / fusion` 也做成 A100 可运行版本：建议怎么做

如果你确实希望把这些历史 CUDA 版本也跑起来，建议按下面顺序推进：

1) 先选一个目标（例如 `topo_base.cu`），给它加 `BFSSolver(Graph&,...)` 适配，让它先能 `make bfs_topo_base` 通过链接并能跑通小图。
2) 对需要 reverse graph 的版本（`bottom_up`/`hybrid_*`），在 `main` 运行参数里开启 `reverse=1`，并在 wrapper 中使用 `g.in_rowptr()/g.in_colidx()`。
3) 逐步把 kernel 的 CSR 类型从 `int*`/`IndexT*` 升级为 `uint64_t*`（rowptr）与 `VertexId*`（colidx），避免大图溢出与多余拷贝。

如果你希望我直接动手改（比如先把 `bfs_topo_base` 在 A100 上跑起来，或把 `bottom_up/hybrid` 也适配到 `Graph` 接口），告诉我你想优先启用哪几个目标（`bfs_topo_base / bfs_topo_lb / bfs_bu / bfs_hybrid_base / bfs_fusion ...`）即可。

