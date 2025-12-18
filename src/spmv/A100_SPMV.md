# SpMV CUDA 版本（A100 / SM80）说明

本文档位于 `gardenia/src/spmv/`，只聚焦本目录下的 CUDA 实现（`*.cu`），并以 **NVIDIA A100（SM80）** 为目标说明：

- 这些 CUDA 文件分别代表什么（联系文件名，通俗解释 + 例子）。
- 目前哪些 CUDA 版本**无法直接编译/链接成可运行的 SpMV 可执行文件**，原因是什么，能否修改。
- 如何在 A100 上生成 `cubin`、反汇编得到 SASS，并查看 SASS ↔ 源码行号对应。

> 背景：`src/spmv/main.cc` 固定调用 `SpmvSolver(Graph&, ...)`（见 `spmv.h` 声明），因此能否“编译成可执行文件”不只取决于 CUDA 语法是否能过，还取决于 **Solver 接口是否匹配**。

---

## 0) A100 快速开始（只编译可运行的 CUDA 版本）

### 0.1 编译（A100 / sm_80）

`src/common.mk` 里已配置 `sm_80`（A100）：`-gencode arch=compute_80,code=sm_80`。

在 SpMV 目录编译（CUDA 版本）：

```bash
cd /root/gpgpu-sim/accel-sim-framework/gpu-app-collection/gardenia/src/spmv
make -j spmv_base spmv_warp spmv_vector spmv_push
```

生成的可执行文件会被移动到 `gardenia/bin/`（由 `src/common.mk` 的 `BIN=../../bin` 控制）。

### 0.2 运行（示例图）

`main.cc` 的用法字符串里写了 3 个可选参数，但**当前代码实际上只读取两个**：

- `argv[3]`：`symmetrize(0/1)`
- `argv[4]`：`reverse(0/1)`

因此推荐按下面形式运行：

```bash
cd /root/gpgpu-sim/accel-sim-framework/gpu-app-collection/gardenia/bin

# 用 symmetrize=1 让图变成无向（同时也保证 in_rowptr/in_colidx 可用，便于校验）
./spmv_base   mtx ../datasets/4 1 0
./spmv_warp   mtx ../datasets/4 1 0
./spmv_vector mtx ../datasets/4 1 0
./spmv_push   mtx ../datasets/4 1 0
```

> 说明：`spmv_base/warp/vector` 的实现使用 `g.in_rowptr()/g.in_colidx()`（即“入边 CSR”）；如果你既不 `symmetrize=1` 也不 `reverse=1`，`in_*` 指针可能未初始化，程序会不稳定。为了简单，建议先用 `symmetrize=1` 跑通小图。

---

## 1) 文件名怎么读：从名字看出“并行粒度/访问方式”

SpMV（Sparse Matrix-Vector Multiply）的核心就是：

> 对每一行 `row`，做 `y[row] += Σ A[row,col] * x[col]`。

在图语境里，你可以把 “一行” 理解成 “一个顶点”，把 “非零元素” 理解成 “边”，不同实现只是**把这些边工作怎么分给线程**不同。

常见命名暗号：

- `base`：最基础版本（通常 **一线程一行** / scalar）。
- `warp`：**一 warp 一行**（warp 内 32 线程协作把这一行的非零扫完再做归约）。
- `vector`：**一 vector 一行**（vector 是可配置长度，比如 2/4/8/16/32 线程；更适合“平均每行 nnz 较小/中等”的矩阵）。
- `tex/texture`：通过 **texture cache** 读取 `x`（属于 legacy 用法，思路是只读缓存更友好）。
- `push`：把 `x[src]` 沿边 “推” 到 `y[dst]`（scatter + `atomicAdd`），常用于图算法的“推送式更新”（等价于计算 `A^T x` 的一种方式）。
- `tiling/partition/push_tile`：更复杂的 **重排/分块/子图** 思路（把矩阵重新组织成更适合 cache/并行的形态，代码复杂度也显著上升）。
- `cusparse`：直接调用 NVIDIA 官方库 cuSPARSE 的 SpMV。

---

## 1.4) 深入微观：以 Base 版本为例看 CSR 计算逻辑

为了不浪费空间存那些 0，代码使用了 **CSR (Compressed Sparse Row)** 格式来压缩存储矩阵。下面以 `base.cu` 为例，用通俗的语言配合一个实际例子来解释它的逻辑。

### 1. 核心逻辑：分工合作

想象一下，我们要计算一个巨大的矩阵乘法 $y = A \times x$。
代码中的策略（`spmv_csr_scalar`）是**“一人包一行”**：
*   **GPU 上的每一个线程（Thread）** 就像一个学生。
*   **每一个学生负责计算矩阵的一行**。
*   学生拿着这一行的非零数字，去和向量 $x$ 里的对应数字相乘，然后加起来，算出这一行的结果。

### 2. 数据是怎么存的？(CSR 格式)

为了让学生知道自己该算哪些数，我们需要三个数组（花名册）：
1.  **`Ax` (Values)**: 只存矩阵里**非零**的数值。
2.  **`Aj` (Column Indices)**: 存上面那些数值在第几**列**。
3.  **`Ap` (Row Pointers)**: 存每一行在 `Ax` 和 `Aj` 里的**起始位置**。

### 3. 实际例子演示

假设我们有这样一个矩阵 $A$ 和向量 $x$：

$$
A = \begin{bmatrix}
1 & 0 & 2 \\
0 & 3 & 0 \\
4 & 0 & 5
\end{bmatrix}, \quad
x = \begin{bmatrix}
10 \\
20 \\
30
\end{bmatrix}
$$

**CSR 存储数组如下：**
*   **`Ax` (非零值)**: `[1, 2, 3, 4, 5]` (按行把非零数读出来)
*   **`Aj` (列号)**: `[0, 2, 1, 0, 2]` (1在第0列，2在第2列，3在第1列...)
*   **`Ap` (行偏移)**: `[0, 2, 3, 5]`
    *   第0行从索引 0 开始（包含1, 2）
    *   第1行从索引 2 开始（包含3）
    *   第2行从索引 3 开始（包含4, 5）
    *   最后补一个总数 5

### 4. 代码执行流程 (模拟 GPU 线程)

现在我们看代码 `spmv_csr_scalar` 是怎么跑的。假设有 3 个线程同时工作。

#### **线程 0 (负责第 0 行)**
1.  **认领任务**: `int row = ...` 算出自己是第 0 号线程，负责第 0 行。
2.  **查范围**:
    *   `row_begin = Ap[0] = 0`
    *   `row_end = Ap[1] = 2`
    *   这意味着它要处理 `Ax` 数组里下标 `0` 到 `1` 的元素。
3.  **开始计算 (循环)**:
    *   **第一次循环 (offset=0)**:
        *   拿到矩阵的值 `Ax[0] = 1`
        *   拿到列号 `Aj[0] = 0`
        *   去向量 $x$ 查值 `x[0] = 10`
        *   乘起来：`1 * 10 = 10`，加到总和里。
    *   **第二次循环 (offset=1)**:
        *   拿到矩阵的值 `Ax[1] = 2`
        *   拿到列号 `Aj[1] = 2`
        *   去向量 $x$ 查值 `x[2] = 30`
        *   乘起来：`2 * 30 = 60`，加到总和里。
4.  **写回结果**: 总和 `10 + 60 = 70`。把 `70` 写入 `y[0]`。

#### **线程 1 (负责第 1 行)**
1.  **查范围**: `Ap[1]=2` 到 `Ap[2]=3`。只处理下标 2 的元素。
2.  **计算**:
    *   `Ax[2] = 3` (矩阵里的3)
    *   `Aj[2] = 1` (在第1列)
    *   `x[1] = 20`
    *   乘积：`3 * 20 = 60`。
3.  **写回**: `y[1] = 60`。

#### **线程 2 (负责第 2 行)**
1.  **查范围**: `Ap[2]=3` 到 `Ap[3]=5`。处理下标 3, 4 的元素。
2.  **计算**:
    *   `4 * x[0] (10) = 40`
    *   `5 * x[2] (30) = 150`
3.  **写回**: `y[2] = 40 + 150 = 190`。

### 总结

这段代码的逻辑就是：**利用 CSR 格式快速跳过矩阵中的 0，每个 GPU 线程独立负责计算结果向量 $y$ 中的一个元素（即矩阵的一行与向量 $x$ 的点积）。**

---

## 1.5) 通俗理解：社交网络“热度统计”

我们把 SpMV ($y = Ax$) 想象成在计算一个社交网络中每个人的**“总影响力”**。

**具体数据例子**：
假设有 3 个用户（Row 0, 1, 2），关注关系如下（矩阵 $A$）：
- **Row 0 (大明星)**：被 1000 个人关注（这一行有 1000 个非零元素）。
- **Row 1 (路人甲)**：只被 1 个人关注（这一行有 1 个非零元素）。
- **Row 2 (路人乙)**：没人关注（这一行是空的）。

**任务**：算出每个人收到了多少热度（$y$）。

| 算法 | 对应文件 | 形象比喻 | 针对该数据的处理过程 | 优缺点 |
| :--- | :--- | :--- | :--- | :--- |
| **Base (Scalar)** | `base.cu` | **专属小秘书**<br>(一人管一人) | **Thread-per-Row**<br>• **Thread 0**：要处理 1000 个粉丝，累得半死。<br>• **Thread 1**：处理 1 个粉丝，瞬间干完。<br>• **Thread 2**：没事干，直接下班。<br>👉 **结果**：Thread 1 和 2 都在等 Thread 0，GPU 整体被拖慢。 | **缺点**：**贫富差距**。大明星的秘书累死，普通人的秘书闲死。<br>**优点**：逻辑简单。 |
| **Warp** | `warp.cu` | **全员会计团**<br>(一组管一人) | **Warp-per-Row**<br>• **Warp 0 (32人)**：一起处理大明星。每人分 30 个粉丝，速度提升 32 倍！<br>• **Warp 1 (32人)**：一起处理路人甲。1 个人干活，31 个人围观（浪费）。<br>👉 **结果**：大明星处理快了，但处理路人时浪费了大量算力。 | **优点**：大明星很开心。<br>**缺点**：**杀鸡用牛刀**。处理小户时资源浪费严重。 |
| **Vector** | `vector.cu` | **看人下菜碟**<br>(按需分配) | **Vector-per-Row**<br>• **Row 0**：派 32 人团去服务。<br>• **Row 1**：派 2 人组去服务。<br>👉 **结果**：既不让大明星卡顿，也不浪费人力在路人身上。 | **优点**：**资源均衡**。效率最高。<br>**缺点**：实现复杂。 |
| **Push** | `push.cu` | **送货上门**<br>(主动送分) | **Push / Scatter**<br>视角反转。不再是 Row 0 去查谁关注了自己，而是那 1000 个粉丝主动把热度加到 Row 0 的账户里。<br>👉 **结果**：这 1000 个人同时试图修改 Row 0 的账户（`atomicAdd`），导致**排队冲突**。 | **优点**：适合源点扩散。<br>**缺点**：**家门口堵车**（原子冲突）。 |

## 1.6) 为什么 SpMV 要用图数据 (.mtx)？

你可能会问：“SpMV 不是矩阵乘法吗？为什么输入全是图（Graph）？”

**答案：SpMV 是图算法的核心原语。**

在图计算中，稀疏矩阵 $A$ 通常代表**邻接矩阵**（Adjacency Matrix），向量 $x$ 代表**顶点属性**。
计算 $y = Ax$ 的物理含义就是：**每个顶点收集并聚合所有邻居的信息**。

- **PageRank**：核心迭代就是 $x_{new} = A^T x_{old}$（网页权重的传播）。
- **GCN (图卷积网络)**：核心操作是特征聚合，本质就是 SpMV。
- **BFS / SSSP**：可以看作是在特殊半环（Semiring）上的 SpMV。

因此，使用真实的图数据（`.mtx` 格式）作为 SpMV 的负载是非常标准且合理的，它模拟了图算法中最耗时的计算环节。

---

## 2) CUDA 文件清单：每个文件代表什么（通俗解释 + 例子）

### 2.1 直接可用（已实现 `SpmvSolver(Graph&,...)`，能编译并运行）

- `base.cu`（`spmv_base`）
  - 含义：CSR 标准写法，一线程处理一行（`row`），逐个累加 `Ax[offset] * x[col]`。
  - 例子：你可以把它理解为“最容易看懂、最像教科书”的 GPU SpMV；当每行 nnz 很小且分布均匀时，它也能跑得不错。

- `warp.cu`（`spmv_warp`）
  - 含义：一 warp 处理一行。warp 内每个 lane 负责扫一部分非零，再做 warp 内归约。
  - 例子：当很多行的 nnz 在几十到几百时，比 `base` 更不容易被“长行/短行不均匀”拖慢。

- `vector.cu`（`spmv_vector`）
  - 含义：一“vector”（2/4/8/16/32 线程）处理一行；并根据 `nnz_per_row` 自动选择 vector 宽度。
  - 例子：如果矩阵平均每行只有 3~10 个非零，使用 4 或 8 线程一组往往更划算（比强行用 32 线程的 warp 模式浪费更少）。
  - 备注：此文件使用 `texture` 读取 `x`（`tex1Dfetch`），属于 legacy 做法，但在很多 CUDA 版本仍可编译。

- `push.cu`（`spmv_push`）
  - 含义：push/scatter 版本：每个 `src` 沿着出边把贡献 `Ax * x[src]` 原子加到 `y[dst]`（`atomicAdd`）。
  - 例子：图算法里常见“把自己的值推给邻居”（比如把 rank/残差推给出邻居）就属于这种形式；当你更愿意按“源点出边”遍历时，这个写法更自然。

### 2.2 目前无法直接生成可运行可执行文件（需修改/有依赖/接口不匹配）

下面这些 `.cu` 多数实现的是旧接口 `SpmvSolver(int m, int nnz, ...)`，而当前 `main.cc` 只会调用 `SpmvSolver(Graph&,...)`，因此目标 `spmv_tex/spmv_cusparse/spmv_partition/spmv_tiling/spmv_push_tile` 往往会在链接阶段失败。

- `tex.cu`（Makefile 里有 `spmv_tex` 目标）
  - 含义：和 `base` 类似，但用 texture cache 读取 `x`。
  - 目前问题：函数签名是旧版，不匹配 `spmv.h` 的声明。
  - 能否修改：可以，最常见做法是新增一个 `SpmvSolver(Graph&,...)` wrapper，从 `Graph` 取 CSR 指针，再调用/改写 kernel 入参。

- `cusparse.cu`（Makefile 里有 `spmv_cusparse` 目标）
  - 含义：调用 cuSPARSE 的 `cusparseScsrmv` 做 SpMV。
  - 目前问题：
    - 旧接口签名不匹配；
    - 代码使用 `cusparseScsrmv`（旧 API），在新 CUDA 上可能会有弃用警告（更推荐 `cusparseSpMV`）。
  - 能否修改：可以，但建议直接改成 `SpmvSolver(Graph&,...)` 并在内部做：
    - CSR 指针与索引类型适配（`Graph` 的 rowptr 是 `uint64_t*`，cuSPARSE 通常用 `int*` 或 `int64_t*`）。
    - API 升级（如目标 CUDA 版本较新）。

- `tiling.cu`（Makefile 里有 `spmv_tiling` 目标）
  - 含义：把矩阵按列/块重排（blocking/tiling），希望获得更好的局部性与吞吐。
  - 目前问题：旧接口签名不匹配；并且这类“重排”通常还需要额外的预处理/辅助结构。
  - 能否修改：可以，但改动会比 `tex.cu` 大（需要理清 blocking 产生的数据结构如何从 `Graph` 构造）。

- `partition.cu`（Makefile 里有 `spmv_partition` 目标）
  - 含义：更激进的分段/分区，把图/矩阵切成多个子图与 range，再做 merge。
  - 目前问题：旧接口签名不匹配；另外包含一些更偏“研究代码”的实现细节（例如内联 PTX load/store），对新编译器/新架构可能需要额外验证。
  - 能否修改：可以，但建议先把接口跑通，再逐步验证性能/正确性。

- `push_tile.cu`（Makefile 里有 `spmv_push_tile` 目标）
  - 含义：push 版本的“分块/子图”实现（配合 `segmenting.h` 等预处理）。
  - 目前问题：旧接口签名不匹配。
  - 能否修改：可以，但同样需要把“分块/idx_map”这套预处理和 `Graph` 对接起来。

---

## 3) 在 A100 上看 SASS ↔ 源码对应：怎么生成 `.cubin/.sass`

本目录 `Makefile` 已加入 BFS 同款的 `cubin/sass` 辅助规则（依赖 `nvcc` 与 `nvdisasm`），并会利用 `src/common.mk` 默认带的 `-lineinfo` 做行号映射。

### 3.1 推荐：对单个文件生成 SASS

```bash
cd /root/gpgpu-sim/accel-sim-framework/gpu-app-collection/gardenia/src/spmv

make sass-base
make sass-warp
make sass-vector
make sass-push

# 更详细的 nvdisasm 输出（指令信息/谓词/机器码/行号）
make SASS_VERBOSE=1 sass-warp

# 需要更强的源码对应（会改变优化与指令形态）
make DEBUG=1 sass-warp
```

### 3.2 生成全目录所有 `.sass`（不推荐作为第一步）

```bash
make sass-all
```

> 注意：`sass-all` 会尝试编译本目录所有 `*.cu`（包括 `cusparse.cu/partition.cu/...`）。如果你的环境缺少某些依赖（或某些文件对新 CUDA 有兼容性问题），建议改用 `make sass-xxx` 单独生成。

---

## 4) 如果你希望我“把旧接口版本也改到 A100 可运行”

我可以按你希望的优先级做适配，常见路径是：

1) 给目标文件新增 `SpmvSolver(Graph&,...)` wrapper；
2) 从 `Graph` 取 `rowptr/colidx` 并处理索引类型（`uint64_t` ↔ `int/int64_t`）；
3) 对需要的预处理（blocking/segmenting/partition）做最小可用的对接；
4) 跑通小图（`datasets/4`）后再考虑性能/大图溢出与优化。

你告诉我想优先启用哪些目标（例如 `spmv_tex`、`spmv_cusparse`、`spmv_partition`、`spmv_tiling`、`spmv_push_tile`），我就按这个顺序改。

