# GARDENIA 在 A100（CUDA 12.6）上的初始化/部署说明

本文档面向你当前仓库路径：`/root/gpgpu-sim/accel-sim-framework/gpu-app-collection/gardenia`，目标是在真实硬件 **A100（SM 8.0）** 上使用 **CUDA 12.6** 编译并运行 GARDENIA（不包含/不依赖 GPGPU-Sim/Accel-Sim 的仿真流程）。

---

## 0. 目录速览

- 源码：`src/`
- 公共头文件：`include/`
- 生成的可执行文件：`bin/`
- 自带小数据集：`datasets/`
- CUB（第三方库副本）：`cub/`（用于 `cub::BlockScan` / `cub::BlockReduce` 等并行原语）

---

## 1. 环境前置条件（A100 + CUDA 12.6）

建议检查：

```bash
nvidia-smi
nvcc --version
gcc --version
```

常见要求：

- CUDA 12.6 Toolkit 已安装（并且 `nvcc` 在 `PATH` 中，或你能在 `src/common.mk` 里显式指定）。
- NVIDIA 驱动版本满足 CUDA 12.6 要求。
- host 编译器版本与 CUDA 12.6 兼容（通常建议 GCC 11.x；若系统 GCC 太新，`nvcc` 可能会报 host 编译器不支持）。

---

## 2. 必改：为 A100 增加 `sm_80`（否则运行会报 “no kernel image”）

GARDENIA 的编译参数集中在：`src/common.mk`。

打开 `src/common.mk`，至少做下面三件事（示例按 CUDA 12.6 安装在 `/usr/local/cuda-12.6` 编写；若你是 `/usr/local/cuda`，按实际调整）：

1）设置 CUDA 路径和 nvcc：

```make
CUDA_HOME=/usr/local/cuda-12.6
NVCC=$(CUDA_HOME)/bin/nvcc
```

2）为 A100 增加/只保留 `sm_80` 架构（推荐只保留你要跑的架构，fatbin 更小）：

```make
CUDA_ARCH := \
	-gencode arch=compute_80,code=sm_80
NVFLAGS=$(CUDA_ARCH)
```

（如果你还希望同一 binary 能在 V100/T4 等上跑，可以额外加 `sm_70/sm_75`）

3）确保链接阶段能找到 CUDA 的库（尤其是编译 `spmv_cusparse` 这类会 `-lcusparse` 的 target 时）：

```make
LIBS = -L$(CUDA_HOME)/lib64
LIBS += -L/usr/lib64
```

---

## 3. 编译（推荐按子目录编译，避免不必要依赖）

进入源码目录：

```bash
cd /root/gpgpu-sim/accel-sim-framework/gpu-app-collection/gardenia/src
```

### 3.1 常用图分析 kernels（不依赖 Intel 编译器）

```bash
make -j -C bfs
make -j -C bc
make -j -C cc
make -j -C pr
make -j -C sssp
make -j -C spmv
make -j -C symgs
make -j -C vc
```

编译产物默认会被移动到 `gardenia/bin/`（由 `src/common.mk` 里的 `BIN=../../bin` 控制）。

### 3.2 可能踩坑的目录/target

- `src/nvGRAPH`：Makefile 依赖 `-lnvgraph`，该库在新 CUDA 版本中通常已不可用；建议跳过。
- `*_omp_target.cpp` 相关目标：Makefile 往往使用 `ICPC/ICC`（Intel 编译器）编译 `.cpp`；如果你没装 Intel 编译器，避免构建这些 target（仅构建 CUDA/G++ 能完成的目标即可）。

---

## 4. 运行方式：注意有三种不同的“输入图”接口

GARDENIA 里并不是所有 kernel 都用同一种图加载器，因此命令行形式不同。

### 4.1 “prefix + filetype” 形式（读取 `prefix.mtx`）

这类程序使用 `include/csr_graph.h` 的 `Graph(prefix, filetype, symmetrize, reverse)`，典型的有：

- BFS / BC / CC / PR / SSSP / SpMV / SymGS / VC

运行示例（以自带数据集为例）：

```bash
cd /root/gpgpu-sim/accel-sim-framework/gpu-app-collection/gardenia/bin

# BFS：文件类型 mtx；prefix 不带 .mtx 后缀
./bfs_linear_base mtx ../datasets/4 0 0 0

# CC：无向图通常需要 symmetrize=1（让边双向）
./cc_base mtx ../datasets/chesapeake 1 0

# PR：通常按有向图跑（symmetrize=0）
./pr_gather_warp mtx ../datasets/4 0
```

重要说明：

- `symmetrize=1` 会在读入时补齐反向边，适合无向算法或“只给上三角/下三角”的 `.mtx`（例如 `chesapeake.mtx` 是 `symmetric`）。
- `reverse=1` 会额外构建转置图（入边列表），`SymGS` 等会用到。

### 4.2 “直接给文件名（带后缀）” 形式（支持 `.mtx/.graph/.gr`）

这类程序使用 `include/graph_io.h` 的 `read_graph()`（通过文件后缀判断格式），典型的有：

- MST（`src/mst/main.cu`）
- SCC（`src/scc/main.cc`）
- SGD（`src/sgd/main.cc`）

运行示例（用自带测试数据）：

```bash
cd /root/gpgpu-sim/accel-sim-framework/gpu-app-collection/gardenia/bin

# MST：带权 mtx（第三列会作为边权读取）
./mst ../datasets/test_mst.mtx

# SCC：第二个参数可指定是否有向（1=有向；0=无向并自动 symmetrize）
./scc ../datasets/test_scc.mtx 1

# SGD：mtx 的第三列作为 rating
./sgd ../datasets/test_sgd.mtx 0.05 0.003 5
```

### 4.3 “二进制前缀（.meta.txt/.vertex.bin/.edge.bin）” 形式

`src/tc`（Triangle Counting）等图挖掘/挖矿相关程序使用另一套 `Graph(prefix)`（见 `include/graph.hh` / `src/common/graph.cc`），需要你提前把图预处理成二进制 CSR 文件（`*.meta.txt`、`*.vertex.bin`、`*.edge.bin`）。

注意：`datasets/` 目录下默认 **不提供** 这类二进制前缀数据，所以 `tc` 不能直接用 `datasets/*.mtx` 跑，需要你先准备转换流程。

---

## 5. `datasets/` 自带数据集都是什么意思？

目录：`datasets/` 里的文件可以分为三类：**格式示例**、**真实小图**、**各 kernel 的最小测试用例**。

### 5.1 格式示例：同一张小图的多种格式（4 / 4w）

- `4.mtx`：14×14，`nnz=256` 的 MatrixMarket（pattern/general）示例图；包含重复边/自环等，用于测试“去重、去自环”等清洗逻辑。
- `4w.mtx`：`4.mtx` 的带权版本（integer/general，第三列是权重）。
- `4.graph`：Metis/DIMACS10 常见的 `.graph` 邻接表格式版本（同一张小图；文件里有空行表示该顶点无邻居）。
- `4.gr`：9th DIMACS challenge 的 `.gr` 格式版本（行以 `a` 开头；如果有第三列权重，`graph_io.h` 当前会忽略并按权重 1 处理）。
- `4w.graph`：带边权的 `.graph` 变体（邻居与权重交替出现）。注意：当前 `graph_io.h::graph2csr()` **不解析** 这种带权 `.graph`，不要直接用它做 `read_graph()` 输入；如果需要权重，请优先用 `4w.mtx`。

### 5.2 真实小图：`chesapeake.mtx`

- `chesapeake.mtx`：来自 UF Sparse Matrix Collection 的 DIMACS10 数据集（clustering/chesapeake），`39×39 nnz=170`，MatrixMarket `pattern symmetric`，表示无向图的稀疏邻接矩阵（通常文件只给出一半边）。如果你用的是 “prefix + filetype” 的程序，建议运行时 `symmetrize=1`。

### 5.3 kernel 最小测试用例（用于快速 sanity check）

- `test_bc.mtx`：BC（Betweenness Centrality）的最小图用例（与 `test/graphs/bc.mtx` 相同）。
- `test_cc.mtx`：CC（Connected Components）的最小图用例（包含多个连通分量，便于验证）。
- `test_pr.mtx`：PR（PageRank）的最小有向图用例（与 `test/graphs/pr.mtx` 相同）。
- `test_scc.mtx` / `test_small_scc.mtx`：SCC（Strongly Connected Components）的最小有向图用例（包含环，便于验证 SCC 分解）。
- `test_mst.mtx`：MST（Minimum Spanning Tree）的最小带权用例（第三列为边权，供 `read_graph()` 读取）。
- `test_sgd.mtx`：SGD（矩阵分解/推荐系统）最小评分矩阵用例（第一列 user，第二列 item，第三列 rating）。
- `test1.graph`~`test6.graph`：若干极小的 `.graph` 邻接表用例，主要用于格式转换/算法调试/边界情况测试。

### 5.4 `test.mk`（不是数据集）

- `test.mk`：只是一些 `wget` 下载大规模公开数据集的链接集合（例如 LiveJournal、web-Google、roadNet-CA 等），不是 Makefile 规则，也不是数据文件本身。

---

## 6. 常见问题排查（A100 + CUDA 12.6）

- 运行时报 `no kernel image is available for execution on the device`：
  - 99% 是因为没编译 `sm_80`。回到第 2 节修改 `src/common.mk` 的 `CUDA_ARCH`。
- `nvcc` 报 host 编译器不支持：
  - 换成 CUDA 12.6 支持的 GCC 版本，或用 `nvcc -ccbin` 指定兼容版本。
- 链接时报 `cannot find -lcusparse`（或运行时找不到 CUDA 动态库）：
  - 检查 `src/common.mk` 里的 `LIBS` 是否包含 `-L$(CUDA_HOME)/lib64`，以及 `LD_LIBRARY_PATH` 是否正确。
- `src/nvGRAPH` 无法编译：
  - 这是预期的（新 CUDA 无 nvGRAPH），跳过即可。
- CUB 相关编译问题：
  - 本仓库自带 `cub/`（v1.8.0）。若与 CUDA 12.6 出现兼容性报错，可考虑改用 CUDA Toolkit 自带的 `<cub/...>`（需要调整 include path 优先级）。
