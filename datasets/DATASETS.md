# GARDENIA `datasets/` 目录说明（新增大图 + `.mtx` 是什么）

本文档面向目录：`gardenia/datasets/`。你新增的数据集主要是一些经典图数据（社交网、Web 网、引文网、道路网、合成图），它们统一用 **`.mtx`（Matrix Market）** 格式保存。

> 运行提示：本仓库里 `bfs_linear_base` 这类程序的参数是  
> `./bfs_linear_base <filetype> <graph-prefix> [symmetrize] [reverse] [source]`  
> 当 `filetype=mtx` 时，`graph-prefix` 需要写 **不带 `.mtx` 后缀** 的前缀路径。

---

## 1) `.mtx` 怎么理解？它是不是图格式？

**`.mtx` 是 Matrix Market 格式**（常用于稀疏矩阵数据交换）。它本质上是“稀疏矩阵的非零元素列表”，但**图的邻接矩阵也是稀疏矩阵**，所以常把图用 `.mtx` 来存。

### 1.1 `.mtx` 和“图”的对应关系（最常见的用法）

把一个图看成邻接矩阵 `A`：

- 顶点 = 矩阵的行/列编号
- 边 `u -> v` = 矩阵里一个非零项 `A[u,v]`

Matrix Market 的 **coordinate（坐标）** 格式大致长这样：

```text
%%MatrixMarket matrix coordinate pattern general
5 5 3
1 2
2 5
4 3
```

含义（按“图”理解）：

- 第一行是格式声明（可当作注释头）
- `5 5 3` 表示 5×5 的矩阵里有 3 个非零项
- 后面每行 `(row, col)` 表示一条边：`1->2`、`2->5`、`4->3`

### 1.2 `pattern / integer / real` 的区别（是否带权）

你会看到类似：

- `pattern`：只给 `(row, col)`，不带权重（无权图）
- `integer` / `real`：一般是 `(row, col, value)`，第三列是权重/数值（有权图或带属性的边）

例如你跑通的 `4w.mtx` 就是 `integer`，每条边多一个整数值；但 **GARDENIA 的 BFS 读取器只读前两列（row/col）**，第三列会被忽略，所以 `4w.mtx` 仍能当成无权图跑 BFS。

### 1.3 `general / symmetric` 的区别（有向 vs 无向的常见映射）

Matrix Market 里常见两类：

- `general`：矩阵不要求对称；常用来表示**有向图**（边 `u->v` 与 `v->u` 可独立存在）
- `symmetric`：只存上三角或下三角；常用来表示**无向图**（一条无向边 `{u,v}` 通常只存一次）

在本仓库里，如果 `.mtx` 文件头是 `symmetric`，通常建议运行时把 `symmetrize=1`，让读入阶段补齐反向边，恢复成真正的无向图。

---

## 2) 你新增的这些数据集分别是什么意思？

下面按目录名解释每个数据集的直观含义，并给一个“把它当作什么现实网络”的例子，以及 GARDENIA 的 BFS 跑法示例。

> 规模说明：下文的 `|V|`/`nnz` 来自 `.mtx` 文件头（矩阵维度/非零数）；实际载入后，程序还会去重/去自环，最终 `|E|` 可能略有变化。

### 2.1 `web-Google/`：Google 网页链接图（Web Graph）

- 文件：`web-Google/web-Google.mtx`
- 名字含义：`web` 表示网页网络；`Google` 表示来源是 Google 的网页抓取数据。
- 现实例子：
  - 顶点 = 一个网页（URL）
  - 有向边 `u->v` = 网页 `u` 上有一个超链接指向网页 `v`
  - BFS 从一个网页出发，可以理解成“点几次链接能到达哪些网页”
- 图类型：通常按 **有向图** 使用（文件头是 `general`）
- 规模（文件头）：`|V|=916,428`，`nnz=5,105,039`
- BFS 示例（从 `gardenia/bin` 运行）：
  - `./bfs_linear_base mtx ../datasets/web-Google/web-Google 0 0 0`

### 2.2 `cit-Patents/`：专利引用网络（Citation Graph）

- 文件：`cit-Patents/cit-Patents.mtx`
- 名字含义：`cit`=citation（引用）；`Patents`=专利。
- 现实例子：
  - 顶点 = 一篇专利
  - 有向边 `u->v` = 专利 `u` 引用了专利 `v`
  - BFS 从某专利出发，可以理解成“追溯引用链/引用传播范围”
- 图类型：**有向图**（文件头标注 `directed graph` / `general`）
- 规模（文件头）：`|V|=3,774,768`，`nnz=16,518,948`
- BFS 示例：
  - `./bfs_linear_base mtx ../datasets/cit-Patents/cit-Patents 0 0 0`

**特别注意：`cit-Patents_nodename.mtx` 不是边列表**

- 文件：`cit-Patents/cit-Patents_nodename.mtx`
- 它的头是 `matrix array real general`，尺寸是 `3774768 1`，后面是一列数字；这是“节点原始编号/名字映射”的辅助数据，不是图的邻接边。
- 不要把它当 `graph-prefix` 传给 BFS；BFS 应该用 `cit-Patents.mtx`。

### 2.3 `soc-LiveJournal1/`：LiveJournal 社交网络

- 文件：`soc-LiveJournal1/soc-LiveJournal1.mtx`
- 名字含义：`soc`=social（社交）；`LiveJournal` 是一个社区/博客平台。
- 现实例子：
  - 顶点 = 用户账号
  - 有向边 `u->v` = 用户 `u` “关注/订阅/好友关系指向”用户 `v`（具体语义取决于数据定义）
  - BFS 可以理解成“社交关系传播的跳数”
- 图类型：常按 **有向图** 使用（文件头是 `general`）
- 规模（文件头）：`|V|=4,847,571`，`nnz=68,993,773`
- BFS 示例：
  - `./bfs_linear_base mtx ../datasets/soc-LiveJournal1/soc-LiveJournal1 0 0 0`

### 2.4 `soc-orkut/`：Orkut 社交网络

- 文件：`soc-orkut/soc-orkut.mtx`
- 名字含义：`soc`=social；`orkut` 是 Google 曾经的社交平台 Orkut。
- 现实例子：
  - 顶点 = Orkut 用户
  - 无向边 `{u,v}` = 两个用户互为好友
  - BFS 从一个人出发就是“几度好友（1 度/2 度/3 度）”
- 图类型：**无向图**（文件头是 `symmetric`）
- 规模（文件头）：`|V|=2,997,166`，`nnz=106,349,209`（注意：symmetric 通常只存一半）
- BFS 示例（建议 `symmetrize=1`）：
  - `./bfs_linear_base mtx ../datasets/soc-orkut/soc-orkut 1 0 0`

### 2.5 `flickr/`：Flickr 网络（图片社区）

- 文件：`flickr/flickr.mtx`
- 名字含义：Flickr 是图片分享社区；这里是对其网络关系的一次抓取/构建（常见是用户关系、关注关系或互动关系）。
- 现实例子：
  - 顶点 = Flickr 用户
  - 有向边 `u->v` = 用户 `u` 关注/联系/指向用户 `v`
  - BFS 可以理解成“从某个用户出发，沿关注链能触达哪些用户”
- 图类型：常按 **有向图** 使用（文件头是 `general`）
- 规模（文件头）：`|V|=820,878`，`nnz=9,837,214`
- BFS 示例：
  - `./bfs_linear_base mtx ../datasets/flickr/flickr 0 0 0`

### 2.6 `roadNet-CA/`：加州道路网络

- 文件：`roadNet-CA/roadNet-CA.mtx`
- 名字含义：`roadNet`=道路网络；`CA`=California（加州）。
- 现实例子：
  - 顶点 = 路口/道路节点
  - 无向边 `{u,v}` = 两个路口之间有一段道路连接
  - BFS（按无权）可以理解成“最少经过多少个路口能到达”（不是最短距离公里数）
- 图类型：**无向图**（文件头是 `symmetric`）
- 规模（文件头）：`|V|=1,971,281`，`nnz=2,766,607`
- BFS 示例（建议 `symmetrize=1`）：
  - `./bfs_linear_base mtx ../datasets/roadNet-CA/roadNet-CA 1 0 0`

### 2.7 `road_usa/`：美国道路网络（DIMACS10）

- 文件：`road_usa/road_usa.mtx`
- 名字含义：`road`=道路；`usa`=美国。
- 现实例子：同道路网络（顶点=路口/节点，边=道路连接），但规模更大。
- 图类型：**无向图**（文件头是 `symmetric`）
- 规模（文件头）：`|V|=23,947,347`，`nnz=28,854,312`（超大规模顶点数）
- BFS 示例（建议 `symmetrize=1`，并准备较大的显存/内存）：
  - `./bfs_linear_base mtx ../datasets/road_usa/road_usa 1 0 0`

### 2.8 `kron_g500-logn21/`：Graph500 Kronecker 合成图（规模 2^21）

- 文件：`kron_g500-logn21/kron_g500-logn21.mtx`
- 名字含义：
  - `kron`=Kronecker（克罗内克积/克罗内克图生成）
  - `g500`=Graph500 基准（BFS 是 Graph500 的核心 benchmark）
  - `logn21` 通常表示顶点规模接近 `2^21 = 2,097,152`
- 现实例子：
  - 这不是现实采样的网络，而是“长得像社交网/互联网”的合成图：通常有明显的幂律度分布、高度不均匀的出度/入度。
  - 常用来测试 BFS/图遍历在“尺度大 + 度分布极不均匀”时的性能。
- 图类型：无向（文件头是 `symmetric`，且标注 `undirected multigraph`）
- 规模（文件头）：`|V|=2,097,152`，`nnz=91,042,010`
- BFS 示例（建议 `symmetrize=1`）：
  - `./bfs_linear_base mtx ../datasets/kron_g500-logn21/kron_g500-logn21 1 0 0`

---

## 3) 运行参数小抄（跟 `.mtx` 关系最大的是 `symmetrize`）

- `filetype`：对这些数据集用 `mtx`
- `graph-prefix`：写“文件路径前缀”，例如 `../datasets/web-Google/web-Google`（不要写 `.mtx`）
- `symmetrize`：
  - `.mtx` 头是 `symmetric`（无向图常见）→ 一般选 `1`
  - `.mtx` 头是 `general`（有向图常见）→ 一般选 `0`
- `reverse`：是否额外构建反向边列表（部分算法会用到入边）；`bfs_linear_base` 不需要时可用 `0`
- `source`：BFS 起点顶点 id（默认 0）

---

## 4) 附：这些数据从哪里来？

`datasets/test.mk` 里记录了下载来源，主要来自：

- UF Sparse Matrix Collection（很多 SNAP / DIMACS10 数据以 `.mtx` 形式提供）
- nrvis / networkrepository（例如 `soc-orkut`）

如果你要写论文/报告，建议同时查看各数据集 `.mtx` 文件头部的注释与 `soc-orkut/readme.txt` 的引用说明。

