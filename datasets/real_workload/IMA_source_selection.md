# Gardenia real_workload 源点选择（偏 IMA 压力）

这份报告用于给 **SSSP/BFS/BC** 这类“从 source 出发扩散”的图算法选 `source_id`，目标是让访存更不规则、frontier 更大，从而更容易体现 IMA/L1 压力。

## 方法（可复现）

- 对每个 `*.mtx` 统计 **out-degree**（按 MTX 原方向），以及 **symmetrized degree**（把每条边同时记到两端，等价于把图当无向看待的度）。
- 计算 non-zero 度的 `median/p90/p99`，并给出一个“度 >= 阈值”的代表顶点；同时给出最大度顶点（hub）。
- 说明：这里不做“去重边”的精确处理（`csr_graph.h` 会去重），所以度数是偏上界；用于挑 source 做压力测试一般够用。

### 你提到的“反例/限制”是否考虑？（回答：目前未显式考虑）

这份报告目前只用 **度（degree）** 做第一阶段筛选，确实没有显式计算下列因素：

- **邻居是否“局部”**：比如某个大度点的邻居编号非常集中/连续，访问 `dist[neighbor]` 时可能更容易落在同一两个 cache line，IMA 会变轻。
- **实现是否有缓解手段**：例如 Gardenia 的 CSR 读入会对邻接表排序/去重，这会让读 `column_indices` 更顺序；但对 `dist[dst]` 这种间接访问是否变好，仍取决于 `dst` 编号是否集中。
- **真正决定 IMA 的其它图特征**：方向性/可达性、社团结构、重复访问、原子更新热点、frontier 的增长曲线等。

为什么仍然先用度？因为它是**最便宜、最稳的“压力放大器”**：在多数真实图里，顶点编号并不刻意按社区连续排列，高度点往往仍会带来更分散的 neighbor id，从而更容易触发随机访存。

如果你希望把这些反例也纳入筛选，我建议做“第二阶段”过滤（下一步可做）：

- 对候选 source（比如 hub/p90/p99）再统计：邻居 id 的跨度（max-min）、相邻邻居 id 的 gap 分布、以及 `neighbor_id // 32` 的唯一值个数（估计触及多少个 128B cache line，越多越随机）。
- 再做一个很浅的采样 BFS（比如只跑前 2~3 层）估计 frontier 是否会快速爆炸，避免选到“度大但不可达/扩散慢”的点。

本次更新已经把“第二阶段”的**邻居局部性**统计加进来了（见每个数据集的 `Locality (out)` 表）。
如何读这个表（都越大越不局部/越随机）：

- `unique_lines`：邻居落在多少个不同的 cache line（默认假设 128B line、4B 元素，所以一条 line 有 32 个元素）。越大通常越随机。
- `neighbors_per_line`：平均每个 cache line 上有多少个邻居，越大说明邻居更“挤在一起”（更局部，IMA 可能更轻）。
- `neighbor_span`：邻居 id 的最大最小跨度，越大说明编号更分散（通常更随机）。
- `line_density`：`unique_lines / line_span`，越接近 1 说明邻居覆盖的 cache line 越连续；越小说明越稀疏/分散。
- `bfs2_frontier2`：从 source 出发走 2 跳，能新到达多少个不同顶点（很粗的“早期扩散速度”指标）。越大通常越容易很快把 frontier 做大。

复现命令（在 repo 根目录）：

```bash
g++ -O3 -std=c++11 gpu-app-collection/gardenia/tools/mtx_degree_stats.cc \
  -o gpu-app-collection/gardenia/tools/mtx_degree_stats
gpu-app-collection/gardenia/tools/mtx_degree_stats <file.mtx...>
gpu-app-collection/gardenia/tools/mtx_degree_stats --symmetrize <file.mtx...>

g++ -O3 -std=c++11 gpu-app-collection/gardenia/tools/mtx_source_locality.cc \
  -o gpu-app-collection/gardenia/tools/mtx_source_locality
# out-neighbors 的局部性 + 2-hop 扩散（--bfs2）
gpu-app-collection/gardenia/tools/mtx_source_locality --bfs2 <file.mtx> <source0> [source1...]
# sym neighbors（把边当无向）
gpu-app-collection/gardenia/tools/mtx_source_locality --bfs2 --symmetrize <file.mtx> <source0> [source1...]
```

## 结论（怎么选 source）

先解释一下报告里两行“推荐 source”的含义（通俗版）：

- 这些数字（例如 `3569341`）就是 **source 的顶点编号（vertex id）**，用于填到命令行里的 `source_id` 位置。
- 这个编号是 **0-based**（从 0 开始数），因为 Gardenia 在读 `.mtx` 时会把文件里的 1-based 顶点编号减 1 存到内部。
- “度（degree）”可以理解成“这个点连出去/连着多少条边”。度越大，算法扩散时越容易遇到 **随机访存、热点竞争、frontier 变大**，对 IMA 更友好。

推荐策略：

- **首选 hub**：`max_vertex`，也就是“度最高的点”（可以理解为图里的“大 V/枢纽点”）。
  - 优点：通常最容易把 IMA/L1 压力拉满。
  - 缺点：也可能让仿真更慢、噪声更大（过于极端）。
- **备选 p90**：`p90_vertex`，也就是“度排进前 10% 的代表点”（更温和）。
  - 优点：通常更稳、更省时，但仍能给出明显压力。

额外提醒：

- **路网（road_usa/roadNet-CA）强烈建议用 `symmetrize=1`**，否则很容易出现 source 可达性差、迭代轮数很少（工作量被“测没了”）。

## 各数据集统计与推荐 source

### cit-Patents

- 文件：`cit-Patents/cit-Patents.mtx`
- |V| = 3774768

**Out-degree（按 MTX 方向）**

| 指标 | 值 |
|---|---|
| avg out-degree | 4.376 |
| zero out-degree vertices | 1685423 |
| hub (max_degree) | deg=770, v=3569341 |
| p99 (non-zero) | deg≈39, v=1692580 |
| p90 (non-zero) | deg≈15, v=1654613 |
| median (non-zero) | deg≈6, v=1654481 |

**Symmetrized degree（把图当无向）**

| 指标 | 值 |
|---|---|
| avg degree | 8.752 |
| zero degree vertices | 0 |
| hub (max_degree) | deg=793, v=2514765 |
| p99 (non-zero) | deg≈47, v=22843 |
| p90 (non-zero) | deg≈19, v=3130 |
| median (non-zero) | deg≈6, v=193 |

**推荐（用于 IMA 压力）**

- 首选 hub（度最高的点）：`3569341`（更激进，压力更大）
- 备选 p90（前 10% 高度点的代表）：`1654613`（更稳，通常更省时）

**第二阶段：邻居局部性检查（防反例）**

Locality (out)：（按 MTX 方向统计 source 的 out-neighbors）

| candidate | source_id | unique_neighbors | bfs2_frontier2 | unique_lines | neighbors_per_line | line_density | neighbor_span |
|---|---:|---:|---:|---:|---:|---:|---:|
| hub (max_degree) |  `3569341` | 770 | 3542 | 715 | 1.08 | 0.007 | 3331212 |
| p99 (non-zero) |  `1692580` | 39 | 0 | 38 | 1.03 | 0.001 | 1492150 |
| p90 (non-zero) |  `1654613` | 15 | 0 | 15 | 1.00 | 0.000 | 1412632 |

> 注意：`p90` 的 `bfs2_frontier2=0`，它可能是“只出 1 跳就到尽头”的点，适合做快速 sanity，但不适合压 IMA。

解读（这张表说明什么）：

- `hub` (`3569341`)：邻居非常分散（接近“一邻居一条 cache line”），更容易触发随机访存/IMA。 2 跳扩散较大，更可能形成多轮 frontier 压力。（neighbor_span≈0.88*|V|）
- `p99` (`1692580`)：邻居非常分散（接近“一邻居一条 cache line”），更容易触发随机访存/IMA。 2 跳扩散为 0，压力很可能只在第 1 轮出现，后续很快结束。（neighbor_span≈0.40*|V|）
- `p90` (`1654613`)：邻居非常分散（接近“一邻居一条 cache line”），更容易触发随机访存/IMA。 2 跳扩散为 0，压力很可能只在第 1 轮出现，后续很快结束。（但该点度本身很小，单轮访问量有限）（neighbor_span≈0.37*|V|）

建议命令行（按 Gardenia 参数约定；以 `./data/{name}` 为前缀举例）：

```bash
# SSSP: <filetype> <prefix> [symmetrize] [reverse] [source] [delta]
./sssp_linear_base mtx ./data/cit-Patents 0 0 3569341 1
./sssp_linear_base mtx ./data/cit-Patents 0 0 1654613 1
# BFS: <filetype> <prefix> [symmetrize] [reverse] [source]
./bfs_linear_base mtx ./data/cit-Patents 0 0 3569341
./bfs_linear_base mtx ./data/cit-Patents 0 0 1654613
# BC: <filetype> <prefix> [symmetrize] [reverse] [source]
./bc_linear_base mtx ./data/cit-Patents 0 0 3569341
./bc_linear_base mtx ./data/cit-Patents 0 0 1654613
```

通俗例子（拿这两行解释）：

- “首选 hub：`3569341`” 就是：把 `source_id` 设为 `3569341`，从一个连接很多边的“大点”出发，通常更容易把 IMA 压力打满。
- “备选 p90：`1654613`” 就是：把 `source_id` 设为 `1654613`，从一个“也很活跃但没那么极端”的点出发，往往更稳更省时。

### flickr

- 文件：`flickr/flickr.mtx`
- |V| = 820878

**Out-degree（按 MTX 方向）**

| 指标 | 值 |
|---|---|
| avg out-degree | 11.984 |
| zero out-degree vertices | 265189 |
| hub (max_degree) | deg=10272, v=1586 |
| p99 (non-zero) | deg≈304, v=57 |
| p90 (non-zero) | deg≈24, v=14 |
| median (non-zero) | deg≈2, v=0 |

**Symmetrized degree（把图当无向）**

| 指标 | 值 |
|---|---|
| avg degree | 23.968 |
| zero degree vertices | 0 |
| hub (max_degree) | deg=14902, v=1586 |
| p99 (non-zero) | deg≈445, v=14 |
| p90 (non-zero) | deg≈28, v=14 |
| median (non-zero) | deg≈2, v=0 |

**推荐（用于 IMA 压力）**

- 首选 hub（度最高的点）：`1586`（更激进，压力更大）
- 备选 p90（前 10% 高度点的代表）：`14`（更稳，通常更省时）

**第二阶段：邻居局部性检查（防反例）**

Locality (out)：（按 MTX 方向统计 source 的 out-neighbors）

| candidate | source_id | unique_neighbors | bfs2_frontier2 | unique_lines | neighbors_per_line | line_density | neighbor_span |
|---|---:|---:|---:|---:|---:|---:|---:|
| hub (max_degree) |  `1586` | 10272 | 180644 | 1895 | 5.42 | 0.795 | 76247 |
| p99 (non-zero) |  `57` | 338 | 26425 | 52 | 6.50 | 0.684 | 2388 |
| p90 (non-zero) |  `14` | 46 | 2658 | 3 | 15.33 | 1.000 | 56 |

解读（这张表说明什么）：

- `hub` (`1586`)：邻居编号更集中（局部性更好），IMA 可能不算重。 2 跳扩散较大，更可能形成多轮 frontier 压力。（neighbor_span≈0.09*|V|）
- `p99` (`57`)：邻居编号更集中（局部性更好），IMA 可能不算重。 2 跳扩散较大，更可能形成多轮 frontier 压力。（neighbor_span≈0.00*|V|）
- `p90` (`14`)：邻居编号更集中（局部性更好），IMA 可能不算重。 2 跳扩散较大，更可能形成多轮 frontier 压力。（neighbor_span≈0.00*|V|）

建议命令行（按 Gardenia 参数约定；以 `./data/{name}` 为前缀举例）：

```bash
# SSSP: <filetype> <prefix> [symmetrize] [reverse] [source] [delta]
./sssp_linear_base mtx ./data/flickr 0 0 1586 1
./sssp_linear_base mtx ./data/flickr 0 0 14 1
# BFS: <filetype> <prefix> [symmetrize] [reverse] [source]
./bfs_linear_base mtx ./data/flickr 0 0 1586
./bfs_linear_base mtx ./data/flickr 0 0 14
# BC: <filetype> <prefix> [symmetrize] [reverse] [source]
./bc_linear_base mtx ./data/flickr 0 0 1586
./bc_linear_base mtx ./data/flickr 0 0 14
```

通俗例子（拿这两行解释）：

- “首选 hub：`1586`” 就是：把 `source_id` 设为 `1586`，从一个连接很多边的“大点”出发，通常更容易把 IMA 压力打满。
- “备选 p90：`14`” 就是：把 `source_id` 设为 `14`，从一个“也很活跃但没那么极端”的点出发，往往更稳更省时。

### kron_g500-logn21

- 文件：`kron_g500-logn21/kron_g500-logn21.mtx`
- |V| = 2097152

**Out-degree（按 MTX 方向）**

| 指标 | 值 |
|---|---|
| avg out-degree | 43.412 |
| zero out-degree vertices | 817617 |
| hub (max_degree) | deg=196832, v=1930586 |
| p99 (non-zero) | deg≈1268, v=73498 |
| p90 (non-zero) | deg≈102, v=12812 |
| median (non-zero) | deg≈6, v=1401 |

**Symmetrized degree（把图当无向）**

| 指标 | 值 |
|---|---|
| avg degree | 86.823 |
| zero degree vertices | 553065 |
| hub (max_degree) | deg=213904, v=1930586 |
| p99 (non-zero) | deg≈1900, v=63 |
| p90 (non-zero) | deg≈194, v=4 |
| median (non-zero) | deg≈8, v=1 |

**推荐（用于 IMA 压力）**

- 首选 hub（度最高的点）：`1930586`（更激进，压力更大）
- 备选 p90（前 10% 高度点的代表）：`12812`（更稳，通常更省时）

**第二阶段：邻居局部性检查（防反例）**

Locality (out)：（按 MTX 方向统计 source 的 out-neighbors）

| candidate | source_id | unique_neighbors | bfs2_frontier2 | unique_lines | neighbors_per_line | line_density | neighbor_span |
|---|---:|---:|---:|---:|---:|---:|---:|
| hub (max_degree) |  `1930586` | 196832 | 928489 | 58422 | 3.37 | 0.968 | 1930567 |
| p99 (non-zero) |  `73498` | 1447 | 9477 | 1072 | 1.35 | 0.468 | 73239 |
| p90 (non-zero) |  `12812` | 108 | 450 | 97 | 1.11 | 0.242 | 12738 |

解读（这张表说明什么）：

- `hub` (`1930586`)：邻居编号更集中（局部性更好），IMA 可能不算重。 2 跳扩散较大，更可能形成多轮 frontier 压力。（neighbor_span≈0.92*|V|）
- `p99` (`73498`)：邻居分散程度中等。 2 跳扩散较大，更可能形成多轮 frontier 压力。（neighbor_span≈0.03*|V|）
- `p90` (`12812`)：邻居非常分散（接近“一邻居一条 cache line”），更容易触发随机访存/IMA。 2 跳扩散较大，更可能形成多轮 frontier 压力。（neighbor_span≈0.01*|V|）

建议命令行（按 Gardenia 参数约定；以 `./data/{name}` 为前缀举例）：

```bash
# SSSP: <filetype> <prefix> [symmetrize] [reverse] [source] [delta]
./sssp_linear_base mtx ./data/kron_g500-logn21 0 0 1930586 1
./sssp_linear_base mtx ./data/kron_g500-logn21 0 0 12812 1
# BFS: <filetype> <prefix> [symmetrize] [reverse] [source]
./bfs_linear_base mtx ./data/kron_g500-logn21 0 0 1930586
./bfs_linear_base mtx ./data/kron_g500-logn21 0 0 12812
# BC: <filetype> <prefix> [symmetrize] [reverse] [source]
./bc_linear_base mtx ./data/kron_g500-logn21 0 0 1930586
./bc_linear_base mtx ./data/kron_g500-logn21 0 0 12812
```

通俗例子（拿这两行解释）：

- “首选 hub：`1930586`” 就是：把 `source_id` 设为 `1930586`，从一个连接很多边的“大点”出发，通常更容易把 IMA 压力打满。
- “备选 p90：`12812`” 就是：把 `source_id` 设为 `12812`，从一个“也很活跃但没那么极端”的点出发，往往更稳更省时。

### roadNet-CA

- 文件：`roadNet-CA/roadNet-CA.mtx`
- |V| = 1971281

**Out-degree（按 MTX 方向）**

| 指标 | 值 |
|---|---|
| avg out-degree | 1.403 |
| zero out-degree vertices | 171478 |
| hub (max_degree) | deg=6, v=117563 |
| p99 (non-zero) | deg≈4, v=176 |
| p90 (non-zero) | deg≈3, v=84 |
| median (non-zero) | deg≈1, v=1 |

**Symmetrized degree（把图当无向）**

| 指标 | 值 |
|---|---|
| avg degree | 2.807 |
| zero degree vertices | 6075 |
| hub (max_degree) | deg=12, v=562818 |
| p99 (non-zero) | deg≈4, v=3 |
| p90 (non-zero) | deg≈4, v=3 |
| median (non-zero) | deg≈3, v=0 |

**推荐（用于 IMA 压力）**

- 首选 hub（度最高的点）：`117563`（更激进，压力更大）
- 备选 p90（前 10% 高度点的代表）：`84`（更稳，通常更省时）
- 路网建议启用 symmetrize=1，并用 hub：`562818` 或 p90：`3`

**第二阶段：邻居局部性检查（防反例）**

Locality (out)：（按 MTX 方向统计 source 的 out-neighbors）

| candidate | source_id | unique_neighbors | bfs2_frontier2 | unique_lines | neighbors_per_line | line_density | neighbor_span |
|---|---:|---:|---:|---:|---:|---:|---:|
| hub (max_degree) |  `117563` | 6 | 7 | 3 | 2.00 | 0.001 | 114042 |
| p99 (non-zero) |  `176` | 4 | 1 | 2 | 2.00 | 1.000 | 30 |
| p90 (non-zero) |  `84` | 3 | 2 | 2 | 1.50 | 0.667 | 74 |

解读（这张表说明什么）：

- `hub` (`117563`)：邻居分散程度中等。 2 跳扩散不大，压力持续性一般。（但该点度本身很小，单轮访问量有限）（neighbor_span≈0.06*|V|）
- `p99` (`176`)：邻居分散程度中等。 2 跳扩散不大，压力持续性一般。（但该点度本身很小，单轮访问量有限）（neighbor_span≈0.00*|V|）
- `p90` (`84`)：邻居分散程度中等。 2 跳扩散不大，压力持续性一般。（但该点度本身很小，单轮访问量有限）（neighbor_span≈0.00*|V|）

Locality (sym)：（把边当无向统计 source 的 neighbors；对应运行时 symmetrize=1 的直觉）

| candidate | source_id | unique_neighbors | bfs2_frontier2 | unique_lines | neighbors_per_line | line_density | neighbor_span |
|---|---:|---:|---:|---:|---:|---:|---:|
| hub (max_degree) |  `562818` | 12 | 15 | 3 | 4.00 | 1.000 | 64 |
| p99 (non-zero) |  `3` | 4 | 8 | 2 | 2.00 | 0.143 | 420 |
| p90 (non-zero) |  `3` | 4 | 8 | 2 | 2.00 | 0.143 | 420 |

解读（sym 这张表说明什么）：

- `hub` (`562818`)：邻居编号更集中（局部性更好），IMA 可能不算重。 2 跳扩散不大，压力持续性一般。（但该点度本身很小，单轮访问量有限）（neighbor_span≈0.00*|V|）
- `p99` (`3`)：邻居分散程度中等。 2 跳扩散不大，压力持续性一般。（但该点度本身很小，单轮访问量有限）（neighbor_span≈0.00*|V|）
- `p90` (`3`)：邻居分散程度中等。 2 跳扩散不大，压力持续性一般。（但该点度本身很小，单轮访问量有限）（neighbor_span≈0.00*|V|）

建议命令行（按 Gardenia 参数约定；以 `./data/{name}` 为前缀举例）：

```bash
# SSSP: <filetype> <prefix> [symmetrize] [reverse] [source] [delta]
./sssp_linear_base mtx ./data/roadNet-CA 0 0 117563 1
./sssp_linear_base mtx ./data/roadNet-CA 0 0 84 1
# BFS: <filetype> <prefix> [symmetrize] [reverse] [source]
./bfs_linear_base mtx ./data/roadNet-CA 0 0 117563
./bfs_linear_base mtx ./data/roadNet-CA 0 0 84
# BC: <filetype> <prefix> [symmetrize] [reverse] [source]
./bc_linear_base mtx ./data/roadNet-CA 0 0 117563
./bc_linear_base mtx ./data/roadNet-CA 0 0 84
```

通俗例子（拿这两行解释）：

- “首选 hub：`117563`” 就是：把 `source_id` 设为 `117563`，从一个连接很多边的“大点”出发，通常更容易把 IMA 压力打满。
- “备选 p90：`84`” 就是：把 `source_id` 设为 `84`，从一个“也很活跃但没那么极端”的点出发，往往更稳更省时。

### road_usa

- 文件：`road_usa/road_usa.mtx`
- |V| = 23947347

**Out-degree（按 MTX 方向）**

| 指标 | 值 |
|---|---|
| avg out-degree | 1.205 |
| zero out-degree vertices | 6392288 |
| hub (max_degree) | deg=8, v=17644255 |
| p99 (non-zero) | deg≈4, v=47945 |
| p90 (non-zero) | deg≈3, v=1270 |
| median (non-zero) | deg≈1, v=1 |

**Symmetrized degree（把图当无向）**

| 指标 | 值 |
|---|---|
| avg degree | 2.410 |
| zero degree vertices | 0 |
| hub (max_degree) | deg=9, v=18944626 |
| p99 (non-zero) | deg≈4, v=28 |
| p90 (non-zero) | deg≈4, v=28 |
| median (non-zero) | deg≈2, v=0 |

**推荐（用于 IMA 压力）**

- 首选 hub（度最高的点）：`17644255`（更激进，压力更大）
- 备选 p90（前 10% 高度点的代表）：`1270`（更稳，通常更省时）
- 路网建议启用 symmetrize=1，并用 hub：`18944626` 或 p90：`28`

**第二阶段：邻居局部性检查（防反例）**

Locality (out)：（按 MTX 方向统计 source 的 out-neighbors）

| candidate | source_id | unique_neighbors | bfs2_frontier2 | unique_lines | neighbors_per_line | line_density | neighbor_span |
|---|---:|---:|---:|---:|---:|---:|---:|
| hub (max_degree) |  `17644255` | 8 | 3 | 6 | 1.33 | 0.000 | 1944381 |
| p99 (non-zero) |  `47945` | 4 | 2 | 3 | 1.33 | 0.750 | 87 |
| p90 (non-zero) |  `1270` | 3 | 2 | 1 | 3.00 | 1.000 | 11 |

解读（这张表说明什么）：

- `hub` (`17644255`)：邻居分散程度中等。 2 跳扩散不大，压力持续性一般。（但该点度本身很小，单轮访问量有限）（neighbor_span≈0.08*|V|）
- `p99` (`47945`)：邻居分散程度中等。 2 跳扩散不大，压力持续性一般。（但该点度本身很小，单轮访问量有限）（neighbor_span≈0.00*|V|）
- `p90` (`1270`)：邻居编号更集中（局部性更好），IMA 可能不算重。 2 跳扩散不大，压力持续性一般。（但该点度本身很小，单轮访问量有限）（neighbor_span≈0.00*|V|）

Locality (sym)：（把边当无向统计 source 的 neighbors；对应运行时 symmetrize=1 的直觉）

| candidate | source_id | unique_neighbors | bfs2_frontier2 | unique_lines | neighbors_per_line | line_density | neighbor_span |
|---|---:|---:|---:|---:|---:|---:|---:|
| hub (max_degree) |  `18944626` | 9 | 10 | 8 | 1.12 | 0.000 | 2242466 |
| p99 (non-zero) |  `28` | 4 | 3 | 4 | 1.00 | 0.000 | 2097136 |
| p90 (non-zero) |  `28` | 4 | 3 | 4 | 1.00 | 0.000 | 2097136 |

解读（sym 这张表说明什么）：

- `hub` (`18944626`)：邻居非常分散（接近“一邻居一条 cache line”），更容易触发随机访存/IMA。 2 跳扩散不大，压力持续性一般。（但该点度本身很小，单轮访问量有限）（neighbor_span≈0.09*|V|）
- `p99` (`28`)：邻居非常分散（接近“一邻居一条 cache line”），更容易触发随机访存/IMA。 2 跳扩散不大，压力持续性一般。（但该点度本身很小，单轮访问量有限）（neighbor_span≈0.09*|V|）
- `p90` (`28`)：邻居非常分散（接近“一邻居一条 cache line”），更容易触发随机访存/IMA。 2 跳扩散不大，压力持续性一般。（但该点度本身很小，单轮访问量有限）（neighbor_span≈0.09*|V|）

建议命令行（按 Gardenia 参数约定；以 `./data/{name}` 为前缀举例）：

```bash
# SSSP: <filetype> <prefix> [symmetrize] [reverse] [source] [delta]
./sssp_linear_base mtx ./data/road_usa 0 0 17644255 1
./sssp_linear_base mtx ./data/road_usa 0 0 1270 1
# BFS: <filetype> <prefix> [symmetrize] [reverse] [source]
./bfs_linear_base mtx ./data/road_usa 0 0 17644255
./bfs_linear_base mtx ./data/road_usa 0 0 1270
# BC: <filetype> <prefix> [symmetrize] [reverse] [source]
./bc_linear_base mtx ./data/road_usa 0 0 17644255
./bc_linear_base mtx ./data/road_usa 0 0 1270
```

通俗例子（拿这两行解释）：

- “首选 hub：`17644255`” 就是：把 `source_id` 设为 `17644255`，从一个连接很多边的“大点”出发，通常更容易把 IMA 压力打满。
- “备选 p90：`1270`” 就是：把 `source_id` 设为 `1270`，从一个“也很活跃但没那么极端”的点出发，往往更稳更省时。

### soc-LiveJournal1

- 文件：`soc-LiveJournal1/soc-LiveJournal1.mtx`
- |V| = 4847571

**Out-degree（按 MTX 方向）**

| 指标 | 值 |
|---|---|
| avg out-degree | 14.126 |
| zero out-degree vertices | 553239 |
| hub (max_degree) | deg=20292, v=10009 |
| p99 (non-zero) | deg≈151, v=1 |
| p90 (non-zero) | deg≈39, v=0 |
| median (non-zero) | deg≈5, v=0 |

**Symmetrized degree（把图当无向）**

| 指标 | 值 |
|---|---|
| avg degree | 28.251 |
| zero degree vertices | 962 |
| hub (max_degree) | deg=22887, v=10009 |
| p99 (non-zero) | deg≈283, v=1 |
| p90 (non-zero) | deg≈70, v=0 |
| median (non-zero) | deg≈8, v=0 |

**推荐（用于 IMA 压力）**

- 首选 hub（度最高的点）：`10009`（更激进，压力更大）
- 备选 p90（前 10% 高度点的代表）：`0`（更稳，通常更省时）

**第二阶段：邻居局部性检查（防反例）**

Locality (out)：（按 MTX 方向统计 source 的 out-neighbors）

| candidate | source_id | unique_neighbors | bfs2_frontier2 | unique_lines | neighbors_per_line | line_density | neighbor_span |
|---|---:|---:|---:|---:|---:|---:|---:|
| hub (max_degree) |  `10009` | 20292 | 186879 | 9864 | 2.06 | 0.066 | 4785574 |
| p99 (non-zero) |  `1` | 209 | 16791 | 186 | 1.12 | 0.004 | 1413125 |
| p90 (non-zero) |  `0` | 46 | 2155 | 2 | 23.00 | 1.000 | 45 |

解读（这张表说明什么）：

- `hub` (`10009`)：邻居分散程度中等。 2 跳扩散较大，更可能形成多轮 frontier 压力。（neighbor_span≈0.99*|V|）
- `p99` (`1`)：邻居非常分散（接近“一邻居一条 cache line”），更容易触发随机访存/IMA。 2 跳扩散较大，更可能形成多轮 frontier 压力。（neighbor_span≈0.29*|V|）
- `p90` (`0`)：邻居编号更集中（局部性更好），IMA 可能不算重。 2 跳扩散较大，更可能形成多轮 frontier 压力。（neighbor_span≈0.00*|V|）

建议命令行（按 Gardenia 参数约定；以 `./data/{name}` 为前缀举例）：

```bash
# SSSP: <filetype> <prefix> [symmetrize] [reverse] [source] [delta]
./sssp_linear_base mtx ./data/soc-LiveJournal1 0 0 10009 1
./sssp_linear_base mtx ./data/soc-LiveJournal1 0 0 0 1
# BFS: <filetype> <prefix> [symmetrize] [reverse] [source]
./bfs_linear_base mtx ./data/soc-LiveJournal1 0 0 10009
./bfs_linear_base mtx ./data/soc-LiveJournal1 0 0 0
# BC: <filetype> <prefix> [symmetrize] [reverse] [source]
./bc_linear_base mtx ./data/soc-LiveJournal1 0 0 10009
./bc_linear_base mtx ./data/soc-LiveJournal1 0 0 0
```

通俗例子（拿这两行解释）：

- “首选 hub：`10009`” 就是：把 `source_id` 设为 `10009`，从一个连接很多边的“大点”出发，通常更容易把 IMA 压力打满。
- “备选 p90：`0`” 就是：把 `source_id` 设为 `0`，从一个“也很活跃但没那么极端”的点出发，往往更稳更省时。

### soc-orkut

- 文件：`soc-orkut/soc-orkut.mtx`
- |V| = 2997166

**Out-degree（按 MTX 方向）**

| 指标 | 值 |
|---|---|
| avg out-degree | 35.483 |
| zero out-degree vertices | 1005 |
| hub (max_degree) | deg=3136, v=376893 |
| p99 (non-zero) | deg≈225, v=38040 |
| p90 (non-zero) | deg≈78, v=403 |
| median (non-zero) | deg≈22, v=302 |

**Symmetrized degree（把图当无向）**

| 指标 | 值 |
|---|---|
| avg degree | 70.966 |
| zero degree vertices | 0 |
| hub (max_degree) | deg=27466, v=42983 |
| p99 (non-zero) | deg≈488, v=268 |
| p90 (non-zero) | deg≈152, v=19 |
| median (non-zero) | deg≈42, v=1 |

**推荐（用于 IMA 压力）**

- 首选 hub（度最高的点）：`376893`（更激进，压力更大）
- 备选 p90（前 10% 高度点的代表）：`403`（更稳，通常更省时）

**第二阶段：邻居局部性检查（防反例）**

Locality (out)：（按 MTX 方向统计 source 的 out-neighbors）

| candidate | source_id | unique_neighbors | bfs2_frontier2 | unique_lines | neighbors_per_line | line_density | neighbor_span |
|---|---:|---:|---:|---:|---:|---:|---:|
| hub (max_degree) |  `376893` | 3136 | 8305 | 572 | 5.48 | 0.050 | 368672 |
| p99 (non-zero) |  `38040` | 233 | 2979 | 100 | 2.33 | 0.085 | 37731 |
| p90 (non-zero) |  `403` | 82 | 11 | 6 | 13.67 | 0.462 | 374 |

解读（这张表说明什么）：

- `hub` (`376893`)：邻居分散程度中等。 2 跳扩散较大，更可能形成多轮 frontier 压力。（neighbor_span≈0.12*|V|）
- `p99` (`38040`)：邻居分散程度中等。 2 跳扩散较大，更可能形成多轮 frontier 压力。（neighbor_span≈0.01*|V|）
- `p90` (`403`)：邻居编号更集中（局部性更好），IMA 可能不算重。 2 跳扩散不大，压力持续性一般。（neighbor_span≈0.00*|V|）

建议命令行（按 Gardenia 参数约定；以 `./data/{name}` 为前缀举例）：

```bash
# SSSP: <filetype> <prefix> [symmetrize] [reverse] [source] [delta]
./sssp_linear_base mtx ./data/soc-orkut 0 0 376893 1
./sssp_linear_base mtx ./data/soc-orkut 0 0 403 1
# BFS: <filetype> <prefix> [symmetrize] [reverse] [source]
./bfs_linear_base mtx ./data/soc-orkut 0 0 376893
./bfs_linear_base mtx ./data/soc-orkut 0 0 403
# BC: <filetype> <prefix> [symmetrize] [reverse] [source]
./bc_linear_base mtx ./data/soc-orkut 0 0 376893
./bc_linear_base mtx ./data/soc-orkut 0 0 403
```

通俗例子（拿这两行解释）：

- “首选 hub：`376893`” 就是：把 `source_id` 设为 `376893`，从一个连接很多边的“大点”出发，通常更容易把 IMA 压力打满。
- “备选 p90：`403`” 就是：把 `source_id` 设为 `403`，从一个“也很活跃但没那么极端”的点出发，往往更稳更省时。

### web-Google

- 文件：`web-Google/web-Google.mtx`
- |V| = 916428

**Out-degree（按 MTX 方向）**

| 指标 | 值 |
|---|---|
| avg out-degree | 5.571 |
| zero out-degree vertices | 176974 |
| hub (max_degree) | deg=456, v=506742 |
| p99 (non-zero) | deg≈25, v=211 |
| p90 (non-zero) | deg≈15, v=7 |
| median (non-zero) | deg≈5, v=1 |

**Symmetrized degree（把图当无向）**

| 指标 | 值 |
|---|---|
| avg degree | 11.141 |
| zero degree vertices | 40715 |
| hub (max_degree) | deg=6353, v=537039 |
| p99 (non-zero) | deg≈78, v=0 |
| p90 (non-zero) | deg≈25, v=0 |
| median (non-zero) | deg≈6, v=0 |

**推荐（用于 IMA 压力）**

- 首选 hub（度最高的点）：`506742`（更激进，压力更大）
- 备选 p90（前 10% 高度点的代表）：`7`（更稳，通常更省时）

**第二阶段：邻居局部性检查（防反例）**

Locality (out)：（按 MTX 方向统计 source 的 out-neighbors）

| candidate | source_id | unique_neighbors | bfs2_frontier2 | unique_lines | neighbors_per_line | line_density | neighbor_span |
|---|---:|---:|---:|---:|---:|---:|---:|
| hub (max_degree) |  `506742` | 456 | 1903 | 448 | 1.02 | 0.016 | 908607 |
| p99 (non-zero) |  `211` | 26 | 160 | 26 | 1.00 | 0.001 | 882018 |
| p90 (non-zero) |  `7` | 16 | 54 | 16 | 1.00 | 0.001 | 857465 |

解读（这张表说明什么）：

- `hub` (`506742`)：邻居非常分散（接近“一邻居一条 cache line”），更容易触发随机访存/IMA。 2 跳扩散较大，更可能形成多轮 frontier 压力。（neighbor_span≈0.99*|V|）
- `p99` (`211`)：邻居非常分散（接近“一邻居一条 cache line”），更容易触发随机访存/IMA。 2 跳扩散较大，更可能形成多轮 frontier 压力。（但该点度本身很小，单轮访问量有限）（neighbor_span≈0.96*|V|）
- `p90` (`7`)：邻居非常分散（接近“一邻居一条 cache line”），更容易触发随机访存/IMA。 2 跳扩散不大，压力持续性一般。（但该点度本身很小，单轮访问量有限）（neighbor_span≈0.94*|V|）

建议命令行（按 Gardenia 参数约定；以 `./data/{name}` 为前缀举例）：

```bash
# SSSP: <filetype> <prefix> [symmetrize] [reverse] [source] [delta]
./sssp_linear_base mtx ./data/web-Google 0 0 506742 1
./sssp_linear_base mtx ./data/web-Google 0 0 7 1
# BFS: <filetype> <prefix> [symmetrize] [reverse] [source]
./bfs_linear_base mtx ./data/web-Google 0 0 506742
./bfs_linear_base mtx ./data/web-Google 0 0 7
# BC: <filetype> <prefix> [symmetrize] [reverse] [source]
./bc_linear_base mtx ./data/web-Google 0 0 506742
./bc_linear_base mtx ./data/web-Google 0 0 7
```

通俗例子（拿这两行解释）：

- “首选 hub：`506742`” 就是：把 `source_id` 设为 `506742`，从一个连接很多边的“大点”出发，通常更容易把 IMA 压力打满。
- “备选 p90：`7`” 就是：把 `source_id` 设为 `7`，从一个“也很活跃但没那么极端”的点出发，往往更稳更省时。

