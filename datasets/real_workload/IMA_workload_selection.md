# IMA 工作负载（Gardenia）输入选择：bfs / spmv / sssp / bc / tc / cc

目标：为了评估/优化 **L1/L2 cache 预取**在图负载中的 **IMA（Indirect Memory Access）**问题，给每个算法挑一个“尽量不规则、间接访问压力大”的 **(dataset, 参数)** 组合，并尽量避免“workload 太短/太小看不出效果”。

候选数据集：`web-Google`、`cit-Patents`、`flickr`（均在本目录下）。

> 说明：BFS/SSSP/BC 的 `source_id` 是 **0-based**（与 `IMA_source_selection.md` 保持一致）；`symmetrize=1` 表示把图当无向（读入时补齐反向边），通常能显著提高可达性/工作量。

---

## 评价口径（用于 BFS/SSSP/BC 的 source 类算法）

source 扩散类算法的核心 IMA 往往来自 `dist[neighbor]` / `visited[neighbor]` 这类“**用邻居 id 做索引**”的间接访问。这里用 `tools/mtx_source_locality --bfs2` 统计的指标做选择：

- `neighbors_per_line` 越接近 `1`：越接近“一邻居一条 cache line”，越 IMA。
- `line_density` 越小：邻居 id 覆盖的 cache line 越稀疏（更随机）。
- `neighbor_span` 越大：邻居 id 跨度越大（更随机）。
- `bfs2_frontier2` 越大：前两跳扩散越快，更容易形成多轮 frontier（避免 workload 过短）。

本文件里的对比数据来自（默认按 128B line、4B 元素 ⇒ `line_elems=32`）：

```bash
gpu-app-collection/gardenia/tools/mtx_source_locality --bfs2 --symmetrize <file.mtx> <source_id>
```

---

## 推荐配置（两档：最大压力 + 更友善但仍 IMA 明显）

下面的命令行以 `gpu-app-collection/gardenia/bin/` 为工作目录、并使用 `./data/<name>` 这种前缀（与 `util/job_launching/apps/define-all-apps.yml` 的写法一致）。

---

## 为什么大图/强扩散会让 trace “又慢又大”

你遇到的“仿真时间很长、CSV 巨大”通常来自两层叠加：

- **算法工作量变大**：更大的图、更强可达性（`symmetrize=1`）、更强扩散的 source（hub）会显著增加活跃 warp/访存事件与 kernel cycles。
- **trace 输出开销**：尤其是 `issue_trace`，它接近“每 cycle × 每 SM × 每 scheduler 一行”，体量几乎随 cycles 线性增长；当 cycles 很大时，写盘 I/O 会把仿真拖得更慢。

因此：如果你必须保留 issue/L1/L2 trace 做后处理，实践上更稳的策略是：
先跑 **更友善但仍 IMA 很强**的配置把 pipeline 跑通与做趋势验证，再切到 **最大压力**配置做最终对比。

---

## A) IMA-Stress（最大 IMA 压力；耗时/体量最大）

这组用于“把 IMA 压力拉满”，适合你最终验证 prefetch 上限，但很容易导致超大 trace。

### bfs（BFS）

**Stress：`cit-Patents` + `symmetrize=1` + `source_id=3569341`**

```bash
./bfs_linear_base mtx ./data/cit-Patents 1 0 3569341
```

### sssp（SSSP）

**Stress：`cit-Patents` + `symmetrize=1` + `source_id=3569341` + `delta=1`**

```bash
./sssp_linear_base mtx ./data/cit-Patents 1 0 3569341 1
```

### bc（BC）

**Stress：`cit-Patents` + `symmetrize=1` + `source_id=3569341`**

```bash
./bc_linear_base mtx ./data/cit-Patents 1 0 3569341
```

### spmv（SpMV）

**Stress：`cit-Patents` + `symmetrize=0`**（SpMV 的 IMA 主要是 `x[col_idx]` gather；保持有向能减少边数，较 `symmetrize=1` 更“可控”）

```bash
./spmv_base mtx ./data/cit-Patents 0 0
```

### cc（Connected Components）

**Stress：`cit-Patents` + `symmetrize=1`**（CC 语义上更贴近无向）

```bash
./cc_base mtx ./data/cit-Patents 1 0
```

### tc（Triangle Counting）

**Stress：`flickr`**（三角/交集更典型，但通常更慢）

```bash
./tc_gpu_base ./data/flickr
```

---

## B) IMA-Friendly（更友善：控制 runtime/体量；但 IMA 仍明显）

这组专门针对你现在的痛点（`issue_trace` 不能关），通过“更小/更 IMA 的图特征 + 更温和的参数”控制运行时间，同时避免选到过于轻量导致 IMA 不明显。

> 重点建议优先替换 **BC / SSSP / CC / TC**；BFS/SpMV 如果你现有 trace 体量还能接受，可以先不动。

### bc / sssp（重点）

**Friendly：`web-Google` + `symmetrize=0` + `source_id=506742`（hub，但图更小且保持有向）**

- 选 `web-Google` 而不是 `flickr`：`web-Google` 的邻居 id 分布更接近“一邻居一条 cache line”（`neighbors_per_line≈1`），IMA 更典型；而 `flickr` 在 BFS/SSSP/BC 这类算法上邻居编号更集中（`neighbors_per_line` 显著大于 1），IMA 反而容易变轻，同时边数也更大。

```bash
./bc_linear_base   mtx ./data/web-Google 0 0 506742
./sssp_linear_base mtx ./data/web-Google 0 0 506742 1
```

如果你觉得 hub 仍然太重，可以把 source 降到 `p99=211`（更温和，但仍保持 `neighbor_span` 很大，IMA 仍明显）：

```bash
./bc_linear_base   mtx ./data/web-Google 0 0 211
./sssp_linear_base mtx ./data/web-Google 0 0 211 1
```

### cc（重点）

**Friendly：`web-Google` + `symmetrize=1`**

```bash
./cc_base mtx ./data/web-Google 1 0
```

### tc（重点）

**Friendly：`web-Google`**（优先把运行时间/体量控制住；需要更强三角/交集压力时再切回 `flickr`）

```bash
./tc_gpu_base ./data/web-Google
```

### bfs / spmv（可选：你现有体量可接受就先不动）

如果你也需要把 BFS/SpMV 变得更友善但不“太轻”：

```bash
./bfs_linear_base mtx ./data/web-Google 0 0 506742
./spmv_base       mtx ./data/web-Google 0 0
```

---

## C) 仍用 `cit-Patents` 但更友善（回答你的问题：可以，但要注意“过轻”风险）

你完全可以继续用 `cit-Patents`，用两种方式降负载：

1) **保持有向图（`symmetrize=0`）**：边数更少、可达性更差，BFS/SSSP/BC 往往更快结束，但 IMA（邻居跨度大）仍然明显。
2) **换更温和的 source**：但对 `cit-Patents` 来说，`p99/p90` 在 out 模式下经常出现 `bfs2_frontier2=0`（过轻，可能一两轮就结束），所以更建议先从 **hub + symmetrize=0** 这条路开始。

示例（BC/SSSP）：

```bash
./bc_linear_base   mtx ./data/cit-Patents 0 0 3569341
./sssp_linear_base mtx ./data/cit-Patents 0 0 3569341 1
```

如果你必须用 `symmetrize=1`（无向）但又想比 hub 更友善，可以换成更温和的 source（`mtx_source_locality --bfs2 --symmetrize` 下的 `bfs2_frontier2` 明显更小）：

- `source=1654613`（p90，`bfs2_frontier2≈250`）
- `source=1692580`（p99，`bfs2_frontier2≈142`）

```bash
./bc_linear_base   mtx ./data/cit-Patents 1 0 1654613
./sssp_linear_base mtx ./data/cit-Patents 1 0 1654613 1
```
