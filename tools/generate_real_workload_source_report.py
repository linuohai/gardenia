#!/usr/bin/env python3
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class StatsRow:
    file: str
    m: int
    avg_degree: float
    max_degree: int
    max_vertex: int
    zero_vertices: int
    median_nz_deg: int
    median_vertex: int
    p90_nz_deg: int
    p90_vertex: int
    p99_nz_deg: int
    p99_vertex: int


@dataclass(frozen=True)
class LocalityRow:
    mode: str
    file: str
    source: int
    raw_neighbors: int
    unique_neighbors: int
    min_neighbor: int
    max_neighbor: int
    neighbor_span: int
    unique_lines: int
    line_span: int
    line_density: float
    neighbors_per_line: float
    lines_per_neighbor: float
    mean_gap: float
    median_gap: int
    p90_gap: int
    bfs2_frontier2: int


def repo_root_from_this_file() -> Path:
    # Returns the accel-sim-framework directory (the one containing gpu-app-collection/, util/, ...).
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "gpu-app-collection").exists() and (parent / "gpu-simulator").exists():
            return parent
    raise RuntimeError("Failed to locate repo root")


def build_degree_tool(repo_root: Path) -> Path:
    src = repo_root / "gpu-app-collection/gardenia/tools/mtx_degree_stats.cc"
    out = Path(os.environ.get("TMPDIR", "/tmp")) / f"mtx_degree_stats_{os.getpid()}"
    cmd = ["g++", "-O3", "-std=c++11", str(src), "-o", str(out)]
    subprocess.check_call(cmd)
    return out


def build_locality_tool(repo_root: Path) -> Path:
    src = repo_root / "gpu-app-collection/gardenia/tools/mtx_source_locality.cc"
    out = Path(os.environ.get("TMPDIR", "/tmp")) / f"mtx_source_locality_{os.getpid()}"
    cmd = ["g++", "-O3", "-std=c++11", str(src), "-o", str(out)]
    subprocess.check_call(cmd)
    return out


def list_real_workload_mtx(repo_root: Path) -> List[Path]:
    rw = repo_root / "gpu-app-collection/gardenia/datasets/real_workload"
    mtx_files: List[Path] = []
    for child in sorted(rw.iterdir()):
        if not child.is_dir():
            continue
        # Prefer a single *.mtx per subdir.
        found = sorted(child.glob("*.mtx"))
        if not found:
            continue
        # Exclude auxiliary nodename files.
        found = [p for p in found if not p.name.endswith("_nodename.mtx")]
        if not found:
            continue
        # If multiple remain, pick the one matching directory name if present.
        exact = [p for p in found if p.stem == child.name]
        mtx_files.append(exact[0] if exact else found[0])
    return mtx_files


def run_tool(tool: Path, files: List[Path], symmetrize: bool) -> Dict[str, StatsRow]:
    cmd = [str(tool)]
    if symmetrize:
        cmd.append("--symmetrize")
    cmd.extend(str(p) for p in files)
    out = subprocess.check_output(cmd, text=True)
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if len(lines) < 2 or not lines[1].startswith("file,"):
        raise RuntimeError("Unexpected tool output")
    reader = csv.DictReader(lines[1:])
    rows: Dict[str, StatsRow] = {}
    for r in reader:
        row = StatsRow(
            file=r["file"],
            m=int(r["m"]),
            avg_degree=float(r["avg_degree"]),
            max_degree=int(r["max_degree"]),
            max_vertex=int(r["max_vertex"]),
            zero_vertices=int(r["zero_vertices"]),
            median_nz_deg=int(r["median_nz_deg"]),
            median_vertex=int(r["median_vertex"]),
            p90_nz_deg=int(r["p90_nz_deg"]),
            p90_vertex=int(r["p90_vertex"]),
            p99_nz_deg=int(r["p99_nz_deg"]),
            p99_vertex=int(r["p99_vertex"]),
        )
        rows[row.file] = row
    return rows


def run_locality_tool(
    tool: Path,
    file_path: Path,
    sources: List[int],
    symmetrize: bool,
    line_elems: int = 32,
) -> Dict[int, LocalityRow]:
    cmd = [str(tool), "--bfs2"]
    if symmetrize:
        cmd.append("--symmetrize")
    cmd.extend(["--line-elems", str(line_elems), str(file_path)])
    cmd.extend(str(s) for s in sources)
    out = subprocess.check_output(cmd, text=True)
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if not lines or not lines[0].startswith("mode,"):
        raise RuntimeError("Unexpected locality tool output")
    reader = csv.DictReader(lines)
    rows: Dict[int, LocalityRow] = {}
    for r in reader:
        row = LocalityRow(
            mode=r["mode"],
            file=r["file"],
            source=int(r["source"]),
            raw_neighbors=int(r["raw_neighbors"]),
            unique_neighbors=int(r["unique_neighbors"]),
            min_neighbor=int(r["min_neighbor"]),
            max_neighbor=int(r["max_neighbor"]),
            neighbor_span=int(r["neighbor_span"]),
            unique_lines=int(r["unique_lines"]),
            line_span=int(r["line_span"]),
            line_density=float(r["line_density"]),
            neighbors_per_line=float(r["neighbors_per_line"]),
            lines_per_neighbor=float(r["lines_per_neighbor"]),
            mean_gap=float(r["mean_gap"]),
            median_gap=int(r["median_gap"]),
            p90_gap=int(r["p90_gap"]),
            bfs2_frontier2=int(r.get("bfs2_frontier2", "0")),
        )
        rows[row.source] = row
    return rows


def dataset_name_from_mtx_path(path: Path) -> str:
    # .../real_workload/<name>/<name>.mtx
    return path.parent.name


def write_report(
    repo_root: Path,
    files: List[Path],
    out_rows: Dict[str, StatsRow],
    sym_rows: Dict[str, StatsRow],
    locality_tool: Path,
) -> Path:
    rw = repo_root / "gpu-app-collection/gardenia/datasets/real_workload"
    out_path = rw / "IMA_source_selection.md"

    def fmt_float(x: float) -> str:
        return f"{x:.3f}"

    def rel_to_real_workload(p: Path) -> str:
        try:
            return str(p.relative_to(rw))
        except Exception:
            return str(p)

    lines: List[str] = []
    lines.append("# Gardenia real_workload 源点选择（偏 IMA 压力）")
    lines.append("")
    lines.append("这份报告用于给 **SSSP/BFS/BC** 这类“从 source 出发扩散”的图算法选 `source_id`，目标是让访存更不规则、frontier 更大，从而更容易体现 IMA/L1 压力。")
    lines.append("")
    lines.append("## 方法（可复现）")
    lines.append("")
    lines.append("- 对每个 `*.mtx` 统计 **out-degree**（按 MTX 原方向），以及 **symmetrized degree**（把每条边同时记到两端，等价于把图当无向看待的度）。")
    lines.append("- 计算 non-zero 度的 `median/p90/p99`，并给出一个“度 >= 阈值”的代表顶点；同时给出最大度顶点（hub）。")
    lines.append("- 说明：这里不做“去重边”的精确处理（`csr_graph.h` 会去重），所以度数是偏上界；用于挑 source 做压力测试一般够用。")
    lines.append("")
    lines.append("### 你提到的“反例/限制”是否考虑？（回答：目前未显式考虑）")
    lines.append("")
    lines.append("这份报告目前只用 **度（degree）** 做第一阶段筛选，确实没有显式计算下列因素：")
    lines.append("")
    lines.append("- **邻居是否“局部”**：比如某个大度点的邻居编号非常集中/连续，访问 `dist[neighbor]` 时可能更容易落在同一两个 cache line，IMA 会变轻。")
    lines.append("- **实现是否有缓解手段**：例如 Gardenia 的 CSR 读入会对邻接表排序/去重，这会让读 `column_indices` 更顺序；但对 `dist[dst]` 这种间接访问是否变好，仍取决于 `dst` 编号是否集中。")
    lines.append("- **真正决定 IMA 的其它图特征**：方向性/可达性、社团结构、重复访问、原子更新热点、frontier 的增长曲线等。")
    lines.append("")
    lines.append("为什么仍然先用度？因为它是**最便宜、最稳的“压力放大器”**：在多数真实图里，顶点编号并不刻意按社区连续排列，高度点往往仍会带来更分散的 neighbor id，从而更容易触发随机访存。")
    lines.append("")
    lines.append("如果你希望把这些反例也纳入筛选，我建议做“第二阶段”过滤（下一步可做）：")
    lines.append("")
    lines.append("- 对候选 source（比如 hub/p90/p99）再统计：邻居 id 的跨度（max-min）、相邻邻居 id 的 gap 分布、以及 `neighbor_id // 32` 的唯一值个数（估计触及多少个 128B cache line，越多越随机）。")
    lines.append("- 再做一个很浅的采样 BFS（比如只跑前 2~3 层）估计 frontier 是否会快速爆炸，避免选到“度大但不可达/扩散慢”的点。")
    lines.append("")
    lines.append("本次更新已经把“第二阶段”的**邻居局部性**统计加进来了（见每个数据集的 `Locality (out)` 表）。")
    lines.append("如何读这个表（都越大越不局部/越随机）：")
    lines.append("")
    lines.append("- `unique_lines`：邻居落在多少个不同的 cache line（默认假设 128B line、4B 元素，所以一条 line 有 32 个元素）。越大通常越随机。")
    lines.append("- `neighbors_per_line`：平均每个 cache line 上有多少个邻居，越大说明邻居更“挤在一起”（更局部，IMA 可能更轻）。")
    lines.append("- `neighbor_span`：邻居 id 的最大最小跨度，越大说明编号更分散（通常更随机）。")
    lines.append("- `line_density`：`unique_lines / line_span`，越接近 1 说明邻居覆盖的 cache line 越连续；越小说明越稀疏/分散。")
    lines.append("- `bfs2_frontier2`：从 source 出发走 2 跳，能新到达多少个不同顶点（很粗的“早期扩散速度”指标）。越大通常越容易很快把 frontier 做大。")
    lines.append("")
    lines.append("复现命令（在 repo 根目录）：")
    lines.append("")
    lines.append("```bash")
    lines.append("g++ -O3 -std=c++11 gpu-app-collection/gardenia/tools/mtx_degree_stats.cc \\")
    lines.append("  -o gpu-app-collection/gardenia/tools/mtx_degree_stats")
    lines.append("gpu-app-collection/gardenia/tools/mtx_degree_stats <file.mtx...>")
    lines.append("gpu-app-collection/gardenia/tools/mtx_degree_stats --symmetrize <file.mtx...>")
    lines.append("")
    lines.append("g++ -O3 -std=c++11 gpu-app-collection/gardenia/tools/mtx_source_locality.cc \\")
    lines.append("  -o gpu-app-collection/gardenia/tools/mtx_source_locality")
    lines.append("# out-neighbors 的局部性 + 2-hop 扩散（--bfs2）")
    lines.append("gpu-app-collection/gardenia/tools/mtx_source_locality --bfs2 <file.mtx> <source0> [source1...]")
    lines.append("# sym neighbors（把边当无向）")
    lines.append("gpu-app-collection/gardenia/tools/mtx_source_locality --bfs2 --symmetrize <file.mtx> <source0> [source1...]")
    lines.append("```")
    lines.append("")
    lines.append("## 结论（怎么选 source）")
    lines.append("")
    lines.append("先解释一下报告里两行“推荐 source”的含义（通俗版）：")
    lines.append("")
    lines.append("- 这些数字（例如 `3569341`）就是 **source 的顶点编号（vertex id）**，用于填到命令行里的 `source_id` 位置。")
    lines.append("- 这个编号是 **0-based**（从 0 开始数），因为 Gardenia 在读 `.mtx` 时会把文件里的 1-based 顶点编号减 1 存到内部。")
    lines.append("- “度（degree）”可以理解成“这个点连出去/连着多少条边”。度越大，算法扩散时越容易遇到 **随机访存、热点竞争、frontier 变大**，对 IMA 更友好。")
    lines.append("")
    lines.append("推荐策略：")
    lines.append("")
    lines.append("- **首选 hub**：`max_vertex`，也就是“度最高的点”（可以理解为图里的“大 V/枢纽点”）。")
    lines.append("  - 优点：通常最容易把 IMA/L1 压力拉满。")
    lines.append("  - 缺点：也可能让仿真更慢、噪声更大（过于极端）。")
    lines.append("- **备选 p90**：`p90_vertex`，也就是“度排进前 10% 的代表点”（更温和）。")
    lines.append("  - 优点：通常更稳、更省时，但仍能给出明显压力。")
    lines.append("")
    lines.append("额外提醒：")
    lines.append("")
    lines.append("- **路网（road_usa/roadNet-CA）强烈建议用 `symmetrize=1`**，否则很容易出现 source 可达性差、迭代轮数很少（工作量被“测没了”）。")
    lines.append("")
    lines.append("## 各数据集统计与推荐 source")
    lines.append("")

    for p in files:
        f = str(p)
        if f not in out_rows or f not in sym_rows:
            continue
        name = dataset_name_from_mtx_path(p)
        out_s = out_rows[f]
        sym_s = sym_rows[f]
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- 文件：`{rel_to_real_workload(p)}`")
        lines.append(f"- |V| = {out_s.m}")
        lines.append("")
        lines.append("**Out-degree（按 MTX 方向）**")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|---|---|")
        lines.append(f"| avg out-degree | {fmt_float(out_s.avg_degree)} |")
        lines.append(f"| zero out-degree vertices | {out_s.zero_vertices} |")
        lines.append(f"| hub (max_degree) | deg={out_s.max_degree}, v={out_s.max_vertex} |")
        lines.append(f"| p99 (non-zero) | deg≈{out_s.p99_nz_deg}, v={out_s.p99_vertex} |")
        lines.append(f"| p90 (non-zero) | deg≈{out_s.p90_nz_deg}, v={out_s.p90_vertex} |")
        lines.append(f"| median (non-zero) | deg≈{out_s.median_nz_deg}, v={out_s.median_vertex} |")
        lines.append("")
        lines.append("**Symmetrized degree（把图当无向）**")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|---|---|")
        lines.append(f"| avg degree | {fmt_float(sym_s.avg_degree)} |")
        lines.append(f"| zero degree vertices | {sym_s.zero_vertices} |")
        lines.append(f"| hub (max_degree) | deg={sym_s.max_degree}, v={sym_s.max_vertex} |")
        lines.append(f"| p99 (non-zero) | deg≈{sym_s.p99_nz_deg}, v={sym_s.p99_vertex} |")
        lines.append(f"| p90 (non-zero) | deg≈{sym_s.p90_nz_deg}, v={sym_s.p90_vertex} |")
        lines.append(f"| median (non-zero) | deg≈{sym_s.median_nz_deg}, v={sym_s.median_vertex} |")
        lines.append("")
        lines.append("**推荐（用于 IMA 压力）**")
        lines.append("")
        lines.append(f"- 首选 hub（度最高的点）：`{out_s.max_vertex}`（更激进，压力更大）")
        lines.append(f"- 备选 p90（前 10% 高度点的代表）：`{out_s.p90_vertex}`（更稳，通常更省时）")
        if name.startswith("road"):
            lines.append(f"- 路网建议启用 symmetrize=1，并用 hub：`{sym_s.max_vertex}` 或 p90：`{sym_s.p90_vertex}`")

        # Second-stage locality check (out-degree adjacency).
        candidates = sorted(
            set([out_s.max_vertex, out_s.p90_vertex, out_s.p99_vertex])
        )
        try:
            locality_out = run_locality_tool(
                locality_tool, p, candidates, symmetrize=False, line_elems=32
            )
        except Exception as e:
            locality_out = {}
            lines.append("")
            lines.append(f"> Locality tool failed for {name}: {e}")
        lines.append("")
        lines.append("**第二阶段：邻居局部性检查（防反例）**")
        lines.append("")
        lines.append("Locality (out)：（按 MTX 方向统计 source 的 out-neighbors）")
        lines.append("")
        lines.append(
            "| candidate | source_id | unique_neighbors | bfs2_frontier2 | unique_lines | neighbors_per_line | line_density | neighbor_span |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        def row_or_blank(src_id: int) -> str:
            r = locality_out.get(src_id)
            if r is None:
                return f"|  | `{src_id}` |  |  |  |  |  |  |"
            return (
                f"|  | `{src_id}` | {r.unique_neighbors} | {r.bfs2_frontier2} | {r.unique_lines} | "
                f"{r.neighbors_per_line:.2f} | {r.line_density:.3f} | {r.neighbor_span} |"
            )
        # Preserve semantic labels
        lines.append(
            "| hub (max_degree) | "
            + row_or_blank(out_s.max_vertex).split("|", 2)[2]
        )
        lines.append(
            "| p99 (non-zero) | "
            + row_or_blank(out_s.p99_vertex).split("|", 2)[2]
        )
        lines.append(
            "| p90 (non-zero) | "
            + row_or_blank(out_s.p90_vertex).split("|", 2)[2]
        )

        # Quick sanity notes for reachability/pressure.
        hub_row = locality_out.get(out_s.max_vertex)
        p90_row = locality_out.get(out_s.p90_vertex)
        if hub_row is not None and hub_row.bfs2_frontier2 == 0:
            lines.append("")
            lines.append(
                "> 注意：`hub` 的 `bfs2_frontier2=0`，说明从它出发 2 跳都扩散不出去（很可能会让 BFS/SSSP 很快结束，压力偏小）。"
            )
        if p90_row is not None and p90_row.bfs2_frontier2 == 0:
            lines.append("")
            lines.append(
                "> 注意：`p90` 的 `bfs2_frontier2=0`，它可能是“只出 1 跳就到尽头”的点，适合做快速 sanity，但不适合压 IMA。"
            )

        # Interpretation helper (plain language).
        def interpret_locality(row: LocalityRow, m: int) -> str:
            if row.unique_neighbors == 0:
                return "出度几乎为 0，扩散很可能马上结束。"
            scatter_ratio = row.unique_lines / max(1.0, float(row.unique_neighbors))
            span_ratio = row.neighbor_span / max(1.0, float(m))
            if row.neighbors_per_line >= 3.0 and row.line_density >= 0.3:
                locality_hint = "邻居编号更集中（局部性更好），IMA 可能不算重。"
            elif scatter_ratio >= 0.8 and row.neighbors_per_line <= 1.3:
                locality_hint = "邻居非常分散（接近“一邻居一条 cache line”），更容易触发随机访存/IMA。"
            else:
                locality_hint = "邻居分散程度中等。"
            if row.bfs2_frontier2 == 0:
                growth_hint = "2 跳扩散为 0，压力很可能只在第 1 轮出现，后续很快结束。"
            else:
                # Avoid over-claiming for tiny-degree graphs.
                if row.bfs2_frontier2 < 256 and row.bfs2_frontier2 < 4 * row.unique_neighbors:
                    growth_hint = "2 跳扩散不大，压力持续性一般。"
                else:
                    growth_hint = "2 跳扩散较大，更可能形成多轮 frontier 压力。"
            volume_hint = ""
            if row.unique_neighbors < 32:
                volume_hint = "（但该点度本身很小，单轮访问量有限）"
            return f"{locality_hint} {growth_hint}{volume_hint}（neighbor_span≈{span_ratio:.2f}*|V|）"

        lines.append("")
        lines.append("解读（这张表说明什么）：")
        lines.append("")
        for label, src_id in [
            ("hub", out_s.max_vertex),
            ("p99", out_s.p99_vertex),
            ("p90", out_s.p90_vertex),
        ]:
            r = locality_out.get(src_id)
            if r is None:
                continue
            lines.append(f"- `{label}` (`{src_id}`)：{interpret_locality(r, out_s.m)}")

        if name.startswith("road"):
            # For road networks, also provide symmetrized locality (what you'd get with symmetrize=1).
            candidates_sym = sorted(
                set([sym_s.max_vertex, sym_s.p90_vertex, sym_s.p99_vertex])
            )
            try:
                locality_sym = run_locality_tool(
                    locality_tool, p, candidates_sym, symmetrize=True, line_elems=32
                )
            except Exception as e:
                locality_sym = {}
                lines.append("")
                lines.append(f"> Locality tool (sym) failed for {name}: {e}")
            lines.append("")
            lines.append("Locality (sym)：（把边当无向统计 source 的 neighbors；对应运行时 symmetrize=1 的直觉）")
            lines.append("")
            lines.append(
                "| candidate | source_id | unique_neighbors | bfs2_frontier2 | unique_lines | neighbors_per_line | line_density | neighbor_span |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            def row_sym_or_blank(src_id: int) -> str:
                r = locality_sym.get(src_id)
                if r is None:
                    return f"|  | `{src_id}` |  |  |  |  |  |  |"
                return (
                    f"|  | `{src_id}` | {r.unique_neighbors} | {r.bfs2_frontier2} | {r.unique_lines} | "
                    f"{r.neighbors_per_line:.2f} | {r.line_density:.3f} | {r.neighbor_span} |"
                )
            lines.append(
                "| hub (max_degree) | "
                + row_sym_or_blank(sym_s.max_vertex).split("|", 2)[2]
            )
            lines.append(
                "| p99 (non-zero) | "
                + row_sym_or_blank(sym_s.p99_vertex).split("|", 2)[2]
            )
            lines.append(
                "| p90 (non-zero) | "
                + row_sym_or_blank(sym_s.p90_vertex).split("|", 2)[2]
            )

            lines.append("")
            lines.append("解读（sym 这张表说明什么）：")
            lines.append("")
            for label, src_id in [
                ("hub", sym_s.max_vertex),
                ("p99", sym_s.p99_vertex),
                ("p90", sym_s.p90_vertex),
            ]:
                r = locality_sym.get(src_id)
                if r is None:
                    continue
                lines.append(f"- `{label}` (`{src_id}`)：{interpret_locality(r, sym_s.m)}")

        lines.append("")
        lines.append("建议命令行（按 Gardenia 参数约定；以 `./data/{name}` 为前缀举例）：")
        lines.append("")
        lines.append("```bash")
        lines.append(f"# SSSP: <filetype> <prefix> [symmetrize] [reverse] [source] [delta]")
        lines.append(f"./sssp_linear_base mtx ./data/{name} 0 0 {out_s.max_vertex} 1")
        lines.append(f"./sssp_linear_base mtx ./data/{name} 0 0 {out_s.p90_vertex} 1")
        lines.append(f"# BFS: <filetype> <prefix> [symmetrize] [reverse] [source]")
        lines.append(f"./bfs_linear_base mtx ./data/{name} 0 0 {out_s.max_vertex}")
        lines.append(f"./bfs_linear_base mtx ./data/{name} 0 0 {out_s.p90_vertex}")
        lines.append(f"# BC: <filetype> <prefix> [symmetrize] [reverse] [source]")
        lines.append(f"./bc_linear_base mtx ./data/{name} 0 0 {out_s.max_vertex}")
        lines.append(f"./bc_linear_base mtx ./data/{name} 0 0 {out_s.p90_vertex}")
        lines.append("```")
        lines.append("")
        lines.append("通俗例子（拿这两行解释）：")
        lines.append("")
        lines.append(f"- “首选 hub：`{out_s.max_vertex}`” 就是：把 `source_id` 设为 `{out_s.max_vertex}`，从一个连接很多边的“大点”出发，通常更容易把 IMA 压力打满。")
        lines.append(f"- “备选 p90：`{out_s.p90_vertex}`” 就是：把 `source_id` 设为 `{out_s.p90_vertex}`，从一个“也很活跃但没那么极端”的点出发，往往更稳更省时。")
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main() -> int:
    repo_root = repo_root_from_this_file()
    tool = build_degree_tool(repo_root)
    locality_tool = build_locality_tool(repo_root)
    mtx_files = list_real_workload_mtx(repo_root)
    if not mtx_files:
        print("No MTX files found under real_workload", file=sys.stderr)
        return 2
    out_rows = run_tool(tool, mtx_files, symmetrize=False)
    sym_rows = run_tool(tool, mtx_files, symmetrize=True)
    out_path = write_report(repo_root, mtx_files, out_rows, sym_rows, locality_tool)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
