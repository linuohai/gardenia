#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

struct MtxHeader {
  uint32_t m = 0;
  uint32_t n = 0;
  uint64_t nnz_declared = 0;
};

static bool parse_three_uint64(const char *line, uint64_t &a, uint64_t &b,
                               uint64_t &c) {
  char *end = nullptr;
  while (*line && std::isspace(*line)) ++line;
  if (!*line) return false;
  a = std::strtoull(line, &end, 10);
  if (end == line) return false;
  line = end;
  b = std::strtoull(line, &end, 10);
  if (end == line) return false;
  line = end;
  c = std::strtoull(line, &end, 10);
  if (end == line) return false;
  return true;
}

static bool parse_two_uint64(const char *line, uint64_t &a, uint64_t &b) {
  char *end = nullptr;
  while (*line && std::isspace(*line)) ++line;
  if (!*line) return false;
  a = std::strtoull(line, &end, 10);
  if (end == line) return false;
  line = end;
  b = std::strtoull(line, &end, 10);
  if (end == line) return false;
  return true;
}

static bool read_mtx_header(std::ifstream &in, MtxHeader &hdr) {
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    if (line[0] == '%') continue;
    uint64_t m = 0, n = 0, nnz = 0;
    if (!parse_three_uint64(line.c_str(), m, n, nnz)) return false;
    if (m > std::numeric_limits<uint32_t>::max() ||
        n > std::numeric_limits<uint32_t>::max()) {
      return false;
    }
    hdr.m = static_cast<uint32_t>(m);
    hdr.n = static_cast<uint32_t>(n);
    hdr.nnz_declared = nnz;
    return true;
  }
  return false;
}

struct LocalityStats {
  uint32_t source = 0;
  uint64_t raw_neighbors = 0;
  uint64_t unique_neighbors = 0;
  uint32_t min_neighbor = 0;
  uint32_t max_neighbor = 0;
  uint64_t neighbor_span = 0;

  uint32_t line_elems = 32;
  uint64_t unique_lines = 0;
  uint64_t line_span = 0;
  double line_density = 0.0;
  double neighbors_per_line = 0.0;
  double lines_per_neighbor = 0.0;

  double mean_gap = 0.0;
  uint32_t median_gap = 0;
  uint32_t p90_gap = 0;

  uint64_t bfs2_frontier2 = 0;
};

static uint32_t percentile_from_sorted(const std::vector<uint32_t> &v,
                                       double q) {
  if (v.empty()) return 0;
  const double clamped = std::max(0.0, std::min(1.0, q));
  const size_t idx =
      static_cast<size_t>(std::ceil(clamped * static_cast<double>(v.size()))) -
      1;
  return v[std::min(idx, v.size() - 1)];
}

static LocalityStats compute_locality(uint32_t source, uint64_t raw_count,
                                     const std::vector<uint32_t> &neighbors,
                                     uint32_t line_elems) {
  LocalityStats s;
  s.source = source;
  s.raw_neighbors = raw_count;
  s.line_elems = line_elems;
  if (neighbors.empty()) return s;
  s.unique_neighbors = neighbors.size();
  s.min_neighbor = neighbors.front();
  s.max_neighbor = neighbors.back();
  s.neighbor_span = static_cast<uint64_t>(s.max_neighbor) - s.min_neighbor;

  uint64_t unique_lines = 0;
  uint64_t min_line = static_cast<uint64_t>(s.min_neighbor) / line_elems;
  uint64_t max_line = static_cast<uint64_t>(s.max_neighbor) / line_elems;
  uint64_t prev_line = std::numeric_limits<uint64_t>::max();
  for (auto nid : neighbors) {
    const uint64_t line = static_cast<uint64_t>(nid) / line_elems;
    if (line != prev_line) {
      unique_lines++;
      prev_line = line;
    }
  }
  s.unique_lines = unique_lines;
  s.line_span = max_line - min_line + 1;
  s.line_density =
      (s.line_span == 0) ? 0.0 : static_cast<double>(s.unique_lines) / s.line_span;
  s.neighbors_per_line =
      (s.unique_lines == 0)
          ? 0.0
          : static_cast<double>(s.unique_neighbors) / s.unique_lines;
  s.lines_per_neighbor =
      (s.unique_neighbors == 0)
          ? 0.0
          : static_cast<double>(s.unique_lines) / s.unique_neighbors;

  if (neighbors.size() < 2) return s;
  std::vector<uint32_t> gaps;
  gaps.reserve(neighbors.size() - 1);
  uint64_t gap_sum = 0;
  for (size_t i = 1; i < neighbors.size(); i++) {
    const uint32_t gap = neighbors[i] - neighbors[i - 1];
    gaps.push_back(gap);
    gap_sum += gap;
  }
  std::sort(gaps.begin(), gaps.end());
  s.mean_gap = static_cast<double>(gap_sum) / gaps.size();
  s.median_gap = percentile_from_sorted(gaps, 0.50);
  s.p90_gap = percentile_from_sorted(gaps, 0.90);
  return s;
}

static void print_usage(const char *argv0) {
  std::cerr << "Usage: " << argv0
            << " [--symmetrize] [--bfs2] [--line-elems N] <file.mtx> <source0> [source1...]\n";
}

int main(int argc, char **argv) {
  bool symmetrize = false;
  uint32_t line_elems = 32;
  bool bfs2 = false;
  std::vector<std::string> positional;
  for (int i = 1; i < argc; i++) {
    if (std::strcmp(argv[i], "--symmetrize") == 0) {
      symmetrize = true;
      continue;
    }
    if (std::strcmp(argv[i], "--bfs2") == 0) {
      bfs2 = true;
      continue;
    }
    if (std::strcmp(argv[i], "--line-elems") == 0) {
      if (i + 1 >= argc) {
        print_usage(argv[0]);
        return 1;
      }
      line_elems = static_cast<uint32_t>(std::strtoul(argv[i + 1], nullptr, 10));
      i++;
      continue;
    }
    positional.emplace_back(argv[i]);
  }

  if (positional.size() < 2) {
    print_usage(argv[0]);
    return 1;
  }

  const std::string file_path = positional[0];
  std::vector<uint32_t> sources;
  sources.reserve(positional.size() - 1);
  for (size_t i = 1; i < positional.size(); i++) {
    sources.push_back(static_cast<uint32_t>(std::strtoul(positional[i].c_str(), nullptr, 10)));
  }
  std::sort(sources.begin(), sources.end());
  sources.erase(std::unique(sources.begin(), sources.end()), sources.end());

  std::ifstream in(file_path);
  if (!in.good()) {
    std::cerr << "Failed to open " << file_path << "\n";
    return 1;
  }
  MtxHeader hdr;
  if (!read_mtx_header(in, hdr)) {
    std::cerr << "Failed to read MTX header from " << file_path << "\n";
    return 1;
  }

  std::vector<std::vector<uint32_t>> neighbors(sources.size());

  auto find_source_index = [&](uint32_t v) -> int {
    auto it = std::lower_bound(sources.begin(), sources.end(), v);
    if (it == sources.end() || *it != v) return -1;
    return static_cast<int>(it - sources.begin());
  };

  std::string line;
  uint64_t src_u64 = 0, dst_u64 = 0;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    if (line[0] == '%') continue;
    if (!parse_two_uint64(line.c_str(), src_u64, dst_u64)) continue;
    if (src_u64 == 0 || dst_u64 == 0) continue;
    const uint64_t src0 = src_u64 - 1;
    const uint64_t dst0 = dst_u64 - 1;
    if (src0 >= hdr.m || dst0 >= hdr.m) continue;
    if (src0 == dst0) continue;

    const int src_idx = find_source_index(static_cast<uint32_t>(src0));
    if (src_idx >= 0) neighbors[src_idx].push_back(static_cast<uint32_t>(dst0));
    if (symmetrize) {
      const int dst_idx = find_source_index(static_cast<uint32_t>(dst0));
      if (dst_idx >= 0) neighbors[dst_idx].push_back(static_cast<uint32_t>(src0));
    }
  }

  std::vector<uint64_t> raw_counts(sources.size(), 0);
  for (size_t i = 0; i < sources.size(); i++) {
    raw_counts[i] = neighbors[i].size();
    if (neighbors[i].empty()) continue;
    std::sort(neighbors[i].begin(), neighbors[i].end());
    neighbors[i].erase(std::unique(neighbors[i].begin(), neighbors[i].end()),
                       neighbors[i].end());
  }

  std::vector<uint64_t> bfs2_counts(sources.size(), 0);
  if (bfs2) {
    // BFS depth-2 expansion estimate:
    // frontier1 = unique out-neighbors of source (or sym neighbors if --symmetrize)
    // frontier2 = unique vertices reached by one more hop from frontier1.
    const size_t m = hdr.m;
    const size_t ns = sources.size();
    std::vector<uint8_t> in_frontier1(ns * m, 0);
    std::vector<uint8_t> visited(ns * m, 0);
    for (size_t si = 0; si < ns; si++) {
      const size_t base = si * m;
      const uint32_t src = sources[si];
      if (src < m) visited[base + src] = 1;
      for (auto v : neighbors[si]) {
        in_frontier1[base + v] = 1;
        visited[base + v] = 1;
      }
    }

    std::ifstream in2(file_path);
    if (in2.good()) {
      MtxHeader hdr2;
      if (read_mtx_header(in2, hdr2)) {
        std::string line2;
        uint64_t su = 0, du = 0;
        while (std::getline(in2, line2)) {
          if (line2.empty()) continue;
          if (line2[0] == '%') continue;
          if (!parse_two_uint64(line2.c_str(), su, du)) continue;
          if (su == 0 || du == 0) continue;
          const uint64_t s0 = su - 1;
          const uint64_t d0 = du - 1;
          if (s0 >= m || d0 >= m) continue;
          if (s0 == d0) continue;
          const uint32_t src = static_cast<uint32_t>(s0);
          const uint32_t dst = static_cast<uint32_t>(d0);
          for (size_t si = 0; si < ns; si++) {
            const size_t base = si * m;
            if (in_frontier1[base + src] && !visited[base + dst]) {
              visited[base + dst] = 1;
              bfs2_counts[si]++;
            }
            if (symmetrize && in_frontier1[base + dst] && !visited[base + src]) {
              visited[base + src] = 1;
              bfs2_counts[si]++;
            }
          }
        }
      }
    }
  }

  std::cout << "mode,file,source,raw_neighbors,unique_neighbors,min_neighbor,max_neighbor,neighbor_span,"
               "unique_lines,line_span,line_density,neighbors_per_line,lines_per_neighbor,mean_gap,median_gap,p90_gap,"
               "bfs2_frontier2\n";
  const std::string mode = symmetrize ? "sym" : "out";
  for (size_t i = 0; i < sources.size(); i++) {
    auto stats = compute_locality(sources[i], raw_counts[i], neighbors[i], line_elems);
    stats.bfs2_frontier2 = bfs2_counts[i];
    std::cout << mode << "," << file_path << "," << stats.source << "," << stats.raw_neighbors
              << "," << stats.unique_neighbors << "," << stats.min_neighbor << ","
              << stats.max_neighbor << "," << stats.neighbor_span << ","
              << stats.unique_lines << "," << stats.line_span << ","
              << stats.line_density << "," << stats.neighbors_per_line << ","
              << stats.lines_per_neighbor << "," << stats.mean_gap << ","
              << stats.median_gap << "," << stats.p90_gap << ","
              << stats.bfs2_frontier2 << "\n";
  }

  return 0;
}
