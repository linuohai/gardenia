#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
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

struct DegreeStats {
  uint32_t num_vertices = 0;
  uint64_t edges_read = 0;
  uint64_t self_loops_skipped = 0;
  uint64_t out_of_range_skipped = 0;
  uint32_t max_degree = 0;
  uint32_t max_degree_vertex = 0;
  uint64_t sum_degree = 0;
  uint32_t zero_degree_vertices = 0;
  uint32_t nonzero_degree_vertices = 0;
  uint32_t median_nonzero_degree = 0;
  uint32_t p90_nonzero_degree = 0;
  uint32_t p99_nonzero_degree = 0;
  uint32_t median_vertex = 0;
  uint32_t p90_vertex = 0;
  uint32_t p99_vertex = 0;
};

static DegreeStats compute_degree_stats(const std::string &mtx_path,
                                       bool symmetrize) {
  std::ifstream in(mtx_path);
  if (!in.good()) {
    std::cerr << "Failed to open " << mtx_path << "\n";
    std::exit(1);
  }

  MtxHeader hdr;
  if (!read_mtx_header(in, hdr)) {
    std::cerr << "Failed to read MTX header from " << mtx_path << "\n";
    std::exit(1);
  }
  if (hdr.m != hdr.n) {
    std::cerr << "Warning: m(" << hdr.m << ") != n(" << hdr.n
              << ") for file " << mtx_path << "\n";
  }

  std::vector<uint32_t> degree(hdr.m, 0);
  std::string line;
  uint64_t src_u64 = 0, dst_u64 = 0;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    if (line[0] == '%') continue;
    if (!parse_two_uint64(line.c_str(), src_u64, dst_u64)) continue;
    if (src_u64 == 0 || dst_u64 == 0) {
      // MTX should be 1-based. Skip malformed lines.
      continue;
    }
    const uint64_t src0 = src_u64 - 1;
    const uint64_t dst0 = dst_u64 - 1;
    if (src0 >= hdr.m || dst0 >= hdr.m) {
      // Some MTX files can have inconsistent dimensions; ignore.
      continue;
    }
    if (src0 == dst0) {
      continue;
    }
    degree[static_cast<size_t>(src0)]++;
    if (symmetrize) degree[static_cast<size_t>(dst0)]++;
  }

  DegreeStats stats;
  stats.num_vertices = hdr.m;
  for (uint32_t v = 0; v < hdr.m; v++) {
    const uint32_t d = degree[v];
    stats.sum_degree += d;
    if (d == 0) {
      stats.zero_degree_vertices++;
      continue;
    }
    stats.nonzero_degree_vertices++;
    if (d > stats.max_degree) {
      stats.max_degree = d;
      stats.max_degree_vertex = v;
    }
  }

  // Build histogram for non-zero degrees to compute quantiles.
  std::vector<uint32_t> hist(stats.max_degree + 1, 0);
  for (uint32_t v = 0; v < hdr.m; v++) {
    const uint32_t d = degree[v];
    if (d == 0) continue;
    hist[d]++;
  }

  const auto nz = stats.nonzero_degree_vertices;
  auto degree_at_quantile = [&](double q) -> uint32_t {
    if (nz == 0) return 0;
    const uint64_t target = static_cast<uint64_t>(std::ceil(q * nz));
    uint64_t cum = 0;
    for (uint32_t d = 1; d <= stats.max_degree; d++) {
      cum += hist[d];
      if (cum >= target) return d;
    }
    return stats.max_degree;
  };

  stats.median_nonzero_degree = degree_at_quantile(0.50);
  stats.p90_nonzero_degree = degree_at_quantile(0.90);
  stats.p99_nonzero_degree = degree_at_quantile(0.99);

  auto pick_vertex_at_least = [&](uint32_t threshold) -> uint32_t {
    if (threshold == 0) return 0;
    for (uint32_t v = 0; v < hdr.m; v++) {
      if (degree[v] >= threshold) return v;
    }
    return stats.max_degree_vertex;
  };

  stats.median_vertex = pick_vertex_at_least(stats.median_nonzero_degree);
  stats.p90_vertex = pick_vertex_at_least(stats.p90_nonzero_degree);
  stats.p99_vertex = pick_vertex_at_least(stats.p99_nonzero_degree);

  return stats;
}

static void print_usage(const char *argv0) {
  std::cerr << "Usage: " << argv0 << " [--symmetrize] <file1.mtx> [file2.mtx...]\n";
}

int main(int argc, char **argv) {
  bool symmetrize = false;
  std::vector<std::string> files;
  for (int i = 1; i < argc; i++) {
    if (std::strcmp(argv[i], "--symmetrize") == 0) {
      symmetrize = true;
      continue;
    }
    files.emplace_back(argv[i]);
  }
  if (files.empty()) {
    print_usage(argv[0]);
    return 1;
  }

  std::cout << "mode=" << (symmetrize ? "symmetrized_degree" : "out_degree") << "\n";
  std::cout << "file,m,avg_degree,max_degree,max_vertex,zero_vertices,"
               "median_nz_deg,median_vertex,p90_nz_deg,p90_vertex,p99_nz_deg,p99_vertex\n";
  for (const auto &path : files) {
    const auto stats = compute_degree_stats(path, symmetrize);
    const double avg =
        (stats.num_vertices == 0)
            ? 0.0
            : static_cast<double>(stats.sum_degree) / stats.num_vertices;
    std::cout << path << "," << stats.num_vertices << "," << avg << ","
              << stats.max_degree << "," << stats.max_degree_vertex << ","
              << stats.zero_degree_vertices << "," << stats.median_nonzero_degree
              << "," << stats.median_vertex << "," << stats.p90_nonzero_degree
              << "," << stats.p90_vertex << "," << stats.p99_nonzero_degree
              << "," << stats.p99_vertex << "\n";
  }
  return 0;
}
