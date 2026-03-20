#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "csr_graph.h"

static void write_binary_or_die(const std::string &path, const void *data,
                                size_t bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::out);
  if (!out) {
    std::cerr << "Failed to open output file: " << path << "\n";
    std::exit(1);
  }
  out.write(reinterpret_cast<const char *>(data),
            static_cast<std::streamsize>(bytes));
  if (!out) {
    std::cerr << "Failed to write output file: " << path << "\n";
    std::exit(1);
  }
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <in_mtx_prefix> <out_bin_prefix> [symmetrize(0/1)=1]"
                 " [need_reverse(0/1)=0]\n"
              << "Example: " << argv[0]
              << " ./data/web-Google ./data/web-Google 1 0\n";
    return 1;
  }
  const std::string in_prefix = argv[1];
  const std::string out_prefix = argv[2];
  const bool symmetrize = (argc > 3) ? (std::atoi(argv[3]) != 0) : true;
  const bool need_reverse = (argc > 4) ? (std::atoi(argv[4]) != 0) : false;

  Graph g(in_prefix, "mtx", symmetrize, need_reverse);

  const auto num_vertices = static_cast<size_t>(g.num_vertices());
  const auto num_edges = static_cast<size_t>(g.num_edges());
  const auto max_degree = static_cast<int>(g.get_max_degree());

  {
    std::ofstream meta(out_prefix + ".meta.txt");
    if (!meta) {
      std::cerr << "Failed to open output file: " << out_prefix << ".meta.txt\n";
      return 1;
    }
    meta << num_vertices << " " << num_edges << " " << sizeof(vidType) << " "
         << max_degree << "\n";
  }

  write_binary_or_die(out_prefix + ".vertex.bin", g.out_rowptr(),
                      sizeof(uint64_t) * (num_vertices + 1));
  write_binary_or_die(out_prefix + ".edge.bin", g.out_colidx(),
                      sizeof(VertexId) * num_edges);

  return 0;
}
