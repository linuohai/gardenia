// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "cc.h"

int main(int argc, char *argv[]) {
  std::cout << "Connected Component by Xuhao Chen\n";
  if (argc < 3) {
    printf("Usage: %s <filetype> <graph> [symmetrize(0/1)] [reverse(0/1)]\n", argv[0]);
    printf("Example: %s mtx web-Google 1 1\n", argv[0]);
    exit(1);
  }
  bool symmetrize = false;
  bool need_reverse = false;
  if (argc > 3) symmetrize = atoi(argv[3]);
  if (argc > 4) need_reverse = atoi(argv[4]);
  Graph g(argv[2], argv[1], symmetrize, need_reverse);
  auto m = g.V();
  std::vector<CompT> comp(m);
  for (int i = 0; i < m; i++) comp[i] = i;
  CCSolver(g, &comp[0]);
  CCVerifier(g, &comp[0]);
  return 0;
}
