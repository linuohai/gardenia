cd $(dirname "$0")

wget http://www.cise.ufl.edu/research/sparse/MM/Gleich/flickr.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/SNAP/web-Google.tar.gz
# wget http://www.cise.ufl.edu/research/sparse/MM/SNAP/roadNet-CA.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/SNAP/cit-Patents.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/SNAP/soc-LiveJournal1.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/DIMACS10/road_usa.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/DIMACS10/kron_g500-logn21.tar.gz
wget http://nrvis.com/download/data/soc/soc-orkut.zip

mkdir -p ./real_workload

# 较小的测例已经上传到仓库
tar -xzvf flickr.tar.gz -C ./real_workload
tar -xzvf web-Google.tar.gz -C ./real_workload
tar -xzvf cit-Patents.tar.gz -C ./real_workload
tar -xzvf kron_g500-logn21.tar.gz -C ./real_workload
tar -xzvf road_usa.tar.gz -C ./real_workload
# tar -xzvf roadNet-CA.tar.gz -C ./real_workload
tar -xzvf soc-LiveJournal1.tar.gz -C ./real_workload
unzip soc-orkut.zip -d ./real_workload/soc-orkut

rm -f  *.zip *.gz

# Link datasets to accel-sim workload directories
mkdir -p ./accelsim/bfs_linear_base/data
mkdir -p ./accelsim/spmv/data
CURRENT_DIR=$(pwd)
find "$CURRENT_DIR/real_workload" -name "*.mtx" -exec ln -sf {} "$CURRENT_DIR/accelsim/bfs_linear_base/data/" \;
find "$CURRENT_DIR/real_workload" -name "*.mtx" -exec ln -sf {} "$CURRENT_DIR/accelsim/spmv/data/" \;

cd -