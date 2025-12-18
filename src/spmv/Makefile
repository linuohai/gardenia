include ../common.mk
SRC=main.cc
OBJS=vector.o omp_base.o verifier.o main.o
INCLUDES += -I${MKLROOT}/include -I$(CUB_DIR)
LIBS += -L${MKLROOT}/lib/intel64

### Optional: cubin & SASS dump helpers
#
# Usage examples:
#   make cubin-base
#   make sass-base
#   make sass-all
#
# Control verbosity / source-line mapping:
#   make SASS_VERBOSE=1 sass-base
#   make DEBUG=1 sass-base          # device debug (-G -g) comes from common.mk

NVCC_CUBIN_FLAGS ?= $(NVFLAGS)

NVCC_SASS_FLAGS ?=
ifeq ($(SASS_VERBOSE),1)
NVCC_SASS_FLAGS += --print-instruction-info --print-guard --print-code --print-line-info
else
NVCC_SASS_FLAGS += --print-line-info
endif

CU_SRCS := $(wildcard *.cu)
CUBINS  := $(CU_SRCS:.cu=.cubin)
SASS    := $(CU_SRCS:.cu=.sass)

### Generate cubin for a single CU file (compile-only)
%.cubin: %.cu
	$(NVCC) $(NVCC_CUBIN_FLAGS) $(INCLUDES) --cubin $< -o $@

### Disassemble cubin into SASS text
%.sass: %.cubin
	nvdisasm $(NVCC_SASS_FLAGS) $< > $@

# Convenience targets
cubin-all: $(CUBINS)
sass-all: $(SASS)

cubin-base: base.cubin
sass-base: base.sass
cubin-warp: warp.cubin
sass-warp: warp.sass
cubin-vector: vector.cubin
sass-vector: vector.sass
cubin-push: push.cubin
sass-push: push.sass
all: $(OBJS) spmv_vector spmv_omp_base spmv_warp spmv_base

%.o: %.cpp
	$(ICPC) $(ICPCFLAGS) $(INCLUDES) -march=core-avx2 -D__PURE_INTEL_C99_HEADERS__ -c $<

ifeq ($(HOST),X86)
%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<
else
%.o: %.cc
	$(CXX) $(SIMFLAGS) $(INCLUDES) -c $<
endif

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

spmv_omp_base: omp_base.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o omp_base.o verifier.o -o $@ -lgomp
	mv $@ $(BIN)

spmv_omp_pb: omp_pb.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o omp_pb.o verifier.o -o $@ -lgomp
	mv $@ $(BIN)

spmv_omp_push: omp_push.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o omp_push.o verifier.o -o $@ -lgomp
	mv $@ $(BIN)

spmv_omp_base_sim: omp_base.o main.o verifier.o
	$(CXX) $(SIMFLAGS) $(INCLUDES) main.o omp_base.o verifier.o $(M5OP) -o $@ -lgomp
	mv $@ $(BIN)

spmv_ocl_base: ocl_base.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o ocl_base.o verifier.o -o $@ -lOpenCL
	mv $@ $(BIN)

spmv_omp_simd: omp_simd.o main.o verifier.o
	$(ICPC) $(ICPCFLAGS) $(INCLUDES) main.o omp_simd.o verifier.o -o $@
	mv $@ $(BIN)

spmv_base: main.o base.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o base.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

spmv_tex: main.o tex.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o tex.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

spmv_warp: main.o warp.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o warp.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

spmv_vector: main.o vector.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o vector.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

spmv_push: main.o push.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o push.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

spmv_push_tile: main.o push_tile.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o push_tile.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

spmv_partition: main.o partition.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o partition.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

spmv_tiling: main.o tiling.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o tiling.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

spmv_cusparse: main.o cusparse.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o cusparse.o verifier.o -o $@ $(LIBS) -lcusparse
	mv $@ $(BIN)

spmv_mkl: main.o mkl.o verifier.o
	$(ICPC) $(ICPCFLAGS) $(INCLUDES) main.o mkl.o verifier.o -o $@ $(LIBS) -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm
	mv $@ $(BIN)

spmv_omp_target: main.o omp_target.o verifier.o
	$(ICPC) $(ICPCFLAGS) $(INCLUDES) $(EXTRA) main.o omp_target.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm -f *.o *.cubin *.sass
