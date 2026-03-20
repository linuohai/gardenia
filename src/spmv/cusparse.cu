// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "spmv.h"
#include "timer.h"
#include "spmv_util.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
#include <cublas_v2.h>
#include <cusparse_v2.h>
#define SPMV_VARIANT "cusparse"

inline void CudaSparseCheckImpl(cusparseStatus_t status, const char *file, int line) {
	if (status != CUSPARSE_STATUS_SUCCESS) {
		fprintf(stderr, "CUSPARSE error at %s:%d: status=%d\n", file, line, (int)status);
		exit(EXIT_FAILURE);
	}
}

#define CudaSparseCheck(call) CudaSparseCheckImpl((call), __FILE__, __LINE__)

void SpmvSolver(Graph &g, const ValueT* h_Ax, const ValueT *h_x, ValueT *h_y) { 
	auto m = g.V();
	auto nnz = g.E();
	auto h_Ap = g.in_rowptr();
	auto h_Aj = g.in_colidx();
	//print_device_info(0);
	std::vector<int> h_Ap32(m + 1);
	for (int i = 0; i < m + 1; i++) {
		if (h_Ap[i] > INT_MAX) {
			fprintf(stderr, "rowptr[%d]=%lu exceeds 32-bit range\n", i, h_Ap[i]);
			exit(EXIT_FAILURE);
		}
		h_Ap32[i] = static_cast<int>(h_Ap[i]);
	}
	int *d_Ap;
	VertexId *d_Aj;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ap, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Aj, nnz * sizeof(VertexId)));
	CUDA_SAFE_CALL(cudaMemcpy(d_Ap, h_Ap32.data(), (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_Aj, h_Aj, nnz * sizeof(VertexId), cudaMemcpyHostToDevice));
	ValueT *d_Ax, *d_x, *d_y;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ax, sizeof(ValueT) * nnz));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_x, sizeof(ValueT) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_y, sizeof(ValueT) * m));
	CUDA_SAFE_CALL(cudaMemcpy(d_Ax, h_Ax, nnz * sizeof(ValueT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_x, h_x, m * sizeof(ValueT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_y, h_y, m * sizeof(ValueT), cudaMemcpyHostToDevice));
	ValueT *y_copy = (ValueT *)malloc(m * sizeof(ValueT));
	for(int i = 0; i < m; i ++) y_copy[i] = h_y[i];
	SpmvSerial(m, nnz, h_Ap, h_Aj, h_Ax, h_x, y_copy);

	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	printf("Launching CUDA SpMV solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	const ValueT alpha = 1.0;
	const ValueT beta = 1.0;

	cudaStream_t streamId;
	cusparseHandle_t cusparseHandle;
	streamId = NULL;
	cusparseHandle = NULL;
	cudaStreamCreateWithFlags(&streamId, cudaStreamNonBlocking);
	CudaSparseCheck(cusparseCreate(&cusparseHandle));
	CudaSparseCheck(cusparseSetStream(cusparseHandle, streamId));

	Timer t;
	t.Start();
#ifdef LONG_TYPES
	const cudaDataType value_type = CUDA_R_64F;
#else
	const cudaDataType value_type = CUDA_R_32F;
#endif
	cusparseSpMatDescr_t matA;
	cusparseDnVecDescr_t vecX;
	cusparseDnVecDescr_t vecY;
	size_t bufferSize = 0;
	void *dBuffer = NULL;

	CudaSparseCheck(cusparseCreateCsr(&matA,
		static_cast<int64_t>(m), static_cast<int64_t>(m), static_cast<int64_t>(nnz),
		d_Ap, d_Aj, d_Ax,
		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
		value_type));
	CudaSparseCheck(cusparseCreateDnVec(&vecX, static_cast<int64_t>(m), d_x, value_type));
	CudaSparseCheck(cusparseCreateDnVec(&vecY, static_cast<int64_t>(m), d_y, value_type));
	CudaSparseCheck(cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		&alpha, matA, vecX, &beta, vecY, value_type, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
	if (bufferSize > 0) CUDA_SAFE_CALL(cudaMalloc(&dBuffer, bufferSize));
	CudaSparseCheck(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		&alpha, matA, vecX, &beta, vecY, value_type, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
	CudaTest("solving failed");
	//CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	if (dBuffer) CUDA_SAFE_CALL(cudaFree(dBuffer));
	CudaSparseCheck(cusparseDestroyDnVec(vecX));
	CudaSparseCheck(cusparseDestroyDnVec(vecY));
	CudaSparseCheck(cusparseDestroySpMat(matA));
	CudaSparseCheck(cusparseDestroy(cusparseHandle));
	CUDA_SAFE_CALL(cudaStreamDestroy(streamId));
	double time = t.Millisecs();
	float gbyte = bytes_per_spmv(m, nnz);
	float GFLOPs = (time == 0) ? 0 : (2 * nnz / time) / 1e6;
	float GBYTEs = (time == 0) ? 0 : (gbyte / time) / 1e6;
	CUDA_SAFE_CALL(cudaMemcpy(h_y, d_y, sizeof(ValueT) * m, cudaMemcpyDeviceToHost));
	double error = l2_error(m, y_copy, h_y);
	printf("\truntime [%s] = %.4f ms ( %5.2f GFLOP/s %5.1f GB/s) [L2 error %f]\n", SPMV_VARIANT, time, GFLOPs, GBYTEs, error);

	CUDA_SAFE_CALL(cudaMemcpy(h_y, d_y, sizeof(ValueT) * m, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_Ap));
	CUDA_SAFE_CALL(cudaFree(d_Aj));
	CUDA_SAFE_CALL(cudaFree(d_Ax));
	CUDA_SAFE_CALL(cudaFree(d_x));
	CUDA_SAFE_CALL(cudaFree(d_y));
}
