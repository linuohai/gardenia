# cuSPARSE SpMV 教程（面向 CUDA 初学者）

## 1) 你将学到什么
- 这段 `cusparse.cu` 的 SpMV 到底在做什么，以及为什么要这样做。
- CSR 稀疏矩阵如何从主机内存搬到 GPU，再交给 cuSPARSE。
- rowptr 转 32-bit 的原因、限制，以及失败时会发生什么。
- alpha/beta 的真实含义与结果公式。
- stream 与计时可能不准的原因（并说明需要进一步确认的文件）。

## 2) 前置知识（通俗解释 + 小例子）
**CSR 格式（Compressed Sparse Row）**  
CSR 的核心想法是：只存“非零元素”的位置和值。  
例子：矩阵 A  
```
[[1, 0, 2],
 [0, 0, 3],
 [4, 0, 0]]
```
CSR 会表示为：  
- Ap（行偏移）：`[0, 2, 3, 4]`  
- Aj（列索引）：`[0, 2, 2, 0]`  
- Ax（非零值）：`[1, 2, 3, 4]`  
含义：第 0 行有 2 个非零，存在 Ax[0..1]；第 1 行有 1 个非零，存在 Ax[2]，以此类推。

**SpMV 公式（稀疏矩阵乘向量）**  
对每一行做点积。  
例子：A=`[[1,0],[2,3]]`，x=`[10,20]`，结果 y=`[10, 80]`。

**alpha / beta 的含义**  
cuSPARSE 的 SpMV 公式通常是 `y = alpha * A * x + beta * y`（需要查看 cuSPARSE 文档确认）。  
例子：alpha=1、beta=1、y_init=[5,5]，则结果是 `A*x + y_init`。

**stream 与异步（基础概念）**  
stream 可以理解为 GPU 的“任务队列”。把拷贝和计算排进队列后，CPU 不必等待 GPU 完成。  
如果计时只在 CPU 上做，不同步 GPU，计时可能偏小。

## 3) 执行流程清单（按代码顺序）
Step 1: 取图规模与 CSR 指针  
`m`、`nnz`、`h_Ap`、`h_Aj` 来自 Graph（需要查看 `csr_graph.h` 确认细节）。  

Step 2: rowptr 转 32-bit  
把 `h_Ap` 转成 `h_Ap32`（`int`），并检查是否超过 `INT_MAX`，超出则退出。  

Step 3: 申请 GPU 内存  
`d_Ap/d_Aj/d_Ax/d_x/d_y` 分别对应 CSR 的行偏移、列索引、非零值、输入向量、输出向量。  

Step 4: 主机到设备拷贝  
把 `h_Ap32/h_Aj/h_Ax/h_x/h_y` 拷到 GPU。  

Step 5: CPU 参考计算  
用 `SpmvSerial` 在 CPU 上算一份 `y_copy` 作为正确性基线。  

Step 6: 创建 stream 与 cuSPARSE handle  
设置 stream，再把 handle 绑定到 stream。  

Step 7: 创建 cuSPARSE 描述符并分配 buffer  
创建 CSR 矩阵描述符、稠密向量描述符，查询临时 buffer 大小并分配。  

Step 8: 调用 `cusparseSpMV`  
在 GPU 上完成 SpMV。  

Step 9: 释放资源  
释放 buffer、描述符、handle、stream。  

Step 10: 拷回结果与误差计算  
把 `d_y` 拷回 `h_y`，计算 L2 error 并打印。  

## 4) 逐块注释式讲解
**整体逻辑 + 逐行解读（分块说明）**  
下面按“功能块”拆开引用，每块配一段说明，避免把全部代码一次性铺开。  

**块 1：错误检查宏**  
```cpp
inline void CudaSparseCheckImpl(cusparseStatus_t status, const char *file, int line) {
  if (status != CUSPARSE_STATUS_SUCCESS) {
    fprintf(stderr, "CUSPARSE error at %s:%d: status=%d\n", file, line, (int)status);
    exit(EXIT_FAILURE);
  }
}
#define CudaSparseCheck(call) CudaSparseCheckImpl((call), __FILE__, __LINE__)
```
说明：把 cuSPARSE 返回码变成可读错误信息，失败时直接退出，避免带着错误继续运行。

**块 2：入口与 CSR 指针获取**  
```cpp
void SpmvSolver(Graph &g, const ValueT* h_Ax, const ValueT *h_x, ValueT *h_y) {
  auto m = g.V();
  auto nnz = g.E();
  auto h_Ap = g.in_rowptr();
  auto h_Aj = g.in_colidx();
```
说明：从 Graph 取矩阵规模与 CSR 指针，这决定了后续 GPU 内存大小与拷贝范围。

**块 3：rowptr 转 32-bit（含溢出检查）**  
```cpp
  std::vector<int> h_Ap32(m + 1);
  for (int i = 0; i < m + 1; i++) {
    if (h_Ap[i] > INT_MAX) { fprintf(stderr, "rowptr[%d]=%lu exceeds 32-bit range\n", i, h_Ap[i]); exit(EXIT_FAILURE); }
    h_Ap32[i] = static_cast<int>(h_Ap[i]);
  }
```
说明：cuSPARSE 创建 CSR 描述符时指定了 32-bit 索引，因此必须保证 rowptr 不溢出；超出范围就直接退出。

**块 4：GPU 内存分配与拷贝**  
```cpp
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
```
说明：把 CSR 三数组和向量 x、y 从主机搬到设备；`d_y` 初值用于 `beta * y` 项。

**块 5：CPU 参考结果（正确性校验）**  
```cpp
  ValueT *y_copy = (ValueT *)malloc(m * sizeof(ValueT));
  for (int i = 0; i < m; i++) y_copy[i] = h_y[i];
  SpmvSerial(m, nnz, h_Ap, h_Aj, h_Ax, h_x, y_copy);
```
说明：在 CPU 上先算一遍“参考结果”，用于后续 L2 error 对比。

**块 6：stream 与 cuSPARSE handle**  
```cpp
  cudaStream_t streamId;
  cusparseHandle_t cusparseHandle;
  cudaStreamCreateWithFlags(&streamId, cudaStreamNonBlocking);
  CudaSparseCheck(cusparseCreate(&cusparseHandle));
  CudaSparseCheck(cusparseSetStream(cusparseHandle, streamId));
```
说明：创建非阻塞 stream，并让 cuSPARSE 在该 stream 上执行，支持异步排队。

**块 7：描述符 + buffer + SpMV**  
```cpp
  cusparseSpMatDescr_t matA;
  cusparseDnVecDescr_t vecX;
  cusparseDnVecDescr_t vecY;
  size_t bufferSize = 0;
  void *dBuffer = NULL;
  CudaSparseCheck(cusparseCreateCsr(&matA, (int64_t)m, (int64_t)m, (int64_t)nnz,
    d_Ap, d_Aj, d_Ax, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, value_type));
  CudaSparseCheck(cusparseCreateDnVec(&vecX, (int64_t)m, d_x, value_type));
  CudaSparseCheck(cusparseCreateDnVec(&vecY, (int64_t)m, d_y, value_type));
  CudaSparseCheck(cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha, matA, vecX, &beta, vecY, value_type, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
  if (bufferSize > 0) CUDA_SAFE_CALL(cudaMalloc(&dBuffer, bufferSize));
  CudaSparseCheck(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha, matA, vecX, &beta, vecY, value_type, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
```
说明：  
`cusparseCreateCsr`/`cusparseCreateDnVec` 把原始数组包装成 cuSPARSE 描述符；  
`cusparseSpMV_bufferSize` 让库决定最合适的临时 buffer 大小；  
`cusparseSpMV` 执行真正的 SpMV（公式与实现细节依赖库版本）。

**块 8：资源释放与结果拷回**  
```cpp
  CudaSparseCheck(cusparseDestroyDnVec(vecX));
  CudaSparseCheck(cusparseDestroyDnVec(vecY));
  CudaSparseCheck(cusparseDestroySpMat(matA));
  CudaSparseCheck(cusparseDestroy(cusparseHandle));
  CUDA_SAFE_CALL(cudaStreamDestroy(streamId));
  CUDA_SAFE_CALL(cudaMemcpy(h_y, d_y, sizeof(ValueT) * m, cudaMemcpyDeviceToHost));
}
```
说明：释放 cuSPARSE 描述符、handle 和 stream，最后把结果 `d_y` 拷回 `h_y`。  

**rowptr 转 32-bit 的原因、限制与失败条件**  
代码里调用 `cusparseCreateCsr` 使用 `CUSPARSE_INDEX_32I`。  
因此 `h_Ap` 必须能装进 32-bit。  
如果 `h_Ap[i] > INT_MAX`，代码会直接 `exit(EXIT_FAILURE)`。

**cudaMalloc/cudaMemcpy 的输入输出关系**  
- `d_Ap`：CSR 行偏移（来自 `h_Ap32`）  
- `d_Aj`：CSR 列索引（来自 `h_Aj`）  
- `d_Ax`：非零值（来自 `h_Ax`）  
- `d_x`：输入向量（来自 `h_x`）  
- `d_y`：输出向量（初值来自 `h_y`，用于 `beta * y` 项）  

**为什么先算 SpmvSerial 并做 L2 error**  
这一步是“正确性校验”：CPU 先算一份结果 `y_copy`，  
GPU 算完后再用 `l2_error` 比较二者。  
`l2_error` 的公式需要查看 `spmv_util.h` 才能完全确认。

**cuSPARSE API 调用关系**  
顺序是：  
`cusparseCreateCsr` → `cusparseCreateDnVec`(x) → `cusparseCreateDnVec`(y) →  
`cusparseSpMV_bufferSize` → 分配 buffer → `cusparseSpMV`。  
`CUSPARSE_SPMV_ALG_DEFAULT` 让库自行选择算法，具体策略随版本变化。

**alpha=1、beta=1 的真实公式**  
按 cuSPARSE 约定（需查文档确认），结果是：  
`y = A * x + y_init`。  
这里 `y_init` 来自传入的 `h_y`，在计算前拷到 `d_y`。

**stream 的作用与计时不准的原因**  
`cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)` 创建非阻塞 stream，  
计算与拷贝是异步排队的。  
`Timer` 不是 GPU event 计时（需看 `timer.h` 确认），  
所以可能测到“发指令”的时间，而不是 GPU 真正执行完的时间。

## 5) cuSPARSE API 参数表
**cusparseCreateCsr**
| 参数 | 代码里的值 | 通俗解释 |
|---|---|---|
| m, n | m, m | 行数、列数 |
| nnz | nnz | 非零数量 |
| rowOffsets | d_Ap | CSR 行偏移 |
| colIndices | d_Aj | CSR 列索引 |
| values | d_Ax | CSR 非零值 |
| rowOffsetsType | CUSPARSE_INDEX_32I | 行偏移类型 |
| colIndicesType | CUSPARSE_INDEX_32I | 列索引类型 |
| indexBase | CUSPARSE_INDEX_BASE_ZERO | 0-based |
| valueType | value_type | 数值类型（float 或 double，需看 typedef） |

**cusparseSpMV_bufferSize / cusparseSpMV**
| 参数 | 代码里的值 | 通俗解释 |
|---|---|---|
| operation | CUSPARSE_OPERATION_NON_TRANSPOSE | 不转置 |
| alpha, beta | &alpha, &beta | 线性组合系数 |
| A | matA | CSR 矩阵描述符 |
| x, y | vecX, vecY | 稠密向量描述符 |
| computeType | value_type | 计算类型 |
| alg | CUSPARSE_SPMV_ALG_DEFAULT | 库自动选算法（版本相关） |
| external buffer | dBuffer | 临时工作区 |

## 6) 超小 CSR 例子（4x4）
矩阵 A：
```
[ [10, 0, 0, 2],
  [ 0, 3, 0, 0],
  [ 4, 0, 5, 0],
  [ 0, 0, 0, 6] ]
```
向量 x = `[1, 2, 3, 4]`，初始 y_init = `[1, 1, 1, 1]`  
CSR：
```
Ap = [0, 2, 3, 5, 6]
Aj = [0, 3, 1, 0, 2, 3]
Ax = [10, 2, 3, 4, 5, 6]
```
手算第 3 行（下标 2）：  
`Ap[2]=3, Ap[3]=5`，看 Aj[3]=0、Aj[4]=2  
`A[2,*]·x = 4*x[0] + 5*x[2] = 4*1 + 5*3 = 19`  
再加上 `y_init[2]=1`，最终 `y[2]=20`。  
`cusparseSpMV` 就是在 GPU 上对每行做这个计算。

## 7) “它哪里体现了优化？”
- **把计算交给 cuSPARSE**：这就是最大的工程优化，避免自己写通用 SpMV kernel。  
- **使用 buffer**：`cusparseSpMV_bufferSize` 让库决定更高效的实现（具体策略随版本变化）。  
- **使用 stream**：为异步与重叠执行留出可能性。  
注意：库内部如何并行、具体 kernel 细节，都可能随 cuSPARSE 版本变化。

## 8) 如何运行与如何验证（模板）
运行命令模板（需根据你的路径补全）：  
```bash
./bin/spmv_cusparse mtx <dataset_prefix_path> <symmetrize> <reverse>
```
示例（取决于你的数据路径）：  
```bash
./bin/spmv_cusparse mtx ./datasets/real_workload/web-Google/web-Google 1 0
```
L2 error 的理解：  
输出越接近 0，越说明 GPU 与 CPU 结果一致。  
float 情况下一般希望在 `1e-4 ~ 1e-6` 量级，但会随规模与数据分布变化。  
阈值的严格性需要结合 `spmv_util.h` 的公式与数据集规模确认。

## 9) 如何用 ncu 做 profile
命令模板：  
```bash
ncu --set full --target-processes all ./bin/spmv_cusparse mtx <dataset_prefix_path> <symmetrize> <reverse>
```
观察点（通俗理解）：  
- 是否出现 cuSPARSE 的 SpMV 相关 kernel（名称随版本变化）。  
- Host 到 Device 的拷贝耗时（HtoD/DtoH）。  
- 总时间里 SpMV kernel 占比是多少。

## 10) 建议清单（不改源码）
- 释放 `y_copy`（当前 `malloc` 后未 `free`）。  
- 计时建议改用 `cudaEvent` 或在关键处 `cudaStreamSynchronize`（需查看 `timer.h`）。  
- 确认 `beta=1` 是否符合你的算法语义（现在计算的是 `A*x + y_init`）。  
- 注意索引溢出风险：`h_Ap[i] > INT_MAX` 会直接退出。  
- 代码里有两次 `cudaMemcpy(h_y, d_y, ...)`，可以评估是否需要。  
- `ValueT`/`VertexId` 的真实类型要以 `common.h` 为准（`LONG_TYPES` 宏会改变类型）。  
