#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_fp16.h>

// cuBLAS is column-major; our matrices are row-major. Computing (B^T * A^T) = (A*B)^T
// lets us hand cuBLAS the transposed problem directly instead of transposing data --
// same trick SGEMM_CUDA used, kept as-is since it's correct.

inline void runCublasFP32(cublasHandle_t handle, int M, int N, int K, const float *A,
                           const float *B, float *C, float alpha, float beta) {
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F, N,
               A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// fp16 inputs, fp32 accumulate/output -- the reference every tensor-core kernel in
// this repo (phases 1-4) gets diffed against.
inline void runCublasFP16TensorOp(cublasHandle_t handle, int M, int N, int K,
                                   const __half *A, const __half *B, float *C,
                                   float alpha, float beta) {
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F, N,
               A, CUDA_R_16F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F_FAST_16F,
               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

inline cublasHandle_t make_cublas_handle() {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cublasCreate failed\n");
    exit(EXIT_FAILURE);
  }
  return handle;
}
