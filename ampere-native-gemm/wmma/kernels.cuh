#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "kernels/wmma_naive.cuh"
#include "kernels/wmma_blocktile.cuh"
#include "kernels/wmma_warptile.cuh"

void run_wmma_naive(__half *A, __half *B, float *C, const int M, const int N, const int K, 
    float alpha, float beta){

    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    const int WN = 4;
    constexpr int WM = 4;
    dim3 gridDim(ceil(N * M / (WMMA_N * WN) * (WMMA_M * WM)));
    // dim3 gridDim(ceil(N / (WMMA_N * WN)), ceil(M / (WMMA_M * WM)));
    dim3 threadPerBlock(512);
    gemm_wmma_naive<WMMA_M, WMMA_N, WMMA_K>
    <<<gridDim, threadPerBlock>>>(A, B, C, M, N, K, alpha, beta);

}


void run_wmma_blocktiling(__half *A, __half *B, float *C, const int M, const int N, const int K, 
    float alpha, float beta){

    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    const int BM = 64;
    const int BN = 64;
    const int BK = 64;
    // constexpr int warpM = BM / WMMA_M;
    // constexpr int warpN = BN / WMMA_N;
    dim3 gridDim(ceil(N / BN) , ceil(M / BM));
    dim3 threadsPerBlock(512);

    gemm_wmma_blocktiling<BM, BN, BK, WMMA_M, WMMA_N, WMMA_K>
    <<<gridDim, threadsPerBlock>>>(A, B, C, M, N, K, alpha, beta);

}

void run_wmma_warptiling(__half *A, __half *B, float *C, const int M, const int N, const int K, 
    float alpha, float beta){
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    const int BM = 64;
    const int BN = 64;
    const int BK = 64;
    const int WM = 32;
    const int WN = 16;
    const int WARPSIZE = 32;

    constexpr int numThreadsPerBlock = (WARPSIZE * WN * WM) / (WMMA_M * WMMA_N);

    static_assert((WM * WN * WARPSIZE) % (WMMA_M * WMMA_N) == 0);
    static_assert(BM % WM == 0);
    static_assert(BN % WN == 0);

    dim3 gridDim((N / BN) , (M / BM));
    dim3 threadsPerBlock(numThreadsPerBlock);

    gemm_wmma_warptiling<BM, BN, BK, WM, WN, WMMA_M, WMMA_N, WMMA_K, numThreadsPerBlock>
    <<<gridDim, threadsPerBlock>>>(A, B, C, M, N, K, alpha, beta);
}

void run_custom_wmma_kernel(std::string kernel, __half *A, __half *B, float *C, const int M, const int N, 
    const int K, float alpha, float beta){
    
        if(kernel == "naive"){
            run_wmma_naive(A, B, C, M, N, K, alpha, beta);
        }
        else if(kernel == "blocktiling"){
            run_wmma_blocktiling(A, B, C, M, N, K, alpha, beta);
        }
        else if(kernel == "warptiling"){
            run_wmma_warptiling(A, B, C, M, N, K, alpha, beta);
        }
}