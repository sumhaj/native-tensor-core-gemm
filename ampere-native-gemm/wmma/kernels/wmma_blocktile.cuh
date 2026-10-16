#include <iostream>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

template <const int BM, const int BN, const int BK>
__device__ void load_data_to_smem(__half *A, __half *B, __half *As, __half *Bs, const int N, const int K, 
    int innerRowA, int innerColA, int strideA, int innerRowB, int innerColB, int strideB){

    for(int loadOffset = 0; loadOffset < BM; loadOffset += strideA){
        As[(innerRowA + loadOffset) * BK + innerColA] = A[(innerRowA + loadOffset) * K + innerColA];
    }

    for(int loadOffset = 0; loadOffset < BK; loadOffset += strideB){
        Bs[(innerRowB + loadOffset) * BN + innerColB] = B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();
}

template <const int BM, const int BN, const int BK, const int WMMA_M, const int WMMA_N, const int WMMA_K>
__global__ void gemm_wmma_blocktiling(__half *A, __half *B, float *C, const int M, const int N, const int K, 
    float alpha, float beta){

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int numThreadsPerBlock = blockDim.x;
    // int numOfWarps = numThreadsPerBlock / warpSize;

    int warpIdx = threadIdx.x / warpSize;
    int warpRow = warpIdx / (BN / WMMA_N);
    int warpCol = warpIdx % (BN / WMMA_N);

    int rowM = warpRow * WMMA_M;
    int colN = warpCol * WMMA_N;

    __shared__ __half As[BM * BK];
    __shared__ __half Bs[BK * BN];

    A += blockRow * BM * K;
    B += blockCol * BN;
    C += blockRow * BM * N + blockCol * BN;

    int innerRowA = threadIdx.x / BK;
    int innerColA = threadIdx.x % BK;
    int strideA = numThreadsPerBlock / BK;

    int innerRowB = threadIdx.x / BN;
    int innerColB = threadIdx.x % BN;
    int strideB = numThreadsPerBlock / BN;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for(int blockTileIdx = 0; blockTileIdx < K; blockTileIdx += BK){

        load_data_to_smem<BM, BN, BK>(A, B, As, Bs, N, K, innerRowA, innerColA, strideA, 
            innerRowB, innerColB, strideB);

        A += BK;
        B += BK * N;

        for(int i = 0; i < BK; i += WMMA_K){

            wmma::load_matrix_sync(a_frag, As + rowM * BK + i, BK);
            wmma::load_matrix_sync(b_frag, Bs + i * BN + colN, BN);

            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        __syncthreads();

    }
    
    wmma::load_matrix_sync(c_frag, C + rowM * N + colN, N, wmma::mem_row_major);
    
    for(int i=0; i < c_frag.num_elements; i++)
        c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];

    wmma::store_matrix_sync(C + rowM * N + colN, c_frag, N, wmma::mem_row_major);
}