#include <iostream>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

template <const int BM, const int BN, const int BK, const int WMMA_M, const int WMMA_N, const int WMMA_K>
__global__ void gemm_wmma_blocktiling(__half *A, __half *B, float *C, const int M, const int N, const int K, 
    float alpha, float beta){

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

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

    int innerRowA = threadIdx.x / (BK / 8);
    int innerColA = threadIdx.x % (BK / 8);

    int innerRowB = threadIdx.x / (BN / 8);
    int innerColB = threadIdx.x % (BN / 8);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for(int blockTileIdx = 0; blockTileIdx < K; blockTileIdx += BK){

        reinterpret_cast<float4 *>(&As[innerRowA * BK + innerColA * 8])[0] = 
        reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 8])[0];

        reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 8])[0] = 
        reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 8])[0];
        __syncthreads();

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