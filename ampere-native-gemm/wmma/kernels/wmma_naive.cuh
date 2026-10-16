#include <iostream>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;


template <const int WMMA_M, const int WMMA_N, const int WMMA_K>
__global__ void gemm_wmma_naive(__half *A, __half *B, float *C, const int M, const int N, const int K, 
    float alpha, float beta){

    int number_of_warps = blockDim.x / warpSize;
    
    // number of warps along N dim
    int warpNDim = N / WMMA_N;

    // Calculate warp no. for the current thread
    int warpIdx = blockIdx.x * number_of_warps + (threadIdx.x / warpSize); 
    // Warp no. along M dim
    int warpM = warpIdx / warpNDim;
    // Warp no. along M dim
    int warpN = warpIdx % warpNDim;

    // starting row idx of matrix A for the current warp(to be loaded for WMMA command)
    int rowM = warpM * WMMA_M;
    // starting col idx of matrix B for the current warp(to be loaded for WMMA command)
    int colN = warpN * WMMA_N;

    if(rowM < M && colN < N){
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

        wmma::fill_fragment(acc_frag, 0.0f);

        for(int dotTileIdx = 0; dotTileIdx < K; dotTileIdx += WMMA_K){ // for(int dotTileIdx = 0; dotTileIdx < K + WMMA_K - 1; dotTileIdx += WMMA_K){
            
            wmma::load_matrix_sync(a_frag, A + rowM * K + dotTileIdx, K);
            wmma::load_matrix_sync(b_frag, B + dotTileIdx * N + colN, N);

            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        
        wmma::load_matrix_sync(c_frag, C + rowM * N + colN, N, wmma::mem_row_major);

        for(int i=0; i < c_frag.num_elements; i++)
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];

        wmma::store_matrix_sync(C + rowM * N + colN, c_frag, N, wmma::mem_row_major);
    }

}