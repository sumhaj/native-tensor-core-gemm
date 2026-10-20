#include <iostream>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

template<const int BM, const int BN, const int BK, const int WM, const int WN, 
const int WMMA_M, const int WMMA_N, const int WMMA_K, const int numThreadsPerBlock>
__global__ void gemm_wmma_warptiling(__half *A, __half *B, float *C, const int M, const int N, const int K, 
    float alpha, float beta){

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int warpIdx = threadIdx.x / warpSize;
    int warpRow = warpIdx / (WN / WMMA_N);
    int warpCol = warpIdx % (WN / WMMA_N);

    int rowM = warpRow * WMMA_M;
    int colN = warpCol * WMMA_N;

    __shared__ __half As[BM * BK];
    __shared__ __half Bs[BK * BN];

    A += blockRow * BM * K;
    B += blockCol * BN;
    C += blockRow * BM * N + blockCol * BN;

    constexpr int WMITER = BM / WM;
    constexpr int WNITER = BN / WN;

    int innerRowA = threadIdx.x / (BK / 8);
    int innerColA = threadIdx.x % (BK / 8);
    int strideA = 8 * numThreadsPerBlock / BK;

    int innerRowB = threadIdx.x / (BN / 8);
    int innerColB = threadIdx.x % (BN / 8);
    int strideB = 8 * numThreadsPerBlock / BN;       

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[WMITER];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[WNITER];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag[WMITER * WNITER];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[WMITER * WNITER];

    for(int blockSubRowIdx = 0; blockSubRowIdx < WMITER; blockSubRowIdx++){
        for(int blockSubColIdx = 0; blockSubColIdx < WNITER; blockSubColIdx++){
            wmma::fill_fragment(acc_frag[blockSubRowIdx * WNITER + blockSubColIdx], 0.0f);
        }
    }

    for(int blockTileIdx = 0; blockTileIdx < K; blockTileIdx += BK){

        for(int loadOffset = 0; loadOffset < BM; loadOffset += strideA)
            reinterpret_cast<float4 *>(&As[(innerRowA + loadOffset) * BK + innerColA * 8])[0] = 
            reinterpret_cast<float4 *>(&A[(innerRowA + loadOffset) * K + innerColA * 8])[0];

        for(int loadOffset = 0; loadOffset < BK; loadOffset += strideB)
            reinterpret_cast<float4 *>(&Bs[(innerRowB + loadOffset) * BN + innerColB * 8])[0] = 
            reinterpret_cast<float4 *>(&B[(innerRowB + loadOffset) * N + innerColB * 8])[0];

        __syncthreads();

        A += BK;
        B += BK * N;

        for(int blockSubRowIdx = 0; blockSubRowIdx < WMITER; blockSubRowIdx++){
            for(int blockSubColIdx = 0; blockSubColIdx < WNITER; blockSubColIdx++){
                const int blockSubIdx = blockSubRowIdx * WNITER + blockSubColIdx;
                for(int i = 0; i < BK; i += WMMA_K){

                    wmma::load_matrix_sync(a_frag[blockSubRowIdx], As + (blockSubRowIdx * WM + rowM) * BK + i, BK);
                    wmma::load_matrix_sync(b_frag[blockSubColIdx], Bs + i * BN + (blockSubColIdx * WN) + colN, BN);

                    wmma::mma_sync(acc_frag[blockSubIdx], a_frag[blockSubRowIdx], 
                        b_frag[blockSubColIdx], acc_frag[blockSubIdx]);
                }
            }
        }

        __syncthreads();
    }

    for(int blockSubRowIdx = 0; blockSubRowIdx < WMITER; blockSubRowIdx++){
        for(int blockSubColIdx = 0; blockSubColIdx < WNITER; blockSubColIdx++){

            const int blockSubIdx = blockSubRowIdx * WNITER + blockSubColIdx;
            float *C_temp = C + blockSubRowIdx * WM * N + blockSubColIdx * WN;

            wmma::load_matrix_sync(c_frag[blockSubIdx], 
                C_temp + rowM * N + colN, N, wmma::mem_row_major);

            for(int i=0; i < c_frag[blockSubIdx].num_elements; i++)
                c_frag[blockSubIdx].x[i] = 
                alpha * acc_frag[blockSubIdx].x[i] + 
                beta * c_frag[blockSubIdx].x[i];

            wmma::store_matrix_sync(C_temp + rowM * N + colN, 
                c_frag[blockSubIdx], N, wmma::mem_row_major);
        }       
        
    }

}