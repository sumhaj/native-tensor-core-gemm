#include <iostream>
#include <cuda/barrier>
#include <cooperative_groups.h>
// #include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

template <const int BM, const int BN, const int BK, const int strideA, const int strideB, typename T>
__device__ void load_data_to_smem(__half *A, __half *B, __half *As, __half *Bs, const int M, const int N, const int K,
                                  int innerRowA, int innerColA, int innerRowB, int innerColB, T &barrier)
{

    for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA)
        cuda::memcpy_async(&As[(innerRowA + loadOffset) * BK + innerColA * 8],
                           &A[(innerRowA + loadOffset) * K + innerColA * 8],
                           cuda::aligned_size_t<16>(sizeof(float4)), barrier);

        

    for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB)
        cuda::memcpy_async(&Bs[(innerRowB + loadOffset) * BN + innerColB * 8],
                           &B[(innerRowB + loadOffset) * N + innerColB * 8],
                           cuda::aligned_size_t<16>(sizeof(float4)), barrier);
}

template <const int BM, const int BN, const int BK, const int WM, const int WN, const int WMITER,
          const int WNITER, const int WMMA_M, const int WMMA_N, const int WMMA_K>
__device__ void compute_gemm_using_smem(__half *As, __half *Bs, auto a_frag, auto b_frag, 
    auto acc_frag, const int rowM, const int colN){
    
    for (int warpSubRowIdx = 0; warpSubRowIdx < WMITER; warpSubRowIdx++){
        for (int warpSubColIdx = 0; warpSubColIdx < WNITER; warpSubColIdx++){
            for (int i = 0; i < BK; i += WMMA_K){

                wmma::load_matrix_sync(a_frag, As + (warpSubRowIdx * WM + rowM) * BK + i, BK);
                wmma::load_matrix_sync(b_frag, Bs + i * BN + warpSubColIdx * WN + colN, BN);

                wmma::mma_sync(acc_frag[warpSubRowIdx * WNITER + warpSubColIdx], a_frag,
                               b_frag, acc_frag[warpSubRowIdx * WNITER + warpSubColIdx]);
            }
        }
    }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMMA_M, const int WMMA_N, const int WMMA_K, const int numThreadsPerBlock>
__global__ void gemm_wmma_double_buffer(__half *A, __half *B, float *C, const int M, const int N, const int K,
                                     float alpha, float beta){

    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    const int warpIdx = threadIdx.x / warpSize;
    const int warpRow = warpIdx / (WN / WMMA_N);
    const int warpCol = warpIdx % (WN / WMMA_N);

    const int WMITER = BM / WM;
    const int WNITER = BN / WN;

    const int rowM = warpRow * WMMA_M;
    const int colN = warpCol * WMMA_N;

    auto block = cooperative_groups::this_thread_block();
    using barrier_t = cuda::barrier<cuda::thread_scope_block>;
    alignas(cuda::barrier<cuda::thread_scope_block>) 
    // __shared__ char barrier_storage[sizeof(cuda::barrier<cuda::thread_scope_block>)];

    __shared__ barrier_t frontBarrier;
    __shared__ barrier_t backBarrier;
    auto frontBarrierPtr = &frontBarrier;
    auto backBarrierPtr = &backBarrier;

    if (block.thread_rank() == 0)
    {
        // A single thread initializes the total expected arrival count.
        init(&frontBarrier, block.size());
        init(&backBarrier, block.size());
    }
    __syncthreads();

    __shared__ __half As[2 * BM * BK];
    __shared__ __half Bs[2 * BK * BN];

    A += blockRow * BM * K;
    B += blockCol * BN;
    C += blockRow * BM * N + blockCol * BN;

    const int innerRowA = threadIdx.x / (BK / 8);
    const int innerColA = threadIdx.x % (BK / 8);
    const int strideA = 8 * numThreadsPerBlock / BK;

    const int innerRowB = threadIdx.x / (BN / 8);
    const int innerColB = threadIdx.x % (BN / 8);
    const int strideB = 8 * numThreadsPerBlock / BN;

    int bufferIdx = 0;

    load_data_to_smem<BM, BN, BK, strideA, strideB>(A, B, As + bufferIdx * BM * BK, Bs + bufferIdx * BK * BN, M, N, K, 
        innerRowA, innerColA, innerRowB, innerColB, (*frontBarrierPtr));

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag[WMITER * WNITER];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    for (int warpSubRowIdx = 0; warpSubRowIdx < WMITER; warpSubRowIdx++){
        for (int warpSubColIdx = 0; warpSubColIdx < WNITER; warpSubColIdx++){
            wmma::fill_fragment(acc_frag[warpSubRowIdx * WNITER + warpSubColIdx], 0.0f);
        }
    }


    for(int blockTileIdx = 0; blockTileIdx < K; blockTileIdx += BK){

        if(blockTileIdx + BK < K)
            load_data_to_smem<BM, BN, BK, strideA, strideB>(A + BK, B + BK * N, As + (1 - bufferIdx) * BM * BK, 
                Bs + (1 - bufferIdx) * BK * BN, M, N, K, innerRowA, innerColA, innerRowB, innerColB, (*backBarrierPtr));

        (*frontBarrierPtr).arrive_and_wait();

        compute_gemm_using_smem<BM, BN, BK, WM, WN, WMITER, WNITER, WMMA_M, WMMA_N, WMMA_K>(As + bufferIdx * BM * BK, 
            Bs + bufferIdx * BK * BN, a_frag, b_frag, acc_frag, rowM, colN);

        A += BK;
        B += BK * N;
        
        auto tmp = frontBarrierPtr;
        frontBarrierPtr = backBarrierPtr;
        backBarrierPtr = tmp;

        bufferIdx = 1 - bufferIdx;

        __syncthreads();
    }

    for (int warpSubRowIdx = 0; warpSubRowIdx < WMITER; warpSubRowIdx++){
        for (int warpSubColIdx = 0; warpSubColIdx < WNITER; warpSubColIdx++){

            const int subWarpIdx = warpSubRowIdx * WNITER + warpSubColIdx;
            float *C_temp = C + warpSubRowIdx * WM * N + warpSubColIdx * WN;
            wmma::load_matrix_sync(c_frag, C_temp + rowM * N + colN, 
                N, wmma::mem_row_major);

            for(int i=0; i < c_frag.num_elements; i++)
                c_frag.x[i] = 
                alpha * acc_frag[subWarpIdx].x[i] + 
                beta * c_frag.x[i];

            wmma::store_matrix_sync(C_temp + rowM * N + colN, 
                c_frag, N, wmma::mem_row_major);
            
        }
    }
}