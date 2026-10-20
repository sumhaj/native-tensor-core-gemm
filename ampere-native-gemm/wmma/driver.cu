#include <cstdio>
#include <functional>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "bench.cuh"
#include "cublas_ref.cuh"
#include "matrix_utils.cuh"
#include "verify.cuh"

#include "kernels.cuh"

void setup_and_execute_kernel(int argc, char **argv){

    BenchConfig config = parse_bench_args(argc, argv);

    const long M = config.M;
    const long N = config.N;
    const long K = config.K;
    const std::string kernel = config.kernel;

    const float alpha = config.alpha;
    const float beta = config.beta;

    __half *hA = nullptr;
    __half *hB = nullptr;
    float *hC = nullptr;
    float *hC_ref = nullptr;

    __half *dA  = nullptr;
    __half *dB  = nullptr;
    float *dC  = nullptr;
    float *dC_ref = nullptr;

    CUDA_CHECK(cudaMallocHost(&hA, sizeof(__half) * M * K));
    CUDA_CHECK(cudaMallocHost(&hB, sizeof(__half) * N * K));
    CUDA_CHECK(cudaMallocHost(&hC, sizeof(float) * M * N));
    CUDA_CHECK(cudaMallocHost(&hC_ref, sizeof(float) * M * N));
    
    CUDA_CHECK(cudaMalloc(&dA, sizeof(__half) * M * K));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(__half) * N * K));
    CUDA_CHECK(cudaMalloc(&dC, sizeof(float) * M * N));
    CUDA_CHECK(cudaMalloc(&dC_ref, sizeof(float) * M * N));

    randomize_matrix(hA, M*K);
    randomize_matrix(hB, N*K);

    CUDA_CHECK(cudaMemcpy(dA, hA, sizeof(__half) * M * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sizeof(__half) * N * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, sizeof(float) * M * N));
    CUDA_CHECK(cudaMemset(dC_ref, 0, sizeof(float) * M * N));

    cublasHandle_t handle;
    if(cublasCreate(&handle)) {
        printf("Create cublas handle error.\n");
        exit(EXIT_FAILURE);
    }

    std::function<void()> cublas_kernel = std::bind(runCublasFP16TensorOp, handle, M, N, K, dA, dB, dC_ref,
    alpha, beta);
    std::function<void()> custom_wmma_kernel = std::bind(run_custom_wmma_kernel, kernel, dA, dB, dC, M, N, K, 
    alpha, beta);

    cublas_kernel();
    custom_wmma_kernel();
    cudaDeviceSynchronize();

    CUDA_CHECK(cudaMemcpy(hC, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hC_ref, dC_ref, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

    if(!verify_matrix(hC_ref, hC, M*N)) return;

    const float avg_ms_cublas_kernel = run_warmup_then_timed(cublas_kernel, 20, 200);
    double gflop_cublas_kernel = gflops(M, N, K, avg_ms_cublas_kernel);
    BenchResult cublas_result = {.kernel_name="cublas", .M = M, .N = N, .K = K, .iters= 200, 
        .avg_ms = avg_ms_cublas_kernel, .gflops_val = gflop_cublas_kernel};

    const float avg_ms_custom_wmma_kernel = run_warmup_then_timed(custom_wmma_kernel, 20, 200);
    double gflop_custom_wmma_kernel = gflops(M, N, K, avg_ms_custom_wmma_kernel);
    BenchResult custom_wmma_result = {.kernel_name="wmma", .M = M, .N = N, .K = K, .iters= 200, 
        .avg_ms = avg_ms_custom_wmma_kernel, .gflops_val = gflop_custom_wmma_kernel};

    report_result(cublas_result, "results/benchmark.csv");
    report_result(custom_wmma_result, "results/benchmark.csv");

}

int main(int argc, char **argv){
    setup_and_execute_kernel(argc, argv);
    return 1;
}