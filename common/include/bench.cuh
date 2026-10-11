#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <functional>
#include <cuda_runtime.h>
#include "matrix_utils.cuh"

// Fixes two weaknesses in SGEMM_CUDA's driver: it drew M/N/K randomly per run
// (no way to benchmark one specific size or a size sweep from the CLI), and its
// only "warmup" was the single correctness-check kernel launch before a fixed
// 50-iteration timed loop. Here warmup and timing are explicit, separate, and
// the problem size is always CLI-controlled.
struct BenchConfig {
  long M = 4096, N = 4096, K = 4096;
  int warmup = 5;
  int iters = 50;
  float alpha = 0.5f, beta = 3.0f;
  std::string csv;  // empty = stdout only
  std::string kernel = "";
};

inline BenchConfig parse_bench_args(int argc, char **argv) {
  BenchConfig cfg;
  for (int i = 1; i < argc; i++) {
    auto next = [&](const char *flag) -> const char * {
      if (i + 1 >= argc) {
        fprintf(stderr, "missing value for %s\n", flag);
        exit(EXIT_FAILURE);
      }
      return argv[++i];
    };
    if (!strcmp(argv[i], "--m")) cfg.M = atol(next("--m"));
    else if (!strcmp(argv[i], "--n")) cfg.N = atol(next("--n"));
    else if (!strcmp(argv[i], "--k")) cfg.K = atol(next("--k"));
    else if (!strcmp(argv[i], "--warmup")) cfg.warmup = atoi(next("--warmup"));
    else if (!strcmp(argv[i], "--iters")) cfg.iters = atoi(next("--iters"));
    else if (!strcmp(argv[i], "--alpha")) cfg.alpha = atof(next("--alpha"));
    else if (!strcmp(argv[i], "--beta")) cfg.beta = atof(next("--beta"));
    else if (!strcmp(argv[i], "--csv")) cfg.csv = next("--csv");
    else if (!strcmp(argv[i], "--kernel")) cfg.kernel = next("--kernel");
    else {
      fprintf(stderr, "unknown arg: %s\n", argv[i]);
      exit(EXIT_FAILURE);
    }
  }
  return cfg;
}

// Runs `launch` `warmup` times (untimed, discarded) then `iters` times wrapped in a
// single cudaEvent pair, returning average milliseconds per iteration.
inline float run_warmup_then_timed(const std::function<void()> &launch, int warmup,
                                    int iters) {
  for (int i = 0; i < warmup; i++) launch();
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t beg, end;
  CUDA_CHECK(cudaEventCreate(&beg));
  CUDA_CHECK(cudaEventCreate(&end));

  CUDA_CHECK(cudaEventRecord(beg));
  for (int i = 0; i < iters; i++) launch();
  CUDA_CHECK(cudaEventRecord(end));
  CUDA_CHECK(cudaEventSynchronize(end));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, beg, end));
  cudaEventDestroy(beg);
  cudaEventDestroy(end);
  return elapsed_ms / iters;
}

inline double gflops(long M, long N, long K, float avg_ms) {
  double flops = 2.0 * (double)M * (double)N * (double)K;
  return flops / (avg_ms * 1e6);  // ms -> s (1e-3) and flops -> Gflops (1e-9): net 1e6
}

struct BenchResult {
  std::string kernel_name, dtype;
  long M, N, K;
  int iters;
  float avg_ms;
  double gflops_val;
  // float max_abs_err, max_rel_err;
  // bool pass;
};

inline void report_result(const BenchResult &r, const std::string &csv_path) {
  // printf("[%s] M=%ld N=%ld K=%ld dtype=%s avg=%.6fms gflops=%.1f pass=%s\n",
  //        r.kernel_name.c_str(), r.M, r.N, r.K, r.dtype.c_str(), r.avg_ms,
  //        r.gflops_val, r.pass ? "yes" : "no");
  printf("[%s] M=%ld N=%ld K=%ld dtype=%s avg=%.6fms gflops=%.1f\n",
         r.kernel_name.c_str(), r.M, r.N, r.K, r.dtype.c_str(), r.avg_ms,
         r.gflops_val);
  if (csv_path.empty()) return;
  bool exists = (fopen(csv_path.c_str(), "r") != nullptr);
  FILE *f = fopen(csv_path.c_str(), "a");
  if (!f) return;
  if (!exists)
    fprintf(f, "kernel,M,N,K,dtype,iters,avg_ms,gflops,max_abs_err,max_rel_err,pass\n");
  // fprintf(f, "%s,%ld,%ld,%ld,%s,%d,%.6f,%.3f,%.6f,%.6f,%d\n", r.kernel_name.c_str(),
  //         r.M, r.N, r.K, r.dtype.c_str(), r.iters, r.avg_ms, r.gflops_val,
  //         r.max_abs_err, r.max_rel_err, r.pass ? 1 : 0);
  fprintf(f, "%s,%ld,%ld,%ld,%s,%d,%.6f,%.3f\n", r.kernel_name.c_str(),
          r.M, r.N, r.K, r.dtype.c_str(), r.iters, r.avg_ms, r.gflops_val);
  fclose(f);
}
