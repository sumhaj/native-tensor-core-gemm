#pragma once

#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t _e = (call);                                                  \
    if (_e != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,           \
              cudaGetErrorString(_e));                                        \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

// Ported from SGEMM_CUDA: gettimeofday-seeded RNG avoids the low time-precision
// collisions that srand(time(NULL)) hits when launching kernels back-to-back.
inline void randomize_matrix(__half *mat, long n) {
  struct timeval time {};
  gettimeofday(&time, nullptr);
  srand(time.tv_usec);
  for (long i = 0; i < n; i++) {
    float tmp = (float)(rand() % 5) + 0.01f * (float)(rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : -tmp;
    mat[i] = __float2half(tmp);
  }
}

inline void float_to_half(const float *src, __half *dst, long n) {
  for (long i = 0; i < n; i++) dst[i] = __float2half(src[i]);
}

inline void float_to_bf16(const float *src, __nv_bfloat16 *dst, long n) {
  for (long i = 0; i < n; i++) dst[i] = __float2bfloat16(src[i]);
}

inline float half_to_float(__half h) { return __half2float(h); }
inline float bf16_to_float(__nv_bfloat16 h) { return __bfloat162float(h); }
