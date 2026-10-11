#pragma once

#include <cmath>
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

inline float nt_to_float(float v) { return v; }
inline float nt_to_float(__half v) { return __half2float(v); }
inline float nt_to_float(__nv_bfloat16 v) { return __bfloat162float(v); }

// Combined absolute+relative tolerance check: |ref - out| <= abs_tol + rel_tol*|ref|.
// A fixed absolute tolerance (as SGEMM_CUDA used for fp32) is the wrong check for
// fp16-accumulate kernels, where error scales with the magnitude of each entry.
template <typename T>
bool verify_matrix(const float *ref, const T *out, long n,
                    float rel_tol = 1e-2f, float abs_tol = 1e-3f,
                    long max_report = 5) {
  bool ok = true;
  long reported = 0;
  float max_abs_err = 0.0f, max_rel_err = 0.0f;
  for (long i = 0; i < n; i++) {
    float r = ref[i];
    float o = nt_to_float(out[i]);
    float diff = std::fabs(r - o);
    float allowed = abs_tol + rel_tol * std::fabs(r);
    max_abs_err = std::max(max_abs_err, diff);
    if (std::fabs(r) > 1e-6f) max_rel_err = std::max(max_rel_err, diff / std::fabs(r));
    if (std::isnan(o) || diff > allowed) {
      ok = false;
      if (reported < max_report) {
        printf("Divergence at %ld: ref=%.6f out=%.6f diff=%.6f allowed=%.6f\n",
               i, r, o, diff, allowed);
        reported++;
      }
    }
  }
  printf("verify_matrix: max_abs_err=%.6f max_rel_err=%.6f -> %s\n", max_abs_err,
         max_rel_err, ok ? "PASS" : "FAIL");
  return ok;
}
