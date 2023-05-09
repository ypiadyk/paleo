#ifndef __MATCHER__
#define __MATCHER__

#include <stdlib.h>

void compute_diffs_scalar(unsigned char* r, int* ps, float* w, int pad,
                          unsigned char* img, int iw, int* ms, int n, double* diffs);

void compute_diffs_simd_avx2(unsigned char* r, int* ps, unsigned short* w, int ww, int pad,
                             unsigned char* img, int iw, int* ms, int n, double* diffs);

void compute_diffs_simd_avx512(unsigned char* r, int* ps, unsigned short* w, int ww, int pad,
                               unsigned char* img, int iw, int* ms, int n, double* diffs);

#endif // __MATCHER__
