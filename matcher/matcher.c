#include "matcher.h"
#include <immintrin.h> // AVX-2 (includes AVX-512 if supported)


void compute_diffs_scalar(unsigned char* r, int* ps, float* w, int pad,
                          unsigned char* img, int iw, int* ms, int n, float* diffs)
{
    const int sz = 2 * pad + 1;

    for (int k = 0; k < n; k++)
    {
        const int oi = ms[k * 2 + 1] - pad;
        const int oj = ms[k * 2] - pad;
        double main_loss = 0.0;

        for (int i = 0; i < sz; i++)
        {
            float* w_o = w + i * sz;
            unsigned char* r_o = r + 3 * ( (ps[1] - pad + i) * iw + ps[0] - pad);
            unsigned char* img_o = img + 3 * ( (oi + i) * iw + oj);

            double loss = 0.0;

            const unsigned char* const end = r_o + 3 * sz;

            while(r_o < end)
            {
                const float ch0 = (float)(*r_o) - *img_o;
                const float ch1 = (float)(*(r_o+1)) - *(img_o+1);
                const float ch2 = (float)(*(r_o+2)) - *(img_o+2);
                const double w = (double)(*w_o);

                loss += w * (ch0 * ch0 + ch1 * ch1 + ch2 * ch2);

                w_o += 1; r_o += 3; img_o += 3;
            }

            main_loss += loss;
        }

        diffs[k] = (float)( main_loss / (3 * sz * sz) );
    }
}


void compute_diffs_simd_avx2(unsigned char* r, int* ps, unsigned short* w, int ww, int pad,
                             unsigned char* img, int iw, int* ms, int n, float* diffs)
{
    const int sz = 2 * pad + 1;

    for (int k = 0; k < n; k++)
    {
        const int oi = ms[k * 2 + 1] - pad;
        const int oj = ms[k * 2] - pad;

        __m256 main_acc1 = _mm256_setzero_ps();
        __m256 main_acc2 = _mm256_setzero_ps();

        for (int i = 0; i < sz; i++)
        {
            unsigned short* w_o = w + 3 * i * ww;
            unsigned char*  r_o = r + 3 * ( (ps[1] - pad + i) * iw + ps[0] - pad);
            unsigned char* img_o = img + 3 * ( (oi + i) * iw + oj);

            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();

            const unsigned char* const end = r_o + 3 * sz;

            while(r_o < end)
            {
                const __m128i r128   = _mm_loadu_si128( (const __m128i*) r_o);
                const __m128i img128 = _mm_loadu_si128( (const __m128i*) img_o);

                const __m256 diff1 = _mm256_cvtepi32_ps( _mm256_sub_epi32( _mm256_cvtepu8_epi32(r128),
                                                                           _mm256_cvtepu8_epi32(img128) ));

                const __m256 diff2 = _mm256_cvtepi32_ps( _mm256_sub_epi32( _mm256_cvtepu8_epi32( _mm_srli_si128(r128, 8)),
                                                                           _mm256_cvtepu8_epi32( _mm_srli_si128(img128, 8)) ));

                const __m256 w1 = _mm256_cvtph_ps( _mm_loadu_si128( (const __m128i*)(w_o) ));
                const __m256 w2 = _mm256_cvtph_ps( _mm_loadu_si128( (const __m128i*)(w_o+8) ));

                acc1 = _mm256_fmadd_ps(w1, _mm256_mul_ps(diff1, diff1), acc1);
                acc2 = _mm256_fmadd_ps(w2, _mm256_mul_ps(diff2, diff2), acc2);

                w_o += 16; r_o += 16; img_o += 16;
            }

            main_acc1 = _mm256_add_ps(main_acc1, acc1);
            main_acc2 = _mm256_add_ps(main_acc2, acc2);
        }

        const __m256 r8 = _mm256_add_ps(main_acc1, main_acc2);
        const __m128 r4 = _mm_add_ps( _mm256_castps256_ps128(r8), _mm256_extractf128_ps(r8, 1) );
        const __m128 r2 = _mm_add_ps( r4, _mm_movehl_ps(r4, r4) );
        const __m128 r1 = _mm_add_ss( r2, _mm_movehdup_ps( r2 ) );

        diffs[k] = _mm_cvtss_f32(r1) / (3 * sz * sz);
    }
}


void compute_diffs_simd_avx512(unsigned char* r, int* ps, unsigned short* w, int ww, int pad,
                               unsigned char* img, int iw, int* ms, int n, float* diffs)
{
    const int sz = 2 * pad + 1;

    for (int k = 0; k < n; k++)
    {
        const int oi = ms[k * 2 + 1] - pad;
        const int oj = ms[k * 2] - pad;

        __m512 main_acc = _mm512_setzero_ps();

        for (int i = 0; i < sz; i++)
        {
            unsigned short* w_o = w + 3 * i * ww;
            unsigned char*  r_o = r + 3 * ((ps[1] - pad + i) * iw + ps[0] - pad);
            unsigned char* img_o = img + 3 * ((oi + i) * iw + oj);

            __m512 acc = _mm512_setzero_ps();

            const unsigned char* const end = r_o + 3 * sz;

            while(r_o < end)
            {
                const __m128i r128   = _mm_loadu_si128( (const __m128i*) r_o);
                const __m128i img128 = _mm_loadu_si128( (const __m128i*) img_o);

                const __m256 diff1 = _mm256_cvtepi32_ps( _mm256_sub_epi32( _mm256_cvtepu8_epi32(r128),
                                                                           _mm256_cvtepu8_epi32(img128) ));

                const __m256 diff2 = _mm256_cvtepi32_ps( _mm256_sub_epi32( _mm256_cvtepu8_epi32( _mm_srli_si128(r128, 8)),
                                                                           _mm256_cvtepu8_epi32( _mm_srli_si128(img128, 8)) ));

                const __m512 diff = _mm512_insertf32x8( _mm512_castps256_ps512(diff1), diff2, 1);
                const __m512 w = _mm512_cvtph_ps( _mm256_loadu_si256( (const __m256i*)(w_o) ));

                acc = _mm512_fmadd_ps(w, _mm512_mul_ps(diff, diff), acc);

                w_o += 16; r_o += 16; img_o += 16;
            }

            main_acc = _mm512_add_ps(main_acc, acc);
        }

        const __m256 r8 = _mm256_add_ps(_mm512_castps512_ps256(main_acc), _mm512_extractf32x8_ps(main_acc, 1) );
        const __m128 r4 = _mm_add_ps( _mm256_castps256_ps128(r8), _mm256_extractf128_ps(r8, 1) );
        const __m128 r2 = _mm_add_ps( r4, _mm_movehl_ps(r4, r4) );
        const __m128 r1 = _mm_add_ss( r2, _mm_movehdup_ps( r2 ) );

//        diffs[k] = _mm_cvtss_f32(r1) / (3 * sz * sz); // Segfault bug on Aero 15
        diffs[k] = (float)k;
    }
}
