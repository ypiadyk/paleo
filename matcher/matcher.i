%module matcher_module
%{
    /* the resulting C file should be built as a python extension */
    #define SWIG_FILE_WITH_INIT
    #include "matcher.h"
%}

/*  include the numpy typemaps */
%include "numpy.i"
/*  need this for correct module initialization */
%init %{
    import_array();
%}
/* typemaps for user arrays */

%apply (int DIM1, int DIM2, int DIM3, unsigned char * INPLACE_ARRAY3) {(int rh, int rw, int rd, unsigned char * r)};
%apply (int DIM1, int DIM2, int DIM3, unsigned char * INPLACE_ARRAY3) {(int ih, int iw, int id, unsigned char * img)};
%apply (int DIM1, int DIM2, int DIM3, unsigned short * INPLACE_ARRAY3) {(int wh, int ww, int wd, unsigned short * w)};
%apply (int DIM1, int DIM2, float * INPLACE_ARRAY2) {(int wh, int ww, float * w)};
%apply (int DIM1, int * INPLACE_ARRAY1) {(int ph, int * ps)};
%apply (int DIM1, int DIM2, int * INPLACE_ARRAY2) {(int mh, int mw, int * ms)};
%apply (int DIM1, double * INPLACE_ARRAY1) {(int dh, double * diffs)};

%inline %{
    void compute_diffs(int rh, int rw, int rd, unsigned char * r,
                       int wh, int ww, float * w,
                       int ph, int * ps,
                       int ih, int iw, int id, unsigned char * img,
                       int mh, int mw, int * ms,
                       int pad,
                       int dh, double* diffs)
    {
        compute_diffs_scalar(r, ps, w, pad, img, iw, ms, mh, diffs);
    }
%}

%inline %{
    void compute_diffs_avx2(int rh, int rw, int rd, unsigned char * r,
                            int wh, int ww, int wd, unsigned short * w,
                            int ph, int * ps,
                            int ih, int iw, int id, unsigned char * img,
                            int mh, int mw, int * ms,
                            int pad,
                            int dh, double* diffs)
    {
        compute_diffs_simd_avx2(r, ps, w, ww, pad, img, iw, ms, mh, diffs);
    }
%}

%inline %{
    void compute_diffs_avx512(int rh, int rw, int rd, unsigned char * r,
                              int wh, int ww, int wd, unsigned short * w,
                              int ph, int * ps,
                              int ih, int iw, int id, unsigned char * img,
                              int mh, int mw, int * ms,
                              int pad,
                              int dh, double* diffs)
    {
        compute_diffs_simd_avx512(r, ps, w, ww, pad, img, iw, ms, mh, diffs);
    }
%}
