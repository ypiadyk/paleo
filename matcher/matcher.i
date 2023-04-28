%module matcher_module
%{
    /* the resulting C file should be built as a python extension */
    #define SWIG_FILE_WITH_INIT
    /*  Includes the header in the wrapper code */
    #include "matcher.h"
%}
/*  Parse the header file to generate wrappers */
/* %include "matcher.h" */

/*  include the numpy typemaps */
%include "numpy.i"
/*  need this for correct module initialization */
%init %{
    import_array();
%}
/*  typemaps for the two arrays, the second will be modified in-place */
//%apply (double* IN_ARRAY1, int DIM1) {(double * in_array, int size_in)}
//%apply (double* INPLACE_ARRAY1, int DIM1) {(double * out_array, int size_out)}
%apply (int DIM1, int DIM2, unsigned int * INPLACE_ARRAY2) {(int height, int width, unsigned int * mask)};
%apply (int DIM1, int DIM2, double * INPLACE_ARRAY2) {(int height_2, int width_2, double * data)};

%apply (int DIM1, int DIM2, int DIM3, unsigned char * INPLACE_ARRAY3) {(int rh, int rw, int rd, unsigned char * r)};
%apply (int DIM1, int DIM2, float * INPLACE_ARRAY2) {(int wh, int ww, float * w)};
%apply (int DIM1, int DIM2, int DIM3, unsigned char * INPLACE_ARRAY3) {(int ih, int iw, int id, unsigned char * img)};
%apply (int DIM1, int DIM2, int * INPLACE_ARRAY2) {(int mh, int mw, int * ms)};
%apply (int DIM1, float * INPLACE_ARRAY1) {(int dh, float * diffs)};

/*  Wrapper for cos_doubles that massages the types */
%inline %{
    /*  takes as input two numpy arrays */
    /*void cos_doubles_func(double * in_array, int size_in, double * out_array, int size_out) {
        cos_doubles(in_array, out_array, size_in);*/
    int in_bounds_func(int x, int y, int w, int h) {
        return in_bounds(x, y, w, h);
    }
%}

%inline %{
    void label(int height, int width, unsigned int * mask, int height_2, int width_2, double * data, double thr)
    {
        if (height != height_2 || width != width_2) printf("Matcher Warning: Dimensions of input arrays don't match so don't expect correct results!\n");
        // TODO: throw("Dimensions of input arrays don't match!");

        image img;
        img.max_grey_value = 255;
        img.width = (unsigned int)width;
        img.height = (unsigned int)height;

        img.mask = (unsigned int **) malloc(height * sizeof(unsigned int *));
        img.data = (double **) malloc(height * sizeof(double *));

        for (int i=0; i<height; i++) {
            img.mask[i] = mask + i * width;
            img.data[i] = data + i * width;
        }

        label_components(&img, thr);

        free(img.mask);
        free(img.data);
    }
%}

%inline %{
    void compute_diffs(int rh, int rw, int rd, unsigned char * r,
                       int wh, int ww, float * w,
                       int ih, int iw, int id, unsigned char * img,
                       int mh, int mw, int * ms,
                       int pad,
                       int dh, float* diffs)
    {
//         printf("In\n");
        compute_diffs_c(r, w, pad, img, iw, ih, ms, mh, diffs);
    }
%}
