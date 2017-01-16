#include <math.h>
#include <complex.h>
#include <stdlib.h>
#include <stdio.h>

struct ccwt_data {
    complex double *input, *output;
    unsigned int sample_count, padding, width, height;
    double frequency_range, frequency_offset, frequency_basis, deviation;
};

void convolve(unsigned int sample_count, complex double* signal, complex double* kernel);
void gabor_wavelet(unsigned int sample_count, complex double* kernel, double center_frequency, double deviation);
int ccwt_calculate(struct ccwt_data* ccwt, void* user_data, int(*callback)(struct ccwt_data*, void*, unsigned int));
int ccwt_render_png(struct ccwt_data* ccwt, FILE* file, unsigned char mode);
