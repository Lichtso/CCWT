#include <math.h>
#include <complex.h>
#include <stdlib.h>
#include <stdio.h>

struct fftw_plan_struct;
typedef struct fftw_plan_s* fftw_plan;
struct ccwt_data {
    complex double *input, *output;
    fftw_plan input_plan, output_plan;
    unsigned int sample_count, padding, width, height;
    double frequency_scale, frequency_range, frequency_offset, frequency_basis, deviation;
};

void gabor_wavelet(unsigned int sample_count, complex double* kernel, double center_frequency, double deviation);
void convolve(unsigned int sample_count, complex double* signal, complex double* kernel);
int ccwt_init(struct ccwt_data* ccwt);
int ccwt_cleanup(struct ccwt_data* ccwt);
int ccwt_calculate(struct ccwt_data* ccwt, void* user_data, int(*callback)(struct ccwt_data*, void*, unsigned int));
int ccwt_render_png(struct ccwt_data* ccwt, FILE* file, unsigned char mode);
