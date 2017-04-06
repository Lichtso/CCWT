#ifndef _MATH_DEFINES_DEFINED
#define _MATH_DEFINES_DEFINED
#define M_E        2.71828182845904523536
#define M_LOG2E    1.44269504088896340736
#define M_LOG10E   0.434294481903251827651
#define M_LN2      0.693147180559945309417
#define M_LN10     2.30258509299404568402
#define M_PI       3.14159265358979323846
#define M_PI_2     1.57079632679489661923
#define M_PI_4     0.785398163397448309616
#define M_1_PI     0.318309886183790671538
#define M_2_PI     0.636619772367581343076
#define M_2_SQRTPI 1.12837916709551257390
#define M_SQRT2    1.41421356237309504880
#define M_SQRT1_2  0.707106781186547524401
#endif

#include <math.h>
#include <complex.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef complex
typedef complex double complex_type;
#else
typedef _Dcomplex complex_type;
#endif

struct ccwt_data;

struct ccwt_thread_data {
    int return_value;
    unsigned int begin_y, end_y, thread_index;
    void* pthread;
    complex_type* output;
    struct ccwt_data* ccwt;
};

struct ccwt_data {
    unsigned int thread_count, height,
                 input_sample_count, input_width, input_padding,
                 output_sample_count, output_width, output_padding;
    complex_type* input;
    double* frequency_band;
    struct ccwt_thread_data* threads;
    void *fftw_plan, *user_data;
    int(*callback)(struct ccwt_thread_data* thread, unsigned int y);
};

void ccwt_frequency_band(double* frequency_band, unsigned int frequencies_count, double frequency_range, double frequency_offset, double frequency_basis, double deviation);
complex_type* ccwt_fft(unsigned int input_width, unsigned int input_padding, unsigned int thread_count, void* input, unsigned char input_type);
int ccwt_numeric_output(struct ccwt_data* ccwt);

enum ccwt_render_mode {
#define macro_wrapper(name) name,
#include <render_mode.h>
};

int ccwt_render_png(struct ccwt_data* ccwt, FILE* file, unsigned char mode, double log_factor);
