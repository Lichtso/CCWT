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
