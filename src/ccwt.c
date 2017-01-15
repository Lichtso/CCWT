#include <ccwt.h>
#include <fftw3.h>

void convolve(unsigned int sample_count, complex double* signal, complex double* kernel) {
    double scaleFactor = 1.0/(double)sample_count;
    for(unsigned int i = 0; i < sample_count; ++i)
        signal[i] *= kernel[i]*scaleFactor;
}

void gabor_wavelet(unsigned int sample_count, complex double* kernel, double center_frequency, double deviation) {
    deviation = 1.0/deviation;
    for(unsigned int i = 0; i < sample_count/2+1; ++i) {
        double f = (i-center_frequency)*deviation;
        kernel[i] = exp(-f*f);
    }
    for(unsigned int i = sample_count/2+1; i < sample_count; ++i) {
        double f = (sample_count-i+center_frequency)*deviation;
        kernel[i] = exp(-f*f);
    }
}

int ccwt_calculate(struct ccwt_data* ccwt, void* user_data, int(*callback)(struct ccwt_data*, void*, unsigned int)) {
    fftw_plan input_plan = fftw_plan_dft_1d(ccwt->sample_count, (fftw_complex*)ccwt->input, (fftw_complex*)ccwt->input, FFTW_FORWARD, FFTW_ESTIMATE);
    if(!input_plan)
        return -1;
    fftw_plan output_plan = fftw_plan_dft_1d(ccwt->sample_count, (fftw_complex*)ccwt->output, (fftw_complex*)ccwt->output, FFTW_BACKWARD, FFTW_MEASURE);
    if(!output_plan) {
        fftw_destroy_plan(input_plan);
        return -1;
    }
    int return_value = 0;
    fftw_execute(input_plan);
    for(unsigned int y = 0; y < ccwt->height && !return_value; ++y) {
        double frequency = ccwt->frequency_scale*(1.0-(double)y/(ccwt->height-1))+ccwt->frequency_offset;
        if(ccwt->frequency_basis > 0.0)
            frequency = pow(ccwt->frequency_basis, frequency);
        gabor_wavelet(ccwt->sample_count, ccwt->output, frequency*ccwt->sample_count/ccwt->width, ccwt->deviation);
        convolve(ccwt->sample_count, ccwt->output, ccwt->input);
        fftw_execute(output_plan);
        return_value = callback(ccwt, user_data, y);
    }
    fftw_destroy_plan(input_plan);
    fftw_destroy_plan(output_plan);
    return return_value;
}
