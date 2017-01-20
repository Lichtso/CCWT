#include <ccwt.h>
#include <fftw3.h>

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

void convolve(unsigned int sample_count, complex double* signal, complex double* kernel) {
    double scaleFactor = 1.0/(double)sample_count;
    for(unsigned int i = 0; i < sample_count; ++i)
        signal[i] *= kernel[i]*scaleFactor;
}

int ccwt_init(struct ccwt_data* ccwt) {
    unsigned int input_size = ccwt->sample_count;
    ccwt->sample_count += 2*ccwt->padding;
    ccwt->frequency_scale = (double)ccwt->sample_count/input_size;
    ccwt->input = (complex double*)fftw_malloc(sizeof(fftw_complex)*ccwt->sample_count);
    ccwt->output = (complex double*)fftw_malloc(sizeof(fftw_complex)*ccwt->sample_count);
    ccwt->input_plan = fftw_plan_dft_1d(ccwt->sample_count, (fftw_complex*)ccwt->input, (fftw_complex*)ccwt->input, FFTW_FORWARD, FFTW_ESTIMATE);
    ccwt->output_plan = fftw_plan_dft_1d(ccwt->sample_count, (fftw_complex*)ccwt->output, (fftw_complex*)ccwt->output, FFTW_BACKWARD, FFTW_MEASURE);
    if(!ccwt->input || !ccwt->output || !ccwt->input_plan || !ccwt->output_plan)
        return -1;
    for(unsigned int i = 0; i < ccwt->padding; ++i)
        ccwt->input[i] = ccwt->input[ccwt->sample_count-i-1] = 0;
    return 0;
}

int ccwt_cleanup(struct ccwt_data* ccwt) {
    fftw_free(ccwt->input);
    fftw_free(ccwt->output);
    fftw_destroy_plan(ccwt->input_plan);
    fftw_destroy_plan(ccwt->output_plan);
    return 0;
}

int ccwt_calculate(struct ccwt_data* ccwt, void* user_data, int(*callback)(struct ccwt_data*, void*, unsigned int)) {
    int return_value = 0;
    fftw_execute(ccwt->input_plan);
    for(unsigned int y = 0; y < ccwt->height && !return_value; ++y) {
        double frequency = ccwt->frequency_range*(1.0-(double)y/(ccwt->height-1))+ccwt->frequency_offset;
        if(ccwt->frequency_basis > 0.0)
            frequency = pow(ccwt->frequency_basis, frequency);
        gabor_wavelet(ccwt->sample_count, ccwt->output, frequency*ccwt->frequency_scale, ccwt->deviation);
        convolve(ccwt->sample_count, ccwt->output, ccwt->input);
        fftw_execute(ccwt->output_plan);
        return_value = callback(ccwt, user_data, y);
    }
    return return_value;
}
