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

void downsample(unsigned int dst_sample_count, unsigned int src_sample_count, complex double* signal) {
    if(dst_sample_count < src_sample_count)
        for(unsigned int i = dst_sample_count; i < src_sample_count; ++i)
            signal[i%dst_sample_count] += signal[i];
}

int ccwt_init(struct ccwt_data* ccwt) {
    ccwt->input_sample_count = ccwt->input_width+2*ccwt->input_padding;
    ccwt->padding_correction = (double)ccwt->input_sample_count/ccwt->input_width;
    ccwt->output_sample_count = ccwt->output_width*ccwt->padding_correction;
    ccwt->output_padding = ccwt->input_padding*(double)ccwt->output_width/ccwt->input_width;
    ccwt->input = (complex double*)fftw_malloc(sizeof(fftw_complex)*ccwt->input_sample_count);
    ccwt->output = (complex double*)fftw_malloc(sizeof(fftw_complex)*ccwt->input_sample_count);
    ccwt->input_plan = fftw_plan_dft_1d(ccwt->input_sample_count, (fftw_complex*)ccwt->input, (fftw_complex*)ccwt->input, FFTW_FORWARD, FFTW_ESTIMATE);
    ccwt->output_plan = fftw_plan_dft_1d(ccwt->output_sample_count, (fftw_complex*)ccwt->output, (fftw_complex*)ccwt->output, FFTW_BACKWARD, FFTW_MEASURE);
    if(!ccwt->input || !ccwt->output || !ccwt->input_plan || !ccwt->output_plan)
        return -1;
    for(unsigned int i = 0; i < ccwt->input_padding; ++i)
        ccwt->input[i] = ccwt->input[ccwt->input_sample_count-i-1] = 0;
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
        gabor_wavelet(ccwt->input_sample_count, ccwt->output, frequency*ccwt->padding_correction, ccwt->deviation);
        convolve(ccwt->input_sample_count, ccwt->output, ccwt->input);
        downsample(ccwt->output_sample_count, ccwt->input_sample_count, ccwt->output);
        fftw_execute(ccwt->output_plan);
        return_value = callback(ccwt, user_data, y);
    }
    return return_value;
}
