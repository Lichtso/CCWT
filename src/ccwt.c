#include <ccwt.h>
#include <fftw3.h>
#include <math.h>

void gabor_wavelet(unsigned int sample_count, complex double* kernel, double center_frequency, double deviation) {
    deviation = 1.0/sqrt(deviation);
    unsigned int half_sample_count = sample_count>>1;
    for(unsigned int i = 0; i < sample_count; ++i) {
        double f = fabs(i-center_frequency);
        f = half_sample_count-fabs(f-half_sample_count);
        f *= deviation;
        kernel[i] = exp(-f*f);
    }
}

int ccwt_init(struct ccwt_data* ccwt) {
    ccwt->input_sample_count = ccwt->input_width+2*ccwt->input_padding;
    ccwt->output_sample_count = ccwt->output_width*((double)ccwt->input_sample_count/ccwt->input_width);
    ccwt->output_padding = ccwt->input_padding*((double)ccwt->output_width/ccwt->input_width);
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

void ccwt_generate_frequencies(double* frequencies, unsigned int frequencies_count, double frequency_range, double frequency_offset, double frequency_basis, double deviation) {
    if(frequency_range == 0.0)
        frequency_range = frequencies_count/2;
    for(unsigned int y = 0; y < frequencies_count; ++y) {
        double frequency = frequency_range*(1.0-(double)y/frequencies_count)+frequency_offset,
               frequency_derivative = frequency_range/frequencies_count;
        if(frequency_basis > 0.0) {
            frequency = pow(frequency_basis, frequency);
            frequency_derivative *= log(frequency_basis)*frequency;
        }
        frequencies[y*2  ] = frequency;
        frequencies[y*2+1] = frequency_derivative*deviation;
    }
}

int ccwt_calculate(struct ccwt_data* ccwt, void* user_data, int(*callback)(struct ccwt_data*, void*, unsigned int)) {
    int return_value = 0;
    double scale_factor = 1.0/(double)ccwt->input_sample_count,
           padding_correction = (double)ccwt->input_sample_count/ccwt->input_width;
    fftw_execute(ccwt->input_plan);
    for(unsigned int y = 0; y < ccwt->height && !return_value; ++y) {
        double frequency = ccwt->frequencies[y*2  ]*padding_correction,
               deviation = ccwt->frequencies[y*2+1]*ccwt->output_sample_count*padding_correction;
        gabor_wavelet(ccwt->input_sample_count, ccwt->output, frequency, deviation);
        for(unsigned int i = 0; i < ccwt->output_sample_count; ++i)
            ccwt->output[i] = ccwt->output[i]*ccwt->input[i]*scale_factor;
        if(ccwt->output_sample_count < ccwt->input_sample_count) {
            unsigned int rest = ccwt->input_sample_count%ccwt->output_sample_count, cut_index = ccwt->input_sample_count-rest;
            for(unsigned int chunk_index = ccwt->output_sample_count; chunk_index < cut_index; chunk_index += ccwt->output_sample_count)
                for(unsigned int i = 0; i < ccwt->output_sample_count; ++i)
                    ccwt->output[i] += ccwt->output[chunk_index+i]*ccwt->input[chunk_index+i]*scale_factor;
            for(unsigned int i = 0; i < rest; ++i)
                ccwt->output[i] += ccwt->output[cut_index+i]*ccwt->input[cut_index+i]*scale_factor;
        }
        fftw_execute(ccwt->output_plan);
        return_value = callback(ccwt, user_data, y);
    }
    return return_value;
}
