#include <math.h>
#include <complex.h>
#include <stdlib.h>
#include <fftw3.h>
#include <png.h>

void convolve(unsigned int sampleCount, complex double* signal, complex double* kernel) {
    double scaleFactor = 1.0/(double)sampleCount;
    for(unsigned long i = 0; i < sampleCount; ++i)
        signal[i] *= kernel[i]*scaleFactor;
}

void gaborWavelet(unsigned int sampleCount, complex double* kernel, double f0, double a) {
    a = 1.0/a;
    for(unsigned long i = 0; i < sampleCount/2; ++i) {
        double f = (i-f0)*a;
        kernel[i] = exp(-f*f);
    }
    for(unsigned long i = sampleCount/2; i < sampleCount; ++i) {
        double f = (sampleCount-i+f0)*a;
        kernel[i] = exp(-f*f);
    }
}

void toFrequencyDomain(unsigned int sampleCount, complex double* frequencyDomain, complex double* timeDomain) {
    fftw_plan plan = fftw_plan_dft_1d(sampleCount, (fftw_complex*)timeDomain, (fftw_complex*)frequencyDomain, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
}

void toTimeDomain(unsigned int sampleCount, complex double* timeDomain, complex double* frequencyDomain) {
    fftw_plan plan = fftw_plan_dft_1d(sampleCount, (fftw_complex*)frequencyDomain, (fftw_complex*)timeDomain, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
}

complex double* allocateFftw(unsigned int sampleCount) {
    return (complex double*)fftw_malloc(sizeof(complex double)*sampleCount);
}

void debugPrintFftw(unsigned int sampleCount, complex double* signal) {
    for(unsigned long i = 0; i < sampleCount; ++i)
        printf("% 2.6f % 2.6f % 2.6f\n", creal(signal[i]), cimag(signal[i]), cabs(signal[i]));
}

const unsigned int maxColorFactor = 255;

void writePixelHsv(unsigned char* pixel, double H, double S, double V) {
    unsigned char h = H*6;
    double f = H*6-h, p = V*(1-S), q = V*(1-S*f), t = V*(1-(S*(1-f)));
    switch(h) {
        default:
            pixel[0] = V*maxColorFactor;
            pixel[1] = t*maxColorFactor;
            pixel[2] = p*maxColorFactor;
        break;
        case 1:
            pixel[0] = q*maxColorFactor;
            pixel[1] = V*maxColorFactor;
            pixel[2] = p*maxColorFactor;
        break;
        case 2:
            pixel[0] = p*maxColorFactor;
            pixel[1] = V*maxColorFactor;
            pixel[2] = t*maxColorFactor;
        break;
        case 3:
            pixel[0] = p*maxColorFactor;
            pixel[1] = q*maxColorFactor;
            pixel[2] = V*maxColorFactor;
        break;
        case 4:
            pixel[0] = t*maxColorFactor;
            pixel[1] = p*maxColorFactor;
            pixel[2] = V*maxColorFactor;
        break;
        case 5:
            pixel[0] = V*maxColorFactor;
            pixel[1] = p*maxColorFactor;
            pixel[2] = q*maxColorFactor;
        break;
    }
}

int main(int agrc, const char** argv) {
    unsigned int mode = 5, border = 50, width = 1024*4, height = 1024*4, sampleCount = width+2*border;

    complex double* input = allocateFftw(sampleCount);
    complex double* output = allocateFftw(sampleCount);

    for(unsigned long i = 0; i < border; ++i) {
        input[i] = 0;
        input[width+border+i] = 0;
    }
    for(unsigned long i = 0; i < width; ++i) {
        double t = 2.0*M_PI*i/width;
        t *= pow(1.009, 0.07*i+100.0);
        input[border+i] = cos(t)+_Complex_I*sin(t);
    }
    toFrequencyDomain(sampleCount, input, input);

    unsigned char* outputRow = (unsigned char*)malloc(width*3);
    FILE* outputFile = fopen("out.png", "wb");
    if(!outputFile) {
        puts("Could not open output file");
        return 1;
    }
    png_structp outputPng = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop outputPngInfo = png_create_info_struct(outputPng);
    if(setjmp(png_jmpbuf(outputPng))) {
        puts("Error during png encoding");
        return 1;
    }
    png_init_io(outputPng, outputFile);
    png_set_IHDR(outputPng, outputPngInfo, width, height,
                 8, (mode < 4) ? PNG_COLOR_TYPE_GRAY: PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
    png_write_info(outputPng, outputPngInfo);

    double borderFactor = 1.0/(1.0-2.0*border/width);
    for(unsigned long y = 0; y < height; ++y) {
        double frequency = borderFactor*pow(2.8, 4.0-5.0*y/height);
        gaborWavelet(sampleCount, output, frequency, 5.5);
        convolve(sampleCount, output, input);
        toTimeDomain(sampleCount, output, output);

        switch(mode) {
            case 0: // Real Grayscale
                for(unsigned long x = 0; x < width; ++x)
                    outputRow[x] = fmin(0.5+0.5*creal(output[border+x]), 1.0)*maxColorFactor;
            break;
            case 1: // Imaginary Grayscale
                for(unsigned long x = 0; x < width; ++x)
                    outputRow[x] = fmin(0.5+0.5*cimag(output[border+x]), 1.0)*maxColorFactor;
            break;
            case 2: // Amplitude Grayscale
                for(unsigned long x = 0; x < width; ++x)
                    outputRow[x] = fmin(cabs(output[border+x]), 1.0)*maxColorFactor;
            break;
            case 3: // Phase Grayscale
                for(unsigned long x = 0; x < width; ++x)
                    outputRow[x] = 2.0*fabs(carg(output[border+x])/M_PI)*maxColorFactor;
            break;
            case 4: // Equipotential
                for(unsigned long x = 0; x < width; ++x)
                    writePixelHsv(&outputRow[x*3], fmin(cabs(output[border+x])*0.9, 0.9), 1.0, 1.0);
            break;
            case 5: // Rainbow
                for(unsigned long x = 0; x < width; ++x)
                    writePixelHsv(&outputRow[x*3], carg(output[border+x])/(2*M_PI)+0.5, 1.0, fmin(cabs(output[border+x]), 1.0));
            break;
        }
        png_write_row(outputPng, outputRow);
    }

    png_write_end(outputPng, NULL);
    fclose(outputFile);
    free(outputRow);

    fftw_free(input);
    fftw_free(output);
    return 0;
}
