#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <png.h>
#include <fftw3.h>

double abs(fftw_complex value) {
    return sqrt(value[0]*value[0]+value[1]*value[1]);
}

double atan(fftw_complex value) {
    return atan2(value[1], value[0])/M_PI*0.5;
}

struct Signal {
    unsigned long sampleCount;
    fftw_complex* timeDomain;
    fftw_complex* frequencyDomain;

    Signal(unsigned long _sampleCount) :sampleCount(_sampleCount) {
        timeDomain = reinterpret_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*sampleCount));
        frequencyDomain = reinterpret_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*sampleCount));
    }

    ~Signal() {
        fftw_free(timeDomain);
        fftw_free(frequencyDomain);
    }

    void toFrequencyDomain() {
        fftw_plan plan = fftw_plan_dft_1d(sampleCount, timeDomain, frequencyDomain, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);
    }

    void toTimeDomain() {
        fftw_plan plan = fftw_plan_dft_1d(sampleCount, frequencyDomain, timeDomain, FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);
    }

    void convolution(const Signal& other) {
        double scaleFactor = 1.0/(double)sampleCount;
        for(unsigned long i = 0; i < sampleCount; ++i) {
            double a = other.frequencyDomain[i][0]*scaleFactor, b = other.frequencyDomain[i][1]*scaleFactor;
            double c = frequencyDomain[i][0], d = frequencyDomain[i][1];
            frequencyDomain[i][0] = a*c-b*d;
            frequencyDomain[i][1] = (a+b)*(c+d)-a*c-b*d;
        }
    }

    void gaborWavelet(double f0, double a) {

        for(unsigned long i = 0; i < sampleCount/2; ++i) {
            double f = i-f0;
            frequencyDomain[i][0] = exp(-f*f*a);
            frequencyDomain[i][1] = 0;
        }
        for(unsigned long i = sampleCount/2; i < sampleCount; ++i) {
            double f = sampleCount-i+f0;
            frequencyDomain[i][0] = exp(-f*f*a);
            frequencyDomain[i][1] = 0;
        }
        toTimeDomain();
    }

    void debugPrint() {
        for(unsigned long i = 0; i < sampleCount; ++i)
            printf("% 2.6f % 2.6f % 2.20f | % 2.6f % 2.6f % 2.6f\n",
                timeDomain[i][0], timeDomain[i][1], abs(timeDomain[i]),
                frequencyDomain[i][0], frequencyDomain[i][1], abs(frequencyDomain[i]));
    }
};

void writePixelHsv(unsigned char* pixel, double H, double S, double V) {
    unsigned char h = H*6;
    double f = H*6-h, p = V*(1-S), q = V*(1-S*f), t = V*(1-(S*(1-f)));
    switch(h) {
        default:
            pixel[0] = V*255;
            pixel[1] = t*255;
            pixel[2] = p*255;
        break;
        case 1:
            pixel[0] = q*255;
            pixel[1] = V*255;
            pixel[2] = p*255;
        break;
        case 2:
            pixel[0] = p*255;
            pixel[1] = V*255;
            pixel[2] = t*255;
        break;
        case 3:
            pixel[0] = p*255;
            pixel[1] = q*255;
            pixel[2] = V*255;
        break;
        case 4:
            pixel[0] = t*255;
            pixel[1] = p*255;
            pixel[2] = V*255;
        break;
        case 5:
            pixel[0] = V*255;
            pixel[1] = p*255;
            pixel[2] = q*255;
        break;
    }
}

int main() {
    unsigned long sampleCount = 256, frequencyCount = 256;

    Signal signal(sampleCount);
    double border = 32;
    for(unsigned long i = 0; i < sampleCount; ++i) {
        double t = (double)i/(double)sampleCount*2*M_PI;
        t *= pow(1.009, i);
        double a = 1.0;
        if(i < border)
            a *= i/border;
        else if(i > sampleCount-border)
            a *= (sampleCount-i)/border;
        signal.timeDomain[i][0] = a*cos(t);
        signal.timeDomain[i][1] = a*sin(t);
    }
    signal.toFrequencyDomain();

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
    png_set_IHDR(outputPng, outputPngInfo, sampleCount, frequencyCount,
                 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
    png_write_info(outputPng, outputPngInfo);
    unsigned char* outputRow = (unsigned char*)malloc(sampleCount*3);

    double frequency = 32.0;
    for(unsigned int i = 0; i < frequencyCount; ++i) {
        Signal kernel(sampleCount);
        kernel.gaborWavelet(frequency, 0.1);
        kernel.convolution(signal);
        kernel.toTimeDomain();
        frequency *= 0.99;

        for(unsigned long i = 0; i < sampleCount; ++i) {
            // writePixelHsv(&outputRow[i*3], 0, 0, fmin(abs(kernel.timeDomain[i]), 1.0)); // AmplitudeGrayscale
            // writePixelHsv(&outputRow[i*3], 0, 0, 2*fabs(atan(kernel.timeDomain[i]))); // PhaseGrayscale
            // writePixelHsv(&outputRow[i*3], fmin(abs(kernel.timeDomain[i])*0.9, 0.9), 1, 1); // Equipotential
            writePixelHsv(&outputRow[i*3], atan(kernel.timeDomain[i])+0.5, 1, fmin(abs(kernel.timeDomain[i]), 1.0)); // Rainbow
        }
        png_write_row(outputPng, outputRow);
    }

    png_write_end(outputPng, NULL);
    free(outputRow);
    fclose(outputFile);

    return 0;
}
