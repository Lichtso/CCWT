#include <math.h>
#include <complex.h>
#include <stdlib.h>
#include <fftw3.h>
#include <png.h>

void convolve(unsigned int sample_count, complex double* signal, complex double* kernel) {
    double scaleFactor = 1.0/(double)sample_count;
    for(unsigned long i = 0; i < sample_count; ++i)
        signal[i] *= kernel[i]*scaleFactor;
}

void gaborWavelet(unsigned int sample_count, complex double* kernel, double centerFrequency, double deviation) {
    deviation = 1.0/deviation;
    for(unsigned long i = 0; i < sample_count/2+1; ++i) {
        double f = (i-centerFrequency)*deviation;
        kernel[i] = exp(-f*f);
    }
    for(unsigned long i = sample_count/2+1; i < sample_count; ++i) {
        double f = (sample_count-i+centerFrequency)*deviation;
        kernel[i] = exp(-f*f);
    }
}

const unsigned int maxColorFactor = 255;

#define writePixelHsvCase(a, b, c) \
    pixel[0] = a*maxColorFactor; \
    pixel[1] = b*maxColorFactor; \
    pixel[2] = c*maxColorFactor; \
    break;

void writePixelHsv(unsigned char* pixel, double H, double S, double V) {
    unsigned char h = H*6;
    double f = H*6-h, p = V*(1-S), q = V*(1-S*f), t = V*(1-(S*(1-f)));
    switch(h) {
        default: writePixelHsvCase(V, t, p)
        case 1: writePixelHsvCase(q, V, p)
        case 2: writePixelHsvCase(p, V, t)
        case 3: writePixelHsvCase(p, q, V)
        case 4: writePixelHsvCase(t, p, V)
        case 5: writePixelHsvCase(V, p, q)
    }
}

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <python2.7/Python.h>
#include <numpy/arrayobject.h>

#define cleanup(method, var) if(var) method(var)

static PyObject* ccwt_png(PyObject* self, PyObject* args) {
    fftw_plan input_plan = NULL, output_plan = NULL;
    complex double *input = NULL, *output = NULL;

    char* output_path = NULL;
    FILE* output_file = NULL;
    png_structp output_png = NULL;
    png_infop output_png_info = NULL;
    unsigned char* output_row = NULL;

    unsigned int sample_count = 0, padding = 0, width = 0, height = 0, rendering_mode = 0;
    double frequency_scale = 1.0, frequency_offset = 0.0, frequency_basis = 0.0, deviation = 5.5;
    PyArrayObject *input_array = NULL, *output_array = NULL;

    if(!PyArg_ParseTuple(args, "sO!iiiddd", &output_path, &PyArray_Type, &input_array, &padding, &height, &rendering_mode, &frequency_scale, &frequency_offset, &frequency_basis))
        return NULL;

    if(PyArray_TYPE(input_array) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_TypeError, "Expected first argument to be float64");
        goto cleanup;
    }

    if(PyArray_NDIM(input_array) != 1) {
        PyErr_SetString(PyExc_TypeError, "Expected first argument to have one dimension");
        goto cleanup;
    }

    output_array = (PyArrayObject*)PyArray_NewLikeArray(input_array, NPY_ANYORDER, NULL, 0);
    if(!output_array) {
        PyErr_SetString(PyExc_StandardError, "Failed to allocate output array");
        goto cleanup;
    }

    width = (unsigned int)PyArray_DIM(input_array, 0);
    sample_count = 2*padding+width;
    input = (complex double*)fftw_malloc(sizeof(complex double)*sample_count);
    output = (complex double*)fftw_malloc(sizeof(complex double)*sample_count);
    if(!input || !output) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate buffers");
        goto cleanup;
    }

    for(unsigned int i = 0; i < padding; ++i)
        input[i] = 0;
    for(unsigned int i = 0; i < width; ++i)
        input[padding+i] = ((double*)PyArray_DATA(input_array))[i];
    for(unsigned int i = padding+width; i < sample_count; ++i)
        input[i] = 0;

    input_plan = fftw_plan_dft_1d(sample_count, (fftw_complex*)input, (fftw_complex*)input, FFTW_FORWARD, FFTW_ESTIMATE);
    output_plan = fftw_plan_dft_1d(sample_count, (fftw_complex*)output, (fftw_complex*)output, FFTW_BACKWARD, FFTW_MEASURE);
    fftw_execute(input_plan);

    for(unsigned int i = 0; i < width; ++i)
        ((double*)PyArray_DATA(output_array))[i] = cabs(input[padding+i]);

    output_row = (unsigned char*)malloc(width*3);
    output_file = fopen(output_path, "wb");
    if(!output_file) {
        PyErr_SetString(PyExc_IOError, "Could not open output file");
        goto cleanup;
    }
    output_png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    output_png_info = png_create_info_struct(output_png);
    if(setjmp(png_jmpbuf(output_png))) {
        PyErr_SetString(PyExc_StandardError, "Failed to encode png file");
        goto cleanup;
    }
    png_init_io(output_png, output_file);
    png_set_IHDR(output_png, output_png_info, width, height,
                 8, (rendering_mode < 4) ? PNG_COLOR_TYPE_GRAY: PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
    png_write_info(output_png, output_png_info);

    for(unsigned long y = 0; y < height; ++y) {
        double frequency = frequency_scale*(1.0-(double)y/height)+frequency_offset;
        if(frequency_basis > 0.0)
            frequency = pow(frequency_basis, frequency);
        gaborWavelet(sample_count, output, frequency*sample_count/width, deviation);
        convolve(sample_count, output, input);
        fftw_execute(output_plan);

        switch(rendering_mode) {
            case 0: // Real Grayscale
                for(unsigned long x = 0; x < width; ++x)
                    output_row[x] = fmin(0.5+0.5*creal(output[padding+x]), 1.0)*maxColorFactor;
            break;
            case 1: // Imaginary Grayscale
                for(unsigned long x = 0; x < width; ++x)
                    output_row[x] = fmin(0.5+0.5*cimag(output[padding+x]), 1.0)*maxColorFactor;
            break;
            case 2: // Amplitude Grayscale
                for(unsigned long x = 0; x < width; ++x)
                    output_row[x] = fmin(cabs(output[padding+x]), 1.0)*maxColorFactor;
            break;
            case 3: // Phase Grayscale
                for(unsigned long x = 0; x < width; ++x)
                    output_row[x] = 2.0*fabs(carg(output[padding+x])/M_PI)*maxColorFactor;
            break;
            case 4: // Equipotential
                for(unsigned long x = 0; x < width; ++x)
                    writePixelHsv(&output_row[x*3], fmin(cabs(output[padding+x])*0.9, 0.9), 1.0, 1.0);
            break;
            case 5: // Rainbow
                for(unsigned long x = 0; x < width; ++x)
                    writePixelHsv(&output_row[x*3], carg(output[padding+x])/(2*M_PI)+0.5, 1.0, fmin(cabs(output[padding+x]), 1.0));
            break;
        }
        png_write_row(output_png, output_row);
    }
    png_write_end(output_png, NULL);

    cleanup:
    cleanup(fftw_destroy_plan, input_plan);
    cleanup(fftw_destroy_plan, output_plan);
    cleanup(fftw_free, input);
    cleanup(fftw_free, output);
    cleanup(fclose, output_file);
    cleanup(free, output_row);
    cleanup(Py_INCREF, output_array);
    return (PyObject*)output_array;
}

static PyMethodDef python_methods[] = {
    { "generatePng", ccwt_png, METH_VARARGS, "Generate a PNG image as result" },
    { NULL, NULL, 0, NULL }
};

PyMODINIT_FUNC initccwt() {
    Py_InitModule("ccwt", python_methods);
    import_array();
}
