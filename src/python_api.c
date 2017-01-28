#include <ccwt.h>
#include <fftw3.h>

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#define python_api_case(python_type, c_type) \
    case python_type: \
    for(unsigned int i = 0; i < ccwt.input_width; ++i) \
        ccwt.input[ccwt.input_padding+i] = ((c_type*)PyArray_DATA(input_array))[i]; \
    break

int row_callback(struct ccwt_data* ccwt, void* user_data, unsigned int row) {
    complex double* array_data = (complex double*)user_data;
    memcpy(&array_data[ccwt->output_width*row], &ccwt->output[ccwt->output_padding], ccwt->output_width*sizeof(complex double));
    return 0;
}

static PyObject* frequency_band(PyObject* self, PyObject* args) {
    unsigned int frequencies_count;
    double frequency_range = 0.0, frequency_offset = 0.0, frequency_basis = 0.0, deviation = M_E/(M_PI*M_PI);
    if(!PyArg_ParseTuple(args, "i|dddd", &frequencies_count, &frequency_range, &frequency_offset, &frequency_basis, &deviation))
        return NULL;

    long int dimensions[] = { frequencies_count, 2 };
    PyArrayObject* frequencies = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, dimensions, NPY_FLOAT64, NULL, NULL, 0, 0, Py_None);
    if(!frequencies)
        return NULL;

    ccwt_frequency_band((double*)PyArray_DATA(frequencies), frequencies_count, frequency_range, frequency_offset, frequency_basis, deviation);

    Py_INCREF(frequencies);
    return (PyObject*)frequencies;
}

static PyObject* fft(PyObject* self, PyObject* args) {
    unsigned int input_padding = 0;
    PyArrayObject* input_array = NULL;
    if(!PyArg_ParseTuple(args, "O!|i", &PyArray_Type, &input_array, &input_padding))
        return NULL;

    if(PyArray_NDIM(input_array) != 1) {
        PyErr_SetString(PyExc_TypeError, "Expected first argument to have exactly one dimension");
        return NULL;
    }

    if(PyArray_TYPE(input_array) < NPY_FLOAT32 || PyArray_TYPE(input_array) > NPY_COMPLEX128) {
        PyErr_SetString(PyExc_TypeError, "Expected first argument to be an array of: float32, float64, complex64 or complex128");
        return NULL;
    }

    unsigned int input_width = (unsigned int)PyArray_DIM(input_array, 0);
    complex double* output_array = ccwt_fft(input_width, input_padding, PyArray_DATA(input_array),
        PyArray_TYPE(input_array) == NPY_FLOAT64 || PyArray_TYPE(input_array) == NPY_COMPLEX128,
        PyArray_TYPE(input_array) == NPY_COMPLEX64 || PyArray_TYPE(input_array) == NPY_COMPLEX128);

    long int dimensions[] = { input_width+input_padding*2 };
    return PyArray_New(&PyArray_Type, 1, dimensions, NPY_COMPLEX128, NULL, output_array, 0, 0, Py_None);
}

static PyObject* python_api(PyObject* args, unsigned int mode) {
    char* path = NULL;
    FILE* file = NULL;
    unsigned int return_value = 0, rendering_mode;
    PyArrayObject *input_array = NULL, *output_array = NULL, *frequencies = NULL;
    struct ccwt_data ccwt;
    ccwt.output_width = 0;
    ccwt.input_padding = 0;

    if(mode == 0) {
        if(!PyArg_ParseTuple(args, "O!O!|ii", &PyArray_Type, &input_array, &PyArray_Type, &frequencies, &ccwt.output_width, &ccwt.input_padding))
            return NULL;
    } else {
        if(!PyArg_ParseTuple(args, "siO!O!|ii", &path, &rendering_mode, &PyArray_Type, &input_array, &PyArray_Type, &frequencies, &ccwt.output_width, &ccwt.input_padding))
            return NULL;
        file = fopen(path, "wb");
        if(!file) {
            PyErr_SetString(PyExc_IOError, "Could not open output file");
            goto cleanup;
        }
    }

    if(PyArray_NDIM(input_array) != 1) {
        PyErr_SetString(PyExc_TypeError, "Expected first argument to have exactly one dimension");
        goto cleanup;
    }

    if(PyArray_TYPE(input_array) != NPY_COMPLEX128) {
        PyErr_SetString(PyExc_TypeError, "Expected first argument to be an array of type complex128");
        goto cleanup;
    }

    if(PyArray_NDIM(frequencies) != 2) {
        PyErr_SetString(PyExc_TypeError, "Expected second argument to have exactly two dimensions");
        goto cleanup;
    }

    if(PyArray_TYPE(frequencies) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_TypeError, "Expected second argument to be an array of type float64");
        goto cleanup;
    }

    ccwt.input_sample_count = (unsigned int)PyArray_DIM(input_array, 0);
    ccwt.input_width = ccwt.input_sample_count-2*ccwt.input_padding;
    ccwt.height = (unsigned int)PyArray_DIM(frequencies, 0);
    ccwt.input = (complex double*)PyArray_DATA(input_array);
    ccwt.frequencies = (double*)PyArray_DATA(frequencies);
    if(ccwt.output_width == 0)
        ccwt.output_width = ccwt.input_width;

    if(mode == 0) {
        long int dimensions[] = { ccwt.height, ccwt.output_width };
        output_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, dimensions, NPY_COMPLEX128, NULL, NULL, 0, 0, Py_None);
        if(!output_array)
            goto cleanup;
        return_value = ccwt_calculate(&ccwt, PyArray_DATA(output_array), row_callback);
    } else
        return_value = ccwt_render_png(&ccwt, file, rendering_mode);

    switch(return_value) {
        default:
            PyErr_SetNone(PyExc_Exception);
            break;
        case -1:
            PyErr_SetNone(PyExc_MemoryError);
            break;
        case -2:
            PyErr_SetString(PyExc_ValueError, "Upsampling is not supported");
            break;
        case 0:;
    }

    cleanup:
    if(mode == 0) {
        if(!output_array)
            return NULL;
        Py_INCREF(output_array);
        return (PyObject*)output_array;
    } else {
        if(file)
            fclose(file);
        Py_INCREF(Py_None);
        return Py_None;
    }
}

static PyObject* calculate(PyObject* self, PyObject* args) {
    return python_api(args, 0);
}

static PyObject* render_png(PyObject* self, PyObject* args) {
    return python_api(args, 1);
}

static struct PyMethodDef module_methods[] = {
    { "fft", fft, METH_VARARGS, "Calculate the FFT of a signal" },
    { "frequency_band", frequency_band, METH_VARARGS, "Generate a frequency band" },
    { "calculate", calculate, METH_VARARGS, "Calculate 2D complex array as result" },
    { "render_png", render_png, METH_VARARGS, "Generate a PNG image as result" },
    { NULL, NULL, 0, NULL }
};

#define module_name ccwt

#define concat2(a, b) a##b
#define concat(a, b) concat2(a, b)
#define str(in) #in
#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT, str(module_name), "", -1, module_methods
};
#define module_init PyInit_
#define module_create(module_name) PyModule_Create(&module_definition)
#else
#define module_init init
#define module_create(module_name) Py_InitModule(str(module_name), module_methods)
#endif

PyMODINIT_FUNC concat(module_init, module_name) (void) {
    import_array();
    PyObject* module = module_create(module_name);
#undef macro_wrapper
#define macro_wrapper(name) PyModule_AddIntConstant(module, #name, name);
#include <render_mode.h>
#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
