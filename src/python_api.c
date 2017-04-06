#include <ccwt.h>
#include <fftw3.h>

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

void fft_array_destructor(PyObject* obj) {
    void* ptr = PyCapsule_GetPointer(obj, "ptr");
    if(ptr)
        fftw_free(ptr);
}

static PyObject* fft(PyObject* self, PyObject* args) {
    unsigned int input_padding = 0, thread_count = 0;
    PyArrayObject* input_signal = NULL;
    if(!PyArg_ParseTuple(args, "O!|ii", &PyArray_Type, &input_signal, &input_padding, &thread_count))
        return NULL;

    if(PyArray_NDIM(input_signal) != 1) {
        PyErr_SetString(PyExc_TypeError, "Expected first argument to have exactly one dimension");
        return NULL;
    }

    if(PyArray_TYPE(input_signal) < NPY_FLOAT32 || PyArray_TYPE(input_signal) > NPY_COMPLEX128) {
        PyErr_SetString(PyExc_TypeError, "Expected first argument to be an array of: float32, float64, complex64 or complex128");
        return NULL;
    }

    unsigned int input_width = (unsigned int)PyArray_DIM(input_signal, 0);
    complex_type* fourier_transformed_signal = ccwt_fft(
        input_width, input_padding, thread_count,
        PyArray_DATA(input_signal), PyArray_TYPE(input_signal)-NPY_FLOAT32
    );

    long int dimensions[] = { input_width+input_padding*2 };
    PyObject* output_array = PyArray_New(&PyArray_Type, 1, dimensions, NPY_COMPLEX128, NULL, fourier_transformed_signal, 0, 0, Py_None);
    PyArray_SetBaseObject((PyArrayObject*)output_array, PyCapsule_New(fourier_transformed_signal, "ptr", &fft_array_destructor));
    return output_array;
}

static PyObject* frequency_band(PyObject* self, PyObject* args) {
    unsigned int frequencies_count;
    double frequency_range = 0.0, frequency_offset = 0.0, frequency_basis = 0.0, deviation = 1.0;
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

int row_callback(struct ccwt_thread_data* thread, unsigned int y) {
    struct ccwt_data* ccwt = thread->ccwt;
    complex_type* array_data = (complex_type*)ccwt->user_data;
    memcpy(&array_data[ccwt->output_width*y], &thread->output[ccwt->output_padding], ccwt->output_width*sizeof(complex_type));
    return 0;
}

static PyObject* python_api(PyObject* args, unsigned int mode) {
    FILE* file = NULL;
    unsigned int return_value = 0, rendering_mode;
    double logarithmic_basis;
    PyArrayObject *fourier_transformed_signal = NULL, *output_array = NULL, *frequency_band = NULL;
    struct ccwt_data ccwt;
    ccwt.thread_count = 1;
    ccwt.output_width = 0;
    ccwt.input_padding = 0;

    if(mode == 0) {
        if(!PyArg_ParseTuple(args, "O!O!|iii", &PyArray_Type, &fourier_transformed_signal, &PyArray_Type, &frequency_band, &ccwt.output_width, &ccwt.input_padding, &ccwt.thread_count))
            return NULL;
    } else {
        PyObject* file_object = NULL;
        if(!PyArg_ParseTuple(args, "OidO!O!|iii", &file_object, &rendering_mode, &logarithmic_basis, &PyArray_Type, &fourier_transformed_signal, &PyArray_Type, &frequency_band, &ccwt.output_width, &ccwt.input_padding, &ccwt.thread_count))
            return NULL;
        file = fdopen(PyObject_AsFileDescriptor(file_object), "wb");
        if(!file) {
            PyErr_SetString(PyExc_IOError, "Could not open output file");
            goto cleanup;
        }
    }

    if(PyArray_NDIM(fourier_transformed_signal) != 1) {
        PyErr_SetString(PyExc_TypeError, "Expected first argument to have exactly one dimension");
        goto cleanup;
    }

    if(PyArray_TYPE(fourier_transformed_signal) != NPY_COMPLEX128) {
        PyErr_SetString(PyExc_TypeError, "Expected first argument to be an array of type complex128");
        goto cleanup;
    }

    if(PyArray_NDIM(frequency_band) != 2) {
        PyErr_SetString(PyExc_TypeError, "Expected second argument to have exactly two dimensions");
        goto cleanup;
    }

    if(PyArray_TYPE(frequency_band) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_TypeError, "Expected second argument to be an array of type float64");
        goto cleanup;
    }

    ccwt.input_sample_count = (unsigned int)PyArray_DIM(fourier_transformed_signal, 0);
    ccwt.input_width = ccwt.input_sample_count-2*ccwt.input_padding;
    ccwt.height = (unsigned int)PyArray_DIM(frequency_band, 0);
    ccwt.input = (complex_type*)PyArray_DATA(fourier_transformed_signal);
    ccwt.frequency_band = (double*)PyArray_DATA(frequency_band);
    if(ccwt.output_width == 0)
        ccwt.output_width = ccwt.input_width;

    if(mode == 0) {
        long int dimensions[] = { ccwt.height, ccwt.output_width };
        output_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, dimensions, NPY_COMPLEX128, NULL, NULL, 0, 0, Py_None);
        if(!output_array)
            goto cleanup;
        ccwt.user_data = PyArray_DATA(output_array);
        ccwt.callback = row_callback;
        return_value = ccwt_numeric_output(&ccwt);
    } else
        return_value = ccwt_render_png(&ccwt, file, rendering_mode, logarithmic_basis);

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
        if(!file)
            return NULL;
        fflush(file);
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
    { "numeric_output", calculate, METH_VARARGS, "Calculate 2D complex array as result" },
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
    fftw_init_threads();
    import_array();
    PyObject* module = module_create(module_name);
#undef macro_wrapper
#define macro_wrapper(name) PyModule_AddIntConstant(module, #name, name);
#include <render_mode.h>
#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
