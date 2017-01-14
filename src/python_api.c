#include <ccwt.h>
#include <fftw3.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <python2.7/Python.h>
#include <numpy/arrayobject.h>

#define cleanup(method, var) if(var) method(var)

static PyObject* ccwt_render_png_python_api(PyObject* self, PyObject* args) {
    char* path = NULL;
    unsigned int rendering_mode = 0;
    PyArrayObject *input_array = NULL;
    struct ccwt_data ccwt;
    ccwt.input = NULL;
    ccwt.output = NULL;

    if(!PyArg_ParseTuple(args, "sO!iiidddd", &path, &PyArray_Type, &input_array, &ccwt.padding, &ccwt.height, &rendering_mode, &ccwt.frequency_scale, &ccwt.frequency_offset, &ccwt.frequency_basis, &ccwt.deviation))
        return NULL;

    FILE* file = fopen(path, "wb");
    if(!file) {
        PyErr_SetString(PyExc_IOError, "Could not open output file");
        goto cleanup;
    }

    if(PyArray_TYPE(input_array) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_TypeError, "Expected first argument to be float64");
        goto cleanup;
    }

    if(PyArray_NDIM(input_array) != 1) {
        PyErr_SetString(PyExc_TypeError, "Expected first argument to have one dimension");
        goto cleanup;
    }

    ccwt.width = (unsigned int)PyArray_DIM(input_array, 0);
    ccwt.sample_count = 2*ccwt.padding+ccwt.width;
    ccwt.input = (complex double*)fftw_malloc(sizeof(complex double)*ccwt.sample_count);
    ccwt.output = (complex double*)fftw_malloc(sizeof(complex double)*ccwt.sample_count);
    if(!ccwt.input || !ccwt.output) {
        PyErr_SetNone(PyExc_MemoryError);
        goto cleanup;
    }

    for(unsigned int i = 0; i < ccwt.padding; ++i)
        ccwt.input[i] = 0;
    for(unsigned int i = 0; i < ccwt.width; ++i)
        ccwt.input[ccwt.padding+i] = ((double*)PyArray_DATA(input_array))[i];
    for(unsigned int i = ccwt.padding+ccwt.width; i < ccwt.sample_count; ++i)
        ccwt.input[i] = 0;

    switch(ccwt_render_png(&ccwt, file, rendering_mode)) {
        default:
            PyErr_SetNone(PyExc_StandardError);
            break;
        case -1:
            PyErr_SetNone(PyExc_MemoryError);
            break;
        case 0:;
    }

    cleanup:
    cleanup(fclose, file);
    cleanup(fftw_free, ccwt.input);
    cleanup(fftw_free, ccwt.output);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef python_methods[] = {
    { "render_png", ccwt_render_png_python_api, METH_VARARGS, "Generate a PNG image as result" },
    { NULL, NULL, 0, NULL }
};

PyMODINIT_FUNC initccwt() {
    Py_InitModule("ccwt", python_methods);
    import_array();
}
