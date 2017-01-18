#include <ccwt.h>
#include <fftw3.h>

#if PY_MAJOR_VERSION >= 3
#include <python3.5/Python.h>
#else
#include <python2.7/Python.h>
#endif
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#define cleanup(method, var) if(var) method(var)

int row_callback(struct ccwt_data* ccwt, void* user_data, unsigned int row) {
    complex double* array_data = (complex double*)user_data;
    memcpy(&array_data[ccwt->width*row], &ccwt->output[ccwt->padding], ccwt->width*sizeof(complex double));
    return 0;
}

static PyObject* python_api(PyObject* args, unsigned int mode) {
    char* path = NULL;
    FILE* file = NULL;
    unsigned int return_value = 0, rendering_mode = 0;
    PyArrayObject *input_array = NULL, *output_array = NULL;
    struct ccwt_data ccwt;
    ccwt.input = NULL;
    ccwt.output = NULL;

    if(mode == 0) {
        if(!PyArg_ParseTuple(args, "O!ddddiiis", &PyArray_Type, &input_array, &ccwt.frequency_range, &ccwt.frequency_offset, &ccwt.frequency_basis, &ccwt.deviation, &ccwt.padding, &ccwt.height, &rendering_mode, &path))
            return NULL;
        file = fopen(path, "wb");
        if(!file) {
            PyErr_SetString(PyExc_IOError, "Could not open output file");
            goto cleanup;
        }
    } else if(!PyArg_ParseTuple(args, "O!ddddii", &PyArray_Type, &input_array, &ccwt.frequency_range, &ccwt.frequency_offset, &ccwt.frequency_basis, &ccwt.deviation, &ccwt.padding, &ccwt.height))
        return NULL;

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

    if(mode == 0)
        return_value = ccwt_render_png(&ccwt, file, rendering_mode);
    else {
        long int dimensions[] = { ccwt.height, ccwt.width };
        output_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, dimensions, NPY_COMPLEX128, NULL, NULL, 0, 0, Py_None);
        if(!output_array)
            goto cleanup;
        return_value = ccwt_calculate(&ccwt, PyArray_DATA(output_array), row_callback);
    }

    switch(return_value) {
        default:
            PyErr_SetNone(PyExc_Exception);
            break;
        case -1:
            PyErr_SetNone(PyExc_MemoryError);
            break;
        case 0:;
    }

    cleanup:
    cleanup(fftw_free, ccwt.input);
    cleanup(fftw_free, ccwt.output);
    if(mode == 0) {
        cleanup(fclose, file);
        Py_INCREF(Py_None);
        return Py_None;
    } else {
        cleanup(Py_INCREF, output_array);
        return (PyObject*)output_array;
    }
}

static PyObject* render_png(PyObject* self, PyObject* args) {
    return python_api(args, 0);
}

static PyObject* calculate(PyObject* self, PyObject* args) {
    return python_api(args, 1);
}

static struct PyMethodDef module_methods[] = {
    { "render_png", render_png, METH_VARARGS, "Generate a PNG image as result" },
    { "calculate", calculate, METH_VARARGS, "Calculate 2D complex array as result" },
    { NULL, NULL, 0, NULL }
};

const char* module_name = "ccwt";
#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT, module_name, "", -1, module_methods
};
PyMODINIT_FUNC PyInit_ccwt() {
    PyModule_Create(&module_definition);
#else
PyMODINIT_FUNC initccwt() {
    Py_InitModule(module_name, module_methods);
#endif
    import_array();
}
