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

static PyObject* python_api(PyObject* args, unsigned int mode) {
    char* path = NULL;
    FILE* file = NULL;
    unsigned int return_value = 0, rendering_mode = 0;
    PyArrayObject *input_array = NULL, *output_array = NULL;
    struct ccwt_data ccwt;

    if(mode == 0) {
        if(!PyArg_ParseTuple(args, "O!ddddiiiis", &PyArray_Type, &input_array, &ccwt.frequency_range, &ccwt.frequency_offset, &ccwt.frequency_basis, &ccwt.deviation, &ccwt.input_padding, &ccwt.output_width, &ccwt.height, &rendering_mode, &path))
            return NULL;
        file = fopen(path, "wb");
        if(!file) {
            PyErr_SetString(PyExc_IOError, "Could not open output file");
            goto cleanup;
        }
    } else if(!PyArg_ParseTuple(args, "O!ddddiii", &PyArray_Type, &input_array, &ccwt.frequency_range, &ccwt.frequency_offset, &ccwt.frequency_basis, &ccwt.deviation, &ccwt.input_padding, &ccwt.output_width, &ccwt.height))
        return NULL;

    if(PyArray_NDIM(input_array) != 1) {
        PyErr_SetString(PyExc_TypeError, "Expected first argument to have one dimension");
        goto cleanup;
    }

    ccwt.input_width = (unsigned int)PyArray_DIM(input_array, 0);
    if(ccwt_init(&ccwt) != 0) {
        PyErr_SetNone(PyExc_MemoryError);
        goto cleanup;
    }
    if(ccwt.output_width > ccwt.input_width) {
        PyErr_SetString(PyExc_ValueError, "Upsampling is not supported");
        goto cleanup;
    }

    switch(PyArray_TYPE(input_array)) {
        python_api_case(NPY_FLOAT32, float);
        python_api_case(NPY_FLOAT64, double);
        python_api_case(NPY_COMPLEX64, complex float);
        python_api_case(NPY_COMPLEX128, complex double);
        default:
            PyErr_SetString(PyExc_TypeError, "Expected first argument to be one of: float32, float64, complex64, complex128");
            goto cleanup;
    }

    if(mode == 0)
        return_value = ccwt_render_png(&ccwt, file, rendering_mode);
    else {
        long int dimensions[] = { ccwt.height, ccwt.output_width };
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
    ccwt_cleanup(&ccwt);
    if(mode == 0) {
        if(file)
            fclose(file);
        Py_INCREF(Py_None);
        return Py_None;
    } else {
        if(!output_array)
            return NULL;
        Py_INCREF(output_array);
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

#define module_name ccwt

#define concat2(a, b) a##b
#define concat(a, b) concat2(a, b)
#define str(in) #in
#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT, str(module_name), "", -1, module_methods
};
#define module_init PyInit_
#define module_create(module_name) return PyModule_Create(&module_definition)
#else
#define module_init init
#define module_create(module_name) Py_InitModule(str(module_name), module_methods)
#endif

PyMODINIT_FUNC concat(module_init, module_name) (void) {
    import_array();
    module_create(module_name);
}
