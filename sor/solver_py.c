#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <string.h>

#include "image.h"
#include "solver.h"

PyObject *solve(PyObject *self, PyObject *args);

static char run_docs[] =
    "usage: foobar \n";

static PyMethodDef module_methods[] = {
    {"solve", (PyCFunction) solve, 
     METH_VARARGS, run_docs},
    {NULL}
};

static struct PyModuleDef Sor =
{
    PyModuleDef_HEAD_INIT,
    "Sor", /* name of module */
    "usage: foobar\n", /* module documentation, may be NULL */
    -1,   /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    module_methods
};

PyMODINIT_FUNC PyInit_Sor(void)
{
    import_array();
    return PyModule_Create(&Sor);
}

void ImgFromArray(image_t *img, PyArrayObject *arr)
{
    img->width = arr->dimensions[1];
    img->height = arr->dimensions[0];
    img->stride = arr->strides[0] / sizeof(float);
    img->data = (float *)PyArray_DATA(arr);
}

PyObject *solve(PyObject *self, PyObject *args)
{
    // Parse input arguments
    PyObject *odu, *odv, *oa11, *oa12, *oa22, *ob1, *ob2, *ohoriz, *overt, *oiterations, *oomega;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOOO", &odu, &odv, &oa11, &oa12, &oa22, &ob1, &ob2, &ohoriz, &overt, &oiterations, &oomega)){
        return NULL;
    }

    PyArrayObject *adu = (PyArrayObject *) PyArray_ContiguousFromObject(odu, PyArray_FLOAT32, 2, 2);
    PyArrayObject *adv = (PyArrayObject *) PyArray_ContiguousFromObject(odv, PyArray_FLOAT32, 2, 2);
    PyArrayObject *aa11 = (PyArrayObject *) PyArray_ContiguousFromObject(oa11, PyArray_FLOAT32, 2, 2);
    PyArrayObject *aa12 = (PyArrayObject *) PyArray_ContiguousFromObject(oa12, PyArray_FLOAT32, 2, 2);
    PyArrayObject *aa22 = (PyArrayObject *) PyArray_ContiguousFromObject(oa22, PyArray_FLOAT32, 2, 2);
    PyArrayObject *ab1 = (PyArrayObject *) PyArray_ContiguousFromObject(ob1, PyArray_FLOAT32, 2, 2);
    PyArrayObject *ab2 = (PyArrayObject *) PyArray_ContiguousFromObject(ob2, PyArray_FLOAT32, 2, 2);
    PyArrayObject *ahoriz = (PyArrayObject *) PyArray_ContiguousFromObject(ohoriz, PyArray_FLOAT32, 2, 2);
    PyArrayObject *avert = (PyArrayObject *) PyArray_ContiguousFromObject(overt, PyArray_FLOAT32, 2, 2);
    if (adu == NULL || adv == NULL || aa11 == NULL || aa12 == NULL || aa22 == NULL || ab1 == NULL || ab2 == NULL || ahoriz == NULL || avert == NULL){
        return NULL;
    }
    int iterations = (int)PyLong_AsLong(oiterations);
    float omega = (float)PyFloat_AsDouble(oomega);
    
    // Convert to image_t
    image_t du, dv, a11, a12, a22, b1, b2, horiz, vert;
    ImgFromArray(&du, adu);
    ImgFromArray(&dv, adv);
    ImgFromArray(&a11, aa11);
    ImgFromArray(&a12, aa12);
    ImgFromArray(&a22, aa22);
    ImgFromArray(&b1, ab1);
    ImgFromArray(&b2, ab2);
    ImgFromArray(&horiz, ahoriz);
    ImgFromArray(&vert, avert);
    
    // Run solver (slower version does not modify the arrays a11, a12, a22, b1, b2, horiz and vert)
    sor_coupled_slow_but_readable(&du, &dv, &a11, &a12, &a22, &b1, &b2, &horiz, &vert, iterations, omega);
    //sor_coupled(&du, &dv, &a11, &a12, &a22, &b1, &b2, &horiz, &vert, iterations, omega);
    
    // Delete arrays
    //Py_DECREF(adu);
    //Py_DECREF(adv);
    Py_DECREF(aa11);
    Py_DECREF(aa12);
    Py_DECREF(aa22);
    Py_DECREF(ab1);
    Py_DECREF(ab2);
    Py_DECREF(ahoriz);
    Py_DECREF(avert);
    
    //Py_RETURN_NONE;
    PyObject *out = PyTuple_New(2);
    PyTuple_SetItem(out, 0, (PyObject *)adu);
    PyTuple_SetItem(out, 1, (PyObject *)adv);
    return out;
}

