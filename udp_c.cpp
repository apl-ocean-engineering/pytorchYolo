#include <Python.h>

// setenv("PYTHONPATH")

Py_Initalize();

PyObject *pName, *pModule, *pDict, *pFunc, *pArgs, *pValue;

pName = PyString_FromString("atom_py");
pModule = PyModule_GetDict(pModule);
pFunc = PyDict_GetItemString(pDict, "run");

pArgs = PyTuple_New(2);

pValue = PyInt_FromLong(2);
