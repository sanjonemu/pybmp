/*
  bmp.h
*/

#ifndef __BMP_H__
#define __BMP_H__

#ifndef UNICODE
#define UNICODE
#endif

#include <Python.h>
#include <structmember.h>
#include <frameobject.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#ifdef __WINNT__
#include <windows.h>
#endif

#ifdef __BMP_MAKE_DLL_
#define __PORT __declspec(dllexport) // make dll mode
#else
#define __PORT __declspec(dllimport) // use dll mode
#endif

typedef unsigned char uchar;
typedef unsigned int uint;

#define BUFSIZE 4096

#define IASPECTX 16
#define IASPECTY 24

#ifdef __WINNT__
__PORT uint writebmp(char *fn, HBITMAP hbmp, int c, int r, HDC hdc);
#endif
__PORT uint drawbmp(int c, int r, int p, int d, char *bmpbuffer, char *fn);

#define _BMP "bmp"

// PyErr_Fetch should be called at the same (stack) layer as MACRO placed on.
// and *MUST* be called PyImport_ImportModule etc *AFTER* PyErr_Fetch
#define BMPPROCESSEXCEPTION(S) do{ \
  if(PyErr_Occurred()){ \
    PyObject *ptyp, *pval, *ptb; \
    PyErr_Fetch(&ptyp, &pval, &ptb); \
    if(0) fprintf(stderr, "%08x %08x: %s\n", ptb, pval, \
      pval ? PyString_AsString(pval) : "!pval"); \
    PyObject *m = PyImport_ImportModule(_BMP); \
    if(!m) fprintf(stderr, "cannot import %s\n", _BMP); \
    else{ \
      PyObject *tpl = Py_BuildValue("(s)", S); \
      PyObject *kw = PyDict_New(); \
      if(ptyp) PyDict_SetItemString(kw, "typ", ptyp); \
      if(pval) PyDict_SetItemString(kw, "val", pval); \
      if(ptb) PyDict_SetItemString(kw, "tb", ptb); \
      PyObject_Call(PyObject_GetAttrString(m, "bmpProcessException"), \
        tpl, kw); \
    } \
    PyErr_NormalizeException(&ptyp, &pval, &ptb); \
    PyErr_Clear(); \
    if(0) fprintf(stderr, "cleanup exceptions inside: %s\n", S); \
  } \
}while(0)

PyObject *bmpProcessException(PyObject *self, PyObject *args, PyObject *kw);
PyObject *BMP(PyObject *self, PyObject *args, PyObject *kw);

#endif // __BMP_H__
