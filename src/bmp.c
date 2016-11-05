/*
  bmp.c

  >mingw32-make -f makefile.tdmgcc64
  >test_bmp.py
*/

#define __BMP_MAKE_DLL_
#include <bmp.h>

#include <numpy/arrayobject.h>

#define DEBUGLOG 0
#define TESTLOG "../dll/_test_dll_.log"

__PORT uint writebmp(char *fn, char *bmpbuffer)
{
  return 0;
}

__PORT uint drawbmp(char *bmpbuffer)
{
  return 0;
}

static int tbInfo(int line, PyCodeObject *f_code)
{
  // struct _frame in frameobject.h
  //   (frame->f_code->..)
  //   (tb->tb_frame->f_code->..)
  char *file = PyString_AsString(f_code->co_filename);
  char *fnc = PyString_AsString(f_code->co_name);
  fprintf(stderr, "    %s(%d): %s\n", file, line, fnc);
  return 0;
}

static int tbDisp(char *s)
{
  fprintf(stderr, "Traceback (most recent call last): --[%s]--\n", s ? s:"");
  PyThreadState *tstat = PyThreadState_GET();
  if(tstat && tstat->frame){
    PyFrameObject *frame = tstat->frame;
    if(!frame) fprintf(stderr, "  error: [!frame] broken stack ?\n");
    else{
      // PyTracebackObject* tb = (PyTracebackObject*)tstat->curexc_traceback;
      // PyTracebackObject* tb = (PyTracebackObject*)tstat->exc_traceback;
      // PyTracebackObject* tb = (PyTracebackObject*)tstat->async_exc;
      PyTracebackObject* tb = (PyTracebackObject*)frame->f_exc_traceback;
      if(!tb){
        fprintf(stderr, "  error: [!tb] another stack ?\n");
        while(frame){ // backword
          // tbInfo(frame->f_lineno, frame->f_code); // not the correct number
          /* need to call PyCode_Addr2Line() */
          tbInfo(PyCode_Addr2Line(frame->f_code, frame->f_lasti), frame->f_code);
          frame = frame->f_back;
        }
      }else{
        while(tb){ // forward
          tbInfo(tb->tb_lineno, tb->tb_frame->f_code); // is tb_lineno correct ?
          tb = tb->tb_next;
        }
      }
    }
  }else{
    fprintf(stderr, "  error: [!tstat || !tstat->frame] another thread ?\n");
  }
  return 0;
}

PyObject *bmpProcessException(PyObject *self, PyObject *args, PyObject *kw)
{
  char *s;
  PyObject *ptyp = NULL;
  PyObject *pval = NULL;
  PyObject *ptb = NULL;

  FILE *fp = fopen(TESTLOG, "ab");
  fprintf(fp, "bmpProcessException %08x %08x %08x\n",
    (uchar *)self, (uchar *)args, (uchar *)kw);
  fclose(fp);

  // if(obj == Py_None){ }

  char *keys[] = {"s", "typ", "val", "tb", NULL};
  if(!PyArg_ParseTupleAndKeywords(args, kw, "|sOOO", keys, &s, &ptyp, &pval, &ptb)){
    FILE *fp = fopen(TESTLOG, "ab");
    fprintf(fp, "ERROR: PyArg_ParseTupleAndKeywords()\n");
    fclose(fp);
    return NULL;
  }else{
    FILE *fp = fopen(TESTLOG, "ab");
    fprintf(fp, "SUCCESS: PyArg_ParseTupleAndKeywords(%s, %08x, %08x, %08x)\n", s, (char *)ptyp, (char *)pval, (char *)ptb);
    fclose(fp);
  }

  tbDisp(s);

  PyObject *mtb = PyImport_ImportModule("traceback");
  if(!mtb) fprintf(stderr, "cannot import traceback\n");
  else{
    char *fmt[] = {"format_exception_only", "format_exception"};
    PyObject *formatted_list;
    if(!ptb) formatted_list = PyObject_CallMethod(mtb, fmt[0],
      "OO", ptyp, pval);
    else formatted_list = PyObject_CallMethod(mtb, fmt[1],
      "OOO", ptyp, pval, ptb);
    if(!formatted_list){
      fprintf(stderr, "None == traceback.%s(...)\n", fmt[ptb ? 1 : 0]);
    }else{
      long len = PyLong_AsLong(
        PyObject_CallMethod(formatted_list, "__len__", NULL));
      if(0) fprintf(stderr, "traceback.%s(...): %d\n", fmt[ptb ? 1 : 0], len);
      long i;
      for(i = 0; i < len; ++i)
        fprintf(stderr, "%s", PyString_AsString(
          PyList_GetItem(formatted_list, i)));
    }
  }
  return Py_BuildValue("{ss}", "s", s);
}

PyObject *BMP(PyObject *self, PyObject *args, PyObject *kw)
{
  PyArrayObject *nda = NULL;

#if 0
  FILE *fp = fopen(TESTLOG, "ab");
  fprintf(fp, "BMP %08x %08x %08x\n",
    (uchar *)self, (uchar *)args, (uchar *)kw);
  fclose(fp);
#endif

  char *keys[] = {"nda", NULL};
  if(!PyArg_ParseTupleAndKeywords(args, kw, "|O!", keys, &PyArray_Type, &nda)){
    FILE *fp = fopen(TESTLOG, "ab");
    fprintf(fp, "ERROR: PyArg_ParseTupleAndKeywords()\n");
    fclose(fp);
    Py_RETURN_NONE; // must raise Exception
  }else{
#if 0
    FILE *fp = fopen(TESTLOG, "ab");
    fprintf(fp, "SUCCESS: PyArg_ParseTupleAndKeywords(ndim=%d)\n",
      PyArray_NDIM(nda));
    fclose(fp);
#endif
  }

  if(nda){
    fprintf(stderr, "TEST\n");
  }
  if(!nda){ // if(nda == Py_None)
    Py_RETURN_NONE; // must raise Exception
  }
  Py_INCREF(nda);
  return (PyObject *)nda;
}

static PyMethodDef bmp_methods[] = {
  {"bmpProcessException", (PyCFunction)bmpProcessException,
    METH_VARARGS | METH_KEYWORDS, "*args, **kw:\n"
    " s:\n"
    " typ:\n"
    " val:\n"
    " tb:\n"
    "result: dict (output to stderr)"},
  {"BMP", (PyCFunction)BMP,
    METH_VARARGS | METH_KEYWORDS, "*args, **kw:\n"
    " nda: .BMP ndarray\n"
    "result: nda (and display to DC)"},
  {NULL, NULL, 0, NULL}
};

static char bmp_docstr[] = \
  "about this module\n"\
  "BMP image handler for Python (to numpy ndarray or PIL) native C .pyd";

PyMODINIT_FUNC initbmp()
{
  PyObject *m = Py_InitModule3(_BMP, bmp_methods, bmp_docstr);
  if(!m) return;
  /* IMPORTANT: this must be called */
  import_array();
}
