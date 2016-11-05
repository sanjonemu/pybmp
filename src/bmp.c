/*
  bmp.c

  >mingw32-make -f makefile.tdmgcc64
  >test_bmp.py

```python
import binascii
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open('../res/cs_checker_7x5_24.bmp') # R,G,B

binascii.b2a_hex(img.tostring()) # R,G,B
# 'ff0000''ffffff''ff0000''ffffff''ff0000''ffffff''ff0000'\
# 'ffffff''00ff00''ffffff''00ff00''ffffff''00ff00''ffffff'\
# '0000ff''ffffff''0000ff''ffffff''0000ff''ffffff''0000ff'\
# 'ffffff''333333''ffffff''666666''ffffff''cccccc''ffffff'\
# '00ffff''ffffff''ff00ff''ffffff''ffff00''ffffff''000000'

binascii.b2a_hex(img.tostring('raw', 'XBGR')) # 00,B,G,R
# '000000ff''00ffffff''000000ff''00ffffff''000000ff''00ffffff''000000ff'\
# '00ffffff''0000ff00''00ffffff''0000ff00''00ffffff''0000ff00''00ffffff'\
# '00ff0000''00ffffff''00ff0000''00ffffff''00ff0000''00ffffff''00ff0000'\
# '00ffffff''00333333''00ffffff''00666666''00ffffff''00cccccc''00ffffff'\
# '00ffff00''00ffffff''00ff00ff''00ffffff''0000ffff''00ffffff''00000000'

binascii.b2a_hex(img.tostring('raw', 'XRGB')) # 00,R,G,B
# '00ff0000''00ffffff''00ff0000''00ffffff''00ff0000''00ffffff''00ff0000'\
# '00ffffff''0000ff00''00ffffff''0000ff00''00ffffff''0000ff00''00ffffff'\
# '000000ff''00ffffff''000000ff''00ffffff''000000ff''00ffffff''000000ff'\
# '00ffffff''00333333''00ffffff''00666666''00ffffff''00cccccc''00ffffff'\
# '0000ffff''00ffffff''00ff00ff''00ffffff''00ffff00''00ffffff''00000000'

binascii.b2a_hex(img.tostring('raw', 'RGBX')) # R,G,B,FF
# 'ff0000ff''ffffffff''ff0000ff''ffffffff''ff0000ff''ffffffff''ff0000ff'\
# 'ffffffff''00ff00ff''ffffffff''00ff00ff''ffffffff''00ff00ff''ffffffff'\
# '0000ffff''ffffffff''0000ffff''ffffffff''0000ffff''ffffffff''0000ffff'\
# 'ffffffff''333333ff''ffffffff''666666ff''ffffffff''ccccccff''ffffffff'\
# '00ffffff''ffffffff''ff00ffff''ffffffff''ffff00ff''ffffffff''000000ff'

binascii.b2a_hex(img.tostring('raw', 'BGRX')) # B,G,R,00
# '0000ff00''ffffff00''0000ff00''ffffff00''0000ff00''ffffff00''0000ff00'\
# 'ffffff00''00ff0000''ffffff00''00ff0000''ffffff00''00ff0000''ffffff00'\
# 'ff000000''ffffff00''ff000000''ffffff00''ff000000''ffffff00''ff000000'\
# 'ffffff00''33333300''ffffff00''66666600''ffffff00''cccccc00''ffffff00'\
# 'ffff0000''ffffff00''ff00ff00''ffffff00''00ffff00''ffffff00''00000000'

# fake convert to ubytes img:R,G,B -> s:R,G,B,FF -> im:B,G,R,A -> bm:B,G,R,A
nda = np.array(img)
r, c, m = nda.shape
s = img.tostring('raw', 'RGBX') # append A as FF
im = Image.fromstring('RGBA', (c, r), s, 'raw', 'BGRA') # fake to B,G,R,A
bm = im.tostring('raw', 'RGBA') # fake to B,G,R,A
binascii.b2a_hex(bm)
# '0000ffff''ffffffff''0000ffff''ffffffff''0000ffff''ffffffff''0000ffff'\
# 'ffffffff''00ff00ff''ffffffff''00ff00ff''ffffffff''00ff00ff''ffffffff'\
# 'ff0000ff''ffffffff''ff0000ff''ffffffff''ff0000ff''ffffffff''ff0000ff'\
# 'ffffffff''333333ff''ffffffff''666666ff''ffffffff''ccccccff''ffffffff'\
# 'ffff00ff''ffffffff''ff00ffff''ffffffff''00ffffff''ffffffff''000000ff'

fig = plt.figure()
axis = [fig.add_subplot(221 + _) for _ in range(4)]
axis[0].imshow(img) # bottom <-> top
axis[2].imshow(nda)
axis[1].imshow(im)
axis[3].imshow(np.array(im))
plt.show()
```
*/

#define __BMP_MAKE_DLL_
#include <bmp.h>

#include <numpy/arrayobject.h>

#define DEBUGLOG 0
#define TESTLOG "../dll/_test_dll_.log"

#ifdef __WINNT__

#define OFFBITS (sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER))

__PORT uint writebmp(char *fn, HBITMAP hbmp, int c, int r, HDC hdc)
{
  size_t wlen, sz;
  LPBYTE p;
  FILE *fp;
  if(!fn) return 1;
  if(!(fp = fopen(fn, "wb"))){
    fprintf(stderr, "cannot create: %s", fn);
    return 1;
  }
  wlen = c * 3;
  if(wlen % 4) wlen += 4 - wlen % 4;
  sz = OFFBITS + r * wlen;
  if(p = (LPBYTE)malloc(sz)){
    LPBITMAPFILEHEADER bh = (LPBITMAPFILEHEADER)p;
    LPBITMAPINFOHEADER bi = (LPBITMAPINFOHEADER)(p + sizeof(BITMAPFILEHEADER));
    LPBYTE pixels = p + OFFBITS;
    memset(bh, 0, sizeof(BITMAPFILEHEADER));
    memset(bi, 0, sizeof(BITMAPINFOHEADER));
    bh->bfType = ('M' << 8) | 'B';
    bh->bfSize = sz;
    bh->bfOffBits = OFFBITS;
    bi->biSize = sizeof(BITMAPINFOHEADER);
    bi->biWidth = c;
    bi->biHeight = r;
    bi->biPlanes = 1; // not be 3
    bi->biBitCount = 24; // not be 8
    bi->biCompression = BI_RGB;
    // rows may be reversed top <-> bottom copying them from bitmap to buffer
    GetDIBits(hdc, hbmp, 0, r, pixels, (LPBITMAPINFO)bi, DIB_RGB_COLORS);
    fwrite(p, sz, 1, fp);
    free(p);
  }
  fclose(fp);
  return 0;
}

// cols rows planes depth-bits ary
__PORT uint drawbmp(int c, int r, int p, int d, char *bmpbuffer, char *fn)
{
  uint result = 0;
  // should call CreateCompatibleBitmap(hdc, c, r) when use unknown device
  HBITMAP hbmp = CreateBitmap(c, r, p, d, bmpbuffer); // (A)RGB L.E. -> B,G,R,A
  HWND hwnd = GetDesktopWindow();
  HDC hdc = GetDC(hwnd);
  HDC hmdc = CreateCompatibleDC(hdc);
  HBITMAP obmp = (HBITMAP)SelectObject(hmdc, hbmp);
  StretchBlt(hdc, 0, 0, c * IASPECTX, r * IASPECTY, hmdc, 0, 0, c, r, SRCCOPY);
  SelectObject(hmdc, obmp);
  if(fn) result = writebmp(fn, hbmp, c, r, hmdc);
  DeleteObject(hbmp);
  DeleteDC(hmdc);
  ReleaseDC(hwnd, hdc);
  return result;
}

#else

// dummy
__PORT uint drawbmp(int c, int r, int p, int d, char *bmpbuffer, char *fn)
{
  fprintf(stderr, "drawbmp works only on Win32API now, sorry\n");
  return 0;
}

#endif

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
  char *fn = NULL;

#if 0
  FILE *fp = fopen(TESTLOG, "ab");
  fprintf(fp, "BMP %08x %08x %08x\n",
    (uchar *)self, (uchar *)args, (uchar *)kw);
  fclose(fp);
#endif

  char *keys[] = {"nda", "fn", NULL};
  if(!PyArg_ParseTupleAndKeywords(args, kw, "|O!s", keys,
    &PyArray_Type, &nda, &fn)){
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
    int ndim = PyArray_NDIM(nda);
    npy_int *dims = PyArray_DIMS(nda);
    if(ndim == 3 && PyArray_TYPE(nda) == NPY_UINT8)
      drawbmp(dims[1], dims[0], dims[2], 8, PyArray_DATA(nda), fn);
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
#ifdef __WINNT__
    "  fn: save to fn (default None: skip)\n"
#endif
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
