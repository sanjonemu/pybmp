#!/usr/local/bin/python
# -*- coding: utf-8 -*-
'''test_c
'''

import sys, os

import numpy as np
from PIL import Image

import ctypes

BMP_PYD = '../dll/bmp.pyd'
BMP_FILE = '../res/cs_checker_7x5_24.bmp'
BMP_OUT = '/tmp/cs_out_.bmp'

def main():
  bmp = ctypes.cdll.LoadLibrary(BMP_PYD)
  img = Image.open(BMP_FILE) # R,G,B
  # fake convert to ubytes img:R,G,B -> s:R,G,B,FF -> im:B,G,R,A -> bm:B,G,R,A
  nda = np.array(img)
  r, c, m = nda.shape
  s = img.tostring('raw', 'RGBX') # append A as FF
  im = Image.fromstring('RGBA', (c, r), s, 'raw', 'BGRA') # fake to B,G,R,A
  bm = im.tostring('raw', 'RGBA') # fake to B,G,R,A
  buf = ctypes.create_string_buffer(bm) # B,G,R,A
  fn = ctypes.create_string_buffer(BMP_OUT)
  sys.stderr.write('%d\n' % bmp.drawbmp(c, r, 4, 8, buf, fn))
  return True

if __name__ == '__main__':
  main()
