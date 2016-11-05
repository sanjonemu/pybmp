#!/usr/local/bin/python
# -*- coding: utf-8 -*-
'''test_bmp
'''

import sys, os

import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
from scipy import misc
from PIL import Image

sys.path.append('..')
from pybmp import bmp

BMP_BASE = '../res'
BMP_INFILE = '%s/cs_checker_7x5_24.bmp' % BMP_BASE
BMP_OUTFILE = '/tmp/cs_dummy_.bmp'

def main():
  fig = plt.figure()
  axis = [fig.add_subplot(211 + _) for _ in range(2)]
  img = Image.open(BMP_INFILE) # R,G,B
  # fake convert to ubytes img:R,G,B -> s:R,G,B,FF -> im:B,G,R,A -> bm:B,G,R,A
  nda = np.array(img)
  r, c, m = nda.shape
  s = img.tostring('raw', 'RGBX') # append A as FF
  im = Image.fromstring('RGBA', (c, r), s, 'raw', 'BGRA') # fake to B,G,R,A
  bm = np.array(im) # fake to B,G,R,A
  axis[0].imshow(nda)
  bmp.BMP(bm, fn=BMP_OUTFILE)
  # plt.show()

if __name__ == '__main__':
  main()
