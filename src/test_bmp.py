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

def main():
  fig = plt.figure()
  axis = [fig.add_subplot(211 + _) for _ in range(2)]
  bm = np.array(Image.open(BMP_INFILE))
  axis[0].imshow(bm)
  bmp.BMP(bm)
  # plt.show()

if __name__ == '__main__':
  main()
