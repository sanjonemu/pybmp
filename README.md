pybmp
=====

BMP image handler for Python (to numpy ndarray or PIL) native C .pyd


How to use
----------

```python
from pybmp import bmp
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
from scipy import misc
from PIL import Image

fig = plt.figure()
axis = [fig.add_subplot(211 + _) for _ in range(2)]
img = Image.open('../res/cs_checker_7x5_24.bmp') # R,G,B
# fake convert to ubytes img:R,G,B -> s:R,G,B,FF -> im:B,G,R,A -> bm:B,G,R,A
nda = np.array(img)
r, c, m = nda.shape
s = img.tostring('raw', 'RGBX') # append A as FF
im = Image.fromstring('RGBA', (c, r), s, 'raw', 'BGRA') # fake to B,G,R,A
bm = np.array(im) # fake to B,G,R,A
axis[0].imshow(nda)
bmp.BMP(bm, fn='/tmp/cs_checker_out_.bmp')
# plt.show()
```


Links
-----

github https://github.com/sanjonemu/pybmp

pybmp https://pypi.python.org/pypi/pybmp


License
-------

MIT License

