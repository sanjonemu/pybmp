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
bm = np.array(Image.open('../res/cs_checker_7x5_24.bmp'))
axis[0].imshow(bm)
bmp.BMP(bm)
# plt.show()
```


Links
-----

github https://github.com/sanjonemu/pybmp

pybmp https://pypi.python.org/pypi/pybmp


License
-------

MIT License

