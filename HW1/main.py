from scipy import misc
import numpy as np
import tensorflow as tf

image = misc.imread("3.png", True)
im = np.asarray(image)
print (image.shape)

