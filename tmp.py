import random
random.seed()

import numpy as np

from pprint import pprint

def _convolve2d(image, kernel):
  shape = (image.shape[0] - kernel.shape[0] + 1, image.shape[1] - kernel.shape[1] + 1) + kernel.shape
  strides = image.strides * 2
  strided_image = np.lib.stride_tricks.as_strided(image, shape, strides)
  return np.einsum('kl,ijkl->ij', kernel, strided_image)

num_row = 10
num_col = 10

# Initialize matrix for if-bombs-there and the number-to-show
bombs_array = []
num_bombs_array = []

extended_bombs_array = np.random.randint(0,2,[num_row+2,num_col+2])
extended_bombs_array[0,:] = 0
extended_bombs_array[-1,:] = 0
extended_bombs_array[:,0] = 0
extended_bombs_array[:,-1] = 0

bombs_array = extended_bombs_array[1:-1, 1:-1]
num_bombs_array = _convolve2d(extended_bombs_array, np.array([[1,1,1],[1,0,1],[1,1,1]]))

pprint(bombs_array)
pprint(num_bombs_array)