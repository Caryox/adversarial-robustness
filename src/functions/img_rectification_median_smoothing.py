  # Smooth image by apllying median filter with kernel size (m*n). Use default kernel size of 2x2 as indicated by CW,
  '''
    Try to calculate median of given pixels inside of kernel window to normalize it's values.
	The function can be used with batch or single image array by measuring its shape and thereby it's dimension.
	The image data is casted into torch sensor because the scipy filter will change it's type otherwise to ndarray format.
	Usage for batch: smoothed_batch = median_smoothing(original_batch)
	Usage for single image: smoothed_image = median_smoothing(original_batch[index_of_image][0])
  '''
from scipy import ndimage
import torch

def median_smoothing(img_data,kernel_size=2):
  from scipy import ndimage
  print(img_data.shape.__len__())
  if (img_data.shape.__len__() > 2):
    for i in range(img_data.shape[0]):
      img_data[i][0] = torch.from_numpy(ndimage.median_filter(img_data[i][0], size=kernel_size))
  else:
    img_data = torch.from_numpy(ndimage.median_filter(img_data, size=kernel_size))
  return(img_data)
