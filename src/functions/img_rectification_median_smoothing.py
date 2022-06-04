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
