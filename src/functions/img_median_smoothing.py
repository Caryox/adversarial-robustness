def median_smoothing(img_data,dataset_name,kernel_size=2):
    # Smooth image by apllying median filter with kernel size (m*n). Use default kernel size of 2x2 as indicated by CW,
  '''
    Try to calculate median of given pixels inside of kernel window to normalize it's values.
    The function can be used with batch or single image array by measuring its shape and thereby it's dimension.
    The image data is casted into torch sensor because the scipy filter will change it's type otherwise to ndarray format.
    Usage for batch: smoothed_batch = median_smoothing(original_batch)
    Usage for single image: smoothed_image = median_smoothing(original_batch[index_of_image][0])
  '''
  import torch
  from scipy import ndimage 
  img_data_median = torch.clone(img_data) #Clone tensor to protect original tensor data
  if  "CIFAR" in str(dataset_name):
    for i in range(img_data_median.shape[0]):
      for j in range(img_data_median.shape[1]): #RGB Values
        img_data_median[i][j] = torch.from_numpy(ndimage.median_filter(img_data_median[i][j], size=kernel_size)) #Use ndimage filter for CIFAR-10
  else:  
    if (img_data_median.shape.__len__() > 2):
      for i in range(img_data_median.shape[0]):
        img_data_median[i][0] = torch.from_numpy(ndimage.median_filter(img_data_median[i][0], size=kernel_size)) #Use ndimage filter for MNIST if batch is used
    else:
      img_data_median = torch.from_numpy(ndimage.median_filter(img_data_median, size=kernel_size)) #Use ndimage filter for MNIST if single image is used
  return(img_data_median)