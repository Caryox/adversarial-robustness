import numpy as np
import torch
def bit_reduction(img_data,dataset_name,clip_min=0.499999,clip_max=0.5,bit=4):
# Function for reducing image dimensionality by reducing range of used color depth  
  '''
    Try denormalize image and clip lower and upper pixel values
	Usage: reduced_batch = bit_reduction(original_batch)
  '''
 
  img_data_bit = torch.clone(img_data)
 
  img_min = min(np.ravel(img_data_bit))
  img_max = max(np.ravel(img_data_bit))

  if  "CIFAR" in str(dataset_name):
    step_size = abs(img_min-img_max)/(pow(2,bit)-1)
    steps = pow(2,bit)

    reduced_data = (img_data_bit-min(np.ravel(img_data_bit))) / (max(np.ravel(img_data_bit)) - min(np.ravel(img_data_bit)))
    step_size_norm = abs(min(np.ravel(reduced_data))-max(np.ravel(reduced_data)))/pow(2,bit)

    for i in range(img_data_bit.shape[0]):
      for j in range(img_data_bit.shape[1]): #RGB Values
        for k in range(steps):
          reduced_data[i][j][(reduced_data[i][j]>=(step_size_norm*(k))) & (reduced_data[i][j]<=step_size_norm*(k+1))] = img_min+(k*step_size)
  else:
    reduced_data = (img_data_bit-min(np.ravel(img_data_bit))) / (max(np.ravel(img_data_bit)) - min(np.ravel(img_data_bit)))
    reduced_data = reduced_data.clip(min=clip_min,max=clip_max)

    if (img_data_bit.shape.__len__() > 2):
      for i in range(img_data_bit.shape[0]):
        reduced_data[i][0][reduced_data[i][0]<clip_max] = torch.from_numpy(np.array(img_min))
        reduced_data[i][0][reduced_data[i][0]==clip_max] = torch.from_numpy(np.array(img_max))
    else:
      reduced_data[0][reduced_data[0]<clip_max] = torch.from_numpy(np.array(img_min))
      reduced_data[0][reduced_data[0]==clip_max] = torch.from_numpy(np.array(img_max))
	  
  return(reduced_data)