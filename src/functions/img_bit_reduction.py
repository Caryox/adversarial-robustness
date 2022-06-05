# Function for reducing image dimensionality by reducing range of used color depth  
  '''
    Try denormalize image and clip lower and upper pixel values
	Usage: reduced_batch = bit_reduction(original_batch)
  '''

def bit_reduction(img_data,clip_min=0.5,clip_max=0.9):
  reduced_data = (img_data-min(np.ravel(img_data))) / (max(np.ravel(img_data)) - min(np.ravel(img_data)))
  reduced_data = reduced_data.clip(min=clip_min,max=clip_max)
  return(reduced_data)
