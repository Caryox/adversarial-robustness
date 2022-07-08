
import torch


#model = simple_net_upgraded(28,28)                           get weight matrix
#model.fc2.register_forward_hook(get_activation('fc2'))
#x = torch.rand(1,28,28,28)
#last_convolution = model(x)
#print(activation['fc2'])

#model.maxpool2.register_forward_hook(get_activation('maxpool2')) get average pooling layer
#x = torch.rand(1,28,28,28)
#max_pool = model(x)
#print(activation['maxpool2'])



last_convolution = torch.rand(1,28,28,28)  # These values are just generated to be able to run the queries
max_pool = torch.rand(1,28,28,28)


v1 = last_convolution*max_pool # 1. Hadamard Multiplication Elementwise Multiplication of matrices (the shape should remain the same)


v2 = torch.sort(v1, dim=1, descending=False, stable=False, out=None) # 2. sort the connections calculation results of each neuron from min to max and get the V2


v3 = torch.clamp(v1,min=-0.5, max=0.5) # In this step the nd Tensor should get the top and bottom 30 percent of the values set to zero


V4 = torch.sum(v1,1) #Prediction Score