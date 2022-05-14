import torch.optim
# Custom Packages
import sys
sys.path.append('./utils')
sys.path.append('./src/functions')

import basic_nn as basic
import param
  
optimizer = torch.optim.SGD(params=basic.parameters(), lr=param.learning_rate, momentum=param.momentum)
