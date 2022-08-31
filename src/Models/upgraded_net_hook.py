'''
ResNet and hook for weight-matrix and avg-pool (Few2Decide)
'''

from pyexpat import model
from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    # Structure of the residual block
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Identity shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, input_channel, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(input_channel, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.flatten = nn.Flatten(1,-1)
        
        self.linear = nn.Linear(64, num_classes)
    
    # downsampling of the input
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out) 
        out = self.flatten(out)
        out = self.linear(out)

        return out

# weight initialize 
def weights_init_uniform(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.uniform_(0.0, 1.0)

# ResNet-20 
'''
def ResNet20(input_channel):
    return ResNet(input_channel, BasicBlock, [3, 3, 3])
'''
# ResNet-32
'''
def ResNet32(input_channel):
    return ResNet(input_channel, BasicBlock, [5, 5, 5])
'''
# ResNet-44
def ResNet44(input_channel):
    return ResNet(input_channel, BasicBlock, [7, 7, 7])

# ResNet-56
'''
def ResNet56(input_channel):
    return ResNet(input_channel, BasicBlock, [9, 9, 9])
'''

# hook (required for Few2Decide)

activation = {} #this will store values from fc2 layer, which would be then used as input for the few2decide
def get_activation(name):
    def hook(model, input, output):
        activation[name] = model.weight
    return hook


average_pooling = {} #this will store values global average pooling, which would be then used as input for the few2decide
def get_pooling(name):
    def hook(model, input, output):
        average_pooling[name] = output.detach()
    return hook
