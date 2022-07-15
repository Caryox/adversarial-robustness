from pyexpat import model
from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
class simple_net_upgraded(nn.Module): # This is the same as the previous one, but with a different (and more easy to understand) architecture. It has a few more neurons in each layer, replaced x.view(-1, 320) with linear layer. Replaced functional dropout with nn.dropout.
	def __init__(self, numChannels, classes):
		super(simple_net_upgraded, self).__init__()

		self.conv1 = nn.Conv2d(in_channels=numChannels, out_channels=20,
			kernel_size=(5, 5))
		self.relu1 = nn.ReLU()
		self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

		self.conv2 = nn.Conv2d(in_channels=20, out_channels=50,
			kernel_size=(5, 5))
		self.relu2 = nn.ReLU()
		#self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		#self.avg_pool2d= nn.AvgPool2d((2,2))
		self.avg_pool2d= nn.AdaptiveAvgPool2d((1, 1))
		#self.conv2_drop = nn.Dropout2d()

		self.fc1 = nn.Linear(in_features=800, out_features=500)
		self.relu3 = nn.ReLU()
		self.dropout_l = nn.Dropout(p=0.5)
		#self.average_pool2d = nn.AvgPool2d((2, 2), stride=(1, 1))
		self.fc2 = nn.Linear(in_features=500, out_features=classes)
		

	def forward(self, x):

		x = self.conv1(x)
		x = self.relu1(x)
		x = self.maxpool1(x)

		x = self.conv2(x)
		x = self.relu2(x)
		#x = self.maxpool2(x)
		x = self.avg_pool2d(x)
		#x = self.conv2_drop(x)
		x = torch.flatten(x, 1)
		x = self.fc1(x)
		x = self.relu3(x)
		x = self.dropout_l(x)
		#x = self.average_pool2d(x)
		x = self.fc2(x)
		

		return x"""

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

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
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.flatten = nn.Flatten(1,-1)
        
        self.linear = nn.Linear(64, num_classes)

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
        #print(out.size()[3])
        out = self.avgpool(out) #F.avg_pool2d(out, out.size()[3])
        out = self.flatten(out)
        #out = out.view(out.size(0), -1)
        #print(out.size())
        out = self.linear(out)
        #print(out.size())
        return out

def weights_init_uniform(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.uniform_(0.0, 1.0)

def ResNet20():
    return ResNet(BasicBlock, [3, 3, 3])

def ResNet32():
    return ResNet(BasicBlock, [5, 5, 5])

def ResNet44():
    return ResNet(BasicBlock, [7, 7, 7])

def ResNet56():
    return ResNet(BasicBlock, [9, 9, 9])

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
