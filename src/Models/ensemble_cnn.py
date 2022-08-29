import torch
import torch.nn as nn
import torch.nn.functional as F

class ensemble_rectification(nn.Module): # This is the same as the previous one, but with a different (and more easy to understand) architecture. It has a few more neurons in each layer, replaced x.view(-1, 320) with linear layer. Replaced functional dropout with nn.dropout.
	def __init__(self, numChannels, classes):
		super(ensemble_rectification, self).__init__()

		#Net1
		self.conv1_1 = nn.Conv2d(in_channels=numChannels, out_channels=20,
			kernel_size=(5, 5))
		self.relu1_1 = nn.ReLU()
		self.maxpool1_1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

		self.conv2_1 = nn.Conv2d(in_channels=20, out_channels=50,
			kernel_size=(5, 5))
		self.relu2_1 = nn.ReLU()
		self.maxpool2_1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

		self.conv2_drop_1 = nn.Dropout2d()

		self.fc1_1 = nn.Linear(in_features=800, out_features=500)
		self.relu3_1 = nn.ReLU()
		self.dropout_l_1 = nn.Dropout(p=0.5)
		self.fc2_1 = nn.Linear(in_features=500, out_features=classes)

		#Net2
		self.conv1_2 = nn.Conv2d(in_channels=numChannels, out_channels=30,
			kernel_size=(5, 5))
		self.relu1_2 = nn.ReLU()
		self.maxpool1_2 = nn.MaxPool2d(kernel_size=(2), stride=(2, 2))

		self.conv2_2 = nn.Conv2d(in_channels=30, out_channels=75,
			kernel_size=(5, 5))
		self.relu2_2 = nn.ReLU()
		self.maxpool2_2 = nn.MaxPool2d(kernel_size=(2), stride=(2, 2))

		self.conv2_drop_2 = nn.Dropout2d()

		self.fc1_2 = nn.Linear(in_features=1200, out_features=900)
		self.relu3_2 = nn.ReLU()
		self.dropout_l_2 = nn.Dropout(p=0.3)
		self.fc2_2 = nn.Linear(in_features=900, out_features=classes)

		#Net3
		self.conv1_3 = nn.Conv2d(in_channels=numChannels, out_channels=50,
			kernel_size=(5, 5))
		self.relu1_3 = nn.ReLU()
		self.maxpool1_3 = nn.MaxPool2d(kernel_size=(2), stride=(2))

		self.conv2_3 = nn.Conv2d(in_channels=50, out_channels=125,
			kernel_size=(5, 5))
		self.relu2_3 = nn.ReLU()
		self.maxpool2_3 = nn.MaxPool2d(kernel_size=(2), stride=(2))

		self.conv2_drop_3 = nn.Dropout2d()

		self.fc1_3 = nn.Linear(in_features=2000, out_features=1000)
		self.relu3_3 = nn.ReLU()
		self.dropout_l_3 = nn.Dropout(p=0.2)
		self.fc2_3 = nn.Linear(in_features=1000, out_features=classes)
			
	def forward(self, orig):
		x = self.conv1_1(orig)
		x = self.relu1_1(x)
		x = self.maxpool1_1(x)
		x = self.conv2_1(x)
		x = self.relu2_1(x)
		x = self.maxpool2_1(x)
		x = self.conv2_drop_1(x)
		x = torch.flatten(x, 1)
		x = self.fc1_1(x)
		x = self.relu3_1(x)
		x = self.dropout_l_1(x)
		x = self.fc2_1(x)

		y = self.conv1_2(orig)
		y = self.relu1_2(y)
		y = self.maxpool1_2(y)
		y = self.conv2_2(y)
		y = self.relu2_2(y)
		y = self.maxpool2_2(y)
		y = self.conv2_drop_2(y)
		y = torch.flatten(y, 1)
		y = self.fc1_2(y)
		y = self.relu3_2(y)
		y = self.dropout_l_2(y)
		y = self.fc2_2(y)

		z = self.conv1_3(orig)
		z = self.relu1_3(z)
		z = self.maxpool1_3(z)
		z = self.conv2_3(z)
		z = self.relu2_3(z)
		z = self.maxpool2_3(z)
		z = self.conv2_drop_3(z)
		z = torch.flatten(z, 1)
		z = self.fc1_3(z)
		z = self.relu3_3(z)
		z = self.dropout_l_3(z)
		z = self.fc2_3(z)

		concat = torch.cat([x, y, z], dim=1)

		return concat