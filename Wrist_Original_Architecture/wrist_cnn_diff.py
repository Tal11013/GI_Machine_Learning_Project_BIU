from torch import nn
import torch.nn.functional as F
import torch


class DiffuserLayer(nn.Module):
    def __init__(self, input_size, diffuser_weights):
        super(DiffuserLayer, self).__init__()
        self.input_size = input_size
        self.diffuser_weights = nn.Parameter(diffuser_weights)

    def forward(self, x):
        diffused_input = x * self.diffuser_weights
        return diffused_input

class ConvolutionalNetDiff(nn.Module):
    def __init__(self, num_of_measurements, batch_size, shape):
        super(ConvolutionalNetDiff, self).__init__()
        self.batch_size = batch_size
        self.shape = shape
        num_of_features = shape[0] * shape[1]
        input_size = num_of_measurements
        diffuser_weights = torch.ones((num_of_features,), requires_grad = True)
        self.diffuser = DiffuserLayer(num_of_features, diffuser_weights)

        self.mp = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(num_of_features, num_of_features)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.do = nn.Dropout()
        self.fc4 = nn.Linear(2048, 1024)
        self.fc5 = nn.Linear(1024, 2)

    def forward(self, x):
        # Pass through the diffuser layer
        x = torch.flatten(x, 1)
        x = self.diffuser(x)
        # fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = torch.reshape(x, (self.batch_size, 1, 32, 64))
        # first convolutional layer
        x = self.conv1(x)
        # batch normalization
        x = self.bn1(x)
        # max pooling
        x = self.mp(x)
        x = F.relu(x)
        # second convolutional layer
        x = self.conv2(x)
        # batch normalization
        x = self.bn2(x)
        # max pooling
        x = self.mp(x)
        x = F.relu(x)
        # third convolutional layer
        x = self.conv3(x)
        # batch normalization
        x = self.bn3(x)
        # max pooling
        x = self.mp(x)
        x = F.relu(x)
        # drop out
        x = self.do(x)
        x = torch.reshape(x, (self.batch_size, 2048))
        # fully connected layers
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        return torch.sigmoid(x)