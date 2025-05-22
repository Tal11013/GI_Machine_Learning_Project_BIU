from torch import nn
import torch.nn.functional as F
import torch

from consts import *

class ConvolutionalNet(nn.Module):
    def __init__(self, batch_size, shape=(IMAGE_SIZE, IMAGE_SIZE)):
        super(ConvolutionalNet, self).__init__()
        self.batch_size = batch_size
        self.shape = shape

        # Convolutional Layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)  # Input: (1, 128, 128), Output: (16, 128, 128)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)  # Input: (16, 64, 64), Output: (32, 64, 64)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2)  # Input: (32, 32, 32), Output: (64, 32, 32)
        self.bn3 = nn.BatchNorm2d(64)

        self.mp = nn.MaxPool2d(2)  # Pooling layer halves the size

        # Fully connected layers after convolution
        # The final size after convolutions and pooling is (64, 16, 16), i.e., 64 * 16 * 16 = 16384
        self.fc1 = nn.Linear(64 * 16 * 16, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, 1024)
        self.fc5 = nn.Linear(1024, 2)

        self.do = nn.Dropout()

    def forward(self, x):
        # Convolutional layers with batch normalization and max pooling
        x = self.conv1(x)  # Input: (batch_size, 1, 128, 128), Output: (batch_size, 16, 128, 128)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.mp(x)  # Max Pooling: Output: (batch_size, 16, 64, 64)

        x = self.conv2(x)  # Input: (batch_size, 16, 64, 64), Output: (batch_size, 32, 64, 64)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.mp(x)  # Max Pooling: Output: (batch_size, 32, 32, 32)

        x = self.conv3(x)  # Input: (batch_size, 32, 32, 32), Output: (batch_size, 64, 32, 32)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.mp(x)  # Max Pooling: Output: (batch_size, 64, 16, 16)

        # Flatten the output from the convolutional layers
        x = torch.flatten(x, 1)  # Flatten: Output: (batch_size, 64 * 16 * 16) = (batch_size, 16384)

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.do(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)

        return torch.sigmoid(x)